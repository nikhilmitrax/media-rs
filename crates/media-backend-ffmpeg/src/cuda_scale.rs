//! Narrow native CUDA component-plane scaling, with checked graph operations.

use ffmpeg_next as ffmpeg;
use media_backend_cuda::CudaContext;
use media_core::error::{Error, Result};

use crate::{
    cuda_encode::{CudaResizeFilter, producer_ready},
    format::backend_error,
    native,
};

pub(crate) struct CudaScaler {
    graph: Option<ffmpeg::filter::Graph>,
    // The graph and all its native buffers must be freed before this owner.
    context: CudaContext,
}
impl CudaScaler {
    pub(crate) fn new(
        frame: &ffmpeg::frame::Video,
        context: &CudaContext,
        base: ffmpeg::Rational,
        rate: ffmpeg::Rational,
        width: u32,
        height: u32,
        filter: CudaResizeFilter,
    ) -> Result<Self> {
        let scaler = ffmpeg::filter::find("scale_cuda").ok_or_else(|| Error::unsupported("linked FFmpeg has no scale_cuda filter; native resize requires CUDA filter support, never a host fallback"))?;
        let source = ffmpeg::filter::find("buffer")
            .ok_or_else(|| Error::unsupported("linked FFmpeg lacks the video buffer source"))?;
        let sink = ffmpeg::filter::find("buffersink")
            .ok_or_else(|| Error::unsupported("linked FFmpeg lacks the video buffer sink"))?;
        let mut graph = ffmpeg::filter::Graph::new();
        ffi_helpers::configure_graph(&mut graph);
        let aspect = frame.aspect_ratio();
        let aspect = if aspect.numerator() > 0 && aspect.denominator() > 0 {
            aspect
        } else {
            ffmpeg::Rational(1, 1)
        };
        let pixel: ffmpeg::ffi::AVPixelFormat = ffmpeg::format::Pixel::CUDA.into();
        let args = format!(
            "video_size={}x{}:pix_fmt={}:time_base={}/{}:pixel_aspect={}/{}",
            frame.width(),
            frame.height(),
            pixel as i32,
            base.numerator(),
            base.denominator(),
            aspect.numerator(),
            aspect.denominator()
        );
        let mut input = graph.add(&source, "input", &args).map_err(backend_error)?;
        ffi_helpers::attach_source(&mut input, frame, base, rate, aspect)?;
        let args = format!(
            "w={width}:h={height}:interp_algo={}:passthrough=0",
            filter.name()
        );
        let mut scale = graph.add(&scaler, "scale", &args).map_err(backend_error)?;
        let mut output = graph.add(&sink, "output", "").map_err(backend_error)?;
        ffi_helpers::link(&mut input, &mut scale)?;
        ffi_helpers::link(&mut scale, &mut output)?;
        graph
            .validate()
            .map_err(|e| backend_error(e).with_operation("configure native CUDA resize graph"))?;
        Ok(Self {
            graph: Some(graph),
            context: context.clone(),
        })
    }
    pub(crate) fn run(&mut self, input: ffmpeg::frame::Video) -> Result<ffmpeg::frame::Video> {
        let source = native::cuda_surface(&input)?;
        producer_ready(&self.context, source.stream)?;
        let mut output = ffmpeg::frame::Video::empty();
        let result = (|| -> Result<()> {
            let graph = self.graph.as_mut().ok_or_else(|| {
                Error::backend(
                    "ffmpeg",
                    "CUDA scale graph was retired after a driver failure",
                )
            })?;
            let mut node = graph
                .get("input")
                .ok_or_else(|| Error::backend("ffmpeg", "CUDA scale source disappeared"))?;
            ffi_helpers::add_ref(&mut node, &input)?;
            graph
                .get("output")
                .ok_or_else(|| Error::backend("ffmpeg", "CUDA scale sink disappeared"))?
                .sink()
                .frame(&mut output)
                .map_err(|e| backend_error(e).with_operation("receive native scaled CUDA frame"))?;
            let scaled = native::cuda_surface(&output)?;
            if scaled.context != source.context || scaled.format != source.format {
                return Err(Error::incompatible(
                    "CUDA scale changed the actual context or native precision without permission",
                ));
            }
            // Retain original decoder input until asynchronous scale work ends.
            producer_ready(&self.context, scaled.stream)?;
            Ok(())
        })();
        if result.is_err()
            && let Err(error) = ffi_helpers::retire_failed_stream(&self.context, source.stream)
        {
            // An unrecoverable driver error cannot establish safe reuse. Retain
            // native ownership, rather than freeing an in-flight allocation.
            std::mem::forget((input, output, self.graph.take(), self.context.clone()));
            return Err(error
                .with_constraint("cuda_retirement_unknown")
                .with_operation(
                    "retire failed native CUDA scale work; native ownership retained",
                ));
        }
        drop(input);
        result.map(|()| output)
    }
}

#[allow(unsafe_code)]
mod ffi_helpers {
    use super::*;
    pub(super) fn retire_failed_stream(context: &CudaContext, stream: u64) -> Result<()> {
        if producer_ready(context, stream).is_ok() {
            return Ok(());
        }
        // SAFETY: the validated source hardware pool owns this exact stream and
        // context, including valid NULL legacy streams; borrowed wrappers never
        // destroy it. Only this producer stream is synchronized, as a fallback
        // when an event could not establish completion on an error path.
        let borrowed = unsafe {
            context
                .native_context()
                .wrap_borrowed_stream(stream as usize as _)
        };
        borrowed
            .synchronize()
            .map_err(|e| Error::backend("cuda", e.to_string()).with_cause(e))
    }
    pub(super) fn configure_graph(graph: &mut ffmpeg::filter::Graph) {
        // SAFETY: uniquely borrowed newly allocated graph, before configuration.
        unsafe {
            (*graph.as_mut_ptr()).nb_threads = 1;
            ffmpeg::ffi::avfilter_graph_set_auto_convert(
                graph.as_mut_ptr(),
                ffmpeg::ffi::AVFILTER_AUTO_CONVERT_NONE as u32,
            );
        }
    }
    pub(super) fn attach_source(
        context: &mut ffmpeg::filter::Context,
        frame: &ffmpeg::frame::Video,
        base: ffmpeg::Rational,
        rate: ffmpeg::Rational,
        aspect: ffmpeg::Rational,
    ) -> Result<()> {
        // SAFETY: live borrowed source/frame; source parameters_set takes its own
        // hardware-pool reference. Temporary scalar parameters own no fields.
        unsafe {
            let parameters = ffmpeg::ffi::av_buffersrc_parameters_alloc();
            if parameters.is_null() {
                return Err(Error::exhausted(
                    "cannot allocate CUDA buffer source parameters",
                ));
            }
            (*parameters).format = ffmpeg::ffi::AVPixelFormat::AV_PIX_FMT_CUDA as i32;
            (*parameters).width = frame.width() as i32;
            (*parameters).height = frame.height() as i32;
            (*parameters).time_base = base.into();
            (*parameters).frame_rate = rate.into();
            (*parameters).sample_aspect_ratio = aspect.into();
            (*parameters).hw_frames_ctx = (*frame.as_ptr()).hw_frames_ctx;
            let result = ffmpeg::ffi::av_buffersrc_parameters_set(context.as_mut_ptr(), parameters);
            ffmpeg::ffi::av_free(parameters.cast());
            if result < 0 {
                return Err(backend_error(ffmpeg::Error::from(result)));
            }
        }
        Ok(())
    }
    pub(super) fn link(
        source: &mut ffmpeg::filter::Context,
        target: &mut ffmpeg::filter::Context,
    ) -> Result<()> {
        // SAFETY: both nodes belong to the same live graph and have one video pad.
        let result =
            unsafe { ffmpeg::ffi::avfilter_link(source.as_mut_ptr(), 0, target.as_mut_ptr(), 0) };
        if result < 0 {
            return Err(backend_error(ffmpeg::Error::from(result)));
        }
        Ok(())
    }
    pub(super) fn add_ref(
        source: &mut ffmpeg::filter::Context,
        frame: &ffmpeg::frame::Video,
    ) -> Result<()> {
        // SAFETY: KEEP_REF leaves this immutable borrowed frame unmodified and
        // gives the graph its own reference. The safe binding's add() omits this
        // flag and would consume the contents of an immutable Rust borrow.
        let result = unsafe {
            ffmpeg::ffi::av_buffersrc_add_frame_flags(
                source.as_mut_ptr(),
                frame.as_ptr() as *mut _,
                ffmpeg::ffi::AV_BUFFERSRC_FLAG_KEEP_REF as i32,
            )
        };
        if result < 0 {
            return Err(backend_error(ffmpeg::Error::from(result)));
        }
        Ok(())
    }
}
