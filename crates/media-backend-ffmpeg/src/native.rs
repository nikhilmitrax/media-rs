//! Narrow FFmpeg FFI operations missing from the safe binding.
use ffmpeg_next as ffmpeg;
use media_core::error::{Error, Result};
use std::ffi::{CStr, CString};
use std::path::Path;
use std::ptr;

pub fn input_start_time(input: &ffmpeg::format::context::Input) -> i64 {
    // SAFETY: the borrowed Input owns a live AVFormatContext for this read.
    unsafe { (*input.as_ptr()).start_time }
}

pub(crate) fn input_from_headers(
    mut io: ffmpeg::format::context::StreamIo,
    filename: Option<&str>,
    options: ffmpeg::Dictionary<'_>,
) -> Result<ffmpeg::format::context::Input> {
    let filename = filename
        .map(CString::new)
        .transpose()
        .map_err(|_| Error::invalid("input filename contains a NUL byte"))?;
    // SAFETY: io owns the live AVIOContext throughout open. On success ownership
    // is transferred exactly once to Input's public custom-I/O destructor; on
    // failure io remains a Rust owner and FFmpeg closes only the format context.
    unsafe {
        let mut context = ffmpeg::ffi::avformat_alloc_context();
        if context.is_null() {
            return Err(Error::exhausted("cannot allocate indexed demux context"));
        }
        (*context).pb = io.as_mut_ptr();
        (*context).flags |= ffmpeg::ffi::AVFMT_FLAG_CUSTOM_IO;
        let mut dictionary = options.disown();
        let status = ffmpeg::ffi::avformat_open_input(
            &mut context,
            filename.as_ref().map_or(ptr::null(), |name| name.as_ptr()),
            ptr::null_mut(),
            &mut dictionary,
        );
        let _remaining = ffmpeg::Dictionary::own(dictionary);
        if status < 0 {
            if !context.is_null() {
                ffmpeg::ffi::avformat_close_input(&mut context);
            }
            return Err(super::format::backend_error(ffmpeg::Error::from(status))
                .with_operation("open indexed container headers"));
        }
        Ok(ffmpeg::format::context::Input::wrap_with_custom_io(
            context, io,
        ))
    }
}

pub(crate) fn seek_video_stream(
    input: &mut ffmpeg::format::context::Input,
    track: u32,
    timestamp: i64,
) -> Result<()> {
    let track = i32::try_from(track)
        .map_err(|_| Error::invalid("video seek track exceeds native stream index"))?;
    // SAFETY: Input exclusively owns the live AVFormatContext; timestamp is in
    // the selected stream's exact time base. FFmpeg retains no Rust pointers.
    let status = unsafe {
        ffmpeg::ffi::av_seek_frame(
            input.as_mut_ptr(),
            track,
            timestamp,
            ffmpeg::ffi::AVSEEK_FLAG_BACKWARD,
        )
    };
    if status < 0 {
        return Err(super::format::backend_error(ffmpeg::Error::from(status))
            .with_track(track as u32)
            .with_operation("indexed keyframe seek"));
    }
    Ok(())
}

pub(crate) fn discover_compressed_headers(
    input: &mut ffmpeg::format::context::Input,
) -> Result<()> {
    // SAFETY: Input exclusively owns a live context/custom AVIO. The open
    // dictionary forbids all probe decoders; only bounded demux/parser discovery
    // is permitted. AVIO enforces the existing byte/cancellation constraints.
    let status =
        unsafe { ffmpeg::ffi::avformat_find_stream_info(input.as_mut_ptr(), ptr::null_mut()) };
    if status < 0 {
        return Err(super::format::backend_error(ffmpeg::Error::from(status))
            .with_operation("discover indexed compressed headers"));
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy)]
pub(crate) struct CudaDeviceDescription {
    pub context: u64,
}

#[cfg(feature = "cuda")]
pub(crate) fn configure_cuda_decoder(
    context: &mut ffmpeg::codec::context::Context,
    ordinal: u32,
    use_current: bool,
    extra_frames: usize,
) -> Result<CudaDeviceDescription> {
    use ffmpeg::ffi;
    if ordinal > i32::MAX as u32 || extra_frames > i32::MAX as usize {
        return Err(Error::invalid(
            "CUDA ordinal or decoder surface count is out of range",
        ));
    }
    // SAFETY: context is uniquely borrowed and owns its AVCodecContext. All
    // option/device strings remain live during the call, and dictionaries and
    // temporary AVBufferRefs are released on every success/failure path.
    unsafe {
        let decoder = ffi::avcodec_find_decoder(context.id().into());
        if decoder.is_null() {
            return Err(Error::unsupported(
                "no FFmpeg decoder is available for the selected codec",
            ));
        }
        let mut supports_cuda = false;
        for index in 0..1024 {
            let configuration = ffi::avcodec_get_hw_config(decoder, index);
            if configuration.is_null() {
                break;
            }
            let configuration = &*configuration;
            if configuration.device_type == ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_CUDA
                && configuration.pix_fmt == ffi::AVPixelFormat::AV_PIX_FMT_CUDA
                && configuration.methods & ffi::AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX as i32 != 0
            {
                supports_cuda = true;
                break;
            }
        }
        if !supports_cuda {
            return Err(Error::unsupported(format!(
                "FFmpeg decoder {} has no CUDA hardware-frame configuration; hardware-required forbids software fallback",
                context.id().name()
            )));
        }
        let mut device = create_cuda_device(ordinal, use_current)?;
        let raw = context.as_mut_ptr();
        if !(*raw).hw_device_ctx.is_null() {
            return Err(Error::incompatible(
                "codec context already owns a hardware device",
            ));
        }
        (*raw).hw_device_ctx = device.reference;
        device.reference = ptr::null_mut();
        (*raw).get_format = Some(select_cuda_format);
        (*raw).extra_hw_frames = extra_frames as i32;
        Ok(device.description)
    }
}

#[cfg(feature = "cuda")]
pub(crate) struct CudaDeviceRef {
    reference: *mut ffmpeg::ffi::AVBufferRef,
    pub(crate) description: CudaDeviceDescription,
}
#[cfg(feature = "cuda")]
impl Drop for CudaDeviceRef {
    fn drop(&mut self) {
        // SAFETY: reference is exclusively owned or NULL after transfer; native
        // AVBufferRef destruction retains the device through its cleanup.
        unsafe {
            ffmpeg::ffi::av_buffer_unref(&mut self.reference);
        }
    }
}
#[cfg(feature = "cuda")]
pub(crate) fn create_cuda_device(ordinal: u32, use_current: bool) -> Result<CudaDeviceRef> {
    use ffmpeg::ffi;
    if ordinal > i32::MAX as u32 {
        return Err(Error::invalid(
            "CUDA ordinal exceeds the native device index range",
        ));
    }
    let device = CString::new(ordinal.to_string()).expect("integer contains no NUL");
    let key = if use_current {
        c"current_ctx"
    } else {
        c"primary_ctx"
    };
    let mut options = ptr::null_mut();
    let mut reference = ptr::null_mut();
    // SAFETY: strings/dictionaries remain live for native device creation;
    // every failure frees temporary native references before returning.
    unsafe {
        let status = ffi::av_dict_set(&mut options, key.as_ptr(), c"1".as_ptr(), 0);
        if status < 0 {
            ffi::av_dict_free(&mut options);
            return Err(crate::format::backend_error(ffmpeg::Error::from(status)));
        }
        let result = ffi::av_hwdevice_ctx_create(
            &mut reference,
            ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_CUDA,
            device.as_ptr(),
            options,
            0,
        );
        ffi::av_dict_free(&mut options);
        if result < 0 {
            ffi::av_buffer_unref(&mut reference);
            return Err(Error::unsupported(format!(
                "cannot initialize FFmpeg CUDA device {ordinal}: {}; ensure this FFmpeg build enables CUDA/NVDEC and the container exposes NVIDIA compute/video driver libraries",
                ffmpeg::Error::from(result)
            )).with_cause(ffmpeg::Error::from(result)));
        }
        if reference.is_null() || (*reference).data.is_null() {
            ffi::av_buffer_unref(&mut reference);
            return Err(Error::backend(
                "ffmpeg",
                "CUDA device creation returned no device context",
            ));
        }
        let device_context = &*((*reference).data as *const ffi::AVHWDeviceContext);
        if device_context.type_ != ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_CUDA
            || device_context.hwctx.is_null()
        {
            ffi::av_buffer_unref(&mut reference);
            return Err(Error::backend(
                "ffmpeg",
                "hardware device is not a CUDA context",
            ));
        }
        let cuda = &*(device_context.hwctx as *const ffi::AVCUDADeviceContext);
        let description = CudaDeviceDescription {
            context: cuda.cuda_ctx as usize as u64,
        };
        if description.context == 0 {
            ffi::av_buffer_unref(&mut reference);
            return Err(Error::backend(
                "ffmpeg",
                "CUDA hardware context has a null native handle",
            ));
        }
        Ok(CudaDeviceRef {
            reference,
            description,
        })
    }
}

#[cfg(all(test, feature = "cuda"))]
mod cuda_contract_tests {
    #[test]
    fn software_formats_cannot_satisfy_required_cuda_selection() {
        use ffmpeg_next::ffi::AVPixelFormat::{
            AV_PIX_FMT_CUDA, AV_PIX_FMT_NONE, AV_PIX_FMT_YUV420P,
        };
        let software_only = [AV_PIX_FMT_YUV420P, AV_PIX_FMT_NONE];
        let hardware = [AV_PIX_FMT_YUV420P, AV_PIX_FMT_CUDA, AV_PIX_FMT_NONE];
        // SAFETY: each local readable array is terminated by AV_PIX_FMT_NONE
        // and remains live for the complete callback invocation.
        unsafe {
            assert_eq!(
                super::select_cuda_format(std::ptr::null_mut(), software_only.as_ptr()),
                AV_PIX_FMT_NONE
            );
            assert_eq!(
                super::select_cuda_format(std::ptr::null_mut(), hardware.as_ptr()),
                AV_PIX_FMT_CUDA
            );
        }
    }

    #[test]
    fn host_frame_is_rejected_by_cuda_surface_adapter() {
        let frame = ffmpeg_next::frame::Video::new(ffmpeg_next::format::Pixel::YUV420P, 96, 64);
        assert!(
            super::cuda_surface(&frame)
                .unwrap_err()
                .to_string()
                .contains("software frames are forbidden")
        );
    }
}

#[cfg(feature = "cuda")]
unsafe extern "C" fn select_cuda_format(
    _context: *mut ffmpeg::ffi::AVCodecContext,
    formats: *const ffmpeg::ffi::AVPixelFormat,
) -> ffmpeg::ffi::AVPixelFormat {
    use ffmpeg::ffi::AVPixelFormat::{AV_PIX_FMT_CUDA, AV_PIX_FMT_NONE};
    if formats.is_null() {
        return AV_PIX_FMT_NONE;
    }
    // SAFETY: libavcodec supplies a readable AV_PIX_FMT_NONE-terminated format
    // list for the duration of this callback. Return only a listed CUDA format
    // or NONE, so a failed hardware path can never silently choose software.
    unsafe {
        let mut pointer = formats;
        loop {
            let format = *pointer;
            if format == AV_PIX_FMT_NONE {
                return AV_PIX_FMT_NONE;
            }
            if format == AV_PIX_FMT_CUDA {
                return AV_PIX_FMT_CUDA;
            }
            pointer = pointer.add(1);
        }
    }
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy)]
pub(crate) struct CudaSurfaceDescription {
    pub context: u64,
    pub stream: u64,
    pub format: media_core::media::PixelFormat,
    pub allocated_width: u32,
    pub allocated_height: u32,
    pub addresses: [u64; 2],
    pub pitches: [usize; 2],
    pub row_bytes: [usize; 2],
    pub rows: [usize; 2],
    pub accounted_bytes: usize,
}

#[cfg(feature = "cuda")]
pub(crate) fn cuda_surface(frame: &ffmpeg::frame::Video) -> Result<CudaSurfaceDescription> {
    use ffmpeg::ffi;
    use media_core::media::PixelFormat;
    // SAFETY: the borrowed AVFrame retains its AVHWFramesContext, which retains
    // its AVHWDeviceContext. Format/type checks precede every CUDA-specific cast.
    // Device addresses are inspected as integers and never host-dereferenced.
    unsafe {
        let raw = &*frame.as_ptr();
        if raw.format != ffi::AVPixelFormat::AV_PIX_FMT_CUDA as i32 || raw.hw_frames_ctx.is_null() {
            return Err(Error::incompatible(
                "hardware-required decoding produced no CUDA surface; software frames are forbidden",
            ));
        }
        if (*raw.hw_frames_ctx).data.is_null() {
            return Err(Error::backend(
                "ffmpeg",
                "CUDA frame has no retained hardware frames context",
            ));
        }
        let frames = &*((*raw.hw_frames_ctx).data as *const ffi::AVHWFramesContext);
        if frames.format != ffi::AVPixelFormat::AV_PIX_FMT_CUDA || frames.device_ctx.is_null() {
            return Err(Error::backend(
                "ffmpeg",
                "CUDA frame context does not describe addressable CUDA surfaces",
            ));
        }
        let device = &*frames.device_ctx;
        if device.type_ != ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_CUDA || device.hwctx.is_null() {
            return Err(Error::backend("ffmpeg", "frame device is not CUDA"));
        }
        let cuda = &*(device.hwctx as *const ffi::AVCUDADeviceContext);
        let format = match frames.sw_format {
            ffi::AVPixelFormat::AV_PIX_FMT_NV12 => PixelFormat::Nv12,
            ffi::AVPixelFormat::AV_PIX_FMT_P010LE => PixelFormat::P010,
            other => {
                return Err(Error::unsupported(format!(
                    "CUDA surface format {other:?} requires a supported GPU conversion; no host transfer was selected"
                )));
            }
        };
        let allocated_width = u32::try_from(frames.width)
            .map_err(|_| Error::invalid("CUDA frame allocation width is invalid"))?;
        let allocated_height = u32::try_from(frames.height)
            .map_err(|_| Error::invalid("CUDA frame allocation height is invalid"))?;
        if raw.width <= 0
            || raw.height <= 0
            || allocated_width < raw.width as u32
            || allocated_height < raw.height as u32
        {
            return Err(Error::invalid(
                "CUDA visible dimensions exceed their hardware allocation",
            ));
        }
        let component = if format == PixelFormat::P010 {
            2usize
        } else {
            1
        };
        let width = allocated_width as usize;
        let row_bytes = [
            width
                .checked_mul(component)
                .ok_or_else(|| Error::overflow("CUDA luma row overflow"))?,
            width
                .div_ceil(2)
                .checked_mul(2)
                .and_then(|n| n.checked_mul(component))
                .ok_or_else(|| Error::overflow("CUDA chroma row overflow"))?,
        ];
        let rows = [
            allocated_height as usize,
            (allocated_height as usize).div_ceil(2),
        ];
        let mut pitches = [0usize; 2];
        let mut addresses = [0u64; 2];
        let mut accounted_bytes = 0usize;
        for index in 0..2 {
            if raw.linesize[index] <= 0 || raw.data[index].is_null() {
                return Err(Error::incompatible(
                    "CUDA surface has no positive pitched device plane",
                ));
            }
            pitches[index] = raw.linesize[index] as usize;
            addresses[index] = raw.data[index] as usize as u64;
            if pitches[index] < row_bytes[index] {
                return Err(Error::invalid(
                    "CUDA plane pitch cannot cover visible samples",
                ));
            }
            accounted_bytes = pitches[index]
                .checked_mul(rows[index])
                .and_then(|n| accounted_bytes.checked_add(n))
                .ok_or_else(|| Error::overflow("CUDA surface accounting overflow"))?;
        }
        let context = cuda.cuda_ctx as usize as u64;
        if context == 0 {
            return Err(Error::backend(
                "ffmpeg",
                "CUDA frame has a null native context",
            ));
        }
        Ok(CudaSurfaceDescription {
            context,
            stream: cuda.stream as usize as u64,
            format,
            allocated_width,
            allocated_height,
            addresses,
            pitches,
            row_bytes,
            rows,
            accounted_bytes,
        })
    }
}
pub fn parameter_description(
    parameters: &ffmpeg::codec::Parameters,
) -> Result<(Option<u32>, Option<u32>, Vec<u8>)> {
    // SAFETY: Parameters retains its AVCodecParameters owner. We only read it
    // and copy the validated configuration bytes while that owner is borrowed.
    unsafe {
        let p = &*parameters.as_ptr();
        if p.extradata_size < 0 || p.extradata_size > 4 * 1024 * 1024 {
            return Err(Error::invalid(
                "codec configuration exceeds the metadata limit",
            ));
        }
        let config = if p.extradata_size == 0 {
            Vec::new()
        } else {
            if p.extradata.is_null() {
                return Err(Error::invalid("codec configuration pointer is null"));
            }
            std::slice::from_raw_parts(p.extradata, p.extradata_size as usize).to_vec()
        };
        Ok((
            u32::try_from(p.width).ok().filter(|v| *v > 0),
            u32::try_from(p.height).ok().filter(|v| *v > 0),
            config,
        ))
    }
}
pub fn clear_codec_tag(stream: &mut ffmpeg::StreamMut<'_>) {
    // SAFETY: unique StreamMut access permits changing its codec parameters;
    // clearing a container-specific tag is required before cross-container mux.
    unsafe {
        (*stream.parameters().as_mut_ptr()).codec_tag = 0;
    }
}
pub fn check_mux_codec(path: &Path, codec: ffmpeg::codec::Id) -> Result<()> {
    let path = CString::new(
        path.to_str()
            .ok_or_else(|| Error::invalid("output path must be UTF-8"))?,
    )
    .map_err(|_| Error::invalid("output path contains a NUL byte"))?;
    // SAFETY: path is a valid terminated string; av_guess_format returns a
    // library-owned static descriptor and avformat_query_codec only reads it.
    unsafe {
        let format = ffmpeg::ffi::av_guess_format(ptr::null(), path.as_ptr(), ptr::null());
        if format.is_null() {
            return Err(Error::unsupported(
                "cannot infer output container from destination",
            ));
        }
        let supported = ffmpeg::ffi::avformat_query_codec(
            format,
            codec.into(),
            ffmpeg::ffi::FF_COMPLIANCE_NORMAL,
        );
        if supported <= 0 {
            return Err(Error::incompatible(format!(
                "output container does not establish support for codec {}",
                codec.name()
            )));
        }
    }
    Ok(())
}
pub fn flush_output(output: &mut ffmpeg::format::context::Output) -> Result<()> {
    // SAFETY: Output exclusively owns a live format context and pb belongs to
    // its retained custom StreamIo. Flushing is explicit and errors are checked.
    unsafe {
        let pb = (*output.as_mut_ptr()).pb;
        if pb.is_null() {
            return Err(Error::backend(
                "ffmpeg",
                "output has no writable I/O context",
            ));
        }
        ffmpeg::ffi::avio_flush(pb);
        if (*pb).error < 0 {
            return Err(crate::format::backend_error(ffmpeg::Error::from(
                (*pb).error,
            )));
        }
    }
    Ok(())
}
pub fn compiled_device_types() -> Vec<String> {
    let mut output = Vec::new();
    // SAFETY: av_hwdevice_iterate_types and name return library-owned enum/static
    // data; names are borrowed only during the call and copied into owned strings.
    unsafe {
        let mut kind = ffmpeg::ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_NONE;
        loop {
            kind = ffmpeg::ffi::av_hwdevice_iterate_types(kind);
            if kind == ffmpeg::ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_NONE {
                break;
            }
            let name = ffmpeg::ffi::av_hwdevice_get_type_name(kind);
            if !name.is_null() {
                output.push(CStr::from_ptr(name).to_string_lossy().into_owned());
            }
        }
    }
    output
}
pub fn copy_plane(
    frame: &ffmpeg::frame::Video,
    index: usize,
    row_bytes: usize,
    rows: usize,
) -> Result<Vec<u8>> {
    if index >= 8 || rows == 0 || row_bytes == 0 {
        return Err(Error::invalid("invalid frame plane geometry"));
    }
    let length = row_bytes
        .checked_mul(rows)
        .ok_or_else(|| Error::overflow("frame plane size overflow"))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(length)
        .map_err(|e| Error::exhausted(e.to_string()))?;
    // SAFETY: the borrowed AVFrame owns references to every source plane buffer.
    // We validate plane pitch, pointer and each row's range against the retained
    // AVBufferRef before constructing a byte slice, and copy before releasing it.
    unsafe {
        let raw = &*frame.as_ptr();
        let stride = raw.linesize[index];
        if stride <= 0 || (stride as usize) < row_bytes || raw.data[index].is_null() {
            return Err(Error::incompatible(
                "FFmpeg frame needs a nonnegative addressable host plane",
            ));
        }
        let buffer = ffmpeg::ffi::av_frame_get_plane_buffer(frame.as_ptr() as *mut _, index as i32);
        if buffer.is_null() {
            return Err(Error::incompatible("FFmpeg plane has no retained buffer"));
        }
        let start = (*buffer).data as usize;
        let end = start
            .checked_add((*buffer).size)
            .ok_or_else(|| Error::overflow("native buffer address overflow"))?;
        for row in 0..rows {
            let offset = (stride as usize)
                .checked_mul(row)
                .ok_or_else(|| Error::overflow("native row address overflow"))?;
            let pointer = (raw.data[index] as usize)
                .checked_add(offset)
                .ok_or_else(|| Error::overflow("native plane address overflow"))?;
            if pointer < start || pointer.checked_add(row_bytes).is_none_or(|last| last > end) {
                return Err(Error::incompatible(
                    "FFmpeg plane rows exceed their retained allocation",
                ));
            }
            bytes.extend_from_slice(std::slice::from_raw_parts(pointer as *const u8, row_bytes));
        }
    }
    Ok(bytes)
}

/// Run a borrowed CPU scaler with checked allocation and complete-row output.
pub(crate) fn scale_checked(
    scaler: &mut ffmpeg::software::scaling::Context,
    input: &ffmpeg::frame::Video,
) -> Result<ffmpeg::frame::Video> {
    let source = *scaler.input();
    let target = *scaler.output();
    if (input.format(), input.width(), input.height())
        != (source.format, source.width, source.height)
    {
        return Err(Error::incompatible(
            "CPU scaler source changed after negotiation",
        ));
    }
    let mut output = ffmpeg::frame::Video::empty();
    output.set_format(target.format);
    output.set_width(target.width);
    output.set_height(target.height);
    // SAFETY: uniquely owned fresh AVFrame with positive dimensions/formats
    // already validated during scaler construction. Errors precede sws_scale.
    let status = unsafe { ffmpeg::ffi::av_frame_get_buffer(output.as_mut_ptr(), 32) };
    if status < 0 {
        return Err(crate::format::backend_error(ffmpeg::Error::from(status)));
    }
    // SAFETY: native decoder and allocated output retain all source/target
    // planes for the complete synchronous call; no borrowed pointer escapes.
    let rows = unsafe {
        ffmpeg::ffi::sws_scale(
            scaler.as_mut_ptr(),
            (*input.as_ptr()).data.as_ptr() as *const *const u8,
            (*input.as_ptr()).linesize.as_ptr(),
            0,
            source.height as i32,
            (*output.as_mut_ptr()).data.as_ptr(),
            (*output.as_ptr()).linesize.as_ptr(),
        )
    };
    if rows < 0 {
        return Err(crate::format::backend_error(ffmpeg::Error::from(rows)));
    }
    if rows != target.height as i32 {
        return Err(Error::backend(
            "ffmpeg",
            "CPU scaler did not produce every output row",
        ));
    }
    Ok(output)
}
pub fn configure_scaler(
    scaler: &mut ffmpeg::software::scaling::Context,
    matrix: i32,
    source_range: i32,
) -> Result<()> {
    // SAFETY: scaler exclusively owns a live SwsContext. The coefficient table
    // is library-owned static data; sws_setColorspaceDetails copies its values.
    unsafe {
        let coefficients = ffmpeg::ffi::sws_getCoefficients(matrix);
        if coefficients.is_null() {
            return Err(Error::unsupported(
                "FFmpeg has no requested YUV color coefficients",
            ));
        }
        let result = ffmpeg::ffi::sws_setColorspaceDetails(
            scaler.as_mut_ptr(),
            coefficients,
            source_range,
            coefficients,
            1,
            0,
            1 << 16,
            1 << 16,
        );
        if result < 0 {
            return Err(crate::format::backend_error(ffmpeg::Error::from(result)));
        }
    }
    Ok(())
}

pub fn display_orientation(bytes: &[u8]) -> Result<media_core::media::Orientation> {
    use media_core::media::Orientation as O;
    if bytes.len() != 36 {
        return Err(Error::invalid("invalid FFmpeg display-matrix payload"));
    }
    let mut matrix = [0i32; 9];
    for (value, encoded) in matrix.iter_mut().zip(bytes.chunks_exact(4)) {
        *value = i32::from_ne_bytes(encoded.try_into().expect("four-byte chunk"));
    }
    if matrix[2] != 0 || matrix[5] != 0 || matrix[8] != 1 << 30 {
        return Err(Error::unsupported(
            "perspective display matrices need an explicit spatial transform",
        ));
    }
    let mut axis = [0; 4];
    for (out, index) in axis.iter_mut().zip([0, 1, 3, 4]) {
        let value = matrix[index];
        *out = if value.abs_diff(0) <= 1 {
            0
        } else if value.abs_diff(1 << 16) <= 1 {
            1
        } else if value.abs_diff(-(1 << 16)) <= 1 {
            -1
        } else {
            return Err(Error::unsupported(
                "nonorthogonal display matrices need an explicit spatial transform",
            ));
        };
    }
    // FFmpeg's row-vector convention maps p'=a*p+c*q and q'=b*p+d*q.
    match axis {
        [1, 0, 0, 1] => Ok(O::Normal),
        [-1, 0, 0, 1] => Ok(O::MirrorHorizontal),
        [-1, 0, 0, -1] => Ok(O::Rotate180),
        [1, 0, 0, -1] => Ok(O::MirrorVertical),
        [0, 1, 1, 0] => Ok(O::Transpose),
        [0, 1, -1, 0] => Ok(O::Rotate90),
        [0, -1, -1, 0] => Ok(O::Transverse),
        [0, -1, 1, 0] => Ok(O::Rotate270),
        _ => Err(Error::unsupported(
            "singular display matrix cannot define orientation",
        )),
    }
}
