//! Required native NVDEC-to-NVENC transcode with compressed audio passthrough.
//!
//! Decoded planes stay in CUDA memory. The encoder borrows the decoder's actual
//! hardware-frame pool and retains native frames through delayed packet output.
//! Optional native CUDA filters resize stored YUV components without upload,
//! download, color conversion, or bit-depth conversion. Codec-private surfaces and
//! driver reservations are not claimed as measured CUDA allocation accounting.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fs::OpenOptions;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;

use ffmpeg_next as ffmpeg;
use media_backend_cuda::{CudaContext, CudaLimits};
use media_core::device::{ContextId, Device};
use media_core::error::{Error, Result};
use media_core::media::{Orientation, PixelFormat, TrackId};
use media_core::time::{Rounding, Time, TimeBase};
use media_io::source::Source;

use crate::format::{self, ProbeOptions, TrackKind, backend_error};
use crate::native;

/// Explicit NVIDIA hardware encoder; software fallback is forbidden.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NvencCodec {
    /// H.264 with native eight-bit NV12 input.
    H264,
    /// HEVC Main or Main10, preserving native NV12 or P010 precision.
    Hevc,
}
impl NvencCodec {
    fn name(self) -> &'static str {
        match self {
            Self::H264 => "h264_nvenc",
            Self::Hevc => "hevc_nvenc",
        }
    }
    fn id(self) -> ffmpeg::codec::Id {
        match self {
            Self::H264 => ffmpeg::codec::Id::H264,
            Self::Hevc => ffmpeg::codec::Id::HEVC,
        }
    }
}

/// Reconstruction of stored YUV component planes, without linear-light filtering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaResizeFilter {
    /// Nearest component samples.
    Nearest,
    /// Bilinear interpolation of stored component code values.
    Bilinear,
    /// FFmpeg's native CUDA bicubic reconstruction.
    Bicubic,
    /// FFmpeg's native CUDA Lanczos reconstruction.
    Lanczos,
}
impl CudaResizeFilter {
    #[cfg(feature = "cuda-filters")]
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Nearest => "nearest",
            Self::Bilinear => "bilinear",
            Self::Bicubic => "bicubic",
            Self::Lanczos => "lanczos",
        }
    }
}

/// Native surface transcode selection and enforced application-side budgets.
#[derive(Debug, Clone)]
pub struct CudaTranscodeOptions {
    /// Selected non-cover video stream, or the first such stream.
    pub video_track: Option<TrackId>,
    /// Selected compressed audio streams; None copies all, Some(empty) none.
    pub audio_tracks: Option<Vec<TrackId>>,
    /// Required native encoder implementation.
    pub codec: NvencCodec,
    /// Requested geometry; both absent preserves the source dimensions.
    /// Different dimensions require the cuda-filters feature and scale_cuda.
    pub width: Option<u32>,
    /// Requested even height; no implicit crop or fit behavior.
    pub height: Option<u32>,
    /// Stored YUV component interpolation; preserves existing chroma location
    /// metadata and does not request chroma-grid rephasing or color conversion.
    pub resize_filter: CudaResizeFilter,
    /// Target encoder bitrate in bits per second.
    pub bit_rate: usize,
    /// NVENC speed/quality preset p1 through p7.
    pub preset: String,
    /// Maximum consecutive B frames, in 0..=4.
    pub max_b_frames: usize,
    /// Maximum key-frame interval.
    pub gop: u32,
    /// Maximum known pitched allocation size of one retained decoded surface.
    pub max_frame_bytes: usize,
    /// Maximum surfaces submitted but not yet matched to encoded packets.
    pub max_pending_frames: usize,
    /// Maximum summed known bytes in those retained surfaces.
    pub max_pending_bytes: usize,
    /// Maximum compressed audio bytes staged while establishing the first frame.
    pub max_preroll_bytes: usize,
    /// Maximum staged compressed audio packets, including zero-length packets.
    pub max_preroll_packets: usize,
    /// Header-only source discovery budget; no software pixel decode is used.
    pub probe: ProbeOptions,
}
impl Default for CudaTranscodeOptions {
    fn default() -> Self {
        Self {
            video_track: None,
            audio_tracks: None,
            codec: NvencCodec::H264,
            width: None,
            height: None,
            resize_filter: CudaResizeFilter::Bilinear,
            bit_rate: 2_000_000,
            preset: "p4".into(),
            max_b_frames: 2,
            gop: 60,
            max_frame_bytes: 256 * 1024 * 1024,
            max_pending_frames: 8,
            max_pending_bytes: 512 * 1024 * 1024,
            max_preroll_bytes: 8 * 1024 * 1024,
            max_preroll_packets: 4096,
            probe: ProbeOptions::default(),
        }
    }
}

/// Finalized native hardware transcode evidence and disclosed metadata loss.
#[derive(Debug, Clone)]
pub struct CudaTranscodeReport {
    /// Finalized new output file.
    pub destination: PathBuf,
    /// Encoded source video track.
    pub video_track: TrackId,
    /// Source audio tracks copied in output stream order.
    pub audio_tracks: Vec<TrackId>,
    /// Source tracks intentionally excluded from output.
    pub unselected_tracks: Vec<TrackId>,
    /// Established source common-clock origin; timestamps are not rebased.
    pub source_origin: Option<Time>,
    /// Actual source bytes read, including header discovery and rereads.
    pub source_bytes_read: u64,
    /// Actual hardware-decoded frames, including decoder drain.
    pub frames_decoded: u64,
    /// Native frames accepted by NVENC.
    pub frames_encoded: u64,
    /// Encoded video packets written, including encoder drain.
    pub video_packets_written: u64,
    /// New packet DTS reconstructed from source presentation input order.
    pub decode_timestamps_reconstructed: u64,
    /// Reconstructed DTS that differed from the native wrapper's packet DTS.
    pub decode_timestamps_adjusted: u64,
    /// Maximum B-frame presentation-ordinal delay used to derive new DTS.
    pub decode_reorder_bound: usize,
    /// Compressed audio packets written without audio decoding.
    pub audio_packets_written: u64,
    /// Previously unknown compressed packet durations established from codec
    /// configuration/sample count, without decoding audio or changing PTS/DTS.
    pub audio_durations_derived: u64,
    /// Actual output width after optional native component scaling.
    pub output_width: u32,
    /// Actual output height after optional native component scaling.
    pub output_height: u32,
    /// Actual native encoder selected.
    pub codec: NvencCodec,
    /// Actual retained native input/output precision.
    pub pixel_format: PixelFormat,
    /// Exact CUDA context used for both native codecs.
    pub device: Device,
    /// Peak known bytes of application-retained pending input surfaces.
    pub peak_pending_bytes: usize,
    /// Metadata not propagated to the new bitstream/container.
    pub metadata_loss: Vec<String>,
}

/// Decode directly to CUDA surfaces and encode them with NVENC into a new file.
///
/// Hardware is required. Unsupported native formats, HDR, interlacing, rotation,
/// and changing source metadata/geometry fail explicitly. Native resizing needs
/// the cuda-filters feature and linked scale_cuda filter. NV12 is
/// accepted for H.264/HEVC; P010 is accepted for HEVC Main10 without truncation.
/// Encoder negotiation precedes output creation and existing files are never
/// overwritten. Errors after creation identify a potentially partial output.
/// PTS/durations retain the source clock. New DTS use presentation input order
/// shifted by the configured B-frame reorder bound. Initial decode slots extend
/// backward by an established positive initial duration/gap; missing timing fails.
pub fn transcode(
    source: &Source,
    destination: &Path,
    ordinal: u32,
    limits: CudaLimits,
    options: CudaTranscodeOptions,
) -> Result<CudaTranscodeReport> {
    transcode_impl(source, destination, ordinal, limits, None, options)
}

/// Transcode on an application/session-owned actual CUDA context.
/// Native codecs share this handle rather than activating a different primary
/// context; the supplied owner remains retained through all native cleanup.
pub fn transcode_on_context(
    source: &Source,
    destination: &Path,
    context: &CudaContext,
    options: CudaTranscodeOptions,
) -> Result<CudaTranscodeReport> {
    let Device::Cuda { ordinal, .. } = context.device() else {
        return Err(Error::invalid("native transcode requires a CUDA context"));
    };
    transcode_impl(
        source,
        destination,
        ordinal,
        context.limits(),
        Some(context),
        options,
    )
}

fn transcode_impl(
    source: &Source,
    destination: &Path,
    ordinal: u32,
    limits: CudaLimits,
    supplied: Option<&CudaContext>,
    options: CudaTranscodeOptions,
) -> Result<CudaTranscodeReport> {
    validate_options(&options)?;
    limits.validate()?;
    let mut opened = format::open_source_without_decoding(source, options.probe)?;
    let info = format::snapshot(
        &opened.context,
        source,
        opened.counter.total.load(Ordering::Relaxed),
    )?;
    let video = info
        .tracks
        .iter()
        .find(|track| {
            track.kind == TrackKind::Video
                && !track.cover_art
                && options.video_track.is_none_or(|id| id == track.id)
        })
        .ok_or_else(|| Error::unsupported("source has no selected non-cover video track"))?;
    let video_track = video.id;
    if let Some(matrix) = &video.display_matrix
        && native::display_orientation(matrix)? != Orientation::Normal
    {
        return Err(Error::unsupported(
            "native transcode needs explicitly applied video orientation",
        ));
    }
    let audio_tracks = options.audio_tracks.clone().unwrap_or_else(|| {
        info.tracks
            .iter()
            .filter(|track| track.kind == TrackKind::Audio)
            .map(|track| track.id)
            .collect()
    });
    let mut unique = BTreeSet::new();
    for id in &audio_tracks {
        if !unique.insert(id.0)
            || !info
                .tracks
                .iter()
                .any(|t| t.id == *id && t.kind == TrackKind::Audio)
        {
            return Err(Error::invalid(
                "audio copy selection contains a duplicate or non-audio track",
            ));
        }
    }
    native::check_mux_codec(destination, options.codec.id())?;
    let mut audio_parameters = BTreeMap::new();
    for id in &audio_tracks {
        let parameters = opened
            .context
            .stream(id.0 as usize)
            .ok_or_else(|| Error::invalid("audio track disappeared"))?
            .parameters();
        native::check_mux_codec(destination, parameters.id())?;
        audio_parameters.insert(id.0, resolve_audio_header(parameters)?);
    }
    let codec = ffmpeg::encoder::find_by_name(options.codec.name()).ok_or_else(|| {
        Error::unsupported(format!("linked FFmpeg lacks {}", options.codec.name()))
    })?;
    let input = opened
        .context
        .stream(video_track.0 as usize)
        .ok_or_else(|| Error::invalid("video track disappeared"))?;
    let input_base = input.time_base();
    let average_rate = input.avg_frame_rate();
    let mut decoder_context = ffmpeg::codec::context::Context::from_parameters(input.parameters())
        .map_err(backend_error)?;
    decoder_context.set_threading(ffmpeg::threading::Config::count(1));
    // Source factories/probe callbacks may have changed the calling thread's
    // current CUDA context. Rebind immediately before native current_ctx setup.
    if let Some(context) = supplied {
        context
            .native_context()
            .bind_to_thread()
            .map_err(|error| Error::backend("cuda", error.to_string()).with_cause(error))?;
    }
    let native_device = native::configure_cuda_decoder(
        &mut decoder_context,
        ordinal,
        supplied.is_some(),
        decoder_headroom(options.max_pending_frames),
    )?;
    let context = match supplied {
        Some(context) => context.clone(),
        None => CudaContext::new(ordinal, limits)?,
    };
    if context.device()
        != (Device::Cuda {
            ordinal,
            context: ContextId::new(native_device.context)?,
        })
    {
        return Err(Error::incompatible(
            "decoder and encoder adapter require the same actual CUDA context",
        ));
    }
    let mut decoder_wrapper = decoder_context.decoder();
    decoder_wrapper.set_packet_time_base(input_base);
    let mut decoder = decoder_wrapper.video().map_err(backend_error)?;
    opened.allow_reading();
    let mut staged = Vec::new();
    let mut staged_bytes = 0usize;
    let mut decoder_draining = false;
    let first = loop {
        let mut frame = ffmpeg::frame::Video::empty();
        match decoder.receive_frame(&mut frame) {
            Ok(()) => break frame,
            Err(ffmpeg::Error::Eof) => {
                return Err(Error::unsupported(
                    "input video contains no decodable hardware frames",
                ));
            }
            Err(error) if is_again(error) && !decoder_draining => {}
            Err(error) => {
                return Err(
                    backend_error(error).with_operation("establish first native hardware frame")
                );
            }
        }
        let mut packet = ffmpeg::Packet::empty();
        match packet.read(&mut opened.context) {
            Ok(()) if packet.stream() == video_track.0 as usize => {
                decoder.send_packet(&packet).map_err(backend_error)?
            }
            Ok(()) if unique.contains(&(packet.stream() as u32)) => {
                stage_audio(&mut staged_bytes, staged.len(), packet.size(), &options)?;
                staged.push(packet);
            }
            Ok(()) => {}
            Err(ffmpeg::Error::Eof) => {
                decoder.send_eof().map_err(backend_error)?;
                decoder_draining = true;
            }
            Err(error) => {
                return Err(backend_error(error).with_operation("native transcode preroll demux"));
            }
        }
    };
    let source_surface = validate_frame(&first, &context, &options)?;
    // NVENC writes codec timing/VUI from this hint. A VFR average describes the
    // whole recording and must not replace the source codec's nominal interval.
    let rate = nominal_rate(decoder.frame_rate())?;
    let source_geometry = (first.width(), first.height(), source_surface.format);
    let source_color = Color::from_frame(&first);
    let mut filter = SurfaceFilter::new(&first, &context, input_base, rate, &options)?;
    let first = filter.run(first)?;
    let surface = validate_frame(&first, &context, &options)?;
    let width = first.width();
    let height = first.height();
    let mut encoder = ffmpeg::codec::context::Context::new_with_codec(codec)
        .encoder()
        .video()
        .map_err(backend_error)?;
    encoder.set_width(width);
    encoder.set_height(height);
    encoder.set_format(ffmpeg::format::Pixel::CUDA);
    encoder.set_time_base(input_base);
    encoder.set_frame_rate(Some(rate));
    encoder.set_aspect_ratio(first.aspect_ratio());
    encoder.set_bit_rate(options.bit_rate);
    encoder.set_max_b_frames(options.max_b_frames);
    encoder.set_gop(options.gop);
    encoder.set_colorspace(first.color_space());
    encoder.set_color_range(first.color_range());
    encoder.set_color_primaries(first.color_primaries());
    encoder.set_color_transfer_characteristic(first.color_transfer_characteristic());
    ffi_helpers::attach_frames(&mut encoder, &first)?;
    if ffi_helpers::requires_global_header(destination)? {
        encoder.set_flags(ffmpeg::codec::Flags::GLOBAL_HEADER);
    }
    let mut settings = ffmpeg::Dictionary::new();
    settings.set("preset", &options.preset);
    settings.set("rc-lookahead", "0");
    settings.set("delay", "0");
    settings.set(
        "profile",
        if surface.format == PixelFormat::P010 {
            "main10"
        } else if options.codec == NvencCodec::H264 {
            "high"
        } else {
            "main"
        },
    );
    let encoder = encoder
        .open_with(settings)
        .map_err(|e| backend_error(e).with_operation("open required NVENC encoder"))?;
    if format::time_base(encoder.time_base())? != format::time_base(input_base)? {
        return Err(Error::incompatible(
            "NVENC changed the source video time base",
        ));
    }
    let name = destination
        .to_str()
        .ok_or_else(|| Error::invalid("output path must be UTF-8"))?;
    let file = OpenOptions::new()
        .create_new(true)
        .read(true)
        .write(true)
        .open(destination)
        .map_err(Error::io)?;
    let sync = file.try_clone().map_err(Error::io)?;
    let output = ffmpeg::format::output_to_stream(
        ffmpeg::format::context::StreamIo::from_write_seek(file).map_err(backend_error)?,
        Some(name),
        None,
    );
    let mut output = output.map_err(|e| {
        backend_error(e)
            .with_source(name)
            .with_operation("native transcode; partial destination may exist")
    })?;
    let clock = DecodeClock::new(options.max_b_frames);
    let mut pipeline = Pipeline {
        encoder,
        decoder,
        pending: BTreeMap::new(),
        context,
        input_base: format::time_base(input_base)?,
        video_output_base: format::time_base(input_base)?,
        audio_mapping: BTreeMap::new(),
        options,
        width,
        height,
        format: surface.format,
        color: Color::from_frame(&first),
        source_geometry,
        source_color,
        filter,
        clock,
        pending_bytes: 0,
        peak_pending_bytes: 0,
        frames_decoded: 1,
        frames_encoded: 0,
        video_packets_written: 0,
        dts_adjustments: 0,
        audio_packets_written: 0,
        audio_durations_derived: 0,
        metadata_loss: BTreeSet::new(),
    };
    let result = (|| -> Result<()> {
        {
            let input = opened
                .context
                .stream(video_track.0 as usize)
                .ok_or_else(|| Error::invalid("video track disappeared"))?;
            let mut stream = output.add_stream(codec).map_err(backend_error)?;
            stream.set_parameters(&pipeline.encoder);
            stream.set_time_base(input_base);
            stream.set_avg_frame_rate(average_rate);
            stream.set_metadata(input.metadata().to_owned());
            ffi_helpers::copy_disposition(&input, &mut stream);
            native::clear_codec_tag(&mut stream);
        }
        for (index, id) in audio_tracks.iter().enumerate() {
            let input = opened
                .context
                .stream(id.0 as usize)
                .ok_or_else(|| Error::invalid("audio track disappeared"))?;
            let mut stream = output
                .add_stream(ffmpeg::encoder::find(ffmpeg::codec::Id::None))
                .map_err(backend_error)?;
            stream.set_parameters(
                audio_parameters
                    .get(&id.0)
                    .ok_or_else(|| Error::invalid("resolved audio parameters missing"))?
                    .clone(),
            );
            stream.set_time_base(input.time_base());
            stream.set_metadata(input.metadata().to_owned());
            ffi_helpers::copy_disposition(&input, &mut stream);
            native::clear_codec_tag(&mut stream);
            pipeline.audio_mapping.insert(
                id.0 as usize,
                (index + 1, format::time_base(input.time_base())?),
            );
        }
        output.set_metadata(opened.context.metadata().to_owned());
        if opened.context.chapters().next().is_some() {
            pipeline
                .metadata_loss
                .insert("container chapters were not propagated".into());
        }
        for data in opened
            .context
            .stream(video_track.0 as usize)
            .ok_or_else(|| Error::invalid("video track disappeared"))?
            .side_data()
        {
            pipeline.metadata_loss.insert(format!(
                "video stream side data {:?} was not propagated",
                data.kind()
            ));
        }
        ffi_helpers::configure_mux(&mut output);
        output.write_header().map_err(backend_error)?;
        pipeline.video_output_base = format::time_base(
            output
                .stream(0)
                .ok_or_else(|| Error::invalid("output video stream missing"))?
                .time_base(),
        )?;
        pipeline.submit_ready(first, &mut output)?;
        for packet in staged {
            pipeline.copy_audio(packet, &mut output)?;
        }
        if !decoder_draining {
            // The packet producing the first frame can leave further frames
            // buffered. Receive them before sending another compressed packet.
            pipeline.receive_frames(&mut output, false)?;
            loop {
                let mut packet = ffmpeg::Packet::empty();
                match packet.read(&mut opened.context) {
                    Ok(()) => {}
                    Err(ffmpeg::Error::Eof) => break,
                    Err(error) => {
                        return Err(backend_error(error).with_operation("native transcode demux"));
                    }
                }
                if packet.stream() == video_track.0 as usize {
                    pipeline
                        .decoder
                        .send_packet(&packet)
                        .map_err(backend_error)?;
                    pipeline.receive_frames(&mut output, false)?;
                } else if pipeline.audio_mapping.contains_key(&packet.stream()) {
                    pipeline.copy_audio(packet, &mut output)?;
                }
            }
            pipeline.decoder.send_eof().map_err(backend_error)?;
        }
        pipeline.receive_frames(&mut output, true)?;
        pipeline.encoder.send_eof().map_err(backend_error)?;
        pipeline.receive_packets(&mut output, true)?;
        if !pipeline.pending.is_empty() || pipeline.frames_decoded != pipeline.frames_encoded {
            return Err(Error::backend(
                "ffmpeg",
                "NVENC drain did not account for every submitted frame",
            ));
        }
        output.write_trailer().map_err(backend_error)?;
        native::flush_output(&mut output)?;
        Ok(())
    })();
    drop(output);
    result.map_err(|e| {
        e.with_source(name)
            .with_operation("native transcode; partial destination may exist")
    })?;
    sync.sync_all().map_err(Error::io)?;
    Ok(CudaTranscodeReport {
        destination: destination.to_owned(),
        video_track,
        audio_tracks: audio_tracks.clone(),
        unselected_tracks: info
            .tracks
            .iter()
            .filter(|t| t.id != video_track && !audio_tracks.contains(&t.id))
            .map(|t| t.id)
            .collect(),
        source_origin: info.timeline_origin,
        source_bytes_read: opened.counter.total.load(Ordering::Relaxed),
        frames_decoded: pipeline.frames_decoded,
        frames_encoded: pipeline.frames_encoded,
        video_packets_written: pipeline.video_packets_written,
        decode_timestamps_reconstructed: pipeline.video_packets_written,
        decode_timestamps_adjusted: pipeline.dts_adjustments,
        decode_reorder_bound: pipeline.options.max_b_frames,
        audio_packets_written: pipeline.audio_packets_written,
        audio_durations_derived: pipeline.audio_durations_derived,
        output_width: width,
        output_height: height,
        codec: pipeline.options.codec,
        pixel_format: pipeline.format,
        device: pipeline.context.device(),
        peak_pending_bytes: pipeline.peak_pending_bytes,
        metadata_loss: pipeline.metadata_loss.into_iter().collect(),
    })
}

#[derive(Debug, PartialEq, Eq)]
struct Color {
    matrix: ffmpeg::color::Space,
    range: ffmpeg::color::Range,
    primaries: ffmpeg::color::Primaries,
    transfer: ffmpeg::color::TransferCharacteristic,
    chroma: ffmpeg::chroma::Location,
    aspect: ffmpeg::Rational,
}
impl Color {
    fn from_frame(f: &ffmpeg::frame::Video) -> Self {
        Self {
            matrix: f.color_space(),
            range: f.color_range(),
            primaries: f.color_primaries(),
            transfer: f.color_transfer_characteristic(),
            chroma: f.chroma_location(),
            aspect: f.aspect_ratio(),
        }
    }
}
struct PendingFrame {
    duration: i64,
    bytes: usize,
    // Hold native references until output; Video::clone would copy pixels.
    _frame: ffmpeg::frame::Video,
}
enum SurfaceFilter {
    Identity,
    #[cfg(feature = "cuda-filters")]
    Scale(crate::cuda_scale::CudaScaler),
}
impl SurfaceFilter {
    fn new(
        frame: &ffmpeg::frame::Video,
        _context: &CudaContext,
        _base: ffmpeg::Rational,
        _rate: ffmpeg::Rational,
        options: &CudaTranscodeOptions,
    ) -> Result<Self> {
        let (width, height) = (
            options.width.unwrap_or(frame.width()),
            options.height.unwrap_or(frame.height()),
        );
        if (width, height) == (frame.width(), frame.height()) {
            return Ok(Self::Identity);
        }
        #[cfg(feature = "cuda-filters")]
        {
            Ok(Self::Scale(crate::cuda_scale::CudaScaler::new(
                frame,
                _context,
                _base,
                _rate,
                width,
                height,
                options.resize_filter,
            )?))
        }
        #[cfg(not(feature = "cuda-filters"))]
        {
            Err(Error::unsupported(
                "native resize requires the cuda-filters feature and linked scale_cuda filter; CPU resize/upload is forbidden",
            ))
        }
    }
    fn run(&mut self, input: ffmpeg::frame::Video) -> Result<ffmpeg::frame::Video> {
        match self {
            Self::Identity => Ok(input),
            #[cfg(feature = "cuda-filters")]
            Self::Scale(filter) => filter.run(input),
        }
    }
}
struct Pipeline {
    // Native codecs and their retained frames must die before the CUDA owner.
    encoder: ffmpeg::encoder::Video,
    decoder: ffmpeg::decoder::Video,
    pending: BTreeMap<i64, PendingFrame>,
    // The filter's own retained context also outlives its graph's native frames.
    context: CudaContext,
    input_base: TimeBase,
    video_output_base: TimeBase,
    audio_mapping: BTreeMap<usize, (usize, TimeBase)>,
    options: CudaTranscodeOptions,
    width: u32,
    height: u32,
    format: PixelFormat,
    color: Color,
    source_geometry: (u32, u32, PixelFormat),
    source_color: Color,
    filter: SurfaceFilter,
    clock: DecodeClock,
    pending_bytes: usize,
    peak_pending_bytes: usize,
    frames_decoded: u64,
    frames_encoded: u64,
    video_packets_written: u64,
    dts_adjustments: u64,
    audio_packets_written: u64,
    audio_durations_derived: u64,
    metadata_loss: BTreeSet<String>,
}
impl Pipeline {
    fn receive_frames(
        &mut self,
        output: &mut ffmpeg::format::context::Output,
        draining: bool,
    ) -> Result<()> {
        loop {
            let mut frame = ffmpeg::frame::Video::empty();
            match self.decoder.receive_frame(&mut frame) {
                Ok(()) => {
                    increment(&mut self.frames_decoded)?;
                    let surface = validate_frame(&frame, &self.context, &self.options)?;
                    if (frame.width(), frame.height(), surface.format) != self.source_geometry
                        || Color::from_frame(&frame) != self.source_color
                    {
                        return Err(Error::unsupported(
                            "native source geometry, precision, color, or aspect changes require explicit renegotiation",
                        ));
                    }
                    let frame = self.filter.run(frame)?;
                    self.submit_ready(frame, output)?;
                }
                Err(ffmpeg::Error::Eof) if draining => return Ok(()),
                Err(error) if is_again(error) && !draining => return Ok(()),
                Err(error) => {
                    return Err(
                        backend_error(error).with_operation("required native decoder receive")
                    );
                }
            }
        }
    }
    fn submit_ready(
        &mut self,
        mut frame: ffmpeg::frame::Video,
        output: &mut ffmpeg::format::context::Output,
    ) -> Result<()> {
        let surface = validate_frame(&frame, &self.context, &self.options)?;
        if (frame.width(), frame.height(), surface.format) != (self.width, self.height, self.format)
            || Color::from_frame(&frame) != self.color
        {
            return Err(Error::unsupported(
                "native source geometry, precision, color, or aspect changes require explicit renegotiation",
            ));
        }
        let pts = frame.pts().ok_or_else(|| {
            Error::unsupported(
                "native encode requires original frame PTS; timestamps are not fabricated",
            )
        })?;
        if self.pending.contains_key(&pts) {
            return Err(Error::incompatible(
                "duplicate source frame PTS is ambiguous through NVENC",
            ));
        }
        let next = self
            .pending_bytes
            .checked_add(surface.accounted_bytes)
            .ok_or_else(|| Error::overflow("pending surface byte accounting overflow"))?;
        if self.pending.len() >= self.options.max_pending_frames
            || next > self.options.max_pending_bytes
        {
            return Err(
                Error::exhausted("NVENC pending native surface budget exceeded")
                    .with_constraint("max_pending_frames/max_pending_bytes"),
            );
        }
        // Wait on the actual producer stream event, including the valid NULL
        // legacy stream. This waits for readiness without copying any pixels.
        producer_ready(&self.context, surface.stream)?;
        self.metadata_loss
            .extend(ffi_helpers::side_data_loss(&frame));
        frame.set_kind(ffmpeg::picture::Type::None);
        self.clock.push(pts, frame.packet().duration)?;
        self.encoder
            .send_frame(&frame)
            .map_err(|e| backend_error(e).with_operation("submit native CUDA surface to NVENC"))?;
        self.pending.insert(
            pts,
            PendingFrame {
                duration: frame.packet().duration.max(0),
                bytes: surface.accounted_bytes,
                _frame: frame,
            },
        );
        self.pending_bytes = next;
        self.peak_pending_bytes = self.peak_pending_bytes.max(next);
        increment(&mut self.frames_encoded)?;
        self.receive_packets(output, false)
    }
    fn receive_packets(
        &mut self,
        output: &mut ffmpeg::format::context::Output,
        draining: bool,
    ) -> Result<()> {
        loop {
            let mut packet = ffmpeg::Packet::empty();
            match self.encoder.receive_packet(&mut packet) {
                Ok(()) => {}
                Err(ffmpeg::Error::Eof) if draining => return Ok(()),
                Err(error) if is_again(error) && !draining => return Ok(()),
                Err(error) => {
                    return Err(backend_error(error).with_operation("drain native NVENC packet"));
                }
            }
            let pts = packet
                .pts()
                .ok_or_else(|| Error::backend("ffmpeg", "NVENC packet has no source PTS"))?;
            let pending = self.pending.remove(&pts).ok_or_else(|| {
                Error::backend(
                    "ffmpeg",
                    "NVENC packet PTS does not match a submitted native frame",
                )
            })?;
            let dts = self.clock.next(pts)?;
            if packet.dts() != Some(dts) {
                increment(&mut self.dts_adjustments)?;
            }
            packet.set_dts(Some(dts));
            self.pending_bytes -= pending.bytes;
            if pending.duration > 0 {
                packet.set_duration(pending.duration);
            }
            rescale_packet(&mut packet, self.input_base, self.video_output_base)?;
            packet.set_stream(0);
            packet.set_position(-1);
            packet.write_interleaved(output).map_err(backend_error)?;
            increment(&mut self.video_packets_written)?;
        }
    }
    fn copy_audio(
        &mut self,
        mut packet: ffmpeg::Packet,
        output: &mut ffmpeg::format::context::Output,
    ) -> Result<()> {
        let (index, base) = self
            .audio_mapping
            .get(&packet.stream())
            .copied()
            .ok_or_else(|| Error::invalid("unselected audio packet"))?;
        if packet.pts().is_none() || packet.dts().is_none() {
            return Err(Error::unsupported(
                "audio copy requires original PTS/DTS; timestamp fabrication is forbidden",
            ));
        }
        let parameters = output
            .stream(index)
            .ok_or_else(|| Error::invalid("output audio stream missing"))?
            .parameters();
        if establish_packet_duration(&mut packet, &parameters, base)? {
            increment(&mut self.audio_durations_derived)?;
        }
        let target = format::time_base(
            output
                .stream(index)
                .ok_or_else(|| Error::invalid("output audio stream missing"))?
                .time_base(),
        )?;
        rescale_packet(&mut packet, base, target)?;
        packet.set_stream(index);
        packet.set_position(-1);
        packet.write_interleaved(output).map_err(backend_error)?;
        increment(&mut self.audio_packets_written)
    }
}

// A presentation-ordinal delay handles VFR gaps; constant-time offsets do not.
// Queue size is bounded by pending submitted frames plus the reorder bound.
struct DecodeClock {
    delay: usize,
    inputs: VecDeque<i64>,
    first: Option<i64>,
    last_input: Option<i64>,
    startup_step: Option<i64>,
    emitted: usize,
    last_dts: Option<i64>,
}
impl DecodeClock {
    fn new(delay: usize) -> Self {
        Self {
            delay,
            inputs: VecDeque::new(),
            first: None,
            last_input: None,
            startup_step: None,
            emitted: 0,
            last_dts: None,
        }
    }
    fn push(&mut self, pts: i64, duration: i64) -> Result<()> {
        if self.last_input.is_some_and(|previous| pts <= previous) {
            return Err(Error::incompatible(
                "native VFR encoding requires strictly increasing presentation input PTS",
            ));
        }
        if let Some(first) = self.first {
            if self.startup_step.is_none() && self.emitted == 0 {
                self.startup_step = Some(
                    pts.checked_sub(first)
                        .ok_or_else(|| Error::overflow("initial video timestamp gap overflow"))?,
                );
            }
        } else {
            self.first = Some(pts);
            if duration > 0 {
                self.startup_step = Some(duration);
            }
        }
        self.last_input = Some(pts);
        self.inputs.push_back(pts);
        Ok(())
    }
    fn next(&mut self, packet_pts: i64) -> Result<i64> {
        let dts = if self.emitted < self.delay {
            let first = self
                .first
                .ok_or_else(|| Error::backend("ffmpeg", "NVENC output arrived before any input"))?;
            let step = self.startup_step.ok_or_else(|| Error::unsupported("initial reordered encode DTS requires an established positive frame duration or initial PTS gap"))?;
            let offset = step
                .checked_mul((self.delay - self.emitted) as i64)
                .ok_or_else(|| Error::overflow("initial native encode reorder offset overflow"))?;
            first
                .checked_sub(offset)
                .ok_or_else(|| Error::overflow("initial native encode DTS overflow"))?
        } else {
            self.inputs.pop_front().ok_or_else(|| {
                Error::backend(
                    "ffmpeg",
                    "NVENC produced more packets than submitted input frames",
                )
            })?
        };
        if dts > packet_pts || self.last_dts.is_some_and(|previous| dts <= previous) {
            return Err(Error::incompatible(
                "NVENC output exceeds the declared B-frame reorder depth or cannot form strictly ordered source-clock DTS",
            ));
        }
        self.last_dts = Some(dts);
        self.emitted = self
            .emitted
            .checked_add(1)
            .ok_or_else(|| Error::overflow("native decode timestamp ordinal overflow"))?;
        Ok(dts)
    }
}

fn validate_options(o: &CudaTranscodeOptions) -> Result<()> {
    if o.bit_rate == 0
        || o.bit_rate > i64::MAX as usize
        || o.max_b_frames > 4
        || o.gop == 0
        || o.gop > i32::MAX as u32
        || !(1..=64).contains(&o.max_pending_frames)
        || o.max_frame_bytes == 0
        || o.max_pending_bytes == 0
        || o.max_preroll_bytes == 0
        || o.max_preroll_packets == 0
        || o.width.is_some() != o.height.is_some()
        || !["p1", "p2", "p3", "p4", "p5", "p6", "p7"].contains(&o.preset.as_str())
    {
        return Err(Error::invalid(
            "invalid native encode geometry, bitrate, preset, GOP, or allocation budget",
        ));
    }
    if let (Some(w), Some(h)) = (o.width, o.height) {
        validate_geometry(w, h)?;
    }
    Ok(())
}
// Application-retained input frames and NVDEC's native decode/reference pool
// are different budgets. Adding an arbitrary pending-frame limit (e.g. 32) to
// the codec's own reference/reorder surface count can exceed NVDEC's 32-surface
// device limit before the first frame. Reserve a bounded eight-surface headroom
// for this no-lookahead, low-delay encoder path; never turn an application queue
// limit into an equally large native decoder reference reservation.
fn decoder_headroom(max_pending_frames: usize) -> usize {
    max_pending_frames.min(8)
}
fn nominal_rate(codec: Option<ffmpeg::Rational>) -> Result<ffmpeg::Rational> {
    codec.filter(|rate| rate.numerator() > 0 && rate.denominator() > 0)
        .ok_or_else(|| Error::unsupported("native encoding requires an established source codec nominal frame rate; VFR averages and guessed demux rates are not substitutes"))
}

// Header-only discovery deliberately forbids probing through decoded samples.
// Some containers leave AAC's channel layout to AudioSpecificConfig. Opening
// its decoder parses that codec configuration; no compressed packet is sent
// and no audio sample frame is received. Formats needing sample decode for
// missing configuration fail explicitly instead of changing passthrough policy.
fn resolve_audio_header(
    mut parameters: ffmpeg::codec::Parameters,
) -> Result<ffmpeg::codec::Parameters> {
    if ffi_helpers::audio_header_complete(&parameters) {
        ffi_helpers::establish_audio_frame_size(&mut parameters)?;
        return Ok(parameters);
    }
    let decoder = ffmpeg::codec::context::Context::from_parameters(parameters.clone())
        .map_err(backend_error)?
        .decoder()
        .audio()
        .map_err(|e| {
            backend_error(e).with_operation(
                "parse compressed audio codec configuration without sample decoding",
            )
        })?;
    if decoder.channels() == 0 || decoder.rate() == 0 {
        return Err(Error::unsupported(format!(
            "audio stream copy requires established channel/rate configuration without decoding samples (configuration returned {} channels at {} Hz)",
            decoder.channels(),
            decoder.rate()
        )));
    }
    ffi_helpers::fill_audio_header(&mut parameters, &decoder)?;
    ffi_helpers::establish_audio_frame_size(&mut parameters)?;
    Ok(parameters)
}
// AAC-LC GASpecificConfig establishes coded samples without audio decoding.
// Unsupported object/PCE/rate configurations remain unknown, never guessed.
fn aac_lc_frame_samples(configuration: &[u8], output_rate: u32) -> Option<u32> {
    struct Bits<'a> {
        data: &'a [u8],
        position: usize,
    }
    impl Bits<'_> {
        fn read(&mut self, count: usize) -> Option<u32> {
            let mut value = 0;
            for _ in 0..count {
                value = (value << 1)
                    | u32::from((self.data.get(self.position / 8)? >> (7 - self.position % 8)) & 1);
                self.position += 1;
            }
            Some(value)
        }
    }
    let mut bits = Bits {
        data: configuration,
        position: 0,
    };
    if bits.read(5)? != 2 {
        return None;
    }
    let frequency = bits.read(4)?;
    let rate = if frequency == 15 {
        bits.read(24)?
    } else {
        *[
            96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000, 7350,
        ]
        .get(frequency as usize)?
    };
    if rate != output_rate || bits.read(4)? == 0 {
        return None;
    }
    let samples = if bits.read(1)? == 0 { 1024 } else { 960 };
    // Core-coder dependencies/extensions need additional interpretation.
    if bits.read(1)? != 0 || bits.read(1)? != 0 {
        return None;
    }
    Some(samples)
}
fn validate_geometry(w: u32, h: u32) -> Result<()> {
    if w == 0
        || h == 0
        || w > i32::MAX as u32
        || h > i32::MAX as u32
        || !w.is_multiple_of(2)
        || !h.is_multiple_of(2)
    {
        return Err(Error::invalid(
            "native YUV420 encoding needs nonzero even dimensions within native limits",
        ));
    }
    Ok(())
}
fn validate_frame(
    frame: &ffmpeg::frame::Video,
    context: &CudaContext,
    options: &CudaTranscodeOptions,
) -> Result<native::CudaSurfaceDescription> {
    validate_integrity(frame)?;
    validate_geometry(frame.width(), frame.height())?;
    if frame.is_interlaced() {
        return Err(Error::unsupported(
            "native encode requires explicit deinterlacing before submission",
        ));
    }
    use ffmpeg::color::TransferCharacteristic as Transfer;
    if matches!(
        frame.color_transfer_characteristic(),
        Transfer::SMPTE2084 | Transfer::ARIB_STD_B67
    ) {
        return Err(Error::unsupported(
            "native HDR metadata preservation needs an explicit HDR encoding policy",
        ));
    }
    let surface = native::cuda_surface(frame)?;
    if !matches!(context.device(), Device::Cuda { context: id, .. } if id.get() == surface.context)
    {
        return Err(Error::incompatible(
            "native decoder surface belongs to a different actual CUDA context",
        ));
    }
    if surface.format == PixelFormat::P010 && options.codec == NvencCodec::H264 {
        return Err(Error::unsupported(
            "P010 requires HEVC Main10; implicit eight-bit truncation is forbidden",
        ));
    }
    if surface.accounted_bytes > options.max_frame_bytes {
        return Err(
            Error::exhausted("native decoded surface exceeds max_frame_bytes")
                .with_constraint("max_frame_bytes"),
        );
    }
    Ok(surface)
}
fn validate_integrity(frame: &ffmpeg::frame::Video) -> Result<()> {
    if frame.is_corrupt() || frame.has_decode_errors() {
        return Err(Error::backend(
            "ffmpeg",
            "required native transcode decoder returned corrupt or incomplete output",
        )
        .with_operation("validate native decoded frame before encoding"));
    }
    Ok(())
}
fn stage_audio(
    bytes: &mut usize,
    packets: usize,
    size: usize,
    options: &CudaTranscodeOptions,
) -> Result<()> {
    let next = bytes
        .checked_add(size)
        .ok_or_else(|| Error::overflow("compressed audio preroll size overflow"))?;
    if next > options.max_preroll_bytes || packets >= options.max_preroll_packets {
        return Err(
            Error::exhausted("compressed audio preroll exceeds its explicit budget")
                .with_constraint("max_preroll_bytes/max_preroll_packets"),
        );
    }
    *bytes = next;
    Ok(())
}
#[allow(unsafe_code)]
pub(crate) fn producer_ready(context: &CudaContext, stream: u64) -> Result<()> {
    // SAFETY: cuda_surface validated this stream's exact owning CUDA context;
    // the frame and its native device owner remain borrowed through this wait.
    unsafe { context.record_external_completion(stream)?.wait() }
}
fn establish_packet_duration(
    packet: &mut ffmpeg::Packet,
    parameters: &ffmpeg::codec::Parameters,
    base: TimeBase,
) -> Result<bool> {
    if packet.duration() > 0 {
        return Ok(false);
    }
    if packet.duration() < 0 || packet.size() == 0 {
        return Err(Error::unsupported(
            "audio copy requires a nonempty compressed packet and established nonnegative timing",
        ));
    }
    let (samples, rate) = ffi_helpers::coded_audio_samples(parameters, packet.size())?;
    let duration = Time::from_samples(samples, rate)?;
    let ticks = duration.to_ticks(base, Rounding::NearestTiesEven)?;
    if ticks <= 0 || base.time(ticks)? != duration {
        return Err(Error::incompatible(
            "source time base cannot represent the codec-established audio packet duration exactly",
        ));
    }
    packet.set_duration(ticks);
    Ok(true)
}
fn rescale_packet(packet: &mut ffmpeg::Packet, source: TimeBase, target: TimeBase) -> Result<()> {
    fn exact(value: i64, source: TimeBase, target: TimeBase) -> Result<i64> {
        let mapped = source.rescale(value, target, Rounding::NearestTiesEven)?;
        if i128::from(value) * i128::from(source.numerator()) * i128::from(target.denominator())
            != i128::from(mapped)
                * i128::from(target.numerator())
                * i128::from(source.denominator())
        {
            return Err(Error::incompatible(
                "muxer time base cannot represent an original packet timestamp exactly",
            ));
        }
        Ok(mapped)
    }
    packet.set_pts(packet.pts().map(|v| exact(v, source, target)).transpose()?);
    packet.set_dts(packet.dts().map(|v| exact(v, source, target)).transpose()?);
    if packet.duration() > 0 {
        packet.set_duration(exact(packet.duration(), source, target)?);
    }
    Ok(())
}
fn is_again(error: ffmpeg::Error) -> bool {
    matches!(error, ffmpeg::Error::Other { errno } if errno == ffmpeg::error::EAGAIN)
}
fn increment(value: &mut u64) -> Result<()> {
    *value = value
        .checked_add(1)
        .ok_or_else(|| Error::overflow("native transcode counter overflow"))?;
    Ok(())
}

#[allow(unsafe_code)]
mod ffi_helpers {
    use super::*;
    use std::{ffi::CString, ptr};
    pub(super) fn coded_audio_samples(
        parameters: &ffmpeg::codec::Parameters,
        bytes: usize,
    ) -> Result<(i64, u32)> {
        let bytes = i32::try_from(bytes)
            .map_err(|_| Error::overflow("compressed audio packet size exceeds native range"))?;
        // SAFETY: immutable borrowed codec parameters. The native metadata-only
        // duration helper does not decode or access a sample buffer.
        unsafe {
            let raw = &*parameters.as_ptr();
            if raw.sample_rate <= 0
                || (raw.codec_id == ffmpeg::ffi::AVCodecID::AV_CODEC_ID_AAC && raw.frame_size <= 0)
            {
                return Err(Error::unsupported(
                    "audio packet duration is unknown and codec configuration cannot establish its sample count",
                ));
            }
            let samples =
                ffmpeg::ffi::av_get_audio_frame_duration2(parameters.as_ptr() as *mut _, bytes);
            if samples <= 0 {
                return Err(Error::unsupported(
                    "audio packet duration cannot be established without decoding samples",
                ));
            }
            Ok((i64::from(samples), raw.sample_rate as u32))
        }
    }
    #[cfg(test)]
    pub(super) fn corrupt_frame() -> ffmpeg::frame::Video {
        let mut frame = ffmpeg::frame::Video::empty();
        // SAFETY: exclusive owned empty AVFrame; a native decode-error flag is
        // scalar metadata and this fixture makes no claim of actual GPU output.
        unsafe {
            (*frame.as_mut_ptr()).decode_error_flags = 1;
        }
        frame
    }
    #[cfg(test)]
    pub(super) fn incomplete_aac_header() -> ffmpeg::codec::Parameters {
        let mut parameters = ffmpeg::codec::Parameters::new();
        parameters.set_medium(ffmpeg::media::Type::Audio);
        parameters.set_id(ffmpeg::codec::Id::AAC);
        // SAFETY: exclusive parameters own this zero-padded codec configuration
        // allocation. The AVCodecParameters destructor frees it exactly once.
        unsafe {
            let p = &mut *parameters.as_mut_ptr();
            // Matches MP4 header-only discovery: sample rate is established by
            // the container, while channel layout needs codec configuration.
            p.sample_rate = 48000;
            p.extradata =
                ffmpeg::ffi::av_mallocz(2 + ffmpeg::ffi::AV_INPUT_BUFFER_PADDING_SIZE as usize)
                    as *mut u8;
            assert!(!p.extradata.is_null());
            // AAC-LC, 48 kHz, two channels, no sample packet required.
            p.extradata.write(0x11);
            p.extradata.add(1).write(0x90);
            p.extradata_size = 2;
        }
        parameters
    }
    pub(super) fn audio_header_complete(parameters: &ffmpeg::codec::Parameters) -> bool {
        // SAFETY: borrowed parameters retain a valid initialized native object.
        unsafe {
            (*parameters.as_ptr()).sample_rate > 0
                && (*parameters.as_ptr()).ch_layout.nb_channels > 0
        }
    }
    pub(super) fn establish_audio_frame_size(
        parameters: &mut ffmpeg::codec::Parameters,
    ) -> Result<()> {
        // SAFETY: exclusive owned parameters and bounded initialized codec bytes
        // are borrowed only for this metadata-only AAC configuration parser.
        unsafe {
            let parameters = &mut *parameters.as_mut_ptr();
            if parameters.frame_size > 0
                || parameters.codec_id != ffmpeg::ffi::AVCodecID::AV_CODEC_ID_AAC
            {
                return Ok(());
            }
            if parameters.extradata_size < 0 || parameters.extradata_size > 4 * 1024 * 1024 {
                return Err(Error::invalid("AAC codec configuration exceeds its bound"));
            }
            if parameters.extradata_size == 0
                || parameters.extradata.is_null()
                || parameters.sample_rate <= 0
            {
                return Ok(());
            }
            let configuration = std::slice::from_raw_parts(
                parameters.extradata,
                parameters.extradata_size as usize,
            );
            if let Some(samples) =
                aac_lc_frame_samples(configuration, parameters.sample_rate as u32)
            {
                parameters.frame_size = samples as i32;
            }
        }
        Ok(())
    }
    pub(super) fn fill_audio_header(
        parameters: &mut ffmpeg::codec::Parameters,
        decoder: &ffmpeg::decoder::Audio,
    ) -> Result<()> {
        let mut resolved = ffmpeg::codec::Parameters::new();
        // SAFETY: independently owned live codec parameters/context. Checked
        // conversion canonicalizes modern and pre-FFmpeg7 legacy audio layout
        // fields without hand-written offsets or ABI-dependent Rust cfg.
        unsafe {
            let result = ffmpeg::ffi::avcodec_parameters_from_context(
                resolved.as_mut_ptr(),
                decoder.as_ptr(),
            );
            if result < 0 {
                return Err(backend_error(ffmpeg::Error::from(result)));
            }
            let original = &*parameters.as_ptr();
            let canonical = &*resolved.as_ptr();
            if original.codec_id != canonical.codec_id
                || original.extradata_size != canonical.extradata_size
            {
                return Err(Error::incompatible(
                    "audio initialization changed copied codec/bitstream configuration",
                ));
            }
            if original.extradata_size > 0 {
                if original.extradata.is_null() || canonical.extradata.is_null() {
                    return Err(Error::incompatible(
                        "audio configuration has missing retained codec bytes",
                    ));
                }
                let length = original.extradata_size as usize;
                if std::slice::from_raw_parts(original.extradata, length)
                    != std::slice::from_raw_parts(canonical.extradata, length)
                {
                    return Err(Error::incompatible(
                        "audio initialization changed compressed codec bytes",
                    ));
                }
            }
        }
        *parameters = resolved;
        Ok(())
    }

    pub(super) fn attach_frames(
        encoder: &mut ffmpeg::codec::encoder::video::Video,
        frame: &ffmpeg::frame::Video,
    ) -> Result<()> {
        // SAFETY: validated CUDA frame retains an AVHWFramesContext. av_buffer_ref
        // creates an independent native reference owned/freed by the encoder;
        // no pointer is borrowed beyond its native reference lifetime.
        unsafe {
            let source = (*frame.as_ptr()).hw_frames_ctx;
            if source.is_null() {
                return Err(Error::incompatible(
                    "NVENC requires the actual native input hardware frame pool",
                ));
            }
            let target = encoder.as_mut_ptr();
            if !(*target).hw_frames_ctx.is_null() {
                return Err(Error::incompatible(
                    "encoder already has a hardware frame pool",
                ));
            }
            let reference = ffmpeg::ffi::av_buffer_ref(source);
            if reference.is_null() {
                return Err(Error::exhausted(
                    "cannot retain the NVDEC hardware frame pool",
                ));
            }
            (*target).hw_frames_ctx = reference;
            (*target).chroma_sample_location = frame.chroma_location().into();
        }
        Ok(())
    }
    pub(super) fn requires_global_header(path: &Path) -> Result<bool> {
        let path = CString::new(
            path.to_str()
                .ok_or_else(|| Error::invalid("output path must be UTF-8"))?,
        )
        .map_err(|_| Error::invalid("output path contains NUL"))?;
        // SAFETY: terminated live filename; result is immutable library-owned data.
        unsafe {
            let format = ffmpeg::ffi::av_guess_format(ptr::null(), path.as_ptr(), ptr::null());
            if format.is_null() {
                return Err(Error::unsupported("unknown output container"));
            }
            Ok((*format).flags & ffmpeg::ffi::AVFMT_GLOBALHEADER != 0)
        }
    }
    pub(super) fn copy_disposition(
        source: &ffmpeg::Stream<'_>,
        target: &mut ffmpeg::StreamMut<'_>,
    ) {
        // SAFETY: independently borrowed live input/output contexts own these streams.
        unsafe {
            (*target.as_mut_ptr()).disposition = source.disposition().bits();
        }
    }
    pub(super) fn configure_mux(output: &mut ffmpeg::format::context::Output) {
        // SAFETY: exclusive context before header. Preserve original negative DTS
        // and bound the native interleaver's sparse-track time horizon.
        unsafe {
            (*output.as_mut_ptr()).avoid_negative_ts = ffmpeg::ffi::AVFMT_AVOID_NEG_TS_DISABLED;
            (*output.as_mut_ptr()).max_interleave_delta = 1_000_000;
        }
    }
    pub(super) fn side_data_loss(frame: &ffmpeg::frame::Video) -> Vec<String> {
        // SAFETY: borrowed frame owns nb_side_data valid side-data pointers.
        unsafe {
            (0..(*frame.as_ptr()).nb_side_data)
                .map(|i| {
                    let data = *(*frame.as_ptr()).side_data.add(i as usize);
                    format!(
                        "video frame side data {:?} was not propagated",
                        ffmpeg::frame::side_data::Type::from((*data).type_)
                    )
                })
                .collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use media_core::error::ErrorKind;
    #[test]
    fn aac_lc_frame_duration_requires_established_configuration() {
        assert_eq!(aac_lc_frame_samples(&[0x11, 0x90], 48000), Some(1024));
        assert_eq!(aac_lc_frame_samples(&[0x11, 0x94], 48000), Some(960));
        assert_eq!(aac_lc_frame_samples(&[0x11, 0x90], 44100), None);
        assert_eq!(aac_lc_frame_samples(&[0x09, 0x90], 48000), None);
        assert_eq!(aac_lc_frame_samples(&[0x11, 0x80], 48000), None);
        assert_eq!(aac_lc_frame_samples(&[0x11], 48000), None);
    }
    #[test]
    fn missing_audio_duration_is_exactly_derived_or_rejected_without_samples() {
        ffmpeg::init().unwrap();
        let parameters = resolve_audio_header(ffi_helpers::incomplete_aac_header()).unwrap();
        let mut packet = ffmpeg::Packet::copy(&[1]);
        packet.set_pts(Some(107000));
        packet.set_dts(Some(107000));
        assert!(
            establish_packet_duration(&mut packet, &parameters, TimeBase::new(1, 48000).unwrap())
                .unwrap()
        );
        assert_eq!(packet.duration(), 1024);
        assert_eq!((packet.pts(), packet.dts()), (Some(107000), Some(107000)));
        packet.set_duration(832);
        assert!(
            !establish_packet_duration(&mut packet, &parameters, TimeBase::new(1, 48000).unwrap())
                .unwrap()
        );
        assert_eq!(
            packet.duration(),
            832,
            "established partial duration stays intact"
        );
        packet.set_duration(0);
        assert_eq!(
            establish_packet_duration(&mut packet, &parameters, TimeBase::new(1, 1000).unwrap())
                .unwrap_err()
                .kind(),
            ErrorKind::Incompatible
        );
        assert_eq!(
            establish_packet_duration(
                &mut packet,
                &ffi_helpers::incomplete_aac_header(),
                TimeBase::new(1, 48000).unwrap()
            )
            .unwrap_err()
            .kind(),
            ErrorKind::Unsupported
        );
    }
    #[test]
    fn marked_incomplete_native_frames_fail_before_encoder_submission() {
        let frame = ffi_helpers::corrupt_frame();
        assert!(frame.has_decode_errors());
        assert_eq!(
            validate_integrity(&frame).unwrap_err().kind(),
            ErrorKind::Backend
        );
        validate_integrity(&ffmpeg::frame::Video::empty()).unwrap();
    }
    #[test]
    fn header_only_audio_copy_preserves_every_packet_and_edit_offset() {
        use std::{
            process::Command,
            sync::atomic::{AtomicU64, Ordering},
        };
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let binary = std::env::var_os("MEDIA_FFMPEG_BIN")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                if Path::new("/opt/homebrew/bin/ffmpeg").is_file() {
                    PathBuf::from("/opt/homebrew/bin/ffmpeg")
                } else {
                    PathBuf::from("ffmpeg")
                }
            });
        if Command::new(&binary).arg("-version").output().is_err() {
            assert!(
                std::env::var_os("MEDIA_REQUIRE_FFMPEG_TESTS").is_none(),
                "required FFmpeg fixture generator unavailable"
            );
            eprintln!("UNEXERCISED: AAC copy fixture generator unavailable");
            return;
        }
        let directory = std::env::temp_dir().join(format!(
            "media-header-copy-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&directory).unwrap();
        struct Cleanup(PathBuf);
        impl Drop for Cleanup {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
        let _cleanup = Cleanup(directory.clone());
        let input = directory.join("source.mp4");
        let target = directory.join("copy.mp4");
        let generated = Command::new(&binary)
            .args([
                "-v",
                "error",
                "-nostdin",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=192x192:rate=10:duration=1.2",
                "-itsoffset",
                "0.25",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:sample_rate=48000:duration=0.7",
                "-c:v",
                "libx264",
                "-threads:v",
                "1",
                "-bf",
                "2",
                "-vf",
                "setpts='if(lt(N,6),N,6+(N-6)*2)/(10*TB)'",
                "-fps_mode",
                "vfr",
                "-c:a",
                "aac",
                "-threads:a",
                "1",
                "-b:a",
                "64000",
                "-output_ts_offset",
                "2",
            ])
            .arg(&input)
            .output()
            .unwrap();
        assert!(
            generated.status.success(),
            "{}",
            String::from_utf8_lossy(&generated.stderr)
        );
        let source = Source::file(&input).unwrap();
        let mut opened =
            format::open_source_without_decoding(&source, ProbeOptions::default()).unwrap();
        let video = opened.context.stream(0).unwrap();
        let nominal = video.rate();
        assert_eq!(nominal, ffmpeg::Rational(10, 1));
        assert_ne!(
            nominal,
            video.avg_frame_rate(),
            "VFR average cannot stand in for codec timing"
        );
        let file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .read(true)
            .open(&target)
            .unwrap();
        let mut output = ffmpeg::format::output_to_stream(
            ffmpeg::format::context::StreamIo::from_write_seek(file).unwrap(),
            target.to_str(),
            None,
        )
        .unwrap();
        for stream in opened.context.streams() {
            let mut parameters = stream.parameters();
            if parameters.medium() == ffmpeg::media::Type::Audio {
                parameters = resolve_audio_header(parameters).unwrap();
            }
            let mut added = output
                .add_stream(ffmpeg::encoder::find(ffmpeg::codec::Id::None))
                .unwrap();
            added.set_parameters(parameters);
            added.set_time_base(stream.time_base());
            added.set_avg_frame_rate(stream.avg_frame_rate());
            added.set_metadata(stream.metadata().to_owned());
            ffi_helpers::copy_disposition(&stream, &mut added);
            native::clear_codec_tag(&mut added);
        }
        output.set_metadata(opened.context.metadata().to_owned());
        ffi_helpers::configure_mux(&mut output);
        output.write_header().unwrap();
        opened.allow_reading();
        let mut sent = 0;
        loop {
            let mut packet = ffmpeg::Packet::empty();
            match packet.read(&mut opened.context) {
                Ok(()) => {}
                Err(ffmpeg::Error::Eof) => break,
                Err(e) => panic!("{e}"),
            }
            let stream = packet.stream();
            let from =
                format::time_base(opened.context.stream(stream).unwrap().time_base()).unwrap();
            let to = format::time_base(output.stream(stream).unwrap().time_base()).unwrap();
            if stream == 1 {
                let parameters = output.stream(stream).unwrap().parameters();
                establish_packet_duration(&mut packet, &parameters, from).unwrap();
            }
            rescale_packet(&mut packet, from, to).unwrap();
            if stream == 1 {
                eprintln!(
                    "native AAC pts {:?} duration {} bytes {}",
                    packet.pts(),
                    packet.duration(),
                    packet.size()
                );
                sent += 1;
            }
            packet.set_position(-1);
            packet.write_interleaved(&mut output).unwrap();
        }
        output.write_trailer().unwrap();
        native::flush_output(&mut output).unwrap();
        drop(output);
        let collect = |source: &Source| {
            let mut reader = format::PacketReader::open(source, ProbeOptions::default()).unwrap();
            let mut packets = Vec::new();
            while let Some(packet) = reader.next_packet().unwrap() {
                if packet.track == TrackId(1) {
                    packets.push(packet);
                }
            }
            packets
        };
        let before = collect(&source);
        let after = collect(&Source::file(&target).unwrap());
        eprintln!(
            "AAC sent {sent}; before {:?}; after {:?}",
            before
                .iter()
                .map(|p| (p.pts, p.duration, p.data.len()))
                .collect::<Vec<_>>(),
            after
                .iter()
                .map(|p| (p.pts, p.duration, p.data.len()))
                .collect::<Vec<_>>()
        );
        assert_eq!(sent, before.len());
        assert_eq!(before.len(), after.len());
        for (before, after) in before.iter().zip(after) {
            assert_eq!(before.data, after.data);
            assert_eq!(
                (before.pts, before.dts, before.duration),
                (after.pts, after.dts, after.duration)
            );
        }
    }
    #[test]
    fn ordinal_decode_delay_preserves_vfr_across_a_late_large_gap() {
        let mut clock = DecodeClock::new(2);
        for pts in [0, 10, 20, 1000, 1010, 1020] {
            clock.push(pts, 10).unwrap();
        }
        let output_pts = [0, 1000, 10, 20, 1020, 1010];
        let dts: Vec<_> = output_pts
            .into_iter()
            .map(|pts| clock.next(pts).unwrap())
            .collect();
        assert_eq!(dts, [-20, -10, 0, 10, 20, 1000]);
        assert!(dts.windows(2).all(|p| p[0] < p[1]));
        assert!(dts.iter().zip(output_pts).all(|(&dts, pts)| dts <= pts));
        assert_eq!(clock.inputs.len(), 2, "only the reorder tail remains");
        assert!(
            1000 - 2 * 10 > output_pts[3],
            "a constant-time offset would be invalid here"
        );
    }
    #[test]
    fn decode_clock_requires_timing_bounds_and_unique_ordered_pts() {
        let mut clock = DecodeClock::new(2);
        clock.push(-100, 0).unwrap();
        assert_eq!(clock.next(-100).unwrap_err().kind(), ErrorKind::Unsupported);
        clock.push(-90, 0).unwrap();
        assert_eq!(clock.next(-100).unwrap(), -120);
        assert_eq!(
            clock.push(-90, 10).unwrap_err().kind(),
            ErrorKind::Incompatible
        );
        assert_eq!(
            clock.push(-200, 10).unwrap_err().kind(),
            ErrorKind::Incompatible
        );
        let mut overflow = DecodeClock::new(2);
        overflow.push(i64::MIN, 1).unwrap();
        assert_eq!(
            overflow.next(i64::MIN).unwrap_err().kind(),
            ErrorKind::Overflow
        );
        let mut bound = DecodeClock::new(0);
        bound.push(100, 10).unwrap();
        assert_eq!(bound.next(90).unwrap_err().kind(), ErrorKind::Incompatible);
    }
    #[test]
    fn invalid_native_requests_and_bounded_preroll() {
        let mut options = CudaTranscodeOptions {
            max_pending_frames: 65,
            ..Default::default()
        };
        assert_eq!(
            validate_options(&options).unwrap_err().kind(),
            ErrorKind::InvalidInput
        );
        options.max_pending_frames = 2;
        options.max_preroll_bytes = 4;
        options.max_preroll_packets = 1;
        validate_options(&options).unwrap();
        let mut bytes = 0;
        stage_audio(&mut bytes, 0, 4, &options).unwrap();
        assert!(stage_audio(&mut bytes, 1, 0, &options).is_err());
        assert_eq!(bytes, 4);
        assert!(stage_audio(&mut bytes, 0, 1, &options).is_err());
        assert_eq!(bytes, 4);
        assert!(validate_geometry(192, 191).is_err());
    }
    #[test]
    fn application_pending_budget_does_not_expand_native_reference_pool() {
        assert_eq!(decoder_headroom(1), 1);
        assert_eq!(decoder_headroom(8), 8);
        assert_eq!(decoder_headroom(32), 8);
        assert_eq!(decoder_headroom(64), 8);
        assert_eq!(CudaTranscodeOptions::default().max_pending_frames, 8);
    }
    #[test]
    fn aac_configuration_establishes_copy_headers_without_decoding_samples() {
        ffmpeg::init().unwrap();
        let parameters = ffi_helpers::incomplete_aac_header();
        assert!(!ffi_helpers::audio_header_complete(&parameters));
        let resolved = resolve_audio_header(parameters).unwrap();
        assert!(ffi_helpers::audio_header_complete(&resolved));
        let mut decoder = ffmpeg::codec::context::Context::from_parameters(resolved)
            .unwrap()
            .decoder()
            .audio()
            .unwrap();
        assert_eq!((decoder.rate(), decoder.channels()), (48000, 2));
        let mut frame = ffmpeg::frame::Audio::empty();
        assert!(
            is_again(decoder.receive_frame(&mut frame).unwrap_err()),
            "configuration parsing must not produce decoded sample frames"
        );
    }
    #[test]
    fn original_signed_packet_clock_is_exact_or_rejected() {
        let mut packet = ffmpeg::Packet::empty();
        packet.set_pts(Some(250));
        packet.set_dts(Some(-200));
        packet.set_duration(100);
        rescale_packet(
            &mut packet,
            TimeBase::new(1, 1000).unwrap(),
            TimeBase::new(1, 48000).unwrap(),
        )
        .unwrap();
        assert_eq!(
            (packet.pts(), packet.dts(), packet.duration()),
            (Some(12000), Some(-9600), 4800)
        );
        packet.set_pts(Some(1));
        assert_eq!(
            rescale_packet(
                &mut packet,
                TimeBase::new(1, 44100).unwrap(),
                TimeBase::new(1, 1000).unwrap()
            )
            .unwrap_err()
            .kind(),
            ErrorKind::Incompatible
        );
    }
}
