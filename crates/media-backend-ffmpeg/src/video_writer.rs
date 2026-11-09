//! Incremental native CPU video-only writer with explicit backpressure/finalize.
//!
//! Accepts shared host RGB8 or YUV420P8 frames, copying visible rows into native
//! reference-counted frames. Encoder-private pools are not whole-process memory
//! accounting. Output remains partial until finish drains EOF and writes trailer.
use crate::{
    encode::{self, CpuVideoCodec, TranscodeOptions, ffi_helpers},
    format::{self, backend_error},
    native,
    packet_decode::{exact_ticks, native_base},
};
use ffmpeg_next as ffmpeg;
use media_core::{
    error::{Error, Result},
    media::{ColorDescription, Orientation, PixelFormat, TrackId, VideoFrame},
    time::{Time, TimeBase},
};
use std::{
    collections::BTreeMap,
    fs::{File, OpenOptions},
    path::{Path, PathBuf},
};

/// Explicit timestamp field supplied to the output encoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoTimestamp {
    /// Original source PTS; preserves nonzero source offsets.
    Source,
    /// Asset-relative PTS; caller explicitly selects this alternate clock.
    Asset,
}
/// Container duration fidelity, separate from exact submitted presentation PTS.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoDurationPolicy {
    /// Require Matroska BlockDuration/default-duration carriage on its millisecond grid.
    Exact,
    /// Explicitly permit muxer-derived packet durations, reported as a limitation.
    AllowContainerRewrite,
}
/// Direct video encoder setup. Dimensions and input format never renegotiate.
#[derive(Debug, Clone)]
pub struct VideoWriterOptions {
    /// Exact even output/input width.
    pub width: u32,
    /// Exact even output/input height.
    pub height: u32,
    /// Host RGB8 or native planar YUV420P8.
    pub input_format: PixelFormat,
    /// Encoder/mux timestamps; every submitted PTS/duration must fit exactly.
    pub time_base: TimeBase,
    /// Explicit nominal seconds per frame for codec/container hints; variable
    /// presentation PTS and durations remain independently supplied on frames.
    pub nominal_frame_duration: TimeBase,
    /// Exact duration carriage defaults to Matroska; other muxers require opt-in.
    pub duration_policy: VideoDurationPolicy,
    /// Required logical source frame identity.
    pub track: TrackId,
    /// Explicit PTS field; Source by default.
    pub timestamp: VideoTimestamp,
    /// Explicit encoded YUV description. Initial path supports BT.709 matrix,
    /// primaries/transfer with full or limited range and left chroma siting.
    /// RGB inputs must declare the same primaries/transfer, RGB matrix/full range.
    pub color: ColorDescription,
    /// Software H.264 or HEVC implementation.
    pub codec: CpuVideoCodec,
    /// Bits per second.
    pub bit_rate: usize,
    /// Explicit supported codec profile, or encoder negotiation.
    pub profile: Option<String>,
    /// Validated x264/x265 speed preset.
    pub preset: String,
    /// Maximum B frames (0..=16).
    pub max_b_frames: usize,
    /// Keyframe interval.
    pub gop: u32,
    /// Bounded native codec workers (1..=64).
    pub threads: usize,
    /// Per-frame conservative storage footprint, excluding codec-private pools.
    pub max_frame_bytes: u64,
    /// Maximum submitted frames awaiting reordered packets/durations.
    pub max_pending_frames: usize,
    /// Maximum one compressed output packet before muxing.
    pub max_packet_bytes: usize,
}
impl VideoWriterOptions {
    /// Strict color/geometry declaration; defaults to one worker and H.264.
    pub fn new(
        width: u32,
        height: u32,
        input_format: PixelFormat,
        time_base: TimeBase,
        nominal_frame_duration: TimeBase,
        color: ColorDescription,
    ) -> Self {
        Self {
            width,
            height,
            input_format,
            time_base,
            nominal_frame_duration,
            duration_policy: VideoDurationPolicy::Exact,
            track: TrackId(0),
            timestamp: VideoTimestamp::Source,
            color,
            codec: CpuVideoCodec::H264,
            bit_rate: 2_000_000,
            profile: None,
            preset: "veryfast".into(),
            max_b_frames: 2,
            gop: 60,
            threads: 1,
            max_frame_bytes: 256 * 1024 * 1024,
            max_pending_frames: 128,
            max_packet_bytes: 16 * 1024 * 1024,
        }
    }
}
/// Acceptance leaves caller ownership intact in every state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoWriteState {
    /// Frame was accepted and its bytes can be released by the caller.
    Accepted,
    /// Drain available encoded packets before retrying this same frame.
    NeedDrain,
    /// Writer was already finalized; frame was not consumed.
    End,
}
/// Packet drain result; available packets are written incrementally to the muxer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoDrainState {
    /// Encoder needs another input frame before it can produce more packets.
    NeedInput,
    /// Encoder EOF drained; trailer still requires finish.
    End,
}
/// File/codec counts. finalized false explicitly describes retained partial output.
#[derive(Debug, Clone)]
pub struct VideoWriterReport {
    /// New destination; never an overwritten existing file.
    pub destination: PathBuf,
    /// Actual frames accepted by native encoder.
    pub frames_accepted: u64,
    /// Actual encoded packets accepted by muxer.
    pub packets_written: u64,
    /// True only after codec EOF, trailer, flush and file synchronization.
    pub finalized: bool,
    /// Native codec/mux I/O has failed; successful finalization is no longer possible.
    pub failed: bool,
    /// Actual software codec implementation.
    pub codec: CpuVideoCodec,
    /// Native encoded pixel format.
    pub pixel_format: &'static str,
    /// Explicit container fidelity limitations accepted by caller policy.
    pub warnings: Vec<String>,
}
/// Incremental writer. Drop never declares successful completion.
pub struct VideoWriter {
    output: Option<ffmpeg::format::context::Output>,
    sync: File,
    encoder: ffmpeg::encoder::Video,
    options: VideoWriterOptions,
    output_base: TimeBase,
    pending: BTreeMap<i64, i64>,
    last_end: Option<Time>,
    last_pts: Option<Time>,
    must_drain: bool,
    draining: bool,
    encoder_end: bool,
    failed: bool,
    report: VideoWriterReport,
}
impl VideoWriter {
    /// Validate/open codec and negotiate container before creating a new file.
    pub fn create(destination: &Path, options: VideoWriterOptions) -> Result<Self> {
        format::initialize()?;
        encode::validate_options(&TranscodeOptions {
            codec: options.codec,
            bit_rate: options.bit_rate,
            profile: options.profile.clone(),
            preset: options.preset.clone(),
            max_b_frames: options.max_b_frames,
            gop: options.gop,
            threads: options.threads,
            max_frame_bytes: options.max_frame_bytes,
            max_pending_frames: options.max_pending_frames,
            ..Default::default()
        })?;
        encode::check_geometry(options.width, options.height, options.max_frame_bytes)?;
        if !matches!(
            options.input_format,
            PixelFormat::Rgb8 | PixelFormat::Yuv420p
        ) || options.max_packet_bytes == 0
        {
            return Err(Error::unsupported(
                "incremental CPU writer requires RGB8/YUV420P8 and a nonzero packet bound",
            ));
        }
        if options.max_pending_frames < options.max_b_frames + options.threads + 24 {
            return Err(Error::invalid(
                "pending-frame bound cannot cover configured lookahead/B-frame/worker delay",
            ));
        }
        validate_color(&options.color, false, &options.color)?;
        native::check_mux_codec(destination, options.codec.id())?;
        if options.duration_policy == VideoDurationPolicy::Exact
            && destination
                .extension()
                .and_then(|extension| extension.to_str())
                != Some("mkv")
        {
            return Err(Error::unsupported(
                "exact incremental frame durations currently require .mkv; explicitly allow container rewriting for other muxers",
            ));
        }
        if options.duration_policy == VideoDurationPolicy::Exact {
            let milliseconds = TimeBase::new(1, 1000)?;
            exact_ticks(options.time_base.time(1)?, milliseconds).map_err(|_| {
                Error::unsupported(
                    "strict Matroska timestamps require an integral millisecond time-base grid",
                )
            })?;
            exact_ticks(options.nominal_frame_duration.time(1)?, milliseconds).map_err(|_| {
                Error::unsupported(
                    "strict Matroska nominal frame duration must fit its millisecond grid exactly",
                )
            })?;
        }
        let base = native_base(options.time_base)?;
        let nominal = native_base(options.nominal_frame_duration)?;
        let frame_rate = ffmpeg::Rational(nominal.denominator(), nominal.numerator());
        let codec = ffmpeg::encoder::find_by_name(options.codec.encoder_name())
            .ok_or_else(|| Error::unsupported("linked FFmpeg lacks requested software encoder"))?;
        let mut encoder = ffmpeg::codec::context::Context::new_with_codec(codec)
            .encoder()
            .video()
            .map_err(backend_error)?;
        encoder.set_width(options.width);
        encoder.set_height(options.height);
        encoder.set_format(ffmpeg::format::Pixel::YUV420P);
        encoder.set_time_base(base);
        encoder.set_frame_rate(Some(frame_rate));
        encoder.set_aspect_ratio(ffmpeg::Rational(1, 1));
        encoder.set_bit_rate(options.bit_rate);
        encoder.set_max_b_frames(options.max_b_frames);
        encoder.set_gop(options.gop);
        encoder.set_threading(ffmpeg::threading::Config::count(options.threads));
        encoder.set_colorspace(ffmpeg::color::Space::BT709);
        encoder.set_color_primaries(ffmpeg::color::Primaries::BT709);
        encoder.set_color_transfer_characteristic(ffmpeg::color::TransferCharacteristic::BT709);
        encoder.set_color_range(if full_range(&options.color)? {
            ffmpeg::color::Range::JPEG
        } else {
            ffmpeg::color::Range::MPEG
        });
        ffi_helpers::set_encoder_chroma(&mut encoder, ffmpeg::chroma::Location::Left);
        if ffi_helpers::requires_global_header(destination)? {
            encoder.set_flags(ffmpeg::codec::Flags::GLOBAL_HEADER);
        }
        let mut settings = ffmpeg::Dictionary::new();
        settings.set("preset", &options.preset);
        if let Some(profile) = &options.profile {
            settings.set("profile", profile);
        }
        match options.codec {
            CpuVideoCodec::H264 => settings.set(
                "x264-params",
                "rc-lookahead=16:sync-lookahead=0:lookahead-threads=1",
            ),
            CpuVideoCodec::Hevc => settings.set(
                "x265-params",
                &format!(
                    "pools=none:frame-threads={}:wpp=0:rc-lookahead=16",
                    options.threads
                ),
            ),
        }
        let encoder = encoder.open_with(settings).map_err(backend_error)?;
        if format::time_base(encoder.time_base())? != options.time_base {
            return Err(Error::incompatible(
                "native encoder changed exact writer time base",
            ));
        }
        let name = destination
            .to_str()
            .ok_or_else(|| Error::invalid("writer destination path must be UTF-8"))?;
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(destination)
            .map_err(Error::io)?;
        let sync = file.try_clone().map_err(Error::io)?;
        let mut output = ffmpeg::format::output_to_stream(
            ffmpeg::format::context::StreamIo::from_write_seek(file).map_err(backend_error)?,
            Some(name),
            None,
        )
        .map_err(|error| partial(backend_error(error), destination))?;
        let setup = (|| {
            let mut stream = output.add_stream(codec).map_err(backend_error)?;
            stream.set_parameters(&encoder);
            stream.set_time_base(base);
            stream.set_avg_frame_rate(frame_rate);
            native::clear_codec_tag(&mut stream);
            ffi_helpers::configure_mux(&mut output);
            output.write_header().map_err(backend_error)?;
            format::time_base(
                output
                    .stream(0)
                    .ok_or_else(|| Error::backend("ffmpeg", "output stream missing"))?
                    .time_base(),
            )
        })();
        let output_base = setup.map_err(|error| partial(error, destination))?;
        let report = VideoWriterReport {
            destination: destination.to_owned(),
            frames_accepted: 0,
            packets_written: 0,
            finalized: false,
            failed: false,
            codec: options.codec,
            pixel_format: "yuv420p",
            warnings: if options.duration_policy == VideoDurationPolicy::AllowContainerRewrite {
                vec!["container may derive packet durations from decode-order DTS; submitted PTS remain exact but per-frame duration carriage is not guaranteed".into()]
            } else {
                Vec::new()
            },
        };
        Ok(Self {
            output: Some(output),
            sync,
            encoder,
            options,
            output_base,
            pending: BTreeMap::new(),
            last_end: None,
            last_pts: None,
            must_drain: false,
            draining: false,
            encoder_end: false,
            failed: false,
            report,
        })
    }
    /// Submit one shared core frame. NeedDrain does not consume or retain it.
    /// Rejected metadata/timing/geometry leaves the healthy writer retryable.
    pub fn write_frame(&mut self, frame: &VideoFrame) -> Result<VideoWriteState> {
        if self.failed {
            return Err(partial(
                Error::invalid("writer previously failed; output cannot be finalized"),
                &self.report.destination,
            ));
        }
        if self.report.finalized || self.draining {
            return Ok(VideoWriteState::End);
        }
        if frame.track != self.options.track
            || frame.image.width() != self.options.width
            || frame.image.height() != self.options.height
            || frame.image.pixel_format() != self.options.input_format
        {
            return Err(Error::incompatible(
                "writer frame changed track, geometry or configured pixel format",
            ));
        }
        let metadata = frame.image.metadata();
        if metadata.orientation != Orientation::Normal
            || metadata.icc_profile.is_some()
            || metadata.exif.is_some()
        {
            return Err(Error::unsupported(
                "writer requires explicitly applied orientation and no unhandled ICC/EXIF metadata; no metadata is silently dropped",
            ));
        }
        validate_color(
            &metadata.color,
            self.options.input_format == PixelFormat::Rgb8,
            &self.options.color,
        )?;
        if frame.image.storage().byte_len() as u64 > self.options.max_frame_bytes {
            return Err(Error::exhausted(
                "input host allocation exceeds writer frame budget",
            ));
        }
        frame.image.host_bytes()?;
        let pts = match self.options.timestamp {
            VideoTimestamp::Source => frame.timing.source_pts,
            VideoTimestamp::Asset => frame.timing.pts,
        }
        .ok_or_else(|| Error::incompatible("writer selected PTS clock is missing"))?;
        let duration = frame
            .timing
            .duration
            .filter(|duration| *duration > Time::ZERO)
            .ok_or_else(|| {
                Error::incompatible("writer requires an explicit positive frame duration")
            })?;
        let ticks = exact_ticks(pts, self.options.time_base)?;
        let duration_ticks = exact_ticks(duration, self.options.time_base)?;
        let end = pts.checked_add(duration)?;
        if self.last_pts.is_some_and(|last| pts <= last)
            || self.last_end.is_some_and(|last| pts < last)
        {
            return Err(Error::incompatible(
                "writer frame PTS is not increasing or declared display durations overlap",
            ));
        }
        if self.must_drain {
            return Ok(VideoWriteState::NeedDrain);
        }
        if self.pending.len() >= self.options.max_pending_frames {
            return Err(partial(
                Error::exhausted("writer pending-frame budget exhausted"),
                &self.report.destination,
            ));
        }
        let mut native = ffi::host_frame(frame, &self.options)?;
        native.set_pts(Some(ticks));
        native.set_kind(ffmpeg::picture::Type::None);
        ffi::set_duration(&mut native, duration_ticks);
        match self.encoder.send_frame(&native) {
            Ok(()) => {
                self.pending.insert(ticks, duration_ticks);
                self.last_pts = Some(pts);
                self.last_end = Some(end);
                self.must_drain = true;
                self.report.frames_accepted = self
                    .report
                    .frames_accepted
                    .checked_add(1)
                    .ok_or_else(|| Error::overflow("accepted writer frame counter"))?;
                Ok(VideoWriteState::Accepted)
            }
            Err(error) if is_again(error) => Ok(VideoWriteState::NeedDrain),
            Err(error) => {
                self.failed = true;
                Err(partial(backend_error(error), &self.report.destination))
            }
        }
    }
    /// Mux all currently available packets without buffering a Rust packet queue.
    pub fn drain(&mut self) -> Result<VideoDrainState> {
        if self.failed {
            return Err(partial(
                Error::invalid("writer previously failed"),
                &self.report.destination,
            ));
        }
        if self.encoder_end || self.report.finalized {
            return Ok(VideoDrainState::End);
        }
        let result = self.drain_inner();
        if result.is_err() {
            self.failed = true;
        }
        result.map_err(|error| partial(error, &self.report.destination))
    }
    fn drain_inner(&mut self) -> Result<VideoDrainState> {
        loop {
            let mut packet = ffmpeg::Packet::empty();
            match self.encoder.receive_packet(&mut packet) {
                Ok(()) => {}
                Err(ffmpeg::Error::Eof) if self.draining => {
                    self.encoder_end = true;
                    return Ok(VideoDrainState::End);
                }
                Err(error) if is_again(error) && !self.draining => {
                    self.must_drain = false;
                    return Ok(VideoDrainState::NeedInput);
                }
                Err(error) => return Err(backend_error(error)),
            }
            if packet.size() > self.options.max_packet_bytes {
                return Err(Error::exhausted(
                    "encoded packet exceeds writer packet bound",
                ));
            }
            let pts = packet
                .pts()
                .ok_or_else(|| Error::backend("ffmpeg", "encoder returned missing PTS"))?;
            let duration = self.pending.remove(&pts).ok_or_else(|| {
                Error::backend("ffmpeg", "encoder packet does not match a submitted frame")
            })?;
            packet.set_duration(duration);
            encode::rescale_packet(&mut packet, self.options.time_base, self.output_base)?;
            packet.set_stream(0);
            packet.set_position(-1);
            packet
                .write_interleaved(
                    self.output
                        .as_mut()
                        .ok_or_else(|| Error::invalid("writer output already retired"))?,
                )
                .map_err(backend_error)?;
            self.report.packets_written = self
                .report
                .packets_written
                .checked_add(1)
                .ok_or_else(|| Error::overflow("written video packet counter"))?;
        }
    }
    /// Drain delayed B frames/packets and write trailer exactly once.
    /// Repeated successful finish returns the same finalized report.
    pub fn finish(&mut self) -> Result<VideoWriterReport> {
        if self.report.finalized {
            return Ok(self.report.clone());
        }
        let result = self.finish_inner();
        if result.is_err() {
            self.failed = true;
        }
        result.map_err(|error| partial(error, &self.report.destination))?;
        self.report.finalized = true;
        Ok(self.report.clone())
    }
    fn finish_inner(&mut self) -> Result<()> {
        if self.failed || self.report.frames_accepted == 0 {
            return Err(Error::invalid(
                "failed/empty writer cannot be finalized successfully",
            ));
        }
        self.drain()?;
        if !self.draining {
            self.encoder.send_eof().map_err(backend_error)?;
            self.draining = true;
        }
        if self.drain()? != VideoDrainState::End || !self.pending.is_empty() {
            return Err(Error::backend(
                "ffmpeg",
                "writer EOF did not account for all accepted frames",
            ));
        }
        let output = self
            .output
            .as_mut()
            .ok_or_else(|| Error::invalid("writer output missing"))?;
        output.write_trailer().map_err(backend_error)?;
        native::flush_output(output)?;
        drop(self.output.take());
        self.sync.sync_all().map_err(Error::io)
    }
    /// Snapshot explicitly distinguishes unfinished/failed partial output.
    pub fn status(&self) -> VideoWriterReport {
        let mut report = self.report.clone();
        report.failed = self.failed;
        report
    }
    /// Close native output without declaring completion; retain a partial-output report.
    /// Already finalized output remains finalized in the returned evidence.
    pub fn abort(mut self) -> VideoWriterReport {
        drop(self.output.take());
        self.report.failed = self.failed;
        self.report
    }
}
fn partial(error: Error, path: &Path) -> Error {
    error
        .with_source(path.display().to_string())
        .with_operation("incremental video writer; partial destination may exist")
}
fn full_range(color: &ColorDescription) -> Result<bool> {
    match color.range.as_deref() {
        Some("full" | "pc") => Ok(true),
        Some("limited" | "tv") => Ok(false),
        _ => Err(Error::incompatible(
            "writer needs an explicit full/limited color range",
        )),
    }
}
fn validate_color(input: &ColorDescription, rgb: bool, output: &ColorDescription) -> Result<()> {
    if input.primaries.as_deref() != Some("bt709")
        || input.transfer.as_deref() != Some("bt709")
        || output.primaries.as_deref() != Some("bt709")
        || output.transfer.as_deref() != Some("bt709")
        || output.matrix.as_deref() != Some("bt709")
        || output.chroma_siting.as_deref() != Some("left")
    {
        return Err(Error::unsupported(
            "initial CPU writer requires explicit BT.709 primaries/transfer/matrix and left output chroma siting",
        ));
    }
    if rgb {
        if input.matrix.as_deref() != Some("rgb")
            || !full_range(input)?
            || input.chroma_siting.is_some()
        {
            return Err(Error::incompatible(
                "RGB input needs declared RGB matrix/full range and no subsampled chroma grid",
            ));
        }
    } else if input.matrix.as_deref() != output.matrix.as_deref()
        || input.chroma_siting.as_deref() != Some("left")
        || full_range(input)? != full_range(output)?
    {
        return Err(Error::incompatible(
            "native YUV input color/range/chroma differs from writer negotiation",
        ));
    }
    Ok(())
}
fn is_again(error: ffmpeg::Error) -> bool {
    matches!(error,ffmpeg::Error::Other{errno} if errno==ffmpeg::error::EAGAIN)
}

#[allow(unsafe_code)]
mod ffi {
    use super::*;
    pub(super) fn set_duration(frame: &mut ffmpeg::frame::Video, duration: i64) {
        // SAFETY: exclusively borrowed live AVFrame; duration is a scalar tick count.
        unsafe {
            (*frame.as_mut_ptr()).duration = duration;
        }
    }
    fn allocate(
        width: u32,
        height: u32,
        format: ffmpeg::format::Pixel,
    ) -> Result<ffmpeg::frame::Video> {
        let mut frame = ffmpeg::frame::Video::empty();
        frame.set_width(width);
        frame.set_height(height);
        frame.set_format(format);
        // SAFETY: fresh exclusively owned AVFrame with validated dimensions/format.
        let status = unsafe { ffmpeg::ffi::av_frame_get_buffer(frame.as_mut_ptr(), 32) };
        if status < 0 {
            return Err(backend_error(ffmpeg::Error::from(status)));
        }
        Ok(frame)
    }
    pub(super) fn host_frame(
        input: &VideoFrame,
        options: &VideoWriterOptions,
    ) -> Result<ffmpeg::frame::Video> {
        let rgb = options.input_format == PixelFormat::Rgb8;
        let mut source = allocate(
            options.width,
            options.height,
            if rgb {
                ffmpeg::format::Pixel::RGB24
            } else {
                ffmpeg::format::Pixel::YUV420P
            },
        )?;
        let bytes = input.image.host_bytes()?;
        let geometry = if rgb {
            vec![(options.width as usize * 3, options.height as usize)]
        } else {
            vec![
                (options.width as usize, options.height as usize),
                (options.width as usize / 2, options.height as usize / 2),
                (options.width as usize / 2, options.height as usize / 2),
            ]
        };
        if input.image.layout().planes.len() != geometry.len() {
            return Err(Error::incompatible(
                "writer requires explicit host planes matching its packed/planar input",
            ));
        }
        for (index, (row_bytes, rows)) in geometry.into_iter().enumerate() {
            let plane = &input.image.layout().planes[index];
            if plane.row_bytes < row_bytes || plane.rows < rows || plane.row_stride < row_bytes {
                return Err(Error::incompatible(
                    "input image planes do not cover visible pixels",
                ));
            }
            let stride = source.stride(index);
            for row in 0..rows {
                let start = plane.offset + row * plane.row_stride;
                let visible = bytes
                    .get(start..start + row_bytes)
                    .ok_or_else(|| Error::invalid("writer visible source plane exceeds storage"))?;
                source.data_mut(index)[row * stride..row * stride + row_bytes]
                    .copy_from_slice(visible);
            }
        }
        let full = full_range(&options.color)?;
        let mut output = if rgb {
            rgb_to_yuv(&source, full)?
        } else {
            source
        };
        output.set_color_space(ffmpeg::color::Space::BT709);
        output.set_color_primaries(ffmpeg::color::Primaries::BT709);
        output.set_color_transfer_characteristic(ffmpeg::color::TransferCharacteristic::BT709);
        output.set_color_range(if full {
            ffmpeg::color::Range::JPEG
        } else {
            ffmpeg::color::Range::MPEG
        });
        ffi_helpers::set_frame_attributes(
            &mut output,
            ffmpeg::chroma::Location::Left,
            ffmpeg::Rational(1, 1),
        );
        Ok(output)
    }
    fn rgb_to_yuv(source: &ffmpeg::frame::Video, full: bool) -> Result<ffmpeg::frame::Video> {
        use ffmpeg::ffi;
        // SAFETY: allocation returns one owned scaler; the local guard frees it on every path.
        let raw = std::ptr::NonNull::new(unsafe { ffi::sws_alloc_context() })
            .ok_or_else(|| Error::exhausted("writer RGB scaler allocation"))?;
        struct Scaler(std::ptr::NonNull<ffi::SwsContext>);
        impl Drop for Scaler {
            fn drop(&mut self) {
                // SAFETY: this guard owns the unique scaler allocation and frees it exactly once.
                unsafe {
                    ffi::sws_freeContext(self.0.as_ptr());
                }
            }
        }
        let scaler = Scaler(raw);
        for (name, value) in [
            (c"srcw", i64::from(source.width())),
            (c"srch", i64::from(source.height())),
            (c"dstw", i64::from(source.width())),
            (c"dsth", i64::from(source.height())),
            (c"src_format", ffi::AVPixelFormat::AV_PIX_FMT_RGB24 as i64),
            (c"dst_format", ffi::AVPixelFormat::AV_PIX_FMT_YUV420P as i64),
            (
                c"sws_flags",
                i64::from(ffmpeg::software::scaling::Flags::BILINEAR.bits()),
            ),
            (c"src_range", 1),
            (c"dst_range", i64::from(full)),
            (c"dst_h_chr_pos", 0),
            (c"dst_v_chr_pos", 128),
            (c"threads", 1),
        ] {
            // SAFETY: live exclusive pre-init AVOptions scaler, static terminated option names.
            let status =
                unsafe { ffi::av_opt_set_int(raw.as_ptr().cast(), name.as_ptr(), value, 0) };
            if status < 0 {
                return Err(backend_error(ffmpeg::Error::from(status)));
            }
        }
        // SAFETY: validated positive formats/dimensions and null optional filters.
        let status = unsafe {
            ffi::sws_init_context(raw.as_ptr(), std::ptr::null_mut(), std::ptr::null_mut())
        };
        if status < 0 {
            return Err(backend_error(ffmpeg::Error::from(status)));
        }
        // SAFETY: immutable library-owned coefficient table used synchronously by owned scaler.
        let status = unsafe {
            let coefficients = ffi::sws_getCoefficients(ffi::SWS_CS_ITU709);
            ffi::sws_setColorspaceDetails(
                raw.as_ptr(),
                coefficients,
                1,
                coefficients,
                i32::from(full),
                0,
                1 << 16,
                1 << 16,
            )
        };
        if status < 0 {
            return Err(backend_error(ffmpeg::Error::from(status)));
        }
        let mut target = allocate(
            source.width(),
            source.height(),
            ffmpeg::format::Pixel::YUV420P,
        )?;
        // SAFETY: owned validated source/destination AVFrames stay borrowed for the complete call.
        let rows = unsafe {
            ffi::sws_scale(
                raw.as_ptr(),
                (*source.as_ptr()).data.as_ptr() as *const *const u8,
                (*source.as_ptr()).linesize.as_ptr(),
                0,
                source.height() as i32,
                (*target.as_mut_ptr()).data.as_ptr(),
                (*target.as_ptr()).linesize.as_ptr(),
            )
        };
        drop(scaler);
        if rows != source.height() as i32 {
            return Err(Error::backend(
                "ffmpeg",
                "writer RGB conversion did not produce every row",
            ));
        }
        Ok(target)
    }
}
