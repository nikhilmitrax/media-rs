//! Caller-fed CPU decoding with explicit acceptance, receive and EOF states.
//!
//! Packets are complete access units in the supplied codec bitstream format;
//! arbitrary byte chunks require a parser outside this API. Accepted input is
//! copied into FFmpeg-owned padded packet storage. Output frames own host bytes.
//! There are no Rust packet/frame queues; codec pools remain bounded best effort
//! by worker count and codec delay and are not whole-process memory accounting.
use crate::{
    codec::{self, VideoDecodeOptions},
    format::{self, PacketWithSideData, backend_error},
};
use ffmpeg_next as ffmpeg;
use media_core::{
    error::{Error, Result},
    media::{FrameTiming, Orientation, Packet, TrackId, VideoFrame},
    policy::FrameDurationPolicy,
    time::{Rounding, Time, TimeBase},
};

/// Backend-specific owned FFmpeg codec configuration, including extradata.
pub use ffmpeg::codec::Parameters as CodecParameters;
/// Owned codec configuration and the explicit source timestamp domain.
pub struct VideoDecoderConfig {
    /// FFmpeg codec parameters, including the original bitstream/extradata format.
    pub parameters: CodecParameters,
    /// Logical identity required on each compressed packet.
    pub track: TrackId,
    /// Time base used by the decoder; rational input times must fit exactly.
    pub time_base: TimeBase,
    /// Common asset origin subtracted from source PTS, without per-track rebasing.
    pub timeline_origin: Option<Time>,
    /// Established stream display orientation when frames omit their own matrix.
    pub orientation: Orientation,
    /// Established global presentation ordinal of the first supplied frame.
    /// None is required for arbitrary midstream/seek input; a local decoder
    /// counter does not establish a source-global frame identity.
    pub first_ordinal: Option<u64>,
}
impl std::fmt::Debug for VideoDecoderConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VideoDecoderConfig")
            .field("codec", &self.parameters.id().name())
            .field("track", &self.track)
            .field("time_base", &self.time_base)
            .field("timeline_origin", &self.timeline_origin)
            .field("orientation", &self.orientation)
            .finish()
    }
}
/// Per-packet and retained-frame limits, with explicit CPU representation.
#[derive(Debug, Clone, Copy)]
pub struct PacketDecodeOptions {
    /// Existing CPU output/color/worker/frame bounds. Probe controls are unused.
    pub video: VideoDecodeOptions,
    /// Maximum compressed access-unit payload copied on any send attempt.
    pub max_packet_bytes: usize,
    /// Maximum owned codec extradata copied during decoder setup.
    pub max_codec_config_bytes: usize,
    /// Maximum ICC frame side-data copied into portable image metadata.
    pub max_frame_metadata_bytes: usize,
}
impl Default for PacketDecodeOptions {
    fn default() -> Self {
        Self {
            video: Default::default(),
            max_packet_bytes: 16 * 1024 * 1024,
            max_codec_config_bytes: 4 * 1024 * 1024,
            max_frame_metadata_bytes: 4 * 1024 * 1024,
        }
    }
}
/// Whether input was consumed; NeedDrain leaves caller ownership unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodecSend {
    /// The codec accepted this complete packet or the explicit EOF marker.
    Accepted,
    /// Receive output through NeedInput before retrying the same input.
    NeedDrain,
    /// EOF was already submitted; further packets are not accepted.
    End,
}
/// One receive step; one packet may produce zero, one, or multiple frames.
#[derive(Debug)]
#[allow(
    clippy::large_enum_variant,
    reason = "Keep each hot frame result inline instead of allocating another box per receive"
)]
pub enum FrameReceive {
    /// Independently retained host frame with exact available timing.
    Frame(VideoFrame),
    /// More input is needed; receive again only after an accepted send.
    NeedInput,
    /// Decoder EOF was fully drained, including delayed B frames.
    End,
}
/// Independent mutable CPU decoder driven by complete caller-owned packets.
pub struct PacketDecoder {
    decoder: ffmpeg::decoder::Video,
    config: VideoDecoderConfig,
    options: PacketDecodeOptions,
    must_receive: bool,
    draining: bool,
    finished: bool,
    failed: bool,
    first_format: Option<(u32, u32, ffmpeg::format::Pixel)>,
    last_dts: Option<Time>,
    next_ordinal: Option<u64>,
    packets_sent: u64,
    frames_decoded: u64,
    assumptions: Vec<String>,
}
impl PacketDecoder {
    /// Open native codec workers without a demuxer or hidden source cursor.
    pub fn new(config: VideoDecoderConfig, options: PacketDecodeOptions) -> Result<Self> {
        format::initialize()?;
        let parameters = clone_parameters(&config.parameters, options.max_codec_config_bytes)?;
        if parameters.medium() != ffmpeg::media::Type::Video
            || options.video.threads == 0
            || options.video.threads > 256
            || options.video.max_frame_bytes == 0
            || options.max_packet_bytes == 0
            || options.max_codec_config_bytes == 0
            || options.max_frame_metadata_bytes == 0
        {
            return Err(Error::invalid(
                "packet decoder requires video parameters and nonzero bounded workers/payloads",
            ));
        }
        if options
            .video
            .track
            .is_some_and(|track| track != config.track)
        {
            return Err(Error::invalid(
                "packet decoder track option contradicts its codec configuration",
            ));
        }
        let base = native_base(config.time_base)?;
        let mut context =
            ffmpeg::codec::context::Context::from_parameters(parameters).map_err(backend_error)?;
        context.set_threading(ffmpeg::threading::Config::count(options.video.threads));
        let mut decoder = context.decoder();
        decoder.set_packet_time_base(base);
        let decoder = decoder.video().map_err(backend_error)?;
        let next_ordinal = config.first_ordinal;
        Ok(Self {
            decoder,
            config,
            options,
            must_receive: false,
            draining: false,
            finished: false,
            failed: false,
            first_format: None,
            last_dts: None,
            next_ordinal,
            packets_sent: 0,
            frames_decoded: 0,
            assumptions: Vec::new(),
        })
    }
    /// Submit a complete portable packet. No packet is consumed on NeedDrain.
    pub fn send_packet(&mut self, packet: &Packet) -> Result<CodecSend> {
        if self.failed {
            return Err(Error::invalid(
                "packet decoder failed; flush or recreate it before reuse",
            ));
        }
        if self.draining || self.finished {
            return Ok(CodecSend::End);
        }
        if packet.track != self.config.track {
            return Err(Error::incompatible(
                "compressed packet belongs to a different configured track",
            ));
        }
        if packet.data.is_empty() {
            return Err(Error::invalid(
                "empty compressed packets are not EOF; call send_eof explicitly",
            ));
        }
        if packet.data.len() > self.options.max_packet_bytes {
            return Err(
                Error::exhausted("compressed packet exceeds max_packet_bytes")
                    .with_constraint("max_packet_bytes"),
            );
        }
        if packet
            .dts
            .zip(self.last_dts)
            .is_some_and(|(next, last)| next < last)
        {
            return Err(Error::incompatible(
                "compressed packet DTS moved backwards in decode order",
            ));
        }
        let pts = packet
            .pts
            .map(|time| exact_ticks(time, self.config.time_base))
            .transpose()?;
        let dts = packet
            .dts
            .map(|time| exact_ticks(time, self.config.time_base))
            .transpose()?;
        let duration = packet
            .duration
            .map(|time| {
                if time < Time::ZERO {
                    return Err(Error::invalid("compressed packet duration is negative"));
                }
                exact_ticks(time, self.config.time_base)
            })
            .transpose()?
            .unwrap_or(0);
        if self.must_receive {
            return Ok(CodecSend::NeedDrain);
        }
        let mut native = ffi::packet_copy(&packet.data)?;
        native.set_stream(self.config.track.0 as usize);
        native.set_pts(pts);
        native.set_dts(dts);
        native.set_duration(duration);
        if packet.keyframe {
            native.set_flags(ffmpeg::codec::packet::Flags::KEY);
        }
        match self.decoder.send_packet(&native) {
            Ok(()) => {
                self.must_receive = true;
                self.last_dts = packet.dts.or(self.last_dts);
                self.packets_sent = self
                    .packets_sent
                    .checked_add(1)
                    .ok_or_else(|| Error::overflow("accepted packet counter"))?;
                Ok(CodecSend::Accepted)
            }
            Err(error) if is_again(error) => Ok(CodecSend::NeedDrain),
            Err(error) => {
                self.failed = true;
                Err(backend_error(error).with_operation("external packet send"))
            }
        }
    }
    /// Full packet projection refuses unsupported per-packet side data and flags.
    /// Configuration updates need explicit decoder renegotiation, never silent loss.
    pub fn send_packet_with_side_data(&mut self, packet: &PacketWithSideData) -> Result<CodecSend> {
        if (packet.flags & ffmpeg::codec::packet::Flags::KEY.bits() != 0) != packet.packet.keyframe
        {
            return Err(Error::incompatible(
                "native and portable compressed packet keyframe flags disagree",
            ));
        }
        if !packet.side_data.is_empty()
            || packet.flags & !ffmpeg::codec::packet::Flags::KEY.bits() != 0
        {
            return Err(Error::unsupported(
                "external CPU packet decode cannot drop native packet side data/flags; renegotiate or use native FFmpeg packet integration",
            ));
        }
        self.send_packet(&packet.packet)
    }
    /// Submit EOF after pending output was received; then receive through End.
    pub fn send_eof(&mut self) -> Result<CodecSend> {
        if self.failed {
            return Err(Error::invalid(
                "failed packet decoder cannot drain successfully",
            ));
        }
        if self.draining || self.finished {
            return Ok(CodecSend::End);
        }
        if self.must_receive {
            return Ok(CodecSend::NeedDrain);
        }
        match self.decoder.send_eof() {
            Ok(()) => {
                self.draining = true;
                self.must_receive = true;
                Ok(CodecSend::Accepted)
            }
            Err(error) if is_again(error) => Ok(CodecSend::NeedDrain),
            Err(error) => {
                self.failed = true;
                Err(backend_error(error))
            }
        }
    }
    /// Receive one frame; this never sleeps or polls when input is required.
    pub fn receive_frame(&mut self) -> Result<FrameReceive> {
        if self.failed {
            return Err(Error::invalid(
                "packet decoder failed; flush or recreate it",
            ));
        }
        if self.finished {
            return Ok(FrameReceive::End);
        }
        let result = self.receive_inner();
        if result.is_err() {
            self.failed = true;
        }
        result
    }
    fn receive_inner(&mut self) -> Result<FrameReceive> {
        let mut decoded = ffmpeg::frame::Video::empty();
        match self.decoder.receive_frame(&mut decoded) {
            Ok(()) => {}
            Err(ffmpeg::Error::Eof) if self.draining => {
                self.finished = true;
                return Ok(FrameReceive::End);
            }
            Err(error) if is_again(error) && !self.draining => {
                self.must_receive = false;
                return Ok(FrameReceive::NeedInput);
            }
            Err(error) => {
                return Err(backend_error(error).with_operation("external packet receive"));
            }
        }
        self.frames_decoded = self
            .frames_decoded
            .checked_add(1)
            .ok_or_else(|| Error::overflow("decoded frame counter"))?;
        if decoded.is_corrupt() || decoded.has_decode_errors() {
            return Err(Error::backend(
                "ffmpeg",
                "external packet decoder returned corrupt/incomplete pixels",
            ));
        }
        let description = (decoded.width(), decoded.height(), decoded.format());
        if self.first_format.is_some_and(|first| first != description) {
            return Err(Error::incompatible(
                "packet decoder format changed; create a decoder with renegotiated parameters",
            ));
        }
        if decoded
            .side_data(ffmpeg::frame::side_data::Type::IccProfile)
            .is_some_and(|data| data.data().len() > self.options.max_frame_metadata_bytes)
        {
            return Err(Error::exhausted(
                "decoded ICC metadata exceeds configured frame metadata bound",
            ));
        }
        let image = codec::materialize_image(
            &decoded,
            self.options.video,
            self.config.orientation,
            &mut self.assumptions,
        )?;
        let source_pts = decoded
            .pts()
            .map(|ticks| self.config.time_base.time(ticks))
            .transpose()?;
        let pts = source_pts
            .map(|pts| {
                self.config
                    .timeline_origin
                    .map_or(Ok(pts), |origin| pts.checked_sub(origin))
            })
            .transpose()?;
        let native_duration = (decoded.packet().duration > 0)
            .then(|| self.config.time_base.time(decoded.packet().duration))
            .transpose()?;
        let ordinal = self.next_ordinal;
        self.next_ordinal = ordinal.and_then(|ordinal| ordinal.checked_add(1));
        self.first_format = Some(description);
        Ok(FrameReceive::Frame(VideoFrame {
            image,
            track: self.config.track,
            timing: FrameTiming {
                pts,
                source_pts,
                native_duration,
                duration: if self.options.video.duration_policy
                    == FrameDurationPolicy::TrustReported
                {
                    native_duration
                } else {
                    None
                },
                ordinal,
            },
        }))
    }
    /// Explicitly discard buffered frames after navigation/error recovery.
    /// This is not an EOF drain; future ordinals become unknown.
    pub fn flush(&mut self) {
        self.decoder.flush();
        self.must_receive = false;
        self.draining = false;
        self.finished = false;
        self.failed = false;
        self.first_format = None;
        self.last_dts = None;
        self.next_ordinal = None;
    }
    /// Actual accepted access units across flush generations.
    pub fn packets_sent(&self) -> u64 {
        self.packets_sent
    }
    /// Actual frames received, including a frame later rejected by validation.
    pub fn frames_decoded(&self) -> u64 {
        self.frames_decoded
    }
    /// Explicit missing-color assumptions actually applied during RGB conversion.
    pub fn color_assumptions(&self) -> &[String] {
        &self.assumptions
    }
}
pub(crate) fn native_base(base: TimeBase) -> Result<ffmpeg::Rational> {
    Ok(ffmpeg::Rational(
        i32::try_from(base.numerator())
            .map_err(|_| Error::overflow("native time-base numerator"))?,
        i32::try_from(base.denominator())
            .map_err(|_| Error::overflow("native time-base denominator"))?,
    ))
}
pub(crate) fn exact_ticks(time: Time, base: TimeBase) -> Result<i64> {
    let ticks = time.to_ticks(base, Rounding::NearestTiesEven)?;
    if base.time(ticks)? != time {
        return Err(Error::incompatible(
            "timestamp is not exactly representable in the configured time base",
        ));
    }
    Ok(ticks)
}
fn is_again(error: ffmpeg::Error) -> bool {
    matches!(error,ffmpeg::Error::Other{errno} if errno==ffmpeg::error::EAGAIN)
}
pub(crate) fn clone_parameters(
    parameters: &CodecParameters,
    limit: usize,
) -> Result<CodecParameters> {
    ffi::clone_parameters(parameters, limit)
}
#[allow(unsafe_code)]
mod ffi {
    use super::*;
    use ffmpeg::packet::Mut;
    pub(super) fn clone_parameters(
        parameters: &CodecParameters,
        limit: usize,
    ) -> Result<CodecParameters> {
        // SAFETY: the borrowed wrapper retains its parameters. Check nullable
        // allocation and extradata extent before any native read/copy allocation.
        unsafe {
            let source = parameters.as_ptr();
            if source.is_null() {
                return Err(Error::exhausted(
                    "native codec parameter allocation is null",
                ));
            }
            if (*source).extradata_size < 0 || (*source).extradata_size as usize > limit {
                return Err(Error::exhausted(
                    "codec configuration exceeds explicit metadata bound",
                ));
            }
            if (*source).extradata_size > 0 && (*source).extradata.is_null() {
                return Err(Error::invalid(
                    "nonempty native codec configuration is null",
                ));
            }
            let raw = ffmpeg::ffi::avcodec_parameters_alloc();
            if raw.is_null() {
                return Err(Error::exhausted("cannot allocate owned codec parameters"));
            }
            let mut copy = ffmpeg::codec::Parameters::wrap(raw, None);
            let status = ffmpeg::ffi::avcodec_parameters_copy(copy.as_mut_ptr(), source);
            if status < 0 {
                return Err(backend_error(ffmpeg::Error::from(status)));
            }
            Ok(copy)
        }
    }
    pub(super) fn packet_copy(bytes: &[u8]) -> Result<ffmpeg::Packet> {
        let length = i32::try_from(bytes.len())
            .map_err(|_| Error::overflow("compressed packet exceeds native size range"))?;
        let mut packet = ffmpeg::Packet::empty();
        // SAFETY: uniquely owned fresh AVPacket. av_new_packet allocates the
        // FFmpeg-required zero padding and reports failure before data is touched.
        let status = unsafe { ffmpeg::ffi::av_new_packet(packet.as_mut_ptr(), length) };
        if status < 0 {
            return Err(backend_error(ffmpeg::Error::from(status)));
        }
        packet
            .data_mut()
            .ok_or_else(|| Error::backend("ffmpeg", "allocated packet lacks data"))?
            .copy_from_slice(bytes);
        Ok(packet)
    }
}
