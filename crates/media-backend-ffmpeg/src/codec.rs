//! Incremental FFmpeg video decode with explicit output representation and drain.
use std::sync::atomic::Ordering;

use crate::format::{self, AssetInfo, OpenedInput, ProbeOptions, TrackKind, backend_error};
use crate::native;
use ffmpeg_next as ffmpeg;
use media_core::error::{Error, Result};
use media_core::media::{
    ColorDescription, FrameTiming, Image, ImageLayout, ImageMetadata, Orientation, PixelFormat,
    PlaneLayout, TrackId, VideoFrame,
};
use media_core::storage::StorageLease;
use media_core::time::{Rounding, Time, TimeBase};
use media_io::source::Source;

/// CPU output representation. Neither choice claims GPU decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoOutput {
    /// Explicitly convert decoded pixels to packed eight-bit RGB.
    Rgb8,
    /// Preserve supported decoded packed/planar pixel formats on the host.
    ///
    /// This path copies visible plane bytes into retained host storage and does
    /// not claim no-copy surface sharing. Unsupported native formats fail.
    Native,
}
/// Explicit assumption when YUV matrix/range metadata is missing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorAssumption {
    /// Assume BT.709 matrix and limited range for missing fields only.
    Bt709Limited,
    /// Assume BT.601 matrix and limited range for missing fields only.
    Bt601Limited,
}
/// Per-reader controls; each open creates independent decoder/cursor state.
#[derive(Debug, Clone, Copy)]
pub struct VideoDecodeOptions {
    /// Whether to explicitly interpret native duration hints as display intervals.
    pub duration_policy: media_core::policy::FrameDurationPolicy,
    /// Specific logical video stream; None chooses default non-cover video.
    pub track: Option<TrackId>,
    /// Requested CPU representation.
    pub output: VideoOutput,
    /// Maximum output bytes per frame, excluding opaque native decoder pools.
    pub max_frame_bytes: u64,
    /// Codec worker count, not an unbounded automatic thread allocation.
    pub threads: usize,
    /// Missing YUV matrix/range policy. None requires established fields.
    pub color_assumption: Option<ColorAssumption>,
    /// Bounded stream discovery controls.
    pub probe: ProbeOptions,
}
/// Actual decoded timing/keyframe evidence without RGB pixel materialization.
#[derive(Debug, Clone, Copy)]
pub struct DecodedTiming {
    /// Source/asset timestamps and sequential presentation ordinal.
    pub timing: FrameTiming,
    /// Native decoded key-frame marker; not inferred from packet count or FPS.
    pub keyframe: bool,
    /// Original compressed key packet DTS, used for demux navigation. It may
    /// precede the decoded frame PTS when the stream contains B frames.
    pub seek_timestamp: Option<Time>,
}

#[derive(Default)]
pub(crate) struct KeyPacketTimes(std::collections::BTreeMap<Time, Time>);
impl KeyPacketTimes {
    pub(crate) fn observe(&mut self, packet: &ffmpeg::Packet, base: TimeBase) -> Result<()> {
        if packet.is_key()
            && let (Some(pts), Some(dts)) = (packet.pts(), packet.dts())
        {
            if self.0.len() >= 1024 {
                return Err(Error::exhausted(
                    "too many pending video key packet timestamps",
                ));
            }
            self.0.insert(base.time(pts)?, base.time(dts)?);
        }
        Ok(())
    }
    pub(crate) fn take(&mut self, pts: Option<Time>) -> Option<Time> {
        pts.and_then(|pts| self.0.remove(&pts))
    }
    pub(crate) fn clear(&mut self) {
        self.0.clear();
    }
}
impl Default for VideoDecodeOptions {
    fn default() -> Self {
        Self {
            duration_policy: media_core::policy::FrameDurationPolicy::Unknown,
            track: None,
            output: VideoOutput::Rgb8,
            max_frame_bytes: 256 * 1024 * 1024,
            threads: 1,
            color_assumption: Some(ColorAssumption::Bt709Limited),
            probe: ProbeOptions::default(),
        }
    }
}
/// Incremental reader that drains delayed frames before reporting EOF.
///
/// Source timestamps are preserved, and asset PTS subtracts the established
/// common container origin. Frames own their bytes independently of the reader.
pub struct VideoReader {
    opened: OpenedInput,
    info: AssetInfo,
    decoder: ffmpeg::decoder::Video,
    track: TrackId,
    base: TimeBase,
    options: VideoDecodeOptions,
    seekable: bool,
    draining: bool,
    finished: bool,
    next_ordinal: Option<u64>,
    frames_decoded: u64,
    packets_sent: u64,
    first_format: Option<(u32, u32, ffmpeg::format::Pixel)>,
    assumptions: Vec<String>,
    orientation: Orientation,
    last_keyframe: Option<bool>,
    key_packet_times: KeyPacketTimes,
    last_seek_timestamp: Option<Time>,
}
impl VideoReader {
    /// Open one video track, with bounded probing and independent mutable state.
    pub fn open(source: &Source, options: VideoDecodeOptions) -> Result<Self> {
        Self::open_impl(source, options, false, None)
    }
    /// Open container headers without stream-info discovery for an explicitly
    /// source-validated index. Incomplete header codec configuration is an error;
    /// no full probe or prefix indexing is performed implicitly.
    pub fn open_indexed(source: &Source, options: VideoDecodeOptions) -> Result<Self> {
        Self::open_impl(source, options, true, None)
    }
    /// Open with cooperative AVIO/demux cancellation. `indexed` skips stream-info
    /// discovery and is intended only with source-validated external evidence.
    pub fn open_controlled(
        source: &Source,
        options: VideoDecodeOptions,
        indexed: bool,
        interrupt: format::ReadInterrupt,
    ) -> Result<Self> {
        Self::open_impl(source, options, indexed, Some(interrupt))
    }
    fn open_impl(
        source: &Source,
        options: VideoDecodeOptions,
        indexed: bool,
        interrupt: Option<format::ReadInterrupt>,
    ) -> Result<Self> {
        if options.max_frame_bytes == 0 || options.threads == 0 || options.threads > 256 {
            return Err(Error::invalid(
                "video needs a nonzero frame budget and 1..=256 codec threads",
            ));
        }
        let opened = if let Some(interrupt) = interrupt {
            format::open_source_controlled(source, options.probe, indexed, !indexed, interrupt)?
        } else if indexed {
            format::open_source_indexed(source, options.probe)?
        } else {
            format::open_source(source, options.probe)?
        };
        let info = format::snapshot(
            &opened.context,
            source,
            opened.counter.total.load(Ordering::Relaxed),
        )?;
        let selected = match options.track {
            Some(id) => info
                .tracks
                .iter()
                .find(|t| t.id == id && t.kind == TrackKind::Video),
            None => info
                .tracks
                .iter()
                .find(|t| t.kind == TrackKind::Video && !t.cover_art),
        }
        .ok_or_else(|| Error::unsupported("source has no selected non-cover video track"))?;
        let track = selected.id;
        let base = selected.time_base;
        let orientation = selected
            .display_matrix
            .as_ref()
            .map(|matrix| native::display_orientation(matrix))
            .transpose()?
            .unwrap_or(Orientation::Normal);
        if options.output == VideoOutput::Rgb8
            && let (Some(w), Some(h)) = (selected.width, selected.height)
        {
            checked_frame_bytes(w, h, 3, options.max_frame_bytes)?;
        }
        let stream = opened
            .context
            .stream(track.0 as usize)
            .ok_or_else(|| Error::invalid("selected video track is absent"))?;
        let mut context = ffmpeg::codec::context::Context::from_parameters(stream.parameters())
            .map_err(backend_error)?;
        context.set_threading(ffmpeg::threading::Config::count(options.threads));
        let decoder = context.decoder().video().map_err(backend_error)?;
        opened.allow_reading();
        Ok(Self {
            opened,
            info,
            decoder,
            track,
            base,
            options,
            seekable: source.capabilities()?.seekable,
            draining: false,
            finished: false,
            next_ordinal: Some(0),
            frames_decoded: 0,
            packets_sent: 0,
            first_format: None,
            assumptions: Vec::new(),
            orientation,
            last_keyframe: None,
            key_packet_times: KeyPacketTimes::default(),
            last_seek_timestamp: None,
        })
    }
    /// Probed tracks and common timeline mapping.
    pub fn info(&self) -> &AssetInfo {
        &self.info
    }
    /// Selected logical stream.
    pub fn track(&self) -> TrackId {
        self.track
    }
    /// Decoded frames across all cursor generations, including after seeks.
    pub fn frames_decoded(&self) -> u64 {
        self.frames_decoded
    }
    /// Compressed selected-track packets submitted to the decoder.
    pub fn packets_sent(&self) -> u64 {
        self.packets_sent
    }
    /// Actual encoded source bytes read, including probing and rereads.
    pub fn bytes_read(&self) -> u64 {
        self.opened.counter.total.load(Ordering::Relaxed)
    }
    /// Explicit color assumptions actually used, never inferred support claims.
    pub fn color_assumptions(&self) -> &[String] {
        &self.assumptions
    }
    /// Explicit interpretation applied to backend-reported duration hints.
    pub fn duration_policy(&self) -> media_core::policy::FrameDurationPolicy {
        self.options.duration_policy
    }
    /// Hard total source-byte cap, including bytes already read during probing.
    pub fn set_read_limit(&self, bytes: u64) -> Result<()> {
        self.opened.set_read_limit(bytes)
    }
    /// Start a bounded request on this live cursor. Already-read setup bytes are
    /// excluded; cancellation and earlier read-budget failures remain terminal.
    pub fn begin_read_batch(&self, bytes: u64) -> Result<()> {
        self.opened.begin_read_batch(bytes)
    }
    /// Replace the cancellation predicate checked between native/demux reads.
    pub fn set_interrupt(&self, interrupt: format::ReadInterrupt) {
        self.opened.set_interrupt(interrupt);
    }
    /// Set a caller-established source-to-asset mapping for an explicitly
    /// validated index. Raw native source timestamps remain unchanged.
    pub fn set_timeline_origin(&mut self, origin: Time) {
        self.info.timeline_origin = Some(origin);
    }
    /// Keyframe evidence for the most recently emitted frame/timing record.
    pub fn last_keyframe(&self) -> Option<bool> {
        self.last_keyframe
    }
    /// Decode one presentation-ordered timing record without copying/converting
    /// output pixels. Full index scans require an explicit caller policy/budget.
    pub fn next_timing(&mut self) -> Result<Option<DecodedTiming>> {
        let Some(decoded) = self.receive_decoded()? else {
            return Ok(None);
        };
        let timing = self.describe_frame(&decoded)?;
        Ok(Some(DecodedTiming {
            timing,
            keyframe: decoded.is_key(),
            seek_timestamp: self.last_seek_timestamp,
        }))
    }
    /// Read one presentation-ordered frame or a fully drained EOF.
    pub fn next_frame(&mut self) -> Result<Option<VideoFrame>> {
        let Some(decoded) = self.receive_decoded()? else {
            return Ok(None);
        };
        self.materialize(decoded).map(Some)
    }
    fn receive_decoded(&mut self) -> Result<Option<ffmpeg::frame::Video>> {
        self.opened.check_read_budget()?;
        if self.finished {
            return Ok(None);
        }
        loop {
            let mut decoded = ffmpeg::frame::Video::empty();
            match self.decoder.receive_frame(&mut decoded) {
                Ok(()) => return Ok(Some(decoded)),
                Err(ffmpeg::Error::Eof) => {
                    self.opened.check_read_budget()?;
                    self.finished = true;
                    return Ok(None);
                }
                Err(error) if is_again(error) => {}
                Err(error) => {
                    return Err(backend_error(error)
                        .with_track(self.track.0)
                        .with_operation("decode receive"));
                }
            }
            if self.draining {
                return Err(Error::backend(
                    "ffmpeg",
                    "decoder requested input after EOF drain was submitted",
                )
                .with_track(self.track.0));
            }
            let mut packet = ffmpeg::Packet::empty();
            loop {
                self.opened.check_read_budget()?;
                match packet.read(&mut self.opened.context) {
                    Ok(()) if packet.stream() == self.track.0 as usize => break,
                    Ok(()) => continue,
                    Err(ffmpeg::Error::Eof) => {
                        self.opened.check_read_budget()?;
                        self.decoder.send_eof().map_err(backend_error)?;
                        self.draining = true;
                        break;
                    }
                    Err(error) => {
                        self.opened.check_read_budget()?;
                        return Err(backend_error(error).with_operation("demux for video"));
                    }
                }
            }
            if !self.draining {
                // receive returned EAGAIN immediately above. FFmpeg promises
                // send cannot also return EAGAIN in that state, so no accepted
                // packet is lost and there is no retry/sleep loop.
                self.decoder.send_packet(&packet).map_err(backend_error)?;
                self.key_packet_times.observe(&packet, self.base)?;
                self.packets_sent = self
                    .packets_sent
                    .checked_add(1)
                    .ok_or_else(|| Error::overflow("video packet count overflow"))?;
            }
        }
    }
    /// Navigate to a random-access point at or before an asset-timeline target.
    ///
    /// This is navigation only: decode and apply a temporal selection policy to
    /// establish the exact displayed/nearest frame. Ordinals become unknown
    /// after seeking because no verified presentation index was constructed.
    pub fn seek(&mut self, target: Time) -> Result<()> {
        if !self.seekable {
            return Err(Error::unsupported(
                "random access on a sequential source requires explicit spooling",
            ));
        }
        let source_target = match self.info.timeline_origin {
            Some(origin) => target.checked_add(origin)?,
            None => target,
        };
        let ticks = source_target.to_ticks(TimeBase::new(1, 1_000_000)?, Rounding::Floor)?;
        self.opened
            .context
            .seek(ticks, ..ticks)
            .map_err(backend_error)?;
        self.decoder.flush();
        self.draining = false;
        self.finished = false;
        self.next_ordinal = None;
        self.first_format = None;
        self.key_packet_times.clear();
        self.last_seek_timestamp = None;
        Ok(())
    }
    /// Seek using exact selected-stream source ticks; output ordinals remain
    /// unknown until a source-validated presentation index aligns actual output.
    pub fn seek_source(&mut self, target: Time) -> Result<()> {
        if !self.seekable {
            return Err(Error::unsupported(
                "indexed video seeking requires a seekable source or explicit spooling",
            ));
        }
        let ticks = target.to_ticks(self.base, Rounding::NearestTiesEven)?;
        if self.base.time(ticks)? != target {
            return Err(Error::incompatible(
                "index source PTS is not exactly representable in this stream time base",
            ));
        }
        let seek = native::seek_video_stream(&mut self.opened.context, self.track.0, ticks);
        self.opened.check_read_budget()?;
        seek?;
        self.decoder.flush();
        self.draining = false;
        self.finished = false;
        self.next_ordinal = None;
        self.first_format = None;
        self.last_keyframe = None;
        self.key_packet_times.clear();
        self.last_seek_timestamp = None;
        Ok(())
    }
    /// Navigate a verified keyframe using the demuxer's timestamp domain. PTS
    /// and optional DTS must come from the actual frame/key packet, respectively.
    pub fn seek_indexed_keyframe(&mut self, pts: Time, packet_dts: Option<Time>) -> Result<()> {
        self.seek_source(format::indexed_seek_time(
            &self.info.format,
            pts,
            packet_dts,
        ))
    }
    fn describe_frame(&mut self, decoded: &ffmpeg::frame::Video) -> Result<FrameTiming> {
        if decoded.is_corrupt() || decoded.has_decode_errors() {
            return Err(Error::backend(
                "ffmpeg",
                "decoder returned a corrupt or incomplete video frame",
            )
            .with_track(self.track.0));
        }
        let description = (decoded.width(), decoded.height(), decoded.format());
        if self.first_format.is_some_and(|first| first != description) {
            return Err(Error::incompatible(
                "video format changed midstream; this reader requires explicit renegotiation",
            )
            .with_track(self.track.0));
        }
        self.first_format = Some(description);
        let source_pts = decoded.pts().map(|v| self.base.time(v)).transpose()?;
        let pts = source_pts
            .map(|t| match self.info.timeline_origin {
                Some(origin) => t.checked_sub(origin),
                None => Ok(t),
            })
            .transpose()?;
        let native_duration = if decoded.packet().duration > 0 {
            Some(self.base.time(decoded.packet().duration)?)
        } else {
            None
        };
        let ordinal = self.next_ordinal;
        self.next_ordinal = ordinal.and_then(|n| n.checked_add(1));
        self.frames_decoded = self
            .frames_decoded
            .checked_add(1)
            .ok_or_else(|| Error::overflow("video frame count overflow"))?;
        self.last_keyframe = Some(decoded.is_key());
        self.last_seek_timestamp = self.key_packet_times.take(source_pts);
        Ok(FrameTiming {
            pts,
            duration: match self.options.duration_policy {
                media_core::policy::FrameDurationPolicy::Unknown => None,
                media_core::policy::FrameDurationPolicy::TrustReported => native_duration,
            },
            native_duration,
            source_pts,
            ordinal,
        })
    }
    fn materialize(&mut self, decoded: ffmpeg::frame::Video) -> Result<VideoFrame> {
        let timing = self.describe_frame(&decoded)?;
        let image = materialize_image(
            &decoded,
            self.options,
            self.orientation,
            &mut self.assumptions,
        )?;
        Ok(VideoFrame {
            image,
            track: self.track,
            timing,
        })
    }
}
pub(crate) fn materialize_image(
    decoded: &ffmpeg::frame::Video,
    options: VideoDecodeOptions,
    orientation: Orientation,
    assumptions: &mut Vec<String>,
) -> Result<Image> {
    let mut metadata = frame_metadata(decoded)?;
    if decoded
        .side_data(ffmpeg::util::frame::side_data::Type::DisplayMatrix)
        .is_none()
    {
        metadata.orientation = orientation;
    }
    let image = match options.output {
        VideoOutput::Native => native_image(decoded, metadata, options.max_frame_bytes)?,
        VideoOutput::Rgb8 => {
            checked_frame_bytes(
                decoded.width(),
                decoded.height(),
                3,
                options.max_frame_bytes,
            )?;
            let mut scaler = ffmpeg::software::scaling::Context::get(
                decoded.format(),
                decoded.width(),
                decoded.height(),
                ffmpeg::format::Pixel::RGB24,
                decoded.width(),
                decoded.height(),
                ffmpeg::software::scaling::Flags::BILINEAR,
            )
            .map_err(backend_error)?;
            let (matrix, range, assumed) =
                conversion_coefficients(decoded, options.color_assumption)?;
            if let Some(text) = assumed
                && !assumptions.contains(&text)
            {
                assumptions.push(text);
            }
            if let Some(matrix) = matrix {
                native::configure_scaler(&mut scaler, matrix, range)?;
            }
            let rgb = native::scale_checked(&mut scaler, decoded)?;
            let bytes = native::copy_plane(
                &rgb,
                0,
                decoded.width() as usize * 3,
                decoded.height() as usize,
            )?;
            metadata.color.matrix = Some("rgb".into());
            metadata.color.range = Some("full".into());
            metadata.color.chroma_siting = None;
            Image::from_packed(decoded.width(), decoded.height(), PixelFormat::Rgb8, bytes)?
                .with_metadata(metadata)
        }
    };
    Ok(image)
}
fn is_again(error: ffmpeg::Error) -> bool {
    matches!(error,ffmpeg::Error::Other{errno} if errno==ffmpeg::error::EAGAIN)
}
fn checked_frame_bytes(width: u32, height: u32, bpp: u64, limit: u64) -> Result<usize> {
    if width == 0 || height == 0 {
        return Err(Error::invalid("video frame has empty geometry"));
    }
    let bytes = u64::from(width)
        .checked_mul(u64::from(height))
        .and_then(|n| n.checked_mul(bpp))
        .ok_or_else(|| Error::overflow("video frame byte count overflow"))?;
    if bytes > limit {
        return Err(Error::exhausted(
            "video frame exceeds configured decoded-byte limit",
        ));
    }
    usize::try_from(bytes).map_err(|_| Error::overflow("frame does not fit address space"))
}
pub(crate) fn frame_metadata(frame: &ffmpeg::frame::Video) -> Result<ImageMetadata> {
    let orientation = frame
        .side_data(ffmpeg::util::frame::side_data::Type::DisplayMatrix)
        .map(|data| native::display_orientation(data.data()))
        .transpose()?
        .unwrap_or(Orientation::Normal);
    let icc_profile = frame
        .side_data(ffmpeg::util::frame::side_data::Type::IccProfile)
        .map(|data| data.data().to_vec());
    let exif = None;
    Ok(ImageMetadata {
        orientation,
        icc_profile,
        exif,
        color: ColorDescription {
            primaries: frame.color_primaries().name().map(str::to_owned),
            transfer: frame
                .color_transfer_characteristic()
                .name()
                .map(str::to_owned),
            matrix: frame.color_space().name().map(str::to_owned),
            range: frame.color_range().name().map(str::to_owned),
            chroma_siting: match frame.chroma_location() {
                ffmpeg::chroma::Location::Unspecified => None,
                ffmpeg::chroma::Location::Left => Some("left".into()),
                ffmpeg::chroma::Location::Center => Some("center".into()),
                ffmpeg::chroma::Location::TopLeft => Some("topleft".into()),
                ffmpeg::chroma::Location::Top => Some("top".into()),
                ffmpeg::chroma::Location::BottomLeft => Some("bottomleft".into()),
                ffmpeg::chroma::Location::Bottom => Some("bottom".into()),
            },
        },
    })
}
fn conversion_coefficients(
    frame: &ffmpeg::frame::Video,
    assumption: Option<ColorAssumption>,
) -> Result<(Option<i32>, i32, Option<String>)> {
    use ffmpeg::color::{Range, Space};
    let mut used = false;
    let matrix = match frame.color_space() {
        Space::RGB => return Ok((None, 1, None)),
        Space::BT709 => ffmpeg::ffi::SWS_CS_ITU709,
        Space::BT470BG | Space::SMPTE170M => ffmpeg::ffi::SWS_CS_ITU601,
        Space::FCC => ffmpeg::ffi::SWS_CS_FCC,
        Space::SMPTE240M => ffmpeg::ffi::SWS_CS_SMPTE240M,
        Space::BT2020NCL => ffmpeg::ffi::SWS_CS_BT2020,
        Space::Unspecified => {
            used = true;
            match assumption {
                Some(ColorAssumption::Bt709Limited) => ffmpeg::ffi::SWS_CS_ITU709,
                Some(ColorAssumption::Bt601Limited) => ffmpeg::ffi::SWS_CS_ITU601,
                None => {
                    return Err(Error::incompatible(
                        "RGB conversion needs an established YUV matrix or explicit assumption",
                    ));
                }
            }
        }
        other => {
            return Err(Error::unsupported(format!(
                "RGB conversion for matrix {other:?} is not implemented"
            )));
        }
    };
    let range = match frame.color_range() {
        Range::MPEG => 0,
        Range::JPEG => 1,
        Range::Unspecified => {
            used = true;
            if assumption.is_none() {
                return Err(Error::incompatible(
                    "RGB conversion needs an established YUV range or explicit assumption",
                ));
            }
            0
        }
    };
    let text = used.then(|| format!("missing YUV fields used explicit {assumption:?} assumption"));
    Ok((Some(matrix), range, text))
}
fn native_image(
    frame: &ffmpeg::frame::Video,
    metadata: ImageMetadata,
    limit: u64,
) -> Result<Image> {
    use ffmpeg::format::Pixel as P;
    let (w, h) = (frame.width() as usize, frame.height() as usize);
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let (format, geometry) = match frame.format() {
        P::RGB24 => (PixelFormat::Rgb8, vec![(w.checked_mul(3), h)]),
        P::RGBA => (PixelFormat::Rgba8, vec![(w.checked_mul(4), h)]),
        P::GRAY8 => (PixelFormat::Gray8, vec![(Some(w), h)]),
        P::GRAY16LE => (PixelFormat::Gray16, vec![(w.checked_mul(2), h)]),
        P::RGB48LE => (PixelFormat::Rgb16, vec![(w.checked_mul(6), h)]),
        P::RGBA64LE => (PixelFormat::Rgba16, vec![(w.checked_mul(8), h)]),
        P::YUV420P | P::YUVJ420P => (
            PixelFormat::Yuv420p,
            vec![(Some(w), h), (Some(cw), ch), (Some(cw), ch)],
        ),
        P::NV12 => (
            PixelFormat::Nv12,
            vec![(Some(w), h), (cw.checked_mul(2), ch)],
        ),
        P::P010LE => (
            PixelFormat::P010,
            vec![(w.checked_mul(2), h), (cw.checked_mul(4), ch)],
        ),
        other => {
            return Err(Error::unsupported(format!(
                "native decoded pixel format {other:?} needs an explicit supported conversion"
            )));
        }
    };
    let mut total = 0usize;
    let mut planes = Vec::new();
    for (row, rows) in geometry {
        let row = row.ok_or_else(|| Error::overflow("native video row size overflow"))?;
        let length = row
            .checked_mul(rows)
            .ok_or_else(|| Error::overflow("native video plane size overflow"))?;
        planes.push(PlaneLayout {
            offset: total,
            row_stride: row,
            row_bytes: row,
            rows,
        });
        total = total
            .checked_add(length)
            .ok_or_else(|| Error::overflow("native video allocation overflow"))?;
    }
    if total as u64 > limit {
        return Err(Error::exhausted(
            "native video frame exceeds decoded-byte limit",
        ));
    }
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(total)
        .map_err(|e| Error::exhausted(e.to_string()))?;
    for (index, plane) in planes.iter().enumerate() {
        bytes.extend_from_slice(&native::copy_plane(
            frame,
            index,
            plane.row_bytes,
            plane.rows,
        )?);
    }
    Image::new(
        ImageLayout {
            width: frame.width(),
            height: frame.height(),
            allocated_width: frame.width(),
            allocated_height: frame.height(),
            format,
            planes,
        },
        StorageLease::from_bytes(bytes),
        metadata,
    )
}
