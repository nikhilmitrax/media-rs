//! Incremental native CPU video resize/encode with compressed audio passthrough.
//!
//! One demux cursor feeds bounded codec workers and the interleaving muxer. Video
//! PTS and audio PTS/DTS retain their common source clock; tracks are never each
//! rebased to zero. Output is finalized only after both codecs drain to EOF.
//! Resizing operates on stored eight-bit YUV components, with no ICC conversion,
//! linear-light filtering, tone mapping, or implicit high-bit-depth conversion.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::OpenOptions;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;

use ffmpeg_next as ffmpeg;
use media_core::error::{Error, Result};
use media_core::media::{Orientation, TrackId};
use media_core::time::{Rounding, Time, TimeBase};
use media_io::source::Source;

use crate::format::{self, ProbeOptions, TrackKind, backend_error};
use crate::native;

/// Explicit software encoder implementation. This does not select NVENC.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuVideoCodec {
    /// H.264 through the linked libx264 encoder.
    H264,
    /// HEVC Main through the linked libx265 encoder.
    Hevc,
}
impl CpuVideoCodec {
    pub(crate) fn encoder_name(self) -> &'static str {
        match self {
            Self::H264 => "libx264",
            Self::Hevc => "libx265",
        }
    }
    pub(crate) fn id(self) -> ffmpeg::codec::Id {
        match self {
            Self::H264 => ffmpeg::codec::Id::H264,
            Self::Hevc => ffmpeg::codec::Id::HEVC,
        }
    }
}
/// Component-domain resize reconstruction filter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResizeFilter {
    /// Nearest neighbour.
    Nearest,
    /// Bilinear reconstruction.
    Bilinear,
    /// Bicubic reconstruction.
    Bicubic,
    /// Lanczos reconstruction.
    Lanczos,
}
impl ResizeFilter {
    fn flags(self) -> ffmpeg::software::scaling::Flags {
        use ffmpeg::software::scaling::Flags;
        match self {
            Self::Nearest => Flags::POINT,
            Self::Bilinear => Flags::BILINEAR,
            Self::Bicubic => Flags::BICUBIC,
            Self::Lanczos => Flags::LANCZOS,
        }
    }
}
/// Explicit video encoding, selected audio stream-copy, and allocation controls.
#[derive(Debug, Clone)]
pub struct TranscodeOptions {
    /// Specific video track; None selects the first non-cover video track.
    pub video_track: Option<TrackId>,
    /// Audio tracks to copy. None copies all audio; Some(empty) copies none.
    pub audio_tracks: Option<Vec<TrackId>>,
    /// Native software encoder to use.
    pub codec: CpuVideoCodec,
    /// Exact output width. Width and height must both be set or both absent.
    pub width: Option<u32>,
    /// Exact output height. Four-two-zero output requires even dimensions.
    pub height: Option<u32>,
    /// Stored-component resize filter; no implicit color conversion.
    pub filter: ResizeFilter,
    /// Encoder target bitrate in bits per second.
    pub bit_rate: usize,
    /// H.264 baseline/main/high or HEVC main; None uses encoder negotiation.
    pub profile: Option<String>,
    /// Named x264/x265 speed preset, validated before creating output.
    pub preset: String,
    /// Maximum consecutive B frames, bounded to 0..=16.
    pub max_b_frames: usize,
    /// Maximum key-frame interval.
    pub gop: u32,
    /// Explicit decoder and encoder worker count (1..=64 each).
    pub threads: usize,
    /// Fail before I/O: a native GPU encoder surface pipeline is not implemented.
    pub hardware_required: bool,
    /// Maximum conservative decoded/output frame footprint, separately each.
    pub max_frame_bytes: u64,
    /// Maximum submitted frames awaiting their encoded packets/durations.
    pub max_pending_frames: usize,
    /// Bounded source discovery controls.
    pub probe: ProbeOptions,
}
impl Default for TranscodeOptions {
    fn default() -> Self {
        Self {
            video_track: None,
            audio_tracks: None,
            codec: CpuVideoCodec::H264,
            width: None,
            height: None,
            filter: ResizeFilter::Bilinear,
            bit_rate: 2_000_000,
            profile: None,
            preset: "veryfast".into(),
            max_b_frames: 2,
            gop: 60,
            threads: 1,
            hardware_required: false,
            max_frame_bytes: 256 * 1024 * 1024,
            max_pending_frames: 128,
            probe: ProbeOptions::default(),
        }
    }
}
/// Finalized result with actual codec/mux counts and explicit discarded metadata.
#[derive(Debug, Clone)]
pub struct TranscodeReport {
    /// Finalized output file.
    pub destination: PathBuf,
    /// Input video stream encoded into output stream zero.
    pub video_track: TrackId,
    /// Selected compressed audio streams, in output stream order.
    pub audio_tracks: Vec<TrackId>,
    /// Discovered input streams intentionally excluded by selection.
    pub unselected_tracks: Vec<TrackId>,
    /// Established original common timeline origin; timestamps are not rebased.
    pub source_origin: Option<Time>,
    /// Actual source bytes read, including probing and rereads.
    pub source_bytes_read: u64,
    /// Frames received from the software decoder, including EOF drain.
    pub frames_decoded: u64,
    /// Frames accepted by the software encoder.
    pub frames_encoded: u64,
    /// Encoded video packets written, including encoder EOF drain.
    pub video_packets_written: u64,
    /// Original compressed audio packets written without decoding.
    pub audio_packets_written: u64,
    /// Actual output width.
    pub output_width: u32,
    /// Actual output height.
    pub output_height: u32,
    /// Selected software encoder.
    pub codec: CpuVideoCodec,
    /// Actual encoder pixel representation.
    pub output_pixel_format: String,
    /// Metadata not carried through encoding/container changes.
    pub metadata_loss: Vec<String>,
    /// Timing fidelity limits; native decoder hints are not promised to survive re-encoding.
    pub timing_limitations: Vec<String>,
}

/// Encode one video stream, copy selected audio, drain, and finalize a new file.
///
/// Codec/container/geometry selection and encoder opening happen before output
/// creation. Existing files are never overwritten. Runtime failures retain a
/// partial destination and identify it in the returned error. At present only
/// progressive eight-bit YUV420P/NV12 sources are accepted; other source formats,
/// rotated video, alpha, HDR, and depth changes require a future explicit policy.
pub fn transcode(
    source: &Source,
    destination: &Path,
    options: TranscodeOptions,
) -> Result<TranscodeReport> {
    validate_options(&options)?;
    let mut opened = format::open_source(source, options.probe)?;
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
                && match options.video_track {
                    Some(id) => track.id == id && !track.cover_art,
                    None => !track.cover_art,
                }
        })
        .ok_or_else(|| Error::unsupported("source has no selected non-cover video track"))?;
    let video_track = video.id;
    if let Some(matrix) = &video.display_matrix
        && native::display_orientation(matrix)? != Orientation::Normal
    {
        return Err(Error::unsupported(
            "video display orientation must be applied explicitly before CPU transcode",
        ));
    }
    let audio_tracks = match &options.audio_tracks {
        Some(tracks) => tracks.clone(),
        None => info
            .tracks
            .iter()
            .filter(|track| track.kind == TrackKind::Audio)
            .map(|track| track.id)
            .collect(),
    };
    let mut unique = BTreeSet::new();
    for id in &audio_tracks {
        if !unique.insert(id.0)
            || !info
                .tracks
                .iter()
                .any(|track| track.id == *id && track.kind == TrackKind::Audio)
        {
            return Err(Error::invalid(
                "audio copy selection contains a duplicate or non-audio track",
            ));
        }
    }
    native::check_mux_codec(destination, options.codec.id())?;
    for id in &audio_tracks {
        let stream = opened
            .context
            .stream(id.0 as usize)
            .ok_or_else(|| Error::invalid("audio track disappeared"))?;
        native::check_mux_codec(destination, stream.parameters().id())?;
    }
    let input = opened
        .context
        .stream(video_track.0 as usize)
        .ok_or_else(|| Error::invalid("video track disappeared"))?;
    let input_base = input.time_base();
    let mut context = ffmpeg::codec::context::Context::from_parameters(input.parameters())
        .map_err(backend_error)?;
    context.set_threading(ffmpeg::threading::Config::count(options.threads));
    let mut decoder = context.decoder();
    decoder.set_packet_time_base(input_base);
    let decoder = decoder.video().map_err(backend_error)?;
    validate_source(&decoder, &options)?;
    let (width, height) = match (options.width, options.height) {
        (Some(w), Some(h)) => (w, h),
        _ => (decoder.width(), decoder.height()),
    };
    check_geometry(width, height, options.max_frame_bytes)?;
    let codec = ffmpeg::encoder::find_by_name(options.codec.encoder_name()).ok_or_else(|| {
        Error::unsupported(format!(
            "linked FFmpeg lacks software encoder {}",
            options.codec.encoder_name()
        ))
    })?;
    let mut encoder = ffmpeg::codec::context::Context::new_with_codec(codec)
        .encoder()
        .video()
        .map_err(backend_error)?;
    encoder.set_width(width);
    encoder.set_height(height);
    encoder.set_format(ffmpeg::format::Pixel::YUV420P);
    encoder.set_time_base(input_base);
    encoder.set_frame_rate(decoder.frame_rate());
    // Preserve display aspect through exact rational sample-aspect adjustment.
    let sar = adjusted_aspect(
        decoder.aspect_ratio(),
        decoder.width(),
        decoder.height(),
        width,
        height,
    )?;
    encoder.set_aspect_ratio(sar);
    encoder.set_bit_rate(options.bit_rate);
    encoder.set_max_b_frames(options.max_b_frames);
    encoder.set_gop(options.gop);
    encoder.set_threading(ffmpeg::threading::Config::count(options.threads));
    encoder.set_colorspace(decoder.color_space());
    encoder.set_color_range(decoder.color_range());
    encoder.set_color_primaries(decoder.color_primaries());
    encoder.set_color_transfer_characteristic(decoder.color_transfer_characteristic());
    ffi_helpers::set_encoder_chroma(&mut encoder, decoder.chroma_location());
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
    let encoder = encoder
        .open_with(settings)
        .map_err(|error| backend_error(error).with_operation("open software video encoder"))?;
    if format::time_base(encoder.time_base())? != format::time_base(input_base)? {
        return Err(Error::incompatible(
            "encoder changed the source time base during negotiation",
        ));
    }
    // All negotiation above is performed without creating or truncating output.
    let name = destination
        .to_str()
        .ok_or_else(|| Error::invalid("FFmpeg output path must be UTF-8"))?;
    let file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .read(true)
        .open(destination)
        .map_err(Error::io)?;
    let sync = file.try_clone().map_err(Error::io)?;
    let mut output = ffmpeg::format::output_to_stream(
        ffmpeg::format::context::StreamIo::from_write_seek(file).map_err(backend_error)?,
        Some(name),
        None,
    )
    .map_err(backend_error)?;
    let unselected_tracks = info
        .tracks
        .iter()
        .filter(|track| track.id != video_track && !audio_tracks.contains(&track.id))
        .map(|track| track.id)
        .collect();
    let mut pipeline = Pipeline {
        decoder,
        encoder,
        scaler: None,
        input_base: format::time_base(input_base)?,
        video_output_base: format::time_base(input_base)?,
        audio_mapping: BTreeMap::new(),
        source_geometry: None,
        width,
        height,
        sar,
        options,
        pending: BTreeMap::new(),
        metadata_loss: BTreeSet::new(),
        frames_decoded: 0,
        frames_encoded: 0,
        video_packets_written: 0,
        audio_packets_written: 0,
    };
    opened.allow_reading();
    let result = (|| -> Result<()> {
        {
            let input = opened
                .context
                .stream(video_track.0 as usize)
                .ok_or_else(|| Error::invalid("video track disappeared"))?;
            let mut stream = output.add_stream(codec).map_err(backend_error)?;
            stream.set_parameters(&pipeline.encoder);
            stream.set_time_base(input_base);
            stream.set_avg_frame_rate(input.avg_frame_rate());
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
            stream.set_parameters(input.parameters());
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
        for stream in opened.context.streams() {
            if stream.index() == video_track.0 as usize {
                for side_data in stream.side_data() {
                    pipeline.metadata_loss.insert(format!(
                        "video stream side data {:?} was not propagated",
                        side_data.kind()
                    ));
                }
            }
        }
        ffi_helpers::configure_mux(&mut output);
        output.write_header().map_err(backend_error)?;
        pipeline.video_output_base = format::time_base(
            output
                .stream(0)
                .ok_or_else(|| Error::invalid("output video stream missing"))?
                .time_base(),
        )?;
        loop {
            let mut packet = ffmpeg::Packet::empty();
            match packet.read(&mut opened.context) {
                Ok(()) => {}
                Err(ffmpeg::Error::Eof) => break,
                Err(error) => return Err(backend_error(error).with_operation("transcode demux")),
            }
            if packet.stream() == video_track.0 as usize {
                pipeline
                    .decoder
                    .send_packet(&packet)
                    .map_err(backend_error)?;
                pipeline.receive_frames(&mut output, false)?;
            } else if let Some((index, input_base)) =
                pipeline.audio_mapping.get(&packet.stream()).copied()
            {
                if packet.pts().is_none() || packet.dts().is_none() {
                    return Err(Error::unsupported(
                        "audio copy requires established packet PTS/DTS; muxer timestamp fabrication is forbidden",
                    ));
                }
                let output_base = format::time_base(
                    output
                        .stream(index)
                        .ok_or_else(|| Error::invalid("output audio stream missing"))?
                        .time_base(),
                )?;
                rescale_packet(&mut packet, input_base, output_base)?;
                packet.set_stream(index);
                packet.set_position(-1);
                packet
                    .write_interleaved(&mut output)
                    .map_err(backend_error)?;
                increment(&mut pipeline.audio_packets_written)?;
            }
        }
        pipeline.decoder.send_eof().map_err(backend_error)?;
        pipeline.receive_frames(&mut output, true)?;
        pipeline.encoder.send_eof().map_err(backend_error)?;
        pipeline.receive_packets(&mut output, true)?;
        if !pipeline.pending.is_empty() || pipeline.frames_decoded == 0 {
            return Err(Error::backend(
                "ffmpeg",
                "encoder drain did not account for every submitted frame, or input video was empty",
            ));
        }
        output.write_trailer().map_err(backend_error)?;
        native::flush_output(&mut output)?;
        Ok(())
    })();
    drop(output);
    result.map_err(|error| {
        error
            .with_source(destination.display().to_string())
            .with_operation("transcode; partial destination may exist")
    })?;
    sync.sync_all().map_err(Error::io)?;
    Ok(TranscodeReport {
        destination: destination.to_owned(),
        video_track,
        audio_tracks,
        unselected_tracks,
        source_origin: info.timeline_origin,
        source_bytes_read: opened.counter.total.load(Ordering::Relaxed),
        frames_decoded: pipeline.frames_decoded,
        frames_encoded: pipeline.frames_encoded,
        video_packets_written: pipeline.video_packets_written,
        audio_packets_written: pipeline.audio_packets_written,
        output_width: width,
        output_height: height,
        codec: pipeline.options.codec,
        output_pixel_format: "yuv420p".into(),
        metadata_loss: pipeline.metadata_loss.into_iter().collect(),
        timing_limitations:vec!["Native packet/codec duration hints may be absent or differ after re-encoding and container muxing; exact presentation PTS are preserved, while unknown display intervals require an explicit duration-inference policy".into()],
    })
}

struct Pipeline {
    decoder: ffmpeg::decoder::Video,
    encoder: ffmpeg::encoder::Video,
    scaler: Option<ffi_helpers::YuvScaler>,
    input_base: TimeBase,
    video_output_base: TimeBase,
    audio_mapping: BTreeMap<usize, (usize, TimeBase)>,
    source_geometry: Option<(u32, u32, ffmpeg::format::Pixel)>,
    width: u32,
    height: u32,
    sar: ffmpeg::Rational,
    options: TranscodeOptions,
    pending: BTreeMap<i64, i64>,
    metadata_loss: BTreeSet<String>,
    frames_decoded: u64,
    frames_encoded: u64,
    video_packets_written: u64,
    audio_packets_written: u64,
}
impl Pipeline {
    fn receive_frames(
        &mut self,
        output: &mut ffmpeg::format::context::Output,
        draining: bool,
    ) -> Result<()> {
        loop {
            let mut decoded = ffmpeg::frame::Video::empty();
            match self.decoder.receive_frame(&mut decoded) {
                Ok(()) => {}
                Err(ffmpeg::Error::Eof) if draining => return Ok(()),
                Err(error) if is_again(error) && !draining => return Ok(()),
                Err(error) => {
                    return Err(backend_error(error).with_operation("transcode decoder receive"));
                }
            }
            increment(&mut self.frames_decoded)?;
            validate_frame_integrity(&decoded)?;
            check_geometry(
                decoded.width(),
                decoded.height(),
                self.options.max_frame_bytes,
            )?;
            let geometry = (decoded.width(), decoded.height(), decoded.format());
            if self.source_geometry.is_some_and(|first| first != geometry) {
                return Err(Error::unsupported(
                    "video changes geometry or pixel format midstream; explicit encoder renegotiation is required",
                ));
            }
            validate_pixels(decoded.format())?;
            if decoded.is_interlaced() {
                return Err(Error::unsupported(
                    "interlaced video needs explicit deinterlacing before transcode",
                ));
            }
            if decoded.color_space() != self.decoder.color_space()
                || decoded.color_range() != self.decoder.color_range()
                || decoded.color_primaries() != self.decoder.color_primaries()
                || decoded.color_transfer_characteristic()
                    != self.decoder.color_transfer_characteristic()
                || decoded.chroma_location() != self.decoder.chroma_location()
            {
                return Err(Error::unsupported(
                    "video color/chroma metadata changes or was not established during encoder negotiation",
                ));
            }
            // Side data that is not copied is reported instead of silently claimed preserved.
            self.metadata_loss
                .extend(ffi_helpers::side_data_loss(&decoded));
            let pts = decoded.pts().ok_or_else(|| {
                Error::unsupported(
                    "video frame has no source PTS; transcode will not fabricate timestamps",
                )
            })?;
            if self.pending.len() >= self.options.max_pending_frames {
                return Err(Error::exhausted("encoder pending-frame budget exceeded")
                    .with_constraint("max_pending_frames"));
            }
            if self
                .pending
                .insert(pts, decoded.packet().duration.max(0))
                .is_some()
            {
                return Err(Error::incompatible(
                    "duplicate video frame PTS cannot be mapped uniquely through the encoder",
                ));
            }
            let mut scaled = ffmpeg::frame::Video::empty();
            let frame = if decoded.width() == self.width
                && decoded.height() == self.height
                && decoded.format() == ffmpeg::format::Pixel::YUV420P
            {
                &mut decoded
            } else {
                if self.scaler.is_none() {
                    self.scaler = Some(ffi_helpers::YuvScaler::new(
                        &decoded,
                        self.width,
                        self.height,
                        self.options.filter,
                    )?);
                }
                self.scaler
                    .as_mut()
                    .ok_or_else(|| Error::backend("ffmpeg", "scaler missing"))?
                    .run(&decoded, &mut scaled)?;
                scaled.set_color_space(decoded.color_space());
                scaled.set_color_range(decoded.color_range());
                scaled.set_color_primaries(decoded.color_primaries());
                scaled.set_color_transfer_characteristic(decoded.color_transfer_characteristic());
                ffi_helpers::set_frame_attributes(&mut scaled, decoded.chroma_location(), self.sar);
                &mut scaled
            };
            frame.set_pts(Some(pts));
            frame.set_kind(ffmpeg::picture::Type::None);
            self.encoder.send_frame(frame).map_err(backend_error)?;
            increment(&mut self.frames_encoded)?;
            self.source_geometry = Some(geometry);
            self.receive_packets(output, false)?;
        }
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
                    return Err(backend_error(error).with_operation("transcode encoder receive"));
                }
            }
            let pts = packet.pts().ok_or_else(|| {
                Error::backend("ffmpeg", "encoder returned a packet without source PTS")
            })?;
            let duration = self.pending.remove(&pts).ok_or_else(|| {
                Error::backend(
                    "ffmpeg",
                    "encoder packet PTS does not match a submitted frame",
                )
            })?;
            if duration > 0 {
                packet.set_duration(duration);
            }
            rescale_packet(&mut packet, self.input_base, self.video_output_base)?;
            packet.set_stream(0);
            packet.set_position(-1);
            packet.write_interleaved(output).map_err(backend_error)?;
            increment(&mut self.video_packets_written)?;
        }
    }
}

pub(crate) fn validate_options(options: &TranscodeOptions) -> Result<()> {
    if options.hardware_required {
        return Err(Error::unsupported(
            "hardware-required encode needs a native NVENC surface pipeline; CPU encode/upload is not a substitute",
        ));
    }
    if options.threads == 0
        || options.threads > 64
        || options.bit_rate == 0
        || options.bit_rate > i64::MAX as usize
        || options.max_b_frames > 16
        || options.gop == 0
        || options.gop > i32::MAX as u32
        || options.max_frame_bytes == 0
        || options.max_pending_frames == 0
        || options.width.is_some() != options.height.is_some()
    {
        return Err(Error::invalid(
            "invalid transcode geometry, bitrate, worker count, GOP, or allocation budget",
        ));
    }
    if ![
        "ultrafast",
        "superfast",
        "veryfast",
        "faster",
        "fast",
        "medium",
        "slow",
        "slower",
        "veryslow",
        "placebo",
    ]
    .contains(&options.preset.as_str())
    {
        return Err(Error::invalid("unknown software encoder speed preset"));
    }
    if let Some(profile) = &options.profile {
        let valid = match options.codec {
            CpuVideoCodec::H264 => ["baseline", "main", "high"].contains(&profile.as_str()),
            CpuVideoCodec::Hevc => profile == "main",
        };
        if !valid {
            return Err(Error::unsupported(
                "profile is unsupported by the eight-bit CPU encoder path",
            ));
        }
    }
    Ok(())
}
fn validate_frame_integrity(frame: &ffmpeg::frame::Video) -> Result<()> {
    if frame.is_corrupt() || frame.has_decode_errors() {
        return Err(Error::backend("ffmpeg","decoder returned a corrupt or partially decoded frame; strict transcode cannot encode it").with_operation("transcode decoded frame integrity"));
    }
    Ok(())
}
fn validate_source(decoder: &ffmpeg::decoder::Video, options: &TranscodeOptions) -> Result<()> {
    check_geometry(decoder.width(), decoder.height(), options.max_frame_bytes)?;
    validate_pixels(decoder.format())?;
    use ffmpeg::color::TransferCharacteristic as Transfer;
    if matches!(
        decoder.color_transfer_characteristic(),
        Transfer::SMPTE2084 | Transfer::ARIB_STD_B67
    ) {
        return Err(Error::unsupported(
            "HDR encoding/resize requires an explicit HDR processing policy",
        ));
    }
    Ok(())
}
fn validate_pixels(format: ffmpeg::format::Pixel) -> Result<()> {
    if !matches!(
        format,
        ffmpeg::format::Pixel::YUV420P | ffmpeg::format::Pixel::NV12
    ) {
        return Err(Error::unsupported(format!(
            "CPU transcode requires eight-bit YUV420P/NV12; {format:?} needs explicit precision/alpha/chroma conversion"
        )));
    }
    Ok(())
}
pub(crate) fn check_geometry(width: u32, height: u32, budget: u64) -> Result<()> {
    if width == 0
        || height == 0
        || width > i32::MAX as u32
        || height > i32::MAX as u32
        || !width.is_multiple_of(2)
        || !height.is_multiple_of(2)
    {
        return Err(Error::invalid(
            "YUV420 encode requires nonzero even dimensions within native limits",
        ));
    }
    // Conservative allowance for native padding and planar storage; codec-private
    // reference surfaces remain controlled separately by codec delay/workers.
    let bytes = u64::from(width)
        .checked_mul(u64::from(height))
        .and_then(|v| v.checked_mul(4))
        .ok_or_else(|| Error::overflow("video geometry overflows"))?;
    if bytes > budget {
        return Err(
            Error::exhausted("video frame exceeds transcode allocation budget")
                .with_constraint("max_frame_bytes"),
        );
    }
    Ok(())
}
fn adjusted_aspect(
    sar: ffmpeg::Rational,
    sw: u32,
    sh: u32,
    dw: u32,
    dh: u32,
) -> Result<ffmpeg::Rational> {
    let (n, d) = if sar.numerator() > 0 && sar.denominator() > 0 {
        (sar.numerator() as u64, sar.denominator() as u64)
    } else {
        return Ok(ffmpeg::Rational(0, 1));
    };
    let n = n
        .checked_mul(u64::from(sw))
        .and_then(|v| v.checked_mul(u64::from(dh)))
        .ok_or_else(|| Error::overflow("sample aspect ratio overflow"))?;
    let d = d
        .checked_mul(u64::from(sh))
        .and_then(|v| v.checked_mul(u64::from(dw)))
        .ok_or_else(|| Error::overflow("sample aspect ratio overflow"))?;
    fn gcd(mut a: u64, mut b: u64) -> u64 {
        while b != 0 {
            (a, b) = (b, a % b);
        }
        a
    }
    let g = gcd(n, d);
    Ok(ffmpeg::Rational(
        i32::try_from(n / g)
            .map_err(|_| Error::overflow("sample aspect ratio numerator exceeds native range"))?,
        i32::try_from(d / g)
            .map_err(|_| Error::overflow("sample aspect ratio denominator exceeds native range"))?,
    ))
}
pub(crate) fn rescale_packet(
    packet: &mut ffmpeg::Packet,
    source: TimeBase,
    target: TimeBase,
) -> Result<()> {
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
    matches!(error,ffmpeg::Error::Other{errno}if errno==ffmpeg::error::EAGAIN)
}
fn increment(value: &mut u64) -> Result<()> {
    *value = value
        .checked_add(1)
        .ok_or_else(|| Error::overflow("transcode counter overflow"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use media_core::error::ErrorKind;

    #[test]
    fn marked_native_decode_failures_are_rejected_before_encoding() {
        for (corrupt, decode_error) in [(true, false), (false, true)] {
            let frame = ffi_helpers::integrity_test_frame(corrupt, decode_error);
            assert_eq!(
                validate_frame_integrity(&frame).unwrap_err().kind(),
                ErrorKind::Backend
            );
        }
        validate_frame_integrity(&ffmpeg::frame::Video::empty()).unwrap();
    }

    #[test]
    fn packet_rescaling_preserves_offsets_and_reordered_negative_dts() {
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
    }

    #[test]
    fn packet_rescaling_rejects_quantization_and_overflow() {
        let mut packet = ffmpeg::Packet::empty();
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
        packet.set_pts(Some(i64::MAX));
        assert_eq!(
            rescale_packet(
                &mut packet,
                TimeBase::new(1, 1).unwrap(),
                TimeBase::new(1, u32::MAX).unwrap()
            )
            .unwrap_err()
            .kind(),
            ErrorKind::Overflow
        );
    }
}

/// FFI not exposed by safe bindings; all pointers remain owned by borrowed wrappers.
#[allow(unsafe_code)]
pub(crate) mod ffi_helpers {
    use super::*;
    use std::{ffi::CString, ptr};
    #[cfg(test)]
    pub(super) fn integrity_test_frame(corrupt: bool, decode_error: bool) -> ffmpeg::frame::Video {
        let mut frame = ffmpeg::frame::Video::empty();
        // SAFETY: this uniquely owned native fixture frame is live. Only scalar
        // FFmpeg integrity flags are set; no planes or pointers are fabricated.
        unsafe {
            (*frame.as_mut_ptr()).flags = if corrupt {
                ffmpeg::ffi::AV_FRAME_FLAG_CORRUPT
            } else {
                0
            };
            (*frame.as_mut_ptr()).decode_error_flags = i32::from(decode_error);
        }
        frame
    }
    pub(super) fn copy_disposition(
        source: &ffmpeg::Stream<'_>,
        target: &mut ffmpeg::StreamMut<'_>,
    ) {
        // SAFETY: independently borrowed input and output contexts retain both streams.
        unsafe {
            (*target.as_mut_ptr()).disposition = source.disposition().bits();
        }
    }
    pub(super) fn side_data_loss(frame: &ffmpeg::frame::Video) -> Vec<String> {
        // SAFETY: the borrowed frame owns a valid array with nb_side_data
        // initialized entries; no side-data pointer escapes this call.
        unsafe {
            (0..(*frame.as_ptr()).nb_side_data)
                .map(|i| {
                    let data = *(*frame.as_ptr()).side_data.add(i as usize);
                    let kind = ffmpeg::frame::side_data::Type::from((*data).type_);
                    format!("video frame side data {kind:?} was not propagated")
                })
                .collect()
        }
    }
    pub(crate) fn requires_global_header(path: &Path) -> Result<bool> {
        let path = CString::new(
            path.to_str()
                .ok_or_else(|| Error::invalid("output path must be UTF-8"))?,
        )
        .map_err(|_| Error::invalid("output path contains NUL"))?;
        // SAFETY: path is terminated and live; descriptor is immutable library-owned storage.
        unsafe {
            let format = ffmpeg::ffi::av_guess_format(ptr::null(), path.as_ptr(), ptr::null());
            if format.is_null() {
                return Err(Error::unsupported("unknown output container"));
            }
            Ok((*format).flags & ffmpeg::ffi::AVFMT_GLOBALHEADER != 0)
        }
    }
    pub(crate) fn set_encoder_chroma(
        encoder: &mut ffmpeg::codec::encoder::video::Video,
        chroma: ffmpeg::chroma::Location,
    ) {
        // SAFETY: exclusive live codec context, enum conversion matches native definition.
        unsafe {
            (*encoder.as_mut_ptr()).chroma_sample_location = chroma.into();
        }
    }
    pub(crate) fn set_frame_attributes(
        frame: &mut ffmpeg::frame::Video,
        chroma: ffmpeg::chroma::Location,
        sar: ffmpeg::Rational,
    ) {
        // SAFETY: exclusive live frame; these fields contain scalars, not borrowed pointers.
        unsafe {
            (*frame.as_mut_ptr()).chroma_location = chroma.into();
            (*frame.as_mut_ptr()).sample_aspect_ratio = sar.into();
        }
    }
    pub(crate) fn configure_mux(output: &mut ffmpeg::format::context::Output) {
        // SAFETY: exclusive format context before write_header. Preserve source clock,
        // including negative reordered DTS; limit the interleaver's time horizon.
        unsafe {
            (*output.as_mut_ptr()).avoid_negative_ts = ffmpeg::ffi::AVFMT_AVOID_NEG_TS_DISABLED;
            (*output.as_mut_ptr()).max_interleave_delta = 1_000_000;
        }
    }
    // The safe binding cannot set chroma positions before initializing its
    // scaler, and ignores av_frame_get_buffer/sws_scale failures. This narrow
    // owned wrapper checks both and never exposes borrowed data pointers.
    pub(super) struct YuvScaler {
        raw: std::ptr::NonNull<ffmpeg::ffi::SwsContext>,
        source: (u32, u32, ffmpeg::format::Pixel),
        width: u32,
        height: u32,
    }
    impl Drop for YuvScaler {
        fn drop(&mut self) {
            // SAFETY: raw is one exclusively owned allocation, freed exactly once.
            unsafe {
                ffmpeg::ffi::sws_freeContext(self.raw.as_ptr());
            }
        }
    }
    impl YuvScaler {
        pub(super) fn new(
            frame: &ffmpeg::frame::Video,
            width: u32,
            height: u32,
            filter: ResizeFilter,
        ) -> Result<Self> {
            use ffmpeg::{chroma::Location, ffi};
            let (horizontal, vertical) = match frame.chroma_location() {
                Location::Left => (0, 128),
                Location::Center => (128, 128),
                Location::TopLeft => (0, 0),
                Location::Top => (128, 0),
                Location::BottomLeft => (0, 256),
                Location::Bottom => (128, 256),
                _ => {
                    return Err(Error::unsupported(
                        "YUV resize needs established chroma siting; no implicit grid assumption is made",
                    ));
                }
            };
            // SAFETY: allocation has no arguments; unique RAII owner immediately
            // retains it, including on every option/init failure below.
            let raw = std::ptr::NonNull::new(unsafe { ffi::sws_alloc_context() })
                .ok_or_else(|| Error::exhausted("FFmpeg scaler allocation failed"))?;
            let scaler = Self {
                raw,
                source: (frame.width(), frame.height(), frame.format()),
                width,
                height,
            };
            let format: ffi::AVPixelFormat = frame.format().into();
            let full = i64::from(frame.color_range() == ffmpeg::color::Range::JPEG);
            for (name, value) in [
                (c"srcw", i64::from(frame.width())),
                (c"srch", i64::from(frame.height())),
                (c"dstw", i64::from(width)),
                (c"dsth", i64::from(height)),
                (c"src_format", format as i64),
                (c"dst_format", ffi::AVPixelFormat::AV_PIX_FMT_YUV420P as i64),
                (c"sws_flags", i64::from(filter.flags().bits())),
                (c"src_h_chr_pos", horizontal),
                (c"dst_h_chr_pos", horizontal),
                (c"src_v_chr_pos", vertical),
                (c"dst_v_chr_pos", vertical),
                (c"src_range", full),
                (c"dst_range", full),
                (c"threads", 1),
            ] {
                // SAFETY: every option name is a live terminated literal and raw
                // is an exclusive uninitialized AVOptions-enabled SwsContext.
                let status =
                    unsafe { ffi::av_opt_set_int(raw.as_ptr().cast(), name.as_ptr(), value, 0) };
                if status < 0 {
                    return Err(backend_error(ffmpeg::Error::from(status)));
                }
            }
            // SAFETY: dimensions/formats/grid/range are validated above; null
            // filter pointers select built-in kernels and are valid per API.
            let status =
                unsafe { ffi::sws_init_context(raw.as_ptr(), ptr::null_mut(), ptr::null_mut()) };
            if status < 0 {
                return Err(backend_error(ffmpeg::Error::from(status)));
            }
            Ok(scaler)
        }
        pub(super) fn run(
            &mut self,
            source: &ffmpeg::frame::Video,
            target: &mut ffmpeg::frame::Video,
        ) -> Result<()> {
            // SAFETY: target is a valid owned AVFrame, borrowed for this read.
            let target_empty = unsafe { target.is_empty() };
            if self.source != (source.width(), source.height(), source.format()) || !target_empty {
                return Err(Error::invalid(
                    "scaler requires matching input geometry and a fresh output frame",
                ));
            }
            target.set_width(self.width);
            target.set_height(self.height);
            target.set_format(ffmpeg::format::Pixel::YUV420P);
            // SAFETY: exclusively borrowed fresh AVFrame with checked positive
            // dimensions/format. A failed allocation never reaches sws_scale.
            let status = unsafe { ffmpeg::ffi::av_frame_get_buffer(target.as_mut_ptr(), 32) };
            if status < 0 {
                return Err(backend_error(ffmpeg::Error::from(status)));
            }
            // SAFETY: libavcodec owns validated source planes; target owns newly
            // allocated destination planes. Their AVFrames remain borrowed for
            // the complete synchronous call and no pointer escapes.
            let rows = unsafe {
                ffmpeg::ffi::sws_scale(
                    self.raw.as_ptr(),
                    (*source.as_ptr()).data.as_ptr() as *const *const u8,
                    (*source.as_ptr()).linesize.as_ptr(),
                    0,
                    source.height() as i32,
                    (*target.as_mut_ptr()).data.as_ptr(),
                    (*target.as_ptr()).linesize.as_ptr(),
                )
            };
            if rows < 0 {
                return Err(backend_error(ffmpeg::Error::from(rows)));
            }
            if rows != self.height as i32 {
                return Err(Error::backend(
                    "ffmpeg",
                    "scaler did not produce every destination row",
                ));
            }
            Ok(())
        }
    }
}
