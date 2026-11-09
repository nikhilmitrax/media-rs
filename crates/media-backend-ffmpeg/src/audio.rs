//! Bounded FFmpeg audio decoding with demux edits and codec skip handling.
//!
//! Requires FFmpeg 5.1+ with libswresample. Container edit interpretation stays
//! entirely in libavformat, and decoder skip/discard handling stays in
//! libavcodec (skip_manual is never enabled). Equal-rate libswresample only
//! normalizes scalar format/packing; continuous rate conversion belongs to the
//! caller. Codec-private allocations remain native and unmeasured.
//!
//! Source clock means libavformat's edited demux presentation timestamps, not
//! an invented pre-edit media clock. Asset clock subtracts a proven common
//! origin before all subsequent sample-grid processing. Semantic extent honors
//! declared metadata; absent encoder delay is never guessed from the codec name.
//! See FFmpeg's [MOV edit-list rules](https://ffmpeg.org/ffmpeg-formats.html#mov_002fmp4_002f3gp)
//! and [automatic skip/discard implementation](https://ffmpeg.org/doxygen/8.1/decode_8c_source.html).

use crate::{
    audio_native,
    format::{self, AssetInfo, OpenedInput, ProbeOptions, TrackKind, backend_error},
};
use ffmpeg_next as ffmpeg;
use media_core::{
    error::{Error, ErrorKind, Result},
    media::{AudioBlock, ChannelLayout, TrackId},
    time::{Time, TimeRange},
};
use media_io::source::Source;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

/// Processing clock for every block, slice and subsequent continuous operation.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum AudioTimeline {
    /// Use demux presentation timestamps after edit mapping. Explicit sample-clock
    /// reconstruction overrides subsequent timestamps while keeping this origin.
    #[default]
    Source,
    /// Subtract an established common container origin. Incomplete video header
    /// origins (notably reordered Matroska video) cannot establish this mapping.
    Asset,
}
/// Extra wrapper-level interpretation after libavcodec has applied skip samples.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum AudioTrimPolicy {
    /// Intersect with a checked declared stream span, including trailing codec
    /// padding that a plain FFmpeg raw decode can leave beyond stream duration.
    #[default]
    Semantic,
    /// Expose codec output without a second container-span intersection. Codec
    /// skip/discard processing remains enabled, exactly as in FFmpeg CLI decode.
    RawDecoded,
}
/// Strict bounded decoder and format-normalization controls.
#[derive(Clone, Copy, Debug)]
pub struct AudioDecodeOptions {
    /// Explicit logical stream, or the first audio stream when absent.
    pub track: Option<TrackId>,
    /// Header discovery forbids all codec-based probing, including video decoding.
    pub probe: ProbeOptions,
    /// Maximum selected compressed packet size.
    pub max_packet_bytes: usize,
    /// Maximum decoded interleaved scalar samples per native frame.
    pub max_buffered_samples: usize,
    /// Maximum sample frames returned by one call.
    pub max_block_frames: usize,
    /// Bounded codec threads, in 1..=256.
    pub threads: usize,
    /// Source or explicitly established asset clock.
    pub timeline: AudioTimeline,
    /// Container semantic end versus raw decoded codec extent.
    pub trim: AudioTrimPolicy,
    /// Explicitly allow one container tick of timestamp rounding, reporting each
    /// adjustment. Larger discontinuities fail unless reconstruction is explicit.
    pub allow_one_tick: bool,
    /// Explicitly rebuild continuous processing timestamps from decoded sample
    /// counts, preserving original PTS on each block. Requires RawDecoded trim;
    /// this repairs neither missing samples nor source synchronization.
    pub reconstruct_timestamps: bool,
}
impl Default for AudioDecodeOptions {
    fn default() -> Self {
        Self {
            track: None,
            probe: ProbeOptions::default(),
            max_packet_bytes: 8 * 1024 * 1024,
            max_buffered_samples: 1024 * 1024,
            max_block_frames: 4096,
            threads: 1,
            timeline: AudioTimeline::Source,
            trim: AudioTrimPolicy::Semantic,
            allow_one_tick: true,
            reconstruct_timestamps: false,
        }
    }
}
/// Established stream, continuous grid, and source/asset mapping.
#[derive(Clone, Debug)]
pub struct AudioInfo {
    /// Logical libavformat track, never a cross-demuxer numeric assumption.
    pub track: TrackId,
    /// Actual decoded frames per second.
    pub sample_rate: u32,
    /// Ordered speaker identities or explicitly unpositioned discrete channels.
    pub layout: ChannelLayout,
    /// Header-derived start in the processing clock; decoder priming can establish
    /// a later first valid output, reported through AudioDecodeReport::leading_gap.
    pub start: Time,
    /// Header-derived start in unmodified demux presentation time.
    pub source_start: Time,
    /// Established common source-to-asset offset, absent when headers cannot prove it.
    pub timeline_origin: Option<Time>,
    /// First valid asset sample, if a common origin is established.
    pub asset_start: Option<Time>,
    /// Declared source-clock half-open interval when sample-aligned and finite.
    pub source_interval: Option<TimeRange>,
    /// Declared playable-span length, absent for raw or unestablished duration.
    /// A leading gap can reduce the actual number of decoded sample frames.
    pub frames: Option<u64>,
    /// Container timestamp quantization, distinct from a sample period.
    pub timestamp_quantum: Time,
    /// Active clock, also used by caller slices/windows/resampling.
    pub timeline: AudioTimeline,
    /// Active semantic versus codec-output extent.
    pub trim: AudioTrimPolicy,
}
/// Actual demux, decoder, and additional semantic-span work.
#[derive(Clone, Debug, Default)]
pub struct AudioDecodeReport {
    /// Processing timestamps rebuilt beyond the permitted quantization bin by
    /// explicit caller policy. Original frame timestamps remain on output blocks.
    pub reconstructed_timestamps: u64,
    /// Declared leading source interval with no valid decoder output. Codec
    /// priming/overlap can establish a later first sample than header probing.
    pub leading_gap: Option<TimeRange>,
    /// Selected compressed packets accepted by the decoder.
    pub packets_sent: u64,
    /// Native audio frames returned after libavcodec's automatic skip/discard.
    pub decoded_frames: u64,
    /// Sample frames returned after all explicit interval/block processing.
    pub output_frames: u64,
    /// Packet skip-start requests passed intact to libavcodec (never reapplied).
    pub codec_skip_start_requests: u64,
    /// Packet discard-end requests passed intact to libavcodec (never reapplied).
    pub codec_skip_end_requests: u64,
    /// Additional leading source samples outside the declared semantic interval.
    pub semantic_trim_start: u64,
    /// Additional trailing source samples outside the declared semantic interval.
    pub semantic_trim_end: u64,
    /// Container rounding adjustments onto the original continuous sample grid.
    pub timestamp_adjustments: u64,
    /// Largest observed timestamp rounding adjustment.
    pub max_timestamp_adjustment: Option<Time>,
    /// Actual bytes read during bounded header probing.
    pub probe_bytes: u64,
    /// Decoder EOF was received and equal-rate format normalization had no tail.
    pub finished: bool,
}
/// Cooperative flag checked between every native receive/demux/returned block.
#[derive(Clone, Debug, Default)]
pub struct AudioCancellation(Arc<AtomicBool>);
impl AudioCancellation {
    /// Cancel all operations sharing this flag.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }
    fn check(&self) -> Result<()> {
        if self.0.load(Ordering::Acquire) {
            Err(Error::new(ErrorKind::Cancelled, "FFmpeg audio cancelled"))
        } else {
            Ok(())
        }
    }
}

/// Independent bounded FFmpeg cursor, decoder, scalar normalizer and drain state.
pub struct AudioReader {
    opened: OpenedInput,
    asset: AssetInfo,
    decoder: ffmpeg::decoder::Audio,
    normalizer: Option<ffmpeg::software::resampling::Context>,
    native_format: Option<ffmpeg::format::Sample>,
    info: AudioInfo,
    options: AudioDecodeOptions,
    report: AudioDecodeReport,
    pending: Option<(AudioBlock, usize)>,
    first_packet: Option<ffmpeg::Packet>,
    next_source: Option<Time>,
    returned_end: Time,
    declared_mpeg_interval: Option<TimeRange>,
    draining: bool,
    failed: bool,
    cancel: AudioCancellation,
}
impl AudioReader {
    /// Header-only bounded probe and decoder construction. No audio is decoded
    /// and no scalar-conversion/resampling working state is constructed here.
    pub fn open(source: &Source, options: AudioDecodeOptions) -> Result<Self> {
        if options.reconstruct_timestamps && options.trim != AudioTrimPolicy::RawDecoded {
            return Err(Error::invalid(
                "decoded-sample timestamp reconstruction requires RawDecoded extent",
            ));
        }
        if options.max_packet_bytes == 0
            || options.max_buffered_samples == 0
            || options.max_block_frames == 0
            || !(1..=256).contains(&options.threads)
        {
            return Err(Error::invalid(
                "FFmpeg audio requires nonzero packet/frame limits and 1..=256 threads",
            ));
        }
        let mut opened = format::open_source_without_decoding(source, options.probe)?;
        let mut asset = format::snapshot(
            &opened.context,
            source,
            opened.counter.total.load(Ordering::Relaxed),
        )?;
        let selected = asset
            .tracks
            .iter()
            .find(|track| {
                track.kind == TrackKind::Audio && options.track.is_none_or(|id| track.id == id)
            })
            .cloned()
            .ok_or_else(|| Error::unsupported("requested FFmpeg audio stream is unavailable"))?;
        let track = selected.id;
        let stream = opened
            .context
            .stream(track.0 as usize)
            .ok_or_else(|| Error::invalid("selected FFmpeg audio stream disappeared"))?;
        let mut context = ffmpeg::codec::context::Context::from_parameters(stream.parameters())
            .map_err(backend_error)?;
        context.set_threading(ffmpeg::threading::Config::count(options.threads));
        let mut unopened = context.decoder();
        unopened.set_packet_time_base(stream.time_base());
        let mut decoder = unopened.audio().map_err(backend_error)?;
        let sample_rate = decoder.rate();
        if sample_rate == 0 || sample_rate > 768_000 {
            return Err(Error::unsupported(
                "FFmpeg audio sample rate is unknown or exceeds 768kHz",
            ));
        }
        let missing_layout = audio_native::decoder_layout(&decoder).is_err();
        let mut first_packet = None;
        let source_start = if let Some(start) = selected.start.filter(|_| !missing_layout) {
            start
        } else {
            let mut found = None;
            for _ in 0..1024 {
                let mut packet = ffmpeg::Packet::empty();
                packet.read(&mut opened.context).map_err(backend_error)?;
                if packet.stream() != track.0 as usize {
                    continue;
                }
                if packet.size() > options.max_packet_bytes {
                    return Err(Error::exhausted(
                        "first FFmpeg audio packet exceeds its byte limit",
                    ));
                }
                let start = selected.start.map(Ok).unwrap_or_else(|| {
                    selected.time_base.time(packet.pts().ok_or_else(|| {
                        Error::unsupported(
                            "FFmpeg audio first packet has no established source PTS",
                        )
                    })?)
                })?;
                first_packet = Some(packet);
                found = Some(start);
                break;
            }
            found.ok_or_else(|| {
                Error::exhausted("FFmpeg audio first-packet probe exceeds 1024 packets")
            })?
        };
        if missing_layout && matches!(selected.codec.as_str(), "mp3" | "mp2" | "mp1") {
            audio_native::establish_mpeg_channels(
                &mut decoder,
                first_packet
                    .as_ref()
                    .expect("missing layout forces bounded first-packet discovery"),
            )?;
        }
        let layout = audio_native::decoder_layout(&decoder)?;
        if layout.channels() > options.max_buffered_samples {
            return Err(Error::exhausted(
                "FFmpeg audio channels exceed decoded sample capacity",
            ));
        }
        asset.probe_bytes_read = opened.counter.total.load(Ordering::Relaxed);
        if asset.timeline_origin.is_none() && asset.tracks.len() == 1 {
            asset.timeline_origin = Some(source_start);
        }
        // Ogg Opus stream duration is a granule span, including the OpusHead
        // pre-skip. RFC 7845 §4.3 subtracts that header value to obtain PCM
        // position. libavcodec already removes samples: adjust metadata only.
        let duration = if asset.format.split(',').any(|name| name == "ogg")
            && selected.codec == "opus"
            && selected.codec_config.len() >= 19
            && selected.codec_config.starts_with(b"OpusHead")
        {
            let pre_skip =
                u16::from_le_bytes([selected.codec_config[10], selected.codec_config[11]]);
            selected
                .duration
                .map(|duration| {
                    duration.checked_sub(Time::from_samples(i64::from(pre_skip), 48000)?)
                })
                .transpose()?
        } else {
            selected.duration
        };
        let declared_interval = duration
            .map(|duration| TimeRange::new(source_start, source_start.checked_add(duration)?))
            .transpose()?;
        // Older MP3 demuxers advertise the coded frame span, including LAME
        // delay/padding. Do not publish that as a known playable sample count.
        // At EOF, automatic skip/discard evidence must explain any shortfall.
        let mpeg_span = asset.format.split(',').any(|name| name == "mp3")
            && matches!(selected.codec.as_str(), "mp3" | "mp2" | "mp1");
        let source_interval = if mpeg_span { None } else { declared_interval };
        let frames = if options.trim == AudioTrimPolicy::Semantic {
            source_interval
                .map(|range| {
                    let ticks = range.duration()?.to_ticks(
                        media_core::time::TimeBase::new(1, sample_rate)?,
                        media_core::time::Rounding::TowardZero,
                    )?;
                    if ticks < 0 || Time::from_samples(ticks, sample_rate)? != range.duration()? {
                        return Err(Error::unsupported(
                            "semantic stream duration is not on the declared sample grid",
                        ));
                    }
                    Ok(ticks as u64)
                })
                .transpose()?
        } else {
            None
        };
        let mov = asset
            .format
            .split(',')
            .any(|name| matches!(name, "mov" | "mp4" | "m4a" | "3gp" | "3g2" | "mj2"));
        if mov && options.trim == AudioTrimPolicy::Semantic && source_interval.is_none() {
            return Err(Error::unsupported(
                "semantic MOV audio needs a finite declared stream interval; choose RawDecoded explicitly if appropriate",
            ));
        }
        // MOV index tables establish stream starts without video codec probing.
        // Matroska reordered-video starts can be incorrect header-only hints.
        let timeline_origin = if mov
            || !asset
                .tracks
                .iter()
                .any(|track| track.kind == TrackKind::Video)
        {
            asset.timeline_origin
        } else {
            None
        };
        let asset_start = timeline_origin
            .map(|origin| source_start.checked_sub(origin))
            .transpose()?;
        let start=match options.timeline {AudioTimeline::Source=>source_start,AudioTimeline::Asset=>asset_start.ok_or_else(||Error::unsupported("asset audio clock requires an established common header origin; use Source or synchronized clip mapping"))?};
        let info = AudioInfo {
            track,
            sample_rate,
            layout,
            start,
            source_start,
            timeline_origin,
            asset_start,
            source_interval,
            frames,
            timestamp_quantum: selected.time_base.time(1)?,
            timeline: options.timeline,
            trim: options.trim,
        };
        let report = AudioDecodeReport {
            probe_bytes: asset.probe_bytes_read,
            ..Default::default()
        };
        opened.allow_reading();
        Ok(Self {
            opened,
            asset,
            decoder,
            normalizer: None,
            native_format: None,
            info,
            options,
            report,
            pending: None,
            first_packet,
            next_source: None,
            returned_end: source_start,
            declared_mpeg_interval: if mpeg_span { declared_interval } else { None },
            draining: false,
            failed: false,
            cancel: Default::default(),
        })
    }
    /// Established audio interpretation before decoding.
    pub fn info(&self) -> &AudioInfo {
        &self.info
    }
    /// Header-only inventory; source clocks may remain partial for other streams.
    pub fn asset_info(&self) -> &AssetInfo {
        &self.asset
    }
    /// Work observed so far, including successful EOF/drain evidence.
    pub fn report(&self) -> &AudioDecodeReport {
        &self.report
    }
    /// Actual source bytes read, including header discovery and demux dependencies.
    pub fn bytes_read(&self) -> u64 {
        self.opened.counter.total.load(Ordering::Relaxed)
    }
    /// Shared cooperative cancellation flag.
    pub fn cancellation(&self) -> AudioCancellation {
        self.cancel.clone()
    }
    /// Read one bounded source/asset-grid block or a fully drained normal EOF.
    pub fn next_block(&mut self) -> Result<Option<AudioBlock>> {
        self.next_block_checked(|| Ok(()))
    }
    /// Also check a caller's cancellation/admission condition between native
    /// receive/demux calls. Errors are terminal and no further output is returned.
    pub fn next_block_checked(
        &mut self,
        mut check: impl FnMut() -> Result<()>,
    ) -> Result<Option<AudioBlock>> {
        if self.failed {
            return Err(Error::invalid("FFmpeg audio execution previously failed"));
        }
        let result = self.next_inner(&mut check);
        if result.is_err() {
            self.failed = true;
            self.pending = None;
            self.normalizer = None;
        }
        result
    }
    fn next_inner(&mut self, check: &mut impl FnMut() -> Result<()>) -> Result<Option<AudioBlock>> {
        loop {
            self.cancel.check()?;
            check()?;
            if let Some((block, offset)) = self.pending.take() {
                let count = (block.frames() - offset).min(self.options.max_block_frames);
                let channels = self.info.layout.channels();
                let start = block
                    .start()
                    .checked_add(Time::from_samples(offset as i64, self.info.sample_rate)?)?;
                let source = block
                    .source_start()
                    .map(|pts| {
                        pts.checked_add(Time::from_samples(offset as i64, self.info.sample_rate)?)
                    })
                    .transpose()?;
                let output = AudioBlock::new(
                    block.samples()[offset * channels..(offset + count) * channels].to_vec(),
                    self.info.sample_rate,
                    self.info.layout.clone(),
                    start,
                )?
                .with_source_start(source);
                if offset + count < block.frames() {
                    self.pending = Some((block, offset + count));
                }
                self.report.output_frames = self
                    .report
                    .output_frames
                    .checked_add(count as u64)
                    .ok_or_else(|| Error::overflow("FFmpeg audio output count"))?;
                return Ok(Some(output));
            }
            if self.report.finished {
                return Ok(None);
            }
            let mut decoded = ffmpeg::frame::Audio::empty();
            match self.decoder.receive_frame(&mut decoded) {
                Ok(()) => {
                    if let Some(block) = self.materialize(decoded)? {
                        self.pending = Some((block, 0));
                    }
                    continue;
                }
                Err(ffmpeg::Error::Eof) => {
                    if self
                        .normalizer
                        .as_ref()
                        .is_some_and(|normalizer| normalizer.delay().is_some())
                    {
                        return Err(Error::unsupported(
                            "equal-rate FFmpeg scalar normalizer has an unexpected computational tail",
                        ));
                    }
                    if self.options.trim == AudioTrimPolicy::Semantic
                        && let Some(interval) =
                            self.info.source_interval.or(self.declared_mpeg_interval)
                        && self.returned_end < interval.end()
                    {
                        let missing = interval.end().checked_sub(self.returned_end)?;
                        let skipped = self
                            .report
                            .codec_skip_start_requests
                            .checked_add(self.report.codec_skip_end_requests)
                            .ok_or_else(|| Error::overflow("MPEG skip evidence count"))?;
                        let explained = self.declared_mpeg_interval.is_some()
                            && skipped > 0
                            && missing
                                == Time::from_samples(
                                    i64::try_from(skipped)
                                        .map_err(|_| Error::overflow("MPEG skip evidence time"))?,
                                    self.info.sample_rate,
                                )?;
                        if !explained {
                            return Err(Error::new(
                                ErrorKind::MissingData,
                                format!(
                                    "FFmpeg audio ended before its declared interval: missing {missing}, codec skip/discard {skipped} samples"
                                ),
                            ));
                        }
                    }
                    self.report.finished = true;
                    return Ok(None);
                }
                Err(error) if is_again(error) => {}
                Err(error) => {
                    return Err(backend_error(error)
                        .with_track(self.info.track.0)
                        .with_operation("audio receive"));
                }
            }
            if self.draining {
                return Err(Error::backend(
                    "ffmpeg",
                    "audio requested input after EOF drain",
                ));
            }
            let mut packet = self
                .first_packet
                .take()
                .unwrap_or_else(ffmpeg::Packet::empty);
            if packet.size() == 0 {
                loop {
                    self.cancel.check()?;
                    check()?;
                    match packet.read(&mut self.opened.context) {
                        Ok(()) if packet.stream() == self.info.track.0 as usize => break,
                        Ok(()) => continue,
                        Err(ffmpeg::Error::Eof) => {
                            self.decoder.send_eof().map_err(backend_error)?;
                            self.draining = true;
                            break;
                        }
                        Err(error) => {
                            return Err(backend_error(error).with_operation("audio demux"));
                        }
                    }
                }
            }
            if !self.draining {
                if packet.size() > self.options.max_packet_bytes {
                    return Err(Error::exhausted(
                        "FFmpeg compressed audio packet exceeds its byte limit",
                    ));
                }
                if packet
                    .flags()
                    .contains(ffmpeg::codec::packet::Flags::CORRUPT)
                {
                    return Err(Error::backend("ffmpeg", "corrupt compressed audio packet"));
                }
                for side in packet.side_data() {
                    if side.kind() == ffmpeg::codec::packet::side_data::Type::SkipSamples {
                        let bytes = side.data();
                        if bytes.len() != 10 {
                            return Err(Error::invalid(
                                "malformed FFmpeg audio skip-sample metadata",
                            ));
                        }
                        increment(
                            &mut self.report.codec_skip_start_requests,
                            u64::from(u32::from_le_bytes(
                                bytes[0..4].try_into().expect("length checked"),
                            )),
                        )?;
                        increment(
                            &mut self.report.codec_skip_end_requests,
                            u64::from(u32::from_le_bytes(
                                bytes[4..8].try_into().expect("length checked"),
                            )),
                        )?;
                    }
                }
                self.decoder.send_packet(&packet).map_err(backend_error)?;
                self.report.packets_sent = self
                    .report
                    .packets_sent
                    .checked_add(1)
                    .ok_or_else(|| Error::overflow("FFmpeg audio packet count"))?;
            }
        }
    }
    fn materialize(&mut self, decoded: ffmpeg::frame::Audio) -> Result<Option<AudioBlock>> {
        if decoded.is_corrupt() || decoded.has_decode_errors() {
            return Err(Error::backend(
                "ffmpeg",
                "corrupt or incomplete decoded audio frame",
            ));
        }
        if decoded
            .side_data(ffmpeg::util::frame::side_data::Type::SkipSamples)
            .is_some()
        {
            return Err(Error::unsupported(
                "decoder exported manual skip metadata despite automatic skip policy; refusing double/unknown trimming",
            ));
        }
        if decoded.rate() != self.info.sample_rate
            || audio_native::frame_layout(&decoded)? != self.info.layout
        {
            return Err(Error::incompatible(
                "FFmpeg audio rate or speaker identities changed during decoding",
            ));
        }
        let count = decoded
            .samples()
            .checked_mul(self.info.layout.channels())
            .ok_or_else(|| Error::overflow("FFmpeg decoded sample count"))?;
        if count > self.options.max_buffered_samples {
            return Err(Error::exhausted(
                "decoded FFmpeg audio frame exceeds its sample limit",
            ));
        }
        if self
            .native_format
            .is_some_and(|format| format != decoded.format())
        {
            return Err(Error::incompatible(
                "FFmpeg audio scalar format changed during decoding",
            ));
        }
        if decoded.samples() == 0 {
            return Ok(None);
        }
        let track = self
            .asset
            .tracks
            .iter()
            .find(|track| track.id == self.info.track)
            .expect("selected stream retained");
        let source_pts =
            track
                .time_base
                .time(
                    decoded
                        .pts()
                        .or_else(|| decoded.timestamp())
                        .ok_or_else(|| {
                            Error::unsupported("decoded FFmpeg audio frame has no original PTS")
                        })?,
                )?;
        let start = if let Some(expected) = self.next_source {
            if source_pts != expected {
                let difference = if source_pts > expected {
                    source_pts.checked_sub(expected)?
                } else {
                    expected.checked_sub(source_pts)?
                };
                let outside_bin =
                    !self.options.allow_one_tick || difference > self.info.timestamp_quantum;
                if outside_bin && !self.options.reconstruct_timestamps {
                    return Err(Error::new(
                        ErrorKind::MissingData,
                        "FFmpeg audio timestamps contain a gap or overlap; explicitly choose decoded_sample_clock() to preserve samples on a reconstructed clock",
                    ).with_time(source_pts));
                }
                if outside_bin {
                    increment(&mut self.report.reconstructed_timestamps, 1)?;
                }
                increment(&mut self.report.timestamp_adjustments, 1)?;
                self.report.max_timestamp_adjustment = Some(
                    self.report
                        .max_timestamp_adjustment
                        .map_or(difference, |old| old.max(difference)),
                );
            }
            expected
        } else {
            source_pts
        };
        let end = start.checked_add(Time::from_samples(
            i64::try_from(decoded.samples())
                .map_err(|_| Error::overflow("FFmpeg decoded frame count"))?,
            self.info.sample_rate,
        )?)?;
        self.next_source = Some(end);
        self.report.decoded_frames = self
            .report
            .decoded_frames
            .checked_add(1)
            .ok_or_else(|| Error::overflow("FFmpeg decoded frame counter"))?;
        let discrete =
            self.info.layout.positions().iter().all(|position| {
                matches!(position, media_core::media::ChannelPosition::Discrete(_))
            });
        if !discrete && self.normalizer.is_none() {
            self.normalizer = Some(
                ffmpeg::software::resampling::Context::get(
                    decoded.format(),
                    decoded.channel_layout(),
                    decoded.rate(),
                    ffmpeg::format::Sample::F32(ffmpeg::format::sample::Type::Packed),
                    decoded.channel_layout(),
                    decoded.rate(),
                )
                .map_err(backend_error)?,
            );
        }
        self.native_format = Some(decoded.format());
        let mut normalized = ffmpeg::frame::Audio::empty();
        let values = if discrete {
            std::borrow::Cow::Owned(audio_native::discrete_samples(
                &decoded,
                self.info.layout.channels(),
            )?)
        } else {
            let delay = self
                .normalizer
                .as_mut()
                .expect("normalizer initialized")
                .run(&decoded, &mut normalized)
                .map_err(backend_error)?;
            if delay.is_some() || normalized.samples() != decoded.samples() {
                return Err(Error::unsupported(
                    "equal-rate scalar normalization changed frame count or introduced delay",
                ));
            }
            std::borrow::Cow::Borrowed(normalized.plane::<f32>(0))
        };
        let (from, to) = if self.options.trim == AudioTrimPolicy::Semantic {
            if let Some(interval) = self.info.source_interval {
                let bounds = interval.sample_bounds(start, self.info.sample_rate)?;
                let from = bounds.start.max(0).min(decoded.samples() as i64) as usize;
                let to = bounds.end.max(0).min(decoded.samples() as i64) as usize;
                (from, to.max(from))
            } else {
                (0, decoded.samples())
            }
        } else {
            (0, decoded.samples())
        };
        increment(&mut self.report.semantic_trim_start, from as u64)?;
        increment(
            &mut self.report.semantic_trim_end,
            (decoded.samples() - to) as u64,
        )?;
        if from == to {
            return Ok(None);
        }
        let valid_start =
            start.checked_add(Time::from_samples(from as i64, self.info.sample_rate)?)?;
        let first = self.report.output_frames == 0 && self.pending.is_none();
        if first && valid_start > self.info.source_start {
            self.report.leading_gap = Some(TimeRange::new(self.info.source_start, valid_start)?);
        }
        let valid_end = start.checked_add(Time::from_samples(to as i64, self.info.sample_rate)?)?;
        if self.options.trim == AudioTrimPolicy::Semantic
            && !first
            && self.returned_end != valid_start
        {
            return Err(Error::new(
                ErrorKind::MissingData,
                "FFmpeg semantic audio interval is discontinuous",
            ));
        }
        self.returned_end = valid_end;
        let mapped = match self.info.timeline {
            AudioTimeline::Source => valid_start,
            AudioTimeline::Asset => valid_start
                .checked_sub(self.info.timeline_origin.expect("asset origin validated"))?,
        };
        let channels = self.info.layout.channels();
        if values.len() != count {
            return Err(Error::backend(
                "ffmpeg",
                "normalized audio scalar count differs from validated shape",
            ));
        }
        let original =
            source_pts.checked_add(Time::from_samples(from as i64, self.info.sample_rate)?)?;
        AudioBlock::new(
            values[from * channels..to * channels].to_vec(),
            self.info.sample_rate,
            self.info.layout.clone(),
            mapped,
        )
        .map(|block| Some(block.with_source_start(Some(original))))
    }
}
fn is_again(error: ffmpeg::Error) -> bool {
    matches!(error,ffmpeg::Error::Other{errno}if errno==ffmpeg::error::EAGAIN)
}
fn increment(counter: &mut u64, amount: u64) -> Result<()> {
    *counter = counter
        .checked_add(amount)
        .ok_or_else(|| Error::overflow("FFmpeg audio report counter"))?;
    Ok(())
}
