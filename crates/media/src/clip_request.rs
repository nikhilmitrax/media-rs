//! Synchronized clips on a verified shared container timeline.
//!
//! Video uses FFmpeg's asset-relative PTS and preserves original source PTS.
//! Native audio supports PCM/FLAC in Matroska with exact first-packet and
//! timestamp verification across demuxers. With `ffmpeg-audio`, CPU MOV/MP4
//! clips use one FFmpeg demux/decoder implementation for edited AAC/audio and
//! video clocks. Internal audio gaps, unestablished/noncontinuous edit mappings,
//! and wrap remain unsupported. Required-CUDA clips keep their narrower verified
//! sole-video/sole-audio Matroska domain.
//! Audio is processed continuously from its original stream origin before range
//! extraction; each independently requested clip therefore uses the same grid.
//! Independent decoder cursors require a stable file/memory/object version;
//! unversioned/live adapters cannot establish this shared recording contract.

use crate::{request::Asset, video_request::VideoReport};
use media_backend_ffmpeg::{
    codec::VideoDecodeOptions,
    format::{self, AssetInfo, PacketReader, ProbeOptions, TrackKind},
};
use media_backend_native::audio::{AudioDecodeOptions, AudioDecodeReport, AudioReader, MixMatrix};
use media_core::{
    error::{Error, ErrorKind, Result},
    media::{AudioBlock, ChannelLayout, FrameTiming, TrackId, VideoFrame},
    policy::Boundary,
    time::{Time, TimeRange},
};
use media_io::source::{Source, SourceFingerprint};
use media_pipeline::{
    execution::CancellationToken,
    sampling::{DurationInference, FrameIndex, FrameIndexEntry, FrameSelection, TemporalSelection},
    session::{BatchReservation, HostReservation},
};
use std::{
    collections::BTreeMap,
    io::{self, Read, Seek, SeekFrom},
    ops::Range,
    sync::Arc,
};

#[cfg(feature = "cuda")]
#[path = "cuda_clip.rs"]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::{
    ClipVideoSelection, CudaClipRequest, CudaClipResult, CudaClipTensorResult, CudaClipTensorVideo,
    CudaClipVideo,
};

/// A finite, exact uniform schedule on the requested half-open clip interval.
#[derive(Clone, Copy, Debug)]
pub struct Sampling {
    count: usize,
}
impl Sampling {
    /// Request `count` times at start + i * duration/count, excluding the end.
    pub fn uniform(count: usize) -> Self {
        Self { count }
    }
}

/// Explicit video track and temporal output requirements.
#[derive(Clone, Debug)]
pub struct VideoOutput {
    track: Option<TrackId>,
    sampling: Sampling,
    durations: DurationInference,
    options: VideoDecodeOptions,
}
impl VideoOutput {
    /// CPU RGB output with one uniform selection; track selection is required.
    pub fn new() -> Self {
        Self {
            track: None,
            sampling: Sampling::uniform(1),
            durations: DurationInference::RequireExplicit,
            options: Default::default(),
        }
    }
    /// Select an FFmpeg logical stream index explicitly.
    pub fn track(mut self, track: TrackId) -> Self {
        self.track = Some(track);
        self
    }
    /// Select an exact finite uniform schedule.
    pub fn sample(mut self, sampling: Sampling) -> Self {
        self.sampling = sampling;
        self
    }
    /// Explicitly permit inference of missing durations from the following PTS.
    pub fn duration_inference(mut self, policy: DurationInference) -> Self {
        self.durations = policy;
        self
    }
    /// Replace output/color/probe controls; the explicit track remains authoritative.
    pub fn decode_options(mut self, options: VideoDecodeOptions) -> Self {
        self.options = options;
        self
    }
}
impl Default for VideoOutput {
    fn default() -> Self {
        Self::new()
    }
}

/// Explicit audio track, continuous output grid, and named channel requirements.
#[derive(Clone, Debug)]
pub struct AudioOutput {
    track: Option<TrackId>,
    rate: Option<u32>,
    layout: Option<ChannelLayout>,
    matrix: Option<MixMatrix>,
    options: AudioDecodeOptions,
}
impl AudioOutput {
    /// Interleaved f32 audio preserving source rate/layout until explicitly changed.
    /// A logical FFmpeg audio track must be selected before execution.
    pub fn f32() -> Self {
        Self {
            track: None,
            rate: None,
            layout: None,
            matrix: None,
            options: Default::default(),
        }
    }
    /// Select an FFmpeg logical stream index; native track IDs are mapped and verified.
    pub fn track(mut self, track: TrackId) -> Self {
        self.track = Some(track);
        self
    }
    /// Continuously resample from the original source origin to this rate.
    pub fn sample_rate(mut self, rate: u32) -> Self {
        self.rate = Some(rate);
        self
    }
    /// Use a checked named mono/stereo mix, preserving speaker identities.
    pub fn channels(mut self, layout: ChannelLayout) -> Self {
        self.layout = Some(layout);
        self.matrix = None;
        self
    }
    /// Supply an explicit speaker/discrete-channel mapping and coefficients.
    pub fn mix(mut self, matrix: MixMatrix) -> Self {
        self.layout = None;
        self.matrix = Some(matrix);
        self
    }
    /// Replace native decode limits; backend-specific track selection is overwritten.
    pub fn decode_options(mut self, options: AudioDecodeOptions) -> Self {
        self.options = options;
        self
    }
}

/// Lazy bounded synchronized extraction with separate modality boundary policies.
#[derive(Clone, Debug)]
pub struct ClipRequest {
    asset: Asset,
    range: TimeRange,
    video: Option<VideoOutput>,
    audio: Option<AudioOutput>,
    video_boundary: Boundary,
    audio_boundary: Boundary,
    cancel: CancellationToken,
}
impl ClipRequest {
    pub(crate) fn new(asset: Asset, range: TimeRange) -> Self {
        Self {
            asset,
            range,
            video: None,
            audio: None,
            video_boundary: Boundary::Error,
            audio_boundary: Boundary::Error,
            cancel: Default::default(),
        }
    }
    /// Include the selected video track.
    pub fn video(mut self, output: VideoOutput) -> Self {
        self.video = Some(output);
        self
    }
    /// Include the selected audio track.
    pub fn audio(mut self, output: AudioOutput) -> Self {
        self.audio = Some(output);
        self
    }
    /// Set both policies; repeat-frame/silence still require their matching modality.
    pub fn boundary(mut self, policy: Boundary) -> Self {
        self.video_boundary = policy;
        self.audio_boundary = policy;
        self
    }
    /// Error, Truncate, or RepeatFrame; Silence and Wrap are unsupported for video.
    pub fn video_boundary(mut self, policy: Boundary) -> Self {
        self.video_boundary = policy;
        self
    }
    /// Error, Truncate, or Silence; RepeatFrame and Wrap are unsupported for audio.
    pub fn audio_boundary(mut self, policy: Boundary) -> Self {
        self.audio_boundary = policy;
        self
    }
    /// Cooperative cancellation checked between bounded packet/frame/block operations.
    pub fn cancellation(&self) -> CancellationToken {
        self.cancel.clone()
    }
    /// Decode a finite CPU clip, preserving source times and actual coverage.
    ///
    /// The asset timeline is source time minus the discovered container origin.
    /// Error requires full requested interval coverage, beyond sampled points.
    /// Truncate keeps the original uniform schedule and omits missing selections;
    /// it does not silently resample a shorter video interval. Silence pads only
    /// leading/trailing audio absence; real internal discontinuities still fail.
    pub fn read(&self) -> Result<ClipResult> {
        self.cancel.check()?;
        self.validate()?;
        let batch = Arc::new(self.asset.session.resources.reserve_batch()?);
        let (source, info) = {
            let _probe = self.asset.session.resources.reserve_cpu(0, 0)?;
            let source = self.asset.source.fingerprint()?;
            if !source.is_versioned() {
                return Err(Error::unsupported(
                    "synchronized clips require a stable source version across independent decoder cursors",
                ));
            }
            let options = self
                .video
                .as_ref()
                .map_or(ProbeOptions::default(), |v| v.options.probe);
            let info = format::probe(&self.asset.source, options)?;
            (source, info)
        };
        let origin = info.timeline_origin.ok_or_else(|| {
            Error::unsupported("clip needs an established shared container timeline origin")
        })?;
        let audio = self
            .audio
            .as_ref()
            .map(|output| self.read_audio(output, &info, origin, &batch))
            .transpose()?;
        let video = self
            .video
            .as_ref()
            .map(|output| self.read_video(output, &info, &batch))
            .transpose()?;
        self.cancel.check()?;
        let _finalize = self.asset.session.resources.reserve_cpu(0, 0)?;
        if source.is_versioned() && source != self.asset.source.fingerprint()? {
            return Err(Error::incompatible("clip source changed during extraction"));
        }
        let maximum_coverage = match (&video, &audio) {
            (Some(video), Some(audio)) => video
                .actual_coverage
                .len()
                .checked_mul(audio.actual_coverage.len())
                .ok_or_else(|| Error::overflow("clip common coverage count"))?,
            (Some(video), None) => video.actual_coverage.len(),
            (None, Some(audio)) => audio.actual_coverage.len(),
            (None, None) => 0,
        };
        let coverage = self.asset.session.resources.host_budget().reserve(
            maximum_coverage
                .checked_mul(std::mem::size_of::<TimeRange>())
                .ok_or_else(|| Error::overflow("clip common coverage storage"))?,
        )?;
        let actual_coverage = match (&video, &audio) {
            (Some(video), Some(audio)) => {
                intersections(&video.actual_coverage, &audio.actual_coverage)?
            }
            (Some(video), None) => video.actual_coverage.clone(),
            (None, Some(audio)) => audio.actual_coverage.clone(),
            (None, None) => unreachable!(),
        };
        Ok(ClipResult {
            requested: self.range,
            timeline_origin: origin,
            timeline_evidence: ClipTimelineEvidence::ContainerProbe,
            source,
            actual_coverage,
            video,
            audio,
            _batch: batch,
            _coverage: coverage,
            _cancel: self.cancel.clone(),
            _session: self.asset.session.clone(),
            #[cfg(feature = "cuda")]
            _cuda: None,
        })
    }

    fn validate(&self) -> Result<()> {
        if self.video.is_none() && self.audio.is_none() {
            return Err(Error::invalid(
                "clip requires an explicit audio or video output",
            ));
        }
        if let Some(video) = &self.video {
            if video.track.is_none() {
                return Err(Error::invalid(
                    "clip video track must be selected explicitly",
                ));
            }
            if video.sampling.count == 0 || video.sampling.count > 65536 {
                return Err(Error::invalid(
                    "clip uniform sampling count must be in 1..=65536",
                ));
            }
            if !matches!(
                self.video_boundary,
                Boundary::Error | Boundary::Truncate | Boundary::RepeatFrame
            ) {
                return Err(Error::unsupported(
                    "clip video supports Error, Truncate, or RepeatFrame boundaries; silence/wrap are undefined",
                ));
            }
        }
        if let Some(audio) = &self.audio {
            if audio.track.is_none() {
                return Err(Error::invalid(
                    "clip audio track must be selected explicitly",
                ));
            }
            if audio.rate == Some(0) {
                return Err(Error::invalid("clip sample rate must be nonzero"));
            }
            if !matches!(
                self.audio_boundary,
                Boundary::Error | Boundary::Truncate | Boundary::Silence
            ) {
                return Err(Error::unsupported(
                    "clip audio supports Error, Truncate, or Silence boundaries; repeat/wrap are undefined",
                ));
            }
        }
        Ok(())
    }

    fn read_video(
        &self,
        output: &VideoOutput,
        info: &AssetInfo,
        batch: &Arc<BatchReservation>,
    ) -> Result<ClipVideo> {
        let track = output.track.expect("validated track");
        if !info
            .tracks
            .iter()
            .any(|t| t.id == track && t.kind == TrackKind::Video && !t.cover_art)
        {
            return Err(
                Error::unsupported("selected clip video stream is unavailable").with_track(track.0),
            );
        }
        let mut reader = self
            .asset
            .video()?
            .decode_options(output.options)
            .track(track)
            .reader()?;
        if reader.info().timeline_origin != info.timeline_origin {
            return Err(Error::incompatible(
                "video decoder changed the shared timeline mapping",
            ));
        }
        let video = self.select_video(output, info, batch, || reader.next_frame())?;
        let mut report = reader.report();
        report.output_frames = video.frames.len() as u64;
        Ok(video.with_report(report))
    }

    fn select_video<F: ClipFrame>(
        &self,
        output: &VideoOutput,
        info: &AssetInfo,
        batch: &Arc<BatchReservation>,
        mut next_frame: impl FnMut() -> Result<Option<F>>,
    ) -> Result<ClipVideo<F, ()>> {
        let track = output.track.expect("validated track");
        let metadata = self.asset.session.resources.host_budget().reserve(
            (if self.range.is_empty() {
                0
            } else {
                output.sampling.count
            })
            .checked_mul(
                1024usize
                    .checked_add(self.asset.source.label().len())
                    .ok_or_else(|| Error::overflow("clip source label size"))?,
            )
            .ok_or_else(|| Error::overflow("clip video selection metadata"))?,
        )?;
        let times = uniform_times(self.range, output.sampling.count)?;
        let mut candidates: Vec<Option<(FrameIndexEntry, F)>> = vec![None; times.len()];
        let mut preceding: Vec<Option<(FrameIndexEntry, F)>> = vec![None; times.len()];
        let mut previous: Option<(FrameIndexEntry, F)> = None;
        let mut first: Option<(FrameIndexEntry, F)> = None;
        let mut entries = Vec::new();
        let mut entry_guards = Vec::new();
        if !self.range.is_empty() {
            while let Some(frame) = next_frame()? {
                self.cancel.check()?;
                let timing = frame.timing();
                let pts = timing
                    .pts
                    .ok_or_else(|| Error::unsupported("clip requires established video PTS"))?;
                let ordinal = timing
                    .ordinal
                    .ok_or_else(|| Error::unsupported("clip requires presentation ordinals"))?;
                if previous.as_ref().is_some_and(|(last, _)| pts < last.pts) {
                    return Err(Error::new(
                        ErrorKind::AmbiguousTiming,
                        "clip decoder returned nonmonotonic presentation timestamps",
                    ));
                }
                let entry = FrameIndexEntry {
                    pts,
                    duration: timing.duration,
                    ordinal,
                    keyframe: false,
                };
                entry_guards.push(
                    self.asset
                        .session
                        .resources
                        .host_budget()
                        .reserve(std::mem::size_of::<FrameIndexEntry>() * 4)?,
                );
                entries.push(entry);
                if first.is_none()
                    && self.video_boundary == Boundary::RepeatFrame
                    && times.iter().any(|time| *time < pts)
                {
                    first = Some((entry, frame.clone()));
                }
                for (i, &time) in times.iter().enumerate() {
                    let mut local = Vec::with_capacity(3);
                    if let Some((candidate, _)) = &candidates[i] {
                        local.push(*candidate);
                    }
                    if let Some((last, _)) = &previous
                        && !local.iter().any(|e| e.ordinal == last.ordinal)
                    {
                        local.push(*last);
                    }
                    if !local.iter().any(|e| e.ordinal == entry.ordinal) {
                        local.push(entry);
                    }
                    let index = FrameIndex::new(info.source.clone(), track.0, local)?;
                    if let Ok(chosen) =
                        index.select(time, TemporalSelection::DisplayedAt, output.durations)
                    {
                        if chosen.identity.ordinal == entry.ordinal {
                            candidates[i] = Some((entry, frame.clone()));
                        } else if let Some((last, last_frame)) = &previous
                            && chosen.identity.ordinal == last.ordinal
                        {
                            candidates[i] = Some((*last, last_frame.clone()));
                        }
                    }
                    if self.video_boundary == Boundary::RepeatFrame
                        && pts <= time
                        && preceding[i].as_ref().is_none_or(|(last, _)| pts > last.pts)
                    {
                        preceding[i] = Some((entry, frame.clone()));
                    }
                }
                previous = Some((entry, frame));
                if pts > self.range.end() {
                    break;
                }
            }
        }
        let index = FrameIndex::new(info.source.clone(), track.0, entries)?;
        let actual_coverage = video_coverage(&index, self.range, output.durations)?;
        let coverage = self.asset.session.resources.host_budget().reserve(
            actual_coverage
                .capacity()
                .checked_mul(std::mem::size_of::<TimeRange>())
                .ok_or_else(|| Error::overflow("clip video coverage storage"))?,
        )?;
        if self.video_boundary == Boundary::Error && !covers(&actual_coverage, self.range) {
            return Err(missing(
                "video does not cover the full requested clip interval",
                track,
            ));
        }
        let mut retained = BTreeMap::new();
        for (entry, frame) in candidates.into_iter().chain(preceding).flatten() {
            retained.insert(entry.ordinal, frame);
        }
        if let Some((entry, frame)) = first {
            retained.insert(entry.ordinal, frame);
        }
        if let Some((entry, frame)) = previous {
            retained.insert(entry.ordinal, frame);
        }
        let mut frames = Vec::with_capacity(times.len());
        let mut validity = vec![false; times.len()];
        let mut repeated = vec![false; times.len()];
        for (request_index, &time) in times.iter().enumerate() {
            self.cancel.check()?;
            let (selection, valid) =
                match index.select(time, TemporalSelection::DisplayedAt, output.durations) {
                    Ok(selection) => (selection, true),
                    Err(error) if error.kind() == ErrorKind::MissingData => {
                        match self.video_boundary {
                            Boundary::Error => return Err(error),
                            Boundary::Truncate => continue,
                            Boundary::RepeatFrame => {
                                let selected = index
                                    .select(time, TemporalSelection::AtOrBefore, output.durations)
                                    .or_else(|_| {
                                        index.select(
                                            time,
                                            TemporalSelection::AtOrAfter,
                                            output.durations,
                                        )
                                    })?;
                                (selected, false)
                            }
                            _ => unreachable!(),
                        }
                    }
                    Err(error) => return Err(error),
                };
            let frame = retained
                .get(&selection.identity.ordinal)
                .ok_or_else(|| {
                    Error::backend(
                        "clip",
                        "selected video frame was not retained by the bounded scheduler",
                    )
                })?
                .clone();
            let frame = frame.retain_batch(batch.clone());
            validity[request_index] = valid;
            repeated[request_index] = !valid;
            frames.push(ClipVideoFrame {
                request_index,
                requested: time,
                frame,
                selection,
                valid,
                repeated: !valid,
            });
        }
        Ok(ClipVideo {
            track,
            requested_times: times,
            frames,
            validity,
            repeated,
            actual_coverage,
            report: (),
            _metadata: metadata,
            _coverage: coverage,
        })
    }

    fn native_audio_track(&self, output: &AudioOutput, info: &AssetInfo) -> Result<(u32, Time)> {
        let _identity = self.asset.session.resources.reserve_cpu(2, 2)?;
        let track = output.track.expect("validated track");
        if !info
            .format
            .split(',')
            .any(|name| matches!(name, "matroska" | "webm"))
        {
            return Err(Error::unsupported(
                "synchronized native audio currently requires Matroska PCM/FLAC with verified packet/timeline identity; edit-list mappings are unsupported",
            ));
        }
        let tracks: Vec<_> = info
            .tracks
            .iter()
            .filter(|t| t.kind == TrackKind::Audio)
            .collect();
        let ordinal = tracks.iter().position(|t| t.id == track).ok_or_else(|| {
            Error::unsupported("selected clip audio stream is unavailable").with_track(track.0)
        })?;
        if !matches!(
            tracks[ordinal].codec.as_str(),
            "pcm_s16le" | "pcm_s24le" | "pcm_s32le" | "pcm_f32le" | "flac"
        ) {
            return Err(Error::unsupported(
                "synchronized native audio codec needs unimplemented priming/edit/packet mapping validation; use PCM or FLAC in Matroska",
            ));
        }
        let mut options = output.options.clone();
        options.track = None;
        options.max_packet_bytes = options
            .max_packet_bytes
            .min(self.asset.session.limits().host_memory / 4);
        options.max_buffered_samples = options
            .max_buffered_samples
            .min(self.asset.session.limits().host_memory / 16);
        let native = AudioReader::open(&self.asset.source, options.clone())?;
        let native_ids: Vec<_> = native.audio_track_ids().collect();
        if native_ids.len() != tracks.len() {
            return Err(Error::unsupported(
                "audio track inventories disagree across demuxers; mapping cannot be established",
            ));
        }
        if native_ids.len() > 32 {
            return Err(Error::exhausted(
                "clip audio identity mapping is limited to 32 audio tracks",
            ));
        }
        let scan_options = options.clone();
        options.max_probe_bytes /= native_ids.len().max(1) as u64;
        if options.max_probe_bytes == 0 {
            return Err(Error::exhausted(
                "clip native track identity probe budget is empty",
            ));
        }
        let matches = |candidate: &AudioReader, packet: &media_core::media::Packet| {
            !candidate.has_declared_trimming()
                && candidate.codec_short_name() == Some(tracks[ordinal].codec.as_str())
                && packet.pts == Some(candidate.info().start)
                && candidate.first_packet_bytes() == Some(packet.data.as_ref())
        };
        let find_native = |packet: &media_core::media::Packet| -> Result<(u32, Time)> {
            let mut matched = None;
            for &id in &native_ids {
                self.cancel.check()?;
                let matched_start = if id == native.info().track {
                    matches(&native, packet).then_some(native.info().start)
                } else {
                    let mut options = options.clone();
                    options.track = Some(id);
                    let candidate = AudioReader::open(&self.asset.source, options)?;
                    matches(&candidate, packet).then_some(candidate.info().start)
                };
                if let Some(start) = matched_start
                    && matched.replace((id, start)).is_some()
                {
                    return Err(Error::new(
                        ErrorKind::AmbiguousTiming,
                        "multiple native audio tracks match the selected stream's first packet/PTS/codec; identity cannot be established uniquely",
                    ));
                }
            }
            matched.ok_or_else(||Error::unsupported("selected audio packet, codec, original PTS, or trim mapping differ between demuxers; shared timeline cannot be assumed"))
        };
        let identity_source =
            bounded_identity_source(&self.asset.source, scan_options.max_probe_bytes)?;
        let mut packets = PacketReader::open_headers(
            &identity_source,
            ProbeOptions {
                max_read_bytes: scan_options.max_probe_bytes,
                ..Default::default()
            },
        )?;
        let mut bytes = 0_u64;
        for _ in 0..1024 {
            self.cancel.check()?;
            let Some(packet) = packets.next_packet_with_side_data()? else {
                return Err(missing(
                    "audio mapping requires a selected first packet",
                    track,
                ));
            };
            bytes = bytes
                .checked_add(packet.packet.data.len() as u64)
                .ok_or_else(|| Error::overflow("clip audio mapping bytes"))?;
            if bytes > scan_options.max_probe_bytes {
                return Err(Error::exhausted(
                    "clip audio identity scan exceeded the probe packet-byte budget",
                ));
            }
            if packet.packet.track != track {
                continue;
            }
            if !packet.side_data.is_empty() {
                return Err(Error::unsupported(
                    "selected audio's packet identity, original PTS, or trim metadata differ between demuxers; shared timeline cannot be assumed",
                ));
            }
            return find_native(&packet.packet);
        }
        Err(Error::exhausted(
            "clip audio identity scan exceeded 1024 packets",
        ))
    }

    fn read_audio(
        &self,
        output: &AudioOutput,
        info: &AssetInfo,
        origin: Time,
        batch: &Arc<BatchReservation>,
    ) -> Result<ClipAudio> {
        #[cfg(feature = "ffmpeg-audio")]
        if info
            .format
            .split(',')
            .any(|name| matches!(name, "mov" | "mp4" | "m4a" | "3gp" | "3g2" | "mj2"))
        {
            let track = output.track.expect("validated explicit track");
            if !info
                .tracks
                .iter()
                .any(|candidate| candidate.id == track && candidate.kind == TrackKind::Audio)
            {
                return Err(
                    Error::unsupported("selected FFmpeg clip audio track is unavailable")
                        .with_track(track.0),
                );
            }
            let mut options = output.options.clone();
            options.track = Some(track.0);
            let request = self.asset.audio()?.decode_options(options).ffmpeg();
            return self.read_audio_request(output, origin, batch, track.0, request);
        }
        let (native_track, _) = self.native_audio_track(output, info)?;
        self.read_audio_verified(output, origin, batch, native_track)
    }

    fn read_audio_verified(
        &self,
        output: &AudioOutput,
        origin: Time,
        batch: &Arc<BatchReservation>,
        native_track: u32,
    ) -> Result<ClipAudio> {
        let mut options = output.options.clone();
        options.track = Some(native_track);
        let request = self.asset.audio()?.decode_options(options);
        self.read_audio_request(output, origin, batch, native_track, request)
    }

    fn read_audio_request(
        &self,
        output: &AudioOutput,
        origin: Time,
        batch: &Arc<BatchReservation>,
        native_track: u32,
        mut request: crate::audio_request::AudioRequest,
    ) -> Result<ClipAudio> {
        let track = output.track.expect("validated track");
        let backend = request.backend();
        if let Some(rate) = output.rate {
            request = request.resample(rate);
        }
        if let Some(matrix) = &output.matrix {
            request = request.mix(matrix.clone());
        } else if let Some(layout) = &output.layout {
            request = request.downmix(layout.clone());
        }
        let mut stream = request.reader()?.with_cancellation(self.cancel.clone());
        let source_grid_origin = stream.info().start;
        let grid_origin = source_grid_origin.checked_sub(origin)?;
        let rate = stream.output_rate();
        let layout = stream.output_layout().clone();
        let bounds = self.range.sample_bounds(grid_origin, rate)?;
        let frames = usize::try_from(
            bounds
                .end
                .checked_sub(bounds.start)
                .ok_or_else(|| Error::overflow("clip sample span"))?,
        )
        .map_err(|_| Error::overflow("clip sample span"))?;
        let samples = frames
            .checked_mul(layout.channels())
            .ok_or_else(|| Error::overflow("clip audio sample count"))?;
        let bytes = samples
            .checked_mul(4)
            .ok_or_else(|| Error::overflow("clip audio byte count"))?;
        let staging = self.asset.session.resources.host_budget().reserve(bytes)?;
        let metadata = self.asset.session.resources.host_budget().reserve(
            frames
                .checked_mul(2)
                .and_then(|bytes| bytes.checked_add(std::mem::size_of::<TimeRange>()))
                .ok_or_else(|| Error::overflow("clip audio masks"))?,
        )?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(samples)
            .map_err(|_| Error::exhausted("clip audio allocation failed"))?;
        values.resize(samples, 0.0);
        let mut validity = vec![false; frames];
        let source_end = self.range.end().checked_add(origin)?;
        let mut first_source_time = None;
        if !self.range.is_empty() && source_end > source_grid_origin {
            while let Some(block) = stream.next_block()? {
                self.cancel.check()?;
                let block_bounds = TimeRange::new(block.start(), block.end()?)?
                    .sample_bounds(source_grid_origin, rate)?;
                let start = block_bounds.start.max(bounds.start);
                let end = block_bounds.end.min(bounds.end);
                if start < end {
                    let from = usize::try_from(start - block_bounds.start)
                        .map_err(|_| Error::overflow("clip audio block offset"))?;
                    let to = usize::try_from(start - bounds.start)
                        .map_err(|_| Error::overflow("clip audio destination offset"))?;
                    let count = usize::try_from(end - start)
                        .map_err(|_| Error::overflow("clip audio copy length"))?;
                    let channels = layout.channels();
                    values[to * channels..(to + count) * channels].copy_from_slice(
                        &block.samples()[from * channels..(from + count) * channels],
                    );
                    validity[to..to + count].fill(true);
                    if first_source_time.is_none() {
                        first_source_time = block
                            .source_start()
                            .map(|time| time.checked_add(Time::from_samples(from as i64, rate)?))
                            .transpose()?;
                    }
                }
                if block.end()? >= source_end {
                    break;
                }
            }
        }
        let observed = stream.source_coverage()?;
        let actual_coverage = intersect_range(
            self.range,
            TimeRange::new(grid_origin, observed.end().checked_sub(origin)?)?,
        )
        .into_iter()
        .collect::<Vec<_>>();
        if self.audio_boundary == Boundary::Error && !covers(&actual_coverage, self.range) {
            return Err(missing(
                "audio does not cover the full requested clip interval",
                track,
            ));
        }
        let first = validity.iter().position(|v| *v).unwrap_or(0);
        let last = validity.iter().rposition(|v| *v).map_or(first, |n| n + 1);
        if validity[first..last].iter().any(|v| !*v) {
            return Err(missing(
                "clip has an internal audio gap; padding across discontinuities is unsupported",
                track,
            ));
        }
        let returned = if self.audio_boundary == Boundary::Truncate {
            values.copy_within(first * layout.channels()..last * layout.channels(), 0);
            values.truncate((last - first) * layout.channels());
            validity.copy_within(first..last, 0);
            validity.truncate(last - first);
            bounds
                .start
                .checked_add(first as i64)
                .ok_or_else(|| Error::overflow("clip sample start"))?
                ..bounds
                    .start
                    .checked_add(last as i64)
                    .ok_or_else(|| Error::overflow("clip sample end"))?
        } else {
            bounds.clone()
        };
        let padding = validity.iter().map(|valid| !*valid).collect();
        let start = grid_origin.checked_add(Time::from_samples(returned.start, rate)?)?;
        let source_start = if validity.first() == Some(&true) {
            first_source_time
        } else {
            None
        };
        let report = stream.decode_report().clone();
        #[cfg(feature = "ffmpeg-audio")]
        let ffmpeg_report = stream.ffmpeg_decode_report().cloned();
        drop(stream);
        let final_guard = self.asset.session.resources.host_budget().reserve(
            values
                .len()
                .checked_mul(4)
                .ok_or_else(|| Error::overflow("clip final audio size"))?,
        )?;
        let block = AudioBlock::new(values, rate, layout, start)?
            .with_source_start(source_start)
            .retain(batch.clone())
            .retain(final_guard);
        drop(staging);
        Ok(ClipAudio {
            track,
            native_track,
            backend,
            #[cfg(feature = "ffmpeg-audio")]
            ffmpeg_report,
            block,
            sample_grid_origin: grid_origin,
            source_grid_origin,
            requested_sample_bounds: bounds,
            returned_sample_bounds: returned,
            validity,
            padding,
            actual_coverage,
            source_coverage: observed,
            report,
            _metadata: metadata,
        })
    }
}

/// One selected display image with both requested asset time and original source PTS.
#[derive(Debug)]
pub struct ClipVideoFrame<F = VideoFrame> {
    /// Position in the original uniform schedule, even when Truncate omitted others.
    pub request_index: usize,
    /// Requested asset-relative time.
    pub requested: Time,
    /// Pixels and preserved asset/source presentation timing.
    pub frame: F,
    /// Exact FrameIndex provenance; actual time may precede the clip start.
    pub selection: FrameSelection,
    /// True when original established display coverage contains the requested time.
    pub valid: bool,
    /// True only for explicit boundary repetition, not ordinary same-frame selections.
    pub repeated: bool,
}

/// Video selections and masks aligned to the full original uniform schedule.
#[derive(Debug)]
pub struct ClipVideo<F = VideoFrame, R = VideoReport> {
    /// Explicit logical video stream.
    pub track: TrackId,
    /// Full requested schedule, including selections omitted by Truncate.
    pub requested_times: Vec<Time>,
    /// Returned selections; request_index maps each to the original schedule.
    pub frames: Vec<ClipVideoFrame<F>>,
    /// Original display validity at each requested position; false for omissions/repeats.
    pub validity: Vec<bool>,
    /// Explicit repeat mask at each requested position.
    pub repeated: Vec<bool>,
    /// Established source coverage intersected with the requested asset interval.
    pub actual_coverage: Vec<TimeRange>,
    /// Actual dependency decoding and output work.
    pub report: R,
    _metadata: HostReservation,
    _coverage: HostReservation,
}

/// Audio on the original continuous sample grid, with explicit padding provenance.
#[derive(Debug)]
pub struct ClipAudio {
    /// Explicit logical FFmpeg stream index.
    pub track: TrackId,
    /// Decoder-specific track ID: verified native ID for Native; logical FFmpeg
    /// ID for Ffmpeg. This never assumes equality across independent demuxers.
    pub native_track: u32,
    /// Actual decoder route; all mix/resample/window stages remain shared.
    pub backend: crate::audio_request::AudioBackend,
    /// FFmpeg automatic skip and additional semantic-tail evidence, when used.
    #[cfg(feature = "ffmpeg-audio")]
    pub ffmpeg_report: Option<media_backend_ffmpeg::audio::AudioDecodeReport>,
    /// Returned waveform; its start is the actual first sample on the asset grid.
    pub block: AudioBlock,
    /// Original audio sample-grid anchor in asset time, including stream offset.
    pub sample_grid_origin: Time,
    /// The same grid anchor on the unmodified source timeline.
    pub source_grid_origin: Time,
    /// Exact ceil-rounded half-open indices requested on that grid, possibly negative.
    pub requested_sample_bounds: Range<i64>,
    /// Actual returned indices; Truncate may reduce this range.
    pub returned_sample_bounds: Range<i64>,
    /// One boolean per returned sample frame, true for actual source samples.
    pub validity: Vec<bool>,
    /// One boolean per returned sample frame, true only for explicit silence padding.
    pub padding: Vec<bool>,
    /// True semantic source coverage intersected with the requested asset interval.
    pub actual_coverage: Vec<TimeRange>,
    /// Original source time interval established by dependency decoding.
    pub source_coverage: TimeRange,
    /// Actual native decoder counters and timestamp adjustments.
    pub report: AudioDecodeReport,
    _metadata: HostReservation,
}

/// Evidence used to establish the clip's common source-to-asset timeline.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ClipTimelineEvidence {
    /// FFmpeg container discovery, which may use codec probing for CPU clips.
    ContainerProbe,
    /// Sole selected Matroska streams, established from actual first source PTS.
    /// Header-only probing can misidentify reordered video start; its hint is
    /// recorded but does not override actual hardware-presentation evidence.
    VerifiedFirstSources {
        /// Baseline header-only origin, including an incorrect/incomplete hint.
        header_origin_hint: Option<Time>,
        /// Original PTS of the first actual hardware video presentation frame.
        first_video_source_pts: Time,
        /// Verified first PCM/FLAC source packet PTS when audio is selected.
        first_audio_source_pts: Option<Time>,
    },
}

/// Synchronized finite clip with requested, established, and original-source timing.
#[derive(Debug)]
pub struct ClipResult<V = ClipVideo> {
    /// Original half-open asset-relative request, never silently shortened.
    pub requested: TimeRange,
    /// Source time mapped to asset zero; source_time = asset_time + timeline_origin.
    pub timeline_origin: Time,
    /// How the original-source timeline was established, including overridden hints.
    pub timeline_evidence: ClipTimelineEvidence,
    /// Shared byte-object identity/version.
    pub source: SourceFingerprint,
    /// Common real coverage of the selected modalities, excluding repeat/silence.
    pub actual_coverage: Vec<TimeRange>,
    /// Selected video, when requested.
    pub video: Option<V>,
    /// Selected audio, when requested.
    pub audio: Option<ClipAudio>,
    _batch: Arc<BatchReservation>,
    _coverage: HostReservation,
    _cancel: CancellationToken,
    _session: crate::Session,
    #[cfg(feature = "cuda")]
    _cuda: Option<(
        media_backend_cuda::CudaContext,
        Option<crate::cuda_request::CudaColorAssumption>,
    )>,
}

impl<V> ClipResult<V> {
    /// Shared extraction/preprocessing cancellation, independent of output owners.
    pub fn cancellation(&self) -> CancellationToken {
        self._cancel.clone()
    }
}

trait ClipFrame: Clone {
    fn timing(&self) -> FrameTiming;
    fn retain_batch(self, batch: Arc<BatchReservation>) -> Self;
}
impl ClipFrame for VideoFrame {
    fn timing(&self) -> FrameTiming {
        self.timing
    }
    fn retain_batch(mut self, batch: Arc<BatchReservation>) -> Self {
        self.image = self.image.retain(batch);
        self
    }
}
impl<F, R> ClipVideo<F, R> {
    fn with_report<T>(self, report: T) -> ClipVideo<F, T> {
        ClipVideo {
            track: self.track,
            requested_times: self.requested_times,
            frames: self.frames,
            validity: self.validity,
            repeated: self.repeated,
            actual_coverage: self.actual_coverage,
            report,
            _metadata: self._metadata,
            _coverage: self._coverage,
        }
    }
}

fn missing(message: &str, track: TrackId) -> Error {
    Error::new(ErrorKind::MissingData, message).with_track(track.0)
}
fn uniform_times(range: TimeRange, count: usize) -> Result<Vec<Time>> {
    if range.is_empty() {
        return Ok(Vec::new());
    }
    let duration = range.duration()?;
    (0..count)
        .map(|i| {
            range
                .start()
                .checked_add(duration.checked_mul_ratio(i as i64, count as u32)?)
        })
        .collect()
}
fn intersect_range(left: TimeRange, right: TimeRange) -> Option<TimeRange> {
    let start = left.start().max(right.start());
    let end = left.end().min(right.end());
    (start < end).then(|| TimeRange::new(start, end).expect("ordered intersection"))
}
fn intersections(left: &[TimeRange], right: &[TimeRange]) -> Result<Vec<TimeRange>> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(
            left.len()
                .checked_mul(right.len())
                .ok_or_else(|| Error::overflow("clip common coverage count"))?,
        )
        .map_err(|_| Error::exhausted("clip common coverage allocation failed"))?;
    for &left in left {
        for &right in right {
            if let Some(range) = intersect_range(left, right) {
                result.push(range);
            }
        }
    }
    result.sort_by_key(|range| range.start());
    Ok(result)
}
fn covers(coverage: &[TimeRange], requested: TimeRange) -> bool {
    requested.is_empty() || coverage.len() == 1 && coverage[0] == requested
}
fn video_coverage(
    index: &FrameIndex,
    requested: TimeRange,
    inference: DurationInference,
) -> Result<Vec<TimeRange>> {
    let mut coverage: Vec<TimeRange> = Vec::new();
    for entry in index.entries() {
        let duration = match entry.duration {
            Some(duration) => Some(duration),
            None if inference == DurationInference::NextTimestamp => index
                .entries()
                .iter()
                .find(|next| next.pts > entry.pts)
                .map(|next| next.pts.checked_sub(entry.pts))
                .transpose()?,
            None => None,
        };
        let Some(duration) = duration else {
            continue;
        };
        let Some(range) = intersect_range(
            requested,
            TimeRange::new(entry.pts, entry.pts.checked_add(duration)?)?,
        ) else {
            continue;
        };
        if let Some(last) = coverage.last_mut() {
            if range.start() < last.end() {
                return Err(Error::new(
                    ErrorKind::AmbiguousTiming,
                    "overlapping video presentation intervals within the clip",
                ));
            }
            if range.start() == last.end() {
                *last = TimeRange::new(last.start(), range.end())?;
                continue;
            }
        }
        coverage.push(range);
    }
    Ok(coverage)
}

// PacketReader deliberately permits continued streaming after its own header
// probe. Identity verification needs a harder bound through the first selected
// packet, so this source counts every actual read, including seek/reread bytes.
struct IdentityReadBudget<R> {
    inner: R,
    remaining: u64,
}
impl<R: Read> Read for IdentityReadBudget<R> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        if buffer.is_empty() {
            return Ok(0);
        }
        if self.remaining == 0 {
            return Err(io::Error::other(
                "clip audio identity physical-read budget exhausted",
            ));
        }
        let allowed = buffer
            .len()
            .min(usize::try_from(self.remaining).unwrap_or(usize::MAX));
        let read = self.inner.read(&mut buffer[..allowed])?;
        self.remaining -= read as u64;
        Ok(read)
    }
}
impl<R: Seek> Seek for IdentityReadBudget<R> {
    fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
        self.inner.seek(position)
    }
}
fn bounded_identity_source(source: &Source, limit: u64) -> Result<Source> {
    let capabilities = source.capabilities()?;
    let source = source.clone();
    if capabilities.seekable {
        Source::from_seekable_factory(
            "clip audio identity probe",
            None,
            capabilities.known_length,
            move || {
                Ok(IdentityReadBudget {
                    inner: source.open().map_err(io::Error::other)?,
                    remaining: limit,
                })
            },
        )
    } else {
        Source::from_reader_factory(
            "clip audio identity probe",
            None,
            capabilities.known_length,
            move || {
                Ok(IdentityReadBudget {
                    inner: source.open_reader().map_err(io::Error::other)?,
                    remaining: limit,
                })
            },
        )
    }
}
