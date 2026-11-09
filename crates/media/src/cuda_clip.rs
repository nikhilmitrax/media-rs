//! Required native CUDA video and CPU audio on one verified recording timeline.
use super::*;
use crate::cuda_request::{CudaColorAssumption, CudaFrameBatch, CudaReport, CudaTensorBatch};
use media_backend_cuda::{CudaContext, CudaLimits, PreprocessSpec};
use media_backend_ffmpeg::cuda::{CudaDecodeOptions, CudaVideoFrame};
use media_pipeline::sampling::SelectionBatch;

/// Synchronized selections whose video surfaces remain on their native CUDA device.
pub type CudaClipVideo = ClipVideo<CudaVideoFrame, CudaReport>;
/// Shared timeline, CPU waveform, and required-hardware video selections.
pub type CudaClipResult = ClipResult<CudaClipVideo>;
/// Shared timeline, CPU waveform, and one final contiguous CUDA video tensor.
pub type CudaClipTensorResult = ClipResult<CudaClipTensorVideo>;

/// Lazy synchronized extraction with required NVDEC video and native CPU audio.
///
/// Discovery and cross-demux audio verification forbid video codec probing.
/// No decoded video is transferred to host and there is no software fallback.
/// CPU audio retains the existing verified Matroska PCM/FLAC mapping limits.
/// The current synchronized GPU route requires the sole moving-video stream
/// and, when present, the sole audio stream. Other timeline tracks are rejected.
/// Origin is the minimum actual first hardware-video PTS and verified audio PTS;
/// incomplete header-derived start hints never establish this shared mapping.
#[derive(Clone, Debug)]
pub struct CudaClipRequest {
    clip: ClipRequest,
    ordinal: u32,
    limits: CudaLimits,
    context: Option<CudaContext>,
    options: CudaDecodeOptions,
    assumption: Option<CudaColorAssumption>,
}
impl ClipRequest {
    /// Require native CUDA video while keeping selected audio on the CPU.
    /// Track, uniform sampling, duration, and boundary choices remain explicit.
    pub fn cuda(self, ordinal: u32, limits: CudaLimits) -> CudaClipRequest {
        let mut options = CudaDecodeOptions::default();
        if let Some(video) = &self.video {
            options.probe = video.options.probe;
            options.duration_policy = video.options.duration_policy;
        }
        CudaClipRequest {
            clip: self,
            ordinal,
            limits,
            context: None,
            options,
            assumption: None,
        }
    }
}
impl CudaClipRequest {
    /// Select video and its exact uniform schedule before extraction.
    pub fn video(mut self, output: VideoOutput) -> Self {
        self.options.probe = output.options.probe;
        self.options.duration_policy = output.options.duration_policy;
        self.clip.video = Some(output);
        self
    }
    /// Select the CPU audio stream, mix, and continuous sample rate.
    pub fn audio(mut self, output: AudioOutput) -> Self {
        self.clip.audio = Some(output);
        self
    }
    /// Set both modality boundary policies, checked against their interpretations.
    pub fn boundary(mut self, boundary: Boundary) -> Self {
        self.clip = self.clip.boundary(boundary);
        self
    }
    /// Error, Truncate, or explicit native-frame repetition for video.
    pub fn video_boundary(mut self, boundary: Boundary) -> Self {
        self.clip.video_boundary = boundary;
        self
    }
    /// Error, Truncate, or explicit leading/trailing silence for CPU audio.
    pub fn audio_boundary(mut self, boundary: Boundary) -> Self {
        self.clip.audio_boundary = boundary;
        self
    }
    /// Share this actual context through the Session's immutable resource limits.
    pub fn on_context(mut self, context: &CudaContext) -> Self {
        self.context = Some(context.clone());
        self
    }
    /// Set explicit native surface/probe limits; VideoOutput's selected track wins.
    /// Retaining many distinct requested frames can require increasing the default
    /// eight leases; exhaustion returns an error rather than waiting on outputs.
    pub fn decode_options(mut self, options: CudaDecodeOptions) -> Self {
        self.options = options;
        self
    }
    /// Disclose missing YUV/RGB metadata during tensor preparation; contradictions fail.
    pub fn assume_missing_color(mut self, assumption: CudaColorAssumption) -> Self {
        self.assumption = Some(assumption);
        self
    }
    /// One cancellation token shared by audio and hardware-video execution.
    pub fn cancellation(&self) -> CancellationToken {
        self.clip.cancel.clone()
    }
    /// Extract on the verified shared asset timeline with no decoded host video.
    pub fn read(&self) -> Result<CudaClipResult> {
        self.clip.cancel.check()?;
        self.clip.validate()?;
        let output = self.clip.video.as_ref().ok_or_else(|| {
            Error::invalid("a CUDA clip requires an explicitly selected video output")
        })?;
        self.limits.validate()?;
        let batch = Arc::new(self.clip.asset.session.resources.reserve_batch()?);
        let source = self.clip.asset.source.fingerprint()?;
        if !source.is_versioned() {
            return Err(Error::unsupported(
                "synchronized CUDA clips require a stable source version across independent cursors",
            ));
        }
        let track = output.track.expect("validated explicit track");
        let mut options = self.options;
        options.track = Some(track);
        let mut request = self
            .clip
            .asset
            .video()?
            .cuda(self.ordinal, self.limits)
            .decode_options(options);
        if let Some(context) = &self.context {
            request = request.on_context(context);
        }
        // The required reader bootstraps the GPU before opening encoded cursors.
        // Its discovery uses probe_headers, which forbids codec-based probing.
        let mut reader = request.reader_with_cancellation(self.clip.cancel.clone())?;
        let mut info = reader
            .info()
            .ok_or_else(|| Error::backend("clip", "new CUDA reader has no header inventory"))?
            .clone();
        if info.source != source {
            return Err(Error::incompatible(
                "CUDA clip source changed during header discovery",
            ));
        }
        if !info
            .tracks
            .iter()
            .any(|t| t.id == track && t.kind == TrackKind::Video && !t.cover_art)
        {
            return Err(
                Error::unsupported("selected CUDA clip video stream is unavailable")
                    .with_track(track.0),
            );
        }
        validate_timeline_inventory(&info, track, self.clip.audio.as_ref().and_then(|a| a.track))?;
        let first = reader.next_frame()?.ok_or_else(|| {
            missing(
                "CUDA clip cannot establish a timeline without a first hardware video frame",
                track,
            )
        })?;
        let video_start = first.frame().timing.source_pts.ok_or_else(|| {
            Error::unsupported("CUDA clip needs original first hardware video PTS")
        })?;
        let audio_mapping = self
            .clip
            .audio
            .as_ref()
            .map(|output| self.clip.native_audio_track(output, &info))
            .transpose()?;
        let header_origin_hint = info.timeline_origin;
        let first_audio_source_pts = audio_mapping.map(|(_, start)| start);
        let origin = audio_mapping.map_or(video_start, |(_, start)| video_start.min(start));
        info.timeline_origin = Some(origin);
        let audio = self
            .clip
            .audio
            .as_ref()
            .zip(audio_mapping)
            .map(|(output, (native_track, _))| {
                self.clip
                    .read_audio_verified(output, origin, &batch, native_track)
            })
            .transpose()?;
        let mut first = Some(first.rebase_timeline(origin)?);
        let video = self.clip.select_video(output, &info, &batch, || {
            if first.is_some() {
                Ok(first.take())
            } else {
                reader
                    .next_frame()?
                    .map(|frame| frame.rebase_timeline(origin))
                    .transpose()
            }
        })?;
        let mut report = reader.report();
        report.output_frames = video.frames.len() as u64;
        let context = reader.context().clone();
        let video = video.with_report(report);
        self.clip.cancel.check()?;
        if source != self.clip.asset.source.fingerprint()? {
            return Err(Error::incompatible(
                "CUDA clip source changed during combined extraction",
            ));
        }
        let maximum_coverage = audio
            .as_ref()
            .map_or(Ok(video.actual_coverage.len()), |audio| {
                video
                    .actual_coverage
                    .len()
                    .checked_mul(audio.actual_coverage.len())
                    .ok_or_else(|| Error::overflow("CUDA clip common coverage count"))
            })?;
        let coverage = self.clip.asset.session.resources.host_budget().reserve(
            maximum_coverage
                .checked_mul(std::mem::size_of::<TimeRange>())
                .ok_or_else(|| Error::overflow("CUDA clip coverage storage"))?,
        )?;
        let actual_coverage = match &audio {
            Some(audio) => intersections(&video.actual_coverage, &audio.actual_coverage)?,
            None => video.actual_coverage.clone(),
        };
        Ok(ClipResult {
            requested: self.clip.range,
            timeline_origin: origin,
            timeline_evidence: ClipTimelineEvidence::VerifiedFirstSources {
                header_origin_hint,
                first_video_source_pts: video_start,
                first_audio_source_pts,
            },
            source,
            actual_coverage,
            video: Some(video),
            audio,
            _batch: batch,
            _coverage: coverage,
            _cancel: self.clip.cancel.clone(),
            _session: self.clip.asset.session.clone(),
            _cuda: Some((context, self.assumption.clone())),
        })
    }
}

fn validate_timeline_inventory(
    info: &AssetInfo,
    video: TrackId,
    audio: Option<TrackId>,
) -> Result<()> {
    if !info
        .format
        .split(',')
        .any(|name| matches!(name, "matroska" | "webm"))
    {
        return Err(Error::unsupported(
            "synchronized CUDA clip origin currently requires Matroska with verified first source timestamps; edit-list mappings are unsupported",
        ));
    }
    if info.tracks.iter().any(|track| {
        !((track.id == video && track.kind == TrackKind::Video && !track.cover_art)
            || (Some(track.id) == audio && track.kind == TrackKind::Audio))
    }) {
        return Err(Error::unsupported(
            "synchronized CUDA clip origin currently requires the sole moving-video stream and its sole explicitly selected audio stream; additional timeline tracks are not mapped",
        ));
    }
    if let Some(audio) = audio
        && !info
            .tracks
            .iter()
            .any(|track| track.id == audio && track.kind == TrackKind::Audio)
    {
        return Err(
            Error::unsupported("selected synchronized CUDA audio stream is unavailable")
                .with_track(audio.0),
        );
    }
    Ok(())
}

impl ClipFrame for CudaVideoFrame {
    fn timing(&self) -> FrameTiming {
        self.frame().timing
    }
    fn retain_batch(self, batch: Arc<BatchReservation>) -> Self {
        self.retain(batch)
    }
}

/// Metadata for one tensor row, preserving source PTS and original schedule position.
#[derive(Debug)]
pub struct ClipVideoSelection {
    /// Position in the original uniform schedule, including truncated gaps.
    pub request_index: usize,
    /// Original requested asset time.
    pub requested: Time,
    /// Actual presentation/source PTS, duration, and source ordinal.
    pub timing: FrameTiming,
    /// Established original display coverage contains the requested time.
    pub valid: bool,
    /// This row comes from explicit boundary repetition.
    pub repeated: bool,
}
/// One final device allocation plus masks, shared source timing, and CPU audio peer.
#[derive(Debug)]
pub struct CudaClipTensorVideo {
    /// Explicit selected logical stream.
    pub track: TrackId,
    /// Full schedule, including positions omitted by Truncate.
    pub requested_times: Vec<Time>,
    /// Contiguous video tensor and exact FrameIndex selection provenance.
    pub batch: CudaTensorBatch,
    /// One record per tensor row; source PTS is never rebased to the clip start.
    pub selections: Vec<ClipVideoSelection>,
    /// Original display validity at each requested schedule position.
    pub validity: Vec<bool>,
    /// Explicit repetition at each requested schedule position.
    pub repeated: Vec<bool>,
    /// Real video coverage, excluding repetitions.
    pub actual_coverage: Vec<TimeRange>,
    _metadata: HostReservation,
    _coverage: HostReservation,
    _tensor_metadata: HostReservation,
}

impl CudaClipResult {
    /// Consume all selected video surfaces into one final CUDA tensor. Audio stays
    /// in its original CPU allocation. Tensor storage/foreign consumers and audio
    /// aliases retain the same single Session batch slot independently.
    pub fn into_tensor(self, spec: PreprocessSpec) -> Result<CudaClipTensorResult> {
        self._cancel.check()?;
        let (context, assumption) = self
            ._cuda
            .clone()
            .ok_or_else(|| Error::incompatible("CUDA clip has no owning context"))?;
        let video = self
            .video
            .ok_or_else(|| Error::invalid("CUDA clip has no video output"))?;
        if video.frames.is_empty() {
            return Err(Error::invalid(
                "an empty/truncated CUDA clip has no tensor rows",
            ));
        }
        let tensor_metadata = self._session.resources.host_budget().reserve(
            video
                .frames
                .len()
                .checked_mul(
                    std::mem::size_of::<CudaVideoFrame>()
                        + std::mem::size_of::<FrameSelection>()
                        + std::mem::size_of::<FrameIndexEntry>()
                        + std::mem::size_of::<ClipVideoSelection>()
                        + 64,
                )
                .ok_or_else(|| Error::overflow("CUDA clip tensor metadata"))?,
        )?;
        let mut native = Vec::with_capacity(video.frames.len());
        let mut selections = Vec::with_capacity(video.frames.len());
        let mut rows = Vec::with_capacity(video.frames.len());
        let mut unique = BTreeMap::new();
        for output in video.frames {
            let timing = output.frame.frame().timing;
            unique
                .entry(output.selection.identity.ordinal)
                .or_insert(FrameIndexEntry::from_timing(timing, false)?);
            selections.push(output.selection);
            rows.push(ClipVideoSelection {
                request_index: output.request_index,
                requested: output.requested,
                timing,
                valid: output.valid,
                repeated: output.repeated,
            });
            native.push(output.frame);
        }
        let unique_frames: Vec<_> = unique.into_values().collect();
        for selection in &mut selections {
            selection.work_index = unique_frames
                .iter()
                .position(|entry| entry.ordinal == selection.identity.ordinal)
                .ok_or_else(|| Error::backend("clip", "selected tensor row has no source entry"))?;
        }
        let batch = CudaFrameBatch::from_native_parts(
            native,
            self.source.clone(),
            Some(SelectionBatch {
                selections,
                unique_frames,
            }),
            video.report,
            context,
            assumption,
            self._batch.clone(),
        )?
        .into_tensor(spec)?;
        self._cancel.check()?;
        Ok(ClipResult {
            requested: self.requested,
            timeline_origin: self.timeline_origin,
            timeline_evidence: self.timeline_evidence,
            source: self.source,
            actual_coverage: self.actual_coverage,
            audio: self.audio,
            video: Some(CudaClipTensorVideo {
                track: video.track,
                requested_times: video.requested_times,
                batch,
                selections: rows,
                validity: video.validity,
                repeated: video.repeated,
                actual_coverage: video.actual_coverage,
                _metadata: video._metadata,
                _coverage: video._coverage,
                _tensor_metadata: tensor_metadata,
            }),
            _batch: self._batch,
            _coverage: self._coverage,
            _cancel: self._cancel,
            _cuda: self._cuda,
            _session: self._session,
        })
    }
}
