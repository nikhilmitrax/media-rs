//! Required NVDEC decoding and GPU-resident, caller-ordered model inputs.
//!
//! This path performs no decoded host transfer or software fallback. RGB output
//! retains the input primaries and nonlinear transfer function; normalization
//! does not imply ICC conversion, transfer-function conversion, or tone mapping.

use media_pipeline::session::{BatchReservation, HostReservation};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

use media_backend_cuda::{
    ChromaSiting, CudaContext, CudaLimits, CudaTensor, CudaYuvImage, PreprocessSpec, YuvMatrix,
    YuvRange, preprocess_batch,
};
use media_backend_ffmpeg::{
    cuda::{CudaDecodeOptions, CudaVideoFrame, CudaVideoReader},
    format::{AssetInfo, ProbeOptions},
};
use media_core::{
    error::{Error, Result},
    media::{ColorDescription, Orientation, TrackId},
    time::Time,
};
use media_io::source::SourceFingerprint;
use media_pipeline::{
    execution::CancellationToken,
    sampling::{DurationInference, FrameIndex, FrameIndexEntry, SelectionBatch, TemporalSelection},
};

use crate::request::Asset;
use crate::video_index::{
    IndexBuildLimits, IndexSeekLimits, IndexedDecodeState, IndexedTiming, VideoIndex,
};

#[derive(Debug, Clone)]
enum Target {
    New { ordinal: u32, limits: CudaLimits },
    Existing(CudaContext),
}

/// Explicit interpretation for absent YUV/RGB metadata.
///
/// Established contradictory metadata is rejected. This describes the input
/// nonlinear RGB interpretation after matrix conversion, rather than requesting
/// a conversion to a different color space.
#[derive(Debug, Clone)]
pub struct CudaColorAssumption {
    /// YUV matrix used when absent.
    pub matrix: YuvMatrix,
    /// YUV code range used when absent.
    pub range: YuvRange,
    /// Chroma sample location used when absent.
    pub chroma: ChromaSiting,
    /// RGB primary name, for example `bt709`.
    pub primaries: String,
    /// Nonlinear transfer name, for example `bt709`.
    pub transfer: String,
}

/// Lazy required-hardware video request with explicit native-surface budgets.
#[derive(Debug, Clone)]
pub struct CudaVideoRequest {
    asset: Asset,
    target: Target,
    options: CudaDecodeOptions,
    times: Option<Vec<Time>>,
    selection: TemporalSelection,
    durations: DurationInference,
    color_assumption: Option<CudaColorAssumption>,
    index: Option<Arc<VideoIndex>>,
    indexed_limits: IndexSeekLimits,
    cancel: Option<CancellationToken>,
}

impl CudaVideoRequest {
    pub(crate) fn new(asset: Asset, ordinal: u32, limits: CudaLimits) -> Self {
        Self {
            asset,
            target: Target::New { ordinal, limits },
            options: CudaDecodeOptions::default(),
            times: None,
            selection: TemporalSelection::DisplayedAt,
            durations: DurationInference::RequireExplicit,
            color_assumption: None,
            index: None,
            indexed_limits: IndexSeekLimits::default(),
            cancel: None,
        }
    }

    pub(crate) fn inherit(
        mut self,
        track: Option<TrackId>,
        probe: ProbeOptions,
        duration_policy: media_core::policy::FrameDurationPolicy,
        times: Option<Vec<Time>>,
        selection: TemporalSelection,
        durations: DurationInference,
    ) -> Self {
        self.options.track = track;
        self.options.probe = probe;
        self.options.duration_policy = duration_policy;
        self.times = times;
        self.selection = selection;
        self.durations = durations;
        self
    }

    /// Share the supplied actual native context; ordinal equality is insufficient.
    pub fn on_context(mut self, context: &CudaContext) -> Self {
        self.target = Target::Existing(context.clone());
        self
    }

    /// Replace hardware decoder probing and retained-surface limits.
    pub fn decode_options(mut self, options: CudaDecodeOptions) -> Self {
        self.options = options;
        self
    }

    /// Select one logical video stream.
    pub fn track(mut self, track: TrackId) -> Self {
        self.options.track = Some(track);
        self
    }

    /// Select exact timestamps in caller order; repeated source frames share leases.
    pub fn sample(mut self, times: impl IntoIterator<Item = Time>) -> Self {
        self.times = Some(times.into_iter().collect());
        self
    }

    /// Choose timestamp-selection semantics independently of hardware placement.
    pub fn selection(mut self, selection: TemporalSelection) -> Self {
        self.selection = selection;
        self
    }

    /// Permit duration inference explicitly when native durations are missing.
    pub fn duration_inference(mut self, inference: DurationInference) -> Self {
        self.durations = inference;
        self
    }
    /// Attach immutable source/version/track/policy-bound decoded timing evidence.
    /// No index construction or CPU video decode is performed implicitly.
    pub fn with_index(mut self, index: Arc<VideoIndex>) -> Self {
        self.index = Some(index);
        self
    }
    /// Open a reusable indexed GPU reader, retaining one decoder until dropped.
    ///
    /// Requires an explicit index. Sample times are supplied to each `read` call;
    /// selection, duration, cancellation and work limits come from this request.
    /// Setup reads are bounded separately from each batch's source-read budget.
    pub fn indexed_reader(&self) -> Result<CudaIndexedVideoReader> {
        self.indexed_limits.validate()?;
        if self.index.is_none() {
            return Err(Error::invalid(
                "a reusable indexed reader requires an attached index",
            ));
        }
        let mut request = self.clone();
        request.times = None;
        let mut reader = request.open_reader()?;
        reader.update_report();
        reader
            .decoder
            .as_ref()
            .ok_or_else(|| Error::invalid("indexed decoder closed"))?
            .set_read_limit(request.indexed_limits.max_source_bytes)?;
        Ok(CudaIndexedVideoReader {
            request,
            reader: Some(reader),
            state: IndexedDecodeState::default(),
        })
    }
    /// Bound real seeks, dependency decoding and source reads for indexed batches.
    pub fn indexed_limits(mut self, limits: IndexSeekLimits) -> Self {
        self.indexed_limits = limits;
        self
    }
    /// Share cooperative cancellation with sampled/indexed executions.
    pub fn with_cancellation(mut self, cancel: CancellationToken) -> Self {
        self.cancel = Some(cancel);
        self
    }
    /// Explicitly build a complete bounded decoded timing index on NVDEC. No
    /// decoded pixels are copied to the host; cold work is reported separately.
    pub fn build_index(&self, limits: IndexBuildLimits) -> Result<Arc<VideoIndex>> {
        self.build_index_with_cancellation(limits, self.cancel.clone().unwrap_or_default())
    }
    /// Load bounded timing metadata without opening a GPU or decoding pixels.
    /// Source, stream and duration interpretation must match this request.
    pub fn load_index(
        &self,
        path: impl AsRef<std::path::Path>,
        limits: IndexBuildLimits,
    ) -> Result<Arc<VideoIndex>> {
        VideoIndex::load(
            &self.asset.source,
            &self.asset.session.resources,
            self.options.track,
            self.options.duration_policy,
            path.as_ref(),
            limits,
        )
    }
    /// Build on NVDEC with cooperative cancellation between native operations.
    pub fn build_index_with_cancellation(
        &self,
        limits: IndexBuildLimits,
        cancel: CancellationToken,
    ) -> Result<Arc<VideoIndex>> {
        cancel.check()?;
        let (source, reservation) = crate::video_index::prepare_build(
            &self.asset.source,
            &self.asset.session.resources,
            limits,
        )?;
        let mut request = self.clone();
        request.index = None;
        request.cancel = Some(cancel.clone());
        request.options.probe.max_read_bytes = request
            .options
            .probe
            .max_read_bytes
            .min(limits.max_source_bytes);
        let mut reader = request.open_reader()?.with_cancellation(cancel.clone());
        reader
            .decoder
            .as_ref()
            .ok_or_else(|| Error::invalid("GPU index decoder closed"))?
            .set_read_limit(limits.max_source_bytes)?;
        let track = reader
            .decoder
            .as_ref()
            .ok_or_else(|| Error::invalid("GPU index decoder closed"))?
            .track();
        let index = VideoIndex::build(
            self.asset.session.resources.clone(),
            source,
            track,
            self.options.duration_policy,
            limits,
            reservation,
            true,
            &cancel,
            || {
                let next = reader.next_frame()?.map(|frame| IndexedTiming {
                    timing: frame.frame().timing,
                    keyframe: frame.keyframe(),
                    seek_timestamp: frame.seek_timestamp(),
                });
                Ok((next, reader.report().bytes_read))
            },
        )?;
        index.validate(
            &self.asset.source,
            self.options.track,
            self.options.duration_policy,
        )?;
        Ok(index)
    }

    /// Disclose assumptions for missing metadata; contradictory fields still fail.
    pub fn assume_missing_color(mut self, assumption: CudaColorAssumption) -> Self {
        self.color_assumption = Some(assumption);
        self
    }

    /// Open an independent required-NVDEC sequential cursor.
    pub fn reader(&self) -> Result<CudaReader> {
        if self.times.is_some() {
            return Err(Error::invalid(
                "timestamp batches use read(), rather than a sequential reader",
            ));
        }
        self.open_reader()
    }
    /// Open with a shared cooperative cancellation token, checking both before
    /// native setup and after potentially blocking source/decoder initialization.
    pub fn reader_with_cancellation(&self, cancellation: CancellationToken) -> Result<CudaReader> {
        cancellation.check()?;
        let reader = self.reader()?.with_cancellation(cancellation.clone());
        cancellation.check()?;
        Ok(reader)
    }

    /// Open an incremental required-NVDEC cursor.
    pub fn stream(&self) -> Result<CudaReader> {
        self.reader()
    }

    fn open_reader(&self) -> Result<CudaReader> {
        self.validate()?;
        let codec = self.asset.session.cuda.reserve_codecs(1)?;
        let context = match &self.target {
            Target::New { ordinal, limits } => {
                self.asset.session.cuda.context(*ordinal, *limits)?
            }
            Target::Existing(context) => self.asset.session.cuda.register(context)?,
        };
        let mut options = self.options;
        if let Some(index) = &self.index {
            index.validate(&self.asset.source, options.track, options.duration_policy)?;
            options.track = Some(index.track());
            options.probe.max_read_bytes = options
                .probe
                .max_read_bytes
                .min(self.indexed_limits.max_source_bytes);
        }
        let cancel = self.cancel.clone().unwrap_or_default();
        cancel.check()?;
        let token = cancel.clone();
        let mut decoder = CudaVideoReader::open_on_context_controlled(
            &self.asset.source,
            &context,
            options,
            self.index.is_some(),
            Arc::new(move || token.is_cancelled()),
        )?;
        if let Some(index) = &self.index {
            decoder.set_timeline_origin(index.timeline_origin());
        }
        cancel.check()?;
        Ok(CudaReader {
            decoder: Some(decoder),
            context,
            cancel,
            report: CudaReport {
                hardware_decode: true,
                duration_policy: self.options.duration_policy,
                ..CudaReport::default()
            },
            failed: false,
            session: self.asset.session.clone(),
            codec: Some(codec),
        })
    }

    fn validate(&self) -> Result<()> {
        if let Some(cancel) = &self.cancel {
            cancel.check()?;
        }
        if let Some(index) = &self.index {
            self.indexed_limits.validate()?;
            index.validate(
                &self.asset.source,
                self.options.track,
                self.options.duration_policy,
            )?;
        }
        if let Target::New { limits, .. } = self.target {
            limits.validate()?;
        }
        if self
            .times
            .as_ref()
            .is_some_and(|times| times.len() > 65_536)
        {
            return Err(Error::invalid(
                "a timestamp batch is limited to 65536 requests",
            ));
        }
        Ok(())
    }

    /// Materialize native GPU frames within explicit retained-surface limits.
    ///
    /// Selection decodes from the stream start through one frame beyond the
    /// latest request, establishing real presentation ordinals. Large retained
    /// results fail with resource exhaustion; they never wait on their own leases.
    pub fn read(&self) -> Result<CudaFrameBatch> {
        if self.index.is_some() && self.times.is_some() {
            return self.read_indexed();
        }
        self.validate()?;
        let batch = Arc::new(self.asset.session.resources.reserve_batch()?);
        let fingerprint = self.asset.source.fingerprint()?;
        let mut reader = self.open_reader()?;
        let Some(times) = &self.times else {
            let mut frames = Vec::new();
            while let Some(frame) = reader.next_frame()? {
                frames.push(frame);
            }
            if fingerprint.is_versioned() && self.asset.source.fingerprint()? != fingerprint {
                return Err(Error::incompatible(
                    "source changed during CUDA materialization",
                ));
            }
            return CudaFrameBatch::from_native_parts(
                frames,
                fingerprint,
                None,
                reader.report(),
                reader.context.clone(),
                self.color_assumption.clone(),
                batch,
            );
        };
        if times.is_empty() {
            return CudaFrameBatch::from_native_parts(
                Vec::new(),
                fingerprint,
                None,
                reader.report(),
                reader.context.clone(),
                self.color_assumption.clone(),
                batch,
            );
        }
        let maximum = *times
            .iter()
            .max()
            .ok_or_else(|| Error::invalid("empty timestamps"))?;
        let track = reader
            .decoder
            .as_ref()
            .ok_or_else(|| Error::invalid("decoder closed"))?
            .track();
        let mut entries = Vec::new();
        let mut guards = Vec::new();
        let mut candidates: Vec<Option<(FrameIndexEntry, CudaVideoFrame)>> =
            vec![None; times.len()];
        let mut previous: Option<(FrameIndexEntry, CudaVideoFrame)> = None;
        while let Some(frame) = reader.next_frame()? {
            let timing = frame.frame().timing;
            let entry = FrameIndexEntry {
                pts: timing.pts.ok_or_else(|| {
                    Error::unsupported("exact sampling requires presentation timestamps")
                })?,
                duration: timing.duration,
                ordinal: timing.ordinal.ok_or_else(|| {
                    Error::unsupported("exact source ordinals were not established")
                })?,
                keyframe: false,
            };
            guards.push(
                self.asset
                    .session
                    .resources
                    .host_budget()
                    .reserve(std::mem::size_of::<FrameIndexEntry>())?,
            );
            entries.push(entry);
            for (i, &time) in times.iter().enumerate() {
                let mut local = Vec::with_capacity(3);
                if let Some((candidate, _)) = &candidates[i] {
                    local.push(*candidate);
                }
                if let Some((last, _)) = &previous
                    && !local
                        .iter()
                        .any(|candidate| candidate.ordinal == last.ordinal)
                {
                    local.push(*last);
                }
                if !local
                    .iter()
                    .any(|candidate| candidate.ordinal == entry.ordinal)
                {
                    local.push(entry);
                }
                let index = FrameIndex::new(fingerprint.clone(), track.0, local)?;
                if let Ok(chosen) = index.select(time, self.selection, self.durations) {
                    if chosen.identity.ordinal == entry.ordinal {
                        candidates[i] = Some((entry, frame.clone()));
                    } else if let Some((last, last_frame)) = &previous
                        && chosen.identity.ordinal == last.ordinal
                    {
                        candidates[i] = Some((*last, last_frame.clone()));
                    }
                }
            }
            previous = Some((entry, frame));
            if entry.pts > maximum {
                break;
            }
        }
        let index = FrameIndex::new(fingerprint.clone(), track.0, entries)?;
        let selection = index.select_batch(times, self.selection, self.durations)?;
        if index.source().is_versioned() {
            index.validate_source(&self.asset.source)?;
        }
        let mut unique = BTreeMap::new();
        for (entry, frame) in candidates.into_iter().flatten() {
            unique.insert(entry.ordinal, frame);
        }
        if let Some((entry, frame)) = previous {
            unique.entry(entry.ordinal).or_insert(frame);
        }
        let mut frames = Vec::with_capacity(times.len());
        for selected in &selection.selections {
            frames.push(
                unique
                    .get(&selected.identity.ordinal)
                    .ok_or_else(|| {
                        Error::backend(
                            "sampling",
                            "selected GPU frame was not retained by the candidate scheduler",
                        )
                    })?
                    .clone(),
            );
        }
        CudaFrameBatch::from_native_parts(
            frames,
            fingerprint,
            Some(selection),
            reader.report(),
            reader.context.clone(),
            self.color_assumption.clone(),
            batch,
        )
    }
    fn read_indexed(&self) -> Result<CudaFrameBatch> {
        let times = self
            .times
            .as_deref()
            .ok_or_else(|| Error::invalid("indexed execution requires timestamps"))?;
        self.read_indexed_on(times, &mut None, &mut IndexedDecodeState::default())
    }
    fn read_indexed_on(
        &self,
        times: &[Time],
        retained: &mut Option<CudaReader>,
        state: &mut IndexedDecodeState,
    ) -> Result<CudaFrameBatch> {
        self.validate()?;
        self.indexed_limits.validate()?;
        let index = self
            .index
            .as_ref()
            .ok_or_else(|| Error::invalid("missing GPU video index"))?;
        index.validate(
            &self.asset.source,
            self.options.track,
            self.options.duration_policy,
        )?;
        let (output_bytes, working_bytes) =
            index.batch_metadata_bytes(times.len(), std::mem::size_of::<CudaVideoFrame>())?;
        let metadata = Arc::new(
            self.asset
                .session
                .resources
                .host_budget()
                .reserve(output_bytes)?,
        );
        let _working = self
            .asset
            .session
            .resources
            .host_budget()
            .reserve(working_bytes)?;
        let (selection, groups) =
            index.schedule(times, self.selection, self.durations, self.indexed_limits)?;
        let batch = Arc::new(self.asset.session.resources.reserve_batch()?);
        let (before_frames, before_bytes) = if let Some(reader) = retained.as_ref() {
            let decoder = reader
                .decoder
                .as_ref()
                .ok_or_else(|| Error::invalid("indexed decoder closed"))?;
            decoder.begin_read_batch(self.indexed_limits.max_source_bytes)?;
            (reader.report().decoded_frames, reader.report().bytes_read)
        } else {
            let mut reader = self.open_reader()?;
            reader.update_report();
            reader
                .decoder
                .as_ref()
                .ok_or_else(|| Error::invalid("indexed decoder closed"))?
                .set_read_limit(self.indexed_limits.max_source_bytes)?;
            *retained = Some(reader);
            (0, 0)
        };
        let reader = retained.as_mut().expect("indexed reader opened");
        let before_seeks = reader.report.seeks;
        let mut unique = BTreeMap::new();
        for group in groups {
            reader.cancel.check()?;
            let anchor = index.frame(group.anchor)?;
            let anchor_pts = anchor
                .timing
                .source_pts
                .ok_or_else(|| Error::incompatible("GPU index anchor lacks source PTS"))?;
            let remaining = self
                .indexed_limits
                .max_decoded_frames
                .saturating_sub(reader.report.decoded_frames - before_frames);
            let mut restart = None;
            let mut position = state.next;
            if state.should_seek(&group, remaining) {
                self.indexed_limits
                    .check_seek(reader.report.seeks - before_seeks)?;
                restart = Some(Instant::now());
                reader
                    .decoder
                    .as_mut()
                    .ok_or_else(|| Error::invalid("indexed decoder closed"))?
                    .seek_indexed_keyframe(anchor_pts, anchor.seek_timestamp)?;
                reader.report.seeks = reader
                    .report
                    .seeks
                    .checked_add(1)
                    .ok_or_else(|| Error::overflow("indexed seek count"))?;
                position = None;
                state.next = None;
            }
            loop {
                self.indexed_limits.check_decoded(
                    (reader.report.decoded_frames - before_frames).saturating_add(1),
                )?;
                let started = Instant::now();
                let frame = reader.next_frame()?.ok_or_else(|| {
                    Error::incompatible("NVDEC reached EOF before indexed GPU selection")
                })?;
                if restart.is_none() {
                    state.observe_frame(started.elapsed());
                }
                let timing = frame.frame().timing;
                let ordinal = match position {
                    Some(position) => position,
                    None if timing.source_pts == Some(anchor_pts) && frame.keyframe() => {
                        group.anchor
                    }
                    None => continue,
                };
                if let Some(started) = restart.take() {
                    state.observe_restart(started.elapsed());
                }
                let verified = index.check_actual(ordinal, timing, frame.keyframe())?;
                let frame = frame.with_indexed_position(
                    verified
                        .source_pts
                        .ok_or_else(|| Error::incompatible("indexed source time absent"))?,
                    verified
                        .pts
                        .ok_or_else(|| Error::incompatible("indexed asset time absent"))?,
                    ordinal as u64,
                )?;
                if group.wanted.binary_search(&(ordinal as u64)).is_ok() {
                    unique.insert(ordinal as u64, frame);
                }
                state.next = Some(
                    ordinal
                        .checked_add(1)
                        .ok_or_else(|| Error::overflow("indexed ordinal"))?,
                );
                position = state.next;
                if ordinal as u64 == group.last {
                    break;
                }
            }
        }
        index.validate(
            &self.asset.source,
            self.options.track,
            self.options.duration_policy,
        )?;
        let mut frames = Vec::new();
        frames
            .try_reserve_exact(times.len())
            .map_err(|_| Error::exhausted("indexed GPU output allocation failed"))?;
        for selected in &selection.selections {
            frames.push(
                unique
                    .get(&selected.identity.ordinal)
                    .ok_or_else(|| Error::incompatible("indexed GPU selection was not decoded"))?
                    .clone(),
            );
        }
        let mut report = reader.report();
        report.decoded_frames -= before_frames;
        report.bytes_read -= before_bytes;
        report.seeks -= before_seeks;
        report.index_cache_hits = 1;
        Ok(CudaFrameBatch::from_native_parts(
            frames,
            index.source().clone(),
            Some(selection),
            report,
            reader.context.clone(),
            self.color_assumption.clone(),
            batch,
        )?
        .retain_metadata(metadata))
    }

    /// Fuse explicit crop/resize, matrix conversion, normalization and packing.
    pub fn to_tensor(self, spec: PreprocessSpec) -> CudaTensorRequest {
        CudaTensorRequest {
            request: self,
            spec,
        }
    }
}

/// Stateful indexed GPU sampling with one retained decoder and verified position.
///
/// Batches retain independent output leases. Work counters and limits apply to
/// each call; setup is performed at construction. Any failed call closes this
/// reader, so stale evidence or partial native state cannot be reused.
pub struct CudaIndexedVideoReader {
    request: CudaVideoRequest,
    reader: Option<CudaReader>,
    state: IndexedDecodeState,
}
impl CudaIndexedVideoReader {
    /// Select this batch's timestamps in caller order, including repetitions.
    pub fn read(&mut self, times: &[Time]) -> Result<CudaFrameBatch> {
        if self.reader.is_none() {
            return Err(Error::invalid(
                "indexed reader is closed after an earlier failure",
            ));
        }
        let result = self
            .request
            .read_indexed_on(times, &mut self.reader, &mut self.state);
        if result.is_err() {
            self.reader.take();
        }
        result
    }
}

/// Independent native GPU cursor with cooperative cancellation.
pub struct CudaReader {
    decoder: Option<CudaVideoReader>,
    context: CudaContext,
    cancel: CancellationToken,
    report: CudaReport,
    failed: bool,
    session: crate::Session,
    codec: Option<std::sync::Arc<crate::cuda_session::CodecPermit>>,
}

impl CudaReader {
    /// Share cancellation with coordinated audio/video or application execution.
    pub fn with_cancellation(mut self, cancellation: CancellationToken) -> Self {
        if let Some(decoder) = &self.decoder {
            let token = cancellation.clone();
            decoder.set_interrupt(Arc::new(move || token.is_cancelled()));
        }
        self.cancel = cancellation;
        self
    }
    /// Probed source tracks and shared timeline, while the cursor is open.
    pub fn info(&self) -> Option<&AssetInfo> {
        self.decoder.as_ref().map(CudaVideoReader::info)
    }
    /// Actual native context retained independently of decoder lifetime.
    pub fn context(&self) -> &CudaContext {
        &self.context
    }
    /// Cooperative cancellation; the next operation releases the decoder promptly.
    pub fn cancellation(&self) -> CancellationToken {
        self.cancel.clone()
    }
    /// Seek a native random-access region; global ordinals become unknown.
    pub fn seek(&mut self, time: Time) -> Result<()> {
        if let Err(error) = self.cancel.check() {
            self.update_report();
            self.decoder.take();
            self.codec.take();
            self.failed = true;
            return Err(error);
        }
        let result = self
            .decoder
            .as_mut()
            .ok_or_else(|| Error::invalid("CUDA cursor is closed"))?
            .seek(time);
        if result.is_err() {
            self.update_report();
            self.decoder.take();
            self.codec.take();
            self.failed = true;
        }
        result
    }
    /// Read a retained native surface, without creating a host representation.
    pub fn next_frame(&mut self) -> Result<Option<CudaVideoFrame>> {
        if self.failed {
            return Err(Error::invalid("CUDA execution previously failed"));
        }
        if let Err(error) = self.cancel.check() {
            self.update_report();
            self.decoder.take();
            self.codec.take();
            self.failed = true;
            return Err(error);
        }
        if self.decoder.is_none() {
            return Ok(None);
        }
        let mut result = self
            .decoder
            .as_mut()
            .ok_or_else(|| Error::invalid("CUDA cursor is closed"))?
            .next_frame();
        self.update_report();
        if self.cancel.is_cancelled() {
            result = Err(Error::new(
                media_core::error::ErrorKind::Cancelled,
                "execution cancelled during CUDA decode",
            ));
        }
        result = result.and_then(|frame| {
            frame
                .map(|frame| {
                    let bytes = self
                        .session
                        .cuda
                        .reserve_native(frame.frame().image.storage().byte_len())?;
                    let metadata = self
                        .session
                        .resources
                        .host_budget()
                        .reserve(native_metadata_bytes(frame.frame().image.metadata())?)?;
                    let codec = self
                        .codec
                        .as_ref()
                        .ok_or_else(|| Error::invalid("CUDA reader has no active codec permit"))?
                        .clone();
                    Ok(frame.retain(bytes).retain(codec).retain(metadata))
                })
                .transpose()
        });
        if result.is_err() {
            self.failed = true;
            self.decoder.take();
            self.codec.take();
        } else if matches!(result, Ok(None)) {
            self.decoder.take();
            self.codec.take();
        }
        result
    }
    fn update_report(&mut self) {
        if let Some(decoder) = &self.decoder {
            self.report.bytes_read = decoder.bytes_read();
            self.report.decoded_frames = decoder.frames_decoded();
            self.report.output_frames = decoder.frames_decoded();
        }
        self.report.hardware_decode = true;
        self.report.peak_device_allocation_bytes = self.context.memory_peak();
    }
    /// Counters from exercised library operations; opaque driver allocations are unknown.
    pub fn report(&self) -> CudaReport {
        self.report.clone()
    }
}

impl Iterator for CudaReader {
    type Item = Result<CudaVideoFrame>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.failed {
            None
        } else {
            self.next_frame().transpose()
        }
    }
}

/// Observed GPU path costs, separated from driver implementation details.
#[derive(Debug, Clone, Default)]
pub struct CudaReport {
    /// Actual native indexed keyframe seeks during this execution.
    pub seeks: u64,
    /// Attached source-validated indexes used; build work is reported separately.
    pub index_cache_hits: u64,
    /// Explicit interpretation of backend-reported packet/codec duration hints.
    pub duration_policy: media_core::policy::FrameDurationPolicy,
    /// Encoded input bytes read, including probing.
    pub bytes_read: u64,
    /// Dependency and output frames actually decoded.
    pub decoded_frames: u64,
    /// Outputs including caller repetitions.
    pub output_frames: u64,
    /// True only after the NVDEC reader is opened successfully.
    pub hardware_decode: bool,
    /// Decoded host crossings; this implementation has no such operation.
    pub decoded_host_transfer_bytes: u64,
    /// Allocation high-water mark enforced by the CUDA allocator, excluding native surfaces.
    pub peak_device_allocation_bytes: usize,
    /// Opaque driver and decoder-reference allocation bytes are unmeasured.
    pub opaque_driver_bytes: Option<u64>,
    /// Explicit missing-color assumptions applied during preprocessing.
    pub color_assumptions: Vec<String>,
    /// Established RGB primaries/transfer after YUV matrix conversion.
    pub output_color: Option<ColorDescription>,
}

/// Retained, caller-ordered native surfaces plus exact selection provenance.
#[derive(Debug)]
pub struct CudaFrameBatch {
    /// Source identity/version observed before opening this execution's decoder.
    pub source: SourceFingerprint,
    /// Native surfaces; duplicates share ownership and readiness.
    pub frames: Vec<CudaVideoFrame>,
    /// Requested and actual selection timing when sampling was requested.
    pub selection: Option<SelectionBatch>,
    /// Work actually performed.
    pub report: CudaReport,
    context: CudaContext,
    assumption: Option<CudaColorAssumption>,
    batch: Arc<BatchReservation>,
    metadata: Option<Arc<HostReservation>>,
}

impl CudaFrameBatch {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_native_parts(
        frames: Vec<CudaVideoFrame>,
        source: SourceFingerprint,
        selection: Option<SelectionBatch>,
        mut report: CudaReport,
        context: CudaContext,
        assumption: Option<CudaColorAssumption>,
        batch: Arc<BatchReservation>,
    ) -> Result<Self> {
        if frames.iter().any(|frame| {
            frame.context().device() != context.device()
                || !frame.context().shares_resources_with(&context)
        }) {
            return Err(Error::incompatible(
                "native batch frames must share the declared actual context and canonical allocator",
            ));
        }
        report.output_frames = u64::try_from(frames.len())
            .map_err(|_| Error::overflow("native batch output count"))?;
        Ok(Self {
            frames: frames
                .into_iter()
                .map(|frame| frame.retain(batch.clone()))
                .collect(),
            source,
            selection,
            report,
            context,
            assumption,
            batch,
            metadata: None,
        })
    }
    fn retain_metadata(mut self, metadata: Arc<HostReservation>) -> Self {
        self.frames = self
            .frames
            .into_iter()
            .map(|frame| frame.retain(metadata.clone()))
            .collect();
        self.metadata = Some(metadata);
        self
    }
    /// Actual owning context, suitable for creating a compatible model stream.
    pub fn context(&self) -> &CudaContext {
        &self.context
    }

    /// Consume the native batch into one final contiguous device allocation.
    /// Native input leases retire after queued preprocessing finishes.
    pub fn into_tensor(mut self, spec: PreprocessSpec) -> Result<CudaTensorBatch> {
        let mut inputs = Vec::with_capacity(self.frames.len());
        let mut common_color = None;
        for frame in &self.frames {
            if frame.frame().image.metadata().orientation != Orientation::Normal {
                return Err(Error::unsupported(
                    "CUDA preprocessing requires orientation applied to pixels; metadata-only rotation is unsupported",
                ));
            }
            if frame.frame().image.metadata().icc_profile.is_some() {
                return Err(Error::unsupported(
                    "CUDA preprocessing does not apply ICC profiles",
                ));
            }
            let (input, color, assumptions) = interpret(frame, self.assumption.as_ref())?;
            if common_color
                .as_ref()
                .is_some_and(|previous| previous != &color)
            {
                return Err(Error::incompatible(
                    "one CUDA tensor batch requires consistent RGB primaries and transfer",
                ));
            }
            common_color = Some(color);
            for assumption in assumptions {
                if !self.report.color_assumptions.contains(&assumption) {
                    self.report.color_assumptions.push(assumption);
                }
            }
            spec.validate(input.layout())?;
            inputs.push(input);
        }
        let stream = self.context.stream()?;
        let mut tensor = preprocess_batch(&inputs, &stream, spec)?.retain(self.batch);
        if let Some(metadata) = &self.metadata {
            tensor = tensor.retain(metadata.clone());
        }
        self.report.peak_device_allocation_bytes = self.context.memory_peak();
        self.report.output_color = common_color;
        Ok(CudaTensorBatch {
            source: self.source,
            tensor,
            selection: self.selection,
            report: self.report,
            context: self.context,
            _metadata: self.metadata,
        })
    }
}

fn native_metadata_bytes(metadata: &media_core::media::ImageMetadata) -> Result<usize> {
    let color = &metadata.color;
    let mut bytes = std::mem::size_of::<media_core::media::ImageMetadata>();
    for field in [
        &color.primaries,
        &color.transfer,
        &color.matrix,
        &color.range,
        &color.chroma_siting,
    ] {
        bytes = bytes
            .checked_add(field.as_ref().map_or(0, String::capacity))
            .ok_or_else(|| Error::overflow("CUDA image color metadata charge"))?;
    }
    for payload in [&metadata.icc_profile, &metadata.exif] {
        bytes = bytes
            .checked_add(payload.as_ref().map_or(0, Vec::capacity))
            .ok_or_else(|| Error::overflow("CUDA image metadata payload charge"))?;
    }
    Ok(bytes)
}
fn interpret(
    frame: &CudaVideoFrame,
    assumption: Option<&CudaColorAssumption>,
) -> Result<(CudaYuvImage, ColorDescription, Vec<String>)> {
    let mut color = frame.frame().image.metadata().color.clone();
    let mut applied = Vec::new();
    if let Some(assumption) = assumption {
        let matrix = match assumption.matrix {
            YuvMatrix::Bt601 => "smpte170m",
            YuvMatrix::Bt709 => "bt709",
            YuvMatrix::Bt2020 => "bt2020nc",
        };
        let range = match assumption.range {
            YuvRange::Limited => "tv",
            YuvRange::Full => "pc",
        };
        let chroma = match assumption.chroma {
            ChromaSiting::Center => "center",
            ChromaSiting::TopLeft => "topleft",
            ChromaSiting::Left => "left",
        };
        fill(&mut color.matrix, matrix, "matrix", &mut applied)?;
        fill(&mut color.range, range, "range", &mut applied)?;
        fill(
            &mut color.chroma_siting,
            chroma,
            "chroma siting",
            &mut applied,
        )?;
        fill(
            &mut color.primaries,
            &assumption.primaries,
            "primaries",
            &mut applied,
        )?;
        fill(
            &mut color.transfer,
            &assumption.transfer,
            "transfer",
            &mut applied,
        )?;
    }
    let primaries = color.primaries.clone().ok_or_else(|| {
        Error::incompatible(
            "GPU RGB output requires established primaries or an explicit missing-color assumption",
        )
    })?;
    let transfer = color.transfer.clone().ok_or_else(|| Error::incompatible("GPU RGB output requires an established transfer function or an explicit missing-color assumption"))?;
    let matrix = match color.matrix.as_deref() {
        Some("bt709") => YuvMatrix::Bt709,
        Some("smpte170m" | "bt470bg") => YuvMatrix::Bt601,
        Some("bt2020nc") => YuvMatrix::Bt2020,
        _ => return Err(Error::unsupported("unsupported or missing CUDA YUV matrix")),
    };
    let range = match color.range.as_deref() {
        Some("tv") => YuvRange::Limited,
        Some("pc") => YuvRange::Full,
        _ => return Err(Error::incompatible("missing or unsupported CUDA YUV range")),
    };
    let chroma = match color.chroma_siting.as_deref() {
        Some("left") => ChromaSiting::Left,
        Some("center") => ChromaSiting::Center,
        Some("topleft") => ChromaSiting::TopLeft,
        _ => {
            return Err(Error::incompatible(
                "missing or unsupported CUDA chroma siting",
            ));
        }
    };
    let rgb_color = ColorDescription {
        primaries: Some(primaries),
        transfer: Some(transfer),
        matrix: Some("rgb".into()),
        range: Some("pc".into()),
        chroma_siting: None,
    };
    Ok((frame.as_yuv(matrix, range, chroma)?, rgb_color, applied))
}

fn fill(
    field: &mut Option<String>,
    expected: &str,
    name: &str,
    applied: &mut Vec<String>,
) -> Result<()> {
    if expected.is_empty() {
        return Err(Error::invalid("color assumption names must be nonempty"));
    }
    match field {
        Some(actual) if actual != expected => Err(Error::incompatible(format!(
            "established {name} {actual} contradicts requested interpretation {expected}"
        ))),
        Some(_) => Ok(()),
        None => {
            *field = Some(expected.into());
            applied.push(format!("assumed missing {name}={expected}"));
            Ok(())
        }
    }
}

/// Lazy fused GPU preprocessing request.
#[derive(Debug, Clone)]
pub struct CudaTensorRequest {
    request: CudaVideoRequest,
    spec: PreprocessSpec,
}

impl CudaTensorRequest {
    /// Attach decoded timing evidence to this fused model-input workflow.
    pub fn with_index(mut self, index: Arc<VideoIndex>) -> Self {
        self.request = self.request.with_index(index);
        self
    }
    /// Bound indexed seek/decode/read work without changing tensor semantics.
    pub fn indexed_limits(mut self, limits: IndexSeekLimits) -> Self {
        self.request = self.request.indexed_limits(limits);
        self
    }
    /// Explicit cold NVDEC index build; preprocessing is not materialized.
    pub fn build_index(&self, limits: IndexBuildLimits) -> Result<Arc<VideoIndex>> {
        self.request.build_index(limits)
    }
    /// Load explicit bounded timing evidence without opening CUDA or decoding.
    pub fn load_index(
        &self,
        path: impl AsRef<std::path::Path>,
        limits: IndexBuildLimits,
    ) -> Result<Arc<VideoIndex>> {
        self.request.load_index(path, limits)
    }
    /// Cooperative cancellation for decode/preprocessing executions.
    pub fn with_cancellation(mut self, cancel: CancellationToken) -> Self {
        self.request = self.request.with_cancellation(cancel);
        self
    }
    /// Validate request structure without decoding or creating a GPU context.
    pub fn compile(&self) -> Result<CudaTensorPlan> {
        self.request.validate()?;
        self.spec.output_elements()?;
        self.spec.normalization.validate()?;
        let (ordinal, context, context_limits) = match &self.request.target {
            Target::New { ordinal, limits } => {
                let context = self.request.asset.session.cuda.peek(*ordinal)?;
                if context
                    .as_ref()
                    .is_some_and(|context| context.limits() != *limits)
                {
                    return Err(Error::incompatible(
                        "compiled CUDA request limits differ from the Session's canonical context",
                    )
                    .with_constraint("cuda_context_limits"));
                }
                (*ordinal, context.map(|context| context.device()), *limits)
            }
            Target::Existing(context) => {
                let media_core::device::Device::Cuda { ordinal, .. } = context.device() else {
                    return Err(Error::invalid("CUDA plan requires a native CUDA context"));
                };
                if self
                    .request
                    .asset
                    .session
                    .cuda
                    .peek(ordinal)?
                    .is_some_and(|canonical| {
                        canonical.device() != context.device()
                            || !canonical.shares_resources_with(context)
                    })
                {
                    return Err(Error::incompatible("compiled CUDA context differs from the Session's canonical context/allocator")
                        .with_constraint("cuda_context_identity"));
                }
                (ordinal, Some(context.device()), context.limits())
            }
        };
        let output_shape = self
            .request
            .times
            .as_ref()
            .map(|times| self.spec.batch_shape(times.len()))
            .transpose()?;
        let description = CudaPlanDescription {
            ordinal, context, context_limits,
            hardware_required: true, decoded_transfers: media_core::policy::TransferPolicy::ForbidHost,
            preprocessing: self.spec, output_shape,
            requested_times: self.request.times.clone(), selection: self.request.selection,
            duration_inference: self.request.durations, decoder: self.request.options,
            session_resources: self.request.asset.session.cuda.limits(),
            missing_color: self.request.color_assumption.clone(),
            index_frames: self.request.index.as_ref().map(|index|index.len()),
            indexed_limits: self.request.index.as_ref().map(|_|self.request.indexed_limits),
            unresolved: vec![
                "Native codec/profile, allocated geometry, color metadata and frame durations require decoding".into(),
                if self.request.index.is_some() { "Source-validated index groups verified ordinals by unique decoded keyframes; actual seek alignment and dependency count remain runtime checks".into() }
                else { "Timestamp batches decode from the start to establish presentation ordinals; dependency count is runtime work".into() },
                "Opaque decoder-reference pools and driver heaps are not measured or included in known-native reservations".into(),
            ],
        };
        Ok(CudaTensorPlan {
            request: self.clone(),
            source: self.request.asset.source.fingerprint()?,
            description,
        })
    }
    /// Execute an independent decode/preprocessing cursor.
    pub fn read(&self) -> Result<CudaTensorBatch> {
        self.compile()?.execute()
    }
}

/// Structured known choices and explicitly unresolved native GPU requirements.
#[derive(Debug, Clone)]
pub struct CudaPlanDescription {
    /// Attached complete decoded timing records, or None for prefix execution.
    pub index_frames: Option<usize>,
    /// Explicit actual-work bounds when an index is attached.
    pub indexed_limits: Option<IndexSeekLimits>,
    /// Requested CUDA ordinal, without inventing a lazy native context ID.
    pub ordinal: u32,
    /// Actual known native context, or None until lazy native bootstrap.
    pub context: Option<media_core::device::Device>,
    /// Owned allocation and retained-stream limits for the canonical wrapper.
    pub context_limits: CudaLimits,
    /// Dedicated hardware decoding is a hard requirement.
    pub hardware_required: bool,
    /// No decoded host crossings are permitted.
    pub decoded_transfers: media_core::policy::TransferPolicy,
    /// Exact crop/resize domain, interpolation, normalization and packing request.
    pub preprocessing: PreprocessSpec,
    /// Known final contiguous tensor shape, absent for full sequential materialization.
    pub output_shape: Option<[usize; 4]>,
    /// Requested timestamps in caller order, absent for sequential materialization.
    pub requested_times: Option<Vec<Time>>,
    /// Temporal selection, independent of decoder navigation.
    pub selection: TemporalSelection,
    /// Explicit permission to infer unknown presentation durations.
    pub duration_inference: DurationInference,
    /// Per-reader probe and retained-native surface bounds.
    pub decoder: CudaDecodeOptions,
    /// Shared native codec/surface reservation limits, separate from owned tensors.
    pub session_resources: crate::cuda_session::CudaSessionLimits,
    /// Explicit missing-color interpretation, when supplied.
    pub missing_color: Option<CudaColorAssumption>,
    /// Decisions/costs not established before actual execution.
    pub unresolved: Vec<String>,
}

/// Reusable source-validated required-hardware plan with independent executions.
#[derive(Debug, Clone)]
pub struct CudaTensorPlan {
    request: CudaTensorRequest,
    source: SourceFingerprint,
    description: CudaPlanDescription,
}

impl CudaTensorPlan {
    /// Programmatic known policies, resource limits, and unresolved requirements.
    pub fn description(&self) -> &CudaPlanDescription {
        &self.description
    }
    /// Resolve native format, timing and metadata at execution; no hidden CPU fallback.
    pub fn explain(&self) -> String {
        format!(
            "NVDEC required -> retained CUDA NV12/P010 surfaces -> fused {:?} crop/resize, source-color RGB, normalization, {:?} contiguous f32 batch; decoded host transfers forbidden. Chroma interpolation {:?}; spatial interpolation {:?}. Native duration interpretation {:?}; interval inference {:?}. Exact timestamp candidates are deduplicated and restored to caller order. Runtime: codec/profile support, dimensions, color metadata, frame durations, native context and decoder reservations.",
            self.request.spec.resize_domain,
            self.request.spec.layout,
            self.request.spec.chroma_interpolation,
            self.request.spec.interpolation,
            self.request.request.options.duration_policy,
            self.request.request.durations
        )
    }
    /// Validate source identity, then execute a fresh bounded reader.
    pub fn execute(&self) -> Result<CudaTensorBatch> {
        if self.source.is_versioned()
            && self.request.request.asset.source.fingerprint()? != self.source
        {
            return Err(Error::incompatible(
                "source changed since CUDA plan compilation",
            ));
        }
        let result = self
            .request
            .request
            .read()?
            .into_tensor(self.request.spec)?;
        if self.source.is_versioned()
            && self.request.request.asset.source.fingerprint()? != self.source
        {
            return Err(Error::incompatible(
                "source changed during CUDA plan execution",
            ));
        }
        Ok(result)
    }
}

/// Final model input allocation with context, exact selections and execution costs.
#[derive(Debug)]
pub struct CudaTensorBatch {
    /// Original source identity/version, including unsampled materializations.
    pub source: SourceFingerprint,
    /// Contiguous f32 CUDA tensor; consuming export checks writable exclusivity.
    pub tensor: CudaTensor,
    /// Caller-order selection timing and source identities.
    pub selection: Option<SelectionBatch>,
    /// Hardware decode and allocation evidence.
    pub report: CudaReport,
    context: CudaContext,
    _metadata: Option<Arc<HostReservation>>,
}
impl CudaTensorBatch {
    /// Owning context for a compatible external consumer stream.
    pub fn context(&self) -> &CudaContext {
        &self.context
    }
}

/// Required native hardware transcode with selected compressed audio passthrough.
/// Optional native component resize requires `cuda-filters` and `scale_cuda`.
#[derive(Debug, Clone)]
pub struct CudaConvertRequest {
    asset: Asset,
    ordinal: u32,
    limits: CudaLimits,
    options: media_backend_ffmpeg::cuda_encode::CudaTranscodeOptions,
    fit: Option<media_pipeline::operation::Fit>,
    context: Option<CudaContext>,
}
impl CudaConvertRequest {
    pub(crate) fn from_software(
        asset: Asset,
        ordinal: u32,
        limits: CudaLimits,
        cpu: media_backend_ffmpeg::encode::TranscodeOptions,
        fit: Option<media_pipeline::operation::Fit>,
    ) -> Self {
        use media_backend_ffmpeg::{
            cuda_encode::{CudaTranscodeOptions, NvencCodec},
            encode::CpuVideoCodec,
        };
        let options = CudaTranscodeOptions {
            video_track: cpu.video_track,
            audio_tracks: cpu.audio_tracks,
            codec: match cpu.codec {
                CpuVideoCodec::H264 => NvencCodec::H264,
                CpuVideoCodec::Hevc => NvencCodec::Hevc,
            },
            width: cpu.width,
            height: cpu.height,
            bit_rate: cpu.bit_rate,
            max_b_frames: cpu.max_b_frames,
            gop: cpu.gop,
            probe: cpu.probe,
            ..Default::default()
        };
        Self {
            asset,
            ordinal,
            limits,
            options,
            fit,
            context: None,
        }
    }
    /// Replace NVENC codec, preset, native surface and compressed preroll limits.
    pub fn options(
        mut self,
        options: media_backend_ffmpeg::cuda_encode::CudaTranscodeOptions,
    ) -> Self {
        self.options = options;
        self
    }
    /// Share this Session's registered actual native context and allocator.
    pub fn on_context(mut self, context: &CudaContext) -> Self {
        self.context = Some(context.clone());
        self
    }
    /// Set the native encoder target bitrate, in bits per second.
    pub fn bitrate(mut self, bits_per_second: usize) -> Self {
        self.options.bit_rate = bits_per_second;
        self
    }
    /// Resize stored YUV components on the GPU, preserving their color interpretation.
    pub fn resize_components(mut self, fit: media_pipeline::operation::Fit) -> Self {
        self.fit = Some(fit);
        self
    }
    /// Drain real hardware decoder/encoder output, copy audio, and finalize a new file.
    pub fn write(
        &self,
        path: impl AsRef<std::path::Path>,
    ) -> Result<media_backend_ffmpeg::cuda_encode::CudaTranscodeReport> {
        self.limits.validate()?;
        let _codecs = self.asset.session.cuda.reserve_codecs(2)?;
        let context = match &self.context {
            Some(context) => self.asset.session.cuda.register(context)?,
            None => self.asset.session.cuda.context(self.ordinal, self.limits)?,
        };
        let mut options = self.options.clone();
        // Reserve the pending allowance plus two maximum working surfaces
        // (decoded input and scale output). Native private/reference pools are
        // explicitly outside these known application-surface reservations.
        let native_limit = self.asset.session.cuda.native_limit();
        options.max_pending_bytes = options
            .max_pending_bytes
            .min(context.memory_limit())
            .min(native_limit / 2);
        options.max_frame_bytes = options
            .max_frame_bytes
            .min(context.memory_limit())
            .min((native_limit - options.max_pending_bytes) / 2);
        if options.max_pending_bytes == 0 || options.max_frame_bytes == 0 {
            return Err(Error::exhausted(
                "Session known-native CUDA budget cannot admit transcode pending/working surfaces",
            )
            .with_constraint("cuda_native_bytes"));
        }
        let allowance = options
            .max_frame_bytes
            .checked_mul(2)
            .and_then(|working| working.checked_add(options.max_pending_bytes))
            .ok_or_else(|| Error::overflow("CUDA transcode native allowance size"))?;
        let _native = self.asset.session.cuda.reserve_native(allowance)?;
        let mut extra_probe_bytes = 0_u64;
        if let Some(fit) = self.fit {
            let info =
                media_backend_ffmpeg::format::probe_headers(&self.asset.source, options.probe)?;
            extra_probe_bytes = info.probe_bytes_read;
            let track = info
                .tracks
                .iter()
                .find(|track| {
                    track.kind == media_backend_ffmpeg::format::TrackKind::Video
                        && !track.cover_art
                        && options.video_track.is_none_or(|id| id == track.id)
                })
                .ok_or_else(|| Error::unsupported("selected video stream is unavailable"))?;
            let width = track.width.ok_or_else(|| {
                Error::unsupported("header-only probe did not establish video width")
            })?;
            let height = track.height.ok_or_else(|| {
                Error::unsupported("header-only probe did not establish video height")
            })?;
            let (width, height) = crate::video_request::inside_dimensions(width, height, fit)?;
            options.width = Some(width);
            options.height = Some(height);
        }
        let available = self
            .asset
            .session
            .resources
            .host_budget()
            .limit()
            .saturating_sub(self.asset.session.live_host_bytes());
        options.max_preroll_bytes = options
            .max_preroll_bytes
            .min(available)
            .min(self.asset.session.limits().encoded_bytes);
        if options.max_preroll_bytes == 0 {
            return Err(Error::exhausted(
                "no host budget remains for compressed transcode preroll",
            ));
        }
        let _preroll = self
            .asset
            .session
            .resources
            .host_budget()
            .reserve(options.max_preroll_bytes)?;
        let result = media_backend_ffmpeg::cuda_encode::transcode_on_context(
            &self.asset.source,
            path.as_ref(),
            &context,
            options,
        );
        let mut report = match result {
            Ok(report) => report,
            Err(error) => {
                if error.context().constraint.as_deref() == Some("cuda_retirement_unknown") {
                    context.record_retirement_failure(error.to_string());
                    // The backend deliberately retained native frames/graph
                    // after an unrecoverable driver failure. Preserve quota too.
                    std::mem::forget((_codecs, _native));
                }
                return Err(error);
            }
        };
        report.source_bytes_read = report
            .source_bytes_read
            .checked_add(extra_probe_bytes)
            .ok_or_else(|| Error::overflow("CUDA transcode probe byte counter"))?;
        Ok(report)
    }
}
