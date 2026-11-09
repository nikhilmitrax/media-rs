//! Incremental CPU video and exact caller-ordered timestamp batches.
use crate::request::Asset;
use crate::video_index::{
    IndexBuildLimits, IndexSeekLimits, IndexedDecodeState, IndexedTiming, VideoIndex,
};
use media_backend_ffmpeg::{
    codec::{VideoDecodeOptions, VideoReader as Decoder},
    format::{self, AssetInfo, ProbeOptions, RemuxReport},
};
use media_core::{
    error::{Error, Result},
    media::{TrackId, VideoFrame},
    time::Time,
};
use media_pipeline::session::{BatchReservation, CpuReservation, HostReservation};
use media_pipeline::{
    execution::CancellationToken,
    sampling::{DurationInference, FrameIndex, FrameIndexEntry, SelectionBatch, TemporalSelection},
};
use std::{collections::BTreeMap, path::Path, sync::Arc, time::Instant};

/// Lazy CPU video selection with explicit temporal semantics.
#[derive(Debug, Clone)]
pub struct VideoRequest {
    asset: Asset,
    options: VideoDecodeOptions,
    times: Option<Vec<Time>>,
    selection: TemporalSelection,
    durations: DurationInference,
    index: Option<Arc<VideoIndex>>,
    indexed_limits: IndexSeekLimits,
    cancel: Option<CancellationToken>,
}
impl VideoRequest {
    pub(crate) fn new(asset: Asset) -> Self {
        Self {
            asset,
            options: VideoDecodeOptions::default(),
            times: None,
            selection: TemporalSelection::DisplayedAt,
            durations: DurationInference::RequireExplicit,
            index: None,
            indexed_limits: IndexSeekLimits::default(),
            cancel: None,
        }
    }
    /// Require NVDEC and CUDA-resident output, with no decoded host crossings.
    /// CPU pixel-output and color-conversion settings do not apply to this path.
    #[cfg(feature = "cuda")]
    pub fn cuda(
        self,
        ordinal: u32,
        limits: media_backend_cuda::CudaLimits,
    ) -> crate::cuda_request::CudaVideoRequest {
        let request = crate::cuda_request::CudaVideoRequest::new(self.asset, ordinal, limits)
            .inherit(
                self.options.track,
                self.options.probe,
                self.options.duration_policy,
                self.times,
                self.selection,
                self.durations,
            )
            .indexed_limits(self.indexed_limits);
        let request = match self.index {
            Some(index) => request.with_index(index),
            None => request,
        };
        match self.cancel {
            Some(cancel) => request.with_cancellation(cancel),
            None => request,
        }
    }
    /// Select an exact logical video stream.
    pub fn track(mut self, track: TrackId) -> Self {
        self.options.track = Some(track);
        self
    }
    /// Replace decoder output/color/probing controls.
    pub fn decode_options(mut self, options: VideoDecodeOptions) -> Self {
        self.options = options;
        self
    }
    /// Request timestamp selections in caller order, including repetitions.
    pub fn sample(mut self, times: impl IntoIterator<Item = Time>) -> Self {
        self.times = Some(times.into_iter().collect());
        self
    }
    /// Choose displayed-at, nearest, or before/after semantics explicitly.
    pub fn selection(mut self, selection: TemporalSelection) -> Self {
        self.selection = selection;
        self
    }
    /// Explicitly permit inferring missing frame durations from the next PTS.
    pub fn duration_inference(mut self, policy: DurationInference) -> Self {
        self.durations = policy;
        self
    }
    /// Attach explicitly constructed, source/version-validated timing evidence.
    /// This never builds an index or scans missing regions implicitly.
    pub fn with_index(mut self, index: Arc<VideoIndex>) -> Self {
        self.index = Some(index);
        self
    }
    /// Open a reusable indexed CPU reader, retaining one decoder until dropped.
    ///
    /// Requires an explicit index. Sample times are supplied to each `read` call;
    /// selection, duration, cancellation and work limits come from this request.
    /// Setup reads are bounded separately from each batch's source-read budget.
    pub fn indexed_reader(&self) -> Result<IndexedVideoReader> {
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
        Ok(IndexedVideoReader {
            request,
            reader: Some(reader),
            state: IndexedDecodeState::default(),
        })
    }
    /// Bound actual seek/decode/source-read work for indexed executions.
    pub fn indexed_limits(mut self, limits: IndexSeekLimits) -> Self {
        self.indexed_limits = limits;
        self
    }
    /// Share cooperative cancellation with sampled/indexed executions.
    /// Without this override each execution receives an independent token.
    pub fn with_cancellation(mut self, cancel: CancellationToken) -> Self {
        self.cancel = Some(cancel);
        self
    }
    /// Explicitly decode a complete bounded timing/keyframe index without RGB
    /// pixel conversion. Its cold scan costs are separate from later reads.
    pub fn build_index(&self, limits: IndexBuildLimits) -> Result<Arc<VideoIndex>> {
        self.build_index_with_cancellation(limits, self.cancel.clone().unwrap_or_default())
    }
    /// Load an explicitly persisted index under frame/metadata bounds, validating
    /// source identity/version and this request's stream/duration interpretation.
    pub fn load_index(
        &self,
        path: impl AsRef<Path>,
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
    /// Build a complete index with cooperative cancellation between native reads.
    pub fn build_index_with_cancellation(
        &self,
        limits: IndexBuildLimits,
        cancel: CancellationToken,
    ) -> Result<Arc<VideoIndex>> {
        cancel.check()?;
        let cpu = self.reserve_cpu()?;
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
        let mut reader = request.open_reader_admitted(cpu)?;
        reader.cancel = cancel.clone();
        reader
            .decoder
            .as_ref()
            .ok_or_else(|| Error::invalid("index decoder closed"))?
            .set_read_limit(limits.max_source_bytes)?;
        let track = reader.track;
        let index = VideoIndex::build(
            self.asset.session.resources.clone(),
            source,
            track,
            self.options.duration_policy,
            limits,
            reservation,
            false,
            &cancel,
            || {
                let next = reader.next_index_timing()?;
                Ok((next, reader.report().bytes_read))
            },
        )?;
        let _metadata = self.asset.session.resources.reserve_cpu(0, 0)?;
        index.validate(
            &self.asset.source,
            self.options.track,
            self.options.duration_policy,
        )?;
        Ok(index)
    }
    /// Probe tracks within the source-read budget.
    pub fn probe(&self) -> Result<AssetInfo> {
        self.validate_cpu_threads()?;
        let _cpu = self.asset.session.resources.reserve_cpu(0, 0)?;
        format::probe(&self.asset.source, self.options.probe)
    }
    /// Compile a source-validated reusable description without keeping a decoder.
    pub fn compile(&self) -> Result<VideoPlan> {
        self.validate_cpu_threads()?;
        let _cpu = self.asset.session.resources.reserve_cpu(0, 0)?;
        let source = self.asset.source.fingerprint()?;
        if let Some(index) = &self.index {
            index.validate(
                &self.asset.source,
                self.options.track,
                self.options.duration_policy,
            )?;
        } else {
            format::probe(&self.asset.source, self.options.probe)?;
        }
        if source.is_versioned() && self.asset.source.fingerprint()? != source {
            return Err(Error::incompatible(
                "source changed during video plan compilation",
            ));
        }
        let mut description = media_pipeline::plan::PlanDescription::default();
        description
            .steps
            .push(media_pipeline::diagnostics::PlanStep {
                operation: format!(
                    "incremental video decode; native duration interpretation {:?}, interval inference {:?}",
                    self.options.duration_policy, self.durations
                ),
                backend: "ffmpeg software".into(),
                output: media_core::device::Device::Cpu,
                format: if self.options.output == media_backend_ffmpeg::codec::VideoOutput::Rgb8 {
                    Some(media_core::media::PixelFormat::Rgb8)
                } else {
                    None
                },
                materializes: true,
            });
        if let Some(times) = &self.times {
            description
                .steps
                .push(media_pipeline::diagnostics::PlanStep {
                    operation: format!(
                        "exact {:?} timestamp batch, {} caller entries",
                        self.selection,
                        times.len()
                    ),
                    backend: "presentation-order candidate scheduler".into(),
                    output: media_core::device::Device::Cpu,
                    format: None,
                    materializes: false,
                });
            if self.index.is_some() {
                description.steps.push(media_pipeline::diagnostics::PlanStep { operation:"source-validated decoded index; group selected ordinals by verified keyframe and decode GOP-local regions".into(),backend:"FFmpeg exact stream-time seek".into(),output:media_core::device::Device::Cpu,format:None,materializes:false });
            } else {
                description.unresolved.push("prefix decode establishes ordinals; attach an explicitly built VideoIndex for GOP-local execution".into());
            }
        }
        description.unresolved.push("runtime codec format, frame intervals and decoder-private reservations remain checked during execution".into());
        Ok(VideoPlan {
            request: self.clone(),
            source,
            description,
        })
    }
    /// Open a fresh decoder/cursor for direct incremental processing.
    pub fn reader(&self) -> Result<VideoReader> {
        if self.times.is_some() {
            return Err(Error::invalid(
                "timestamp batches use read(); direct readers expose sequential frames",
            ));
        }
        self.open_reader()
    }
    fn open_reader(&self) -> Result<VideoReader> {
        let cpu = self.reserve_cpu()?;
        self.open_reader_admitted(cpu)
    }
    fn validate_cpu_threads(&self) -> Result<()> {
        if let Some(cancel) = &self.cancel {
            cancel.check()?;
        }
        if self.options.threads == 0 || self.options.threads > 256 {
            return Err(Error::invalid(
                "video requires 1..=256 explicitly requested codec workers",
            ));
        }
        self.asset
            .session
            .resources
            .validate_cpu_request(1, self.options.threads)
    }
    fn reserve_cpu(&self) -> Result<CpuReservation> {
        self.validate_cpu_threads()?;
        self.asset
            .session
            .resources
            .reserve_cpu(1, self.options.threads)
    }
    fn open_reader_admitted(&self, cpu: CpuReservation) -> Result<VideoReader> {
        let cancel = self.cancel.clone().unwrap_or_default();
        cancel.check()?;
        let mut options = self.options;
        if let Some(index) = &self.index {
            index.validate(&self.asset.source, options.track, options.duration_policy)?;
            options.track = Some(index.track());
            options.probe.max_read_bytes = options
                .probe
                .max_read_bytes
                .min(self.indexed_limits.max_source_bytes);
        }
        options.max_frame_bytes = options
            .max_frame_bytes
            .min(self.asset.session.limits().host_memory as u64);
        let token = cancel.clone();
        let mut decoder = Decoder::open_controlled(
            &self.asset.source,
            options,
            self.index.is_some(),
            Arc::new(move || token.is_cancelled()),
        )?;
        if let Some(index) = &self.index {
            decoder.set_timeline_origin(index.timeline_origin());
        }
        cancel.check()?;
        let info = decoder.info().clone();
        let track = decoder.track();
        let report = VideoReport {
            bytes_read: decoder.bytes_read(),
            decoded_frames: 0,
            output_frames: 0,
            hardware_decode: false,
            color_assumptions: Vec::new(),
            duration_policy: options.duration_policy,
            seeks: 0,
            index_cache_hits: 0,
        };
        Ok(VideoReader {
            decoder: Some(decoder),
            cpu: Some(cpu),
            info,
            track,
            report,
            asset: self.asset.clone(),
            cancel,
            failed: false,
        })
    }
    /// Return an incremental sequential reader.
    pub fn stream(&self) -> Result<VideoReader> {
        self.reader()
    }
    /// Materialize a bounded finite result or exact timestamp batch.
    ///
    /// Timestamp selection decodes from the stream beginning only through the
    /// largest requested PTS plus one future frame. This establishes real
    /// ordinals and does not implicitly build or cache a full-file seek index.
    /// Dependency frames still count as decoded work.
    pub fn read(&self) -> Result<FrameBatch> {
        if self.index.is_some() && self.times.is_some() {
            return self.read_indexed();
        }
        let cpu = self.reserve_cpu()?;
        let batch = Arc::new(self.asset.session.resources.reserve_batch()?);
        let fingerprint = self.asset.source.fingerprint()?;
        let mut reader = self.open_reader_admitted(cpu)?;
        let Some(times) = &self.times else {
            let mut frames = Vec::new();
            while let Some(frame) = reader.next_frame()? {
                frames.push(frame);
            }
            // EOF retired the decoder scope; range-version checks may call a
            // source provider, so use a temporary metadata scope for that check.
            let _metadata = self.asset.session.resources.reserve_cpu(0, 0)?;
            if fingerprint.is_versioned() && self.asset.source.fingerprint()? != fingerprint {
                return Err(Error::incompatible(
                    "source changed during video materialization",
                ));
            }
            return Ok(FrameBatch::new(
                frames,
                None,
                reader.report(),
                self.asset.session.clone(),
                batch,
                fingerprint,
            ));
        };
        if times.is_empty() {
            return Ok(FrameBatch::new(
                Vec::new(),
                None,
                reader.report(),
                self.asset.session.clone(),
                batch,
                fingerprint,
            ));
        }
        if times.len() > 65536 {
            return Err(Error::invalid(
                "a timestamp batch is limited to 65536 requests",
            ));
        }
        let maximum = *times
            .iter()
            .max()
            .ok_or_else(|| Error::invalid("empty timestamp batch"))?;
        let mut entries = Vec::new();
        let mut entry_guards = Vec::new();
        let mut candidates: Vec<Option<(FrameIndexEntry, VideoFrame)>> = vec![None; times.len()];
        let mut previous: Option<(FrameIndexEntry, VideoFrame)> = None;
        while let Some(frame) = reader.next_frame()? {
            let pts = frame.timing.pts.ok_or_else(|| {
                Error::unsupported("exact sampling requires established presentation timestamps")
            })?;
            let ordinal = frame.timing.ordinal.ok_or_else(|| {
                Error::unsupported("source presentation ordinals were not established")
            })?;
            let entry = FrameIndexEntry {
                pts,
                duration: frame.timing.duration,
                ordinal,
                keyframe: false,
            };
            entry_guards.push(
                self.asset
                    .session
                    .resources
                    .host_budget()
                    .reserve(std::mem::size_of::<FrameIndexEntry>())?,
            );
            entries.push(entry);
            for (i, &time) in times.iter().enumerate() {
                let mut local = Vec::new();
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
                let index = FrameIndex::new(fingerprint.clone(), reader.track.0, local)?;
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
            if pts > maximum {
                break;
            }
        }
        let index = FrameIndex::new(fingerprint.clone(), reader.track.0, entries)?;
        let selection = index.select_batch(times, self.selection, self.durations)?;
        let _metadata = reader
            .cpu
            .is_none()
            .then(|| self.asset.session.resources.reserve_cpu(0, 0))
            .transpose()?;
        if fingerprint.is_versioned() {
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
                            "selected frame was not retained by the bounded candidate scheduler",
                        )
                    })?
                    .clone(),
            );
        }
        let mut report = reader.report();
        report.output_frames = frames.len() as u64;
        Ok(FrameBatch::new(
            frames,
            Some(selection),
            report,
            self.asset.session.clone(),
            batch,
            fingerprint,
        ))
    }
    fn read_indexed(&self) -> Result<FrameBatch> {
        let times = self
            .times
            .as_deref()
            .ok_or_else(|| Error::invalid("indexed execution requires timestamps"))?;
        self.read_indexed_on(times, &mut None, &mut IndexedDecodeState::default())
    }
    fn read_indexed_on(
        &self,
        times: &[Time],
        retained: &mut Option<VideoReader>,
        state: &mut IndexedDecodeState,
    ) -> Result<FrameBatch> {
        self.validate_cpu_threads()?;
        self.indexed_limits.validate()?;
        let index = self
            .index
            .as_ref()
            .ok_or_else(|| Error::invalid("missing attached index"))?;
        index.validate(
            &self.asset.source,
            self.options.track,
            self.options.duration_policy,
        )?;
        let (output_bytes, working_bytes) =
            index.batch_metadata_bytes(times.len(), std::mem::size_of::<VideoFrame>())?;
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
                .ok_or_else(|| Error::incompatible("index anchor has no source PTS"))?;
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
                let mut frame = reader.next_frame()?.ok_or_else(|| {
                    Error::incompatible("decoder reached EOF before indexed selection")
                })?;
                if restart.is_none() {
                    state.observe_frame(started.elapsed());
                }
                let keyframe = reader
                    .decoder
                    .as_ref()
                    .and_then(Decoder::last_keyframe)
                    .ok_or_else(|| Error::incompatible("decoded keyframe evidence absent"))?;
                let ordinal = match position {
                    Some(position) => position,
                    None if frame.timing.source_pts == Some(anchor_pts) && keyframe => group.anchor,
                    None => continue,
                };
                if let Some(started) = restart.take() {
                    state.observe_restart(started.elapsed());
                }
                frame.timing = index.check_actual(ordinal, frame.timing, keyframe)?;
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
            .map_err(|_| Error::exhausted("indexed result allocation failed"))?;
        for selected in &selection.selections {
            frames.push(
                unique
                    .get(&selected.identity.ordinal)
                    .ok_or_else(|| Error::incompatible("selected indexed frame was not decoded"))?
                    .clone(),
            );
        }
        let mut report = reader.report();
        report.decoded_frames -= before_frames;
        report.bytes_read -= before_bytes;
        report.seeks -= before_seeks;
        report.index_cache_hits = 1;
        report.output_frames = frames.len() as u64;
        Ok(FrameBatch::new(
            frames,
            Some(selection),
            report,
            self.asset.session.clone(),
            batch,
            index.source().clone(),
        )
        .retain_metadata(metadata))
    }
}
/// Stateful indexed CPU sampling with one retained decoder and verified position.
///
/// Batches retain independent output leases. Work counters and limits apply to
/// each call; setup is performed at construction. Any failed call closes this
/// reader, so stale evidence or partial native state cannot be reused.
pub struct IndexedVideoReader {
    request: VideoRequest,
    reader: Option<VideoReader>,
    state: IndexedDecodeState,
}
impl IndexedVideoReader {
    /// Select this batch's timestamps in caller order, including repetitions.
    pub fn read(&mut self, times: &[Time]) -> Result<FrameBatch> {
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

/// Reusable CPU video work; each execution owns a fresh cursor and decoder.
#[derive(Debug, Clone)]
pub struct VideoPlan {
    request: VideoRequest,
    source: media_io::source::SourceFingerprint,
    description: media_pipeline::plan::PlanDescription,
}
impl VideoPlan {
    fn validate_source(&self) -> Result<()> {
        self.request.validate_cpu_threads()?;
        let _cpu = self.request.asset.session.resources.reserve_cpu(0, 0)?;
        if self.source.is_versioned() && self.request.asset.source.fingerprint()? != self.source {
            return Err(Error::incompatible(
                "source changed since video plan compilation",
            ));
        }
        Ok(())
    }
    /// Structured backend, placement, allocation and unresolved choices.
    pub fn description(&self) -> &media_pipeline::plan::PlanDescription {
        &self.description
    }
    /// Explicit interpretation of packet/codec duration hints for this plan.
    pub fn duration_policy(&self) -> media_core::policy::FrameDurationPolicy {
        self.request.options.duration_policy
    }
    /// Explicit permission to infer presentation intervals from following PTS.
    pub fn duration_inference(&self) -> DurationInference {
        self.request.durations
    }
    /// Human-readable execution choices and their costs.
    pub fn explain(&self) -> String {
        self.description.explain()
    }
    /// Materialize a bounded result with source checks before and after execution.
    pub fn execute(&self) -> Result<FrameBatch> {
        self.validate_source()?;
        let result = self.request.read()?;
        self.validate_source()?;
        Ok(result)
    }
    /// Open an independent sequential reader; sampled plans use execute().
    pub fn reader(&self) -> Result<VideoReader> {
        self.validate_source()?;
        self.request.reader()
    }
}
/// Independent sequential video execution with output-retained allocation charges.
pub struct VideoReader {
    decoder: Option<Decoder>,
    cpu: Option<CpuReservation>,
    info: AssetInfo,
    track: TrackId,
    report: VideoReport,
    asset: Asset,
    cancel: CancellationToken,
    failed: bool,
}
impl VideoReader {
    /// Share cancellation with caller work, including native demux/AVIO reads.
    pub fn with_cancellation(mut self, cancel: CancellationToken) -> Self {
        if let Some(decoder) = &self.decoder {
            let token = cancel.clone();
            decoder.set_interrupt(Arc::new(move || token.is_cancelled()));
        }
        self.cancel = cancel;
        self
    }
    fn next_index_timing(&mut self) -> Result<Option<IndexedTiming>> {
        if self.failed {
            return Err(Error::invalid("video index execution previously failed"));
        }
        let result = (|| {
            self.cancel.check()?;
            let next = self
                .decoder
                .as_mut()
                .ok_or_else(|| Error::invalid("index decoder retired"))?
                .next_timing()?;
            self.cancel.check()?;
            Ok(next.map(|record| IndexedTiming {
                timing: record.timing,
                keyframe: record.keyframe,
                seek_timestamp: record.seek_timestamp,
            }))
        })();
        self.update_report();
        if result.is_err() {
            self.failed = true;
            self.retire();
        } else if matches!(result, Ok(None)) {
            self.retire();
        }
        result
    }
    /// Probed source tracks and common timeline.
    pub fn info(&self) -> &AssetInfo {
        &self.info
    }
    /// Cooperative cancellation handle.
    pub fn cancellation(&self) -> CancellationToken {
        self.cancel.clone()
    }
    /// Seek to a validated decoder random-access region, not an exact selection.
    ///
    /// Returned frames keep source timing; global presentation ordinals become
    /// unknown after navigation. Use a timestamp batch for exact selection.
    pub fn seek(&mut self, time: Time) -> Result<()> {
        let result = self.cancel.check().and_then(|()| {
            self.decoder
                .as_mut()
                .ok_or_else(|| {
                    Error::invalid("video decoder retired; open a new reader for navigation")
                })?
                .seek(time)
        });
        if result.is_err() {
            self.failed = true;
            self.retire();
        }
        result
    }
    /// Read one frame; retained outputs remain valid after this reader drops.
    pub fn next_frame(&mut self) -> Result<Option<VideoFrame>> {
        if self.failed {
            return Err(Error::invalid("video execution previously failed"));
        }
        let result = (|| {
            self.cancel.check()?;
            let Some(decoder) = self.decoder.as_mut() else {
                return Ok(None);
            };
            let Some(mut frame) = decoder.next_frame()? else {
                return Ok(None);
            };
            self.cancel.check()?;
            let guard = self
                .asset
                .session
                .resources
                .host_budget()
                .reserve(frame.image.storage().byte_len())?;
            frame.image = frame.image.retain(guard);
            Ok(Some(frame))
        })();
        self.update_report();
        if result.is_err() {
            self.failed = true;
            self.retire();
        } else if matches!(result, Ok(None)) {
            self.retire();
        }
        result
    }
    fn update_report(&mut self) {
        if let Some(decoder) = &self.decoder {
            self.report.bytes_read = decoder.bytes_read();
            self.report.decoded_frames = decoder.frames_decoded();
            self.report.output_frames = decoder.frames_decoded();
            if self.report.color_assumptions.as_slice() != decoder.color_assumptions() {
                self.report.color_assumptions = decoder.color_assumptions().to_vec();
            }
        }
    }
    fn retire(&mut self) {
        self.update_report();
        self.decoder.take();
        self.cpu.take();
    }
    /// Actual source-read, decode, and color-assumption counters.
    pub fn report(&self) -> VideoReport {
        self.report.clone()
    }
}
impl Iterator for VideoReader {
    type Item = Result<VideoFrame>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.failed {
            return None;
        }
        self.next_frame().transpose()
    }
}
/// Finite caller-ordered video result with selection provenance.
#[derive(Debug)]
pub struct FrameBatch {
    /// Source object identity/version validated for this result.
    pub source: media_io::source::SourceFingerprint,
    session: crate::Session,
    _batch: Arc<BatchReservation>,
    /// Returned frames; repeated selections share image ownership.
    pub frames: Vec<VideoFrame>,
    /// Requested/actual times and unique source work, when timestamp sampling.
    pub selection: Option<SelectionBatch>,
    /// Observed backend work.
    pub report: VideoReport,
    _metadata: Option<Arc<HostReservation>>,
}
impl FrameBatch {
    fn new(
        mut frames: Vec<VideoFrame>,
        selection: Option<SelectionBatch>,
        report: VideoReport,
        session: crate::Session,
        batch: Arc<BatchReservation>,
        source: media_io::source::SourceFingerprint,
    ) -> Self {
        for frame in &mut frames {
            frame.image = frame.image.clone().retain(batch.clone());
        }
        Self {
            source,
            frames,
            selection,
            report,
            session,
            _batch: batch,
            _metadata: None,
        }
    }
    fn retain_metadata(mut self, metadata: Arc<HostReservation>) -> Self {
        for frame in &mut self.frames {
            frame.image = frame.image.clone().retain(metadata.clone());
        }
        self._metadata = Some(metadata);
        self
    }
    /// Shared session retained by the batch.
    pub fn session(&self) -> &crate::Session {
        &self.session
    }
    /// Materialize a checked contiguous host RGB model-input batch.
    #[cfg(feature = "interop")]
    pub fn to_tensor(
        &self,
        spec: &media_interop::tensor::TensorSpec,
    ) -> Result<media_interop::tensor::Tensor> {
        let images: Vec<_> = self
            .frames
            .iter()
            .map(|frame| frame.image.clone())
            .collect();
        crate::tensor_request::pack_images(&images, &self.session, spec)
    }
}
/// Video execution counters, separate from opaque driver/codec memory.
#[derive(Debug, Clone)]
pub struct VideoReport {
    /// Native indexed keyframe seeks actually performed by this execution.
    pub seeks: u64,
    /// Source-validated attached indexes used; cold construction is separate.
    pub index_cache_hits: u64,
    /// Explicit interpretation of backend-reported duration hints.
    pub duration_policy: media_core::policy::FrameDurationPolicy,
    /// Actual encoded bytes read, including probe rereads.
    pub bytes_read: u64,
    /// Decoded dependency and output frames.
    pub decoded_frames: u64,
    /// Returned frames, including repeated caller selections.
    pub output_frames: u64,
    /// Actual hardware decode; this CPU reader reports false.
    pub hardware_decode: bool,
    /// Explicit matrix/range assumptions used by the backend.
    pub color_assumptions: Vec<String>,
}
/// Compressed stream-copy request; never silently re-encodes or drops tracks.
#[derive(Debug, Clone)]
pub struct RemuxRequest {
    asset: Asset,
    tracks: Option<Vec<TrackId>>,
}
impl RemuxRequest {
    pub(crate) fn new(asset: Asset) -> Self {
        Self {
            asset,
            tracks: None,
        }
    }
    /// Copy exactly these streams; an empty list is invalid.
    pub fn tracks(mut self, tracks: impl IntoIterator<Item = TrackId>) -> Self {
        self.tracks = Some(tracks.into_iter().collect());
        self
    }
    /// Copy selected (or all) streams into a new file and explicitly finalize.
    pub fn write(&self, path: impl AsRef<Path>) -> Result<RemuxReport> {
        let _cpu = self.asset.session.resources.reserve_cpu(0, 0)?;
        format::remux_with_options(
            &self.asset.source,
            path.as_ref(),
            self.tracks.as_deref(),
            ProbeOptions::default(),
        )
    }
}

/// Incremental software video encoding with selected compressed audio passthrough.
#[derive(Debug, Clone)]
pub struct ConvertRequest {
    asset: Asset,
    options: media_backend_ffmpeg::encode::TranscodeOptions,
    fit: Option<media_pipeline::operation::Fit>,
}
impl ConvertRequest {
    pub(crate) fn new(asset: Asset) -> Self {
        Self {
            asset,
            options: Default::default(),
            fit: None,
        }
    }
    /// Replace exact codec, stream-copy, profile, worker, and allocation settings.
    pub fn options(mut self, options: media_backend_ffmpeg::encode::TranscodeOptions) -> Self {
        self.options = options;
        self
    }
    /// Resize stored YUV components within these bounds; no transfer-function conversion is implied.
    pub fn resize_components(mut self, fit: media_pipeline::operation::Fit) -> Self {
        self.fit = Some(fit);
        self
    }
    /// Set the software encoder target bitrate in bits per second.
    pub fn bitrate(mut self, bits_per_second: usize) -> Self {
        self.options.bit_rate = bits_per_second;
        self
    }
    /// Require true hardware encoding; this software path fails before output creation.
    pub fn hardware_required(mut self) -> Self {
        self.options.hardware_required = true;
        self
    }
    /// Select the native NVDEC/NVENC path, using NVENC's default preset.
    /// Selected tracks, bitrate, codec and exact dimensions are retained; CPU
    /// resize filters require a native implementation and fail explicitly.
    #[cfg(feature = "cuda")]
    pub fn cuda(
        self,
        ordinal: u32,
        limits: media_backend_cuda::CudaLimits,
    ) -> crate::cuda_request::CudaConvertRequest {
        crate::cuda_request::CudaConvertRequest::from_software(
            self.asset,
            ordinal,
            limits,
            self.options,
            self.fit,
        )
    }
    /// Drain the decoder/encoder, copy selected audio, and finalize a new output file.
    pub fn write(
        &self,
        path: impl AsRef<Path>,
    ) -> Result<media_backend_ffmpeg::encode::TranscodeReport> {
        if self.options.threads == 0 || self.options.threads > 64 {
            return Err(Error::invalid(
                "CPU transcode requires 1..=64 requested workers per codec",
            ));
        }
        let threads = self
            .options
            .threads
            .checked_mul(2)
            .ok_or_else(|| Error::overflow("transcode codec worker count"))?;
        let _cpu = self.asset.session.resources.reserve_cpu(2, threads)?;
        let mut options = self.options.clone();
        let mut extra_probe_bytes = 0_u64;
        options.max_frame_bytes = options
            .max_frame_bytes
            .min(self.asset.session.limits().host_memory as u64);
        if let Some(fit) = self.fit {
            let info = format::probe(&self.asset.source, options.probe)?;
            extra_probe_bytes = info.probe_bytes_read;
            let track = info
                .tracks
                .iter()
                .find(|t| {
                    t.kind == format::TrackKind::Video
                        && !t.cover_art
                        && options.video_track.is_none_or(|id| t.id == id)
                })
                .ok_or_else(|| Error::unsupported("selected video stream is unavailable"))?;
            let (iw, ih) = (
                track
                    .width
                    .ok_or_else(|| Error::unsupported("video width is not established"))?,
                track
                    .height
                    .ok_or_else(|| Error::unsupported("video height is not established"))?,
            );
            let (width, height) = inside_dimensions(iw, ih, fit)?;
            options.width = Some(width);
            options.height = Some(height);
        }
        let mut report =
            media_backend_ffmpeg::encode::transcode(&self.asset.source, path.as_ref(), options)?;
        report.source_bytes_read = report
            .source_bytes_read
            .checked_add(extra_probe_bytes)
            .ok_or_else(|| Error::overflow("transcode probe byte counter"))?;
        Ok(report)
    }
}

pub(crate) fn inside_dimensions(
    iw: u32,
    ih: u32,
    fit: media_pipeline::operation::Fit,
) -> Result<(u32, u32)> {
    let media_pipeline::operation::Fit::Inside { width, height } = fit else {
        return Err(Error::unsupported(
            "encoder cover-crop is unavailable; select inside bounds or exact output dimensions",
        ));
    };
    if width == 0 || height == 0 || iw == 0 || ih == 0 {
        return Err(Error::invalid("resize dimensions must be nonzero"));
    }
    if u64::from(width) * u64::from(ih) <= u64::from(height) * u64::from(iw) {
        Ok((
            width,
            u32::try_from(u64::from(ih) * u64::from(width) / u64::from(iw))
                .map_err(|_| Error::overflow("resize height"))?
                .max(1),
        ))
    } else {
        Ok((
            u32::try_from(u64::from(iw) * u64::from(height) / u64::from(ih))
                .map_err(|_| Error::overflow("resize width"))?
                .max(1),
            height,
        ))
    }
}
