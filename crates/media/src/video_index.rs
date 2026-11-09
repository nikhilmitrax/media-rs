//! Explicit decoded timing indexes and bounded GOP-local selection schedules.
//!
//! Index construction is a visible full sequential decode. Reuse stores timing,
//! never decoded image objects, and never infers ordinals from packet count/FPS.
use media_core::{
    error::{Error, ErrorKind, Result},
    media::{FrameTiming, TrackId},
    policy::FrameDurationPolicy,
    time::Time,
};
use media_io::source::{Source, SourceFingerprint, SourceIdentity, SourceVersion};
use media_pipeline::{
    execution::CancellationToken,
    sampling::{DurationInference, FrameIndex, FrameIndexEntry, SelectionBatch, TemporalSelection},
    session::{HostReservation, Session as Resources},
};
use std::{collections::BTreeMap, sync::Arc, time::Duration};
#[path = "video_index_cache.rs"]
mod cache;

/// Hard bounds for explicitly authorized complete decoded-index construction.
#[derive(Debug, Clone, Copy)]
pub struct IndexBuildLimits {
    /// Maximum decoded presentation records; no partial index is returned.
    pub max_frames: usize,
    /// Reserved metadata payload bytes, shared with the Session host budget.
    pub max_index_bytes: usize,
    /// Maximum actual encoded bytes read, including discovery and rereads.
    pub max_source_bytes: u64,
}
impl Default for IndexBuildLimits {
    fn default() -> Self {
        Self {
            max_frames: 1_000_000,
            max_index_bytes: 128 * 1024 * 1024,
            max_source_bytes: 1024 * 1024 * 1024,
        }
    }
}
impl IndexBuildLimits {
    /// Reject empty limits before source I/O or metadata allocation.
    pub fn validate(self) -> Result<()> {
        if self.max_frames == 0 || self.max_index_bytes == 0 || self.max_source_bytes == 0 {
            return Err(Error::invalid(
                "video index frame, metadata and source-read limits must be positive",
            ));
        }
        Ok(())
    }
}
/// Hard work/read limits for one execution using an attached index.
#[derive(Debug, Clone, Copy)]
pub struct IndexSeekLimits {
    /// Includes preroll/dependency frames and every requested GOP.
    pub max_decoded_frames: u64,
    /// Maximum actual native seeks in one caller batch; adjacent regions may
    /// share a decoder continuation without spending another seek.
    pub max_seeks: usize,
    /// Actual encoded bytes including header discovery, seeks and rereads.
    pub max_source_bytes: u64,
}
impl Default for IndexSeekLimits {
    fn default() -> Self {
        Self {
            max_decoded_frames: 65_536,
            max_seeks: 4096,
            max_source_bytes: 1024 * 1024 * 1024,
        }
    }
}
impl IndexSeekLimits {
    pub(crate) fn validate(self) -> Result<()> {
        if self.max_decoded_frames == 0 || self.max_seeks == 0 || self.max_source_bytes == 0 {
            return Err(Error::invalid(
                "indexed execution work/read limits must be positive",
            ));
        }
        Ok(())
    }
    pub(crate) fn check_decoded(self, count: u64) -> Result<()> {
        if count > self.max_decoded_frames {
            return Err(Error::exhausted(
                "indexed video execution exceeded its decoded-frame limit",
            )
            .with_constraint("indexed_decoded_frames"));
        }
        Ok(())
    }
    pub(crate) fn check_seek(self, count: u64) -> Result<()> {
        if count >= self.max_seeks as u64 {
            return Err(Error::exhausted("indexed batch exceeds its seek limit")
                .with_constraint("indexed_seeks"));
        }
        Ok(())
    }
}
/// Cold construction work, separate from later indexed execution counters.
#[derive(Debug, Clone, Copy, Default)]
pub struct IndexBuildReport {
    /// Actual presentation frames decoded to establish global ordinals.
    pub decoded_frames: u64,
    /// Actual encoded bytes read, including initial discovery.
    pub bytes_read: u64,
    /// True only for a required-NVDEC index build.
    pub hardware_decode: bool,
    /// Retained library metadata payload charge; codec-private heaps are excluded.
    pub retained_index_bytes: usize,
    /// Whether these records were loaded from an explicit persisted index.
    pub loaded: bool,
}
/// Immutable decoded evidence, bound to source version, stream and time policy.
#[derive(Debug)]
pub struct VideoIndex {
    source: SourceFingerprint,
    track: TrackId,
    origin: Time,
    policy: FrameDurationPolicy,
    frames: Vec<IndexedTiming>,
    selection: FrameIndex,
    anchors: Vec<usize>,
    report: IndexBuildReport,
    _metadata: HostReservation,
    resources: Resources,
}
#[derive(Debug, Clone, Copy)]
pub(crate) struct IndexedTiming {
    pub timing: FrameTiming,
    pub keyframe: bool,
    pub seek_timestamp: Option<Time>,
}
#[derive(Debug)]
pub(crate) struct SeekGroup {
    pub anchor: usize,
    pub wanted: Vec<u64>,
    pub last: u64,
}

/// Cursor position is established only by matching decoded index evidence.
/// Cost observations choose navigation, never presentation-frame identity.
#[derive(Default)]
pub(crate) struct IndexedDecodeState {
    pub next: Option<usize>,
    frame_ns: Option<u128>,
    restart_ns: Option<u128>,
}
impl IndexedDecodeState {
    pub fn should_seek(&self, group: &SeekGroup, remaining_frames: u64) -> bool {
        let Some(next) = self.next else {
            return true;
        };
        if next as u64 > group.wanted[0] {
            return true;
        }
        // Continuing must not spend the work budget on an avoidable gap.
        if group.last.saturating_sub(next as u64).saturating_add(1) > remaining_frames {
            return true;
        }
        let gap = group.anchor.saturating_sub(next) as u128;
        if gap == 0 {
            return false;
        }
        // A lazy decoder initialization or unusually expensive restart must not
        // turn a sparse indexed request into a long prefix scan. Spend at most
        // one region's local decode work on bridging its preceding gap.
        let local_frames = group
            .last
            .saturating_sub(group.anchor as u64)
            .saturating_add(1);
        if gap > local_frames as u128 {
            return true;
        }
        match (self.frame_ns, self.restart_ns) {
            (Some(frame), Some(restart)) => frame.saturating_mul(gap) > restart,
            _ => true,
        }
    }
    pub fn observe_frame(&mut self, elapsed: Duration) {
        Self::observe(&mut self.frame_ns, elapsed);
    }
    pub fn observe_restart(&mut self, elapsed: Duration) {
        Self::observe(&mut self.restart_ns, elapsed);
    }
    fn observe(cost: &mut Option<u128>, elapsed: Duration) {
        let sample = elapsed.as_nanos().max(1);
        *cost = Some(cost.map_or(sample, |previous| (previous / 2 + sample / 2).max(1)));
    }
}
impl VideoIndex {
    /// Established source identity/version; private adapter identities stay redacted.
    pub fn source(&self) -> &SourceFingerprint {
        &self.source
    }
    /// Selected logical video stream.
    pub fn track(&self) -> TrackId {
        self.track
    }
    /// Shared source time mapped to asset zero during construction.
    pub fn timeline_origin(&self) -> Time {
        self.origin
    }
    /// Explicit interpretation used for native duration hints.
    pub fn duration_policy(&self) -> FrameDurationPolicy {
        self.policy
    }
    /// Number of established presentation ordinals.
    pub fn len(&self) -> usize {
        self.frames.len()
    }
    /// Whether no decoded presentation frames were established.
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }
    /// Actual cold build/load costs; indexed reads report their own work.
    pub fn report(&self) -> &IndexBuildReport {
        &self.report
    }
    /// Established timing/keyframe records in presentation-time order; each
    /// entry retains its actual global source ordinal, including repeated PTS.
    pub fn entries(&self) -> &[FrameIndexEntry] {
        self.selection.entries()
    }
    pub(crate) fn validate(
        &self,
        source: &Source,
        track: Option<TrackId>,
        policy: FrameDurationPolicy,
    ) -> Result<()> {
        if !source.capabilities()?.seekable {
            return Err(Error::unsupported(
                "indexed video execution requires a seekable source or explicit spooling",
            ));
        }
        if !self.source.is_versioned() || source.fingerprint()? != self.source {
            return Err(
                Error::incompatible("video index source identity/version is stale")
                    .with_constraint("video_index_source"),
            );
        }
        if track.is_some_and(|track| track != self.track) || policy != self.policy {
            return Err(Error::incompatible(
                "video index stream or duration interpretation does not match this request",
            )
            .with_constraint("video_index_interpretation"));
        }
        Ok(())
    }
    pub(crate) fn frame(&self, ordinal: usize) -> Result<IndexedTiming> {
        self.frames
            .get(ordinal)
            .copied()
            .ok_or_else(|| Error::incompatible("decoded output extends beyond verified index"))
    }
    pub(crate) fn batch_metadata_bytes(
        &self,
        count: usize,
        frame_bytes: usize,
    ) -> Result<(usize, usize)> {
        if count > 65_536 {
            return Err(Error::invalid(
                "indexed timestamp batches are limited to65536 requests",
            ));
        }
        let identity = source_charge(&self.source)?.saturating_sub(4096) / 2;
        let output = std::mem::size_of::<media_pipeline::sampling::FrameSelection>()
            .checked_add(std::mem::size_of::<FrameIndexEntry>())
            .and_then(|n| n.checked_add(frame_bytes))
            .and_then(|n| n.checked_add(identity))
            .and_then(|n| n.checked_add(1024))
            .and_then(|n| n.checked_mul(count))
            .and_then(|n| n.checked_add(4096))
            .ok_or_else(|| Error::overflow("indexed result metadata size"))?;
        let working = frame_bytes
            .checked_add(640)
            .and_then(|n| n.checked_mul(count))
            .and_then(|n| n.checked_add(4096))
            .ok_or_else(|| Error::overflow("indexed scheduling metadata size"))?;
        Ok((output, working))
    }
    pub(crate) fn schedule(
        &self,
        times: &[Time],
        selection: TemporalSelection,
        inference: DurationInference,
        limits: IndexSeekLimits,
    ) -> Result<(SelectionBatch, Vec<SeekGroup>)> {
        limits.validate()?;
        let selection = self.selection.select_batch(times, selection, inference)?;
        let mut groups: BTreeMap<usize, Vec<u64>> = BTreeMap::new();
        for selected in &selection.unique_frames {
            let ordinal = usize::try_from(selected.ordinal)
                .map_err(|_| Error::overflow("indexed ordinal"))?;
            let preceding = self.anchors.partition_point(|&anchor| anchor <= ordinal);
            let anchor = *self
                .anchors
                .get(preceding.checked_sub(1).ok_or_else(|| {
                    Error::unsupported(
                        "no uniquely identifiable decoded keyframe precedes this selection",
                    )
                })?)
                .ok_or_else(|| Error::unsupported("no verified random-access anchor"))?;
            groups.entry(anchor).or_default().push(selected.ordinal);
        }
        let groups = groups
            .into_iter()
            .map(|(anchor, mut wanted)| {
                wanted.sort_unstable();
                let last = *wanted.last().expect("nonempty index group");
                SeekGroup {
                    anchor,
                    wanted,
                    last,
                }
            })
            .collect();
        Ok((selection, groups))
    }
    pub(crate) fn check_actual(
        &self,
        position: usize,
        actual: FrameTiming,
        keyframe: bool,
    ) -> Result<FrameTiming> {
        let expected = self.frame(position)?;
        if actual.source_pts != expected.timing.source_pts
            || keyframe != expected.keyframe
            || (self.policy == FrameDurationPolicy::TrustReported
                && actual.duration != expected.timing.duration)
        {
            return Err(Error::incompatible("actual decoded presentation sequence differs from the source-validated video index")
                .with_constraint("video_index_alignment"));
        }
        // Native SDK duration hints remain actual output evidence. Only verified
        // asset mapping/global ordinal is restored after decoder navigation.
        Ok(FrameTiming {
            pts: expected.timing.pts,
            ordinal: expected.timing.ordinal,
            ..actual
        })
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn build(
        resources: Resources,
        source: SourceFingerprint,
        track: TrackId,
        policy: FrameDurationPolicy,
        limits: IndexBuildLimits,
        reservation: HostReservation,
        hardware: bool,
        cancel: &CancellationToken,
        mut next: impl FnMut() -> Result<(Option<IndexedTiming>, u64)>,
    ) -> Result<Arc<Self>> {
        let mut frames = Vec::new();
        let maximum = limits.max_frames.min(
            limits
                .max_index_bytes
                .saturating_sub(source_charge(&source)?)
                / record_charge(),
        );
        if maximum == 0 {
            return Err(
                Error::exhausted("video index metadata budget cannot hold one frame")
                    .with_constraint("video_index_bytes"),
            );
        }
        let mut origin = None;
        let bytes_read = loop {
            cancel.check()?;
            let (frame, bytes) = next()?;
            if bytes > limits.max_source_bytes {
                return Err(Error::exhausted("video index source-read limit exceeded")
                    .with_constraint("video_source_bytes"));
            }
            let Some(frame) = frame else {
                break bytes;
            };
            if frames.len() >= maximum {
                return Err(Error::exhausted(
                    "complete video index exceeds its frame/metadata limit",
                )
                .with_constraint("video_index_bytes"));
            }
            let position = frames.len() as u64;
            let (Some(source_pts), Some(pts), Some(ordinal)) = (
                frame.timing.source_pts,
                frame.timing.pts,
                frame.timing.ordinal,
            ) else {
                return Err(Error::new(
                    ErrorKind::MissingData,
                    "index construction requires actual source/asset PTS and global sequential ordinals",
                ));
            };
            if ordinal != position {
                return Err(Error::incompatible(
                    "decoded presentation index is not sequential from stream start",
                ));
            }
            let mapped = source_pts.checked_sub(pts)?;
            if origin.is_some_and(|origin| origin != mapped) {
                return Err(Error::incompatible(
                    "asset timeline mapping changed during index construction",
                ));
            }
            origin = Some(mapped);
            if frames.len() == frames.capacity() {
                frames
                    .try_reserve_exact((maximum - frames.len()).min(1024))
                    .map_err(|_| Error::exhausted("video index allocation failed"))?;
            }
            frames.push(frame);
        };
        let origin = origin.ok_or_else(|| {
            Error::new(ErrorKind::MissingData, "video index has no decoded frames")
        })?;
        Self::finish(
            resources,
            source,
            track,
            origin,
            policy,
            frames,
            reservation,
            IndexBuildReport {
                decoded_frames: 0,
                bytes_read,
                hardware_decode: hardware,
                retained_index_bytes: 0,
                loaded: false,
            },
        )
    }
    #[allow(clippy::too_many_arguments, clippy::redundant_locals)]
    fn finish(
        resources: Resources,
        source: SourceFingerprint,
        track: TrackId,
        origin: Time,
        policy: FrameDurationPolicy,
        frames: Vec<IndexedTiming>,
        reservation: HostReservation,
        mut report: IndexBuildReport,
    ) -> Result<Arc<Self>> {
        // Scope ordering keeps the full reservation alive while every working
        // vector/map drops on an error, before transferring final ownership.
        let reservation = reservation;
        let frames = frames;
        let mut key_counts = BTreeMap::new();
        for frame in &frames {
            if frame.keyframe {
                *key_counts.entry(frame.timing.source_pts).or_insert(0usize) += 1;
            }
        }
        let mut anchors = Vec::new();
        let mut entries = Vec::new();
        entries
            .try_reserve_exact(frames.len())
            .map_err(|_| Error::exhausted("selection index allocation failed"))?;
        anchors
            .try_reserve_exact(frames.len())
            .map_err(|_| Error::exhausted("keyframe index allocation failed"))?;
        for (position, frame) in frames.iter().enumerate() {
            entries.push(FrameIndexEntry::from_timing(frame.timing, frame.keyframe)?);
            if frame.keyframe && key_counts.get(&frame.timing.source_pts) == Some(&1) {
                anchors.push(position);
            }
        }
        let bytes = frames
            .capacity()
            .checked_mul(std::mem::size_of::<IndexedTiming>())
            .and_then(|n| {
                n.checked_add(entries.capacity() * std::mem::size_of::<FrameIndexEntry>())
            })
            .and_then(|n| n.checked_add(anchors.capacity() * std::mem::size_of::<usize>()))
            .and_then(|n| n.checked_add(source_charge(&source).ok()?))
            .ok_or_else(|| Error::overflow("retained video index charge"))?;
        if !report.loaded {
            report.decoded_frames = frames.len() as u64;
        }
        report.retained_index_bytes = bytes;
        let selection =
            FrameIndex::new(source.clone(), track.0, entries)?.with_verified_keyframe_flags();
        drop(key_counts);
        if bytes > reservation.bytes() {
            drop((frames, selection, anchors));
            return Err(
                Error::exhausted("retained index exceeds its reserved metadata capacity")
                    .with_constraint("video_index_bytes"),
            );
        }
        let metadata = reservation.shrink_to(bytes)?;
        Ok(Arc::new(Self {
            source,
            track,
            origin,
            policy,
            frames,
            selection,
            anchors,
            report,
            _metadata: metadata,
            resources,
        }))
    }
}
fn record_charge() -> usize {
    std::mem::size_of::<IndexedTiming>()
        + std::mem::size_of::<FrameIndexEntry>()
        + std::mem::size_of::<usize>()
        + 128
}
fn source_charge(source: &SourceFingerprint) -> Result<usize> {
    let identity = match &source.identity {
        SourceIdentity::File(path) => path.capacity(),
        SourceIdentity::Custom(label) => label.capacity(),
        _ => 0,
    };
    let version = match &source.version {
        SourceVersion::Custom(version) | SourceVersion::Range { version, .. } => version.capacity(),
        _ => 0,
    };
    identity
        .checked_add(version)
        .and_then(|n| n.checked_mul(2))
        .and_then(|n| n.checked_add(4096))
        .ok_or_else(|| Error::overflow("video index source metadata size"))
}

pub(crate) fn prepare_build(
    source: &Source,
    resources: &Resources,
    limits: IndexBuildLimits,
) -> Result<(SourceFingerprint, HostReservation)> {
    limits.validate()?;
    let reservation = resources.host_budget().reserve(limits.max_index_bytes)?;
    let fingerprint = source.fingerprint()?;
    let minimum = source_charge(&fingerprint)?
        .checked_add(record_charge())
        .and_then(|n| n.checked_add(8192))
        .ok_or_else(|| Error::overflow("video index minimum metadata size"))?;
    if minimum > limits.max_index_bytes {
        return Err(Error::exhausted("video index metadata limit cannot admit source identity, scratch and one timing record").with_constraint("video_index_bytes"));
    }
    if !fingerprint.is_versioned() || !source.capabilities()?.seekable {
        return Err(Error::unsupported(
            "reusable decoded indexes require a versioned seekable source; spool explicitly otherwise",
        ));
    }
    Ok((fingerprint, reservation))
}

#[cfg(test)]
mod navigation_tests {
    use super::*;

    #[test]
    fn observed_restart_cost_avoids_short_gaps_but_work_limits_and_rewinds_take_priority() {
        let mut state = IndexedDecodeState {
            next: Some(10),
            ..Default::default()
        };
        state.observe_frame(Duration::from_millis(1));
        state.observe_restart(Duration::from_millis(8));
        let nearby = SeekGroup {
            anchor: 12,
            wanted: vec![14],
            last: 14,
        };
        assert!(!state.should_seek(&nearby, 20));
        let distant = SeekGroup {
            anchor: 30,
            wanted: vec![35],
            last: 35,
        };
        assert!(state.should_seek(&distant, 100));
        state.observe_restart(Duration::from_secs(10));
        assert!(
            state.should_seek(&distant, 100),
            "a startup outlier cannot justify walking a large sparse gap"
        );
        assert!(
            state.should_seek(&nearby, 4),
            "continuing cannot burn a limited frame budget on the gap"
        );
        let backwards = SeekGroup {
            anchor: 0,
            wanted: vec![5],
            last: 5,
        };
        assert!(state.should_seek(&backwards, 100));
        state.next = None;
        assert!(
            state.should_seek(&nearby, 100),
            "unverified cursor positions always realign"
        );
    }
}
