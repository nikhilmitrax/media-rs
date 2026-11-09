//! Deterministic temporal selection and schedule-independent dataset sampling.
//!
//! A frame index is established timing evidence supplied by an actual decoder or
//! indexer. Selection never scans input, decodes frames, silently approximates a
//! requested time, or infers a missing duration without an explicit policy.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

use media_core::error::{Error, ErrorKind, Result};
use media_core::time::{Time, TimeRange};
use media_io::source::{Source, SourceFingerprint, SourceIdentity, SourceVersion};

/// Bounds for pure selection/schedule output and its planned payload allocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SamplingLimits {
    /// Maximum requests or auxiliary coverage intervals; checked before storage.
    pub max_requests: usize,
    /// Maximum planned vector/identity/interval payload bytes, excluding allocator overhead.
    pub max_bytes: usize,
}
impl Default for SamplingLimits {
    fn default() -> Self {
        Self {
            max_requests: 65536,
            max_bytes: 64 * 1024 * 1024,
        }
    }
}
impl SamplingLimits {
    /// Reject zero resource bounds.
    pub fn validate(self) -> Result<()> {
        if self.max_requests == 0 || self.max_bytes == 0 {
            return Err(Error::invalid(
                "sampling request and byte bounds must be nonzero",
            ));
        }
        Ok(())
    }
    pub(crate) fn check(self, count: usize, bytes: usize) -> Result<()> {
        self.validate()?;
        if count > self.max_requests {
            return Err(
                Error::exhausted("sampling exceeds the configured request count")
                    .with_constraint("max_requests"),
            );
        }
        if bytes > self.max_bytes {
            return Err(
                Error::exhausted("sampling exceeds planned payload-byte capacity")
                    .with_constraint("max_bytes"),
            );
        }
        Ok(())
    }
}

/// Provenance for an exact verified presentation-ordinal query, not a timestamp query.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OrdinalSelection {
    /// Requested global presentation ordinal in the selected logical stream.
    pub requested_ordinal: u64,
    /// Entry with that exact established ordinal; no vector position is substituted.
    pub frame: FrameIndexEntry,
    /// Source version and logical source-frame identity.
    pub identity: FrameIdentity,
    /// Index into the batch's presentation-ordered unique_frames.
    pub work_index: usize,
}
/// Caller-ordered ordinal selections and deduplicated source-frame work.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OrdinalBatch {
    /// One exact result per request, retaining repetitions and caller order.
    pub selections: Vec<OrdinalSelection>,
    /// Selected source entries, sorted by PTS and source ordinal.
    pub unique_frames: Vec<FrameIndexEntry>,
}
/// One known keyframe; a random-access marker is not a decoded output frame.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KeyframeSelection {
    /// Verified source entry and original presentation time.
    pub frame: FrameIndexEntry,
    /// Original source/track/presentation ordinal.
    pub identity: FrameIdentity,
}
/// Known indexed keyframes whose PTS lies in the requested half-open interval.
/// A partial index does not claim coverage of unindexed recording regions.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KeyframeBatch {
    /// Original half-open PTS query.
    pub requested: TimeRange,
    /// Verified keyframe entries, in presentation order.
    pub frames: Vec<KeyframeSelection>,
}

/// Timing and identity established for a frame in presentation order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FrameIndexEntry {
    /// Presentation timestamp on the common asset timeline.
    pub pts: Time,
    /// Established positive presentation duration, if known.
    pub duration: Option<Time>,
    /// Stable presentation ordinal within the selected logical stream.
    pub ordinal: u64,
    /// Whether the source identifies this frame as a random-access keyframe.
    pub keyframe: bool,
}
impl FrameIndexEntry {
    /// Copy established decoder timing without inventing PTS or global ordinals.
    /// Decoder navigation can invalidate ordinal knowledge; such timing fails.
    pub fn from_timing(timing: media_core::media::FrameTiming, keyframe: bool) -> Result<Self> {
        Ok(Self{pts:timing.pts.ok_or_else(||Error::new(ErrorKind::MissingData,"frame PTS is not established"))?,
            duration:timing.duration,ordinal:timing.ordinal.ok_or_else(||Error::unsupported("global presentation ordinal is unknown after navigation or incomplete indexing"))?,keyframe})
    }
}

/// Temporal selection, independent of navigation/index-building policy.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum TemporalSelection {
    /// Select the uniquely established presentation interval containing a time.
    #[default]
    DisplayedAt,
    /// Nearest PTS; ties choose earlier PTS, then earlier presentation ordinal.
    Nearest,
    /// Greatest PTS at or before the request; duplicate PTS choose earlier ordinal.
    AtOrBefore,
    /// Least PTS at or after the request; duplicate PTS choose earlier ordinal.
    AtOrAfter,
}

/// Permission to derive missing presentation durations from subsequent PTS.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum DurationInference {
    /// Require an established duration; unknown intervals remain unresolved.
    #[default]
    RequireExplicit,
    /// End a missing interval at the next strictly greater PTS, if one exists.
    ///
    /// This does not establish the last frame's duration. Duplicate PTS remain
    /// potentially overlapping intervals rather than being silently discarded.
    NextTimestamp,
}

/// Source identity/version, logical stream, and stable source-frame ordinal.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FrameIdentity {
    /// Byte object identity/version at index construction time.
    pub source: SourceFingerprint,
    /// Logical stream identifier.
    pub track: u32,
    /// Established presentation ordinal in that stream.
    pub ordinal: u64,
}

/// One exact temporal selection, retaining requested and actual source timing.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FrameSelection {
    /// The caller's requested time, not a rebased output timestamp.
    pub requested: Time,
    /// Selected frame's original presentation timestamp.
    pub actual: Time,
    /// Established or explicitly inferred presentation duration, when known.
    pub duration: Option<Time>,
    /// Whether the duration was derived under `NextTimestamp`.
    pub duration_inferred: bool,
    /// Stable source identity, preserved across repeated selections.
    pub identity: FrameIdentity,
    /// Index into [`SelectionBatch::unique_frames`]; zero for a single selection.
    pub work_index: usize,
}

/// Caller-ordered results and presentation-ordered unique selected frames.
///
/// `unique_frames` deduplicates requested source frames. It is not a complete
/// decoder schedule: dependency frames and valid random-access points still
/// belong to a backend's seek/decode planner.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SelectionBatch {
    /// Exactly one result per input request, in the same order.
    pub selections: Vec<FrameSelection>,
    /// Selected source frames, ordered by PTS then presentation ordinal.
    pub unique_frames: Vec<FrameIndexEntry>,
}

/// Timing evidence for one source version and one logical track.
///
/// Construction normalizes entry storage by PTS then ordinal. The caller must
/// establish ordinals from presentation order rather than decode/packet order.
/// Before reusing an index with a mutable source, call [`Self::validate_source`].
#[derive(Clone, Debug)]
pub struct FrameIndex {
    source: SourceFingerprint,
    track: u32,
    entries: Vec<FrameIndexEntry>,
    verified_keyframe_flags: bool,
}

impl FrameIndex {
    /// Validate entries and associate them with their original source version.
    ///
    /// Empty indexes are permitted. Ordinals must be unique, and established
    /// durations must be positive with representable end timestamps. Duplicate
    /// PTS are retained so temporal selection can report ambiguity explicitly.
    pub fn new(
        source: SourceFingerprint,
        track: u32,
        mut entries: Vec<FrameIndexEntry>,
    ) -> Result<Self> {
        let mut ordinals = BTreeSet::new();
        for entry in &entries {
            if !ordinals.insert(entry.ordinal) {
                return Err(Error::invalid(
                    "frame index contains a duplicate presentation ordinal",
                )
                .with_track(track));
            }
            if let Some(duration) = entry.duration {
                if duration <= Time::ZERO {
                    return Err(
                        Error::invalid("established frame duration must be positive")
                            .with_track(track)
                            .with_time(entry.pts),
                    );
                }
                entry.pts.checked_add(duration)?;
            }
        }
        entries.sort_by_key(|entry| (entry.pts, entry.ordinal));
        Ok(Self {
            source,
            track,
            entries,
            verified_keyframe_flags: false,
        })
    }

    /// The source identity/version this timing evidence describes.
    pub fn source(&self) -> &SourceFingerprint {
        &self.source
    }

    /// The selected logical stream.
    pub const fn track(&self) -> u32 {
        self.track
    }

    /// Presentation-sorted timing entries, retaining duplicate PTS.
    pub fn entries(&self) -> &[FrameIndexEntry] {
        &self.entries
    }
    /// Declare that every supplied entry's keyframe flag was established by the
    /// source decoder/indexer. This performs no I/O and invents no missing flags.
    /// Default indexes lack this evidence because false can also mean unknown.
    pub fn with_verified_keyframe_flags(mut self) -> Self {
        self.verified_keyframe_flags = true;
        self
    }
    /// Whether all supplied keyframe flags carry explicit caller-provided evidence.
    pub fn keyframe_flags_verified(&self) -> bool {
        self.verified_keyframe_flags
    }
    /// Select exactly an established presentation ordinal from this index.
    /// Local decoder positions after a seek must never be supplied as global
    /// ordinals; absent ordinals fail rather than using vector positions.
    pub fn select_ordinal(&self, ordinal: u64) -> Result<OrdinalSelection> {
        let mut batch = self.select_ordinal_batch(&[ordinal], SamplingLimits::default())?;
        Ok(batch.selections.remove(0))
    }
    /// Resolve a bounded ordinal batch with exact provenance and caller order.
    /// Scans only this in-memory index, not the media or an implicit full-file index.
    pub fn select_ordinal_batch(
        &self,
        requests: &[u64],
        limits: SamplingLimits,
    ) -> Result<OrdinalBatch> {
        let per = std::mem::size_of::<OrdinalSelection>()
            .checked_add(std::mem::size_of::<FrameIndexEntry>())
            .and_then(|n| n.checked_add(self.identity_heap_bytes()))
            .and_then(|n| n.checked_add(128))
            .ok_or_else(|| Error::overflow("ordinal selection metadata size"))?;
        limits.check(
            requests.len(),
            requests
                .len()
                .checked_mul(per)
                .ok_or_else(|| Error::overflow("ordinal batch size"))?,
        )?;
        let wanted: BTreeSet<_> = requests.iter().copied().collect();
        let mut found = BTreeMap::new();
        for (position, entry) in self.entries.iter().enumerate() {
            if wanted.contains(&entry.ordinal) {
                found.insert(entry.ordinal, position);
            }
        }
        for ordinal in requests {
            if !found.contains_key(ordinal) {
                return Err(Error::new(
                    ErrorKind::MissingData,
                    format!("presentation ordinal {ordinal} is not established by this index"),
                )
                .with_track(self.track));
            }
        }
        let positions: BTreeSet<_> = found.values().copied().collect();
        let work: BTreeMap<_, _> = positions
            .iter()
            .copied()
            .enumerate()
            .map(|(work, position)| (position, work))
            .collect();
        let mut unique_frames = Vec::new();
        unique_frames
            .try_reserve_exact(positions.len())
            .map_err(|e| Error::exhausted(e.to_string()))?;
        unique_frames.extend(positions.iter().map(|&position| self.entries[position]));
        let mut selections = Vec::new();
        selections
            .try_reserve_exact(requests.len())
            .map_err(|e| Error::exhausted(e.to_string()))?;
        for &ordinal in requests {
            let position = found[&ordinal];
            let frame = self.entries[position];
            selections.push(OrdinalSelection {
                requested_ordinal: ordinal,
                frame,
                identity: self.identity(frame.ordinal),
                work_index: work[&position],
            });
        }
        Ok(OrdinalBatch {
            selections,
            unique_frames,
        })
    }
    /// Enumerate verified keyframe flags in a half-open PTS interval.
    /// Unknown flags error; unknown display duration is harmless for PTS selection.
    /// This returns only entries already indexed and does no decoding/navigation.
    pub fn keyframes_in(&self, range: TimeRange, limits: SamplingLimits) -> Result<KeyframeBatch> {
        if !self.verified_keyframe_flags {
            return Err(
                Error::unsupported("keyframe flags were not verified for this index")
                    .with_track(self.track),
            );
        }
        let count = self
            .entries
            .iter()
            .filter(|entry| entry.keyframe && range.contains(entry.pts))
            .count();
        let bytes = std::mem::size_of::<KeyframeSelection>()
            .checked_add(self.identity_heap_bytes())
            .and_then(|per| per.checked_mul(count))
            .ok_or_else(|| Error::overflow("keyframe selection size"))?;
        limits.check(count, bytes)?;
        let mut frames = Vec::new();
        frames
            .try_reserve_exact(count)
            .map_err(|e| Error::exhausted(e.to_string()))?;
        for &frame in &self.entries {
            if frame.keyframe && range.contains(frame.pts) {
                frames.push(KeyframeSelection {
                    frame,
                    identity: self.identity(frame.ordinal),
                });
            }
        }
        Ok(KeyframeBatch {
            requested: range,
            frames,
        })
    }
    fn identity(&self, ordinal: u64) -> FrameIdentity {
        FrameIdentity {
            source: self.source.clone(),
            track: self.track,
            ordinal,
        }
    }
    fn identity_heap_bytes(&self) -> usize {
        let identity = match &self.source.identity {
            SourceIdentity::File(path) => path.as_os_str().as_encoded_bytes().len(),
            SourceIdentity::Custom(text) => text.len(),
            SourceIdentity::Memory(_) => 0,
            SourceIdentity::Private(bytes) => bytes.len(),
        };
        identity.saturating_add(match &self.source.version {
            SourceVersion::Custom(text) => text.len(),
            _ => 0,
        })
    }

    /// Reject reuse against an unversioned or changed source.
    pub fn validate_source(&self, source: &Source) -> Result<()> {
        let current = source.fingerprint()?;
        if !self.source.is_versioned() || !current.is_versioned() {
            return Err(
                Error::unsupported("frame index reuse requires a stable source version")
                    .with_track(self.track),
            );
        }
        if current != self.source {
            return Err(
                Error::incompatible("frame index is stale for the current source version")
                    .with_track(self.track),
            );
        }
        Ok(())
    }

    /// Select a frame under explicit temporal and duration-inference policies.
    pub fn select(
        &self,
        requested: Time,
        policy: TemporalSelection,
        inference: DurationInference,
    ) -> Result<FrameSelection> {
        let index = self.select_index(requested, policy, inference)?;
        self.make_selection(requested, index, inference, 0)
    }

    /// Select a finite batch, preserving caller order and deduplicating source frames.
    ///
    /// Empty requests produce an empty batch. Any missing or ambiguous request
    /// fails the batch; skip/padding behavior must be explicitly added by a higher
    /// layer rather than being hidden by this exact selector.
    pub fn select_batch(
        &self,
        requests: &[Time],
        policy: TemporalSelection,
        inference: DurationInference,
    ) -> Result<SelectionBatch> {
        self.select_batch_with_limits(requests, policy, inference, SamplingLimits::default())
    }
    /// Exact timestamp batch with explicit request and planned metadata bounds.
    /// The original select_batch API uses the default SamplingLimits.
    pub fn select_batch_with_limits(
        &self,
        requests: &[Time],
        policy: TemporalSelection,
        inference: DurationInference,
        limits: SamplingLimits,
    ) -> Result<SelectionBatch> {
        let per = std::mem::size_of::<FrameSelection>()
            .checked_add(std::mem::size_of::<FrameIndexEntry>())
            .and_then(|n| n.checked_add(self.identity_heap_bytes()))
            .and_then(|n| n.checked_add(128))
            .ok_or_else(|| Error::overflow("timestamp selection metadata size"))?;
        limits.check(
            requests.len(),
            requests
                .len()
                .checked_mul(per)
                .ok_or_else(|| Error::overflow("timestamp batch size"))?,
        )?;
        let mut selected = Vec::new();
        selected
            .try_reserve_exact(requests.len())
            .map_err(|error| {
                Error::exhausted(format!("cannot allocate frame selections: {error}"))
            })?;
        for &requested in requests {
            selected.push(self.select_index(requested, policy, inference)?);
        }
        let unique: BTreeSet<usize> = selected.iter().copied().collect();
        let work_indices: BTreeMap<usize, usize> = unique
            .iter()
            .copied()
            .enumerate()
            .map(|(work_index, entry_index)| (entry_index, work_index))
            .collect();
        let unique_frames = unique
            .into_iter()
            .map(|index| self.entries[index])
            .collect();
        let selections = requests
            .iter()
            .copied()
            .zip(selected)
            .map(|(requested, index)| {
                self.make_selection(requested, index, inference, work_indices[&index])
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(SelectionBatch {
            selections,
            unique_frames,
        })
    }

    fn select_index(
        &self,
        requested: Time,
        policy: TemporalSelection,
        inference: DurationInference,
    ) -> Result<usize> {
        let selected = match policy {
            TemporalSelection::DisplayedAt => return self.displayed_at(requested, inference),
            TemporalSelection::Nearest => self
                .entries
                .iter()
                .enumerate()
                .min_by(|(_, left), (_, right)| {
                    distance_cmp(requested, left.pts, right.pts)
                        .then_with(|| left.pts.cmp(&right.pts))
                        .then_with(|| left.ordinal.cmp(&right.ordinal))
                })
                .map(|(index, _)| index),
            TemporalSelection::AtOrAfter => {
                let index = self.entries.partition_point(|entry| entry.pts < requested);
                (index < self.entries.len()).then_some(index)
            }
            TemporalSelection::AtOrBefore => {
                let end = self.entries.partition_point(|entry| entry.pts <= requested);
                end.checked_sub(1).map(|index| {
                    let pts = self.entries[index].pts;
                    self.entries.partition_point(|entry| entry.pts < pts)
                })
            }
        };
        selected.ok_or_else(|| {
            self.failure(
                ErrorKind::MissingData,
                requested,
                "no indexed presentation timestamp satisfies the requested selection",
            )
        })
    }

    fn displayed_at(&self, requested: Time, inference: DurationInference) -> Result<usize> {
        let mut selected = None;
        let mut unresolved = false;
        for (index, entry) in self
            .entries
            .iter()
            .enumerate()
            .take_while(|(_, entry)| entry.pts <= requested)
        {
            match self.duration(index, inference)? {
                (Some(duration), _) if requested < entry.pts.checked_add(duration)? => {
                    if selected.is_some() {
                        return Err(self.failure(
                            ErrorKind::AmbiguousTiming,
                            requested,
                            "multiple indexed presentation intervals contain the requested time",
                        ));
                    }
                    selected = Some(index);
                }
                (None, _) => unresolved = true,
                _ => {}
            }
        }
        if unresolved {
            return Err(self.failure(ErrorKind::MissingData, requested,
                "an earlier frame has no established end; select an explicit duration-inference policy"));
        }
        selected.ok_or_else(|| {
            self.failure(
                ErrorKind::MissingData,
                requested,
                "requested time lies outside established presentation intervals",
            )
        })
    }

    fn duration(&self, index: usize, inference: DurationInference) -> Result<(Option<Time>, bool)> {
        let entry = &self.entries[index];
        if entry.duration.is_some() || inference == DurationInference::RequireExplicit {
            return Ok((entry.duration, false));
        }
        let next = self
            .entries
            .partition_point(|candidate| candidate.pts <= entry.pts);
        self.entries
            .get(next)
            .map(|next| {
                next.pts
                    .checked_sub(entry.pts)
                    .map(|duration| (Some(duration), true))
            })
            .unwrap_or(Ok((None, false)))
    }

    fn make_selection(
        &self,
        requested: Time,
        index: usize,
        inference: DurationInference,
        work_index: usize,
    ) -> Result<FrameSelection> {
        let entry = &self.entries[index];
        let (duration, duration_inferred) = self.duration(index, inference)?;
        Ok(FrameSelection {
            requested,
            actual: entry.pts,
            duration,
            duration_inferred,
            work_index,
            identity: FrameIdentity {
                source: self.source.clone(),
                track: self.track,
                ordinal: entry.ordinal,
            },
        })
    }

    fn failure(&self, kind: ErrorKind, requested: Time, message: &str) -> Error {
        Error::new(kind, message)
            .with_track(self.track)
            .with_time(requested)
            .with_source(format!("{:?}", self.source.identity))
            .with_operation("select_frame")
    }
}

/// Exact uniform grid of `count` points over a half-open interval.
///
/// Point `i` is `start + i * (end-start) / count`, so the exclusive end is never
/// emitted for a nonempty interval. A zero count or empty interval is rejected.
/// Default SamplingLimits bound request count and planned storage; use
/// uniform_times_with_limits to replace those caps. Representation limits and
/// allocation failure are reported, never approximated.
pub fn uniform_times(range: TimeRange, count: usize) -> Result<Vec<Time>> {
    uniform_times_with_limits(range, count, SamplingLimits::default())
}
/// Exact uniform grid with explicit request/payload bounds; no point equals end.
pub fn uniform_times_with_limits(
    range: TimeRange,
    count: usize,
    limits: SamplingLimits,
) -> Result<Vec<Time>> {
    if count == 0 || range.is_empty() {
        return Err(Error::invalid(
            "uniform sampling requires a nonempty interval and a positive count",
        ));
    }
    let count_u32 = u32::try_from(count).map_err(|_| {
        Error::overflow("uniform sampling count exceeds the rational time denominator")
    })?;
    limits.check(
        count,
        count
            .checked_mul(std::mem::size_of::<Time>())
            .ok_or_else(|| Error::overflow("uniform grid payload size"))?,
    )?;
    let mut times = Vec::new();
    times.try_reserve_exact(count).map_err(|error| {
        Error::exhausted(format!("cannot allocate uniform sampling grid: {error}"))
    })?;
    for index in 0..count_u32 {
        times.push(crate::clip_sampling::interpolate_time(
            range, index, count_u32,
        )?);
    }
    Ok(times)
}

/// Stable inputs for a sample's deterministic pseudorandom draws.
///
/// The fixed SplitMix64-style integer mixer is schedule-independent and defined
/// using wrapping `u64` arithmetic. It is not a cryptographic generator. Selection
/// reproducibility does not imply bit-identical media decoding across backends.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct SampleKey {
    /// Dataset seed.
    pub seed: u64,
    /// Dataset epoch.
    pub epoch: u64,
    /// Stable dataset sample identity, independent of worker assignment.
    pub sample_id: u64,
    /// Identity of the augmentation applied to this sample.
    pub augmentation_id: u64,
}

impl SampleKey {
    /// Construct a schedule-independent sampling key.
    pub const fn new(seed: u64, epoch: u64, sample_id: u64, augmentation_id: u64) -> Self {
        Self {
            seed,
            epoch,
            sample_id,
            augmentation_id,
        }
    }

    /// Draw a reproducible 64-bit pseudorandom value at an explicit counter.
    ///
    /// This integer algorithm is part of the reproducibility contract; changing
    /// it requires a documented sampling-algorithm version change.
    pub fn draw_u64(self, counter: u64) -> u64 {
        let mut state = mix64(self.seed ^ 0x243f_6a88_85a3_08d3);
        state = mix64(state ^ mix64(self.epoch ^ 0x1319_8a2e_0370_7344));
        state = mix64(state ^ mix64(self.sample_id ^ 0xa409_3822_299f_31d0));
        state = mix64(state ^ mix64(self.augmentation_id ^ 0x082e_fa98_ec4e_6c89));
        mix64(state ^ mix64(counter ^ 0x4528_21e6_38d0_1377))
    }
}

/// Deterministic strided dataset shard, keyed only by stable sample identity.
///
/// Shard membership is `sample_id % count == index`; epoch/augmentation changes
/// do not reassign samples. Combined shards cover each input sample exactly once.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DatasetShard {
    index: u32,
    count: u32,
}

impl DatasetShard {
    /// Validate the worker/shard index and positive total shard count.
    pub fn new(index: u32, count: u32) -> Result<Self> {
        if count == 0 || index >= count {
            return Err(Error::invalid(
                "dataset shard requires count > 0 and index < count",
            ));
        }
        Ok(Self { index, count })
    }

    /// This shard's zero-based index.
    pub const fn index(self) -> u32 {
        self.index
    }

    /// Number of disjoint shards.
    pub const fn count(self) -> u32 {
        self.count
    }

    /// Whether a stable sample identity belongs to this shard.
    pub fn contains(self, sample_id: u64) -> bool {
        sample_id % u64::from(self.count) == u64::from(self.index)
    }
}

fn distance_cmp(requested: Time, left: Time, right: Time) -> Ordering {
    // The common requested denominator cancels. Each absolute difference is
    // below 2^96; multiplication by a u32 denominator fits unsigned u128 even at
    // i64::MIN/MAX. No intermediate narrow Time difference is required.
    let distance = |time: Time| {
        (i128::from(requested.numerator()) * i128::from(time.denominator())
            - i128::from(time.numerator()) * i128::from(requested.denominator()))
        .unsigned_abs()
    };
    (distance(left) * u128::from(right.denominator()))
        .cmp(&(distance(right) * u128::from(left.denominator())))
}

fn mix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}
