//! Pure bounded clip schedules on exact caller-declared time grids.
//!
//! These functions do no I/O, decode, seek, padding, or implicit index building.
//! Random algorithm version 1 samples integer start-grid positions with unbiased
//! bounded rejection, deriving each draw from SampleKey and its stable counter.
//! Worker identifiers and mutable worker RNG state never enter the algorithm.

use crate::sampling::{SampleKey, SamplingLimits};
use media_core::{
    error::{Error, ErrorKind, Result},
    time::{Time, TimeRange},
};

/// Exact clip length and discrete random-start resolution.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ClipSampling {
    /// Positive interval length, including a clip ending exactly at coverage end.
    pub duration: Time,
    /// Positive start spacing anchored at the declared coverage start.
    pub start_step: Time,
}
impl ClipSampling {
    /// Declare both duration and random-start resolution; no float jitter is used.
    pub fn new(duration: Time, start_step: Time) -> Result<Self> {
        let result = Self {
            duration,
            start_step,
        };
        result.validate()?;
        Ok(result)
    }
    fn validate(self) -> Result<()> {
        if self.duration <= Time::ZERO || self.start_step <= Time::ZERO {
            return Err(Error::invalid(
                "clip duration and start step must be positive",
            ));
        }
        Ok(())
    }
}

/// Independent draws with replacement from every fitting start-grid position.
/// Duplicate clips are permitted. Each output counter is independent of count,
/// so a longer request preserves a shorter request's prefix under the same key.
pub fn random_clips(
    coverage: TimeRange,
    spec: ClipSampling,
    count: usize,
    key: SampleKey,
    limits: SamplingLimits,
) -> Result<Vec<TimeRange>> {
    spec.validate()?;
    check_ranges(count, limits)?;
    let mut output = allocate(count)?;
    if count == 0 {
        return Ok(output);
    }
    let slots = start_slots(coverage, spec)?;
    for counter in 0..count {
        let index = draw_below(key, counter as u64, slots, 0x7261_6e64_636c_6970)?;
        output.push(clip_at(coverage, spec, index)?);
    }
    Ok(output)
}

/// Draw one clip from each nearly equal, nonempty contiguous start-grid stratum.
/// Requires at least count fitting start positions. Strata cover the complete
/// fitting start domain once; clip intervals themselves may overlap.
pub fn stratified_clips(
    coverage: TimeRange,
    spec: ClipSampling,
    count: usize,
    key: SampleKey,
    limits: SamplingLimits,
) -> Result<Vec<TimeRange>> {
    spec.validate()?;
    check_ranges(count, limits)?;
    let mut output = allocate(count)?;
    if count == 0 {
        return Ok(output);
    }
    let slots = start_slots(coverage, spec)?;
    if count as u128 > u128::from(slots) {
        return Err(Error::new(
            ErrorKind::MissingData,
            "stratified clips need at least one fitting grid start per stratum",
        ));
    }
    for counter in 0..count {
        let low = (counter as u128 * u128::from(slots) / count as u128) as u64;
        let high = ((counter as u128 + 1) * u128::from(slots) / count as u128) as u64;
        let index = low + draw_below(key, counter as u64, high - low, 0x7374_7261_7463_6c69)?;
        output.push(clip_at(coverage, spec, index)?);
    }
    Ok(output)
}

/// Treatment of a final sliding window that exceeds the requested interval.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PartialTail {
    /// Omit every incomplete window.
    Drop,
    /// End an incomplete window at the requested exclusive end and mark it.
    Truncate,
    /// Reject any incomplete window rather than changing its duration.
    Error,
}
/// Response to an interval without established source coverage.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MissingCoverage {
    /// Reject the schedule at the first missing interval.
    Error,
    /// Omit windows with any unestablished coverage.
    Skip,
    /// Retain the requested interval and report established/missing subintervals.
    /// This never synthesizes samples, repeated images, or silence.
    Mark,
}
/// Explicit geometry and boundary semantics for sliding clip requests.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SlidingWindows {
    /// Positive full window duration.
    pub duration: Time,
    /// Positive distance between starts, anchored at requested.start().
    pub hop: Time,
    /// Explicit final partial-window handling.
    pub partial_tail: PartialTail,
    /// Explicit missing-coverage handling.
    pub missing: MissingCoverage,
}
/// One clip interval ready to feed to an asset clip request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ClipWindow {
    /// Actual scheduled half-open interval; a truncated tail ends at request end.
    pub range: TimeRange,
    /// Originally requested full duration, even when the tail was truncated.
    pub nominal_duration: Time,
    /// Whether this window's duration was shortened under explicit tail policy.
    pub partial_tail: bool,
    /// Established portions of range, sorted and disjoint.
    pub coverage: Vec<TimeRange>,
    /// Unestablished portions of range, without invented source data.
    pub missing: Vec<TimeRange>,
}
impl ClipWindow {
    /// Whether the complete returned interval has established coverage.
    pub fn is_valid(&self) -> bool {
        self.missing.is_empty()
    }
}

/// Build bounded sliding windows from caller-supplied established coverage.
///
/// Coverage intervals are normalized as a set; overlap/adjacency does not invent
/// frames or prove unique display selection. Empty coverage means no established
/// data, not that the requested interval is implicitly available. Resource caps
/// include all attempted windows before Skip filtering; dropped tails are
/// excluded arithmetically without enumeration or allocation.
pub fn sliding_clips(
    requested: TimeRange,
    spec: SlidingWindows,
    coverage: &[TimeRange],
    limits: SamplingLimits,
) -> Result<Vec<ClipWindow>> {
    limits.validate()?;
    if spec.duration <= Time::ZERO || spec.hop <= Time::ZERO {
        return Err(Error::invalid("sliding duration and hop must be positive"));
    }
    if requested.is_empty() {
        return Ok(Vec::new());
    }
    let span = Wide::from(requested.end()).sub(Wide::from(requested.start()))?;
    let candidates = match spec.partial_tail {
        PartialTail::Drop => {
            let fitting = span.sub(Wide::from(spec.duration))?;
            if fitting.num < 0 {
                0
            } else {
                fitting
                    .floor_ratio(spec.hop)?
                    .0
                    .checked_add(1)
                    .ok_or_else(|| Error::overflow("sliding window count"))?
            }
        }
        _ => {
            let (whole, remainder) = span.floor_ratio(spec.hop)?;
            whole
                .checked_add(u64::from(remainder))
                .ok_or_else(|| Error::overflow("sliding window count"))?
        }
    };
    let count =
        usize::try_from(candidates).map_err(|_| Error::overflow("sliding request count"))?;
    let output_bytes = count
        .checked_mul(std::mem::size_of::<ClipWindow>())
        .ok_or_else(|| Error::overflow("sliding metadata size"))?;
    limits.check(count, output_bytes)?;
    if count == 0 {
        return Ok(Vec::new());
    }
    if spec.partial_tail == PartialTail::Error {
        let last = grid_time(requested.start(), spec.hop, candidates - 1)?;
        if Wide::from(last)
            .add(Wide::from(spec.duration))?
            .greater_than(Wide::from(requested.end()))?
        {
            return Err(Error::new(
                ErrorKind::MissingData,
                "sliding request has an incomplete final window",
            )
            .with_time(last));
        }
    }
    limits.check(
        coverage.len(),
        coverage
            .len()
            .checked_mul(std::mem::size_of::<TimeRange>())
            .ok_or_else(|| Error::overflow("coverage input size"))?,
    )?;
    let mut planned = output_bytes
        .checked_add(
            coverage
                .len()
                .checked_mul(std::mem::size_of::<TimeRange>() * 2)
                .ok_or_else(|| Error::overflow("coverage normalization size"))?,
        )
        .ok_or_else(|| Error::overflow("sliding payload"))?;
    limits.check(count, planned)?;
    let mut established = allocate(coverage.len())?;
    established.extend_from_slice(coverage);
    established.sort_by_key(|range| range.start());
    let mut normalized: Vec<TimeRange> = allocate(established.len())?;
    for range in established
        .iter()
        .copied()
        .filter(|range| !range.is_empty())
    {
        if let Some(last) = normalized.last_mut()
            && range.start() <= last.end()
        {
            *last = TimeRange::new(last.start(), last.end().max(range.end()))?;
        } else {
            normalized.push(range);
        }
    }
    let mut output = allocate(count)?;
    for index in 0..candidates {
        let start = grid_time(requested.start(), spec.hop, index)?;
        let end = Wide::from(start).add(Wide::from(spec.duration))?;
        let partial = end.greater_than(Wide::from(requested.end()))?;
        let end = if partial {
            requested.end()
        } else {
            end.time()?
        };
        let range = TimeRange::new(start, end)?;
        let first = normalized.partition_point(|interval| interval.end() <= range.start());
        let after = normalized.partition_point(|interval| interval.start() < range.end());
        let relevant = &normalized[first..after];
        let portions = coverage_counts(range, relevant)?;
        if portions.1 > 0 {
            match spec.missing {
                MissingCoverage::Error => {
                    return Err(Error::new(
                        ErrorKind::MissingData,
                        "sliding window lacks established source coverage",
                    )
                    .with_time(start));
                }
                MissingCoverage::Skip => continue,
                MissingCoverage::Mark => {}
            }
        }
        planned = planned
            .checked_add(
                portions
                    .0
                    .checked_add(portions.1)
                    .and_then(|n| n.checked_mul(std::mem::size_of::<TimeRange>()))
                    .ok_or_else(|| Error::overflow("window coverage payload"))?,
            )
            .ok_or_else(|| Error::overflow("sliding payload"))?;
        limits.check(count, planned)?;
        let mut actual = allocate(portions.0)?;
        let mut missing = allocate(portions.1)?;
        let mut cursor = range.start();
        for interval in relevant {
            if let Some(piece) = intersection(range, *interval) {
                if cursor < piece.start() {
                    missing.push(TimeRange::new(cursor, piece.start())?);
                }
                cursor = piece.end();
                actual.push(piece);
            }
        }
        if cursor < range.end() {
            missing.push(TimeRange::new(cursor, range.end())?);
        }
        output.push(ClipWindow {
            range,
            nominal_duration: spec.duration,
            partial_tail: partial,
            coverage: actual,
            missing,
        });
    }
    Ok(output)
}

fn coverage_counts(range: TimeRange, coverage: &[TimeRange]) -> Result<(usize, usize)> {
    let (mut actual, mut missing, mut cursor) = (0, 0, range.start());
    for interval in coverage {
        if let Some(piece) = intersection(range, *interval) {
            actual += 1;
            if cursor < piece.start() {
                missing += 1;
            }
            cursor = piece.end();
        }
    }
    if cursor < range.end() {
        missing += 1;
    }
    Ok((actual, missing))
}
fn intersection(left: TimeRange, right: TimeRange) -> Option<TimeRange> {
    let (start, end) = (left.start().max(right.start()), left.end().min(right.end()));
    (start < end).then(|| TimeRange::new(start, end).expect("ordered intersection"))
}
fn check_ranges(count: usize, limits: SamplingLimits) -> Result<()> {
    limits.check(
        count,
        count
            .checked_mul(std::mem::size_of::<TimeRange>())
            .ok_or_else(|| Error::overflow("clip schedule payload"))?,
    )
}
fn allocate<T>(count: usize) -> Result<Vec<T>> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(count)
        .map_err(|e| Error::exhausted(e.to_string()))?;
    Ok(result)
}
fn start_slots(coverage: TimeRange, spec: ClipSampling) -> Result<u64> {
    let fitting = Wide::from(coverage.end())
        .sub(Wide::from(coverage.start()))?
        .sub(Wide::from(spec.duration))?;
    if fitting.num < 0 {
        return Err(Error::new(
            ErrorKind::MissingData,
            "clip duration exceeds declared continuous coverage",
        ));
    }
    fitting
        .floor_ratio(spec.start_step)?
        .0
        .checked_add(1)
        .ok_or_else(|| Error::overflow("clip start lattice exceeds u64 positions"))
}
fn clip_at(coverage: TimeRange, spec: ClipSampling, index: u64) -> Result<TimeRange> {
    let start = grid_time(coverage.start(), spec.start_step, index)?;
    let end = start.checked_add(spec.duration)?;
    if start < coverage.start() || end > coverage.end() {
        return Err(Error::backend(
            "sampling",
            "exact clip start escaped its declared fitting grid",
        ));
    }
    TimeRange::new(start, end)
}
fn draw_below(key: SampleKey, counter: u64, bound: u64, domain: u64) -> Result<u64> {
    if bound == 0 {
        return Err(Error::invalid("random stratum must not be empty"));
    }
    let derived = SampleKey {
        seed: key.draw_u64(counter),
        augmentation_id: key.augmentation_id ^ domain,
        ..key
    };
    let threshold = bound.wrapping_neg() % bound;
    for attempt in 0..128 {
        let draw = derived.draw_u64(attempt);
        if draw >= threshold {
            return Ok(draw % bound);
        }
    }
    Err(Error::exhausted(
        "bounded unbiased sampling draw exhausted 128 attempts",
    ))
}
fn grid_time(origin: Time, step: Time, index: u64) -> Result<Time> {
    let offset = Wide::new(
        i128::from(step.numerator())
            .checked_mul(i128::from(index))
            .ok_or_else(|| Error::overflow("time grid offset"))?,
        u128::from(step.denominator()),
    );
    Wide::from(origin).add(offset)?.time()
}

// Wide intermediate fractions avoid first narrowing an extreme recording span
// to Time. Final timestamps must still fit Time's exact public representation.
#[derive(Clone, Copy)]
struct Wide {
    num: i128,
    den: u128,
}
impl From<Time> for Wide {
    fn from(time: Time) -> Self {
        Self {
            num: i128::from(time.numerator()),
            den: u128::from(time.denominator()),
        }
    }
}
impl Wide {
    fn new(num: i128, den: u128) -> Self {
        let g = gcd(num.unsigned_abs(), den);
        let magnitude = num.unsigned_abs() / g;
        let numerator = if num >= 0 {
            magnitude as i128
        } else if magnitude == 1_u128 << 127 {
            i128::MIN
        } else {
            -(magnitude as i128)
        };
        Self {
            num: numerator,
            den: den / g,
        }
    }
    fn add(self, other: Self) -> Result<Self> {
        let g = gcd(self.den, other.den);
        let left = other.den / g;
        let right = self.den / g;
        let numerator = self
            .num
            .checked_mul(i128::try_from(left).map_err(|_| Error::overflow("wide time factor"))?)
            .and_then(|n| {
                other
                    .num
                    .checked_mul(i128::try_from(right).ok()?)
                    .and_then(|m| n.checked_add(m))
            })
            .ok_or_else(|| Error::overflow("wide time numerator"))?;
        Ok(Self::new(
            numerator,
            self.den
                .checked_mul(left)
                .ok_or_else(|| Error::overflow("wide time denominator"))?,
        ))
    }
    fn sub(self, other: Self) -> Result<Self> {
        self.add(Self {
            num: other
                .num
                .checked_neg()
                .ok_or_else(|| Error::overflow("wide time negation"))?,
            den: other.den,
        })
    }
    fn floor_ratio(self, step: Time) -> Result<(u64, bool)> {
        if self.num < 0 || step <= Time::ZERO {
            return Err(Error::invalid(
                "grid ratio requires nonnegative span and positive step",
            ));
        }
        let (mut a, mut b, mut c, mut d) = (
            self.num as u128,
            u128::from(step.denominator()),
            self.den,
            step.numerator() as u128,
        );
        let g = gcd(a, d);
        a /= g;
        d /= g;
        let g = gcd(b, c);
        b /= g;
        c /= g;
        let numerator = a
            .checked_mul(b)
            .ok_or_else(|| Error::overflow("grid ratio numerator"))?;
        let denominator = c
            .checked_mul(d)
            .ok_or_else(|| Error::overflow("grid ratio denominator"))?;
        Ok((
            u64::try_from(numerator / denominator)
                .map_err(|_| Error::overflow("time grid exceeds u64 positions"))?,
            numerator % denominator != 0,
        ))
    }
    fn greater_than(self, other: Self) -> Result<bool> {
        Ok(self.sub(other)?.num > 0)
    }
    fn time(self) -> Result<Time> {
        Time::new(
            i64::try_from(self.num)
                .map_err(|_| Error::overflow("scheduled time numerator exceeds i64"))?,
            u32::try_from(self.den)
                .map_err(|_| Error::overflow("scheduled time denominator exceeds u32"))?,
        )
    }
}
pub(crate) fn interpolate_time(range: TimeRange, numerator: u32, denominator: u32) -> Result<Time> {
    if denominator == 0 || numerator > denominator {
        return Err(Error::invalid(
            "time interpolation ratio must lie in zero through one",
        ));
    }
    let left = i128::from(range.start().numerator())
        .checked_mul(i128::from(range.end().denominator()))
        .and_then(|n| n.checked_mul(i128::from(denominator - numerator)))
        .ok_or_else(|| Error::overflow("uniform time numerator"))?;
    let right = i128::from(range.end().numerator())
        .checked_mul(i128::from(range.start().denominator()))
        .and_then(|n| n.checked_mul(i128::from(numerator)))
        .ok_or_else(|| Error::overflow("uniform time numerator"))?;
    let denominator = u128::from(range.start().denominator())
        * u128::from(range.end().denominator())
        * u128::from(denominator);
    Wide::new(
        left.checked_add(right)
            .ok_or_else(|| Error::overflow("uniform time numerator"))?,
        denominator,
    )
    .time()
}
fn gcd(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        let next = a % b;
        a = b;
        b = next;
    }
    a.max(1)
}
