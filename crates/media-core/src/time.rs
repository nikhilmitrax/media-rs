//! Exact signed media time and explicitly rounded stream timestamps.
//!
//! Times are canonical reduced fractions with a positive denominator. Arithmetic
//! is checked; comparison always uses exact wide integers rather than floats.

use std::cmp::Ordering;
use std::fmt;
use std::ops::Range;

use crate::error::{Error, Result};

/// A signed rational number of seconds with an `i64` numerator and `u32` denominator.
///
/// Use `Option<Time>` for missing timestamps, rather than a sentinel value.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Time {
    numerator: i64,
    denominator: u32,
}

impl Time {
    /// Zero seconds, canonically represented as `0 / 1`.
    pub const ZERO: Self = Self {
        numerator: 0,
        denominator: 1,
    };

    /// Construct a reduced fraction of a second, rejecting zero denominators.
    pub fn new(numerator: i64, denominator: u32) -> Result<Self> {
        if denominator == 0 {
            return Err(Error::invalid("media time denominator must be nonzero"));
        }
        Ok(Self::reduce(numerator, denominator))
    }

    /// Construct exact time from signed milliseconds.
    pub fn from_millis(milliseconds: i64) -> Self {
        Self::reduce(milliseconds, 1_000)
    }

    /// Construct exact sample-grid time at a nonzero sample rate.
    pub fn from_samples(sample: i64, sample_rate: u32) -> Result<Self> {
        Self::new(sample, sample_rate)
    }

    /// Convert finite floating-point seconds to the nearest nanosecond, ties to even.
    ///
    /// This is convenience input. Use [`Self::new`] for exact media timestamps.
    /// The reduced result must fit this type's numerator and denominator.
    pub fn from_seconds(seconds: f64) -> Result<Self> {
        if !seconds.is_finite() {
            return Err(Error::invalid("media time seconds must be finite"));
        }
        let whole = seconds.trunc();
        // i64::MAX rounds up to 2^63 as f64; the upper check is exclusive.
        if whole < i64::MIN as f64 || whole >= 9_223_372_036_854_775_808.0 {
            return Err(Error::overflow(
                "floating-point media time exceeds i64 seconds",
            ));
        }
        let nanos = ((seconds - whole) * 1_000_000_000.0).round_ties_even() as i64;
        Self::from_wide(
            i128::from(whole as i64) * 1_000_000_000 + i128::from(nanos),
            1_000_000_000,
        )
    }

    /// The canonical signed numerator.
    pub const fn numerator(self) -> i64 {
        self.numerator
    }

    /// The canonical positive denominator.
    pub const fn denominator(self) -> u32 {
        self.denominator
    }

    /// Approximate seconds for display or APIs requiring floating-point input.
    pub fn seconds(self) -> f64 {
        self.numerator as f64 / f64::from(self.denominator)
    }

    /// Add times, reducing before checking representation limits.
    pub fn checked_add(self, other: Self) -> Result<Self> {
        Self::from_wide(
            i128::from(self.numerator) * i128::from(other.denominator)
                + i128::from(other.numerator) * i128::from(self.denominator),
            u64::from(self.denominator) * u64::from(other.denominator),
        )
    }

    /// Subtract times, reducing before checking representation limits.
    pub fn checked_sub(self, other: Self) -> Result<Self> {
        Self::from_wide(
            i128::from(self.numerator) * i128::from(other.denominator)
                - i128::from(other.numerator) * i128::from(self.denominator),
            u64::from(self.denominator) * u64::from(other.denominator),
        )
    }

    /// Multiply by a signed rational scale with a nonzero denominator.
    pub fn checked_mul_ratio(self, numerator: i64, denominator: u32) -> Result<Self> {
        if denominator == 0 {
            return Err(Error::invalid("time scale denominator must be nonzero"));
        }
        Self::from_wide(
            i128::from(self.numerator) * i128::from(numerator),
            u64::from(self.denominator) * u64::from(denominator),
        )
    }

    /// Compare exactly using `i128`; all products fit, so comparison is infallible.
    pub fn checked_cmp(self, other: Self) -> Ordering {
        (i128::from(self.numerator) * i128::from(other.denominator))
            .cmp(&(i128::from(other.numerator) * i128::from(self.denominator)))
    }

    /// Convert to integer ticks with an explicitly selected rounding policy.
    pub fn to_ticks(self, time_base: TimeBase, rounding: Rounding) -> Result<i64> {
        let numerator = i128::from(self.numerator) * i128::from(time_base.denominator);
        let denominator = i128::from(self.denominator) * i128::from(time_base.numerator);
        narrow_ticks(div_round(numerator, denominator, rounding))
    }

    fn reduce(numerator: i64, denominator: u32) -> Self {
        let divisor = gcd(
            u128::from(numerator.unsigned_abs()),
            u128::from(denominator),
        );
        Self {
            numerator: numerator / divisor as i64,
            denominator: denominator / divisor as u32,
        }
    }

    fn from_wide(numerator: i128, denominator: u64) -> Result<Self> {
        let divisor = gcd(numerator.unsigned_abs(), u128::from(denominator));
        Ok(Self {
            numerator: i64::try_from(numerator / divisor as i128)
                .map_err(|_| Error::overflow("reduced media time numerator exceeds i64"))?,
            denominator: u32::try_from(u128::from(denominator) / divisor)
                .map_err(|_| Error::overflow("reduced media time denominator exceeds u32"))?,
        })
    }
}

impl Ord for Time {
    fn cmp(&self, other: &Self) -> Ordering {
        self.checked_cmp(*other)
    }
}

impl PartialOrd for Time {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl fmt::Display for Time {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}/{} s", self.numerator, self.denominator)
    }
}

/// The positive rational number of seconds represented by one stream tick.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct TimeBase {
    numerator: u32,
    denominator: u32,
}

impl TimeBase {
    /// Construct a positive time base, rejecting either zero component.
    pub fn new(numerator: u32, denominator: u32) -> Result<Self> {
        if numerator == 0 || denominator == 0 {
            return Err(Error::invalid(
                "time base numerator and denominator must be nonzero",
            ));
        }
        let divisor = gcd(u128::from(numerator), u128::from(denominator)) as u32;
        Ok(Self {
            numerator: numerator / divisor,
            denominator: denominator / divisor,
        })
    }

    /// Numerator of seconds per tick.
    pub const fn numerator(self) -> u32 {
        self.numerator
    }

    /// Denominator of seconds per tick.
    pub const fn denominator(self) -> u32 {
        self.denominator
    }

    /// Convert stream ticks to exact media time when representable.
    pub fn time(self, ticks: i64) -> Result<Time> {
        Time::from_wide(
            i128::from(ticks) * i128::from(self.numerator),
            u64::from(self.denominator),
        )
    }

    /// Rescale ticks directly without requiring an intermediate `Time` to fit.
    pub fn rescale(self, ticks: i64, target: Self, rounding: Rounding) -> Result<i64> {
        let numerator =
            i128::from(ticks) * i128::from(self.numerator) * i128::from(target.denominator);
        let denominator = i128::from(self.denominator) * i128::from(target.numerator);
        narrow_ticks(div_round(numerator, denominator, rounding))
    }
}

/// Rounding policy for conversions from rational time to integer ticks.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Rounding {
    /// Toward negative infinity.
    Floor,
    /// Toward positive infinity.
    Ceil,
    /// Nearest integer, choosing the even integer on an exact tie.
    NearestTiesEven,
    /// Toward zero.
    TowardZero,
}

/// A validated half-open interval `[start, end)` on a media timeline.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct TimeRange {
    start: Time,
    end: Time,
}

impl TimeRange {
    /// Construct a range, permitting emptiness and rejecting reversed bounds.
    pub fn new(start: Time, end: Time) -> Result<Self> {
        if start > end {
            return Err(Error::invalid("media time range end precedes its start"));
        }
        Ok(Self { start, end })
    }

    /// Inclusive beginning.
    pub const fn start(self) -> Time {
        self.start
    }

    /// Exclusive end.
    pub const fn end(self) -> Time {
        self.end
    }

    /// Whether the range contains no time points.
    pub fn is_empty(self) -> bool {
        self.start == self.end
    }

    /// Whether this half-open range contains a time.
    pub fn contains(self, time: Time) -> bool {
        self.start <= time && time < self.end
    }

    /// Exact duration, or an error if the difference exceeds `Time`'s limits.
    pub fn duration(self) -> Result<Time> {
        self.end.checked_sub(self.start)
    }

    /// Sample indices on an explicitly anchored continuous output grid.
    ///
    /// Returns `ceil((start-origin)*rate)..ceil((end-origin)*rate)`. Negative
    /// indices remain negative so callers can apply truncation or padding policy.
    pub fn sample_bounds(self, origin: Time, sample_rate: u32) -> Result<Range<i64>> {
        if sample_rate == 0 {
            return Err(Error::invalid("sample grid rate must be nonzero"));
        }
        Ok(sample_bound(self.start, origin, sample_rate)?
            ..sample_bound(self.end, origin, sample_rate)?)
    }
}

fn sample_bound(time: Time, origin: Time, rate: u32) -> Result<i64> {
    let numerator = i128::from(time.numerator) * i128::from(origin.denominator)
        - i128::from(origin.numerator) * i128::from(time.denominator);
    let denominator = i128::from(time.denominator) * i128::from(origin.denominator);
    // Divide before scaling to avoid overflowing an unreduced numerator.
    let whole = numerator.div_euclid(denominator);
    let remainder = numerator.rem_euclid(denominator);
    let rate = i128::from(rate);
    let whole = whole
        .checked_mul(rate)
        .ok_or_else(|| Error::overflow("sample grid index arithmetic overflowed"))?;
    let fraction = remainder
        .checked_mul(rate)
        .ok_or_else(|| Error::overflow("sample grid fraction arithmetic overflowed"))?;
    let scaled = whole
        .checked_add(div_round(fraction, denominator, Rounding::Ceil))
        .ok_or_else(|| Error::overflow("sample grid index arithmetic overflowed"))?;
    narrow_ticks(scaled)
}

fn narrow_ticks(ticks: i128) -> Result<i64> {
    i64::try_from(ticks).map_err(|_| Error::overflow("media timestamp exceeds i64 ticks"))
}

fn div_round(numerator: i128, denominator: i128, rounding: Rounding) -> i128 {
    let floor = numerator.div_euclid(denominator);
    let remainder = numerator.rem_euclid(denominator);
    match rounding {
        Rounding::TowardZero => numerator / denominator,
        Rounding::Floor => floor,
        Rounding::Ceil => floor + i128::from(remainder != 0),
        Rounding::NearestTiesEven => match (remainder * 2).cmp(&denominator) {
            Ordering::Less => floor,
            Ordering::Greater => floor + 1,
            Ordering::Equal => floor + floor.rem_euclid(2),
        },
    }
}

fn gcd(mut left: u128, mut right: u128) -> u128 {
    while right != 0 {
        (left, right) = (right, left % right);
    }
    left
}
