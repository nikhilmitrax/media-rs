//! Exact timing, rescaling, rounding, and continuous sample-grid contracts.

use media_core::time::{Rounding, Time, TimeBase, TimeRange};
use std::collections::HashSet;

fn t(n: i64, d: u32) -> Time {
    Time::new(n, d).unwrap()
}

#[test]
fn canonical_equality_ordering_and_hashes_agree() {
    let equivalent = [t(1, 2), t(2, 4), t(100, 200)];
    assert!(equivalent.windows(2).all(|p| p[0] == p[1]));
    assert_eq!(equivalent.into_iter().collect::<HashSet<_>>().len(), 1);
    assert_eq!(t(0, u32::MAX), Time::ZERO);
    assert_eq!((t(-6, 8).numerator(), t(-6, 8).denominator()), (-3, 4));
    assert!(t(i64::MIN, u32::MAX) < t(i64::MAX, u32::MAX - 1));
    assert!(Time::new(1, 0).is_err());
}

#[test]
fn arithmetic_reduces_before_narrowing_and_rejects_overflow() {
    let max = t(i64::MAX, 1);
    assert_eq!(max.checked_mul_ratio(2, 2).unwrap(), max);
    assert_eq!(max.checked_sub(max).unwrap(), Time::ZERO);
    assert_eq!(t(1, 6).checked_add(t(1, 3)).unwrap(), t(1, 2));
    assert_eq!(t(-1, 6).checked_sub(t(1, 3)).unwrap(), t(-1, 2));
    assert!(max.checked_add(t(1, 1)).is_err());
    assert!(t(i64::MIN, 1).checked_mul_ratio(-1, 1).is_err());
    assert!(t(1, u32::MAX).checked_mul_ratio(1, u32::MAX).is_err());
    assert!(t(1, 1).checked_mul_ratio(0, 0).is_err());
    assert_eq!(t(i64::MIN, 1).checked_mul_ratio(0, 1).unwrap(), Time::ZERO);
}

#[test]
fn rounding_handles_negative_values_and_even_ties() {
    let seconds = TimeBase::new(1, 1).unwrap();
    for (value, floor, ceil, nearest, truncate) in [
        (t(3, 2), 1, 2, 2, 1),
        (t(5, 2), 2, 3, 2, 2),
        (t(-3, 2), -2, -1, -2, -1),
        (t(-5, 2), -3, -2, -2, -2),
        (t(-1, 3), -1, 0, 0, 0),
    ] {
        assert_eq!(value.to_ticks(seconds, Rounding::Floor).unwrap(), floor);
        assert_eq!(value.to_ticks(seconds, Rounding::Ceil).unwrap(), ceil);
        assert_eq!(
            value.to_ticks(seconds, Rounding::NearestTiesEven).unwrap(),
            nearest
        );
        assert_eq!(
            value.to_ticks(seconds, Rounding::TowardZero).unwrap(),
            truncate
        );
    }
}

#[test]
fn timebase_rescaling_does_not_narrow_an_intermediate_time() {
    let mpeg = TimeBase::new(1, 90_000).unwrap();
    let samples = TimeBase::new(1, 48_000).unwrap();
    assert_eq!(mpeg.time(45_000).unwrap(), t(1, 2));
    assert_eq!(
        mpeg.rescale(45_000, samples, Rounding::Floor).unwrap(),
        24_000
    );
    assert_eq!(mpeg.rescale(-1, samples, Rounding::Floor).unwrap(), -1);
    assert_eq!(mpeg.rescale(-1, samples, Rounding::Ceil).unwrap(), 0);
    let large = TimeBase::new(u32::MAX, 1).unwrap();
    assert!(large.time(i64::MAX).is_err());
    assert_eq!(
        large.rescale(i64::MAX, large, Rounding::Ceil).unwrap(),
        i64::MAX
    );
    assert_eq!(
        large.rescale(i64::MIN, large, Rounding::Floor).unwrap(),
        i64::MIN
    );
    assert!(TimeBase::new(0, 1).is_err());
    assert!(TimeBase::new(1, 0).is_err());
    assert_eq!(TimeBase::new(2, 180_000).unwrap(), mpeg);
    assert!(t(i64::MAX, 1).to_ticks(samples, Rounding::Ceil).is_err());
}

#[test]
fn ranges_are_validated_half_open_and_can_be_empty() {
    let range = TimeRange::new(t(-1, 1), t(2, 1)).unwrap();
    assert!(range.contains(t(-1, 1)));
    assert!(range.contains(Time::ZERO));
    assert!(!range.contains(t(2, 1)));
    assert_eq!(range.duration().unwrap(), t(3, 1));
    assert_eq!((range.start(), range.end()), (t(-1, 1), t(2, 1)));
    let empty = TimeRange::new(t(1, 2), t(1, 2)).unwrap();
    assert!(empty.is_empty());
    assert!(!empty.contains(t(1, 2)));
    assert!(TimeRange::new(t(2, 1), t(1, 1)).is_err());
    assert!(
        TimeRange::new(t(i64::MIN, 1), t(i64::MAX, 1))
            .unwrap()
            .duration()
            .is_err()
    );
}

#[test]
fn adjacent_audio_queries_share_a_continuous_sample_grid() {
    let origin = t(1, 7);
    let middle = origin.checked_add(t(13, 100_000)).unwrap();
    let end = origin.checked_add(t(31, 100_000)).unwrap();
    let left = TimeRange::new(origin, middle).unwrap();
    let right = TimeRange::new(middle, end).unwrap();
    let whole = TimeRange::new(origin, end).unwrap();
    assert_eq!(left.sample_bounds(origin, 16_000).unwrap(), 0..3);
    assert_eq!(right.sample_bounds(origin, 16_000).unwrap(), 3..5);
    assert_eq!(whole.sample_bounds(origin, 16_000).unwrap(), 0..5);
    assert!(whole.sample_bounds(origin, 0).is_err());
}

#[test]
fn sample_grids_preserve_negative_indices_and_fractional_bounds() {
    let range = TimeRange::new(t(-7, 10), t(-1, 10)).unwrap();
    assert_eq!(range.sample_bounds(Time::ZERO, 4).unwrap(), -2..0);
    assert_eq!(
        TimeRange::new(t(1, 3), t(1, 3))
            .unwrap()
            .sample_bounds(Time::ZERO, 10)
            .unwrap(),
        4..4
    );
    assert_eq!(Time::from_samples(-4, 8).unwrap(), t(-1, 2));
    assert!(Time::from_samples(1, 0).is_err());
    let end = t(i64::MAX, u32::MAX);
    let close = t(i64::MAX - 1, u32::MAX);
    assert_eq!(
        TimeRange::new(close, end)
            .unwrap()
            .sample_bounds(close, u32::MAX)
            .unwrap(),
        0..1
    );
    assert!(
        TimeRange::new(t(i64::MAX, 1), t(i64::MAX, 1))
            .unwrap()
            .sample_bounds(t(i64::MIN, 1), 1)
            .is_err()
    );
}

#[test]
fn floating_input_is_finite_quantized_and_range_checked() {
    assert_eq!(Time::from_seconds(0.125).unwrap(), t(1, 8));
    assert_eq!(Time::from_seconds(-0.125).unwrap(), t(-1, 8));
    assert_eq!(Time::from_seconds(0.000_000_000_4).unwrap(), Time::ZERO);
    assert_eq!(Time::from_seconds(0.000_000_000_5).unwrap(), Time::ZERO);
    assert_eq!(
        Time::from_seconds(0.000_000_001_5).unwrap(),
        t(2, 1_000_000_000)
    );
    assert_eq!(
        Time::from_seconds(10_000_000_000.0).unwrap(),
        t(10_000_000_000, 1)
    );
    assert_eq!(Time::from_seconds(i64::MIN as f64).unwrap(), t(i64::MIN, 1));
    assert!(Time::from_seconds(i64::MAX as f64).is_err());
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(Time::from_seconds(value).is_err());
    }
    assert_eq!(Time::from_millis(i64::MIN).numerator(), i64::MIN / 8);
    assert_eq!(t(1, 8).seconds(), 0.125);
    assert_eq!(t(-1, 2).to_string(), "-1/2 s");
}
