//! Exact ordinal/keyframe evidence, deterministic clips, and bounded window schedules.
use media_core::{
    error::ErrorKind,
    media::FrameTiming,
    time::{Time, TimeRange},
};
use media_io::source::Source;
use media_pipeline::{
    clip_sampling::{
        ClipSampling, MissingCoverage, PartialTail, SlidingWindows, random_clips, sliding_clips,
        stratified_clips,
    },
    sampling::{
        DatasetShard, FrameIndex, FrameIndexEntry, SampleKey, SamplingLimits, uniform_times,
        uniform_times_with_limits,
    },
};

fn s(value: i64) -> Time {
    Time::new(value, 1).unwrap()
}
fn range(start: i64, end: i64) -> TimeRange {
    TimeRange::new(s(start), s(end)).unwrap()
}
fn index() -> FrameIndex {
    FrameIndex::new(
        Source::memory(vec![1, 2, 3]).fingerprint().unwrap(),
        7,
        vec![
            FrameIndexEntry {
                pts: s(2),
                duration: None,
                ordinal: 900,
                keyframe: false,
            },
            FrameIndexEntry {
                pts: s(0),
                duration: None,
                ordinal: 100,
                keyframe: true,
            },
            FrameIndexEntry {
                pts: s(1),
                duration: None,
                ordinal: 500,
                keyframe: false,
            },
            FrameIndexEntry {
                pts: s(3),
                duration: None,
                ordinal: 1000,
                keyframe: true,
            },
            FrameIndexEntry {
                pts: s(4),
                duration: None,
                ordinal: u64::MAX,
                keyframe: true,
            },
        ],
    )
    .unwrap()
}

#[test]
fn ordinal_queries_preserve_actual_identity_instead_of_index_positions_or_fake_times() {
    let index = index();
    assert_eq!(index.select_ordinal(500).unwrap().frame.pts, s(1));
    assert_eq!(
        index.select_ordinal(0).unwrap_err().kind(),
        ErrorKind::MissingData
    );
    assert_eq!(
        index.select_ordinal(u64::MAX).unwrap().identity.ordinal,
        u64::MAX
    );
    let batch = index
        .select_ordinal_batch(&[900, 100, 900, 500], SamplingLimits::default())
        .unwrap();
    assert_eq!(
        batch
            .selections
            .iter()
            .map(|v| v.requested_ordinal)
            .collect::<Vec<_>>(),
        vec![900, 100, 900, 500]
    );
    assert_eq!(
        batch
            .unique_frames
            .iter()
            .map(|v| v.ordinal)
            .collect::<Vec<_>>(),
        vec![100, 500, 900]
    );
    assert_eq!(
        batch
            .selections
            .iter()
            .map(|v| v.work_index)
            .collect::<Vec<_>>(),
        vec![2, 0, 2, 1]
    );
    assert_eq!(batch.selections[0].identity, batch.selections[2].identity);
    assert_eq!(batch.selections[0].identity.track, 7);
    assert!(
        index
            .select_ordinal_batch(&[], SamplingLimits::default())
            .unwrap()
            .selections
            .is_empty()
    );
    assert!(
        index
            .select_ordinal_batch(&[100, 101], SamplingLimits::default())
            .is_err()
    );
}

#[test]
fn navigation_without_verified_global_ordinals_cannot_create_ordinal_index_entries() {
    let after_seek = FrameTiming {
        pts: Some(s(2)),
        duration: Some(s(1)),
        ordinal: None,
        ..Default::default()
    };
    assert_eq!(
        FrameIndexEntry::from_timing(after_seek, false)
            .unwrap_err()
            .kind(),
        ErrorKind::Unsupported
    );
    let missing_pts = FrameTiming {
        ordinal: Some(10),
        ..Default::default()
    };
    assert_eq!(
        FrameIndexEntry::from_timing(missing_pts, false)
            .unwrap_err()
            .kind(),
        ErrorKind::MissingData
    );
    let known = FrameIndexEntry::from_timing(
        FrameTiming {
            ordinal: Some(900),
            ..after_seek
        },
        true,
    )
    .unwrap();
    assert_eq!(
        (known.ordinal, known.pts, known.keyframe),
        (900, s(2), true)
    );
}

#[test]
fn keyframes_require_verified_flags_but_not_invented_display_durations() {
    let index = index();
    assert!(!index.keyframe_flags_verified());
    assert_eq!(
        index
            .keyframes_in(range(0, 5), SamplingLimits::default())
            .unwrap_err()
            .kind(),
        ErrorKind::Unsupported
    );
    let index = index.with_verified_keyframe_flags();
    assert!(index.keyframe_flags_verified());
    let frames = index
        .keyframes_in(range(0, 4), SamplingLimits::default())
        .unwrap();
    assert_eq!(
        frames
            .frames
            .iter()
            .map(|v| v.frame.ordinal)
            .collect::<Vec<_>>(),
        vec![100, 1000]
    );
    assert_eq!(frames.requested, range(0, 4));
    assert!(frames.frames.iter().all(|v| v.frame.duration.is_none()));
    assert!(
        index
            .keyframes_in(range(3, 3), SamplingLimits::default())
            .unwrap()
            .frames
            .is_empty()
    );
    assert!(
        index
            .keyframes_in(range(5, 10), SamplingLimits::default())
            .unwrap()
            .frames
            .is_empty()
    );
    assert!(
        index
            .keyframes_in(
                range(0, 5),
                SamplingLimits {
                    max_requests: 1,
                    ..Default::default()
                }
            )
            .is_err()
    );
}

#[test]
fn selection_counts_and_identity_payloads_are_checked_before_batch_allocation() {
    let index = index();
    assert!(
        index
            .select_batch_with_limits(
                &[s(0), s(1)],
                media_pipeline::sampling::TemporalSelection::Nearest,
                media_pipeline::sampling::DurationInference::RequireExplicit,
                SamplingLimits {
                    max_requests: 1,
                    ..Default::default()
                }
            )
            .is_err()
    );
    assert_eq!(
        index
            .select_ordinal_batch(
                &[100, 500],
                SamplingLimits {
                    max_requests: 1,
                    ..Default::default()
                }
            )
            .unwrap_err()
            .kind(),
        ErrorKind::ResourceExhausted
    );
    assert!(
        index
            .select_ordinal_batch(
                &[100],
                SamplingLimits {
                    max_bytes: 1,
                    ..Default::default()
                }
            )
            .is_err()
    );
    assert!(uniform_times(range(0, 1), 65537).is_err());
    assert!(
        uniform_times_with_limits(
            range(0, 1),
            4,
            SamplingLimits {
                max_requests: 3,
                ..Default::default()
            }
        )
        .is_err()
    );
    assert!(
        uniform_times_with_limits(
            range(0, 1),
            4,
            SamplingLimits {
                max_bytes: 1,
                ..Default::default()
            }
        )
        .is_err()
    );
}

#[test]
fn uniform_and_random_schedules_handle_extreme_spans_without_narrow_duration_intermediates() {
    let extremes = TimeRange::new(s(i64::MIN), s(i64::MAX)).unwrap();
    assert_eq!(
        uniform_times(extremes, 2).unwrap(),
        vec![s(i64::MIN), Time::new(-1, 2).unwrap()]
    );
    let clips = random_clips(
        extremes,
        ClipSampling::new(s(1), s(1)).unwrap(),
        8,
        SampleKey::new(5, 2, 42, 9),
        SamplingLimits::default(),
    )
    .unwrap();
    for clip in clips {
        assert!(clip.start() >= extremes.start());
        assert!(clip.end() <= extremes.end());
        assert_eq!(clip.duration().unwrap(), s(1));
    }
    assert_eq!(
        random_clips(
            extremes,
            ClipSampling::new(s(1), Time::new(1, u32::MAX).unwrap()).unwrap(),
            1,
            SampleKey::new(0, 0, 0, 0),
            SamplingLimits::default()
        )
        .unwrap_err()
        .kind(),
        ErrorKind::Overflow
    );
}

#[test]
fn random_algorithm_v1_has_a_stable_exact_start_grid_and_count_independent_prefix() {
    let spec = ClipSampling::new(s(1), Time::new(1, 4).unwrap()).unwrap();
    let key = SampleKey::new(5, 2, 42, 9);
    let clips = random_clips(range(0, 5), spec, 8, key, SamplingLimits::default()).unwrap();
    assert_eq!(
        clips.iter().map(|clip| clip.start()).collect::<Vec<_>>(),
        [750, 2750, 3500, 1750, 3750, 3000, 3250, 3000].map(Time::from_millis)
    );
    assert_eq!(
        random_clips(range(0, 5), spec, 3, key, SamplingLimits::default()).unwrap(),
        clips[..3]
    );
    for clip in clips {
        assert!(clip.start() >= s(0) && clip.end() <= s(5));
        assert_eq!(clip.duration().unwrap(), s(1));
    }
}

#[test]
fn deterministic_clip_schedules_survive_worker_sharding_reordering_and_seed_changes() {
    let coverage = TimeRange::new(Time::new(1, 3).unwrap(), Time::new(10, 3).unwrap()).unwrap();
    let spec = ClipSampling::new(Time::new(1, 2).unwrap(), Time::new(1, 12).unwrap()).unwrap();
    let sample = |id| {
        random_clips(
            coverage,
            spec,
            5,
            SampleKey::new(123, 4, id, 9),
            SamplingLimits::default(),
        )
        .unwrap()
    };
    let expected = (0..100).map(|id| (id, sample(id))).collect::<Vec<_>>();
    for workers in [1, 2, 7, 101] {
        let mut observed = Vec::new();
        for worker in (0..workers).rev() {
            let shard = DatasetShard::new(worker, workers).unwrap();
            for id in (0..100).rev().filter(|&id| shard.contains(id)) {
                observed.push((id, sample(id)));
            }
        }
        observed.sort_by_key(|v| v.0);
        assert_eq!(observed, expected);
    }
    let base = sample(42);
    for key in [
        SampleKey::new(124, 4, 42, 9),
        SampleKey::new(123, 5, 42, 9),
        SampleKey::new(123, 4, 42, 10),
    ] {
        assert_ne!(
            random_clips(coverage, spec, 5, key, SamplingLimits::default()).unwrap(),
            base
        );
    }
}

#[test]
fn stratified_clips_cover_nonempty_start_strata_and_do_not_fake_extra_strata() {
    let clips = stratified_clips(
        range(0, 5),
        ClipSampling::new(s(1), Time::new(1, 4).unwrap()).unwrap(),
        4,
        SampleKey::new(1, 2, 3, 4),
        SamplingLimits::default(),
    )
    .unwrap();
    for (clip, (low, high)) in clips.iter().zip([(0, 4), (4, 8), (8, 12), (12, 17)]) {
        let index = clip.start().checked_mul_ratio(4, 1).unwrap();
        assert!(index >= s(low) && index < s(high));
        assert_eq!(clip.duration().unwrap(), s(1));
    }
    assert_eq!(
        stratified_clips(
            range(0, 5),
            ClipSampling::new(s(1), Time::new(1, 4).unwrap()).unwrap(),
            18,
            SampleKey::new(0, 0, 0, 0),
            SamplingLimits::default()
        )
        .unwrap_err()
        .kind(),
        ErrorKind::MissingData
    );
    let one = random_clips(
        range(0, 1),
        ClipSampling::new(s(1), s(1)).unwrap(),
        3,
        SampleKey::new(0, 0, 0, 0),
        SamplingLimits::default(),
    )
    .unwrap();
    assert_eq!(one, vec![range(0, 1); 3]);
}

fn windows(tail: PartialTail, missing: MissingCoverage) -> SlidingWindows {
    SlidingWindows {
        duration: s(4),
        hop: s(2),
        partial_tail: tail,
        missing,
    }
}

#[test]
fn sliding_windows_report_actual_coverage_missing_intervals_and_partial_tail_separately() {
    let coverage = [range(6, 10), range(0, 2), range(1, 3)];
    let result = sliding_clips(
        range(0, 10),
        windows(PartialTail::Truncate, MissingCoverage::Mark),
        &coverage,
        SamplingLimits::default(),
    )
    .unwrap();
    assert_eq!(result.len(), 5);
    assert_eq!(result[0].range, range(0, 4));
    assert_eq!(result[0].coverage, vec![range(0, 3)]);
    assert_eq!(result[0].missing, vec![range(3, 4)]);
    assert_eq!(result[1].missing, vec![range(3, 6)]);
    assert_eq!(result[2].coverage, vec![range(6, 8)]);
    assert_eq!(result[2].missing, vec![range(4, 6)]);
    assert!(result[3].is_valid());
    assert_eq!(result[3].range, range(6, 10));
    assert!(result[4].partial_tail);
    assert!(result[4].is_valid());
    assert_eq!(result[4].range, range(8, 10));
    assert_eq!(result[4].nominal_duration, s(4));
    let skipped = sliding_clips(
        range(0, 10),
        windows(PartialTail::Truncate, MissingCoverage::Skip),
        &coverage,
        SamplingLimits::default(),
    )
    .unwrap();
    assert_eq!(
        skipped.iter().map(|v| v.range).collect::<Vec<_>>(),
        vec![range(6, 10), range(8, 10)]
    );
    assert_eq!(
        sliding_clips(
            range(0, 10),
            windows(PartialTail::Truncate, MissingCoverage::Error),
            &coverage,
            SamplingLimits::default()
        )
        .unwrap_err()
        .kind(),
        ErrorKind::MissingData
    );
}

#[test]
fn rational_sliding_windows_have_exact_starts_and_explicit_drop_or_error_tails() {
    let range = TimeRange::new(Time::new(1, 3).unwrap(), Time::new(5, 3).unwrap()).unwrap();
    let spec = SlidingWindows {
        duration: Time::new(2, 3).unwrap(),
        hop: Time::new(1, 3).unwrap(),
        partial_tail: PartialTail::Truncate,
        missing: MissingCoverage::Error,
    };
    let windows = sliding_clips(range, spec, &[range], SamplingLimits::default()).unwrap();
    assert_eq!(
        windows.iter().map(|v| v.range.start()).collect::<Vec<_>>(),
        [
            Time::new(1, 3).unwrap(),
            Time::new(2, 3).unwrap(),
            s(1),
            Time::new(4, 3).unwrap()
        ]
    );
    assert_eq!(windows[3].range.end(), range.end());
    assert!(windows[3].partial_tail);
    assert_eq!(
        sliding_clips(
            range,
            SlidingWindows {
                partial_tail: PartialTail::Drop,
                ..spec
            },
            &[range],
            SamplingLimits::default()
        )
        .unwrap()
        .len(),
        3
    );
    assert_eq!(
        sliding_clips(
            range,
            SlidingWindows {
                partial_tail: PartialTail::Error,
                ..spec
            },
            &[range],
            SamplingLimits::default()
        )
        .unwrap_err()
        .kind(),
        ErrorKind::MissingData
    );
}

#[test]
fn truncated_extreme_tail_does_not_require_representing_an_out_of_range_nominal_end() {
    let range = TimeRange::new(s(i64::MAX - 1), s(i64::MAX)).unwrap();
    let spec = SlidingWindows {
        duration: s(1),
        hop: Time::new(1, 2).unwrap(),
        partial_tail: PartialTail::Truncate,
        missing: MissingCoverage::Error,
    };
    // The half-second start itself cannot be represented by Time near i64::MAX;
    // that exact representation failure is reported instead of rounded.
    assert_eq!(
        sliding_clips(range, spec, &[range], SamplingLimits::default())
            .unwrap_err()
            .kind(),
        ErrorKind::Overflow
    );
    let spec = SlidingWindows {
        duration: s(2),
        hop: s(1),
        ..spec
    };
    let only = sliding_clips(range, spec, &[range], SamplingLimits::default()).unwrap();
    assert_eq!(only.len(), 1);
    assert_eq!(only[0].range, range);
    assert!(only[0].partial_tail);
}

#[test]
fn empty_invalid_missing_and_excessive_requests_fail_without_large_preallocation() {
    let key = SampleKey::new(0, 0, 0, 0);
    let spec = ClipSampling::new(s(1), s(1)).unwrap();
    assert!(
        random_clips(range(0, 0), spec, 0, key, SamplingLimits::default())
            .unwrap()
            .is_empty()
    );
    assert!(random_clips(range(0, 0), spec, 1, key, SamplingLimits::default()).is_err());
    assert!(ClipSampling::new(s(0), s(1)).is_err());
    assert!(ClipSampling::new(s(1), s(-1)).is_err());
    assert!(random_clips(range(0, 10), spec, 65537, key, SamplingLimits::default()).is_err());
    assert!(
        random_clips(
            range(0, 10),
            spec,
            1,
            key,
            SamplingLimits {
                max_bytes: 1,
                ..Default::default()
            }
        )
        .is_err()
    );
    assert!(
        sliding_clips(
            range(0, 0),
            windows(PartialTail::Drop, MissingCoverage::Error),
            &[],
            SamplingLimits::default()
        )
        .unwrap()
        .is_empty()
    );
    assert!(
        sliding_clips(
            range(0, 0),
            SlidingWindows {
                duration: Time::ZERO,
                ..windows(PartialTail::Drop, MissingCoverage::Error)
            },
            &[],
            SamplingLimits::default()
        )
        .is_err()
    );
    assert!(
        sliding_clips(
            range(0, 1_000_000),
            SlidingWindows {
                duration: s(1),
                hop: s(1),
                ..windows(PartialTail::Drop, MissingCoverage::Error)
            },
            &[],
            SamplingLimits::default()
        )
        .is_err()
    );
    let marked = sliding_clips(
        range(0, 4),
        windows(PartialTail::Drop, MissingCoverage::Mark),
        &[],
        SamplingLimits::default(),
    )
    .unwrap();
    assert_eq!(marked[0].missing, vec![range(0, 4)]);
    assert!(marked[0].coverage.is_empty());
    let request = TimeRange::new(Time::ZERO, Time::new(3, 65539).unwrap()).unwrap();
    let too_fine = SlidingWindows {
        duration: Time::new(1, 65537).unwrap(),
        hop: Time::new(1, 65539).unwrap(),
        partial_tail: PartialTail::Truncate,
        missing: MissingCoverage::Error,
    };
    assert_eq!(
        sliding_clips(request, too_fine, &[request], SamplingLimits::default())
            .unwrap_err()
            .kind(),
        ErrorKind::Overflow
    );
}
