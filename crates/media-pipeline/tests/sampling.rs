//! Deterministic VFR selection, interval evidence, batching, and dataset sampling.

use media_core::error::ErrorKind;
use media_core::time::{Time, TimeRange};
use media_io::source::Source;
use media_pipeline::sampling::{
    DatasetShard, DurationInference, FrameIndex, FrameIndexEntry, SampleKey, TemporalSelection,
    uniform_times,
};

fn ms(value: i64) -> Time {
    Time::from_millis(value)
}

fn entry(pts: i64, duration: Option<i64>, ordinal: u64) -> FrameIndexEntry {
    FrameIndexEntry {
        pts: ms(pts),
        duration: duration.map(ms),
        ordinal,
        keyframe: ordinal == 0,
    }
}

fn index(entries: Vec<FrameIndexEntry>) -> FrameIndex {
    FrameIndex::new(
        Source::memory(vec![1, 2, 3]).fingerprint().unwrap(),
        7,
        entries,
    )
    .unwrap()
}

fn vfr() -> FrameIndex {
    index(vec![
        entry(0, Some(40), 0),
        entry(40, Some(60), 1),
        entry(100, Some(40), 2),
        entry(140, Some(60), 3),
    ])
}

#[test]
fn design_vfr_selection_uses_display_intervals_and_earlier_pts_ties() {
    let index = vfr();
    for (request, expected) in [(0, 0), (39, 0), (40, 40), (75, 40), (100, 100), (199, 140)] {
        let selected = index
            .select(
                ms(request),
                TemporalSelection::DisplayedAt,
                DurationInference::RequireExplicit,
            )
            .unwrap();
        assert_eq!(selected.requested, ms(request));
        assert_eq!(selected.actual, ms(expected));
        assert!(!selected.duration_inferred);
        assert_eq!(selected.identity.track, 7);
    }
    assert_eq!(
        index
            .select(
                ms(70),
                TemporalSelection::Nearest,
                DurationInference::RequireExplicit
            )
            .unwrap()
            .actual,
        ms(40)
    );
    assert_eq!(
        index
            .select(
                ms(99),
                TemporalSelection::AtOrBefore,
                DurationInference::RequireExplicit
            )
            .unwrap()
            .actual,
        ms(40)
    );
    assert_eq!(
        index
            .select(
                ms(41),
                TemporalSelection::AtOrAfter,
                DurationInference::RequireExplicit
            )
            .unwrap()
            .actual,
        ms(100)
    );
    assert_eq!(
        index
            .select(
                ms(200),
                TemporalSelection::DisplayedAt,
                DurationInference::RequireExplicit
            )
            .unwrap_err()
            .kind(),
        ErrorKind::MissingData
    );
}

#[test]
fn missing_coverage_and_unknown_durations_do_not_silently_hold_a_frame() {
    let gap = index(vec![entry(0, Some(40), 0), entry(100, Some(40), 1)]);
    assert_eq!(
        gap.select(
            ms(50),
            TemporalSelection::DisplayedAt,
            DurationInference::RequireExplicit
        )
        .unwrap_err()
        .kind(),
        ErrorKind::MissingData
    );
    assert!(
        gap.select(
            ms(-1),
            TemporalSelection::AtOrBefore,
            DurationInference::RequireExplicit
        )
        .is_err()
    );
    assert!(
        gap.select(
            ms(101),
            TemporalSelection::AtOrAfter,
            DurationInference::RequireExplicit
        )
        .is_err()
    );
    let unknown = index(vec![
        entry(0, None, 0),
        entry(40, Some(60), 1),
        entry(100, None, 2),
    ]);
    let error = unknown
        .select(
            ms(75),
            TemporalSelection::DisplayedAt,
            DurationInference::RequireExplicit,
        )
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::MissingData);
    assert_eq!(error.context().time, Some(ms(75)));
    assert_eq!(error.context().track, Some(7));
    assert_eq!(
        unknown
            .select(
                ms(75),
                TemporalSelection::DisplayedAt,
                DurationInference::NextTimestamp
            )
            .unwrap()
            .actual,
        ms(40)
    );
    let inferred = unknown
        .select(
            ms(20),
            TemporalSelection::DisplayedAt,
            DurationInference::NextTimestamp,
        )
        .unwrap();
    assert_eq!(inferred.duration, Some(ms(40)));
    assert!(inferred.duration_inferred);
    assert!(
        unknown
            .select(
                ms(100),
                TemporalSelection::DisplayedAt,
                DurationInference::NextTimestamp
            )
            .is_err()
    );
    assert_eq!(
        unknown
            .select(
                ms(100),
                TemporalSelection::Nearest,
                DurationInference::RequireExplicit
            )
            .unwrap()
            .duration,
        None
    );
}

#[test]
fn overlapping_and_duplicate_intervals_are_ambiguous_while_pts_ties_are_stable() {
    let overlap = index(vec![entry(0, Some(100), 0), entry(40, Some(80), 1)]);
    assert_eq!(
        overlap
            .select(
                ms(50),
                TemporalSelection::DisplayedAt,
                DurationInference::RequireExplicit
            )
            .unwrap_err()
            .kind(),
        ErrorKind::AmbiguousTiming
    );
    let duplicates = index(vec![
        entry(0, Some(40), 9),
        entry(0, Some(40), 3),
        entry(40, Some(40), 10),
    ]);
    assert_eq!(
        duplicates
            .select(
                ms(0),
                TemporalSelection::DisplayedAt,
                DurationInference::RequireExplicit
            )
            .unwrap_err()
            .kind(),
        ErrorKind::AmbiguousTiming
    );
    for policy in [
        TemporalSelection::Nearest,
        TemporalSelection::AtOrBefore,
        TemporalSelection::AtOrAfter,
    ] {
        assert_eq!(
            duplicates
                .select(ms(0), policy, DurationInference::RequireExplicit)
                .unwrap()
                .identity
                .ordinal,
            3
        );
    }
    let inferred_duplicates = index(vec![
        entry(0, None, 9),
        entry(0, None, 3),
        entry(40, Some(40), 10),
    ]);
    assert_eq!(
        inferred_duplicates
            .select(
                ms(20),
                TemporalSelection::DisplayedAt,
                DurationInference::NextTimestamp
            )
            .unwrap_err()
            .kind(),
        ErrorKind::AmbiguousTiming
    );
}

#[test]
fn unordered_repeated_queries_restore_caller_order_and_share_source_work() {
    let index = vfr();
    let requests = [ms(145), ms(1), ms(150), ms(75), ms(2)];
    let batch = index
        .select_batch(
            &requests,
            TemporalSelection::DisplayedAt,
            DurationInference::RequireExplicit,
        )
        .unwrap();
    assert_eq!(
        batch
            .selections
            .iter()
            .map(|selection| selection.requested)
            .collect::<Vec<_>>(),
        requests
    );
    assert_eq!(
        batch
            .selections
            .iter()
            .map(|selection| selection.actual)
            .collect::<Vec<_>>(),
        [ms(140), ms(0), ms(140), ms(40), ms(0)]
    );
    assert_eq!(
        batch
            .unique_frames
            .iter()
            .map(|frame| frame.ordinal)
            .collect::<Vec<_>>(),
        [0, 1, 3]
    );
    assert_eq!(
        batch
            .selections
            .iter()
            .map(|selection| selection.work_index)
            .collect::<Vec<_>>(),
        [2, 0, 2, 1, 0]
    );
    assert_eq!(batch.selections[0].identity, batch.selections[2].identity);
    assert_eq!(batch.selections[1].identity, batch.selections[4].identity);
    assert_eq!(batch.selections[0].identity.source, *index.source());
    assert_eq!(index.track(), 7);
}

#[test]
fn indexes_validate_timing_and_reuse_source_version_evidence() {
    let source = Source::memory(vec![1, 2, 3]);
    let duplicate_ordinals = vec![entry(0, Some(40), 0), entry(40, Some(40), 0)];
    assert!(FrameIndex::new(source.fingerprint().unwrap(), 0, duplicate_ordinals).is_err());
    for invalid in [entry(0, Some(0), 0), entry(0, Some(-1), 0)] {
        assert!(FrameIndex::new(source.fingerprint().unwrap(), 0, vec![invalid]).is_err());
    }
    let ordered = index(vec![entry(140, Some(60), 3), entry(0, Some(40), 0)]);
    assert_eq!(ordered.entries()[0].ordinal, 0);
    ordered.validate_source(&source).unwrap();
    assert_eq!(
        ordered
            .validate_source(&Source::memory(vec![1, 2, 4]))
            .unwrap_err()
            .kind(),
        ErrorKind::Incompatible
    );
    let empty = index(vec![]);
    assert!(
        empty
            .select(
                ms(0),
                TemporalSelection::Nearest,
                DurationInference::RequireExplicit
            )
            .is_err()
    );
    assert!(
        empty
            .select_batch(
                &[],
                TemporalSelection::DisplayedAt,
                DurationInference::RequireExplicit
            )
            .unwrap()
            .selections
            .is_empty()
    );
}

#[test]
fn nearest_comparison_handles_extreme_timestamps_without_narrow_differences() {
    let frames = index(vec![
        FrameIndexEntry {
            pts: Time::new(i64::MIN, 1).unwrap(),
            duration: None,
            ordinal: 0,
            keyframe: true,
        },
        FrameIndexEntry {
            pts: Time::new(i64::MAX, 1).unwrap(),
            duration: None,
            ordinal: 1,
            keyframe: true,
        },
    ]);
    assert_eq!(
        frames
            .select(
                Time::ZERO,
                TemporalSelection::Nearest,
                DurationInference::RequireExplicit
            )
            .unwrap()
            .identity
            .ordinal,
        1
    );
    let frames = index(vec![
        FrameIndexEntry {
            pts: Time::new(i64::MIN, u32::MAX).unwrap(),
            duration: None,
            ordinal: 0,
            keyframe: true,
        },
        FrameIndexEntry {
            pts: Time::new(i64::MAX, u32::MAX - 1).unwrap(),
            duration: None,
            ordinal: 1,
            keyframe: true,
        },
    ]);
    assert_eq!(
        frames
            .select(
                Time::new(i64::MAX, u32::MAX).unwrap(),
                TemporalSelection::Nearest,
                DurationInference::RequireExplicit
            )
            .unwrap()
            .identity
            .ordinal,
        1
    );
}

#[test]
fn uniform_grid_is_exact_and_half_open() {
    let range = TimeRange::new(Time::new(1, 3).unwrap(), Time::new(2, 3).unwrap()).unwrap();
    let points = uniform_times(range, 3).unwrap();
    assert_eq!(
        points,
        [
            Time::new(1, 3).unwrap(),
            Time::new(4, 9).unwrap(),
            Time::new(5, 9).unwrap()
        ]
    );
    assert!(points.into_iter().all(|point| range.contains(point)));
    assert_eq!(uniform_times(range, 1).unwrap(), [range.start()]);
    assert!(uniform_times(range, 0).is_err());
    assert!(uniform_times(TimeRange::new(ms(1), ms(1)).unwrap(), 3).is_err());
    #[cfg(target_pointer_width = "64")]
    assert!(uniform_times(range, u32::MAX as usize + 1).is_err());
}

#[test]
fn dataset_selection_is_identical_under_different_worker_counts_and_orders() {
    let expected: Vec<_> = (0..100)
        .map(|id| (id, SampleKey::new(123, 4, id, 9).draw_u64(0)))
        .collect();
    for count in [1, 2, 7, 101] {
        let mut observed = Vec::new();
        for worker in (0..count).rev() {
            let shard = DatasetShard::new(worker, count).unwrap();
            assert_eq!((shard.index(), shard.count()), (worker, count));
            for id in (0..100).rev().filter(|&id| shard.contains(id)) {
                observed.push((id, SampleKey::new(123, 4, id, 9).draw_u64(0)));
            }
        }
        observed.sort_unstable();
        assert_eq!(observed, expected);
    }
    let base = SampleKey::new(123, 4, 5, 9).draw_u64(0);
    for key in [
        SampleKey::new(124, 4, 5, 9),
        SampleKey::new(123, 5, 5, 9),
        SampleKey::new(123, 4, 6, 9),
        SampleKey::new(123, 4, 5, 10),
    ] {
        assert_ne!(base, key.draw_u64(0));
    }
    assert_ne!(base, SampleKey::new(123, 4, 5, 9).draw_u64(1));
    assert!(DatasetShard::new(0, 0).is_err());
    assert!(DatasetShard::new(2, 2).is_err());
}
