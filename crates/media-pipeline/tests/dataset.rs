//! Stable dataset selection and real ownership/backpressure/cancellation behavior.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use media_core::error::{Error, ErrorKind};
use media_pipeline::dataset::*;
use media_pipeline::sampling::{DatasetShard, SampleKey};

fn samples(n: u64) -> Vec<DatasetSample<u64>> {
    (0..n)
        .map(|sample_id| DatasetSample {
            sample_id,
            request: sample_id + 100,
        })
        .collect()
}
fn keys(plan: &DatasetPlan<u64>) -> Vec<SampleKey> {
    plan.batches()
        .iter()
        .flat_map(|batch| batch.slots.iter().filter_map(|slot| slot.key))
        .collect()
}
fn work<T, U>(queue: &mut DatasetPrefetch<T, U>) -> PrefetchWork<T, U> {
    match queue.try_next_work().unwrap() {
        WorkPoll::Ready(work) => work,
        _ => panic!("expected admitted work"),
    }
}
fn limits() -> PrefetchLimits {
    PrefetchLimits {
        max_results: 2,
        max_bytes: 16,
        max_result_bytes: 8,
    }
}

#[test]
fn shuffle_keys_and_shards_are_independent_of_collection_and_worker_order() {
    let options = DatasetOptions {
        seed: 17,
        epoch: 4,
        augmentation_id: 3,
        batch_size: 4,
        shuffle: true,
        ..Default::default()
    };
    let original = DatasetPlan::new(samples(31), options).unwrap();
    let mut reversed = samples(31);
    reversed.reverse();
    assert_eq!(
        keys(&original),
        keys(&DatasetPlan::new(reversed, options).unwrap())
    );
    let expected: BTreeMap<_, _> = keys(&original)
        .into_iter()
        .map(|key| (key.sample_id, key))
        .collect();
    for workers in [1, 2, 3, 7] {
        let mut observed = BTreeMap::new();
        for index in 0..workers {
            let shard = DatasetPlan::new(
                samples(31),
                DatasetOptions {
                    shard: Some(DatasetShard::new(index, workers).unwrap()),
                    ..options
                },
            )
            .unwrap();
            for key in keys(&shard) {
                assert!(observed.insert(key.sample_id, key).is_none());
            }
        }
        assert_eq!(observed, expected);
    }
}

#[test]
fn incomplete_batches_have_explicit_ragged_drop_and_padding_identity() {
    let options = DatasetOptions {
        batch_size: 3,
        ..Default::default()
    };
    let keep = DatasetPlan::new(samples(4), options).unwrap();
    assert_eq!(
        keep.batches()
            .iter()
            .map(|b| b.slots.len())
            .collect::<Vec<_>>(),
        [3, 1]
    );
    let drop = DatasetPlan::new(
        samples(4),
        DatasetOptions {
            incomplete: IncompleteBatch::Drop,
            ..options
        },
    )
    .unwrap();
    assert_eq!(
        (
            drop.report().scheduled_samples,
            drop.report().dropped_incomplete_samples
        ),
        (3, 1)
    );
    let repeat = DatasetPlan::new(
        samples(4),
        DatasetOptions {
            incomplete: IncompleteBatch::RepeatLast,
            ..options
        },
    )
    .unwrap();
    let tail = &repeat.batches()[1].slots;
    assert_eq!(
        tail.iter()
            .map(|s| s.key.unwrap().sample_id)
            .collect::<Vec<_>>(),
        [3, 3, 3]
    );
    assert_eq!(tail[1].validity, SlotValidity::RepeatedPadding);
    assert!(Arc::ptr_eq(
        tail[0].sample.as_ref().unwrap(),
        tail[2].sample.as_ref().unwrap()
    ));
    assert_eq!(
        (
            repeat.report().scheduled_samples,
            repeat.report().padding_slots
        ),
        (4, 2)
    );
    let missing = DatasetPlan::new(
        samples(4),
        DatasetOptions {
            incomplete: IncompleteBatch::PadMissing,
            ..options
        },
    )
    .unwrap();
    assert!(
        missing.batches()[1].slots[1..]
            .iter()
            .all(|s| s.sample.is_none()
                && s.key.is_none()
                && s.validity == SlotValidity::MissingPadding)
    );
    assert!(
        DatasetPlan::new(samples(0), options)
            .unwrap()
            .batches()
            .is_empty()
    );
    assert!(
        DatasetPlan::new(
            samples(1),
            DatasetOptions {
                batch_size: 0,
                ..options
            }
        )
        .is_err()
    );
    assert!(
        DatasetPlan::new(
            vec![
                DatasetSample {
                    sample_id: 1,
                    request: 0
                },
                DatasetSample {
                    sample_id: 1,
                    request: 1
                }
            ],
            options
        )
        .is_err()
    );
}

#[test]
fn explicit_misses_padding_and_corrupt_skips_keep_keys_and_diagnostics() {
    let key = |id| SampleKey::new(1, 2, id, 3);
    let result = resolve_samples(
        [
            (key(0), Ok(7)),
            (
                key(1),
                Err(Error::new(ErrorKind::MissingData, "gap").with_track(5)),
            ),
            (key(2), Err(Error::backend("codec", "corrupt source"))),
            (key(3), Ok(9)),
        ],
        FailurePolicies {
            missing: FailurePolicy::Pad,
            corrupt: FailurePolicy::Skip,
        },
        |actual, error| {
            assert_eq!(actual, key(1));
            assert_eq!(error.context().track, Some(5));
            Ok(0)
        },
    )
    .unwrap();
    assert_eq!(
        result
            .samples
            .iter()
            .map(|s| (s.key.sample_id, s.value, s.valid))
            .collect::<Vec<_>>(),
        [(0, 7, true), (1, 0, false), (3, 9, true)]
    );
    assert_eq!(
        result.samples[1].failure.as_ref().unwrap().kind(),
        ErrorKind::MissingData
    );
    assert_eq!(result.skipped[0].key, key(2));
    assert_eq!(
        result.skipped[0].error.context().backend.as_deref(),
        Some("codec")
    );
    for kind in [
        ErrorKind::Cancelled,
        ErrorKind::ResourceExhausted,
        ErrorKind::AmbiguousTiming,
        ErrorKind::InvalidInput,
    ] {
        let result = resolve_samples::<u8>(
            [(key(0), Err(Error::new(kind, "must fail")))],
            FailurePolicies {
                missing: FailurePolicy::Pad,
                corrupt: FailurePolicy::Skip,
            },
            |_, _| panic!("cannot hide cancellation/resource/semantic errors"),
        );
        assert_eq!(result.unwrap_err().kind(), kind);
    }
}

#[test]
fn reverse_completion_restores_order_and_retained_results_hold_admission() {
    let plan = DatasetPlan::new(samples(3), DatasetOptions::default()).unwrap();
    let mut queue = DatasetPrefetch::<_, u64>::new(plan, limits()).unwrap();
    let first = work(&mut queue);
    let second = work(&mut queue);
    assert!(matches!(
        queue.try_next_work().unwrap(),
        WorkPoll::Backpressure
    ));
    second.complete(Ok(11), 4).unwrap();
    assert!(
        queue.try_next_result().unwrap().is_none(),
        "later completion cannot bypass earlier work"
    );
    first.complete(Ok(10), 6).unwrap();
    assert_eq!(queue.retained_bytes().unwrap(), 10);
    let first = queue.try_next_result().unwrap().unwrap();
    let second = queue.try_next_result().unwrap().unwrap();
    assert_eq!((first.sequence(), second.sequence()), (0, 1));
    assert_eq!(*first.result().as_ref().unwrap(), 10);
    assert!(matches!(
        queue.try_next_work().unwrap(),
        WorkPoll::Backpressure
    ));
    drop(first);
    let third = work(&mut queue);
    assert_eq!(
        queue.retained_bytes().unwrap(),
        12,
        "retained4 plus reserved8"
    );
    third.complete(Ok(12), 3).unwrap();
    let third = queue.try_next_result().unwrap().unwrap();
    assert_eq!(*third.result().as_ref().unwrap(), 12);
    assert!(queue.is_finished());
    drop(second);
    drop(third);
    assert_eq!(queue.retained_bytes().unwrap(), 0);
}

#[test]
fn byte_admission_precedes_work_and_shrinks_only_after_completion() {
    let plan = DatasetPlan::new(samples(3), DatasetOptions::default()).unwrap();
    let mut queue = DatasetPrefetch::<_, u8>::new(
        plan,
        PrefetchLimits {
            max_results: 3,
            max_bytes: 12,
            max_result_bytes: 8,
        },
    )
    .unwrap();
    let first = work(&mut queue);
    assert!(matches!(
        queue.try_next_work().unwrap(),
        WorkPoll::Backpressure
    ));
    first.complete(Ok(1), 4).unwrap();
    let second = work(&mut queue);
    assert_eq!(queue.retained_bytes().unwrap(), 12);
    assert!(matches!(
        queue.try_next_work().unwrap(),
        WorkPoll::Backpressure
    ));
    second.complete(Ok(2), 4).unwrap();
    assert!(
        matches!(queue.try_next_work().unwrap(), WorkPoll::Backpressure),
        "two4byte results leave less than next8byte reservation"
    );
    drop(queue.try_next_result().unwrap().unwrap());
    assert!(matches!(queue.try_next_work().unwrap(), WorkPoll::Ready(_)));
}

struct Tracked(Arc<AtomicUsize>);
impl Drop for Tracked {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn oversized_and_abandoned_work_produce_ordered_errors_without_leaks() {
    let drops = Arc::new(AtomicUsize::new(0));
    let plan = DatasetPlan::new(samples(2), DatasetOptions::default()).unwrap();
    let mut queue = DatasetPrefetch::<_, Tracked>::new(plan, limits()).unwrap();
    let first = work(&mut queue);
    let second = work(&mut queue);
    second.complete(Ok(Tracked(drops.clone())), 9).unwrap();
    assert_eq!(
        drops.load(Ordering::SeqCst),
        1,
        "oversized output is actually dropped"
    );
    drop(first);
    let first = queue.try_next_result().unwrap().unwrap();
    assert_eq!(
        first.result().as_ref().err().unwrap().kind(),
        ErrorKind::Cancelled
    );
    let second = queue.try_next_result().unwrap().unwrap();
    assert_eq!(
        second.result().as_ref().err().unwrap().kind(),
        ErrorKind::ResourceExhausted
    );
    assert_eq!(queue.retained_bytes().unwrap(), 0);
    assert!(queue.is_finished());
}

#[test]
fn actual_workers_preserve_keys_and_result_order() {
    let plan = DatasetPlan::new(
        samples(8),
        DatasetOptions {
            shuffle: true,
            seed: 41,
            batch_size: 1,
            ..Default::default()
        },
    )
    .unwrap();
    let expected = keys(&plan);
    let mut queue = DatasetPrefetch::<_, SampleKey>::new(
        plan,
        PrefetchLimits {
            max_results: 8,
            max_bytes: 64,
            max_result_bytes: 8,
        },
    )
    .unwrap();
    let mut workers = Vec::new();
    for _ in 0..8 {
        let ticket = work(&mut queue);
        workers.push(std::thread::spawn(move || {
            let key = ticket.batch().slots[0].key.unwrap();
            ticket.complete(Ok(key), 8).unwrap();
        }));
    }
    for worker in workers {
        worker.join().unwrap();
    }
    let mut observed = Vec::new();
    while let Some(result) = queue.try_next_result().unwrap() {
        observed.push(*result.result().as_ref().unwrap());
    }
    assert_eq!(observed, expected);
    assert_eq!(
        observed
            .iter()
            .map(|key| key.sample_id)
            .collect::<BTreeSet<_>>()
            .len(),
        8
    );
    assert!(queue.is_finished());
}

#[test]
fn cancellation_and_receiver_drop_release_results_and_stop_new_work() {
    let drops = Arc::new(AtomicUsize::new(0));
    let plan = DatasetPlan::new(samples(3), DatasetOptions::default()).unwrap();
    let mut queue = DatasetPrefetch::<_, Tracked>::new(plan, limits()).unwrap();
    let first = work(&mut queue);
    let second = work(&mut queue);
    first.complete(Ok(Tracked(drops.clone())), 4).unwrap();
    queue.cancellation().cancel();
    assert!(matches!(queue.try_next_work(), Err(e) if e.kind() == ErrorKind::Cancelled));
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    assert!(second.cancellation().is_cancelled());
    drop(queue);
    assert!(second.complete(Ok(Tracked(drops.clone())), 4).is_err());
    assert_eq!(drops.load(Ordering::SeqCst), 2);
}
