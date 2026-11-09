//! Shared budget lifetime, backpressure, and cancellation behavior.
use media_core::{
    error::ErrorKind,
    media::{Image, PixelFormat},
};
use media_pipeline::{
    execution::{BoundedQueue, CancellationToken},
    session::{HostBudget, Limits, Session},
};
#[test]
fn retained_outputs_release_budget_only_after_final_clone() {
    let budget = HostBudget::new(6).unwrap();
    let image = Image::from_packed(1, 1, PixelFormat::Rgb8, vec![0; 3])
        .unwrap()
        .retain(budget.reserve(3).unwrap());
    let retained = image.clone();
    drop(image);
    let reservation = budget.reserve(3).unwrap();
    assert_eq!(
        budget.reserve(1).unwrap_err().kind(),
        ErrorKind::ResourceExhausted
    );
    drop(reservation);
    assert_eq!(budget.current(), 3);
    drop(retained);
    assert_eq!(budget.current(), 0);
    assert_eq!(budget.peak(), 6);
}
#[test]
fn concurrent_reservations_never_exceed_the_shared_limit() {
    let budget = HostBudget::new(32).unwrap();
    let threads: Vec<_> = (0..8)
        .map(|_| {
            let budget = budget.clone();
            std::thread::spawn(move || {
                for _ in 0..200 {
                    if let Ok(guard) = budget.reserve(8) {
                        std::thread::yield_now();
                        drop(guard);
                    }
                }
            })
        })
        .collect();
    for thread in threads {
        thread.join().unwrap();
    }
    assert_eq!(budget.current(), 0);
    assert!(budget.peak() <= 32);
    assert!(budget.reserve(usize::MAX).is_err());
}
#[test]
fn backpressure_returns_owned_work_and_cancel_releases_queued_guards() {
    let budget = HostBudget::new(6).unwrap();
    let cancel = CancellationToken::default();
    let queue = BoundedQueue::new(1).unwrap();
    queue.try_push(budget.reserve(3).unwrap(), &cancel).unwrap();
    let returned = queue
        .try_push(budget.reserve(3).unwrap(), &cancel)
        .unwrap_err();
    assert_eq!(budget.current(), 6);
    drop(returned);
    cancel.cancel();
    assert_eq!(cancel.check().unwrap_err().kind(), ErrorKind::Cancelled);
    queue.clear().unwrap();
    assert_eq!(budget.current(), 0);
    assert!(queue.try_pop().unwrap().is_none());
}
#[test]
fn invalid_limits_are_rejected_without_allocating() {
    assert!(Session::new(Limits::new().host_memory(0)).is_err());
    assert!(Session::new(Limits::new().in_flight_batches(0)).is_err());
}
#[test]
fn working_reservations_split_and_shrink_without_a_gap_in_accounting() {
    let budget = HostBudget::new(64).unwrap();
    let mut working = budget.reserve(64).unwrap();
    let first = working.split_off(12).unwrap();
    assert_eq!(budget.current(), 64);
    assert!(working.split_off(53).is_err());
    let second = working.shrink_to(20).unwrap();
    assert_eq!(budget.current(), 32);
    drop(first);
    assert_eq!(budget.current(), 20);
    drop(second);
    assert_eq!(budget.current(), 0);
}
#[test]
fn session_batch_slots_are_shared_and_never_wait_on_retained_output() {
    let session = Session::new(Limits::new().in_flight_batches(1)).unwrap();
    let clone = session.clone();
    let slot = std::sync::Arc::new(session.reserve_batch().unwrap());
    let retained = slot.clone();
    drop(slot);
    assert_eq!(session.live_batches(), 1);
    assert_eq!(
        clone.reserve_batch().unwrap_err().kind(),
        ErrorKind::ResourceExhausted
    );
    drop(retained);
    assert_eq!(session.live_batches(), 0);
    assert!(clone.reserve_batch().is_ok());
}
