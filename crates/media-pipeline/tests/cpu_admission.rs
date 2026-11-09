//! Atomic CPU admission is bounded independently of retained payload allocations.
use media_core::error::ErrorKind;
use media_pipeline::session::{CpuUsage, Limits, Session};
use std::sync::{Arc, Barrier};

#[test]
fn failed_multi_resource_reservations_charge_nothing_and_retry_after_drop() {
    let session = Session::new(
        Limits::new()
            .cpu_executions(3)
            .cpu_codecs(3)
            .cpu_codec_threads(4),
    )
    .unwrap();
    let held = session.reserve_cpu(1, 2).unwrap();
    let error = session.reserve_cpu(2, 3).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::ResourceExhausted);
    assert_eq!(
        error.context().constraint.as_deref(),
        Some("cpu_codec_threads")
    );
    assert_eq!(
        session.live_cpu_usage(),
        CpuUsage {
            executions: 1,
            codecs: 1,
            codec_threads: 2
        }
    );
    let second = session.reserve_cpu(2, 2).unwrap();
    assert_eq!(session.live_cpu_usage().codecs, 3);
    assert_eq!(
        session
            .reserve_cpu(1, 1)
            .unwrap_err()
            .context()
            .constraint
            .as_deref(),
        Some("cpu_codecs")
    );
    drop(second);
    drop(held);
    assert_eq!(session.live_cpu_usage(), CpuUsage::default());
    assert!(session.reserve_cpu(2, 4).is_ok());
    assert!(session.reserve_cpu(1, 0).is_err());
    assert!(Limits::new().cpu_executions(0).validate().is_err());
    assert!(Limits::new().cpu_codecs(0).validate().is_err());
    assert!(Limits::new().cpu_codec_threads(0).validate().is_err());
}

#[test]
fn simultaneous_contenders_share_one_nonblocking_execution_limit() {
    let session = Session::new(Limits::new().cpu_executions(1)).unwrap();
    let held = session.reserve_cpu(0, 0).unwrap();
    let barrier = Arc::new(Barrier::new(5));
    let workers: Vec<_> = (0..4)
        .map(|_| {
            let session = session.clone();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                barrier.wait();
                let error = session.reserve_cpu(1, 1).unwrap_err();
                assert_eq!(
                    error.context().constraint.as_deref(),
                    Some("cpu_executions")
                );
            })
        })
        .collect();
    barrier.wait();
    for worker in workers {
        worker.join().unwrap();
    }
    assert_eq!(session.live_cpu_usage().executions, 1);
    drop(held);
    let retry = session.reserve_cpu(1, 1).unwrap();
    assert_eq!(session.live_cpu_usage().codec_threads, 1);
    drop(retry);
    assert_eq!(session.live_cpu_usage(), CpuUsage::default());
}
