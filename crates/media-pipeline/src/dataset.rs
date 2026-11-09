//! Stable finite dataset schedules and runtime-independent bounded prefetch.
//!
//! Callers choose their own worker threads or executor. Admission reserves a
//! declared worst-case result size before work starts, completion restores
//! schedule order, and retained result guards continue to hold their permits.
//! This bounds this adapter's results; arbitrary clones made by a consumer and
//! opaque backend reservations require their own storage/session accounting.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::{Arc, Mutex, mpsc};

use media_core::error::{Error, ErrorKind, Result};

use crate::execution::CancellationToken;
use crate::sampling::{DatasetShard, SampleKey};

/// One stable sample identity and its application-defined asset/clip request.
#[derive(Debug)]
pub struct DatasetSample<T> {
    /// Identity stable across epochs, workers, and scheduling order.
    pub sample_id: u64,
    /// Asset, clip request, or application descriptor; no decode is implied.
    pub request: T,
}
/// Explicit treatment of a shard's final incomplete batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IncompleteBatch {
    /// Preserve a smaller, ragged final batch.
    Keep,
    /// Drop its real samples and report their count.
    Drop,
    /// Repeat the final real sample, keeping its original key and a padding mask.
    RepeatLast,
    /// Add masked empty slots, without inventing sample identities or pixels.
    PadMissing,
}
/// Deterministic selection, ordering, and finite batch shape.
#[derive(Debug, Clone, Copy)]
pub struct DatasetOptions {
    /// Global sampling seed.
    pub seed: u64,
    /// Epoch key; affects shuffle/random draws, never shard membership.
    pub epoch: u64,
    /// Stable augmentation identity.
    pub augmentation_id: u64,
    /// Optional stable-ID shard. Every original sample belongs to one shard.
    pub shard: Option<DatasetShard>,
    /// Sort by the versioned SampleKey draw, then sample ID, before batching.
    pub shuffle: bool,
    /// Nonzero maximum number of slots in a batch.
    pub batch_size: usize,
    /// Treatment of the selected shard's final incomplete batch.
    pub incomplete: IncompleteBatch,
}
impl Default for DatasetOptions {
    fn default() -> Self {
        Self {
            seed: 0,
            epoch: 0,
            augmentation_id: 0,
            shard: None,
            shuffle: false,
            batch_size: 1,
            incomplete: IncompleteBatch::Keep,
        }
    }
}
/// Whether a scheduled slot is original data or explicit tail padding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotValidity {
    /// Original selected sample.
    Valid,
    /// Explicit repeat of the last real sample, retaining source identity.
    RepeatedPadding,
    /// No source sample exists; downstream must supply explicit padding.
    MissingPadding,
}
/// One requested slot with stable identity and a visible padding mask.
#[derive(Debug)]
pub struct DatasetSlot<T> {
    /// Shared source request, absent only for missing padding.
    pub sample: Option<Arc<DatasetSample<T>>>,
    /// Sampling key, absent when no source sample exists.
    pub key: Option<SampleKey>,
    /// Original/repeated/missing status.
    pub validity: SlotValidity,
}
impl<T> Clone for DatasetSlot<T> {
    fn clone(&self) -> Self {
        Self {
            sample: self.sample.clone(),
            key: self.key,
            validity: self.validity,
        }
    }
}
/// Caller-ordered batch independent of worker completion order.
#[derive(Debug)]
pub struct DatasetBatch<T> {
    /// Zero-based sequence within this shard/epoch schedule.
    pub ordinal: u64,
    /// Requests and explicit padding in batch order.
    pub slots: Vec<DatasetSlot<T>>,
}
/// Finite selection and incomplete-batch accounting.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DatasetPlanReport {
    /// Number of distinct original input samples.
    pub input_samples: usize,
    /// Original samples assigned to this shard before final-batch handling.
    pub shard_samples: usize,
    /// Original samples excluded by this shard.
    pub other_shard_samples: usize,
    /// Original samples removed by the explicit Drop policy.
    pub dropped_incomplete_samples: usize,
    /// Original samples actually scheduled, excluding padding repeats.
    pub scheduled_samples: usize,
    /// Added repeated or missing tail slots.
    pub padding_slots: usize,
}
/// Validated finite batches across arbitrary assets, without opening a backend.
#[derive(Debug)]
pub struct DatasetPlan<T> {
    batches: Vec<DatasetBatch<T>>,
    report: DatasetPlanReport,
}
impl<T> DatasetPlan<T> {
    /// Validate unique stable IDs, shard, optionally shuffle, and form batches.
    ///
    /// Shuffle uses `(key.draw_u64(0), sample_id)` order, independent of worker
    /// count and input collection order. Without shuffle, caller order is kept.
    /// Shard-local ordering is documented separately from global merge order.
    pub fn new(samples: Vec<DatasetSample<T>>, options: DatasetOptions) -> Result<Self> {
        if options.batch_size == 0 || options.batch_size > 65_536 {
            return Err(Error::invalid("dataset batch size must be in 1..=65536"));
        }
        let input_samples = samples.len();
        let mut ids = BTreeSet::new();
        let mut selected = Vec::new();
        for sample in samples {
            if !ids.insert(sample.sample_id) {
                return Err(Error::invalid("dataset sample IDs must be unique"));
            }
            if options
                .shard
                .is_some_and(|shard| !shard.contains(sample.sample_id))
            {
                continue;
            }
            let key = SampleKey::new(
                options.seed,
                options.epoch,
                sample.sample_id,
                options.augmentation_id,
            );
            selected.push(DatasetSlot {
                sample: Some(Arc::new(sample)),
                key: Some(key),
                validity: SlotValidity::Valid,
            });
        }
        if options.shuffle {
            selected.sort_by_key(|slot| {
                let key = slot.key.expect("original slot always has a key");
                (key.draw_u64(0), key.sample_id)
            });
        }
        let mut report = DatasetPlanReport {
            input_samples,
            shard_samples: selected.len(),
            other_shard_samples: input_samples - selected.len(),
            ..Default::default()
        };
        let mut batches = Vec::new();
        let mut iter = selected.into_iter();
        loop {
            let mut slots: Vec<_> = iter.by_ref().take(options.batch_size).collect();
            if slots.is_empty() {
                break;
            }
            let real = slots.len();
            if real < options.batch_size {
                match options.incomplete {
                    IncompleteBatch::Drop => {
                        report.dropped_incomplete_samples = real;
                        break;
                    }
                    IncompleteBatch::Keep => {}
                    IncompleteBatch::RepeatLast => {
                        let mut repeated = slots.last().expect("nonempty batch").clone();
                        repeated.validity = SlotValidity::RepeatedPadding;
                        slots.resize_with(options.batch_size, || repeated.clone());
                    }
                    IncompleteBatch::PadMissing => {
                        slots.resize_with(options.batch_size, || DatasetSlot {
                            sample: None,
                            key: None,
                            validity: SlotValidity::MissingPadding,
                        })
                    }
                }
            }
            report.scheduled_samples += real;
            report.padding_slots += slots.len() - real;
            batches.push(DatasetBatch {
                ordinal: u64::try_from(batches.len())
                    .map_err(|_| Error::overflow("dataset batch ordinal overflow"))?,
                slots,
            });
        }
        Ok(Self { batches, report })
    }
    /// Finite batches in deterministic schedule order.
    pub fn batches(&self) -> &[DatasetBatch<T>] {
        &self.batches
    }
    /// Selection/drop/padding counts; skipped corrupt results remain separate.
    pub fn report(&self) -> DatasetPlanReport {
        self.report
    }
}

/// Explicit response to missing samples or corrupt inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailurePolicy {
    /// Return the structured error immediately.
    Error,
    /// Remove that slot and retain a structured skipped-sample report.
    Skip,
    /// Invoke the caller's explicit padding operation and mark its output invalid.
    Pad,
}
/// Separate boundary/gap and corrupt-input behavior.
#[derive(Debug, Clone, Copy)]
pub struct FailurePolicies {
    /// MissingData failures, including established timing gaps.
    pub missing: FailurePolicy,
    /// Backend or I/O failures; cancellation/resource/programming errors still fail.
    pub corrupt: FailurePolicy,
}
impl Default for FailurePolicies {
    fn default() -> Self {
        Self {
            missing: FailurePolicy::Error,
            corrupt: FailurePolicy::Error,
        }
    }
}
/// One original identity and structured reason for a skipped or padded result.
#[derive(Debug)]
pub struct SampleFailure {
    /// Original stable sampling key.
    pub key: SampleKey,
    /// Full failure, including source/track/operation context when supplied.
    pub error: Error,
}
/// One actual or explicitly padded result; source key is always retained.
#[derive(Debug)]
pub struct ResolvedSample<T> {
    /// Original sampling identity, unchanged by padding.
    pub key: SampleKey,
    /// Produced value or caller-supplied padding.
    pub value: T,
    /// False for padding; padding never claims an actual successful selection.
    pub valid: bool,
    /// Reason for padding, when present.
    pub failure: Option<Error>,
}
/// Ordered valid/padded output plus every explicit skipped diagnostic.
#[derive(Debug)]
pub struct ResolvedSamples<T> {
    /// Ordered retained slots. Skip creates a deliberately ragged output.
    pub samples: Vec<ResolvedSample<T>>,
    /// Skipped identities/errors, in original input order.
    pub skipped: Vec<SampleFailure>,
}
/// Apply explicit gap/corruption behavior without hiding cancellation or budgets.
///
/// Padding is supplied by the caller because video repetition, silence, and
/// zero tensors have different semantics. No synthetic source time is assigned.
pub fn resolve_samples<T>(
    results: impl IntoIterator<Item = (SampleKey, Result<T>)>,
    policies: FailurePolicies,
    mut pad: impl FnMut(SampleKey, &Error) -> Result<T>,
) -> Result<ResolvedSamples<T>> {
    let mut output = ResolvedSamples {
        samples: Vec::new(),
        skipped: Vec::new(),
    };
    for (key, result) in results {
        match result {
            Ok(value) => output.samples.push(ResolvedSample {
                key,
                value,
                valid: true,
                failure: None,
            }),
            Err(error) => {
                let policy = match error.kind() {
                    ErrorKind::MissingData => policies.missing,
                    ErrorKind::Backend | ErrorKind::Io => policies.corrupt,
                    _ => return Err(error),
                };
                match policy {
                    FailurePolicy::Error => return Err(error),
                    FailurePolicy::Skip => output.skipped.push(SampleFailure { key, error }),
                    FailurePolicy::Pad => {
                        let value = pad(key, &error)?;
                        output.samples.push(ResolvedSample {
                            key,
                            value,
                            valid: false,
                            failure: Some(error),
                        });
                    }
                }
            }
        }
    }
    Ok(output)
}

/// Admission bounds for a worker-driven prefetch adapter.
#[derive(Debug, Clone, Copy)]
pub struct PrefetchLimits {
    /// Work plus completed/returned result guards retained simultaneously.
    pub max_results: usize,
    /// Aggregate declared or completed result bytes retained by this adapter.
    pub max_bytes: usize,
    /// Worst-case bytes reserved before each batch is handed to a worker.
    pub max_result_bytes: usize,
}
impl PrefetchLimits {
    fn validate(self) -> Result<()> {
        if self.max_results == 0
            || self.max_results > 65_536
            || self.max_result_bytes == 0
            || self.max_result_bytes > self.max_bytes
        {
            return Err(Error::invalid(
                "prefetch needs 1..=65536 results and a positive per-result reservation within max_bytes",
            ));
        }
        Ok(())
    }
}
#[derive(Debug, Default)]
struct Usage {
    results: usize,
    bytes: usize,
}
#[derive(Debug)]
struct Budget {
    limits: PrefetchLimits,
    usage: Mutex<Usage>,
}
#[derive(Debug)]
struct Permit {
    budget: Arc<Budget>,
    bytes: usize,
}
impl Permit {
    fn acquire(budget: &Arc<Budget>) -> Result<Option<Self>> {
        let mut usage = budget
            .usage
            .lock()
            .map_err(|_| Error::backend("dataset", "prefetch budget lock poisoned"))?;
        let next = usage.bytes.checked_add(budget.limits.max_result_bytes);
        if usage.results >= budget.limits.max_results
            || next.is_none_or(|n| n > budget.limits.max_bytes)
        {
            return Ok(None);
        }
        usage.results += 1;
        usage.bytes = next.expect("validated byte bound");
        Ok(Some(Self {
            budget: budget.clone(),
            bytes: budget.limits.max_result_bytes,
        }))
    }
    fn shrink(&mut self, actual: usize) -> Result<()> {
        if actual > self.bytes {
            return Err(Error::exhausted(
                "worker result exceeds its declared prefetch byte reservation",
            )
            .with_constraint("max_result_bytes"));
        }
        let mut usage = self
            .budget
            .usage
            .lock()
            .map_err(|_| Error::backend("dataset", "prefetch budget lock poisoned"))?;
        usage.bytes -= self.bytes - actual;
        self.bytes = actual;
        Ok(())
    }
}
impl Drop for Permit {
    fn drop(&mut self) {
        let mut usage = self.budget.usage.lock().unwrap_or_else(|e| e.into_inner());
        usage.results -= 1;
        usage.bytes -= self.bytes;
    }
}
struct Completed<U> {
    sequence: u64,
    result: Result<U>,
    bytes: usize,
    _permit: Permit,
}

/// Worker-owned request. Dropping it produces an ordered abandonment diagnostic.
pub struct PrefetchWork<T, U> {
    batch: DatasetBatch<T>,
    sender: mpsc::Sender<Completed<U>>,
    permit: Option<Permit>,
    cancel: CancellationToken,
}
impl<T, U> PrefetchWork<T, U> {
    /// Scheduled requests; order and random keys never depend on completion time.
    pub fn batch(&self) -> &DatasetBatch<T> {
        &self.batch
    }
    /// Cooperative cancellation handle for use between blocking backend calls.
    pub fn cancellation(&self) -> CancellationToken {
        self.cancel.clone()
    }
    /// Complete this work and report its actual retained result bytes.
    ///
    /// The worker must enforce the declared worst-case bound while allocating;
    /// this check cannot undo a worker's already-performed external allocation.
    /// Oversized results are dropped and become ordered resource errors.
    pub fn complete(mut self, result: Result<U>, actual_bytes: usize) -> Result<()> {
        let mut permit = self.permit.take().expect("work owns one admission permit");
        let (result, bytes) = if self.cancel.is_cancelled() {
            drop(result);
            (
                Err(Error::new(ErrorKind::Cancelled, "dataset work cancelled")),
                0,
            )
        } else if let Err(error) = permit.shrink(actual_bytes) {
            drop(result);
            (Err(error), 0)
        } else {
            (result, actual_bytes)
        };
        // Failed outputs hold no value, so retain no claimed result bytes.
        let bytes = if result.is_err() { 0 } else { bytes };
        permit.shrink(bytes)?;
        self.sender
            .send(Completed {
                sequence: self.batch.ordinal,
                result,
                bytes,
                _permit: permit,
            })
            .map_err(|_| {
                Error::new(
                    ErrorKind::Cancelled,
                    "dataset prefetch receiver was dropped",
                )
            })
    }
}
impl<T, U> Drop for PrefetchWork<T, U> {
    fn drop(&mut self) {
        if let Some(mut permit) = self.permit.take() {
            // No successful result exists. Even if poisoned, dropping the permit
            // still releases its original reservation safely.
            let _ = permit.shrink(0);
            let _ = self.sender.send(Completed {
                sequence: self.batch.ordinal,
                result: Err(Error::new(
                    ErrorKind::Cancelled,
                    "dataset worker abandoned its scheduled request",
                )),
                bytes: 0,
                _permit: permit,
            });
        }
    }
}
/// Admission distinguishes backpressure from exhaustion of the finite schedule.
pub enum WorkPoll<T, U> {
    /// One admitted batch and its ownership/completion ticket.
    Ready(PrefetchWork<T, U>),
    /// Existing work/results hold a byte or count permit; release or consume one.
    Backpressure,
    /// Every batch has been submitted; completions may still be outstanding.
    Finished,
}
/// An ordered result whose guard retains its bounded prefetch reservation.
pub struct Prefetched<U> {
    completed: Completed<U>,
}
impl<U> Prefetched<U> {
    /// Original batch sequence, independent of worker order.
    pub fn sequence(&self) -> u64 {
        self.completed.sequence
    }
    /// Success or structured failure for this batch; no failure is silently skipped.
    pub fn result(&self) -> &Result<U> {
        &self.completed.result
    }
    /// Actual library-retained result size reported by the producer.
    pub fn bytes(&self) -> usize {
        self.completed.bytes
    }
}
/// Nonblocking, finite, caller-driven bounded prefetch with ordered completion.
pub struct DatasetPrefetch<T, U> {
    pending: VecDeque<DatasetBatch<T>>,
    sender: mpsc::Sender<Completed<U>>,
    receiver: mpsc::Receiver<Completed<U>>,
    completed: BTreeMap<u64, Completed<U>>,
    budget: Arc<Budget>,
    cancel: CancellationToken,
    issued: u64,
    next_result: u64,
}
impl<T, U> DatasetPrefetch<T, U> {
    /// Reserve before work starts; no threads, runtime, or blocking waits are created.
    pub fn new(plan: DatasetPlan<T>, limits: PrefetchLimits) -> Result<Self> {
        limits.validate()?;
        let (sender, receiver) = mpsc::channel();
        Ok(Self {
            pending: plan.batches.into(),
            sender,
            receiver,
            completed: BTreeMap::new(),
            budget: Arc::new(Budget {
                limits,
                usage: Mutex::new(Usage::default()),
            }),
            cancel: CancellationToken::default(),
            issued: 0,
            next_result: 0,
        })
    }
    /// Shared cancellation checked by admission, workers, and result retrieval.
    pub fn cancellation(&self) -> CancellationToken {
        self.cancel.clone()
    }
    /// Admit the next deterministic batch or report explicit backpressure.
    pub fn try_next_work(&mut self) -> Result<WorkPoll<T, U>> {
        self.check_cancelled()?;
        if self.pending.is_empty() {
            return Ok(WorkPoll::Finished);
        }
        let Some(permit) = Permit::acquire(&self.budget)? else {
            return Ok(WorkPoll::Backpressure);
        };
        let batch = self.pending.pop_front().expect("nonempty pending schedule");
        self.issued += 1;
        Ok(WorkPoll::Ready(PrefetchWork {
            batch,
            sender: self.sender.clone(),
            permit: Some(permit),
            cancel: self.cancel.clone(),
        }))
    }
    /// Poll the next schedule-ordered result. None means not available yet or EOF.
    ///
    /// A slow earlier worker causes bounded head-of-line waiting; a faster later
    /// worker can never change ordering or cause more work to bypass admission.
    pub fn try_next_result(&mut self) -> Result<Option<Prefetched<U>>> {
        self.check_cancelled()?;
        for result in self.receiver.try_iter() {
            self.completed.insert(result.sequence, result);
        }
        Ok(self.completed.remove(&self.next_result).map(|completed| {
            self.next_result += 1;
            Prefetched { completed }
        }))
    }
    /// Whether every scheduled result was delivered, independent of retained guards.
    pub fn is_finished(&self) -> bool {
        self.pending.is_empty() && self.next_result == self.issued
    }
    /// Current application-declared bytes, including work and retained result guards.
    pub fn retained_bytes(&self) -> Result<usize> {
        Ok(self
            .budget
            .usage
            .lock()
            .map_err(|_| Error::backend("dataset", "prefetch budget lock poisoned"))?
            .bytes)
    }
    fn check_cancelled(&mut self) -> Result<()> {
        if let Err(error) = self.cancel.check() {
            self.pending.clear();
            self.completed.clear();
            while self.receiver.try_recv().is_ok() {}
            return Err(error);
        }
        Ok(())
    }
}
impl<T, U> Drop for DatasetPrefetch<T, U> {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}
