//! Shared limits and host-allocation accounting retained by output leases.
use media_core::error::{Error, Result};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};
/// Enforced library-level limits. Opaque driver allocations are not included.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    /// Maximum live library-accounted host bytes.
    pub host_memory: usize,
    /// Maximum encoded object materialized by an image convenience request.
    pub encoded_bytes: usize,
    /// Maximum decoded image pixels.
    pub image_pixels: u64,
    /// Maximum outstanding facade video/tensor batches; acquisition never blocks.
    pub in_flight_batches: usize,
    /// Simultaneously active CPU facade execution scopes, including readers.
    pub cpu_executions: usize,
    /// Known CPU codec contexts admitted across those execution scopes.
    pub cpu_codecs: usize,
    /// Sum of explicitly requested workers for admitted CPU codec contexts.
    /// This does not count opaque demux-probe, FFT, SDK or process workers.
    pub cpu_codec_threads: usize,
}
impl Default for Limits {
    fn default() -> Self {
        Self {
            host_memory: 512 * 1024 * 1024,
            encoded_bytes: 256 * 1024 * 1024,
            image_pixels: 64 * 1024 * 1024,
            in_flight_batches: 3,
            cpu_executions: 8,
            cpu_codecs: 16,
            cpu_codec_threads: 32,
        }
    }
}
impl Limits {
    /// Default bounded configuration.
    pub fn new() -> Self {
        Self::default()
    }
    /// Set enforced host allocation accounting limit.
    pub fn host_memory(mut self, bytes: usize) -> Self {
        self.host_memory = bytes;
        self
    }
    /// Set encoded object limit.
    pub fn encoded_bytes(mut self, bytes: usize) -> Self {
        self.encoded_bytes = bytes;
        self
    }
    /// Set image pixel count limit.
    pub fn image_pixels(mut self, pixels: u64) -> Self {
        self.image_pixels = pixels;
        self
    }
    /// Set outstanding batch count limit.
    pub fn in_flight_batches(mut self, count: usize) -> Self {
        self.in_flight_batches = count;
        self
    }
    /// Set active CPU execution/reader scopes; acquisition never blocks.
    pub fn cpu_executions(mut self, count: usize) -> Self {
        self.cpu_executions = count;
        self
    }
    /// Set known CPU codec-context admission across executions.
    pub fn cpu_codecs(mut self, count: usize) -> Self {
        self.cpu_codecs = count;
        self
    }
    /// Set the aggregate explicitly requested CPU codec-worker count.
    pub fn cpu_codec_threads(mut self, count: usize) -> Self {
        self.cpu_codec_threads = count;
        self
    }
    /// Reject configurations that cannot execute any work.
    pub fn validate(self) -> Result<Self> {
        if self.host_memory == 0
            || self.encoded_bytes == 0
            || self.image_pixels == 0
            || self.in_flight_batches == 0
            || self.cpu_executions == 0
            || self.cpu_codecs == 0
            || self.cpu_codec_threads == 0
        {
            return Err(Error::invalid(
                "memory, input, pixel, in-flight, and CPU admission limits must be nonzero",
            ));
        }
        Ok(self)
    }
}
/// Active known CPU admission, independent of retained host output bytes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CpuUsage {
    /// Active execution scopes or reader lifetimes.
    pub executions: usize,
    /// Known admitted codec contexts.
    pub codecs: usize,
    /// Sum of explicitly requested codec worker counts.
    pub codec_threads: usize,
}
#[derive(Debug)]
struct CpuState {
    usage: Mutex<CpuUsage>,
}
/// Atomic multi-resource CPU admission, released after the execution's codecs.
/// Retained copied frames/samples do not retain a CPU codec permit.
#[derive(Debug)]
pub struct CpuReservation {
    state: Arc<CpuState>,
    codecs: usize,
    threads: usize,
}
impl Drop for CpuReservation {
    fn drop(&mut self) {
        let mut usage = self
            .state
            .usage
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        usage.executions -= 1;
        usage.codecs -= self.codecs;
        usage.codec_threads -= self.threads;
    }
}
#[derive(Debug)]
struct BudgetState {
    limit: usize,
    current: AtomicUsize,
    peak: AtomicUsize,
}
/// Shared allocation budget, safe to use across concurrent executions.
#[derive(Debug, Clone)]
pub struct HostBudget(Arc<BudgetState>);
impl HostBudget {
    /// Create a nonzero host byte budget.
    pub fn new(limit: usize) -> Result<Self> {
        if limit == 0 {
            return Err(Error::invalid("host memory limit must be nonzero"));
        }
        Ok(Self(Arc::new(BudgetState {
            limit,
            current: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
        })))
    }
    /// Acquire without waiting on application-held output leases.
    pub fn reserve(&self, bytes: usize) -> Result<HostReservation> {
        let previous = self
            .0
            .current
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                current
                    .checked_add(bytes)
                    .filter(|&next| next <= self.0.limit)
            })
            .map_err(|_| {
                Error::exhausted(format!(
                    "host budget exhausted: requested {bytes} bytes, {} live, {} limit",
                    self.current(),
                    self.0.limit
                ))
                .with_constraint("host_memory")
            })?;
        self.0.peak.fetch_max(previous + bytes, Ordering::Relaxed);
        Ok(HostReservation {
            state: Arc::clone(&self.0),
            bytes,
        })
    }
    /// Currently retained accounted bytes.
    pub fn current(&self) -> usize {
        self.0.current.load(Ordering::Acquire)
    }
    /// Highest observed live accounted bytes.
    pub fn peak(&self) -> usize {
        self.0.peak.load(Ordering::Relaxed)
    }
    /// Enforced byte limit.
    pub fn limit(&self) -> usize {
        self.0.limit
    }
}
/// Releases allocation accounting only when its owning output/storage retires.
#[derive(Debug)]
pub struct HostReservation {
    state: Arc<BudgetState>,
    bytes: usize,
}
impl HostReservation {
    /// Retained byte charge.
    pub fn bytes(&self) -> usize {
        self.bytes
    }
    /// Retain only final output bytes after a conservatively reserved operation.
    /// This never grows a reservation or temporarily releases the output charge.
    pub fn shrink_to(mut self, bytes: usize) -> Result<Self> {
        if bytes > self.bytes {
            return Err(Error::invalid(
                "cannot grow a reservation while shrinking it",
            ));
        }
        self.state
            .current
            .fetch_sub(self.bytes - bytes, Ordering::AcqRel);
        self.bytes = bytes;
        Ok(self)
    }
    /// Divide an existing reservation between independently retained outputs.
    pub fn split_off(&mut self, bytes: usize) -> Result<Self> {
        if bytes > self.bytes {
            return Err(Error::invalid("split exceeds reserved bytes"));
        }
        self.bytes -= bytes;
        Ok(Self {
            state: self.state.clone(),
            bytes,
        })
    }
}
impl Drop for HostReservation {
    fn drop(&mut self) {
        self.state.current.fetch_sub(self.bytes, Ordering::AcqRel);
    }
}
/// Shared execution resources; each source read still opens an independent cursor.
#[derive(Debug, Clone)]
pub struct Session {
    limits: Limits,
    host: HostBudget,
    batches: HostBudget,
    cpu: Arc<CpuState>,
}
impl Default for Session {
    fn default() -> Self {
        let limits = Limits::default();
        Self {
            limits,
            host: HostBudget(Arc::new(BudgetState {
                limit: limits.host_memory,
                current: AtomicUsize::new(0),
                peak: AtomicUsize::new(0),
            })),
            batches: HostBudget(Arc::new(BudgetState {
                limit: limits.in_flight_batches,
                current: AtomicUsize::new(0),
                peak: AtomicUsize::new(0),
            })),
            cpu: Arc::new(CpuState {
                usage: Mutex::new(CpuUsage::default()),
            }),
        }
    }
}
impl Session {
    /// Create a session with validated limits.
    pub fn new(limits: Limits) -> Result<Self> {
        let limits = limits.validate()?;
        Ok(Self {
            limits,
            host: HostBudget::new(limits.host_memory)?,
            batches: HostBudget::new(limits.in_flight_batches)?,
            cpu: Arc::new(CpuState {
                usage: Mutex::new(CpuUsage::default()),
            }),
        })
    }
    /// Execution limits.
    pub fn limits(&self) -> Limits {
        self.limits
    }
    /// Shared host allocation accounting.
    pub fn host_budget(&self) -> &HostBudget {
        &self.host
    }
    /// Reserve one result batch without waiting on the caller's retained outputs.
    pub fn reserve_batch(&self) -> Result<BatchReservation> {
        self.batches.reserve(1).map(BatchReservation).map_err(|_|Error::exhausted(format!(
            "in-flight batch limit exhausted: {} retained, {} limit; release a result before submitting more work",
            self.live_batches(),self.limits.in_flight_batches)).with_constraint("in_flight_batches"))
    }
    /// Outstanding materialized batch reservations, including retained exports.
    pub fn live_batches(&self) -> usize {
        self.batches.current()
    }
    /// Reject requests above configured known-codec/worker capacity before I/O.
    /// This is a preflight only; occupied capacity is checked by reserve_cpu.
    pub fn validate_cpu_request(&self, codecs: usize, threads: usize) -> Result<()> {
        if codecs == 0 && threads != 0 || codecs != 0 && threads < codecs {
            return Err(Error::invalid(
                "CPU codec admission requires at least one requested worker per known codec",
            ));
        }
        if codecs > self.limits.cpu_codecs {
            return Err(
                Error::exhausted("requested known CPU codecs exceed the Session limit")
                    .with_constraint("cpu_codecs"),
            );
        }
        if threads > self.limits.cpu_codec_threads {
            return Err(
                Error::exhausted("requested CPU codec workers exceed the Session limit")
                    .with_constraint("cpu_codec_threads"),
            );
        }
        Ok(())
    }
    /// Reserve one CPU execution plus known codec/worker counts atomically.
    /// Acquisition never waits, and no user callback runs while the lock is held.
    /// Use zero codecs/workers for a non-codec transform or metadata-only scope.
    pub fn reserve_cpu(&self, codecs: usize, threads: usize) -> Result<CpuReservation> {
        self.validate_cpu_request(codecs, threads)?;
        let mut usage = self
            .cpu
            .usage
            .lock()
            .map_err(|_| Error::backend("cpu", "CPU admission lock poisoned"))?;
        let executions = usage
            .executions
            .checked_add(1)
            .filter(|&n| n <= self.limits.cpu_executions)
            .ok_or_else(|| {
                Error::exhausted("Session CPU execution slots are retained by active work/readers")
                    .with_constraint("cpu_executions")
            })?;
        let admitted_codecs = usage
            .codecs
            .checked_add(codecs)
            .filter(|&n| n <= self.limits.cpu_codecs)
            .ok_or_else(|| {
                Error::exhausted("Session known CPU codec-context slots are exhausted")
                    .with_constraint("cpu_codecs")
            })?;
        let admitted_threads = usage
            .codec_threads
            .checked_add(threads)
            .filter(|&n| n <= self.limits.cpu_codec_threads)
            .ok_or_else(|| {
                Error::exhausted(
                    "Session explicitly requested CPU codec-worker capacity is exhausted",
                )
                .with_constraint("cpu_codec_threads")
            })?;
        *usage = CpuUsage {
            executions,
            codecs: admitted_codecs,
            codec_threads: admitted_threads,
        };
        Ok(CpuReservation {
            state: self.cpu.clone(),
            codecs,
            threads,
        })
    }
    /// Snapshot of active known CPU admission; opaque native workers are unknown.
    pub fn live_cpu_usage(&self) -> CpuUsage {
        *self
            .cpu
            .usage
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
    }
}

/// One batch slot, retained by result allocations and foreign exports.
#[derive(Debug)]
pub struct BatchReservation(HostReservation);
impl BatchReservation {
    /// Number of slots held; a batch always holds one.
    pub fn slots(&self) -> usize {
        self.0.bytes()
    }
}
