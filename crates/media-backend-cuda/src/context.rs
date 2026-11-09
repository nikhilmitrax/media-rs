//! Actual CUDA context/queue identity and enforced library allocation limits.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, Weak};

use cudarc::driver::{CudaContext as NativeContext, CudaFunction, CudaStream as NativeStream};
use media_core::device::{ContextId, Device, DeviceQueue};
use media_core::error::{Error, Result};

use crate::memory::DeviceBudget;
use crate::sync::RetirementTask;

/// Enforced limits for allocations and streams created through this adapter.
///
/// Driver reservations and external allocations are outside these counters.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaLimits {
    /// Maximum live device bytes owned by this context wrapper's allocations.
    pub memory_bytes: usize,
    /// Maximum simultaneously retained native stream wrappers.
    pub streams: usize,
}

impl Default for CudaLimits {
    fn default() -> Self {
        Self {
            memory_bytes: 512 * 1024 * 1024,
            streams: 8,
        }
    }
}

impl CudaLimits {
    /// Reject limits that cannot permit any work, without opening a GPU.
    pub fn validate(self) -> Result<()> {
        if self.memory_bytes == 0 || self.streams == 0 {
            return Err(Error::invalid(
                "CUDA memory and stream limits must be positive",
            ));
        }
        Ok(())
    }
}

/// Retained primary, independently owned, or imported CUDA context with identity
/// from the actual native handle.
///
/// The native primary context can be shared with compatible external runtimes.
/// Stream creation uses a GPU-side dependency on the default queue rather than
/// cudarc's context-wide first-stream synchronization. Operation boundaries use
/// stream/event ordering, including when sharing an already-active context.
#[derive(Clone, Debug)]
pub struct CudaContext {
    pub(crate) inner: Arc<ContextInner>,
}

#[derive(Debug)]
pub(crate) struct ContextInner {
    pub(crate) native: Arc<NativeContext>,
    pub(crate) device: Device,
    pub(crate) budget: Arc<DeviceBudget>,
    pub(crate) stream_count: AtomicUsize,
    pub(crate) stream_limit: usize,
    pub(crate) preprocess: Mutex<Option<CudaFunction>>,
    pub(crate) retirement: Sender<RetirementTask>,
    retirement_error: Arc<Mutex<Option<String>>>,
}

impl CudaContext {
    /// Open a real device; missing drivers or unavailable ordinals return errors.
    pub fn new(ordinal: u32, limits: CudaLimits) -> Result<Self> {
        Self::open(ordinal, limits, false)
    }

    /// Create an independent native context, incompatible with the primary one.
    ///
    /// This path requires a CUDA 12.8-compatible driver. Even on the same GPU
    /// ordinal, its allocations cannot be submitted to another native context.
    pub fn new_owned(ordinal: u32, limits: CudaLimits) -> Result<Self> {
        Self::open(ordinal, limits, true)
    }

    /// Retain an existing cudarc context without creating or releasing a second
    /// native context. Allocations already owned by the caller are uncharged;
    /// allocations created through this adapter share its explicit byte budget.
    ///
    /// Event tracking must stay enabled, and callers must not destroy native
    /// handles retained by cudarc. Register this wrapper with a Session to share
    /// its canonical allocator; importing twice creates independent budgets.
    pub fn from_native(native: Arc<NativeContext>, limits: CudaLimits) -> Result<Self> {
        limits.validate()?;
        validate_driver_version(Self::driver_api_version()?)?;
        if !native.is_event_tracking() {
            return Err(Error::incompatible(
                "imported CUDA contexts require enabled cudarc event tracking",
            ));
        }
        let ordinal = u32::try_from(native.ordinal())
            .ok()
            .filter(|&n| n <= i32::MAX as u32)
            .ok_or_else(|| {
                Error::invalid("imported CUDA ordinal exceeds the driver index range")
            })?;
        native.bind_to_thread().map_err(cuda_error)?;
        if cudarc::driver::result::device::get(ordinal as i32).map_err(cuda_error)?
            != native.cu_device()
        {
            return Err(Error::incompatible(
                "imported CUDA context ordinal does not identify its actual native device",
            ));
        }
        Self::retain_native(native, ordinal, limits)
    }

    /// Whether the driver library is present; does not establish GPU availability.
    pub fn driver_available() -> bool {
        driver_present()
    }

    /// Installed driver API version (`12080` means CUDA 12.8), without opening a GPU.
    pub fn driver_api_version() -> Result<i32> {
        if !driver_present() {
            return Err(Error::unsupported(
                "CUDA driver library is unavailable; no CPU fallback was selected",
            ));
        }
        read_driver_version()
    }

    fn open(ordinal: u32, limits: CudaLimits, independent: bool) -> Result<Self> {
        limits.validate()?;
        if ordinal > i32::MAX as u32 {
            return Err(Error::invalid(
                "CUDA ordinal exceeds the driver index range",
            ));
        }
        validate_driver_version(Self::driver_api_version()?)?;
        let native = if independent {
            NativeContext::new_non_primary(ordinal as usize, 0)
        } else {
            NativeContext::new(ordinal as usize)
        }
        .map_err(cuda_error)?;
        Self::retain_native(native, ordinal, limits)
    }

    fn retain_native(native: Arc<NativeContext>, ordinal: u32, limits: CudaLimits) -> Result<Self> {
        let context = ContextId::new(native.cu_ctx() as usize as u64)?;
        let (retirement, receiver) = mpsc::channel::<RetirementTask>();
        let retirement_error = Arc::new(Mutex::new(None));
        std::thread::Builder::new()
            .name("media-cuda-retirement".into())
            .spawn(move || {
                while let Ok(task) = receiver.recv() {
                    // Drop establishes retirement even when queued tasks are
                    // unwound or a worker can no longer accept work.
                    drop(task);
                }
            })
            .map_err(Error::io)?;
        Ok(Self {
            inner: Arc::new(ContextInner {
                native,
                device: Device::Cuda { ordinal, context },
                budget: Arc::new(DeviceBudget::new(limits.memory_bytes)),
                stream_count: AtomicUsize::new(0),
                stream_limit: limits.streams,
                preprocess: Mutex::new(None),
                retirement,
                retirement_error,
            }),
        })
    }

    /// Actual device ordinal and native context identity.
    pub fn device(&self) -> Device {
        self.inner.device
    }

    /// Create a nonblocking stream while enforcing the retained stream limit.
    pub fn stream(&self) -> Result<CudaStream> {
        self.check_errors()?;
        self.reserve_stream()?;
        let native = match self.inner.native.default_stream().fork() {
            Ok(stream) => stream,
            Err(error) => {
                self.inner.stream_count.fetch_sub(1, Ordering::AcqRel);
                return Err(cuda_error(error));
            }
        };
        self.register_stream(native)
    }

    /// Retain an existing cudarc queue without creating, synchronizing, or
    /// independently destroying it. The queue must share the exact native Arc
    /// owner, including its event-tracking state, with this context wrapper.
    ///
    /// Imported wrappers consume the same stream quota as created queues. A
    /// native queue can be registered once; clone that wrapper for aliases.
    /// Context-default stream zero is supported. Per-thread/legacy sentinel
    /// handles are rejected because they do not identify a concrete shared queue.
    /// Native work submitted by the caller must obey explicit readiness and
    /// consumer-retention rules, just as work through native_stream().
    pub fn import_stream(&self, native: Arc<NativeStream>) -> Result<CudaStream> {
        self.check_errors()?;
        if !Arc::ptr_eq(native.context(), &self.inner.native)
            || native.context().cu_ctx() != self.inner.native.cu_ctx()
        {
            return Err(Error::incompatible(
                "imported CUDA stream must share the exact native context owner and tracking state",
            ));
        }
        if !self.inner.native.is_event_tracking() {
            return Err(Error::incompatible(
                "imported CUDA streams require enabled cudarc event tracking",
            ));
        }
        let id = native.cu_stream() as usize as u64;
        if id == 1 || id == 2 {
            return Err(Error::unsupported(
                "CUDA special per-thread/legacy stream sentinels cannot identify a shared concrete queue; use context-default zero or an owned stream",
            ));
        }
        let queue = DeviceQueue {
            device: self.device(),
            id,
        };
        if stream_registry().lookup(queue)?.is_some() {
            return Err(Error::incompatible(
                "CUDA native queue is already registered; clone its existing wrapper",
            ));
        }
        self.reserve_stream()?;
        self.register_stream(native)
    }

    fn reserve_stream(&self) -> Result<()> {
        #[allow(
            deprecated,
            reason = "try_update requires a newer Rust than the 1.88 MSRV"
        )]
        self.inner
            .stream_count
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |count| {
                if count < self.inner.stream_limit {
                    count.checked_add(1)
                } else {
                    None
                }
            })
            .map_err(|_| Error::exhausted("CUDA retained stream limit reached"))?;
        Ok(())
    }

    fn register_stream(&self, native: Arc<NativeStream>) -> Result<CudaStream> {
        let id = native.cu_stream() as usize as u64;
        let stream = CudaStream {
            inner: Arc::new(StreamInner {
                native,
                context: self.clone(),
                id,
            }),
        };
        stream_registry().register(stream.queue(), &stream.inner)?;
        Ok(stream)
    }

    /// Enforced live allocation bytes; excludes opaque driver reservations.
    pub fn memory_used(&self) -> usize {
        self.inner.budget.used()
    }

    /// Peak live bytes allocated through this wrapper.
    pub fn memory_peak(&self) -> usize {
        self.inner.budget.peak()
    }

    /// Library device-byte allocation limit.
    pub fn memory_limit(&self) -> usize {
        self.inner.budget.limit()
    }
    /// Immutable allocation and retained-stream limits shared by this wrapper.
    pub fn limits(&self) -> CudaLimits {
        CudaLimits {
            memory_bytes: self.memory_limit(),
            streams: self.inner.stream_limit,
        }
    }
    /// Whether both wrappers share the same allocator, stream limits, kernel
    /// cache, and retirement worker; native handle equality alone is insufficient.
    pub fn shares_resources_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
    /// Record an external backend's retirement diagnostic so later operations
    /// can report it through check_errors(). No GPU work is submitted.
    pub fn record_retirement_failure(&self, message: impl Into<String>) {
        let message = message.into();
        let mut stored = self
            .inner
            .retirement_error
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        if stored.is_none() {
            *stored = Some(message);
        }
    }

    /// Inspect errors from earlier asynchronous work or native destruction.
    pub fn check_errors(&self) -> Result<()> {
        if let Some(error) = lock(&self.inner.retirement_error)?.take() {
            return Err(Error::backend(
                "cuda",
                format!("native surface retirement failed: {error}"),
            ));
        }
        self.inner.native.check_err().map_err(cuda_error)
    }

    /// Borrow the native context for backend-specific integration.
    ///
    /// External allocations are not charged to this wrapper's budget. Native
    /// consumers must preserve cudarc event tracking and retain storage until
    /// consumer completion; a device ordinal alone is not compatibility evidence.
    pub fn native_context(&self) -> &Arc<NativeContext> {
        &self.inner.native
    }
}

/// Cloneable queue wrapper retained by every allocation and completion it owns.
#[derive(Clone, Debug)]
pub struct CudaStream {
    pub(crate) inner: Arc<StreamInner>,
}

#[derive(Debug)]
pub(crate) struct StreamInner {
    pub(crate) native: Arc<NativeStream>,
    pub(crate) context: CudaContext,
    pub(crate) id: u64,
}

impl Drop for StreamInner {
    fn drop(&mut self) {
        let queue = DeviceQueue {
            device: self.context.device(),
            id: self.id,
        };
        stream_registry().remove(queue, self);
        self.context
            .inner
            .stream_count
            .fetch_sub(1, Ordering::AcqRel);
    }
}

// Registration is keyed by actual native context and stream, rather than by one
// wrapper's private map. Two independently retained primary-context wrappers can
// therefore establish correct ordering on the same actual context.
struct StreamRegistry<T> {
    streams: Mutex<HashMap<DeviceQueue, Weak<T>>>,
}
impl<T> StreamRegistry<T> {
    fn new() -> Self {
        Self {
            streams: Mutex::new(HashMap::new()),
        }
    }
    fn register(&self, queue: DeviceQueue, stream: &Arc<T>) -> Result<()> {
        let mut streams = lock(&self.streams)?;
        if streams.get(&queue).and_then(Weak::upgrade).is_some() {
            return Err(Error::incompatible(
                "CUDA native queue already has a retained registered owner",
            ));
        }
        streams.insert(queue, Arc::downgrade(stream));
        Ok(())
    }
    fn lookup(&self, queue: DeviceQueue) -> Result<Option<Arc<T>>> {
        Ok(lock(&self.streams)?.get(&queue).and_then(Weak::upgrade))
    }
    fn remove(&self, queue: DeviceQueue, owner: &T) {
        if let Ok(mut streams) = self.streams.lock()
            && streams
                .get(&queue)
                .is_some_and(|registered| std::ptr::eq(registered.as_ptr(), owner))
        {
            streams.remove(&queue);
        }
    }
}
fn stream_registry() -> &'static StreamRegistry<StreamInner> {
    static REGISTRY: OnceLock<StreamRegistry<StreamInner>> = OnceLock::new();
    REGISTRY.get_or_init(StreamRegistry::new)
}
pub(crate) fn registered_stream(queue: DeviceQueue) -> Result<Arc<StreamInner>> {
    stream_registry().lookup(queue)?.ok_or_else(|| {
        Error::incompatible(
            "CUDA queue is not a retained registered stream in the specified native context",
        )
    })
}

impl CudaStream {
    /// Actual queue identity used by core completion tokens.
    pub fn queue(&self) -> DeviceQueue {
        DeviceQueue {
            device: self.inner.context.device(),
            id: self.inner.id,
        }
    }

    /// Retained owning context.
    pub fn context(&self) -> &CudaContext {
        &self.inner.context
    }

    /// Wait only for this stream's queued work.
    pub fn synchronize(&self) -> Result<()> {
        self.inner.native.synchronize().map_err(cuda_error)
    }

    /// Native queue escape hatch for compatible backend integrations.
    ///
    /// External consumers must retain buffers through a completion recorded after
    /// their use. Never destroy the underlying handle or disable event tracking.
    pub fn native_stream(&self) -> &Arc<NativeStream> {
        &self.inner.native
    }

    pub(crate) fn ensure_device(&self, device: Device) -> Result<()> {
        if self.context().device() != device {
            return Err(Error::incompatible(
                "CUDA operation requires the allocation's actual owning context",
            ));
        }
        self.context().check_errors()
    }
}

pub(crate) fn lock<T>(mutex: &Mutex<T>) -> Result<MutexGuard<'_, T>> {
    mutex
        .lock()
        .map_err(|_| Error::backend("cuda", "CUDA ownership lock was poisoned"))
}

pub(crate) fn cuda_error(error: cudarc::driver::DriverError) -> Error {
    Error::backend("cuda", error.to_string()).with_cause(error)
}

fn validate_driver_version(version: i32) -> Result<()> {
    if version < 12_080 {
        return Err(Error::unsupported(format!(
            "CUDA driver API {version} is older than the required CUDA 12.8 (12080); no CPU fallback was selected"
        )));
    }
    Ok(())
}

#[allow(unsafe_code)]
fn read_driver_version() -> Result<i32> {
    let mut version = 0;
    // SAFETY: The driver presence probe succeeded. This long-established CUDA
    // entrypoint writes one i32 to a live local output and retains no pointer.
    unsafe { cudarc::driver::sys::cuDriverGetVersion(&mut version) }
        .result()
        .map_err(cuda_error)?;
    Ok(version)
}

#[allow(unsafe_code)]
fn driver_present() -> bool {
    // SAFETY: This cudarc probe opens only the platform CUDA driver library and
    // does not invoke functions or expose symbols; absence is returned as false.
    unsafe { cudarc::driver::sys::is_culib_present() }
}

#[allow(unsafe_code)]
pub(crate) fn nvrtc_present() -> bool {
    // SAFETY: As above, the probe only opens/closes the installed NVRTC library.
    unsafe { cudarc::nvrtc::sys::is_culib_present() }
}

#[cfg(test)]
mod registry_tests {
    use super::*;
    #[test]
    fn old_driver_apis_are_rejected_before_loading_new_context_symbols() {
        assert!(validate_driver_version(12_040).is_err());
        assert!(validate_driver_version(0).is_err());
        validate_driver_version(12_080).unwrap();
        validate_driver_version(13_200).unwrap();
    }
    #[test]
    fn independent_wrapper_keys_share_actual_context_but_never_just_an_ordinal() {
        let registry = StreamRegistry::<usize>::new();
        let make = |context, id| DeviceQueue {
            device: Device::Cuda {
                ordinal: 0,
                context: ContextId::new(context).unwrap(),
            },
            id,
        };
        let stream = Arc::new(7);
        registry.register(make(11, 5), &stream).unwrap();
        assert!(Arc::ptr_eq(
            &registry.lookup(make(11, 5)).unwrap().unwrap(),
            &stream
        ));
        assert!(registry.lookup(make(12, 5)).unwrap().is_none());
        assert!(registry.lookup(make(11, 6)).unwrap().is_none());
        drop(stream);
        assert!(registry.lookup(make(11, 5)).unwrap().is_none());
        let replacement = Arc::new(9);
        registry.register(make(11, 5), &replacement).unwrap();
        let rejected = Arc::new(10);
        assert!(registry.register(make(11, 5), &rejected).is_err());
        registry.remove(make(11, 5), rejected.as_ref());
        assert!(
            Arc::ptr_eq(
                &registry.lookup(make(11, 5)).unwrap().unwrap(),
                &replacement
            ),
            "a rejected/older owner cannot remove another queue registration"
        );
        registry.remove(make(11, 5), replacement.as_ref());
        assert!(registry.lookup(make(11, 5)).unwrap().is_none());
    }
}
