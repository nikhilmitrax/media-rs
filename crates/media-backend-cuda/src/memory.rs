//! Explicit transfers and retained, budgeted native device allocations.

use std::fmt;
use std::mem::size_of;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use cudarc::driver::{CudaSlice, DevicePtrMut, DeviceRepr, ValidAsZeroBits};
use media_core::device::Device;
use media_core::error::{Error, Result};
use media_core::storage::{MemoryKind, StorageLease, StorageOwner};

use crate::context::{CudaStream, cuda_error, lock};
use crate::sync::{ConsumerLease, CudaCompletion};

#[derive(Debug)]
pub(crate) struct DeviceBudget {
    limit: usize,
    used: AtomicUsize,
    peak: AtomicUsize,
}

impl DeviceBudget {
    pub(crate) fn new(limit: usize) -> Self {
        Self {
            limit,
            used: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
        }
    }
    pub(crate) fn used(&self) -> usize {
        self.used.load(Ordering::Acquire)
    }
    pub(crate) fn peak(&self) -> usize {
        self.peak.load(Ordering::Acquire)
    }
    pub(crate) fn limit(&self) -> usize {
        self.limit
    }
    fn reserve(self: &Arc<Self>, bytes: usize) -> Result<Reservation> {
        #[allow(
            deprecated,
            reason = "try_update requires a newer Rust than the 1.88 MSRV"
        )]
        let previous = self
            .used
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |used| {
                used.checked_add(bytes).filter(|&next| next <= self.limit)
            })
            .map_err(|_| {
                Error::exhausted(
                    "CUDA device allocation would exceed the configured library budget",
                )
            })?;
        self.peak.fetch_max(previous + bytes, Ordering::AcqRel);
        Ok(Reservation {
            budget: self.clone(),
            bytes,
        })
    }
}

#[derive(Debug)]
pub(crate) struct Reservation {
    budget: Arc<DeviceBudget>,
    bytes: usize,
}
impl Reservation {
    fn retain_unknown<O>(self, owner: O) {
        // Unproven physical retirement must never return accounted capacity or
        // release the context/session ownership that makes the resource valid.
        std::mem::forget((self, owner));
    }
}
impl Drop for Reservation {
    fn drop(&mut self) {
        self.budget.used.fetch_sub(self.bytes, Ordering::AcqRel);
    }
}

/// Internal pre-publication owner: every partial-launch error retires storage
/// before its reservation is released.
pub(crate) struct PendingAllocation<T: DeviceRepr + Send + Sync + 'static> {
    slice: Option<CudaSlice<T>>,
    stream: CudaStream,
    reservation: Option<Reservation>,
}
impl<T: DeviceRepr + Send + Sync + 'static> PendingAllocation<T> {
    pub(crate) fn slice_mut(&mut self) -> Result<&mut CudaSlice<T>> {
        self.slice
            .as_mut()
            .ok_or_else(|| Error::backend("cuda", "allocation was already published"))
    }
    pub(crate) fn finish(mut self) -> Result<CudaBuffer<T>> {
        let slice = self
            .slice
            .take()
            .ok_or_else(|| Error::backend("cuda", "allocation was already published"))?;
        let reservation = self.reservation.take().ok_or_else(|| {
            Error::backend("cuda", "allocation reservation was already published")
        })?;
        CudaBuffer::finish(slice, self.stream.clone(), reservation)
    }
}
impl<T: DeviceRepr + Send + Sync + 'static> Drop for PendingAllocation<T> {
    fn drop(&mut self) {
        if let Some(slice) = self.slice.take()
            && free_native_slice(slice, &self.stream).is_err()
        {
            if let Some(reservation) = self.reservation.take() {
                reservation.retain_unknown(self.stream.clone());
            } else {
                std::mem::forget(self.stream.clone());
            }
        }
    }
}

/// Shared immutable typed allocation in one actual CUDA context.
///
/// Cloning shares ownership; it never duplicates device bytes. The final drop
/// retires cudarc's tracked producer/consumer events and synchronizes only the
/// allocation stream after freeing, so its budget is not released prematurely.
/// An external consumer must be retained through [`Self::retain_until`].
/// Reads of one shared allocation are ordered across streams so cudarc's single
/// reusable read event cannot forget an earlier outstanding consumer. Different
/// allocations may still be processed concurrently.
pub struct CudaBuffer<T: DeviceRepr + Send + Sync + 'static> {
    pub(crate) inner: Arc<Allocation<T>>,
}

impl<T: DeviceRepr + Send + Sync + 'static> Clone for CudaBuffer<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}
impl<T: DeviceRepr + Send + Sync + 'static> fmt::Debug for CudaBuffer<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CudaBuffer")
            .field("elements", &self.len())
            .field("device", &self.device())
            .finish()
    }
}

pub(crate) struct Allocation<T: DeviceRepr + Send + Sync + 'static> {
    pub(crate) slice: Mutex<Option<CudaSlice<T>>>,
    pub(crate) stream: CudaStream,
    ready: CudaCompletion,
    // cudarc maintains one reusable read event. Chain read operations across
    // streams so overwriting it can never forget an earlier outstanding read.
    last_read: Mutex<CudaCompletion>,
    length: usize,
    bytes: usize,
    _reservation: Option<Reservation>,
    retained: Mutex<Vec<Arc<dyn Retained>>>,
}

trait Retained: fmt::Debug + Send + Sync {}
impl<T: fmt::Debug + Send + Sync> Retained for T {}

impl<T: DeviceRepr + Send + Sync + 'static> fmt::Debug for Allocation<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CudaAllocation")
            .field("elements", &self.length)
            .field("bytes", &self.bytes)
            .finish()
    }
}

impl<T: DeviceRepr + Send + Sync + 'static> Drop for Allocation<T> {
    fn drop(&mut self) {
        let last_read = self
            .last_read
            .get_mut()
            .unwrap_or_else(|poison| poison.into_inner())
            .clone();
        // Own completion evidence supplements cudarc's reusable event tracking.
        // Queue this dependency before native free, so even a failed internal
        // cudarc event cannot permit recycling ahead of an outstanding reader.
        if last_read.depend_on_stream(&self.stream).is_err()
            && last_read.wait_for_retirement().is_err()
        {
            let slice = self
                .slice
                .get_mut()
                .unwrap_or_else(|poison| poison.into_inner())
                .take();
            self.retain_unknown(slice);
            return;
        }
        // cudarc waits on tracked reads/writes before enqueuing its free. Wait
        // only on this allocation's stream after that free before releasing the
        // reservation. No device/context-wide synchronization is used here.
        let slice = self
            .slice
            .get_mut()
            .unwrap_or_else(|poison| poison.into_inner())
            .take();
        if slice.is_some_and(|slice| free_native_slice(slice, &self.stream).is_err()) {
            self.retain_unknown(None);
        }
    }
}
impl<T: DeviceRepr + Send + Sync + 'static> Allocation<T> {
    fn retain_unknown(&mut self, slice: Option<CudaSlice<T>>) {
        let guards = std::mem::take(
            self.retained
                .get_mut()
                .unwrap_or_else(|poison| poison.into_inner()),
        );
        let owners = (
            slice,
            guards,
            self.stream.clone(),
            self.ready.clone(),
            self.last_read
                .get_mut()
                .unwrap_or_else(|poison| poison.into_inner())
                .clone(),
        );
        if let Some(reservation) = self._reservation.take() {
            reservation.retain_unknown(owners);
        } else {
            std::mem::forget(owners);
        }
    }
}

impl<T: DeviceRepr + Send + Sync + 'static> StorageOwner for Allocation<T> {
    fn byte_len(&self) -> usize {
        self.bytes
    }
    fn kind(&self) -> MemoryKind {
        MemoryKind::DeviceLinear
    }
    fn device(&self) -> Device {
        self.stream.context().device()
    }
}

impl<T: DeviceRepr + Send + Sync + 'static> CudaBuffer<T> {
    pub(crate) fn is_uniquely_owned(&self) -> bool {
        Arc::strong_count(&self.inner) == 1
    }

    pub(crate) fn exclusive_pointer(&mut self, stream: &CudaStream) -> Result<u64> {
        stream.ensure_device(self.device())?;
        if !self.is_uniquely_owned() {
            return Err(Error::incompatible(
                "CUDA allocation has aliases; writable export requires exclusive storage or an explicit device copy",
            ));
        }
        let mut slice = lock(&self.inner.slice)?;
        let slice = slice
            .as_mut()
            .ok_or_else(|| Error::backend("cuda", "allocation was already retired"))?;
        lock(&self.inner.last_read)?.depend_on_stream(stream)?;
        let (pointer, _write_marker) = slice.device_ptr_mut(stream.native_stream());
        stream.context().check_errors()?;
        Ok(pointer)
    }

    pub(crate) fn finish(
        slice: CudaSlice<T>,
        stream: CudaStream,
        reservation: Reservation,
    ) -> Result<Self> {
        let length = slice.len();
        let bytes = allocation_bytes::<T>(length)?;
        let ready = match stream.record_completion() {
            Ok(ready) => ready,
            Err(error) => {
                // Retire an allocation whose producer event could not be made
                // before releasing the byte reservation on this error path.
                if let Err(retirement) = free_native_slice(slice, &stream) {
                    reservation.retain_unknown(stream);
                    return Err(retirement);
                }
                return Err(error);
            }
        };
        Ok(Self {
            inner: Arc::new(Allocation {
                slice: Mutex::new(Some(slice)),
                stream,
                last_read: Mutex::new(ready.clone()),
                ready,
                length,
                bytes,
                _reservation: Some(reservation),
                retained: Mutex::new(Vec::new()),
            }),
        })
    }

    /// Number of typed elements.
    pub fn len(&self) -> usize {
        self.inner.length
    }
    /// Retain a resource through every buffer/storage alias and foreign consumer.
    /// Guards are released only after the native allocation has retired.
    pub fn retain<G: fmt::Debug + Send + Sync + 'static>(self, guard: G) -> Self {
        self.inner
            .retained
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .push(Arc::new(guard));
        self
    }

    /// Whether the allocation has no elements; public constructors reject emptiness.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Device-byte allocation size.
    pub fn byte_len(&self) -> usize {
        self.inner.bytes
    }

    /// Actual owning device/context.
    pub fn device(&self) -> Device {
        self.inner.stream.context().device()
    }

    /// Producer readiness retained with the allocation.
    pub fn readiness(&self) -> &CudaCompletion {
        &self.inner.ready
    }

    pub(crate) fn read_on<R>(
        &self,
        stream: &CudaStream,
        operation: impl FnOnce(&CudaSlice<T>) -> Result<R>,
    ) -> Result<R> {
        stream.ensure_device(self.device())?;
        let slice = lock(&self.inner.slice)?;
        let slice = slice
            .as_ref()
            .ok_or_else(|| Error::backend("cuda", "allocation was already retired"))?;
        let mut last_read = lock(&self.inner.last_read)?;
        last_read.depend_on_stream(stream)?;
        let result = operation(slice);
        match stream.record_completion() {
            Ok(completion) => *last_read = completion,
            Err(error) => {
                // If an event cannot be recorded, finish this stream's read
                // before releasing the lock or allowing another event overwrite.
                if let Err(retirement) = confirm_retirement(stream) {
                    // Keep input storage and all attached session quotas: the
                    // unrecorded consumer could still be using this allocation.
                    std::mem::forget((self.clone(), stream.clone()));
                    return Err(retirement);
                }
                return Err(error);
            }
        }
        result
    }

    /// Checked core storage owner with no implicit host view or transfer.
    pub fn storage_lease(&self) -> StorageLease {
        StorageLease::from_owner(self.inner.clone(), Arc::new(self.inner.ready.clone()))
    }

    /// Retain this allocation until an external consumer's recorded event completes.
    ///
    /// Record the event after all consumer work referencing this allocation.
    /// Dropping the returned lease waits on that event, not the entire device.
    pub fn retain_until(&self, consumer_done: CudaCompletion) -> Result<ConsumerLease<T>> {
        if self.device() != consumer_done.device() {
            return Err(Error::incompatible(
                "consumer completion belongs to another CUDA context",
            ));
        }
        Ok(ConsumerLease::new(self.clone(), consumer_done))
    }
}

impl CudaStream {
    /// Explicitly upload host elements into a new budgeted device allocation.
    ///
    /// The host transfer is completed before its borrowed input may be released.
    pub fn upload<T: DeviceRepr + Send + Sync + 'static>(
        &self,
        values: &[T],
    ) -> Result<CudaBuffer<T>> {
        let reservation = self.reserve::<T>(values.len())?;
        let slice = match self.inner.native.clone_htod(values) {
            Ok(slice) => slice,
            Err(error) => return Err(self.failed_allocation(cuda_error(error), reservation)),
        };
        CudaBuffer::finish(slice, self.clone(), reservation)
    }

    /// Allocate initialized zero elements on this stream within the device budget.
    pub fn zeros<T: DeviceRepr + ValidAsZeroBits + Send + Sync + 'static>(
        &self,
        length: usize,
    ) -> Result<CudaBuffer<T>> {
        self.allocate_zeros::<T>(length)?.finish()
    }

    /// Explicitly download an allocation, waiting only for relevant stream work.
    pub fn download<T: DeviceRepr + Send + Sync + 'static>(
        &self,
        buffer: &CudaBuffer<T>,
    ) -> Result<Vec<T>> {
        buffer.read_on(self, |slice| {
            self.inner.native.clone_dtoh(slice).map_err(cuda_error)
        })
    }

    /// Explicit same-device detachment into independently owned storage.
    pub fn copy<T: DeviceRepr + Send + Sync + 'static>(
        &self,
        buffer: &CudaBuffer<T>,
    ) -> Result<CudaBuffer<T>> {
        self.ensure_device(buffer.device())?;
        let reservation = self.reserve::<T>(buffer.len())?;
        let copy = match buffer.read_on(self, |slice| {
            self.inner.native.clone_dtod(slice).map_err(cuda_error)
        }) {
            Ok(copy) => copy,
            Err(error) => return Err(self.failed_allocation(error, reservation)),
        };
        let result = CudaBuffer::finish(copy, self.clone(), reservation)?;
        // Logical session guards survive explicit detachment as well as aliases.
        let guards = buffer
            .inner
            .retained
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clone();
        result
            .inner
            .retained
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .extend(guards);
        Ok(result)
    }

    pub(crate) fn allocate_zeros<T: DeviceRepr + ValidAsZeroBits + Send + Sync + 'static>(
        &self,
        length: usize,
    ) -> Result<PendingAllocation<T>> {
        let reservation = self.reserve::<T>(length)?;
        self.allocate_reserved_zeros(length, reservation)
    }

    pub(crate) fn allocate_reserved_zeros<
        T: DeviceRepr + ValidAsZeroBits + Send + Sync + 'static,
    >(
        &self,
        length: usize,
        reservation: Reservation,
    ) -> Result<PendingAllocation<T>> {
        if allocation_bytes::<T>(length)? != reservation.bytes {
            return Err(Error::invalid(
                "CUDA allocation does not match its reserved byte count",
            ));
        }
        let slice = match self.inner.native.alloc_zeros(length) {
            Ok(slice) => slice,
            Err(error) => {
                return Err(self.failed_allocation(cuda_error(error), reservation));
            }
        };
        Ok(PendingAllocation {
            slice: Some(slice),
            stream: self.clone(),
            reservation: Some(reservation),
        })
    }

    pub(crate) fn reserve<T>(&self, length: usize) -> Result<Reservation> {
        self.context().check_errors()?;
        self.context()
            .inner
            .budget
            .reserve(allocation_bytes::<T>(length)?)
    }
    fn failed_allocation(&self, error: Error, reservation: Reservation) -> Error {
        // cudarc can allocate a pointer, then fail creating its tracking events
        // before returning a CudaSlice owner. Without an owned pointer we cannot
        // prove whether cleanup happened, even if a stream wait succeeds.
        self.context().record_retirement_failure(format!(
            "native allocation/copy failed with unresolved partial ownership: {error}"
        ));
        reservation.retain_unknown(self.clone());
        error.with_constraint("cuda_retirement_unknown")
    }
}

// Check free submission directly, rather than reading cudarc's shared error
// slot after Drop (another thread can consume that slot). This owner came from
// the adapter's allocation stream and all library consumers are ordered first.
#[allow(unsafe_code)]
fn free_native_slice<T>(slice: CudaSlice<T>, stream: &CudaStream) -> Result<()> {
    if slice.stream().cu_stream() != stream.inner.native.cu_stream()
        || slice.context().cu_ctx() != stream.context().inner.native.cu_ctx()
    {
        std::mem::forget(slice);
        let error = Error::incompatible(
            "CUDA allocation retirement has a different native allocation stream/context",
        )
        .with_constraint("cuda_retirement_unknown");
        stream
            .context()
            .record_retirement_failure(error.to_string());
        return Err(error);
    }
    let pointer = slice.leak();
    let result = (|| {
        stream.context().inner.native.bind_to_thread()?;
        if stream.context().inner.native.has_async_alloc() {
            // SAFETY: leak transferred sole pointer ownership without freeing;
            // it was allocated on this exact retained stream, with every library
            // read ordered before free. The pointer is freed exactly once here.
            unsafe {
                cudarc::driver::result::free_async(pointer, stream.inner.native.cu_stream())?;
            }
        } else {
            stream.inner.native.synchronize()?;
            // SAFETY: this pointer was allocated by cudarc's synchronous branch;
            // all pending accesses on its allocation stream have completed.
            unsafe {
                cudarc::driver::result::free_sync(pointer)?;
            }
        }
        stream.inner.native.synchronize()
    })();
    result.map_err(|error| {
        stream
            .context()
            .inner
            .native
            .record_err(Err::<(), _>(error));
        stream
            .context()
            .record_retirement_failure(error.to_string());
        cuda_error(error).with_constraint("cuda_retirement_unknown")
    })
}

pub(crate) fn confirm_retirement(stream: &CudaStream) -> Result<()> {
    stream.inner.native.synchronize().map_err(|error| {
        stream
            .context()
            .inner
            .native
            .record_err(Err::<(), _>(error));
        stream
            .context()
            .record_retirement_failure(error.to_string());
        cuda_error(error).with_constraint("cuda_retirement_unknown")
    })
}

fn allocation_bytes<T>(length: usize) -> Result<usize> {
    if length == 0 || size_of::<T>() == 0 {
        return Err(Error::invalid(
            "CUDA allocations require nonzero elements and element size",
        ));
    }
    length
        .checked_mul(size_of::<T>())
        .ok_or_else(|| Error::overflow("CUDA allocation byte count overflow"))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unknown_retirement_preserves_accounted_capacity_and_attached_owners() {
        struct Owner(Arc<AtomicUsize>);
        impl Drop for Owner {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }
        let dropped = Arc::new(AtomicUsize::new(0));
        let budget = Arc::new(DeviceBudget::new(32));
        budget
            .reserve(24)
            .unwrap()
            .retain_unknown(Owner(dropped.clone()));
        assert_eq!(budget.used(), 24);
        assert_eq!(dropped.load(Ordering::SeqCst), 0);
        assert!(
            budget.reserve(9).is_err(),
            "unknown physical retirement cannot make capacity reusable"
        );
        let remaining = budget.reserve(8).unwrap();
        drop(remaining);
        assert_eq!(budget.used(), 24);
        // This is deliberately the CPU quota-retention policy only. It neither
        // allocates a GPU nor claims to inject a real driver fault.
    }
    #[test]
    fn reservation_bounds_are_checked_and_released_once() {
        let budget = Arc::new(DeviceBudget::new(32));
        let first = budget.reserve(24).unwrap();
        assert!(budget.reserve(9).is_err());
        let second = budget.reserve(8).unwrap();
        assert_eq!((budget.used(), budget.peak()), (32, 32));
        drop(first);
        assert_eq!(budget.used(), 8);
        drop(second);
        assert_eq!(budget.used(), 0);
        assert!(allocation_bytes::<f32>(usize::MAX).is_err());
        assert!(allocation_bytes::<u8>(0).is_err());
        assert!(allocation_bytes::<()>(1).is_err());
        let large = Arc::new(DeviceBudget::new(usize::MAX));
        let all = large.reserve(usize::MAX).unwrap();
        assert!(large.reserve(1).is_err());
        drop(all);
    }
}
