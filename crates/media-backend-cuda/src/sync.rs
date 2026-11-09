//! Context-checked producer dependencies and consumer-retained allocation leases.

use std::sync::Arc;

use cudarc::driver::{CudaEvent, CudaStream as NativeStream, DeviceRepr};
use media_core::device::{Device, DeviceQueue};
use media_core::error::{Error, Result};
use media_core::storage::{Completion, StorageLease};

use crate::context::{CudaContext, CudaStream, cuda_error, registered_stream};
use crate::memory::CudaBuffer;

/// Recorded producer/consumer event retaining its native context and stream.
#[derive(Clone, Debug)]
pub struct CudaCompletion {
    event: Arc<CudaEvent>,
    context: CudaContext,
    _native_stream: Arc<NativeStream>,
    _retained_stream: Option<CudaStream>,
}

impl CudaCompletion {
    /// Actual context of this event.
    pub fn device(&self) -> Device {
        self.context.device()
    }

    /// Wait for this event only, propagating asynchronous launch errors.
    pub fn wait(&self) -> Result<()> {
        self.event.synchronize().map_err(cuda_error)
    }
    /// Establish final retirement, falling back to this token's producer stream
    /// if its event fails. Event failures remain visible in context diagnostics
    /// even when the stream fallback establishes safe recycling.
    pub fn wait_for_retirement(&self) -> Result<()> {
        if let Err(event_error) = self.wait() {
            if let Err(stream_error) = self._native_stream.synchronize() {
                self.context
                    .inner
                    .native
                    .record_err(Err::<(), _>(stream_error));
                self.context
                    .record_retirement_failure(stream_error.to_string());
                return Err(cuda_error(stream_error).with_constraint("cuda_retirement_unknown"));
            }
            self.context.record_retirement_failure(format!(
                "completion event failed; producer stream fallback completed safely: {event_error}"
            ));
        }
        Ok(())
    }

    /// Establish GPU-side ordering on a compatible stream, without a CPU wait.
    pub fn depend_on_stream(&self, stream: &CudaStream) -> Result<()> {
        stream.ensure_device(self.device())?;
        stream.inner.native.wait(&self.event).map_err(cuda_error)
    }
}

impl Completion for CudaCompletion {
    fn is_complete(&self) -> Result<bool> {
        self.context
            .inner
            .native
            .bind_to_thread()
            .map_err(cuda_error)?;
        self.event.try_is_complete().map_err(cuda_error)
    }
    fn wait(&self) -> Result<()> {
        self.wait()
    }
    fn depend_on(&self, queue: DeviceQueue) -> Result<()> {
        if queue.device != self.device() {
            return Err(Error::incompatible(
                "CUDA completion and consumer queue have different native contexts",
            ));
        }
        let stream = registered_stream(queue)?;
        stream.native.wait(&self.event).map_err(cuda_error)
    }
}

impl CudaStream {
    /// Record all work currently submitted to this queue as a retained event.
    pub fn record_completion(&self) -> Result<CudaCompletion> {
        let event = self.inner.native.record_event(None).map_err(cuda_error)?;
        Ok(CudaCompletion {
            event: Arc::new(event),
            context: self.context().clone(),
            _native_stream: self.inner.native.clone(),
            _retained_stream: Some(self.clone()),
        })
    }
}

impl CudaContext {
    /// Record an event after work on an externally owned CUDA stream.
    ///
    /// The returned wrapper retains this context and does not destroy the stream.
    ///
    /// # Safety
    /// `raw_stream` must be zero (this context's default stream) or a live native
    /// `CUstream` belonging to this exact native context. Its owner must keep it
    /// valid through event recording, and must not invalidate the context.
    #[allow(unsafe_code)]
    pub unsafe fn record_external_completion(&self, raw_stream: u64) -> Result<CudaCompletion> {
        // SAFETY: The caller establishes the native stream/context ownership
        // contract above; cudarc's borrowed wrapper never destroys this handle.
        let stream = unsafe {
            self.inner
                .native
                .wrap_borrowed_stream(raw_stream as usize as _)
        };
        let event = stream.record_event(None).map_err(cuda_error)?;
        Ok(CudaCompletion {
            event: Arc::new(event),
            context: self.clone(),
            _native_stream: stream,
            _retained_stream: None,
        })
    }
}

pub(crate) struct RetirementTask {
    pub(crate) completion: CudaCompletion,
    pub(crate) _owner: StorageLease,
}
impl Drop for RetirementTask {
    fn drop(&mut self) {
        if self.completion.wait_for_retirement().is_err() {
            std::mem::forget((self._owner.clone(), self.completion.clone()));
        }
    }
}

impl CudaStream {
    pub(crate) fn retire_external_after(
        &self,
        completion: CudaCompletion,
        owner: StorageLease,
    ) -> Result<()> {
        let task = RetirementTask {
            completion,
            _owner: owner,
        };
        if let Err(error) = self.context().inner.retirement.send(task) {
            // A failed worker must not cause early reuse of the decoder surface.
            error.0.completion.wait()?;
            return Err(Error::backend(
                "cuda",
                "native surface retirement worker stopped",
            ));
        }
        Ok(())
    }
}

/// Allocation lease retained until work recorded after an external consumer ends.
///
/// Its final drop waits on the consumer's event before releasing storage. Keeping
/// a producer-ready event alone would not satisfy this retirement contract.
#[derive(Debug)]
pub struct ConsumerLease<T: DeviceRepr + Send + Sync + 'static> {
    _buffer: CudaBuffer<T>,
    completion: CudaCompletion,
}

impl<T: DeviceRepr + Send + Sync + 'static> ConsumerLease<T> {
    pub(crate) fn new(buffer: CudaBuffer<T>, completion: CudaCompletion) -> Self {
        Self {
            _buffer: buffer,
            completion,
        }
    }
    /// Explicitly wait for consumer completion; the storage remains retained.
    pub fn wait(&self) -> Result<()> {
        self.completion.wait()
    }
}

impl<T: DeviceRepr + Send + Sync + 'static> Drop for ConsumerLease<T> {
    fn drop(&mut self) {
        if self.completion.wait_for_retirement().is_err() {
            std::mem::forget((self._buffer.clone(), self.completion.clone()));
        }
    }
}
