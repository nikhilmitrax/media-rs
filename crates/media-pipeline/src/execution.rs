//! Cancellation and bounded hand-off without unbounded internal queues.
use media_core::error::{Error, ErrorKind, Result};
use std::{
    collections::VecDeque,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
};
/// Cooperative cancellation shared with an execution.
#[derive(Debug, Clone, Default)]
pub struct CancellationToken(Arc<AtomicBool>);
impl CancellationToken {
    /// Request cancellation; native work still retains its own resources.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }
    /// Query cancellation.
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
    /// Stop a new unit of work if cancellation was requested.
    pub fn check(&self) -> Result<()> {
        if self.is_cancelled() {
            Err(Error::new(ErrorKind::Cancelled, "execution cancelled"))
        } else {
            Ok(())
        }
    }
}
/// Nonblocking bounded queue; a full queue is explicit backpressure.
#[derive(Debug)]
pub struct BoundedQueue<T> {
    capacity: usize,
    items: Mutex<VecDeque<T>>,
}
impl<T> BoundedQueue<T> {
    /// Create a queue with a nonzero capacity.
    pub fn new(capacity: usize) -> Result<Self> {
        if capacity == 0 {
            return Err(Error::invalid("queue capacity must be nonzero"));
        }
        Ok(Self {
            capacity,
            items: Mutex::new(VecDeque::new()),
        })
    }
    /// Submit or return ownership of the item under backpressure/cancellation.
    pub fn try_push(&self, item: T, cancel: &CancellationToken) -> std::result::Result<(), T> {
        if cancel.is_cancelled() {
            return Err(item);
        }
        let Ok(mut queue) = self.items.lock() else {
            return Err(item);
        };
        if queue.len() == self.capacity || cancel.is_cancelled() {
            return Err(item);
        }
        queue.push_back(item);
        Ok(())
    }
    /// Retrieve the next item, preserving submission order.
    pub fn try_pop(&self) -> Result<Option<T>> {
        Ok(self
            .items
            .lock()
            .map_err(|_| Error::backend("execution", "queue lock poisoned"))?
            .pop_front())
    }
    /// Release queued unused resources after cancellation.
    pub fn clear(&self) -> Result<()> {
        self.items
            .lock()
            .map_err(|_| Error::backend("execution", "queue lock poisoned"))?
            .clear();
        Ok(())
    }
}
