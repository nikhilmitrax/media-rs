//! Session-owned CUDA contexts and known native resource admission.
//!
//! Owned tensor allocations share their canonical context's CudaLimits. This
//! separate budget covers retained native surface leases and conservative
//! transcode allowances; opaque decoder-reference and driver heaps are unknown.

use std::collections::BTreeMap;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};

use media_backend_cuda::{CudaContext, CudaLimits};
use media_core::{
    device::Device,
    error::{Error, Result},
};

/// Aggregate native resource limits shared by executions of one Session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaSessionLimits {
    /// Retained native codec sessions: a reader reserves one; transcode reserves
    /// decoder and encoder slots atomically. Native frames retain reader slots.
    pub max_codec_sessions: usize,
    /// Known retained native surface bytes plus reserved transcode allowances.
    /// This excludes owned tensors (CudaLimits) and opaque backend reservations.
    pub max_native_bytes: usize,
}
impl Default for CudaSessionLimits {
    fn default() -> Self {
        Self {
            max_codec_sessions: 2,
            max_native_bytes: 512 * 1024 * 1024,
        }
    }
}
impl CudaSessionLimits {
    fn validate(self) -> Result<()> {
        if !(1..=64).contains(&self.max_codec_sessions) || self.max_native_bytes == 0 {
            return Err(Error::invalid(
                "CUDA session requires 1..=64 codec slots and a positive known-native byte limit",
            ));
        }
        Ok(())
    }
}
#[derive(Debug)]
struct Counters {
    limits: CudaSessionLimits,
    codecs: AtomicUsize,
    bytes: AtomicUsize,
    peak_bytes: AtomicUsize,
}
/// Shared admission permit; cloning an Arc retains rather than duplicates charge.
#[derive(Debug)]
#[cfg(any(feature = "ffmpeg", test))]
pub(crate) struct CodecPermit {
    counters: Arc<Counters>,
    count: usize,
}
#[cfg(any(feature = "ffmpeg", test))]
impl Drop for CodecPermit {
    fn drop(&mut self) {
        self.counters.codecs.fetch_sub(self.count, Ordering::AcqRel);
    }
}
/// Known native byte reservation, retained by native image storage and consumers.
#[derive(Debug)]
#[cfg(any(feature = "ffmpeg", test))]
pub(crate) struct NativeReservation {
    counters: Arc<Counters>,
    bytes: usize,
}
#[cfg(any(feature = "ffmpeg", test))]
impl Drop for NativeReservation {
    fn drop(&mut self) {
        self.counters.bytes.fetch_sub(self.bytes, Ordering::AcqRel);
    }
}

#[derive(Debug)]
pub(crate) struct CudaResources {
    contexts: Mutex<BTreeMap<u32, CudaContext>>,
    counters: Arc<Counters>,
}
impl Default for CudaResources {
    fn default() -> Self {
        Self::new(CudaSessionLimits::default(), Vec::new()).expect("valid CUDA defaults")
    }
}
impl CudaResources {
    pub(crate) fn new(limits: CudaSessionLimits, registered: Vec<CudaContext>) -> Result<Self> {
        limits.validate()?;
        let resources = Self {
            contexts: Mutex::new(BTreeMap::new()),
            counters: Arc::new(Counters {
                limits,
                codecs: AtomicUsize::new(0),
                bytes: AtomicUsize::new(0),
                peak_bytes: AtomicUsize::new(0),
            }),
        };
        for context in registered {
            resources.register(&context)?;
        }
        Ok(resources)
    }
    pub(crate) fn context(&self, ordinal: u32, limits: CudaLimits) -> Result<CudaContext> {
        limits.validate()?;
        let mut registry = self
            .contexts
            .lock()
            .map_err(|_| Error::backend("cuda", "session context registry lock poisoned"))?;
        if let Some(context) = registry.get(&ordinal) {
            if context.limits() != limits {
                return Err(Error::incompatible("this Session already has a CUDA context with different allocation/stream limits; share its limits or use a separate Session")
                    .with_constraint("cuda_context_limits"));
            }
            return Ok(context.clone());
        }
        // Only native device bootstrap occurs under this lock: no source,
        // demuxer, user transform, allocator callback, or source factory runs.
        #[cfg(feature = "ffmpeg")]
        let context = media_backend_ffmpeg::cuda::create_context(ordinal, limits)?;
        #[cfg(not(feature = "ffmpeg"))]
        let context = CudaContext::new(ordinal, limits)?;
        registry.insert(ordinal, context.clone());
        Ok(context)
    }
    pub(crate) fn register(&self, supplied: &CudaContext) -> Result<CudaContext> {
        let Device::Cuda { ordinal, .. } = supplied.device() else {
            return Err(Error::invalid("registered context must be CUDA"));
        };
        let mut registry = self
            .contexts
            .lock()
            .map_err(|_| Error::backend("cuda", "session context registry lock poisoned"))?;
        if let Some(context) = registry.get(&ordinal) {
            if context.device() != supplied.device() || !context.shares_resources_with(supplied) {
                return Err(Error::incompatible("a Session requires one canonical native context and shared allocator per CUDA ordinal; the supplied wrapper has different context or resource ownership")
                    .with_constraint("cuda_context_identity"));
            }
            return Ok(context.clone());
        }
        registry.insert(ordinal, supplied.clone());
        Ok(supplied.clone())
    }
    #[cfg(any(feature = "ffmpeg", test))]
    pub(crate) fn reserve_codecs(&self, count: usize) -> Result<Arc<CodecPermit>> {
        if count == 0 {
            return Err(Error::invalid("codec reservation must be positive"));
        }
        self.counters.codecs.fetch_update(Ordering::AcqRel, Ordering::Acquire,
            |used| used.checked_add(count).filter(|&next| next <= self.counters.limits.max_codec_sessions))
            .map_err(|_| Error::exhausted("Session CUDA codec slots are retained by active readers/transcodes or native frames")
                .with_constraint("cuda_codec_sessions"))?;
        Ok(Arc::new(CodecPermit {
            counters: self.counters.clone(),
            count,
        }))
    }
    #[cfg(any(feature = "ffmpeg", test))]
    pub(crate) fn reserve_native(&self, bytes: usize) -> Result<NativeReservation> {
        let previous = self.counters.bytes.fetch_update(Ordering::AcqRel, Ordering::Acquire,
            |used| used.checked_add(bytes).filter(|&next| next <= self.counters.limits.max_native_bytes))
            .map_err(|_| Error::exhausted("Session known CUDA native-surface byte budget is exhausted; release retained frames or reduce the transcode allowance")
                .with_constraint("cuda_native_bytes"))?;
        self.counters
            .peak_bytes
            .fetch_max(previous + bytes, Ordering::AcqRel);
        Ok(NativeReservation {
            counters: self.counters.clone(),
            bytes,
        })
    }
    pub(crate) fn live_codecs(&self) -> usize {
        self.counters.codecs.load(Ordering::Acquire)
    }
    pub(crate) fn native_bytes(&self) -> usize {
        self.counters.bytes.load(Ordering::Acquire)
    }
    pub(crate) fn peak_native_bytes(&self) -> usize {
        self.counters.peak_bytes.load(Ordering::Acquire)
    }
    #[cfg(feature = "ffmpeg")]
    pub(crate) fn native_limit(&self) -> usize {
        self.counters.limits.max_native_bytes
    }
    pub(crate) fn limits(&self) -> CudaSessionLimits {
        self.counters.limits
    }
    #[cfg(feature = "ffmpeg")]
    pub(crate) fn peek(&self, ordinal: u32) -> Result<Option<CudaContext>> {
        Ok(self
            .contexts
            .lock()
            .map_err(|_| Error::backend("cuda", "session context registry lock poisoned"))?
            .get(&ordinal)
            .cloned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use media_core::error::ErrorKind;
    #[test]
    fn codec_admission_is_atomic_and_retained_permits_do_not_expand_capacity() {
        let resources = CudaResources::new(
            CudaSessionLimits {
                max_codec_sessions: 2,
                max_native_bytes: 10,
            },
            Vec::new(),
        )
        .unwrap();
        let first = resources.reserve_codecs(1).unwrap();
        let retained = first.clone();
        assert_eq!(
            resources.reserve_codecs(2).unwrap_err().kind(),
            ErrorKind::ResourceExhausted
        );
        assert_eq!(
            resources.live_codecs(),
            1,
            "failed multi-slot admission charges nothing"
        );
        drop(first);
        assert_eq!(resources.live_codecs(), 1);
        drop(retained);
        let transcode = resources.reserve_codecs(2).unwrap();
        assert_eq!(resources.live_codecs(), 2);
        drop(transcode);
        assert_eq!(resources.live_codecs(), 0);
    }
    #[test]
    fn known_native_bytes_are_shared_and_returned_only_after_final_guard() {
        let resources = CudaResources::new(
            CudaSessionLimits {
                max_codec_sessions: 2,
                max_native_bytes: 10,
            },
            Vec::new(),
        )
        .unwrap();
        let first = Arc::new(resources.reserve_native(6).unwrap());
        let retained = first.clone();
        assert!(resources.reserve_native(5).is_err());
        let second = resources.reserve_native(4).unwrap();
        assert_eq!(resources.peak_native_bytes(), 10);
        drop(first);
        drop(second);
        assert_eq!(resources.native_bytes(), 6);
        drop(retained);
        assert_eq!(resources.native_bytes(), 0);
    }
}
