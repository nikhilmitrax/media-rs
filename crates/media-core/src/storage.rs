//! Shared allocation ownership and producer readiness.
use crate::{
    device::{Device, DeviceQueue},
    error::{Error, Result},
};
use std::{fmt, sync::Arc};
/// Native allocation representation, independent of pixel/sample format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryKind {
    /// Ordinary host bytes.
    Host,
    /// Backend-owned pinned host memory.
    PinnedHost,
    /// Linear device allocation.
    DeviceLinear,
    /// Pitched device allocation.
    DevicePitched,
    /// Non-addressable native surface.
    Opaque,
}
/// Producer dependency retaining its native context.
pub trait Completion: fmt::Debug + Send + Sync {
    /// Query without waiting.
    fn is_complete(&self) -> Result<bool>;
    /// Wait for this dependency, without a routine device-wide barrier.
    fn wait(&self) -> Result<()>;
    /// Establish compatible queue ordering, or reject an unsupported route.
    fn depend_on(&self, _queue: DeviceQueue) -> Result<()> {
        if self.is_complete()? {
            Ok(())
        } else {
            Err(Error::unsupported(
                "completion token cannot establish queue ordering",
            ))
        }
    }
}
/// Synchronously complete CPU work.
#[derive(Debug, Default)]
pub struct Ready;
impl Completion for Ready {
    fn is_complete(&self) -> Result<bool> {
        Ok(true)
    }
    fn wait(&self) -> Result<()> {
        Ok(())
    }
}
/// Backend allocation owner.
///
/// Native owners must retain decoder/pool/context references and retire native
/// resources safely. Mutable host access requires exclusivity including backend
/// references. Queued consumers must retain a lease until consumer completion;
/// producer readiness alone does not establish consumer retirement.
pub trait StorageOwner: fmt::Debug + Send + Sync {
    /// Addressable allocation length or known opaque accounting size.
    fn byte_len(&self) -> usize;
    /// Allocation representation.
    fn kind(&self) -> MemoryKind;
    /// Actual owning device/context.
    fn device(&self) -> Device;
    /// Existing host view; no implicit device transfer.
    fn host_bytes(&self) -> Option<&[u8]> {
        None
    }
    /// Exclusive host view only when native references allow mutation.
    fn host_bytes_mut(&mut self) -> Option<&mut [u8]> {
        None
    }
}
#[derive(Debug)]
struct HostBytes(Vec<u8>);
impl StorageOwner for HostBytes {
    fn byte_len(&self) -> usize {
        self.0.len()
    }
    fn kind(&self) -> MemoryKind {
        MemoryKind::Host
    }
    fn device(&self) -> Device {
        Device::Cpu
    }
    fn host_bytes(&self) -> Option<&[u8]> {
        Some(&self.0)
    }
    fn host_bytes_mut(&mut self) -> Option<&mut [u8]> {
        Some(&mut self.0)
    }
}
/// Cloneable allocation ownership plus producer readiness.
#[derive(Debug, Clone)]
pub struct StorageLease {
    owner: Arc<dyn StorageOwner>,
    readiness: Arc<dyn Completion>,
    retained: Vec<Arc<dyn RetainedGuard>>,
}
trait RetainedGuard: fmt::Debug + Send + Sync {}
impl<T: fmt::Debug + Send + Sync> RetainedGuard for T {}
impl StorageLease {
    /// Own synchronous host bytes without copying them.
    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        Self {
            owner: Arc::new(HostBytes(bytes)),
            readiness: Arc::new(Ready),
            retained: Vec::new(),
        }
    }
    /// Retain a native owner and producer dependency.
    pub fn from_owner(owner: Arc<dyn StorageOwner>, readiness: Arc<dyn Completion>) -> Self {
        Self {
            owner,
            readiness,
            retained: Vec::new(),
        }
    }
    /// Retain a resource guard until every clone of this lease is released.
    pub fn retain<T: fmt::Debug + Send + Sync + 'static>(mut self, guard: T) -> Self {
        self.retained.push(Arc::new(guard));
        self
    }
    /// Allocation/accounting size.
    pub fn byte_len(&self) -> usize {
        self.owner.byte_len()
    }
    /// Allocation representation.
    pub fn kind(&self) -> MemoryKind {
        self.owner.kind()
    }
    /// Owning placement.
    pub fn device(&self) -> Device {
        self.owner.device()
    }
    /// Retained producer dependency.
    pub fn readiness(&self) -> &Arc<dyn Completion> {
        &self.readiness
    }
    /// Wait for readiness and borrow existing host bytes.
    pub fn host_bytes(&self) -> Result<&[u8]> {
        self.readiness.wait()?;
        self.owner.host_bytes().ok_or_else(|| {
            Error::incompatible("no host byte view; request an explicit transfer or conversion")
        })
    }
    /// Mutate only when application aliases and backend references permit it.
    pub fn try_host_bytes_mut(&mut self) -> Result<&mut [u8]> {
        self.readiness.wait()?;
        Arc::get_mut(&mut self.owner)
            .and_then(|owner| owner.host_bytes_mut())
            .ok_or_else(|| {
                Error::incompatible(
                    "storage is shared or not exclusively host-addressable; detach explicitly",
                )
            })
    }
    /// Whether both handles share an allocation.
    pub fn shares_allocation(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.owner, &other.owner)
    }
    /// Explicitly duplicate host bytes.
    pub fn deep_copy_host(&self) -> Result<Self> {
        Ok(Self::from_bytes(self.host_bytes()?.to_vec()))
    }
}
