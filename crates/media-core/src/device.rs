//! Native context identity and placement, independent of hardware discovery.
use crate::error::{Error, Result};
/// Backend-assigned identity for an actual native context.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ContextId(u64);
impl ContextId {
    /// Construct a nonzero context identity.
    pub fn new(value: u64) -> Result<Self> {
        if value == 0 {
            return Err(Error::invalid("native context identity must be nonzero"));
        }
        Ok(Self(value))
    }
    /// Opaque identity value.
    pub fn get(self) -> u64 {
        self.0
    }
}
/// Established allocation placement; construction does not discover hardware.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum Device {
    /// Ordinary host placement.
    #[default]
    Cpu,
    /// A CUDA ordinal and the actual owning context.
    Cuda {
        /// Process-local GPU ordinal.
        ordinal: u32,
        /// Backend-established native context identity.
        context: ContextId,
    },
}
impl Device {
    /// Whether bytes are in host memory.
    pub fn is_host(self) -> bool {
        matches!(self, Self::Cpu)
    }
}
/// Queue identity for native ordering compatibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeviceQueue {
    /// Actual device/context.
    pub device: Device,
    /// Backend-assigned queue identity.
    pub id: u64,
}
