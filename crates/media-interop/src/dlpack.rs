//! Owned CPU and optional CUDA DLPack exports using legacy and versioned C ABIs.
//!
//! Layout follows the primary DLPack 1.3 header:
//! <https://github.com/dmlc/dlpack/blob/main/include/dlpack/dlpack.h>
//! These are C descriptors, not Python capsules or Python protocol objects.
//! Only uniquely owned f32 storage can be exported for foreign writes.
//! The consumer must retain the managed descriptor until all reads/writes complete,
//! then call its deleter exactly once. CPU producer work is already complete.
//! CUDA exports order producer readiness onto one checked consumer stream; their
//! deleter waits for work submitted to that stream before retiring the allocation.
//! No Python capsule, implicit device download, or device-wide barrier is provided.
#![allow(unsafe_code)]

use crate::tensor::{Tensor, TensorParts};
use media_core::error::{Error, Result};
use std::{ffi::c_void, ptr::NonNull};

/// DLPack ABI/API version.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DLPackVersion {
    /// ABI major version.
    pub major: u32,
    /// API minor version.
    pub minor: u32,
}
/// Native device descriptor, using the DLPack integer device type.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DLDevice {
    /// CPU is 1; actual CUDA device storage is 2.
    pub device_type: i32,
    /// Actual CUDA device ordinal; ordinary CPU storage uses zero.
    pub device_id: i32,
}
/// Native-endian scalar data type.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DLDataType {
    /// IEEE floating point is code 2.
    pub code: u8,
    /// Scalar width, 32 for this adapter.
    pub bits: u8,
    /// Scalar lanes, always 1.
    pub lanes: u16,
}
/// Non-owning DLPack tensor descriptor.
#[repr(C)]
pub struct DLTensor {
    /// Aligned logical data address, or null for an empty tensor.
    pub data: *mut c_void,
    /// Actual allocation device.
    pub device: DLDevice,
    /// Number of dimensions.
    pub ndim: i32,
    /// Native-endian scalar representation.
    pub dtype: DLDataType,
    /// Exactly ndim dimensions, or null for a scalar.
    pub shape: *mut i64,
    /// Exactly ndim element strides; non-null for rank > 0, including contiguous.
    pub strides: *mut i64,
    /// Offset from data in bytes, zero for these aligned allocations.
    pub byte_offset: u64,
}
/// Deprecated, pre-1.0 managed C descriptor, available for explicit compatibility.
#[repr(C)]
pub struct DLManagedTensor {
    /// Borrowed tensor description.
    pub dl_tensor: DLTensor,
    /// Private producer-owned state; consumers must not replace this pointer.
    pub manager_ctx: *mut c_void,
    /// Must be called once after all consumer use, with the original descriptor.
    pub deleter: Option<unsafe extern "C" fn(*mut DLManagedTensor)>,
}
/// Current versioned C descriptor, exported as DLPack 1.3.
#[repr(C)]
pub struct DLManagedTensorVersioned {
    /// ABI/API version; unsupported major versions must be released via deleter.
    pub version: DLPackVersion,
    /// Private producer-owned state; consumers must not replace this pointer.
    pub manager_ctx: *mut c_void,
    /// Must be called once after all consumer use, with the original descriptor.
    pub deleter: Option<unsafe extern "C" fn(*mut DLManagedTensorVersioned)>,
    /// Writable; zero for sharing unique storage, IS_COPIED for explicit detachment.
    pub flags: u64,
    /// Borrowed tensor description.
    pub dl_tensor: DLTensor,
}

struct Context {
    parts: TensorParts,
    shape: Vec<i64>,
    strides: Vec<i64>,
}
impl Context {
    fn new(tensor: Tensor) -> Result<Box<Self>> {
        let shape = tensor
            .shape()
            .iter()
            .map(|v| i64::try_from(*v).map_err(|_| Error::overflow("DLPack dimension exceeds i64")))
            .collect::<Result<Vec<_>>>()?;
        let strides = tensor
            .strides()
            .iter()
            .map(|v| i64::try_from(*v).map_err(|_| Error::overflow("DLPack stride exceeds i64")))
            .collect::<Result<Vec<_>>>()?;
        Ok(Box::new(Self {
            parts: tensor.into_unique()?,
            shape,
            strides,
        }))
    }
    fn descriptor(&mut self) -> DLTensor {
        let empty = self.parts.shape.contains(&0);
        let scalar = self.shape.is_empty();
        DLTensor {
            data: if empty {
                std::ptr::null_mut()
            } else {
                self.parts
                    .data
                    .as_mut_ptr()
                    .wrapping_add(self.parts.offset)
                    .cast::<c_void>()
            },
            device: DLDevice {
                device_type: 1,
                device_id: 0,
            },
            ndim: self.shape.len() as i32,
            dtype: DLDataType {
                code: 2,
                bits: 32,
                lanes: 1,
            },
            shape: if scalar {
                std::ptr::null_mut()
            } else {
                self.shape.as_mut_ptr()
            },
            strides: if scalar {
                std::ptr::null_mut()
            } else {
                self.strides.as_mut_ptr()
            },
            byte_offset: 0,
        }
    }
}

/// RAII owner of an unconsumed versioned descriptor.
///
/// Foreign code may inspect the borrowed pointer but must not release it unless
/// ownership has first been transferred with into_raw. Dropping the wrapper
/// releases an unconsumed descriptor and all retained allocation/context guards.
pub struct VersionedExport {
    raw: NonNull<DLManagedTensorVersioned>,
}
impl VersionedExport {
    /// Borrow the descriptor pointer; ownership stays with this wrapper.
    pub fn as_ptr(&self) -> *mut DLManagedTensorVersioned {
        self.raw.as_ptr()
    }
    /// Transfer the descriptor and its lifetime to a C-ABI consumer.
    ///
    /// The consumer must use the original shape/stride/manager/deleter fields
    /// unchanged and call the deleter once. Host accesses must finish before
    /// release. CUDA work must all be submitted on the selected consumer stream
    /// before release; the deleter waits for that stream's recorded completion.
    pub fn into_raw(self) -> *mut DLManagedTensorVersioned {
        let raw = self.raw.as_ptr();
        std::mem::forget(self);
        raw
    }
}
impl Drop for VersionedExport {
    fn drop(&mut self) {
        // SAFETY: This wrapper uniquely owns its live, untransferred descriptor.
        unsafe {
            let raw = self.raw.as_ptr();
            ((*raw).deleter.expect("export always supplies a deleter"))(raw);
        }
    }
}

/// RAII owner of an unconsumed legacy descriptor; prefer VersionedExport.
pub struct LegacyExport {
    raw: NonNull<DLManagedTensor>,
}
impl LegacyExport {
    /// Borrow the descriptor pointer without transferring ownership.
    pub fn as_ptr(&self) -> *mut DLManagedTensor {
        self.raw.as_ptr()
    }
    /// Transfer ownership; the C consumer must eventually call deleter once.
    pub fn into_raw(self) -> *mut DLManagedTensor {
        let raw = self.raw.as_ptr();
        std::mem::forget(self);
        raw
    }
}
impl Drop for LegacyExport {
    fn drop(&mut self) {
        // SAFETY: This wrapper uniquely owns its live, untransferred descriptor.
        unsafe {
            let raw = self.raw.as_ptr();
            ((*raw).deleter.expect("export always supplies a deleter"))(raw);
        }
    }
}

/// Export a unique, writable host f32 allocation using DLPack 1.3.
///
/// Cloned aliases are rejected; use Tensor::deep_copy explicitly when independent
/// writable storage is required. No extra numeric copy or CPU/GPU transfer occurs.
pub fn export_versioned(tensor: Tensor) -> Result<VersionedExport> {
    let mut context = Context::new(tensor)?;
    let descriptor = context.descriptor();
    let manager_ctx = Box::into_raw(context).cast::<c_void>();
    let managed = Box::new(DLManagedTensorVersioned {
        version: DLPackVersion { major: 1, minor: 3 },
        manager_ctx,
        deleter: Some(release_versioned),
        flags: 0,
        dl_tensor: descriptor,
    });
    Ok(VersionedExport {
        raw: NonNull::from(Box::leak(managed)),
    })
}

/// Export explicitly to the deprecated legacy ABI, with the same uniqueness rule.
pub fn export_legacy(tensor: Tensor) -> Result<LegacyExport> {
    let mut context = Context::new(tensor)?;
    let descriptor = context.descriptor();
    let manager_ctx = Box::into_raw(context).cast::<c_void>();
    let managed = Box::new(DLManagedTensor {
        dl_tensor: descriptor,
        manager_ctx,
        deleter: Some(release_legacy),
    });
    Ok(LegacyExport {
        raw: NonNull::from(Box::leak(managed)),
    })
}

/// DLPack 1.3 flag indicating an explicit copy made by the producer.
pub const FLAG_IS_COPIED: u64 = 1 << 1;

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use crate::native::{NativeCudaTensor, export_cuda_tensor};
    use media_backend_cuda::{CudaStream, CudaTensor};
    use media_core::{device::Device, policy::CopyPolicy};

    struct CudaContext {
        owner: NativeCudaTensor,
        shape: Vec<i64>,
        strides: Vec<i64>,
        address: usize,
        ordinal: i32,
    }

    impl CudaContext {
        fn new(tensor: CudaTensor, consumer: &CudaStream, copy: CopyPolicy) -> Result<Box<Self>> {
            let owner = export_cuda_tensor(tensor, consumer, copy)?;
            let descriptor = owner.descriptor();
            let shape = descriptor
                .shape()
                .into_iter()
                .map(|n| i64::try_from(n).map_err(|_| Error::overflow("DLPack CUDA dimension")))
                .collect::<Result<Vec<_>>>()?;
            let strides = descriptor
                .strides()
                .into_iter()
                .map(|n| i64::try_from(n).map_err(|_| Error::overflow("DLPack CUDA stride")))
                .collect::<Result<Vec<_>>>()?;
            let Device::Cuda { ordinal, .. } = descriptor.device() else {
                return Err(Error::incompatible(
                    "DLPack CUDA export requires actual CUDA storage",
                ));
            };
            let ordinal =
                i32::try_from(ordinal).map_err(|_| Error::overflow("DLPack CUDA ordinal"))?;
            let address = usize::try_from(descriptor.address())
                .map_err(|_| Error::overflow("DLPack CUDA address cannot fit native pointer"))?;
            Ok(Box::new(Self {
                owner,
                shape,
                strides,
                address,
                ordinal,
            }))
        }

        fn descriptor(&mut self) -> DLTensor {
            DLTensor {
                data: self.address as *mut c_void,
                device: DLDevice {
                    device_type: 2,
                    device_id: self.ordinal,
                },
                ndim: 4,
                dtype: DLDataType {
                    code: 2,
                    bits: 32,
                    lanes: 1,
                },
                shape: self.shape.as_mut_ptr(),
                strides: self.strides.as_mut_ptr(),
                byte_offset: 0,
            }
        }
    }

    /// Export actual CUDA f32 RGB storage using DLPack 1.3 without a download.
    ///
    /// The consumer stream must have the allocation's exact native context.
    /// Producer/earlier reader events are ordered onto that stream. Forbid
    /// rejects allocation aliases (including buffer clones and storage leases);
    /// Allow explicitly detaches aliased storage on the same GPU, setting
    /// FLAG_IS_COPIED. A unique allocation is shared directly with flags zero.
    ///
    /// All foreign work must use this consumer stream and be submitted before
    /// calling the deleter. The deleter records/waits its completion before
    /// freeing storage, even when the caller releases immediately after launch.
    /// Retained allocation/context/budget guards survive early Rust wrapper drops.
    pub fn export_cuda_versioned(
        tensor: CudaTensor,
        consumer: &CudaStream,
        copy: CopyPolicy,
    ) -> Result<VersionedExport> {
        let mut context = CudaContext::new(tensor, consumer, copy)?;
        let flags = if context.owner.descriptor().copied() {
            FLAG_IS_COPIED
        } else {
            0
        };
        let descriptor = context.descriptor();
        let managed = Box::new(DLManagedTensorVersioned {
            version: DLPackVersion { major: 1, minor: 3 },
            manager_ctx: Box::into_raw(context).cast::<c_void>(),
            deleter: Some(release_cuda_versioned),
            flags,
            dl_tensor: descriptor,
        });
        Ok(VersionedExport {
            raw: NonNull::from(Box::leak(managed)),
        })
    }

    /// Export actual CUDA storage using the deprecated legacy ABI.
    ///
    /// Ownership, uniqueness, stream/context checks, and consumer-completion
    /// retirement match export_cuda_versioned. The legacy ABI has no copied flag;
    /// pass Forbid if an explicitly detached allocation is unacceptable.
    pub fn export_cuda_legacy(
        tensor: CudaTensor,
        consumer: &CudaStream,
        copy: CopyPolicy,
    ) -> Result<LegacyExport> {
        let mut context = CudaContext::new(tensor, consumer, copy)?;
        let descriptor = context.descriptor();
        let managed = Box::new(DLManagedTensor {
            dl_tensor: descriptor,
            manager_ctx: Box::into_raw(context).cast::<c_void>(),
            deleter: Some(release_cuda_legacy),
        });
        Ok(LegacyExport {
            raw: NonNull::from(Box::leak(managed)),
        })
    }

    unsafe extern "C" fn release_cuda_versioned(raw: *mut DLManagedTensorVersioned) {
        if raw.is_null() {
            return;
        }
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // SAFETY: Only the matching CUDA exporter provides this callback.
            // The consumer returns its original live descriptor exactly once,
            // after the final submission on the retained consumer stream.
            unsafe {
                let managed = Box::from_raw(raw);
                let context = Box::from_raw(managed.manager_ctx.cast::<CudaContext>());
                drop(managed);
                // ForeignCudaTensor Drop records/waits the consumer stream before
                // releasing allocation ownership and its device-byte reservation.
                drop(context);
            }
        }));
    }

    unsafe extern "C" fn release_cuda_legacy(raw: *mut DLManagedTensor) {
        if raw.is_null() {
            return;
        }
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // SAFETY: Same original-pointer, exact-once, final-submission contract
            // as the versioned CUDA callback, with its matching legacy type.
            unsafe {
                let managed = Box::from_raw(raw);
                let context = Box::from_raw(managed.manager_ctx.cast::<CudaContext>());
                drop(managed);
                drop(context);
            }
        }));
    }
}

#[cfg(feature = "cuda")]
pub use cuda::{export_cuda_legacy, export_cuda_versioned};

// These callbacks are only supplied with descriptors allocated by the matching
// export function. They must be called once with the original, live descriptor;
// callers may mutate data, but never the bookkeeping pointers or shape/strides.
unsafe extern "C" fn release_versioned(raw: *mut DLManagedTensorVersioned) {
    if raw.is_null() {
        return;
    }
    // Prevent a user-supplied retained guard's Drop panic crossing the C boundary.
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // SAFETY: The consumer contract returns the original unique descriptor
        // exactly once, so both pointers were produced by matching Box::into_raw.
        unsafe {
            let managed = Box::from_raw(raw);
            let context = Box::from_raw(managed.manager_ctx.cast::<Context>());
            drop(managed);
            drop(context);
        }
    }));
}
unsafe extern "C" fn release_legacy(raw: *mut DLManagedTensor) {
    if raw.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // SAFETY: Same producer/allocation/one-call invariant as versioned export.
        unsafe {
            let managed = Box::from_raw(raw);
            let context = Box::from_raw(managed.manager_ctx.cast::<Context>());
            drop(managed);
            drop(context);
        }
    }));
}
