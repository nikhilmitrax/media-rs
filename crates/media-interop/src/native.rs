//! Checked native tensor ownership without a host transfer or an ordinal-only pointer.
//!
//! CUDA adapters require the `cuda` feature. Native video surfaces must first be
//! converted to an actual compatible tensor; an opaque surface is not a tensor.

#[cfg(feature = "cuda")]
mod cuda {
    use crate::tensor::TensorLayout;
    use media_backend_cuda::{CudaStream, CudaTensor, ForeignCudaTensor};
    use media_core::{
        device::Device,
        error::{Error, Result},
        policy::CopyPolicy,
    };

    /// Established physical element format of a native tensor allocation.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub enum NativeTensorFormat {
        /// Native-endian IEEE f32 RGB, with channel position given by layout.
        RgbF32,
    }

    /// Established native scalar representation.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub enum NativeTensorDType {
        /// Native-endian IEEE floating point, 32 bits, one scalar lane.
        F32,
    }

    /// Checked physical tensor description in its actual owning CUDA context.
    ///
    /// The address is a device address, never a host pointer. Its validity is
    /// bounded by the owning NativeCudaTensor or foreign managed descriptor.
    /// Consumer work must use the stream selected during export, and stop
    /// submitting work before that owner is released.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct NativeTensorDescriptor {
        address: u64,
        device: Device,
        format: NativeTensorFormat,
        shape: [usize; 4],
        strides: [usize; 4],
        layout: TensorLayout,
        byte_len: usize,
        copied: bool,
    }

    impl NativeTensorDescriptor {
        /// Device address of the first f32 element; never synthesized from an ordinal.
        pub fn address(&self) -> u64 {
            self.address
        }
        /// Actual device ordinal and native context identity.
        pub fn device(&self) -> Device {
            self.device
        }
        /// Physical scalar and channel representation.
        pub fn format(&self) -> NativeTensorFormat {
            self.format
        }
        /// Scalar representation established by the typed CUDA allocation.
        pub fn dtype(&self) -> NativeTensorDType {
            NativeTensorDType::F32
        }
        /// Physical dimensions, including the leading batch dimension.
        pub fn shape(&self) -> [usize; 4] {
            self.shape
        }
        /// Strides in f32 elements, not bytes.
        pub fn strides(&self) -> [usize; 4] {
            self.strides
        }
        /// Physical RGB channel-axis packing.
        pub fn layout(&self) -> TensorLayout {
            self.layout
        }
        /// Owned device allocation extent in bytes.
        pub fn byte_len(&self) -> usize {
            self.byte_len
        }
        /// Whether export detached aliased storage through an explicit GPU copy.
        pub fn copied(&self) -> bool {
            self.copied
        }
    }

    /// Exclusive foreign-writable tensor lease ordered onto one real consumer stream.
    ///
    /// This type cannot be cloned or converted back into a shared allocation.
    /// Dropping it records and waits for work already submitted to that stream
    /// before releasing allocation/context/budget guards. It does not use a
    /// device-wide barrier. Foreign code must stop submitting uses before drop.
    #[derive(Debug)]
    pub struct NativeCudaTensor {
        descriptor: NativeTensorDescriptor,
        _owner: ForeignCudaTensor,
    }

    impl NativeCudaTensor {
        /// Borrow the established address, format, shape, strides, and context.
        pub fn descriptor(&self) -> &NativeTensorDescriptor {
            &self.descriptor
        }
    }

    /// Consume an actual CUDA tensor for exclusive native read/write use.
    ///
    /// The consumer must belong to the allocation's exact native context.
    /// Producer work is ordered by GPU event onto that stream. Forbid rejects
    /// every allocation alias, including tensor/buffer clones and storage leases;
    /// Allow may explicitly detach on the same GPU and reports that copy.
    /// No download or color/numeric conversion occurs here.
    pub fn export_cuda_tensor(
        tensor: CudaTensor,
        consumer: &CudaStream,
        copy: CopyPolicy,
    ) -> Result<NativeCudaTensor> {
        let shape = tensor.shape();
        let strides = tensor.strides();
        let layout = match tensor.layout() {
            media_backend_cuda::TensorLayout::Nchw => TensorLayout::Nchw,
            media_backend_cuda::TensorLayout::Nhwc => TensorLayout::Nhwc,
        };
        let bytes = shape
            .iter()
            .try_fold(4usize, |bytes, dimension| bytes.checked_mul(*dimension))
            .ok_or_else(|| Error::overflow("native CUDA tensor size"))?;
        if bytes != tensor.buffer().byte_len() {
            return Err(Error::incompatible(
                "native CUDA tensor shape does not match its allocation",
            ));
        }
        let owner = tensor.into_foreign_writable(consumer, copy)?;
        let descriptor = NativeTensorDescriptor {
            address: owner.device_pointer(),
            device: owner.device(),
            format: NativeTensorFormat::RgbF32,
            shape,
            strides,
            layout,
            byte_len: bytes,
            copied: owner.copied(),
        };
        if descriptor.address == 0 || !descriptor.address.is_multiple_of(256) {
            return Err(Error::incompatible(
                "native CUDA tensor allocation must have a nonzero 256-byte-aligned address",
            ));
        }
        Ok(NativeCudaTensor {
            descriptor,
            _owner: owner,
        })
    }
}

#[cfg(feature = "cuda")]
pub use cuda::{
    NativeCudaTensor, NativeTensorDType, NativeTensorDescriptor, NativeTensorFormat,
    export_cuda_tensor,
};
