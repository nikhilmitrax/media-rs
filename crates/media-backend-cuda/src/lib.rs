//! Explicit CUDA transfers, retained device allocations, and video preprocessing.
//!
//! Driver and NVRTC libraries are loaded dynamically. Building does not require a
//! CUDA installation; executing GPU operations requires actual supported hardware.
//! This backend does not perform hardware video decoding or choose CPU fallback.

pub mod context;
pub mod kernel;
pub mod memory;
pub mod sync;

pub use context::{CudaContext, CudaLimits, CudaStream};
pub use kernel::{
    ChromaInterpolation, ChromaSiting, CropRect, CudaTensor, CudaYuvImage, ForeignCudaTensor,
    Interpolation, Normalization, PreprocessSpec, ResizeDomain, TensorLayout, YuvMatrix, YuvRange,
    preprocess_batch,
};
pub use memory::CudaBuffer;
pub use sync::{ConsumerLease, CudaCompletion};
