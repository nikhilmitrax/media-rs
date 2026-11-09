//! Checked host image views, explicit tensor packing, and owned foreign exports.
//!
//! The optional `cuda` feature exports actual CUDA tensors with checked native
//! context, exclusive writable storage, and consumer-stream completion ownership.
//! The optional `icc` feature adds explicit bounded ICC-to-sRGB tensor conversion.

pub mod color;
pub mod dlpack;
pub mod image;
pub mod native;
pub mod ndarray;
pub mod tensor;
