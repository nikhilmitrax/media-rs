//! Optional native FFmpeg demuxing, decoding, and container operations.
//!
//! Enable `linked` to link system FFmpeg libraries. Without it, portable builds
//! can inspect the backend's compile-time availability through `hardware`.

#[cfg(feature = "audio")]
pub mod audio;
#[cfg(feature = "audio")]
#[allow(unsafe_code)]
mod audio_native;
#[cfg(feature = "linked")]
pub mod codec;
#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "cuda")]
pub mod cuda_encode;
#[cfg(feature = "cuda-filters")]
mod cuda_scale;
#[cfg(feature = "linked")]
pub mod encode;
#[cfg(feature = "linked")]
pub mod format;
pub mod hardware;
#[cfg(feature = "linked")]
pub mod packet_decode;
#[cfg(feature = "linked")]
pub mod video_writer;

#[cfg(feature = "linked")]
#[allow(unsafe_code)]
mod native;
