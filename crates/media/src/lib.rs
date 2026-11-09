//! Bounded multimedia requests, shared contracts, and optional native backends.

#[doc(inline)]
pub use media_core as core;
#[doc(inline)]
pub use media_io as io;
#[doc(inline)]
pub use media_pipeline as pipeline;

/// Optional backend implementations and native escape hatches.
pub mod backend {
    #[cfg(feature = "cuda")]
    pub use media_backend_cuda as cuda;
    #[cfg(feature = "ffmpeg")]
    pub use media_backend_ffmpeg as ffmpeg;
    #[cfg(feature = "native")]
    pub use media_backend_native as native;
}

#[cfg(feature = "interop")]
#[doc(inline)]
pub use media_interop as interop;

pub mod prelude;
pub mod reader;
pub mod request;
pub mod writer;

#[cfg(feature = "native")]
pub mod image_request;

#[cfg(feature = "native")]
pub mod sequence_request;

#[cfg(feature = "native")]
pub mod audio_request;

#[cfg(feature = "ffmpeg")]
pub mod video_index;
#[cfg(feature = "ffmpeg")]
pub mod video_request;

#[cfg(all(feature = "native", feature = "ffmpeg"))]
pub mod clip_request;

#[cfg(all(feature = "ffmpeg", feature = "cuda"))]
pub mod cuda_request;
#[cfg(feature = "cuda")]
pub mod cuda_session;

#[cfg(feature = "interop")]
pub mod tensor_request;

pub use media_core::error::{Error, Result};
pub use request::{Asset, Session, SessionBuilder};

/// Open a local asset in a default bounded session without decoding it.
pub fn open(path: impl AsRef<std::path::Path>) -> Result<Asset> {
    Session::default().open(path)
}
