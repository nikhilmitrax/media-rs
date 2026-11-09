//! Blocking readers and compressed packet access.
#[cfg(feature = "native")]
pub use crate::audio_request::AudioStream;
#[cfg(all(feature = "ffmpeg", feature = "cuda"))]
pub use crate::cuda_request::CudaReader;
#[cfg(feature = "ffmpeg")]
pub use crate::video_request::VideoReader;
#[cfg(feature = "ffmpeg")]
pub use media_backend_ffmpeg::format::PacketReader;
#[cfg(feature = "ffmpeg")]
pub use media_backend_ffmpeg::packet_decode::{
    CodecParameters, CodecSend, FrameReceive, PacketDecodeOptions, PacketDecoder,
    VideoDecoderConfig,
};
#[cfg(feature = "ffmpeg")]
pub use media_core::policy::FrameDurationPolicy;
