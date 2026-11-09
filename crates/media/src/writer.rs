//! Writers with explicit completion and finite encoded destinations.
#[cfg(feature = "ffmpeg")]
pub use media_backend_ffmpeg::video_writer::{
    VideoDrainState, VideoDurationPolicy, VideoTimestamp, VideoWriteState, VideoWriter,
    VideoWriterOptions, VideoWriterReport,
};
#[cfg(feature = "native")]
pub use media_backend_native::audio::WavWriter;
pub use media_io::sink::{FinishedSink, Sink};
