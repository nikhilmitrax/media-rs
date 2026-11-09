//! Common public media types and request geometry.
#[cfg(feature = "native")]
pub use crate::audio_request::Window;
#[cfg(feature = "native")]
pub use crate::audio_request::{
    AudioPlan, AudioSamplesRequest, AudioSamplesStream, AudioSamplesWriteResult,
};
#[cfg(all(feature = "native", feature = "ffmpeg"))]
pub use crate::clip_request::{AudioOutput, ClipResult, Sampling, VideoOutput};
#[cfg(feature = "native")]
pub use crate::image_request::ColorResizeSpec;
pub use crate::request::{Asset, Session, SessionBuilder};
#[cfg(feature = "native")]
pub use crate::sequence_request::{
    AnimationBatch, AnimationEncodeOptions, AnimationFormat, LoopCount, PageBatch,
    SequenceEncodeLimits, SequenceEncodedResult, SequenceLimits, SequenceWriteResult,
    TiffEncodeOptions,
};
#[cfg(feature = "ffmpeg")]
pub use media_backend_ffmpeg::encode::{CpuVideoCodec, ResizeFilter, TranscodeOptions};
#[cfg(feature = "native")]
pub use media_backend_native::audio_samples::{
    ConversionReport as AudioConversionReport, ConvertedAudio, SampleFormat, SampleGamut,
    SampleOutput, SamplePacking, SampleSpec, TpdfDither,
};
#[cfg(feature = "native")]
pub use media_backend_native::image::{EncodeOptions, ImageFormat, MetadataPolicy};
#[cfg(feature = "native")]
pub use media_backend_native::image_color::{
    AlphaMode, ColorConvertSpec, ColorSpace, GamutPolicy, ResizeDomain as ColorResizeDomain,
};
#[cfg(feature = "native")]
pub use media_backend_native::image_pixels::{
    AlphaChange, FloatRange, PadSpec, PixelChange, PixelConvertSpec, PixelFill, Quantization,
};
pub use media_core::{
    error::{Error, ErrorKind, Result},
    media::{AudioBlock, ChannelLayout, ChannelPosition, Image, PixelFormat, TrackId, VideoFrame},
    policy::{Boundary, CopyPolicy, DecodePolicy, FrameDurationPolicy, Placement, TransferPolicy},
    time::{Rounding, Time, TimeBase, TimeRange},
};
#[cfg(feature = "interop")]
pub use media_interop::tensor::{
    AlphaPolicy, ColorAssumption, IccGamut, IccLimits, IccPolicy, NumericRange, Tensor,
    TensorLayout, TensorSpec,
};
pub use media_pipeline::sampling::{DurationInference, TemporalSelection};
pub use media_pipeline::{
    operation::{Filter, Fit},
    session::Limits,
};
