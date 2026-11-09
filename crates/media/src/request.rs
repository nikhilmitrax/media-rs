//! Public assets, bounded sessions, and lazy spatial requests.
use media_core::error::Result;
use media_io::source::Source;
use media_pipeline::session::{Limits, Session as Resources};
use std::path::Path;

/// Shared budgets and backend resources; executions own independent cursors.
#[derive(Debug, Clone, Default)]
pub struct Session {
    pub(crate) resources: Resources,
    #[cfg(feature = "cuda")]
    pub(crate) cuda: std::sync::Arc<crate::cuda_session::CudaResources>,
}
impl Session {
    /// Begin configuring a bounded session.
    pub fn builder() -> SessionBuilder {
        SessionBuilder::default()
    }
    /// Open a local asset without decoding it.
    pub fn open(&self, path: impl AsRef<Path>) -> Result<Asset> {
        Ok(self.from_source(Source::file(path)?))
    }
    /// Retain an existing source adapter without decoding it.
    pub fn from_source(&self, source: Source) -> Asset {
        Asset {
            source,
            session: self.clone(),
        }
    }
    /// Current accounted host allocations retained across executions.
    pub fn live_host_bytes(&self) -> usize {
        self.resources.host_budget().current()
    }
    /// High-water mark for library-accounted host payloads and working reservations.
    pub fn peak_host_bytes(&self) -> usize {
        self.resources.host_budget().peak()
    }
    /// Materialized video/tensor batches retained by outputs or foreign consumers.
    pub fn live_batches(&self) -> usize {
        self.resources.live_batches()
    }
    /// Active CPU facade scopes, known codec contexts and requested workers.
    /// Opaque native/demux-probe/FFT workers and direct backend calls are excluded.
    pub fn live_cpu_usage(&self) -> media_pipeline::session::CpuUsage {
        self.resources.live_cpu_usage()
    }
    /// Configured limits.
    pub fn limits(&self) -> Limits {
        self.resources.limits()
    }
    /// Lazily obtain this Session's canonical context with matching immutable
    /// allocation/stream limits. Native bootstrap runs before source callbacks.
    #[cfg(feature = "cuda")]
    pub fn cuda_context(
        &self,
        ordinal: u32,
        limits: media_backend_cuda::CudaLimits,
    ) -> Result<media_backend_cuda::CudaContext> {
        self.cuda.context(ordinal, limits)
    }
    /// Native codec sessions retained by readers, transcodes, and exported frames.
    #[cfg(feature = "cuda")]
    pub fn live_cuda_codec_sessions(&self) -> usize {
        self.cuda.live_codecs()
    }
    /// Reserved known native surface bytes and conservative transcode allowances;
    /// excludes owned tensors and opaque driver/reference reservations.
    #[cfg(feature = "cuda")]
    pub fn reserved_cuda_native_bytes(&self) -> usize {
        self.cuda.native_bytes()
    }
    /// Peak known native reservation bytes across executions of this Session.
    #[cfg(feature = "cuda")]
    pub fn peak_reserved_cuda_native_bytes(&self) -> usize {
        self.cuda.peak_native_bytes()
    }
    /// Aggregate native codec and known-surface reservation bounds.
    #[cfg(feature = "cuda")]
    pub fn cuda_limits(&self) -> crate::cuda_session::CudaSessionLimits {
        self.cuda.limits()
    }
}
/// Session configuration.
#[derive(Debug, Clone, Default)]
pub struct SessionBuilder {
    limits: Limits,
    #[cfg(feature = "cuda")]
    cuda_limits: crate::cuda_session::CudaSessionLimits,
    #[cfg(feature = "cuda")]
    cuda_contexts: Vec<media_backend_cuda::CudaContext>,
}
impl SessionBuilder {
    /// Replace resource limits.
    pub fn limits(mut self, limits: Limits) -> Self {
        self.limits = limits;
        self
    }
    /// Configure aggregate native codec admission and known retained-surface bytes.
    #[cfg(feature = "cuda")]
    pub fn cuda_limits(mut self, limits: crate::cuda_session::CudaSessionLimits) -> Self {
        self.cuda_limits = limits;
        self
    }
    /// Register an application-owned actual context and its shared allocator.
    /// One canonical wrapper per ordinal is validated during build().
    #[cfg(feature = "cuda")]
    pub fn cuda_context(mut self, context: media_backend_cuda::CudaContext) -> Self {
        self.cuda_contexts.push(context);
        self
    }
    /// Validate and create shared resources.
    pub fn build(self) -> Result<Session> {
        Ok(Session {
            resources: Resources::new(self.limits)?,
            #[cfg(feature = "cuda")]
            cuda: std::sync::Arc::new(crate::cuda_session::CudaResources::new(
                self.cuda_limits,
                self.cuda_contexts,
            )?),
        })
    }
}
/// Cheap, cloneable source identity and session handle.
#[derive(Debug, Clone)]
pub struct Asset {
    pub(crate) source: Source,
    pub(crate) session: Session,
}
impl Asset {
    /// Byte adapter and its real source capabilities.
    pub fn source(&self) -> &Source {
        &self.source
    }
    /// Session sharing budgets and backend resources with this asset.
    pub fn session(&self) -> &Session {
        &self.session
    }
    /// Prepare a lazy still-image request.
    #[cfg(feature = "native")]
    pub fn image(&self) -> Result<crate::image_request::ImageRequest> {
        Ok(crate::image_request::ImageRequest::new(self.clone()))
    }
    /// Prepare bounded composited GIF/APNG/WebP display frames and loop timing.
    #[cfg(feature = "native")]
    pub fn animation(&self) -> crate::sequence_request::AnimationRequest {
        crate::sequence_request::AnimationRequest::new(self.clone())
    }
    /// Prepare bounded independent TIFF pages, preserving geometry and precision.
    #[cfg(feature = "native")]
    pub fn pages(&self) -> crate::sequence_request::PagesRequest {
        crate::sequence_request::PagesRequest::new(self.clone())
    }
    /// Prepare continuous audio decoding and preprocessing.
    #[cfg(feature = "native")]
    pub fn audio(&self) -> Result<crate::audio_request::AudioRequest> {
        Ok(crate::audio_request::AudioRequest::new(self.clone()))
    }
    /// Prepare video probing, sequential decode, or exact temporal sampling.
    #[cfg(feature = "ffmpeg")]
    pub fn video(&self) -> Result<crate::video_request::VideoRequest> {
        Ok(crate::video_request::VideoRequest::new(self.clone()))
    }
    /// Extract selected audio/video tracks on this container's shared timeline.
    #[cfg(all(feature = "native", feature = "ffmpeg"))]
    pub fn clip(&self, range: media_core::time::TimeRange) -> crate::clip_request::ClipRequest {
        crate::clip_request::ClipRequest::new(self.clone(), range)
    }
    /// Prepare explicit compressed-stream copying.
    #[cfg(feature = "ffmpeg")]
    pub fn remux(&self) -> crate::video_request::RemuxRequest {
        crate::video_request::RemuxRequest::new(self.clone())
    }
    /// Prepare video conversion with explicit software encoding and audio passthrough.
    #[cfg(feature = "ffmpeg")]
    pub fn convert(&self) -> crate::video_request::ConvertRequest {
        crate::video_request::ConvertRequest::new(self.clone())
    }
}
