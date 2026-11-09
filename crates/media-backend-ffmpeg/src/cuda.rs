//! Required NVDEC/CUDA decoding with retained native surfaces and no host pixels.
//!
//! The decoder selects only AV_PIX_FMT_CUDA. CPU fallback and device-to-host
//! frame transfer are absent from this path. Each output retains the AVFrame,
//! its hardware-frame pool/device references, and an event recorded on FFmpeg's
//! actual CUDA stream. External consumers retain that lease until their work ends.

use std::fmt;
use std::sync::{Arc, Mutex, atomic::Ordering};

use crate::format::{self, AssetInfo, OpenedInput, ProbeOptions, TrackKind, backend_error};
use crate::native;
use ffmpeg_next as ffmpeg;
use media_backend_cuda::{
    ChromaSiting, CudaCompletion, CudaContext, CudaLimits, CudaYuvImage, YuvMatrix, YuvRange,
};
use media_core::device::{ContextId, Device};
use media_core::error::{Error, Result};
use media_core::media::{
    FrameTiming, Image, ImageLayout, Orientation, PlaneLayout, TrackId, VideoFrame,
};
use media_core::storage::{MemoryKind, StorageLease, StorageOwner};
use media_core::time::{Rounding, Time, TimeBase};
use media_io::source::Source;

/// Initialize an FFmpeg-compatible primary CUDA context without source I/O.
///
/// Native device creation establishes FFmpeg's scheduling flags first. The
/// returned actual context retains the device after the temporary native device
/// reference is released, and can be shared by independent reader executions.
pub fn create_context(ordinal: u32, limits: CudaLimits) -> Result<CudaContext> {
    limits.validate()?;
    ffmpeg::init().map_err(backend_error)?;
    let native = native::create_cuda_device(ordinal, false)?;
    let context = CudaContext::new(ordinal, limits)?;
    if context.device()
        != (Device::Cuda {
            ordinal,
            context: ContextId::new(native.description.context)?,
        })
    {
        return Err(Error::incompatible(
            "FFmpeg bootstrap and retained CUDA adapter have different native context handles",
        ));
    }
    Ok(context)
}

/// Required hardware decoding and retained-surface budgets.
#[derive(Debug, Clone, Copy)]
pub struct CudaDecodeOptions {
    /// Explicit interpretation of native packet/codec duration hints.
    pub duration_policy: media_core::policy::FrameDurationPolicy,
    /// Selected non-cover video stream, or default non-cover video.
    pub track: Option<TrackId>,
    /// Limits on source reads used for metadata discovery.
    pub probe: ProbeOptions,
    /// Maximum known allocation bytes retained by one output surface.
    pub max_frame_bytes: usize,
    /// Maximum simultaneously retained output surface leases.
    pub max_retained_frames: usize,
    /// Maximum known bytes retained by output leases.
    ///
    /// This excludes opaque driver and decoder-reference reservations, which are
    /// not claimed as enforced device-memory accounting.
    pub max_retained_bytes: usize,
}
impl Default for CudaDecodeOptions {
    fn default() -> Self {
        Self {
            duration_policy: media_core::policy::FrameDurationPolicy::Unknown,
            track: None,
            probe: ProbeOptions::default(),
            max_frame_bytes: 256 * 1024 * 1024,
            max_retained_frames: 8,
            max_retained_bytes: 512 * 1024 * 1024,
        }
    }
}

/// One addressable device plane, never a CPU pointer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaPlane {
    /// CUDA device address; only a compatible native CUDA API may dereference it.
    pub address: u64,
    /// Byte pitch between allocated rows.
    pub pitch: usize,
    /// Allocated sample bytes in each row, excluding pitch padding.
    pub row_bytes: usize,
    /// Allocated row count, including coded padding.
    pub rows: usize,
}

/// Immutable decoded surface, retained independently of reader lifetime.
#[derive(Debug, Clone)]
pub struct CudaVideoFrame {
    frame: VideoFrame,
    owner: Arc<CudaFrameOwner>,
    planes: [CudaPlane; 2],
    accounting_layout: ImageLayout,
    keyframe: bool,
    seek_timestamp: Option<Time>,
}
impl CudaVideoFrame {
    /// Retain a resource through the native image lease and queued consumers.
    pub fn retain<G: fmt::Debug + Send + Sync + 'static>(self, guard: G) -> Self {
        self.owner
            .retained
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .push(Arc::new(guard));
        self
    }
    /// Portable image/timing metadata. Its opaque storage has no host byte view.
    pub fn frame(&self) -> &VideoFrame {
        &self.frame
    }
    /// Native decoded key-frame marker, independent of compressed packet count.
    pub fn keyframe(&self) -> bool {
        self.keyframe
    }
    /// Original compressed key packet DTS, separate from presentation time.
    pub fn seek_timestamp(&self) -> Option<Time> {
        self.seek_timestamp
    }
    /// Apply caller-verified index position after matching the actual source PTS.
    /// This metadata operation does not construct or authenticate an index.
    /// Native duration evidence, image ownership and readiness remain unchanged.
    pub fn with_indexed_position(
        mut self,
        source_pts: Time,
        asset_pts: Time,
        ordinal: u64,
    ) -> Result<Self> {
        if self.frame.timing.source_pts != Some(source_pts) {
            return Err(Error::incompatible(
                "actual CUDA frame source PTS differs from verified index position",
            ));
        }
        self.frame.timing.pts = Some(asset_pts);
        self.frame.timing.ordinal = Some(ordinal);
        Ok(self)
    }
    /// Apply an explicitly verified shared timeline origin to this frame's
    /// presentation metadata. Native source PTS, duration evidence, ordinal,
    /// allocation owner, and device readiness remain unchanged.
    pub fn rebase_timeline(mut self, origin: Time) -> Result<Self> {
        let source = self.frame.timing.source_pts.ok_or_else(|| {
            Error::unsupported(
                "CUDA frame timeline rebasing requires an established native source PTS",
            )
        })?;
        self.frame.timing.pts = Some(source.checked_sub(origin)?);
        Ok(self)
    }
    /// Actual owning CUDA primary or supplied context.
    pub fn context(&self) -> &CudaContext {
        &self.owner.context
    }
    /// Retained native device plane descriptions.
    pub fn planes(&self) -> [CudaPlane; 2] {
        self.planes
    }
    /// Allocation geometry for the native planar CUDA view.
    pub fn allocation_layout(&self) -> &ImageLayout {
        &self.accounting_layout
    }
    /// Borrow a GPU view using established matrix, range, and chroma metadata.
    ///
    /// Missing or unsupported fields require an explicit interpretation through
    /// `as_yuv`; this method never fills them by guessing from image dimensions.
    pub fn yuv_view(&self) -> Result<CudaYuvImage> {
        let color = &self.frame.image.metadata().color;
        let matrix = match color.matrix.as_deref() {
            Some("bt709") => YuvMatrix::Bt709,
            Some("bt470bg" | "smpte170m") => YuvMatrix::Bt601,
            Some("bt2020nc") => YuvMatrix::Bt2020,
            _ => {
                return Err(Error::incompatible(
                    "CUDA preprocessing requires an established supported YUV matrix or explicit interpretation",
                ));
            }
        };
        let range = match color.range.as_deref() {
            Some("tv") => YuvRange::Limited,
            Some("pc") => YuvRange::Full,
            _ => {
                return Err(Error::incompatible(
                    "CUDA preprocessing requires an established YUV range or explicit interpretation",
                ));
            }
        };
        let chroma = match color.chroma_siting.as_deref() {
            Some("left") => ChromaSiting::Left,
            Some("center") => ChromaSiting::Center,
            Some("topleft") => ChromaSiting::TopLeft,
            _ => {
                return Err(Error::incompatible(
                    "CUDA preprocessing requires established supported chroma siting or explicit interpretation",
                ));
            }
        };
        self.as_yuv(matrix, range, chroma)
    }
    /// Borrow a CUDA kernel view with explicitly selected color interpretation.
    ///
    /// This does not copy or transfer decoded bytes. The GPU adapter establishes
    /// producer readiness and retains this surface through kernel completion.
    /// The arguments are an explicit interpretation choice; callers should use
    /// established metadata or a disclosed missing-metadata assumption.
    pub fn as_yuv(
        &self,
        matrix: YuvMatrix,
        range: YuvRange,
        chroma: ChromaSiting,
    ) -> Result<CudaYuvImage> {
        external_yuv(
            self.frame.image.storage().clone(),
            self.accounting_layout.clone(),
            [self.planes[0].address, self.planes[1].address],
            matrix,
            range,
            chroma,
        )
    }
}

/// Incremental decoder that requires actual CUDA hardware-frame negotiation.
pub struct CudaVideoReader {
    opened: OpenedInput,
    info: AssetInfo,
    decoder: ffmpeg::decoder::Video,
    // Drop a pending AVFrame before the supplied native context is released.
    pending: Option<ffmpeg::frame::Video>,
    context: CudaContext,
    track: TrackId,
    base: TimeBase,
    options: CudaDecodeOptions,
    seekable: bool,
    draining: bool,
    finished: bool,
    next_ordinal: Option<u64>,
    frames_decoded: u64,
    packets_sent: u64,
    key_packet_times: crate::codec::KeyPacketTimes,
    first_format: Option<(u32, u32, media_core::media::PixelFormat)>,
    orientation: Orientation,
    retained: Arc<RetainedBudget>,
    retirement_errors: Arc<Mutex<Option<String>>>,
}
impl CudaVideoReader {
    /// Open a fresh FFmpeg primary context before retaining it through cudarc.
    ///
    /// FFmpeg requires primary-context scheduling flags compatible with its own
    /// CUDA backend. If another runtime already activated incompatible flags,
    /// use `open_on_context` with that runtime's compatible context explicitly.
    pub fn open(
        source: &Source,
        ordinal: u32,
        limits: CudaLimits,
        options: CudaDecodeOptions,
    ) -> Result<Self> {
        limits.validate()?;
        Self::open_impl(source, ordinal, Some(limits), None, options, false, None)
    }
    /// Use an already-owned CUDA context, verifying the actual native handle.
    ///
    /// The context is bound on the calling thread and retained by every output.
    /// Neither an ordinal match nor an implicit host upload establishes sharing.
    pub fn open_on_context(
        source: &Source,
        context: &CudaContext,
        options: CudaDecodeOptions,
    ) -> Result<Self> {
        let Device::Cuda { ordinal, .. } = context.device() else {
            return Err(Error::incompatible(
                "CUDA decoder needs an actual CUDA context",
            ));
        };
        context
            .native_context()
            .bind_to_thread()
            .map_err(|e| Error::backend("cuda", e.to_string()).with_cause(e))?;
        Self::open_impl(
            source,
            ordinal,
            None,
            Some(context.clone()),
            options,
            false,
            None,
        )
    }
    /// Open only native container headers using an explicitly validated index;
    /// no stream-info probe or CPU video discovery decoder is opened.
    pub fn open_on_context_indexed(
        source: &Source,
        context: &CudaContext,
        options: CudaDecodeOptions,
    ) -> Result<Self> {
        let Device::Cuda { ordinal, .. } = context.device() else {
            return Err(Error::invalid("indexed decoder requires CUDA context"));
        };
        context
            .native_context()
            .bind_to_thread()
            .map_err(|e| Error::backend("cuda", e.to_string()).with_cause(e))?;
        Self::open_impl(
            source,
            ordinal,
            None,
            Some(context.clone()),
            options,
            true,
            None,
        )
    }
    /// Open with cooperative native-read cancellation. Indexed mode requires
    /// source-validated timing evidence and never invokes CPU codec discovery.
    pub fn open_on_context_controlled(
        source: &Source,
        context: &CudaContext,
        options: CudaDecodeOptions,
        indexed: bool,
        interrupt: format::ReadInterrupt,
    ) -> Result<Self> {
        let Device::Cuda { ordinal, .. } = context.device() else {
            return Err(Error::invalid("controlled decoder requires CUDA context"));
        };
        context
            .native_context()
            .bind_to_thread()
            .map_err(|e| Error::backend("cuda", e.to_string()).with_cause(e))?;
        Self::open_impl(
            source,
            ordinal,
            None,
            Some(context.clone()),
            options,
            indexed,
            Some(interrupt),
        )
    }
    #[allow(clippy::too_many_arguments)]
    fn open_impl(
        source: &Source,
        ordinal: u32,
        limits: Option<CudaLimits>,
        external: Option<CudaContext>,
        options: CudaDecodeOptions,
        indexed: bool,
        interrupt: Option<format::ReadInterrupt>,
    ) -> Result<Self> {
        if options.max_frame_bytes == 0
            || options.max_retained_frames == 0
            || options.max_retained_frames > 64
            || options.max_retained_bytes == 0
        {
            return Err(Error::invalid(
                "CUDA decode needs positive byte budgets and 1..=64 retained frames",
            ));
        }
        let opened = if let Some(interrupt) = interrupt {
            format::open_source_controlled(source, options.probe, indexed, false, interrupt)?
        } else if indexed {
            format::open_source_indexed(source, options.probe)?
        } else {
            format::open_source_without_decoding(source, options.probe)?
        };
        let info = format::snapshot(
            &opened.context,
            source,
            opened.counter.total.load(Ordering::Relaxed),
        )?;
        let selected = match options.track {
            Some(id) => info
                .tracks
                .iter()
                .find(|t| t.id == id && t.kind == TrackKind::Video),
            None => info
                .tracks
                .iter()
                .find(|t| t.kind == TrackKind::Video && !t.cover_art),
        }
        .ok_or_else(|| Error::unsupported("source has no selected non-cover video stream"))?;
        let track = selected.id;
        let base = selected.time_base;
        let orientation = selected
            .display_matrix
            .as_ref()
            .map(|m| native::display_orientation(m))
            .transpose()?
            .unwrap_or(Orientation::Normal);
        let stream = opened
            .context
            .stream(track.0 as usize)
            .ok_or_else(|| Error::invalid("selected video stream disappeared"))?;
        let mut decoder_context =
            ffmpeg::codec::context::Context::from_parameters(stream.parameters())
                .map_err(backend_error)?;
        decoder_context.set_threading(ffmpeg::threading::Config::count(1));
        // Source/probe callbacks may execute user code that changes the calling
        // thread's current CUDA context. Bind immediately before current_ctx.
        if let Some(context) = external.as_ref() {
            context
                .native_context()
                .bind_to_thread()
                .map_err(|error| Error::backend("cuda", error.to_string()).with_cause(error))?;
        }
        let device = native::configure_cuda_decoder(
            &mut decoder_context,
            ordinal,
            external.is_some(),
            options.max_retained_frames,
        )
        .map_err(|e| {
            e.with_track(track.0)
                .with_operation("required CUDA decoder setup")
        })?;
        let context = match external {
            Some(context) => context,
            None => CudaContext::new(
                ordinal,
                limits.ok_or_else(|| {
                    Error::invalid("CUDA context creation requires allocation limits")
                })?,
            )?,
        };
        let expected = Device::Cuda {
            ordinal,
            context: ContextId::new(device.context)?,
        };
        if context.device() != expected {
            return Err(Error::incompatible(
                "FFmpeg and CUDA kernel adapter opened different native contexts",
            ));
        }
        let decoder = decoder_context.decoder().video().map_err(|e| {
            backend_error(e)
                .with_track(track.0)
                .with_operation("open hardware-required decoder")
        })?;
        opened.allow_reading();
        Ok(Self {
            opened,
            info,
            decoder,
            pending: None,
            context,
            track,
            base,
            options,
            seekable: source.capabilities()?.seekable,
            draining: false,
            finished: false,
            next_ordinal: Some(0),
            frames_decoded: 0,
            packets_sent: 0,
            key_packet_times: crate::codec::KeyPacketTimes::default(),
            first_format: None,
            orientation,
            retained: Arc::new(RetainedBudget {
                state: Mutex::new(RetainedState::default()),
                max_frames: options.max_retained_frames,
                max_bytes: options.max_retained_bytes,
            }),
            retirement_errors: Arc::new(Mutex::new(None)),
        })
    }
    /// Actual CUDA context shared with native decoded output.
    pub fn context(&self) -> &CudaContext {
        &self.context
    }
    /// Discovered source timing and tracks. A header-derived timeline origin
    /// contradicted by the first actual frame is cleared; raw source PTS remains
    /// available even when asset presentation timing cannot be established.
    pub fn info(&self) -> &AssetInfo {
        &self.info
    }
    /// Selected video stream.
    pub fn track(&self) -> TrackId {
        self.track
    }
    /// Actual source bytes read by AVIO, including probing and rereads.
    pub fn bytes_read(&self) -> u64 {
        self.opened.counter.total.load(Ordering::Relaxed)
    }
    /// Actual decoded hardware frames.
    pub fn frames_decoded(&self) -> u64 {
        self.frames_decoded
    }
    /// Compressed selected-track packets accepted by the decoder.
    pub fn packets_sent(&self) -> u64 {
        self.packets_sent
    }
    /// Hard total source-byte cap, including discovery reads already performed.
    pub fn set_read_limit(&self, bytes: u64) -> Result<()> {
        self.opened.set_read_limit(bytes)
    }
    /// Start a bounded request on this live cursor, excluding setup/earlier reads.
    /// Earlier cancellation or source-read failures remain terminal.
    pub fn begin_read_batch(&self, bytes: u64) -> Result<()> {
        self.opened.begin_read_batch(bytes)
    }
    /// Replace the cooperative native-read cancellation predicate.
    pub fn set_interrupt(&self, interrupt: format::ReadInterrupt) {
        self.opened.set_interrupt(interrupt);
    }
    /// Apply a caller-established shared asset origin from a validated index.
    /// No frame timestamps or pixels are synthesized by this metadata setting.
    pub fn set_timeline_origin(&mut self, origin: Time) {
        self.info.timeline_origin = Some(origin);
    }
    /// Currently retained surface lease count.
    pub fn retained_frames(&self) -> Result<usize> {
        Ok(self
            .retained
            .state
            .lock()
            .map_err(|_| Error::backend("ffmpeg", "retained-surface budget lock poisoned"))?
            .frames)
    }
    /// Known bytes currently retained in output surface leases.
    pub fn retained_bytes(&self) -> Result<usize> {
        Ok(self
            .retained
            .state
            .lock()
            .map_err(|_| Error::backend("ffmpeg", "retained-surface budget lock poisoned"))?
            .bytes)
    }
    /// Read the next CUDA surface; a software frame is always an error.
    pub fn next_frame(&mut self) -> Result<Option<CudaVideoFrame>> {
        self.opened.check_read_budget()?;
        self.context.check_errors()?;
        if let Some(error) = self
            .retirement_errors
            .lock()
            .map_err(|_| Error::backend("ffmpeg", "retirement lock poisoned"))?
            .as_ref()
        {
            return Err(Error::backend(
                "cuda",
                format!("earlier decoded surface retirement failed: {error}"),
            ));
        }
        if self.finished {
            return Ok(None);
        }
        if self.retained_frames()? >= self.options.max_retained_frames {
            return Err(Error::exhausted("retained CUDA decode surfaces reached the configured limit; release a frame or explicitly detach on device")
                .with_constraint("max_retained_frames"));
        }
        if let Some(decoded) = self.pending.take() {
            return self.materialize(decoded).map(Some);
        }
        loop {
            let mut decoded = ffmpeg::frame::Video::empty();
            match self.decoder.receive_frame(&mut decoded) {
                Ok(()) => {
                    self.frames_decoded = self
                        .frames_decoded
                        .checked_add(1)
                        .ok_or_else(|| Error::overflow("hardware frame count overflow"))?;
                    return self.materialize(decoded).map(Some);
                }
                Err(ffmpeg::Error::Eof) => {
                    self.opened.check_read_budget()?;
                    self.finished = true;
                    return Ok(None);
                }
                Err(ffmpeg::Error::Other { errno }) if errno == ffmpeg::error::EAGAIN => {}
                Err(error) => {
                    return Err(backend_error(error)
                        .with_track(self.track.0)
                        .with_operation("required CUDA decode; software fallback forbidden"));
                }
            }
            if self.draining {
                return Err(Error::backend(
                    "ffmpeg",
                    "hardware decoder requested input after drain",
                ));
            }
            let mut packet = ffmpeg::Packet::empty();
            loop {
                self.opened.check_read_budget()?;
                match packet.read(&mut self.opened.context) {
                    Ok(()) if packet.stream() == self.track.0 as usize => break,
                    Ok(()) => continue,
                    Err(ffmpeg::Error::Eof) => {
                        self.opened.check_read_budget()?;
                        self.decoder.send_eof().map_err(backend_error)?;
                        self.draining = true;
                        break;
                    }
                    Err(error) => {
                        self.opened.check_read_budget()?;
                        return Err(backend_error(error).with_operation("demux for CUDA decode"));
                    }
                }
            }
            if !self.draining {
                self.decoder.send_packet(&packet).map_err(|e| {
                    backend_error(e)
                        .with_track(self.track.0)
                        .with_operation("required CUDA decode; software fallback forbidden")
                })?;
                self.key_packet_times.observe(&packet, self.base)?;
                self.packets_sent = self
                    .packets_sent
                    .checked_add(1)
                    .ok_or_else(|| Error::overflow("hardware decode packet counter overflow"))?;
            }
        }
    }
    /// Seek navigation on the asset timeline, flushing previous decoder state.
    ///
    /// Retained old frames remain valid. Ordinals are unknown until an explicit
    /// presentation index establishes them; seek is not temporal selection.
    pub fn seek(&mut self, target: Time) -> Result<()> {
        if !self.seekable {
            return Err(Error::unsupported(
                "CUDA reader random access needs a seekable source or explicit spooling",
            ));
        }
        let origin = self.info.timeline_origin.ok_or_else(|| {
            Error::unsupported("CUDA asset-time seeking requires an established timeline origin")
        })?;
        let source_target = target.checked_add(origin)?;
        let ticks = source_target.to_ticks(TimeBase::new(1, 1_000_000)?, Rounding::Floor)?;
        self.opened
            .context
            .seek(ticks, ..ticks)
            .map_err(backend_error)?;
        self.decoder.flush();
        self.pending = None;
        self.draining = false;
        self.finished = false;
        self.next_ordinal = None;
        self.first_format = None;
        self.key_packet_times.clear();
        Ok(())
    }
    /// Navigate by exact source-stream timestamp for source-validated index
    /// alignment. No host pixels are created; ordinals remain unknown here.
    pub fn seek_source(&mut self, target: Time) -> Result<()> {
        if !self.seekable {
            return Err(Error::unsupported(
                "indexed CUDA seeking requires a seekable source or explicit spooling",
            ));
        }
        let ticks = target.to_ticks(self.base, Rounding::NearestTiesEven)?;
        if self.base.time(ticks)? != target {
            return Err(Error::incompatible(
                "index source PTS does not fit this CUDA stream time base exactly",
            ));
        }
        let seek = native::seek_video_stream(&mut self.opened.context, self.track.0, ticks);
        self.opened.check_read_budget()?;
        seek?;
        self.decoder.flush();
        self.pending = None;
        self.draining = false;
        self.finished = false;
        self.next_ordinal = None;
        self.first_format = None;
        self.key_packet_times.clear();
        Ok(())
    }
    /// Navigate a verified keyframe in this demuxer's actual timestamp domain.
    pub fn seek_indexed_keyframe(&mut self, pts: Time, packet_dts: Option<Time>) -> Result<()> {
        self.seek_source(format::indexed_seek_time(
            &self.info.format,
            pts,
            packet_dts,
        ))
    }
    fn materialize(&mut self, decoded: ffmpeg::frame::Video) -> Result<CudaVideoFrame> {
        if decoded.is_corrupt() || decoded.has_decode_errors() {
            return Err(Error::backend(
                "ffmpeg",
                "hardware decoder returned corrupt/incomplete output",
            ));
        }
        let keyframe = decoded.is_key();
        let description = native::cuda_surface(&decoded)?;
        let Device::Cuda { ordinal, .. } = self.context.device() else {
            return Err(Error::incompatible("CUDA context placement is invalid"));
        };
        let device = Device::Cuda {
            ordinal,
            context: ContextId::new(description.context)?,
        };
        if device != self.context.device() {
            return Err(Error::incompatible(
                "decoded surface belongs to another native CUDA context",
            ));
        }
        if description.accounted_bytes > self.options.max_frame_bytes {
            return Err(Error::exhausted(
                "decoded CUDA surface exceeds per-frame byte limit",
            ));
        }
        let format = (decoded.width(), decoded.height(), description.format);
        if self.first_format.is_some_and(|previous| previous != format) {
            return Err(Error::incompatible(
                "CUDA video format changed midstream; explicit renegotiation is required",
            ));
        }
        self.first_format = Some(format);
        let (decoded, reservation) = self.retained.reserve_decoded(
            decoded,
            description.accounted_bytes,
            &mut self.pending,
        )?;
        let ready = external_completion(&self.context, description.stream)?;
        let mut metadata = crate::codec::frame_metadata(&decoded)?;
        if decoded
            .side_data(ffmpeg::util::frame::side_data::Type::DisplayMatrix)
            .is_none()
        {
            metadata.orientation = self.orientation;
        }
        let source_pts = decoded.pts().map(|v| self.base.time(v)).transpose()?;
        let seek_timestamp = self.key_packet_times.take(source_pts);
        if self.next_ordinal == Some(0) {
            self.info.timeline_origin =
                verified_header_origin(self.info.timeline_origin, source_pts);
        }
        let pts = source_pts
            .zip(self.info.timeline_origin)
            .map(|(source, origin)| source.checked_sub(origin))
            .transpose()?;
        let native_duration = if decoded.packet().duration > 0 {
            Some(self.base.time(decoded.packet().duration)?)
        } else {
            None
        };
        let owner = Arc::new(CudaFrameOwner {
            frame: Arc::new(decoded),
            context: self.context.clone(),
            ready: ready.clone(),
            device,
            bytes: description.accounted_bytes,
            _reservation: Arc::new(reservation),
            retirement_errors: Arc::clone(&self.retirement_errors),
            retained: Mutex::new(Vec::new()),
        });
        let storage = StorageLease::from_owner(owner.clone(), Arc::new(ready));
        let layout = ImageLayout {
            width: format.0,
            height: format.1,
            allocated_width: description.allocated_width,
            allocated_height: description.allocated_height,
            format: description.format,
            planes: Vec::new(),
        };
        let image = Image::new(layout, storage, metadata)?;
        let ordinal = self.next_ordinal;
        self.next_ordinal = ordinal.and_then(|n| n.checked_add(1));
        let planes = std::array::from_fn(|i| CudaPlane {
            address: description.addresses[i],
            pitch: description.pitches[i],
            row_bytes: description.row_bytes[i],
            rows: description.rows[i],
        });
        let y_bytes = description.pitches[0]
            .checked_mul(description.rows[0])
            .ok_or_else(|| Error::overflow("CUDA luma accounting overflow"))?;
        let accounting_layout = ImageLayout {
            width: format.0,
            height: format.1,
            allocated_width: description.allocated_width,
            allocated_height: description.allocated_height,
            format: description.format,
            planes: vec![
                PlaneLayout {
                    offset: 0,
                    row_stride: description.pitches[0],
                    row_bytes: description.row_bytes[0],
                    rows: description.rows[0],
                },
                PlaneLayout {
                    offset: y_bytes,
                    row_stride: description.pitches[1],
                    row_bytes: description.row_bytes[1],
                    rows: description.rows[1],
                },
            ],
        };
        Ok(CudaVideoFrame {
            frame: VideoFrame {
                image,
                track: self.track,
                timing: FrameTiming {
                    pts,
                    duration: match self.options.duration_policy {
                        media_core::policy::FrameDurationPolicy::Unknown => None,
                        media_core::policy::FrameDurationPolicy::TrustReported => native_duration,
                    },
                    native_duration,
                    source_pts,
                    ordinal,
                },
            },
            owner,
            planes,
            accounting_layout,
            keyframe,
            seek_timestamp,
        })
    }
}

// Header-only discovery cannot use a CPU codec to repair an invented container
// start (notably reordered Matroska video). An actual first frame before the
// advertised common origin disproves that mapping; retaining raw source timing
// permits a caller with independent cross-track evidence to rebase explicitly.
fn verified_header_origin(header: Option<Time>, first_source: Option<Time>) -> Option<Time> {
    match (header, first_source) {
        (Some(origin), Some(first)) if first >= origin => Some(origin),
        _ => None,
    }
}

#[derive(Default)]
struct RetainedState {
    frames: usize,
    bytes: usize,
}
struct RetainedBudget {
    state: Mutex<RetainedState>,
    max_frames: usize,
    max_bytes: usize,
}
impl RetainedBudget {
    fn reserve_decoded(
        self: &Arc<Self>,
        decoded: ffmpeg::frame::Video,
        bytes: usize,
        pending: &mut Option<ffmpeg::frame::Video>,
    ) -> Result<(ffmpeg::frame::Video, Reservation)> {
        match self.reserve(bytes) {
            Ok(reservation) => Ok((decoded, reservation)),
            Err(error) => {
                if error.kind() == media_core::error::ErrorKind::ResourceExhausted {
                    // Backpressure is recoverable: preserve this same decoded
                    // frame, not the following frame, for retry after release.
                    *pending = Some(decoded);
                }
                Err(error)
            }
        }
    }
    fn reserve(self: &Arc<Self>, bytes: usize) -> Result<Reservation> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| Error::backend("ffmpeg", "retained-surface budget lock poisoned"))?;
        let new_bytes = state
            .bytes
            .checked_add(bytes)
            .ok_or_else(|| Error::overflow("retained CUDA bytes overflow"))?;
        if state.frames >= self.max_frames || new_bytes > self.max_bytes {
            return Err(Error::exhausted(
                "retained CUDA surfaces exceed the configured count/byte budget; release a frame",
            ));
        }
        state.frames += 1;
        state.bytes = new_bytes;
        Ok(Reservation {
            budget: Arc::clone(self),
            bytes,
        })
    }
}

struct Reservation {
    budget: Arc<RetainedBudget>,
    bytes: usize,
}
impl Drop for Reservation {
    fn drop(&mut self) {
        let mut state = self
            .budget
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        state.frames -= 1;
        state.bytes -= self.bytes;
    }
}
struct CudaFrameOwner {
    frame: Arc<ffmpeg::frame::Video>,
    context: CudaContext,
    ready: CudaCompletion,
    device: Device,
    bytes: usize,
    _reservation: Arc<Reservation>,
    retirement_errors: Arc<Mutex<Option<String>>>,
    // Last field: guards release only after producer/consumer retirement and
    // native AVFrame/pool/context cleanup, including wrapper-only owner aliases.
    retained: Mutex<Vec<Arc<dyn NativeGuard>>>,
}
trait NativeGuard: fmt::Debug + Send + Sync {}
impl<T: fmt::Debug + Send + Sync> NativeGuard for T {}
impl fmt::Debug for CudaFrameOwner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CudaFrameOwner")
            .field("device", &self.device)
            .field("bytes", &self.bytes)
            .field("width", &self.frame.width())
            .field("height", &self.frame.height())
            .finish()
    }
}
impl StorageOwner for CudaFrameOwner {
    fn byte_len(&self) -> usize {
        self.bytes
    }
    fn kind(&self) -> MemoryKind {
        MemoryKind::Opaque
    }
    fn device(&self) -> Device {
        self.device
    }
}
impl Drop for CudaFrameOwner {
    fn drop(&mut self) {
        // Final retirement waits only this producer event. Imported GPU kernel
        // views additionally retain the owner through their post-consumption
        // event, so decoder surfaces cannot recycle during queued reads.
        if let Err(error) = self.ready.wait_for_retirement() {
            self.context.record_retirement_failure(error.to_string());
            let mut recorded = self
                .retirement_errors
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            if recorded.is_none() {
                *recorded = Some(error.to_string());
            }
            // No safe native retirement was established. Keep both the native
            // owner and every resource quota that accounts for its retention.
            let guards = self
                .retained
                .get_mut()
                .unwrap_or_else(|poison| poison.into_inner())
                .clone();
            std::mem::forget((
                self.frame.clone(),
                self.context.clone(),
                self.ready.clone(),
                self._reservation.clone(),
                guards,
            ));
        }
    }
}

#[allow(unsafe_code)]
fn external_completion(context: &CudaContext, stream: u64) -> Result<CudaCompletion> {
    // SAFETY: native::cuda_surface verified that this FFmpeg stream belongs to
    // the exact CUcontext retained by context. FFmpeg's AVFrame keeps the
    // hardware device alive while an event is recorded on that borrowed stream.
    unsafe { context.record_external_completion(stream) }
}
#[allow(unsafe_code)]
fn external_yuv(
    owner: StorageLease,
    layout: ImageLayout,
    addresses: [u64; 2],
    matrix: YuvMatrix,
    range: YuvRange,
    chroma: ChromaSiting,
) -> Result<CudaYuvImage> {
    // SAFETY: only native::cuda_surface constructs these immutable planes.
    // AVFrame ownership, device identity and the producer event are retained by
    // owner. The adapter retains owner until post-kernel consumer completion.
    unsafe { CudaYuvImage::from_external_planes(owner, layout, addresses, matrix, range, chroma) }
}

#[cfg(test)]
mod budget_tests {
    use super::*;
    #[test]
    fn actual_first_source_pts_rejects_contradictory_header_origin_without_rebasing() {
        let video = Time::from_millis(500);
        let audio_header = Time::from_millis(600);
        assert_eq!(
            verified_header_origin(Some(audio_header), Some(video)),
            None
        );
        assert_eq!(verified_header_origin(None, Some(video)), None);
        assert_eq!(verified_header_origin(Some(audio_header), None), None);
        assert_eq!(
            verified_header_origin(Some(video), Some(video)),
            Some(video)
        );
        assert_eq!(
            verified_header_origin(Some(video), Some(audio_header)),
            Some(video)
        );
    }
    #[test]
    fn failed_byte_reservation_preserves_the_same_decoded_frame_for_retry() {
        let budget = Arc::new(RetainedBudget {
            state: Mutex::new(RetainedState::default()),
            max_frames: 2,
            max_bytes: 64,
        });
        let held = budget.reserve(48).unwrap();
        let mut decoded = ffmpeg::frame::Video::empty();
        decoded.set_pts(Some(37));
        let mut pending = None;
        let failed = budget.reserve_decoded(decoded, 32, &mut pending);
        assert!(failed.is_err());
        assert_eq!(pending.as_ref().unwrap().pts(), Some(37));
        assert_eq!(budget.state.lock().unwrap().frames, 1);
        drop(held);
        let (retried, reservation) = budget
            .reserve_decoded(pending.take().unwrap(), 32, &mut pending)
            .unwrap();
        assert_eq!(retried.pts(), Some(37));
        assert!(pending.is_none());
        assert_eq!(budget.state.lock().unwrap().bytes, 32);
        drop(reservation);
        assert_eq!(budget.state.lock().unwrap().bytes, 0);
    }
}
