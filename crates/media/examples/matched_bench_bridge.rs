//! Benchmark-only C ABI exporting CPU/CUDA crops into a common Python consumer.
//!
//! Never use this ABI as a production binding. The single caller owns each handle,
//! consumes each returned legacy DLPack descriptor exactly once, and submits CUDA
//! consumer work only on mb_consumer_stream(handle), via torch.cuda.ExternalStream.
//! A consumed capsule belongs to Torch; an unconsumed capsule must call mb_release.
//! All CUDA output/readiness/consumer lifetime guards are retained by the descriptor.
use media::interop::{
    dlpack::{self, DLManagedTensor},
    tensor::Tensor,
};
use media::{
    backend::{
        cuda::{
            ChromaInterpolation, CropRect, CudaContext, CudaLimits, CudaStream, PreprocessSpec,
            preprocess_batch,
        },
        ffmpeg::{
            codec::{VideoDecodeOptions, VideoReader as NativeCpu},
            cuda::{CudaDecodeOptions, CudaVideoFrame, CudaVideoReader as NativeGpu},
        },
    },
    core::media::Orientation,
    io::source::Source,
    prelude::*,
    video_index::{IndexBuildLimits, VideoIndex},
};
use std::{
    collections::BTreeMap,
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

fn gpu_limits() -> CudaLimits {
    CudaLimits {
        memory_bytes: 512 * 1024 * 1024,
        streams: 12,
    }
}
fn decode_options() -> CudaDecodeOptions {
    CudaDecodeOptions {
        max_frame_bytes: 32 * 1024 * 1024,
        max_retained_frames: 20,
        max_retained_bytes: 512 * 1024 * 1024,
        ..Default::default()
    }
}
#[derive(Clone, Copy)]
enum Backend {
    Media,
    Native,
    NativeSeek,
    Indexed,
    MediaCursor,
}
enum Cursor {
    MediaCpu(media::reader::VideoReader),
    NativeCpu(NativeCpu),
    MediaGpu(media::cuda_request::CudaReader),
    NativeGpu(NativeGpu),
}
impl Cursor {
    fn seek(&mut self, time: Time) -> Result<()> {
        match self {
            Self::MediaCpu(reader) => reader.seek(time),
            Self::NativeCpu(reader) => reader.seek(time),
            Self::MediaGpu(reader) => reader.seek(time),
            Self::NativeGpu(reader) => reader.seek(time),
        }
    }
    fn decoded(&self) -> u64 {
        match self {
            Self::MediaCpu(reader) => reader.report().decoded_frames,
            Self::NativeCpu(reader) => reader.frames_decoded(),
            Self::MediaGpu(reader) => reader.report().decoded_frames,
            Self::NativeGpu(reader) => reader.frames_decoded(),
        }
    }
    fn bytes(&self) -> u64 {
        match self {
            Self::MediaCpu(reader) => reader.report().bytes_read,
            Self::NativeCpu(reader) => reader.bytes_read(),
            Self::MediaGpu(reader) => reader.report().bytes_read,
            Self::NativeGpu(reader) => reader.bytes_read(),
        }
    }
}
#[derive(Debug, Default)]
struct NativeLive {
    bytes: AtomicUsize,
    peak: AtomicUsize,
}
#[derive(Debug)]
struct NativeLease {
    live: Arc<NativeLive>,
    bytes: usize,
}
impl NativeLease {
    fn new(live: &Arc<NativeLive>, bytes: usize) -> Self {
        let current = live.bytes.fetch_add(bytes, Ordering::AcqRel) + bytes;
        live.peak.fetch_max(current, Ordering::AcqRel);
        Self {
            live: live.clone(),
            bytes,
        }
    }
}
impl Drop for NativeLease {
    fn drop(&mut self) {
        self.live.bytes.fetch_sub(self.bytes, Ordering::AcqRel);
    }
}
struct Handle {
    source: Source,
    session: Session,
    context: Option<CudaContext>,
    consumer: Option<CudaStream>,
    backend: Backend,
    gpu: bool,
    warm: bool,
    cursor: Option<Cursor>,
    index: Option<Arc<VideoIndex>>,
    native_live: Arc<NativeLive>,
    width: u32,
    height: u32,
    point_chroma: bool,
    bt601_center: bool,
}
impl Handle {
    #[allow(clippy::too_many_arguments)]
    fn open(
        path: &Path,
        backend: Backend,
        gpu: bool,
        warm: bool,
        width: u32,
        height: u32,
        point_chroma: bool,
        bt601_center: bool,
    ) -> Result<Self> {
        if point_chroma && !gpu {
            return Err(Error::unsupported(
                "point-chroma quality is a CUDA benchmark policy",
            ));
        }
        if width < 224 || height < 224 {
            return Err(Error::invalid(
                "matched task requires a 224-pixel center crop",
            ));
        }
        let source = Source::memory(std::fs::read(path).map_err(Error::io)?);
        let mut builder = Session::builder()
            .limits(Limits::new().host_memory(2 * 1024 * 1024 * 1024))
            .cuda_limits(media::cuda_session::CudaSessionLimits {
                max_codec_sessions: 2,
                max_native_bytes: 512 * 1024 * 1024,
            });
        // Python initializes its primary context before entering this constructor.
        // Retain that exact context; supplied-context FFmpeg uses current_ctx=1.
        let context = if gpu {
            Some(CudaContext::new(0, gpu_limits())?)
        } else {
            None
        };
        if let Some(context) = &context {
            builder = builder.cuda_context(context.clone());
        }
        let session = builder.build()?;
        let consumer = context.as_ref().map(CudaContext::stream).transpose()?;
        let mut handle = Self {
            source,
            session,
            context,
            consumer,
            backend,
            gpu,
            warm,
            cursor: None,
            index: None,
            native_live: Arc::new(NativeLive::default()),
            width,
            height,
            point_chroma,
            bt601_center,
        };
        if warm
            && matches!(
                backend,
                Backend::Native | Backend::NativeSeek | Backend::MediaCursor
            )
        {
            handle.cursor = Some(handle.new_cursor()?);
        }
        if warm && matches!(backend, Backend::Indexed) {
            handle.index = Some(handle.build_index()?);
        }
        Ok(handle)
    }
    fn build_index(&self) -> Result<Arc<VideoIndex>> {
        let request = self.session.from_source(self.source.clone()).video()?;
        let limits = IndexBuildLimits {
            max_frames: 256,
            max_index_bytes: 512 * 1024,
            max_source_bytes: 1024 * 1024 * 1024,
        };
        if self.gpu {
            request
                .cuda(0, gpu_limits())
                .on_context(self.context.as_ref().expect("GPU context"))
                .decode_options(decode_options())
                .build_index(limits)
        } else {
            request.build_index(limits)
        }
    }
    fn facade_frames(
        &mut self,
        indices: &[u64],
        stats: &mut Stats,
    ) -> Result<(BTreeMap<u64, VideoFrame>, BTreeMap<u64, CudaVideoFrame>)> {
        let index = if matches!(self.backend, Backend::Indexed) {
            if let Some(index) = &self.index {
                Some(index.clone())
            } else {
                let start = Instant::now();
                let index = self.build_index()?;
                stats.index_build_ns = start.elapsed().as_nanos() as u64;
                stats.index_decoded_frames = index.report().decoded_frames;
                stats.index_source_bytes = index.report().bytes_read;
                Some(index)
            }
        } else {
            None
        };
        let times = indices
            .iter()
            .map(|index| Time::new(*index as i64, 30))
            .collect::<Result<Vec<_>>>()?;
        let request = self.session.from_source(self.source.clone()).video()?;
        let mut cpu = BTreeMap::new();
        let mut gpu = BTreeMap::new();
        if self.gpu {
            let mut request = request
                .cuda(0, gpu_limits())
                .on_context(self.context.as_ref().expect("GPU context"))
                .decode_options(decode_options())
                .sample(times)
                .selection(TemporalSelection::DisplayedAt)
                .duration_inference(DurationInference::NextTimestamp);
            if let Some(index) = index {
                request = request.with_index(index);
            }
            let batch = request.read()?;
            stats.decoded_frames = batch.report.decoded_frames;
            stats.source_bytes = batch.report.bytes_read;
            stats.seeks = batch.report.seeks;
            stats.index_cache_hits = batch.report.index_cache_hits;
            for (index, frame) in indices.iter().zip(&batch.frames) {
                gpu.insert(*index, frame.clone());
            }
        } else {
            let mut request = request
                .sample(times)
                .selection(TemporalSelection::DisplayedAt)
                .duration_inference(DurationInference::NextTimestamp);
            if let Some(index) = index {
                request = request.with_index(index);
            }
            let batch = request.read()?;
            stats.decoded_frames = batch.report.decoded_frames;
            stats.source_bytes = batch.report.bytes_read;
            stats.seeks = batch.report.seeks;
            stats.index_cache_hits = batch.report.index_cache_hits;
            for (index, frame) in indices.iter().zip(&batch.frames) {
                cpu.insert(*index, frame.clone());
            }
        }
        Ok((cpu, gpu))
    }
    fn retire(&self) -> Result<()> {
        let start = Instant::now();
        if let Some(context) = &self.context {
            while context.memory_used() != 0 || self.native_live.bytes.load(Ordering::Acquire) != 0
            {
                if start.elapsed() > Duration::from_secs(5) {
                    return Err(Error::exhausted(
                        "benchmark native input/output retirement did not finish",
                    ));
                }
                std::thread::yield_now();
            }
            context.check_errors()?;
        }
        Ok(())
    }
    fn new_cursor(&self) -> Result<Cursor> {
        match (self.backend, self.gpu) {
            (Backend::Media | Backend::Indexed | Backend::MediaCursor, false) => {
                Ok(Cursor::MediaCpu(
                    self.session
                        .from_source(self.source.clone())
                        .video()?
                        .reader()?,
                ))
            }
            (Backend::Native | Backend::NativeSeek, false) => Ok(Cursor::NativeCpu(
                NativeCpu::open(&self.source, VideoDecodeOptions::default())?,
            )),
            (Backend::Media | Backend::Indexed | Backend::MediaCursor, true) => {
                Ok(Cursor::MediaGpu(
                    self.session
                        .from_source(self.source.clone())
                        .video()?
                        .cuda(0, gpu_limits())
                        .on_context(self.context.as_ref().expect("GPU context"))
                        .decode_options(decode_options())
                        .reader()?,
                ))
            }
            (Backend::Native | Backend::NativeSeek, true) => {
                Ok(Cursor::NativeGpu(NativeGpu::open_on_context(
                    &self.source,
                    self.context.as_ref().expect("GPU context"),
                    decode_options(),
                )?))
            }
        }
    }
    fn next_cpu(cursor: &mut Cursor) -> Result<Option<VideoFrame>> {
        match cursor {
            Cursor::MediaCpu(reader) => reader.next_frame(),
            Cursor::NativeCpu(reader) => reader.next_frame(),
            _ => Err(Error::invalid("CPU task has GPU cursor")),
        }
    }
    fn next_gpu(cursor: &mut Cursor) -> Result<Option<CudaVideoFrame>> {
        match cursor {
            Cursor::MediaGpu(reader) => reader.next_frame(),
            Cursor::NativeGpu(reader) => reader.next_frame(),
            _ => Err(Error::invalid("GPU task has CPU cursor")),
        }
    }
    fn run(&mut self, indices: &[u64], stats: &mut Stats) -> Result<*mut DLManagedTensor> {
        if indices.is_empty() || indices.len() > 32 || indices.iter().any(|index| *index >= 180) {
            return Err(Error::invalid(
                "fixture request indices must contain 1..32 entries within six seconds",
            ));
        }
        let (cpu, gpu) = if matches!(self.backend, Backend::Media | Backend::Indexed) {
            self.facade_frames(indices, stats)?
        } else {
            let mut unique = indices.to_vec();
            unique.sort_unstable();
            unique.dedup();
            let mut cursor = match self.cursor.take() {
                Some(cursor) => cursor,
                None => self.new_cursor()?,
            };
            let before = if self.warm { cursor.decoded() } else { 0 };
            let before_bytes = if self.warm { cursor.bytes() } else { 0 };
            let mut cpu = BTreeMap::new();
            let mut gpu = BTreeMap::new();
            if self.warm && matches!(self.backend, Backend::Native) {
                cursor.seek(Time::ZERO)?;
                stats.seeks += 1;
            }
            if matches!(self.backend, Backend::NativeSeek)
                || (self.warm && matches!(self.backend, Backend::MediaCursor))
            {
                for index in &unique {
                    let target = Time::new(*index as i64, 30)?;
                    cursor.seek(target)?;
                    stats.seeks += 1;
                    if self.gpu {
                        loop {
                            let frame = Self::next_gpu(&mut cursor)?.ok_or_else(|| {
                                Error::unsupported("seeked GPU fixture ended before request")
                            })?;
                            let pts = frame
                                .frame()
                                .timing
                                .pts
                                .ok_or_else(|| Error::unsupported("GPU fixture PTS missing"))?;
                            if pts < target {
                                continue;
                            }
                            if pts != target {
                                return Err(Error::incompatible(
                                    "seeked GPU presentation time does not match exact CFR request",
                                ));
                            }
                            gpu.insert(*index, frame);
                            break;
                        }
                    } else {
                        loop {
                            let frame = Self::next_cpu(&mut cursor)?.ok_or_else(|| {
                                Error::unsupported("seeked CPU fixture ended before request")
                            })?;
                            let pts = frame
                                .timing
                                .pts
                                .ok_or_else(|| Error::unsupported("CPU fixture PTS missing"))?;
                            if pts < target {
                                continue;
                            }
                            if pts != target {
                                return Err(Error::incompatible(
                                    "seeked CPU presentation time does not match exact CFR request",
                                ));
                            }
                            cpu.insert(*index, frame);
                            break;
                        }
                    }
                }
            } else {
                let maximum = Time::new(*unique.last().expect("nonempty") as i64, 30)?;
                loop {
                    if self.gpu {
                        let Some(frame) = Self::next_gpu(&mut cursor)? else {
                            break;
                        };
                        let pts = frame
                            .frame()
                            .timing
                            .pts
                            .ok_or_else(|| Error::unsupported("GPU fixture PTS missing"))?;
                        if pts > maximum {
                            break;
                        }
                        for index in &unique {
                            if pts == Time::new(*index as i64, 30)? {
                                gpu.insert(*index, frame.clone());
                            }
                        }
                    } else {
                        let Some(frame) = Self::next_cpu(&mut cursor)? else {
                            break;
                        };
                        let pts = frame
                            .timing
                            .pts
                            .ok_or_else(|| Error::unsupported("CPU fixture PTS missing"))?;
                        if pts > maximum {
                            break;
                        }
                        for index in &unique {
                            if pts == Time::new(*index as i64, 30)? {
                                cpu.insert(*index, frame.clone());
                            }
                        }
                    }
                }
            }
            stats.decoded_frames = cursor.decoded() - before;
            stats.source_bytes = cursor.bytes() - before_bytes;
            if self.warm {
                self.cursor = Some(cursor);
            } else {
                drop(cursor);
            }
            (cpu, gpu)
        };
        if self.gpu {
            self.native_live = Arc::new(NativeLive::default());
            let gpu = gpu
                .into_iter()
                .map(|(index, frame)| {
                    let bytes = frame.frame().image.storage().byte_len();
                    (
                        index,
                        frame.retain(NativeLease::new(&self.native_live, bytes)),
                    )
                })
                .collect::<BTreeMap<_, _>>();
            let mut inputs = Vec::with_capacity(indices.len());
            for index in indices {
                let frame = gpu
                    .get(index)
                    .ok_or_else(|| Error::unsupported("GPU exact candidate was not retained"))?;
                let metadata = frame.frame().image.metadata();
                let color = &metadata.color;
                let (signal, chroma) = if self.bt601_center {
                    ("smpte170m", "center")
                } else {
                    ("bt709", "left")
                };
                if metadata.orientation != Orientation::Normal
                    || metadata.icc_profile.is_some()
                    || color.primaries.as_deref() != Some(signal)
                    || color.transfer.as_deref() != Some(signal)
                    || color.matrix.as_deref() != Some(signal)
                    || color.range.as_deref() != Some("tv")
                    || color.chroma_siting.as_deref() != Some(chroma)
                {
                    let bounded = |value: &Option<String>| {
                        value
                            .as_deref()
                            .map(|text| text.chars().take(64).collect::<String>())
                    };
                    return Err(Error::incompatible(format!(
                        "benchmark profile requires Normal/noICC/{signal}/tv/{chroma}; received orientation={:?},ICC_bytes={:?},primaries={:?},transfer={:?},matrix={:?},range={:?},chroma={:?},format={:?},size={}x{}",
                        metadata.orientation,
                        metadata.icc_profile.as_ref().map(Vec::len),
                        bounded(&color.primaries),
                        bounded(&color.transfer),
                        bounded(&color.matrix),
                        bounded(&color.range),
                        bounded(&color.chroma_siting),
                        frame.frame().image.pixel_format(),
                        frame.frame().image.width(),
                        frame.frame().image.height(),
                    )));
                }
                inputs.push(frame.yuv_view()?);
            }
            let stream = self.context.as_ref().expect("GPU context").stream()?;
            let mut spec = PreprocessSpec::new(224, 224)?;
            spec.chroma_interpolation = if self.point_chroma {
                ChromaInterpolation::NearestTiesEarlier
            } else {
                ChromaInterpolation::MatchSpatial
            };
            spec.crop = Some(CropRect {
                x: (self.width - 224) / 2,
                y: (self.height - 224) / 2,
                width: 224,
                height: 224,
            });
            let tensor = preprocess_batch(&inputs, &stream, spec)?;
            drop(inputs);
            drop(gpu);
            // Explicit event wait is timed by the Python caller. No host pixels
            // or whole-device synchronization is introduced by this boundary.
            tensor.buffer().readiness().wait()?;
            stats.native_selected_peak_bytes = self.native_live.peak.load(Ordering::Acquire) as u64;
            stats.session_native_peak_bytes = self.session.peak_reserved_cuda_native_bytes() as u64;
            stats.host_peak_bytes = self.session.peak_host_bytes() as u64;
            stats.owned_peak_bytes =
                self.context.as_ref().expect("GPU context").memory_peak() as u64;
            Ok(dlpack::export_cuda_legacy(
                tensor,
                self.consumer.as_ref().expect("consumer"),
                CopyPolicy::Forbid,
            )?
            .into_raw())
        } else {
            let mut output = Vec::with_capacity(indices.len() * 3 * 224 * 224);
            for index in indices {
                let frame = cpu
                    .get(index)
                    .ok_or_else(|| Error::unsupported("CPU exact candidate was not retained"))?;
                let image = &frame.image;
                if image.pixel_format() != PixelFormat::Rgb8
                    || image.width() != self.width
                    || image.height() != self.height
                {
                    return Err(Error::incompatible("CPU output geometry/format mismatch"));
                }
                let bytes = image.packed_bytes()?;
                let left = (self.width as usize - 224) / 2;
                let top = (self.height as usize - 224) / 2;
                for channel in 0..3 {
                    for y in 0..224 {
                        for x in 0..224 {
                            output.push(
                                f32::from(
                                    bytes[((top + y) * self.width as usize + left + x) * 3
                                        + channel],
                                ) / 255.0,
                            );
                        }
                    }
                }
            }
            Ok(
                dlpack::export_legacy(Tensor::from_vec(output, vec![indices.len(), 3, 224, 224])?)?
                    .into_raw(),
            )
        }
    }
}
/// Fixed-layout benchmark counters; unknown private heaps are not represented.
#[repr(C)]
#[derive(Default)]
pub struct Stats {
    /// Actually received frames, including dependency/future frames.
    pub decoded_frames: u64,
    /// Actual encoded source payload read by the mutable cursor.
    pub source_bytes: u64,
    /// Actual seek calls in this batch.
    pub seeks: u64,
    /// Canonical owned allocation high-water mark, cumulative.
    pub owned_peak_bytes: u64,
    /// Constructor/index setup wall time.
    pub setup_ns: u64,
    /// Full metadata-index construction elapsed time, zero when reusing a setup index.
    pub index_build_ns: u64,
    /// Actual cold construction decode count.
    pub index_decoded_frames: u64,
    /// Actual cold construction encoded bytes read.
    pub index_source_bytes: u64,
    /// Attached source-validated timing indexes reused by this batch.
    pub index_cache_hits: u64,
    /// Known unique selected surface payload peak, excludes decoder-private pools.
    pub native_selected_peak_bytes: u64,
    /// Facade native reservation high-water mark, cumulative (direct bypasses Session).
    pub session_native_peak_bytes: u64,
    /// Facade accounted host payload high-water mark, cumulative.
    pub host_peak_bytes: u64,
}

#[allow(unsafe_code)]
mod abi {
    use super::*;
    use std::{
        cell::RefCell,
        ffi::{CStr, CString, c_char, c_void},
        panic::{AssertUnwindSafe, catch_unwind},
    };
    thread_local! {static ERROR:RefCell<CString>=RefCell::new(CString::new("no error").expect("literal"));}
    fn error(message: impl ToString) {
        ERROR.with(|error| {
            *error.borrow_mut() =
                CString::new(message.to_string().replace('\0', " ")).expect("NUL removed")
        });
    }
    fn boundary<T>(fallback: T, operation: impl FnOnce() -> Result<T>) -> T {
        match catch_unwind(AssertUnwindSafe(operation)) {
            Ok(Ok(result)) => result,
            Ok(Err(cause)) => {
                error(cause);
                fallback
            }
            Err(_) => {
                error("benchmark bridge panic");
                fallback
            }
        }
    }
    /// Benchmark-only constructor ABI version. Rebuild clients on any change.
    #[unsafe(no_mangle)]
    pub extern "C" fn mb_abi_version() -> u32 {
        2
    }
    /// Borrow the last error on this calling thread until its next bridge call.
    #[unsafe(no_mangle)]
    pub extern "C" fn mb_last_error() -> *const c_char {
        ERROR.with(|error| error.borrow().as_ptr())
    }
    /// Own one single-threaded handle. backend0=prefix/1=direct-prefix/2=indexed/3=facade-cursor/4=direct-seek, gpu0/1, warm0/1.
    /// # Safety
    /// path is NUL-terminated UTF-8 and stats points to one writable Stats.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn mb_open(
        path: *const c_char,
        backend: u32,
        gpu: u32,
        warm: u32,
        width: u32,
        height: u32,
        quality: u32,
        profile: u32,
        stats: *mut Stats,
    ) -> *mut c_void {
        boundary(std::ptr::null_mut(), || {
            if path.is_null()
                || stats.is_null()
                || backend > 4
                || gpu > 1
                || warm > 1
                || quality > 1
                || profile > 1
            {
                return Err(Error::invalid("invalid bridge constructor arguments"));
            }
            let start = Instant::now();
            // SAFETY: the caller guarantees the live terminated path and writable stats.
            let path = unsafe { CStr::from_ptr(path) }
                .to_str()
                .map_err(|_| Error::invalid("non-UTF8 fixture path"))?;
            let handle = Handle::open(
                Path::new(path),
                match backend {
                    0 => Backend::Media,
                    1 => Backend::Native,
                    2 => Backend::Indexed,
                    3 => Backend::MediaCursor,
                    _ => Backend::NativeSeek,
                },
                gpu == 1,
                warm == 1,
                width,
                height,
                quality == 1,
                profile == 1,
            )?;
            // SAFETY: caller supplied one live output Stats allocation.
            unsafe {
                *stats = Stats {
                    setup_ns: start.elapsed().as_nanos() as u64,
                    index_decoded_frames: handle
                        .index
                        .as_ref()
                        .map_or(0, |index| index.report().decoded_frames),
                    index_source_bytes: handle
                        .index
                        .as_ref()
                        .map_or(0, |index| index.report().bytes_read),
                    ..Default::default()
                };
            }
            Ok(Box::into_raw(Box::new(handle)).cast())
        })
    }
    /// Return the retained CUDA consumer queue. CPU handles return zero.
    /// # Safety
    /// handle is one live original mb_open result with no concurrent operation.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn mb_consumer_stream(handle: *mut c_void) -> u64 {
        if handle.is_null() {
            return 0;
        }
        // SAFETY: caller retains the original live, exclusively accessed handle.
        let handle = unsafe { &*handle.cast::<Handle>() };
        handle.consumer.as_ref().map_or(0, |stream| {
            stream.native_stream().cu_stream() as usize as u64
        })
    }
    /// Return one original legacy DLPack descriptor, never a host GPU pointer.
    /// # Safety
    /// Handle/stats are live/exclusive; indices covers count readable u64s. The
    /// returned descriptor is consumed exactly once by Torch on mb_consumer_stream
    /// or released once by mb_release if no consumer takes ownership.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn mb_run(
        handle: *mut c_void,
        indices: *const u64,
        count: usize,
        stats: *mut Stats,
    ) -> *mut DLManagedTensor {
        boundary(std::ptr::null_mut(), || {
            if handle.is_null() || indices.is_null() || stats.is_null() || count == 0 || count > 32
            {
                return Err(Error::invalid("invalid bridge batch arguments"));
            }
            // SAFETY: caller supplies one live exclusive handle and bounded input/output arrays.
            unsafe {
                *stats = Stats::default();
                (&mut *handle.cast::<Handle>())
                    .run(std::slice::from_raw_parts(indices, count), &mut *stats)
            }
        })
    }
    /// Wait only benchmark-owned GPU input/output retirement after the Python
    /// consumer queue completed and the imported input tensor was dropped.
    /// # Safety
    /// handle is a live exclusively accessed original mb_open result.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn mb_retire(handle: *mut c_void) -> i32 {
        boundary(-1, || {
            if handle.is_null() {
                return Err(Error::invalid("null benchmark handle"));
            }
            // SAFETY: caller retains the original exclusively accessed handle.
            unsafe { &*handle.cast::<Handle>() }.retire()?;
            Ok(0)
        })
    }
    /// Close the original handle after dropping all Python outputs on their queue.
    /// # Safety
    /// handle is a live original result, not previously closed or concurrently used.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn mb_close(handle: *mut c_void) {
        if !handle.is_null() {
            // SAFETY: caller transfers the unique original allocation back exactly once.
            drop(unsafe { Box::from_raw(handle.cast::<Handle>()) });
        }
    }
    /// Release an original descriptor only if Torch never consumed its capsule.
    /// # Safety
    /// pointer is one live unconsumed original mb_run result and has not been released.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn mb_release(pointer: *mut DLManagedTensor) {
        if !pointer.is_null() {
            // SAFETY: caller transfers the unique live legacy descriptor back exactly once.
            unsafe {
                if let Some(deleter) = (*pointer).deleter {
                    deleter(pointer);
                }
            }
        }
    }
}
