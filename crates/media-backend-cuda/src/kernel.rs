//! Fused NV12/P010 crop, resize, YUV conversion, normalization, and f32 packing.
//!
//! Output is clamped nonlinear RGB in `[0, 1]` before channel-wise normalization.
//! This does not linearize transfer functions, apply ICC profiles, or tone-map HDR.
//! Such conversions require an explicit implementation rather than an assumption.

use cudarc::driver::{CudaFunction, DevicePtr, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};
use media_core::device::Device;
use media_core::error::{Error, Result};
use media_core::media::{ImageLayout, PixelFormat};
use media_core::policy::CopyPolicy;
use media_core::storage::StorageLease;

use crate::context::{CudaContext, CudaStream, cuda_error, lock, nvrtc_present};
use crate::memory::CudaBuffer;

/// Established luma coefficients for nonlinear YUV-to-RGB conversion.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum YuvMatrix {
    /// ITU-R BT.601 coefficients.
    Bt601,
    /// ITU-R BT.709 coefficients.
    Bt709,
    /// BT.2020 non-constant-luminance coefficients; no HDR tone mapping.
    Bt2020,
}

/// Established YUV code-value range.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum YuvRange {
    /// Studio-swing Y/UV ranges.
    Limited,
    /// Full-range code values.
    Full,
}

/// Position of the 4:2:0 chroma sample relative to luma pixel centers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChromaSiting {
    /// Chroma samples centered over each 2x2 luma group.
    Center,
    /// Chroma sample co-sited with the top-left luma sample.
    TopLeft,
    /// Horizontally co-sited, vertically centered.
    Left,
}

/// Contiguous four-dimensional tensor packing with a leading batch dimension of one.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum TensorLayout {
    /// Batch, RGB channels, height, width.
    #[default]
    Nchw,
    /// Batch, height, width, RGB channels.
    Nhwc,
}

/// Spatial interpolation and the default chroma reconstruction filter.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Interpolation {
    /// Round source coordinates to the nearest sample center, ties upward.
    Nearest,
    /// Bilinear sampling with half-pixel resize coordinates and clamped edges.
    #[default]
    Bilinear,
}

/// Chroma reconstruction independently of luma/RGB spatial interpolation.
/// Source chroma siting, YUV matrix and code range retain their declared meaning.
/// Output remains f32; matching an integer-RGB pipeline also requires accounting
/// for that pipeline's separate quantization step.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ChromaInterpolation {
    /// Preserve the spatial interpolation rule, including Nearest's upward ties.
    #[default]
    MatchSpatial,
    /// Reconstruct UV using the nearest sample center, choosing the earlier
    /// sample at exact half-way ties. Declared chroma siting is still respected.
    /// At integer luma pixels this repeats each 4:2:0 UV pair over its 2x2 block
    /// for Left, TopLeft and Center siting. RGB/luma resize interpolation and
    /// floating-point output precision are unchanged.
    NearestTiesEarlier,
}

/// Domain in which resize interpolation occurs.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ResizeDomain {
    /// Convert/clamp source RGB pixels, then interpolate RGB values.
    #[default]
    ConvertedRgb,
    /// Interpolate YUV planes first, then convert and clamp the result.
    ///
    /// This explicitly selected operation differs on saturated edges and
    /// subsampled chroma; it is not a semantics-preserving RGB resize fusion.
    YuvPlanes,
}

/// Validated channel-wise normalization applied after conversion to RGB `[0,1]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Normalization {
    /// RGB channel means.
    pub mean: [f32; 3],
    /// Positive RGB standard deviations.
    pub std: [f32; 3],
}
impl Default for Normalization {
    fn default() -> Self {
        Self {
            mean: [0.0; 3],
            std: [1.0; 3],
        }
    }
}
impl Normalization {
    /// Reject nonfinite means and nonpositive/nonfinite standard deviations.
    pub fn validate(self) -> Result<()> {
        if self.mean.iter().any(|value| !value.is_finite())
            || self
                .std
                .iter()
                .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(Error::invalid(
                "CUDA normalization requires finite means and finite positive standard deviations",
            ));
        }
        if self.mean.iter().zip(self.std).any(|(&mean, std)| {
            !((0.0 - mean) / std).is_finite() || !((1.0 - mean) / std).is_finite()
        }) {
            return Err(Error::invalid(
                "CUDA normalization would overflow finite f32 output",
            ));
        }
        Ok(())
    }
}

/// Integer crop rectangle in visible source coordinates.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CropRect {
    /// Left pixel index.
    pub x: u32,
    /// Top pixel index.
    pub y: u32,
    /// Positive crop width.
    pub width: u32,
    /// Positive crop height.
    pub height: u32,
}

/// Fully explicit preprocessing request for one source frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PreprocessSpec {
    /// Output width.
    pub width: u32,
    /// Output height.
    pub height: u32,
    /// Crop before resizing; absent means the full visible image.
    pub crop: Option<CropRect>,
    /// Physical contiguous RGB packing.
    pub layout: TensorLayout,
    /// Luma/RGB spatial sampling, also used for chroma under MatchSpatial.
    pub interpolation: Interpolation,
    /// Source chroma reconstruction; MatchSpatial preserves existing defaults.
    pub chroma_interpolation: ChromaInterpolation,
    /// Explicit RGB-versus-plane interpolation order.
    pub resize_domain: ResizeDomain,
    /// RGB normalization.
    pub normalization: Normalization,
}

impl PreprocessSpec {
    /// Create a request with bilinear spatial/chroma sampling, NCHW packing,
    /// and RGB `[0,1]` output.
    pub fn new(width: u32, height: u32) -> Result<Self> {
        let value = Self {
            width,
            height,
            crop: None,
            layout: TensorLayout::Nchw,
            interpolation: Interpolation::Bilinear,
            chroma_interpolation: ChromaInterpolation::MatchSpatial,
            resize_domain: ResizeDomain::ConvertedRgb,
            normalization: Normalization::default(),
        };
        value.output_elements()?;
        Ok(value)
    }

    /// Validate all dimensions, crop bounds, and normalization without a GPU.
    pub fn validate(&self, input: &ImageLayout) -> Result<()> {
        self.output_elements()?;
        self.normalization.validate()?;
        if input.width == 0
            || input.height == 0
            || input.width > 16_777_215
            || input.height > 16_777_215
        {
            return Err(Error::invalid(
                "CUDA input dimensions must be positive and within exact f32 integer coordinates",
            ));
        }
        let crop = self.resolved_crop(input);
        if crop.width == 0
            || crop.height == 0
            || crop
                .x
                .checked_add(crop.width)
                .is_none_or(|end| end > input.width)
            || crop
                .y
                .checked_add(crop.height)
                .is_none_or(|end| end > input.height)
        {
            return Err(Error::invalid(
                "CUDA crop must lie within the visible input image",
            ));
        }
        Ok(())
    }

    /// Required contiguous float elements, with checked kernel/address limits.
    pub fn output_elements(&self) -> Result<usize> {
        if self.width == 0
            || self.height == 0
            || self.width > 16_777_215
            || self.height > 16_777_215
        {
            return Err(Error::invalid(
                "CUDA output dimensions must be positive and within exact f32 integer coordinates",
            ));
        }
        let pixels = self
            .width
            .checked_mul(self.height)
            .ok_or_else(|| Error::overflow("CUDA pixel launch count exceeds u32"))?;
        (pixels as usize)
            .checked_mul(3)
            .and_then(|elements| elements.checked_mul(4).map(|_| elements))
            .ok_or_else(|| Error::overflow("CUDA output tensor byte count overflow"))
    }

    /// Four-dimensional shape in the requested physical layout.
    pub fn shape(&self) -> [usize; 4] {
        match self.layout {
            TensorLayout::Nchw => [1, 3, self.height as usize, self.width as usize],
            TensorLayout::Nhwc => [1, self.height as usize, self.width as usize, 3],
        }
    }

    /// Float elements in one final batch allocation, checked before GPU work.
    pub fn batch_elements(&self, count: usize) -> Result<usize> {
        if count == 0 {
            return Err(Error::invalid(
                "CUDA preprocessing batch must contain at least one image",
            ));
        }
        self.output_elements()?
            .checked_mul(count)
            .and_then(|elements| elements.checked_mul(4).map(|_| elements))
            .ok_or_else(|| Error::overflow("CUDA batch tensor byte count overflow"))
    }

    /// Contiguous batch shape, rejecting empty or unrepresentable allocations.
    pub fn batch_shape(&self, count: usize) -> Result<[usize; 4]> {
        self.batch_elements(count)?;
        let mut shape = self.shape();
        shape[0] = count;
        Ok(shape)
    }

    fn resolved_crop(&self, input: &ImageLayout) -> CropRect {
        self.crop.unwrap_or(CropRect {
            x: 0,
            y: 0,
            width: input.width,
            height: input.height,
        })
    }
}

/// Owned addressable CUDA NV12/P010 image with established color interpretation.
///
/// P010 samples have ten valid high bits in little-endian 16-bit storage. Native
/// opaque decoder surfaces require a compatible materialization or an adapter;
/// this constructor does not invent device pointers for them.
#[derive(Clone, Debug)]
pub struct CudaYuvImage {
    storage: YuvStorage,
    layout: ImageLayout,
    matrix: YuvMatrix,
    range: YuvRange,
    chroma: ChromaSiting,
}

#[derive(Clone, Debug)]
enum YuvStorage {
    Owned(CudaBuffer<u8>),
    External {
        owner: StorageLease,
        plane_addresses: [u64; 2],
    },
}

impl CudaYuvImage {
    /// Validate and retain an addressable NV12/P010 allocation and color contract.
    pub fn new(
        buffer: CudaBuffer<u8>,
        layout: ImageLayout,
        matrix: YuvMatrix,
        range: YuvRange,
        chroma: ChromaSiting,
    ) -> Result<Self> {
        Self::validate_layout(&layout, buffer.byte_len())?;
        layout.validate(&buffer.storage_lease())?;
        Ok(Self {
            storage: YuvStorage::Owned(buffer),
            layout,
            matrix,
            range,
            chroma,
        })
    }

    /// Import independently addressed native Y/UV planes without downloading them.
    ///
    /// Each address points directly at that plane's first sample. Layout offsets
    /// describe accounting only. An opaque owner receives no fabricated host view.
    ///
    /// # Safety
    /// Both addresses must be live CUDA allocations in `owner.device()`'s exact
    /// native context and cover the checked plane pitches/extents. The owner must
    /// retain decoder/frame/context resources, and its readiness token must order
    /// producer writes before queued reads. No external writer may mutate these
    /// planes while this image or a queued operation retains the owner.
    #[allow(unsafe_code)]
    pub unsafe fn from_external_planes(
        owner: StorageLease,
        layout: ImageLayout,
        plane_addresses: [u64; 2],
        matrix: YuvMatrix,
        range: YuvRange,
        chroma: ChromaSiting,
    ) -> Result<Self> {
        Self::validate_layout(&layout, owner.byte_len())?;
        if !matches!(owner.device(), Device::Cuda { .. }) || plane_addresses.contains(&0) {
            return Err(Error::incompatible(
                "native CUDA planes need nonzero addresses and actual context ownership",
            ));
        }
        if layout.format == PixelFormat::P010
            && plane_addresses.iter().any(|address| address % 2 != 0)
        {
            return Err(Error::invalid(
                "native P010 plane addresses must be 16-bit aligned",
            ));
        }
        Ok(Self {
            storage: YuvStorage::External {
                owner,
                plane_addresses,
            },
            layout,
            matrix,
            range,
            chroma,
        })
    }

    /// Validate plane/address requirements without touching a GPU.
    pub fn validate_layout(layout: &ImageLayout, byte_len: usize) -> Result<()> {
        if !matches!(layout.format, PixelFormat::Nv12 | PixelFormat::P010) {
            return Err(Error::unsupported(
                "CUDA preprocessing accepts addressable NV12 or P010",
            ));
        }
        if layout.width == 0
            || layout.height == 0
            || layout.allocated_width < layout.width
            || layout.allocated_height < layout.height
            || layout.planes.len() != 2
        {
            return Err(Error::invalid(
                "CUDA YUV image requires visible dimensions and two valid planes",
            ));
        }
        let bytes = if layout.format == PixelFormat::P010 {
            2
        } else {
            1
        };
        let y_row = (layout.allocated_width as usize)
            .checked_mul(bytes)
            .ok_or_else(|| Error::overflow("CUDA YUV row overflow"))?;
        let uv_row = (layout.allocated_width as usize)
            .div_ceil(2)
            .checked_mul(2 * bytes)
            .ok_or_else(|| Error::overflow("CUDA UV row overflow"))?;
        let y = &layout.planes[0];
        let uv = &layout.planes[1];
        if y.row_bytes < y_row
            || uv.row_bytes < uv_row
            || y.rows < layout.allocated_height as usize
            || uv.rows < (layout.allocated_height as usize).div_ceil(2)
            || y.extent()? > uv.offset
            || uv.extent()? > byte_len
        {
            return Err(Error::invalid(
                "CUDA YUV planes overlap, exceed storage, or have insufficient geometry",
            ));
        }
        if bytes == 2
            && [y.offset, y.row_stride, uv.offset, uv.row_stride]
                .iter()
                .any(|value| value % 2 != 0)
        {
            return Err(Error::invalid(
                "P010 plane offsets and pitches must be 16-bit aligned",
            ));
        }
        Ok(())
    }

    /// Owned input allocation, or `None` for an imported native decoder surface.
    pub fn buffer(&self) -> Option<&CudaBuffer<u8>> {
        match &self.storage {
            YuvStorage::Owned(buffer) => Some(buffer),
            YuvStorage::External { .. } => None,
        }
    }

    /// Established visible and allocated plane geometry.
    pub fn layout(&self) -> &ImageLayout {
        &self.layout
    }

    /// Execute the fused transform on a compatible stream with no host pixel crossing.
    pub fn preprocess(&self, stream: &CudaStream, spec: PreprocessSpec) -> Result<CudaTensor> {
        preprocess_batch(std::slice::from_ref(self), stream, spec)
    }

    /// Actual owning device/context, including imported native surfaces.
    pub fn device(&self) -> Device {
        match &self.storage {
            YuvStorage::Owned(buffer) => buffer.device(),
            YuvStorage::External { owner, .. } => owner.device(),
        }
    }

    fn enqueue_into(
        &self,
        stream: &CudaStream,
        function: &CudaFunction,
        output: &mut cudarc::driver::CudaSlice<f32>,
        spec: PreprocessSpec,
        output_offset: usize,
    ) -> Result<()> {
        match &self.storage {
            YuvStorage::Owned(buffer) => {
                buffer.read_on(stream, |input| {
                    let (base, _mark_read) = input.device_ptr(stream.native_stream());
                    stream.context().check_errors()?;
                    let y = base
                        .checked_add(self.layout.planes[0].offset as u64)
                        .ok_or_else(|| Error::overflow("CUDA Y address overflow"))?;
                    let uv = base
                        .checked_add(self.layout.planes[1].offset as u64)
                        .ok_or_else(|| Error::overflow("CUDA UV address overflow"))?;
                    launch_preprocess(
                        stream,
                        function,
                        [y, uv],
                        output,
                        &self.layout,
                        self.matrix,
                        self.range,
                        self.chroma,
                        spec,
                        output_offset,
                    )
                })?;
            }
            YuvStorage::External {
                owner,
                plane_addresses,
            } => {
                stream.ensure_device(owner.device())?;
                owner.readiness().depend_on(stream.queue())?;
                launch_preprocess(
                    stream,
                    function,
                    *plane_addresses,
                    output,
                    &self.layout,
                    self.matrix,
                    self.range,
                    self.chroma,
                    spec,
                    output_offset,
                )?;
                let completed = match stream.record_completion() {
                    Ok(completed) => completed,
                    Err(error) => {
                        if let Err(retirement) = crate::memory::confirm_retirement(stream) {
                            std::mem::forget((owner.clone(), stream.clone()));
                            return Err(retirement);
                        }
                        return Err(error);
                    }
                };
                stream.retire_external_after(completed, owner.clone())?;
            }
        }
        Ok(())
    }
}

/// Prepare caller-ordered images directly into one contiguous GPU allocation.
///
/// Every input's parameters/context and the full output byte size are checked
/// before module compilation, allocation, or launch. Repeated input aliases get
/// distinct output slots. Each native owner is retained through its queued read.
pub fn preprocess_batch(
    images: &[CudaYuvImage],
    stream: &CudaStream,
    spec: PreprocessSpec,
) -> Result<CudaTensor> {
    let elements = spec.batch_elements(images.len())?;
    for (index, image) in images.iter().enumerate() {
        spec.validate(&image.layout)
            .map_err(|error| error.with_operation(format!("CUDA batch input {index}")))?;
        stream.ensure_device(image.device())?;
    }
    // Reserve the whole final tensor before compilation or any CUDA allocation.
    let reservation = stream.reserve::<f32>(elements)?;
    let function = preprocess_function(stream.context())?;
    let mut output = stream.allocate_reserved_zeros::<f32>(elements, reservation)?;
    let per_image = spec.output_elements()?;
    for (index, image) in images.iter().enumerate() {
        image.enqueue_into(
            stream,
            &function,
            output.slice_mut()?,
            spec,
            index * per_image,
        )?;
    }
    let buffer = output.finish()?;
    Ok(CudaTensor {
        buffer,
        shape: spec.batch_shape(images.len())?,
        layout: spec.layout,
    })
}

impl CudaStream {
    /// Prepare a batch on this stream directly into its final tensor allocation.
    pub fn preprocess_batch(
        &self,
        images: &[CudaYuvImage],
        spec: PreprocessSpec,
    ) -> Result<CudaTensor> {
        preprocess_batch(images, self, spec)
    }
}

/// Contiguous single-frame RGB f32 tensor retaining native allocation and readiness.
#[derive(Clone, Debug)]
pub struct CudaTensor {
    buffer: CudaBuffer<f32>,
    shape: [usize; 4],
    layout: TensorLayout,
}

impl CudaTensor {
    /// Retain session/pool resources through all native allocation aliases.
    pub fn retain<G: std::fmt::Debug + Send + Sync + 'static>(mut self, guard: G) -> Self {
        self.buffer = self.buffer.retain(guard);
        self
    }
    /// Consume into exclusively owned foreign-writable storage on a checked queue.
    ///
    /// Every buffer/storage alias is counted. `Forbid` rejects aliases, while
    /// `Allow` detaches on device only when needed. Prior queued readers and
    /// producer writes are ordered before foreign work without a device barrier.
    pub fn into_foreign_writable(
        self,
        consumer: &CudaStream,
        copy: CopyPolicy,
    ) -> Result<ForeignCudaTensor> {
        consumer.ensure_device(self.device())?;
        let Self {
            mut buffer,
            shape,
            layout,
        } = self;
        let copied = !buffer.is_uniquely_owned();
        if copied {
            if copy == CopyPolicy::Forbid {
                return Err(Error::incompatible(
                    "CUDA tensor has buffer/storage aliases; writable export forbids the required device copy",
                ));
            }
            buffer = consumer.copy(&buffer)?;
        }
        let pointer = buffer.exclusive_pointer(consumer)?;
        Ok(ForeignCudaTensor {
            buffer,
            consumer: consumer.clone(),
            pointer,
            shape,
            layout,
            copied,
        })
    }
    /// Native typed allocation for checked external integrations.
    pub fn buffer(&self) -> &CudaBuffer<f32> {
        &self.buffer
    }
    /// Physical tensor shape including leading batch dimension one.
    pub fn shape(&self) -> [usize; 4] {
        self.shape
    }
    /// Contiguous strides in float elements, not bytes.
    pub fn strides(&self) -> [usize; 4] {
        [
            self.shape[1] * self.shape[2] * self.shape[3],
            self.shape[2] * self.shape[3],
            self.shape[3],
            1,
        ]
    }
    /// Physical RGB tensor layout.
    pub fn layout(&self) -> TensorLayout {
        self.layout
    }
    /// Actual owning context.
    pub fn device(&self) -> Device {
        self.buffer.device()
    }
    /// Retained core storage with no implicit host view.
    pub fn storage_lease(&self) -> StorageLease {
        self.buffer.storage_lease()
    }
    /// Explicit host transfer after relevant producer work has completed.
    pub fn download(&self, stream: &CudaStream) -> Result<Vec<f32>> {
        stream.download(&self.buffer)
    }
}

/// Exclusive foreign-writable allocation retained through consumer completion.
///
/// No buffer/storage alias APIs are exposed. Foreign work must be queued on the
/// supplied consumer stream before its owner/deleter releases this object. Drop
/// records and waits a consumer event, falling back to stream synchronization if
/// recording fails; producer readiness alone is never used for retirement.
#[derive(Debug)]
pub struct ForeignCudaTensor {
    buffer: CudaBuffer<f32>,
    consumer: CudaStream,
    pointer: u64,
    shape: [usize; 4],
    layout: TensorLayout,
    copied: bool,
}
impl ForeignCudaTensor {
    /// Device address of the exclusively retained contiguous f32 allocation.
    pub fn device_pointer(&self) -> u64 {
        self.pointer
    }
    /// Four-dimensional physical shape.
    pub fn shape(&self) -> [usize; 4] {
        self.shape
    }
    /// Contiguous strides in float elements.
    pub fn strides(&self) -> [usize; 4] {
        [
            self.shape[1] * self.shape[2] * self.shape[3],
            self.shape[2] * self.shape[3],
            self.shape[3],
            1,
        ]
    }
    /// Actual owning native context and ordinal.
    pub fn device(&self) -> Device {
        self.buffer.device()
    }
    /// Physical tensor packing.
    pub fn layout(&self) -> TensorLayout {
        self.layout
    }
    /// Whether explicit device detachment was needed because aliases existed.
    pub fn copied(&self) -> bool {
        self.copied
    }
    /// Wait for currently submitted consumer work, keeping storage retained.
    pub fn wait(&self) -> Result<()> {
        self.consumer.record_completion()?.wait()
    }
}
impl Drop for ForeignCudaTensor {
    fn drop(&mut self) {
        let waited = self
            .consumer
            .inner
            .native
            .record_event(None)
            .and_then(|event| event.synchronize());
        if let Err(error) = waited {
            let context = self.consumer.context();
            if let Err(stream_error) = self.consumer.inner.native.synchronize() {
                context.inner.native.record_err(Err::<(), _>(stream_error));
                context.record_retirement_failure(stream_error.to_string());
                std::mem::forget((self.buffer.clone(), self.consumer.clone()));
            } else {
                context.record_retirement_failure(format!(
                    "foreign consumer event failed; stream fallback completed safely: {error}"
                ));
            }
        }
    }
}

fn preprocess_function(context: &CudaContext) -> Result<CudaFunction> {
    let mut cached = lock(&context.inner.preprocess)?;
    if let Some(function) = cached.as_ref() {
        return Ok(function.clone());
    }
    if !nvrtc_present() {
        return Err(Error::unsupported(
            "CUDA preprocessing requires NVRTC; install the CUDA development runtime or use a devel image",
        ));
    }
    let options = CompileOptions {
        arch: Some("compute_75"),
        fmad: Some(false),
        options: vec!["--std=c++11".into()],
        ..CompileOptions::default()
    };
    let ptx = compile_ptx_with_opts(include_str!("preprocess.cu"), options).map_err(|error| {
        Error::backend(
            "nvrtc",
            format!("preprocessing compilation failed: {error}"),
        )
    })?;
    let module = context.inner.native.load_module(ptx).map_err(cuda_error)?;
    let function = module.load_function("preprocess_yuv").map_err(cuda_error)?;
    *cached = Some(function.clone());
    Ok(function)
}

#[allow(clippy::too_many_arguments, unsafe_code)]
fn launch_preprocess(
    stream: &CudaStream,
    function: &CudaFunction,
    plane_addresses: [u64; 2],
    output: &mut cudarc::driver::CudaSlice<f32>,
    image: &ImageLayout,
    matrix: YuvMatrix,
    range: YuvRange,
    chroma: ChromaSiting,
    spec: PreprocessSpec,
    output_offset: usize,
) -> Result<()> {
    let crop = spec.resolved_crop(image);
    let input_width = image.width as i32;
    let input_height = image.height as i32;
    let y_address = plane_addresses[0];
    let uv_address = plane_addresses[1];
    let y_offset = 0_u64;
    let y_pitch = image.planes[0].row_stride as u64;
    let uv_offset = 0_u64;
    let uv_pitch = image.planes[1].row_stride as u64;
    let (crop_x, crop_y, crop_width, crop_height) = (
        crop.x as i32,
        crop.y as i32,
        crop.width as i32,
        crop.height as i32,
    );
    let (width, height) = (spec.width as i32, spec.height as i32);
    let p010 = i32::from(image.format == PixelFormat::P010);
    let nhwc = i32::from(spec.layout == TensorLayout::Nhwc);
    let bilinear = i32::from(spec.interpolation == Interpolation::Bilinear);
    let chroma_filter = match spec.chroma_interpolation {
        ChromaInterpolation::MatchSpatial => bilinear,
        ChromaInterpolation::NearestTiesEarlier => 2,
    };
    let resize_rgb = i32::from(spec.resize_domain == ResizeDomain::ConvertedRgb);
    let (y_bias, y_scale, c_bias, c_scale): (f32, f32, f32, f32) = match (p010, range) {
        (0, YuvRange::Limited) => (16.0, 219.0, 128.0, 224.0),
        (0, YuvRange::Full) => (0.0, 255.0, 128.0, 255.0),
        (_, YuvRange::Limited) => (64.0, 876.0, 512.0, 896.0),
        (_, YuvRange::Full) => (0.0, 1023.0, 512.0, 1023.0),
    };
    let (kr, kb): (f32, f32) = match matrix {
        YuvMatrix::Bt601 => (0.299, 0.114),
        YuvMatrix::Bt709 => (0.2126, 0.0722),
        YuvMatrix::Bt2020 => (0.2627, 0.0593),
    };
    let (chroma_x, chroma_y): (f32, f32) = match chroma {
        ChromaSiting::Center => (0.5, 0.5),
        ChromaSiting::TopLeft => (0.0, 0.0),
        ChromaSiting::Left => (0.0, 0.5),
    };
    let mean = spec.normalization.mean;
    let std = spec.normalization.std;
    let mut args = stream.inner.native.launch_builder(function);
    let output_offset = output_offset as u64;
    args.arg(&y_address)
        .arg(output)
        .arg(&output_offset)
        .arg(&uv_address)
        .arg(&input_width)
        .arg(&input_height)
        .arg(&y_offset)
        .arg(&y_pitch)
        .arg(&uv_offset)
        .arg(&uv_pitch)
        .arg(&crop_x)
        .arg(&crop_y)
        .arg(&crop_width)
        .arg(&crop_height)
        .arg(&width)
        .arg(&height)
        .arg(&p010)
        .arg(&nhwc)
        .arg(&bilinear)
        .arg(&chroma_filter)
        .arg(&resize_rgb)
        .arg(&y_bias)
        .arg(&y_scale)
        .arg(&c_bias)
        .arg(&c_scale)
        .arg(&kr)
        .arg(&kb)
        .arg(&chroma_x)
        .arg(&chroma_y)
        .arg(&mean[0])
        .arg(&mean[1])
        .arg(&mean[2])
        .arg(&std[0])
        .arg(&std[1])
        .arg(&std[2]);
    let pixels = spec.width * spec.height;
    let config = LaunchConfig {
        block_dim: (256, 1, 1),
        grid_dim: (pixels.div_ceil(256), 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: The checked layout bounds every luma/chroma read, crop and output
    // dimensions were validated, and output holds exactly width*height*3 floats.
    // Argument order/types match preprocess.cu. Only output is mutated; cudarc
    // records output writes. Owned input uses a DevicePtr read-event guard;
    // external input is retained until a post-kernel completion event.
    unsafe { args.launch(config) }.map_err(cuda_error)?;
    Ok(())
}
