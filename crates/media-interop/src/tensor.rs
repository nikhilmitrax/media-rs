//! Explicit f32 RGB tensor packing and shared, aligned host tensor ownership.

use crate::color::IccTransform;
pub use crate::color::{IccGamut, IccLimits, IccPolicy};
use crate::image::HostImageView;
use media_core::{
    error::{Error, Result},
    media::{Image, Orientation, PixelFormat},
};
use std::{collections::BTreeMap, fmt, sync::Arc};

/// Physical packing of a four-dimensional image tensor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TensorLayout {
    /// Batch, channel, height, width.
    Nchw,
    /// Batch, height, width, channel.
    Nhwc,
}
/// Numeric scale before optional channel normalization.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NumericRange {
    /// Integer full scale maps to zero through one; input f32 is already unit-scaled.
    Unit,
    /// Unit-scaled values are multiplied by 255, including 16-bit input.
    Byte,
}
/// Policy for missing RGB color descriptions.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ColorAssumption {
    /// Require established sRGB/BT.709 metadata, or an explicitly converted RGB ICC.
    #[default]
    RequireKnown,
    /// Fill only missing fields with sRGB/full-range assumptions and report them.
    AssumeSrgb,
}
/// Treatment of stored alpha while producing RGB model input.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum AlphaPolicy {
    /// Reject alpha-bearing input until a policy is explicit.
    #[default]
    Reject,
    /// Discard alpha and preserve stored RGB; does not unpremultiply or composite.
    Discard,
}

/// Exact output color/numeric/layout requirements for RGB f32 preparation.
#[derive(Clone, Debug)]
pub struct TensorSpec {
    layout: TensorLayout,
    range: NumericRange,
    mean: [f32; 3],
    std: [f32; 3],
    color: ColorAssumption,
    alpha: AlphaPolicy,
    max_bytes: usize,
    icc: IccPolicy,
    icc_limits: IccLimits,
}
impl TensorSpec {
    /// RGB f32, NCHW, unit scale, no normalization, strict color/alpha policy.
    pub fn rgb_f32() -> Self {
        Self {
            layout: TensorLayout::Nchw,
            range: NumericRange::Unit,
            mean: [0.0; 3],
            std: [1.0; 3],
            color: ColorAssumption::RequireKnown,
            alpha: AlphaPolicy::Reject,
            max_bytes: 256 * 1024 * 1024,
            icc: IccPolicy::Reject,
            icc_limits: Default::default(),
        }
    }
    /// Choose physical output packing, not merely an axis label.
    pub fn layout(mut self, layout: TensorLayout) -> Self {
        self.layout = layout;
        self
    }
    /// Choose numeric scale before normalization.
    pub fn range(mut self, range: NumericRange) -> Self {
        self.range = range;
        self
    }
    /// Subtract each channel's mean and divide by its standard deviation.
    /// Non-finite coefficients or non-positive standard deviations are rejected.
    pub fn normalize(mut self, mean: [f32; 3], std: [f32; 3]) -> Self {
        self.mean = mean;
        self.std = std;
        self
    }
    /// Select missing-color behavior for input without an authoritative ICC profile.
    /// Contradictory known metadata is never replaced by an sRGB assumption.
    pub fn color_assumption(mut self, policy: ColorAssumption) -> Self {
        self.color = policy;
        self
    }
    /// Select alpha treatment.
    pub fn alpha(mut self, policy: AlphaPolicy) -> Self {
        self.alpha = policy;
        self
    }
    /// Explicitly convert embedded RGB ICC profiles to nonlinear sRGB or reject.
    /// Alpha Discard first treats stored RGB as opaque device RGB, without
    /// unpremultiplication/composition. CMS parsing requires the `icc` feature.
    pub fn icc_policy(mut self, policy: IccPolicy) -> Self {
        self.icc = policy;
        self
    }
    /// Bound ICC profile/CLUT/TRC parsing separately from final tensor storage.
    /// CMS private heap is unmeasured and reported, rather than counted as pixels.
    pub fn icc_limits(mut self, limits: IccLimits) -> Self {
        self.icc_limits = limits;
        self
    }
    /// Bound the output allocation, including alignment padding.
    pub fn max_bytes(mut self, bytes: usize) -> Self {
        self.max_bytes = bytes;
        self
    }
    /// Inspect the requested physical layout.
    pub fn tensor_layout(&self) -> TensorLayout {
        self.layout
    }
    /// Configured output-allocation limit, including alignment padding.
    pub fn allocation_limit(&self) -> usize {
        self.max_bytes
    }
    /// Validate all numeric settings before allocating output.
    pub fn validate(&self) -> Result<()> {
        self.icc_limits.validate()?;
        if self.mean.iter().any(|v| !v.is_finite())
            || self.std.iter().any(|v| !v.is_finite() || *v <= 0.0)
        {
            return Err(Error::invalid(
                "normalization requires finite means and finite positive standard deviations",
            ));
        }
        Ok(())
    }
}
impl Default for TensorSpec {
    fn default() -> Self {
        Self::rgb_f32()
    }
}

pub(crate) trait Retained: fmt::Debug + Send + Sync {}
impl<T: fmt::Debug + Send + Sync> Retained for T {}

/// Shared immutable f32 host tensor with checked shape and element strides.
///
/// Host constructors explicitly establish a 256-byte-aligned logical data region
/// inside a Vec allocation, with at most 63 extra f32s. Cloning shares that vector;
/// no shape permutation is represented as a physical data conversion.
#[derive(Clone)]
pub struct Tensor {
    pub(crate) data: Arc<Vec<f32>>,
    pub(crate) offset: usize,
    shape: Vec<usize>,
    strides: Vec<usize>,
    layout: Option<TensorLayout>,
    assumptions: Vec<String>,
    color_operations: Vec<String>,
    retained: Vec<Arc<dyn Retained>>,
}
impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tensor")
            .field("shape", &self.shape)
            .field("strides", &self.strides)
            .field("layout", &self.layout)
            .field("assumptions", &self.assumptions)
            .field("color_operations", &self.color_operations)
            .finish()
    }
}
impl Tensor {
    /// Own numeric values with a checked row-major shape, establishing aligned storage.
    /// This is an explicit materialization; it may reserve and move the data in place.
    pub fn from_vec(mut values: Vec<f32>, shape: Vec<usize>) -> Result<Self> {
        if shape.len() > 32 {
            return Err(Error::invalid("host tensors support at most 32 dimensions"));
        }
        let elements = shape
            .iter()
            .try_fold(1_usize, |size, dim| size.checked_mul(*dim))
            .ok_or_else(|| Error::overflow("tensor dimensions overflow"))?;
        if elements != values.len() {
            return Err(Error::invalid("tensor shape does not match numeric data"));
        }
        let mut strides = vec![1; shape.len()];
        let mut stride = 1_usize;
        for (index, size) in shape.iter().enumerate().rev() {
            strides[index] = stride;
            stride = stride
                .checked_mul(*size)
                .ok_or_else(|| Error::overflow("tensor strides overflow"))?;
        }
        let offset = if elements == 0 {
            0
        } else {
            let padded = elements
                .checked_add(63)
                .ok_or_else(|| Error::overflow("aligned tensor storage"))?;
            values
                .try_reserve_exact(63)
                .map_err(|e| Error::exhausted(e.to_string()))?;
            values.resize(padded, 0.0);
            let offset = values.as_ptr().align_offset(256);
            if offset > 63 {
                return Err(Error::incompatible(
                    "host f32 allocation cannot establish DLPack alignment",
                ));
            }
            values.copy_within(0..elements, offset);
            offset
        };
        Ok(Self {
            data: Arc::new(values),
            offset,
            shape,
            strides,
            layout: None,
            assumptions: Vec::new(),
            color_operations: Vec::new(),
            retained: Vec::new(),
        })
    }
    /// Existing numeric data. No representation conversion or copy occurs.
    pub fn data(&self) -> &[f32] {
        let length = self.shape.iter().product::<usize>();
        &self.data[self.offset..self.offset + length]
    }
    /// Allocated f32 buffer bytes, including alignment padding and vector capacity.
    pub fn allocation_bytes(&self) -> usize {
        self.data.capacity() * std::mem::size_of::<f32>()
    }
    /// Checked logical shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    /// Strides measured in f32 elements, not bytes.
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }
    /// Image packing when created by to_tensor.
    pub fn layout(&self) -> Option<TensorLayout> {
        self.layout
    }
    /// Missing color metadata that was explicitly assumed.
    pub fn assumptions(&self) -> &[String] {
        &self.assumptions
    }
    /// Explicit ICC conversions, gamut choices, and unmeasured CMS heap notes.
    pub fn color_operations(&self) -> &[String] {
        &self.color_operations
    }
    /// Whether another handle shares this numeric allocation.
    pub fn shares_storage(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.data, &other.data)
    }
    /// Retain a budget/native lifetime guard with the tensor.
    pub fn retain<T: fmt::Debug + Send + Sync + 'static>(mut self, guard: T) -> Self {
        self.retained.push(Arc::new(guard));
        self
    }
    /// Explicitly copy into independently owned aligned storage.
    pub fn deep_copy(&self) -> Result<Self> {
        let mut result = Self::from_vec(self.data().to_vec(), self.shape.clone())?;
        result.layout = self.layout;
        result.assumptions = self.assumptions.clone();
        result.color_operations = self.color_operations.clone();
        Ok(result)
    }
    fn zeroed(shape: Vec<usize>) -> Result<Self> {
        let (elements, strides) = shape_info(&shape)?;
        let length = elements
            .checked_add(if elements == 0 { 0 } else { 63 })
            .ok_or_else(|| Error::overflow("aligned tensor allocation"))?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(length)
            .map_err(|e| Error::exhausted(e.to_string()))?;
        values.resize(length, 0.0);
        let offset = if elements == 0 {
            0
        } else {
            values.as_ptr().align_offset(256)
        };
        if offset > 63 {
            return Err(Error::incompatible(
                "cannot establish aligned host tensor storage",
            ));
        }
        Ok(Self {
            data: Arc::new(values),
            offset,
            shape,
            strides,
            layout: None,
            assumptions: Vec::new(),
            color_operations: Vec::new(),
            retained: Vec::new(),
        })
    }
    pub(crate) fn into_unique(self) -> Result<TensorParts> {
        let data = Arc::try_unwrap(self.data).map_err(|_| Error::incompatible(
            "writable foreign export requires unique tensor storage; release aliases or deep_copy explicitly"))?;
        Ok(TensorParts {
            data,
            offset: self.offset,
            shape: self.shape,
            _retained: self.retained,
        })
    }
}
pub(crate) struct TensorParts {
    pub(crate) data: Vec<f32>,
    pub(crate) offset: usize,
    pub(crate) shape: Vec<usize>,
    _retained: Vec<Arc<dyn Retained>>,
}

/// Convert compatible host pixels into the requested RGB f32 packing.
///
/// This explicitly allocates and converts values. Pitched row layout is honored.
/// ICC input requires an explicit conversion policy and the optional `icc` feature.
/// Non-ICC non-sRGB, unapplied orientation, and multiplanar YUV require earlier
/// conversion; they are never silently reinterpreted.
pub fn to_tensor(image: &Image, spec: &TensorSpec) -> Result<Tensor> {
    images_to_tensor(std::slice::from_ref(image), spec)
}

/// Pack a nonempty batch directly into one NCHW or NHWC f32 allocation.
/// All images must have matching visible dimensions and pixel representation;
/// ragged batches require an explicit earlier resize/pad operation. No temporary
/// per-image tensor is created, and per-image color assumptions remain reported.
pub fn images_to_tensor(images: &[Image], spec: &TensorSpec) -> Result<Tensor> {
    spec.validate()?;
    let first = images
        .first()
        .ok_or_else(|| Error::invalid("tensor image batch must not be empty"))?;
    let (width, height, format) = (
        first.width() as usize,
        first.height() as usize,
        first.pixel_format(),
    );
    let pixels = width
        .checked_mul(height)
        .ok_or_else(|| Error::overflow("tensor image area"))?;
    let per_image = pixels
        .checked_mul(3)
        .ok_or_else(|| Error::overflow("RGB tensor size"))?;
    let elements = per_image
        .checked_mul(images.len())
        .ok_or_else(|| Error::overflow("tensor batch size"))?;
    let allocation_bytes = elements
        .checked_add(63)
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| Error::overflow("aligned tensor byte size"))?;
    if allocation_bytes > spec.max_bytes {
        return Err(Error::exhausted(
            "tensor output exceeds configured byte limit",
        ));
    }
    let bpp = format
        .bytes_per_pixel()
        .ok_or_else(|| Error::incompatible("YUV needs explicit color conversion"))?;
    let mut assumption_counts = BTreeMap::<String, usize>::new();
    for image in images {
        if image.width() as usize != width
            || image.height() as usize != height
            || image.pixel_format() != format
        {
            return Err(Error::incompatible(
                "tensor batch is ragged or changes pixel format; resize/convert explicitly",
            ));
        }
        if image.metadata().icc_profile.is_some() {
            if spec.icc == IccPolicy::Reject {
                return Err(Error::unsupported(
                    "ICC input requires explicit IccPolicy::ConvertToSrgb and the icc feature",
                ));
            }
            if !matches!(
                format,
                PixelFormat::Rgb8
                    | PixelFormat::Rgba8
                    | PixelFormat::Rgb16
                    | PixelFormat::Rgba16
                    | PixelFormat::RgbF32
                    | PixelFormat::RgbaF32
            ) {
                return Err(Error::unsupported(
                    "ICC tensor conversion supports packed RGB/RGBA representations; grayscale needs an explicit conversion",
                ));
            }
            if image
                .metadata()
                .icc_profile
                .as_ref()
                .is_some_and(|profile| profile.len() > spec.icc_limits.max_profile_bytes)
            {
                return Err(Error::exhausted(
                    "ICC profile exceeds the requested parser byte limit",
                )
                .with_constraint("max_profile_bytes"));
            }
        } else {
            for assumption in validate_color(image, spec.color)? {
                *assumption_counts.entry(assumption).or_default() += 1;
            }
        }
        if image.metadata().orientation != Orientation::Normal {
            return Err(Error::incompatible(
                "apply display orientation before tensor preparation",
            ));
        }
        let alpha = matches!(
            format,
            PixelFormat::GrayAlpha8
                | PixelFormat::GrayAlpha16
                | PixelFormat::Rgba8
                | PixelFormat::Rgba16
                | PixelFormat::RgbaF32
        );
        if alpha && spec.alpha == AlphaPolicy::Reject {
            return Err(Error::incompatible(
                "RGB tensor preparation needs an explicit alpha discard or composition policy",
            ));
        }
        HostImageView::new(image)?;
    }
    let shape = match spec.layout {
        TensorLayout::Nchw => vec![images.len(), 3, height, width],
        TensorLayout::Nhwc => vec![images.len(), height, width, 3],
    };
    let mut tensor = Tensor::zeroed(shape)?;
    let allocation = Arc::get_mut(&mut tensor.data)
        .ok_or_else(|| Error::incompatible("new tensor allocation is unexpectedly aliased"))?;
    let values = &mut allocation[tensor.offset..tensor.offset + elements];
    let mut color_operation_counts = BTreeMap::<String, usize>::new();
    for (batch, image) in images.iter().enumerate() {
        let view = HostImageView::new(image)?;
        let transform = if image.metadata().icc_profile.is_some() {
            let IccPolicy::ConvertToSrgb { gamut } = spec.icc else {
                unreachable!("ICC policy prevalidated")
            };
            Some(IccTransform::new(image, spec.icc_limits, gamut)?)
        } else {
            None
        };
        let mut image_clipped = false;
        for y in 0..height {
            for (x, bytes) in view.row(y)?.chunks_exact(bpp).enumerate() {
                let rgb = rgb_unit(bytes, format)?;
                let rgb = if let Some(transform) = &transform {
                    let (converted, clipped) = transform.apply(rgb)?;
                    image_clipped |= clipped;
                    converted
                } else {
                    rgb
                };
                for (channel, component) in rgb.iter().enumerate() {
                    let scaled = component
                        * if spec.range == NumericRange::Byte {
                            255.0
                        } else {
                            1.0
                        };
                    let value = (scaled - spec.mean[channel]) / spec.std[channel];
                    if !value.is_finite() {
                        return Err(Error::invalid(
                            "tensor input or normalized result is not finite",
                        ));
                    }
                    let index = batch * per_image
                        + match spec.layout {
                            TensorLayout::Nchw => channel * pixels + y * width + x,
                            TensorLayout::Nhwc => (y * width + x) * 3 + channel,
                        };
                    values[index] = value;
                }
            }
        }
        if transform.is_some() {
            *color_operation_counts.entry("ICC RGB converted to nonlinear sRGB with relative-colorimetric intent; 1e-4 linear-sRGB endpoint rounding tolerance".into()).or_default()+=1;
            *color_operation_counts.entry("ICC parser bounds applied; CMS private heap/transform tables are unmeasured and outside tensor output-byte counters".into()).or_default()+=1;
        }
        if image_clipped {
            *color_operation_counts
                .entry("out-of-sRGB-gamut components explicitly clipped".into())
                .or_default() += 1;
        }
    }
    tensor.layout = Some(spec.layout);
    tensor.assumptions = assumption_counts
        .into_iter()
        .map(|(assumption, count)| format!("{assumption}; {count}/{} images", images.len()))
        .collect();
    tensor.color_operations = color_operation_counts
        .into_iter()
        .map(|(operation, count)| format!("{operation}; {count}/{} images", images.len()))
        .collect();
    Ok(tensor)
}

fn shape_info(shape: &[usize]) -> Result<(usize, Vec<usize>)> {
    if shape.len() > 32 {
        return Err(Error::invalid("host tensors support at most 32 dimensions"));
    }
    let elements = shape
        .iter()
        .try_fold(1_usize, |n, dim| n.checked_mul(*dim))
        .ok_or_else(|| Error::overflow("tensor dimensions overflow"))?;
    let mut strides = vec![1; shape.len()];
    let mut stride = 1_usize;
    for (index, dim) in shape.iter().enumerate().rev() {
        strides[index] = stride;
        stride = stride
            .checked_mul(*dim)
            .ok_or_else(|| Error::overflow("tensor strides overflow"))?;
    }
    Ok((elements, strides))
}

fn rgb_unit(bytes: &[u8], format: PixelFormat) -> Result<[f32; 3]> {
    let gray = matches!(
        format,
        PixelFormat::Gray8
            | PixelFormat::GrayAlpha8
            | PixelFormat::Gray16
            | PixelFormat::GrayAlpha16
    );
    let component = |index: usize| -> Result<f32> {
        Ok(match format {
            PixelFormat::Gray8
            | PixelFormat::GrayAlpha8
            | PixelFormat::Rgb8
            | PixelFormat::Rgba8 => f32::from(bytes[index]) / 255.0,
            PixelFormat::Gray16
            | PixelFormat::GrayAlpha16
            | PixelFormat::Rgb16
            | PixelFormat::Rgba16 => {
                f32::from(u16::from_le_bytes([bytes[index * 2], bytes[index * 2 + 1]])) / 65535.0
            }
            PixelFormat::RgbF32 | PixelFormat::RgbaF32 => f32::from_le_bytes(
                bytes[index * 4..index * 4 + 4]
                    .try_into()
                    .map_err(|_| Error::invalid("short f32 component"))?,
            ),
            _ => {
                return Err(Error::incompatible(
                    "YUV needs an explicit color transform before RGB tensor packing",
                ));
            }
        })
    };
    if gray {
        let value = component(0)?;
        Ok([value; 3])
    } else {
        Ok([component(0)?, component(1)?, component(2)?])
    }
}
fn validate_color(image: &Image, policy: ColorAssumption) -> Result<Vec<String>> {
    if image.metadata().icc_profile.is_some() {
        return Err(Error::unsupported(
            "ICC-tagged input requires color-managed conversion before tensor preparation",
        ));
    }
    let color = &image.metadata().color;
    if color.chroma_siting.is_some() {
        return Err(Error::incompatible(
            "packed RGB tensor input cannot carry chroma-siting metadata",
        ));
    }
    let fields = [
        (
            "primaries",
            color.primaries.as_deref(),
            &["srgb", "bt709", "rec709", "bt.709"][..],
            "sRGB/BT.709",
        ),
        (
            "transfer",
            color.transfer.as_deref(),
            &["srgb", "iec61966-2-1", "iec 61966-2-1"][..],
            "sRGB",
        ),
        (
            "matrix",
            color.matrix.as_deref(),
            &["rgb", "identity"][..],
            "identity RGB",
        ),
        ("range", color.range.as_deref(), &["full"][..], "full range"),
    ];
    let mut assumptions = Vec::new();
    for (name, value, accepted, default) in fields {
        match value {
            Some(value) if accepted.iter().any(|s| value.eq_ignore_ascii_case(s)) => {}
            Some(value) => {
                return Err(Error::unsupported(format!(
                    "tensor preparation cannot reinterpret {name}={value}; request an explicit color conversion"
                )));
            }
            None if policy == ColorAssumption::AssumeSrgb => {
                assumptions.push(format!("assumed {name}: {default}"))
            }
            None => {
                return Err(Error::incompatible(format!(
                    "unknown {name}; supply established metadata or explicitly assume sRGB"
                )));
            }
        }
    }
    Ok(assumptions)
}
