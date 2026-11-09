//! Checked padding and explicit numeric RGB/RGBA component conversion.
//!
//! These operations do not transform primaries, transfer functions, or ICC color.
//! Integer components use normalized full-scale values; f32 values require an
//! explicit normalized unit/extended-range interpretation. Alpha is coverage in
//! `[0,1]`. Conversion requires straight input alpha; use `image_color::convert`
//! first to unassociate premultiplied inputs. Padding copies stored values and
//! therefore does not reinterpret their association. Visible pitched host rows
//! are supported; opaque surfaces and grayscale/profile mismatches fail.
use crate::{
    image::{MetadataLoss, metadata_payload_bytes},
    image_color::{AlphaMode, GamutPolicy},
};
use media_core::{
    error::{Error, Result},
    media::{Image, ImageMetadata, PixelFormat, PlaneLayout},
};

/// One stored-component pixel; its variant must exactly match the source format.
#[derive(Debug, Clone, Copy)]
pub enum PixelFill {
    /// Three eight-bit components.
    Rgb8([u8; 3]),
    /// Three eight-bit components and alpha.
    Rgba8([u8; 4]),
    /// Three sixteen-bit components.
    Rgb16([u16; 3]),
    /// Three sixteen-bit components and alpha.
    Rgba16([u16; 4]),
    /// Three finite f32 components.
    RgbF32([f32; 3]),
    /// Three finite f32 components and normalized alpha.
    RgbaF32([f32; 4]),
}
impl PixelFill {
    fn encoded(self) -> Result<(PixelFormat, [u8; 16])> {
        let mut bytes = [0; 16];
        let format = match self {
            Self::Rgb8(values) => {
                bytes[..3].copy_from_slice(&values);
                PixelFormat::Rgb8
            }
            Self::Rgba8(values) => {
                bytes[..4].copy_from_slice(&values);
                PixelFormat::Rgba8
            }
            Self::Rgb16(values) => {
                for (i, value) in values.into_iter().enumerate() {
                    bytes[i * 2..i * 2 + 2].copy_from_slice(&value.to_le_bytes());
                }
                PixelFormat::Rgb16
            }
            Self::Rgba16(values) => {
                for (i, value) in values.into_iter().enumerate() {
                    bytes[i * 2..i * 2 + 2].copy_from_slice(&value.to_le_bytes());
                }
                PixelFormat::Rgba16
            }
            Self::RgbF32(values) => {
                validate_float(&values)?;
                for (i, value) in values.into_iter().enumerate() {
                    bytes[i * 4..i * 4 + 4].copy_from_slice(&value.to_le_bytes());
                }
                PixelFormat::RgbF32
            }
            Self::RgbaF32(values) => {
                validate_float(&values)?;
                check_alpha(f64::from(values[3]))?;
                for (i, value) in values.into_iter().enumerate() {
                    bytes[i * 4..i * 4 + 4].copy_from_slice(&value.to_le_bytes());
                }
                PixelFormat::RgbaF32
            }
        };
        Ok((format, bytes))
    }
}
/// Border extents in stored pixel coordinates and exact typed fill values.
#[derive(Debug, Clone, Copy)]
pub struct PadSpec {
    /// Left border width.
    pub left: u32,
    /// Top border height.
    pub top: u32,
    /// Right border width.
    pub right: u32,
    /// Bottom border height.
    pub bottom: u32,
    /// Exact source-format stored values; no color interpretation is inferred.
    pub fill: PixelFill,
    /// Maximum new pixel and cloned metadata payload bytes.
    pub max_working_bytes: usize,
}
impl PadSpec {
    /// Specify left/top/right/bottom borders; default new-payload budget is 256 MiB.
    pub const fn new(left: u32, top: u32, right: u32, bottom: u32, fill: PixelFill) -> Self {
        Self {
            left,
            top,
            right,
            bottom,
            fill,
            max_working_bytes: 256 * 1024 * 1024,
        }
    }
    /// Apply one border width on all four sides.
    pub const fn uniform(width: u32, fill: PixelFill) -> Self {
        Self::new(width, width, width, width, fill)
    }
    fn changes_geometry(self) -> bool {
        self.left != 0 || self.right != 0 || self.top != 0 || self.bottom != 0
    }
}
/// Explicit change in RGB component/alpha channel count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlphaChange {
    /// Keep the same channel count; default conversion policy.
    Preserve,
    /// Add alpha with exact opaque coverage (MAX for integers,1.0 for float).
    AddOpaque,
    /// Remove alpha only if every source pixel is exactly opaque.
    DropIfOpaque,
    /// Discard declared straight alpha, preserving RGB without compositing.
    DiscardStraight,
}
/// Interpretation of normalized f32 RGB values; alpha always remains `[0,1]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatRange {
    /// Nominal RGB `[0,1]`; outliers require explicit GamutPolicy::Clip.
    Unit,
    /// Finite extended normalized RGB; float outputs retain negative/>1 values.
    /// Integer outputs still require explicit clipping or reject out-of-gamut.
    Extended,
}
/// Information-loss policy for conversion onto a smaller component grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quantization {
    /// Require reversible normalized values; no implicit precision loss.
    Exact,
    /// Round to nearest representable integer, ties to even; report actual loss.
    NearestTiesEven,
}
/// Explicit numeric conversion, independent of any color-space transformation.
#[derive(Debug, Clone, Copy)]
pub struct PixelConvertSpec {
    /// Requested RGB/RGBA packed component type and channel count.
    pub target: PixelFormat,
    /// Explicit channel-count policy.
    pub alpha: AlphaChange,
    /// RGBA sources must declare Straight; premultiplied input is rejected.
    pub input_alpha: Option<AlphaMode>,
    /// Required whenever input or output is f32; irrelevant integer-only settings reject.
    pub float_range: Option<FloatRange>,
    /// Explicit component-grid precision policy.
    pub quantization: Quantization,
    /// Explicit normalized RGB clipping/rejection policy.
    pub gamut: GamutPolicy,
    /// Maximum new pixel and cloned metadata payload bytes.
    pub max_working_bytes: usize,
}
impl PixelConvertSpec {
    /// Strict defaults: preserve alpha, exact precision, reject gamut loss.
    pub const fn new(target: PixelFormat) -> Self {
        Self {
            target,
            alpha: AlphaChange::Preserve,
            input_alpha: None,
            float_range: None,
            quantization: Quantization::Exact,
            gamut: GamutPolicy::Reject,
            max_working_bytes: 256 * 1024 * 1024,
        }
    }
}
/// Actual declared component changes, separate from discarded metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelChange {
    /// Components changed onto a coarser precision grid.
    PrecisionQuantized,
    /// Out-of-gamut RGB values were explicitly clamped to `[0,1]`.
    GamutClipped,
    /// An opaque alpha channel was added.
    AlphaAddedOpaque,
    /// An exactly opaque alpha channel was removed.
    AlphaDroppedOpaque,
    /// Straight alpha was discarded without compositing.
    StraightAlphaDiscarded,
}
/// Packed output and explicit metadata/component change reports.
#[derive(Debug, Clone)]
pub struct PixelTransformOutput {
    /// New tightly packed host pixels.
    pub image: Image,
    /// Stale opaque metadata removed by the operation.
    pub metadata_loss: Vec<MetadataLoss>,
    /// Actual component/alpha losses or additions explicitly requested.
    pub changes: Vec<PixelChange>,
}

/// Planned new payload for padding, without reading pixels or parsing ICC.
pub fn pad_working_bytes(image: &Image, spec: &PadSpec) -> Result<usize> {
    let format = Format::new(image.pixel_format())?;
    if spec.fill.encoded()?.0 != image.pixel_format() {
        return Err(Error::incompatible(
            "padding fill must exactly match source format/channel precision",
        ));
    }
    let (width, height) = pad_geometry(image, spec)?;
    with_metadata(
        pixel_bytes(width, height, format.bytes())?,
        image,
        !spec.changes_geometry(),
    )
}
/// Planned conservative new conversion payload, without reading pixels/parsing ICC.
pub fn convert_working_bytes(image: &Image, spec: &PixelConvertSpec) -> Result<usize> {
    let (_, target) = validate_conversion(image, spec)?;
    with_metadata(
        pixel_bytes(image.width(), image.height(), target.bytes())?,
        image,
        spec.target == image.pixel_format() && spec.alpha == AlphaChange::Preserve,
    )
}
/// Add borders using exact stored values and visible host row strides.
pub fn pad(image: &Image, spec: PadSpec) -> Result<PixelTransformOutput> {
    check_budget(pad_working_bytes(image, &spec)?, spec.max_working_bytes)?;
    validate_profile(image)?;
    let source = Rows::new(image)?;
    let format = Format::new(image.pixel_format())?;
    let (width, height) = pad_geometry(image, &spec)?;
    let (_, fill) = spec.fill.encoded()?;
    let mut output = allocate(pixel_bytes(width, height, format.bytes())?)?;
    for pixel in output.chunks_exact_mut(format.bytes()) {
        pixel.copy_from_slice(&fill[..format.bytes()]);
    }
    for y in 0..image.height() as usize {
        let row = source.row(y)?;
        validate_row(row, format)?;
        let start =
            ((y + spec.top as usize) * width as usize + spec.left as usize) * format.bytes();
        output[start..start + row.len()].copy_from_slice(row);
    }
    let (metadata, metadata_loss) = metadata(
        image,
        spec.changes_geometry(),
        MetadataLoss::ExifAfterGeometry,
    );
    Ok(PixelTransformOutput {
        image: Image::from_packed(width, height, image.pixel_format(), output)?
            .with_metadata(metadata),
        metadata_loss,
        changes: Vec::new(),
    })
}
/// Convert packed RGB/RGBA numeric components with declared precision/alpha rules.
///
/// Integer `[0,MAX]` scales to normalized components. Float components are already
/// normalized according to FloatRange; this never guesses a byte-like float scale.
/// Explicit limited-range RGB metadata is unsupported for precision changes.
pub fn convert(image: &Image, spec: PixelConvertSpec) -> Result<PixelTransformOutput> {
    let (source_format, target) = validate_conversion(image, &spec)?;
    check_budget(convert_working_bytes(image, &spec)?, spec.max_working_bytes)?;
    validate_profile(image)?;
    let rows = Rows::new(image)?;
    let mut output = allocate(pixel_bytes(image.width(), image.height(), target.bytes())?)?;
    let mut quantized = false;
    let mut clipped = false;
    for y in 0..image.height() as usize {
        let row = rows.row(y)?;
        for (x, pixel) in row.chunks_exact(source_format.bytes()).enumerate() {
            let mut values = [Sample::constant(1.); 4];
            for (channel, value) in values.iter_mut().enumerate().take(source_format.channels) {
                *value = source_format.read(pixel, channel)?;
            }
            if source_format.channels == 4 {
                check_alpha(values[3].value)?;
                if spec.alpha == AlphaChange::DropIfOpaque && values[3].value != 1. {
                    return Err(Error::incompatible(
                        "DropIfOpaque found nonopaque alpha; explicitly discard straight alpha or composite before conversion",
                    ));
                }
            }
            for (channel, sample) in values.iter_mut().enumerate().take(target.channels) {
                let needs_unit = channel == 3
                    || target.scalar != Scalar::F32
                    || spec.float_range == Some(FloatRange::Unit);
                if needs_unit && !(0. ..=1.).contains(&sample.value) {
                    if channel == 3 || spec.gamut == GamutPolicy::Reject {
                        return Err(Error::incompatible(
                            "component lies outside declared unit/integer output range",
                        ));
                    }
                    *sample = Sample::constant(sample.value.clamp(0., 1.));
                    clipped = true;
                }
                let offset = (y * image.width() as usize + x) * target.bytes()
                    + channel * target.scalar.bytes();
                quantized |= target.write(
                    *sample,
                    &mut output[offset..offset + target.scalar.bytes()],
                    spec.quantization,
                )?;
            }
        }
    }
    let mut changes = Vec::new();
    if quantized {
        changes.push(PixelChange::PrecisionQuantized);
    }
    if clipped {
        changes.push(PixelChange::GamutClipped);
    }
    match spec.alpha {
        AlphaChange::AddOpaque => changes.push(PixelChange::AlphaAddedOpaque),
        AlphaChange::DropIfOpaque => changes.push(PixelChange::AlphaDroppedOpaque),
        AlphaChange::DiscardStraight => changes.push(PixelChange::StraightAlphaDiscarded),
        AlphaChange::Preserve => {}
    }
    let changed = source_format != target || quantized || clipped;
    let (metadata, metadata_loss) =
        metadata(image, changed, MetadataLoss::ExifAfterComponentTransform);
    Ok(PixelTransformOutput {
        image: Image::from_packed(image.width(), image.height(), spec.target, output)?
            .with_metadata(metadata),
        metadata_loss,
        changes,
    })
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Scalar {
    U8,
    U16,
    F32,
}
impl Scalar {
    fn bytes(self) -> usize {
        match self {
            Self::U8 => 1,
            Self::U16 => 2,
            Self::F32 => 4,
        }
    }
    fn maximum(self) -> u32 {
        match self {
            Self::U8 => 255,
            Self::U16 => 65535,
            Self::F32 => 1,
        }
    }
}
#[derive(Clone, Copy, PartialEq, Eq)]
struct Format {
    channels: usize,
    scalar: Scalar,
}
impl Format {
    fn new(format: PixelFormat) -> Result<Self> {
        let (channels, scalar) = match format {
            PixelFormat::Rgb8 => (3, Scalar::U8),
            PixelFormat::Rgba8 => (4, Scalar::U8),
            PixelFormat::Rgb16 => (3, Scalar::U16),
            PixelFormat::Rgba16 => (4, Scalar::U16),
            PixelFormat::RgbF32 => (3, Scalar::F32),
            PixelFormat::RgbaF32 => (4, Scalar::F32),
            _ => {
                return Err(Error::unsupported(
                    "pixel operations currently require packed RGB/RGBA8/16/f32; Gray/YUV need an explicit supported conversion",
                ));
            }
        };
        Ok(Self { channels, scalar })
    }
    fn bytes(self) -> usize {
        self.channels * self.scalar.bytes()
    }
    fn read(self, pixel: &[u8], channel: usize) -> Result<Sample> {
        let start = channel * self.scalar.bytes();
        let value = &pixel[start..start + self.scalar.bytes()];
        let sample = match self.scalar {
            Scalar::U8 => Sample::integer(u32::from(value[0]), 255),
            Scalar::U16 => Sample::integer(
                u32::from(u16::from_le_bytes(value.try_into().expect("two bytes"))),
                65535,
            ),
            Scalar::F32 => {
                let value = f32::from_le_bytes(value.try_into().expect("four bytes"));
                if !value.is_finite() {
                    return Err(Error::invalid("float pixel component must be finite"));
                }
                Sample {
                    value: f64::from(value),
                    integer: None,
                    float: Some(value),
                }
            }
        };
        Ok(sample)
    }
    fn write(self, sample: Sample, output: &mut [u8], policy: Quantization) -> Result<bool> {
        if self.scalar == Scalar::F32 {
            let value = sample.value as f32;
            if !value.is_finite() {
                return Err(Error::overflow(
                    "float component conversion exceeds finite f32",
                ));
            }
            let reversible = sample.integer.is_none_or(|(code, max)| {
                ((f64::from(value) * f64::from(max)).round_ties_even() as u32) == code
            });
            if !reversible && policy == Quantization::Exact {
                return Err(Error::incompatible(
                    "float conversion is not reversible under Exact quantization",
                ));
            }
            output.copy_from_slice(&value.to_le_bytes());
            return Ok(!reversible);
        }
        let max = self.scalar.maximum();
        let (code, exact) = if let Some((code, source_max)) = sample.integer {
            let numerator = u64::from(code) * u64::from(max);
            let denominator = u64::from(source_max);
            let quotient = numerator / denominator;
            let remainder = numerator % denominator;
            let rounded = quotient
                + u64::from(
                    remainder * 2 > denominator
                        || (remainder * 2 == denominator && !quotient.is_multiple_of(2)),
                );
            (rounded as u32, remainder == 0)
        } else {
            let rounded = (sample.value * f64::from(max)).round_ties_even() as u32;
            let exact = sample.float.map_or(
                sample.value == f64::from(rounded) / f64::from(max),
                |value| value == rounded as f32 / max as f32,
            );
            (rounded, exact)
        };
        if !exact && policy == Quantization::Exact {
            return Err(Error::incompatible(
                "component precision would be lost; explicitly select NearestTiesEven quantization",
            ));
        }
        match self.scalar {
            Scalar::U8 => output[0] = code as u8,
            Scalar::U16 => output.copy_from_slice(&(code as u16).to_le_bytes()),
            Scalar::F32 => unreachable!(),
        };
        Ok(!exact)
    }
}
#[derive(Clone, Copy)]
struct Sample {
    value: f64,
    integer: Option<(u32, u32)>,
    float: Option<f32>,
}
impl Sample {
    fn integer(code: u32, max: u32) -> Self {
        Self {
            value: f64::from(code) / f64::from(max),
            integer: Some((code, max)),
            float: None,
        }
    }
    fn constant(value: f64) -> Self {
        Self {
            value,
            integer: None,
            float: None,
        }
    }
}
struct Rows<'a> {
    bytes: &'a [u8],
    plane: &'a PlaneLayout,
    row_bytes: usize,
}
impl<'a> Rows<'a> {
    fn new(image: &'a Image) -> Result<Self> {
        let format = Format::new(image.pixel_format())?;
        if image.layout().planes.len() != 1 {
            return Err(Error::unsupported(
                "packed host operation needs one addressable plane",
            ));
        }
        let bytes = image.host_bytes()?;
        let plane = &image.layout().planes[0];
        let row_bytes = (image.width() as usize)
            .checked_mul(format.bytes())
            .ok_or_else(|| Error::overflow("visible image row size overflow"))?;
        if plane.rows < image.height() as usize
            || plane.row_bytes < row_bytes
            || plane.row_stride < row_bytes
            || plane.extent()? > bytes.len()
        {
            return Err(Error::invalid("image plane cannot cover visible rows"));
        }
        Ok(Self {
            bytes,
            plane,
            row_bytes,
        })
    }
    fn row(&self, y: usize) -> Result<&[u8]> {
        let start = y
            .checked_mul(self.plane.row_stride)
            .and_then(|v| v.checked_add(self.plane.offset))
            .ok_or_else(|| Error::overflow("visible image row address overflow"))?;
        self.bytes
            .get(
                start
                    ..start
                        .checked_add(self.row_bytes)
                        .ok_or_else(|| Error::overflow("visible row extent overflow"))?,
            )
            .ok_or_else(|| Error::invalid("visible image row exceeds storage"))
    }
}
fn validate_conversion(image: &Image, spec: &PixelConvertSpec) -> Result<(Format, Format)> {
    let (source, target) = (
        Format::new(image.pixel_format())?,
        Format::new(spec.target)?,
    );
    match (source.channels, target.channels, spec.alpha) {
        (3, 3, AlphaChange::Preserve)
        | (4, 4, AlphaChange::Preserve)
        | (3, 4, AlphaChange::AddOpaque)
        | (4, 3, AlphaChange::DropIfOpaque | AlphaChange::DiscardStraight) => {}
        _ => {
            return Err(Error::invalid(
                "alpha change policy does not match source/target channel counts",
            ));
        }
    }
    if source.channels == 4 && spec.input_alpha != Some(AlphaMode::Straight) {
        return Err(Error::unsupported(
            "pixel conversion requires explicitly declared straight RGBA; unassociate with convert_color first",
        ));
    }
    if source.channels == 3 && spec.input_alpha.is_some() {
        return Err(Error::invalid(
            "RGB input has no alpha association to declare",
        ));
    }
    let floats = source.scalar == Scalar::F32 || target.scalar == Scalar::F32;
    if floats != spec.float_range.is_some() {
        return Err(Error::invalid(
            "FloatRange must be supplied exactly when source or target uses f32",
        ));
    }
    if source.scalar != target.scalar
        && image
            .metadata()
            .color
            .range
            .as_deref()
            .is_some_and(|v| !matches!(v, "full" | "pc"))
    {
        return Err(Error::unsupported(
            "precision conversion of explicitly limited-range RGB requires a range-specific policy",
        ));
    }
    Ok((source, target))
}
fn validate_profile(image: &Image) -> Result<()> {
    if let Some(profile) = &image.metadata().icc_profile {
        let profile = moxcms::ColorProfile::new_from_slice(profile)
            .map_err(|error| Error::backend("moxcms", error.to_string()).with_cause(error))?;
        if profile.color_space != moxcms::DataColorSpace::Rgb {
            return Err(Error::incompatible(
                "RGB/RGBA pixels require an RGB ICC device profile; channel mismatch cannot be guessed",
            ));
        }
    }
    Ok(())
}
fn validate_float(values: &[f32]) -> Result<()> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(Error::invalid("float fill components must be finite"));
    }
    Ok(())
}
fn validate_row(row: &[u8], format: Format) -> Result<()> {
    if format.scalar == Scalar::F32 {
        for pixel in row.chunks_exact(format.bytes()) {
            for channel in 0..format.channels {
                let sample = format.read(pixel, channel)?;
                if channel == 3 {
                    check_alpha(sample.value)?;
                }
            }
        }
    }
    Ok(())
}
fn check_alpha(alpha: f64) -> Result<()> {
    if !(0. ..=1.).contains(&alpha) {
        return Err(Error::invalid(
            "alpha coverage must be finite and between zero and one",
        ));
    }
    Ok(())
}
fn pad_geometry(image: &Image, spec: &PadSpec) -> Result<(u32, u32)> {
    Ok((
        image
            .width()
            .checked_add(spec.left)
            .and_then(|v| v.checked_add(spec.right))
            .ok_or_else(|| Error::overflow("padding width exceeds u32"))?,
        image
            .height()
            .checked_add(spec.top)
            .and_then(|v| v.checked_add(spec.bottom))
            .ok_or_else(|| Error::overflow("padding height exceeds u32"))?,
    ))
}
fn pixel_bytes(width: u32, height: u32, bpp: usize) -> Result<usize> {
    (width as usize)
        .checked_mul(height as usize)
        .and_then(|v| v.checked_mul(bpp))
        .ok_or_else(|| Error::overflow("pixel output allocation exceeds address space"))
}
fn with_metadata(bytes: usize, image: &Image, preserve_exif: bool) -> Result<usize> {
    let metadata = metadata_payload_bytes(image.metadata())?;
    let metadata = if preserve_exif {
        metadata
    } else {
        metadata
            .checked_sub(image.metadata().exif.as_ref().map_or(0, Vec::len))
            .ok_or_else(|| Error::overflow("metadata accounting underflow"))?
    };
    bytes
        .checked_add(metadata)
        .ok_or_else(|| Error::overflow("pixel operation payload size overflow"))
}
fn metadata(
    image: &Image,
    changed: bool,
    loss: MetadataLoss,
) -> (ImageMetadata, Vec<MetadataLoss>) {
    let source = image.metadata();
    (
        ImageMetadata {
            orientation: source.orientation,
            icc_profile: source.icc_profile.clone(),
            exif: if changed { None } else { source.exif.clone() },
            color: source.color.clone(),
        },
        if changed && source.exif.is_some() {
            vec![loss]
        } else {
            Vec::new()
        },
    )
}
fn check_budget(bytes: usize, limit: usize) -> Result<()> {
    if bytes > limit {
        return Err(
            Error::exhausted("pixel operation exceeds new-payload working-byte budget")
                .with_constraint("max_working_bytes"),
        );
    }
    Ok(())
}
fn allocate(bytes: usize) -> Result<Vec<u8>> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(bytes)
        .map_err(|_| Error::exhausted("cannot allocate pixel operation output"))?;
    output.resize(bytes, 0);
    Ok(output)
}
