//! Explicit ICC-managed color conversion and alpha-correct image resizing.
//!
//! RGB/RGBA eight-bit, sixteen-bit, and normalized f32 representations retain
//! their component precision. Filtering uses premultiplied alpha, with source
//! association explicitly supplied. Linear-light operations use extended f32
//! linear sRGB as a working space, preserving out-of-gamut values until output.
//! This is an SDR color transform, not an HDR tone mapper.

use crate::image::{Filter, Fit, MetadataLoss};
use media_core::{
    error::{Error, Result},
    media::{ColorDescription, Image, PixelFormat},
};
use moxcms::{ColorProfile, DataColorSpace, Layout, ToneReprCurve, TransformOptions};

/// Named RGB interpretation used as a target or an explicit missing-data assumption.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorSpace {
    /// IEC sRGB, D65 BT.709 primaries.
    Srgb,
    /// Scene-linear sRGB primaries and D65 white.
    LinearSrgb,
    /// Display P3, D65 white, sRGB transfer.
    DisplayP3,
}
impl ColorSpace {
    fn profile(self) -> ColorProfile {
        match self {
            Self::Srgb => ColorProfile::new_srgb(),
            Self::DisplayP3 => ColorProfile::new_display_p3(),
            Self::LinearSrgb => {
                let mut profile = ColorProfile::new_srgb();
                profile.red_trc = Some(ToneReprCurve::Parametric(vec![1.]));
                profile.green_trc = profile.red_trc.clone();
                profile.blue_trc = profile.red_trc.clone();
                profile.cicp = None;
                profile.description = Some(moxcms::ProfileText::PlainString("Linear sRGB".into()));
                profile
            }
        }
    }
    fn description(self) -> ColorDescription {
        ColorDescription {
            primaries: Some(
                match self {
                    Self::DisplayP3 => "display-p3",
                    _ => "bt709",
                }
                .into(),
            ),
            transfer: Some(
                if self == Self::LinearSrgb {
                    "linear"
                } else {
                    "srgb"
                }
                .into(),
            ),
            matrix: Some("rgb".into()),
            range: Some("full".into()),
            chroma_siting: None,
        }
    }
}
/// Domain in which reconstruction kernels operate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResizeDomain {
    /// Encoded components of the selected destination color space.
    StoredComponents,
    /// Linear-light working RGB after applying the source ICC/transfer function.
    LinearLight,
}
/// Alpha association in the stored component domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlphaMode {
    /// RGB components are independent of alpha.
    Straight,
    /// Stored RGB components have been multiplied by alpha.
    Premultiplied,
}
/// Treatment of transformed RGB outside the integer output gamut.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GamutPolicy {
    /// Reject out-of-range components, preserving a strict no-clipping contract.
    Reject,
    /// Explicitly clamp RGB to `[0,1]` for integer output; report an assumption.
    Clip,
}
/// Color conversion controls shared with resize.
#[derive(Debug, Clone, Copy)]
pub struct ColorConvertSpec {
    /// Destination interpretation; None retains the established source profile.
    pub target: Option<ColorSpace>,
    /// Fill absent source interpretation only; never override established tags.
    pub missing_color: Option<ColorSpace>,
    /// Required for RGBA input; packed image storage does not establish association.
    pub input_alpha: Option<AlphaMode>,
    /// Requested output alpha association.
    pub output_alpha: AlphaMode,
    /// Explicit integer output gamut policy.
    pub gamut: GamutPolicy,
    /// Hard upper bound on this operation's planned image working allocations.
    pub max_working_bytes: usize,
}
impl Default for ColorConvertSpec {
    fn default() -> Self {
        Self {
            target: None,
            missing_color: None,
            input_alpha: None,
            output_alpha: AlphaMode::Straight,
            gamut: GamutPolicy::Reject,
            max_working_bytes: 256 * 1024 * 1024,
        }
    }
}
/// Geometry, filtering, and explicit color/alpha contract.
#[derive(Debug, Clone, Copy)]
pub struct ResizeSpec {
    /// Aspect-preserving inside or cover geometry.
    pub fit: Fit,
    /// Reconstruction kernel.
    pub filter: Filter,
    /// Stored components or linear-light reconstruction.
    pub domain: ResizeDomain,
    /// Color interpretation and working-memory controls.
    pub color: ColorConvertSpec,
}
/// Color transform result; assumptions are separate from discarded metadata.
#[derive(Debug, Clone)]
pub struct ColorTransformOutput {
    /// Transformed pixels with an updated output profile and color description.
    pub image: Image,
    /// Opaque metadata that could not be propagated faithfully.
    pub metadata_loss: Vec<MetadataLoss>,
    /// Missing source interpretation, alpha association, or clipping choices used.
    pub assumptions: Vec<String>,
}

/// Convert established ICC/RGB color without changing dimensions or precision.
pub fn convert(image: &Image, spec: ColorConvertSpec) -> Result<ColorTransformOutput> {
    process(image, None, spec)
}
/// Resize with explicit color domain and premultiplied-alpha filtering.
pub fn resize(image: &Image, spec: ResizeSpec) -> Result<ColorTransformOutput> {
    process(image, Some(spec), spec.color)
}

/// Planned conservative image-working allocation without parsing profiles.
///
/// None estimates color conversion. This is the same pixel-buffer estimate
/// enforced by `max_working_bytes`; native CMS profile/LUT allocations have
/// separate parser bounds and are not included as image buffers.
pub fn working_bytes(image: &Image, resize: Option<ResizeSpec>) -> Result<usize> {
    let (width, height, rw, rh) = match resize {
        None => (image.width(), image.height(), image.width(), image.height()),
        Some(spec) => geometry(image.width(), image.height(), spec.fit)?,
    };
    planned_bytes(image.width(), image.height(), rw, rh, width, height)
}

/// Exact retained ICC/EXIF/color-description payload planned for this operation.
///
/// A named target generates its actual encoded RGB profile. With no target,
/// an existing profile is retained; an absent profile is generated from the
/// validated source interpretation or explicit missing-color assumption.
/// Geometry, a named target, or alpha reassociation discards stale EXIF.
/// This does not estimate CMS
/// parser/LUT private heap allocations or pixel working buffers.
pub fn metadata_bytes(image: &Image, spec: &ColorConvertSpec, resize: bool) -> Result<usize> {
    let (profile, color) = if let Some(target) = spec.target {
        (
            target.profile().encode().map_err(cms_error)?.len(),
            target.description(),
        )
    } else if let Some(profile) = &image.metadata().icc_profile {
        let mut color = image.metadata().color.clone();
        color.matrix = Some("rgb".into());
        color.range = Some("full".into());
        (profile.len(), color)
    } else {
        let (source, color) = resolve_source(image, spec.missing_color, &mut Vec::new())?;
        (source.encode().map_err(cms_error)?.len(), color)
    };
    let exif = if resize || spec.target.is_some() || alpha_changes(image, spec) {
        0
    } else {
        image.metadata().exif.as_ref().map_or(0, Vec::len)
    };
    let color = crate::image::metadata_payload_bytes(&media_core::media::ImageMetadata {
        color,
        ..Default::default()
    })?;
    profile
        .checked_add(exif)
        .and_then(|bytes| bytes.checked_add(color))
        .ok_or_else(|| Error::overflow("color output metadata length"))
}

fn process(
    image: &Image,
    resize: Option<ResizeSpec>,
    spec: ColorConvertSpec,
) -> Result<ColorTransformOutput> {
    let channels = match image.pixel_format() {
        PixelFormat::Rgb8 | PixelFormat::Rgb16 | PixelFormat::RgbF32 => 3,
        PixelFormat::Rgba8 | PixelFormat::Rgba16 | PixelFormat::RgbaF32 => 4,
        _ => {
            return Err(Error::unsupported(
                "color management currently requires packed RGB/RGBA; grayscale/YUV need explicit conversion",
            ));
        }
    };
    let input_alpha = if channels == 4 {
        spec.input_alpha.ok_or_else(|| {
            Error::invalid("RGBA color processing needs an explicit input alpha association")
        })?
    } else {
        AlphaMode::Straight
    };
    let mut assumptions = Vec::new();
    if channels == 4 {
        assumptions.push(format!(
            "input alpha association explicitly interpreted as {input_alpha:?}"
        ));
    }
    let (source, source_description) = resolve_source(image, spec.missing_color, &mut assumptions)?;
    let target = spec
        .target
        .map(ColorSpace::profile)
        .unwrap_or_else(|| source.clone());
    let target_description = spec
        .target
        .map(ColorSpace::description)
        .unwrap_or(source_description);
    let (width, height, scaled_width, scaled_height) = match resize {
        None => (image.width(), image.height(), image.width(), image.height()),
        Some(spec) => geometry(image.width(), image.height(), spec.fit)?,
    };
    check_working(
        image.width(),
        image.height(),
        scaled_width,
        scaled_height,
        width,
        height,
        spec.max_working_bytes,
    )?;
    let mut pixels = unpack(image, channels)?;
    if input_alpha == AlphaMode::Premultiplied {
        unassociate(&mut pixels);
    }
    let linear = resize.is_some_and(|v| v.domain == ResizeDomain::LinearLight);
    let working = if linear {
        ColorSpace::LinearSrgb.profile()
    } else {
        target.clone()
    };
    if linear || spec.target.is_some() {
        pixels = transform(&source, &working, &pixels)?;
    }
    if let Some(resize) = resize {
        associate(&mut pixels);
        pixels = resample(
            &pixels,
            image.width(),
            image.height(),
            scaled_width,
            scaled_height,
            resize.filter,
        )?;
        unassociate(&mut pixels);
        if scaled_width != width || scaled_height != height {
            let x = (scaled_width - width) / 2;
            let y = (scaled_height - height) / 2;
            let mut cropped = Vec::with_capacity(width as usize * height as usize * 4);
            for row in y..y + height {
                let start = (row as usize * scaled_width as usize + x as usize) * 4;
                cropped.extend_from_slice(&pixels[start..start + width as usize * 4]);
            }
            pixels = cropped;
        }
    }
    if linear {
        pixels = transform(&working, &target, &pixels)?;
    }
    if spec.output_alpha == AlphaMode::Premultiplied {
        associate(&mut pixels);
    }
    let mut metadata = image.metadata().clone();
    let mut metadata_loss = Vec::new();
    if (resize.is_some() || spec.target.is_some() || alpha_changes(image, &spec))
        && metadata.exif.take().is_some()
    {
        metadata_loss.push(if resize.is_some() {
            MetadataLoss::ExifAfterGeometry
        } else {
            MetadataLoss::ExifAfterColorTransform
        });
    }
    // A transformed profile replaces rather than contradicts old opaque ICC.
    if spec.target.is_some() || metadata.icc_profile.is_none() {
        metadata.icc_profile = Some(target.encode().map_err(cms_error)?);
    }
    metadata.color = target_description;
    let output = pack(
        image,
        width,
        height,
        channels,
        &pixels,
        spec.gamut,
        &mut assumptions,
    )?
    .with_metadata(metadata);
    Ok(ColorTransformOutput {
        image: output,
        metadata_loss,
        assumptions,
    })
}
fn alpha_changes(image: &Image, spec: &ColorConvertSpec) -> bool {
    image.pixel_format().channels() == 4
        && spec
            .input_alpha
            .is_some_and(|alpha| alpha != spec.output_alpha)
}

fn resolve_source(
    image: &Image,
    missing: Option<ColorSpace>,
    assumptions: &mut Vec<String>,
) -> Result<(ColorProfile, ColorDescription)> {
    let mut description = image.metadata().color.clone();
    if description
        .matrix
        .as_deref()
        .is_some_and(|v| !matches!(v, "rgb" | "identity" | "0"))
        || description
            .range
            .as_deref()
            .is_some_and(|v| !matches!(v, "full" | "pc"))
        || description.chroma_siting.is_some()
    {
        return Err(Error::unsupported(
            "packed RGB color management requires full-range identity-matrix metadata without chroma siting",
        ));
    }
    if let Some(bytes) = &image.metadata().icc_profile {
        let profile = ColorProfile::new_from_slice(bytes).map_err(cms_error)?;
        if profile.color_space != DataColorSpace::Rgb {
            return Err(Error::incompatible(
                "ICC profile is not an RGB device profile",
            ));
        }
        if profile.cicp.is_some_and(|v| {
            matches!(
                v.transfer_characteristics,
                moxcms::TransferCharacteristics::Smpte2084 | moxcms::TransferCharacteristics::Hlg
            )
        }) {
            return Err(Error::unsupported(
                "HDR ICC processing requires an explicit luminance/tone-mapping contract",
            ));
        }
        if description.primaries.is_some() || description.transfer.is_some() {
            let expected = description_profile(&description)?;
            if !profiles_agree(&profile, &expected)? {
                return Err(Error::incompatible(
                    "ICC profile contradicts explicit primaries/transfer metadata",
                ));
            }
        }
        description.matrix = Some("rgb".into());
        description.range = Some("full".into());
        return Ok((profile, description));
    }
    if description.primaries.is_none()
        || description.transfer.is_none()
        || description.range.is_none()
    {
        let assumed = missing.ok_or_else(|| {
            Error::invalid("missing source color metadata requires an explicit assumption")
        })?;
        let defaults = assumed.description();
        if description.primaries.is_none() {
            description.primaries = defaults.primaries;
        }
        if description.transfer.is_none() {
            description.transfer = defaults.transfer;
        }
        if description.range.is_none() {
            description.range = defaults.range;
        }
        assumptions.push(format!("missing source color fields assumed {assumed:?}"));
    }
    description.matrix = Some("rgb".into());
    let profile = description_profile(&description)?;
    Ok((profile, description))
}
fn description_profile(description: &ColorDescription) -> Result<ColorProfile> {
    use moxcms::{
        CicpColorPrimaries as Primaries, MatrixCoefficients, TransferCharacteristics as Transfer,
    };
    let primaries = match description.primaries.as_deref() {
        Some("bt709" | "srgb") => Primaries::Bt709,
        Some("display-p3" | "smpte432" | "smpte432-1") => Primaries::Smpte432,
        Some("bt2020") => Primaries::Bt2020,
        _ => {
            return Err(Error::unsupported(
                "source RGB primaries need a supported named interpretation or an ICC profile",
            ));
        }
    };
    let transfer = match description.transfer.as_deref() {
        Some("srgb" | "iec61966-2-1" | "iec61966_2_1") => Transfer::Srgb,
        Some("linear") => Transfer::Linear,
        Some("bt709") => Transfer::Bt709,
        Some(v) if v.starts_with("png-gamma:") => {
            let gamma = v[10..]
                .parse::<u32>()
                .map_err(|_| Error::invalid("invalid PNG gamma description"))?;
            if gamma == 0 {
                return Err(Error::invalid("gamma must be positive"));
            }
            let mut profile = ColorProfile::new_from_cicp(moxcms::CicpProfile {
                color_primaries: primaries,
                transfer_characteristics: Transfer::Linear,
                matrix_coefficients: MatrixCoefficients::Identity,
                full_range: true,
            });
            let curve = Some(ToneReprCurve::Parametric(vec![100_000. / gamma as f32]));
            profile.red_trc = curve.clone();
            profile.green_trc = curve.clone();
            profile.blue_trc = curve;
            profile.cicp = None;
            return Ok(profile);
        }
        _ => {
            return Err(Error::unsupported(
                "source transfer needs supported SDR metadata or ICC; HDR tone mapping is not implicit",
            ));
        }
    };
    Ok(ColorProfile::new_from_cicp(moxcms::CicpProfile {
        color_primaries: primaries,
        transfer_characteristics: transfer,
        matrix_coefficients: MatrixCoefficients::Identity,
        full_range: true,
    }))
}
fn profiles_agree(profile: &ColorProfile, expected: &ColorProfile) -> Result<bool> {
    if profile.lut_a_to_b_perceptual.is_some()
        || profile.lut_a_to_b_colorimetric.is_some()
        || profile.lut_a_to_b_saturation.is_some()
        || profile.lut_b_to_a_perceptual.is_some()
        || profile.lut_b_to_a_colorimetric.is_some()
        || profile.lut_b_to_a_saturation.is_some()
    {
        return Err(Error::unsupported(
            "redundant named color tags cannot be validated against an arbitrary ICC LUT profile; provide one authoritative interpretation",
        ));
    }
    // Check matrix/TRC chromatic endpoints and neutral samples through the
    // actual ICC transform; profile descriptions/CICP labels alone are not used.
    let values = [
        0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 1., 1., 1., 0.02, 0.02, 0.02, 0.18, 0.18,
        0.18, 0.5, 0.5, 0.5,
    ];
    // Compare colorimetric agreement in one linear reference space. Comparing
    // encoded sRGB channels at zero amplifies ordinary fixed-point ICC matrix
    // rounding through the transfer curve, falsely rejecting our own profiles.
    let mut linear = expected.clone();
    linear.red_trc = Some(ToneReprCurve::Parametric(vec![1.]));
    linear.green_trc = linear.red_trc.clone();
    linear.blue_trc = linear.red_trc.clone();
    linear.cicp = None;
    let mut converted = [0.; 24];
    let mut reference = [0.; 24];
    profile
        .create_transform_f32(Layout::Rgb, &linear, Layout::Rgb, transform_options())
        .map_err(cms_error)?
        .transform(&values, &mut converted)
        .map_err(cms_error)?;
    expected
        .create_transform_f32(Layout::Rgb, &linear, Layout::Rgb, transform_options())
        .map_err(cms_error)?
        .transform(&values, &mut reference)
        .map_err(cms_error)?;
    Ok(reference
        .iter()
        .zip(converted)
        .all(|(a, b)| (*a - b).abs() < 0.001))
}
fn transform_options() -> TransformOptions {
    TransformOptions {
        prefer_fixed_point: false,
        allow_extended_range_rgb_xyz: true,
        ..TransformOptions::default()
    }
}
fn transform(source: &ColorProfile, target: &ColorProfile, pixels: &[f32]) -> Result<Vec<f32>> {
    let mut output = vec![0.; pixels.len()];
    source
        .create_transform_f32(Layout::Rgba, target, Layout::Rgba, transform_options())
        .map_err(cms_error)?
        .transform(pixels, &mut output)
        .map_err(cms_error)?;
    if output.iter().any(|v| !v.is_finite()) {
        return Err(Error::backend(
            "moxcms",
            "color transform returned nonfinite components",
        ));
    }
    // Alpha is coverage, never a tone-curve input. Preserve it independently.
    for (src, dst) in pixels.chunks_exact(4).zip(output.chunks_exact_mut(4)) {
        dst[3] = src[3];
    }
    Ok(output)
}
fn unpack(image: &Image, channels: usize) -> Result<Vec<f32>> {
    let bytes = image.packed_bytes()?;
    let components = match image.pixel_format() {
        PixelFormat::Rgb8 | PixelFormat::Rgba8 => bytes
            .iter()
            .map(|v| f32::from(*v) / 255.)
            .collect::<Vec<_>>(),
        PixelFormat::Rgb16 | PixelFormat::Rgba16 => bytes
            .chunks_exact(2)
            .map(|v| f32::from(u16::from_le_bytes([v[0], v[1]])) / 65535.)
            .collect(),
        _ => bytes
            .chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().expect("four bytes")))
            .collect(),
    };
    let mut output = Vec::with_capacity(image.width() as usize * image.height() as usize * 4);
    for pixel in components.chunks_exact(channels) {
        if pixel.iter().any(|v| !v.is_finite()) {
            return Err(Error::invalid("input image has nonfinite color components"));
        }
        let alpha = if channels == 4 { pixel[3] } else { 1. };
        if !(0. ..=1.).contains(&alpha) {
            return Err(Error::invalid(
                "alpha coverage must be between zero and one",
            ));
        }
        output.extend_from_slice(&[pixel[0], pixel[1], pixel[2], alpha]);
    }
    Ok(output)
}
fn pack(
    source: &Image,
    width: u32,
    height: u32,
    channels: usize,
    pixels: &[f32],
    gamut: GamutPolicy,
    assumptions: &mut Vec<String>,
) -> Result<Image> {
    let integer = !matches!(
        source.pixel_format(),
        PixelFormat::RgbF32 | PixelFormat::RgbaF32
    );
    let mut clipped = false;
    let mut bytes = Vec::new();
    for pixel in pixels.chunks_exact(4) {
        for (index, value) in pixel[..channels].iter().enumerate() {
            let mut value = if index == 3 {
                value.clamp(0., 1.)
            } else {
                *value
            };
            if integer && !(-0.0001..=1.0001).contains(&value) {
                if gamut == GamutPolicy::Reject {
                    return Err(Error::incompatible(
                        "color conversion exceeds integer output gamut; select Clip explicitly or retain floating-point output",
                    ));
                }
                clipped = true;
            }
            if integer {
                value = value.clamp(0., 1.);
            }
            match source.pixel_format() {
                PixelFormat::Rgb8 | PixelFormat::Rgba8 => {
                    bytes.push((value * 255.).round_ties_even() as u8)
                }
                PixelFormat::Rgb16 | PixelFormat::Rgba16 => bytes
                    .extend_from_slice(&((value * 65535.).round_ties_even() as u16).to_le_bytes()),
                _ => bytes.extend_from_slice(&value.to_le_bytes()),
            }
        }
    }
    if clipped {
        assumptions.push("out-of-gamut integer RGB components explicitly clipped".into());
    }
    Image::from_packed(width, height, source.pixel_format(), bytes)
}
fn associate(pixels: &mut [f32]) {
    for p in pixels.chunks_exact_mut(4) {
        p[0] *= p[3];
        p[1] *= p[3];
        p[2] *= p[3];
    }
}
fn unassociate(pixels: &mut [f32]) {
    for p in pixels.chunks_exact_mut(4) {
        let alpha = p[3];
        if alpha > 1e-8 {
            p[0] /= alpha;
            p[1] /= alpha;
            p[2] /= alpha;
        } else {
            p[0] = 0.;
            p[1] = 0.;
            p[2] = 0.;
        }
        p[3] = alpha.clamp(0., 1.);
    }
}

fn geometry(sw: u32, sh: u32, fit: Fit) -> Result<(u32, u32, u32, u32)> {
    let (w, h) = match fit {
        Fit::Inside { width, height } | Fit::Cover { width, height } => (width, height),
    };
    if w == 0 || h == 0 {
        return Err(Error::invalid("resize dimensions must be nonzero"));
    }
    let wide = u64::from(w) * u64::from(sh);
    let tall = u64::from(h) * u64::from(sw);
    let (rw, rh) = match fit {
        Fit::Inside { .. } => {
            if wide <= tall {
                (u64::from(w), (wide + u64::from(sw) / 2) / u64::from(sw))
            } else {
                ((tall + u64::from(sh) / 2) / u64::from(sh), u64::from(h))
            }
        }
        Fit::Cover { .. } => {
            if wide >= tall {
                (u64::from(w), wide.div_ceil(u64::from(sw)))
            } else {
                (tall.div_ceil(u64::from(sh)), u64::from(h))
            }
        }
    };
    let (rw, rh) = (
        u32::try_from(rw.max(1)).map_err(|_| Error::overflow("resize width overflow"))?,
        u32::try_from(rh.max(1)).map_err(|_| Error::overflow("resize height overflow"))?,
    );
    Ok(match fit {
        Fit::Inside { .. } => (rw, rh, rw, rh),
        Fit::Cover { .. } => (w, h, rw, rh),
    })
}
fn check_working(sw: u32, sh: u32, rw: u32, rh: u32, w: u32, h: u32, limit: usize) -> Result<()> {
    let total = planned_bytes(sw, sh, rw, rh, w, h)?;
    if total > limit {
        return Err(
            Error::exhausted("color operation exceeds working-byte budget")
                .with_constraint("max_working_bytes"),
        );
    }
    Ok(())
}
fn planned_bytes(sw: u32, sh: u32, rw: u32, rh: u32, w: u32, h: u32) -> Result<usize> {
    let mut total = 0u64;
    for (width, height, count) in [(sw, sh, 4u64), (rw, sh, 1), (rw, rh, 2), (w, h, 1)] {
        total = total
            .checked_add(
                u64::from(width)
                    .checked_mul(u64::from(height))
                    .and_then(|v| v.checked_mul(16 * count))
                    .ok_or_else(|| Error::overflow("color working allocation overflow"))?,
            )
            .ok_or_else(|| Error::overflow("color working allocation overflow"))?;
    }
    usize::try_from(total)
        .map_err(|_| Error::overflow("color working buffers exceed address space"))
}
fn resample(
    source: &[f32],
    sw: u32,
    sh: u32,
    dw: u32,
    dh: u32,
    filter: Filter,
) -> Result<Vec<f32>> {
    if (sw, sh) == (dw, dh) {
        return Ok(source.to_vec());
    }
    let mut intermediate = vec![0.; dw as usize * sh as usize * 4];
    let mut output = vec![0.; dw as usize * dh as usize * 4];
    for x in 0..dw {
        let weights = weights(x, sw, dw, filter);
        for y in 0..sh {
            for (sx, weight) in &weights {
                for c in 0..4 {
                    intermediate[(y as usize * dw as usize + x as usize) * 4 + c] +=
                        source[(y as usize * sw as usize + *sx) * 4 + c] * *weight;
                }
            }
        }
    }
    for y in 0..dh {
        let weights = weights(y, sh, dh, filter);
        for x in 0..dw {
            for (sy, weight) in &weights {
                for c in 0..4 {
                    output[(y as usize * dw as usize + x as usize) * 4 + c] +=
                        intermediate[(*sy * dw as usize + x as usize) * 4 + c] * *weight;
                }
            }
        }
    }
    Ok(output)
}
fn weights(destination: u32, source: u32, target: u32, filter: Filter) -> Vec<(usize, f32)> {
    let scale = f64::from(source) / f64::from(target);
    let center = (f64::from(destination) + 0.5) * scale;
    if filter == Filter::Nearest {
        return vec![(center.floor().clamp(0., f64::from(source - 1)) as usize, 1.)];
    }
    let support = match filter {
        Filter::Triangle => 1.,
        Filter::CatmullRom => 2.,
        Filter::Gaussian | Filter::Lanczos3 => 3.,
        Filter::Nearest => 0.,
    };
    let spread = scale.max(1.);
    let start = (center - support * spread).floor().max(0.) as u32;
    let end = (center + support * spread).ceil().min(f64::from(source)) as u32;
    let mut output = Vec::with_capacity((end - start) as usize);
    let mut sum = 0.;
    for sample in start..end {
        let x = ((f64::from(sample) + 0.5 - center) / spread).abs();
        let weight = match filter {
            Filter::Triangle => (1. - x).max(0.),
            Filter::CatmullRom => {
                if x < 1. {
                    1.5 * x * x * x - 2.5 * x * x + 1.
                } else if x < 2. {
                    -0.5 * x * x * x + 2.5 * x * x - 4. * x + 2.
                } else {
                    0.
                }
            }
            Filter::Gaussian => (-2. * x * x).exp(),
            Filter::Lanczos3 => {
                if x == 0. {
                    1.
                } else if x < 3. {
                    let pi = std::f64::consts::PI;
                    (pi * x).sin() / (pi * x) * (pi * x / 3.).sin() / (pi * x / 3.)
                } else {
                    0.
                }
            }
            Filter::Nearest => 1.,
        };
        sum += weight;
        output.push((sample as usize, weight as f32));
    }
    for (_, weight) in &mut output {
        *weight /= sum as f32;
    }
    output
}
fn cms_error(error: moxcms::CmsError) -> Error {
    Error::backend("moxcms", error.to_string()).with_cause(error)
}
