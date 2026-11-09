//! Explicit optional ICC RGB interpretation for host tensor preparation.
//!
//! The `icc` feature uses moxcms with bounded profile, CLUT, and TRC parsing.
//! Its private parser/transform heap is not measured by tensor output counters.
//! Each image prepares its pixel transform once; pixels use stack buffers and
//! go directly into final tensor storage. No whole-image color scratch is used.

use media_core::{
    error::{Error, Result},
    media::Image,
};

/// Handling of an embedded ICC profile during RGB model-input preparation.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum IccPolicy {
    /// Reject ICC-bearing input; no profile is silently discarded.
    #[default]
    Reject,
    /// Convert profile-defined RGB to nonlinear sRGB with relative-colorimetric
    /// intent. Requires the `icc` feature and an explicit output gamut policy.
    ConvertToSrgb {
        /// Treatment of actual output components outside the sRGB unit cube.
        gamut: IccGamut,
    },
}
/// Output gamut semantics for an explicit ICC-to-sRGB conversion.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IccGamut {
    /// Fail on out-of-gamut output, including floating-point source images.
    /// Endpoint residuals within 1e-4 linear-sRGB units are rounded to the unit cube.
    Reject,
    /// Explicitly clamp sRGB components to zero through one and report clipping.
    Clip,
}
/// Hard input/parser bounds, separate from the final tensor allocation limit.
///
/// These constrain accepted ICC objects and native table parsing, not an exact
/// measurement of CMS private heap or precomputed transform tables.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IccLimits {
    /// Maximum embedded profile payload bytes, inclusive.
    pub max_profile_bytes: usize,
    /// Maximum parsed CLUT payload bytes.
    pub max_clut_bytes: usize,
    /// Maximum elements in a parsed tone-reproduction curve.
    pub max_trc_entries: usize,
}
impl Default for IccLimits {
    fn default() -> Self {
        Self {
            max_profile_bytes: 4 * 1024 * 1024,
            max_clut_bytes: 1024 * 1024,
            max_trc_entries: 65536,
        }
    }
}
impl IccLimits {
    /// Reject zero or unrepresentable parser limits before allocation.
    pub fn validate(self) -> Result<()> {
        if self.max_profile_bytes == 0
            || self.max_profile_bytes == usize::MAX
            || self.max_clut_bytes == 0
            || self.max_trc_entries == 0
        {
            return Err(Error::invalid(
                "ICC parser bounds must be nonzero and profile bytes must leave room for an inclusive bound",
            ));
        }
        Ok(())
    }
}

#[cfg(not(feature = "icc"))]
pub(crate) struct IccTransform;
#[cfg(not(feature = "icc"))]
impl IccTransform {
    pub(crate) fn new(_: &Image, _: IccLimits, _: IccGamut) -> Result<Self> {
        Err(Error::unsupported(
            "ICC tensor conversion requires the media-interop/icc feature (media/icc through the facade)",
        ))
    }
    pub(crate) fn apply(&self, _: [f32; 3]) -> Result<([f32; 3], bool)> {
        unreachable!("ICC construction fails when the feature is disabled")
    }
}

#[cfg(feature = "icc")]
mod managed {
    use super::*;
    use media_core::media::ColorDescription;
    use moxcms::{
        ColorProfile, DataColorSpace, Layout, ParsingOptions, RenderingIntent, ToneReprCurve,
        TransformF32Executor, TransformOptions,
    };
    use std::sync::Arc;

    pub(crate) struct IccTransform {
        transform: Arc<TransformF32Executor>,
        gamut: IccGamut,
        extended_source: bool,
    }
    impl IccTransform {
        pub(crate) fn new(image: &Image, limits: IccLimits, gamut: IccGamut) -> Result<Self> {
            limits.validate()?;
            let bytes = image
                .metadata()
                .icc_profile
                .as_deref()
                .ok_or_else(|| Error::invalid("ICC conversion needs an embedded profile"))?;
            if bytes.len() > limits.max_profile_bytes {
                return Err(Error::exhausted(
                    "ICC profile exceeds the requested parser byte limit",
                )
                .with_constraint("max_profile_bytes"));
            }
            let profile = ColorProfile::new_from_slice_with_options(
                bytes,
                ParsingOptions {
                    max_profile_size: limits.max_profile_bytes + 1,
                    max_allowed_clut_size: limits.max_clut_bytes,
                    max_allowed_trc_size: limits.max_trc_entries,
                },
            )
            .map_err(cms_error)?;
            if profile.color_space != DataColorSpace::Rgb {
                return Err(Error::incompatible(
                    "RGB tensor packing requires an actual RGB ICC device profile",
                ));
            }
            if profile.cicp.is_some_and(|c| {
                matches!(
                    c.transfer_characteristics,
                    moxcms::TransferCharacteristics::Smpte2084
                        | moxcms::TransferCharacteristics::Hlg
                )
            }) {
                return Err(Error::unsupported(
                    "HDR ICC tensor conversion requires an explicit luminance/tone-mapping contract",
                ));
            }
            validate_tags(&profile, &image.metadata().color)?;
            let extended_source = !has_lut(&profile);
            let transform = profile
                .create_transform_f32(
                    Layout::Rgb,
                    &ColorProfile::new_srgb(),
                    Layout::Rgb,
                    options(),
                )
                .map_err(cms_error)?;
            Ok(Self {
                transform,
                gamut,
                extended_source,
            })
        }
        pub(crate) fn apply(&self, input: [f32; 3]) -> Result<([f32; 3], bool)> {
            if input.iter().any(|v| !v.is_finite()) {
                return Err(Error::invalid(
                    "ICC RGB source contains nonfinite components",
                ));
            }
            if !self.extended_source && input.iter().any(|v| !(0.0..=1.0).contains(v)) {
                return Err(Error::incompatible(
                    "ICC LUT device input must lie within its defined unit domain",
                ));
            }
            let mut output = [0.0; 3];
            self.transform
                .transform(&input, &mut output)
                .map_err(cms_error)?;
            if output.iter().any(|v| !v.is_finite()) {
                return Err(Error::backend(
                    "moxcms",
                    "ICC transform produced nonfinite output",
                ));
            }
            // ICC fixed-point colorant/curve quantization can leave tiny endpoint
            // residuals. Check the 1e-4 tolerance in linear sRGB, where a transfer
            // curve cannot amplify a near-zero matrix rounding error.
            let clipped = output.iter().any(|&v| {
                let linear = if v <= 0.04045 {
                    v / 12.92
                } else {
                    ((v + 0.055) / 1.055).powf(2.4)
                };
                !(-0.0001..=1.0001).contains(&linear)
            });
            if clipped && self.gamut == IccGamut::Reject {
                return Err(Error::incompatible(
                    "ICC conversion exceeds the sRGB tensor gamut; explicitly request clipping if acceptable",
                ));
            }
            output.iter_mut().for_each(|v| *v = v.clamp(0.0, 1.0));
            Ok((output, clipped))
        }
    }
    fn options() -> TransformOptions {
        TransformOptions {
            rendering_intent: RenderingIntent::RelativeColorimetric,
            allow_use_cicp_transfer: false,
            prefer_fixed_point: false,
            allow_extended_range_rgb_xyz: true,
            ..Default::default()
        }
    }
    fn has_lut(profile: &ColorProfile) -> bool {
        profile.lut_a_to_b_perceptual.is_some()
            || profile.lut_a_to_b_colorimetric.is_some()
            || profile.lut_a_to_b_saturation.is_some()
            || profile.lut_b_to_a_perceptual.is_some()
            || profile.lut_b_to_a_colorimetric.is_some()
            || profile.lut_b_to_a_saturation.is_some()
    }
    fn linear(mut profile: ColorProfile) -> ColorProfile {
        profile.red_trc = Some(ToneReprCurve::Parametric(vec![1.]));
        profile.green_trc = profile.red_trc.clone();
        profile.blue_trc = profile.red_trc.clone();
        profile.cicp = None;
        profile
    }
    fn validate_tags(profile: &ColorProfile, color: &ColorDescription) -> Result<()> {
        if color
            .matrix
            .as_deref()
            .is_some_and(|v| !matches!(v.to_ascii_lowercase().as_str(), "rgb" | "identity" | "0"))
            || color
                .range
                .as_deref()
                .is_some_and(|v| !matches!(v.to_ascii_lowercase().as_str(), "full" | "pc"))
            || color.chroma_siting.is_some()
        {
            return Err(Error::incompatible(
                "RGB ICC tensor input contradicts full-range identity RGB storage metadata",
            ));
        }
        if color.primaries.is_none() && color.transfer.is_none() {
            return Ok(());
        }
        if has_lut(profile) {
            return Err(Error::unsupported(
                "named primaries/transfer cannot be independently validated against an arbitrary ICC LUT profile; provide one authoritative interpretation",
            ));
        }
        if let Some(name) = &color.primaries {
            let named = named_primaries(name)?;
            if !equivalent(&linear(profile.clone()), &linear(named))? {
                return Err(Error::incompatible(
                    "ICC profile contradicts explicit RGB primaries",
                ));
            }
        }
        if let Some(name) = &color.transfer {
            let curve = named_transfer(name)?;
            let mut expected = profile.clone();
            expected.red_trc = curve.red_trc;
            expected.green_trc = curve.green_trc;
            expected.blue_trc = curve.blue_trc;
            expected.cicp = None;
            if !equivalent(profile, &expected)? {
                return Err(Error::incompatible(
                    "ICC profile contradicts explicit RGB transfer metadata",
                ));
            }
        }
        Ok(())
    }
    fn named_primaries(name: &str) -> Result<ColorProfile> {
        use moxcms::{
            CicpColorPrimaries as P, MatrixCoefficients as M, TransferCharacteristics as T,
        };
        let p = match name.to_ascii_lowercase().as_str() {
            "srgb" | "bt709" | "rec709" | "bt.709" => P::Bt709,
            "display-p3" | "smpte432" | "smpte432-1" => P::Smpte432,
            "bt2020" | "rec2020" => P::Bt2020,
            _ => {
                return Err(Error::unsupported(
                    "ICC input has unsupported named RGB primaries",
                ));
            }
        };
        Ok(ColorProfile::new_from_cicp(moxcms::CicpProfile {
            color_primaries: p,
            transfer_characteristics: T::Linear,
            matrix_coefficients: M::Identity,
            full_range: true,
        }))
    }
    fn named_transfer(name: &str) -> Result<ColorProfile> {
        use moxcms::{
            CicpColorPrimaries as P, MatrixCoefficients as M, TransferCharacteristics as T,
        };
        let name = name.to_ascii_lowercase();
        let transfer = match name.as_str() {
            "srgb" | "iec61966-2-1" | "iec 61966-2-1" | "iec61966_2_1" => T::Srgb,
            "linear" => T::Linear,
            "bt709" => T::Bt709,
            value if value.starts_with("png-gamma:") => {
                let gamma = value[10..]
                    .parse::<u32>()
                    .map_err(|_| Error::invalid("invalid named PNG gamma"))?;
                if gamma == 0 {
                    return Err(Error::invalid("PNG gamma must be positive"));
                }
                let mut profile = ColorProfile::new_srgb();
                profile.red_trc = Some(ToneReprCurve::Parametric(vec![100000.0 / gamma as f32]));
                profile.green_trc = profile.red_trc.clone();
                profile.blue_trc = profile.red_trc.clone();
                profile.cicp = None;
                return Ok(profile);
            }
            _ => {
                return Err(Error::unsupported(
                    "ICC input has unsupported SDR transfer metadata; HDR tone mapping is not implicit",
                ));
            }
        };
        Ok(ColorProfile::new_from_cicp(moxcms::CicpProfile {
            color_primaries: P::Bt709,
            transfer_characteristics: transfer,
            matrix_coefficients: M::Identity,
            full_range: true,
        }))
    }
    fn equivalent(actual: &ColorProfile, expected: &ColorProfile) -> Result<bool> {
        let input = [
            0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 1., 1., 1., 0.02, 0.02, 0.02, 0.18,
            0.18, 0.18, 0.5, 0.5, 0.5,
        ];
        let target = linear(ColorProfile::new_srgb());
        let mut actual_values = [0.; 24];
        let mut expected_values = [0.; 24];
        actual
            .create_transform_f32(Layout::Rgb, &target, Layout::Rgb, options())
            .map_err(cms_error)?
            .transform(&input, &mut actual_values)
            .map_err(cms_error)?;
        expected
            .create_transform_f32(Layout::Rgb, &target, Layout::Rgb, options())
            .map_err(cms_error)?
            .transform(&input, &mut expected_values)
            .map_err(cms_error)?;
        Ok(actual_values
            .iter()
            .zip(expected_values)
            .all(|(a, b)| a.is_finite() && b.is_finite() && (*a - b).abs() < 0.001))
    }
    fn cms_error(error: moxcms::CmsError) -> Error {
        Error::backend("moxcms", error.to_string()).with_cause(error)
    }
}
#[cfg(feature = "icc")]
pub(crate) use managed::IccTransform;
