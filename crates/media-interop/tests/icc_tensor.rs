//! Actual ICC conversion into final tensor packing, with explicit color policies.
#![cfg(feature = "icc")]

use media_core::{
    error::ErrorKind,
    media::{
        ColorDescription, Image, ImageLayout, ImageMetadata, Orientation, PixelFormat, PlaneLayout,
    },
    storage::StorageLease,
};
use media_interop::tensor::{
    AlphaPolicy, ColorAssumption, IccGamut, IccLimits, IccPolicy, NumericRange, TensorLayout,
    TensorSpec, images_to_tensor, to_tensor,
};
use moxcms::{ColorProfile, Layout, RenderingIntent, ToneReprCurve, TransformOptions};

fn profile_image(profile: ColorProfile, values: &[f32]) -> Image {
    Image::from_packed(
        (values.len() / 3) as u32,
        1,
        PixelFormat::RgbF32,
        values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    )
    .unwrap()
    .with_metadata(ImageMetadata {
        icc_profile: Some(profile.encode().unwrap()),
        ..Default::default()
    })
}
fn spec() -> TensorSpec {
    TensorSpec::rgb_f32().icc_policy(IccPolicy::ConvertToSrgb {
        gamut: IccGamut::Reject,
    })
}
fn reference(image: &Image, input: &[f32]) -> Vec<f32> {
    let profile =
        ColorProfile::new_from_slice(image.metadata().icc_profile.as_ref().unwrap()).unwrap();
    let transform = profile
        .create_transform_f32(
            Layout::Rgb,
            &ColorProfile::new_srgb(),
            Layout::Rgb,
            TransformOptions {
                rendering_intent: RenderingIntent::RelativeColorimetric,
                allow_use_cicp_transfer: false,
                prefer_fixed_point: false,
                allow_extended_range_rgb_xyz: true,
                ..Default::default()
            },
        )
        .unwrap();
    let mut output = vec![0.0; input.len()];
    transform.transform(input, &mut output).unwrap();
    output
}
fn close(actual: &[f32], expected: &[f32], epsilon: f32) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() < epsilon,
            "actual {actual}, expected {expected}"
        );
    }
}

#[test]
fn display_p3_pixels_are_actually_converted_and_icc_is_authoritative_without_srgb_assumptions() {
    let input = [0.65_f32, 0.35, 0.2];
    let image = profile_image(ColorProfile::new_display_p3(), &input);
    assert!(to_tensor(&image, &TensorSpec::rgb_f32()).is_err());
    let tensor = to_tensor(&image, &spec()).unwrap();
    let expected = reference(&image, &input);
    close(tensor.data(), &expected, 0.00001);
    assert!(tensor.data()[0] > 0.69 && tensor.data()[1] < 0.34 && tensor.data()[2] < 0.18);
    assert!(tensor.assumptions().is_empty());
    assert!(
        tensor
            .color_operations()
            .iter()
            .any(|v| v.contains("nonlinear sRGB"))
    );
    assert!(
        tensor
            .color_operations()
            .iter()
            .any(|v| v.contains("unmeasured"))
    );
    let assumed = to_tensor(
        &image,
        &spec().color_assumption(ColorAssumption::AssumeSrgb),
    )
    .unwrap();
    assert_eq!(tensor.data(), assumed.data());
    assert!(assumed.assumptions().is_empty());
}

#[test]
fn out_of_srgb_gamut_is_an_explicit_rejection_or_reported_clip_before_normalization() {
    let image = profile_image(ColorProfile::new_display_p3(), &[0., 1., 0.]);
    assert_eq!(
        to_tensor(&image, &spec()).unwrap_err().kind(),
        ErrorKind::Incompatible
    );
    let clipped = spec().icc_policy(IccPolicy::ConvertToSrgb {
        gamut: IccGamut::Clip,
    });
    let tensor = to_tensor(&image, &clipped.clone().range(NumericRange::Byte)).unwrap();
    assert_eq!(tensor.data(), &[0., 255., 0.]);
    assert!(
        tensor
            .color_operations()
            .iter()
            .any(|v| v.contains("explicitly clipped"))
    );
    assert_eq!(
        to_tensor(
            &image,
            &clipped.normalize([0.5, 0.25, 0.0], [0.5, 0.25, 1.0])
        )
        .unwrap()
        .data(),
        &[-1., 3., 0.]
    );
}

#[test]
fn known_partial_and_full_metadata_must_agree_with_actual_icc_colorimetry() {
    let image = profile_image(ColorProfile::new_display_p3(), &[0.65, 0.35, 0.2]);
    let mut metadata = image.metadata().clone();
    metadata.color = ColorDescription {
        primaries: Some("display-p3".into()),
        transfer: Some("srgb".into()),
        matrix: Some("rgb".into()),
        range: Some("full".into()),
        chroma_siting: None,
    };
    assert!(to_tensor(&image.clone().with_metadata(metadata.clone()), &spec()).is_ok());
    metadata.color.primaries = Some("bt709".into());
    metadata.color.transfer = None;
    assert_eq!(
        to_tensor(
            &image.clone().with_metadata(metadata.clone()),
            &spec().color_assumption(ColorAssumption::AssumeSrgb)
        )
        .unwrap_err()
        .kind(),
        ErrorKind::Incompatible
    );
    metadata.color.primaries = None;
    metadata.color.transfer = Some("linear".into());
    assert_eq!(
        to_tensor(&image.clone().with_metadata(metadata.clone()), &spec())
            .unwrap_err()
            .kind(),
        ErrorKind::Incompatible
    );
    metadata.color.transfer = None;
    metadata.color.range = Some("limited".into());
    assert!(to_tensor(&image.clone().with_metadata(metadata.clone()), &spec()).is_err());
    metadata.color.range = Some("full".into());
    metadata.color.matrix = Some("bt709".into());
    assert!(to_tensor(&image.clone().with_metadata(metadata.clone()), &spec()).is_err());
    metadata.color.matrix = None;
    metadata.color.chroma_siting = Some("left".into());
    assert!(to_tensor(&image.with_metadata(metadata), &spec()).is_err());
}

#[test]
fn pitched_sixteen_bit_rgb_keeps_precision_through_profile_conversion_and_nhwc_packing() {
    let values = [20000_u16, 30000, 40000, 40000, 30000, 20000];
    let mut bytes = vec![99_u8; 16];
    for (row, pixel) in values.chunks_exact(3).enumerate() {
        for (channel, value) in pixel.iter().enumerate() {
            bytes[2 + row * 8 + channel * 2..4 + row * 8 + channel * 2]
                .copy_from_slice(&value.to_le_bytes());
        }
    }
    let image = Image::new(
        ImageLayout {
            width: 1,
            height: 2,
            allocated_width: 1,
            allocated_height: 2,
            format: PixelFormat::Rgb16,
            planes: vec![PlaneLayout {
                offset: 2,
                row_stride: 8,
                row_bytes: 6,
                rows: 2,
            }],
        },
        StorageLease::from_bytes(bytes),
        ImageMetadata {
            icc_profile: Some(ColorProfile::new_display_p3().encode().unwrap()),
            ..Default::default()
        },
    )
    .unwrap();
    let input = values
        .iter()
        .map(|v| *v as f32 / 65535.0)
        .collect::<Vec<_>>();
    let tensor = to_tensor(&image, &spec().layout(TensorLayout::Nhwc)).unwrap();
    close(tensor.data(), &reference(&image, &input), 0.00001);
    assert_eq!(tensor.shape(), &[1, 2, 1, 3]);
    let source_copy = Image::from_packed(
        1,
        2,
        PixelFormat::Rgb16,
        values.into_iter().flat_map(u16::to_le_bytes).collect(),
    )
    .unwrap();
    assert_eq!(
        image.host_bytes().unwrap()[2],
        source_copy.host_bytes().unwrap()[0]
    );
}

#[test]
fn mixed_source_profiles_transform_per_image_into_one_final_batch_without_source_mutation() {
    let input = [0.25_f32, 0.5, 0.75];
    let srgb = profile_image(ColorProfile::new_srgb(), &input);
    let mut profile = ColorProfile::new_srgb();
    profile.red_trc = Some(ToneReprCurve::Parametric(vec![1.]));
    profile.green_trc = profile.red_trc.clone();
    profile.blue_trc = profile.red_trc.clone();
    profile.cicp = None;
    let linear = profile_image(profile, &input);
    let tensor = images_to_tensor(
        &[srgb.clone(), linear.clone()],
        &spec().layout(TensorLayout::Nhwc),
    )
    .unwrap();
    close(&tensor.data()[..3], &input, 0.0001);
    close(&tensor.data()[3..], &[0.5371, 0.7354, 0.8808], 0.0002);
    assert_eq!(tensor.shape(), &[2, 1, 1, 3]);
    assert!(
        tensor
            .color_operations()
            .iter()
            .all(|v| v.ends_with("2/2 images"))
    );
    let original = input
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    assert_eq!(linear.packed_bytes().unwrap(), original);
    assert_eq!(srgb.packed_bytes().unwrap(), original);
}

#[test]
fn parser_limits_alpha_orientation_malformed_profiles_and_storage_contracts_are_checked() {
    let image = profile_image(ColorProfile::new_srgb(), &[0.25, 0.5, 0.75]);
    let length = image.metadata().icc_profile.as_ref().unwrap().len();
    assert_eq!(
        to_tensor(
            &image,
            &spec().icc_limits(IccLimits {
                max_profile_bytes: length - 1,
                ..Default::default()
            })
        )
        .unwrap_err()
        .kind(),
        ErrorKind::ResourceExhausted
    );
    assert!(
        to_tensor(
            &image,
            &spec().icc_limits(IccLimits {
                max_clut_bytes: 0,
                ..Default::default()
            })
        )
        .is_err()
    );
    assert!(to_tensor(&image, &spec().max_bytes(4)).is_err());
    let mut metadata = image.metadata().clone();
    metadata.icc_profile = Some(vec![1, 2, 3]);
    assert!(to_tensor(&image.clone().with_metadata(metadata), &spec()).is_err());
    let mut metadata = image.metadata().clone();
    metadata.orientation = Orientation::Rotate90;
    assert!(to_tensor(&image.clone().with_metadata(metadata), &spec()).is_err());
    let rgba = Image::from_packed(1, 1, PixelFormat::Rgba8, vec![64, 128, 192, 0])
        .unwrap()
        .with_metadata(image.metadata().clone());
    assert!(to_tensor(&rgba, &spec()).is_err());
    close(
        to_tensor(&rgba, &spec().alpha(AlphaPolicy::Discard))
            .unwrap()
            .data(),
        &[64.0 / 255.0, 128.0 / 255.0, 192.0 / 255.0],
        0.0001,
    );
    let gray = Image::from_packed(1, 1, PixelFormat::Gray8, vec![128])
        .unwrap()
        .with_metadata(image.metadata().clone());
    assert!(to_tensor(&gray, &spec()).is_err());
    let mut metadata = image.metadata().clone();
    metadata.icc_profile.as_mut().unwrap()[16..20].copy_from_slice(b"CMYK");
    assert!(to_tensor(&image.with_metadata(metadata), &spec()).is_err());
}

#[test]
fn tone_curve_table_limits_are_applied_before_pixel_conversion() {
    let mut profile = ColorProfile::new_srgb();
    profile.red_trc = Some(ToneReprCurve::Lut((0..256).map(|n| n * 257).collect()));
    profile.green_trc = profile.red_trc.clone();
    profile.blue_trc = profile.red_trc.clone();
    profile.cicp = None;
    let image = profile_image(profile, &[0.25, 0.5, 0.75]);
    assert!(
        to_tensor(
            &image,
            &spec().icc_limits(IccLimits {
                max_trc_entries: 8,
                ..Default::default()
            })
        )
        .is_err()
    );
    assert!(to_tensor(&image, &spec()).is_ok());
}
