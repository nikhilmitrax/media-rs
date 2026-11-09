//! Numerical fixtures for ICC conversion, linear-light resize, and alpha semantics.
use media_backend_native::{
    image::{self, DecodeLimits, EncodeOptions, Filter, Fit, ImageFormat, MetadataLoss},
    image_color::*,
};
use media_core::{
    error::ErrorKind,
    media::{ColorDescription, Image, ImageMetadata, PixelFormat},
};
use moxcms::{ColorProfile, Layout, TransformOptions};

fn tagged(format: PixelFormat, bytes: Vec<u8>) -> Image {
    Image::from_packed(2, 1, format, bytes)
        .unwrap()
        .with_metadata(ImageMetadata {
            color: ColorDescription {
                primaries: Some("bt709".into()),
                transfer: Some("srgb".into()),
                matrix: Some("rgb".into()),
                range: Some("full".into()),
                chroma_siting: None,
            },
            ..ImageMetadata::default()
        })
}
fn spec(domain: ResizeDomain) -> ResizeSpec {
    ResizeSpec {
        fit: Fit::inside(1, 1),
        filter: Filter::Triangle,
        domain,
        color: ColorConvertSpec::default(),
    }
}

#[test]
fn linear_srgb_black_white_average_is_188_instead_of_128() {
    let input = tagged(PixelFormat::Rgb8, vec![0, 0, 0, 255, 255, 255]);
    let linear = resize(&input, spec(ResizeDomain::LinearLight)).unwrap();
    let stored = resize(&input, spec(ResizeDomain::StoredComponents)).unwrap();
    assert_eq!(linear.image.packed_bytes().unwrap(), [188, 188, 188]);
    assert_eq!(stored.image.packed_bytes().unwrap(), [128, 128, 128]);
    assert!(linear.assumptions.is_empty());
    assert_eq!(linear.image.metadata().color, input.metadata().color);
    assert!(linear.image.metadata().icc_profile.is_some());
}

#[test]
fn transparent_color_does_not_bleed_into_visible_edges() {
    let input = tagged(PixelFormat::Rgba8, vec![255, 0, 0, 0, 0, 0, 255, 255]);
    let mut options = spec(ResizeDomain::LinearLight);
    options.color.input_alpha = Some(AlphaMode::Straight);
    let output = resize(&input, options).unwrap();
    assert_eq!(output.image.packed_bytes().unwrap(), [0, 0, 255, 128]);
    assert!(output.assumptions[0].contains("Straight"));
    options.color.output_alpha = AlphaMode::Premultiplied;
    let output = resize(&input, options).unwrap();
    let pixel = output.image.packed_bytes().unwrap();
    assert_eq!((pixel[0], pixel[1], pixel[3]), (0, 0, 128));
    assert!(
        pixel[2].abs_diff(128) <= 1,
        "CMS float precision may straddle one quantization tie"
    );
    assert!(resize(&input, spec(ResizeDomain::LinearLight)).is_err());
}

#[test]
fn stored_premultiplied_input_is_unassociated_before_color_conversion() {
    let input = tagged(PixelFormat::Rgba8, vec![0, 0, 0, 0, 0, 0, 128, 128]);
    let mut options = spec(ResizeDomain::LinearLight);
    options.color.input_alpha = Some(AlphaMode::Premultiplied);
    let output = resize(&input, options).unwrap();
    assert_eq!(output.image.packed_bytes().unwrap(), [0, 0, 255, 64]);
}

#[test]
fn sixteen_bit_precision_is_retained_through_linear_resize_and_png() {
    let input = tagged(
        PixelFormat::Rgb16,
        [0u16, 0, 0, 65535, 65535, 65535]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect(),
    );
    let output = resize(&input, spec(ResizeDomain::LinearLight))
        .unwrap()
        .image;
    assert_eq!(output.pixel_format(), PixelFormat::Rgb16);
    let value = u16::from_le_bytes(output.packed_bytes().unwrap()[..2].try_into().unwrap());
    assert!((i32::from(value) - 48192).abs() <= 4);
    assert_ne!(
        value % 257,
        0,
        "must not pass through an eight-bit intermediate"
    );
    let encoded = image::encode(&output, EncodeOptions::new(ImageFormat::Png)).unwrap();
    let decoded = image::decode(&encoded.bytes, DecodeLimits::default()).unwrap();
    assert_eq!(decoded.pixel_format(), PixelFormat::Rgb16);
    assert_eq!(
        decoded.packed_bytes().unwrap(),
        output.packed_bytes().unwrap()
    );
    assert_eq!(
        decoded.metadata().icc_profile,
        output.metadata().icc_profile
    );
}

#[test]
fn icc_conversion_matches_direct_moxcms_and_updates_output_interpretation() {
    let source_profile = ColorProfile::new_display_p3();
    let input_components = [0.8f32, 0.3, 0.1];
    let input = Image::from_packed(
        1,
        1,
        PixelFormat::RgbF32,
        input_components
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect(),
    )
    .unwrap()
    .with_metadata(ImageMetadata {
        icc_profile: Some(source_profile.encode().unwrap()),
        exif: Some(vec![1, 2, 3]),
        ..ImageMetadata::default()
    });
    let output = convert(
        &input,
        ColorConvertSpec {
            target: Some(ColorSpace::Srgb),
            ..ColorConvertSpec::default()
        },
    )
    .unwrap();
    let actual = output
        .image
        .packed_bytes()
        .unwrap()
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect::<Vec<_>>();
    let mut reference = [0.; 3];
    let parsed =
        ColorProfile::new_from_slice(input.metadata().icc_profile.as_ref().unwrap()).unwrap();
    parsed
        .create_transform_f32(
            Layout::Rgb,
            &ColorProfile::new_srgb(),
            Layout::Rgb,
            TransformOptions {
                prefer_fixed_point: false,
                allow_extended_range_rgb_xyz: true,
                ..TransformOptions::default()
            },
        )
        .unwrap()
        .transform(&input_components, &mut reference)
        .unwrap();
    for (actual, expected) in actual.iter().zip(reference) {
        assert!((*actual - expected).abs() < 0.0001);
    }
    assert_eq!(
        output.image.metadata().color.primaries.as_deref(),
        Some("bt709")
    );
    assert_eq!(
        output.image.metadata().color.transfer.as_deref(),
        Some("srgb")
    );
    assert_ne!(
        output.image.metadata().icc_profile,
        input.metadata().icc_profile
    );
    assert!(output.image.metadata().exif.is_none());
    assert_eq!(
        output.metadata_loss,
        vec![MetadataLoss::ExifAfterColorTransform]
    );
}

#[test]
fn missing_and_contradictory_interpretations_do_not_get_overridden() {
    let untagged = Image::from_packed(1, 1, PixelFormat::Rgb8, vec![20, 40, 60]).unwrap();
    assert_eq!(
        convert(&untagged, ColorConvertSpec::default())
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidInput
    );
    let output = convert(
        &untagged,
        ColorConvertSpec {
            missing_color: Some(ColorSpace::Srgb),
            ..ColorConvertSpec::default()
        },
    )
    .unwrap();
    assert_eq!(output.image.packed_bytes().unwrap(), [20, 40, 60]);
    assert!(output.assumptions[0].contains("Srgb"));
    let mut metadata = tagged(PixelFormat::Rgb8, vec![0; 6]).metadata().clone();
    metadata.icc_profile = Some(ColorProfile::new_display_p3().encode().unwrap());
    let contradictory = untagged.clone().with_metadata(metadata);
    assert_eq!(
        convert(&contradictory, ColorConvertSpec::default())
            .unwrap_err()
            .kind(),
        ErrorKind::Incompatible
    );
    let mut metadata = contradictory.metadata().clone();
    metadata.icc_profile = None;
    metadata.color.range = Some("limited".into());
    assert!(
        convert(
            &untagged.with_metadata(metadata),
            ColorConvertSpec {
                missing_color: Some(ColorSpace::Srgb),
                ..ColorConvertSpec::default()
            }
        )
        .is_err()
    );
}

#[test]
fn integer_gamut_clipping_is_an_explicit_reported_choice() {
    let source = Image::from_packed(1, 1, PixelFormat::Rgb8, vec![255, 0, 0])
        .unwrap()
        .with_metadata(ImageMetadata {
            icc_profile: Some(ColorProfile::new_display_p3().encode().unwrap()),
            ..ImageMetadata::default()
        });
    let spec = ColorConvertSpec {
        target: Some(ColorSpace::Srgb),
        ..ColorConvertSpec::default()
    };
    assert_eq!(
        convert(&source, spec).unwrap_err().kind(),
        ErrorKind::Incompatible
    );
    let output = convert(
        &source,
        ColorConvertSpec {
            gamut: GamutPolicy::Clip,
            ..spec
        },
    )
    .unwrap();
    assert_eq!(output.image.packed_bytes().unwrap(), [255, 0, 0]);
    assert!(output.assumptions.iter().any(|v| v.contains("clipped")));
}

#[test]
fn cover_geometry_and_working_allocation_are_bounded() {
    let input = tagged(PixelFormat::Rgb8, vec![0, 0, 0, 255, 255, 255]);
    let mut options = spec(ResizeDomain::LinearLight);
    options.fit = Fit::cover(2, 2);
    let output = resize(&input, options).unwrap();
    assert_eq!((output.image.width(), output.image.height()), (2, 2));
    options.color.max_working_bytes = 1;
    assert_eq!(
        resize(&input, options).unwrap_err().kind(),
        ErrorKind::ResourceExhausted
    );
    options.fit = Fit::cover(u32::MAX, u32::MAX);
    options.color.max_working_bytes = usize::MAX;
    assert!(resize(&input, options).is_err());
}

#[test]
fn png_srgb_cicp_and_gamma_metadata_are_established_without_guessing() {
    fn encoded(info: png::Info<'static>) -> Vec<u8> {
        let cicp = info.coding_independent_code_points;
        let mut bytes = Vec::new();
        let mut encoder = png::Encoder::with_info(&mut bytes, info).unwrap();
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().unwrap();
        if let Some(cicp) = cicp {
            writer
                .write_chunk(
                    png::chunk::cICP,
                    &[
                        cicp.color_primaries,
                        cicp.transfer_function,
                        cicp.matrix_coefficients,
                        u8::from(cicp.is_video_full_range_image),
                    ],
                )
                .unwrap();
        }
        writer.write_image_data(&[128, 128, 128]).unwrap();
        drop(writer);
        bytes
    }
    let mut info = png::Info::with_size(1, 1);
    info.srgb = Some(png::SrgbRenderingIntent::RelativeColorimetric);
    let bytes = encoded(info);
    let decoded = image::decode(&bytes, DecodeLimits::default()).unwrap();
    assert_eq!(decoded.metadata().color.transfer.as_deref(), Some("srgb"));
    assert!(
        convert(&decoded, ColorConvertSpec::default())
            .unwrap()
            .assumptions
            .is_empty()
    );
    let mut info = png::Info::with_size(1, 1);
    info.coding_independent_code_points = Some(png::CodingIndependentCodePoints {
        color_primaries: 12,
        transfer_function: 13,
        matrix_coefficients: 0,
        is_video_full_range_image: true,
    });
    let bytes = encoded(info);
    assert_eq!(
        image::probe(&bytes, DecodeLimits::default())
            .unwrap()
            .metadata
            .color
            .primaries
            .as_deref(),
        Some("display-p3")
    );
    let mut contradictory = png::Info::with_size(1, 1);
    contradictory.srgb = Some(png::SrgbRenderingIntent::RelativeColorimetric);
    contradictory.coding_independent_code_points = Some(png::CodingIndependentCodePoints {
        color_primaries: 12,
        transfer_function: 13,
        matrix_coefficients: 0,
        is_video_full_range_image: true,
    });
    assert!(image::decode(&encoded(contradictory), DecodeLimits::default()).is_err());
    let mut info = png::Info::with_size(1, 1);
    info.source_gamma = Some(png::ScaledFloat::new(0.5));
    let bytes = encoded(info);
    let decoded = image::decode(&bytes, DecodeLimits::default()).unwrap();
    assert_eq!(
        decoded.metadata().color.transfer.as_deref(),
        Some("png-gamma:50000")
    );
    let output = convert(
        &decoded,
        ColorConvertSpec {
            missing_color: Some(ColorSpace::Srgb),
            target: Some(ColorSpace::LinearSrgb),
            ..ColorConvertSpec::default()
        },
    )
    .unwrap();
    assert!(
        output.image.packed_bytes().unwrap()[0].abs_diff(64) <= 1,
        "PNG gamma0.5 establishes a square-law decoding transfer"
    );
}

#[test]
fn floating_output_retains_extended_gamut_and_public_working_estimate_matches_guard() {
    let source = Image::from_packed(
        1,
        1,
        PixelFormat::RgbF32,
        [1f32, 0., 0.]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect(),
    )
    .unwrap()
    .with_metadata(ImageMetadata {
        icc_profile: Some(ColorProfile::new_display_p3().encode().unwrap()),
        ..ImageMetadata::default()
    });
    let output = convert(
        &source,
        ColorConvertSpec {
            target: Some(ColorSpace::Srgb),
            ..ColorConvertSpec::default()
        },
    )
    .unwrap();
    let values = output
        .image
        .packed_bytes()
        .unwrap()
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert!(values[0] > 1.);
    assert!(values[1] < 0.);
    assert!(values[2] < 0.);
    let input = tagged(PixelFormat::Rgb8, vec![0, 0, 0, 255, 255, 255]);
    let mut options = spec(ResizeDomain::LinearLight);
    let planned = working_bytes(&input, Some(options)).unwrap();
    options.color.max_working_bytes = planned;
    resize(&input, options).unwrap();
    options.color.max_working_bytes = planned - 1;
    assert_eq!(
        resize(&input, options).unwrap_err().kind(),
        ErrorKind::ResourceExhausted
    );
}

#[test]
fn unassociating_alpha_discards_stale_exif_and_plans_matching_metadata_payload() {
    let image = tagged(PixelFormat::Rgba8, vec![128, 0, 0, 128, 0, 128, 0, 128]);
    let mut metadata = image.metadata().clone();
    metadata.exif = Some(vec![1, 2, 3]);
    let image = image.with_metadata(metadata);
    let spec = ColorConvertSpec {
        input_alpha: Some(AlphaMode::Premultiplied),
        ..ColorConvertSpec::default()
    };
    let planned = metadata_bytes(&image, &spec, false).unwrap();
    let output = convert(&image, spec).unwrap();
    assert!(output.image.metadata().exif.is_none());
    assert_eq!(
        output.metadata_loss,
        vec![MetadataLoss::ExifAfterColorTransform]
    );
    assert_eq!(
        planned,
        image::metadata_payload_bytes(output.image.metadata()).unwrap()
    );
}
