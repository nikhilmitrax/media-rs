//! Numerical component conversion, padding, metadata, pitch and budget fixtures.
use media_backend_native::{
    image::{MetadataLoss, metadata_payload_bytes},
    image_color::{AlphaMode, GamutPolicy},
    image_pixels::*,
};
use media_core::{
    error::ErrorKind,
    media::{ColorDescription, Image, ImageLayout, ImageMetadata, PixelFormat, PlaneLayout},
    storage::StorageLease,
};

fn rgb8() -> Image {
    Image::from_packed(2, 1, PixelFormat::Rgb8, vec![1, 2, 3, 253, 254, 255]).unwrap()
}
fn floats(format: PixelFormat, values: &[f32]) -> Image {
    Image::from_packed(
        1,
        1,
        format,
        values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    )
    .unwrap()
}

#[test]
fn padding_uses_exact_sixteen_bit_fill_and_reports_stale_geometry_metadata() {
    let pixels = [1u16, 2, 3, 40000, 50000, 60000];
    let source = Image::from_packed(
        2,
        1,
        PixelFormat::Rgb16,
        pixels.into_iter().flat_map(u16::to_le_bytes).collect(),
    )
    .unwrap()
    .with_metadata(ImageMetadata {
        icc_profile: Some(moxcms::ColorProfile::new_srgb().encode().unwrap()),
        exif: Some(vec![5, 6, 7]),
        color: ColorDescription {
            primaries: Some("bt709".into()),
            transfer: Some("srgb".into()),
            matrix: Some("rgb".into()),
            range: Some("full".into()),
            chroma_siting: None,
        },
        ..Default::default()
    });
    let mut spec = PadSpec::new(1, 1, 2, 1, PixelFill::Rgb16([101, 202, 303]));
    spec.max_working_bytes = pad_working_bytes(&source, &spec).unwrap();
    let output = pad(&source, spec).unwrap();
    assert_eq!((output.image.width(), output.image.height()), (5, 3));
    let samples = output
        .image
        .packed_bytes()
        .unwrap()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|v| u16::from_le_bytes(*v))
        .collect::<Vec<_>>();
    for y in 0..3 {
        for x in 0..5 {
            let expected = if y == 1 && (1..=2).contains(&x) {
                &pixels[(x - 1) * 3..x * 3]
            } else {
                &[101, 202, 303][..]
            };
            assert_eq!(&samples[(y * 5 + x) * 3..(y * 5 + x) * 3 + 3], expected);
        }
    }
    assert_eq!(output.metadata_loss, vec![MetadataLoss::ExifAfterGeometry]);
    assert!(output.image.metadata().exif.is_none());
    assert_eq!(
        output.image.metadata().icc_profile,
        source.metadata().icc_profile
    );
    assert_eq!(output.image.metadata().color, source.metadata().color);
    assert_eq!(
        output.image.storage().byte_len()
            + metadata_payload_bytes(output.image.metadata()).unwrap(),
        spec.max_working_bytes
    );
    spec.max_working_bytes -= 1;
    assert_eq!(
        pad(&source, spec).unwrap_err().kind(),
        ErrorKind::ResourceExhausted
    );
}

#[test]
fn pitched_host_rows_copy_only_visible_pixels() {
    let mut bytes = vec![0xee; 25];
    bytes[3..9].copy_from_slice(&[1, 2, 3, 4, 5, 6]);
    bytes[14..20].copy_from_slice(&[7, 8, 9, 10, 11, 12]);
    let source = Image::new(
        ImageLayout {
            width: 2,
            height: 2,
            allocated_width: 2,
            allocated_height: 2,
            format: PixelFormat::Rgb8,
            planes: vec![PlaneLayout {
                offset: 3,
                row_stride: 11,
                row_bytes: 6,
                rows: 2,
            }],
        },
        StorageLease::from_bytes(bytes),
        ImageMetadata::default(),
    )
    .unwrap();
    assert!(source.packed_bytes().is_err());
    let padded = pad(
        &source,
        PadSpec::new(1, 0, 0, 0, PixelFill::Rgb8([90, 91, 92])),
    )
    .unwrap();
    assert_eq!(
        padded.image.packed_bytes().unwrap(),
        [
            90, 91, 92, 1, 2, 3, 4, 5, 6, 90, 91, 92, 7, 8, 9, 10, 11, 12
        ]
    );
    let converted = convert(&source, PixelConvertSpec::new(PixelFormat::Rgb16)).unwrap();
    assert_eq!(
        converted.image.packed_bytes().unwrap(),
        (1u16..=12)
            .flat_map(|v| (v * 257).to_le_bytes())
            .collect::<Vec<_>>()
    );
}

#[test]
fn integer_and_float_roundtrips_are_exact_when_the_information_is_reversible() {
    let source = Image::from_packed(
        256,
        1,
        PixelFormat::Rgb8,
        (0..=255u8).flat_map(|v| [v, v, v]).collect(),
    )
    .unwrap();
    let up = convert(&source, PixelConvertSpec::new(PixelFormat::Rgb16)).unwrap();
    assert!(up.changes.is_empty());
    let down = convert(&up.image, PixelConvertSpec::new(PixelFormat::Rgb8)).unwrap();
    assert_eq!(
        down.image.packed_bytes().unwrap(),
        source.packed_bytes().unwrap()
    );
    let float = convert(
        &source,
        PixelConvertSpec {
            float_range: Some(FloatRange::Unit),
            ..PixelConvertSpec::new(PixelFormat::RgbF32)
        },
    )
    .unwrap();
    let back = convert(
        &float.image,
        PixelConvertSpec {
            float_range: Some(FloatRange::Unit),
            ..PixelConvertSpec::new(PixelFormat::Rgb8)
        },
    )
    .unwrap();
    assert_eq!(
        back.image.packed_bytes().unwrap(),
        source.packed_bytes().unwrap()
    );
    assert!(back.changes.is_empty());
}

#[test]
fn every_sixteen_bit_value_survives_float_storage_without_precision_loss() {
    let values = (0..=u16::MAX)
        .flat_map(|v| [v, v, v])
        .flat_map(u16::to_le_bytes)
        .collect();
    let source = Image::from_packed(65536, 1, PixelFormat::Rgb16, values).unwrap();
    let float = convert(
        &source,
        PixelConvertSpec {
            float_range: Some(FloatRange::Unit),
            ..PixelConvertSpec::new(PixelFormat::RgbF32)
        },
    )
    .unwrap();
    let back = convert(
        &float.image,
        PixelConvertSpec {
            float_range: Some(FloatRange::Unit),
            ..PixelConvertSpec::new(PixelFormat::Rgb16)
        },
    )
    .unwrap();
    assert_eq!(
        back.image.packed_bytes().unwrap(),
        source.packed_bytes().unwrap()
    );
    assert!(float.changes.is_empty());
    assert!(back.changes.is_empty());
}

#[test]
fn precision_loss_rounding_and_clipping_are_explicit() {
    let input = floats(PixelFormat::RgbF32, &[0.5, 1., 0.]);
    let options = PixelConvertSpec {
        float_range: Some(FloatRange::Unit),
        ..PixelConvertSpec::new(PixelFormat::Rgb8)
    };
    assert_eq!(
        convert(&input, options).unwrap_err().kind(),
        ErrorKind::Incompatible
    );
    let output = convert(
        &input,
        PixelConvertSpec {
            quantization: Quantization::NearestTiesEven,
            ..options
        },
    )
    .unwrap();
    assert_eq!(output.image.packed_bytes().unwrap(), [128, 255, 0]);
    assert_eq!(output.changes, vec![PixelChange::PrecisionQuantized]);
    let extended = floats(PixelFormat::RgbF32, &[0.5, 1.5, -0.25]);
    let options = PixelConvertSpec {
        float_range: Some(FloatRange::Extended),
        quantization: Quantization::NearestTiesEven,
        ..PixelConvertSpec::new(PixelFormat::Rgb8)
    };
    assert!(convert(&extended, options).is_err());
    let output = convert(
        &extended,
        PixelConvertSpec {
            gamut: GamutPolicy::Clip,
            ..options
        },
    )
    .unwrap();
    assert_eq!(output.image.packed_bytes().unwrap(), [128, 255, 0]);
    assert_eq!(
        output.changes,
        vec![PixelChange::PrecisionQuantized, PixelChange::GamutClipped]
    );
    let preserved = convert(
        &extended,
        PixelConvertSpec {
            float_range: Some(FloatRange::Extended),
            ..PixelConvertSpec::new(PixelFormat::RgbF32)
        },
    )
    .unwrap();
    assert_eq!(
        preserved.image.packed_bytes().unwrap(),
        extended.packed_bytes().unwrap()
    );
}

#[test]
fn alpha_add_drop_and_discard_have_distinct_contracts() {
    let source = rgb8();
    assert!(convert(&source, PixelConvertSpec::new(PixelFormat::Rgba8)).is_err());
    let output = convert(
        &source,
        PixelConvertSpec {
            alpha: AlphaChange::AddOpaque,
            ..PixelConvertSpec::new(PixelFormat::Rgba8)
        },
    )
    .unwrap();
    assert_eq!(
        output.image.packed_bytes().unwrap(),
        [1, 2, 3, 255, 253, 254, 255, 255]
    );
    let drop = PixelConvertSpec {
        alpha: AlphaChange::DropIfOpaque,
        input_alpha: Some(AlphaMode::Straight),
        ..PixelConvertSpec::new(PixelFormat::Rgb8)
    };
    assert_eq!(
        convert(&output.image, drop)
            .unwrap()
            .image
            .packed_bytes()
            .unwrap(),
        source.packed_bytes().unwrap()
    );
    let transparent = Image::from_packed(1, 1, PixelFormat::Rgba8, vec![100, 50, 25, 0]).unwrap();
    assert!(convert(&transparent, drop).is_err());
    let discarded = convert(
        &transparent,
        PixelConvertSpec {
            alpha: AlphaChange::DiscardStraight,
            ..drop
        },
    )
    .unwrap();
    assert_eq!(discarded.image.packed_bytes().unwrap(), [100, 50, 25]);
    assert_eq!(discarded.changes, vec![PixelChange::StraightAlphaDiscarded]);
    assert!(
        convert(
            &transparent,
            PixelConvertSpec {
                input_alpha: Some(AlphaMode::Premultiplied),
                ..drop
            }
        )
        .is_err()
    );
}

#[test]
fn unsupported_profiles_formats_nonfinite_values_and_invalid_alpha_reject() {
    let gray = Image::from_packed(1, 1, PixelFormat::Gray8, vec![2]).unwrap();
    assert_eq!(
        pad(&gray, PadSpec::uniform(1, PixelFill::Rgb8([0; 3])))
            .unwrap_err()
            .kind(),
        ErrorKind::Unsupported
    );
    assert!(convert(&rgb8(), PixelConvertSpec::new(PixelFormat::Gray8)).is_err());
    let source = rgb8().with_metadata(ImageMetadata {
        icc_profile: Some(
            moxcms::ColorProfile::new_gray_with_gamma(2.2)
                .encode()
                .unwrap(),
        ),
        ..Default::default()
    });
    assert_eq!(
        pad(&source, PadSpec::uniform(1, PixelFill::Rgb8([0; 3])))
            .unwrap_err()
            .kind(),
        ErrorKind::Incompatible
    );
    assert!(convert(&source, PixelConvertSpec::new(PixelFormat::Rgb16)).is_err());
    assert!(
        pad(
            &floats(PixelFormat::RgbF32, &[0., 1., 0.]),
            PadSpec::uniform(1, PixelFill::RgbF32([f32::NAN, 0., 0.]))
        )
        .is_err()
    );
    assert!(
        convert(
            &floats(PixelFormat::RgbF32, &[f32::INFINITY, 0., 0.]),
            PixelConvertSpec {
                float_range: Some(FloatRange::Extended),
                ..PixelConvertSpec::new(PixelFormat::RgbF32)
            }
        )
        .is_err()
    );
    let invalid = floats(PixelFormat::RgbaF32, &[0., 0., 0., 1.1]);
    assert!(pad(&invalid, PadSpec::uniform(1, PixelFill::RgbaF32([0.; 4]))).is_err());
    assert!(
        convert(
            &invalid,
            PixelConvertSpec {
                input_alpha: Some(AlphaMode::Straight),
                float_range: Some(FloatRange::Extended),
                ..PixelConvertSpec::new(PixelFormat::RgbaF32)
            }
        )
        .is_err()
    );
}

#[test]
fn geometry_fill_working_bounds_and_exif_conversion_are_checked() {
    let source = rgb8().with_metadata(ImageMetadata {
        exif: Some(vec![1, 2, 3]),
        ..Default::default()
    });
    assert!(pad(&source, PadSpec::uniform(1, PixelFill::Rgba8([0; 4]))).is_err());
    assert_eq!(
        pad_working_bytes(
            &source,
            &PadSpec::new(u32::MAX, 0, 0, 0, PixelFill::Rgb8([0; 3]))
        )
        .unwrap_err()
        .kind(),
        ErrorKind::Overflow
    );
    let mut spec = PixelConvertSpec::new(PixelFormat::Rgb16);
    spec.max_working_bytes = convert_working_bytes(&source, &spec).unwrap();
    let output = convert(&source, spec).unwrap();
    assert!(output.image.metadata().exif.is_none());
    assert_eq!(
        output.metadata_loss,
        vec![MetadataLoss::ExifAfterComponentTransform]
    );
    assert_eq!(
        output.image.storage().byte_len()
            + metadata_payload_bytes(output.image.metadata()).unwrap(),
        spec.max_working_bytes
    );
    spec.max_working_bytes -= 1;
    assert_eq!(
        convert(&source, spec).unwrap_err().kind(),
        ErrorKind::ResourceExhausted
    );
    let same = convert(&source, PixelConvertSpec::new(PixelFormat::Rgb8)).unwrap();
    assert_eq!(same.image.metadata().exif, source.metadata().exif);
    assert!(same.metadata_loss.is_empty());
}
