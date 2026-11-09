//! Generated-fixture tests of native image pixels, metadata, and limits.

use media_backend_native::image::{
    DecodeLimits, EncodeOptions, Filter, Fit, ImageFormat, MetadataLoss, apply_orientation, crop,
    decode, encode, probe, resize,
};
use media_core::media::{Image, ImageMetadata, Orientation, PixelFormat};

fn packed(width: u32, height: u32, format: PixelFormat, bytes: Vec<u8>) -> Image {
    Image::from_packed(width, height, format, bytes).unwrap()
}

fn encoded(image: &Image, format: ImageFormat) -> Vec<u8> {
    encode(image, EncodeOptions::new(format)).unwrap().bytes
}

fn minimal_exif(orientation: u16) -> Vec<u8> {
    let mut exif = vec![
        b'I', b'I', 42, 0, 8, 0, 0, 0, 1, 0, 0x12, 0x01, 3, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ];
    exif[18..20].copy_from_slice(&orientation.to_le_bytes());
    exif
}

#[test]
fn png_roundtrip_retains_rgb_pixels() {
    let original = packed(
        2,
        2,
        PixelFormat::Rgb8,
        vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 128, 64, 32],
    );
    let bytes = encoded(&original, ImageFormat::Png);
    let info = probe(&bytes, DecodeLimits::default()).unwrap();
    assert_eq!((info.width, info.height), (2, 2));
    assert_eq!(info.format, ImageFormat::Png);
    assert_eq!(info.pixel_format, PixelFormat::Rgb8);
    assert!(info.bytes_read > 0);
    assert!(info.bytes_read <= DecodeLimits::default().max_probe_bytes);
    let decoded = decode(&bytes, DecodeLimits::default()).unwrap();
    assert_eq!(
        decoded.packed_bytes().unwrap(),
        original.packed_bytes().unwrap()
    );
}

#[test]
fn sixteen_bit_png_retains_little_endian_samples() {
    let values = [0u16, 1, 255, 256, 30_000, 65_535];
    let original = packed(
        3,
        2,
        PixelFormat::Gray16,
        values.into_iter().flat_map(u16::to_le_bytes).collect(),
    );
    let decoded = decode(
        &encoded(&original, ImageFormat::Png),
        DecodeLimits::default(),
    )
    .unwrap();
    assert_eq!(decoded.pixel_format(), PixelFormat::Gray16);
    assert_eq!(
        decoded.packed_bytes().unwrap(),
        original.packed_bytes().unwrap()
    );
    let scaled = resize(&decoded, Fit::inside(6, 4), Filter::Nearest).unwrap();
    assert_eq!(scaled.image.pixel_format(), PixelFormat::Gray16);
    assert_eq!((scaled.image.width(), scaled.image.height()), (6, 4));
    assert_eq!(&scaled.image.packed_bytes().unwrap()[..4], &[0, 0, 0, 0]);
}

#[test]
fn jpeg_roundtrip_has_correct_geometry_and_declared_loss() {
    let original = packed(8, 8, PixelFormat::Rgb8, [45u8, 100, 180].repeat(64));
    let mut options = EncodeOptions::new(ImageFormat::Jpeg);
    options.jpeg_quality = 100;
    let result = encode(&original, options).unwrap();
    assert!(result.metadata_loss.is_empty());
    let decoded = decode(&result.bytes, DecodeLimits::default()).unwrap();
    assert_eq!((decoded.width(), decoded.height()), (8, 8));
    assert_eq!(decoded.pixel_format(), PixelFormat::Rgb8);
    for (actual, expected) in decoded
        .packed_bytes()
        .unwrap()
        .iter()
        .zip(original.packed_bytes().unwrap())
    {
        assert!((i16::from(*actual) - i16::from(*expected)).abs() <= 3);
    }
}

#[test]
fn webp_lossless_retains_rgba() {
    let original = packed(
        2,
        2,
        PixelFormat::Rgba8,
        vec![255, 0, 0, 255, 0, 255, 0, 128, 0, 0, 255, 255, 0, 0, 0, 255],
    );
    let decoded = decode(
        &encoded(&original, ImageFormat::WebP),
        DecodeLimits::default(),
    )
    .unwrap();
    assert_eq!(decoded.pixel_format(), PixelFormat::Rgba8);
    assert_eq!(
        decoded.packed_bytes().unwrap(),
        original.packed_bytes().unwrap()
    );
}

#[test]
fn metadata_only_probe_does_not_decompress_corrupt_pixels() {
    let source = packed(4, 4, PixelFormat::Rgb8, vec![123; 48]);
    let mut bytes = encoded(&source, ImageFormat::Png);
    let index = bytes.windows(4).position(|chunk| chunk == b"IDAT").unwrap();
    // Keep headers and dimensions intact but corrupt the compressed pixel stream.
    bytes[index + 4] ^= 0xff;
    let info = probe(&bytes, DecodeLimits::default()).unwrap();
    assert_eq!((info.width, info.height), (4, 4));
    assert!(decode(&bytes, DecodeLimits::default()).is_err());
}

#[test]
fn probe_and_decode_enforce_different_explicit_budgets() {
    let source = packed(2, 2, PixelFormat::Rgb8, vec![0; 12]);
    let bytes = encoded(&source, ImageFormat::Png);
    let low_probe = DecodeLimits {
        max_probe_bytes: 8,
        ..DecodeLimits::default()
    };
    assert!(probe(&bytes, low_probe).is_err());
    assert!(decode(&bytes, low_probe).is_ok());
    for limits in [
        DecodeLimits {
            max_width: 1,
            ..DecodeLimits::default()
        },
        DecodeLimits {
            max_height: 1,
            ..DecodeLimits::default()
        },
        DecodeLimits {
            max_pixels: 3,
            ..DecodeLimits::default()
        },
        DecodeLimits {
            max_decoded_bytes: 11,
            ..DecodeLimits::default()
        },
        DecodeLimits {
            max_encoded_bytes: 1,
            ..DecodeLimits::default()
        },
    ] {
        assert!(probe(&bytes, limits).is_err());
        assert!(decode(&bytes, limits).is_err());
    }
}

#[test]
fn prepared_image_reuses_inspected_codec_and_releases_only_metadata_read_cap_for_pixels() {
    let mut seed = 123_u32;
    let values = (0..256 * 128 * 3)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed as u8
        })
        .collect();
    let original = packed(256, 128, PixelFormat::Rgb8, values);
    let bytes = encoded(&original, ImageFormat::Png);
    let info = probe(&bytes, DecodeLimits::default()).unwrap();
    assert!(info.bytes_read < bytes.len());
    let prepared = media_backend_native::image::prepare(
        &bytes,
        DecodeLimits {
            max_probe_bytes: info.bytes_read,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(prepared.info().bytes_read, info.bytes_read);
    assert_eq!(prepared.info().pixel_format, original.pixel_format());
    let decoded = prepared.decode().unwrap();
    assert_eq!(
        decoded.packed_bytes().unwrap(),
        original.packed_bytes().unwrap()
    );
    assert!(
        media_backend_native::image::prepare(
            &bytes,
            DecodeLimits {
                max_probe_bytes: 1,
                ..Default::default()
            }
        )
        .is_err()
    );
}

#[test]
fn prepared_image_preserves_high_precision_and_metadata_without_decompression_during_inspection() {
    let original = packed(
        3,
        2,
        PixelFormat::Gray16,
        [0_u16, 1, 255, 256, 30000, 65535]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect(),
    );
    let mut bytes = Vec::new();
    use image::ImageEncoder;
    let mut encoder = image::codecs::png::PngEncoder::new(&mut bytes);
    encoder.set_exif_metadata(minimal_exif(6)).unwrap();
    encoder
        .write_image(
            original.packed_bytes().unwrap(),
            3,
            2,
            image::ExtendedColorType::L16,
        )
        .unwrap();
    let prepared = media_backend_native::image::prepare(&bytes, DecodeLimits::default()).unwrap();
    assert_eq!(prepared.info().metadata.orientation, Orientation::Rotate90);
    assert_eq!(prepared.info().pixel_format, PixelFormat::Gray16);
    let decoded = prepared.decode().unwrap();
    let reference = decode(&bytes, DecodeLimits::default()).unwrap();
    assert_eq!(
        decoded.packed_bytes().unwrap(),
        reference.packed_bytes().unwrap()
    );
    assert_eq!(decoded.metadata(), reference.metadata());
    let idat = bytes.windows(4).position(|v| v == b"IDAT").unwrap();
    bytes[idat + 4] ^= 0xff;
    let prepared = media_backend_native::image::prepare(&bytes, DecodeLimits::default()).unwrap();
    assert_eq!(prepared.info().width, 3);
    assert!(prepared.decode().is_err());
}

#[test]
fn all_eight_orientations_transform_pixels_and_geometry() {
    let cases: &[(Orientation, u32, u32, &[u8])] = &[
        (Orientation::Normal, 2, 3, &[1, 2, 3, 4, 5, 6]),
        (Orientation::MirrorHorizontal, 2, 3, &[2, 1, 4, 3, 6, 5]),
        (Orientation::Rotate180, 2, 3, &[6, 5, 4, 3, 2, 1]),
        (Orientation::MirrorVertical, 2, 3, &[5, 6, 3, 4, 1, 2]),
        (Orientation::Transpose, 3, 2, &[1, 3, 5, 2, 4, 6]),
        (Orientation::Rotate90, 3, 2, &[5, 3, 1, 6, 4, 2]),
        (Orientation::Transverse, 3, 2, &[6, 4, 2, 5, 3, 1]),
        (Orientation::Rotate270, 3, 2, &[2, 4, 6, 1, 3, 5]),
    ];
    for (orientation, width, height, expected) in cases {
        let source =
            packed(2, 3, PixelFormat::Gray8, vec![1, 2, 3, 4, 5, 6]).with_metadata(ImageMetadata {
                orientation: *orientation,
                ..ImageMetadata::default()
            });
        let result = apply_orientation(&source).unwrap();
        assert_eq!(
            (result.image.width(), result.image.height()),
            (*width, *height)
        );
        assert_eq!(
            result.image.packed_bytes().unwrap(),
            *expected,
            "{orientation:?}"
        );
        assert_eq!(result.image.metadata().orientation, Orientation::Normal);
    }
}

#[test]
fn exif_orientation_is_discovered_but_not_implicitly_applied() {
    let source = packed(2, 3, PixelFormat::Gray8, vec![1, 2, 3, 4, 5, 6]);
    // Use the codec directly to create a genuine stored orientation fixture.
    use image::ImageEncoder;
    let mut bytes = Vec::new();
    let mut encoder = image::codecs::png::PngEncoder::new(&mut bytes);
    encoder.set_exif_metadata(minimal_exif(6)).unwrap();
    encoder
        .write_image(
            source.packed_bytes().unwrap(),
            2,
            3,
            image::ExtendedColorType::L8,
        )
        .unwrap();
    let decoded = decode(&bytes, DecodeLimits::default()).unwrap();
    assert_eq!((decoded.width(), decoded.height()), (2, 3));
    assert_eq!(decoded.metadata().orientation, Orientation::Rotate90);
    assert!(encode(&decoded, EncodeOptions::new(ImageFormat::Png)).is_err());
    let result = apply_orientation(&decoded).unwrap();
    assert_eq!((result.image.width(), result.image.height()), (3, 2));
    assert_eq!(result.metadata_loss, vec![MetadataLoss::ExifAfterGeometry]);
    assert!(result.image.metadata().exif.is_none());
    let reloaded = decode(
        &encoded(&result.image, ImageFormat::Png),
        DecodeLimits::default(),
    )
    .unwrap();
    assert_eq!(reloaded.metadata().orientation, Orientation::Normal);
    assert_eq!(
        reloaded.packed_bytes().unwrap(),
        result.image.packed_bytes().unwrap()
    );
}

#[test]
fn geometry_changes_report_exif_loss_and_preserve_icc_in_memory() {
    let original = packed(4, 2, PixelFormat::Gray8, vec![1, 2, 3, 4, 5, 6, 7, 8]).with_metadata(
        ImageMetadata {
            exif: Some(minimal_exif(1)),
            icc_profile: Some(vec![11, 22, 33]),
            ..ImageMetadata::default()
        },
    );
    let resized = resize(&original, Fit::inside(2, 2), Filter::Nearest).unwrap();
    assert_eq!((resized.image.width(), resized.image.height()), (2, 1));
    assert_eq!(resized.metadata_loss, vec![MetadataLoss::ExifAfterGeometry]);
    assert_eq!(
        resized.image.metadata().icc_profile.as_deref(),
        Some(&[11, 22, 33][..])
    );
    let cropped = crop(&original, 1, 0, 2, 2).unwrap();
    assert_eq!(cropped.image.packed_bytes().unwrap(), &[2, 3, 6, 7]);
    assert_eq!(cropped.metadata_loss, vec![MetadataLoss::ExifAfterGeometry]);
}

#[test]
fn cover_center_crops_and_invalid_geometry_fails_before_allocation() {
    let original = packed(4, 2, PixelFormat::Gray8, vec![1, 2, 3, 4, 5, 6, 7, 8]);
    let result = resize(&original, Fit::cover(2, 2), Filter::Nearest).unwrap();
    assert_eq!(result.image.packed_bytes().unwrap(), &[2, 3, 6, 7]);
    assert!(resize(&original, Fit::inside(0, 2), Filter::Nearest).is_err());
    assert!(resize(&original, Fit::cover(u32::MAX, u32::MAX), Filter::Nearest).is_err());
    assert!(crop(&original, u32::MAX, 0, 2, 1).is_err());
    assert!(crop(&original, 3, 0, 2, 1).is_err());
    assert!(crop(&original, 0, 0, 0, 1).is_err());
    let thin = packed(1, 32_768, PixelFormat::Gray8, vec![0; 32_768]);
    assert!(resize(&thin, Fit::cover(32_768, 1), Filter::Nearest).is_err());
}

#[test]
fn encoding_requires_explicit_alpha_and_bit_depth_conversion() {
    let rgba = packed(1, 1, PixelFormat::Rgba8, vec![10, 20, 30, 40]);
    assert!(encode(&rgba, EncodeOptions::new(ImageFormat::Jpeg)).is_err());
    let rgb16 = packed(1, 1, PixelFormat::Rgb16, vec![0; 6]);
    assert!(encode(&rgb16, EncodeOptions::new(ImageFormat::WebP)).is_err());
    let rgbf32 = packed(1, 1, PixelFormat::RgbF32, vec![0; 12]);
    assert!(encode(&rgbf32, EncodeOptions::new(ImageFormat::Png)).is_err());
    let mut options = EncodeOptions::new(ImageFormat::Png);
    options.max_output_bytes = 8;
    assert!(encode(&rgba, options).is_err());
}

#[test]
fn metadata_limit_is_enforced() {
    let source = packed(1, 1, PixelFormat::Gray8, vec![0]).with_metadata(ImageMetadata {
        exif: Some(minimal_exif(1)),
        ..ImageMetadata::default()
    });
    let bytes = encoded(&source, ImageFormat::Png);
    let limited = DecodeLimits {
        max_metadata_bytes: 0,
        ..DecodeLimits::default()
    };
    assert!(probe(&bytes, limited).is_err());
    assert!(decode(&bytes, limited).is_err());
}

#[test]
fn still_api_rejects_animation_containers() {
    assert!(decode(b"GIF89a", DecodeLimits::default()).is_err());
    assert!(decode(b"II*\0", DecodeLimits::default()).is_err());
    let mut apng = b"\x89PNG\r\n\x1a\n".to_vec();
    apng.extend_from_slice(&8u32.to_be_bytes());
    apng.extend_from_slice(b"acTL");
    apng.extend_from_slice(&[0; 12]);
    let error = decode(&apng, DecodeLimits::default())
        .unwrap_err()
        .to_string();
    assert!(error.contains("animated PNG"));
    let webp = b"RIFF\x0c\0\0\0WEBPANIM\0\0\0\0";
    let error = decode(webp, DecodeLimits::default())
        .unwrap_err()
        .to_string();
    assert!(error.contains("animated WebP"));
}

#[test]
fn unprofiled_color_declarations_are_never_silently_discarded_by_encoders() {
    use media_backend_native::image::MetadataPolicy;
    use media_core::media::ColorDescription;
    let image = packed(1, 1, PixelFormat::Rgb8, vec![255, 0, 0]).with_metadata(ImageMetadata {
        color: ColorDescription {
            primaries: Some("bt709".into()),
            transfer: Some("iec61966-2-1".into()),
            matrix: Some("rgb".into()),
            range: Some("pc".into()),
            chroma_siting: None,
        },
        ..Default::default()
    });
    for format in [ImageFormat::Png, ImageFormat::Jpeg, ImageFormat::WebP] {
        assert!(
            encode(&image, EncodeOptions::new(format))
                .unwrap_err()
                .to_string()
                .contains("named color")
        );
        let mut options = EncodeOptions::new(format);
        options.metadata_policy = MetadataPolicy::AllowLoss;
        let result = encode(&image, options).unwrap();
        assert!(
            result
                .metadata_loss
                .contains(&MetadataLoss::ColorDescription)
        );
    }
}
