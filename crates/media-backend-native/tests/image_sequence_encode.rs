//! Generated animation and multipage round trips with exact timing and pixels.
use media_backend_native::{
    image::MetadataPolicy,
    image_color::AlphaMode,
    image_sequence::{self, Animation, DisplayFrame, ImagePage, LoopCount, SequenceLimits},
    image_sequence_encode::{
        self as encode, AnimationEncodeOptions, AnimationFormat, TiffEncodeOptions,
    },
};
use media_core::{
    error::ErrorKind,
    media::{ColorDescription, Image, ImageMetadata, Orientation, PixelFormat},
    time::Time,
};

fn animation(partial_alpha: bool) -> Animation {
    let durations = if partial_alpha {
        [
            Time::new(1, 24).unwrap(),
            Time::new(1, 60).unwrap(),
            Time::ZERO,
        ]
    } else {
        [Time::from_millis(10), Time::from_millis(20), Time::ZERO]
    };
    let pixels = [
        vec![
            255,
            0,
            0,
            255,
            0,
            255,
            0,
            if partial_alpha { 128 } else { 255 },
        ],
        vec![0, 0, 0, 0, 0, 0, 255, 255],
        vec![255, 255, 255, 255, 0, 0, 0, 0],
    ];
    let mut pts = Time::ZERO;
    let frames = pixels
        .into_iter()
        .zip(durations)
        .enumerate()
        .map(|(index, (bytes, duration))| {
            let frame = DisplayFrame {
                image: Image::from_packed(2, 1, PixelFormat::Rgba8, bytes).unwrap(),
                pts,
                duration,
                index: index as u32,
            };
            pts = pts.checked_add(duration).unwrap();
            frame
        })
        .collect();
    Animation {
        frames,
        loop_count: LoopCount::Finite(3),
        total_duration: pts,
    }
}
fn assert_animation(source: &Animation, bytes: &[u8]) {
    let decoded = image_sequence::decode_animation(bytes, SequenceLimits::default()).unwrap();
    assert_eq!(decoded.loop_count, source.loop_count);
    assert_eq!(decoded.total_duration, source.total_duration);
    assert_eq!(decoded.frames.len(), source.frames.len());
    for (actual, expected) in decoded.frames.iter().zip(&source.frames) {
        assert_eq!(
            (actual.pts, actual.duration, actual.index),
            (expected.pts, expected.duration, expected.index)
        );
        assert_eq!(
            actual.image.packed_bytes().unwrap(),
            expected.image.packed_bytes().unwrap()
        );
    }
}
#[test]
fn apng_preserves_partial_alpha_composited_canvases_exact_fractions_and_global_metadata() {
    let mut source = animation(true);
    let profile = moxcms::ColorProfile::new_srgb().encode().unwrap();
    let exif = b"II*\0\x08\0\0\0\0\0\0\0\0\0".to_vec();
    for frame in &mut source.frames {
        frame.image = frame.image.clone().with_metadata(ImageMetadata {
            icc_profile: Some(profile.clone()),
            exif: Some(exif.clone()),
            ..Default::default()
        });
    }
    let output =
        encode::encode_animation(&source, AnimationEncodeOptions::new(AnimationFormat::Apng))
            .unwrap();
    assert!(output.metadata_loss.is_empty());
    assert_eq!(output.count, 3);
    assert_animation(&source, &output.bytes);
    let decoded =
        image_sequence::decode_animation(&output.bytes, SequenceLimits::default()).unwrap();
    assert_eq!(
        decoded.frames[1].image.metadata().icc_profile.as_ref(),
        Some(&profile)
    );
    assert_eq!(
        decoded.frames[1].image.metadata().exif.as_ref(),
        Some(&exif)
    );
    assert!(
        image_sequence::decode_animation(
            &output.bytes[..output.bytes.len() / 2],
            SequenceLimits::default()
        )
        .is_err()
    );
}
#[test]
fn gif_preserves_exact_palette_transparent_clears_and_total_loop_conventions() {
    for loops in [
        LoopCount::Once,
        LoopCount::Infinite,
        LoopCount::Finite(3),
        LoopCount::Finite(65_536),
    ] {
        let mut source = animation(false);
        source.loop_count = loops;
        let output =
            encode::encode_animation(&source, AnimationEncodeOptions::new(AnimationFormat::Gif))
                .unwrap();
        assert_animation(&source, &output.bytes);
        assert!(
            image_sequence::decode_animation(
                &output.bytes[..output.bytes.len() / 2],
                SequenceLimits::default()
            )
            .is_err()
        );
    }
}
#[test]
fn gif_handles_full_exact_opaque_palette_and_fully_transparent_canvas_without_quantization() {
    for pixels in [
        (0..256)
            .flat_map(|value| [value as u8, 0, 0, 255])
            .collect(),
        vec![0; 256 * 4],
    ] {
        let source = Animation {
            frames: vec![DisplayFrame {
                image: Image::from_packed(256, 1, PixelFormat::Rgba8, pixels).unwrap(),
                pts: Time::ZERO,
                duration: Time::from_millis(10),
                index: 0,
            }],
            loop_count: LoopCount::Once,
            total_duration: Time::from_millis(10),
        };
        let output =
            encode::encode_animation(&source, AnimationEncodeOptions::new(AnimationFormat::Gif))
                .unwrap();
        assert_animation(&source, &output.bytes);
    }
}
#[test]
fn apng_retains_explicit_cicp_color_description_without_inventing_a_profile() {
    let mut source = animation(true);
    let color = ColorDescription {
        primaries: Some("bt709".into()),
        transfer: Some("srgb".into()),
        matrix: Some("rgb".into()),
        range: Some("full".into()),
        ..Default::default()
    };
    for frame in &mut source.frames {
        frame.image = frame.image.clone().with_metadata(ImageMetadata {
            color: color.clone(),
            ..Default::default()
        });
    }
    let output =
        encode::encode_animation(&source, AnimationEncodeOptions::new(AnimationFormat::Apng))
            .unwrap();
    assert!(output.metadata_loss.is_empty());
    let decoded =
        image_sequence::decode_animation(&output.bytes, SequenceLimits::default()).unwrap();
    for frame in decoded.frames {
        assert_eq!(frame.image.metadata().color, color);
        assert!(frame.image.metadata().icc_profile.is_none());
    }
}
#[test]
fn unsupported_gif_alpha_palette_precision_and_timing_never_quantize_implicitly() {
    assert!(
        encode::encode_animation(
            &animation(true),
            AnimationEncodeOptions::new(AnimationFormat::Gif)
        )
        .is_err()
    );
    let mut source = animation(false);
    source.frames[0].image =
        Image::from_packed(2, 1, PixelFormat::Rgba8, vec![255, 0, 0, 128, 0, 0, 0, 0]).unwrap();
    assert!(
        encode::encode_animation(&source, AnimationEncodeOptions::new(AnimationFormat::Gif))
            .is_err()
    );
    source.frames[0].image =
        Image::from_packed(2, 1, PixelFormat::Rgba8, vec![1, 0, 0, 0, 0, 0, 0, 0]).unwrap();
    assert!(
        encode::encode_animation(&source, AnimationEncodeOptions::new(AnimationFormat::Gif))
            .is_err()
    );
    let pixels = (0..257)
        .flat_map(|value| [value as u8, (value >> 8) as u8, 0, 255])
        .collect();
    let image = Image::from_packed(257, 1, PixelFormat::Rgba8, pixels).unwrap();
    let source = Animation {
        frames: vec![DisplayFrame {
            image,
            pts: Time::ZERO,
            duration: Time::from_millis(10),
            index: 0,
        }],
        loop_count: LoopCount::Once,
        total_duration: Time::from_millis(10),
    };
    assert!(
        encode::encode_animation(&source, AnimationEncodeOptions::new(AnimationFormat::Gif))
            .is_err()
    );
}
#[test]
fn sequence_timing_geometry_metadata_and_payload_limits_fail_explicitly() {
    let options = AnimationEncodeOptions::new(AnimationFormat::Apng);
    let mut source = animation(false);
    source.frames[1].pts = Time::ZERO;
    assert!(encode::encode_animation(&source, options).is_err());
    let mut source = animation(false);
    source.total_duration = Time::ZERO;
    assert!(encode::encode_animation(&source, options).is_err());
    let mut source = animation(false);
    source.frames[0].duration = Time::new(1, 65_537).unwrap();
    assert!(encode::encode_animation(&source, options).is_err());
    let source = animation(false);
    for kind in 0..4 {
        let mut bounded = options;
        match kind {
            0 => bounded.limits.max_frames = 2,
            1 => bounded.limits.max_input_bytes = 1,
            2 => bounded.limits.max_encoded_bytes = 32,
            _ => bounded.limits.max_working_bytes = 1,
        };
        assert!(encode::encode_animation(&source, bounded).is_err());
    }
    let mut source = animation(false);
    source.frames[1].image = source.frames[1].image.clone().with_metadata(ImageMetadata {
        exif: Some(vec![1, 2, 3]),
        ..Default::default()
    });
    assert!(encode::encode_animation(&source, options).is_err());
    let mut allowed = options;
    allowed.metadata_policy = MetadataPolicy::AllowLoss;
    let encoded = encode::encode_animation(&source, allowed).unwrap();
    assert!(
        encoded
            .metadata_loss
            .iter()
            .any(|loss| loss.contains("distinct metadata"))
    );
}
fn pages() -> Vec<ImagePage> {
    let formats = [
        PixelFormat::Gray8,
        PixelFormat::GrayAlpha8,
        PixelFormat::Gray16,
        PixelFormat::GrayAlpha16,
        PixelFormat::Rgb8,
        PixelFormat::Rgba8,
        PixelFormat::Rgb16,
        PixelFormat::Rgba16,
        PixelFormat::RgbF32,
        PixelFormat::RgbaF32,
    ];
    formats
        .into_iter()
        .enumerate()
        .map(|(index, format)| {
            let count = 2 * format.channels();
            let bytes = match format {
                PixelFormat::Gray16
                | PixelFormat::GrayAlpha16
                | PixelFormat::Rgb16
                | PixelFormat::Rgba16 => (0..count)
                    .flat_map(|value| {
                        [0u16, 1, 255, 256, 32768, 65534, 65535, 258][value].to_le_bytes()
                    })
                    .collect(),
                PixelFormat::RgbF32 | PixelFormat::RgbaF32 => (0..count)
                    .flat_map(|value| {
                        let samples = [
                            -0.5f32,
                            -0.0,
                            f32::MIN_POSITIVE,
                            0.5,
                            1.5,
                            f32::from_bits(0x3f00_0001),
                            2.0,
                            1.0,
                        ];
                        samples[value].to_le_bytes()
                    })
                    .collect(),
                _ => (0..count)
                    .map(|value| [0u8, 255, 1, 127, 128, 254, 2, 253][value])
                    .collect(),
            };
            let alpha = matches!(
                format,
                PixelFormat::GrayAlpha8
                    | PixelFormat::GrayAlpha16
                    | PixelFormat::Rgba8
                    | PixelFormat::Rgba16
                    | PixelFormat::RgbaF32
            )
            .then_some(if index % 2 == 0 {
                AlphaMode::Premultiplied
            } else {
                AlphaMode::Straight
            });
            ImagePage {
                image: Image::from_packed(
                    if index % 2 == 0 { 2 } else { 1 },
                    if index % 2 == 0 { 1 } else { 2 },
                    format,
                    bytes,
                )
                .unwrap()
                .with_metadata(ImageMetadata {
                    orientation: Orientation::Rotate90,
                    ..Default::default()
                }),
                index: index as u32,
                alpha,
            }
        })
        .collect()
}
#[test]
fn every_supported_tiff_page_retains_precision_independent_identity_orientation_and_alpha() {
    let mut source = pages();
    let profile = moxcms::ColorProfile::new_srgb().encode().unwrap();
    source[7].image = source[7].image.clone().with_metadata(ImageMetadata {
        orientation: Orientation::Transverse,
        icc_profile: Some(profile.clone()),
        ..Default::default()
    });
    let output = encode::encode_tiff(&source, TiffEncodeOptions::default()).unwrap();
    assert_eq!(output.count, source.len());
    assert!(output.metadata_loss.is_empty());
    let decoded = image_sequence::decode_pages(&output.bytes, SequenceLimits::default()).unwrap();
    assert_eq!(decoded.pages.len(), source.len());
    for (actual, expected) in decoded.pages.iter().zip(&source) {
        assert_eq!(actual.image.pixel_format(), expected.image.pixel_format());
        assert_eq!(
            (actual.image.width(), actual.image.height()),
            (expected.image.width(), expected.image.height())
        );
        assert_eq!(actual.alpha, expected.alpha);
        assert_eq!(actual.image.metadata(), expected.image.metadata());
        assert_eq!(
            actual.image.packed_bytes().unwrap(),
            expected.image.packed_bytes().unwrap()
        );
    }
    assert!(
        image_sequence::decode_pages(
            &output.bytes[..output.bytes.len() / 2],
            SequenceLimits::default()
        )
        .is_err()
    );
}
#[test]
fn tiff_exif_profiles_alpha_and_seek_output_bounds_are_explicit() {
    let mut source = pages();
    source[0].image = source[0].image.clone().with_metadata(ImageMetadata {
        exif: Some(vec![1, 2, 3]),
        ..Default::default()
    });
    assert!(encode::encode_tiff(&source, TiffEncodeOptions::default()).is_err());
    let mut allowed = TiffEncodeOptions {
        metadata_policy: MetadataPolicy::AllowLoss,
        ..Default::default()
    };
    let output = encode::encode_tiff(&source, allowed).unwrap();
    assert!(
        output
            .metadata_loss
            .iter()
            .any(|loss| loss.contains("EXIF"))
    );
    allowed.limits.max_encoded_bytes = 64;
    assert_eq!(
        encode::encode_tiff(&source, allowed).unwrap_err().kind(),
        ErrorKind::ResourceExhausted
    );
    source[0].image = source[0].image.clone().with_metadata(ImageMetadata {
        icc_profile: Some(moxcms::ColorProfile::new_srgb().encode().unwrap()),
        ..Default::default()
    });
    assert!(encode::encode_tiff(&source, TiffEncodeOptions::default()).is_err());
    source = pages();
    source[1].alpha = None;
    assert!(encode::encode_tiff(&source, TiffEncodeOptions::default()).is_err());
}
