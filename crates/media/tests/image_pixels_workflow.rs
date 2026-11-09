//! Facade padding/conversion, plan output representation and lease accounting.
#![cfg(feature = "native")]
use media::{backend::native::image as codec, io::source::Source, prelude::*};

fn source(rgba: bool) -> Source {
    let image = Image::from_packed(
        2,
        1,
        if rgba {
            PixelFormat::Rgba8
        } else {
            PixelFormat::Rgb8
        },
        if rgba {
            vec![255, 0, 0, 0, 0, 255, 0, 255]
        } else {
            vec![1, 2, 3, 4, 5, 6]
        },
    )
    .unwrap();
    Source::memory(
        codec::encode(&image, EncodeOptions::new(ImageFormat::Png))
            .unwrap()
            .bytes,
    )
}

#[test]
fn fluent_padding_and_exact_pixel_conversion_are_real_and_reusable() {
    let session = Session::default();
    let request = session
        .from_source(source(false))
        .image()
        .unwrap()
        .pad(PadSpec::new(1, 1, 0, 0, PixelFill::Rgb8([10, 20, 30])))
        .convert_pixels(PixelConvertSpec::new(PixelFormat::Rgb16));
    let plan = request.compile().unwrap();
    assert_eq!(
        plan.description().steps.last().unwrap().format,
        Some(PixelFormat::Rgb16)
    );
    let first = plan.execute().unwrap();
    assert_eq!((first.image.width(), first.image.height()), (3, 2));
    let samples = first
        .image
        .packed_bytes()
        .unwrap()
        .chunks_exact(2)
        .map(|v| u16::from_le_bytes(v.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(
        &samples[9..],
        [2570, 5140, 7710, 257, 514, 771, 1028, 1285, 1542]
    );
    let second = plan.execute().unwrap();
    assert_eq!(
        first.image.packed_bytes().unwrap(),
        second.image.packed_bytes().unwrap()
    );
    assert!(
        !first
            .image
            .storage()
            .shares_allocation(second.image.storage())
    );
    assert_eq!(
        session.live_host_bytes(),
        first.image.storage().byte_len() + second.image.storage().byte_len()
    );
    drop(first);
    drop(second);
    assert_eq!(session.live_host_bytes(), 0);
}

#[test]
fn explicit_straight_alpha_discard_enables_rgb_without_compositing() {
    let session = Session::default();
    let asset = session.from_source(source(true));
    let base = PixelConvertSpec {
        target: PixelFormat::Rgb8,
        alpha: AlphaChange::DropIfOpaque,
        input_alpha: Some(AlphaMode::Straight),
        ..PixelConvertSpec::new(PixelFormat::Rgb8)
    };
    assert!(asset.image().unwrap().convert_pixels(base).read().is_err());
    assert_eq!(session.live_host_bytes(), 0);
    let output = asset
        .image()
        .unwrap()
        .convert_pixels(PixelConvertSpec {
            alpha: AlphaChange::DiscardStraight,
            ..base
        })
        .read_with_report()
        .unwrap();
    assert_eq!(output.image.packed_bytes().unwrap(), [255, 0, 0, 0, 255, 0]);
    assert!(
        output
            .report
            .warnings
            .iter()
            .any(|v| v.contains("StraightAlphaDiscarded"))
    );
    drop(output);
    assert_eq!(session.live_host_bytes(), 0);
    let encoded = asset
        .image()
        .unwrap()
        .convert_pixels(PixelConvertSpec {
            alpha: AlphaChange::DiscardStraight,
            ..base
        })
        .write_with_options(
            media::io::sink::Sink::memory(),
            EncodeOptions::new(ImageFormat::Jpeg),
        )
        .unwrap();
    let decoded = codec::decode(&encoded.output.into_bytes().unwrap(), Default::default()).unwrap();
    assert_eq!(decoded.pixel_format(), PixelFormat::Rgb8);
    assert_eq!((decoded.width(), decoded.height()), (2, 1));
    assert_eq!(session.live_host_bytes(), 0);
}

#[test]
fn pixel_operations_retain_icc_and_color_string_payload_until_last_alias_drops() {
    let session = Session::default();
    let output = session
        .from_source(source(false))
        .image()
        .unwrap()
        .convert_color(ColorConvertSpec {
            missing_color: Some(ColorSpace::Srgb),
            target: Some(ColorSpace::Srgb),
            ..ColorConvertSpec::default()
        })
        .pad(PadSpec::uniform(1, PixelFill::Rgb8([0; 3])))
        .convert_pixels(PixelConvertSpec::new(PixelFormat::Rgb16))
        .read()
        .unwrap();
    assert!(output.metadata().icc_profile.is_some());
    assert_eq!(output.metadata().color.transfer.as_deref(), Some("srgb"));
    let expected =
        output.storage().byte_len() + codec::metadata_payload_bytes(output.metadata()).unwrap();
    assert_eq!(session.live_host_bytes(), expected);
    let alias = output.clone();
    drop(output);
    assert_eq!(session.live_host_bytes(), expected);
    drop(alias);
    assert_eq!(session.live_host_bytes(), 0);
}

#[test]
fn new_payload_budgets_and_float_rules_fail_without_retained_allocations() {
    let session = Session::default();
    let asset = session.from_source(source(false));
    let spec = PadSpec {
        max_working_bytes: 1,
        ..PadSpec::uniform(1, PixelFill::Rgb8([0; 3]))
    };
    assert_eq!(
        asset.image().unwrap().pad(spec).read().unwrap_err().kind(),
        ErrorKind::ResourceExhausted
    );
    assert_eq!(session.live_host_bytes(), 0);
    assert!(
        asset
            .image()
            .unwrap()
            .convert_pixels(PixelConvertSpec::new(PixelFormat::RgbF32))
            .read()
            .is_err()
    );
    let image = asset
        .image()
        .unwrap()
        .convert_pixels(PixelConvertSpec {
            float_range: Some(FloatRange::Unit),
            ..PixelConvertSpec::new(PixelFormat::RgbF32)
        })
        .read()
        .unwrap();
    assert_eq!(image.pixel_format(), PixelFormat::RgbF32);
    assert_eq!(session.live_host_bytes(), 24);
    drop(image);
    assert_eq!(session.live_host_bytes(), 0);
}
