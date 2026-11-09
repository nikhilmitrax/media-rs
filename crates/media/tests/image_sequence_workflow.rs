//! Real animation/page facade, retained budgets, and explicit color workflow checks.
#![cfg(feature = "native")]

use media::{
    backend::native::{
        image as codec,
        image_color::{AlphaMode, ColorConvertSpec, ColorSpace, ResizeDomain, ResizeSpec},
    },
    io::source::Source,
    prelude::*,
};
use std::{borrow::Cow, io::Cursor};

fn gif() -> Vec<u8> {
    let mut bytes = Vec::new();
    {
        let mut encoder =
            gif::Encoder::new(&mut bytes, 2, 1, &[255, 0, 0, 0, 0, 255, 0, 255, 0]).unwrap();
        encoder.set_repeat(gif::Repeat::Finite(2)).unwrap();
        for (left, width, pixels, dispose, delay) in [
            (0, 2, vec![0, 0], gif::DisposalMethod::Keep, 3),
            (0, 1, vec![1], gif::DisposalMethod::Previous, 7),
            (1, 1, vec![2], gif::DisposalMethod::Background, 5),
            (0, 1, vec![1], gif::DisposalMethod::Keep, 2),
        ] {
            encoder
                .write_frame(&gif::Frame {
                    left,
                    width,
                    height: 1,
                    buffer: Cow::Owned(pixels),
                    dispose,
                    delay,
                    ..Default::default()
                })
                .unwrap();
        }
    }
    bytes
}
fn apng() -> Vec<u8> {
    let mut bytes = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut bytes, 1, 1);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.set_source_srgb(png::SrgbRenderingIntent::RelativeColorimetric);
        encoder.set_animated(2, 0).unwrap();
        encoder.set_frame_delay(1, 3).unwrap();
        let mut writer = encoder.write_header().unwrap();
        writer.write_image_data(&[255, 0, 0, 255]).unwrap();
        writer.set_frame_delay(1, 7).unwrap();
        writer.write_image_data(&[0, 0, 255, 255]).unwrap();
        writer.finish().unwrap();
    }
    bytes
}
fn tiff(loss: bool) -> Vec<u8> {
    struct Undefined<'a>(&'a [u8]);
    impl tiff::encoder::TiffValue for Undefined<'_> {
        const BYTE_LEN: u8 = 1;
        const FIELD_TYPE: tiff::tags::Type = tiff::tags::Type::UNDEFINED;
        fn count(&self) -> usize {
            self.0.len()
        }
        fn data(&self) -> Cow<'_, [u8]> {
            Cow::Borrowed(self.0)
        }
    }
    let mut bytes = Cursor::new(Vec::new());
    {
        let mut encoder = tiff::encoder::TiffEncoder::new(&mut bytes).unwrap();
        let mut page = encoder
            .new_image::<tiff::encoder::colortype::RGB16>(2, 1)
            .unwrap();
        page.encoder()
            .write_tag(tiff::tags::Tag::Orientation, 6u16)
            .unwrap();
        page.encoder()
            .write_tag(tiff::tags::Tag::IccProfile, Undefined(&[1_u8, 2, 3, 4]))
            .unwrap();
        if loss {
            page.encoder()
                .write_tag(tiff::tags::Tag::Unknown(700), b"opaque XMP".as_slice())
                .unwrap();
        }
        page.write_data(&[1_u16, 258, 65535, 1234, 5678, 9012])
            .unwrap();
        encoder
            .write_image::<tiff::encoder::colortype::Gray8>(1, 2, &[0, 255])
            .unwrap();
    }
    bytes.into_inner()
}
fn encoded(format: PixelFormat, pixels: Vec<u8>, exif: bool) -> Vec<u8> {
    let mut image = Image::from_packed(2, 1, format, pixels).unwrap();
    if exif {
        let mut metadata = image.metadata().clone();
        metadata.exif = Some(b"II*\0\x08\0\0\0\0\0\0\0\0\0".to_vec());
        image = image.with_metadata(metadata);
    }
    codec::encode(&image, EncodeOptions::new(ImageFormat::Png))
        .unwrap()
        .bytes
}
fn color() -> ColorConvertSpec {
    ColorConvertSpec {
        target: Some(ColorSpace::Srgb),
        missing_color: Some(ColorSpace::Srgb),
        input_alpha: Some(AlphaMode::Straight),
        ..Default::default()
    }
}
fn resize(domain: ResizeDomain) -> ResizeSpec {
    let mut color = color();
    color.target = None;
    ResizeSpec {
        fit: codec::Fit::inside(1, 1),
        filter: codec::Filter::Triangle,
        domain,
        color,
    }
}

#[test]
fn composited_gif_frames_keep_loop_timing_pixels_and_independent_output_lifetimes() {
    let session = Session::builder()
        .limits(Limits::new().host_memory(64 * 1024).in_flight_batches(1))
        .build()
        .unwrap();
    let bytes = gif();
    let expected_bytes = bytes.len();
    let asset = session.from_source(Source::memory(bytes));
    assert!(asset.image().unwrap().read().is_err());
    let result = asset.animation().read().unwrap();
    assert_eq!(result.frames.len(), 4);
    assert_eq!(result.loop_count, LoopCount::Finite(3));
    assert_eq!(result.total_duration, Time::from_millis(170));
    assert_eq!(result.frames[1].pts, Time::from_millis(30));
    assert_eq!(result.frames[1].duration, Time::from_millis(70));
    assert_eq!(
        result.frames[2].image.packed_bytes().unwrap(),
        &[255, 0, 0, 255, 0, 255, 0, 255]
    );
    assert_eq!(
        result.frames[3].image.packed_bytes().unwrap(),
        &[0, 0, 255, 255, 0, 0, 0, 0]
    );
    assert_eq!(result.report.bytes_read, expected_bytes as u64);
    assert_eq!(result.report.output_count, 4);
    assert!(!result.report.hardware_decode);
    assert!(
        result
            .report
            .warnings
            .iter()
            .any(|v| v.contains("unmeasured"))
    );
    let before = session.live_host_bytes();
    let retained = result.frames[1].image.clone();
    assert_eq!(session.live_batches(), 1);
    drop(result);
    assert_eq!(session.live_host_bytes(), before - 3 * 8);
    assert_eq!(session.live_batches(), 1);
    assert_eq!(
        asset.animation().read().unwrap_err().kind(),
        ErrorKind::ResourceExhausted
    );
    assert_eq!(
        retained.packed_bytes().unwrap(),
        &[0, 0, 255, 255, 255, 0, 0, 255]
    );
    drop(retained);
    assert_eq!(session.live_host_bytes(), 0);
    assert_eq!(session.live_batches(), 0);
    assert!(asset.animation().read().is_ok());
}

#[test]
fn apng_uses_exact_fractional_encoded_delays_and_explicit_animation_api() {
    let asset = Session::default().from_source(Source::memory(apng()));
    assert!(asset.image().unwrap().read().is_err());
    let result = asset.animation().read().unwrap();
    assert_eq!(result.loop_count, LoopCount::Infinite);
    assert_eq!(result.frames[0].duration, Time::new(1, 3).unwrap());
    assert_eq!(result.frames[1].duration, Time::new(1, 7).unwrap());
    assert_eq!(result.total_duration, Time::new(10, 21).unwrap());
    assert_eq!(
        result.frames[1].image.packed_bytes().unwrap(),
        &[0, 0, 255, 255]
    );
    assert!(asset.pages().read().is_err());
    let still = Session::default().from_source(Source::memory(encoded(
        PixelFormat::Rgb8,
        vec![0; 6],
        false,
    )));
    assert!(still.animation().read().is_err());
}

#[test]
fn independent_tiff_pages_keep_geometry_precision_orientation_and_metadata_leases() {
    let session = Session::default();
    let asset = session.from_source(Source::memory(tiff(false)));
    assert!(asset.image().unwrap().read().is_err());
    let result = asset.pages().read().unwrap();
    assert_eq!(result.pages.len(), 2);
    let first = &result.pages[0].image;
    let second = &result.pages[1].image;
    assert_eq!(
        (first.width(), first.height(), first.pixel_format()),
        (2, 1, PixelFormat::Rgb16)
    );
    assert_eq!(
        first.metadata().orientation,
        media::core::media::Orientation::Rotate90
    );
    assert_eq!(
        first.metadata().icc_profile.as_deref(),
        Some([1_u8, 2, 3, 4].as_slice())
    );
    assert_eq!(
        first.packed_bytes().unwrap(),
        [1_u16, 258, 65535, 1234, 5678, 9012]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        (second.width(), second.height(), second.pixel_format()),
        (1, 2, PixelFormat::Gray8)
    );
    assert_eq!(second.packed_bytes().unwrap(), &[0, 255]);
    assert_eq!(result.pages[0].alpha, None);
    let before = session.live_host_bytes();
    let retained = first.clone();
    drop(result);
    assert_eq!(session.live_host_bytes(), before - 2);
    assert_eq!(
        retained.metadata().icc_profile.as_deref(),
        Some([1_u8, 2, 3, 4].as_slice())
    );
    drop(retained);
    assert_eq!(session.live_host_bytes(), 0);
    assert_eq!(session.live_batches(), 0);
}

#[test]
fn tiff_metadata_projection_requires_explicit_loss_and_reports_what_was_removed() {
    let session = Session::default();
    let request = session.from_source(Source::memory(tiff(true))).pages();
    assert_eq!(request.read().unwrap_err().kind(), ErrorKind::Incompatible);
    assert_eq!(session.live_host_bytes(), 0);
    assert_eq!(session.live_batches(), 0);
    let result = request.allow_metadata_loss().read().unwrap();
    assert!(result.report.warnings.iter().any(|v| v.contains("XMP")));
    assert_eq!(result.pages.len(), 2);
}

#[test]
fn encoded_pixel_total_and_count_limits_fail_without_leaking_sequence_or_batch_permits() {
    let bytes = gif();
    for session in [
        Session::builder()
            .limits(Limits::new().encoded_bytes(8))
            .build()
            .unwrap(),
        Session::builder()
            .limits(Limits::new().image_pixels(1))
            .build()
            .unwrap(),
        Session::builder()
            .limits(Limits::new().host_memory(1024))
            .build()
            .unwrap(),
    ] {
        assert!(
            session
                .from_source(Source::memory(bytes.clone()))
                .animation()
                .read()
                .is_err()
        );
        assert_eq!(session.live_host_bytes(), 0);
        assert_eq!(session.live_batches(), 0);
    }
    let session = Session::default();
    for limits in [
        SequenceLimits {
            max_frames: 1,
            ..Default::default()
        },
        SequenceLimits {
            max_total_decoded_bytes: 1,
            ..Default::default()
        },
    ] {
        assert!(
            session
                .from_source(Source::memory(bytes.clone()))
                .animation()
                .limits(limits)
                .read()
                .is_err()
        );
        assert_eq!(session.live_host_bytes(), 0);
        assert_eq!(session.live_batches(), 0);
    }
}

#[test]
fn facade_color_resize_declares_linear_light_alpha_and_missing_color_semantics() {
    let asset = Session::default().from_source(Source::memory(encoded(
        PixelFormat::Rgb8,
        vec![0, 0, 0, 255, 255, 255],
        false,
    )));
    assert!(
        asset
            .image()
            .unwrap()
            .resize_color(ResizeSpec {
                color: ColorConvertSpec::default(),
                ..resize(ResizeDomain::LinearLight)
            })
            .read()
            .is_err()
    );
    let linear = asset
        .image()
        .unwrap()
        .resize_color(resize(ResizeDomain::LinearLight))
        .read_with_report()
        .unwrap();
    let stored = asset
        .image()
        .unwrap()
        .resize_color(resize(ResizeDomain::StoredComponents))
        .read()
        .unwrap();
    assert_eq!(linear.image.packed_bytes().unwrap(), &[188, 188, 188]);
    assert_eq!(stored.packed_bytes().unwrap(), &[128, 128, 128]);
    assert_eq!(
        linear.image.metadata().color.transfer.as_deref(),
        Some("srgb")
    );
    assert!(linear.report.warnings.iter().any(|v| v.contains("assumed")));
    let transparent = Session::default().from_source(Source::memory(encoded(
        PixelFormat::Rgba8,
        vec![255, 0, 0, 0, 0, 0, 255, 255],
        false,
    )));
    let output = transparent
        .image()
        .unwrap()
        .resize_color(resize(ResizeDomain::LinearLight))
        .read()
        .unwrap();
    assert_eq!(output.packed_bytes().unwrap(), &[0, 0, 255, 128]);
    let mut no_alpha = resize(ResizeDomain::LinearLight);
    no_alpha.color.input_alpha = None;
    assert!(
        transparent
            .image()
            .unwrap()
            .resize_color(no_alpha)
            .read()
            .is_err()
    );
}

#[test]
fn color_transform_loss_diagnostics_and_retained_icc_bytes_obey_session_lifetime() {
    let session = Session::builder()
        .limits(Limits::new().host_memory(4096))
        .build()
        .unwrap();
    let asset = session.from_source(Source::memory(encoded(
        PixelFormat::Rgb8,
        vec![30, 60, 90, 120, 150, 180],
        true,
    )));
    let request = asset.image().unwrap().convert_color(color());
    assert_eq!(request.read().unwrap_err().kind(), ErrorKind::Incompatible);
    assert_eq!(session.live_host_bytes(), 0);
    let result = request.allow_metadata_loss().read_with_report().unwrap();
    assert!(
        result
            .report
            .warnings
            .iter()
            .any(|v| v.contains("ExifAfterColorTransform"))
    );
    assert!(result.image.metadata().exif.is_none());
    assert!(result.image.metadata().icc_profile.is_some());
    let bytes = result.image.storage().byte_len()
        + media::backend::native::image::metadata_payload_bytes(result.image.metadata()).unwrap();
    assert_eq!(session.live_host_bytes(), bytes);
    let retained = result.image.clone();
    drop(result);
    assert_eq!(session.live_host_bytes(), bytes);
    drop(retained);
    assert_eq!(session.live_host_bytes(), 0);
}

#[test]
fn sequential_animation_adapters_and_float_tiff_pages_remain_explicit_and_bounded() {
    let bytes: std::sync::Arc<[u8]> = gif().into();
    let len = bytes.len() as u64;
    let source = Source::from_reader_factory("sequential GIF", None, Some(len), move || {
        Ok(Cursor::new(bytes.clone()))
    })
    .unwrap();
    assert!(!source.capabilities().unwrap().seekable);
    assert_eq!(
        Session::default()
            .from_source(source)
            .animation()
            .read()
            .unwrap()
            .frames
            .len(),
        4
    );
    let mut bytes = Cursor::new(Vec::new());
    {
        let mut encoder = tiff::encoder::TiffEncoder::new(&mut bytes).unwrap();
        encoder
            .write_image::<tiff::encoder::colortype::RGB32Float>(1, 1, &[0.125_f32, 1.25, -0.25])
            .unwrap();
    }
    let pages = Session::default()
        .from_source(Source::memory(bytes.into_inner()))
        .pages()
        .read()
        .unwrap();
    assert_eq!(pages.pages[0].image.pixel_format(), PixelFormat::RgbF32);
    let values = pages.pages[0]
        .image
        .packed_bytes()
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect::<Vec<_>>();
    assert_eq!(values, vec![0.125, 1.25, -0.25]);
}

#[test]
fn color_working_limit_failure_releases_accounting_and_existing_icc_planning_matches_output() {
    let session = Session::builder()
        .limits(Limits::new().host_memory(4096))
        .build()
        .unwrap();
    let asset = session.from_source(Source::memory(encoded(
        PixelFormat::Rgb8,
        vec![30, 60, 90, 120, 150, 180],
        false,
    )));
    let failed = asset.image().unwrap().convert_color(ColorConvertSpec {
        max_working_bytes: 1,
        ..color()
    });
    assert_eq!(
        failed.read().unwrap_err().kind(),
        ErrorKind::ResourceExhausted
    );
    assert_eq!(session.live_host_bytes(), 0);
    let output = asset
        .image()
        .unwrap()
        .convert_color(color())
        .resize_color(resize(ResizeDomain::LinearLight))
        .read()
        .unwrap();
    let bytes = output.storage().byte_len()
        + media::backend::native::image::metadata_payload_bytes(output.metadata()).unwrap();
    assert_eq!(session.live_host_bytes(), bytes);
    drop(output);
    assert_eq!(session.live_host_bytes(), 0);
}
