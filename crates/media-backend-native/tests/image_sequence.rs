//! Generated animation disposal/blend and multipage precision/limit fixtures.
use media_backend_native::{
    image::{self, DecodeLimits, EncodeOptions, ImageFormat},
    image_color::AlphaMode,
    image_sequence::*,
};
use media_core::{
    error::ErrorKind,
    media::{Image, Orientation, PixelFormat},
    time::Time,
};
use std::{borrow::Cow, io::Cursor};

fn gif_fixture(repeat: bool) -> Vec<u8> {
    let mut bytes = Vec::new();
    {
        let mut encoder =
            gif::Encoder::new(&mut bytes, 2, 1, &[255, 0, 0, 0, 0, 255, 0, 255, 0]).unwrap();
        if repeat {
            encoder.set_repeat(gif::Repeat::Finite(2)).unwrap();
        }
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
                    ..gif::Frame::default()
                })
                .unwrap();
        }
    }
    bytes
}
fn apng_fixture() -> Vec<u8> {
    let mut bytes = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut bytes, 2, 1);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.set_source_srgb(png::SrgbRenderingIntent::RelativeColorimetric);
        encoder.set_animated(4, 2).unwrap();
        encoder.set_frame_delay(1, 10).unwrap();
        let mut writer = encoder.write_header().unwrap();
        writer
            .write_image_data(&[255, 0, 0, 255, 255, 0, 0, 255])
            .unwrap();
        writer.set_frame_dimension(1, 1).unwrap();
        writer.set_frame_position(0, 0).unwrap();
        writer.set_blend_op(png::BlendOp::Over).unwrap();
        writer.set_dispose_op(png::DisposeOp::Previous).unwrap();
        writer.set_frame_delay(1, 5).unwrap();
        writer.write_image_data(&[0, 0, 255, 128]).unwrap();
        writer.set_frame_position(1, 0).unwrap();
        writer.set_blend_op(png::BlendOp::Source).unwrap();
        writer.set_dispose_op(png::DisposeOp::Background).unwrap();
        writer.set_frame_delay(3, 10).unwrap();
        writer.write_image_data(&[0, 255, 0, 255]).unwrap();
        writer.set_frame_position(0, 0).unwrap();
        writer.set_dispose_op(png::DisposeOp::None).unwrap();
        writer.set_frame_delay(1, 10).unwrap();
        writer.write_image_data(&[0, 0, 255, 255]).unwrap();
        writer.finish().unwrap();
    }
    bytes
}
fn webp_fixture() -> Vec<u8> {
    fn chunk(output: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
        output.extend_from_slice(kind);
        output.extend_from_slice(&(data.len() as u32).to_le_bytes());
        output.extend_from_slice(data);
        if !data.len().is_multiple_of(2) {
            output.push(0);
        }
    }
    fn u24(output: &mut Vec<u8>, value: u32) {
        output.extend_from_slice(&value.to_le_bytes()[..3]);
    }
    let mut body = Vec::new();
    chunk(&mut body, b"VP8X", &[0x12, 0, 0, 0, 3, 0, 0, 1, 0, 0]);
    chunk(&mut body, b"ANIM", &[0, 0, 0, 0, 3, 0]);
    for (x, width, rgba, delay, flags) in [
        (0, 4, [255, 0, 0, 255], 100, 2),
        (2, 2, [0, 0, 255, 128], 200, 1),
        (0, 2, [0, 255, 0, 255], 300, 2),
    ] {
        let image = Image::from_packed(
            width,
            2,
            PixelFormat::Rgba8,
            rgba.into_iter()
                .cycle()
                .take(width as usize * 2 * 4)
                .collect(),
        )
        .unwrap();
        let encoded = image::encode(&image, EncodeOptions::new(ImageFormat::WebP))
            .unwrap()
            .bytes;
        let mut frame = Vec::new();
        u24(&mut frame, x / 2);
        u24(&mut frame, 0);
        u24(&mut frame, width - 1);
        u24(&mut frame, 1);
        u24(&mut frame, delay);
        frame.push(flags);
        // Native lossless still encoder supplies the compressed VP8L subchunk.
        let mut offset = 12;
        while offset + 8 <= encoded.len() {
            let size =
                u32::from_le_bytes(encoded[offset + 4..offset + 8].try_into().unwrap()) as usize;
            if &encoded[offset..offset + 4] == b"VP8L" {
                frame.extend_from_slice(&encoded[offset..offset + 8 + size + (size & 1)]);
                break;
            }
            offset += 8 + size + (size & 1);
        }
        chunk(&mut body, b"ANMF", &frame);
    }
    let mut output = b"RIFF".to_vec();
    output.extend_from_slice(&((body.len() + 4) as u32).to_le_bytes());
    output.extend_from_slice(b"WEBP");
    output.extend(body);
    output
}

#[test]
fn gif_previous_and_background_disposal_produce_composited_canvases() {
    let bytes = gif_fixture(true);
    assert!(image::decode(&bytes, DecodeLimits::default()).is_err());
    let output = decode_animation(&bytes, SequenceLimits::default()).unwrap();
    assert_eq!(output.loop_count, LoopCount::Finite(3));
    assert_eq!(output.frames.len(), 4);
    assert_eq!(
        output.frames[0].image.packed_bytes().unwrap(),
        [255, 0, 0, 255, 255, 0, 0, 255]
    );
    assert_eq!(
        output.frames[1].image.packed_bytes().unwrap(),
        [0, 0, 255, 255, 255, 0, 0, 255]
    );
    assert_eq!(
        output.frames[2].image.packed_bytes().unwrap(),
        [255, 0, 0, 255, 0, 255, 0, 255]
    );
    assert_eq!(
        output.frames[3].image.packed_bytes().unwrap(),
        [0, 0, 255, 255, 0, 0, 0, 0]
    );
    assert_eq!(output.frames[1].pts, Time::from_millis(30));
    assert_eq!(output.frames[1].duration, Time::from_millis(70));
    assert_eq!(output.total_duration, Time::from_millis(170));
    assert_eq!(
        decode_animation(&gif_fixture(false), SequenceLimits::default())
            .unwrap()
            .loop_count,
        LoopCount::Once
    );
}

#[test]
fn apng_over_previous_and_background_match_display_pixels_and_rational_delays() {
    let bytes = apng_fixture();
    let output = decode_animation(&bytes, SequenceLimits::default()).unwrap();
    assert_eq!(output.loop_count, LoopCount::Finite(2));
    assert_eq!(output.frames.len(), 4);
    let second = output.frames[1].image.packed_bytes().unwrap();
    assert!(second[0].abs_diff(127) <= 1);
    assert!(second[2].abs_diff(128) <= 1);
    assert_eq!(&second[4..], &[255, 0, 0, 255]);
    assert_eq!(
        output.frames[2].image.packed_bytes().unwrap(),
        [255, 0, 0, 255, 0, 255, 0, 255]
    );
    assert_eq!(
        output.frames[3].image.packed_bytes().unwrap(),
        [0, 0, 255, 255, 0, 0, 0, 0]
    );
    assert_eq!(output.frames[2].pts, Time::from_millis(300));
    assert_eq!(output.total_duration, Time::from_millis(700));
    assert_eq!(
        output.frames[0].image.metadata().color.transfer.as_deref(),
        Some("srgb")
    );
}

#[test]
fn webp_blend_background_and_loop_metadata_are_honored() {
    let bytes = webp_fixture();
    let output = decode_animation(&bytes, SequenceLimits::default()).unwrap();
    assert_eq!(output.loop_count, LoopCount::Finite(3));
    assert_eq!(output.frames.len(), 3);
    let second = output.frames[1].image.packed_bytes().unwrap();
    assert_eq!(&second[..4], &[255, 0, 0, 255]);
    assert!(second[8].abs_diff(127) <= 1);
    assert!(second[10].abs_diff(128) <= 1);
    let third = output.frames[2].image.packed_bytes().unwrap();
    assert_eq!(&third[..4], &[0, 255, 0, 255]);
    assert_eq!(&third[8..12], &[0, 0, 0, 0]);
    assert_eq!(output.total_duration, Time::from_millis(600));
}

#[test]
fn animation_frame_and_retained_byte_budgets_fail_before_collecting_pixels() {
    for bytes in [gif_fixture(true), apng_fixture(), webp_fixture()] {
        let limits = SequenceLimits {
            max_frames: 1,
            ..SequenceLimits::default()
        };
        assert_eq!(
            decode_animation(&bytes, limits).unwrap_err().kind(),
            ErrorKind::ResourceExhausted
        );
        let limits = SequenceLimits {
            max_total_decoded_bytes: 1,
            ..SequenceLimits::default()
        };
        assert_eq!(
            decode_animation(&bytes, limits).unwrap_err().kind(),
            ErrorKind::ResourceExhausted
        );
    }
}

#[test]
fn apng_sixteen_bit_compositing_is_rejected_without_precision_loss() {
    let mut bytes = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut bytes, 1, 1);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Sixteen);
        encoder.set_animated(1, 1).unwrap();
        encoder
            .write_header()
            .unwrap()
            .write_image_data(&[255, 255, 0, 0, 0, 0, 255, 255])
            .unwrap();
    }
    assert_eq!(
        decode_animation(&bytes, SequenceLimits::default())
            .unwrap_err()
            .kind(),
        ErrorKind::Unsupported
    );
}

fn tiff_fixture() -> Vec<u8> {
    let mut bytes = Cursor::new(Vec::new());
    {
        let mut encoder = tiff::encoder::TiffEncoder::new(&mut bytes).unwrap();
        let mut page = encoder
            .new_image::<tiff::encoder::colortype::RGB16>(2, 1)
            .unwrap();
        page.encoder()
            .write_tag(tiff::tags::Tag::Orientation, 6u16)
            .unwrap();
        page.write_data(&[1u16, 258, 65535, 1234, 5678, 9012])
            .unwrap();
        encoder
            .write_image::<tiff::encoder::colortype::Gray8>(1, 2, &[0, 255])
            .unwrap();
    }
    bytes.into_inner()
}
#[test]
fn tiff_pages_keep_independent_geometry_native_depth_and_orientation() {
    let bytes = tiff_fixture();
    assert!(image::decode(&bytes, DecodeLimits::default()).is_err());
    let output = decode_pages(&bytes, SequenceLimits::default()).unwrap();
    assert_eq!(output.pages.len(), 2);
    assert_eq!(output.pages[0].image.pixel_format(), PixelFormat::Rgb16);
    assert_eq!(
        output.pages[0].image.metadata().orientation,
        Orientation::Rotate90
    );
    assert_eq!(
        output.pages[0].image.packed_bytes().unwrap(),
        [1u16, 258, 65535, 1234, 5678, 9012]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        (
            output.pages[1].image.width(),
            output.pages[1].image.height()
        ),
        (1, 2)
    );
    assert_eq!(output.pages[1].image.pixel_format(), PixelFormat::Gray8);
    assert_eq!(output.pages[1].image.packed_bytes().unwrap(), [0, 255]);
    assert!(output.pages.iter().all(|page| page.alpha.is_none()));
    assert!(output.metadata_loss.is_empty());
    assert_eq!(
        decode_pages(
            &bytes,
            SequenceLimits {
                max_frames: 1,
                ..SequenceLimits::default()
            }
        )
        .unwrap_err()
        .kind(),
        ErrorKind::ResourceExhausted
    );
    assert_eq!(
        decode_pages(
            &bytes,
            SequenceLimits {
                max_total_decoded_bytes: 1,
                ..SequenceLimits::default()
            }
        )
        .unwrap_err()
        .kind(),
        ErrorKind::ResourceExhausted
    );
}

#[test]
fn tiff_alpha_association_is_retained_in_the_page_contract() {
    let mut bytes = Cursor::new(Vec::new());
    {
        let mut encoder = tiff::encoder::TiffEncoder::new(&mut bytes).unwrap();
        let mut page = encoder
            .new_image::<tiff::encoder::colortype::RGBA8>(1, 1)
            .unwrap();
        page.encoder()
            .write_tag(tiff::tags::Tag::ExtraSamples, &[1u16][..])
            .unwrap();
        page.write_data(&[64, 0, 0, 128]).unwrap();
    }
    let output = decode_pages(&bytes.into_inner(), SequenceLimits::default()).unwrap();
    assert_eq!(output.pages[0].alpha, Some(AlphaMode::Premultiplied));
    assert_eq!(
        output.pages[0].image.packed_bytes().unwrap(),
        [64, 0, 0, 128]
    );
}
