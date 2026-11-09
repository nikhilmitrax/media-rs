//! Real bounded sequence writes, retained encoded bytes and failure-safe destinations.
#![cfg(feature = "native")]
use media::{
    backend::native::image_sequence,
    io::{sink::FinishedSink, source::Source},
    prelude::*,
};
use std::{
    borrow::Cow,
    io::Cursor,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

fn gif_source() -> Vec<u8> {
    let mut bytes = Vec::new();
    let mut encoder = gif::Encoder::new(&mut bytes, 2, 1, &[255, 0, 0, 0, 0, 255]).unwrap();
    encoder.set_repeat(gif::Repeat::Finite(2)).unwrap();
    for (pixels, delay) in [(vec![0, 0], 3), (vec![1, 0], 7), (vec![0, 1], 2)] {
        encoder
            .write_frame(&gif::Frame {
                width: 2,
                height: 1,
                delay,
                buffer: Cow::Owned(pixels),
                dispose: gif::DisposalMethod::Background,
                ..Default::default()
            })
            .unwrap();
    }
    encoder.into_inner().unwrap();
    bytes
}
fn fractional_source() -> Vec<u8> {
    let mut bytes = Vec::new();
    let mut encoder = png::Encoder::new(&mut bytes, 1, 1);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.set_source_srgb(png::SrgbRenderingIntent::RelativeColorimetric);
    encoder.set_animated(2, 0).unwrap();
    encoder.set_frame_delay(1, 24).unwrap();
    let mut writer = encoder.write_header().unwrap();
    writer.write_image_data(&[255, 0, 0, 128]).unwrap();
    writer.set_frame_delay(1, 60).unwrap();
    writer.write_image_data(&[0, 255, 0, 255]).unwrap();
    writer.finish().unwrap();
    bytes
}
fn tiff_source(loss: bool) -> Vec<u8> {
    let mut output = Cursor::new(Vec::new());
    {
        let mut encoder = tiff::encoder::TiffEncoder::new(&mut output).unwrap();
        let mut page = encoder
            .new_image::<tiff::encoder::colortype::RGB16>(1, 1)
            .unwrap();
        page.encoder()
            .write_tag(tiff::tags::Tag::Orientation, 6u16)
            .unwrap();
        if loss {
            page.encoder()
                .write_tag(tiff::tags::Tag::Unknown(700), b"opaque XMP".as_slice())
                .unwrap();
        }
        page.write_data(&[1, 258, 65535]).unwrap();
        encoder
            .write_image::<tiff::encoder::colortype::RGB32Float>(
                2,
                1,
                &[-0.5, 0., 0.5, 1., 1.5, 2.],
            )
            .unwrap();
    }
    output.into_inner()
}
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        Self(std::env::temp_dir().join(format!(
            "media-sequence-encode-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        )))
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

#[test]
fn animation_encoding_retains_only_final_bytes_and_writes_every_frame_with_exact_loops() {
    let session = Session::builder()
        .limits(Limits::new().host_memory(64 * 1024))
        .build()
        .unwrap();
    let input = gif_source();
    let request = session
        .from_source(Source::memory(input.clone()))
        .animation();
    let identity_source = Source::memory(input.clone());
    let identity = identity_source.fingerprint().unwrap();
    let batch = session
        .from_source(identity_source)
        .animation()
        .read()
        .unwrap();
    assert_eq!(batch.source, identity);
    drop(batch);
    for format in [AnimationFormat::Apng, AnimationFormat::Gif] {
        let output = request.encode(AnimationEncodeOptions::new(format)).unwrap();
        assert_eq!(session.live_host_bytes(), output.bytes().len());
        assert_eq!(session.live_batches(), 0);
        assert_eq!(output.report.bytes_read, input.len() as u64);
        assert_eq!(
            (output.report.decoded_count, output.report.output_count),
            (3, 3)
        );
        assert!(!output.report.hardware_decode);
        let actual =
            image_sequence::decode_animation(output.bytes(), SequenceLimits::default()).unwrap();
        let expected = image_sequence::decode_animation(&input, SequenceLimits::default()).unwrap();
        assert_eq!(
            (actual.loop_count, actual.total_duration),
            (expected.loop_count, expected.total_duration)
        );
        for (a, b) in actual.frames.iter().zip(expected.frames) {
            assert_eq!(
                a.image.packed_bytes().unwrap(),
                b.image.packed_bytes().unwrap()
            );
            assert_eq!((a.pts, a.duration), (b.pts, b.duration));
        }
        drop(output);
        assert_eq!(session.live_host_bytes(), 0);
    }
    let path = Temp::new();
    let output = request
        .write(&path.0, AnimationEncodeOptions::new(AnimationFormat::Apng))
        .unwrap();
    assert!(matches!(output.output, FinishedSink::File { .. }));
    assert_eq!(session.live_host_bytes(), 0);
    assert_eq!(
        image_sequence::decode_animation(
            &std::fs::read(&path.0).unwrap(),
            SequenceLimits::default()
        )
        .unwrap()
        .frames
        .len(),
        3
    );
}
#[test]
fn timing_decode_and_encoded_extent_failures_preserve_existing_destination_and_release_budgets() {
    let session = Session::default();
    let request = session
        .from_source(Source::memory(fractional_source()))
        .animation();
    let path = Temp::new();
    std::fs::write(&path.0, b"keep existing destination").unwrap();
    assert!(
        request
            .write(&path.0, AnimationEncodeOptions::new(AnimationFormat::Gif))
            .is_err()
    );
    let mut limited = AnimationEncodeOptions::new(AnimationFormat::Apng);
    limited.limits.max_encoded_bytes = 32;
    assert!(request.write(&path.0, limited).is_err());
    limited = AnimationEncodeOptions::new(AnimationFormat::Apng);
    limited.limits.max_working_bytes = 1;
    assert!(request.encode(limited).is_err());
    let bytes = fractional_source();
    let broken = session
        .from_source(Source::memory(bytes[..bytes.len() / 2].to_vec()))
        .animation();
    assert!(
        broken
            .write(&path.0, AnimationEncodeOptions::new(AnimationFormat::Apng))
            .is_err()
    );
    assert_eq!(
        std::fs::read(&path.0).unwrap(),
        b"keep existing destination"
    );
    assert_eq!(session.live_host_bytes(), 0);
    assert_eq!(session.live_batches(), 0);
    let exact = request
        .encode(AnimationEncodeOptions::new(AnimationFormat::Apng))
        .unwrap();
    let decoded =
        image_sequence::decode_animation(exact.bytes(), SequenceLimits::default()).unwrap();
    assert_eq!(decoded.frames[0].duration, Time::new(1, 24).unwrap());
    assert_eq!(decoded.frames[1].duration, Time::new(1, 60).unwrap());
}
#[test]
fn tiff_writes_every_native_precision_page_and_reports_allowed_projection_loss() {
    let session = Session::default();
    let source = tiff_source(false);
    let identity_source = Source::memory(source.clone());
    let identity = identity_source.fingerprint().unwrap();
    let batch = session.from_source(identity_source).pages().read().unwrap();
    assert_eq!(batch.source, identity);
    drop(batch);
    let request = session.from_source(Source::memory(source.clone())).pages();
    let output = request.encode_tiff(TiffEncodeOptions::default()).unwrap();
    assert_eq!(session.live_host_bytes(), output.bytes().len());
    let actual = image_sequence::decode_pages(output.bytes(), SequenceLimits::default()).unwrap();
    let expected = image_sequence::decode_pages(&source, SequenceLimits::default()).unwrap();
    assert_eq!(actual.pages.len(), 2);
    for (a, b) in actual.pages.iter().zip(expected.pages) {
        assert_eq!(a.image.pixel_format(), b.image.pixel_format());
        assert_eq!(a.image.metadata(), b.image.metadata());
        assert_eq!(
            a.image.packed_bytes().unwrap(),
            b.image.packed_bytes().unwrap()
        );
    }
    drop(output);
    assert_eq!(session.live_host_bytes(), 0);
    let path = Temp::new();
    request
        .write_tiff(&path.0, TiffEncodeOptions::default())
        .unwrap();
    assert_eq!(
        image_sequence::decode_pages(&std::fs::read(&path.0).unwrap(), SequenceLimits::default())
            .unwrap()
            .pages
            .len(),
        2
    );
    let loss = session
        .from_source(Source::memory(tiff_source(true)))
        .pages();
    assert!(loss.encode_tiff(TiffEncodeOptions::default()).is_err());
    let permitted = loss
        .allow_metadata_loss()
        .encode_tiff(TiffEncodeOptions::default())
        .unwrap();
    assert!(
        permitted
            .report
            .warnings
            .iter()
            .any(|warning| warning.contains("XMP"))
    );
    assert_eq!(permitted.report.output_count, 2);
}
#[test]
fn same_path_write_finishes_after_source_decode_and_tiff_limit_failure_does_not_overwrite() {
    let session = Session::default();
    let path = Temp::new();
    std::fs::write(&path.0, gif_source()).unwrap();
    session
        .open(&path.0)
        .unwrap()
        .animation()
        .write(&path.0, AnimationEncodeOptions::new(AnimationFormat::Apng))
        .unwrap();
    assert_eq!(
        image_sequence::decode_animation(
            &std::fs::read(&path.0).unwrap(),
            SequenceLimits::default()
        )
        .unwrap()
        .frames
        .len(),
        3
    );
    let original = std::fs::read(&path.0).unwrap();
    let mut limited = TiffEncodeOptions::default();
    limited.limits.max_encoded_bytes = 32;
    assert!(
        session
            .from_source(Source::memory(tiff_source(false)))
            .pages()
            .write_tiff(&path.0, limited)
            .is_err()
    );
    assert_eq!(std::fs::read(&path.0).unwrap(), original);
    assert_eq!(session.live_host_bytes(), 0);
    assert_eq!(session.live_batches(), 0);
}
