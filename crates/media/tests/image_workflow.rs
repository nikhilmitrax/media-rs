//! End-to-end facade behavior, plan reuse, resource lifetime, and safe output.
#![cfg(feature = "native")]
use media::{
    backend::native::image as codec,
    io::{sink::Sink, source::Source},
    prelude::*,
};
use std::{
    io::{self, Cursor, Read},
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};
fn png() -> Vec<u8> {
    let image = Image::from_packed(4, 2, PixelFormat::Rgb8, (0..24).collect()).unwrap();
    codec::encode(&image, EncodeOptions::new(ImageFormat::Png))
        .unwrap()
        .bytes
}
fn path() -> PathBuf {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    std::env::temp_dir().join(format!(
        "media-facade-{}-{}.png",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ))
}
#[test]
fn fluent_decode_resize_write_roundtrip_is_real() {
    let asset = Session::default().from_source(Source::memory(png()));
    let request = asset
        .image()
        .unwrap()
        .resize_with_filter(Fit::cover(2, 2), Filter::Nearest);
    let plan = request.compile().unwrap();
    assert!(plan.explain().contains("decode still image via image"));
    let read = plan.execute().unwrap();
    assert_eq!((read.image.width(), read.image.height()), (2, 2));
    assert_eq!(read.report.decoded_count, 1);
    assert!(!read.report.hardware_decode);
    let output = request
        .write_with_options(Sink::memory(), EncodeOptions::new(ImageFormat::Png))
        .unwrap();
    let decoded = codec::decode(&output.output.into_bytes().unwrap(), Default::default()).unwrap();
    assert_eq!(
        decoded.packed_bytes().unwrap(),
        read.image.packed_bytes().unwrap()
    );
}
#[test]
fn retained_images_keep_session_accounting_until_last_alias_retires() {
    let session = Session::default();
    let asset = session.from_source(Source::memory(png()));
    let image = asset.image().unwrap().read().unwrap();
    let alias = image.clone();
    assert_eq!(session.live_host_bytes(), 24);
    drop(image);
    drop(asset);
    assert_eq!(session.live_host_bytes(), 24);
    assert_eq!(alias.width(), 4);
    drop(alias);
    assert_eq!(session.live_host_bytes(), 0);
}
#[test]
fn reusable_plans_open_independent_cursors_and_return_distinct_outputs() {
    let plan = Session::default()
        .from_source(Source::memory(png()))
        .image()
        .unwrap()
        .compile()
        .unwrap();
    let other = plan.clone();
    let first = std::thread::spawn(move || other.execute().unwrap())
        .join()
        .unwrap();
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
}
#[test]
fn changed_sources_invalidate_compiled_plans() {
    let file = path();
    std::fs::write(&file, png()).unwrap();
    let plan = media::open(&file)
        .unwrap()
        .image()
        .unwrap()
        .compile()
        .unwrap();
    std::fs::write(&file, [0; 10]).unwrap();
    assert_eq!(plan.execute().unwrap_err().kind(), ErrorKind::Incompatible);
    std::fs::remove_file(file).unwrap();
}
#[test]
fn input_pixel_and_host_limits_fail_without_leaking_reservations() {
    let input = png();
    for limits in [
        Limits::new().encoded_bytes(4),
        Limits::new().image_pixels(2),
        Limits::new().host_memory(10),
    ] {
        let session = Session::builder().limits(limits).build().unwrap();
        assert!(
            session
                .from_source(Source::memory(input.clone()))
                .image()
                .unwrap()
                .read()
                .is_err()
        );
        assert_eq!(session.live_host_bytes(), 0);
    }
}
#[test]
fn same_file_output_is_read_before_it_can_be_truncated() {
    let file = path();
    std::fs::write(&file, png()).unwrap();
    media::open(&file)
        .unwrap()
        .image()
        .unwrap()
        .resize(Fit::inside(2, 2))
        .write(&file)
        .unwrap();
    let image = media::open(&file).unwrap().image().unwrap().read().unwrap();
    assert_eq!((image.width(), image.height()), (2, 1));
    std::fs::remove_file(file).unwrap();
}
#[test]
fn malformed_input_does_not_truncate_existing_destination() {
    let file = path();
    std::fs::write(&file, b"existing data").unwrap();
    let request = Session::default()
        .from_source(Source::memory(vec![0; 10]))
        .image()
        .unwrap();
    assert!(request.write(&file).is_err());
    assert_eq!(std::fs::read(&file).unwrap(), b"existing data");
    std::fs::remove_file(file).unwrap();
}
struct InterruptedOnce {
    first: bool,
    inner: Cursor<Vec<u8>>,
}
impl Read for InterruptedOnce {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        if self.first {
            self.first = false;
            return Err(io::ErrorKind::Interrupted.into());
        }
        self.inner.read(buffer)
    }
}
#[test]
fn sequential_sources_retry_interrupted_io_without_claiming_random_access() {
    let bytes = Arc::new(png());
    let len = bytes.len();
    let source = Source::from_reader_factory("sequential", None, Some(len as u64), move || {
        Ok(InterruptedOnce {
            first: true,
            inner: Cursor::new(bytes.as_ref().clone()),
        })
    })
    .unwrap();
    assert!(!source.capabilities().unwrap().seekable);
    let result = Session::default()
        .from_source(source)
        .image()
        .unwrap()
        .read()
        .unwrap();
    assert_eq!(result.width(), 4);
}

#[test]
fn encoded_results_keep_their_allocation_budget_until_released() {
    let input = png();
    let session = Session::builder()
        .limits(Limits::new().host_memory(input.len() * 2 - 1))
        .build()
        .unwrap();
    let request = session.from_source(Source::memory(input)).image().unwrap();
    let encoded = request
        .encode(EncodeOptions::new(ImageFormat::Png))
        .unwrap();
    assert!(session.live_host_bytes() >= encoded.bytes().len());
    assert_eq!(
        request
            .encode(EncodeOptions::new(ImageFormat::Png))
            .unwrap_err()
            .kind(),
        ErrorKind::ResourceExhausted
    );
    drop(encoded);
    assert_eq!(session.live_host_bytes(), 0);
    assert!(request.encode(EncodeOptions::new(ImageFormat::Png)).is_ok());
}
