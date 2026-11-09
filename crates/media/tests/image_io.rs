//! Count real source reads across convenience decoding and independent plans.
#![cfg(feature = "native")]
use media::{
    Session,
    backend::native::image as codec,
    io::{
        range::{RangeMetadata, RangeOptions, RangeResponse, RangeSource},
        source::Source,
    },
    prelude::*,
};
use std::{
    io::{self, Cursor, Read},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};
fn png(width: u32, height: u32) -> Vec<u8> {
    let image = Image::from_packed(
        width,
        height,
        PixelFormat::Rgb8,
        (0..width as usize * height as usize * 3)
            .map(|v| v as u8)
            .collect(),
    )
    .unwrap();
    codec::encode(&image, codec::EncodeOptions::new(codec::ImageFormat::Png))
        .unwrap()
        .bytes
}
struct Counted {
    reader: Cursor<Arc<[u8]>>,
    bytes: Arc<AtomicUsize>,
}
impl Read for Counted {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        let count = output.len().min(7);
        let read = self.reader.read(&mut output[..count])?;
        self.bytes.fetch_add(read, Ordering::Relaxed);
        Ok(read)
    }
}
fn source(bytes: Vec<u8>) -> (Source, Arc<AtomicUsize>, Arc<AtomicUsize>) {
    let bytes: Arc<[u8]> = bytes.into();
    let opens = Arc::new(AtomicUsize::new(0));
    let reads = Arc::new(AtomicUsize::new(0));
    let opening = opens.clone();
    let reading = reads.clone();
    let source = Source::from_reader_factory(
        "counted immutable image",
        Some("v1".into()),
        Some(bytes.len() as u64),
        move || {
            opening.fetch_add(1, Ordering::Relaxed);
            Ok(Counted {
                reader: Cursor::new(bytes.clone()),
                bytes: reading.clone(),
            })
        },
    )
    .unwrap();
    (source, opens, reads)
}
#[test]
fn ordinary_image_read_has_one_full_object_load_and_exact_execution_counters() {
    let bytes = png(4, 2);
    let expected = codec::decode(&bytes, Default::default()).unwrap();
    let len = bytes.len();
    let (input, opens, reads) = source(bytes);
    let session = Session::builder()
        .limits(
            Limits::new()
                .cpu_executions(1)
                .cpu_codecs(1)
                .cpu_codec_threads(1),
        )
        .build()
        .unwrap();
    let result = session
        .from_source(input)
        .image()
        .unwrap()
        .read_with_report()
        .unwrap();
    assert_eq!(opens.load(Ordering::Relaxed), 1);
    assert_eq!(reads.load(Ordering::Relaxed), len);
    assert_eq!(result.report.bytes_read, len as u64);
    assert_eq!(
        result.image.packed_bytes().unwrap(),
        expected.packed_bytes().unwrap()
    );
    assert_eq!(result.image.metadata(), expected.metadata());
    assert_eq!(session.live_cpu_usage().executions, 0);
    assert_eq!(
        session.live_host_bytes(),
        result.image.storage().byte_len()
            + codec::metadata_payload_bytes(result.image.metadata()).unwrap()
    );
    drop(result);
    assert_eq!(session.live_host_bytes(), 0);
}
#[test]
fn image_plan_compile_and_each_execution_use_fresh_single_cursors_without_input_cache() {
    let bytes = png(8, 4);
    let len = bytes.len();
    let (input, opens, reads) = source(bytes);
    let session = Session::default();
    let plan = session
        .from_source(input)
        .image()
        .unwrap()
        .resize_with_filter(Fit::inside(4, 2), Filter::Nearest)
        .compile()
        .unwrap();
    assert_eq!(opens.load(Ordering::Relaxed), 1);
    assert_eq!(reads.load(Ordering::Relaxed), len);
    assert_eq!(session.live_host_bytes(), 0);
    let one = plan.execute().unwrap();
    let other = plan.clone();
    let two = std::thread::spawn(move || other.execute().unwrap())
        .join()
        .unwrap();
    assert_eq!(opens.load(Ordering::Relaxed), 3);
    assert_eq!(reads.load(Ordering::Relaxed), 3 * len);
    assert_eq!(one.report.bytes_read, len as u64);
    assert_eq!(
        one.image.packed_bytes().unwrap(),
        two.image.packed_bytes().unwrap()
    );
    assert!(!one.image.storage().shares_allocation(two.image.storage()));
    drop((one, two));
    assert_eq!(session.live_host_bytes(), 0);
}
#[test]
fn unversioned_convenience_does_not_replay_its_source_and_plans_revalidate_fresh_geometry() {
    let first = png(4, 2);
    let second = png(9, 1);
    let opens = Arc::new(AtomicUsize::new(0));
    let counter = opens.clone();
    let source = Source::from_reader_factory("changing unversioned image", None, None, move || {
        Ok(Cursor::new(
            if counter.fetch_add(1, Ordering::AcqRel) == 0 {
                first.clone()
            } else {
                second.clone()
            },
        ))
    })
    .unwrap();
    let session = Session::default();
    let asset = session.from_source(source);
    let one = asset.image().unwrap().read().unwrap();
    assert_eq!((one.width(), one.height()), (4, 2));
    assert_eq!(opens.load(Ordering::Acquire), 1);
    drop(one);
    let plan = asset.image().unwrap().compile().unwrap();
    let two = plan.execute().unwrap();
    assert_eq!((two.image.width(), two.image.height()), (9, 1));
    assert_eq!(opens.load(Ordering::Acquire), 3);
    drop(two);
    assert_eq!(session.live_host_bytes(), 0);
}
#[test]
fn single_pass_image_still_rejects_a_version_replacement_during_read() {
    struct Changed {
        bytes: Vec<u8>,
        changed: AtomicBool,
    }
    impl RangeSource for Changed {
        fn metadata(&self) -> media::Result<RangeMetadata> {
            Ok(RangeMetadata {
                length: self.bytes.len() as u64,
                version: if self.changed.load(Ordering::Acquire) {
                    "v2"
                } else {
                    "v1"
                }
                .into(),
            })
        }
        fn read_range(
            &self,
            expected: &RangeMetadata,
            offset: u64,
            length: usize,
        ) -> media::Result<RangeResponse> {
            if &self.metadata()? != expected {
                return Err(media::Error::incompatible(
                    "conditional image version changed",
                ));
            }
            let bytes = self.bytes[offset as usize..offset as usize + length].to_vec();
            self.changed.store(true, Ordering::Release);
            Ok(RangeResponse {
                metadata: expected.clone(),
                bytes,
            })
        }
    }
    let session = Session::default();
    let provider = Arc::new(Changed {
        bytes: png(4, 2),
        changed: AtomicBool::new(false),
    });
    let source =
        Source::from_range_source("replaced image", provider, RangeOptions::default()).unwrap();
    assert_eq!(
        session
            .from_source(source)
            .image()
            .unwrap()
            .read()
            .unwrap_err()
            .kind(),
        ErrorKind::Incompatible
    );
    assert_eq!(session.live_host_bytes(), 0);
    assert_eq!(session.live_cpu_usage().executions, 0);
}
