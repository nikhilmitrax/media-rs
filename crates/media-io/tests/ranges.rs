//! Conditional generic object ranges: independent cursors, mutation, and budgets.
use media_core::error::{Error, Result};
use media_io::{index::VersionedIndex, range::*, source::Source};
use std::{
    io::{Read, Seek, SeekFrom},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
};

struct Object {
    value: Mutex<(String, Vec<u8>)>,
    requests: Mutex<Vec<(u64, usize, String)>>,
    short: AtomicBool,
    lie: AtomicBool,
}
impl Object {
    fn new() -> Self {
        Self {
            value: Mutex::new(("v1".into(), b"abcdefghij".to_vec())),
            requests: Mutex::new(Vec::new()),
            short: AtomicBool::new(false),
            lie: AtomicBool::new(false),
        }
    }
    fn change(&self) {
        *self.value.lock().unwrap() = ("v2".into(), b"ABCDEFGHIJ".to_vec());
    }
}
impl RangeSource for Object {
    fn metadata(&self) -> Result<RangeMetadata> {
        let value = self.value.lock().unwrap();
        Ok(RangeMetadata {
            version: value.0.clone(),
            length: value.1.len() as u64,
        })
    }
    fn read_range(
        &self,
        expected: &RangeMetadata,
        offset: u64,
        length: usize,
    ) -> Result<RangeResponse> {
        let value = self.value.lock().unwrap();
        self.requests
            .lock()
            .unwrap()
            .push((offset, length, expected.version.clone()));
        let metadata = RangeMetadata {
            version: value.0.clone(),
            length: value.1.len() as u64,
        };
        if metadata != *expected {
            return Err(Error::incompatible(
                "test object version precondition failed",
            ));
        }
        let end = offset as usize + length - usize::from(self.short.load(Ordering::Relaxed));
        Ok(RangeResponse {
            metadata: if self.lie.load(Ordering::Relaxed) {
                RangeMetadata {
                    version: "v-other".into(),
                    length: metadata.length,
                }
            } else {
                metadata
            },
            bytes: value.1[offset as usize..end].to_vec(),
        })
    }
}
fn source(object: &Arc<Object>) -> Source {
    Source::from_range_source(
        "object",
        object.clone(),
        RangeOptions {
            max_request_bytes: 3,
            cancellation: None,
        },
    )
    .unwrap()
}

#[test]
fn independent_seek_cursors_and_concurrent_ranges_do_not_share_position() {
    let object = Arc::new(Object::new());
    let source = source(&object);
    let mut a = source.open().unwrap();
    let mut b = source.open().unwrap();
    a.seek(SeekFrom::Start(4)).unwrap();
    let mut bytes = [0; 2];
    a.read_exact(&mut bytes).unwrap();
    assert_eq!(&bytes, b"ef");
    b.read_exact(&mut bytes).unwrap();
    assert_eq!(&bytes, b"ab");
    assert_eq!(a.stream_position().unwrap(), 6);
    assert_eq!(b.stream_position().unwrap(), 2);
    std::thread::scope(|scope| {
        let one = scope.spawn(|| source.read_range(0, 7).unwrap());
        let two = scope.spawn(|| source.read_range(3, 7).unwrap());
        assert_eq!(one.join().unwrap(), b"abcdefg");
        assert_eq!(two.join().unwrap(), b"defghij");
    });
    assert!(
        object
            .requests
            .lock()
            .unwrap()
            .iter()
            .all(|(_, len, version)| *len <= 3 && version == "v1")
    );
    let stats = source.range_statistics().unwrap();
    assert_eq!(stats.payload_bytes_received, 18);
    assert_eq!(stats.range_requests, 8);
    let capabilities = source.capabilities().unwrap();
    assert!(capabilities.range_readable && capabilities.seekable && capabilities.stable_version);
    assert_eq!(capabilities.known_length, Some(10));
}

#[test]
fn object_change_invalidates_old_cursor_and_versioned_index_without_mixing_bytes() {
    let object = Arc::new(Object::new());
    let source = source(&object);
    let index = VersionedIndex::new(&source, "video:0", "source-clock", vec![9]).unwrap();
    let mut cursor = source.open().unwrap();
    let mut prefix = [0; 3];
    cursor.read_exact(&mut prefix).unwrap();
    assert_eq!(&prefix, b"abc");
    object.change();
    let mut untouched = [99; 3];
    assert!(cursor.read(&mut untouched).is_err());
    assert_eq!(untouched, [99; 3]);
    assert_eq!(cursor.stream_position().unwrap(), 3);
    assert!(
        index
            .get(&source, "video:0", "source-clock")
            .unwrap()
            .is_none()
    );
    assert_eq!(source.read_range(0, 3).unwrap(), b"ABC");
}

#[test]
fn short_or_misversioned_responses_do_not_advance_or_modify_the_cursor_buffer() {
    let object = Arc::new(Object::new());
    let source = source(&object);
    let mut cursor = source.open().unwrap();
    object.short.store(true, Ordering::Relaxed);
    let mut bytes = [88; 3];
    assert!(cursor.read(&mut bytes).is_err());
    assert_eq!(bytes, [88; 3]);
    assert_eq!(cursor.stream_position().unwrap(), 0);
    object.short.store(false, Ordering::Relaxed);
    object.lie.store(true, Ordering::Relaxed);
    assert!(cursor.read(&mut bytes).is_err());
    assert_eq!(bytes, [88; 3]);
    assert_eq!(cursor.stream_position().unwrap(), 0);
    assert!(source.read_range(9, 2).is_err());
    assert!(source.read_range(u64::MAX, 2).is_err());
}

#[test]
fn cancellation_stops_new_requests_and_seek_arithmetic_is_checked() {
    let object = Arc::new(Object::new());
    let cancel = RangeCancellation::default();
    let source = Source::from_range_source(
        "object",
        object.clone(),
        RangeOptions {
            max_request_bytes: 2,
            cancellation: Some(cancel.clone()),
        },
    )
    .unwrap();
    let mut cursor = source.open().unwrap();
    assert_eq!(cursor.seek(SeekFrom::End(-2)).unwrap(), 8);
    let mut bytes = [0; 2];
    cursor.read_exact(&mut bytes).unwrap();
    assert_eq!(&bytes, b"ij");
    assert!(cursor.seek(SeekFrom::Start(0)).is_ok());
    assert!(cursor.seek(SeekFrom::Current(-1)).is_err());
    assert_eq!(cursor.seek(SeekFrom::Start(u64::MAX)).unwrap(), u64::MAX);
    assert!(cursor.seek(SeekFrom::Current(1)).is_err());
    let requests = object.requests.lock().unwrap().len();
    cancel.cancel();
    assert!(cursor.read(&mut bytes).is_err());
    assert!(source.open().is_err());
    assert_eq!(object.requests.lock().unwrap().len(), requests);
    assert!(
        Source::from_range_source(
            "object",
            object,
            RangeOptions {
                max_request_bytes: 0,
                cancellation: None
            }
        )
        .is_err()
    );
}
