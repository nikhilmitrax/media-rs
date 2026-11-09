//! Byte-source, versioned-index, and explicit sink completion contracts.

use std::fs;
use std::io::{Cursor, Read, Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use media_io::index::VersionedIndex;
use media_io::sink::{FinishedSink, Sink};
use media_io::source::Source;

static NEXT_FILE: AtomicU64 = AtomicU64::new(0);

struct TestFile(PathBuf);
impl TestFile {
    fn new() -> Self {
        Self(std::env::temp_dir().join(format!(
            "media-io-{}-{}",
            std::process::id(),
            NEXT_FILE.fetch_add(1, Ordering::Relaxed)
        )))
    }
}
impl Drop for TestFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

#[test]
fn memory_readers_are_independent_and_ranges_preserve_other_cursors() {
    let source = Source::memory(Arc::<[u8]>::from(&b"abcdef"[..]));
    let mut first = source.open().unwrap();
    let mut second = source.clone().open().unwrap();
    let mut bytes = [0; 2];
    first.read_exact(&mut bytes).unwrap();
    assert_eq!(&bytes, b"ab");
    second.read_exact(&mut bytes).unwrap();
    assert_eq!(&bytes, b"ab");
    assert_eq!(source.read_range(3, 2).unwrap(), b"de");
    first.read_exact(&mut bytes).unwrap();
    assert_eq!(&bytes, b"cd");
    assert_eq!(source.memory_bytes(), Some(&b"abcdef"[..]));
    assert!(source.read_range(5, 2).is_err());
    assert!(source.read_range(u64::MAX, 1).is_err());
}

#[test]
fn file_cursors_are_independent_and_changed_files_invalidate_indexes() {
    let file = TestFile::new();
    fs::write(&file.0, b"original").unwrap();
    let source = Source::file(&file.0).unwrap();
    let index = VersionedIndex::new(&source, "video:0", "timeline:v1", vec![0, 4]).unwrap();
    let mut first = source.open().unwrap();
    first.seek(SeekFrom::Start(4)).unwrap();
    let mut second = source.open().unwrap();
    let mut byte = [0];
    second.read_exact(&mut byte).unwrap();
    assert_eq!(byte, [b'o']);
    assert_eq!(
        index.get(&source, "video:0", "timeline:v1").unwrap(),
        Some(&vec![0, 4])
    );
    assert!(
        index
            .get(&source, "audio:0", "timeline:v1")
            .unwrap()
            .is_none()
    );
    assert!(
        index
            .get(&source, "video:0", "timeline:v2")
            .unwrap()
            .is_none()
    );
    fs::write(&file.0, b"different length").unwrap();
    assert!(
        index
            .get(&source, "video:0", "timeline:v1")
            .unwrap()
            .is_none()
    );
    assert_eq!(source.capabilities().unwrap().known_length, Some(16));
}

#[test]
fn memory_indexes_compare_complete_content() {
    let source = Source::memory(Arc::<[u8]>::from(&b"abcd"[..]));
    let equivalent = Source::memory(Arc::<[u8]>::from(&b"abcd"[..]));
    let changed = Source::memory(Arc::<[u8]>::from(&b"abce"[..]));
    let index = VersionedIndex::new(&source, "0", "default", [7]).unwrap();
    assert_eq!(index.get(&equivalent, "0", "default").unwrap(), Some(&[7]));
    assert!(index.get(&changed, "0", "default").unwrap().is_none());
}

#[test]
fn sequential_factories_do_not_claim_seeking_or_reusable_versions() {
    let source = Source::from_reader_factory("pipe-replay", None, Some(3), || {
        Ok(Cursor::new(b"abc".to_vec()))
    })
    .unwrap();
    let capabilities = source.capabilities().unwrap();
    assert!(capabilities.sequential);
    assert!(!capabilities.seekable);
    assert!(!capabilities.range_readable);
    assert!(!capabilities.stable_version);
    assert!(source.open().is_err());
    assert!(source.read_range(0, 1).is_err());
    assert!(VersionedIndex::new(&source, "0", "default", [0]).is_err());
    let mut bytes = Vec::new();
    source
        .open_reader()
        .unwrap()
        .read_to_end(&mut bytes)
        .unwrap();
    assert_eq!(bytes, b"abc");
}

#[test]
fn caller_versions_distinguish_seekable_factory_objects() {
    let source = Source::from_seekable_factory("object", Some("v1".into()), Some(3), || {
        Ok(Cursor::new(b"abc".to_vec()))
    })
    .unwrap();
    let changed = Source::from_seekable_factory("object", Some("v2".into()), Some(3), || {
        Ok(Cursor::new(b"def".to_vec()))
    })
    .unwrap();
    let index = VersionedIndex::new(&source, "0", "default", [0]).unwrap();
    assert_eq!(source.read_range(1, 2).unwrap(), b"bc");
    assert!(index.get(&changed, "0", "default").unwrap().is_none());
}

#[test]
fn invalid_metadata_and_reader_failures_are_not_silenced() {
    assert!(Source::file(std::env::temp_dir()).is_err());
    assert!(
        Source::from_reader_factory("", None, None, || Ok(Cursor::new(Vec::<u8>::new()))).is_err()
    );
    assert!(
        Source::from_reader_factory("object", Some(String::new()), None, || Ok(Cursor::new(
            Vec::<u8>::new()
        )))
        .is_err()
    );
    let source = Source::from_reader_factory("failure", Some("v1".into()), None, || {
        Err::<Cursor<Vec<u8>>, _>(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "denied",
        ))
    })
    .unwrap();
    assert!(source.open_reader().is_err());
    let source = Source::memory(Vec::new());
    assert!(VersionedIndex::new(&source, "", "v1", Vec::<u8>::new()).is_err());
    assert_eq!(source.read_range(0, 0).unwrap(), Vec::<u8>::new());
}

#[test]
fn memory_sink_rewrites_headers_and_requires_explicit_finish() {
    let mut sink = Sink::memory();
    sink.write_all(b"0000payload").unwrap();
    sink.seek(SeekFrom::Start(0)).unwrap();
    sink.write_all(b"0007").unwrap();
    assert_eq!(sink.bytes_written(), 15);
    assert!(sink.capabilities().seekable);
    assert_eq!(sink.finish().unwrap().into_bytes().unwrap(), b"0007payload");
}

#[test]
fn prior_io_error_prevents_successful_finish() {
    let mut sink = Sink::memory();
    sink.write_all(b"partial").unwrap();
    assert!(sink.seek(SeekFrom::Current(-100)).is_err());
    assert!(sink.has_failed());
    assert!(sink.finish().is_err());
}

#[test]
fn file_finish_and_abort_preserve_explicit_completion_state() {
    let file = TestFile::new();
    let mut sink = Sink::file(&file.0).unwrap();
    sink.write_all(b"partial").unwrap();
    let partial = sink.abort();
    assert_eq!(partial.path.as_deref(), Some(file.0.as_path()));
    assert_eq!(partial.bytes_written, 7);
    assert!(!partial.io_failed);
    assert_eq!(fs::read(&file.0).unwrap(), b"partial");
    let mut sink = Sink::file(&file.0).unwrap();
    sink.write_all(b"complete").unwrap();
    match sink.finish().unwrap() {
        FinishedSink::File {
            path,
            bytes_written,
        } => {
            assert_eq!(path, file.0);
            assert_eq!(bytes_written, 8);
        }
        FinishedSink::Memory(_) => panic!("file destination must remain a file"),
    }
    assert_eq!(fs::read(&file.0).unwrap(), b"complete");
}
