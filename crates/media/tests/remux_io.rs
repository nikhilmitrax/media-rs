//! All-track stream copy uses one demux cursor and retains packet/timing fidelity.
#![cfg(feature = "ffmpeg")]
use media::{
    Session,
    backend::ffmpeg::format::{self, PacketReader, ProbeOptions},
    core::{error::ErrorKind, media::TrackId, time::Time},
    io::{
        range::{RangeMetadata, RangeOptions, RangeResponse, RangeSource},
        source::Source,
    },
};
use std::{
    fs,
    io::{self, Cursor, Read, Seek, SeekFrom},
    path::PathBuf,
    process::Command,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    },
};
struct Fixture {
    directory: PathBuf,
    path: PathBuf,
}
impl Fixture {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let directory = std::env::temp_dir().join(format!(
            "media-remux-io-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&directory).unwrap();
        let path = directory.join("offset.mkv");
        let binary = std::env::var_os("MEDIA_FFMPEG_BIN").unwrap_or_else(|| "ffmpeg".into());
        let output = Command::new(binary)
            .args([
                "-nostdin",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=32x24:rate=10:duration=1.2",
                "-itsoffset",
                "0.25",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:sample_rate=8000:duration=0.5",
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "libx264",
                "-threads:v",
                "1",
                "-bf",
                "0",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "pcm_s16le",
                "-y",
            ])
            .arg(&path)
            .output()
            .expect("FFmpeg fixture generator required");
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        Self { directory, path }
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.directory);
    }
}
struct Counted {
    inner: Cursor<Arc<[u8]>>,
    bytes: Arc<AtomicU64>,
}
impl Read for Counted {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        let read = self.inner.read(buffer)?;
        self.bytes.fetch_add(read as u64, Ordering::Relaxed);
        Ok(read)
    }
}
impl Seek for Counted {
    fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
        self.inner.seek(position)
    }
}
fn counted(bytes: Vec<u8>) -> (Source, Arc<AtomicUsize>, Arc<AtomicU64>) {
    let bytes: Arc<[u8]> = bytes.into();
    let opens = Arc::new(AtomicUsize::new(0));
    let reads = Arc::new(AtomicU64::new(0));
    let count_opens = opens.clone();
    let count_reads = reads.clone();
    let source = Source::from_seekable_factory(
        "immutable remux fixture",
        Some("v1".into()),
        Some(bytes.len() as u64),
        move || {
            count_opens.fetch_add(1, Ordering::Relaxed);
            Ok(Counted {
                inner: Cursor::new(bytes.clone()),
                bytes: count_reads.clone(),
            })
        },
    )
    .unwrap();
    (source, opens, reads)
}
#[derive(Debug, Eq, PartialEq)]
struct Snapshot {
    track: TrackId,
    pts: Option<Time>,
    dts: Option<Time>,
    duration: Option<Time>,
    data: Vec<u8>,
    flags: i32,
    side: Vec<(String, Vec<u8>)>,
}
fn packets(source: &Source) -> (Vec<(String, Vec<u8>)>, Vec<Snapshot>) {
    let mut reader = PacketReader::open(source, ProbeOptions::default()).unwrap();
    let configs = reader
        .info()
        .tracks
        .iter()
        .map(|track| (track.codec.clone(), track.codec_config.clone()))
        .collect();
    let mut packets = Vec::new();
    while let Some(item) = reader.next_packet_with_side_data().unwrap() {
        packets.push(Snapshot {
            track: item.packet.track,
            pts: item.packet.pts,
            dts: item.packet.dts,
            duration: item.packet.duration,
            data: item.packet.data.to_vec(),
            flags: item.flags,
            side: item
                .side_data
                .into_iter()
                .map(|side| (side.kind, side.data.to_vec()))
                .collect(),
        });
    }
    (configs, packets)
}
#[test]
fn default_all_track_facade_opens_and_probes_once_with_exact_packet_equivalence() {
    let fixture = Fixture::new();
    let bytes = fs::read(&fixture.path).unwrap();
    let (input, opens, reads) = counted(bytes);
    let fingerprint = input.fingerprint().unwrap();
    let session = Session::default();
    let asset = session.from_source(input.clone());
    let all = asset
        .remux()
        .write(fixture.directory.join("all.mkv"))
        .unwrap();
    assert_eq!(opens.load(Ordering::Relaxed), 1);
    assert_eq!(all.bytes_read, reads.load(Ordering::Relaxed));
    assert!(all.probe_bytes_read > 0 && all.probe_bytes_read <= all.bytes_read);
    assert_eq!(all.source, fingerprint);
    assert_eq!(all.tracks, [TrackId(0), TrackId(1)]);
    let previous = reads.load(Ordering::Relaxed);
    let explicit = asset
        .remux()
        .tracks([TrackId(0), TrackId(1)])
        .write(fixture.directory.join("explicit.mkv"))
        .unwrap();
    assert_eq!(opens.load(Ordering::Relaxed), 2);
    assert_eq!(
        explicit.bytes_read,
        reads.load(Ordering::Relaxed) - previous
    );
    assert_eq!(
        all.bytes_read, explicit.bytes_read,
        "all-track discovery must add no second demux probe"
    );
    let expected = packets(&Source::file(&fixture.path).unwrap());
    assert_eq!(packets(&Source::file(&all.destination).unwrap()), expected);
    assert_eq!(
        packets(&Source::file(&explicit.destination).unwrap()),
        expected
    );
    assert_eq!(all.packets_written, expected.1.len() as u64);
    assert_eq!(session.live_cpu_usage().executions, 0);
}
#[test]
fn invalid_selection_or_probe_limit_does_not_create_output_and_existing_files_survive() {
    let fixture = Fixture::new();
    let (input, opens, _) = counted(fs::read(&fixture.path).unwrap());
    let asset = Session::default().from_source(input.clone());
    let output = fixture.directory.join("invalid.mkv");
    assert!(asset.remux().tracks([]).write(&output).is_err());
    assert!(
        asset
            .remux()
            .tracks([TrackId(0), TrackId(0)])
            .write(&output)
            .is_err()
    );
    assert_eq!(opens.load(Ordering::Relaxed), 0);
    assert!(!output.exists());
    assert!(
        format::remux_with_options(
            &input,
            &output,
            None,
            ProbeOptions {
                max_read_bytes: 64,
                ..Default::default()
            }
        )
        .is_err()
    );
    assert!(!output.exists());
    fs::write(&output, b"retain old destination").unwrap();
    assert!(asset.remux().write(&output).is_err());
    assert_eq!(fs::read(&output).unwrap(), b"retain old destination");
}
#[test]
fn finite_remux_rechecks_source_version_after_copy_and_discloses_destination_on_failure() {
    struct VersionChanged {
        bytes: Vec<u8>,
        destination: PathBuf,
        changed: AtomicBool,
    }
    impl RangeSource for VersionChanged {
        fn metadata(&self) -> media::Result<RangeMetadata> {
            if self.destination.exists() {
                self.changed.store(true, Ordering::Release);
            }
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
            let version = if self.changed.load(Ordering::Acquire) {
                "v2"
            } else {
                "v1"
            };
            if expected.version != version {
                return Err(media::Error::incompatible("remux object version changed"));
            }
            Ok(RangeResponse {
                metadata: expected.clone(),
                bytes: self.bytes[offset as usize..offset as usize + length].to_vec(),
            })
        }
    }
    let fixture = Fixture::new();
    let destination = fixture.directory.join("changed.mkv");
    let provider = Arc::new(VersionChanged {
        bytes: fs::read(&fixture.path).unwrap(),
        destination: destination.clone(),
        changed: AtomicBool::new(false),
    });
    let source =
        Source::from_range_source("replaced copy input", provider, RangeOptions::default())
            .unwrap();
    let session = Session::default();
    let error = session
        .from_source(source)
        .remux()
        .write(&destination)
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Incompatible);
    assert_eq!(error.context().source.as_deref(), destination.to_str());
    assert!(error.to_string().contains("changed"));
    assert!(destination.exists());
    assert_eq!(session.live_cpu_usage().executions, 0);
}
