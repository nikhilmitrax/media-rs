//! Hardware-SDK integration tests using tiny reproducible generated fixtures.
//!
//! The ffmpeg executable is used only to generate fixtures and independent
//! reference output. All operations under test call linked libraries directly.
#![cfg(feature = "linked")]

use media_backend_ffmpeg::codec::{VideoDecodeOptions, VideoOutput, VideoReader};
use media_backend_ffmpeg::format::{
    PacketReader, ProbeOptions, TrackKind, probe, probe_headers, remux,
};
use media_backend_ffmpeg::hardware;
use media_core::media::{PixelFormat, TrackId};
use media_core::time::Time;
use media_io::source::Source;
use std::fs;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

struct Fixture {
    directory: PathBuf,
    video: PathBuf,
    binary: PathBuf,
}
impl Fixture {
    fn new(av: bool) -> Option<Self> {
        let binary = std::env::var_os("MEDIA_FFMPEG_BIN")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                if Path::new("/opt/homebrew/bin/ffmpeg").is_file() {
                    PathBuf::from("/opt/homebrew/bin/ffmpeg")
                } else {
                    PathBuf::from("ffmpeg")
                }
            });
        if Command::new(&binary).arg("-version").output().is_err() {
            assert!(
                std::env::var_os("MEDIA_REQUIRE_FFMPEG_TESTS").is_none(),
                "required native FFmpeg integration test lacks the fixture generator"
            );
            eprintln!("SKIP: native SDK is linked, but no ffmpeg fixture generator is installed");
            return None;
        }
        static SEQUENCE: AtomicU64 = AtomicU64::new(0);
        let id = SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let directory =
            std::env::temp_dir().join(format!("media-rs-ffmpeg-test-{}-{id}", std::process::id()));
        fs::create_dir(&directory).unwrap();
        let video = directory.join(if av { "av-offset.mkv" } else { "b-frames.mp4" });
        let mut command = Command::new(&binary);
        command.args([
            "-v",
            "error",
            "-nostdin",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=32x24:rate=10:duration=1.2",
        ]);
        if av {
            command.args([
                "-itsoffset",
                "0.25",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:sample_rate=8000:duration=0.5",
            ]);
        }
        command.args([
            "-c:v",
            "libx264",
            "-threads",
            "1",
            "-bf",
            if av { "0" } else { "2" },
            "-g",
            "12",
            "-pix_fmt",
            "yuv420p",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            "-colorspace",
            "bt709",
            "-color_range",
            "tv",
        ]);
        if av {
            command.args(["-c:a", "pcm_s16le"]);
        } else {
            command.arg("-an");
        }
        let output = command.arg(&video).output().unwrap();
        assert!(
            output.status.success(),
            "fixture generation: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        Some(Self {
            directory,
            video,
            binary,
        })
    }
    fn source(&self) -> Source {
        Source::file(&self.video).unwrap()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.directory);
    }
}

#[test]
fn b_frames_are_drained_with_exact_presentation_timing() {
    let Some(fixture) = Fixture::new(false) else {
        return;
    };
    let mut reader = VideoReader::open(&fixture.source(), VideoDecodeOptions::default()).unwrap();
    let mut frames = Vec::new();
    while let Some(frame) = reader.next_frame().unwrap() {
        frames.push(frame);
    }
    assert_eq!(frames.len(), 12, "EOF drain must retain delayed B frames");
    assert_eq!(reader.frames_decoded(), 12);
    assert_eq!(reader.packets_sent(), 12);
    assert!(reader.next_frame().unwrap().is_none());
    for (index, frame) in frames.iter().enumerate() {
        assert_eq!(frame.timing.ordinal, Some(index as u64));
        assert_eq!(frame.timing.pts, Some(Time::new(index as i64, 10).unwrap()));
        assert_eq!(frame.timing.source_pts, frame.timing.pts);
        assert_eq!(frame.image.pixel_format(), PixelFormat::Rgb8);
        assert_eq!((frame.image.width(), frame.image.height()), (32, 24));
        assert_eq!(frame.image.packed_bytes().unwrap().len(), 32 * 24 * 3);
    }
    let retained = frames.remove(0);
    let bytes = retained.image.packed_bytes().unwrap().to_vec();
    drop(reader);
    drop(frames);
    assert_eq!(retained.image.packed_bytes().unwrap(), bytes);
}

#[test]
fn rgb_conversion_matches_direct_ffmpeg_reference() {
    let Some(fixture) = Fixture::new(false) else {
        return;
    };
    let mut reader = VideoReader::open(&fixture.source(), VideoDecodeOptions::default()).unwrap();
    let frame = reader.next_frame().unwrap().unwrap();
    let reference = Command::new(&fixture.binary)
        .args(["-v", "error", "-nostdin", "-i"])
        .arg(&fixture.video)
        .args([
            "-frames:v",
            "1",
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "pipe:1",
        ])
        .output()
        .unwrap();
    assert!(reference.status.success());
    assert_eq!(frame.image.packed_bytes().unwrap(), reference.stdout);
    assert!(
        reader.color_assumptions().is_empty(),
        "fixture establishes matrix/range explicitly"
    );
}

#[test]
fn native_yuv_planes_and_memory_source_work() {
    let Some(fixture) = Fixture::new(false) else {
        return;
    };
    let source = Source::memory(fs::read(&fixture.video).unwrap());
    let options = VideoDecodeOptions {
        output: VideoOutput::Native,
        ..VideoDecodeOptions::default()
    };
    let mut reader = VideoReader::open(&source, options).unwrap();
    let frame = reader.next_frame().unwrap().unwrap();
    assert_eq!(frame.image.pixel_format(), PixelFormat::Yuv420p);
    let planes = &frame.image.layout().planes;
    assert_eq!(planes.len(), 3);
    assert_eq!((planes[0].row_bytes, planes[0].rows), (32, 24));
    assert_eq!((planes[1].row_bytes, planes[1].rows), (16, 12));
    assert_eq!(
        frame.image.host_bytes().unwrap().len(),
        32 * 24 + 2 * 16 * 12
    );
    assert!(
        frame.image.packed_bytes().is_err(),
        "planar YUV cannot masquerade as packed RGB"
    );
}

#[test]
fn compressed_packets_keep_pts_dts_and_configuration() {
    let Some(fixture) = Fixture::new(false) else {
        return;
    };
    let mut packets = PacketReader::open(&fixture.source(), ProbeOptions::default()).unwrap();
    assert_eq!(packets.info().tracks.len(), 1);
    assert!(!packets.info().tracks[0].codec_config.is_empty());
    let mut count = 0;
    let mut reordered = false;
    let mut keyframes = 0;
    while let Some(packet) = packets.next_packet().unwrap() {
        assert_eq!(packet.track, TrackId(0));
        assert!(!packet.data.is_empty());
        assert!(packet.duration.is_some());
        reordered |= packet.pts != packet.dts;
        keyframes += usize::from(packet.keyframe);
        count += 1;
    }
    assert_eq!(count, 12);
    assert!(reordered, "B-frame packets need different PTS and DTS");
    assert!(keyframes >= 1);
}

#[test]
fn seek_after_drain_restores_decode_and_does_not_invent_ordinals() {
    let Some(fixture) = Fixture::new(false) else {
        return;
    };
    let mut reader = VideoReader::open(&fixture.source(), VideoDecodeOptions::default()).unwrap();
    let mut expected = None;
    while let Some(frame) = reader.next_frame().unwrap() {
        if frame.timing.pts == Some(Time::from_millis(600)) {
            expected = Some(frame.image.packed_bytes().unwrap().to_vec());
        }
    }
    reader.seek(Time::from_millis(600)).unwrap();
    let frame = loop {
        let frame = reader
            .next_frame()
            .unwrap()
            .expect("seek must restore data after drained EOF");
        if frame.timing.pts == Some(Time::from_millis(600)) {
            break frame;
        }
    };
    assert_eq!(frame.timing.ordinal, None);
    assert_eq!(frame.image.packed_bytes().unwrap(), expected.unwrap());
}

#[test]
fn source_and_output_budgets_fail_explicitly() {
    let Some(fixture) = Fixture::new(false) else {
        return;
    };
    let source = fixture.source();
    assert!(
        probe(
            &source,
            ProbeOptions {
                max_read_bytes: 64,
                ..ProbeOptions::default()
            }
        )
        .is_err()
    );
    assert!(
        VideoReader::open(
            &source,
            VideoDecodeOptions {
                max_frame_bytes: 10,
                ..VideoDecodeOptions::default()
            }
        )
        .is_err()
    );
    let bytes: Arc<[u8]> = fs::read(&fixture.video).unwrap().into();
    let reader_bytes = Arc::clone(&bytes);
    let sequential = Source::from_reader_factory(
        "sequential mp4",
        Some("v1".into()),
        Some(bytes.len() as u64),
        move || Ok(Cursor::new(Arc::clone(&reader_bytes))),
    )
    .unwrap();
    let mut reader = VideoReader::open(&sequential, VideoDecodeOptions::default()).unwrap();
    assert!(reader.next_frame().unwrap().is_some());
    assert!(reader.seek(Time::ZERO).is_err());
}

#[test]
fn remux_preserves_selected_av_streams_and_offset() {
    let Some(fixture) = Fixture::new(true) else {
        return;
    };
    let source = fixture.source();
    let original = probe(&source, ProbeOptions::default()).unwrap();
    let video = original
        .tracks
        .iter()
        .find(|t| t.kind == TrackKind::Video)
        .unwrap();
    let audio = original
        .tracks
        .iter()
        .find(|t| t.kind == TrackKind::Audio)
        .unwrap();
    assert_eq!(
        audio
            .start
            .unwrap()
            .checked_sub(video.start.unwrap())
            .unwrap(),
        Time::from_millis(250)
    );
    let destination = fixture.directory.join("copy.mkv");
    let ids = original.tracks.iter().map(|t| t.id).collect::<Vec<_>>();
    let report = remux(&source, &destination, &ids).unwrap();
    assert_eq!(report.tracks, ids);
    assert!(report.packets_written > 12);
    let remuxed = probe(
        &Source::file(&destination).unwrap(),
        ProbeOptions::default(),
    )
    .unwrap();
    assert_eq!(remuxed.tracks.len(), original.tracks.len());
    for (before, after) in original.tracks.iter().zip(&remuxed.tracks) {
        assert_eq!(before.codec, after.codec);
        assert_eq!(before.kind, after.kind);
        assert_eq!(before.start, after.start);
        assert_eq!(before.codec_config, after.codec_config);
    }
    let before = all_packets(&source);
    let after = all_packets(&Source::file(destination).unwrap());
    assert_eq!(
        before, after,
        "compressed bytes and original timing must survive stream copy"
    );
}

type PacketSnapshot = (TrackId, Option<Time>, Option<Time>, Vec<u8>);

fn all_packets(source: &Source) -> Vec<PacketSnapshot> {
    let mut reader = PacketReader::open(source, ProbeOptions::default()).unwrap();
    let mut output = Vec::new();
    while let Some(packet) = reader.next_packet().unwrap() {
        output.push((packet.track, packet.pts, packet.dts, packet.data.to_vec()));
    }
    output
}

#[test]
fn incompatible_remux_does_not_create_output_or_drop_streams() {
    let Some(fixture) = Fixture::new(true) else {
        return;
    };
    let destination = fixture.directory.join("incompatible.wav");
    let error = remux(&fixture.source(), &destination, &[TrackId(0), TrackId(1)]).unwrap_err();
    assert!(error.to_string().contains("support"));
    assert!(!destination.exists());
    assert!(
        remux(
            &fixture.source(),
            &fixture.directory.join("duplicates.mkv"),
            &[TrackId(0), TrackId(0)]
        )
        .is_err()
    );
    let existing = fixture.directory.join("existing.mkv");
    fs::write(&existing, b"keep existing data").unwrap();
    assert!(remux(&fixture.source(), &existing, &[TrackId(0)]).is_err());
    assert_eq!(fs::read(existing).unwrap(), b"keep existing data");
}

#[test]
fn linked_library_diagnostics_do_not_claim_hardware_decode() {
    assert!(hardware::is_linked());
    assert!(hardware::version().unwrap().starts_with("libavformat "));
    let _compiled = hardware::compiled_device_types().unwrap();
    assert!(hardware::require_hardware_decode().is_err());
}

#[test]
fn header_probe_does_not_require_opening_a_codec() {
    let Some(fixture) = Fixture::new(false) else {
        return;
    };
    let info = probe_headers(&fixture.source(), ProbeOptions::default()).unwrap();
    assert_eq!(info.tracks[0].codec, "h264");
    assert_eq!(
        (info.tracks[0].width, info.tracks[0].height),
        (Some(32), Some(24))
    );
}
