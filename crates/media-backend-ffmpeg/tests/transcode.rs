//! Native incremental transcode integration tests with generated B-frame/AAC fixtures.
//! The CLI is used only for fixture generation, never as the implementation.
#![cfg(feature = "linked")]

use media_backend_ffmpeg::codec::{VideoDecodeOptions, VideoReader};
use media_backend_ffmpeg::encode::{CpuVideoCodec, ResizeFilter, TranscodeOptions, transcode};
use media_backend_ffmpeg::format::{PacketReader, ProbeOptions, TrackKind, probe};
use media_core::error::ErrorKind;
use media_core::media::{Packet, TrackId};
use media_core::time::Time;
use media_io::source::Source;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

struct Fixture {
    directory: PathBuf,
    source: PathBuf,
    binary: PathBuf,
}
impl Fixture {
    fn new() -> Option<Self> {
        Self::with_clock_offset(false)
    }
    fn with_clock_offset(offset: bool) -> Option<Self> {
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
                "required native transcode tests lack fixture generator"
            );
            eprintln!("SKIP: linked SDK available but no FFmpeg CLI fixture generator");
            return None;
        }
        static SEQUENCE: AtomicU64 = AtomicU64::new(0);
        let directory = std::env::temp_dir().join(format!(
            "media-rs-transcode-{}-{}",
            std::process::id(),
            SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&directory).unwrap();
        let source = directory.join("offset-bframes-aac.mp4");
        let mut command = Command::new(&binary);
        command.args([
            "-v",
            "error",
            "-nostdin",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x48:rate=10:duration=1.2",
            "-itsoffset",
            "0.25",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=0.7",
            "-c:v",
            "libx264",
            "-threads:v",
            "1",
            "-bf",
            "2",
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
            "-c:a",
            "aac",
            "-threads:a",
            "1",
            "-b:a",
            "64000",
        ]);
        if offset {
            command.args(["-output_ts_offset", "2"]);
        }
        let result = command.arg(&source).output().unwrap();
        assert!(
            result.status.success(),
            "fixture: {}",
            String::from_utf8_lossy(&result.stderr)
        );
        Some(Self {
            directory,
            source,
            binary,
        })
    }
    fn input(&self) -> Source {
        Source::file(&self.source).unwrap()
    }
    fn output(&self, name: &str) -> PathBuf {
        self.directory.join(name)
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.directory);
    }
}

fn packets(source: &Source, track: TrackId) -> Vec<Packet> {
    let mut reader = PacketReader::open(source, ProbeOptions::default()).unwrap();
    let mut output = Vec::new();
    while let Some(packet) = reader.next_packet().unwrap() {
        if packet.track == track {
            output.push(packet);
        }
    }
    output
}
fn audio_track(source: &Source) -> TrackId {
    probe(source, ProbeOptions::default())
        .unwrap()
        .tracks
        .into_iter()
        .find(|track| track.kind == TrackKind::Audio)
        .unwrap()
        .id
}

#[test]
fn resize_b_frames_and_copy_aac_preserve_payload_and_shared_timestamps() {
    let Some(fixture) = Fixture::new() else {
        return;
    };
    let output = fixture.output("resized.mp4");
    let report = transcode(
        &fixture.input(),
        &output,
        TranscodeOptions {
            width: Some(32),
            height: Some(24),
            filter: ResizeFilter::Lanczos,
            profile: Some("high".into()),
            max_b_frames: 3,
            gop: 12,
            ..TranscodeOptions::default()
        },
    )
    .unwrap();
    assert_eq!(
        (
            report.frames_decoded,
            report.frames_encoded,
            report.video_packets_written
        ),
        (12, 12, 12),
        "both decoder and encoder must drain delayed B frames"
    );
    assert_eq!((report.output_width, report.output_height), (32, 24));
    assert_eq!(report.output_pixel_format, "yuv420p");
    assert_eq!(report.codec, CpuVideoCodec::H264);
    assert_eq!(report.video_track, TrackId(0));
    assert_eq!(report.audio_tracks, vec![TrackId(1)]);
    assert!(report.unselected_tracks.is_empty());
    assert!(report.source_bytes_read > 0);
    let output_source = Source::file(&output).unwrap();
    let original_audio = packets(&fixture.input(), audio_track(&fixture.input()));
    let copied_audio = packets(&output_source, audio_track(&output_source));
    assert!(!original_audio.is_empty());
    assert_eq!(report.audio_packets_written, original_audio.len() as u64);
    assert_eq!(original_audio.len(), copied_audio.len());
    // Full compressed byte equality is stronger than a payload hash comparison.
    for (original, copied) in original_audio.iter().zip(&copied_audio) {
        assert_eq!(
            original.data, copied.data,
            "audio must remain compressed stream copy"
        );
        assert_eq!(
            (original.pts, original.dts, original.duration),
            (copied.pts, copied.dts, copied.duration)
        );
    }
    assert!(
        original_audio[0].pts.unwrap() > Time::from_millis(200),
        "fixture has a real quarter-second A/V offset (minus AAC priming)"
    );
    let video_packets = packets(&output_source, TrackId(0));
    let original_video_packets = packets(&fixture.input(), TrackId(0));
    let mut original_pts = original_video_packets
        .iter()
        .map(|packet| packet.pts.unwrap())
        .collect::<Vec<_>>();
    let mut output_pts = video_packets
        .iter()
        .map(|packet| packet.pts.unwrap())
        .collect::<Vec<_>>();
    original_pts.sort();
    output_pts.sort();
    assert_eq!(
        output_pts, original_pts,
        "re-encoding must preserve the exact presentation clock"
    );
    assert!(
        output_pts
            .windows(2)
            .all(|pair| pair[1].checked_sub(pair[0]).unwrap() == Time::new(1, 10).unwrap()),
        "explicit next-PTS intervals remain one tenth second; this does not establish the final frame interval"
    );
    assert!(
        report
            .timing_limitations
            .iter()
            .any(|limitation| limitation.contains("duration hints"))
    );
    assert!(video_packets.iter().any(|packet| packet.pts != packet.dts));
    assert!(
        video_packets
            .windows(2)
            .all(|pair| pair[0].dts < pair[1].dts)
    );
    let mut original = VideoReader::open(&fixture.input(), VideoDecodeOptions::default()).unwrap();
    let mut resized = VideoReader::open(&output_source, VideoDecodeOptions::default()).unwrap();
    let mut count = 0;
    while let Some(frame) = resized.next_frame().unwrap() {
        let reference = original.next_frame().unwrap().unwrap();
        assert_eq!((frame.image.width(), frame.image.height()), (32, 24));
        assert_eq!(frame.timing.source_pts, reference.timing.source_pts);
        // Native decoder hints may legitimately be absent/different across
        // FFmpeg versions and re-encoding. The verified clock proof is above.
        assert_eq!(frame.timing.duration, None);
        assert_eq!(
            frame.image.metadata().color,
            reference.image.metadata().color
        );
        count += 1;
    }
    assert_eq!(count, 12);
    assert!(original.next_frame().unwrap().is_none());
}

#[test]
fn memory_source_and_explicit_audio_selection_finalize_video_only() {
    let Some(fixture) = Fixture::new() else {
        return;
    };
    let source = Source::memory(fs::read(&fixture.source).unwrap());
    let output = fixture.output("video-only.mp4");
    let report = transcode(
        &source,
        &output,
        TranscodeOptions {
            audio_tracks: Some(vec![]),
            max_b_frames: 0,
            preset: "ultrafast".into(),
            ..TranscodeOptions::default()
        },
    )
    .unwrap();
    assert_eq!(report.frames_encoded, 12);
    assert_eq!(report.video_packets_written, 12);
    assert_eq!(report.audio_packets_written, 0);
    assert_eq!(report.unselected_tracks, vec![TrackId(1)]);
    assert_eq!(
        probe(&Source::file(output).unwrap(), ProbeOptions::default())
            .unwrap()
            .tracks
            .len(),
        1
    );
}

#[test]
fn hevc_software_encoder_is_explicit_and_drains() {
    let Some(fixture) = Fixture::new() else {
        return;
    };
    let output = fixture.output("hevc.mp4");
    let report = transcode(
        &fixture.input(),
        &output,
        TranscodeOptions {
            codec: CpuVideoCodec::Hevc,
            profile: Some("main".into()),
            preset: "ultrafast".into(),
            max_b_frames: 2,
            audio_tracks: Some(vec![]),
            ..TranscodeOptions::default()
        },
    )
    .unwrap();
    assert_eq!(report.frames_encoded, 12);
    assert_eq!(report.video_packets_written, 12);
    let source = Source::file(output).unwrap();
    let info = probe(&source, ProbeOptions::default()).unwrap();
    assert_eq!(info.tracks[0].codec, "hevc");
    let mut reader = VideoReader::open(&source, VideoDecodeOptions::default()).unwrap();
    let mut count = 0;
    while reader.next_frame().unwrap().is_some() {
        count += 1;
    }
    assert_eq!(count, 12);
}

#[test]
fn preflight_failures_and_existing_destinations_do_not_create_or_overwrite_output() {
    let Some(fixture) = Fixture::new() else {
        return;
    };
    let incompatible = fixture.output("incompatible.webm");
    assert_eq!(
        transcode(&fixture.input(), &incompatible, TranscodeOptions::default())
            .unwrap_err()
            .kind(),
        ErrorKind::Incompatible
    );
    assert!(!incompatible.exists());
    let existing = fixture.output("existing.mp4");
    fs::write(&existing, b"original destination").unwrap();
    assert!(transcode(&fixture.input(), &existing, TranscodeOptions::default()).is_err());
    assert_eq!(fs::read(&existing).unwrap(), b"original destination");
    for (name, options) in [
        (
            "odd.mp4",
            TranscodeOptions {
                width: Some(31),
                height: Some(24),
                ..TranscodeOptions::default()
            },
        ),
        (
            "threads.mp4",
            TranscodeOptions {
                threads: 0,
                ..TranscodeOptions::default()
            },
        ),
        (
            "profile.mp4",
            TranscodeOptions {
                profile: Some("high444".into()),
                ..TranscodeOptions::default()
            },
        ),
        (
            "audio.mp4",
            TranscodeOptions {
                audio_tracks: Some(vec![TrackId(0)]),
                ..TranscodeOptions::default()
            },
        ),
        (
            "budget.mp4",
            TranscodeOptions {
                max_frame_bytes: 1,
                ..TranscodeOptions::default()
            },
        ),
    ] {
        let output = fixture.output(name);
        assert!(transcode(&fixture.input(), &output, options).is_err());
        assert!(!output.exists());
    }
}

#[test]
fn hardware_required_rejects_before_source_or_output_io() {
    let source = Source::memory(b"not a media file".to_vec());
    let output = std::env::temp_dir().join(format!("media-no-gpu-{}.mp4", std::process::id()));
    let error = transcode(
        &source,
        &output,
        TranscodeOptions {
            hardware_required: true,
            ..TranscodeOptions::default()
        },
    )
    .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(error.message().contains("NVENC"));
    assert!(!output.exists());
}

#[test]
fn pending_encoder_budget_failure_names_partial_destination() {
    let Some(fixture) = Fixture::new() else {
        return;
    };
    let output = fixture.output("partial.mp4");
    let error = transcode(
        &fixture.input(),
        &output,
        TranscodeOptions {
            max_pending_frames: 1,
            ..TranscodeOptions::default()
        },
    )
    .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::ResourceExhausted);
    assert_eq!(
        error.context().constraint.as_deref(),
        Some("max_pending_frames")
    );
    assert_eq!(error.context().source.as_deref(), output.to_str());
    assert!(
        error
            .context()
            .operation
            .as_deref()
            .unwrap()
            .contains("partial")
    );
    assert!(output.exists());
}

#[test]
fn nonzero_common_origin_is_preserved_in_both_tracks() {
    let Some(fixture) = Fixture::with_clock_offset(true) else {
        return;
    };
    let input = fixture.input();
    let output = fixture.output("source-clock.mp4");
    let report = transcode(&input, &output, TranscodeOptions::default()).unwrap();
    assert_eq!(report.source_origin, Some(Time::from_millis(2000)));
    let output = Source::file(output).unwrap();
    let original_video = packets(&input, TrackId(0));
    let encoded_video = packets(&output, TrackId(0));
    let mut original_pts = original_video
        .iter()
        .map(|packet| packet.pts)
        .collect::<Vec<_>>();
    let mut encoded_pts = encoded_video
        .iter()
        .map(|packet| packet.pts)
        .collect::<Vec<_>>();
    original_pts.sort();
    encoded_pts.sort();
    assert_eq!(original_pts, encoded_pts);
    assert_eq!(encoded_pts[0], Some(Time::from_millis(2000)));
    let original_audio = packets(&input, audio_track(&input));
    let copied_audio = packets(&output, audio_track(&output));
    assert_eq!(original_audio[0].pts, copied_audio[0].pts);
    assert!(copied_audio[0].pts.unwrap() > Time::from_millis(2200));
}

#[test]
fn precision_and_display_orientation_need_explicit_processing_policy() {
    let Some(fixture) = Fixture::new() else {
        return;
    };
    let high_depth = fixture.output("ten-bit.mkv");
    let generated = Command::new(&fixture.binary)
        .args(["-v", "error", "-nostdin", "-y", "-i"])
        .arg(&fixture.source)
        .args(["-an", "-c:v", "ffv1", "-pix_fmt", "yuv420p10le"])
        .arg(&high_depth)
        .output()
        .unwrap();
    assert!(
        generated.status.success(),
        "{}",
        String::from_utf8_lossy(&generated.stderr)
    );
    let output = fixture.output("must-not-truncate-depth.mp4");
    let error = transcode(
        &Source::file(high_depth).unwrap(),
        &output,
        TranscodeOptions::default(),
    )
    .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(error.message().contains("precision"));
    assert!(!output.exists());
    let rotated = fixture.output("rotated.mp4");
    let generated = Command::new(&fixture.binary)
        .args([
            "-v",
            "error",
            "-nostdin",
            "-y",
            "-display_rotation:v:0",
            "90",
            "-i",
        ])
        .arg(&fixture.source)
        .args(["-c", "copy"])
        .arg(&rotated)
        .output()
        .unwrap();
    assert!(
        generated.status.success(),
        "{}",
        String::from_utf8_lossy(&generated.stderr)
    );
    let output = fixture.output("must-not-drop-orientation.mp4");
    let error = transcode(
        &Source::file(rotated).unwrap(),
        &output,
        TranscodeOptions::default(),
    )
    .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(error.message().contains("orientation"));
    assert!(!output.exists());
}
