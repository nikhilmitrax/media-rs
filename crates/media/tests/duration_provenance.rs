//! Real reordered VFR media separates native duration hints from display intervals.
#![cfg(feature = "ffmpeg")]

use media::{
    backend::ffmpeg::{
        codec::VideoDecodeOptions,
        format::{PacketReader, ProbeOptions},
    },
    core::policy::FrameDurationPolicy,
    io::source::Source,
    prelude::*,
};
use std::{
    path::PathBuf,
    process::Command,
    sync::atomic::{AtomicU64, Ordering},
};

struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Option<Self> {
        let binary = std::env::var_os("MEDIA_FFMPEG_BIN").unwrap_or_else(|| "ffmpeg".into());
        if Command::new(&binary).arg("-version").output().is_err() {
            assert!(
                std::env::var_os("MEDIA_REQUIRE_FFMPEG_TESTS").is_none(),
                "FFmpeg fixture generator is required"
            );
            eprintln!("UNEXERCISED: FFmpeg fixture generator unavailable");
            return None;
        }
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "media-duration-{}-{}.mp4",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let output = Command::new(binary)
            .args([
                "-v",
                "error",
                "-nostdin",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=32x32:rate=10",
                "-vf",
                "select=not(eq(n\\,3)+eq(n\\,4)+eq(n\\,7))",
                "-frames:v",
                "12",
                "-fps_mode",
                "vfr",
                "-c:v",
                "libx264",
                "-threads:v",
                "1",
                "-bf",
                "3",
                "-g",
                "30",
                "-x264-params",
                "scenecut=0",
                "-pix_fmt",
                "yuv420p",
                "-colorspace",
                "bt709",
                "-color_range",
                "tv",
                "-video_track_timescale",
                "1000",
                "-y",
            ])
            .arg(&path)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        Some(Self(path))
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

#[test]
fn reordered_vfr_mp4_requires_explicit_interval_interpretation() {
    let Some(fixture) = Fixture::new() else {
        return;
    };
    let source = Source::file(&fixture.0).unwrap();
    let mut packets = PacketReader::open(&source, ProbeOptions::default()).unwrap();
    let mut reordered = false;
    while let Some(packet) = packets.next_packet().unwrap() {
        reordered |= packet.pts != packet.dts;
    }
    assert!(reordered, "fixture must contain genuine B-frame reordering");

    let request = media::open(&fixture.0).unwrap().video().unwrap();
    let all = request.read().unwrap();
    assert_eq!(all.frames.len(), 12);
    assert_eq!(all.report.duration_policy, FrameDurationPolicy::Unknown);
    assert!(
        all.frames
            .iter()
            .all(|frame| frame.timing.duration.is_none())
    );
    assert!(
        all.frames
            .iter()
            .any(|frame| frame.timing.native_duration.is_some())
    );
    assert_eq!(all.frames[2].timing.pts, Some(Time::from_millis(200)));
    assert_eq!(all.frames[3].timing.pts, Some(Time::from_millis(500)));

    let time = Time::from_millis(350);
    let strict = request.clone().sample([time]);
    let plan = strict.compile().unwrap();
    assert_eq!(plan.duration_policy(), FrameDurationPolicy::Unknown);
    assert_eq!(
        plan.duration_inference(),
        DurationInference::RequireExplicit
    );
    assert_eq!(plan.execute().unwrap_err().kind(), ErrorKind::MissingData);

    let inferred = strict
        .duration_inference(DurationInference::NextTimestamp)
        .read()
        .unwrap();
    assert_eq!(inferred.frames[0].timing.pts, Some(Time::from_millis(200)));
    assert_eq!(
        inferred.frames[0].timing.duration, None,
        "selection inference must not rewrite source timing evidence"
    );
    assert!(inferred.selection.as_ref().unwrap().selections[0].duration_inferred);
    assert_eq!(
        inferred.report.duration_policy,
        FrameDurationPolicy::Unknown
    );

    let trusted = request
        .decode_options(VideoDecodeOptions {
            duration_policy: FrameDurationPolicy::TrustReported,
            ..Default::default()
        })
        .read()
        .unwrap();
    assert_eq!(
        trusted.report.duration_policy,
        FrameDurationPolicy::TrustReported
    );
    for (unknown, interpreted) in all.frames.iter().zip(&trusted.frames) {
        assert_eq!(interpreted.timing.duration, unknown.timing.native_duration);
        assert_eq!(
            interpreted.timing.native_duration,
            unknown.timing.native_duration
        );
        assert_eq!(interpreted.timing.source_pts, unknown.timing.source_pts);
    }
}
