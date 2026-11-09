//! Real CPU facade admission, prompt decoder retirement and callback reentrance.
#![cfg(feature = "native")]

use media::{backend::native::image as codec, io::source::Source, prelude::*};
use std::{
    io::Cursor,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc,
    },
};

fn png() -> Vec<u8> {
    codec::encode(
        &Image::from_packed(2, 2, PixelFormat::Rgb8, vec![20; 12]).unwrap(),
        EncodeOptions::new(ImageFormat::Png),
    )
    .unwrap()
    .bytes
}

#[test]
fn image_factory_is_admitted_before_callbacks_and_retained_pixels_hold_no_codec() {
    let session = Session::builder()
        .limits(
            Limits::new()
                .cpu_executions(1)
                .cpu_codecs(1)
                .cpu_codec_threads(1),
        )
        .build()
        .unwrap();
    let bytes = png();
    let block = Arc::new(AtomicBool::new(false));
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let release_rx = Arc::new(Mutex::new(release_rx));
    let calls = Arc::new(AtomicUsize::new(0));
    let source = Source::from_seekable_factory(
        "cpu-image-callback",
        Some("v1".into()),
        Some(bytes.len() as u64),
        {
            let session = session.clone();
            let block = block.clone();
            let release = release_rx.clone();
            let calls = calls.clone();
            move || {
                calls.fetch_add(1, Ordering::SeqCst);
                let usage = session.live_cpu_usage();
                assert_eq!(
                    (usage.executions, usage.codecs, usage.codec_threads),
                    (1, 1, 1)
                );
                // Inspecting/reentering Session happens outside its admission mutex.
                assert_eq!(session.limits().cpu_executions, 1);
                let nested = session
                    .from_source(Source::memory(Vec::<u8>::new()))
                    .image()
                    .unwrap()
                    .probe()
                    .unwrap_err();
                assert_eq!(
                    nested.context().constraint.as_deref(),
                    Some("cpu_executions"),
                    "reentrant work fails nonblocking instead of locking recursively"
                );
                if block.load(Ordering::SeqCst) {
                    entered_tx.send(()).unwrap();
                    release
                        .lock()
                        .unwrap()
                        .recv_timeout(std::time::Duration::from_secs(10))
                        .unwrap();
                }
                Ok(Cursor::new(bytes.clone()))
            }
        },
    )
    .unwrap();
    let request = session.from_source(source).image().unwrap();
    let plan = request.compile().unwrap();
    block.store(true, Ordering::SeqCst);
    let running = {
        let plan = plan.clone();
        std::thread::spawn(move || plan.execute().unwrap())
    };
    entered_rx
        .recv_timeout(std::time::Duration::from_secs(10))
        .unwrap();
    let before = calls.load(Ordering::SeqCst);
    let error = plan.execute().unwrap_err();
    assert_eq!(error.kind(), ErrorKind::ResourceExhausted);
    assert_eq!(
        error.context().constraint.as_deref(),
        Some("cpu_executions")
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        before,
        "rejected execution cannot call its source factory"
    );
    block.store(false, Ordering::SeqCst);
    release_tx.send(()).unwrap();
    let first = running.join().unwrap().image;
    let second = plan.execute().unwrap().image;
    assert_eq!(session.live_cpu_usage().executions, 0);
    assert!(!first.storage().shares_allocation(second.storage()));
    assert!(
        session.live_host_bytes() > 0,
        "retained pixels are independent of codec admission"
    );
    // Compound convenience stages acquire sequential permits, never a parent
    // permit around a child. This must work even with capacity exactly one.
    request
        .encode(EncodeOptions::new(ImageFormat::Png))
        .unwrap();
}

#[cfg(feature = "ffmpeg")]
mod video {
    use super::*;
    use media::backend::ffmpeg::codec::VideoDecodeOptions;
    use std::{path::PathBuf, process::Command};

    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Option<Self> {
            let binary = std::env::var_os("MEDIA_FFMPEG_BIN").unwrap_or_else(|| "ffmpeg".into());
            if Command::new(&binary).arg("-version").output().is_err() {
                assert!(
                    std::env::var_os("MEDIA_REQUIRE_FFMPEG_TESTS").is_none(),
                    "FFmpeg fixture generator required"
                );
                eprintln!("UNEXERCISED: CPU codec fixture requires ffmpeg");
                return None;
            }
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            let path = std::env::temp_dir().join(format!(
                "media-cpu-admission-{}-{}.mp4",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::SeqCst)
            ));
            let output = Command::new(binary)
                .args([
                    "-v",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc2=size=32x32:rate=10:duration=0.4",
                    "-c:v",
                    "libx264",
                    "-threads:v",
                    "1",
                    "-bf",
                    "2",
                    "-pix_fmt",
                    "yuv420p",
                    "-colorspace",
                    "bt709",
                    "-color_range",
                    "tv",
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
    fn independent_video_readers_retire_on_eof_cancel_error_and_drop() {
        let Some(fixture) = Fixture::new() else {
            return;
        };
        let session = Session::builder()
            .limits(
                Limits::new()
                    .cpu_executions(2)
                    .cpu_codecs(2)
                    .cpu_codec_threads(2),
            )
            .build()
            .unwrap();
        let request = session.open(&fixture.0).unwrap().video().unwrap();
        let plan = request.compile().unwrap();
        let mut first = plan.reader().unwrap();
        let mut second = plan.reader().unwrap();
        assert_eq!(session.live_cpu_usage().codecs, 2);
        assert_eq!(
            request.reader().err().unwrap().kind(),
            ErrorKind::ResourceExhausted
        );
        let retained = first.next_frame().unwrap().unwrap();
        second.cancellation().cancel();
        assert_eq!(
            second.next_frame().unwrap_err().kind(),
            ErrorKind::Cancelled
        );
        assert_eq!(
            session.live_cpu_usage().codecs,
            1,
            "cancel retires without wrapper drop"
        );
        let third = request.reader().unwrap();
        drop(third);
        while first.next_frame().unwrap().is_some() {}
        assert_eq!(
            session.live_cpu_usage().codecs,
            0,
            "EOF retires native decoder while metadata wrapper remains"
        );
        assert_eq!(first.report().decoded_frames, 4);
        assert!(!first.info().tracks.is_empty());
        assert_eq!(retained.image.packed_bytes().unwrap().len(), 32 * 32 * 3);
        assert!(
            first.seek(Time::ZERO).is_err(),
            "retired CPU readers need a new execution for navigation"
        );

        let constrained = Session::builder()
            .limits(Limits::new().host_memory(32 * 32 * 3).cpu_executions(1))
            .build()
            .unwrap();
        let mut reader = constrained
            .open(&fixture.0)
            .unwrap()
            .video()
            .unwrap()
            .reader()
            .unwrap();
        let held = reader.next_frame().unwrap().unwrap();
        assert!(reader.next_frame().is_err());
        assert_eq!(
            constrained.live_cpu_usage().executions,
            0,
            "fatal frame allocation limit retires decoder immediately"
        );
        assert_eq!(held.image.width(), 32);
    }

    #[test]
    fn requested_thread_caps_and_atomic_two_codec_convert_fail_before_source_callbacks() {
        let session = Session::builder()
            .limits(Limits::new().cpu_codecs(1).cpu_codec_threads(1))
            .build()
            .unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let source = Source::from_seekable_factory("must-not-open", Some("v1".into()), Some(0), {
            let calls = calls.clone();
            move || {
                calls.fetch_add(1, Ordering::SeqCst);
                Ok(Cursor::new(Vec::<u8>::new()))
            }
        })
        .unwrap();
        let asset = session.from_source(source);
        let request = asset.video().unwrap().decode_options(VideoDecodeOptions {
            threads: 2,
            ..Default::default()
        });
        for error in [
            request.reader().err().unwrap(),
            request.compile().unwrap_err(),
            request.read().unwrap_err(),
        ] {
            assert_eq!(
                error.context().constraint.as_deref(),
                Some("cpu_codec_threads")
            );
        }
        let destination =
            std::env::temp_dir().join(format!("media-no-cpu-admission-{}.mp4", std::process::id()));
        assert_eq!(
            asset
                .convert()
                .write(&destination)
                .unwrap_err()
                .context()
                .constraint
                .as_deref(),
            Some("cpu_codecs")
        );
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert_eq!(session.live_cpu_usage().executions, 0);
        assert!(!destination.exists());
    }
}
