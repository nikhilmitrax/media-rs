//! Real linked-FFmpeg facade workflows with generated video fixtures.
#![cfg(feature = "ffmpeg")]
use media::prelude::*;
use std::{
    path::PathBuf,
    process::Command,
    sync::atomic::{AtomicUsize, Ordering},
};
fn fixture() -> Option<PathBuf> {
    if Command::new("ffmpeg").arg("-version").output().is_err() {
        assert!(
            std::env::var_os("MEDIA_REQUIRE_FFMPEG_TESTS").is_none(),
            "FFmpeg fixture generator is required"
        );
        eprintln!("UNEXERCISED: FFmpeg fixture generator unavailable");
        return None;
    }
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let path = std::env::temp_dir().join(format!(
        "media-video-facade-{}-{}.mp4",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let output = Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=16x16:rate=25",
            "-frames:v",
            "10",
            "-c:v",
            "libx264",
            "-bf",
            "2",
            "-g",
            "5",
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
    Some(path)
}
#[test]
fn actual_timestamp_batch_restores_order_and_shares_repeated_frame_storage() {
    let Some(path) = fixture() else {
        return;
    };
    let request = media::open(&path).unwrap().video().unwrap();
    let all = request.read().unwrap();
    assert_eq!(all.frames.len(), 10);
    let times = [
        Time::from_millis(200),
        Time::from_millis(40),
        Time::from_millis(200),
    ];
    let batch = request
        .sample(times)
        .selection(TemporalSelection::Nearest)
        .read()
        .unwrap();
    assert_eq!(batch.frames.len(), 3);
    assert_eq!(batch.report.decoded_frames, 7);
    assert!(
        batch.frames[0]
            .image
            .storage()
            .shares_allocation(batch.frames[2].image.storage())
    );
    assert_eq!(
        batch.frames[0].image.packed_bytes().unwrap(),
        all.frames[5].image.packed_bytes().unwrap()
    );
    assert_eq!(
        batch.frames[1].image.packed_bytes().unwrap(),
        all.frames[1].image.packed_bytes().unwrap()
    );
    let selection = batch.selection.unwrap();
    assert_eq!(selection.selections[0].requested, times[0]);
    assert_eq!(
        selection.selections[0].identity,
        selection.selections[2].identity
    );
    assert_eq!(selection.unique_frames.len(), 2);
    std::fs::remove_file(path).unwrap();
}
#[test]
fn direct_reader_drop_and_cancellation_preserve_owned_outputs() {
    let Some(path) = fixture() else {
        return;
    };
    let session = Session::default();
    let mut reader = session
        .open(&path)
        .unwrap()
        .video()
        .unwrap()
        .reader()
        .unwrap();
    let frame = reader.next_frame().unwrap().unwrap();
    reader.cancellation().cancel();
    assert_eq!(
        reader.next_frame().unwrap_err().kind(),
        ErrorKind::Cancelled
    );
    drop(reader);
    assert_eq!(frame.image.packed_bytes().unwrap().len(), 16 * 16 * 3);
    assert_eq!(session.live_host_bytes(), 768);
    drop(frame);
    assert_eq!(session.live_host_bytes(), 0);
    std::fs::remove_file(path).unwrap();
}
#[test]
fn remux_facade_finishes_and_does_not_overwrite_existing_output() {
    let Some(path) = fixture() else {
        return;
    };
    let output = path.with_extension("mkv");
    media::open(&path).unwrap().remux().write(&output).unwrap();
    assert_eq!(
        media::open(&output)
            .unwrap()
            .video()
            .unwrap()
            .read()
            .unwrap()
            .frames
            .len(),
        10
    );
    assert!(media::open(&path).unwrap().remux().write(&output).is_err());
    std::fs::remove_file(path).unwrap();
    std::fs::remove_file(output).unwrap();
}

#[test]
fn software_convert_facade_resizes_drains_and_rejects_hardware_substitution() {
    let Some(path) = fixture() else {
        return;
    };
    let output = path.with_extension("converted.mp4");
    let request = media::open(&path)
        .unwrap()
        .convert()
        .resize_components(Fit::inside(8, 8));
    assert!(request.clone().hardware_required().write(&output).is_err());
    assert!(!output.exists());
    let report = request.write(&output).unwrap();
    assert_eq!(report.frames_encoded, 10);
    let decoded = media::open(&output)
        .unwrap()
        .video()
        .unwrap()
        .read()
        .unwrap();
    assert_eq!(decoded.frames.len(), 10);
    assert_eq!(decoded.frames[0].image.width(), 8);
    std::fs::remove_file(path).unwrap();
    std::fs::remove_file(output).unwrap();
}
