//! Generated shared-container offset, VFR, fractional-grid, and boundary fixtures.
#![cfg(all(feature = "native", feature = "ffmpeg"))]

use media::{
    clip_request::{AudioOutput, Sampling, VideoOutput},
    prelude::*,
};
use std::{
    path::PathBuf,
    process::Command,
    sync::atomic::{AtomicUsize, Ordering},
};

struct Fixture(PathBuf);
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}
fn fixture(vfr: bool, multiple_audio: bool) -> Option<Fixture> {
    if Command::new("ffmpeg").arg("-version").output().is_err() {
        assert!(
            std::env::var_os("MEDIA_REQUIRE_FFMPEG_TESTS").is_none(),
            "FFmpeg fixture generator is required"
        );
        eprintln!("UNEXERCISED: FFmpeg clip fixture generator unavailable");
        return None;
    }
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let path = std::env::temp_dir().join(format!(
        "media-clips-{}-{}.mkv",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let mut command = Command::new("ffmpeg");
    command.args([
        "-nostdin",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=16x16:rate=25:duration=0.4",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:sample_rate=48000:duration=0.3",
    ]);
    if multiple_audio {
        command.args([
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=880:sample_rate=48000:duration=0.3",
        ]);
    }
    let mut filters = if vfr {
        "[0:v]select=lt(n\\,5),settb=1/1000,setpts=PTS+gte(N\\,2)*40+gte(N\\,3)*80+500[v];[1:a]asetpts=PTS+0.5/TB[a]".to_owned()
    } else {
        "[0:v]settb=1/1000,setpts=PTS+500[v];[1:a]asetpts=PTS+0.6/TB[a]".to_owned()
    };
    if multiple_audio {
        filters.push_str(";[2:a]asetpts=PTS+0.7/TB[b]");
    }
    command.args(["-filter_complex", &filters, "-map", "[v]", "-map", "[a]"]);
    if multiple_audio {
        command.args(["-map", "[b]"]);
    }
    command
        .args([
            "-c:v",
            "ffv1",
            "-c:a",
            "pcm_f32le",
            "-fps_mode:v",
            "vfr",
            "-enc_time_base:v",
            "1/1000",
            "-colorspace",
            "bt709",
            "-color_range",
            "tv",
            "-copyts",
            "-avoid_negative_ts",
            "disabled",
            "-y",
        ])
        .arg(&path);
    let result = command.output().unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    Some(Fixture(path))
}
fn range(start: i64, end: i64) -> TimeRange {
    TimeRange::new(Time::from_millis(start), Time::from_millis(end)).unwrap()
}
fn video_output(count: usize) -> VideoOutput {
    VideoOutput::new()
        .track(TrackId(0))
        .sample(Sampling::uniform(count))
        // These generated Matroska fixtures intentionally model each native
        // reported 40ms interval, including gaps. Trust is explicit here.
        .decode_options(media::backend::ffmpeg::codec::VideoDecodeOptions {
            duration_policy: media::core::policy::FrameDurationPolicy::TrustReported,
            ..Default::default()
        })
}
fn audio(track: u32) -> AudioOutput {
    AudioOutput::f32().track(TrackId(track)).sample_rate(16000)
}

#[test]
fn shared_nonzero_container_origin_preserves_stream_offset_and_preclip_display_pts() {
    let Some(fixture) = fixture(false, false) else {
        return;
    };
    let session = Session::default();
    let asset = session.open(&fixture.0).unwrap();
    let requested = range(150, 250);
    let clip = asset
        .clip(requested)
        .video(video_output(4))
        .audio(audio(1))
        .read()
        .unwrap();
    assert_eq!(clip.timeline_origin, Time::from_millis(500));
    assert_eq!(clip.requested, requested);
    assert_eq!(clip.actual_coverage, vec![requested]);
    let video = clip.video.as_ref().unwrap();
    assert_eq!(
        video.requested_times,
        vec![
            Time::from_millis(150),
            Time::from_millis(175),
            Time::from_millis(200),
            Time::from_millis(225)
        ]
    );
    assert_eq!(
        video.frames[0].frame.timing.pts,
        Some(Time::from_millis(120))
    );
    assert_eq!(
        video.frames[0].frame.timing.source_pts,
        Some(Time::from_millis(620))
    );
    assert!(
        video
            .frames
            .iter()
            .all(|frame| frame.valid && !frame.repeated)
    );
    let audio = clip.audio.as_ref().unwrap();
    assert_eq!(audio.sample_grid_origin, Time::from_millis(100));
    assert_eq!(audio.source_grid_origin, Time::from_millis(600));
    assert_eq!(audio.requested_sample_bounds, 800..2400);
    assert_eq!(audio.block.start(), Time::from_millis(150));
    assert_eq!(audio.block.frames(), 1600);
    assert!(audio.validity.iter().all(|v| *v));
    assert!(audio.padding.iter().all(|v| !*v));
    let expected = asset
        .audio()
        .unwrap()
        .decode_options(media::backend::native::audio::AudioDecodeOptions {
            track: Some(audio.native_track),
            ..Default::default()
        })
        .resample(16000)
        .slice(range(650, 750))
        .read()
        .unwrap();
    assert_eq!(audio.block.samples(), expected.samples());
    let retained = audio.block.clone();
    drop(expected);
    drop(clip);
    assert_eq!(session.live_host_bytes(), retained.samples().len() * 4);
    assert_eq!(retained.start(), Time::from_millis(150));
    drop(retained);
    assert_eq!(session.live_host_bytes(), 0);
}

#[test]
fn adjacent_fractional_clips_share_one_continuous_resampler_history_and_sample_grid() {
    let Some(fixture) = fixture(false, false) else {
        return;
    };
    let asset = media::open(&fixture.0).unwrap();
    let start = Time::new(4801, 32000).unwrap();
    let split = Time::new(6401, 32000).unwrap();
    let end = Time::new(8001, 32000).unwrap();
    let full = asset
        .clip(TimeRange::new(start, end).unwrap())
        .audio(audio(1))
        .read()
        .unwrap()
        .audio
        .unwrap();
    let left = asset
        .clip(TimeRange::new(start, split).unwrap())
        .audio(audio(1))
        .read()
        .unwrap()
        .audio
        .unwrap();
    let right = asset
        .clip(TimeRange::new(split, end).unwrap())
        .audio(audio(1))
        .read()
        .unwrap()
        .audio
        .unwrap();
    assert_eq!(full.requested_sample_bounds, 801..2401);
    assert_eq!(
        left.returned_sample_bounds.end,
        right.returned_sample_bounds.start
    );
    assert_eq!(left.block.end().unwrap(), right.block.start());
    assert_eq!(full.block.start(), Time::new(2401, 16000).unwrap());
    assert_eq!(
        [left.block.samples(), right.block.samples()].concat(),
        full.block.samples()
    );
    assert_eq!(left.block.frames() + right.block.frames(), 1600);
}

#[test]
fn explicit_silence_and_repeat_masks_preserve_requested_timeline_and_real_common_coverage() {
    let Some(fixture) = fixture(false, false) else {
        return;
    };
    let asset = media::open(&fixture.0).unwrap();
    let request = asset
        .clip(range(0, 450))
        .video(video_output(9))
        .audio(audio(1));
    assert_eq!(request.read().unwrap_err().kind(), ErrorKind::MissingData);
    let clip = request
        .video_boundary(Boundary::RepeatFrame)
        .audio_boundary(Boundary::Silence)
        .read()
        .unwrap();
    assert_eq!(clip.requested, range(0, 450));
    assert_eq!(clip.actual_coverage, vec![range(100, 400)]);
    let video = clip.video.unwrap();
    assert_eq!(video.frames.len(), 9);
    assert_eq!(
        video.repeated,
        vec![false, false, false, false, false, false, false, false, true]
    );
    assert_eq!(
        video.frames[8].frame.timing.pts,
        Some(Time::from_millis(360))
    );
    assert_eq!(video.frames[8].requested, Time::from_millis(400));
    let audio = clip.audio.unwrap();
    assert_eq!(audio.block.start(), Time::ZERO);
    assert_eq!(audio.block.frames(), 7200);
    assert_eq!(audio.returned_sample_bounds, -1600..5600);
    assert_eq!(audio.validity.iter().filter(|v| **v).count(), 4800);
    assert_eq!(audio.padding.iter().filter(|v| **v).count(), 2400);
    assert!(audio.block.samples()[..1600].iter().all(|v| *v == 0.0));
    assert!(audio.block.samples()[6400..].iter().all(|v| *v == 0.0));
    assert_eq!(audio.block.source_start(), None);
    assert_eq!(audio.source_coverage, range(600, 900));
}

#[test]
fn truncate_omits_missing_selections_without_resampling_the_shorter_interval() {
    let Some(fixture) = fixture(false, false) else {
        return;
    };
    let clip = media::open(&fixture.0)
        .unwrap()
        .clip(range(0, 450))
        .video(video_output(9))
        .audio(audio(1))
        .boundary(Boundary::Truncate)
        .read()
        .unwrap();
    assert_eq!(clip.requested, range(0, 450));
    assert_eq!(clip.actual_coverage, vec![range(100, 400)]);
    let video = clip.video.unwrap();
    assert_eq!(video.frames.len(), 8);
    assert_eq!(video.requested_times.len(), 9);
    assert_eq!(video.frames.last().unwrap().request_index, 7);
    assert_eq!(
        video.frames.last().unwrap().requested,
        Time::from_millis(350)
    );
    assert!(!video.validity[8]);
    let audio = clip.audio.unwrap();
    assert_eq!(audio.requested_sample_bounds, -1600..5600);
    assert_eq!(audio.returned_sample_bounds, 0..4800);
    assert_eq!(audio.block.start(), Time::from_millis(100));
    assert_eq!(audio.block.frames(), 4800);
    assert!(audio.validity.iter().all(|v| *v));
    assert!(audio.padding.iter().all(|v| !*v));
}

#[test]
fn vfr_display_selection_uses_established_intervals_and_reports_internal_video_absence() {
    let Some(fixture) = fixture(true, false) else {
        return;
    };
    let asset = media::open(&fixture.0).unwrap();
    // One selected time is valid, but Error must still find the unsampled gap.
    assert_eq!(
        asset
            .clip(range(20, 300))
            .video(video_output(1))
            .read()
            .unwrap_err()
            .kind(),
        ErrorKind::MissingData
    );
    let clip = asset
        .clip(range(20, 300))
        .video(video_output(7))
        .video_boundary(Boundary::RepeatFrame)
        .read()
        .unwrap();
    let video = clip.video.unwrap();
    assert_eq!(
        video.validity,
        vec![true, true, false, true, false, false, true]
    );
    assert_eq!(
        video
            .frames
            .iter()
            .map(|frame| frame.selection.identity.ordinal)
            .collect::<Vec<_>>(),
        vec![0, 1, 1, 2, 2, 2, 3]
    );
    assert_eq!(
        video.actual_coverage,
        vec![range(20, 80), range(120, 160), range(240, 300)]
    );
    let trimmed = asset
        .clip(range(20, 300))
        .video(video_output(7))
        .video_boundary(Boundary::Truncate)
        .read()
        .unwrap()
        .video
        .unwrap();
    assert_eq!(
        trimmed
            .frames
            .iter()
            .map(|frame| frame.request_index)
            .collect::<Vec<_>>(),
        vec![0, 1, 3, 6]
    );
}

#[test]
fn explicit_audio_track_mapping_never_confuses_stream_index_with_native_track_number() {
    let Some(fixture) = fixture(false, true) else {
        return;
    };
    let asset = media::open(&fixture.0).unwrap();
    let clip = asset.clip(range(220, 240)).audio(audio(2)).read().unwrap();
    let audio = clip.audio.unwrap();
    assert_eq!(audio.track, TrackId(2));
    assert_ne!(audio.native_track, audio.track.0);
    assert_eq!(audio.sample_grid_origin, Time::from_millis(200));
    let expected = asset
        .audio()
        .unwrap()
        .decode_options(media::backend::native::audio::AudioDecodeOptions {
            track: Some(audio.native_track),
            ..Default::default()
        })
        .resample(16000)
        .slice(range(720, 740))
        .read()
        .unwrap();
    assert_eq!(audio.block.samples(), expected.samples());
}

#[test]
fn empty_requests_cancellation_and_budget_errors_leave_no_retained_execution_work() {
    let Some(fixture) = fixture(false, false) else {
        return;
    };
    let session = Session::builder()
        .limits(Limits::new().host_memory(256 * 1024))
        .build()
        .unwrap();
    let asset = session.open(&fixture.0).unwrap();
    let empty = asset
        .clip(range(500, 500))
        .video(video_output(3))
        .audio(audio(1))
        .read()
        .unwrap();
    assert!(empty.actual_coverage.is_empty());
    assert!(empty.video.as_ref().unwrap().frames.is_empty());
    assert_eq!(empty.audio.as_ref().unwrap().block.frames(), 0);
    drop(empty);
    assert_eq!(session.live_host_bytes(), 0);
    let cancelled = asset.clip(range(150, 250)).audio(audio(1));
    cancelled.cancellation().cancel();
    assert_eq!(cancelled.read().unwrap_err().kind(), ErrorKind::Cancelled);
    assert_eq!(session.live_host_bytes(), 0);
    let oversized = asset
        .clip(range(0, 600000))
        .audio(audio(1))
        .audio_boundary(Boundary::Silence);
    assert_eq!(
        oversized.read().unwrap_err().kind(),
        ErrorKind::ResourceExhausted
    );
    assert_eq!(session.live_host_bytes(), 0);
    assert!(asset.clip(range(150, 250)).audio(audio(1)).read().is_ok());
}

#[test]
fn invalid_tracks_and_undefined_boundary_operations_fail_explicitly() {
    let Some(fixture) = fixture(false, false) else {
        return;
    };
    let asset = media::open(&fixture.0).unwrap();
    assert!(
        asset
            .clip(range(150, 250))
            .video(VideoOutput::new())
            .read()
            .is_err()
    );
    assert!(
        asset
            .clip(range(150, 250))
            .audio(AudioOutput::f32())
            .read()
            .is_err()
    );
    assert_eq!(
        asset
            .clip(range(150, 250))
            .audio(audio(0))
            .read()
            .unwrap_err()
            .kind(),
        ErrorKind::Unsupported
    );
    assert_eq!(
        asset
            .clip(range(150, 250))
            .video(video_output(2))
            .video_boundary(Boundary::Wrap)
            .read()
            .unwrap_err()
            .kind(),
        ErrorKind::Unsupported
    );
    assert_eq!(
        asset
            .clip(range(150, 250))
            .audio(audio(1))
            .audio_boundary(Boundary::RepeatFrame)
            .read()
            .unwrap_err()
            .kind(),
        ErrorKind::Unsupported
    );
    assert_eq!(
        asset
            .clip(range(150, 250))
            .video(video_output(2))
            .video_boundary(Boundary::Silence)
            .read()
            .unwrap_err()
            .kind(),
        ErrorKind::Unsupported
    );
}

#[test]
fn wholly_missing_and_fractional_tail_audio_have_explicit_grid_and_padding_metadata() {
    let Some(fixture) = fixture(false, false) else {
        return;
    };
    let asset = media::open(&fixture.0).unwrap();
    let missing = asset
        .clip(range(500, 600))
        .audio(audio(1))
        .audio_boundary(Boundary::Truncate)
        .read()
        .unwrap()
        .audio
        .unwrap();
    assert_eq!(missing.block.frames(), 0);
    assert_eq!(missing.block.start(), Time::from_millis(500));
    assert_eq!(missing.returned_sample_bounds, 6400..6400);
    assert!(missing.actual_coverage.is_empty());
    let tail = TimeRange::new(
        Time::new(39999, 100000).unwrap(),
        Time::new(40001, 100000).unwrap(),
    )
    .unwrap();
    let request = asset.clip(tail).audio(audio(1));
    assert_eq!(request.read().unwrap_err().kind(), ErrorKind::MissingData);
    let padded = request
        .clone()
        .audio_boundary(Boundary::Silence)
        .read()
        .unwrap()
        .audio
        .unwrap();
    assert_eq!(padded.block.frames(), 1);
    assert_eq!(padded.block.start(), Time::from_millis(400));
    assert_eq!(padded.block.samples(), &[0.0]);
    assert_eq!(padded.validity, vec![false]);
    assert_eq!(padded.padding, vec![true]);
    assert_eq!(
        padded.actual_coverage,
        vec![TimeRange::new(tail.start(), Time::from_millis(400)).unwrap()]
    );
    let trimmed = request
        .audio_boundary(Boundary::Truncate)
        .read()
        .unwrap()
        .audio
        .unwrap();
    assert_eq!(trimmed.block.frames(), 0);
    assert_eq!(trimmed.returned_sample_bounds, 4800..4800);
}

#[test]
fn unknown_speaker_positions_require_explicit_mapping_and_cross_modal_failure_retires_audio() {
    use media::backend::native::audio::MixMatrix;
    let Some(fixture) = fixture(true, false) else {
        return;
    };
    let session = Session::default();
    let asset = session.open(&fixture.0).unwrap();
    assert!(
        asset
            .clip(range(20, 60))
            .audio(audio(1).channels(ChannelLayout::mono()))
            .read()
            .is_err()
    );
    assert_eq!(session.live_host_bytes(), 0);
    let matrix = MixMatrix::new(
        ChannelLayout::new(vec![ChannelPosition::Discrete(0)]).unwrap(),
        ChannelLayout::mono(),
        vec![1.0],
    )
    .unwrap();
    let prepared = audio(1).mix(matrix);
    let clip = asset
        .clip(range(20, 60))
        .audio(prepared.clone())
        .read()
        .unwrap();
    assert_eq!(
        clip.audio.as_ref().unwrap().block.channel_layout(),
        &ChannelLayout::mono()
    );
    drop(clip);
    assert_eq!(session.live_host_bytes(), 0);
    assert_eq!(
        asset
            .clip(range(20, 280))
            .audio(prepared)
            .video(video_output(2))
            .read()
            .unwrap_err()
            .kind(),
        ErrorKind::MissingData
    );
    assert_eq!(session.live_host_bytes(), 0);
}

#[test]
fn compressed_codec_priming_is_rejected_until_a_cross_backend_trim_mapping_exists() {
    let Some(input) = fixture(false, false) else {
        return;
    };
    let output = Fixture(input.0.with_extension("aac.mkv"));
    let result = Command::new("ffmpeg")
        .args(["-nostdin", "-v", "error", "-i"])
        .arg(&input.0)
        .args(["-c:v", "copy", "-c:a", "aac", "-y"])
        .arg(&output.0)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let error = media::open(&output.0)
        .unwrap()
        .clip(range(150, 250))
        .audio(audio(1))
        .read()
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(error.message().contains("priming"));
}

#[test]
fn clip_batch_slot_is_shared_by_both_modalities_and_retained_extracted_aliases() {
    let Some(fixture) = fixture(false, false) else {
        return;
    };
    let session = Session::builder()
        .limits(Limits::new().in_flight_batches(1))
        .build()
        .unwrap();
    let request = session
        .open(&fixture.0)
        .unwrap()
        .clip(range(150, 250))
        .video(video_output(2))
        .audio(audio(1));
    let clip = request.read().unwrap();
    assert_eq!(session.live_batches(), 1);
    let image = clip.video.as_ref().unwrap().frames[0].frame.image.clone();
    let audio = clip.audio.as_ref().unwrap().block.clone();
    drop(clip);
    assert_eq!(session.live_batches(), 1);
    assert_eq!(
        request.read().unwrap_err().kind(),
        ErrorKind::ResourceExhausted
    );
    drop(image);
    assert_eq!(session.live_batches(), 1);
    drop(audio);
    assert_eq!(session.live_batches(), 0);
    assert_eq!(session.live_host_bytes(), 0);
    let next = request.read().unwrap();
    drop(next);
    assert_eq!(session.live_batches(), 0);
}

#[test]
fn ambiguous_audio_identity_is_rejected_instead_of_assuming_track_order() {
    let Some(input) = fixture(false, false) else {
        return;
    };
    let output = Fixture(input.0.with_extension("identical-tracks.mkv"));
    let result = Command::new("ffmpeg")
        .args(["-nostdin", "-v", "error", "-i"])
        .arg(&input.0)
        .args([
            "-map", "0:v:0", "-map", "0:a:0", "-map", "0:a:0", "-c", "copy", "-copyts", "-y",
        ])
        .arg(&output.0)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let session = Session::default();
    let error = session
        .open(&output.0)
        .unwrap()
        .clip(range(150, 250))
        .audio(audio(1))
        .read()
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::AmbiguousTiming);
    assert_eq!(session.live_host_bytes(), 0);
    assert_eq!(session.live_batches(), 0);
}

#[test]
fn lossless_flac_packet_and_timeline_identity_is_verified_for_actual_clip_decoding() {
    let Some(input) = fixture(false, false) else {
        return;
    };
    let output = Fixture(input.0.with_extension("flac.mkv"));
    let result = Command::new("ffmpeg")
        .args(["-nostdin", "-v", "error", "-i"])
        .arg(&input.0)
        .args(["-c:v", "copy", "-c:a", "flac", "-copyts", "-y"])
        .arg(&output.0)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let clip = media::open(&output.0)
        .unwrap()
        .clip(range(150, 250))
        .audio(audio(1))
        .read()
        .unwrap();
    let audio = clip.audio.unwrap();
    assert_eq!(audio.sample_grid_origin, Time::from_millis(100));
    assert_eq!(audio.block.frames(), 1600);
    assert!(audio.validity.iter().all(|valid| *valid));
}

#[test]
fn unversioned_cursors_cannot_silently_claim_a_shared_recording_clock() {
    let source =
        media::io::source::Source::from_reader_factory("unversioned capture", None, None, || {
            Ok(std::io::Cursor::new(Vec::<u8>::new()))
        })
        .unwrap();
    let session = Session::default();
    let error = session
        .from_source(source)
        .clip(range(0, 100))
        .audio(audio(1))
        .read()
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(error.message().contains("stable source version"));
    assert_eq!(session.live_batches(), 0);
    assert_eq!(session.live_host_bytes(), 0);
}
