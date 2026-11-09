//! Shared FFmpeg MOV/AAC semantics, continuous DSP, clocks and facade admission.
#![cfg(feature = "ffmpeg-audio")]
use media::{
    Session,
    audio_request::{AudioBackend, AudioRequest},
    backend::{
        ffmpeg::audio::{AudioDecodeOptions, AudioTimeline},
        native::audio::WavWriter,
    },
    clip_request::{AudioOutput, Sampling, VideoOutput},
    io::{sink::Sink, source::Source},
    prelude::*,
};
use std::{
    fs,
    io::Cursor,
    path::PathBuf,
    process::Command,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};
struct Fixture {
    directory: PathBuf,
    path: PathBuf,
}
fn binary() -> std::ffi::OsString {
    std::env::var_os("MEDIA_FFMPEG_BIN").unwrap_or_else(|| "ffmpeg".into())
}
impl Fixture {
    fn new(video: bool, offset: bool) -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let directory = std::env::temp_dir().join(format!(
            "media-facade-ffaudio-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&directory).unwrap();
        let path = directory.join(if video { "clip.mp4" } else { "audio.m4a" });
        let mut command = Command::new(binary());
        command.args(["-nostdin", "-v", "error"]);
        if video {
            command.args([
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=16x16:rate=25:duration=0.4",
            ]);
        }
        command.args([
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=0.101",
        ]);
        if video {
            if offset {
                command.args([
                    "-filter_complex",
                    "[0:v]setpts=PTS+0.52/TB[v];[1:a]asetpts=PTS+0.72/TB[a]",
                    "-map",
                    "[v]",
                    "-map",
                    "[a]",
                ]);
            } else {
                command.args(["-map", "0:v:0", "-map", "1:a:0"]);
            }
            command.args([
                "-c:v",
                "libx264",
                "-threads:v",
                "1",
                "-bf",
                "0",
                "-color_primaries",
                "bt709",
                "-color_trc",
                "bt709",
                "-colorspace",
                "bt709",
                "-color_range",
                "tv",
            ]);
        }
        command
            .args([
                "-c:a",
                "aac",
                "-threads:a",
                "1",
                "-copyts",
                "-avoid_negative_ts",
                "disabled",
                "-y",
            ])
            .arg(&path);
        let output = command.output().expect("FFmpeg fixture generator required");
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        Self { directory, path }
    }
    fn reference(&self, semantic_frames: Option<usize>) -> Vec<f32> {
        let mut command = Command::new(binary());
        command
            .args(["-nostdin", "-v", "error", "-i"])
            .arg(&self.path)
            .args(["-map", "0:a:0", "-threads:a", "1"]);
        if let Some(frames) = semantic_frames {
            command.args(["-af", &format!("atrim=end_sample={frames}")]);
        }
        let output = command.args(["-f", "f32le", "-"]).output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        output
            .stdout
            .as_chunks::<4>()
            .0
            .iter()
            .map(|bytes| f32::from_le_bytes(*bytes))
            .collect()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.directory);
    }
}
fn requested() -> TimeRange {
    TimeRange::new(
        Time::new(1001, 48000).unwrap(),
        Time::new(3501, 48000).unwrap(),
    )
    .unwrap()
}
fn request(asset: &media::request::Asset, block: usize) -> AudioRequest {
    asset.audio().unwrap().ffmpeg_options(AudioDecodeOptions {
        max_block_frames: block,
        ..Default::default()
    })
}
fn assert_idle(session: &Session) {
    assert_eq!(
        session.live_cpu_usage(),
        media::pipeline::session::CpuUsage::default()
    );
    assert_eq!(session.live_host_bytes(), 0);
}

#[test]
fn native_mp4_requires_explicit_raw_while_ffmpeg_semantic_matches_cli_trim_once() {
    let fixture = Fixture::new(false, false);
    let session = Session::default();
    let asset = session.open(&fixture.path).unwrap();
    let error = asset.audio().unwrap().read().unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(error.to_string().contains("ffmpeg"));
    let native_raw = asset.audio().unwrap().raw_decoded().read().unwrap();
    assert!(native_raw.frames() >= 4848);
    drop(native_raw);
    let plan = request(&asset, 137).compile().unwrap();
    assert!(plan.explain().contains("ffmpeg"));
    assert_idle(&session);
    assert_eq!(plan.source_info().frames, Some(4848));
    let decoded = plan.execute().unwrap();
    assert_eq!(decoded.samples(), fixture.reference(Some(4848)));
    assert_eq!(decoded.frames(), 4848);
    drop(decoded);
    let raw = request(&asset, 4096).raw_decoded().read().unwrap();
    assert_eq!(raw.samples(), fixture.reference(None));
    assert_eq!(raw.frames(), 5120);
    drop(raw);
    let mut reader = request(&asset, 137).reader().unwrap();
    while reader.next_block().unwrap().is_some() {}
    let report = reader.ffmpeg_decode_report().unwrap();
    assert_eq!(report.codec_skip_start_requests, 1024);
    assert_eq!(report.semantic_trim_end, 272);
    assert_eq!(report.semantic_trim_start, 0);
    assert!(report.finished);
    assert_idle(&session);
}

#[test]
fn continuous_resampling_fractional_slices_and_independent_chunk_sizes_share_one_grid() {
    let fixture = Fixture::new(false, false);
    let session = Session::default();
    let asset = session.open(&fixture.path).unwrap();
    let all = request(&asset, 4096).resample(16000).read().unwrap();
    assert_eq!(all.frames(), 1616);
    let chunked = request(&asset, 137).resample(16000).read().unwrap();
    assert_eq!(chunked.samples(), all.samples());
    let sliced = request(&asset, 1)
        .resample(16000)
        .slice(requested())
        .read()
        .unwrap();
    assert_eq!(sliced.start(), Time::from_samples(334, 16000).unwrap());
    assert_eq!(sliced.frames(), 833);
    assert_eq!(sliced.samples(), &all.samples()[334..1167]);
    drop((all, chunked, sliced));
    assert_idle(&session);
}

#[test]
fn common_mp4_clip_uses_one_ffmpeg_clock_and_gapless_audio_without_native_track_mapping() {
    let fixture = Fixture::new(true, false);
    let session = Session::builder()
        .limits(
            Limits::new()
                .cpu_executions(1)
                .cpu_codecs(1)
                .cpu_codec_threads(1)
                .in_flight_batches(1),
        )
        .build()
        .unwrap();
    let asset = session.open(&fixture.path).unwrap();
    let clip = asset
        .clip(requested())
        .video(
            VideoOutput::new()
                .track(TrackId(0))
                .sample(Sampling::uniform(4))
                .duration_inference(DurationInference::NextTimestamp),
        )
        .audio(AudioOutput::f32().track(TrackId(1)).sample_rate(16000))
        .read()
        .unwrap();
    assert_eq!(clip.timeline_origin, Time::ZERO);
    assert_eq!(clip.actual_coverage, [requested()]);
    assert_eq!(clip.audio.as_ref().unwrap().backend, AudioBackend::Ffmpeg);
    let audio = clip.audio.as_ref().unwrap();
    assert_eq!(audio.requested_sample_bounds, 334..1167);
    assert_eq!(audio.block.frames(), 833);
    assert_eq!(
        audio
            .ffmpeg_report
            .as_ref()
            .unwrap()
            .codec_skip_start_requests,
        1024
    );
    assert!(audio.validity.iter().all(|valid| *valid));
    assert!(audio.padding.iter().all(|pad| !*pad));
    let reference = request(&asset, 137).resample(16000).read().unwrap();
    assert_eq!(audio.block.samples(), &reference.samples()[334..1167]);
    drop(reference);
    let video = clip.video.as_ref().unwrap();
    assert!(video.frames[0].selection.actual < requested().start());
    for frame in &video.frames {
        assert_eq!(frame.frame.timing.source_pts, frame.frame.timing.pts);
    }
    let retained = audio.block.clone();
    drop(clip);
    assert_eq!(session.live_batches(), 1);
    assert_eq!(session.live_cpu_usage().executions, 0);
    drop(retained);
    assert_eq!(session.live_batches(), 0);
    assert_idle(&session);
}

#[test]
fn declared_offset_source_and_asset_clocks_are_explicit_and_clip_keeps_raw_source_pts() {
    let fixture = Fixture::new(true, true);
    let session = Session::default();
    let asset = session.open(&fixture.path).unwrap();
    let mut reader = request(&asset, 137).reader().unwrap();
    let info = reader.ffmpeg_info().unwrap().clone();
    let first = reader.next_block().unwrap().unwrap();
    assert_eq!(first.start(), info.source_start);
    assert_eq!(first.source_start(), Some(info.source_start));
    drop(first);
    drop(reader);
    let origin = info.timeline_origin.unwrap();
    let grid = info.source_start.checked_sub(origin).unwrap();
    let source_range = TimeRange::new(
        info.source_start.checked_add(requested().start()).unwrap(),
        info.source_start.checked_add(requested().end()).unwrap(),
    )
    .unwrap();
    let asset_range = TimeRange::new(
        source_range.start().checked_sub(origin).unwrap(),
        source_range.end().checked_sub(origin).unwrap(),
    )
    .unwrap();
    let source = request(&asset, 137).slice(source_range).read().unwrap();
    let mapped = request(&asset, 137)
        .timeline(AudioTimeline::Asset)
        .slice(asset_range)
        .read()
        .unwrap();
    assert_eq!(source.samples(), mapped.samples());
    assert_eq!(mapped.start(), source.start().checked_sub(origin).unwrap());
    assert_eq!(source.source_start(), mapped.source_start());
    let clip = asset
        .clip(asset_range)
        .video(
            VideoOutput::new()
                .track(TrackId(0))
                .sample(Sampling::uniform(3))
                .duration_inference(DurationInference::NextTimestamp),
        )
        .audio(AudioOutput::f32().track(TrackId(1)))
        .read()
        .unwrap();
    assert_eq!(clip.timeline_origin, origin);
    let audio = clip.audio.as_ref().unwrap();
    assert_eq!(audio.source_grid_origin, info.source_start);
    assert_eq!(audio.sample_grid_origin, grid);
    assert_eq!(audio.requested_sample_bounds, 1001..3501);
    assert_eq!(audio.block.samples(), source.samples());
    assert_eq!(audio.block.source_start(), source.source_start());
    drop((source, mapped, clip));
    assert_idle(&session);
}

#[test]
fn audio_admission_precedes_source_callbacks_and_eof_cancel_errors_retire_permits() {
    let block =
        AudioBlock::new(vec![0.25; 8192], 48000, ChannelLayout::mono(), Time::ZERO).unwrap();
    let mut writer = WavWriter::new(Sink::memory(), 48000, ChannelLayout::mono()).unwrap();
    writer.write(&block).unwrap();
    let bytes = writer.finish().unwrap().into_bytes().unwrap();
    let opens = Arc::new(AtomicUsize::new(0));
    let counter = opens.clone();
    let source = Source::from_seekable_factory(
        "admitted audio",
        Some("immutable-v1".into()),
        Some(bytes.len() as u64),
        move || {
            counter.fetch_add(1, Ordering::AcqRel);
            Ok(Cursor::new(bytes.clone()))
        },
    )
    .unwrap();
    let session = Session::builder()
        .limits(
            Limits::new()
                .cpu_executions(1)
                .cpu_codecs(1)
                .cpu_codec_threads(1),
        )
        .build()
        .unwrap();
    let asset = session.from_source(source);
    let request = asset.audio().unwrap();
    let mut reader = request.reader().unwrap();
    assert_eq!(session.live_cpu_usage().executions, 1);
    let before = opens.load(Ordering::Acquire);
    assert_eq!(
        request.probe().unwrap_err().kind(),
        ErrorKind::ResourceExhausted
    );
    assert_eq!(
        request.compile().unwrap_err().kind(),
        ErrorKind::ResourceExhausted
    );
    assert_eq!(opens.load(Ordering::Acquire), before);
    let held = reader.next_block().unwrap().unwrap();
    reader.cancellation().cancel();
    assert_eq!(
        reader.next_block().unwrap_err().kind(),
        ErrorKind::Cancelled
    );
    assert_eq!(session.live_cpu_usage().executions, 0);
    drop(held);
    assert_idle(&session);
    let plan = request.compile().unwrap();
    let output = plan.execute().unwrap();
    assert_eq!(session.live_cpu_usage().executions, 0);
    drop(output);
    let samples = request
        .sample_format(media::backend::native::audio_samples::SampleFormat::I16)
        .read()
        .unwrap();
    assert_eq!(session.live_cpu_usage().executions, 0);
    drop(samples);
    assert_idle(&session);
    assert!(
        session
            .from_source(Source::memory(vec![0; 32]))
            .audio()
            .unwrap()
            .reader()
            .is_err()
    );
    assert_idle(&session);
}

#[test]
fn ffmpeg_audio_and_mp4_clip_cancellation_quota_and_terminal_failures_release_resources() {
    let fixture = Fixture::new(true, false);
    let session = Session::builder()
        .limits(
            Limits::new()
                .cpu_executions(1)
                .cpu_codecs(1)
                .cpu_codec_threads(1),
        )
        .build()
        .unwrap();
    let asset = session.open(&fixture.path).unwrap();
    let mut stream = request(&asset, 137).reader().unwrap();
    let held = stream.next_block().unwrap().unwrap();
    let clip = asset
        .clip(requested())
        .video(
            VideoOutput::new()
                .track(TrackId(0))
                .sample(Sampling::uniform(2))
                .duration_inference(DurationInference::NextTimestamp),
        )
        .audio(AudioOutput::f32().track(TrackId(1)));
    assert_eq!(
        clip.read().unwrap_err().kind(),
        ErrorKind::ResourceExhausted
    );
    assert_eq!(session.live_batches(), 0);
    assert_eq!(session.live_cpu_usage().executions, 1);
    stream.cancellation().cancel();
    assert_eq!(
        stream.next_block().unwrap_err().kind(),
        ErrorKind::Cancelled
    );
    assert_eq!(session.live_cpu_usage().executions, 0);
    drop(held);
    assert_idle(&session);
    let clip = clip.read().unwrap();
    drop(clip);
    assert_idle(&session);
    let cancelled = asset
        .clip(requested())
        .video(VideoOutput::new().track(TrackId(0)))
        .audio(AudioOutput::f32().track(TrackId(1)));
    cancelled.cancellation().cancel();
    assert_eq!(cancelled.read().unwrap_err().kind(), ErrorKind::Cancelled);
    assert_idle(&session);
    let mut invalid = asset
        .audio()
        .unwrap()
        .ffmpeg_options(AudioDecodeOptions {
            max_buffered_samples: 64,
            ..Default::default()
        })
        .reader()
        .unwrap();
    assert_eq!(
        invalid.next_block().unwrap_err().kind(),
        ErrorKind::ResourceExhausted
    );
    assert!(invalid.next().is_none());
    assert_idle(&session);
}
