//! Mixed CUDA-video/CPU-audio clip proof against generated offset/VFR Matroska.
//! Run explicitly with native,ffmpeg,cuda,interop and --ignored --test-threads=1.
//! Hardware tests never skip absent NVDEC/NVRTC/FFmpeg support when invoked.
#![cfg(all(feature = "native", feature = "ffmpeg", feature = "cuda"))]
use media::{
    Session,
    backend::{
        cuda::{CudaLimits, PreprocessSpec},
        ffmpeg::{
            cuda::CudaDecodeOptions,
            format::{self, PacketReader, ProbeOptions},
        },
    },
    clip_request::{AudioOutput, Sampling, VideoOutput},
    core::{
        error::ErrorKind,
        policy::Boundary,
        time::{Time, TimeRange},
    },
    io::source::Source,
    prelude::{Limits, TrackId},
};
use std::{
    fs,
    io::Cursor,
    path::PathBuf,
    process::Command,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

struct Fixture {
    directory: PathBuf,
    path: PathBuf,
}

#[test]
#[ignore = "requires NVDEC and an NVIDIA GPU; explicit invocation never skips"]
fn changed_shared_range_version_during_mixed_execution_never_returns_a_clip() {
    use media::io::range::{RangeMetadata, RangeOptions, RangeResponse, RangeSource};
    struct ReplacedObject {
        bytes: Vec<u8>,
        session: Session,
        changed: AtomicBool,
    }
    impl RangeSource for ReplacedObject {
        fn metadata(&self) -> media::Result<RangeMetadata> {
            // Replace the object after a real first hardware surface was
            // retained, before the independent audio cursor completes.
            if self.session.reserved_cuda_native_bytes() > 0 {
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
            let metadata = self.metadata()?;
            if &metadata != expected {
                return Err(media::Error::incompatible(
                    "conditional mixed-clip source version changed",
                ));
            }
            Ok(RangeResponse {
                metadata,
                bytes: self.bytes[offset as usize..offset as usize + length].to_vec(),
            })
        }
    }
    let fixture = Fixture::new(false, false);
    let session = Session::default();
    let provider = Arc::new(ReplacedObject {
        bytes: fs::read(&fixture.path).unwrap(),
        session: session.clone(),
        changed: AtomicBool::new(false),
    });
    let source = Source::from_range_source(
        "replaced shared container",
        provider.clone(),
        RangeOptions::default(),
    )
    .unwrap();
    let result = session
        .from_source(source)
        .clip(range(150, 250))
        .video(video(4))
        .audio(audio())
        .cuda(0, limits())
        .read();
    assert!(
        result.is_err(),
        "old video and new audio object versions must not form a clip"
    );
    assert!(
        provider.changed.load(Ordering::Acquire),
        "replacement must occur after actual hardware decoding begins"
    );
    retired(&session);
}
impl Fixture {
    fn new(gaps: bool, flac: bool) -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let directory = std::env::temp_dir().join(format!(
            "media-cuda-clips-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&directory).unwrap();
        let path = directory.join("offsets.mkv");
        let ffmpeg = std::env::var_os("MEDIA_FFMPEG_BIN").unwrap_or_else(|| "ffmpeg".into());
        let filter = if gaps {
            "[0:v]select=lt(n\\,5),settb=1/1000,setpts=PTS+gte(N\\,2)*40+gte(N\\,3)*80+500[v];[1:a]asetpts=PTS+0.5/TB[a]"
        } else {
            "[0:v]settb=1/1000,setpts=PTS+500[v];[1:a]asetpts=PTS+0.6/TB[a]"
        };
        let output = Command::new(ffmpeg)
            .args([
                "-nostdin",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=192x192:rate=25:duration=0.4",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:sample_rate=48000:duration=0.3",
                "-filter_complex",
                filter,
                "-map",
                "[v]",
                "-map",
                "[a]",
                "-c:v",
                "libx264",
                "-threads:v",
                "1",
                "-pix_fmt",
                "yuv420p",
                "-bf",
                if gaps { "0" } else { "2" },
                "-g",
                "10",
                "-c:a",
                if flac { "flac" } else { "pcm_f32le" },
                "-fps_mode:v",
                "vfr",
                "-enc_time_base:v",
                "1/1000",
                "-color_primaries",
                "bt709",
                "-color_trc",
                "bt709",
                "-colorspace",
                "bt709",
                "-color_range",
                "tv",
                "-chroma_sample_location",
                "left",
                "-copyts",
                "-avoid_negative_ts",
                "disabled",
                "-y",
            ])
            .arg(&path)
            .output()
            .expect("FFmpeg with libx264 is required for generated hardware clip fixtures");
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
fn range(start: i64, end: i64) -> TimeRange {
    TimeRange::new(Time::from_millis(start), Time::from_millis(end)).unwrap()
}
fn video(count: usize) -> VideoOutput {
    VideoOutput::new()
        .track(TrackId(0))
        .sample(Sampling::uniform(count))
        .decode_options(media::backend::ffmpeg::codec::VideoDecodeOptions {
            duration_policy: media::core::policy::FrameDurationPolicy::TrustReported,
            ..Default::default()
        })
}
fn audio() -> AudioOutput {
    AudioOutput::f32().track(TrackId(1)).sample_rate(16_000)
}
fn limits() -> CudaLimits {
    CudaLimits {
        memory_bytes: 4 * 1024 * 1024,
        streams: 8,
    }
}
fn retired(session: &Session) {
    let deadline = Instant::now() + Duration::from_secs(3);
    while (session.live_batches() != 0
        || session.live_cuda_codec_sessions() != 0
        || session.reserved_cuda_native_bytes() != 0)
        && Instant::now() < deadline
    {
        std::thread::sleep(Duration::from_millis(5));
    }
    assert_eq!(session.live_batches(), 0);
    assert_eq!(session.live_cuda_codec_sessions(), 0);
    assert_eq!(session.reserved_cuda_native_bytes(), 0);
    assert_eq!(session.live_host_bytes(), 0);
}

#[test]
fn missing_video_and_preflight_cancellation_fail_without_driver_or_source_callbacks() {
    let opens = Arc::new(AtomicUsize::new(0));
    let counter = opens.clone();
    let source =
        Source::from_seekable_factory("no reads", Some("v1".into()), Some(16), move || {
            counter.fetch_add(1, Ordering::Relaxed);
            Ok(Cursor::new(vec![0; 16]))
        })
        .unwrap();
    let session = Session::default();
    let asset = session.from_source(source);
    let error = asset
        .clip(range(0, 10))
        .audio(audio())
        .cuda(0, limits())
        .read()
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::InvalidInput);
    let request = asset
        .clip(range(0, 10))
        .video(video(1))
        .audio(audio())
        .cuda(0, limits());
    request.cancellation().cancel();
    assert_eq!(request.read().unwrap_err().kind(), ErrorKind::Cancelled);
    assert_eq!(opens.load(Ordering::Relaxed), 0);
    retired(&session);
}

#[test]
fn generated_h264_headers_and_pcm_mapping_establish_offsets_without_probe_decoding() {
    let fixture = Fixture::new(false, false);
    let source = Source::file(&fixture.path).unwrap();
    let info = format::probe_headers(&source, ProbeOptions::default()).unwrap();
    // Header-only FFmpeg may report video.start/container.start as the audio
    // origin until H.264 reordering is decoded. Do not use that hint as proof.
    assert_eq!(info.tracks[0].codec, "h264");
    assert_eq!(info.tracks[1].codec, "pcm_f32le");
    let mut packets = PacketReader::open_headers(&source, ProbeOptions::default()).unwrap();
    let first_audio = loop {
        let packet = packets.next_packet().unwrap().unwrap();
        if packet.track == TrackId(1) {
            break packet;
        }
    };
    assert_eq!(first_audio.pts, Some(Time::from_millis(600)));
    let clip = Session::default()
        .from_source(source)
        .clip(range(150, 250))
        .video(video(4))
        .audio(audio())
        .read()
        .unwrap();
    assert_eq!(
        clip.audio.unwrap().sample_grid_origin,
        Time::from_millis(100)
    );
    assert_eq!(
        clip.video.unwrap().frames[0].frame.timing.pts,
        Some(Time::from_millis(120))
    );
}

#[test]
fn generated_vfr_gap_is_not_hidden_by_valid_uniform_sample_positions() {
    let fixture = Fixture::new(true, true);
    let asset = Session::default().open(&fixture.path).unwrap();
    let request = asset.clip(range(0, 300)).video(video(2)).audio(audio());
    assert_eq!(request.read().unwrap_err().kind(), ErrorKind::MissingData);
    let output = request.video_boundary(Boundary::Truncate).read().unwrap();
    assert_eq!(output.video.as_ref().unwrap().validity, [true, true]);
    assert_eq!(
        output.video.as_ref().unwrap().actual_coverage,
        [range(0, 80), range(120, 160), range(240, 300)]
    );
}

#[test]
#[ignore = "requires NVDEC/NVRTC and an NVIDIA GPU; explicit invocation never skips"]
fn mixed_offsets_fractional_audio_and_tensor_preserve_source_grid_and_one_shared_slot() {
    let fixture = Fixture::new(false, false);
    let session = Session::builder()
        .limits(Limits::new().host_memory(256 * 1024).in_flight_batches(1))
        .build()
        .unwrap();
    let asset = session.open(&fixture.path).unwrap();
    {
        let mut standalone = asset.video().unwrap().cuda(0, limits()).reader().unwrap();
        let header_hint = standalone.info().unwrap().timeline_origin;
        let first = standalone.next_frame().unwrap().unwrap();
        assert_eq!(
            first.frame().timing.source_pts,
            Some(Time::from_millis(500))
        );
        let native_duration = first.frame().timing.native_duration;
        if header_hint.is_some_and(|origin| origin > Time::from_millis(500)) {
            assert_eq!(first.frame().timing.pts, None);
            assert_eq!(standalone.info().unwrap().timeline_origin, None);
        }
        let image_owner = first.frame().image.storage().clone();
        let rebased = first.rebase_timeline(Time::from_millis(500)).unwrap();
        assert_eq!(rebased.frame().timing.pts, Some(Time::ZERO));
        assert_eq!(
            rebased.frame().timing.source_pts,
            Some(Time::from_millis(500))
        );
        assert_eq!(rebased.frame().timing.native_duration, native_duration);
        assert_eq!(rebased.frame().timing.ordinal, Some(0));
        assert!(
            rebased
                .frame()
                .image
                .storage()
                .shares_allocation(&image_owner)
        );
        drop(standalone);
        drop(rebased);
        assert!(
            session.reserved_cuda_native_bytes() > 0,
            "portable image owner retains its real native lease after rebasing and reader drop"
        );
        drop(image_owner);
    }
    retired(&session);
    let requested = TimeRange::new(
        Time::new(7201, 48_000).unwrap(),
        Time::new(12001, 48_000).unwrap(),
    )
    .unwrap();
    let request = asset
        .clip(requested)
        .video(video(4))
        .audio(audio())
        .cuda(0, limits());
    let clip = request
        .read()
        .expect("required mixed CUDA/CPU clip must execute");
    assert_eq!(clip.timeline_origin, Time::from_millis(500));
    assert!(
        matches!(clip.timeline_evidence, media::clip_request::ClipTimelineEvidence::VerifiedFirstSources { first_video_source_pts, first_audio_source_pts: Some(audio), .. }
        if first_video_source_pts == Time::from_millis(500) && audio == Time::from_millis(600))
    );
    assert_eq!(clip.actual_coverage, [requested]);
    assert_eq!(session.live_batches(), 1);
    assert_eq!(
        request.read().unwrap_err().kind(),
        ErrorKind::ResourceExhausted
    );
    let video = clip.video.as_ref().unwrap();
    assert_eq!(
        video
            .frames
            .iter()
            .map(|frame| frame.selection.actual)
            .collect::<Vec<_>>(),
        [120, 160, 200, 200].map(Time::from_millis)
    );
    assert!(video.report.hardware_decode);
    assert_eq!(video.report.decoded_host_transfer_bytes, 0);
    assert!(video.frames[0].frame.frame().image.packed_bytes().is_err());
    for frame in &video.frames {
        assert_eq!(frame.selection.identity.source, clip.source);
        assert_eq!(
            frame.frame.frame().timing.source_pts,
            Some(
                frame
                    .selection
                    .actual
                    .checked_add(clip.timeline_origin)
                    .unwrap()
            )
        );
    }
    let waveform = &clip.audio.as_ref().unwrap().block;
    assert_eq!(
        clip.audio.as_ref().unwrap().requested_sample_bounds,
        801..2401
    );
    assert_eq!(waveform.frames(), 1600);
    assert_eq!(
        clip.audio.as_ref().unwrap().source_grid_origin,
        Time::from_millis(600)
    );
    let reference = asset.audio().unwrap().resample(16_000).read().unwrap();
    assert_eq!(waveform.samples(), &reference.samples()[801..2401]);
    drop(reference);
    let (context, tensor, audio) = {
        let mut tensor_clip = clip
            .into_tensor(PreprocessSpec::new(16, 16).unwrap())
            .unwrap();
        let video = tensor_clip.video.take().unwrap();
        assert_eq!(video.batch.tensor.shape(), [4, 3, 16, 16]);
        assert_eq!(
            video.selections[0].timing.source_pts,
            Some(Time::from_millis(620))
        );
        assert_eq!(video.selections[0].request_index, 0);
        assert_eq!(
            video.batch.selection.as_ref().unwrap().selections[2].identity,
            video.batch.selection.as_ref().unwrap().selections[3].identity
        );
        assert_eq!(video.batch.report.decoded_host_transfer_bytes, 0);
        video
            .batch
            .tensor
            .storage_lease()
            .readiness()
            .wait()
            .unwrap();
        let context = video.batch.context().clone();
        assert_eq!(context.memory_used(), 4 * 3 * 16 * 16 * 4);
        let audio = tensor_clip.audio.take().unwrap().block;
        let tensor = video.batch.tensor;
        drop(tensor_clip);
        (context, tensor, audio)
    };
    let held_audio = audio.clone();
    assert_eq!(session.live_batches(), 1);
    #[cfg(feature = "interop")]
    {
        let consumer = context.stream().unwrap();
        let foreign = media::interop::dlpack::export_cuda_legacy(
            tensor,
            &consumer,
            media::core::policy::CopyPolicy::Forbid,
        )
        .unwrap();
        drop(audio);
        drop(held_audio);
        assert_eq!(
            session.live_batches(),
            1,
            "the foreign descriptor retains the shared slot"
        );
        drop(foreign);
    }
    #[cfg(not(feature = "interop"))]
    {
        drop(tensor);
        assert_eq!(
            session.live_batches(),
            1,
            "CPU waveform aliases retain the shared slot"
        );
        drop(audio);
        drop(held_audio);
    }
    retired(&session);
    let deadline = Instant::now() + Duration::from_secs(3);
    while context.memory_used() != 0 && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(5));
    }
    assert_eq!(context.memory_used(), 0);
    let cancelled = request.read().unwrap();
    cancelled.cancellation().cancel();
    assert_eq!(
        cancelled
            .into_tensor(PreprocessSpec::new(16, 16).unwrap())
            .unwrap_err()
            .kind(),
        ErrorKind::Cancelled
    );
    retired(&session);
}

#[test]
#[ignore = "requires NVDEC and an NVIDIA GPU; explicit invocation never skips"]
fn mixed_vfr_boundaries_preserve_masks_and_reject_unsampled_coverage_gaps() {
    let fixture = Fixture::new(true, true);
    let session = Session::default();
    let asset = session.open(&fixture.path).unwrap();
    let request = asset
        .clip(range(0, 300))
        .video(video(2))
        .audio(audio())
        .cuda(0, limits());
    assert_eq!(request.read().unwrap_err().kind(), ErrorKind::MissingData);
    retired(&session);
    let clip = request
        .clone()
        .video_boundary(Boundary::Truncate)
        .read()
        .unwrap();
    assert_eq!(clip.video.as_ref().unwrap().validity, [true, true]);
    assert_eq!(
        clip.actual_coverage,
        [range(0, 80), range(120, 160), range(240, 300)]
    );
    drop(clip);
    retired(&session);
    let clip = asset
        .clip(range(-40, 100))
        .video(video(4))
        .audio(audio())
        .video_boundary(Boundary::RepeatFrame)
        .audio_boundary(Boundary::Silence)
        .cuda(0, limits())
        .read()
        .unwrap();
    assert_eq!(
        clip.video.as_ref().unwrap().requested_times,
        [-40, -5, 30, 65].map(Time::from_millis)
    );
    assert_eq!(
        clip.video.as_ref().unwrap().validity,
        [false, false, true, true]
    );
    assert_eq!(
        clip.video.as_ref().unwrap().repeated,
        [true, true, false, false]
    );
    assert_eq!(
        clip.audio
            .as_ref()
            .unwrap()
            .padding
            .iter()
            .filter(|&&value| value)
            .count(),
        640
    );
    assert_eq!(clip.actual_coverage, [range(0, 80)]);
    drop(clip);
    retired(&session);
}

#[test]
#[ignore = "requires NVDEC and an NVIDIA GPU; explicit invocation never skips"]
fn mixed_cancellation_and_native_surface_exhaustion_retire_both_modalities() {
    let fixture = Fixture::new(false, false);
    let bytes = fs::read(&fixture.path).unwrap();
    let token = Arc::new(Mutex::new(
        None::<media::pipeline::execution::CancellationToken>,
    ));
    let callback_token = token.clone();
    let opens = Arc::new(AtomicUsize::new(0));
    let callback_opens = opens.clone();
    let source = Source::from_seekable_factory(
        "cancel at audio execution",
        Some("immutable-v1".into()),
        Some(bytes.len() as u64),
        move || {
            if callback_opens.fetch_add(1, Ordering::AcqRel) == 3 {
                callback_token.lock().unwrap().as_ref().unwrap().cancel();
            }
            Ok(Cursor::new(bytes.clone()))
        },
    )
    .unwrap();
    let session = Session::default();
    let request = session
        .from_source(source)
        .clip(range(150, 250))
        .video(video(4))
        .audio(audio())
        .cuda(0, limits());
    *token.lock().unwrap() = Some(request.cancellation());
    assert_eq!(request.read().unwrap_err().kind(), ErrorKind::Cancelled);
    assert!(opens.load(Ordering::Acquire) >= 4);
    retired(&session);
    let request = session
        .open(&fixture.path)
        .unwrap()
        .clip(range(100, 300))
        .video(video(4))
        .audio(audio())
        .cuda(0, limits())
        .decode_options(CudaDecodeOptions {
            max_retained_frames: 1,
            ..Default::default()
        });
    assert_eq!(
        request.read().unwrap_err().kind(),
        ErrorKind::ResourceExhausted
    );
    retired(&session);
}
