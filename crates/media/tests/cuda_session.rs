//! GPU context/admission/lifetime tests plus an independent CPU format fixture
//! check. Explicit GPU execution requires NVIDIA hardware and never skips it.
#![cfg(all(feature = "ffmpeg", feature = "cuda"))]

use std::{
    fs,
    io::Cursor,
    path::PathBuf,
    process::Command,
    sync::{
        Arc,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
};

use media::core::{error::ErrorKind, time::Time};
use media::{
    Session,
    backend::{
        cuda::{CudaContext, CudaLimits, PreprocessSpec},
        ffmpeg::cuda_encode::CudaTranscodeOptions,
    },
    cuda_session::CudaSessionLimits,
    io::source::Source,
};

struct Fixture {
    directory: PathBuf,
    path: PathBuf,
}
impl Fixture {
    fn changing_format() -> Self {
        let mut fixture = Self::new();
        let binary = std::env::var_os("MEDIA_FFMPEG_BIN")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                if std::path::Path::new("/opt/homebrew/bin/ffmpeg").is_file() {
                    PathBuf::from("/opt/homebrew/bin/ffmpeg")
                } else {
                    PathBuf::from("ffmpeg")
                }
            });
        let mut bytes = Vec::new();
        for (index, width, color) in [(0, 192, "white"), (1, 256, "black")] {
            let segment = fixture.directory.join(format!("part{index}.h264"));
            let input = format!("color=c={color}:size={width}x192:rate=10:duration=0.4");
            let output = Command::new(&binary)
                .args([
                    "-v",
                    "error",
                    "-nostdin",
                    "-f",
                    "lavfi",
                    "-i",
                    &input,
                    "-c:v",
                    "libx264",
                    "-threads:v",
                    "1",
                    "-bf",
                    "0",
                    "-g",
                    "1",
                    "-pix_fmt",
                    "yuv420p",
                    "-x264-params",
                    "repeat-headers=1:aud=1",
                    "-colorspace",
                    "bt709",
                    "-color_primaries",
                    "bt709",
                    "-color_trc",
                    "bt709",
                    "-color_range",
                    "tv",
                    "-chroma_sample_location",
                    "left",
                    "-f",
                    "h264",
                ])
                .arg(&segment)
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            bytes.extend(fs::read(segment).unwrap());
        }
        fixture.path = fixture.directory.join("changing.h264");
        fs::write(&fixture.path, bytes).unwrap();
        // Keep the AnnexB transition intact while using an explicitly supported
        // demuxer. CLI stream copy only prepares the fixture; no pixels decode.
        let wrapped = fixture.directory.join("changing.ts");
        let output = Command::new(&binary)
            .args([
                "-v", "error", "-nostdin", "-fflags", "+genpts", "-f", "h264", "-r", "10", "-i",
            ])
            .arg(&fixture.path)
            .args(["-map", "0:v:0", "-c:v", "copy", "-f", "mpegts"])
            .arg(&wrapped)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        fixture.path = wrapped;
        fixture
    }
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let directory = std::env::temp_dir().join(format!(
            "media-cuda-session-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&directory).unwrap();
        let path = directory.join("source.mp4");
        let binary = std::env::var_os("MEDIA_FFMPEG_BIN")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                if std::path::Path::new("/opt/homebrew/bin/ffmpeg").is_file() {
                    PathBuf::from("/opt/homebrew/bin/ffmpeg")
                } else {
                    PathBuf::from("ffmpeg")
                }
            });
        let output = Command::new(binary)
            .args([
                "-v",
                "error",
                "-nostdin",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=192x192:rate=10:duration=0.6",
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
                "-color_primaries",
                "bt709",
                "-color_trc",
                "bt709",
                "-color_range",
                "tv",
                "-chroma_sample_location",
                "left",
            ])
            .arg(&path)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        Self { directory, path }
    }
}

#[test]
#[ignore = "requires NVDEC resolution changes with retained native surfaces and CUDA preprocessing"]
fn format_change_errors_preserve_old_surface_and_its_codec_admission() {
    let fixture = Fixture::changing_format();
    let session = Session::default();
    let mut reader = session
        .open(&fixture.path)
        .unwrap()
        .video()
        .unwrap()
        .cuda(0, CudaLimits::default())
        .reader()
        .unwrap();
    let first = reader.next_frame().unwrap().unwrap();
    assert_eq!(
        (first.frame().image.width(), first.frame().image.height()),
        (192, 192)
    );
    let context = first.context().clone();
    let error = loop {
        match reader.next_frame() {
            Ok(Some(frame)) => {
                assert_eq!(
                    frame.frame().image.width(),
                    192,
                    "new geometry cannot be silently published"
                );
            }
            Ok(None) => panic!("format transition must produce a precise error"),
            Err(error) => break error,
        }
    };
    assert_eq!(error.kind(), ErrorKind::Incompatible);
    assert!(error.to_string().contains("format changed"));
    drop(reader);
    assert_eq!(
        session.live_cuda_codec_sessions(),
        1,
        "old frame still retains its native decoder generation"
    );
    let input = first.yuv_view().unwrap();
    drop(first);
    let stream = context.stream().unwrap();
    let tensor = input
        .preprocess(&stream, PreprocessSpec::new(8, 8).unwrap())
        .unwrap();
    drop(input);
    let values = tensor.download(&stream).unwrap();
    assert!(
        values.iter().all(|v| *v > 0.98 && *v <= 1.0),
        "retained original white surface must remain readable after native decoder reconfiguration/drop"
    );
    for _ in 0..200 {
        if session.live_cuda_codec_sessions() == 0 && session.reserved_cuda_native_bytes() == 0 {
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    panic!("completed native input failed to retire its session admission");
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.directory);
    }
}

#[test]
fn actual_annexb_resolution_fixture_reports_change_without_invalidating_old_cpu_frame() {
    let binary = std::env::var_os("MEDIA_FFMPEG_BIN")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            if std::path::Path::new("/opt/homebrew/bin/ffmpeg").is_file() {
                PathBuf::from("/opt/homebrew/bin/ffmpeg")
            } else {
                PathBuf::from("ffmpeg")
            }
        });
    if Command::new(binary).arg("-version").output().is_err() {
        assert!(
            std::env::var_os("MEDIA_REQUIRE_FFMPEG_TESTS").is_none(),
            "required format-change fixture generator unavailable"
        );
        eprintln!("UNEXERCISED: CPU format-change fixture requires FFmpeg executable");
        return;
    }
    let fixture = Fixture::changing_format();
    let mut reader = media::backend::ffmpeg::codec::VideoReader::open(
        &Source::file(&fixture.path).unwrap(),
        Default::default(),
    )
    .unwrap();
    let first = reader.next_frame().unwrap().unwrap();
    let original = first.image.packed_bytes().unwrap().to_vec();
    let error = loop {
        match reader.next_frame() {
            Ok(Some(frame)) => assert_eq!(frame.image.width(), 192),
            Ok(None) => panic!("fixture failed to contain an observable format transition"),
            Err(error) => break error,
        }
    };
    assert_eq!(error.kind(), ErrorKind::Incompatible);
    assert!(error.to_string().contains("format changed"));
    drop(reader);
    assert_eq!(first.image.packed_bytes().unwrap(), original);
    assert_eq!((first.image.width(), first.image.height()), (192, 192));
}

#[test]
#[ignore = "requires NVIDIA NVDEC and CUDA 12.8-compatible runtime"]
fn canonical_context_and_codec_permits_survive_reader_drop_with_native_frames() {
    let fixture = Fixture::new();
    let session = Session::default();
    let request = session
        .open(&fixture.path)
        .unwrap()
        .video()
        .unwrap()
        .cuda(0, CudaLimits::default());
    let mut first = request.reader().unwrap();
    let second = request.reader().unwrap();
    assert!(first.context().shares_resources_with(second.context()));
    assert_eq!(session.live_cuda_codec_sessions(), 2);
    let error = request
        .reader()
        .err()
        .expect("third reader must hit aggregate codec limit");
    assert_eq!(error.kind(), ErrorKind::ResourceExhausted);
    assert_eq!(
        error.context().constraint.as_deref(),
        Some("cuda_codec_sessions")
    );
    let frame = first.next_frame().unwrap().unwrap();
    let retained = frame.frame().image.storage().byte_len();
    assert_eq!(session.reserved_cuda_native_bytes(), retained);
    drop(first);
    drop(second);
    assert_eq!(
        session.live_cuda_codec_sessions(),
        1,
        "native frame retains its decoder admission slot"
    );
    assert_eq!(session.reserved_cuda_native_bytes(), retained);
    let image = frame.frame().image.clone();
    drop(frame);
    assert_eq!(
        session.live_cuda_codec_sessions(),
        1,
        "core image/native exports keep admission, not just the frame wrapper"
    );
    assert_eq!(session.reserved_cuda_native_bytes(), retained);
    drop(image);
    assert_eq!(session.live_cuda_codec_sessions(), 0);
    assert_eq!(session.reserved_cuda_native_bytes(), 0);
    let batch = request.read().unwrap();
    assert!(batch.selection.is_none());
    assert_eq!(
        batch.source,
        Source::file(&fixture.path).unwrap().fingerprint().unwrap(),
        "unsampled batches retain source provenance"
    );
    drop(batch);
    let mut cancelled = request.reader().unwrap();
    cancelled.cancellation().cancel();
    assert_eq!(
        cancelled.seek(Time::ZERO).unwrap_err().kind(),
        ErrorKind::Cancelled
    );
    assert_eq!(
        session.live_cuda_codec_sessions(),
        0,
        "cancelled seek retires an unused decoder immediately"
    );
    let mut changed = CudaLimits::default();
    changed.streams += 1;
    assert_eq!(
        session.cuda_context(0, changed).unwrap_err().kind(),
        ErrorKind::Incompatible
    );
}

#[test]
#[ignore = "requires NVIDIA NVDEC and CUDA preprocessing"]
fn aggregate_native_bytes_and_shared_owned_allocation_budget_are_enforced() {
    let fixture = Fixture::new();
    let limited = Session::builder()
        .cuda_limits(CudaSessionLimits {
            max_codec_sessions: 2,
            max_native_bytes: 1,
        })
        .build()
        .unwrap();
    let mut reader = limited
        .open(&fixture.path)
        .unwrap()
        .video()
        .unwrap()
        .cuda(0, CudaLimits::default())
        .reader()
        .unwrap();
    let error = reader.next_frame().unwrap_err();
    assert_eq!(
        error.context().constraint.as_deref(),
        Some("cuda_native_bytes")
    );
    assert_eq!(limited.reserved_cuda_native_bytes(), 0);
    assert_eq!(limited.live_cuda_codec_sessions(), 0);
    drop(reader);
    let limits = CudaLimits {
        memory_bytes: 1535,
        streams: 8,
    };
    let session = Session::default();
    let request = session
        .open(&fixture.path)
        .unwrap()
        .video()
        .unwrap()
        .cuda(0, limits)
        .sample([Time::ZERO])
        .duration_inference(media::prelude::DurationInference::NextTimestamp)
        .to_tensor(PreprocessSpec::new(8, 8).unwrap());
    let first = request.read().unwrap();
    assert_eq!(
        first.source,
        Source::file(&fixture.path).unwrap().fingerprint().unwrap()
    );
    assert_eq!(first.context().memory_used(), 768);
    assert_eq!(
        request
            .read()
            .expect_err("second allocation exceeds canonical aggregate byte budget")
            .kind(),
        ErrorKind::ResourceExhausted
    );
    let context = first.context().clone();
    drop(first);
    assert_eq!(context.memory_used(), 0);
}

#[test]
#[ignore = "requires NVIDIA NVDEC/NVENC and independent CUDA context support"]
fn registered_owned_context_is_used_by_native_decode_and_transcode() {
    let fixture = Fixture::new();
    let context = CudaContext::new_owned(0, CudaLimits::default()).unwrap();
    let session = Session::builder()
        .cuda_context(context.clone())
        .build()
        .unwrap();
    let request = session
        .open(&fixture.path)
        .unwrap()
        .video()
        .unwrap()
        .cuda(0, context.limits());
    let mut reader = request.reader().unwrap();
    let frame = reader.next_frame().unwrap().unwrap();
    assert_eq!(frame.context().device(), context.device());
    drop(frame);
    drop(reader);
    let plan = request
        .sample([Time::ZERO])
        .to_tensor(PreprocessSpec::new(8, 8).unwrap())
        .compile()
        .unwrap();
    assert_eq!(plan.description().context, Some(context.device()));
    let target = fixture.directory.join("encoded.mp4");
    let report = session
        .open(&fixture.path)
        .unwrap()
        .convert()
        .cuda(0, context.limits())
        .on_context(&context)
        .options(CudaTranscodeOptions {
            max_frame_bytes: 2 * 1024 * 1024,
            max_pending_bytes: 2 * 1024 * 1024,
            ..Default::default()
        })
        .write(&target)
        .unwrap();
    assert_eq!(report.device, context.device());
    assert_eq!(report.frames_encoded, 6);
    assert_eq!(session.live_cuda_codec_sessions(), 0);
    assert_eq!(session.reserved_cuda_native_bytes(), 0);
}

#[test]
#[ignore = "requires NVIDIA NVDEC and CUDA native context ownership"]
fn context_policy_rejects_allocator_aliases_and_source_callbacks_can_reenter_session() {
    let fixture = Fixture::new();
    let session = Session::default();
    let canonical = session.cuda_context(0, CudaLimits::default()).unwrap();
    let alias = CudaContext::new(0, CudaLimits::default()).unwrap();
    assert_eq!(alias.device(), canonical.device());
    assert!(!alias.shares_resources_with(&canonical));
    let calls = Arc::new(AtomicUsize::new(0));
    let counted = calls.clone();
    let bytes: Arc<[u8]> = fs::read(&fixture.path).unwrap().into();
    let length = bytes.len() as u64;
    let reentrant = session.clone();
    let source = Source::from_seekable_factory(
        "session-context-callback",
        Some("fixture-v1".into()),
        Some(length),
        move || {
            counted.fetch_add(1, Ordering::SeqCst);
            let context = reentrant
                .cuda_context(0, CudaLimits::default())
                .expect("registry lock cannot span this source callback");
            assert_eq!(context.limits(), CudaLimits::default());
            Ok(Cursor::new(bytes.clone()))
        },
    )
    .unwrap();
    let request = session
        .from_source(source)
        .video()
        .unwrap()
        .cuda(0, CudaLimits::default());
    let error = request
        .clone()
        .on_context(&alias)
        .reader()
        .err()
        .expect("same native handle cannot silently get an independent allocator");
    assert_eq!(
        error.context().constraint.as_deref(),
        Some("cuda_context_identity")
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        0,
        "context mismatch fails before source I/O"
    );
    let mut reader = request.reader().unwrap();
    assert!(reader.next_frame().unwrap().is_some());
    assert!(calls.load(Ordering::SeqCst) > 0);
}
