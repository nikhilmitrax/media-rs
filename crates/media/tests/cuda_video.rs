//! Full facade GPU paths. Explicit hardware runs fail on unavailable support.
#![cfg(all(feature = "ffmpeg", feature = "cuda"))]

use media::backend::cuda::{CudaLimits, Interpolation, PreprocessSpec};
use media::core::{error::ErrorKind, time::Time};
use media::{Session, io::source::Source};

#[test]
fn gpu_plan_preflight_is_lazy_and_rejects_invalid_structure_without_a_driver() {
    let source = Source::memory(vec![0_u8; 16]);
    let asset = Session::default().from_source(source);
    let request = asset
        .video()
        .unwrap()
        .cuda(0, CudaLimits::default())
        .sample([Time::ZERO])
        .to_tensor(PreprocessSpec::new(16, 16).unwrap());
    let plan = request.compile().unwrap();
    assert!(plan.explain().contains("NVDEC required"));
    assert!(plan.explain().contains("host transfers forbidden"));
    let invalid = asset
        .video()
        .unwrap()
        .cuda(
            0,
            CudaLimits {
                memory_bytes: 0,
                streams: 1,
            },
        )
        .to_tensor(PreprocessSpec::new(16, 16).unwrap())
        .compile()
        .unwrap_err();
    assert_eq!(invalid.kind(), ErrorKind::InvalidInput);
}

#[test]
#[ignore = "requires NVDEC, an NVIDIA GPU and NVRTC; explicit invocation never skips"]
fn unsorted_repeated_gpu_sampling_executes_a_reusable_plan_into_one_batch() {
    use std::{
        fs,
        io::Write,
        process::{Command, Stdio},
    };
    let directory =
        std::env::temp_dir().join(format!("media-rs-gpu-facade-{}", std::process::id()));
    fs::create_dir(&directory).unwrap();
    struct Cleanup(std::path::PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    let _cleanup = Cleanup(directory.clone());
    let path = directory.join("colors.mp4");
    let ffmpeg = std::env::var_os("MEDIA_FFMPEG_BIN").unwrap_or_else(|| "ffmpeg".into());
    let mut child = Command::new(ffmpeg)
        .args([
            "-v",
            "error",
            "-nostdin",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            "192x192",
            "-framerate",
            "10",
            "-i",
            "pipe:0",
            "-an",
            "-vf",
            "scale=out_color_matrix=bt709:out_range=tv",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-threads",
            "1",
            "-bf",
            "2",
            "-g",
            "6",
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
        ])
        .arg(&path)
        .stdin(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let colors = [
        [255_u8, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [0, 255, 255],
        [255, 0, 255],
    ];
    let mut input = child.stdin.take().unwrap();
    for color in colors {
        let pixels: Vec<u8> = color.into_iter().cycle().take(192 * 192 * 3).collect();
        input.write_all(&pixels).unwrap();
    }
    drop(input);
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let times = [
        Time::from_millis(250),
        Time::from_millis(50),
        Time::from_millis(250),
        Time::from_millis(350),
    ];
    let mut spec = PreprocessSpec::new(16, 16).unwrap();
    spec.interpolation = Interpolation::Nearest;
    let session = Session::builder()
        .limits(media::prelude::Limits::new().in_flight_batches(1))
        .build()
        .unwrap();
    let plan = session
        .open(&path)
        .unwrap()
        .video()
        .unwrap()
        .sample(times)
        .duration_inference(media::prelude::DurationInference::NextTimestamp)
        .cuda(0, CudaLimits::default())
        .to_tensor(spec)
        .compile()
        .unwrap();
    let batch = plan
        .execute()
        .expect("required GPU path must succeed on the hardware runner");
    assert_eq!(session.live_batches(), 1);
    assert_eq!(
        plan.execute().unwrap_err().kind(),
        ErrorKind::ResourceExhausted
    );
    assert_eq!(batch.tensor.shape(), [4, 3, 16, 16]);
    assert!(batch.report.hardware_decode);
    assert_eq!(batch.report.decoded_host_transfer_bytes, 0);
    assert!(batch.report.opaque_driver_bytes.is_none());
    let selections = &batch.selection.as_ref().unwrap().selections;
    assert_eq!(
        selections
            .iter()
            .map(|s| s.identity.ordinal)
            .collect::<Vec<_>>(),
        [2, 0, 2, 3]
    );
    assert_eq!(selections[0].identity, selections[2].identity);
    assert_eq!(
        batch
            .report
            .output_color
            .as_ref()
            .unwrap()
            .primaries
            .as_deref(),
        Some("bt709")
    );
    let consumer = batch.context().stream().unwrap();
    let values = batch.tensor.download(&consumer).unwrap();
    assert_eq!(&values[..768], &values[1536..2304]);
    for (slot, color_index) in [2, 0, 2, 3].into_iter().enumerate() {
        for channel in 0..3 {
            let expected = f32::from(colors[color_index][channel]) / 255.0;
            for value in &values[slot * 768 + channel * 256..slot * 768 + (channel + 1) * 256] {
                assert!(
                    (*value - expected).abs() < 0.035,
                    "slot {slot} channel {channel}: {value} vs {expected}"
                );
            }
        }
    }
    let retained = batch.tensor.storage_lease();
    drop(batch);
    assert_eq!(session.live_batches(), 1);
    drop(retained);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    while session.live_batches() != 0 && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    assert_eq!(
        session.live_batches(),
        0,
        "queued input/native export leases must retire the batch slot"
    );
    let again = plan.execute().unwrap();
    assert_eq!(again.tensor.shape(), [4, 3, 16, 16]);
    assert_eq!(
        again.selection.as_ref().unwrap().selections[0]
            .identity
            .ordinal,
        2
    );
    // Transcode needs both codec slots. Tensor submission can still retain the
    // decoder inputs; readiness and asynchronous input retirement are distinct.
    again.tensor.buffer().readiness().wait().unwrap();
    drop(again);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    while (session.live_cuda_codec_sessions() != 0 || session.reserved_cuda_native_bytes() != 0)
        && std::time::Instant::now() < deadline
    {
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    assert_eq!(session.live_cuda_codec_sessions(), 0);
    assert_eq!(session.reserved_cuda_native_bytes(), 0);
    #[cfg(feature = "cuda-filters")]
    {
        let output = directory.join("converted.mp4");
        let report = session
            .open(&path)
            .unwrap()
            .convert()
            .resize_components(media::prelude::Fit::inside(160, 160))
            .cuda(0, CudaLimits::default())
            .write(&output)
            .unwrap();
        assert_eq!((report.output_width, report.output_height), (160, 160));
        assert_eq!(report.frames_encoded, 6);
        assert_eq!(report.audio_packets_written, 0);
    }
}
