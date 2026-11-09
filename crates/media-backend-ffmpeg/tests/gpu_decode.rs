//! Explicit hardware tests. Ignored by default and fail when required GPU/SDK
//! support is missing; invoking these tests never reports a hardware skip.
#![cfg(feature = "cuda")]

use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use media_backend_cuda::{
    ChromaSiting, CudaContext, CudaLimits, PreprocessSpec, YuvMatrix, YuvRange,
};
use media_backend_ffmpeg::codec::{VideoDecodeOptions, VideoReader};
use media_backend_ffmpeg::cuda::{CudaDecodeOptions, CudaVideoReader};
use media_core::media::PixelFormat;
use media_core::storage::MemoryKind;
use media_io::source::Source;

struct Fixture {
    directory: PathBuf,
    path: PathBuf,
}
impl Fixture {
    fn new(ten_bit: bool) -> Self {
        static SEQUENCE: AtomicU64 = AtomicU64::new(0);
        let directory = std::env::temp_dir().join(format!(
            "media-rs-nvdec-test-{}-{}",
            std::process::id(),
            SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&directory).unwrap();
        let path = directory.join(if ten_bit { "p010.mp4" } else { "nv12.mp4" });
        let binary = std::env::var_os("MEDIA_FFMPEG_BIN")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                if std::path::Path::new("/opt/homebrew/bin/ffmpeg").is_file() {
                    PathBuf::from("/opt/homebrew/bin/ffmpeg")
                } else {
                    PathBuf::from("ffmpeg")
                }
            });
        let mut command = Command::new(binary);
        command.args([
            "-v",
            "error",
            "-nostdin",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:size=192x192:rate=10:duration=0.6",
            "-vf",
            "scale=in_color_matrix=bt601:out_color_matrix=bt709:out_range=tv",
            "-an",
            "-threads",
            "1",
        ]);
        if ten_bit {
            command.args([
                "-c:v",
                "libx265",
                "-pix_fmt",
                "yuv420p10le",
                "-x265-params",
                "pools=1:frame-threads=1:bframes=2:keyint=6",
            ]);
        } else {
            command.args([
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-bf", "2", "-g", "6",
            ]);
        }
        command.args([
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
        ]);
        let output = command
            .arg(&path)
            .output()
            .expect("hardware test requires the ffmpeg fixture generator");
        assert!(
            output.status.success(),
            "fixture generation failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        Self { directory, path }
    }
    fn source(&self) -> Source {
        Source::file(&self.path).unwrap()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.directory);
    }
}

#[test]
fn generated_gpu_fixtures_decode_as_bt709_red_on_cpu() {
    // Catch fixture matrix/range errors before allocating a cloud GPU. The
    // lavfi color source starts as BT.601 YUV and must be converted, not merely
    // relabeled, before asserting BT.709 NV12/P010 kernel output is pure red.
    for ten_bit in [false, true] {
        let fixture = Fixture::new(ten_bit);
        let mut reader =
            VideoReader::open(&fixture.source(), VideoDecodeOptions::default()).unwrap();
        let frame = reader.next_frame().unwrap().unwrap();
        for pixel in frame
            .image
            .packed_bytes()
            .unwrap()
            .as_chunks::<3>()
            .0
            .iter()
        {
            assert!(f32::from(pixel[0]) / 255. > 0.975, "red={}", pixel[0]);
            assert!(f32::from(pixel[1]) / 255. < 0.025, "green={}", pixel[1]);
            assert!(f32::from(pixel[2]) / 255. < 0.025, "blue={}", pixel[2]);
        }
    }
}

#[test]
#[ignore = "requires an NVIDIA GPU, NVDEC driver libraries, FFmpeg CUDA support, and NVRTC"]
fn nvdec_to_cuda_tensor_has_no_decoded_host_crossing_and_survives_early_drop() {
    let fixture = Fixture::new(false);
    let source = fixture.source();
    let mut reader = CudaVideoReader::open(
        &source,
        0,
        CudaLimits::default(),
        CudaDecodeOptions::default(),
    )
    .expect("hardware-required NVDEC setup must succeed on the GPU runner");
    let frame = reader
        .next_frame()
        .unwrap()
        .expect("hardware-decoded frame");
    assert_eq!(frame.frame().image.pixel_format(), PixelFormat::Nv12);
    assert_eq!(frame.frame().image.storage().kind(), MemoryKind::Opaque);
    assert!(
        frame.frame().image.host_bytes().is_err(),
        "decoded CUDA storage must not expose a host view"
    );
    assert_eq!(frame.context().device(), reader.context().device());
    assert!(
        frame
            .planes()
            .iter()
            .all(|plane| plane.address != 0 && plane.pitch >= plane.row_bytes)
    );
    let context = reader.context().clone();
    let producer = context.stream().unwrap();
    let consumer = context.stream().unwrap();
    let input = frame.yuv_view().unwrap();
    drop(reader);
    let tensor = input
        .preprocess(&producer, PreprocessSpec::new(16, 16).unwrap())
        .unwrap();
    drop(input);
    drop(frame);
    // The only host transfer is this explicitly requested final-tensor download.
    // The native input owner must survive the queued preprocessing after both
    // the wrapper and decoder have been dropped.
    let actual = tensor.download(&consumer).unwrap();
    assert_eq!(tensor.shape(), [1, 3, 16, 16]);
    assert_eq!(actual.len(), 3 * 16 * 16);
    assert!(actual.iter().all(|v| v.is_finite()));
    let mut software = VideoReader::open(&source, VideoDecodeOptions::default()).unwrap();
    let reference = software.next_frame().unwrap().unwrap();
    let rgb = reference.image.packed_bytes().unwrap();
    for channel in 0..3 {
        let expected = f32::from(rgb[channel]) / 255.0;
        for value in &actual[channel * 256..(channel + 1) * 256] {
            assert!(
                (value - expected).abs() < 0.02,
                "channel {channel}: actual {value}, expected {expected}"
            );
        }
    }
}

#[test]
#[ignore = "requires NVIDIA HEVC Main10 NVDEC, FFmpeg CUDA support, and NVRTC"]
fn hevc_main10_preserves_p010_before_gpu_preprocessing() {
    let fixture = Fixture::new(true);
    let mut reader = CudaVideoReader::open(
        &fixture.source(),
        0,
        CudaLimits::default(),
        CudaDecodeOptions::default(),
    )
    .unwrap();
    let frame = reader.next_frame().unwrap().unwrap();
    assert_eq!(frame.frame().image.pixel_format(), PixelFormat::P010);
    assert_eq!(frame.allocation_layout().format, PixelFormat::P010);
    assert!(frame.planes()[0].row_bytes >= 192 * 2);
    assert!(frame.frame().image.host_bytes().is_err());
    let stream = reader.context().stream().unwrap();
    let tensor = frame
        .as_yuv(YuvMatrix::Bt709, YuvRange::Limited, ChromaSiting::Left)
        .unwrap()
        .preprocess(&stream, PreprocessSpec::new(8, 8).unwrap())
        .unwrap();
    drop(frame);
    drop(reader);
    let values = tensor.download(&stream).unwrap();
    assert_eq!(values.len(), 3 * 8 * 8);
    assert!(values[..64].iter().all(|v| *v > 0.95));
    assert!(values[64..].iter().all(|v| v.abs() < 0.025));
}

#[test]
#[ignore = "requires NVIDIA NVDEC and CUDA 12.8-compatible context APIs"]
fn retained_surface_limit_and_context_mismatch_are_enforced() {
    let fixture = Fixture::new(false);
    let source = fixture.source();
    let mut reader = CudaVideoReader::open(
        &source,
        0,
        CudaLimits::default(),
        CudaDecodeOptions {
            max_retained_frames: 2,
            ..CudaDecodeOptions::default()
        },
    )
    .unwrap();
    let first = reader.next_frame().unwrap().unwrap();
    let second = reader.next_frame().unwrap().unwrap();
    assert_eq!(reader.retained_frames().unwrap(), 2);
    assert!(
        reader
            .next_frame()
            .unwrap_err()
            .to_string()
            .contains("retained CUDA")
    );
    drop(first);
    assert_eq!(reader.retained_frames().unwrap(), 1);
    let third = reader.next_frame().unwrap().unwrap();
    let different = CudaContext::new_owned(0, CudaLimits::default()).unwrap();
    let wrong_stream = different.stream().unwrap();
    let input = second
        .as_yuv(YuvMatrix::Bt709, YuvRange::Limited, ChromaSiting::Left)
        .unwrap();
    assert!(
        input
            .preprocess(&wrong_stream, PreprocessSpec::new(8, 8).unwrap())
            .is_err()
    );
    drop(input);
    drop(second);
    drop(third);
    assert_eq!(reader.retained_frames().unwrap(), 0);
    // Explicit existing-context path must use and retain exactly that context.
    let mut on_context =
        CudaVideoReader::open_on_context(&source, reader.context(), CudaDecodeOptions::default())
            .unwrap();
    let frame = on_context.next_frame().unwrap().unwrap();
    assert_eq!(frame.context().device(), reader.context().device());
}
