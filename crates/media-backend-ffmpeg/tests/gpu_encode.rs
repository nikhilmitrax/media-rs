//! Real native surface NVENC tests. Fixture generation and independent output
//! validation may use the CPU; the library transcode path requires NVDEC/NVENC.
//! Explicit execution fails when required hardware/driver support is absent.
#![cfg(feature = "cuda")]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use media_backend_cuda::CudaLimits;
use media_backend_ffmpeg::codec::{VideoDecodeOptions, VideoOutput, VideoReader};
use media_backend_ffmpeg::cuda::{CudaDecodeOptions, CudaVideoReader};
use media_backend_ffmpeg::cuda_encode::{CudaTranscodeOptions, NvencCodec, transcode};
use media_backend_ffmpeg::format::{PacketReader, ProbeOptions, TrackKind, probe};
use media_core::media::{Packet, PixelFormat, TrackId};
use media_io::source::Source;

struct Fixture {
    directory: PathBuf,
    source: PathBuf,
    binary: PathBuf,
}
impl Fixture {
    fn new(ten_bit: bool) -> Self {
        static SEQUENCE: AtomicU64 = AtomicU64::new(0);
        let directory = std::env::temp_dir().join(format!(
            "media-nvenc-test-{}-{}",
            std::process::id(),
            SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&directory).unwrap();
        let source = directory.join("input.mp4");
        let binary = std::env::var_os("MEDIA_FFMPEG_BIN")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                if Path::new("/opt/homebrew/bin/ffmpeg").is_file() {
                    PathBuf::from("/opt/homebrew/bin/ffmpeg")
                } else {
                    PathBuf::from("ffmpeg")
                }
            });
        let mut command = Command::new(&binary);
        command.args([
            "-v",
            "error",
            "-nostdin",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=192x192:rate=10:duration=1.2",
            "-itsoffset",
            "0.25",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=0.7",
            "-threads:v",
            "1",
            "-bf",
            "2",
            "-g",
            "12",
        ]);
        if ten_bit {
            command.args([
                "-c:v",
                "libx265",
                "-pix_fmt",
                "yuv420p10le",
                "-x265-params",
                "pools=1:frame-threads=1:log-level=error",
            ]);
        } else {
            // A VFR clock exercises real presentation timestamps rather than
            // silently assigning output times from a nominal frame rate.
            command.args([
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-vf",
                "setpts='if(lt(N,6),N,6+(N-6)*2)/(10*TB)'",
                "-fps_mode",
                "vfr",
            ]);
        }
        let generated = command
            .args([
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
                "-c:a",
                "aac",
                "-threads:a",
                "1",
                "-b:a",
                "64000",
                "-output_ts_offset",
                "2",
            ])
            .arg(&source)
            .output()
            .expect("required FFmpeg fixture generator");
        assert!(
            generated.status.success(),
            "fixture failed: {}",
            String::from_utf8_lossy(&generated.stderr)
        );
        Self {
            directory,
            source,
            binary,
        }
    }
    fn input(&self) -> Source {
        Source::file(&self.source).unwrap()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.directory);
    }
}
fn audio(source: &Source) -> TrackId {
    probe(source, ProbeOptions::default())
        .unwrap()
        .tracks
        .into_iter()
        .find(|t| t.kind == TrackKind::Audio)
        .unwrap()
        .id
}
fn packets(source: &Source, track: TrackId) -> Vec<Packet> {
    let mut reader = PacketReader::open(source, ProbeOptions::default()).unwrap();
    let mut result = Vec::new();
    while let Some(packet) = reader.next_packet().unwrap() {
        if packet.track == track {
            result.push(packet);
        }
    }
    result
}
fn validate(ten_bit: bool, resize: bool) {
    let fixture = Fixture::new(ten_bit);
    let output = fixture.directory.join("output.mp4");
    let options = CudaTranscodeOptions {
        codec: if ten_bit {
            NvencCodec::Hevc
        } else {
            NvencCodec::H264
        },
        gop: 12,
        width: resize.then_some(160),
        height: resize.then_some(160),
        ..Default::default()
    };
    let report = transcode(
        &fixture.input(),
        &output,
        0,
        CudaLimits::default(),
        options.clone(),
    )
    .expect("required NVDEC-to-NVENC transcode");
    assert_eq!(
        (
            report.frames_decoded,
            report.frames_encoded,
            report.video_packets_written
        ),
        (12, 12, 12),
        "all delayed frames must drain"
    );
    assert_eq!(
        (report.output_width, report.output_height),
        if resize { (160, 160) } else { (192, 192) }
    );
    assert_eq!(report.decode_timestamps_reconstructed, 12);
    assert_eq!(report.decode_reorder_bound, 2);
    assert_eq!(
        report.pixel_format,
        if ten_bit {
            PixelFormat::P010
        } else {
            PixelFormat::Nv12
        }
    );
    assert!(
        report.peak_pending_bytes > 0 && report.peak_pending_bytes <= options.max_pending_bytes
    );
    assert!(report.source_bytes_read > 0);
    let result = Source::file(&output).unwrap();
    let original = packets(&fixture.input(), audio(&fixture.input()));
    let copied = packets(&result, audio(&result));
    assert!(!original.is_empty());
    assert_eq!(report.audio_packets_written, original.len() as u64);
    if original.len() != copied.len() {
        eprintln!(
            "copied AAC source packet indices: {:?}",
            copied
                .iter()
                .map(|after| original.iter().position(|before| before.data == after.data))
                .collect::<Vec<_>>()
        );
        eprintln!(
            "AAC original timing: {:?}; copied timing: {:?}",
            original
                .iter()
                .map(|p| (p.pts, p.dts, p.duration, p.data.len()))
                .collect::<Vec<_>>(),
            copied
                .iter()
                .map(|p| (p.pts, p.dts, p.duration, p.data.len()))
                .collect::<Vec<_>>()
        );
    }
    assert_eq!(original.len(), copied.len());
    for (before, after) in original.iter().zip(copied) {
        assert_eq!(
            before.data, after.data,
            "audio remains compressed, bit-identical packet copy"
        );
        assert_eq!(
            (before.pts, before.dts, before.duration),
            (after.pts, after.dts, after.duration),
            "common-clock offset must survive mux"
        );
    }
    let decode = VideoDecodeOptions {
        // The portable CPU adapter does not expose planar YUV420P10. Timing
        // validation can explicitly convert it; actual precision is established
        // separately by reopening the encoded bitstream as native P010 below.
        output: if ten_bit {
            VideoOutput::Rgb8
        } else {
            VideoOutput::Native
        },
        ..Default::default()
    };
    let mut source = VideoReader::open(&fixture.input(), decode).unwrap();
    let mut encoded = VideoReader::open(&result, decode).unwrap();
    let mut count = 0;
    while let Some(frame) = encoded.next_frame().unwrap() {
        let reference = source.next_frame().unwrap().unwrap();
        assert_eq!(frame.timing.source_pts, reference.timing.source_pts);
        // Compare the retained native hint; a container/codec duration does not
        // establish a VFR frame's actual displayed interval.
        assert_eq!(
            frame.timing.native_duration,
            reference.timing.native_duration
        );
        assert_eq!(frame.timing.duration, None);
        if !ten_bit {
            assert_eq!(frame.image.pixel_format(), reference.image.pixel_format());
        }
        assert_eq!(
            frame.image.metadata().color,
            reference.image.metadata().color
        );
        count += 1;
    }
    assert_eq!(count, 12);
    assert!(source.next_frame().unwrap().is_none());
    let output_rgb = VideoReader::open(&result, VideoDecodeOptions::default())
        .unwrap()
        .next_frame()
        .unwrap()
        .unwrap();
    let mut reference = Command::new(&fixture.binary);
    reference
        .args(["-v", "error", "-nostdin", "-i"])
        .arg(&fixture.source);
    if resize {
        reference.args(["-vf", "scale=160:160:flags=bilinear"]);
    }
    let reference = reference
        .args(["-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
        .output()
        .unwrap();
    assert!(
        reference.status.success(),
        "reference decode: {}",
        String::from_utf8_lossy(&reference.stderr)
    );
    let actual = output_rgb.image.packed_bytes().unwrap();
    assert_eq!(actual.len(), reference.stdout.len());
    let mse = actual
        .iter()
        .zip(&reference.stdout)
        .map(|(&a, &b)| {
            let difference = (f64::from(a) - f64::from(b)) / 255.0;
            difference * difference
        })
        .sum::<f64>()
        / actual.len() as f64;
    assert!(
        mse.sqrt() < 0.08,
        "independent decode/resize RGB RMSE {}",
        mse.sqrt()
    );
    let verify = Command::new(&fixture.binary)
        .args(["-v", "error", "-nostdin", "-i"])
        .arg(&output)
        .args(["-map", "0", "-f", "null", "-"])
        .output()
        .unwrap();
    assert!(
        verify.status.success(),
        "independent full-file decode: {}",
        String::from_utf8_lossy(&verify.stderr)
    );
    // Reopen as actual GPU surfaces; no CPU upload can satisfy this assertion.
    let mut hardware = CudaVideoReader::open(
        &result,
        0,
        CudaLimits::default(),
        CudaDecodeOptions::default(),
    )
    .unwrap();
    let frame = hardware.next_frame().unwrap().unwrap();
    assert_eq!(frame.frame().image.pixel_format(), report.pixel_format);
    assert!(frame.frame().image.host_bytes().is_err());
    drop(frame);
    drop(hardware);
    // A second transcode cannot truncate a finalized output.
    let bytes = fs::read(&output).unwrap();
    assert!(transcode(&fixture.input(), &output, 0, CudaLimits::default(), options).is_err());
    assert_eq!(fs::read(&output).unwrap(), bytes);
}

#[test]
#[ignore = "requires NVIDIA NVDEC/NVENC, CUDA 12.8-compatible driver, and video driver libraries"]
fn native_h264_vfr_b_frames_and_audio_passthrough_finalize() {
    validate(false, false);
}

#[test]
#[ignore = "requires NVIDIA HEVC Main10 NVDEC/NVENC and CUDA/video driver libraries"]
fn native_hevc_main10_preserves_precision_and_audio_passthrough() {
    validate(true, false);
}

#[cfg(feature = "cuda-filters")]
#[test]
#[ignore = "requires NVIDIA NVDEC/NVENC and FFmpeg scale_cuda filter"]
fn native_h264_component_resize_stays_on_device() {
    validate(false, true);
}

#[cfg(feature = "cuda-filters")]
#[test]
#[ignore = "requires HEVC Main10 NVDEC/NVENC and P010 scale_cuda support"]
fn native_hevc_main10_component_resize_preserves_p010() {
    validate(true, true);
}
