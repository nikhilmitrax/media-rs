//! Real caller-fed CPU decoding and incremental encoding without subprocess implementation.
#![cfg(feature = "linked")]
use media_backend_ffmpeg::{
    codec::{VideoDecodeOptions, VideoOutput, VideoReader},
    encode::CpuVideoCodec,
    format::{PacketReader, PacketSideData, PacketWithSideData, ProbeOptions, TrackKind},
    packet_decode::{CodecSend, FrameReceive, PacketDecodeOptions, PacketDecoder},
    video_writer::{
        VideoDrainState, VideoDurationPolicy, VideoTimestamp, VideoWriteState, VideoWriter,
        VideoWriterOptions,
    },
};
use media_core::{
    error::ErrorKind,
    media::{
        ColorDescription, FrameTiming, Image, ImageMetadata, Packet, PixelFormat, TrackId,
        VideoFrame,
    },
    policy::FrameDurationPolicy,
    time::{Time, TimeBase},
};
use media_io::source::Source;
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

struct Directory(PathBuf);
impl Directory {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "media-direct-video-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn file(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn yuv_color() -> ColorDescription {
    ColorDescription {
        primaries: Some("bt709".into()),
        transfer: Some("bt709".into()),
        matrix: Some("bt709".into()),
        range: Some("tv".into()),
        chroma_siting: Some("left".into()),
    }
}
fn frames() -> Vec<VideoFrame> {
    let mut pts = Time::new(2, 1).unwrap();
    (0..12)
        .map(|index| {
            let duration = Time::from_millis(if index % 3 == 2 { 80 } else { 40 });
            let color = ColorDescription {
                matrix: Some("rgb".into()),
                range: Some("full".into()),
                chroma_siting: None,
                ..yuv_color()
            };
            let mut pixels = vec![0; 64 * 48 * 3];
            for y in 0..48 {
                for x in 0..64 {
                    let start = (y * 64 + x) * 3;
                    pixels[start] = if (x + index) % 24 < 12 { 230 } else { 180 };
                    pixels[start + 1] = if y < 20 { 20 } else { 40 };
                    pixels[start + 2] = 10;
                }
            }
            let image = Image::from_packed(64, 48, PixelFormat::Rgb8, pixels)
                .unwrap()
                .with_metadata(ImageMetadata {
                    color,
                    ..Default::default()
                });
            let frame = VideoFrame {
                image,
                track: TrackId(0),
                timing: FrameTiming {
                    pts: Some(pts.checked_sub(Time::new(2, 1).unwrap()).unwrap()),
                    source_pts: Some(pts),
                    duration: Some(duration),
                    native_duration: None,
                    ordinal: Some(index as u64),
                },
            };
            pts = pts.checked_add(duration).unwrap();
            frame
        })
        .collect()
}
fn options(format: PixelFormat) -> VideoWriterOptions {
    VideoWriterOptions::new(
        64,
        48,
        format,
        TimeBase::new(1, 1000).unwrap(),
        TimeBase::new(1, 25).unwrap(),
        yuv_color(),
    )
}
fn write(
    path: &Path,
    frames: &[VideoFrame],
    mut options: VideoWriterOptions,
) -> media_backend_ffmpeg::video_writer::VideoWriterReport {
    options.bit_rate = 200_000;
    let mut writer = VideoWriter::create(path, options).unwrap();
    for frame in frames {
        loop {
            match writer.write_frame(frame).unwrap() {
                VideoWriteState::Accepted => break,
                VideoWriteState::NeedDrain => {
                    assert_eq!(writer.drain().unwrap(), VideoDrainState::NeedInput);
                }
                VideoWriteState::End => panic!("early writer EOF"),
            }
        }
    }
    let report = writer.finish().unwrap();
    assert!(report.finalized);
    report
}
fn decoded(path: &Path, output: VideoOutput) -> Vec<VideoFrame> {
    let mut reader = VideoReader::open(
        &Source::file(path).unwrap(),
        VideoDecodeOptions {
            output,
            duration_policy: FrameDurationPolicy::TrustReported,
            ..Default::default()
        },
    )
    .unwrap();
    let mut frames = Vec::new();
    while let Some(frame) = reader.next_frame().unwrap() {
        frames.push(frame);
    }
    frames
}
fn packets(
    source: &Source,
) -> (
    media_backend_ffmpeg::packet_decode::VideoDecoderConfig,
    Vec<Packet>,
) {
    let mut reader = PacketReader::open(source, ProbeOptions::default()).unwrap();
    let track = reader
        .info()
        .tracks
        .iter()
        .find(|track| track.kind == TrackKind::Video)
        .unwrap()
        .id;
    let config = reader.video_decoder_config(track).unwrap();
    let mut packets = Vec::new();
    while let Some(packet) = reader.next_packet().unwrap() {
        if packet.track == track {
            packets.push(packet);
        }
    }
    (config, packets)
}
fn receive(decoder: &mut PacketDecoder, frames: &mut Vec<VideoFrame>) -> bool {
    loop {
        match decoder.receive_frame().unwrap() {
            FrameReceive::Frame(frame) => frames.push(frame),
            FrameReceive::NeedInput => return false,
            FrameReceive::End => return true,
        }
    }
}
#[test]
fn rgb_vfr_writer_preserves_source_clock_bframes_and_explicit_finalization_states() {
    let directory = Directory::new();
    let path = directory.file("vfr.mkv");
    let frames = frames();
    let mut writer = VideoWriter::create(&path, options(PixelFormat::Rgb8)).unwrap();
    assert!(!writer.status().finalized);
    assert_eq!(writer.drain().unwrap(), VideoDrainState::NeedInput);
    assert_eq!(
        writer.write_frame(&frames[0]).unwrap(),
        VideoWriteState::Accepted
    );
    assert_eq!(
        writer.write_frame(&frames[1]).unwrap(),
        VideoWriteState::NeedDrain
    );
    assert_eq!(writer.status().frames_accepted, 1);
    assert_eq!(
        writer.write_frame(&frames[0]).unwrap_err().kind(),
        ErrorKind::Incompatible
    );
    writer.drain().unwrap();
    for frame in &frames[1..] {
        assert_eq!(
            writer.write_frame(frame).unwrap(),
            VideoWriteState::Accepted
        );
        writer.drain().unwrap();
    }
    assert!(
        writer.status().packets_written < 12,
        "actual codec delay must exercise EOF drain"
    );
    let report = writer.finish().unwrap();
    assert_eq!((report.frames_accepted, report.packets_written), (12, 12));
    assert!(report.finalized);
    assert_eq!(writer.finish().unwrap().packets_written, 12);
    assert_eq!(
        writer.write_frame(&frames[0]).unwrap(),
        VideoWriteState::End
    );
    assert_eq!(writer.drain().unwrap(), VideoDrainState::End);
    drop(writer);
    let (_, packets) = packets(&Source::file(&path).unwrap());
    assert!(
        packets.iter().any(|packet| packet.pts != packet.dts),
        "real B-frame reorder was not exercised"
    );
    for frame in &frames {
        let packet = packets
            .iter()
            .find(|packet| packet.pts == frame.timing.source_pts)
            .unwrap();
        assert_eq!(packet.duration, frame.timing.duration);
    }
    let actual = decoded(&path, VideoOutput::Rgb8);
    assert_eq!(actual.len(), 12);
    for (a, b) in actual.iter().zip(&frames) {
        assert_eq!(a.timing.source_pts, b.timing.source_pts);
        assert_eq!(a.timing.duration, b.timing.duration);
        assert_eq!((a.image.width(), a.image.height()), (64, 48));
    }
    let first = actual[0].image.packed_bytes().unwrap();
    assert!(
        first[0] > 200 && first[1] < 50 && first[2] < 40,
        "BT.709 conversion changed primary color unexpectedly: {:?}",
        &first[..3]
    );
}
#[test]
fn owned_external_packets_decode_zero_or_many_outputs_and_eof_survives_reader_drop() {
    let directory = Directory::new();
    let path = directory.file("packets.mkv");
    write(&path, &frames(), options(PixelFormat::Rgb8));
    let expected = decoded(&path, VideoOutput::Rgb8);
    let source = Source::memory(fs::read(&path).unwrap());
    let (config, packets) = packets(&source);
    drop(source);
    let mut decoder = PacketDecoder::new(
        config,
        PacketDecodeOptions {
            video: VideoDecodeOptions {
                duration_policy: FrameDurationPolicy::TrustReported,
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .unwrap();
    let mut output = Vec::new();
    assert!(matches!(
        decoder.receive_frame().unwrap(),
        FrameReceive::NeedInput
    ));
    for packet in &packets {
        assert_eq!(decoder.send_packet(packet).unwrap(), CodecSend::Accepted);
        assert_eq!(decoder.send_packet(packet).unwrap(), CodecSend::NeedDrain);
        assert_eq!(decoder.send_eof().unwrap(), CodecSend::NeedDrain);
        assert!(!receive(&mut decoder, &mut output));
    }
    assert!(
        output.len() < expected.len(),
        "real delayed B frames must remain before EOF"
    );
    assert_eq!(decoder.send_eof().unwrap(), CodecSend::Accepted);
    assert!(receive(&mut decoder, &mut output));
    assert!(matches!(
        decoder.receive_frame().unwrap(),
        FrameReceive::End
    ));
    assert_eq!(decoder.send_packet(&packets[0]).unwrap(), CodecSend::End);
    assert_eq!(decoder.packets_sent(), packets.len() as u64);
    assert_eq!(decoder.frames_decoded(), expected.len() as u64);
    let retained = output[0].clone();
    drop(decoder);
    for (actual, expected) in output.iter().zip(&expected) {
        assert_eq!(
            actual.image.packed_bytes().unwrap(),
            expected.image.packed_bytes().unwrap()
        );
        assert_eq!(actual.timing.pts, expected.timing.pts);
        assert_eq!(actual.timing.source_pts, expected.timing.source_pts);
        assert_eq!(actual.timing.duration, expected.timing.duration);
        assert_eq!(
            actual.timing.native_duration,
            expected.timing.native_duration
        );
        assert_eq!(actual.timing.ordinal, None);
    }
    drop(output);
    assert_eq!(
        retained.image.packed_bytes().unwrap(),
        expected[0].image.packed_bytes().unwrap()
    );
}
#[test]
fn native_yuv_writer_and_explicit_asset_clock_use_the_same_incremental_pipeline() {
    let directory = Directory::new();
    let source = directory.file("source.mkv");
    write(&source, &frames(), options(PixelFormat::Rgb8));
    let native = decoded(&source, VideoOutput::Native);
    assert!(
        native
            .iter()
            .all(|frame| frame.image.pixel_format() == PixelFormat::Yuv420p)
    );
    let output = directory.file("native.hevc.mkv");
    let mut native_options = options(PixelFormat::Yuv420p);
    native_options.codec = CpuVideoCodec::Hevc;
    native_options.timestamp = VideoTimestamp::Asset;
    let report = write(&output, &native, native_options);
    assert_eq!(report.frames_accepted, 12);
    let actual = decoded(&output, VideoOutput::Native);
    assert_eq!(actual.len(), 12);
    for (actual, input) in actual.iter().zip(&native) {
        assert_eq!(actual.timing.source_pts, input.timing.pts);
        assert_eq!(actual.timing.duration, input.timing.duration);
    }
    let rgb = decoded(&source, VideoOutput::Rgb8);
    assert!(
        rgb.iter()
            .all(|frame| frame.image.metadata().color.chroma_siting.is_none())
    );
    let rgb_output = directory.file("rgb-again.mkv");
    assert_eq!(
        write(&rgb_output, &rgb, options(PixelFormat::Rgb8)).frames_accepted,
        12
    );
    assert_eq!(decoded(&rgb_output, VideoOutput::Rgb8).len(), 12);
}
#[test]
fn invalid_inputs_preflight_no_overwrite_and_runtime_failure_remains_explicit_partial_output() {
    let directory = Directory::new();
    let path = directory.file("existing.mkv");
    fs::write(&path, b"keep").unwrap();
    assert!(VideoWriter::create(&path, options(PixelFormat::Rgb8)).is_err());
    assert_eq!(fs::read(&path).unwrap(), b"keep");
    let bad = directory.file("bad.mp4");
    let mut invalid = options(PixelFormat::Rgb8);
    invalid.width = 63;
    assert!(VideoWriter::create(&bad, invalid).is_err());
    assert!(!bad.exists());
    let incompatible = directory.file("bad.wav");
    assert!(VideoWriter::create(&incompatible, options(PixelFormat::Rgb8)).is_err());
    assert!(!incompatible.exists());
    let mut writer =
        VideoWriter::create(&directory.file("retry.mkv"), options(PixelFormat::Rgb8)).unwrap();
    let mut bad_frame = frames().remove(0);
    bad_frame.track = TrackId(9);
    assert!(writer.write_frame(&bad_frame).is_err());
    let mut bad_frame = frames().remove(0);
    bad_frame.image = Image::from_packed(2, 2, PixelFormat::Rgb8, vec![0; 12]).unwrap();
    assert!(writer.write_frame(&bad_frame).is_err());
    let mut bad_frame = frames().remove(0);
    bad_frame.timing.duration = None;
    assert!(writer.write_frame(&bad_frame).is_err());
    let mut bad_frame = frames().remove(0);
    bad_frame.timing.source_pts = Some(Time::new(1, 3001).unwrap());
    assert!(writer.write_frame(&bad_frame).is_err());
    assert_eq!(writer.status().frames_accepted, 0);
    assert_eq!(
        writer.write_frame(&frames()[0]).unwrap(),
        VideoWriteState::Accepted
    );
    writer.finish().unwrap();
    let partial = directory.file("partial.mkv");
    let mut limited = options(PixelFormat::Rgb8);
    limited.max_packet_bytes = 1;
    let mut writer = VideoWriter::create(&partial, limited).unwrap();
    writer.write_frame(&frames()[0]).unwrap();
    assert!(writer.finish().is_err());
    assert!(partial.exists());
    assert!(!writer.status().finalized);
    assert!(writer.status().failed);
    assert!(writer.finish().is_err());
    let report = writer.abort();
    assert!(report.failed);
    assert!(!report.finalized);
}
#[test]
fn mp4_duration_rewriting_requires_explicit_policy_and_preserves_exact_pts() {
    let directory = Directory::new();
    let path = directory.file("explicit.mp4");
    assert_eq!(
        VideoWriter::create(&path, options(PixelFormat::Rgb8))
            .err()
            .unwrap()
            .kind(),
        ErrorKind::Unsupported
    );
    assert!(!path.exists());
    let mut allowed = options(PixelFormat::Rgb8);
    allowed.duration_policy = VideoDurationPolicy::AllowContainerRewrite;
    let input = frames();
    let report = write(&path, &input, allowed);
    assert_eq!(report.warnings.len(), 1);
    let (_, packets) = packets_from_path(&path);
    for frame in &input {
        assert!(
            packets
                .iter()
                .any(|packet| packet.pts == frame.timing.source_pts)
        );
    }
    assert!(
        input.iter().any(|frame| packets
            .iter()
            .find(|packet| packet.pts == frame.timing.source_pts)
            .unwrap()
            .duration
            != frame.timing.duration),
        "generated VFR/B-frame fixture must exercise actual muxer duration rewriting"
    );
}
#[test]
fn full_range_native_pixels_retain_range_and_strict_submillisecond_preflight_is_non_destructive() {
    let directory = Directory::new();
    let path = directory.file("full.mkv");
    let mut setup = options(PixelFormat::Rgb8);
    setup.color.range = Some("full".into());
    write(&path, &frames(), setup);
    let native = decoded(&path, VideoOutput::Native);
    assert_eq!(native.len(), 12);
    assert!(
        native
            .iter()
            .all(|frame| frame.image.pixel_format() == PixelFormat::Yuv420p
                && frame.image.metadata().color.range.as_deref() == Some("pc"))
    );
    let mut strict = options(PixelFormat::Rgb8);
    strict.time_base = TimeBase::new(1, 30_000).unwrap();
    let invalid = directory.file("inexact.mkv");
    assert_eq!(
        VideoWriter::create(&invalid, strict).err().unwrap().kind(),
        ErrorKind::Unsupported
    );
    assert!(!invalid.exists());
}
#[test]
fn packet_track_side_data_clock_and_payload_validation_are_retryable_and_flush_is_explicit() {
    let directory = Directory::new();
    let path = directory.file("validate.mkv");
    write(&path, &frames(), options(PixelFormat::Rgb8));
    let (mut config, packets) = packets(&Source::file(&path).unwrap());
    config.first_ordinal = Some(120);
    let mut decoder = PacketDecoder::new(config, PacketDecodeOptions::default()).unwrap();
    let mut bad = packets[0].clone();
    bad.track = TrackId(90);
    assert_eq!(
        decoder.send_packet(&bad).unwrap_err().kind(),
        ErrorKind::Incompatible
    );
    bad = packets[0].clone();
    bad.data = Arc::from([]);
    assert!(decoder.send_packet(&bad).is_err());
    bad = packets[0].clone();
    bad.pts = Some(Time::new(1, 3001).unwrap());
    assert!(decoder.send_packet(&bad).is_err());
    let native = PacketWithSideData {
        packet: packets[0].clone(),
        flags: 1,
        side_data: vec![PacketSideData {
            kind: "NewExtraData".into(),
            data: Arc::from([1, 2, 3]),
        }],
    };
    assert_eq!(
        decoder
            .send_packet_with_side_data(&native)
            .unwrap_err()
            .kind(),
        ErrorKind::Unsupported
    );
    assert_eq!(decoder.packets_sent(), 0);
    let mut output = Vec::new();
    for packet in &packets {
        assert_eq!(decoder.send_packet(packet).unwrap(), CodecSend::Accepted);
        receive(&mut decoder, &mut output);
    }
    decoder.send_eof().unwrap();
    receive(&mut decoder, &mut output);
    assert_eq!(output.len(), 12);
    assert!(
        output
            .iter()
            .enumerate()
            .all(|(index, frame)| frame.timing.ordinal == Some(120 + index as u64))
    );
    assert!(
        output
            .iter()
            .all(|frame| frame.timing.duration.is_none() && frame.timing.native_duration.is_some()),
        "default external decoding must keep native duration hints separate from established display durations"
    );
    decoder.flush();
    output.clear();
    for packet in &packets {
        assert_eq!(decoder.send_packet(packet).unwrap(), CodecSend::Accepted);
        receive(&mut decoder, &mut output);
    }
    decoder.send_eof().unwrap();
    receive(&mut decoder, &mut output);
    assert_eq!(output.len(), 12);
    assert!(output.iter().all(|frame| frame.timing.ordinal.is_none()));
    let (config, packets) = packets_from_path(&path);
    let mut bounded = PacketDecoder::new(
        config,
        PacketDecodeOptions {
            max_packet_bytes: 1,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(
        bounded.send_packet(&packets[0]).unwrap_err().kind(),
        ErrorKind::ResourceExhausted
    );
    assert_eq!(bounded.packets_sent(), 0);
}
fn packets_from_path(
    path: &Path,
) -> (
    media_backend_ffmpeg::packet_decode::VideoDecoderConfig,
    Vec<Packet>,
) {
    packets(&Source::file(path).unwrap())
}

#[test]
fn native_decoder_format_change_requires_renegotiation() {
    let binary = if Path::new("/opt/homebrew/bin/ffmpeg").is_file() {
        PathBuf::from("/opt/homebrew/bin/ffmpeg")
    } else {
        PathBuf::from("ffmpeg")
    };
    if Command::new(&binary).arg("-version").output().is_err() {
        assert!(
            std::env::var_os("MEDIA_REQUIRE_FFMPEG_TESTS").is_none(),
            "required FFmpeg fixture generator missing"
        );
        eprintln!("SKIP: no FFmpeg CLI fixture generator");
        return;
    }
    let directory = Directory::new();
    let mut sets = Vec::new();
    for width in [64, 80] {
        let path = directory.file(&format!("{width}.ts"));
        let output = Command::new(&binary)
            .args([
                "-v",
                "error",
                "-nostdin",
                "-f",
                "lavfi",
                "-i",
                &format!("testsrc2=size={width}x48:rate=10:duration=0.5"),
                "-c:v",
                "libx264",
                "-threads",
                "1",
                "-x264-params",
                "keyint=1:repeat-headers=1",
                "-pix_fmt",
                "yuv420p",
                "-color_primaries",
                "bt709",
                "-color_trc",
                "bt709",
                "-colorspace",
                "bt709",
                "-color_range",
                "tv",
            ])
            .arg(&path)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        sets.push(packets_from_path(&path));
    }
    let (config, first) = sets.remove(0);
    let (_, mut second) = sets.remove(0);
    for packet in &mut second {
        packet.pts = packet
            .pts
            .map(|time| time.checked_add(Time::new(1, 1).unwrap()).unwrap());
        packet.dts = packet
            .dts
            .map(|time| time.checked_add(Time::new(1, 1).unwrap()).unwrap());
    }
    let mut decoder = PacketDecoder::new(config, PacketDecodeOptions::default()).unwrap();
    let mut output = Vec::new();
    let mut rejected = false;
    for packet in first.iter().chain(&second) {
        decoder.send_packet(packet).unwrap();
        loop {
            match decoder.receive_frame() {
                Ok(FrameReceive::Frame(frame)) => output.push(frame),
                Ok(FrameReceive::NeedInput | FrameReceive::End) => break,
                Err(error) => {
                    assert_eq!(error.kind(), ErrorKind::Incompatible);
                    rejected = true;
                    break;
                }
            }
        }
        if rejected {
            break;
        }
    }
    assert!(
        rejected,
        "real H.264 SPS resolution change was not rejected"
    );
}
