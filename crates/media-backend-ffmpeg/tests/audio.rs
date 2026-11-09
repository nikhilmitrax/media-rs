//! Real libavcodec AAC skip, semantic tail, block and timeline references.
#![cfg(feature = "audio")]
use media_backend_ffmpeg::audio::{
    AudioDecodeOptions, AudioReader, AudioTimeline, AudioTrimPolicy,
};
use media_core::{error::ErrorKind, media::ChannelLayout, time::Time};
use media_io::source::Source;
use std::{
    fs,
    path::PathBuf,
    process::Command,
    sync::atomic::{AtomicUsize, Ordering},
};
struct Fixture {
    path: PathBuf,
    directory: PathBuf,
}
fn binary() -> std::ffi::OsString {
    std::env::var_os("MEDIA_FFMPEG_BIN").unwrap_or_else(|| "ffmpeg".into())
}
impl Fixture {
    fn aac() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let directory = std::env::temp_dir().join(format!(
            "media-ffaudio-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&directory).unwrap();
        let path = directory.join("tone.m4a");
        let output = Command::new(binary())
            .args([
                "-nostdin",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:sample_rate=48000:duration=0.101",
                "-c:a",
                "aac",
                "-threads:a",
                "1",
                "-y",
            ])
            .arg(&path)
            .output()
            .expect("FFmpeg AAC fixture generator required");
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        Self { path, directory }
    }
    fn reference(&self) -> Vec<f32> {
        let output = Command::new(binary())
            .args(["-nostdin", "-v", "error", "-i"])
            .arg(&self.path)
            .args(["-map", "0:a:0", "-threads:a", "1", "-f", "f32le", "-"])
            .output()
            .unwrap();
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
fn decode(
    source: &Source,
    options: AudioDecodeOptions,
) -> (Vec<f32>, media_backend_ffmpeg::audio::AudioDecodeReport) {
    let mut reader = AudioReader::open(source, options).unwrap();
    let mut samples = Vec::new();
    while let Some(block) = reader.next_block().unwrap() {
        assert!(block.frames() <= options.max_block_frames);
        assert_eq!(
            block.start(),
            Time::from_samples(samples.len() as i64, 48000).unwrap()
        );
        assert_eq!(block.source_start(), Some(block.start()));
        samples.extend_from_slice(block.samples());
    }
    (samples, reader.report().clone())
}
#[test]
fn aac_priming_is_applied_once_and_semantic_end_excludes_declared_tail_padding() {
    let fixture = Fixture::aac();
    let source = Source::file(&fixture.path).unwrap();
    let reader = AudioReader::open(&source, AudioDecodeOptions::default()).unwrap();
    assert_eq!(reader.info().start, Time::ZERO);
    assert_eq!(reader.info().layout, ChannelLayout::mono());
    assert_eq!(reader.info().frames, Some(4848));
    let cli = fixture.reference();
    assert_eq!(cli.len(), 5120);
    let (semantic, report) = decode(&source, AudioDecodeOptions::default());
    assert_eq!(semantic.len(), 4848);
    assert_eq!(semantic, &cli[..4848]);
    assert_eq!(report.codec_skip_start_requests, 1024);
    assert_eq!(report.semantic_trim_end, 272);
    assert_eq!(report.semantic_trim_start, 0);
    assert!(report.finished);
    let (raw, report) = decode(
        &source,
        AudioDecodeOptions {
            trim: AudioTrimPolicy::RawDecoded,
            ..Default::default()
        },
    );
    assert_eq!(raw, cli);
    assert_eq!(report.semantic_trim_end, 0);
    assert!(report.finished);
}
#[test]
fn output_chunking_is_exact_and_probe_does_not_decode_or_create_normalizer_work() {
    let fixture = Fixture::aac();
    let source = Source::file(&fixture.path).unwrap();
    let reader = AudioReader::open(&source, AudioDecodeOptions::default()).unwrap();
    assert_eq!(reader.report().decoded_frames, 0);
    assert_eq!(reader.report().packets_sent, 0);
    let (full, _) = decode(&source, AudioDecodeOptions::default());
    for frames in [1, 137, 4096] {
        let (chunked, report) = decode(
            &source,
            AudioDecodeOptions {
                max_block_frames: frames,
                ..Default::default()
            },
        );
        assert_eq!(chunked, full);
        assert_eq!(report.output_frames, 4848);
    }
    let (asset, _) = decode(
        &source,
        AudioDecodeOptions {
            timeline: AudioTimeline::Asset,
            ..Default::default()
        },
    );
    assert_eq!(asset, full);
}
#[test]
fn limits_cancellation_and_failed_readers_are_explicit() {
    let fixture = Fixture::aac();
    let source = Source::file(&fixture.path).unwrap();
    assert!(
        AudioReader::open(
            &source,
            AudioDecodeOptions {
                max_block_frames: 0,
                ..Default::default()
            }
        )
        .is_err()
    );
    let mut reader = AudioReader::open(&source, AudioDecodeOptions::default()).unwrap();
    reader.cancellation().cancel();
    assert_eq!(
        reader.next_block().unwrap_err().kind(),
        ErrorKind::Cancelled
    );
    assert_eq!(
        reader.next_block().unwrap_err().kind(),
        ErrorKind::InvalidInput
    );
    let mut reader = AudioReader::open(
        &source,
        AudioDecodeOptions {
            max_buffered_samples: 64,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(
        reader.next_block().unwrap_err().kind(),
        ErrorKind::ResourceExhausted
    );
    assert_eq!(
        reader.next_block().unwrap_err().kind(),
        ErrorKind::InvalidInput
    );
    let mut reader = AudioReader::open(
        &source,
        AudioDecodeOptions {
            max_packet_bytes: 1,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(
        reader.next_block().unwrap_err().kind(),
        ErrorKind::ResourceExhausted
    );
}

#[test]
fn normalized_surround_pcm_retains_explicit_speaker_positions_and_sample_values() {
    use media_core::media::ChannelPosition;
    let fixture = Fixture::aac();
    let path = fixture.directory.join("surround.wav");
    let output = Command::new(binary())
        .args([
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "aevalsrc=0.1|0.2|0.3|0.4|0.5|0.6:s=48000:d=0.005:c=5.1(side)",
            "-c:a",
            "pcm_f32le",
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
    let source = Source::file(&path).unwrap();
    let mut reader = AudioReader::open(
        &source,
        AudioDecodeOptions {
            max_block_frames: 77,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(
        reader.info().layout.positions(),
        &[
            ChannelPosition::FrontLeft,
            ChannelPosition::FrontRight,
            ChannelPosition::FrontCenter,
            ChannelPosition::Lfe,
            ChannelPosition::SideLeft,
            ChannelPosition::SideRight
        ]
    );
    let mut frames = 0;
    while let Some(block) = reader.next_block().unwrap() {
        frames += block.frames();
        for frame in block.samples().as_chunks::<6>().0 {
            assert_eq!(frame, &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        }
    }
    assert_eq!(frames, 240);
    assert!(reader.report().finished);
}
