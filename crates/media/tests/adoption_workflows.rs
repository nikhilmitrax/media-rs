//! Practical codec/container coverage against independent FFmpeg CLI output.
//! Run with MEDIA_REQUIRE_FFMPEG_TESTS=1 and --features ffmpeg-audio.
#![cfg(feature = "ffmpeg-audio")]

use media::{
    Session,
    backend::{ffmpeg::codec::VideoDecodeOptions, native::audio::AudioDecodeOptions},
    io::source::Source,
    pipeline::{sampling::TemporalSelection, session::CpuUsage},
    prelude::*,
    video_index::IndexBuildLimits,
};
use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
    sync::atomic::{AtomicU64, Ordering},
};

struct Fixtures(PathBuf);
impl Fixtures {
    fn new() -> Option<Self> {
        if Command::new(binary()).arg("-version").output().is_err() {
            assert!(
                std::env::var_os("MEDIA_REQUIRE_FFMPEG_TESTS").is_none(),
                "FFmpeg fixture generator is required"
            );
            eprintln!("UNEXERCISED: adoption workflows require ffmpeg/ffprobe");
            return None;
        }
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "media-adoption-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Some(Self(path))
    }
    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}
impl Drop for Fixtures {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn binary() -> std::ffi::OsString {
    std::env::var_os("MEDIA_FFMPEG_BIN").unwrap_or_else(|| "ffmpeg".into())
}
fn checked(command: &mut Command) -> Output {
    let output = command.output().unwrap();
    assert!(
        output.status.success(),
        "{command:?}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    output
}
fn idle(session: &Session) {
    assert_eq!(session.live_cpu_usage(), CpuUsage::default());
    assert_eq!(session.live_batches(), 0);
    assert_eq!(session.live_host_bytes(), 0);
}
fn video(path: &Path, codec: &str, extra: &[&str], vfr: bool) {
    let filter = if vfr {
        "select=not(eq(mod(n\\,5)\\,2)),scale=in_color_matrix=bt601:out_color_matrix=bt709:out_range=tv"
    } else {
        "scale=in_color_matrix=bt601:out_color_matrix=bt709:out_range=tv"
    };
    checked(
        Command::new(binary())
            .args([
                "-nostdin",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=192x144:rate=30000/1001:duration=1.8",
                "-vf",
                filter,
                "-an",
                "-c:v",
                codec,
                "-threads:v",
                "1",
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
                "-fps_mode",
                "vfr",
            ])
            .args(extra)
            .arg(path),
    );
}
fn ffprobe(path: &Path, arguments: &[&str]) -> String {
    let executable = std::env::var_os("MEDIA_FFPROBE_BIN").unwrap_or_else(|| "ffprobe".into());
    String::from_utf8(
        checked(
            Command::new(executable)
                .args(["-v", "error", "-select_streams", "v:0"])
                .args(arguments)
                .arg(path),
        )
        .stdout,
    )
    .unwrap()
}
fn reference_video(path: &Path) -> (Vec<Time>, Vec<u8>) {
    let base = ffprobe(
        path,
        &[
            "-show_entries",
            "stream=time_base",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
        ],
    );
    let mut parts = base.lines().next().unwrap().split('/');
    let numerator: i64 = parts.next().unwrap().parse().unwrap();
    let denominator: u32 = parts.next().unwrap().parse().unwrap();
    let pts = ffprobe(path, &["-show_entries", "frame=pts", "-of", "csv=p=0"])
        .lines()
        .filter_map(|line| line.split(',').next().unwrap().parse::<i64>().ok())
        .map(|ticks| Time::new(ticks * numerator, denominator).unwrap())
        .collect();
    let pixels = checked(
        Command::new(binary())
            .args(["-nostdin", "-v", "error", "-noautorotate", "-i"])
            .arg(path)
            .args([
                "-map",
                "0:v:0",
                "-threads:v",
                "1",
                "-sws_flags",
                "bilinear",
                "-pix_fmt",
                "rgb24",
                "-fps_mode",
                "passthrough",
                "-f",
                "rawvideo",
                "-",
            ]),
    )
    .stdout;
    (pts, pixels)
}
fn check_video(path: &Path) {
    let (expected_pts, expected_pixels) = reference_video(path);
    assert!(!expected_pts.is_empty());
    let session = Session::default();
    let asset = session.open(path).unwrap();
    let request = asset.video().unwrap();
    let decoded = request.read().unwrap();
    assert_eq!(
        decoded.frames.len(),
        expected_pts.len(),
        "{}",
        path.display()
    );
    let mut pixels = Vec::new();
    for (ordinal, (frame, expected)) in decoded.frames.iter().zip(&expected_pts).enumerate() {
        assert_eq!(
            frame.timing.source_pts,
            Some(*expected),
            "{} frame {ordinal}",
            path.display()
        );
        assert_eq!(frame.timing.ordinal, Some(ordinal as u64));
        pixels.extend_from_slice(frame.image.packed_bytes().unwrap());
    }
    assert_eq!(pixels.len(), expected_pixels.len());
    assert!(
        pixels == expected_pixels,
        "RGB pixels differ from CLI: {}",
        path.display()
    );
    let index = request.build_index(IndexBuildLimits::default()).unwrap();
    let ordinals = [decoded.frames.len() - 4, 2, decoded.frames.len() / 2, 2];
    let times: Vec<_> = ordinals
        .iter()
        .map(|&i| decoded.frames[i].timing.pts.unwrap())
        .collect();
    for policy in [
        TemporalSelection::DisplayedAt,
        TemporalSelection::Nearest,
        TemporalSelection::AtOrBefore,
        TemporalSelection::AtOrAfter,
    ] {
        let actual = request
            .clone()
            .with_index(index.clone())
            .sample(times.clone())
            .selection(policy)
            .duration_inference(DurationInference::NextTimestamp)
            .read()
            .unwrap_or_else(|error| panic!("{} {policy:?}: {error}", path.display()));
        assert_eq!(actual.frames.len(), ordinals.len());
        for (frame, &ordinal) in actual.frames.iter().zip(&ordinals) {
            assert_eq!(
                frame.timing.ordinal,
                Some(ordinal as u64),
                "{} {policy:?}",
                path.display()
            );
            assert_eq!(
                frame.image.packed_bytes().unwrap(),
                decoded.frames[ordinal].image.packed_bytes().unwrap()
            );
        }
        assert!(
            actual.frames[1]
                .image
                .storage()
                .shares_allocation(actual.frames[3].image.storage())
        );
    }
    drop((decoded, index));
    idle(&session);
}

#[test]
fn everyday_video_codecs_containers_and_open_gops_match_cli_and_indexed_reads() {
    let Some(fixtures) = Fixtures::new() else {
        return;
    };
    let cases: &[(&str, &str, &[&str], bool)] = &[
        ("h264.mp4", "libx264", &["-bf", "3", "-g", "12"], false),
        (
            "open-gop-vfr.mp4",
            "libx264",
            &[
                "-bf",
                "3",
                "-x264-params",
                "keyint=12:min-keyint=12:scenecut=0:open-gop=1",
            ],
            true,
        ),
        (
            "fragmented.mp4",
            "libx264",
            &[
                "-bf",
                "2",
                "-g",
                "12",
                "-movflags",
                "frag_keyframe+empty_moov",
            ],
            false,
        ),
        ("transport.ts", "libx264", &["-bf", "2", "-g", "12"], false),
        (
            "hevc-main10.mov",
            "libx265",
            &[
                "-pix_fmt",
                "yuv420p10le",
                "-x265-params",
                "pools=none:frame-threads=1:keyint=12:min-keyint=12:scenecut=0:open-gop=1:log-level=error",
                "-tag:v",
                "hvc1",
            ],
            true,
        ),
        (
            "vp9.webm",
            "libvpx-vp9",
            &["-deadline", "realtime", "-cpu-used", "8", "-g", "12"],
            true,
        ),
        (
            "av1.mkv",
            "libsvtav1",
            &["-preset", "12", "-svtav1-params", "lp=1", "-g", "12"],
            false,
        ),
    ];
    for &(name, codec, extra, vfr) in cases {
        let path = fixtures.path(name);
        video(&path, codec, extra, vfr);
        check_video(&path);
    }
    let rotated = fixtures.path("rotated.mov");
    checked(
        Command::new(binary())
            .args(["-nostdin", "-v", "error", "-i"])
            .arg(fixtures.path("h264.mp4"))
            .args(["-c", "copy", "-metadata:s:v:0", "rotate=90"])
            .arg(&rotated),
    );
    check_video(&rotated);
}

fn reference_audio(path: &Path) -> Vec<f32> {
    checked(
        Command::new(binary())
            .args(["-nostdin", "-v", "error", "-i"])
            .arg(path)
            .args(["-map", "0:a:0", "-threads:a", "1", "-f", "f32le", "-"]),
    )
    .stdout
    .chunks_exact(4)
    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
    .collect()
}

#[test]
fn common_audio_codecs_preserve_gapless_extent_and_continuous_resampling() {
    let Some(fixtures) = Fixtures::new() else {
        return;
    };
    for (name, codec, native) in [
        ("pcm.wav", "pcm_s16le", true),
        ("float.wav", "pcm_f32le", true),
        ("unsigned.wav", "pcm_u8", true),
        ("pcm24.wav", "pcm_s24le", true),
        ("pcm32.wav", "pcm_s32le", true),
        ("pcm64.wav", "pcm_s64le", false),
        ("double.wav", "pcm_f64le", true),
        ("lossless.flac", "flac", true),
        ("music.mp3", "libmp3lame", true),
        ("voice.ogg", "libopus", false),
        ("voice.webm", "libopus", false),
        ("phone.m4a", "aac", false),
        ("alac.m4a", "alac", false),
    ] {
        let path = fixtures.path(name);
        checked(
            Command::new(binary())
                .args([
                    "-nostdin",
                    "-v",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "aevalsrc=0.2*sin(2*PI*440*t)|0.15*sin(2*PI*880*t):s=48000:d=0.37",
                    "-c:a",
                    codec,
                    "-strict",
                    "experimental",
                    "-threads:a",
                    "1",
                ])
                .arg(&path),
        );
        let session = Session::default();
        let asset = session.open(&path).unwrap();
        let reference = reference_audio(&path);
        let request = asset.audio().unwrap().ffmpeg();
        // AAC's declared semantic tail may be shorter than untrimmed CLI output.
        let raw = request
            .clone()
            .raw_decoded()
            .read()
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        assert_eq!(raw.samples(), reference, "FFmpeg sample mismatch: {name}");
        drop(raw);
        let decoded = request
            .clone()
            .read()
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        if name == "pcm.wav" {
            assert_eq!(
                decoded.channel_layout().positions(),
                &[
                    media::core::media::ChannelPosition::Discrete(0),
                    media::core::media::ChannelPosition::Discrete(1)
                ]
            );
            assert!(
                request
                    .clone()
                    .downmix(ChannelLayout::mono())
                    .read()
                    .is_err(),
                "unpositioned channels require an explicit mix"
            );
        }
        assert_eq!(decoded.frames(), 17_760, "semantic sample extent: {name}");
        let matrix = media::backend::native::audio::MixMatrix::new(
            decoded.channel_layout().clone(),
            ChannelLayout::mono(),
            vec![0.5, 0.5],
        )
        .unwrap();
        let expected = request
            .clone()
            .mix(matrix.clone())
            .resample(16000)
            .read()
            .unwrap();
        let chunked = request
            .clone()
            .ffmpeg_options(media::backend::ffmpeg::audio::AudioDecodeOptions {
                max_block_frames: 137,
                ..Default::default()
            })
            .mix(matrix)
            .resample(16000)
            .read()
            .unwrap();
        assert_eq!(
            expected.samples(),
            chunked.samples(),
            "chunk boundary changed DSP: {name}"
        );
        assert_eq!(expected.frames(), 5920, "resampled sample count: {name}");
        if native {
            let native = asset
                .audio()
                .unwrap()
                .decode_options(AudioDecodeOptions {
                    max_block_frames: 137,
                    ..Default::default()
                })
                .read()
                .unwrap_or_else(|error| panic!("native {name}: {error}"));
            assert_eq!(
                native.frames(),
                decoded.frames(),
                "native gapless extent: {name}"
            );
            let worst = native
                .samples()
                .iter()
                .zip(decoded.samples())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            assert!(worst <= 0.0001, "native {name} differs by {worst}");
        }
        drop((decoded, expected, chunked));
        idle(&session);
    }
}

#[test]
fn speech_and_music_mp3_rates_handle_delay_spanning_multiple_packets() {
    let Some(fixtures) = Fixtures::new() else {
        return;
    };
    let session = Session::default();
    for (i, rate) in [8000_u32, 16000, 22050, 24000, 32000, 44100, 48000]
        .into_iter()
        .enumerate()
    {
        let path = fixtures.path(&format!("rate-{rate}.mp3"));
        let frames = rate as usize * 3 / 10 + 17;
        let input = format!("sine=frequency=440:sample_rate={rate}:duration=1");
        checked(
            Command::new(binary())
                .args([
                    "-nostdin",
                    "-v",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    &input,
                    "-af",
                    &format!("atrim=end_sample={frames}"),
                    "-ac",
                    if i % 2 == 0 { "1" } else { "2" },
                    "-c:a",
                    "libmp3lame",
                    "-threads:a",
                    "1",
                ])
                .arg(&path),
        );
        let asset = session.open(&path).unwrap();
        let native = asset
            .audio()
            .unwrap()
            .read()
            .unwrap_or_else(|error| panic!("native {rate} Hz: {error}"));
        assert_eq!(native.start(), Time::ZERO, "{rate} Hz priming origin");
        assert_eq!(native.frames(), frames, "{rate} Hz gapless frame count");
        let ffmpeg = asset
            .audio()
            .unwrap()
            .ffmpeg()
            .read()
            .unwrap_or_else(|error| panic!("FFmpeg {rate} Hz: {error}"));
        assert_eq!(ffmpeg.frames(), frames);
        let resampled = asset.audio().unwrap().resample(16000).read().unwrap();
        assert_eq!(resampled.frames(), (frames * 16000).div_ceil(rate as usize));
        drop((native, ffmpeg, resampled));
        idle(&session);
    }
}

#[test]
fn decoder_priming_exposes_actual_leading_coverage_and_reanchors_continuous_dsp() {
    let Some(fixtures) = Fixtures::new() else {
        return;
    };
    let path = fixtures.path("overlap.ogg");
    checked(
        Command::new(binary())
            .args([
                "-nostdin",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "aevalsrc=0.2*sin(2*PI*440*t)|0.15*sin(2*PI*880*t):s=48000:d=0.37",
                "-c:a",
                "vorbis",
                "-strict",
                "experimental",
                "-threads:a",
                "1",
            ])
            .arg(&path),
    );
    let session = Session::default();
    let request = session.open(&path).unwrap().audio().unwrap().ffmpeg();
    let header = request.probe().unwrap();
    let raw = request.clone().raw_decoded().read().unwrap();
    assert_eq!(raw.samples(), reference_audio(&path));
    let start = Time::from_samples(1024, 48000).unwrap();
    assert_eq!(raw.start(), start);
    assert!(start > header.start);
    let mut reader = request.clone().reader().unwrap();
    let mut frames = 0;
    while let Some(block) = reader.next_block().unwrap() {
        frames += block.frames();
    }
    assert_eq!(reader.source_coverage().unwrap().start(), start);
    assert_eq!(
        reader.ffmpeg_decode_report().unwrap().leading_gap,
        Some(TimeRange::new(header.start, start).unwrap())
    );
    let waveform = request.clone().read().unwrap();
    assert_eq!(waveform.frames(), frames);
    assert_eq!(
        waveform.samples(),
        &raw.samples()[..waveform.samples().len()]
    );
    let resampled = request.clone().resample(16000).read().unwrap();
    assert_eq!(resampled.start(), start);
    assert_eq!(resampled.frames(), (frames * 16000).div_ceil(48000));
    assert!(
        request
            .clone()
            .slice(TimeRange::new(Time::ZERO, Time::from_millis(100)).unwrap())
            .read()
            .is_err()
    );
    let plan = request.compile().unwrap();
    assert_eq!(plan.execute().unwrap().samples(), waveform.samples());
    drop((raw, reader, waveform, resampled, plan));
    idle(&session);
}

#[test]
fn explicit_sample_clock_recovers_broken_audio_timestamps_without_losing_samples() {
    let Some(fixtures) = Fixtures::new() else {
        return;
    };
    let path = fixtures.path("overlap.mka");
    checked(
        Command::new(binary())
            .args([
                "-nostdin",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "sine=sample_rate=48000:duration=0.37",
                "-af",
                "asetpts=PTS-gte(T\\,0.1)*0.04/TB",
                "-c:a",
                "pcm_s16le",
            ])
            .arg(&path),
    );
    let session = Session::default();
    let asset = session.open(&path).unwrap();
    assert_eq!(
        asset.audio().unwrap().ffmpeg().read().unwrap_err().kind(),
        ErrorKind::MissingData
    );
    let request = asset.audio().unwrap().decoded_sample_clock();
    let plan = request.compile().unwrap();
    assert!(plan.explain().contains("rebuilt"));
    let waveform = plan.execute().unwrap();
    assert_eq!(waveform.frames(), 17_760);
    assert_eq!(waveform.samples(), reference_audio(&path));
    let mut reader = request.clone().reader().unwrap();
    let mut next = None;
    let mut differing_sources = 0;
    while let Some(block) = reader.next_block().unwrap() {
        if let Some(next) = next {
            assert_eq!(block.start(), next);
        }
        if block.source_start() != Some(block.start()) {
            differing_sources += 1;
        }
        next = Some(block.end().unwrap());
    }
    assert!(differing_sources > 0);
    assert!(
        reader
            .ffmpeg_decode_report()
            .unwrap()
            .reconstructed_timestamps
            > 0
    );
    let resampled = request.resample(16000).read().unwrap();
    assert_eq!(resampled.frames(), 5920);
    drop((waveform, reader, resampled, plan));
    idle(&session);
}

fn image_bytes() -> Vec<u8> {
    media::backend::native::image::encode(
        &Image::from_packed(
            64,
            48,
            PixelFormat::Rgb8,
            (0..64 * 48 * 3).map(|i| (i * 17) as u8).collect::<Vec<_>>(),
        )
        .unwrap(),
        EncodeOptions::new(ImageFormat::Png),
    )
    .unwrap()
    .bytes
}

#[test]
#[ignore = "bounded multiworker adoption soak; run explicitly in release mode"]
fn concurrent_mixed_work_retained_outputs_cancellation_and_failures_retire_resources() {
    use std::{
        collections::VecDeque,
        sync::{Arc, Barrier, atomic::AtomicUsize},
        time::{Duration, Instant},
    };
    let fixtures = Fixtures::new().expect("explicit soak requires ffmpeg");
    let path = fixtures.path("video.mp4");
    video(&path, "libx264", &["-bf", "3", "-g", "12"], true);
    let wav = fixtures.path("audio.wav");
    checked(
        Command::new(binary())
            .args([
                "-nostdin",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "sine=sample_rate=48000:duration=0.37",
                "-c:a",
                "pcm_s16le",
            ])
            .arg(&wav),
    );
    let iterations: usize = std::env::var("MEDIA_ADOPTION_ITERATIONS")
        .unwrap_or_else(|_| "2400".into())
        .parse()
        .unwrap();
    let workers: usize = std::env::var("MEDIA_ADOPTION_WORKERS")
        .unwrap_or_else(|_| "8".into())
        .parse()
        .unwrap();
    assert!((1..=32).contains(&workers) && (100..=100_000).contains(&iterations));
    let session = Session::builder()
        .limits(
            Limits::new()
                .host_memory(512 * 1024 * 1024)
                .cpu_executions(workers)
                .cpu_codecs(workers)
                .cpu_codec_threads(workers)
                // Six retained image aliases per worker can each keep a
                // materialized batch alive, in addition to current work.
                .in_flight_batches(workers * 8),
        )
        .build()
        .unwrap();
    let request = session.open(&path).unwrap().video().unwrap();
    let index = request.build_index(IndexBuildLimits::default()).unwrap();
    let baseline = session.live_host_bytes();
    let image = session.from_source(Source::memory(image_bytes()));
    let audio = session.open(&wav).unwrap();
    let next = Arc::new(AtomicUsize::new(0));
    let barrier = Arc::new(Barrier::new(workers));
    let started = Instant::now();
    let mut threads = Vec::new();
    for _ in 0..workers {
        let (image, audio, request, index, next, barrier) = (
            image.clone(),
            audio.clone(),
            request.clone(),
            index.clone(),
            next.clone(),
            barrier.clone(),
        );
        threads.push(std::thread::spawn(move || {
            let mut retained = VecDeque::new();
            let mut latencies = Vec::new();
            barrier.wait();
            loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= iterations {
                    break;
                }
                let start = Instant::now();
                match i % 6 {
                    0 => {
                        let frame = image
                            .image()
                            .unwrap()
                            .resize(Fit::inside(32, 32))
                            .read()
                            .unwrap();
                        retained.push_back(frame);
                    }
                    1 => {
                        let mut stream = audio
                            .audio()
                            .unwrap()
                            .resample(16000)
                            .windows(Window::samples(320).hop(160))
                            .stream()
                            .unwrap();
                        let mut count = 0;
                        while let Some(block) = stream.next_block().unwrap() {
                            assert_eq!(block.frames(), 320);
                            count += 1;
                        }
                        assert_eq!(count, 36);
                    }
                    2 => {
                        let batch = request
                            .clone()
                            .with_index(index.clone())
                            .sample([
                                Time::from_millis(1200),
                                Time::from_millis(100),
                                Time::from_millis(1200),
                            ])
                            .duration_inference(DurationInference::NextTimestamp)
                            .read()
                            .unwrap();
                        assert_eq!(batch.frames.len(), 3);
                        retained.push_back(batch.frames[0].image.clone());
                        drop(batch);
                    }
                    3 => {
                        let mut reader = request.reader().unwrap();
                        let frame = reader.next_frame().unwrap().unwrap();
                        reader.cancellation().cancel();
                        assert_eq!(
                            reader.next_frame().unwrap_err().kind(),
                            ErrorKind::Cancelled
                        );
                        drop(reader);
                        assert!(!frame.image.packed_bytes().unwrap().is_empty());
                        retained.push_back(frame.image);
                    }
                    4 => {
                        let bad = image
                            .session()
                            .from_source(Source::memory(b"broken PNG".to_vec()));
                        assert!(bad.image().unwrap().read().is_err());
                        assert!(
                            request
                                .clone()
                                .decode_options(VideoDecodeOptions {
                                    max_frame_bytes: 1,
                                    ..Default::default()
                                })
                                .read()
                                .is_err()
                        );
                    }
                    _ => {
                        let mut reader = audio.audio().unwrap().reader().unwrap();
                        assert!(reader.next_block().unwrap().is_some());
                        reader.cancellation().cancel();
                        assert_eq!(
                            reader.next_block().unwrap_err().kind(),
                            ErrorKind::Cancelled
                        );
                    }
                }
                if retained.len() > 6 {
                    retained.pop_front();
                }
                for image in &retained {
                    assert!(!image.packed_bytes().unwrap().is_empty());
                }
                if i % 31 == 0 {
                    std::thread::sleep(Duration::from_millis(1));
                }
                latencies.push(start.elapsed().as_nanos());
            }
            drop(retained);
            latencies
        }));
    }
    let joined: Vec<_> = threads.into_iter().map(|thread| thread.join()).collect();
    let mut latencies: Vec<_> = joined
        .into_iter()
        .flat_map(|result| result.unwrap())
        .collect();
    latencies.sort_unstable();
    assert_eq!(latencies.len(), iterations);
    assert_eq!(session.live_host_bytes(), baseline);
    assert_eq!(session.live_batches(), 0);
    assert_eq!(session.live_cpu_usage(), CpuUsage::default());
    eprintln!(
        "ADOPTION_SOAK iterations={iterations} workers={workers} seconds={:.3} p50_ms={:.3} p95_ms={:.3} p99_ms={:.3} peak_accounted_bytes={}",
        started.elapsed().as_secs_f64(),
        latencies[iterations / 2] as f64 / 1e6,
        latencies[iterations * 95 / 100] as f64 / 1e6,
        latencies[iterations * 99 / 100] as f64 / 1e6,
        session.peak_host_bytes()
    );
    drop((index, request, image, audio));
    idle(&session);
}

#[test]
#[ignore = "one-hour encoded input and repeated index persistence; run explicitly"]
fn hour_long_video_indexes_persists_and_samples_without_prefix_redecoding() {
    let fixtures = Fixtures::new().expect("explicit long recording test requires ffmpeg");
    let path = fixtures.path("one-hour.mp4");
    checked(
        Command::new(binary())
            .args([
                "-nostdin",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=64x48:rate=30:duration=3600",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-threads:v",
                "1",
                "-g",
                "300",
                "-bf",
                "2",
                "-colorspace",
                "bt709",
                "-color_range",
                "tv",
            ])
            .arg(&path),
    );
    let session = Session::default();
    let request = session.open(&path).unwrap().video().unwrap();
    let start = std::time::Instant::now();
    let index = request.build_index(IndexBuildLimits::default()).unwrap();
    assert_eq!(index.len(), 108_000);
    let cache = fixtures.path("one-hour.index");
    index.save(&cache).unwrap();
    drop(index);
    let index = request
        .load_index(&cache, IndexBuildLimits::default())
        .unwrap();
    let times = [
        Time::from_millis(3_599_000),
        Time::from_millis(31_000),
        Time::from_millis(1_801_000),
        Time::from_millis(31_000),
    ];
    for _ in 0..8 {
        let batch = request
            .clone()
            .with_index(index.clone())
            .sample(times)
            .duration_inference(DurationInference::NextTimestamp)
            .read()
            .unwrap();
        assert_eq!(
            batch
                .frames
                .iter()
                .map(|frame| frame.timing.ordinal.unwrap())
                .collect::<Vec<_>>(),
            vec![107_970, 930, 54_030, 930]
        );
        assert!(
            batch.report.decoded_frames < 1000,
            "{} dependency frames",
            batch.report.decoded_frames
        );
        assert!(batch.report.seeks <= 3);
    }
    eprintln!(
        "ADOPTION_LONG frames=108000 repeated_queries=8 seconds={:.3}",
        start.elapsed().as_secs_f64()
    );
    drop(index);
    idle(&session);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires real NVDEC/CUDA; transport header and packet DTS regression"]
fn indexed_cuda_transport_stream_and_open_gop_use_real_decode_timestamps() {
    let fixtures = Fixtures::new().expect("GPU adoption test requires ffmpeg");
    let session = Session::default();
    for name in ["transport.ts", "open-gop.mp4"] {
        let path = fixtures.path(name);
        video(
            &path,
            "libx264",
            &[
                "-bf",
                "3",
                "-x264-params",
                "keyint=12:min-keyint=12:scenecut=0:open-gop=1",
            ],
            true,
        );
        let asset = session.open(&path).unwrap();
        let cpu = asset.video().unwrap().read().unwrap();
        let ordinals = [cpu.frames.len() - 4, 2, cpu.frames.len() / 2, 2];
        let times: Vec<_> = ordinals
            .iter()
            .map(|&ordinal| cpu.frames[ordinal].timing.pts.unwrap())
            .collect();
        let request = asset
            .video()
            .unwrap()
            .cuda(0, media::backend::cuda::CudaLimits::default());
        let index = request.build_index(IndexBuildLimits::default()).unwrap();
        assert!(index.report().hardware_decode);
        let sampled = request
            .with_index(index.clone())
            .sample(times)
            .selection(TemporalSelection::Nearest)
            .read()
            .unwrap();
        for (frame, &ordinal) in sampled.frames.iter().zip(&ordinals) {
            assert_eq!(frame.frame().timing.ordinal, Some(ordinal as u64));
            assert_eq!(
                frame.frame().timing.source_pts,
                cpu.frames[ordinal].timing.source_pts
            );
        }
        assert_eq!(sampled.report.decoded_host_transfer_bytes, 0);
        drop((sampled, index, cpu));
        for _ in 0..200 {
            if session.live_batches() == 0 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        idle(&session);
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "real CUDA/NVDEC two-worker adoption soak with retained tensors"]
fn indexed_cuda_concurrent_repeated_requests_retire_native_and_owned_resources() {
    use media::backend::cuda::{CudaLimits, PreprocessSpec};
    use std::{
        collections::VecDeque,
        sync::{Arc, Barrier},
    };
    let fixtures = Fixtures::new().expect("GPU soak requires ffmpeg");
    let path = fixtures.path("gpu.mp4");
    video(&path, "libx264", &["-bf", "3", "-g", "12"], false);
    let limits = CudaLimits {
        memory_bytes: 8 * 1024 * 1024,
        ..Default::default()
    };
    let session = Session::builder()
        .limits(Limits::new().in_flight_batches(8))
        .cuda_limits(media::cuda_session::CudaSessionLimits {
            max_codec_sessions: 8,
            ..Default::default()
        })
        .build()
        .unwrap();
    let request = session
        .open(&path)
        .unwrap()
        .video()
        .unwrap()
        .cuda(0, limits);
    let index = request.build_index(IndexBuildLimits::default()).unwrap();
    let plan = request
        .with_index(index.clone())
        .sample([
            Time::from_millis(1200),
            Time::from_millis(100),
            Time::from_millis(1200),
        ])
        .duration_inference(DurationInference::NextTimestamp)
        .to_tensor(PreprocessSpec::new(16, 16).unwrap())
        .compile()
        .unwrap();
    let context = session.cuda_context(0, limits).unwrap();
    let barrier = Arc::new(Barrier::new(2));
    let mut threads = Vec::new();
    for _ in 0..2 {
        let (plan, context, barrier) = (plan.clone(), context.clone(), barrier.clone());
        threads.push(std::thread::spawn(move || {
            let stream = context.stream().unwrap();
            let mut retained = VecDeque::new();
            let mut expected = None;
            barrier.wait();
            for _ in 0..64 {
                let batch = plan.execute().unwrap();
                assert_eq!(batch.report.decoded_host_transfer_bytes, 0);
                let values = batch.tensor.download(&stream).unwrap();
                if let Some(expected) = &expected {
                    assert_eq!(&values, expected);
                } else {
                    expected = Some(values);
                }
                retained.push_back(batch.tensor.clone());
                if retained.len() > 2 {
                    retained.pop_front();
                }
                drop(batch);
            }
            drop(retained);
            stream.synchronize().unwrap();
        }));
    }
    let joined: Vec<_> = threads.into_iter().map(|thread| thread.join()).collect();
    for result in joined {
        result.unwrap();
    }
    drop((plan, index));
    for _ in 0..400 {
        if session.live_cuda_codec_sessions() == 0 && session.live_batches() == 0 {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    assert_eq!(context.memory_used(), 0);
    assert_eq!(session.live_cuda_codec_sessions(), 0);
    assert_eq!(session.reserved_cuda_native_bytes(), 0);
    eprintln!(
        "ADOPTION_CUDA_SOAK requests=128 workers=2 retained_tensors_per_worker=2 peak_owned_bytes={} peak_known_native_bytes={}",
        context.memory_peak(),
        session.peak_reserved_cuda_native_bytes()
    );
    idle(&session);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "real NVDEC 1080p/4K H264 and HEVC Main10 tensor conformance"]
fn indexed_cuda_large_surfaces_match_prefix_tensors_and_cpu_frame_identities() {
    use media::backend::cuda::{CudaLimits, PreprocessSpec};
    let fixtures = Fixtures::new().expect("large-surface GPU test requires ffmpeg");
    let session = Session::builder()
        .limits(Limits::new().in_flight_batches(8))
        .cuda_limits(media::cuda_session::CudaSessionLimits {
            max_codec_sessions: 8,
            ..Default::default()
        })
        .build()
        .unwrap();
    for (name, size, codec, pixels) in [
        ("1080.mp4", "1920x1080", "libx264", "yuv420p"),
        ("4k.mp4", "3840x2160", "libx264", "yuv420p"),
        ("4k-main10.mp4", "3840x2160", "libx265", "yuv420p10le"),
    ] {
        let path = fixtures.path(name);
        let input = format!("testsrc2=size={size}:rate=30:duration=0.2");
        let mut command = Command::new(binary());
        command.args([
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            &input,
            "-vf",
            "scale=in_color_matrix=bt601:out_color_matrix=bt709:out_range=tv",
            "-c:v",
            codec,
            "-preset",
            "ultrafast",
            "-threads:v",
            "1",
            "-bf",
            "2",
            "-g",
            "3",
            "-pix_fmt",
            pixels,
            "-colorspace",
            "bt709",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            "-color_range",
            "tv",
        ]);
        if codec == "libx265" {
            command.args(["-x265-params","pools=none:frame-threads=1:keyint=3:min-keyint=3:scenecut=0:open-gop=0:log-level=error","-tag:v","hvc1"]);
        }
        checked(command.arg(&path));
        let asset = session.open(&path).unwrap();
        let cpu = asset.video().unwrap().read().unwrap();
        assert_eq!(cpu.frames.len(), 6);
        let ordinals = [4, 1, 2, 1];
        let times: Vec<_> = ordinals
            .iter()
            .map(|&ordinal| cpu.frames[ordinal].timing.pts.unwrap())
            .collect();
        let request = asset.video().unwrap().cuda(0, CudaLimits::default());
        let sampled = request
            .clone()
            .sample(times)
            .duration_inference(DurationInference::NextTimestamp);
        let prefix = sampled.read().unwrap();
        for (frame, &ordinal) in prefix.frames.iter().zip(&ordinals) {
            assert_eq!(
                frame.frame().timing.source_pts,
                cpu.frames[ordinal].timing.source_pts
            );
            assert_eq!(frame.frame().timing.ordinal, Some(ordinal as u64));
        }
        let prefix = prefix
            .into_tensor(PreprocessSpec::new(224, 224).unwrap())
            .unwrap();
        let context = prefix.context().clone();
        let stream = context.stream().unwrap();
        let expected = prefix.tensor.download(&stream).unwrap();
        drop(prefix);
        let index = request.build_index(IndexBuildLimits::default()).unwrap();
        let actual = sampled
            .with_index(index.clone())
            .to_tensor(PreprocessSpec::new(224, 224).unwrap())
            .read()
            .unwrap();
        assert_eq!(actual.report.decoded_host_transfer_bytes, 0);
        assert_eq!(actual.tensor.download(&stream).unwrap(), expected);
        drop((actual, index, cpu, stream));
        for _ in 0..400 {
            if session.live_batches() == 0 && session.live_cuda_codec_sessions() == 0 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert_eq!(context.memory_used(), 0);
        assert_eq!(session.live_cuda_codec_sessions(), 0);
        idle(&session);
        eprintln!(
            "ADOPTION_CUDA_LARGE source={name} size={size} prefix_index_tensors_equal=true cpu_frame_ids_match=true"
        );
    }
}

#[test]
#[ignore = "deterministic malformed-input mutation smoke, explicitly bounded"]
fn mutated_media_returns_errors_or_valid_bounded_outputs_without_panics_or_leaks() {
    let fixtures = Fixtures::new().expect("explicit mutation test requires ffmpeg");
    let video_path = fixtures.path("seed.mp4");
    video(&video_path, "libx264", &["-bf", "2"], false);
    let wav = fixtures.path("seed.wav");
    checked(
        Command::new(binary())
            .args([
                "-nostdin",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "sine=sample_rate=8000:duration=0.04",
                "-c:a",
                "pcm_s16le",
            ])
            .arg(&wav),
    );
    let seeds = [
        image_bytes(),
        fs::read(wav).unwrap(),
        fs::read(video_path).unwrap(),
    ];
    let session = Session::builder()
        .limits(
            Limits::new()
                .host_memory(32 * 1024 * 1024)
                .encoded_bytes(1024 * 1024)
                .image_pixels(512 * 512),
        )
        .build()
        .unwrap();
    let count: usize = std::env::var("MEDIA_ADOPTION_MUTATIONS")
        .unwrap_or_else(|_| "1200".into())
        .parse()
        .unwrap();
    assert!((1..=100_000).contains(&count));
    let mut random = 0x1944_a765_9912_0813_u64;
    for i in 0..count {
        let mut bytes = seeds[i % seeds.len()].clone();
        random ^= random << 13;
        random ^= random >> 7;
        random ^= random << 17;
        match i % 4 {
            0 => bytes.truncate(random as usize % bytes.len()),
            1 => {
                let offset = random as usize % bytes.len();
                bytes[offset] ^= (random >> 32) as u8;
            }
            2 => {
                let offset = random as usize % bytes.len();
                let end = (offset + 16).min(bytes.len());
                bytes[offset..end].fill(0xff);
            }
            _ => bytes = bytes[(random as usize % bytes.len())..].to_vec(),
        }
        let asset = session.from_source(Source::memory(bytes));
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            drop(asset.image().unwrap().read());
            drop(
                asset
                    .audio()
                    .unwrap()
                    .decode_options(AudioDecodeOptions {
                        max_probe_bytes: 64 * 1024,
                        max_packet_bytes: 64 * 1024,
                        max_buffered_samples: 16384,
                        ..Default::default()
                    })
                    .read(),
            );
            drop(
                asset
                    .video()
                    .unwrap()
                    .decode_options(VideoDecodeOptions {
                        max_frame_bytes: 1024 * 1024,
                        probe: media::backend::ffmpeg::format::ProbeOptions {
                            max_read_bytes: 64 * 1024,
                            analyze_duration_us: 100_000,
                        },
                        ..Default::default()
                    })
                    .sample([Time::ZERO])
                    .selection(TemporalSelection::Nearest)
                    .read(),
            );
        }));
        assert!(
            result.is_ok(),
            "mutation {i} seed {} caused a panic",
            i % seeds.len()
        );
        idle(&session);
    }
    eprintln!("ADOPTION_MUTATIONS cases={count} routes=3");
}

#[test]
#[ignore = "bounded mutation of persisted timing-cache lengths, flags and times"]
fn mutated_media_index_cache_is_bounded_even_with_recomputed_checksums() {
    let fixtures = Fixtures::new().expect("cache mutation requires ffmpeg");
    let path = fixtures.path("source.mp4");
    video(&path, "libx264", &["-bf", "2", "-g", "12"], false);
    let session = Session::default();
    let request = session.open(&path).unwrap().video().unwrap();
    let index = request.build_index(IndexBuildLimits::default()).unwrap();
    let cache = fixtures.path("source.index");
    index.save(&cache).unwrap();
    let seed = fs::read(&cache).unwrap();
    drop(index);
    let mut random = 0x8a17_b663_908f_5421_u64;
    for i in 0..1200 {
        random ^= random << 13;
        random ^= random >> 7;
        random ^= random << 17;
        let mut bytes = seed.clone();
        let position = random as usize % (bytes.len() - 8);
        if i % 3 == 0 {
            bytes.truncate(position);
        } else {
            bytes[position] ^= ((random >> 32) as u8) | 1;
            if i % 2 == 0 {
                // Trusted caches are not authenticated. Exercise the actual
                // bounded parser rather than stopping every case at checksum.
                let mut checksum = 0xcbf2_9ce4_8422_2325_u64;
                let body = bytes.len() - 8;
                for &byte in &bytes[..body] {
                    checksum ^= u64::from(byte);
                    checksum = checksum.wrapping_mul(0x100_0000_01b3);
                }
                bytes[body..].copy_from_slice(&checksum.to_le_bytes());
            }
        }
        fs::write(&cache, bytes).unwrap();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            request.load_index(
                &cache,
                IndexBuildLimits {
                    max_frames: 1024,
                    max_index_bytes: 1024 * 1024,
                    max_source_bytes: 1024 * 1024,
                },
            )
        }));
        let loaded = result.unwrap_or_else(|_| panic!("cache mutation {i} panicked"));
        if let Ok(index) = &loaded {
            assert!(index.len() <= 1024);
        }
        drop(loaded);
        idle(&session);
    }
    eprintln!("ADOPTION_CACHE_MUTATIONS cases=1200 recomputed_checksums=true");
}

#[test]
fn wrong_media_empty_inputs_and_short_sources_fail_without_leaking_or_overwriting() {
    let Some(fixtures) = Fixtures::new() else {
        return;
    };
    let session = Session::default();
    let destination = fixtures.path("keep.png");
    fs::write(&destination, b"existing destination").unwrap();
    for bytes in [
        vec![],
        b"not a media file".to_vec(),
        b"\x89PNG\r\n\x1a\n".to_vec(),
        b"RIFF\xff\xff\xff\xffWAVEfmt ".to_vec(),
    ] {
        let asset = session.from_source(Source::memory(bytes));
        assert!(
            asset
                .image()
                .unwrap()
                .allow_metadata_loss()
                .write(&destination)
                .is_err()
        );
        assert_eq!(fs::read(&destination).unwrap(), b"existing destination");
        assert!(asset.audio().unwrap().read().is_err());
        assert!(asset.audio().unwrap().ffmpeg().read().is_err());
        assert!(asset.video().unwrap().read().is_err());
        idle(&session);
    }
    let path = fixtures.path("video.mp4");
    video(&path, "libx264", &["-bf", "2"], false);
    let asset = session.open(&path).unwrap();
    assert!(
        asset
            .video()
            .unwrap()
            .sample(Vec::<Time>::new())
            .read()
            .unwrap()
            .frames
            .is_empty()
    );
    assert!(
        asset
            .video()
            .unwrap()
            .sample([Time::from_millis(9_000)])
            .duration_inference(DurationInference::NextTimestamp)
            .read()
            .is_err()
    );
    assert!(
        asset
            .video()
            .unwrap()
            .decode_options(VideoDecodeOptions {
                max_frame_bytes: 1,
                ..Default::default()
            })
            .read()
            .is_err()
    );
    idle(&session);
}
