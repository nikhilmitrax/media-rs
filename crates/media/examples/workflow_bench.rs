//! Reproducible complete CPU workflow measurements against the same backends.
//! See benches/README.md for semantic equivalence, counters, and interpretation.
use media::io::{
    sink::{FinishedSink, Sink},
    source::Source,
};
use media::{
    backend::{
        ffmpeg::{
            codec::{VideoDecodeOptions, VideoReader as NativeVideoReader},
            encode,
            format::{self, PacketReader, ProbeOptions},
        },
        native::{
            audio::{AudioDecodeOptions, AudioReader, MixMatrix},
            image as native_image,
            resample::{AudioWindows, Resampler},
        },
    },
    prelude::*,
};
use std::{
    collections::{VecDeque, hash_map::DefaultHasher},
    fs::{self, File, OpenOptions},
    hash::Hasher,
    io::{self, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    process::Command,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

const HOST_BYTES: usize = 512 * 1024 * 1024;
const ENCODED_BYTES: usize = 16 * 1024 * 1024;
const MODES: [&str; 8] = [
    "image",
    "audio",
    "video",
    "sparse",
    "dense",
    "remux",
    "transcode",
    "slowconsumer",
];

struct CountedFile {
    file: File,
    bytes: Arc<AtomicU64>,
}
impl Read for CountedFile {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        let n = self.file.read(buffer)?;
        self.bytes.fetch_add(n as u64, Ordering::Relaxed);
        Ok(n)
    }
}
impl Seek for CountedFile {
    fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
        self.file.seek(position)
    }
}
fn counted_source(path: &Path) -> Result<(Source, Arc<AtomicU64>)> {
    let path = fs::canonicalize(path).map_err(Error::io)?;
    let length = fs::metadata(&path).map_err(Error::io)?.len();
    let bytes = Arc::new(AtomicU64::new(0));
    let counter = bytes.clone();
    let source = Source::from_seekable_factory(
        format!("benchmark:{}", path.display()),
        Some(format!("immutable-fixture-length:{length}")),
        Some(length),
        move || {
            Ok(CountedFile {
                file: File::open(&path)?,
                bytes: counter.clone(),
            })
        },
    )?;
    Ok((source, bytes))
}
fn session() -> Result<Session> {
    Session::builder()
        .limits(
            Limits::default()
                .host_memory(HOST_BYTES)
                .encoded_bytes(ENCODED_BYTES),
        )
        .build()
}
fn audio_options() -> AudioDecodeOptions {
    AudioDecodeOptions {
        max_packet_bytes: 65536,
        max_buffered_samples: 65536,
        max_block_frames: 4096,
        ..AudioDecodeOptions::default()
    }
}
fn video_options() -> VideoDecodeOptions {
    VideoDecodeOptions {
        threads: 1,
        max_frame_bytes: 16 * 1024 * 1024,
        ..VideoDecodeOptions::default()
    }
}
fn transcode_options() -> TranscodeOptions {
    TranscodeOptions {
        width: Some(160),
        height: Some(90),
        filter: ResizeFilter::Bilinear,
        bit_rate: 500_000,
        preset: "veryfast".into(),
        threads: 1,
        gop: 30,
        max_b_frames: 2,
        max_frame_bytes: 16 * 1024 * 1024,
        ..TranscodeOptions::default()
    }
}

#[derive(Default)]
struct Digest(DefaultHasher);
impl Digest {
    fn bytes(&mut self, bytes: &[u8]) {
        self.0.write_u64(bytes.len() as u64);
        self.0.write(bytes);
    }
    fn number(&mut self, value: u64) {
        self.0.write_u64(value);
    }
    fn time(&mut self, value: Option<Time>) {
        self.number(u64::from(value.is_some()));
        if let Some(time) = value {
            self.0.write_i64(time.numerator());
            self.0.write_u32(time.denominator());
        }
    }
    fn frame(&mut self, frame: &VideoFrame) -> Result<usize> {
        if frame.image.pixel_format() != PixelFormat::Rgb8 {
            return Err(Error::incompatible("benchmark video must return RGB8"));
        }
        self.number(u64::from(frame.image.width()));
        self.number(u64::from(frame.image.height()));
        self.time(frame.timing.pts);
        self.time(frame.timing.source_pts);
        self.time(frame.timing.duration);
        self.time(frame.timing.native_duration);
        let bytes = frame.image.packed_bytes()?;
        self.bytes(bytes);
        Ok(bytes.len())
    }
    fn audio(&mut self, block: &AudioBlock) {
        self.time(Some(block.start()));
        self.number(u64::from(block.sample_rate()));
        self.number(block.channel_layout().channels() as u64);
        for value in block.samples() {
            self.0.write(&value.to_le_bytes());
        }
    }
    fn finish(&self) -> u64 {
        self.0.finish()
    }
}
struct Observation {
    elapsed: u128,
    first: u128,
    outputs: u64,
    decoded: u64,
    source_bytes: u64,
    output_bytes: u64,
    digest: u64,
    peak: Option<usize>,
}
struct Consumer {
    digest: Digest,
    outputs: u64,
    bytes: u64,
    first: Option<u128>,
}
impl Consumer {
    fn new() -> Self {
        Self {
            digest: Digest::default(),
            outputs: 0,
            bytes: 0,
            first: None,
        }
    }
    fn audio(&mut self, block: AudioBlock, start: Instant) {
        self.first.get_or_insert_with(|| start.elapsed().as_nanos());
        self.digest.audio(&block);
        self.outputs += 1;
        self.bytes += block.samples().len() as u64 * 4;
    }
}

fn image_workflow(path: &Path, facade: bool) -> Result<Observation> {
    let start = Instant::now();
    let (source, counter) = counted_source(path)?;
    let session = session()?;
    let options = EncodeOptions {
        metadata_policy: MetadataPolicy::AllowLoss,
        max_output_bytes: ENCODED_BYTES,
        ..EncodeOptions::new(ImageFormat::Png)
    };
    let (bytes, warnings) = if facade {
        let result = session
            .from_source(source)
            .image()?
            .apply_orientation()
            .resize_with_filter(Fit::inside(320, 320), Filter::Lanczos3)
            .allow_metadata_loss()
            .write_with_options(Sink::memory(), options)?;
        let bytes = match result.output {
            FinishedSink::Memory(bytes) => bytes,
            _ => return Err(Error::backend("bench", "expected memory PNG sink")),
        };
        (bytes, result.report.warnings)
    } else {
        let mut encoded = Vec::new();
        source
            .open_reader()?
            .take(ENCODED_BYTES as u64 + 1)
            .read_to_end(&mut encoded)
            .map_err(Error::io)?;
        let limits = native_image::DecodeLimits {
            max_encoded_bytes: ENCODED_BYTES as u64,
            max_decoded_bytes: HOST_BYTES as u64,
            ..Default::default()
        };
        let image = native_image::decode(&encoded, limits)?;
        let oriented = native_image::apply_orientation(&image)?;
        drop(image);
        let resized = native_image::resize(
            &oriented.image,
            native_image::Fit::inside(320, 320),
            native_image::Filter::Lanczos3,
        )?;
        drop(oriented.image);
        drop(encoded);
        let output = native_image::encode(&resized.image, options)?;
        drop(resized.image);
        let warnings = oriented
            .metadata_loss
            .iter()
            .chain(resized.metadata_loss.iter())
            .chain(output.metadata_loss.iter())
            .map(|loss| format!("{loss:?}"))
            .collect::<Vec<_>>();
        // Match the facade's byte sink completion as well as codec completion.
        let mut sink = Sink::memory();
        sink.write_all(&output.bytes).map_err(Error::io)?;
        let bytes = match sink.finish()? {
            FinishedSink::Memory(bytes) => bytes,
            _ => unreachable!(),
        };
        (bytes, warnings)
    };
    let first = start.elapsed().as_nanos();
    let mut digest = Digest::default();
    digest.bytes(&bytes);
    for warning in warnings {
        digest.bytes(warning.as_bytes());
    }
    let header = bytes
        .get(16..24)
        .ok_or_else(|| Error::incompatible("PNG output lacks dimensions"))?;
    let dimensions = (
        u32::from_be_bytes(header[..4].try_into().expect("four bytes")),
        u32::from_be_bytes(header[4..].try_into().expect("four bytes")),
    );
    if dimensions != (213, 320) {
        return Err(Error::incompatible(
            "oriented PNG resize did not produce213x320",
        ));
    }
    let output_bytes = bytes.len() as u64;
    drop(bytes);
    let peak = facade.then(|| session.peak_host_bytes());
    if session.live_host_bytes() != 0 {
        return Err(Error::backend(
            "bench",
            "image output retirement left a live facade allocation",
        ));
    }
    drop(session);
    Ok(Observation {
        elapsed: start.elapsed().as_nanos(),
        first,
        outputs: 1,
        decoded: 1,
        source_bytes: counter.load(Ordering::Relaxed),
        output_bytes,
        digest: digest.finish(),
        peak,
    })
}
fn audio_workflow(path: &Path, facade: bool) -> Result<Observation> {
    let start = Instant::now();
    let (source, counter) = counted_source(path)?;
    let session = session()?;
    let mut consumer = Consumer::new();
    let decoded;
    if facade {
        let mut stream = session
            .from_source(source)
            .audio()?
            .decode_options(audio_options())
            .downmix(ChannelLayout::mono())
            .resample(16000)
            .windows(Window::samples(320).hop(160))
            .stream()?;
        while let Some(block) = stream.next_block()? {
            consumer.audio(block, start);
        }
        decoded = stream.decode_report().decoded_packets;
        if !stream.decode_report().finished {
            return Err(Error::backend("bench", "facade audio did not finalize"));
        }
        consumer
            .digest
            .number(stream.discarded_window_frames() as u64);
        drop(stream);
    } else {
        let mut reader = AudioReader::open(&source, audio_options())?;
        if reader.info().layout.channels() != 1 {
            return Err(Error::incompatible("benchmark fixture is not mono"));
        }
        let mix = if reader.info().layout != ChannelLayout::mono() {
            Some(MixMatrix::mono(reader.info().layout.clone())?)
        } else {
            None
        };
        let mut resampler = Resampler::new(
            reader.info().sample_rate,
            16000,
            ChannelLayout::mono(),
            reader.info().start,
            1024,
        )?;
        let mut windows = AudioWindows::new(320, 160)?;
        while let Some(block) = reader.next_block()? {
            let block = if let Some(mix) = &mix {
                mix.process(&block)?
            } else {
                block
            };
            for converted in resampler.push(&block)? {
                for window in windows.push(&converted)? {
                    consumer.audio(window, start);
                }
            }
        }
        for converted in resampler.finish()? {
            for window in windows.push(&converted)? {
                consumer.audio(window, start);
            }
        }
        for window in windows.finish()? {
            consumer.audio(window, start);
        }
        decoded = reader.report().decoded_packets;
        if !reader.report().finished {
            return Err(Error::backend("bench", "direct audio did not finalize"));
        }
        consumer.digest.number(windows.discarded_frames() as u64);
        drop(windows);
        drop(resampler);
        drop(reader);
    }
    if consumer.outputs != 399 {
        return Err(Error::incompatible(
            "audio fixture must produce399 complete windows",
        ));
    }
    let peak = facade.then(|| session.peak_host_bytes());
    if session.live_host_bytes() != 0 {
        return Err(Error::backend("bench", "audio buffers did not retire"));
    }
    drop(session);
    Ok(Observation {
        elapsed: start.elapsed().as_nanos(),
        first: consumer.first.unwrap_or_default(),
        outputs: consumer.outputs,
        decoded,
        source_bytes: counter.load(Ordering::Relaxed),
        output_bytes: consumer.bytes,
        digest: consumer.digest.finish(),
        peak,
    })
}
fn video_workflow(path: &Path, facade: bool, slow: bool) -> Result<Observation> {
    let start = Instant::now();
    let (source, counter) = counted_source(path)?;
    let session = session()?;
    let mut first = None;
    let mut digest = Digest::default();
    let mut outputs = 0;
    let mut output_bytes = 0;
    let mut retained = VecDeque::new();
    if facade {
        let mut reader = session
            .from_source(source)
            .video()?
            .decode_options(video_options())
            .reader()?;
        loop {
            if retained.len() == 8 {
                retained.pop_front();
            }
            let Some(frame) = reader.next_frame()? else {
                break;
            };
            first.get_or_insert_with(|| start.elapsed().as_nanos());
            output_bytes += digest.frame(&frame)? as u64;
            outputs += 1;
            if slow {
                retained.push_back(frame);
                std::thread::sleep(Duration::from_millis(1));
            }
        }
        if reader.report().decoded_frames != outputs {
            return Err(Error::incompatible("sequential video accounting mismatch"));
        }
        drop(reader);
    } else {
        let mut reader = NativeVideoReader::open(&source, video_options())?;
        loop {
            if retained.len() == 8 {
                retained.pop_front();
            }
            let Some(frame) = reader.next_frame()? else {
                break;
            };
            first.get_or_insert_with(|| start.elapsed().as_nanos());
            output_bytes += digest.frame(&frame)? as u64;
            outputs += 1;
            if slow {
                retained.push_back(frame);
                std::thread::sleep(Duration::from_millis(1));
            }
        }
        drop(reader);
    }
    drop(retained);
    if outputs != 180 {
        return Err(Error::incompatible(
            "video fixture must decode180 frames including delayed EOF frames",
        ));
    }
    let peak = facade.then(|| session.peak_host_bytes());
    if session.live_host_bytes() != 0 {
        return Err(Error::backend("bench", "video surfaces did not retire"));
    }
    drop(session);
    Ok(Observation {
        elapsed: start.elapsed().as_nanos(),
        first: first.unwrap_or_default(),
        outputs,
        decoded: outputs,
        source_bytes: counter.load(Ordering::Relaxed),
        output_bytes,
        digest: digest.finish(),
        peak,
    })
}
fn sample_times(dense: bool) -> Vec<Time> {
    if dense {
        (0..60).map(|i| Time::from_millis(i * 100)).collect()
    } else {
        [250, 2750, 5250, 2750]
            .into_iter()
            .map(Time::from_millis)
            .collect()
    }
}
fn sampled_workflow(path: &Path, dense: bool) -> Result<Observation> {
    let start = Instant::now();
    let (source, counter) = counted_source(path)?;
    let session = session()?;
    let batch = session
        .from_source(source)
        .video()?
        .decode_options(video_options())
        .sample(sample_times(dense))
        .duration_inference(media::prelude::DurationInference::NextTimestamp)
        .read()?;
    let first = start.elapsed().as_nanos();
    let mut digest = Digest::default();
    let mut output_bytes = 0;
    for frame in &batch.frames {
        output_bytes += digest.frame(frame)? as u64;
    }
    let (outputs, decoded) = (batch.frames.len() as u64, batch.report.decoded_frames);
    drop(batch);
    if session.live_batches() != 0 || session.live_host_bytes() != 0 {
        return Err(Error::backend("bench", "sampled batch did not retire"));
    }
    let peak = Some(session.peak_host_bytes());
    drop(session);
    Ok(Observation {
        elapsed: start.elapsed().as_nanos(),
        first,
        outputs,
        decoded,
        source_bytes: counter.load(Ordering::Relaxed),
        output_bytes,
        digest: digest.finish(),
        peak,
    })
}
fn sampled_reference(path: &Path, dense: bool) -> Result<(u64, u64)> {
    // Correctness oracle only. It scans to EOF and is never presented as an
    // equivalent direct scheduler timing or included in measured execution.
    let source = Source::file(path)?;
    let times = sample_times(dense);
    let mut selected = vec![None; times.len()];
    let mut reader = NativeVideoReader::open(&source, video_options())?;
    let mut previous: Option<VideoFrame> = None;
    while let Some(frame) = reader.next_frame()? {
        let end = frame
            .timing
            .pts
            .ok_or_else(|| Error::unsupported("reference requires PTS"))?;
        let Some(previous) = previous.replace(frame) else {
            continue;
        };
        let pts = previous
            .timing
            .pts
            .ok_or_else(|| Error::unsupported("reference requires PTS"))?;
        if end <= pts {
            return Err(Error::incompatible("reference PTS must increase"));
        }
        // The measured request explicitly selects NextTimestamp inference.
        // Establish that same interval independently from adjacent decoded PTS,
        // never from a native packet duration or nominal frame rate. These
        // fixture queries do not require guessing the last frame's interval.
        for (index, time) in times.iter().enumerate() {
            if *time >= pts && *time < end {
                selected[index] = Some(previous.clone());
            }
        }
    }
    let mut digest = Digest::default();
    for frame in &selected {
        digest.frame(
            frame
                .as_ref()
                .ok_or_else(|| Error::incompatible("reference sample lacks coverage"))?,
        )?;
    }
    Ok((digest.finish(), selected.len() as u64))
}
fn packet_digest(path: &Path) -> Result<u64> {
    let source = Source::file(path)?;
    let mut reader = PacketReader::open(&source, ProbeOptions::default())?;
    let mut digest = Digest::default();
    for track in &reader.info().tracks {
        digest.number(u64::from(track.id.0));
        digest.bytes(track.codec.as_bytes());
        digest.bytes(&track.codec_config);
    }
    while let Some(value) = reader.next_packet_with_side_data()? {
        let packet = value.packet;
        digest.number(u64::from(packet.track.0));
        digest.time(packet.pts);
        digest.time(packet.dts);
        digest.time(packet.duration);
        digest.number(u64::from(packet.keyframe));
        digest.bytes(&packet.data);
        for side in value.side_data {
            digest.bytes(side.kind.as_bytes());
            digest.bytes(&side.data);
        }
    }
    Ok(digest.finish())
}
fn container_workflow(
    input: &Path,
    output: &Path,
    facade: bool,
    transcode: bool,
) -> Result<Observation> {
    let start = Instant::now();
    let (source, counter) = counted_source(input)?;
    let session = session()?;
    let (outputs, decoded) = if transcode {
        let report = if facade {
            session
                .from_source(source)
                .convert()
                .options(transcode_options())
                .write(output)?
        } else {
            encode::transcode(&source, output, transcode_options())?
        };
        if report.frames_encoded != 180
            || report.audio_packets_written == 0
            || report.output_width != 160
            || report.output_height != 90
        {
            return Err(Error::incompatible(
                "transcode output does not match workflow contract",
            ));
        }
        (report.frames_encoded, report.frames_decoded)
    } else {
        let tracks = [TrackId(0), TrackId(1)];
        let report = if facade {
            session
                .from_source(source)
                .remux()
                .tracks(tracks)
                .write(output)?
        } else {
            format::remux(&source, output, &tracks)?
        };
        (report.packets_written, 0)
    };
    // The backend returns only after codec drain/trailer/flush/file close.
    // This CPU encode/remux facade currently does not reserve native codec/mux
    // buffers in Session, so its peak is unmeasured rather than falsely zero.
    drop(session);
    let elapsed = start.elapsed().as_nanos();
    let source_bytes = counter.load(Ordering::Relaxed);
    let digest = packet_digest(output)?;
    let output_bytes = fs::metadata(output).map_err(Error::io)?.len();
    Ok(Observation {
        elapsed,
        first: elapsed,
        outputs,
        decoded,
        source_bytes,
        output_bytes,
        digest,
        peak: None,
    })
}

fn record(
    file: &mut File,
    mode: &str,
    variant: &str,
    iteration: usize,
    position: usize,
    observation: &Observation,
) -> Result<()> {
    let phase = if iteration == 0 {
        "initial_use"
    } else {
        "warm"
    };
    let peak = observation
        .peak
        .map(|v| v.to_string())
        .unwrap_or_else(|| "unmeasured".into());
    writeln!(file,"{mode},{variant},{iteration},{phase},{position},{},{},{},{},{},{},{:016x},{peak},unmeasured,unmeasured,true,passed",observation.elapsed,observation.first,observation.outputs,observation.decoded,observation.source_bytes,observation.output_bytes,observation.digest).map_err(Error::io)
}
fn environment(directory: &Path, fixtures: &Path, iterations: usize) -> Result<()> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(directory.join("environment.txt"))
        .map_err(Error::io)?;
    writeln!(file,"unix_seconds={}\nfixture_directory={}\niterations={}\narch={}\nos={}\nffmpeg_linked={}\nhost_budget={}\nencoded_limit={}\nslow_consumer_delay_ms=1\nslow_retained_frames=8\nresampler_chunk_hint=1024\nvideo_threads=1\nbuild_profile=release",SystemTime::now().duration_since(UNIX_EPOCH).map_err(|e|Error::backend("clock",e.to_string()))?.as_secs(),fixtures.display(),iterations,std::env::consts::ARCH,std::env::consts::OS,media::backend::ffmpeg::hardware::version()?,HOST_BYTES,ENCODED_BYTES).map_err(Error::io)?;
    for (program, args) in [
        ("rustc", vec!["-Vv"]),
        ("cargo", vec!["-V"]),
        ("uname", vec!["-a"]),
        ("sysctl", vec!["-n", "machdep.cpu.brand_string"]),
    ] {
        if let Ok(output) = Command::new(program).args(args).output() {
            writeln!(
                file,
                "\n{program}: exit={:?}\n{}{}",
                output.status.code(),
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            )
            .map_err(Error::io)?;
        }
    }
    fs::copy(
        fixtures.join("manifest.json"),
        directory.join("fixture-manifest.json"),
    )
    .map_err(Error::io)?;
    fs::write(
        directory.join("runner.rs"),
        include_bytes!("workflow_bench.rs"),
    )
    .map_err(Error::io)?;
    fs::write(
        directory.join("Cargo.lock"),
        include_bytes!("../../../Cargo.lock"),
    )
    .map_err(Error::io)?;
    Ok(())
}
fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let fixtures = PathBuf::from(args.next().ok_or_else(|| {
        Error::invalid("usage: workflow_bench FIXTURES OUTPUT_DIRECTORY [ITERATIONS=7] [MODE=all]")
    })?);
    let directory = PathBuf::from(
        args.next()
            .ok_or_else(|| Error::invalid("benchmark needs an output directory"))?,
    );
    let iterations = args
        .next()
        .map_or(Ok(7), |v| v.parse::<usize>())
        .map_err(|_| Error::invalid("invalid iteration count"))?;
    if !(5..=100).contains(&iterations) {
        return Err(Error::invalid(
            "use5..=100 iterations, with initial use separated from warm runs",
        ));
    }
    let chosen = args.next().unwrap_or_else(|| "all".into());
    if chosen != "all" && !MODES.contains(&chosen.as_str()) {
        return Err(Error::invalid("unknown benchmark mode"));
    }
    fs::create_dir_all(&directory).map_err(Error::io)?;
    let mut csv = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(directory.join("raw.csv"))
        .map_err(Error::io)?;
    environment(&directory, &fixtures, iterations)?;
    writeln!(csv,"mode,variant,iteration,phase,within_pair_order,end_to_end_ns,first_result_ns,outputs,decoded_units,source_payload_bytes_read,output_bytes,digest64_noncrypto,peak_host_accounted_bytes,opaque_host_peak_bytes,driver_peak_bytes,retirement_included,validation").map_err(Error::io)?;
    for mode in MODES {
        if chosen != "all" && chosen != mode {
            continue;
        }
        let mut stable_digest = None;
        for iteration in 0..iterations {
            if matches!(mode, "sparse" | "dense") {
                let observation = sampled_workflow(&fixtures.join("av6s.mp4"), mode == "dense")?;
                if iteration == 0 {
                    let expected = sampled_reference(&fixtures.join("av6s.mp4"), mode == "dense")?;
                    if (observation.digest, observation.outputs) != expected {
                        return Err(Error::incompatible(
                            "facade sampling differs from independently decoded display intervals",
                        ));
                    }
                }
                if stable_digest.is_some_and(|value| value != observation.digest) {
                    return Err(Error::incompatible(
                        "sample outputs changed between iterations",
                    ));
                }
                stable_digest = Some(observation.digest);
                record(&mut csv, mode, "facade_only", iteration, 0, &observation)?;
                continue;
            }
            let order = if iteration % 2 == 0 {
                [true, false]
            } else {
                [false, true]
            };
            let mut pair = Vec::new();
            for (position, facade) in order.into_iter().enumerate() {
                let variant = if facade {
                    "facade"
                } else {
                    "direct_same_backend"
                };
                let observation = match mode {
                    "image" => image_workflow(&fixtures.join("oriented.png"), facade)?,
                    "audio" => audio_workflow(&fixtures.join("mono48k.wav"), facade)?,
                    "video" | "slowconsumer" => {
                        video_workflow(&fixtures.join("av6s.mp4"), facade, mode == "slowconsumer")?
                    }
                    "remux" | "transcode" => container_workflow(
                        &fixtures.join("av6s.mp4"),
                        &directory.join(format!("{mode}-{variant}-{iteration}.mp4")),
                        facade,
                        mode == "transcode",
                    )?,
                    _ => unreachable!(),
                };
                pair.push((variant, position, observation));
            }
            let a = &pair[0].2;
            let b = &pair[1].2;
            if (a.digest, a.outputs, a.decoded) != (b.digest, b.outputs, b.decoded) {
                return Err(Error::incompatible(format!(
                    "{mode}: facade/direct outputs or decoded counts differ"
                )));
            }
            if stable_digest.is_some_and(|value| value != a.digest) {
                return Err(Error::incompatible(format!(
                    "{mode}: output digest changed between iterations"
                )));
            }
            stable_digest = Some(a.digest);
            for (variant, position, observation) in &pair {
                record(&mut csv, mode, variant, iteration, *position, observation)?;
            }
        }
        csv.flush().map_err(Error::io)?;
        eprintln!("validated {mode}: {iterations} iterations");
    }
    println!("{}", directory.join("raw.csv").display());
    Ok(())
}
