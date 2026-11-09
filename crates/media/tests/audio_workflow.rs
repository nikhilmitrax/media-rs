//! Continuous facade audio semantics against real WAV and resampling backends.
#![cfg(feature = "native")]
use media::{
    backend::native::{audio::WavWriter, resample::slice_block},
    io::{sink::Sink, source::Source},
    prelude::*,
};
fn wav(frames: usize) -> Vec<u8> {
    let samples: Vec<_> = (0..frames)
        .map(|i| ((i as f32 * 0.07).sin()) * 0.5)
        .collect();
    let input = AudioBlock::new(samples, 48000, ChannelLayout::mono(), Time::ZERO).unwrap();
    let mut writer = WavWriter::new(Sink::memory(), 48000, ChannelLayout::mono()).unwrap();
    writer.write(&input).unwrap();
    writer.finish().unwrap().into_bytes().unwrap()
}
#[test]
fn fluent_continuous_resample_and_fractional_slice_match_shared_grid() {
    let session = Session::default();
    let asset = session.from_source(Source::memory(wav(4800)));
    let request = asset
        .audio()
        .unwrap()
        .downmix(ChannelLayout::mono())
        .resample(16000);
    let full = request.read().unwrap();
    assert_eq!(full.frames(), 1600);
    let range = TimeRange::new(
        Time::new(1001, 48000).unwrap(),
        Time::new(3501, 48000).unwrap(),
    )
    .unwrap();
    let expected = slice_block(&full, range).unwrap();
    let sliced = request.slice(range).read().unwrap();
    assert_eq!(sliced.samples(), expected.samples());
    assert_eq!(sliced.start(), expected.start());
    drop(full);
    drop(sliced);
    assert_eq!(session.live_host_bytes(), 0);
}
#[test]
fn overlapping_windows_are_bounded_continuous_and_timed() {
    let asset = Session::default().from_source(Source::memory(wav(4800)));
    let request = asset
        .audio()
        .unwrap()
        .resample(16000)
        .windows(Window::samples(320).hop(160));
    assert!(request.read().is_err());
    let mut stream = request.stream().unwrap();
    let mut count = 0;
    let mut previous: Option<AudioBlock> = None;
    while let Some(block) = stream.next_block().unwrap() {
        assert_eq!(block.frames(), 320);
        assert_eq!(block.start(), Time::from_millis(count * 10));
        if let Some(last) = previous {
            assert_eq!(&last.samples()[160..], &block.samples()[..160]);
        }
        previous = Some(block);
        count += 1;
    }
    assert_eq!(count, 9);
    assert_eq!(stream.discarded_window_frames(), 160);
}
#[test]
fn cancellation_and_out_of_coverage_are_explicit_errors() {
    let asset = Session::default().from_source(Source::memory(wav(4800)));
    let request = asset.audio().unwrap();
    let mut stream = request.stream().unwrap();
    stream.cancellation().cancel();
    assert_eq!(
        stream.next_block().unwrap_err().kind(),
        ErrorKind::Cancelled
    );
    assert!(
        request
            .clone()
            .slice(TimeRange::new(Time::from_millis(-1), Time::from_millis(10)).unwrap())
            .read()
            .is_err()
    );
    assert_eq!(
        request
            .slice(TimeRange::new(Time::ZERO, Time::from_millis(101)).unwrap())
            .read()
            .unwrap_err()
            .kind(),
        ErrorKind::MissingData
    );
}
#[test]
fn materialized_audio_and_clones_keep_the_output_budget() {
    let session = Session::default();
    let audio = session
        .from_source(Source::memory(wav(20)))
        .audio()
        .unwrap()
        .read()
        .unwrap();
    assert_eq!(session.live_host_bytes(), 80);
    let clone = audio.clone();
    drop(audio);
    assert_eq!(session.live_host_bytes(), 80);
    drop(clone);
    assert_eq!(session.live_host_bytes(), 0);
}

use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};
static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);
struct TestDirectory(PathBuf);
impl TestDirectory {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "media-audio-facade-{}-{}",
            std::process::id(),
            NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
    fn names(&self) -> Vec<String> {
        let mut names: Vec<_> = fs::read_dir(&self.0)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    }
}
impl Drop for TestDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn densely_overlapping_large_windows_do_not_drop_results_or_hold_unbounded_outputs() {
    let session = Session::builder()
        .limits(Limits::new().host_memory(2 * 1024 * 1024))
        .build()
        .unwrap();
    let request = session
        .from_source(Source::memory(wav(65_537)))
        .audio()
        .unwrap()
        .windows(Window::samples(65_536).hop(1));
    let mut stream = request.stream().unwrap();
    let first = stream.next_block().unwrap().unwrap();
    assert_eq!(first.frames(), 65_536);
    assert_eq!(first.start(), Time::ZERO);
    let second = stream.next_block().unwrap().unwrap();
    assert_eq!(second.start(), Time::from_samples(1, 48_000).unwrap());
    assert_eq!(&first.samples()[1..], &second.samples()[..65_535]);
    assert!(stream.next_block().unwrap().is_none());
    assert_eq!(session.live_host_bytes(), 2 * 65_536 * 4);
    drop(stream);
    drop(first);
    drop(second);
    assert_eq!(session.live_host_bytes(), 0);
}

#[test]
fn cancellation_retires_working_state_but_preserves_retained_output_guards() {
    let session = Session::default();
    let request = session
        .from_source(Source::memory(wav(5000)))
        .audio()
        .unwrap()
        .resample(16000)
        .windows(Window::samples(320).hop(1));
    let mut stream = request.stream().unwrap();
    let output = stream.next_block().unwrap().unwrap();
    let clone = output.clone();
    assert!(session.live_host_bytes() > output.samples().len() * 4);
    stream.cancellation().cancel();
    assert_eq!(
        stream.next_block().unwrap_err().kind(),
        ErrorKind::Cancelled
    );
    assert!(stream.next().is_none());
    assert_eq!(session.live_host_bytes(), output.samples().len() * 4);
    drop(stream);
    drop(output);
    assert_eq!(session.live_host_bytes(), clone.samples().len() * 4);
    drop(clone);
    assert_eq!(session.live_host_bytes(), 0);
}

#[test]
fn retained_window_budget_exhaustion_fails_without_deadlock_or_leaked_work() {
    let session = Session::builder()
        .limits(Limits::new().host_memory(128 * 1024))
        .build()
        .unwrap();
    let request = session
        .from_source(Source::memory(wav(5000)))
        .audio()
        .unwrap()
        .windows(Window::samples(512).hop(1));
    let mut stream = request.stream().unwrap();
    let mut retained = Vec::new();
    loop {
        match stream.next_block() {
            Ok(Some(block)) => retained.push(block),
            Ok(None) => panic!("retained overlapping windows should exhaust this budget"),
            Err(error) => {
                assert_eq!(error.kind(), ErrorKind::ResourceExhausted);
                break;
            }
        }
        assert!(retained.len() < 100, "budget must bound retained windows");
    }
    assert_eq!(
        session.live_host_bytes(),
        retained.iter().map(|block| block.samples().len() * 4).sum()
    );
    assert!(stream.next().is_none());
    drop(stream);
    drop(retained);
    assert_eq!(session.live_host_bytes(), 0);
}

#[test]
fn empty_ranges_and_fractional_source_ends_do_not_invent_coverage() {
    let session = Session::default();
    let request = session
        .from_source(Source::memory(wav(4801)))
        .audio()
        .unwrap()
        .resample(16000);
    let mut empty = request
        .clone()
        .slice(TimeRange::new(Time::ZERO, Time::ZERO).unwrap())
        .stream()
        .unwrap();
    assert!(empty.next_block().unwrap().is_none());
    assert_eq!(empty.decode_report().decoded_packets, 0);
    assert_eq!(session.live_host_bytes(), 0);
    let point = Time::new(1, 48_000).unwrap();
    let output = request
        .clone()
        .slice(TimeRange::new(point, point).unwrap())
        .read()
        .unwrap();
    assert_eq!(output.frames(), 0);
    assert_eq!(output.start(), Time::from_samples(1, 16_000).unwrap());
    let source_end = Time::from_samples(4801, 48_000).unwrap();
    let valid = request
        .clone()
        .slice(TimeRange::new(Time::ZERO, source_end).unwrap())
        .read()
        .unwrap();
    assert_eq!(valid.frames(), 1601);
    assert!(valid.end().unwrap() > source_end);
    let beyond = Time::new(2001, 20_000).unwrap(); // 100.05ms, before the last sample interval ends.
    assert_eq!(
        request
            .slice(TimeRange::new(Time::ZERO, beyond).unwrap())
            .read()
            .unwrap_err()
            .kind(),
        ErrorKind::MissingData
    );
}

#[test]
fn probe_does_not_allocate_window_or_resampler_execution_state() {
    let session = Session::builder()
        .limits(Limits::new().host_memory(1024))
        .build()
        .unwrap();
    let request = session
        .from_source(Source::memory(wav(20)))
        .audio()
        .unwrap()
        .resample(16_000)
        .windows(Window::samples(65_536).hop(1));
    assert_eq!(request.probe().unwrap().sample_rate, 48_000);
    assert_eq!(session.live_host_bytes(), 0);
    assert!(request.stream().is_err());
    assert_eq!(session.live_host_bytes(), 0);
}

#[test]
fn failed_writes_preserve_existing_output_and_remove_staging_files() {
    use media::backend::native::audio::MixMatrix;
    use media::core::media::ChannelPosition;
    let directory = TestDirectory::new();
    let output = directory.path("existing.wav");
    fs::write(&output, b"keep existing output").unwrap();
    let request = Session::default()
        .from_source(Source::memory(wav(4800)))
        .audio()
        .unwrap();
    let late_error = request
        .clone()
        .slice(TimeRange::new(Time::ZERO, Time::from_millis(101)).unwrap());
    assert!(late_error.write(&output).is_err());
    assert_eq!(fs::read(&output).unwrap(), b"keep existing output");
    assert_eq!(directory.names(), vec!["existing.wav"]);
    let invalid_layout = ChannelLayout::new(vec![
        ChannelPosition::FrontLeft,
        ChannelPosition::FrontRight,
        ChannelPosition::FrontCenter,
    ])
    .unwrap();
    let matrix = MixMatrix::new(ChannelLayout::mono(), invalid_layout, vec![1., 1., 1.]).unwrap();
    assert!(request.clone().mix(matrix).write(&output).is_err());
    assert_eq!(fs::read(&output).unwrap(), b"keep existing output");
    let unsupported = directory.path("existing.mp3");
    fs::write(&unsupported, b"keep non-WAV").unwrap();
    assert!(request.write(&unsupported).is_err());
    assert_eq!(fs::read(&unsupported).unwrap(), b"keep non-WAV");
}

#[test]
fn successful_writes_replace_only_after_finalization_and_reject_input_aliases() {
    let directory = TestDirectory::new();
    let input = directory.path("input.wav");
    let original = wav(31);
    fs::write(&input, &original).unwrap();
    let output = directory.path("output.wav");
    fs::write(&output, b"old output").unwrap();
    let request = Session::default().open(&input).unwrap().audio().unwrap();
    assert!(request.write(&input).is_err());
    assert_eq!(fs::read(&input).unwrap(), original);
    let finished = request.write(&output).unwrap();
    assert!(matches!(
        finished,
        media::io::sink::FinishedSink::File { .. }
    ));
    let audio = media::open(&output)
        .unwrap()
        .audio()
        .unwrap()
        .read()
        .unwrap();
    assert_eq!(audio.frames(), 31);
    assert_eq!(directory.names(), vec!["input.wav", "output.wav"]);
    #[cfg(unix)]
    {
        let alias = directory.path("hardlink.wav");
        fs::hard_link(&input, &alias).unwrap();
        assert!(request.write(&alias).is_err());
        assert_eq!(fs::read(&input).unwrap(), original);
        assert_eq!(fs::read(&alias).unwrap(), original);
    }
}

#[test]
fn quantized_container_timestamps_keep_a_continuous_grid_and_raw_source_pts() {
    use media::backend::native::audio::{AudioDecodeOptions, AudioReader, AudioTimestampPolicy};
    let source = Source::memory(include_bytes!("fixtures/audio_quantized.mka").as_slice());
    let mut reader = AudioReader::open(&source, Default::default()).unwrap();
    assert_eq!(reader.info().timestamp_quantum, Time::from_millis(1));
    let first = reader.next_block().unwrap().unwrap();
    let second = reader.next_block().unwrap().unwrap();
    assert_eq!(first.frames(), 1024);
    assert_eq!(second.start(), first.end().unwrap());
    assert_eq!(second.source_start(), Some(Time::from_millis(21)));
    assert_ne!(second.start(), second.source_start().unwrap());
    assert!(reader.next_block().unwrap().is_none());
    assert_eq!(reader.report().timestamp_adjustments, 1);
    assert!(reader.report().max_timestamp_adjustment.unwrap() <= reader.info().timestamp_quantum);
    let mut exact = AudioReader::open(
        &source,
        AudioDecodeOptions {
            timestamps: AudioTimestampPolicy::Exact,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(exact.next_block().unwrap().is_some());
    assert_eq!(
        exact.next_block().unwrap_err().kind(),
        ErrorKind::MissingData
    );
    let request = Session::default().from_source(source).audio().unwrap();
    let piece = request
        .clone()
        .slice(TimeRange::new(Time::from_millis(22), Time::from_millis(25)).unwrap())
        .read()
        .unwrap();
    assert_eq!(piece.start(), Time::from_millis(22));
    assert_eq!(
        piece.source_start(),
        Some(
            Time::from_millis(21)
                .checked_add(Time::from_samples(32, 48000).unwrap())
                .unwrap()
        )
    );
    let audio = request.resample(16000).read().unwrap();
    assert_eq!(audio.frames(), 480);
    assert_eq!(audio.source_start(), None);
}

#[test]
fn real_timestamp_gaps_are_errors_and_failed_export_does_not_replace_output() {
    let source = Source::memory(include_bytes!("fixtures/audio_gap.mka").as_slice());
    let request = Session::default().from_source(source).audio().unwrap();
    let mut stream = request.stream().unwrap();
    assert!(stream.next_block().unwrap().is_some());
    assert_eq!(
        stream.next_block().unwrap_err().kind(),
        ErrorKind::MissingData
    );
    assert!(stream.next().is_none());
    let directory = TestDirectory::new();
    let output = directory.path("existing.wav");
    fs::write(&output, b"old").unwrap();
    assert!(request.write(&output).is_err());
    assert_eq!(fs::read(output).unwrap(), b"old");
    assert_eq!(directory.names(), vec!["existing.wav"]);
}

#[test]
fn delayed_audio_origin_is_discovered_before_constructing_a_resampler() {
    let source = Source::memory(include_bytes!("fixtures/audio_offset.mka").as_slice());
    let request = Session::default()
        .from_source(source)
        .audio()
        .unwrap()
        .resample(16000);
    assert_eq!(request.probe().unwrap().start, Time::from_millis(200));
    let audio = request.read().unwrap();
    assert_eq!(audio.start(), Time::from_millis(200));
    assert_eq!(audio.frames(), 480);
    assert_eq!(audio.end().unwrap(), Time::from_millis(230));
    let piece = request
        .slice(TimeRange::new(Time::from_millis(210), Time::from_millis(220)).unwrap())
        .read()
        .unwrap();
    assert_eq!(piece.start(), Time::from_millis(210));
    assert_eq!(piece.frames(), 160);
}

#[test]
fn compiled_audio_plans_describe_exact_grids_and_execute_independently_across_threads() {
    use media::audio_request::AudioSampleFormat;
    let session = Session::default();
    let range = TimeRange::new(
        Time::new(1001, 48000).unwrap(),
        Time::new(3501, 48000).unwrap(),
    )
    .unwrap();
    let plan = session
        .from_source(Source::memory(wav(4800)))
        .audio()
        .unwrap()
        .resample(16000)
        .slice(range)
        .compile()
        .unwrap();
    assert_eq!(session.live_host_bytes(), 0);
    assert_eq!(plan.output().sample_rate, 16000);
    assert_eq!(plan.output().format, AudioSampleFormat::F32);
    assert!(plan.output().interleaved);
    assert_eq!(plan.output().channel_layout, ChannelLayout::mono());
    assert_eq!(plan.sample_grid_origin(), Time::ZERO);
    assert_eq!(plan.sample_bounds(), Some(334..1167));
    assert_eq!(plan.working_limits().resample_chunk_hint_frames, Some(1024));
    assert!(
        plan.description()
            .steps
            .iter()
            .any(|step| step.backend == "symphonia")
    );
    assert!(
        plan.description()
            .steps
            .iter()
            .any(|step| step.backend == "rubato FFT")
    );
    assert!(plan.explain().contains("ceil sample bounds 334..1167"));
    assert!(plan.explain().contains("private"));
    let first = plan.clone();
    let second = plan.clone();
    let first = std::thread::spawn(move || first.execute().unwrap());
    let second = std::thread::spawn(move || second.execute().unwrap());
    let first = first.join().unwrap();
    let second = second.join().unwrap();
    assert_eq!(first.frames(), 833);
    assert_eq!(first.start(), Time::from_samples(334, 16000).unwrap());
    assert_eq!(
        first.end().unwrap(),
        Time::from_samples(1167, 16000).unwrap()
    );
    assert_eq!(first.samples(), second.samples());
    assert_ne!(first.samples().as_ptr(), second.samples().as_ptr());
    assert_eq!(
        session.live_host_bytes(),
        (first.samples().len() + second.samples().len()) * 4
    );
    drop(first);
    drop(second);
    assert_eq!(session.live_host_bytes(), 0);
}

#[test]
fn compiled_readers_have_independent_filter_cursors_and_cancellation() {
    let session = Session::default();
    let plan = session
        .from_source(Source::memory(wav(4800)))
        .audio()
        .unwrap()
        .resample(16000)
        .compile()
        .unwrap();
    let mut first = plan.reader().unwrap();
    let mut second = plan.stream().unwrap();
    let first_block = first.next_block().unwrap().unwrap();
    let next = first.next_block().unwrap().unwrap();
    let second_block = second.next_block().unwrap().unwrap();
    assert_eq!(first_block.start(), Time::ZERO);
    assert_eq!(second_block.start(), Time::ZERO);
    assert_eq!(first_block.samples(), second_block.samples());
    assert_eq!(first_block.end().unwrap(), next.start());
    first.cancellation().cancel();
    assert_eq!(first.next_block().unwrap_err().kind(), ErrorKind::Cancelled);
    assert!(second.next_block().unwrap().is_some());
    drop(first);
    drop(second);
    drop(first_block);
    drop(second_block);
    drop(next);
    assert_eq!(session.live_host_bytes(), 0);
}

#[test]
fn compile_retains_no_execution_reservation_and_windows_keep_materialization_rejection() {
    let session = Session::builder()
        .limits(Limits::new().host_memory(1024))
        .build()
        .unwrap();
    let plan = session
        .from_source(Source::memory(wav(4800)))
        .audio()
        .unwrap()
        .resample(16000)
        .windows(Window::samples(65536).hop(1))
        .compile()
        .unwrap();
    assert_eq!(plan.working_limits().host_memory, 1024);
    assert_eq!(plan.working_limits().max_packet_bytes, 256);
    assert_eq!(plan.working_limits().max_decoded_packet_samples, 64);
    assert_eq!(plan.working_limits().window_frames, Some(65536));
    assert_eq!(session.live_host_bytes(), 0);
    assert_eq!(plan.execute().unwrap_err().kind(), ErrorKind::InvalidInput);
    assert!(plan.reader().is_err());
    assert_eq!(session.live_host_bytes(), 0);
    let windowed = Session::default()
        .from_source(Source::memory(wav(4800)))
        .audio()
        .unwrap()
        .resample(16000)
        .windows(Window::samples(320).hop(160))
        .compile()
        .unwrap();
    assert!(windowed.execute().is_err());
    let mut reader = windowed.stream().unwrap();
    assert_eq!(reader.next_block().unwrap().unwrap().frames(), 320);
}

#[test]
fn stale_versioned_sources_are_rejected_before_compiled_audio_execution() {
    let directory = TestDirectory::new();
    let path = directory.path("source.wav");
    fs::write(&path, wav(4800)).unwrap();
    let session = Session::default();
    let plan = session
        .open(&path)
        .unwrap()
        .audio()
        .unwrap()
        .compile()
        .unwrap();
    fs::write(&path, wav(4801)).unwrap();
    assert_eq!(plan.execute().unwrap_err().kind(), ErrorKind::Incompatible);
    assert!(matches!(plan.reader(),Err(error) if error.kind()==ErrorKind::Incompatible));
    assert_eq!(session.live_host_bytes(), 0);
}

#[test]
fn unversioned_plan_executions_revalidate_probed_audio_interpretation() {
    use std::sync::{Arc, atomic::AtomicUsize};
    let first: Arc<[u8]> = wav(4800).into();
    let input = AudioBlock::new(vec![0.0; 3200], 32000, ChannelLayout::mono(), Time::ZERO).unwrap();
    let mut writer = WavWriter::new(Sink::memory(), 32000, ChannelLayout::mono()).unwrap();
    writer.write(&input).unwrap();
    let second: Arc<[u8]> = writer.finish().unwrap().into_bytes().unwrap().into();
    let opens = Arc::new(AtomicUsize::new(0));
    let counter = opens.clone();
    let source = Source::from_seekable_factory("changing source", None, None, move || {
        Ok(std::io::Cursor::new(
            if counter.fetch_add(1, Ordering::AcqRel) == 0 {
                first.clone()
            } else {
                second.clone()
            },
        ))
    })
    .unwrap();
    let session = Session::default();
    let plan = session
        .from_source(source)
        .audio()
        .unwrap()
        .compile()
        .unwrap();
    assert!(plan.explain().contains("unversioned"));
    assert_eq!(opens.load(Ordering::Acquire), 1);
    assert_eq!(plan.source_info().sample_rate, 48000);
    assert_eq!(plan.execute().unwrap_err().kind(), ErrorKind::Incompatible);
    assert_eq!(session.live_host_bytes(), 0);
}

#[test]
fn finite_plan_execution_checks_source_version_again_after_decoding_has_started() {
    use std::{
        io::Write,
        sync::{Arc, Barrier, atomic::AtomicBool},
        time::{Duration, Instant},
    };
    let directory = TestDirectory::new();
    let path = directory.path("changing.wav");
    fs::write(&path, wav(2_000_000)).unwrap();
    let session = Session::default();
    let plan = session
        .open(&path)
        .unwrap()
        .audio()
        .unwrap()
        .compile()
        .unwrap();
    let limits = plan.working_limits();
    let threshold =
        limits.max_decoded_packet_samples * 4 + limits.max_decoder_block_frames * 8 + 1024 * 1024;
    let observer_session = session.clone();
    let observer_path = path.clone();
    let barrier = Arc::new(Barrier::new(2));
    let observer_barrier = barrier.clone();
    let done = Arc::new(AtomicBool::new(false));
    let observer_done = done.clone();
    let observer = std::thread::spawn(move || {
        observer_barrier.wait();
        let deadline = Instant::now() + Duration::from_secs(5);
        while observer_session.live_host_bytes() <= threshold {
            if observer_done.load(Ordering::Acquire) || Instant::now() >= deadline {
                return false;
            }
            std::thread::yield_now();
        }
        // Append a harmless chunk rather than corrupting already-decoded samples.
        // The RIFF data interval is unchanged, but the source version is now stale.
        fs::OpenOptions::new()
            .append(true)
            .open(observer_path)
            .unwrap()
            .write_all(b"JUNK\0\0\0\0")
            .unwrap();
        true
    });
    barrier.wait();
    let result = plan.execute();
    done.store(true, Ordering::Release);
    assert!(
        observer.join().unwrap(),
        "mutation must happen after waveform materialization starts"
    );
    assert_eq!(result.unwrap_err().kind(), ErrorKind::Incompatible);
    assert_eq!(session.live_host_bytes(), 0);
}
