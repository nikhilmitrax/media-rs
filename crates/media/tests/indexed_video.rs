//! Real VFR/B-frame indexed seeking must reduce decoded work and retain provenance.
#![cfg(feature = "ffmpeg")]
use media::{
    Session,
    backend::ffmpeg::codec::VideoDecodeOptions,
    core::{error::ErrorKind, policy::FrameDurationPolicy, time::Time},
    io::source::Source,
    pipeline::{
        execution::CancellationToken,
        sampling::{DurationInference, TemporalSelection},
    },
    video_index::{IndexBuildLimits, IndexSeekLimits},
};
use std::{
    fs,
    path::PathBuf,
    process::Command,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};
#[derive(Clone, Default)]
struct ReadCounts {
    bytes: Arc<AtomicU64>,
    calls: Arc<AtomicU64>,
    opens: Arc<AtomicU64>,
    cancel_after: Arc<AtomicU64>,
    cancel: CancellationToken,
}
impl ReadCounts {
    fn reset(&self) {
        self.bytes.store(0, Ordering::SeqCst);
        self.calls.store(0, Ordering::SeqCst);
        self.opens.store(0, Ordering::SeqCst);
    }
}
struct CountedReader {
    data: std::io::Cursor<Arc<[u8]>>,
    counts: ReadCounts,
}
impl std::io::Read for CountedReader {
    fn read(&mut self, bytes: &mut [u8]) -> std::io::Result<usize> {
        let read = std::io::Read::read(&mut self.data, bytes)?;
        self.counts.bytes.fetch_add(read as u64, Ordering::SeqCst);
        let call = self.counts.calls.fetch_add(1, Ordering::SeqCst) + 1;
        let cancel = self.counts.cancel_after.load(Ordering::SeqCst);
        if cancel > 0 && call >= cancel {
            self.counts.cancel.cancel();
        }
        Ok(read)
    }
}
impl std::io::Seek for CountedReader {
    fn seek(&mut self, position: std::io::SeekFrom) -> std::io::Result<u64> {
        std::io::Seek::seek(&mut self.data, position)
    }
}
fn counted_source(fixture: &Fixture, version: &str, counts: &ReadCounts) -> Source {
    let bytes: Arc<[u8]> = fs::read(&fixture.path).unwrap().into();
    let length = bytes.len() as u64;
    let counts = counts.clone();
    Source::from_seekable_factory(
        format!("index-counted-{}", fixture.path.display()),
        Some(version.into()),
        Some(length),
        move || {
            counts.opens.fetch_add(1, Ordering::SeqCst);
            Ok(CountedReader {
                data: std::io::Cursor::new(bytes.clone()),
                counts: counts.clone(),
            })
        },
    )
    .unwrap()
}
struct Fixture {
    directory: PathBuf,
    path: PathBuf,
}
impl Fixture {
    fn new(gop: usize) -> Option<Self> {
        Self::with_rate(gop, "30")
    }
    fn with_rate(gop: usize, rate: &str) -> Option<Self> {
        let binary = std::env::var_os("MEDIA_FFMPEG_BIN").unwrap_or_else(|| "ffmpeg".into());
        if Command::new(&binary).arg("-version").output().is_err() {
            assert!(
                std::env::var_os("MEDIA_REQUIRE_FFMPEG_TESTS").is_none(),
                "FFmpeg fixture generator required"
            );
            eprintln!("UNEXERCISED: indexed VFR fixtures require ffmpeg");
            return None;
        }
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let directory = std::env::temp_dir().join(format!(
            "media-index-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::SeqCst)
        ));
        fs::create_dir(&directory).unwrap();
        let path = directory.join("vfr.mp4");
        let params = format!("keyint={gop}:min-keyint={gop}:scenecut=0:open-gop=0");
        let input = format!("testsrc2=size=192x192:rate={rate}:duration=20");
        let output=Command::new(binary).args(["-v","error","-nostdin","-f","lavfi","-i",&input,"-vf","select=not(eq(mod(n\\,7)\\,3)+eq(mod(n\\,7)\\,4)),scale=in_color_matrix=bt601:out_color_matrix=bt709:out_range=tv","-fps_mode","vfr","-c:v","libx264","-threads:v","1","-bf","3","-x264-params",&params,"-pix_fmt","yuv420p","-colorspace","bt709","-color_primaries","bt709","-color_trc","bt709","-color_range","tv","-chroma_sample_location","left","-y"]).arg(&path).output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        Some(Self { directory, path })
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.directory);
    }
}
fn times() -> [Time; 4] {
    [
        Time::from_millis(18050),
        Time::from_millis(1250),
        Time::from_millis(18050),
        Time::from_millis(9750),
    ]
}

#[test]
fn vfr_b_frames_use_gop_local_decode_with_exact_selection_and_real_global_ordinals() {
    let Some(fixture) = Fixture::new(60) else {
        return;
    };
    let session = Session::default();
    let request = session.open(&fixture.path).unwrap().video().unwrap();
    // Missing color interpretation cannot obstruct a metadata-only index build.
    let index = request
        .clone()
        .decode_options(VideoDecodeOptions {
            color_assumption: None,
            ..Default::default()
        })
        .build_index(IndexBuildLimits::default())
        .unwrap();
    assert!(index.len() > 400);
    assert_eq!(index.report().decoded_frames, index.len() as u64);
    assert!(!index.report().hardware_decode);
    assert!(index.report().bytes_read > 0);
    for policy in [
        TemporalSelection::DisplayedAt,
        TemporalSelection::Nearest,
        TemporalSelection::AtOrBefore,
        TemporalSelection::AtOrAfter,
    ] {
        let sampled = request
            .clone()
            .sample(times())
            .selection(policy)
            .duration_inference(DurationInference::NextTimestamp);
        let reference = sampled.read().unwrap();
        let plan = sampled.with_index(index.clone()).compile().unwrap();
        assert!(plan.explain().contains("GOP-local"));
        let actual = plan.execute().unwrap();
        eprintln!(
            "indexed {policy:?}: warm decoded={} seeks={} bytes={}; prefix decoded={} bytes={}; cold decoded={} bytes={}",
            actual.report.decoded_frames,
            actual.report.seeks,
            actual.report.bytes_read,
            reference.report.decoded_frames,
            reference.report.bytes_read,
            index.report().decoded_frames,
            index.report().bytes_read
        );
        assert_eq!(actual.report.index_cache_hits, 1);
        assert!(actual.report.seeks > 0 && actual.report.seeks <= 3);
        assert!(
            actual.report.decoded_frames < reference.report.decoded_frames / 2,
            "warm decoded{} vs prefix{}",
            actual.report.decoded_frames,
            reference.report.decoded_frames
        );
        assert!(actual.report.bytes_read > 0);
        for (actual, reference) in actual.frames.iter().zip(&reference.frames) {
            assert_eq!(actual.timing, reference.timing);
            assert_eq!(
                actual.image.packed_bytes().unwrap(),
                reference.image.packed_bytes().unwrap()
            );
        }
        let selected = actual.selection.as_ref().unwrap();
        let expected = reference.selection.as_ref().unwrap();
        for (selected, expected) in selected.selections.iter().zip(&expected.selections) {
            assert_eq!(
                (selected.requested, selected.actual, &selected.identity),
                (expected.requested, expected.actual, &expected.identity)
            );
        }
        assert!(
            actual.frames[0]
                .image
                .storage()
                .shares_allocation(actual.frames[2].image.storage())
        );
    }
}

#[test]
fn persisted_index_reuse_and_long_gop_keep_version_policy_and_budget_checks() {
    let Some(fixture) = Fixture::new(120) else {
        return;
    };
    let session = Session::default();
    let request = session.open(&fixture.path).unwrap().video().unwrap();
    let index = request.build_index(IndexBuildLimits::default()).unwrap();
    let cache = fixture.directory.join("timing.idx");
    index.save(&cache).unwrap();
    assert!(index.save(&cache).is_err());
    let loaded = request
        .load_index(&cache, IndexBuildLimits::default())
        .unwrap();
    assert!(loaded.report().loaded);
    assert_eq!(loaded.report().decoded_frames, 0);
    assert_eq!(loaded.len(), index.len());
    let sample = request
        .clone()
        .sample([Time::from_millis(18050)])
        .selection(TemporalSelection::Nearest);
    let baseline = sample.read().unwrap();
    let actual = sample.clone().with_index(loaded.clone()).read().unwrap();
    assert_eq!(actual.report.seeks, 1);
    assert!(actual.report.decoded_frames < baseline.report.decoded_frames / 4);
    assert_eq!(actual.frames[0].timing, baseline.frames[0].timing);
    assert!(
        sample
            .clone()
            .with_index(loaded.clone())
            .indexed_limits(IndexSeekLimits {
                max_decoded_frames: 1,
                ..Default::default()
            })
            .read()
            .is_err()
    );
    assert!(
        request
            .clone()
            .decode_options(VideoDecodeOptions {
                duration_policy: FrameDurationPolicy::TrustReported,
                ..Default::default()
            })
            .load_index(&cache, IndexBuildLimits::default())
            .is_err()
    );
    let mut broken = fs::read(&cache).unwrap();
    let end = broken.len() - 1;
    broken[end] ^= 1;
    let damaged = fixture.directory.join("damaged.idx");
    fs::write(&damaged, broken).unwrap();
    assert!(
        request
            .load_index(&damaged, IndexBuildLimits::default())
            .is_err()
    );
    let mut source = fs::read(&fixture.path).unwrap();
    source.push(0);
    fs::write(&fixture.path, source).unwrap();
    assert_eq!(
        sample
            .with_index(loaded)
            .read()
            .unwrap_err()
            .context()
            .constraint
            .as_deref(),
        Some("video_index_source")
    );
    assert!(
        request
            .load_index(&cache, IndexBuildLimits::default())
            .is_err()
    );
}

#[test]
fn fractional_rate_keyframes_seek_in_exact_stream_ticks_without_microsecond_flooring() {
    let Some(fixture) = Fixture::with_rate(61, "30000/1001") else {
        return;
    };
    let request = Session::default()
        .open(&fixture.path)
        .unwrap()
        .video()
        .unwrap();
    let index = request.build_index(IndexBuildLimits::default()).unwrap();
    assert!(
        index
            .entries()
            .iter()
            .filter(|entry| entry.keyframe)
            .any(|entry| {
                let source = entry.pts.checked_add(index.timeline_origin()).unwrap();
                1_000_000 % source.denominator() != 0
            }),
        "fixture must contain native keyframe PTS that cannot be represented as integer microseconds"
    );
    let sample = request
        .sample([Time::from_millis(18050)])
        .selection(TemporalSelection::Nearest);
    let baseline = sample.read().unwrap();
    let actual = sample.with_index(index).read().unwrap();
    assert_eq!(actual.report.seeks, 1);
    assert!(actual.report.decoded_frames < baseline.report.decoded_frames / 2);
    assert_eq!(actual.frames[0].timing, baseline.frames[0].timing);
    assert_eq!(
        actual.frames[0].image.packed_bytes().unwrap(),
        baseline.frames[0].image.packed_bytes().unwrap()
    );
}

#[test]
fn explicit_index_limits_cancellation_and_nonseekable_sources_never_scan_implicitly() {
    let Some(fixture) = Fixture::new(60) else {
        return;
    };
    let session = Session::default();
    let request = session.open(&fixture.path).unwrap().video().unwrap();
    let cancel = CancellationToken::default();
    cancel.cancel();
    assert_eq!(
        request
            .build_index_with_cancellation(IndexBuildLimits::default(), cancel.clone())
            .unwrap_err()
            .kind(),
        ErrorKind::Cancelled
    );
    assert_eq!(session.live_cpu_usage().executions, 0);
    assert_eq!(session.live_host_bytes(), 0);
    assert!(
        request
            .build_index(IndexBuildLimits {
                max_frames: 2,
                ..Default::default()
            })
            .is_err()
    );
    assert_eq!(session.live_cpu_usage().executions, 0);
    assert_eq!(session.live_host_bytes(), 0);
    let bytes = fs::read(&fixture.path).unwrap();
    let sequential = Source::from_reader_factory(
        "sequential-index",
        Some("v1".into()),
        Some(bytes.len() as u64),
        move || Ok(std::io::Cursor::new(bytes.clone())),
    )
    .unwrap();
    assert_eq!(
        session
            .from_source(sequential)
            .video()
            .unwrap()
            .build_index(IndexBuildLimits::default())
            .unwrap_err()
            .kind(),
        ErrorKind::Unsupported
    );
    let index = request.build_index(IndexBuildLimits::default()).unwrap();
    let retained = session.live_host_bytes();
    assert_eq!(retained, index.report().retained_index_bytes);
    assert_eq!(
        request
            .with_index(index.clone())
            .sample(times())
            .with_cancellation(cancel)
            .read()
            .unwrap_err()
            .kind(),
        ErrorKind::Cancelled
    );
    assert_eq!(session.live_cpu_usage().executions, 0);
    assert_eq!(session.live_host_bytes(), retained);
    drop(index);
    assert_eq!(session.live_host_bytes(), 0);
}

#[test]
fn indexed_header_bytes_metadata_admission_and_native_read_cancellation_are_enforced() {
    let Some(fixture) = Fixture::new(60) else {
        return;
    };
    let counts = ReadCounts::default();
    let source = counted_source(&fixture, "v1", &counts);
    let session = Session::default();
    let request = session.from_source(source.clone()).video().unwrap();
    let index = request.build_index(IndexBuildLimits::default()).unwrap();
    counts.reset();
    let sampled = request
        .clone()
        .with_index(index.clone())
        .sample([Time::from_millis(18050)])
        .selection(TemporalSelection::Nearest);
    let error = sampled
        .clone()
        .indexed_limits(IndexSeekLimits {
            max_source_bytes: 64,
            ..Default::default()
        })
        .read()
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::ResourceExhausted);
    assert!(
        counts.bytes.load(Ordering::SeqCst) <= 64,
        "initial indexed headers must obey the source cap before later seek setup"
    );
    assert!(counts.opens.load(Ordering::SeqCst) > 0);
    assert_eq!(session.live_cpu_usage().executions, 0);

    counts.reset();
    let constrained = Session::builder()
        .limits(media::prelude::Limits::new().host_memory(16 * 1024))
        .build()
        .unwrap();
    let error = constrained
        .from_source(source)
        .video()
        .unwrap()
        .with_index(index.clone())
        .sample(std::iter::repeat_n(Time::from_millis(100), 65536))
        .selection(TemporalSelection::Nearest)
        .read()
        .unwrap_err();
    assert_eq!(error.context().constraint.as_deref(), Some("host_memory"));
    assert_eq!(
        counts.opens.load(Ordering::SeqCst),
        0,
        "query metadata must be admitted before native open/scheduling allocations"
    );
    assert_eq!(constrained.live_host_bytes(), 0);
    assert_eq!(constrained.live_cpu_usage().executions, 0);
    assert!(
        request
            .build_index(IndexBuildLimits {
                max_index_bytes: 32,
                ..Default::default()
            })
            .is_err()
    );
    assert_eq!(counts.opens.load(Ordering::SeqCst), 0);

    counts.reset();
    counts.cancel_after.store(3, Ordering::SeqCst);
    let error = sampled
        .with_cancellation(counts.cancel.clone())
        .read()
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Cancelled);
    assert!(
        counts.calls.load(Ordering::SeqCst) <= 3,
        "cancel must stop native reads rather than scanning remaining packets"
    );
    assert_eq!(session.live_cpu_usage().executions, 0);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires real NVDEC/CUDA/NVRTC; indexed reads never fall back or skip missing hardware"]
fn indexed_cuda_vfr_matches_prefix_preserves_ownership_and_enforces_work_io_and_cancel_limits() {
    use media::backend::cuda::{CudaLimits, PreprocessSpec};
    let fixture = Fixture::new(60).expect("real GPU test requires FFmpeg fixture generator");
    let counts = ReadCounts::default();
    let source = counted_source(&fixture, "v1", &counts);
    let session = Session::default();
    let base = session
        .from_source(source.clone())
        .video()
        .unwrap()
        .cuda(0, CudaLimits::default());
    let index = base
        .build_index(IndexBuildLimits::default())
        .expect("index construction must use actual NVDEC");
    assert!(index.report().hardware_decode);
    assert_eq!(index.report().decoded_frames, index.len() as u64);
    assert!(index.len() > 400);
    let sampled = base
        .clone()
        .sample(times())
        .duration_inference(DurationInference::NextTimestamp);
    let reference = sampled.read().unwrap();
    let prefix_decoded = reference.report.decoded_frames;
    let expected = reference
        .frames
        .iter()
        .map(|frame| frame.frame().timing)
        .collect::<Vec<_>>();
    let spec = PreprocessSpec::new(8, 8).unwrap();
    let reference = reference.into_tensor(spec).unwrap();
    let context = reference.context().clone();
    let transfer = context.stream().unwrap();
    // Explicit test-oracle tensor downloads are separate from source decode;
    // no decoded source frame is transferred or substituted by a CPU decoder.
    let expected_values = reference.tensor.download(&transfer).unwrap();
    drop(reference);

    let indexed = sampled.clone().with_index(index.clone());
    let actual = indexed.read().unwrap();
    assert!(actual.report.decoded_frames < prefix_decoded / 2);
    assert!(actual.report.seeks > 0 && actual.report.seeks <= 3);
    assert_eq!(actual.report.index_cache_hits, 1);
    assert_eq!(actual.report.decoded_host_transfer_bytes, 0);
    for (frame, expected) in actual.frames.iter().zip(&expected) {
        assert_eq!(frame.frame().timing, *expected);
    }
    assert!(
        actual.frames[0]
            .frame()
            .image
            .storage()
            .shares_allocation(actual.frames[2].frame().image.storage())
    );
    let retained_frame = actual.frames[0].clone();
    let actual = actual.into_tensor(spec).unwrap();
    assert_eq!(actual.tensor.download(&transfer).unwrap(), expected_values);
    let retained_tensor = actual.tensor.clone();
    drop(actual);
    for _ in 0..200 {
        if session.live_cuda_codec_sessions() == 1 {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    assert_eq!(
        session.live_cuda_codec_sessions(),
        1,
        "only the deliberately retained source frame keeps a codec admission"
    );

    counts.reset();
    let error = indexed
        .clone()
        .indexed_limits(IndexSeekLimits {
            max_source_bytes: 64,
            ..Default::default()
        })
        .read()
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::ResourceExhausted);
    assert!(counts.bytes.load(Ordering::SeqCst) <= 64);
    let error = indexed
        .clone()
        .indexed_limits(IndexSeekLimits {
            max_decoded_frames: 1,
            ..Default::default()
        })
        .read()
        .unwrap_err();
    assert_eq!(
        error.context().constraint.as_deref(),
        Some("indexed_decoded_frames")
    );
    counts.reset();
    let limited = Session::builder()
        .limits(media::prelude::Limits::new().host_memory(16 * 1024))
        .build()
        .unwrap();
    let error = limited
        .from_source(source)
        .video()
        .unwrap()
        .cuda(0, CudaLimits::default())
        .with_index(index.clone())
        .sample(std::iter::repeat_n(Time::from_millis(100), 65536))
        .selection(TemporalSelection::Nearest)
        .read()
        .unwrap_err();
    assert_eq!(error.context().constraint.as_deref(), Some("host_memory"));
    assert_eq!(counts.opens.load(Ordering::SeqCst), 0);
    assert_eq!(limited.live_cuda_codec_sessions(), 0);
    assert_eq!(limited.live_host_bytes(), 0);
    let changed = counted_source(&fixture, "v2", &counts);
    counts.reset();
    assert_eq!(
        session
            .from_source(changed)
            .video()
            .unwrap()
            .cuda(0, CudaLimits::default())
            .with_index(index.clone())
            .sample([Time::ZERO])
            .read()
            .unwrap_err()
            .context()
            .constraint
            .as_deref(),
        Some("video_index_source")
    );
    assert_eq!(counts.opens.load(Ordering::SeqCst), 0);
    counts.reset();
    counts.cancel_after.store(3, Ordering::SeqCst);
    assert_eq!(
        indexed
            .clone()
            .with_cancellation(counts.cancel.clone())
            .read()
            .unwrap_err()
            .kind(),
        ErrorKind::Cancelled
    );
    assert!(counts.calls.load(Ordering::SeqCst) <= 3);

    // A retained indexed native surface remains usable after all requests/index
    // objects retire, then tensor aliases keep output metadata/batch ownership.
    drop(indexed);
    drop(sampled);
    drop(base);
    drop(index);
    let input = retained_frame.yuv_view().unwrap();
    drop(retained_frame);
    let pixel = input
        .preprocess(&transfer, PreprocessSpec::new(1, 1).unwrap())
        .unwrap();
    drop(input);
    assert_eq!(pixel.download(&transfer).unwrap().len(), 3);
    drop(pixel);
    for _ in 0..200 {
        if session.reserved_cuda_native_bytes() == 0 {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    assert_eq!(session.reserved_cuda_native_bytes(), 0);
    assert_eq!(session.live_cuda_codec_sessions(), 0);
    assert!(
        session.live_host_bytes() > 0,
        "retained tensor owns indexed-result metadata admission"
    );
    assert_eq!(
        retained_tensor.download(&transfer).unwrap(),
        expected_values
    );
    drop(retained_tensor);
    assert_eq!(session.live_host_bytes(), 0);
    assert_eq!(session.live_batches(), 0);
}
