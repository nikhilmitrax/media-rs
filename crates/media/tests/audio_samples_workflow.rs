//! Typed PCM conversion, retained budgets, caller storage, and transactional WAV.
#![cfg(feature = "native")]
use media::{
    backend::native::{
        audio::WavWriter,
        audio_samples::{SampleFormat, SampleGamut, SampleOutput, SamplePacking, TpdfDither},
    },
    io::{sink::Sink, source::Source},
    prelude::*,
};
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

fn wav(samples: Vec<f32>, layout: ChannelLayout) -> Vec<u8> {
    let input = AudioBlock::new(samples, 48_000, layout.clone(), Time::ZERO).unwrap();
    let mut writer = WavWriter::new(Sink::memory(), 48_000, layout).unwrap();
    writer.write(&input).unwrap();
    writer.finish().unwrap().into_bytes().unwrap()
}
fn mono(samples: Vec<f32>) -> Source {
    Source::memory(wav(samples, ChannelLayout::mono()))
}

#[test]
fn planar_typed_output_retains_only_its_aligned_allocation() {
    let session = Session::default();
    let output = session
        .from_source(Source::memory(wav(
            vec![-1., 1., -0.5, 0.5, 0., 0.25],
            ChannelLayout::stereo(),
        )))
        .audio()
        .unwrap()
        .sample_format(SampleFormat::I16)
        .packing(SamplePacking::Planar)
        .read()
        .unwrap();
    assert_eq!(
        output.as_i16().unwrap(),
        &[-32768, -16384, 0, 32767, 16384, 8192]
    );
    assert_eq!(output.as_i16().unwrap().as_ptr() as usize % 64, 0);
    assert_eq!(output.shape(), [2, 3]);
    assert_eq!(output.strides(), [3, 1]);
    assert_eq!(output.sample_rate(), 48_000);
    assert_eq!(output.start(), Time::ZERO);
    assert_eq!(output.source_start(), Some(Time::ZERO));
    assert_eq!(
        output.end().unwrap(),
        Time::from_samples(3, 48_000).unwrap()
    );
    assert_eq!(output.report().quantized_samples, 6);
    assert_eq!(session.live_host_bytes(), output.allocation_bytes());
    let bytes = output.allocation_bytes();
    let clone = output.clone();
    assert!(clone.shares_storage(&output));
    drop(output);
    assert_eq!(session.live_host_bytes(), bytes);
    drop(clone);
    assert_eq!(session.live_host_bytes(), 0);
}

#[test]
fn fractional_resample_slice_and_chunked_dither_share_the_absolute_grid() {
    let session = Session::default();
    let base = session
        .from_source(mono(vec![0.; 8192]))
        .audio()
        .unwrap()
        .resample(16_000);
    let dither = TpdfDither {
        seed: 9127,
        grid_origin: Time::ZERO,
    };
    let typed = base
        .clone()
        .sample_format(SampleFormat::I16)
        .dither(Some(dither));
    let all = typed.read().unwrap();
    let mut stream = typed.stream().unwrap();
    let mut values = Vec::new();
    let mut count = 0;
    while let Some(block) = stream.next_buffer().unwrap() {
        assert_eq!(
            block.start(),
            Time::from_samples(values.len() as i64, 16_000).unwrap()
        );
        values.extend_from_slice(block.as_i16().unwrap());
        count += 1;
    }
    assert!(count > 1);
    assert_eq!(values, all.as_i16().unwrap());
    assert_eq!(stream.conversion_report(), all.report());
    assert!(stream.decode_report().finished);
    assert_eq!(session.live_host_bytes(), all.allocation_bytes());
    let range = TimeRange::new(
        Time::new(1001, 48_000).unwrap(),
        Time::new(3501, 48_000).unwrap(),
    )
    .unwrap();
    let slice = base
        .slice(range)
        .sample_format(SampleFormat::I16)
        .dither(Some(dither))
        .read()
        .unwrap();
    assert_eq!(slice.start(), Time::from_samples(334, 16_000).unwrap());
    assert_eq!(slice.frames(), 833);
    assert_eq!(slice.as_i16().unwrap(), &all.as_i16().unwrap()[334..1167]);
    drop(slice);
    drop(all);
    assert_eq!(session.live_host_bytes(), 0);
}

#[test]
fn caller_storage_is_exclusive_and_unchanged_on_any_failure() {
    let session = Session::default();
    let request = session
        .from_source(Source::memory(wav(
            vec![0.5, -0.5, 1., -1.],
            ChannelLayout::stereo(),
        )))
        .audio()
        .unwrap()
        .sample_format(SampleFormat::I32)
        .packing(SamplePacking::Planar);
    let mut left = [77; 2];
    let mut right = [77; 2];
    let report = request
        .process_into(SampleOutput::PlanarI32(&mut [&mut left, &mut right]))
        .unwrap();
    assert_eq!(left, [1_073_741_824, i32::MAX]);
    assert_eq!(right, [-1_073_741_824, i32::MIN]);
    assert_eq!(report.quantized_samples, 4);
    assert_eq!(session.live_host_bytes(), 0);
    let mut wrong = [77; 4];
    assert!(
        request
            .process_into(SampleOutput::InterleavedI32(&mut wrong))
            .is_err()
    );
    assert_eq!(wrong, [77; 4]);
    let malformed = session
        .from_source(mono(vec![0., f32::NAN]))
        .audio()
        .unwrap()
        .sample_format(SampleFormat::I16);
    let mut destination = [91; 2];
    assert!(
        malformed
            .process_into(SampleOutput::InterleavedI16(&mut destination))
            .is_err()
    );
    assert_eq!(destination, [91; 2]);
    assert_eq!(session.live_host_bytes(), 0);
}

#[test]
fn cancellation_and_conversion_failure_retire_working_state() {
    let session = Session::default();
    let mut stream = session
        .from_source(mono(vec![0.25; 8192]))
        .audio()
        .unwrap()
        .sample_format(SampleFormat::I16)
        .stream()
        .unwrap();
    let output = stream.next_buffer().unwrap().unwrap();
    stream.cancellation().cancel();
    assert_eq!(
        stream.next_buffer().unwrap_err().kind(),
        ErrorKind::Cancelled
    );
    assert!(stream.next().is_none());
    assert_eq!(session.live_host_bytes(), output.allocation_bytes());
    drop(output);
    assert_eq!(session.live_host_bytes(), 0);
    let mut failing = session
        .from_source(mono(vec![1.1; 8192]))
        .audio()
        .unwrap()
        .sample_format(SampleFormat::I16)
        .stream()
        .unwrap();
    assert!(failing.next_buffer().is_err());
    assert_eq!(session.live_host_bytes(), 0);
    assert!(failing.next().is_none());
}

#[test]
fn retained_window_outputs_enforce_shared_session_capacity() {
    let session = Session::builder()
        .limits(Limits::new().host_memory(128 * 1024))
        .build()
        .unwrap();
    let mut stream = session
        .from_source(mono(vec![0.; 8192]))
        .audio()
        .unwrap()
        .windows(Window::samples(512).hop(1))
        .sample_format(SampleFormat::I16)
        .dither(Some(TpdfDither {
            seed: 3,
            grid_origin: Time::ZERO,
        }))
        .stream()
        .unwrap();
    let mut outputs = Vec::new();
    loop {
        match stream.next_buffer() {
            Ok(Some(output)) => {
                if let Some(previous) = outputs.last() {
                    let previous: &media::backend::native::audio_samples::ConvertedAudio = previous;
                    assert_eq!(
                        &previous.as_i16().unwrap()[1..],
                        &output.as_i16().unwrap()[..511]
                    );
                }
                outputs.push(output);
                assert!(outputs.len() < 200);
            }
            Err(error) => {
                assert_eq!(error.kind(), ErrorKind::ResourceExhausted);
                break;
            }
            Ok(None) => panic!("retained outputs should exhaust the bounded budget"),
        }
    }
    assert!(outputs.len() > 1);
    assert_eq!(
        session.live_host_bytes(),
        outputs
            .iter()
            .map(|output| output.allocation_bytes())
            .sum::<usize>()
    );
    drop(outputs);
    assert_eq!(session.live_host_bytes(), 0);
}

static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);
struct TestDirectory(PathBuf);
impl TestDirectory {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "media-pcm-facade-{}-{}",
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
fn pcm_wav_headers_payload_and_real_decode_match_explicit_format() {
    let directory = TestDirectory::new();
    let session = Session::default();
    for format in [SampleFormat::I16, SampleFormat::I32] {
        let path = directory.path(&format!("{format:?}.wav"));
        let result = session
            .from_source(mono(vec![-1., -0.5, 0., 0.5, 1.]))
            .audio()
            .unwrap()
            .sample_format(format)
            .write(&path)
            .unwrap();
        assert_eq!(result.conversion_report.quantized_samples, 5);
        assert_eq!(result.conversion_report.endpoint_saturated_samples, 1);
        assert!(result.decode_report.finished);
        let bytes = fs::read(&path).unwrap();
        assert_eq!(
            u16::from_le_bytes(bytes[20..22].try_into().unwrap()),
            0xfffe
        );
        assert_eq!(
            u16::from_le_bytes(bytes[34..36].try_into().unwrap()),
            (format.bytes() * 8) as u16
        );
        assert_eq!(u32::from_le_bytes(bytes[44..48].try_into().unwrap()), 1);
        assert_eq!(u32::from_le_bytes(bytes[68..72].try_into().unwrap()), 5);
        assert_eq!(
            u32::from_le_bytes(bytes[76..80].try_into().unwrap()),
            (5 * format.bytes()) as u32
        );
        let expected: Vec<u8> = match format {
            SampleFormat::I16 => [i16::MIN, -16384, 0, 16384, i16::MAX]
                .into_iter()
                .flat_map(i16::to_le_bytes)
                .collect(),
            SampleFormat::I32 => [i32::MIN, -1_073_741_824, 0, 1_073_741_824, i32::MAX]
                .into_iter()
                .flat_map(i32::to_le_bytes)
                .collect(),
            SampleFormat::F32 => unreachable!(),
        };
        assert_eq!(&bytes[80..], expected);
        let decoded = session
            .open(&path)
            .unwrap()
            .audio()
            .unwrap()
            .read()
            .unwrap();
        assert_eq!(decoded.frames(), 5);
        for (actual, expected) in decoded.samples().iter().zip([-1., -0.5, 0., 0.5, 1.]) {
            assert!((actual - expected).abs() <= 1. / 32768.);
        }
    }
    assert_eq!(session.live_host_bytes(), 0);
}

#[test]
fn failed_quantization_aliases_and_invalid_exports_preserve_existing_files() {
    let directory = TestDirectory::new();
    let path = directory.path("output.wav");
    fs::write(&path, b"old destination").unwrap();
    let session = Session::default();
    let mut samples = vec![0.25; 6000];
    samples[5999] = 1.5;
    let request = session
        .from_source(mono(samples))
        .audio()
        .unwrap()
        .sample_format(SampleFormat::I16);
    assert!(request.write(&path).is_err());
    assert_eq!(fs::read(&path).unwrap(), b"old destination");
    assert_eq!(directory.names(), ["output.wav"]);
    let result = request
        .clone()
        .gamut(SampleGamut::Clip)
        .write(&path)
        .unwrap();
    assert_eq!(result.conversion_report.gamut_clipped_samples, 1);
    let original = fs::read(&path).unwrap();
    let input = session
        .open(&path)
        .unwrap()
        .audio()
        .unwrap()
        .sample_format(SampleFormat::I32);
    assert!(input.write(&path).is_err());
    let hardlink = directory.path("alias.wav");
    fs::hard_link(&path, &hardlink).unwrap();
    assert!(input.write(&hardlink).is_err());
    assert_eq!(fs::read(&path).unwrap(), original);
    assert!(
        request
            .clone()
            .packing(SamplePacking::Planar)
            .write(&path)
            .is_err()
    );
    let wrong_extension = directory.path("output.mp3");
    fs::write(&wrong_extension, b"old mp3").unwrap();
    assert!(request.write(&wrong_extension).is_err());
    assert_eq!(fs::read(&wrong_extension).unwrap(), b"old mp3");
    assert_eq!(session.live_host_bytes(), 0);
}
