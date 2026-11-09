//! Native audio decode, sample grids, mixing, filter state, and explicit WAV completion.

use media_backend_native::{
    audio::{AudioDecodeOptions, AudioReader, MixMatrix, WavWriter},
    resample::{AudioWindows, Resampler, slice_block},
};
use media_core::{
    media::{AudioBlock, ChannelLayout, ChannelPosition},
    time::{Time, TimeRange},
};
use media_io::{sink::Sink, source::Source};
use std::io::{Cursor, Read};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

fn pcm16_wav(samples: &[i16], rate: u32, channels: u16) -> Vec<u8> {
    let bytes = samples.len() as u32 * 2;
    let mut data = Vec::new();
    data.extend_from_slice(b"RIFF");
    data.extend_from_slice(&(36 + bytes).to_le_bytes());
    data.extend_from_slice(b"WAVEfmt ");
    data.extend_from_slice(&16_u32.to_le_bytes());
    data.extend_from_slice(&1_u16.to_le_bytes());
    data.extend_from_slice(&channels.to_le_bytes());
    data.extend_from_slice(&rate.to_le_bytes());
    data.extend_from_slice(&(rate * u32::from(channels) * 2).to_le_bytes());
    data.extend_from_slice(&(channels * 2).to_le_bytes());
    data.extend_from_slice(&16_u16.to_le_bytes());
    data.extend_from_slice(b"data");
    data.extend_from_slice(&bytes.to_le_bytes());
    for sample in samples {
        data.extend_from_slice(&sample.to_le_bytes());
    }
    data
}

fn block(samples: Vec<f32>, rate: u32, layout: ChannelLayout, start: Time) -> AudioBlock {
    AudioBlock::new(samples, rate, layout, start).unwrap()
}

#[test]
fn streaming_pcm_decode_preserves_values_and_exact_timing() {
    let samples: Vec<i16> = (0..4321)
        .flat_map(|index| [index as i16, -(index as i16)])
        .collect();
    let source = Source::memory(pcm16_wav(&samples, 48_000, 2));
    let options = AudioDecodeOptions {
        max_block_frames: 173,
        ..Default::default()
    };
    let mut reader = AudioReader::open(&source, options).unwrap();
    assert_eq!(reader.info().sample_rate, 48_000);
    assert_eq!(reader.info().layout, ChannelLayout::stereo());
    assert!(reader.report().probe_bytes <= 1024 * 1024);
    let mut decoded = Vec::new();
    let mut next = Time::ZERO;
    while let Some(audio) = reader.next_block().unwrap() {
        assert!(audio.frames() <= 173);
        assert_eq!(audio.start(), next);
        next = audio.end().unwrap();
        decoded.extend_from_slice(audio.samples());
    }
    let expected: Vec<f32> = samples
        .iter()
        .map(|value| f32::from(*value) / 32768.0)
        .collect();
    assert_eq!(decoded, expected);
    assert_eq!(next, Time::from_samples(4321, 48_000).unwrap());
    assert_eq!(reader.report().output_frames, 4321);
    assert!(reader.report().finished);
    assert!(reader.next_block().unwrap().is_none());
}

#[test]
fn independent_audio_readers_and_nonseekable_sources_work() {
    let bytes = pcm16_wav(&[0, 1000, -1000, 32767, -32768], 16_000, 1);
    let source =
        Source::from_reader_factory("wav-stream", None, Some(bytes.len() as u64), move || {
            Ok(Cursor::new(bytes.clone()))
        })
        .unwrap();
    assert!(!source.capabilities().unwrap().seekable);
    let mut first = AudioReader::open(&source, Default::default()).unwrap();
    let mut second = AudioReader::open(&source, Default::default()).unwrap();
    assert_eq!(
        first.info().layout.positions(),
        &[ChannelPosition::FrontLeft]
    );
    assert_eq!(
        first.next_block().unwrap().unwrap().samples(),
        second.next_block().unwrap().unwrap().samples()
    );
}

struct Counted {
    cursor: Cursor<Vec<u8>>,
    reads: Arc<AtomicUsize>,
}
impl Read for Counted {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        let count = self.cursor.read(buffer)?;
        self.reads.fetch_add(count, Ordering::Relaxed);
        Ok(count)
    }
}

#[test]
fn probing_and_owned_buffers_are_limited() {
    let bytes = pcm16_wav(&vec![0; 8192], 16_000, 1);
    let reads = Arc::new(AtomicUsize::new(0));
    let counter = reads.clone();
    let source = Source::from_reader_factory(
        "counted",
        Some("v1".into()),
        Some(bytes.len() as u64),
        move || {
            Ok(Counted {
                cursor: Cursor::new(bytes.clone()),
                reads: counter.clone(),
            })
        },
    )
    .unwrap();
    let result = AudioReader::open(
        &source,
        AudioDecodeOptions {
            max_probe_bytes: 8,
            ..Default::default()
        },
    );
    assert!(result.is_err());
    assert!(reads.load(Ordering::Relaxed) <= 8);
    let source = Source::memory(pcm16_wav(&vec![0; 8192], 16_000, 1));
    let mut reader = AudioReader::open(
        &source,
        AudioDecodeOptions {
            max_buffered_samples: 8,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(reader.next_block().is_err());
    assert!(reader.next_block().is_err());
    assert!(
        AudioReader::open(
            &source,
            AudioDecodeOptions {
                track: Some(u32::MAX),
                ..Default::default()
            }
        )
        .is_err()
    );
}

#[test]
fn truncated_input_is_not_normal_eof() {
    let mut bytes = pcm16_wav(&vec![4; 3000], 16_000, 1);
    bytes.truncate(bytes.len() - 17);
    let source = Source::memory(bytes);
    match AudioReader::open(&source, Default::default()) {
        Err(_) => {}
        Ok(mut reader) => {
            let mut failed = false;
            loop {
                match reader.next_block() {
                    Ok(Some(_)) => {}
                    Ok(None) => break,
                    Err(_) => {
                        failed = true;
                        break;
                    }
                }
            }
            assert!(
                failed,
                "truncated declared PCM data must not be accepted as successful EOF"
            );
            assert!(!reader.report().finished);
        }
    }
}

#[test]
fn mixing_uses_speaker_identity_and_explicit_coefficients() {
    let stereo = block(
        vec![1.0, -1.0, 0.5, 0.25],
        16_000,
        ChannelLayout::stereo(),
        Time::from_millis(23),
    );
    let mono = MixMatrix::mono(ChannelLayout::stereo())
        .unwrap()
        .process(&stereo)
        .unwrap();
    assert_eq!(mono.samples(), &[0.0, 0.375]);
    assert_eq!(mono.start(), stereo.start());
    let discrete = ChannelLayout::new(vec![
        ChannelPosition::Discrete(0),
        ChannelPosition::Discrete(1),
    ])
    .unwrap();
    assert!(MixMatrix::mono(discrete.clone()).is_err());
    assert!(MixMatrix::stereo(discrete.clone()).is_err());
    let matrix = MixMatrix::new(discrete.clone(), ChannelLayout::mono(), vec![0.0, 2.0]).unwrap();
    let output = matrix
        .process(&block(vec![0.25, 0.75], 16_000, discrete, Time::ZERO))
        .unwrap();
    assert_eq!(output.samples(), &[1.5]); // float headroom is retained, not clipped.
    assert!(matrix.process(&stereo).is_err());
    assert!(MixMatrix::new(ChannelLayout::mono(), ChannelLayout::mono(), vec![f32::NAN]).is_err());
}

#[test]
fn surround_mix_omits_lfe_and_maps_named_surrounds() {
    let layout = ChannelLayout::new(vec![
        ChannelPosition::FrontLeft,
        ChannelPosition::FrontRight,
        ChannelPosition::FrontCenter,
        ChannelPosition::Lfe,
        ChannelPosition::SideLeft,
        ChannelPosition::SideRight,
    ])
    .unwrap();
    let mix = MixMatrix::stereo(layout.clone()).unwrap();
    let input = block(
        vec![0.0, 0.0, 0.0, 100.0, 1.0, -1.0],
        48_000,
        layout,
        Time::ZERO,
    );
    let out = mix.process(&input).unwrap();
    assert!((out.samples()[0] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6);
    assert!((out.samples()[1] + std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6);
}

#[test]
fn wav_float_output_is_finalized_and_decodes_without_clipping() {
    let first = block(
        vec![-1.5, 0.0, 0.5, 1.5],
        16_000,
        ChannelLayout::mono(),
        Time::ZERO,
    );
    let mut writer = WavWriter::new(Sink::memory(), 16_000, ChannelLayout::mono()).unwrap();
    writer.write(&first).unwrap();
    let bytes = writer.finish().unwrap().into_bytes().unwrap();
    assert_eq!(&bytes[0..4], b"RIFF");
    assert_eq!(
        u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize + 8,
        bytes.len()
    );
    assert_eq!(u32::from_le_bytes(bytes[68..72].try_into().unwrap()), 4);
    let mut reader = AudioReader::open(&Source::memory(bytes), Default::default()).unwrap();
    let audio = reader.next_block().unwrap().unwrap();
    assert_eq!(audio.samples(), first.samples());
    assert_eq!(audio.channel_layout(), &ChannelLayout::mono());
    assert!(reader.next_block().unwrap().is_none());
}

fn resample(input: &[f32], chunk: usize, from: u32, to: u32) -> (Vec<f32>, Vec<Time>) {
    let origin = Time::from_millis(17);
    let mut resampler = Resampler::new(from, to, ChannelLayout::mono(), origin, 512).unwrap();
    let mut samples = Vec::new();
    let mut starts = Vec::new();
    for (index, part) in input.chunks(chunk).enumerate() {
        let start = origin
            .checked_add(Time::from_samples((index * chunk) as i64, from).unwrap())
            .unwrap();
        for block in resampler
            .push(&block(part.to_vec(), from, ChannelLayout::mono(), start))
            .unwrap()
        {
            starts.push(block.start());
            samples.extend_from_slice(block.samples());
        }
        assert!(resampler.buffered_input_frames() < resampler.input_chunk_frames());
    }
    for block in resampler.finish().unwrap() {
        starts.push(block.start());
        samples.extend_from_slice(block.samples());
    }
    assert!(resampler.finish().unwrap().is_empty());
    assert!(
        resampler
            .push(&block(Vec::new(), from, ChannelLayout::mono(), origin))
            .is_err()
    );
    (samples, starts)
}

#[test]
fn resampling_is_continuous_across_arbitrary_chunk_boundaries_and_drains_exactly() {
    let input: Vec<f32> = (0..10_001).map(|n| ((n as f32) * 0.035).sin()).collect();
    let (all, all_starts) = resample(&input, input.len(), 44_100, 16_000);
    let (chunked, starts) = resample(&input, 137, 44_100, 16_000);
    assert_eq!(all.len(), (input.len() * 16_000).div_ceil(44_100));
    assert_eq!(all, chunked);
    assert_eq!(all_starts, starts);
    assert_eq!(starts[0], Time::from_millis(17));
    for length in [0, 1, 2, 440, 441, 442] {
        assert_eq!(
            resample(&input[..length], 17, 44_100, 16_000).0.len(),
            (length * 16_000).div_ceil(44_100)
        );
    }
}

#[test]
fn downsampling_rejects_above_nyquist_energy() {
    let tone = |frequency: f32| -> Vec<f32> {
        (0..48_000)
            .map(|index| (2.0 * std::f32::consts::PI * frequency * index as f32 / 48_000.0).sin())
            .collect()
    };
    let low = resample(&tone(1000.0), 511, 48_000, 16_000).0;
    let high = resample(&tone(12_000.0), 511, 48_000, 16_000).0;
    let rms = |samples: &[f32]| {
        (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
    };
    assert!(rms(&low[256..low.len() - 256]) > 0.65);
    assert!(rms(&high[256..high.len() - 256]) < 0.01);
}

#[test]
fn resampler_rejects_format_changes_gaps_and_unbounded_working_sets() {
    let origin = Time::ZERO;
    let mut resampler = Resampler::new(48_000, 16_000, ChannelLayout::mono(), origin, 512).unwrap();
    assert!(resampler.latency_frames() > 0);
    assert!(
        resampler
            .push(&block(
                vec![0.0],
                48_000,
                ChannelLayout::mono(),
                Time::from_millis(1)
            ))
            .is_err()
    );
    assert!(
        resampler
            .push(&block(vec![0.0], 44_100, ChannelLayout::mono(), origin))
            .is_err()
    );
    assert!(Resampler::new(0, 16_000, ChannelLayout::mono(), origin, 512).is_err());
    assert!(Resampler::new(1, u32::MAX, ChannelLayout::mono(), origin, 512).is_err());
}

#[test]
fn fractional_audio_slices_share_one_continuous_grid() {
    let block = block(
        (0..100).map(|n| n as f32).collect(),
        16_000,
        ChannelLayout::mono(),
        Time::ZERO,
    )
    .with_source_start(Some(Time::from_millis(3)));
    let a = Time::new(1, 32_000).unwrap();
    let b = Time::new(11, 32_000).unwrap();
    let c = Time::new(31, 32_000).unwrap();
    let left = slice_block(&block, TimeRange::new(a, b).unwrap()).unwrap();
    let right = slice_block(&block, TimeRange::new(b, c).unwrap()).unwrap();
    let full = slice_block(&block, TimeRange::new(a, c).unwrap()).unwrap();
    assert_eq!(left.start(), Time::from_samples(1, 16_000).unwrap());
    assert_eq!(
        left.source_start(),
        Some(Time::from_millis(3).checked_add(left.start()).unwrap())
    );
    assert_eq!(left.end().unwrap(), right.start());
    assert_eq!([left.samples(), right.samples()].concat(), full.samples());
    assert!(slice_block(&block, TimeRange::new(Time::from_millis(-1), a).unwrap()).is_err());
}

#[test]
fn overlapping_windows_preserve_grid_across_input_chunks_and_bound_memory() {
    let mut windows = AudioWindows::new(320, 160).unwrap();
    let origin = Time::from_millis(50);
    let data: Vec<f32> = (0..3300).map(|n| n as f32).collect();
    let mut output = Vec::new();
    for (index, samples) in data.chunks(127).enumerate() {
        output.extend(
            windows
                .push(&block(
                    samples.to_vec(),
                    16_000,
                    ChannelLayout::mono(),
                    origin
                        .checked_add(Time::from_samples((index * 127) as i64, 16_000).unwrap())
                        .unwrap(),
                ))
                .unwrap(),
        );
        assert!(windows.buffered_samples() < 320);
    }
    assert_eq!(output.len(), 19);
    for (index, window) in output.iter().enumerate() {
        assert_eq!(window.frames(), 320);
        assert_eq!(window.samples(), &data[index * 160..index * 160 + 320]);
        assert_eq!(
            window.start(),
            origin
                .checked_add(Time::from_samples((index * 160) as i64, 16_000).unwrap())
                .unwrap()
        );
    }
    assert!(windows.finish().unwrap().is_empty());
    assert_eq!(windows.discarded_frames(), 260);
    assert!(
        windows
            .push(&block(Vec::new(), 16_000, ChannelLayout::mono(), origin))
            .is_err()
    );
}

#[test]
fn separated_windows_skip_explicit_hop_gaps() {
    let mut windows = AudioWindows::new(3, 5).unwrap();
    let data: Vec<f32> = (0..15).map(|n| n as f32).collect();
    let out = windows
        .push(&block(
            data.clone(),
            16_000,
            ChannelLayout::mono(),
            Time::ZERO,
        ))
        .unwrap();
    assert_eq!(out.len(), 3);
    for (index, window) in out.iter().enumerate() {
        assert_eq!(window.samples(), &data[index * 5..index * 5 + 3]);
    }
}
