//! PCM endpoints, ties-even quantization, planar output, and grid-stable TPDF.
use media_backend_native::audio_samples::*;
use media_core::{
    error::ErrorKind,
    media::{AudioBlock, ChannelLayout},
    time::Time,
};

fn block(values: Vec<f32>, layout: ChannelLayout, start: Time) -> AudioBlock {
    AudioBlock::new(values, 48000, layout, start).unwrap()
}

#[test]
fn pcm_endpoints_ties_even_and_explicit_clipping_have_defined_numeric_semantics() {
    let input = block(
        vec![-1., -0.5, 0., 0.5, 1.],
        ChannelLayout::mono(),
        Time::ZERO,
    );
    let output = convert(&input, SampleSpec::new(SampleFormat::I16)).unwrap();
    assert_eq!(
        output.as_i16().unwrap(),
        &[i16::MIN, -16384, 0, 16384, i16::MAX]
    );
    assert_eq!(output.report().quantized_samples, 5);
    assert_eq!(output.report().endpoint_saturated_samples, 1);
    assert_eq!(
        convert(&input, SampleSpec::new(SampleFormat::I32))
            .unwrap()
            .as_i32()
            .unwrap(),
        &[i32::MIN, -1073741824, 0, 1073741824, i32::MAX]
    );
    for format in [SampleFormat::I16, SampleFormat::I32] {
        let scale = if format == SampleFormat::I16 {
            32768.0
        } else {
            2147483648.0
        };
        let values =
            [0.5_f64, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5, -3.5].map(|value| (value / scale) as f32);
        let output = convert(
            &block(values.to_vec(), ChannelLayout::mono(), Time::ZERO),
            SampleSpec::new(format),
        )
        .unwrap();
        let actual = match format {
            SampleFormat::I16 => output
                .as_i16()
                .unwrap()
                .iter()
                .map(|v| i64::from(*v))
                .collect::<Vec<_>>(),
            SampleFormat::I32 => output
                .as_i32()
                .unwrap()
                .iter()
                .map(|v| i64::from(*v))
                .collect(),
            _ => unreachable!(),
        };
        assert_eq!(actual, vec![0, 2, 2, 4, 0, -2, -2, -4]);
    }
    let input = block(
        vec![-2., -1., 0., 1., 2.],
        ChannelLayout::mono(),
        Time::ZERO,
    );
    assert!(convert(&input, SampleSpec::new(SampleFormat::I16)).is_err());
    let output = convert(
        &input,
        SampleSpec {
            gamut: SampleGamut::Clip,
            ..SampleSpec::new(SampleFormat::I16)
        },
    )
    .unwrap();
    assert_eq!(
        output.as_i16().unwrap(),
        &[i16::MIN, i16::MIN, 0, i16::MAX, i16::MAX]
    );
    assert_eq!(output.report().gamut_clipped_samples, 2);
}

#[test]
fn planar_storage_is_physically_reordered_aligned_shared_and_preserves_source_timing() {
    let start = Time::from_millis(50);
    let input = block(
        vec![1., 0.125, 0.5, 0.25, -1., 0.375],
        ChannelLayout::stereo(),
        start,
    )
    .with_source_start(Some(Time::from_millis(49)));
    let output = convert(
        &input,
        SampleSpec {
            packing: SamplePacking::Planar,
            ..SampleSpec::new(SampleFormat::F32)
        },
    )
    .unwrap();
    assert_eq!(
        output.as_f32().unwrap(),
        &[1., 0.5, -1., 0.125, 0.25, 0.375]
    );
    assert_eq!(output.shape(), [2, 3]);
    assert_eq!(output.strides(), [3, 1]);
    assert_eq!(output.as_f32().unwrap().as_ptr().align_offset(64), 0);
    assert_eq!(output.start(), start);
    assert_eq!(output.end().unwrap(), input.end().unwrap());
    assert_eq!(output.source_start(), Some(Time::from_millis(49)));
    assert_eq!(output.report().quantized_samples, 0);
    let alias = output.clone();
    assert!(output.shares_storage(&alias));
    assert!(output.as_i16().is_err());
    let copied = convert(&input, SampleSpec::new(SampleFormat::F32)).unwrap();
    assert!(!output.shares_storage(&copied));
    assert_eq!(copied.as_f32().unwrap(), input.samples());
    assert!(output.allocation_bytes() <= allocation_bytes(3, 2, SampleFormat::F32).unwrap());
    for format in [SampleFormat::I16, SampleFormat::I32] {
        let converted = convert(&input, SampleSpec::new(format)).unwrap();
        let offset = match format {
            SampleFormat::I16 => converted.as_i16().unwrap().as_ptr().align_offset(64),
            SampleFormat::I32 => converted.as_i32().unwrap().as_ptr().align_offset(64),
            _ => unreachable!(),
        };
        assert_eq!(offset, 0);
    }
}

#[test]
fn preallocated_planar_buffers_are_checked_and_errors_never_partially_modify_outputs() {
    let input = block(
        vec![-1., 1., 0.25, -0.25],
        ChannelLayout::stereo(),
        Time::ZERO,
    );
    let spec = SampleSpec {
        packing: SamplePacking::Planar,
        ..SampleSpec::new(SampleFormat::I16)
    };
    let (mut left, mut right) = ([9_i16; 2], [9_i16; 2]);
    process_into(
        &input,
        spec,
        SampleOutput::PlanarI16(&mut [&mut left, &mut right]),
    )
    .unwrap();
    assert_eq!(left, [i16::MIN, 8192]);
    assert_eq!(right, [i16::MAX, -8192]);
    let mut packed = [9_i16; 4];
    assert!(process_into(&input, spec, SampleOutput::InterleavedI16(&mut packed)).is_err());
    assert_eq!(packed, [9; 4]);
    let (mut left, mut short) = ([9_i16; 2], [9_i16; 1]);
    assert!(
        process_into(
            &input,
            spec,
            SampleOutput::PlanarI16(&mut [&mut left, &mut short])
        )
        .is_err()
    );
    assert_eq!(left, [9; 2]);
    assert_eq!(short, [9]);
    let late_nan = block(
        vec![0., 0., 0., f32::NAN],
        ChannelLayout::stereo(),
        Time::ZERO,
    );
    assert!(
        process_into(
            &late_nan,
            SampleSpec::new(SampleFormat::I16),
            SampleOutput::InterleavedI16(&mut packed)
        )
        .is_err()
    );
    assert_eq!(packed, [9; 4]);
    let mut wrong_type = [99_i32; 4];
    assert!(
        process_into(
            &input,
            SampleSpec::new(SampleFormat::I16),
            SampleOutput::InterleavedI32(&mut wrong_type)
        )
        .is_err()
    );
    assert_eq!(wrong_type, [99; 4]);
}

#[test]
fn deterministic_tpdf_is_identical_across_chunks_and_worker_completion_order() {
    let frames = 8192;
    let origin = Time::new(1, 3).unwrap();
    let input = block(vec![0.; frames * 2], ChannelLayout::stereo(), origin);
    let spec = SampleSpec {
        dither: Some(TpdfDither {
            seed: 123,
            grid_origin: origin,
        }),
        ..SampleSpec::new(SampleFormat::I16)
    };
    let full = convert(&input, spec).unwrap();
    // Published v1 hash/numeric contract: catch accidental seed/key changes,
    // independently of whether all chunks still agree with the changed code.
    assert_eq!(
        &full.as_i16().unwrap()[..32],
        &[
            1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0,
            -1, 0, 0, 0
        ]
    );
    let before_origin = block(
        vec![0.; 8],
        ChannelLayout::stereo(),
        origin
            .checked_sub(Time::from_samples(4, 48000).unwrap())
            .unwrap(),
    );
    assert_eq!(
        convert(&before_origin, spec).unwrap().as_i16().unwrap(),
        &[0, 0, 0, 0, 0, 0, 0, 1]
    );
    let planar = convert(
        &input,
        SampleSpec {
            packing: SamplePacking::Planar,
            ..spec
        },
    )
    .unwrap();
    for frame in 0..frames {
        for channel in 0..2 {
            assert_eq!(
                planar.as_i16().unwrap()[channel * frames + frame],
                full.as_i16().unwrap()[frame * 2 + channel]
            );
        }
    }
    let mut chunks = Vec::new();
    let mut offset = 0;
    for count in [1, 127, 503, 1024, 67, 4096, 2374] {
        let count = count.min(frames - offset);
        let start = origin
            .checked_add(Time::from_samples(offset as i64, 48000).unwrap())
            .unwrap();
        chunks.push((
            offset,
            block(vec![0.; count * 2], ChannelLayout::stereo(), start),
        ));
        offset += count;
    }
    assert_eq!(offset, frames);
    for workers in [1, 2, 7] {
        let mut jobs = Vec::new();
        for worker in 0..workers {
            let work = chunks
                .iter()
                .rev()
                .filter(|(offset, _)| offset % workers == worker)
                .cloned()
                .collect::<Vec<_>>();
            jobs.push(std::thread::spawn(move || {
                work.into_iter()
                    .map(|(offset, input)| (offset, convert(&input, spec).unwrap()))
                    .collect::<Vec<_>>()
            }));
        }
        let mut outputs = jobs
            .into_iter()
            .flat_map(|job| job.join().unwrap())
            .collect::<Vec<_>>();
        outputs.sort_by_key(|v| v.0);
        assert_eq!(
            outputs
                .iter()
                .flat_map(|(_, output)| output.as_i16().unwrap().iter().copied())
                .collect::<Vec<_>>(),
            full.as_i16().unwrap()
        );
        let report = outputs
            .iter()
            .try_fold(ConversionReport::default(), |report, (_, output)| {
                report.checked_add(output.report())
            })
            .unwrap();
        assert_eq!(report, full.report());
    }
    assert_eq!(full.report().dithered_samples, (frames * 2) as u64);
    assert!(full.as_i16().unwrap().iter().all(|v| v.abs() <= 1));
    let positive = full.as_i16().unwrap().iter().filter(|&&v| v == 1).count();
    let negative = full.as_i16().unwrap().iter().filter(|&&v| v == -1).count();
    assert!(positive > 1000 && negative > 1000 && positive.abs_diff(negative) < 500);
    assert_ne!(
        full.as_i16().unwrap(),
        convert(
            &input,
            SampleSpec {
                dither: Some(TpdfDither {
                    seed: 124,
                    grid_origin: origin
                }),
                ..spec
            }
        )
        .unwrap()
        .as_i16()
        .unwrap()
    );
}

#[test]
fn wav_requires_explicit_matching_pcm_and_preflight_failure_does_not_append() {
    use media_backend_native::audio::WavWriter;
    use media_io::sink::Sink;
    let input = block(vec![-1., 1.], ChannelLayout::mono(), Time::ZERO);
    let mut writer = WavWriter::new_with_format(
        Sink::memory(),
        48000,
        ChannelLayout::mono(),
        SampleFormat::I16,
    )
    .unwrap();
    assert!(writer.write(&input).is_err());
    let packed = convert(&input, SampleSpec::new(SampleFormat::I16)).unwrap();
    let planar = convert(
        &input,
        SampleSpec {
            packing: SamplePacking::Planar,
            ..SampleSpec::new(SampleFormat::I16)
        },
    )
    .unwrap();
    assert!(writer.write_samples(&planar).is_err());
    writer.write_samples(&packed).unwrap();
    // A repeated timestamp would overlap the previously written interval.
    assert!(writer.write_samples(&packed).is_err());
    let bytes = writer.finish().unwrap().into_bytes().unwrap();
    assert_eq!(&bytes[80..], &[0, 128, 255, 127]);
    assert_eq!(u32::from_le_bytes(bytes[68..72].try_into().unwrap()), 2);
}

#[test]
fn dither_requires_exact_grid_alignment_and_checked_absolute_frame_indices() {
    let spec = SampleSpec {
        dither: Some(TpdfDither {
            seed: 1,
            grid_origin: Time::ZERO,
        }),
        ..SampleSpec::new(SampleFormat::I16)
    };
    let mut output = [9_i16; 2];
    let off_grid = block(
        vec![0., 0.],
        ChannelLayout::mono(),
        Time::new(1, 96000).unwrap(),
    );
    assert_eq!(
        process_into(&off_grid, spec, SampleOutput::InterleavedI16(&mut output))
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidInput
    );
    assert_eq!(output, [9; 2]);
    let overflowing = block(
        vec![0., 0.],
        ChannelLayout::mono(),
        Time::new(i64::MAX, 48000).unwrap(),
    );
    assert!(overflowing.end().is_ok());
    assert_eq!(
        process_into(
            &overflowing,
            spec,
            SampleOutput::InterleavedI16(&mut output)
        )
        .unwrap_err()
        .kind(),
        ErrorKind::Overflow
    );
    assert_eq!(output, [9; 2]);
    assert!(
        convert(
            &block(vec![0.], ChannelLayout::mono(), Time::ZERO),
            SampleSpec {
                dither: spec.dither,
                ..SampleSpec::new(SampleFormat::F32)
            }
        )
        .is_err()
    );
}

#[test]
fn empty_nonfinite_and_invalid_allocation_requests_have_explicit_safe_results() {
    let empty = convert(
        &block(Vec::new(), ChannelLayout::stereo(), Time::ZERO),
        SampleSpec::new(SampleFormat::I32),
    )
    .unwrap();
    assert_eq!(empty.frames(), 0);
    assert_eq!(empty.allocation_bytes(), 0);
    assert!(empty.as_i32().unwrap().is_empty());
    for sample in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(
            convert(
                &block(vec![sample], ChannelLayout::mono(), Time::ZERO),
                SampleSpec {
                    gamut: SampleGamut::Clip,
                    ..SampleSpec::new(SampleFormat::I16)
                }
            )
            .is_err()
        );
    }
    assert!(allocation_bytes(1, 0, SampleFormat::I16).is_err());
    assert!(allocation_bytes(usize::MAX, 2, SampleFormat::I32).is_err());
}
