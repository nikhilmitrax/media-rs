//! Explicit unit-float/PCM conversion, packing, and preallocated sample processing.
//!
//! Canonical AudioBlock storage remains interleaved f32. This module explicitly
//! requests unit-domain output: finite samples outside [-1,1] either fail or are
//! reported and clipped. Nonfinite samples always fail, including Clip mode.
//! Integer conversion scales by 2^(bits-1), rounds nearest with ties to even, then
//! saturates integer endpoints: -1 maps to MIN and +1 maps to MAX. Optional TPDF
//! noise is added in integer LSB units before rounding. No dither is implicit.

use media_core::{
    error::{Error, Result},
    media::{AudioBlock, ChannelLayout},
    time::{Rounding, Time, TimeBase},
};
use std::{fmt, sync::Arc};

const ALIGNMENT: usize = 64;

/// Explicit scalar representation of a converted sample buffer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SampleFormat {
    /// Native f32 unit samples; conversion performs no integer quantization.
    F32,
    /// Signed sixteen-bit PCM with scale 32768 and asymmetric endpoints.
    I16,
    /// Signed thirty-two-bit PCM with scale 2147483648 and asymmetric endpoints.
    I32,
}
impl SampleFormat {
    /// Storage bytes in one scalar sample.
    pub const fn bytes(self) -> usize {
        match self {
            Self::I16 => 2,
            Self::F32 | Self::I32 => 4,
        }
    }
}
/// Physical sample ordering in one contiguous converted allocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SamplePacking {
    /// [frame, channel], with all channels of one frame adjacent.
    Interleaved,
    /// [channel, frame], with each complete channel in a contiguous plane.
    Planar,
}
/// Handling of finite samples outside the explicitly requested unit domain.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SampleGamut {
    /// Reject before allocating output or modifying caller buffers.
    Reject,
    /// Clamp to [-1,1] and count every affected input sample.
    Clip,
}
/// Deterministic one-LSB triangular dither keyed to the continuous source grid.
///
/// Algorithm version 1 hashes seed, canonical grid-origin numerator/denominator,
/// signed absolute frame index and channel, then subtracts two 53-bit uniforms.
/// It is reproducible across chunk/worker ordering and is not cryptographic.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TpdfDither {
    /// Caller's reproducible dither seed.
    pub seed: u64,
    /// Explicit sample-grid origin; block starts must lie exactly on this grid.
    pub grid_origin: Time,
}
/// Explicit numeric, packing, gamut and optional dither requirements.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SampleSpec {
    /// Requested scalar representation.
    pub format: SampleFormat,
    /// Physical layout of returned or caller-provided storage.
    pub packing: SamplePacking,
    /// Explicit finite out-of-unit behavior.
    pub gamut: SampleGamut,
    /// Optional deterministic TPDF, only meaningful for integer PCM output.
    pub dither: Option<TpdfDither>,
}
impl SampleSpec {
    /// Interleaved output with strict unit-domain input and no dither.
    pub const fn new(format: SampleFormat) -> Self {
        Self {
            format,
            packing: SamplePacking::Interleaved,
            gamut: SampleGamut::Reject,
            dither: None,
        }
    }
    /// Validate format-specific choices without touching sample storage.
    pub fn validate(self) -> Result<()> {
        if self.format == SampleFormat::F32 && self.dither.is_some() {
            return Err(Error::invalid(
                "TPDF dither requires an explicit integer PCM format",
            ));
        }
        Ok(())
    }
}
/// Observed numerical conversion choices for one or more blocks.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ConversionReport {
    /// Source sample frames processed.
    pub frames: u64,
    /// Interleaved scalar input samples processed.
    pub samples: u64,
    /// Finite out-of-unit source samples explicitly clipped.
    pub gamut_clipped_samples: u64,
    /// Samples rounded into an explicitly requested integer format.
    pub quantized_samples: u64,
    /// Samples to which caller-authorized one-LSB TPDF was added.
    pub dithered_samples: u64,
    /// Rounded codes saturated at an integer endpoint, including defined +1→MAX.
    pub endpoint_saturated_samples: u64,
}
impl ConversionReport {
    /// Combine completed block reports without wrapping counters.
    pub fn checked_add(self, other: Self) -> Result<Self> {
        let add = |a: u64, b: u64| {
            a.checked_add(b)
                .ok_or_else(|| Error::overflow("sample conversion counter"))
        };
        Ok(Self {
            frames: add(self.frames, other.frames)?,
            samples: add(self.samples, other.samples)?,
            gamut_clipped_samples: add(self.gamut_clipped_samples, other.gamut_clipped_samples)?,
            quantized_samples: add(self.quantized_samples, other.quantized_samples)?,
            dithered_samples: add(self.dithered_samples, other.dithered_samples)?,
            endpoint_saturated_samples: add(
                self.endpoint_saturated_samples,
                other.endpoint_saturated_samples,
            )?,
        })
    }
}

trait Guard: fmt::Debug + Send + Sync {}
impl<T: fmt::Debug + Send + Sync> Guard for T {}
#[derive(Clone, Debug)]
struct Aligned<T> {
    data: Arc<Vec<T>>,
    offset: usize,
    len: usize,
}
impl<T> Aligned<T> {
    fn values(&self) -> &[T] {
        &self.data[self.offset..self.offset + self.len]
    }
    fn bytes(&self) -> usize {
        self.data.capacity() * std::mem::size_of::<T>()
    }
}
#[derive(Clone, Debug)]
enum Storage {
    F32(Aligned<f32>),
    I16(Aligned<i16>),
    I32(Aligned<i32>),
}

/// Shared immutable converted samples with source-time provenance.
///
/// Nonempty logical data is 64-byte aligned; empty buffers allocate nothing.
/// Cloning shares the typed allocation and retained guards. Flat typed views do
/// not copy; shape/strides describe physical planar/interleaved packing.
#[derive(Clone, Debug)]
pub struct ConvertedAudio {
    storage: Storage,
    format: SampleFormat,
    packing: SamplePacking,
    rate: u32,
    layout: ChannelLayout,
    frames: usize,
    start: Time,
    source_start: Option<Time>,
    report: ConversionReport,
    retained: Vec<Arc<dyn Guard>>,
}
impl ConvertedAudio {
    /// Actual scalar representation.
    pub fn sample_format(&self) -> SampleFormat {
        self.format
    }
    /// Actual physical packing.
    pub fn packing(&self) -> SamplePacking {
        self.packing
    }
    /// Sample frames, excluding alignment padding.
    pub fn frames(&self) -> usize {
        self.frames
    }
    /// Sample frames per second.
    pub fn sample_rate(&self) -> u32 {
        self.rate
    }
    /// Ordered speaker identities.
    pub fn channel_layout(&self) -> &ChannelLayout {
        &self.layout
    }
    /// Continuous media-time position of the first sample frame.
    pub fn start(&self) -> Time {
        self.start
    }
    /// Original source timestamp when directly established by the producer.
    pub fn source_start(&self) -> Option<Time> {
        self.source_start
    }
    /// Exclusive sample-grid end, with checked rational timing.
    pub fn end(&self) -> Result<Time> {
        self.start.checked_add(Time::from_samples(
            i64::try_from(self.frames).map_err(|_| Error::overflow("converted frame count"))?,
            self.rate,
        )?)
    }
    /// Logical storage shape: `[frames, channels]` or `[channels, frames]`.
    pub fn shape(&self) -> [usize; 2] {
        match self.packing {
            SamplePacking::Interleaved => [self.frames, self.layout.channels()],
            SamplePacking::Planar => [self.layout.channels(), self.frames],
        }
    }
    /// Scalar strides for the declared logical shape.
    pub fn strides(&self) -> [usize; 2] {
        [self.shape()[1], 1]
    }
    /// Aligned owned storage bytes, including vector capacity and alignment tail.
    pub fn allocation_bytes(&self) -> usize {
        match &self.storage {
            Storage::F32(v) => v.bytes(),
            Storage::I16(v) => v.bytes(),
            Storage::I32(v) => v.bytes(),
        }
    }
    /// Checked zero-copy flat f32 view, in the declared packing.
    pub fn as_f32(&self) -> Result<&[f32]> {
        match &self.storage {
            Storage::F32(v) => Ok(v.values()),
            _ => Err(Error::incompatible("converted audio is not f32")),
        }
    }
    /// Checked zero-copy flat i16 view, in the declared packing.
    pub fn as_i16(&self) -> Result<&[i16]> {
        match &self.storage {
            Storage::I16(v) => Ok(v.values()),
            _ => Err(Error::incompatible("converted audio is not i16")),
        }
    }
    /// Checked zero-copy flat i32 view, in the declared packing.
    pub fn as_i32(&self) -> Result<&[i32]> {
        match &self.storage {
            Storage::I32(v) => Ok(v.values()),
            _ => Err(Error::incompatible("converted audio is not i32")),
        }
    }
    /// Actual clipping, quantization, dither and endpoint counters.
    pub fn report(&self) -> ConversionReport {
        self.report
    }
    /// Retain a session/allocation lifetime guard across all buffer clones.
    pub fn retain<T: fmt::Debug + Send + Sync + 'static>(mut self, guard: T) -> Self {
        self.retained.push(Arc::new(guard));
        self
    }
    /// Whether another result shares this exact typed allocation.
    pub fn shares_storage(&self, other: &Self) -> bool {
        match (&self.storage, &other.storage) {
            (Storage::F32(a), Storage::F32(b)) => Arc::ptr_eq(&a.data, &b.data),
            (Storage::I16(a), Storage::I16(b)) => Arc::ptr_eq(&a.data, &b.data),
            (Storage::I32(a), Storage::I32(b)) => Arc::ptr_eq(&a.data, &b.data),
            _ => false,
        }
    }
}

/// Typed exclusive caller buffers, explicitly declaring their physical packing.
/// Every planar slice must have exactly frames entries and every plane must be
/// independently mutable; Rust's mutable borrows provide the exclusivity contract.
pub enum SampleOutput<'a> {
    /// Interleaved f32 scalar storage, length frames * channels.
    InterleavedF32(&'a mut [f32]),
    /// One f32 plane per channel, each exactly frames long.
    PlanarF32(&'a mut [&'a mut [f32]]),
    /// Interleaved i16 scalar storage, length frames * channels.
    InterleavedI16(&'a mut [i16]),
    /// One i16 plane per channel, each exactly frames long.
    PlanarI16(&'a mut [&'a mut [i16]]),
    /// Interleaved i32 scalar storage, length frames * channels.
    InterleavedI32(&'a mut [i32]),
    /// One i32 plane per channel, each exactly frames long.
    PlanarI32(&'a mut [&'a mut [i32]]),
}

/// Upper bound on the aligned typed vector payload needed for conversion.
pub fn allocation_bytes(frames: usize, channels: usize, format: SampleFormat) -> Result<usize> {
    if channels == 0 {
        return Err(Error::invalid(
            "sample allocation requires at least one channel",
        ));
    }
    let len = frames
        .checked_mul(channels)
        .ok_or_else(|| Error::overflow("converted sample count"))?;
    len.checked_add(if len == 0 {
        0
    } else {
        ALIGNMENT / format.bytes() - 1
    })
    .and_then(|n| n.checked_mul(format.bytes()))
    .ok_or_else(|| Error::overflow("aligned converted sample allocation"))
}
/// Explicit packing/numeric materialization, without changing sample timing.
pub fn convert(block: &AudioBlock, spec: SampleSpec) -> Result<ConvertedAudio> {
    let prepared = prepare(block, spec)?;
    let (storage, report) = match spec.format {
        SampleFormat::F32 => {
            let (storage, report) = make::<f32>(block, spec, prepared)?;
            (Storage::F32(storage), report)
        }
        SampleFormat::I16 => {
            let (storage, report) = make::<i16>(block, spec, prepared)?;
            (Storage::I16(storage), report)
        }
        SampleFormat::I32 => {
            let (storage, report) = make::<i32>(block, spec, prepared)?;
            (Storage::I32(storage), report)
        }
    };
    Ok(ConvertedAudio {
        storage,
        format: spec.format,
        packing: spec.packing,
        rate: block.sample_rate(),
        layout: block.channel_layout().clone(),
        frames: block.frames(),
        start: block.start(),
        source_start: block.source_start(),
        report,
        retained: Vec::new(),
    })
}

/// Convert into typed caller-owned storage without an output allocation.
///
/// Shape, packing, numeric domain, nonfinite values, timing, and dither alignment
/// are all validated before the first write. Any returned error leaves every
/// output element unchanged. The successful fill contains no fallible operation.
pub fn process_into(
    block: &AudioBlock,
    spec: SampleSpec,
    output: SampleOutput<'_>,
) -> Result<ConversionReport> {
    spec.validate()?;
    let channels = block.channel_layout().channels();
    let frames = block.frames();
    let samples = block.samples().len();
    let (format, packing, shape) = match &output {
        SampleOutput::InterleavedF32(v) => (
            SampleFormat::F32,
            SamplePacking::Interleaved,
            v.len() == samples,
        ),
        SampleOutput::InterleavedI16(v) => (
            SampleFormat::I16,
            SamplePacking::Interleaved,
            v.len() == samples,
        ),
        SampleOutput::InterleavedI32(v) => (
            SampleFormat::I32,
            SamplePacking::Interleaved,
            v.len() == samples,
        ),
        SampleOutput::PlanarF32(v) => (
            SampleFormat::F32,
            SamplePacking::Planar,
            v.len() == channels && v.iter().all(|p| p.len() == frames),
        ),
        SampleOutput::PlanarI16(v) => (
            SampleFormat::I16,
            SamplePacking::Planar,
            v.len() == channels && v.iter().all(|p| p.len() == frames),
        ),
        SampleOutput::PlanarI32(v) => (
            SampleFormat::I32,
            SamplePacking::Planar,
            v.len() == channels && v.iter().all(|p| p.len() == frames),
        ),
    };
    if format != spec.format || packing != spec.packing || !shape {
        return Err(Error::incompatible(
            "caller sample output has incompatible scalar type, packing, plane count, or frame count",
        ));
    }
    let prepared = prepare(block, spec)?;
    Ok(match output {
        SampleOutput::InterleavedF32(v) => {
            fill::<f32>(block, spec, prepared, |f, c, x| v[f * channels + c] = x)
        }
        SampleOutput::InterleavedI16(v) => {
            fill::<i16>(block, spec, prepared, |f, c, x| v[f * channels + c] = x)
        }
        SampleOutput::InterleavedI32(v) => {
            fill::<i32>(block, spec, prepared, |f, c, x| v[f * channels + c] = x)
        }
        SampleOutput::PlanarF32(v) => fill::<f32>(block, spec, prepared, |f, c, x| v[c][f] = x),
        SampleOutput::PlanarI16(v) => fill::<i16>(block, spec, prepared, |f, c, x| v[c][f] = x),
        SampleOutput::PlanarI32(v) => fill::<i32>(block, spec, prepared, |f, c, x| v[c][f] = x),
    })
}

#[derive(Clone, Copy)]
struct Prepared {
    grid_start: Option<i64>,
    report: ConversionReport,
}
fn prepare(block: &AudioBlock, spec: SampleSpec) -> Result<Prepared> {
    spec.validate()?;
    block.end()?;
    let mut clipped = 0_u64;
    for sample in block.samples() {
        if !sample.is_finite() {
            return Err(Error::invalid(
                "sample conversion rejects nonfinite source values",
            ));
        }
        if !(-1.0..=1.0).contains(sample) {
            if spec.gamut == SampleGamut::Reject {
                return Err(Error::incompatible(
                    "sample conversion exceeds unit range; explicitly choose Clip if acceptable",
                ));
            }
            clipped += 1;
        }
    }
    let grid_start = spec
        .dither
        .map(|dither| {
            let relative = block.start().checked_sub(dither.grid_origin)?;
            let index =
                relative.to_ticks(TimeBase::new(1, block.sample_rate())?, Rounding::TowardZero)?;
            if Time::from_samples(index, block.sample_rate())? != relative {
                return Err(Error::invalid(
                    "dither input start is not on the explicitly declared sample grid",
                ));
            }
            if block.frames() > 0 {
                index
                    .checked_add(
                        i64::try_from(block.frames() - 1)
                            .map_err(|_| Error::overflow("dither frame index"))?,
                    )
                    .ok_or_else(|| Error::overflow("absolute dither sample index"))?;
            }
            Ok(index)
        })
        .transpose()?;
    Ok(Prepared {
        grid_start,
        report: ConversionReport {
            frames: block.frames() as u64,
            samples: block.samples().len() as u64,
            gamut_clipped_samples: clipped,
            quantized_samples: if spec.format == SampleFormat::F32 {
                0
            } else {
                block.samples().len() as u64
            },
            dithered_samples: if spec.dither.is_some() {
                block.samples().len() as u64
            } else {
                0
            },
            endpoint_saturated_samples: 0,
        },
    })
}
trait Scalar: Copy + Default {
    fn encode(value: f64, noise: f64, report: &mut ConversionReport) -> Self;
}
impl Scalar for f32 {
    fn encode(value: f64, _: f64, _: &mut ConversionReport) -> Self {
        value as f32
    }
}
impl Scalar for i16 {
    fn encode(value: f64, noise: f64, report: &mut ConversionReport) -> Self {
        integer(
            value,
            noise,
            32768.0,
            i16::MIN as f64,
            i16::MAX as f64,
            report,
        ) as i16
    }
}
impl Scalar for i32 {
    fn encode(value: f64, noise: f64, report: &mut ConversionReport) -> Self {
        integer(
            value,
            noise,
            2147483648.0,
            i32::MIN as f64,
            i32::MAX as f64,
            report,
        ) as i32
    }
}
fn integer(
    value: f64,
    noise: f64,
    scale: f64,
    minimum: f64,
    maximum: f64,
    report: &mut ConversionReport,
) -> f64 {
    let code = (value * scale + noise).round_ties_even();
    if code < minimum || code > maximum {
        report.endpoint_saturated_samples += 1;
    }
    code.clamp(minimum, maximum)
}
fn fill<T: Scalar>(
    block: &AudioBlock,
    spec: SampleSpec,
    prepared: Prepared,
    mut store: impl FnMut(usize, usize, T),
) -> ConversionReport {
    let mut report = prepared.report;
    let channels = block.channel_layout().channels();
    for (index, &value) in block.samples().iter().enumerate() {
        let (frame, channel) = (index / channels, index % channels);
        let value = if spec.gamut == SampleGamut::Clip {
            value.clamp(-1.0, 1.0)
        } else {
            value
        };
        let noise = if let (Some(dither), Some(start)) = (spec.dither, prepared.grid_start) {
            tpdf(dither, start + frame as i64, channel as u64)
        } else {
            0.0
        };
        store(
            frame,
            channel,
            T::encode(f64::from(value), noise, &mut report),
        );
    }
    report
}
fn make<T: Scalar>(
    block: &AudioBlock,
    spec: SampleSpec,
    prepared: Prepared,
) -> Result<(Aligned<T>, ConversionReport)> {
    let len = block.samples().len();
    let extra = if len == 0 {
        0
    } else {
        ALIGNMENT / std::mem::size_of::<T>() - 1
    };
    let total = len
        .checked_add(extra)
        .ok_or_else(|| Error::overflow("aligned sample count"))?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(total)
        .map_err(|e| Error::exhausted(e.to_string()))?;
    values.resize(total, T::default());
    let offset = if len == 0 {
        0
    } else {
        values.as_ptr().align_offset(ALIGNMENT)
    };
    if offset > extra {
        return Err(Error::incompatible(
            "cannot establish aligned sample storage",
        ));
    }
    let channels = block.channel_layout().channels();
    let frames = block.frames();
    let report = fill::<T>(block, spec, prepared, |frame, channel, value| {
        let index = match spec.packing {
            SamplePacking::Interleaved => frame * channels + channel,
            SamplePacking::Planar => channel * frames + frame,
        };
        values[offset + index] = value;
    });
    Ok((
        Aligned {
            data: Arc::new(values),
            offset,
            len,
        },
        report,
    ))
}
fn tpdf(dither: TpdfDither, index: i64, channel: u64) -> f64 {
    let mut key = mix(dither.seed ^ 0x6175_6469_6f64_6974);
    key = mix(key ^ mix(dither.grid_origin.numerator() as u64));
    key = mix(key ^ mix(u64::from(dither.grid_origin.denominator())));
    key = mix(key ^ mix(index as u64));
    key = mix(key ^ mix(channel));
    let uniform = |value: u64| (value >> 11) as f64 / 9007199254740992.0;
    uniform(mix(key ^ 0x7470_6466_5f75_6e31)) - uniform(mix(key ^ 0x7470_6466_5f75_6e32))
}
fn mix(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}
