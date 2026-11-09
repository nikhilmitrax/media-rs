//! Stateful anti-aliased resampling and bounded overlapping audio windows.

use audioadapter_buffers::direct::InterleavedSlice;
use media_core::{
    error::{Error, Result},
    media::{AudioBlock, ChannelLayout},
    time::{Time, TimeRange},
};
use rubato::{Fft, FixedSync, Indexing, Resampler as RubatoResampler};
use std::collections::VecDeque;

const MAX_WORKING_SAMPLES: usize = 1_048_576;

/// Continuous fixed-rate FFT resampling with explicit startup-delay compensation.
///
/// Output is anchored at origin on the output-rate grid. push retains at most
/// one input chunk; returned blocks are caller-owned. finish drains filter state
/// and trims to the semantic interval [origin, origin + input_frames/input_rate).
/// Changing input chunk boundaries does not reset the filter or sample grid.
pub struct Resampler {
    processor: Fft<f32>,
    input_rate: u32,
    output_rate: u32,
    layout: ChannelLayout,
    origin: Time,
    input: Vec<f32>,
    output: Vec<f32>,
    filled_frames: usize,
    total_input: u64,
    total_output: u64,
    delay: usize,
    delay_remaining: usize,
    finished: bool,
}
impl Resampler {
    /// Create a quality, anti-aliased resampler with a bounded chunk-size hint.
    ///
    /// Internal chunks are rounded to the exact rational-rate alignment. Rate
    /// pairs or chunks requiring over 1,048,576 interleaved working samples are
    /// rejected before constructing the FFT. Backend FFT scratch is additional.
    pub fn new(
        input_rate: u32,
        output_rate: u32,
        layout: ChannelLayout,
        origin: Time,
        chunk_frames: usize,
    ) -> Result<Self> {
        if input_rate == 0 || output_rate == 0 || chunk_frames == 0 || chunk_frames > 65_536 {
            return Err(Error::invalid(
                "resampling requires nonzero rates and a chunk hint in 1..=65536",
            ));
        }
        let divisor = gcd(input_rate, output_rate);
        let input_unit = (input_rate / divisor) as usize;
        let output_unit = (output_rate / divisor) as usize;
        let chunks = chunk_frames.div_ceil(input_unit).max(1);
        let largest = input_unit
            .max(output_unit)
            .checked_mul(chunks)
            .and_then(|n| n.checked_mul(layout.channels()))
            .ok_or_else(|| Error::overflow("resampler working-buffer dimensions"))?;
        if largest > MAX_WORKING_SAMPLES {
            return Err(Error::exhausted(
                "rate ratio and chunk exceed resampler working-buffer limit",
            ));
        }
        let processor = Fft::<f32>::new(
            input_rate as usize,
            output_rate as usize,
            chunk_frames,
            layout.channels(),
            FixedSync::Both,
        )
        .map_err(|e| Error::backend("rubato", e.to_string()).with_cause(e))?;
        let input_len = processor
            .input_frames_max()
            .checked_mul(layout.channels())
            .ok_or_else(|| Error::overflow("resampler input buffer"))?;
        let output_len = processor
            .output_frames_max()
            .checked_mul(layout.channels())
            .ok_or_else(|| Error::overflow("resampler output buffer"))?;
        if input_len.max(output_len) > MAX_WORKING_SAMPLES {
            return Err(Error::exhausted(
                "resampler selected working buffers above the limit",
            ));
        }
        let delay = if input_rate == output_rate {
            0
        } else {
            processor.output_delay()
        };
        Ok(Self {
            processor,
            input_rate,
            output_rate,
            layout,
            origin,
            input: vec![0.0; input_len],
            output: vec![0.0; output_len],
            filled_frames: 0,
            total_input: 0,
            total_output: 0,
            delay,
            delay_remaining: delay,
            finished: false,
        })
    }

    /// Filter startup delay, measured in output-rate frames and removed from results.
    pub fn latency_frames(&self) -> usize {
        self.delay
    }
    /// Input frames retained while waiting for the next complete processing chunk.
    pub fn buffered_input_frames(&self) -> usize {
        self.filled_frames
    }
    /// Maximum input frames in a processing chunk.
    pub fn input_chunk_frames(&self) -> usize {
        self.processor.input_frames_max()
    }
    /// Output sample-grid origin.
    pub fn origin(&self) -> Time {
        self.origin
    }

    /// Feed contiguous input without resetting state between blocks.
    pub fn push(&mut self, block: &AudioBlock) -> Result<Vec<AudioBlock>> {
        if self.finished {
            return Err(Error::invalid("cannot feed a finalized resampler"));
        }
        if block.sample_rate() != self.input_rate || block.channel_layout() != &self.layout {
            return Err(Error::incompatible(
                "resampler input rate or channel identities changed",
            ));
        }
        let expected = sample_time(self.origin, self.total_input, self.input_rate)?;
        if block.start() != expected {
            return Err(Error::incompatible(
                "resampler input is discontinuous; reset or handle gaps explicitly",
            ));
        }
        let total = self
            .total_input
            .checked_add(block.frames() as u64)
            .ok_or_else(|| Error::overflow("continuous input frame count"))?;
        // Preflight end timing so a failed request does not partially advance state.
        sample_time(self.origin, total, self.input_rate)?;
        if self.input_rate == self.output_rate {
            self.total_input = total;
            self.total_output = total;
            return Ok(if block.frames() == 0 {
                Vec::new()
            } else {
                vec![block.clone()]
            });
        }
        let channels = self.layout.channels();
        let mut result = Vec::new();
        let mut consumed = 0;
        while consumed < block.frames() {
            let needed = self.processor.input_frames_next();
            let count = (needed - self.filled_frames).min(block.frames() - consumed);
            let begin = self.filled_frames * channels;
            self.input[begin..begin + count * channels].copy_from_slice(
                &block.samples()[consumed * channels..(consumed + count) * channels],
            );
            self.filled_frames += count;
            self.total_input += count as u64;
            consumed += count;
            if self.filled_frames == needed {
                if let Some(output) = self.process_chunk(None)? {
                    result.push(output);
                }
                self.filled_frames = 0;
            }
        }
        Ok(result)
    }

    /// Drain the computational tail, then trim to the valid semantic output grid.
    /// Repeated finish calls return no additional samples.
    pub fn finish(&mut self) -> Result<Vec<AudioBlock>> {
        if self.finished {
            return Ok(Vec::new());
        }
        let target = self.target_frames()?;
        let mut result = Vec::new();
        if self.filled_frames > 0 {
            let frames = self.filled_frames;
            if let Some(block) = self.process_chunk(Some(frames))? {
                result.push(block);
            }
            self.filled_frames = 0;
        }
        while self.total_output < target {
            let previous = self.total_output;
            let old_delay = self.delay_remaining;
            if let Some(block) = self.process_chunk(Some(0))? {
                result.push(block);
            }
            if self.total_output == previous && self.delay_remaining == old_delay {
                return Err(Error::backend(
                    "rubato",
                    "resampler made no progress while draining",
                ));
            }
        }
        self.finished = true;
        Ok(result)
    }

    fn target_frames(&self) -> Result<u64> {
        let count = u128::from(self.total_input) * u128::from(self.output_rate);
        u64::try_from(count.div_ceil(u128::from(self.input_rate)))
            .map_err(|_| Error::overflow("resampled semantic frame count"))
    }

    fn process_chunk(&mut self, partial: Option<usize>) -> Result<Option<AudioBlock>> {
        let channels = self.layout.channels();
        let input_frames = self.processor.input_frames_next();
        let output_frames = self.output.len() / channels;
        let input = InterleavedSlice::new(&self.input, channels, input_frames)
            .map_err(|e| Error::invalid(e.to_string()))?;
        let mut output = InterleavedSlice::new_mut(&mut self.output, channels, output_frames)
            .map_err(|e| Error::invalid(e.to_string()))?;
        let indexing = partial.map(|frames| Indexing::new().partial_len(frames));
        let (_, written) = self
            .processor
            .process_into_buffer(&input, &mut output, indexing.as_ref())
            .map_err(|e| Error::backend("rubato", e.to_string()).with_cause(e))?;
        let skipped = self.delay_remaining.min(written);
        self.delay_remaining -= skipped;
        let available = written - skipped;
        let remaining = self.target_frames()?.saturating_sub(self.total_output);
        let frames = available.min(usize::try_from(remaining).unwrap_or(usize::MAX));
        if frames == 0 {
            return Ok(None);
        }
        let start = sample_time(self.origin, self.total_output, self.output_rate)?;
        let block = AudioBlock::new(
            self.output[skipped * channels..(skipped + frames) * channels].to_vec(),
            self.output_rate,
            self.layout.clone(),
            start,
        )?;
        self.total_output += frames as u64;
        Ok(Some(block))
    }
}

/// Extract a half-open interval from an already continuously processed block.
///
/// Bounds use ceil on the block's explicit sample grid, so adjacent fractional
/// ranges concatenate without duplicate or omitted samples. Out-of-coverage
/// requests return an error rather than silently padding.
pub fn slice_block(block: &AudioBlock, range: TimeRange) -> Result<AudioBlock> {
    if range.start() < block.start() || range.end() > block.end()? {
        return Err(Error::invalid(
            "audio slice falls outside available coverage",
        ));
    }
    let bounds = range.sample_bounds(block.start(), block.sample_rate())?;
    let start = usize::try_from(bounds.start).map_err(|_| Error::overflow("audio slice start"))?;
    let end = usize::try_from(bounds.end).map_err(|_| Error::overflow("audio slice end"))?;
    if end > block.frames() {
        return Err(Error::invalid("audio sample grid exceeds block"));
    }
    let channels = block.channel_layout().channels();
    let source_start = block
        .source_start()
        .map(|origin| sample_time(origin, start as u64, block.sample_rate()))
        .transpose()?;
    Ok(AudioBlock::new(
        block.samples()[start * channels..end * channels].to_vec(),
        block.sample_rate(),
        block.channel_layout().clone(),
        sample_time(block.start(), start as u64, block.sample_rate())?,
    )?
    .with_source_start(source_start))
}

/// Bounded overlapping windows on a continuous input grid.
///
/// Only complete windows are emitted. finish explicitly discards the trailing
/// incomplete window and exposes its frame count through discarded_frames.
/// No silence is invented, and input discontinuities are errors.
pub struct AudioWindows {
    window: usize,
    hop: usize,
    samples: VecDeque<f32>,
    layout: Option<ChannelLayout>,
    rate: u32,
    origin: Time,
    input_frames: u64,
    buffer_start: u64,
    next_start: u64,
    discarded: usize,
    finished: bool,
}
impl AudioWindows {
    /// Choose window and hop sizes in sample frames.
    pub fn new(window_frames: usize, hop_frames: usize) -> Result<Self> {
        if window_frames == 0 || hop_frames == 0 || window_frames > 65_536 {
            return Err(Error::invalid(
                "audio windows need nonzero window/hop and a window <=65536 frames",
            ));
        }
        Ok(Self {
            window: window_frames,
            hop: hop_frames,
            samples: VecDeque::new(),
            layout: None,
            rate: 0,
            origin: Time::ZERO,
            input_frames: 0,
            buffer_start: 0,
            next_start: 0,
            discarded: 0,
            finished: false,
        })
    }
    /// Feed contiguous input and return the newly complete windows.
    pub fn push(&mut self, block: &AudioBlock) -> Result<Vec<AudioBlock>> {
        if self.finished {
            return Err(Error::invalid("cannot feed finalized audio windows"));
        }
        if let Some(layout) = &self.layout {
            if layout != block.channel_layout()
                || self.rate != block.sample_rate()
                || block.start() != sample_time(self.origin, self.input_frames, self.rate)?
            {
                return Err(Error::incompatible(
                    "window input format or continuous sample grid changed",
                ));
            }
        } else {
            if self
                .window
                .checked_mul(block.channel_layout().channels())
                .is_none_or(|n| n > MAX_WORKING_SAMPLES)
            {
                return Err(Error::exhausted("audio window exceeds sample-buffer limit"));
            }
            self.layout = Some(block.channel_layout().clone());
            self.rate = block.sample_rate();
            self.origin = block.start();
        }
        let channels = block.channel_layout().channels();
        self.input_frames
            .checked_add(block.frames() as u64)
            .ok_or_else(|| Error::overflow("window input count"))?;
        let mut result = Vec::new();
        for frame in block.samples().chunks_exact(channels) {
            if self.input_frames >= self.next_start {
                self.samples.extend(frame);
            }
            self.input_frames += 1;
            if self.samples.len() / channels == self.window {
                result.push(AudioBlock::new(
                    self.samples.iter().copied().collect::<Vec<_>>(),
                    self.rate,
                    block.channel_layout().clone(),
                    sample_time(self.origin, self.next_start, self.rate)?,
                )?);
                self.next_start = self
                    .next_start
                    .checked_add(self.hop as u64)
                    .ok_or_else(|| Error::overflow("window start"))?;
                let remove = self
                    .next_start
                    .saturating_sub(self.buffer_start)
                    .min(self.samples.len() as u64 / channels as u64);
                self.samples.drain(..remove as usize * channels);
                self.buffer_start = self.next_start.min(self.input_frames);
            } else if self.input_frames < self.next_start {
                self.buffer_start = self.input_frames;
            }
        }
        Ok(result)
    }
    /// Finalize without padding the last incomplete window.
    pub fn finish(&mut self) -> Result<Vec<AudioBlock>> {
        if !self.finished {
            self.discarded = self
                .layout
                .as_ref()
                .map_or(0, |layout| self.samples.len() / layout.channels());
            self.samples.clear();
            self.finished = true;
        }
        Ok(Vec::new())
    }
    /// Frames retained for an incomplete final window and discarded at finish.
    pub fn discarded_frames(&self) -> usize {
        self.discarded
    }
    /// Current retained interleaved samples, useful for budget diagnostics.
    pub fn buffered_samples(&self) -> usize {
        self.samples.len()
    }
}

fn sample_time(origin: Time, frame: u64, rate: u32) -> Result<Time> {
    origin.checked_add(Time::from_samples(
        i64::try_from(frame).map_err(|_| Error::overflow("audio sample time"))?,
        rate,
    )?)
}
fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let remainder = a % b;
        a = b;
        b = remainder;
    }
    a
}
