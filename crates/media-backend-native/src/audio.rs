//! Streaming native audio decoding, explicit channel mixing, and WAV writing.
//!
//! Decoding is packet-oriented and does not materialize the whole recording.
//! Output blocks have bounded frame counts; demuxer/decoder internal allocations
//! remain backend-managed and are not included in the owned sample-buffer bound.

use std::io::{self, Read, Seek, SeekFrom, Write};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, Ordering},
};

use crate::audio_samples::{ConvertedAudio, SampleFormat, SamplePacking};
use media_core::{
    error::{Error, ErrorKind, Result},
    media::{AudioBlock, ChannelLayout, ChannelPosition},
    time::Time,
};
use media_io::{
    sink::{FinishedSink, Sink},
    source::{ReadSeek, Source},
};
use symphonia::core::{
    audio::{ChannelLabel, Channels, Position},
    codecs::audio::{AudioCodecId, AudioDecoder, AudioDecoderOptions},
    common::Limit,
    errors::Error as DecodeError,
    formats::{FormatOptions, FormatReader, TrackType, probe::Hint},
    io::{MediaSource, MediaSourceStream},
    meta::MetadataOptions,
    packet::Packet,
};

/// Behavior when a compressed packet is corrupt.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum CorruptAudioPolicy {
    /// Stop with an actionable error.
    #[default]
    Fail,
    /// Skip a corrupt decoded packet and record it in the report.
    Skip,
}

/// Interpretation of container timestamp quantization relative to decoded frames.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum AudioTimestampPolicy {
    /// Every packet timestamp must equal the established continuous sample grid.
    Exact,
    /// Permit at most one container time-base tick of quantization, report every
    /// adjustment, and retain the unadjusted timestamp on each decoded block.
    /// Larger gaps/overlaps are errors; this never fills missing source samples.
    #[default]
    AllowOneTick,
}

/// Decode limits and logical audio-track selection.
#[derive(Clone, Debug)]
pub struct AudioDecodeOptions {
    /// Explicit logical track; default is the demuxer's default audio track.
    pub track: Option<u32>,
    /// Maximum bytes read while probing headers, including read-ahead.
    pub max_probe_bytes: u64,
    /// Maximum compressed packet bytes accepted from the demuxer.
    pub max_packet_bytes: usize,
    /// Maximum interleaved samples copied from a decoded packet.
    pub max_buffered_samples: usize,
    /// Maximum frames returned by one next-block call.
    pub max_block_frames: usize,
    /// Verify decoded content when supported by the codec.
    pub verify: bool,
    /// Explicit corrupt-packet policy.
    pub corrupt: CorruptAudioPolicy,
    /// Container timestamp-quantization policy.
    pub timestamps: AudioTimestampPolicy,
}
impl Default for AudioDecodeOptions {
    fn default() -> Self {
        Self {
            track: None,
            max_probe_bytes: 1024 * 1024,
            max_packet_bytes: 8 * 1024 * 1024,
            max_buffered_samples: 1024 * 1024,
            max_block_frames: 4096,
            verify: true,
            corrupt: CorruptAudioPolicy::Fail,
            timestamps: AudioTimestampPolicy::AllowOneTick,
        }
    }
}

/// Established native audio-track metadata.
#[derive(Clone, Debug)]
pub struct AudioInfo {
    /// Logical track identity.
    pub track: u32,
    /// Frames per second.
    pub sample_rate: u32,
    /// Ordered speaker identities.
    pub layout: ChannelLayout,
    /// Declared playable frame count, if supplied by the demuxer.
    pub frames: Option<u64>,
    /// Original first-frame timing.
    pub start: Time,
    /// Container timestamp resolution, distinct from one decoded sample period.
    pub timestamp_quantum: Time,
}

/// Decode counters; skipped packets remain visible.
#[derive(Clone, Debug, Default)]
pub struct AudioDecodeReport {
    /// Successfully decoded compressed packets.
    pub decoded_packets: u64,
    /// Corrupt packets skipped under an explicit policy.
    pub skipped_packets: u64,
    /// Packet timestamps adjusted under the explicit processing-clock policy.
    pub timestamp_adjustments: u64,
    /// Largest absolute timestamp adjustment observed.
    pub max_timestamp_adjustment: Option<Time>,
    /// Gaps preserved after explicitly skipped corrupt packets.
    pub discontinuities: u64,
    /// Sample frames returned to callers.
    pub output_frames: u64,
    /// Bytes read during probing.
    pub probe_bytes: u64,
    /// Codec verification status after successful EOF.
    pub verified: Option<bool>,
    /// Whether normal EOF and codec finalization were reached.
    pub finished: bool,
}

/// A blocking packet decoder with independent source state and bounded outputs.
pub struct AudioReader {
    format: Box<dyn FormatReader>,
    decoder: Box<dyn AudioDecoder>,
    info: AudioInfo,
    options: AudioDecodeOptions,
    label: String,
    time_base: symphonia::core::units::TimeBase,
    report: AudioDecodeReport,
    pending: Option<(AudioBlock, usize)>,
    next_time: Time,
    skipped_since_output: bool,
    first_packet: Option<Packet>,
    failed: bool,
}
impl AudioReader {
    /// Probe headers and one selected packet, then open an independent decoder.
    /// Physical probe reads stay under max_probe_bytes; no audio is decoded here.
    pub fn open(source: &Source, options: AudioDecodeOptions) -> Result<Self> {
        if options.max_probe_bytes == 0
            || options.max_packet_bytes == 0
            || options.max_buffered_samples == 0
            || options.max_block_frames == 0
        {
            return Err(Error::invalid("audio decode limits must be nonzero"));
        }
        let caps = source.capabilities()?;
        let reader = if caps.seekable {
            InputReader::Seekable(source.open()?)
        } else {
            InputReader::Sequential(source.open_reader()?)
        };
        let remaining = Arc::new(AtomicU64::new(options.max_probe_bytes));
        let input = Input {
            reader: Mutex::new(reader),
            length: caps.known_length,
            seekable: caps.seekable,
            probe_remaining: remaining.clone(),
        };
        let stream = MediaSourceStream::new(Box::new(input), Default::default());
        let mut hint = Hint::new();
        if let Some(extension) = source
            .path()
            .and_then(|p| p.extension())
            .and_then(|s| s.to_str())
        {
            hint.with_extension(extension);
        }
        let mut format = symphonia::default::get_probe()
            .probe(
                &hint,
                stream,
                FormatOptions::default(),
                MetadataOptions::default()
                    .limit_tag_bytes(Limit::Maximum(64 * 1024))
                    .limit_visual_bytes(Limit::Maximum(1024 * 1024)),
            )
            .map_err(|e| backend_error(e, source.label(), "probe"))?;
        let track = if let Some(id) = options.track {
            format.tracks().iter().find(|t| t.id == id)
        } else {
            format.default_track(TrackType::Audio)
        }
        .ok_or_else(|| {
            Error::unsupported("requested audio track was not found").with_source(source.label())
        })?;
        let params = track
            .codec_params
            .as_ref()
            .and_then(|p| p.audio())
            .ok_or_else(|| {
                Error::unsupported("selected track has no supported audio codec parameters")
            })?;
        let decoder = symphonia::default::get_codecs()
            .make_audio_decoder(
                params,
                &AudioDecoderOptions::default().verify(options.verify),
            )
            .map_err(|e| backend_error(e, source.label(), "create audio decoder"))?;
        let params = decoder.codec_params();
        let rate = params
            .sample_rate
            .filter(|r| *r > 0)
            .ok_or_else(|| Error::unsupported("audio sample rate is unknown at probe time"))?;
        let layout = codec_channel_layout(
            params.codec,
            params
                .channels
                .as_ref()
                .ok_or_else(|| Error::unsupported("audio channel identities are unknown"))?,
        )?;
        if layout.channels() > options.max_buffered_samples {
            return Err(Error::exhausted(
                "audio channel count exceeds sample-buffer budget",
            ));
        }
        let time_base = track.time_base.unwrap_or_else(|| {
            symphonia::core::units::TimeBase::try_from_recip(rate).expect("rate validated")
        });
        let track_id = track.id;
        let frame_count = track.num_frames;
        let declared_start = track.start_ts.get();
        let declared_delay = u64::from(track.delay.unwrap_or(0));
        // Some demuxers cannot establish stream offsets in their track header.
        // Discover one selected packet under the same physical-read probe budget;
        // retain it for decoding rather than reopening or scanning the recording.
        let first_packet = loop {
            let Some(packet) = format
                .next_packet()
                .map_err(|e| backend_error(e, source.label(), "probe first audio packet"))?
            else {
                break None;
            };
            if packet.track_id == track_id {
                if packet.data.len() > options.max_packet_bytes {
                    return Err(Error::exhausted(
                        "first audio packet exceeds configured byte limit",
                    ));
                }
                break Some(packet);
            }
        };
        let start = convert_timestamp(
            first_packet
                .as_ref()
                .map_or(declared_start, |packet| packet.pts.get()),
            time_base,
        )?
        .checked_add(Time::from_samples(
            i64::try_from(first_packet.as_ref().map_or(0, |packet| {
                let trim = packet.trim_start.get();
                if trim > 0 {
                    trim.max(declared_delay)
                } else {
                    0
                }
            }))
            .map_err(|_| Error::overflow("initial audio trim timestamp"))?,
            rate,
        )?)?;
        let probe_bytes = options
            .max_probe_bytes
            .saturating_sub(remaining.load(Ordering::Relaxed));
        remaining.store(u64::MAX, Ordering::Relaxed);
        let info = AudioInfo {
            track: track_id,
            sample_rate: rate,
            layout,
            frames: frame_count,
            start,
            timestamp_quantum: Time::new(i64::from(time_base.numer.get()), time_base.denom.get())?,
        };
        Ok(Self {
            format,
            decoder,
            info,
            options,
            label: source.label(),
            time_base,
            report: AudioDecodeReport {
                probe_bytes,
                ..Default::default()
            },
            pending: None,
            next_time: start,
            skipped_since_output: false,
            first_packet,
            failed: false,
        })
    }

    /// Metadata established without a complete decode.
    pub fn info(&self) -> &AudioInfo {
        &self.info
    }
    /// Audio track identifiers in original demuxer order, without decoding.
    ///
    /// IDs are backend/container identities, not another backend's stream index.
    /// Unsupported tracks without audio parameters are omitted; callers mapping
    /// independent demuxers must verify both inventory and packet/timing identity.
    pub fn audio_track_ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.format.tracks().iter().filter_map(|track| {
            track
                .codec_params
                .as_ref()
                .and_then(|params| params.audio())
                .map(|_| track.id)
        })
    }
    /// Selected first compressed packet retained during the bounded probe.
    ///
    /// Available only until decoding consumes that packet. This borrowed view
    /// permits exact cross-demuxer identity checks without copying its bytes.
    /// info().start includes declared priming trim and can differ from the
    /// compressed packet's original timestamp. Differing packetization/trim
    /// mappings must be validated before aligning independent decoder timelines.
    pub fn first_packet_bytes(&self) -> Option<&[u8]> {
        self.first_packet
            .as_ref()
            .map(|packet| packet.data.as_ref())
    }
    /// Probed container family, independently of a path or extension hint.
    pub fn format_short_name(&self) -> &'static str {
        self.format.format_info().short_name
    }
    /// Registered codec identity for exact cross-demuxer compatibility checks.
    pub fn codec_short_name(&self) -> Option<&'static str> {
        symphonia::default::get_codecs()
            .get_audio_decoder(self.decoder.codec_params().codec)
            .map(|registered| registered.codec.info.short_name)
    }
    /// Whether selected track/first packet declares codec delay or padding.
    /// Cross-backend timeline adapters must validate such trim mappings explicitly.
    pub fn has_declared_trimming(&self) -> bool {
        self.format
            .tracks()
            .iter()
            .find(|track| track.id == self.info.track)
            .is_some_and(|track| track.delay.unwrap_or(0) > 0 || track.padding.unwrap_or(0) > 0)
            || self
                .first_packet
                .as_ref()
                .is_some_and(|packet| packet.trim_start.get() > 0 || packet.trim_end.get() > 0)
    }
    /// Counters and finalization state for this execution.
    pub fn report(&self) -> &AudioDecodeReport {
        &self.report
    }

    /// Decode the next bounded block. Only normal demuxer EOF returns None.
    ///
    /// Format changes are reported as errors before differently formatted output.
    /// Symphonia audio decoders return each packet's valid, gapless-trimmed
    /// output directly; finalize checks verification rather than synthesizing tail.
    pub fn next_block(&mut self) -> Result<Option<AudioBlock>> {
        if self.failed {
            return Err(Error::invalid("audio reader is in a failed state"));
        }
        let result = self.next_inner();
        if result.is_err() {
            self.failed = true;
        }
        result
    }

    fn next_inner(&mut self) -> Result<Option<AudioBlock>> {
        if let Some((block, offset)) = self.pending.take() {
            return self.take_pending(block, offset).map(Some);
        }
        if self.report.finished {
            return Ok(None);
        }
        loop {
            let packet = if let Some(packet) = self.first_packet.take() {
                Some(packet)
            } else {
                self.format
                    .next_packet()
                    .map_err(|e| backend_error(e, self.label.clone(), "demux audio packet"))?
            };
            let Some(packet) = packet else {
                let finalization = self.decoder.finalize();
                self.report.verified = finalization.verify_ok;
                if finalization.verify_ok == Some(false) {
                    return Err(Error::backend("symphonia", "audio verification failed")
                        .with_source(self.label.clone()));
                }
                if self.report.skipped_packets == 0
                    && self
                        .info
                        .frames
                        .is_some_and(|n| self.report.output_frames < n)
                {
                    return Err(Error::backend(
                        "symphonia",
                        "audio ended before its declared playable frame count",
                    )
                    .with_source(self.label.clone()));
                }
                self.report.finished = true;
                return Ok(None);
            };
            if packet.track_id != self.info.track {
                continue;
            }
            if packet.data.len() > self.options.max_packet_bytes {
                return Err(Error::exhausted(
                    "compressed audio packet exceeds configured byte limit",
                )
                .with_source(self.label.clone())
                .with_track(self.info.track));
            }
            let decoded = match self.decoder.decode(&packet) {
                Ok(buffer) => buffer,
                Err(error @ (DecodeError::DecodeError(_) | DecodeError::IoError(_)))
                    if self.options.corrupt == CorruptAudioPolicy::Skip =>
                {
                    self.report.skipped_packets += 1;
                    self.skipped_since_output = true;
                    // Only decoded-packet failures may be skipped. Demux errors remain fatal.
                    let _ = error;
                    continue;
                }
                Err(error) => {
                    return Err(backend_error(error, self.label.clone(), "decode audio")
                        .with_track(self.info.track));
                }
            };
            self.report.decoded_packets += 1;
            // Fully trimmed priming packets can return an empty placeholder
            // buffer with no negotiated output format. Validate actual samples.
            if decoded.is_empty() {
                continue;
            }
            let layout = channel_layout(decoded.spec().channels())?;
            if decoded.spec().rate() != self.info.sample_rate || layout != self.info.layout {
                return Err(Error::incompatible(
                    format!("audio format changed: probed {} Hz {:?}, decoded {} Hz {:?}; explicit renegotiation is required",self.info.sample_rate,self.info.layout,decoded.spec().rate(),layout),
                )
                .with_source(self.label.clone())
                .with_track(self.info.track));
            }
            if decoded.samples_interleaved() > self.options.max_buffered_samples {
                return Err(Error::exhausted(
                    "decoded packet exceeds configured owned sample-buffer limit",
                ));
            }
            let mut samples = vec![0.0_f32; decoded.samples_interleaved()];
            decoded.copy_to_slice_interleaved(&mut samples);
            // Symphonia already removed encoder-delay samples from the buffer.
            // Its packet PTS still describes the untrimmed block: advance only
            // the timestamp, without removing those samples a second time.
            let source_start = convert_timestamp(packet.pts.get(), self.time_base)?.checked_add(
                Time::from_samples(
                    i64::try_from(packet.trim_start.get())
                        .map_err(|_| Error::overflow("audio trim timestamp"))?,
                    self.info.sample_rate,
                )?,
            )?;
            let delta = source_start.checked_sub(self.next_time)?;
            let tolerance = if self.options.timestamps == AudioTimestampPolicy::AllowOneTick {
                self.info.timestamp_quantum
            } else {
                Time::ZERO
            };
            let within_bin = delta <= tolerance && delta >= Time::ZERO.checked_sub(tolerance)?;
            let start = if within_bin {
                if delta != Time::ZERO {
                    self.report.timestamp_adjustments += 1;
                    let adjustment = if delta < Time::ZERO {
                        Time::ZERO.checked_sub(delta)?
                    } else {
                        delta
                    };
                    self.report.max_timestamp_adjustment = Some(
                        self.report
                            .max_timestamp_adjustment
                            .map_or(adjustment, |old| old.max(adjustment)),
                    );
                }
                self.next_time
            } else if self.skipped_since_output {
                self.report.discontinuities += 1;
                source_start
            } else {
                return Err(Error::new(ErrorKind::MissingData,"audio packet timing has a gap/overlap exceeding the permitted timestamp quantization")
                    .with_source(self.label.clone()).with_track(self.info.track).with_time(source_start));
            };
            self.skipped_since_output = false;
            let block = AudioBlock::new(
                samples,
                self.info.sample_rate,
                self.info.layout.clone(),
                start,
            )?
            .with_source_start(Some(source_start));
            self.next_time = block.end()?;
            return self.take_pending(block, 0).map(Some);
        }
    }

    fn take_pending(&mut self, block: AudioBlock, offset: usize) -> Result<AudioBlock> {
        let frames = (block.frames() - offset).min(self.options.max_block_frames);
        let channels = block.channel_layout().channels();
        let start = block.start().checked_add(Time::from_samples(
            i64::try_from(offset).map_err(|_| Error::overflow("audio frame offset"))?,
            block.sample_rate(),
        )?)?;
        let result = if offset == 0 && frames == block.frames() {
            block.clone()
        } else {
            AudioBlock::new(
                block.samples()[offset * channels..(offset + frames) * channels].to_vec(),
                block.sample_rate(),
                block.channel_layout().clone(),
                start,
            )?
            .with_source_start(
                block
                    .source_start()
                    .map(|source| {
                        source.checked_add(Time::from_samples(offset as i64, block.sample_rate())?)
                    })
                    .transpose()?,
            )
        };
        if offset + frames < block.frames() {
            self.pending = Some((block, offset + frames));
        }
        self.report.output_frames = self
            .report
            .output_frames
            .checked_add(frames as u64)
            .ok_or_else(|| Error::overflow("audio output count"))?;
        Ok(result)
    }
}

fn convert_timestamp(timestamp: i64, base: symphonia::core::units::TimeBase) -> Result<Time> {
    Time::new(timestamp, base.denom.get())?.checked_mul_ratio(i64::from(base.numer.get()), 1)
}
fn backend_error(error: DecodeError, source: String, operation: &str) -> Error {
    Error::backend("symphonia", error.to_string())
        .with_cause(error)
        .with_source(source)
        .with_operation(operation)
}

enum InputReader {
    Seekable(Box<dyn ReadSeek + Send>),
    Sequential(Box<dyn Read + Send>),
}
struct Input {
    reader: Mutex<InputReader>,
    length: Option<u64>,
    seekable: bool,
    probe_remaining: Arc<AtomicU64>,
}
impl Read for Input {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        if bytes.is_empty() {
            return Ok(0);
        }
        let remaining = self.probe_remaining.load(Ordering::Relaxed);
        if remaining == 0 {
            return Err(io::Error::other("audio probe exceeded its read budget"));
        }
        let count = bytes
            .len()
            .min(usize::try_from(remaining).unwrap_or(usize::MAX));
        let reader = self
            .reader
            .get_mut()
            .map_err(|_| io::Error::other("audio input lock poisoned"))?;
        let read = match reader {
            InputReader::Seekable(reader) => reader.read(&mut bytes[..count]),
            InputReader::Sequential(reader) => reader.read(&mut bytes[..count]),
        }?;
        if remaining != u64::MAX {
            self.probe_remaining
                .fetch_sub(read as u64, Ordering::Relaxed);
        }
        Ok(read)
    }
}
impl Seek for Input {
    fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
        match self
            .reader
            .get_mut()
            .map_err(|_| io::Error::other("audio input lock poisoned"))?
        {
            InputReader::Seekable(reader) => reader.seek(position),
            InputReader::Sequential(_) => Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "sequential audio source",
            )),
        }
    }
}
impl MediaSource for Input {
    fn is_seekable(&self) -> bool {
        self.seekable
    }
    fn byte_len(&self) -> Option<u64> {
        self.length
    }
}

/// Map decoder channel identities without guessing speaker meaning from count.
fn channel_layout(channels: &Channels) -> Result<ChannelLayout> {
    let positions = match channels {
        Channels::Positioned(mask) => {
            let mono = mask.bits().count_ones() == 1 && *mask == Position::FRONT_CENTER;
            mask.iter()
                .map(|p| position(p, mono))
                .collect::<Result<Vec<_>>>()?
        }
        Channels::Discrete(count) => (0..*count).map(ChannelPosition::Discrete).collect(),
        Channels::Custom(labels) => labels
            .iter()
            .map(|label| match label {
                ChannelLabel::Positioned(p) => position(*p, false),
                ChannelLabel::Discrete(index) => Ok(ChannelPosition::Discrete(*index)),
                _ => Err(Error::unsupported(
                    "ambisonic channel identities require a dedicated adapter",
                )),
            })
            .collect::<Result<Vec<_>>>()?,
        _ => {
            return Err(Error::unsupported(
                "unknown or ambisonic audio channel layout",
            ));
        }
    };
    ChannelLayout::new(positions)
}

fn codec_channel_layout(codec: AudioCodecId, channels: &Channels) -> Result<ChannelLayout> {
    use symphonia::core::codecs::audio::well_known::{CODEC_ID_MP1, CODEC_ID_MP2, CODEC_ID_MP3};
    let layout = channel_layout(channels)?;
    // MPEG audio's one-channel mode is mono. Symphonia's demux metadata can
    // label it FrontLeft while the decoder correctly labels the same mode Mono.
    if matches!(codec, CODEC_ID_MP1 | CODEC_ID_MP2 | CODEC_ID_MP3)
        && layout.positions() == [ChannelPosition::FrontLeft]
    {
        Ok(ChannelLayout::mono())
    } else {
        Ok(layout)
    }
}
fn position(p: Position, mono: bool) -> Result<ChannelPosition> {
    Ok(match p {
        Position::FRONT_LEFT => ChannelPosition::FrontLeft,
        Position::FRONT_RIGHT => ChannelPosition::FrontRight,
        Position::FRONT_CENTER if mono => ChannelPosition::Mono,
        Position::FRONT_CENTER => ChannelPosition::FrontCenter,
        Position::LFE1 => ChannelPosition::Lfe,
        Position::REAR_LEFT => ChannelPosition::BackLeft,
        Position::REAR_RIGHT => ChannelPosition::BackRight,
        Position::SIDE_LEFT => ChannelPosition::SideLeft,
        Position::SIDE_RIGHT => ChannelPosition::SideRight,
        _ => {
            return Err(Error::unsupported(
                "speaker position cannot be represented by the current media contract",
            ));
        }
    })
}

/// Explicit row-major channel-mix coefficients; output retains float headroom.
#[derive(Clone, Debug)]
pub struct MixMatrix {
    input: ChannelLayout,
    output: ChannelLayout,
    coefficients: Vec<f32>,
}
impl MixMatrix {
    /// Define one coefficient for each output-channel/input-channel pair.
    pub fn new(
        input: ChannelLayout,
        output: ChannelLayout,
        coefficients: Vec<f32>,
    ) -> Result<Self> {
        if coefficients.len() != input.channels() * output.channels()
            || coefficients.iter().any(|c| !c.is_finite())
        {
            return Err(Error::invalid(
                "mix matrix needs finite row-major output × input coefficients",
            ));
        }
        Ok(Self {
            input,
            output,
            coefficients,
        })
    }
    /// Mono mix: average stereo fronts, -3 dB center/surround contribution, omit LFE.
    ///
    /// Single named non-LFE channels retain unity gain. Discrete channels need an
    /// explicit matrix. No implicit normalization, clipping, or limiter is applied.
    pub fn mono(input: ChannelLayout) -> Result<Self> {
        let coefficients = if input.channels() == 1
            && !matches!(
                input.positions()[0],
                ChannelPosition::Lfe | ChannelPosition::Discrete(_)
            ) {
            vec![1.0]
        } else {
            let stereo = Self::stereo(input.clone())?;
            (0..input.channels())
                .map(|i| (stereo.coefficients[i] + stereo.coefficients[input.channels() + i]) * 0.5)
                .collect()
        };
        Self::new(input, ChannelLayout::mono(), coefficients)
    }
    /// Stereo mix using named positions; surrounds/center use -3 dB, LFE is omitted.
    pub fn stereo(input: ChannelLayout) -> Result<Self> {
        let mut coefficients = vec![0.0; input.channels() * 2];
        for (index, position) in input.positions().iter().enumerate() {
            let (left, right) = match position {
                ChannelPosition::Mono => (1.0, 1.0),
                ChannelPosition::FrontLeft => (1.0, 0.0),
                ChannelPosition::FrontRight => (0.0, 1.0),
                ChannelPosition::FrontCenter if input.channels() == 1 => (1.0, 1.0),
                ChannelPosition::FrontCenter => (
                    std::f32::consts::FRAC_1_SQRT_2,
                    std::f32::consts::FRAC_1_SQRT_2,
                ),
                ChannelPosition::BackLeft | ChannelPosition::SideLeft => {
                    (std::f32::consts::FRAC_1_SQRT_2, 0.0)
                }
                ChannelPosition::BackRight | ChannelPosition::SideRight => {
                    (0.0, std::f32::consts::FRAC_1_SQRT_2)
                }
                ChannelPosition::Lfe => (0.0, 0.0),
                ChannelPosition::Discrete(_) => {
                    return Err(Error::unsupported(
                        "discrete channels require an explicit mix matrix",
                    ));
                }
            };
            coefficients[index] = left;
            coefficients[input.channels() + index] = right;
        }
        Self::new(input, ChannelLayout::stereo(), coefficients)
    }
    /// The explicit input speaker order.
    pub fn input_layout(&self) -> &ChannelLayout {
        &self.input
    }
    /// Output speaker order.
    pub fn output_layout(&self) -> &ChannelLayout {
        &self.output
    }
    /// Row-major coefficients.
    pub fn coefficients(&self) -> &[f32] {
        &self.coefficients
    }
    /// Apply the matrix while preserving rate and first-sample timing.
    pub fn process(&self, block: &AudioBlock) -> Result<AudioBlock> {
        if block.channel_layout() != &self.input {
            return Err(Error::incompatible("mix input channel identities differ"));
        }
        let count = block
            .frames()
            .checked_mul(self.output.channels())
            .ok_or_else(|| Error::overflow("mixed sample count"))?;
        let mut samples = vec![0.0; count];
        for (frame, output) in block
            .samples()
            .chunks_exact(self.input.channels())
            .zip(samples.chunks_exact_mut(self.output.channels()))
        {
            for (channel, value) in output.iter_mut().enumerate() {
                *value = frame
                    .iter()
                    .zip(
                        &self.coefficients[channel * self.input.channels()
                            ..(channel + 1) * self.input.channels()],
                    )
                    .map(|(sample, gain)| sample * gain)
                    .sum();
            }
        }
        AudioBlock::new(
            samples,
            block.sample_rate(),
            self.output.clone(),
            block.start(),
        )
        .map(|output| output.with_source_start(block.source_start()))
    }
}

/// Streaming IEEE-float or explicitly converted PCM WAV for named mono/stereo.
pub struct WavWriter {
    sink: Sink,
    rate: u32,
    layout: ChannelLayout,
    frames: u64,
    next: Option<Time>,
    format: SampleFormat,
}
impl WavWriter {
    /// Start RIFF/WAVE output. Header completion happens only in finish.
    pub fn new(sink: Sink, sample_rate: u32, layout: ChannelLayout) -> Result<Self> {
        Self::new_with_format(sink, sample_rate, layout, SampleFormat::F32)
    }
    /// Start explicit f32/PCM16/PCM32 RIFF output, without implicit quantization.
    /// Integer writers accept only matching ConvertedAudio via write_samples.
    pub fn new_with_format(
        mut sink: Sink,
        sample_rate: u32,
        layout: ChannelLayout,
        format: SampleFormat,
    ) -> Result<Self> {
        if sample_rate == 0
            || !(layout == ChannelLayout::mono() || layout == ChannelLayout::stereo())
        {
            return Err(Error::unsupported(
                "WAV output requires nonzero rate and named mono/stereo layout",
            ));
        }
        let channels = layout.channels() as u16;
        let byte_rate = sample_rate
            .checked_mul(u32::from(channels) * format.bytes() as u32)
            .ok_or_else(|| Error::overflow("WAV byte rate"))?;
        sink.write_all(b"RIFF\0\0\0\0WAVEfmt ").map_err(Error::io)?;
        sink.write_all(&40_u32.to_le_bytes()).map_err(Error::io)?;
        sink.write_all(&0xfffe_u16.to_le_bytes())
            .map_err(Error::io)?;
        sink.write_all(&channels.to_le_bytes()).map_err(Error::io)?;
        sink.write_all(&sample_rate.to_le_bytes())
            .map_err(Error::io)?;
        sink.write_all(&byte_rate.to_le_bytes())
            .map_err(Error::io)?;
        sink.write_all(&(channels * format.bytes() as u16).to_le_bytes())
            .map_err(Error::io)?;
        sink.write_all(&(format.bytes() as u16 * 8).to_le_bytes())
            .map_err(Error::io)?;
        sink.write_all(&22_u16.to_le_bytes()).map_err(Error::io)?;
        sink.write_all(&(format.bytes() as u16 * 8).to_le_bytes())
            .map_err(Error::io)?;
        let channel_mask = if channels == 1 { 4_u32 } else { 3_u32 };
        sink.write_all(&channel_mask.to_le_bytes())
            .map_err(Error::io)?;
        sink.write_all(&[
            if format == SampleFormat::F32 { 3 } else { 1 },
            0,
            0,
            0,
            0,
            0,
            16,
            0,
            128,
            0,
            0,
            170,
            0,
            56,
            155,
            113,
        ])
        .map_err(Error::io)?;
        sink.write_all(b"fact").map_err(Error::io)?;
        sink.write_all(&4_u32.to_le_bytes()).map_err(Error::io)?;
        sink.write_all(&0_u32.to_le_bytes()).map_err(Error::io)?;
        sink.write_all(b"data\0\0\0\0").map_err(Error::io)?;
        Ok(Self {
            sink,
            rate: sample_rate,
            layout,
            frames: 0,
            next: None,
            format,
        })
    }
    /// Append contiguous audio, preserving f32 values without implicit clipping.
    pub fn write(&mut self, block: &AudioBlock) -> Result<()> {
        if self.format != SampleFormat::F32 {
            return Err(Error::incompatible(
                "integer WAV output requires explicit sample conversion before writing",
            ));
        }
        let frames = self.validate_append(
            block.sample_rate(),
            block.channel_layout(),
            block.start(),
            block.frames(),
        )?;
        let next = block.end()?;
        self.write_typed(block.samples())?;
        self.frames = frames;
        self.next = Some(next);
        Ok(())
    }
    /// Append matching, physically interleaved converted samples.
    /// Shape/rate/layout/timing/RIFF bounds are checked before writes. An I/O
    /// failure poisons the underlying sink, so finish cannot report success.
    pub fn write_samples(&mut self, block: &ConvertedAudio) -> Result<()> {
        if block.sample_format() != self.format || block.packing() != SamplePacking::Interleaved {
            return Err(Error::incompatible(
                "WAV sample type/packing differs from its explicit interleaved output format",
            ));
        }
        let frames = self.validate_append(
            block.sample_rate(),
            block.channel_layout(),
            block.start(),
            block.frames(),
        )?;
        let next = block.end()?;
        match self.format {
            SampleFormat::F32 => self.write_typed(block.as_f32()?)?,
            SampleFormat::I16 => self.write_typed(block.as_i16()?)?,
            SampleFormat::I32 => self.write_typed(block.as_i32()?)?,
        }
        self.frames = frames;
        self.next = Some(next);
        Ok(())
    }
    fn validate_append(
        &self,
        rate: u32,
        layout: &ChannelLayout,
        start: Time,
        count: usize,
    ) -> Result<u64> {
        if rate != self.rate || layout != &self.layout {
            return Err(Error::incompatible(
                "WAV input rate or speaker layout changed",
            ));
        }
        if self.next.is_some_and(|time| time != start) {
            return Err(Error::incompatible(
                "WAV input is discontinuous; choose a gap policy explicitly",
            ));
        }
        let frames = self
            .frames
            .checked_add(count as u64)
            .ok_or_else(|| Error::overflow("WAV frame count"))?;
        let data_bytes = frames
            .checked_mul(self.layout.channels() as u64 * self.format.bytes() as u64)
            .ok_or_else(|| Error::overflow("WAV data size"))?;
        if data_bytes > u64::from(u32::MAX - 72) {
            return Err(Error::unsupported(
                "RIFF WAV exceeds 32-bit size; RF64 is required",
            ));
        }
        Ok(frames)
    }
    fn write_typed<T: WavSample>(&mut self, values: &[T]) -> Result<()> {
        let mut bytes = [0_u8; 4096];
        for chunk in values.chunks(bytes.len() / T::BYTES) {
            for (sample, slot) in chunk.iter().zip(bytes.chunks_exact_mut(T::BYTES)) {
                sample.write_le(slot);
            }
            self.sink
                .write_all(&bytes[..chunk.len() * T::BYTES])
                .map_err(Error::io)?;
        }
        Ok(())
    }
    /// Finalize RIFF lengths and fact count, then flush the destination.
    pub fn finish(mut self) -> Result<FinishedSink> {
        let bytes =
            (self.frames * self.layout.channels() as u64 * self.format.bytes() as u64) as u32;
        self.sink.seek(SeekFrom::Start(4)).map_err(Error::io)?;
        self.sink
            .write_all(&(bytes + 72).to_le_bytes())
            .map_err(Error::io)?;
        self.sink.seek(SeekFrom::Start(68)).map_err(Error::io)?;
        self.sink
            .write_all(&(self.frames as u32).to_le_bytes())
            .map_err(Error::io)?;
        self.sink.seek(SeekFrom::Start(76)).map_err(Error::io)?;
        self.sink
            .write_all(&bytes.to_le_bytes())
            .map_err(Error::io)?;
        self.sink.finish()
    }
}
trait WavSample {
    const BYTES: usize;
    fn write_le(&self, bytes: &mut [u8]);
}
impl WavSample for f32 {
    const BYTES: usize = 4;
    fn write_le(&self, bytes: &mut [u8]) {
        bytes.copy_from_slice(&self.to_le_bytes());
    }
}
impl WavSample for i16 {
    const BYTES: usize = 2;
    fn write_le(&self, bytes: &mut [u8]) {
        bytes.copy_from_slice(&self.to_le_bytes());
    }
}
impl WavSample for i32 {
    const BYTES: usize = 4;
    fn write_le(&self, bytes: &mut [u8]) {
        bytes.copy_from_slice(&self.to_le_bytes());
    }
}
