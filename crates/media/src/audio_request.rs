//! Continuous audio requests with bounded incremental output.
use crate::request::Asset;
use media_backend_native::{
    audio::{AudioDecodeOptions, AudioDecodeReport, AudioInfo, AudioReader, MixMatrix, WavWriter},
    resample::{AudioWindows, Resampler, slice_block},
};
use media_core::{
    device::Device,
    error::{Error, ErrorKind, Result},
    media::{AudioBlock, ChannelLayout},
    time::{Time, TimeRange},
};
use media_io::sink::{FinishedSink, Sink};
use media_io::source::SourceFingerprint;
use media_pipeline::{
    diagnostics::PlanStep,
    execution::CancellationToken,
    plan::PlanDescription,
    session::{CpuReservation, HostReservation},
};
use std::{
    collections::VecDeque,
    fs::{File, OpenOptions},
    ops::Range,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};
const RESAMPLE_CHUNK_HINT: usize = 1024;

#[path = "audio_samples_request.rs"]
mod converted_samples;
pub use converted_samples::{AudioSamplesRequest, AudioSamplesStream, AudioSamplesWriteResult};
#[path = "audio_decoder.rs"]
mod audio_decoder;
use audio_decoder::DecoderAdapter;

/// Explicit audio decoder implementation; continuous downstream stages are shared.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum AudioBackend {
    /// Pure Rust decoding; unsupported container-edit semantics require opt-in raw output.
    #[default]
    Native,
    /// One FFmpeg demux/decoder path applies MOV edits and automatic skip metadata.
    #[cfg(feature = "ffmpeg-audio")]
    Ffmpeg,
}
/// Whether the request promises a playable container interval or raw decoder output.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum AudioSemantics {
    /// Require established edit/trim interpretation; default native ISO MP4 is rejected.
    #[default]
    Semantic,
    /// Explicitly accept backend decoder extent without a container-semantic claim.
    RawDecoded,
}

/// A sample-count window with an explicit hop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Window {
    frames: usize,
    hop: usize,
}
impl Window {
    /// Complete windows of this many sample frames.
    pub fn samples(frames: usize) -> Self {
        Self {
            frames,
            hop: frames,
        }
    }
    /// Set distance between successive window starts.
    pub fn hop(mut self, frames: usize) -> Self {
        self.hop = frames;
        self
    }
}
/// Lazy audio decode, mix, continuous resample, slice, and window operations.
#[derive(Debug, Clone)]
pub struct AudioRequest {
    asset: Asset,
    layout: Option<ChannelLayout>,
    matrix: Option<MixMatrix>,
    rate: Option<u32>,
    range: Option<TimeRange>,
    window: Option<Window>,
    options: AudioDecodeOptions,
    backend: AudioBackend,
    semantics: AudioSemantics,
    #[cfg(feature = "ffmpeg-audio")]
    ffmpeg_options: media_backend_ffmpeg::audio::AudioDecodeOptions,
}
impl AudioRequest {
    pub(crate) fn new(asset: Asset) -> Self {
        Self {
            asset,
            layout: None,
            matrix: None,
            rate: None,
            range: None,
            window: None,
            options: AudioDecodeOptions::default(),
            backend: AudioBackend::Native,
            semantics: AudioSemantics::Semantic,
            #[cfg(feature = "ffmpeg-audio")]
            ffmpeg_options: Default::default(),
        }
    }
    /// Use a named mono/stereo downmix; other layouts need an explicit matrix.
    pub fn downmix(mut self, layout: ChannelLayout) -> Self {
        self.layout = Some(layout);
        self.matrix = None;
        self
    }
    /// Use exact caller-provided channel coefficients.
    pub fn mix(mut self, matrix: MixMatrix) -> Self {
        self.layout = None;
        self.matrix = Some(matrix);
        self
    }
    /// Convert continuously to this sample rate with antialiasing.
    pub fn resample(mut self, rate: u32) -> Self {
        self.rate = Some(rate);
        self
    }
    /// Select a half-open interval after continuous resampling on the source grid.
    pub fn slice(mut self, range: TimeRange) -> Self {
        self.range = Some(range);
        self
    }
    /// Form complete overlapping windows; incomplete tail frames are reported.
    pub fn windows(mut self, window: Window) -> Self {
        self.window = Some(window);
        self
    }
    /// Replace native track/corruption/probing options.
    pub fn decode_options(mut self, options: AudioDecodeOptions) -> Self {
        self.options = options;
        self
    }
    /// Select pure Rust decoding, retaining the explicit semantic/raw policy.
    pub fn native(mut self) -> Self {
        self.backend = AudioBackend::Native;
        self
    }
    /// Explicitly accept raw decoder extent; no gapless/edit correctness is promised.
    pub fn raw_decoded(mut self) -> Self {
        self.semantics = AudioSemantics::RawDecoded;
        self
    }
    /// Choose whether a playable container interval must be established.
    pub fn semantics(mut self, semantics: AudioSemantics) -> Self {
        self.semantics = semantics;
        self
    }
    /// Current explicit decoder route.
    pub fn backend(&self) -> AudioBackend {
        self.backend
    }
    /// Select one FFmpeg demux/decoder path, preserving the original source clock.
    #[cfg(feature = "ffmpeg-audio")]
    pub fn ffmpeg(mut self) -> Self {
        if self.backend != AudioBackend::Ffmpeg {
            self.ffmpeg_options.track = self.options.track.map(media_core::media::TrackId);
            self.ffmpeg_options.probe.max_read_bytes = self.options.max_probe_bytes;
            self.ffmpeg_options.max_packet_bytes = self.options.max_packet_bytes;
            self.ffmpeg_options.max_buffered_samples = self.options.max_buffered_samples;
            self.ffmpeg_options.max_block_frames = self.options.max_block_frames;
            self.ffmpeg_options.allow_one_tick = self.options.timestamps
                == media_backend_native::audio::AudioTimestampPolicy::AllowOneTick;
        }
        self.backend = AudioBackend::Ffmpeg;
        self
    }
    /// Select FFmpeg and replace its explicit decoder/probe/numeric-time controls.
    #[cfg(feature = "ffmpeg-audio")]
    pub fn ffmpeg_options(
        mut self,
        options: media_backend_ffmpeg::audio::AudioDecodeOptions,
    ) -> Self {
        self.semantics = match options.trim {
            media_backend_ffmpeg::audio::AudioTrimPolicy::Semantic => AudioSemantics::Semantic,
            media_backend_ffmpeg::audio::AudioTrimPolicy::RawDecoded => AudioSemantics::RawDecoded,
        };
        self.ffmpeg_options = options;
        self.backend = AudioBackend::Ffmpeg;
        self
    }
    /// Explicitly decode through FFmpeg and rebuild a continuous processing
    /// clock from sample counts when container timestamps are broken. All valid
    /// decoded samples are kept. Decoder blocks retain original timestamps in
    /// source_start(); resampling does not preserve that per-block provenance.
    /// This selects RawDecoded extent and does not promise synchronization with
    /// video or repair missing samples. Inspect ffmpeg_decode_report() for counts.
    #[cfg(feature = "ffmpeg-audio")]
    pub fn decoded_sample_clock(mut self) -> Self {
        self = self.ffmpeg();
        self.semantics = AudioSemantics::RawDecoded;
        self.ffmpeg_options.reconstruct_timestamps = true;
        self.ffmpeg_options.trim = media_backend_ffmpeg::audio::AudioTrimPolicy::RawDecoded;
        self
    }
    /// Set the FFmpeg processing clock before interpreting slices/windows; this
    /// explicitly selects FFmpeg. Asset requires an established common origin.
    /// Source (the default) interprets slice ranges on demux presentation PTS
    /// after container edits. Asset interprets them after subtracting the common
    /// origin, throughout decode, continuous resampling, slicing and windowing.
    /// This changes the processing grid itself, rather than retiming a result.
    ///
    /// ```no_run
    /// # fn example(asset: &media::request::Asset) -> media::Result<()> {
    /// use media::backend::ffmpeg::audio::AudioTimeline;
    /// use media::core::time::{Time, TimeRange};
    /// let range = TimeRange::new(Time::ZERO, Time::from_millis(10))?;
    /// let waveform = asset.audio()?.ffmpeg()
    ///     .timeline(AudioTimeline::Asset).slice(range).read()?;
    /// # drop(waveform); Ok(()) }
    /// ```
    #[cfg(feature = "ffmpeg-audio")]
    pub fn timeline(mut self, timeline: media_backend_ffmpeg::audio::AudioTimeline) -> Self {
        self = self.ffmpeg();
        self.ffmpeg_options.timeline = timeline;
        self
    }
    /// Explicitly convert unit-float output to a typed numeric/packing contract.
    pub fn to_samples(
        self,
        spec: media_backend_native::audio_samples::SampleSpec,
    ) -> AudioSamplesRequest {
        AudioSamplesRequest::new(self, spec)
    }
    /// Request strict interleaved f32/PCM16/PCM32 samples with no implicit dither.
    /// Existing read() remains canonical interleaved f32 and preserves headroom.
    pub fn sample_format(
        self,
        format: media_backend_native::audio_samples::SampleFormat,
    ) -> AudioSamplesRequest {
        self.to_samples(media_backend_native::audio_samples::SampleSpec::new(format))
    }
    /// Bounded header/first-packet probe without decoding audio or building filters.
    pub fn probe(&self) -> Result<AudioInfo> {
        let _cpu = self.reserve_decoder()?;
        self.probe_admitted()
    }
    fn probe_admitted(&self) -> Result<AudioInfo> {
        Ok(DecoderAdapter::open(self, decoder_options(self))?
            .info()
            .clone())
    }
    fn reserve_decoder(&self) -> Result<CpuReservation> {
        self.asset
            .session
            .resources
            .reserve_cpu(1, self.decoder_threads())
    }
    fn decoder_threads(&self) -> usize {
        #[cfg(feature = "ffmpeg-audio")]
        if self.backend == AudioBackend::Ffmpeg {
            return self.ffmpeg_options.threads;
        }
        1
    }
    /// Compile a reusable description using bounded header/first-packet probing.
    ///
    /// No audio is decoded, no FFT/window working state or shared host capacity
    /// is retained. Structural choices are checked now; current budget admission,
    /// filter chunk alignment/delay, and true final coverage remain execution work.
    pub fn compile(&self) -> Result<AudioPlan> {
        let (source, info) = {
            let _cpu = self.reserve_decoder()?;
            let source = self.asset.source.fingerprint()?;
            let discovered = self.probe_admitted();
            if source.is_versioned() && source != self.asset.source.fingerprint()? {
                return Err(
                    Error::incompatible("source changed during audio plan compilation")
                        .with_source(self.asset.source.label()),
                );
            }
            (source, discovered?)
        };
        let mix = resolve_mix(self, &info)?;
        let layout = mix.as_ref().map_or_else(
            || info.layout.clone(),
            |matrix| matrix.output_layout().clone(),
        );
        let rate = self.rate.unwrap_or(info.sample_rate);
        if rate == 0 {
            return Err(Error::invalid("output sample rate must be nonzero"));
        }
        if self.range.is_some_and(|range| range.start() < info.start) {
            return Err(Error::new(
                ErrorKind::MissingData,
                "audio query starts before source coverage",
            ));
        }
        if self
            .window
            .is_some_and(|window| window.frames == 0 || window.frames > 65536 || window.hop == 0)
        {
            return Err(Error::invalid(
                "audio windows require 1..=65536 frames and a nonzero hop",
            ));
        }
        let bounds = self
            .range
            .map(|range| range.sample_bounds(info.start, rate))
            .transpose()?;
        let options = execution_options(self);
        let limits = AudioWorkingLimits {
            host_memory: self.asset.session.limits().host_memory,
            max_probe_bytes: options.max_probe_bytes,
            max_packet_bytes: options.max_packet_bytes,
            max_decoded_packet_samples: options.max_buffered_samples,
            max_decoder_block_frames: options
                .max_block_frames
                .min(options.max_buffered_samples / info.layout.channels()),
            resample_chunk_hint_frames: (rate != info.sample_rate).then_some(RESAMPLE_CHUNK_HINT),
            window_frames: self.window.map(|window| window.frames),
            window_hop_frames: self.window.map(|window| window.hop),
        };
        let output = AudioPlanOutput {
            sample_rate: rate,
            channel_layout: layout,
            format: AudioSampleFormat::F32,
            interleaved: true,
        };
        let mut description = PlanDescription::default();
        description.steps.push(PlanStep {
            operation: format!(
                "decode track {} to interleaved f32 at {} Hz, {:?}; {:?} interval on the explicitly selected processing clock",
                info.track, info.sample_rate, info.layout, self.semantics
            ),
            backend: match self.backend {
                AudioBackend::Native => "symphonia",
                #[cfg(feature = "ffmpeg-audio")]
                AudioBackend::Ffmpeg => "ffmpeg/libavcodec + libswresample",
            }
            .into(),
            output: Device::Cpu,
            format: None,
            materializes: true,
        });
        if mix.is_some() {
            description.steps.push(PlanStep {
                operation: format!("explicit channel mapping to {:?}", output.channel_layout),
                backend: "native mixing matrix".into(),
                output: Device::Cpu,
                format: None,
                materializes: true,
            });
        }
        if rate != info.sample_rate {
            description.steps.push(PlanStep{operation:format!("continuous {} -> {} Hz; grid origin {}; startup delay removed and final computational tail trimmed",
            info.sample_rate,rate,info.start),backend:"rubato FFT".into(),output:Device::Cpu,format:None,materializes:true});
        }
        if let (Some(range), Some(bounds)) = (self.range, &bounds) {
            description.steps.push(PlanStep {
                operation: format!(
                    "half-open [{}, {}) on continuous origin {}; ceil sample bounds {}..{}",
                    range.start(),
                    range.end(),
                    info.start,
                    bounds.start,
                    bounds.end
                ),
                backend: "continuous sample-grid slice".into(),
                output: Device::Cpu,
                format: None,
                materializes: true,
            });
        }
        if let Some(window) = self.window {
            description.steps.push(PlanStep{operation:format!("complete {}-frame windows with {}-frame hop; incomplete tail explicitly discarded",window.frames,window.hop),
            backend:"bounded overlapping audio windows".into(),output:Device::Cpu,format:None,materializes:true});
        }
        description.steps.push(PlanStep{operation:format!("host limit {} bytes; probe {} bytes; compressed packet {} bytes; decoded packet {} f32 samples; decoder block <= {} frames",
            limits.host_memory,limits.max_probe_bytes,limits.max_packet_bytes,limits.max_decoded_packet_samples,limits.max_decoder_block_frames),
            backend:"shared session admission".into(),output:Device::Cpu,format:None,materializes:false});
        description.unresolved.push(
            "codec-private and FFT scratch/heap allocations are backend-managed and unmeasured"
                .into(),
        );
        if self.semantics == AudioSemantics::RawDecoded {
            description.unresolved.push("RawDecoded explicitly exposes this backend's decoder extent; container edits, priming and padding are not a semantic/gapless guarantee".into());
        }
        #[cfg(feature = "ffmpeg-audio")]
        if self.backend == AudioBackend::Ffmpeg && self.ffmpeg_options.reconstruct_timestamps {
            description.unresolved.push("explicit decoded-sample clock: processing timestamps are rebuilt, decoder blocks retain original timestamps before resampling, and cross-stream synchronization is not promised".into());
        }
        description.unresolved.push("actual filter chunk alignment/startup delay, final semantic coverage, and current shared-budget admission are checked during execution".into());
        description.unresolved.push("range extraction processes dependencies from the original recording origin; compilation does not build a seek index or decode samples".into());
        if !source.is_versioned() {
            description.unresolved.push("unversioned source: no freshness guarantee; probed track/rate/layout/origin are revalidated on every fresh execution".into());
        }
        Ok(AudioPlan {
            request: self.clone(),
            source,
            info,
            output,
            limits,
            bounds,
            description,
        })
    }
    /// Open independent decoder/resampler state.
    pub fn reader(&self) -> Result<AudioStream> {
        AudioStream::open(self.clone())
    }
    pub(crate) fn reader_with_source_snapshot(&self) -> Result<(AudioStream, SourceFingerprint)> {
        let cpu = self.reserve_decoder()?;
        let source = self.asset.source.fingerprint()?;
        let reader = AudioStream::open_admitted(self.clone(), cpu)?;
        if source.is_versioned() && source != self.asset.source.fingerprint()? {
            return Err(Error::incompatible(
                "audio source changed while opening execution",
            ));
        }
        Ok((reader, source))
    }
    /// Return bounded incremental blocks/windows.
    pub fn stream(&self) -> Result<AudioStream> {
        self.reader()
    }
    /// Materialize one finite continuous waveform within shared host limits.
    ///
    /// Window requests use stream() so overlapping samples are not silently
    /// concatenated into a misleading waveform.
    pub fn read(&self) -> Result<AudioBlock> {
        self.ensure_materializable()?;
        self.materialize_stream(self.reader()?)
    }
    fn ensure_materializable(&self) -> Result<()> {
        if self.window.is_some() {
            return Err(Error::invalid(
                "materializing overlapping windows requires stream(), not a concatenated waveform",
            ));
        }
        Ok(())
    }
    fn materialize_stream(&self, mut stream: AudioStream) -> Result<AudioBlock> {
        let mut samples = Vec::new();
        let mut guards: Vec<HostReservation> = Vec::new();
        let mut start = None;
        let mut source_start = None;
        let mut next = None;
        while let Some(block) = stream.next_block()? {
            if next.is_some_and(|v| v != block.start()) {
                return Err(Error::incompatible(
                    "audio waveform has a timing discontinuity",
                ));
            }
            if start.is_none() {
                source_start = block.source_start();
                start = Some(block.start());
            }
            next = Some(block.end()?);
            let bytes = block
                .samples()
                .len()
                .checked_mul(4)
                .ok_or_else(|| Error::overflow("audio materialization size"))?;
            let guard = self.asset.session.resources.host_budget().reserve(bytes)?;
            samples
                .try_reserve_exact(block.samples().len())
                .map_err(|_| Error::exhausted("audio materialization allocation failed"))?;
            samples.extend_from_slice(block.samples());
            guards.push(guard);
        }
        let bytes = samples
            .len()
            .checked_mul(4)
            .ok_or_else(|| Error::overflow("audio output size"))?;
        let final_guard = self.asset.session.resources.host_budget().reserve(bytes)?;
        let block = AudioBlock::new(
            samples,
            stream.rate,
            stream.layout.clone(),
            start.unwrap_or(stream.output_origin()?),
        )?
        .with_source_start(source_start)
        .retain(final_guard);
        drop(guards);
        Ok(block)
    }
    /// Write IEEE-float WAV through a same-directory staging file and finalize it.
    /// Only .wav/.wave destinations are accepted. Existing output is replaced only
    /// after successful decoding, encoding, header finalization, and close; a failure
    /// leaves the existing destination unchanged and removes unused staging data.
    /// Standalone WAV sample zero is the first emitted sample; original source
    /// offsets remain available from probe/read timing, rather than in the WAV header.
    pub fn write(&self, path: impl AsRef<Path>) -> Result<FinishedSink> {
        let path = path.as_ref();
        if !path
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("wav") || ext.eq_ignore_ascii_case("wave"))
        {
            return Err(Error::invalid(
                "native audio output is WAV; use a .wav or .wave destination",
            ));
        }
        if self.window.is_some() {
            return Err(Error::invalid(
                "overlapping windows cannot be written as one continuous WAV",
            ));
        }
        if let Some(input) = self.asset.source.path()
            && same_file(input, path)?
        {
            return Err(Error::invalid(
                "streaming audio output must differ from its input file",
            ));
        }
        let mut stream = self.reader()?;
        if !(stream.layout == ChannelLayout::mono() || stream.layout == ChannelLayout::stereo()) {
            return Err(Error::unsupported(
                "native WAV output needs named mono/stereo channels; mix explicitly",
            ));
        }
        stream
            .rate
            .checked_mul(stream.layout.channels() as u32 * 4)
            .ok_or_else(|| Error::overflow("WAV output byte rate"))?;
        let (stage, file) = StagedOutput::create(path)?;
        let mut writer = WavWriter::new(
            Sink::from_file(file, stage.path.clone()),
            stream.rate,
            stream.layout.clone(),
        )?;
        while let Some(block) = stream.next_block()? {
            writer.write(&block)?;
        }
        let bytes_written = match writer.finish()? {
            FinishedSink::File { bytes_written, .. } => bytes_written,
            FinishedSink::Memory(_) => {
                return Err(Error::backend(
                    "audio",
                    "file writer returned memory output",
                ));
            }
        };
        stage.commit(path)?;
        Ok(FinishedSink::File {
            path: path.to_path_buf(),
            bytes_written,
        })
    }
}
/// Native scalar representation produced by an audio plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioSampleFormat {
    /// Native f32 floating-point samples, without an implicit clipping operation.
    F32,
}
/// Probed and resolved output representation, separate from pixel formats.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioPlanOutput {
    /// Output sample frames per second.
    pub sample_rate: u32,
    /// Ordered output speaker/channel identities.
    pub channel_layout: ChannelLayout,
    /// Stored numeric sample format.
    pub format: AudioSampleFormat,
    /// True: all samples in a frame precede the next frame's samples.
    pub interleaved: bool,
}
/// Known execution caps; these are bounds, not measured private heap estimates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioWorkingLimits {
    /// Shared host budget, including application-retained outputs.
    pub host_memory: usize,
    /// Hard physical reads during native header/first-packet discovery.
    pub max_probe_bytes: u64,
    /// Maximum accepted compressed packet bytes after the session clamp.
    pub max_packet_bytes: usize,
    /// Maximum copied interleaved decoded samples in one packet.
    pub max_decoded_packet_samples: usize,
    /// Maximum decoder frames before mixing/resampling/window processing.
    pub max_decoder_block_frames: usize,
    /// Requested filter chunk hint; actual rational alignment is resolved at execution.
    pub resample_chunk_hint_frames: Option<usize>,
    /// Maximum complete window length, when selected.
    pub window_frames: Option<usize>,
    /// Explicit distance between window starts, when selected.
    pub window_hop_frames: Option<usize>,
}
/// Cloneable audio work with no retained cursor, decoder, FFT, or execution reservation.
#[derive(Debug, Clone)]
pub struct AudioPlan {
    request: AudioRequest,
    source: SourceFingerprint,
    info: AudioInfo,
    output: AudioPlanOutput,
    limits: AudioWorkingLimits,
    bounds: Option<Range<i64>>,
    description: PlanDescription,
}
impl AudioPlan {
    fn validate_source(&self) -> Result<()> {
        if self.source.is_versioned() && self.source != self.request.asset.source.fingerprint()? {
            return Err(
                Error::incompatible("source changed since audio plan compilation")
                    .with_source(self.request.asset.source.label()),
            );
        }
        Ok(())
    }
    /// Source identity/version observed around the bounded compilation probe.
    pub fn source(&self) -> &SourceFingerprint {
        &self.source
    }
    /// Header/first-packet source interpretation established during compilation.
    pub fn source_info(&self) -> &AudioInfo {
        &self.info
    }
    /// Resolved numeric format, channel identities, rate and packing.
    pub fn output(&self) -> &AudioPlanOutput {
        &self.output
    }
    /// Native and session caps resolved without creating filter working state.
    pub fn working_limits(&self) -> AudioWorkingLimits {
        self.limits
    }
    /// Original recording's output-grid anchor; clips do not reanchor this origin.
    pub fn sample_grid_origin(&self) -> Time {
        self.info.start
    }
    /// Ceil-rounded requested indices on the continuous output grid, when sliced.
    pub fn sample_bounds(&self) -> Option<Range<i64>> {
        self.bounds.clone()
    }
    /// Ordered backend/materialization choices and explicit runtime limitations.
    pub fn description(&self) -> &PlanDescription {
        &self.description
    }
    /// Human-readable backend, representation, grid, allocation and runtime choices.
    pub fn explain(&self) -> String {
        self.description.explain()
    }
    /// Materialize one finite waveform with source validation before and after work.
    /// Window plans use reader()/stream(), matching AudioRequest::read rejection.
    pub fn execute(&self) -> Result<AudioBlock> {
        self.request.ensure_materializable()?;
        let result = self.request.materialize_stream(self.reader()?);
        let _metadata = self.request.asset.session.resources.reserve_cpu(0, 0)?;
        self.validate_source()?;
        result
    }
    /// Open independent state, checking source freshness and probed interpretation.
    /// Streaming checks freshness at open; finite execute also validates at the end.
    pub fn reader(&self) -> Result<AudioStream> {
        let cpu = self.request.reserve_decoder()?;
        self.validate_source()?;
        let reader = AudioStream::open_admitted(self.request.clone(), cpu)?;
        self.validate_source()?;
        let actual = reader.info();
        if actual.track != self.info.track
            || actual.sample_rate != self.info.sample_rate
            || actual.layout != self.info.layout
            || actual.start != self.info.start
            || actual.frames != self.info.frames
            || actual.timestamp_quantum != self.info.timestamp_quantum
        {
            return Err(Error::incompatible(
                "audio source interpretation changed since compilation; compile again",
            ));
        }
        Ok(reader)
    }
    /// Open another independent continuous decoder/filter/window execution.
    pub fn stream(&self) -> Result<AudioStream> {
        self.reader()
    }
}

fn decoder_options(request: &AudioRequest) -> AudioDecodeOptions {
    #[allow(unused_mut)] // Optional FFmpeg route fills its explicitly separate controls.
    let mut options = request.options.clone();
    #[cfg(feature = "ffmpeg-audio")]
    if request.backend == AudioBackend::Ffmpeg {
        options.track = request.ffmpeg_options.track.map(|id| id.0);
        options.max_probe_bytes = request.ffmpeg_options.probe.max_read_bytes;
        options.max_packet_bytes = request.ffmpeg_options.max_packet_bytes;
        options.max_buffered_samples = request.ffmpeg_options.max_buffered_samples;
        options.max_block_frames = request.ffmpeg_options.max_block_frames;
    }
    options
}
fn execution_options(request: &AudioRequest) -> AudioDecodeOptions {
    let mut options = decoder_options(request);
    let capacity = request.asset.session.limits().host_memory;
    options.max_buffered_samples = options.max_buffered_samples.min(
        capacity
            / if request.backend == AudioBackend::Native {
                16
            } else {
                64
            },
    );
    options.max_packet_bytes = options.max_packet_bytes.min(capacity / 4);
    options
}
fn resolve_mix(request: &AudioRequest, info: &AudioInfo) -> Result<Option<MixMatrix>> {
    if let Some(matrix) = &request.matrix {
        if matrix.input_layout() != &info.layout {
            return Err(Error::incompatible(
                "mix matrix does not match source channel identities",
            ));
        }
        Ok(Some(matrix.clone()))
    } else if let Some(layout) = &request.layout {
        if layout == &info.layout {
            Ok(None)
        } else if layout == &ChannelLayout::mono() {
            Ok(Some(MixMatrix::mono(info.layout.clone())?))
        } else if layout == &ChannelLayout::stereo() {
            Ok(Some(MixMatrix::stereo(info.layout.clone())?))
        } else {
            Err(Error::unsupported(
                "target channel layout needs an explicit mix matrix",
            ))
        }
    } else {
        Ok(None)
    }
}
/// An independent continuous audio execution. Dropped outputs retain their own samples.
pub struct AudioStream {
    request: AudioRequest,
    decoder: Option<DecoderAdapter>,
    cpu_admission: Option<CpuReservation>,
    final_report: AudioDecodeReport,
    #[cfg(feature = "ffmpeg-audio")]
    final_ffmpeg_report: Option<media_backend_ffmpeg::audio::AudioDecodeReport>,
    #[cfg(feature = "ffmpeg-audio")]
    ffmpeg_info: Option<media_backend_ffmpeg::audio::AudioInfo>,
    work_reservation: Option<HostReservation>,
    discarded_window_frames: usize,
    mix: Option<MixMatrix>,
    resampler: Option<Resampler>,
    windows: Option<AudioWindows>,
    pending: VecDeque<AudioBlock>,
    window_input: Option<(AudioBlock, usize)>,
    info: AudioInfo,
    layout: ChannelLayout,
    rate: u32,
    source_end: Time,
    started: bool,
    finished: bool,
    failed: bool,
    cancel: CancellationToken,
}
impl AudioStream {
    fn open(request: AudioRequest) -> Result<Self> {
        let cpu = request.reserve_decoder()?;
        Self::open_admitted(request, cpu)
    }
    fn open_admitted(mut request: AudioRequest, cpu: CpuReservation) -> Result<Self> {
        request.options = execution_options(&request);
        let decoder = DecoderAdapter::open(&request, request.options.clone())?;
        let info = decoder.info().clone();
        let mix = resolve_mix(&request, &info)?;
        let layout = mix
            .as_ref()
            .map_or_else(|| info.layout.clone(), |m| m.output_layout().clone());
        let rate = request.rate.unwrap_or(info.sample_rate);
        if rate == 0 {
            return Err(Error::invalid("output sample rate must be nonzero"));
        }
        if request
            .range
            .is_some_and(|range| range.start() < info.start)
        {
            return Err(Error::new(
                ErrorKind::MissingData,
                "audio query starts before source coverage",
            ));
        }
        let resampler = if rate != info.sample_rate {
            Some(Resampler::new(
                info.sample_rate,
                rate,
                layout.clone(),
                info.start,
                RESAMPLE_CHUNK_HINT,
            )?)
        } else {
            None
        };
        let windows = request
            .window
            .map(|w| AudioWindows::new(w.frames, w.hop))
            .transpose()?;
        // Reserve the library-owned packet copy, processing blocks, filter I/O,
        // and window staging before execution. Native codec/FFT private scratch
        // remains backend-managed; it is not misreported as observed allocation.
        let block_frames = request
            .options
            .max_block_frames
            .min(request.options.max_buffered_samples / info.layout.channels());
        let block_samples = block_frames
            .checked_mul(layout.channels())
            .ok_or_else(|| Error::overflow("audio working block"))?;
        let mut work_bytes = request
            .options
            .max_buffered_samples
            .checked_mul(if request.backend == AudioBackend::Native {
                4
            } else {
                16
            })
            .and_then(|n| {
                block_samples
                    .checked_mul(8)
                    .and_then(|block| n.checked_add(block))
            })
            .ok_or_else(|| Error::overflow("audio working memory"))?;
        if let Some(resampler) = &resampler {
            let input = resampler.input_chunk_frames();
            let output = (input as u128 * rate as u128).div_ceil(info.sample_rate as u128);
            let frames = (input as u128)
                .checked_add(
                    output
                        .checked_mul(3)
                        .ok_or_else(|| Error::overflow("resampler staging"))?,
                )
                .ok_or_else(|| Error::overflow("resampler working frames"))?;
            let bytes = usize::try_from(
                frames
                    .checked_mul(layout.channels() as u128 * 4)
                    .ok_or_else(|| Error::overflow("resampler working memory"))?,
            )
            .map_err(|_| Error::overflow("resampler working memory"))?;
            work_bytes = work_bytes
                .checked_add(bytes)
                .ok_or_else(|| Error::overflow("audio working memory"))?;
        }
        if let Some(window) = request.window {
            let bytes = window
                .frames
                .checked_mul(layout.channels())
                .and_then(|n| n.checked_mul(8))
                .ok_or_else(|| Error::overflow("window working memory"))?;
            work_bytes = work_bytes
                .checked_add(bytes)
                .ok_or_else(|| Error::overflow("audio working memory"))?;
        }
        let work_reservation = Some(
            request
                .asset
                .session
                .resources
                .host_budget()
                .reserve(work_bytes)?,
        );
        let finished = request
            .range
            .is_some_and(|range| range.start() == info.start && range.start() == range.end());
        Ok(Self {
            #[cfg(feature = "ffmpeg-audio")]
            ffmpeg_info: decoder.ffmpeg_info().cloned(),
            request,
            decoder: Some(decoder),
            cpu_admission: Some(cpu),
            final_report: AudioDecodeReport::default(),
            #[cfg(feature = "ffmpeg-audio")]
            final_ffmpeg_report: None,
            work_reservation,
            discarded_window_frames: 0,
            mix,
            resampler,
            windows,
            pending: VecDeque::new(),
            window_input: None,
            layout,
            rate,
            source_end: info.start,
            started: false,
            info,
            finished,
            failed: false,
            cancel: CancellationToken::default(),
        })
    }
    /// Probed source-track metadata. The first decoded block refines start when
    /// codec priming establishes later actual coverage than the header.
    pub fn info(&self) -> &AudioInfo {
        &self.info
    }
    /// Interval actually decoded so far on the processing clock, excluding tail.
    /// A finished reader establishes the true source end; a partial reader only
    /// establishes coverage through the last decoded dependency block.
    /// With explicit decoded_sample_clock(), this is reconstructed time; original
    /// source timestamps remain on decoder blocks before resampling.
    pub fn source_coverage(&self) -> Result<TimeRange> {
        TimeRange::new(self.info.start, self.source_end)
    }
    #[cfg(feature = "ffmpeg")]
    pub(crate) fn output_rate(&self) -> u32 {
        self.rate
    }
    #[cfg(feature = "ffmpeg")]
    pub(crate) fn output_layout(&self) -> &ChannelLayout {
        &self.layout
    }
    /// Cooperative cancellation handle.
    pub fn cancellation(&self) -> CancellationToken {
        self.cancel.clone()
    }
    #[cfg(feature = "ffmpeg")]
    pub(crate) fn with_cancellation(mut self, token: CancellationToken) -> Self {
        self.cancel = token;
        self
    }
    /// Frames discarded from an incomplete final window.
    pub fn discarded_window_frames(&self) -> usize {
        self.windows
            .as_ref()
            .map_or(self.discarded_window_frames, AudioWindows::discarded_frames)
    }
    /// Source decoder counters, including corrupt packets and finalization.
    pub fn decode_report(&self) -> &AudioDecodeReport {
        self.decoder
            .as_ref()
            .map_or(&self.final_report, DecoderAdapter::report)
    }
    /// FFmpeg edit/skip/semantic-tail counters, preserved after EOF or failure.
    #[cfg(feature = "ffmpeg-audio")]
    pub fn ffmpeg_decode_report(&self) -> Option<&media_backend_ffmpeg::audio::AudioDecodeReport> {
        self.decoder
            .as_ref()
            .and_then(DecoderAdapter::ffmpeg_report)
            .or(self.final_ffmpeg_report.as_ref())
    }
    /// Original FFmpeg source/asset-clock interpretation, even after retirement.
    #[cfg(feature = "ffmpeg-audio")]
    pub fn ffmpeg_info(&self) -> Option<&media_backend_ffmpeg::audio::AudioInfo> {
        self.ffmpeg_info.as_ref()
    }
    /// Return one bounded block or complete window; failures are terminal.
    pub fn next_block(&mut self) -> Result<Option<AudioBlock>> {
        if self.failed {
            return Err(Error::invalid("audio execution previously failed"));
        }
        let result = self.next_inner();
        if result.is_err() {
            self.failed = true;
            self.retire();
        }
        result
    }
    fn next_inner(&mut self) -> Result<Option<AudioBlock>> {
        loop {
            self.cancel.check()?;
            if let Some((block, offset)) = self.window_input.take() {
                let window = self
                    .request
                    .window
                    .ok_or_else(|| Error::invalid("window state without a window"))?;
                let frames = (block.frames() - offset).min(window.hop.min(4096));
                let channels = self.layout.channels();
                let part = AudioBlock::new(
                    block.samples()[offset * channels..(offset + frames) * channels].to_vec(),
                    self.rate,
                    self.layout.clone(),
                    block
                        .start()
                        .checked_add(Time::from_samples(offset as i64, self.rate)?)?,
                )?;
                if offset + frames < block.frames() {
                    self.window_input = Some((block, offset + frames));
                }
                let mut outputs = self
                    .windows
                    .as_mut()
                    .ok_or_else(|| Error::invalid("missing window state"))?
                    .push(&part)?;
                // At most one window can become ready when at most one hop was
                // fed. Assert that bound rather than silently dropping outputs.
                if outputs.len() > 1 {
                    return Err(Error::backend(
                        "audio",
                        "window stage exceeded its one-hop output bound",
                    ));
                }
                if let Some(output) = outputs.pop() {
                    return self.account(output).map(Some);
                }
                continue;
            }
            if let Some(block) = self.pending.pop_front() {
                if self
                    .request
                    .range
                    .is_some_and(|range| block.end().is_ok_and(|end| end >= range.end()))
                {
                    self.finished = true;
                    self.pending.clear();
                }
                let block = if let Some(range) = self.request.range {
                    let start = range.start().max(block.start());
                    let end = range.end().min(block.end()?);
                    if start >= end {
                        continue;
                    }
                    slice_block(&block, TimeRange::new(start, end)?)?
                } else {
                    block
                };
                if block.frames() == 0 {
                    continue;
                }
                if self.windows.is_some() {
                    self.window_input = Some((block, 0));
                    continue;
                }
                return self.account(block).map(Some);
            }
            if self.finished {
                if let Some(windows) = &mut self.windows {
                    windows.finish()?;
                }
                self.retire();
                if self
                    .request
                    .range
                    .is_some_and(|range| range.end() > self.source_end)
                {
                    return Err(Error::new(
                        ErrorKind::MissingData,
                        "audio query ends beyond source coverage",
                    ));
                }
                return Ok(None);
            }
            match self
                .decoder
                .as_mut()
                .ok_or_else(|| Error::invalid("audio decoder already retired"))?
                .next_block(&self.cancel)?
            {
                Some(block) => {
                    if !self.started && block.start() != self.source_end {
                        // A header grid may precede the first valid decoder
                        // output (for example older AAC or Vorbis overlap).
                        // Whole-stream reads expose actual coverage. A query
                        // requesting the missing leading interval still fails.
                        if self
                            .request
                            .range
                            .is_some_and(|range| range.start() < block.start())
                        {
                            return Err(Error::new(
                                ErrorKind::MissingData,
                                "audio query starts before actual decoded coverage",
                            ));
                        }
                        self.info.start = block.start();
                        self.source_end = block.start();
                        if self.resampler.is_some() {
                            self.resampler = Some(Resampler::new(
                                self.info.sample_rate,
                                self.rate,
                                self.layout.clone(),
                                block.start(),
                                RESAMPLE_CHUNK_HINT,
                            )?);
                        }
                    }
                    self.started = true;
                    if block.start() != self.source_end {
                        return Err(Error::new(ErrorKind::MissingData,"audio source has a gap, overlap, or discontinuity; choose an explicit recovery policy")
                            .with_source(self.request.asset.source.label()).with_time(block.start()));
                    }
                    self.source_end = block.end()?;
                    let block = if let Some(mix) = &self.mix {
                        mix.process(&block)?
                    } else {
                        block
                    };
                    let outputs = if let Some(resampler) = &mut self.resampler {
                        resampler.push(&block)?
                    } else {
                        vec![block]
                    };
                    self.pending.extend(outputs);
                }
                None => {
                    if let Some(resampler) = &mut self.resampler {
                        self.pending.extend(resampler.finish()?);
                    }
                    self.finished = true;
                }
            }
        }
    }
    fn retire(&mut self) {
        self.pending.clear();
        self.window_input = None;
        if let Some(decoder) = self.decoder.take() {
            self.final_report = decoder.report().clone();
            #[cfg(feature = "ffmpeg-audio")]
            {
                self.final_ffmpeg_report = decoder.ffmpeg_report().cloned();
            }
        }
        if let Some(windows) = self.windows.take() {
            self.discarded_window_frames = windows.discarded_frames();
        }
        self.resampler = None;
        self.mix = None;
        self.work_reservation = None;
        self.cpu_admission = None;
    }
    fn account(&self, block: AudioBlock) -> Result<AudioBlock> {
        let bytes = block
            .samples()
            .len()
            .checked_mul(4)
            .ok_or_else(|| Error::overflow("audio block byte count"))?;
        Ok(block.retain(
            self.request
                .asset
                .session
                .resources
                .host_budget()
                .reserve(bytes)?,
        ))
    }
    fn output_origin(&self) -> Result<Time> {
        let Some(range) = self.request.range else {
            return Ok(self.info.start);
        };
        let index = range.sample_bounds(self.info.start, self.rate)?.start;
        self.info
            .start
            .checked_add(Time::from_samples(index, self.rate)?)
    }
}
impl Iterator for AudioStream {
    type Item = Result<AudioBlock>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.failed {
            return None;
        }
        self.next_block().transpose()
    }
}

static NEXT_STAGE: AtomicU64 = AtomicU64::new(0);
struct StagedOutput {
    path: PathBuf,
    committed: bool,
}
impl StagedOutput {
    fn create(destination: &Path) -> Result<(Self, File)> {
        let parent = destination
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        for _ in 0..128 {
            let sequence = NEXT_STAGE.fetch_add(1, Ordering::Relaxed);
            let path = parent.join(format!(
                ".media-audio-{}-{sequence}.stage",
                std::process::id()
            ));
            match OpenOptions::new()
                .write(true)
                .read(true)
                .create_new(true)
                .open(&path)
            {
                Ok(file) => {
                    let stage = Self {
                        path,
                        committed: false,
                    };
                    if let Ok(metadata) = std::fs::metadata(destination)
                        && let Err(error) = file.set_permissions(metadata.permissions())
                    {
                        drop(file);
                        return Err(Error::io(error).with_operation("preserve output permissions"));
                    }
                    return Ok((stage, file));
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => {
                    return Err(Error::io(error).with_operation("create audio staging file"));
                }
            }
        }
        Err(Error::exhausted(
            "could not create an unused audio staging path",
        ))
    }
    fn commit(mut self, destination: &Path) -> Result<()> {
        std::fs::rename(&self.path, destination)
            .map_err(|e| Error::io(e).with_operation("commit finalized WAV"))?;
        self.committed = true;
        Ok(())
    }
}
impl Drop for StagedOutput {
    fn drop(&mut self) {
        if !self.committed {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}
fn same_file(input: &Path, output: &Path) -> Result<bool> {
    let output_metadata = match std::fs::metadata(output) {
        Ok(value) => value,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(Error::io(error)),
    };
    if std::fs::canonicalize(output).is_ok_and(|path| path == input) {
        return Ok(true);
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let input_metadata = std::fs::metadata(input).map_err(Error::io)?;
        Ok(input_metadata.dev() == output_metadata.dev()
            && input_metadata.ino() == output_metadata.ino())
    }
    #[cfg(not(unix))]
    {
        let _ = output_metadata;
        Ok(false)
    }
}
