//! Explicit typed sample conversion after the canonical continuous audio stages.
use super::{AudioRequest, AudioStream, StagedOutput, same_file};
use media_backend_native::{
    audio::{AudioDecodeReport, AudioInfo, WavWriter},
    audio_samples::{
        self, ConversionReport, ConvertedAudio, SampleGamut, SampleOutput, SamplePacking,
        SampleSpec, TpdfDither,
    },
};
use media_core::{
    error::{Error, Result},
    media::ChannelLayout,
};
use media_io::{
    sink::{FinishedSink, Sink},
    source::SourceFingerprint,
};
use media_pipeline::execution::CancellationToken;
use std::path::Path;

/// Explicit output representation following continuous decode/mix/resample/slice.
#[derive(Clone, Debug)]
pub struct AudioSamplesRequest {
    request: AudioRequest,
    spec: SampleSpec,
}
impl AudioSamplesRequest {
    pub(super) fn new(request: AudioRequest, spec: SampleSpec) -> Self {
        Self { request, spec }
    }
    /// Choose actual contiguous plane/interleaved ordering.
    pub fn packing(mut self, packing: SamplePacking) -> Self {
        self.spec.packing = packing;
        self
    }
    /// Choose finite out-of-unit rejection or explicitly reported clipping.
    pub fn gamut(mut self, gamut: SampleGamut) -> Self {
        self.spec.gamut = gamut;
        self
    }
    /// Set or remove deterministic one-LSB TPDF on an explicit continuous grid.
    pub fn dither(mut self, dither: Option<TpdfDither>) -> Self {
        self.spec.dither = dither;
        self
    }
    /// Inspect the exact requested format/packing/numeric policy.
    pub fn specification(&self) -> SampleSpec {
        self.spec
    }
    /// Materialize a finite canonical waveform, then explicitly pack/quantize it.
    /// Both input and aligned output allocations count against the shared budget.
    /// Use reader()/stream() for blockwise conversion without a full waveform.
    pub fn read(&self) -> Result<ConvertedAudio> {
        self.spec.validate()?;
        self.request.ensure_materializable()?;
        let (inner, source) = self.request.reader_with_source_snapshot()?;
        let block = self.request.materialize_stream(inner)?;
        let _cpu = self.request.asset.session.resources.reserve_cpu(0, 0)?;
        validate_source(&self.request, &source)?;
        let output = accounted_convert(&self.request, &block, self.spec)?;
        validate_source(&self.request, &source)?;
        Ok(output)
    }
    /// Fill typed exclusive caller-owned storage after finite waveform decoding.
    /// Caller allocation is external to the session; canonical input remains
    /// budgeted. Any decode/validation/conversion error leaves output unchanged.
    /// Native audio_samples::process_into accepts individual streaming blocks.
    pub fn process_into(&self, output: SampleOutput<'_>) -> Result<ConversionReport> {
        self.spec.validate()?;
        self.request.ensure_materializable()?;
        let (inner, source) = self.request.reader_with_source_snapshot()?;
        let block = self.request.materialize_stream(inner)?;
        let _cpu = self.request.asset.session.resources.reserve_cpu(0, 0)?;
        validate_source(&self.request, &source)?;
        audio_samples::process_into(&block, self.spec, output)
    }
    /// Open independent bounded canonical and converted-block state.
    pub fn reader(&self) -> Result<AudioSamplesStream> {
        AudioSamplesStream::open(self.clone())
    }
    /// Open a fresh converted-block execution; overlapping windows remain distinct.
    pub fn stream(&self) -> Result<AudioSamplesStream> {
        self.reader()
    }
    /// Stream interleaved f32/PCM16/PCM32 WAV into a same-directory stage.
    /// All quantization/dither is explicit. Existing destination content changes
    /// only after successful decode, conversion, finalization, close and rename.
    /// Planar/window exports are rejected before touching the destination.
    pub fn write(&self, path: impl AsRef<Path>) -> Result<AudioSamplesWriteResult> {
        self.spec.validate()?;
        self.request.ensure_materializable()?;
        if self.spec.packing != SamplePacking::Interleaved {
            return Err(Error::unsupported(
                "WAV requires explicitly interleaved sample output",
            ));
        }
        let path = path.as_ref();
        validate_path(&self.request, path)?;
        let mut reader = self.reader()?;
        let rate = reader.output_rate;
        let layout = reader.output_layout.clone();
        if !(layout == ChannelLayout::mono() || layout == ChannelLayout::stereo()) {
            return Err(Error::unsupported(
                "WAV needs named mono/stereo channels; supply an explicit mix matrix",
            ));
        }
        rate.checked_mul((layout.channels() * self.spec.format.bytes()) as u32)
            .ok_or_else(|| Error::overflow("WAV byte rate"))?;
        let (stage, file) = StagedOutput::create(path)?;
        let mut writer = WavWriter::new_with_format(
            Sink::from_file(file, stage.path.clone()),
            rate,
            layout,
            self.spec.format,
        )?;
        while let Some(block) = reader.next_buffer()? {
            writer.write_samples(&block)?;
        }
        let conversion_report = reader.conversion_report();
        let decode_report = reader.decode_report().clone();
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
        Ok(AudioSamplesWriteResult {
            output: FinishedSink::File {
                path: path.to_path_buf(),
                bytes_written,
            },
            conversion_report,
            decode_report,
        })
    }
}

/// Bounded converted sample blocks, each retaining only its own typed allocation.
pub struct AudioSamplesStream {
    request: AudioSamplesRequest,
    inner: Option<AudioStream>,
    source: SourceFingerprint,
    info: AudioInfo,
    output_rate: u32,
    output_layout: ChannelLayout,
    cancel: CancellationToken,
    failed: bool,
    conversion_report: ConversionReport,
    decode_report: AudioDecodeReport,
}
impl AudioSamplesStream {
    fn open(request: AudioSamplesRequest) -> Result<Self> {
        request.spec.validate()?;
        let (inner, source) = request.request.reader_with_source_snapshot()?;
        let info = inner.info().clone();
        let cancel = inner.cancellation();
        let output_rate = inner.rate;
        let output_layout = inner.layout.clone();
        Ok(Self {
            request,
            inner: Some(inner),
            source,
            info,
            output_rate,
            output_layout,
            cancel,
            failed: false,
            conversion_report: Default::default(),
            decode_report: Default::default(),
        })
    }
    /// Probed original source stream interpretation, before conversion/resampling.
    pub fn info(&self) -> &AudioInfo {
        &self.info
    }
    /// Effective output sample rate after continuous resampling.
    pub fn output_sample_rate(&self) -> u32 {
        self.output_rate
    }
    /// Effective output channel identities after an explicit mix.
    pub fn output_channel_layout(&self) -> &ChannelLayout {
        &self.output_layout
    }
    /// Cooperative cancellation also retires canonical working buffers on failure.
    pub fn cancellation(&self) -> CancellationToken {
        self.cancel.clone()
    }
    /// Conversion counters for successfully returned blocks, including repeats.
    pub fn conversion_report(&self) -> ConversionReport {
        self.conversion_report
    }
    /// Native decode counters, preserved after EOF/failure retires execution state.
    pub fn decode_report(&self) -> &AudioDecodeReport {
        self.inner
            .as_ref()
            .map_or(&self.decode_report, AudioStream::decode_report)
    }
    /// Return one independent converted block; failures are terminal.
    /// Source freshness is checked at open and successful finite completion.
    pub fn next_buffer(&mut self) -> Result<Option<ConvertedAudio>> {
        if self.failed {
            return Err(Error::invalid(
                "converted audio execution previously failed",
            ));
        }
        let result = self.next_inner();
        if result.is_err() {
            self.failed = true;
            self.retire();
        }
        result
    }
    fn next_inner(&mut self) -> Result<Option<ConvertedAudio>> {
        self.cancel.check()?;
        let Some(inner) = self.inner.as_mut() else {
            return Ok(None);
        };
        let Some(block) = inner.next_block()? else {
            let _metadata = self
                .request
                .request
                .asset
                .session
                .resources
                .reserve_cpu(0, 0)?;
            validate_source(&self.request.request, &self.source)?;
            self.retire();
            return Ok(None);
        };
        self.cancel.check()?;
        let converted = accounted_convert(&self.request.request, &block, self.request.spec)?;
        self.conversion_report = self.conversion_report.checked_add(converted.report())?;
        Ok(Some(converted))
    }
    fn retire(&mut self) {
        if let Some(inner) = self.inner.take() {
            self.decode_report = inner.decode_report().clone();
        }
    }
}
impl Iterator for AudioSamplesStream {
    type Item = Result<ConvertedAudio>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.failed {
            None
        } else {
            self.next_buffer().transpose()
        }
    }
}
/// Explicitly finalized WAV and observed conversion/decode evidence.
#[derive(Debug)]
pub struct AudioSamplesWriteResult {
    /// Completed destination after atomic same-directory stage replacement.
    pub output: FinishedSink,
    /// Actual clipping, integer quantization, dither, and endpoint counts.
    pub conversion_report: ConversionReport,
    /// Actual decoder work and timestamp/finalization evidence.
    pub decode_report: AudioDecodeReport,
}

fn accounted_convert(
    request: &AudioRequest,
    block: &media_core::media::AudioBlock,
    spec: SampleSpec,
) -> Result<ConvertedAudio> {
    let bytes = audio_samples::allocation_bytes(
        block.frames(),
        block.channel_layout().channels(),
        spec.format,
    )?;
    let guard = request
        .asset
        .session
        .resources
        .host_budget()
        .reserve(bytes)?;
    let converted = audio_samples::convert(block, spec)?;
    if converted.allocation_bytes() > bytes {
        return Err(Error::exhausted(
            "converted sample allocation exceeded its reserved payload",
        ));
    }
    let actual = converted.allocation_bytes();
    Ok(converted.retain(guard.shrink_to(actual)?))
}
fn validate_source(request: &AudioRequest, source: &SourceFingerprint) -> Result<()> {
    if source.is_versioned() && source != &request.asset.source.fingerprint()? {
        return Err(Error::incompatible(
            "source changed during converted audio execution",
        ));
    }
    Ok(())
}
fn validate_path(request: &AudioRequest, path: &Path) -> Result<()> {
    if !path
        .extension()
        .and_then(|v| v.to_str())
        .is_some_and(|v| v.eq_ignore_ascii_case("wav") || v.eq_ignore_ascii_case("wave"))
    {
        return Err(Error::invalid(
            "typed audio WAV output requires a .wav or .wave extension",
        ));
    }
    if let Some(input) = request.asset.source.path()
        && same_file(input, path)?
    {
        return Err(Error::invalid(
            "audio output must differ from its input file",
        ));
    }
    Ok(())
}
