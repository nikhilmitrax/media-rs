//! Decoder adapter retaining the established canonical continuous audio contract.
use super::*;
enum Decoder {
    Native(Box<AudioReader>),
    #[cfg(feature = "ffmpeg-audio")]
    Ffmpeg(Box<media_backend_ffmpeg::audio::AudioReader>),
}
pub(super) struct DecoderAdapter {
    decoder: Decoder,
    info: AudioInfo,
    #[cfg(feature = "ffmpeg-audio")]
    report: AudioDecodeReport,
}
impl DecoderAdapter {
    pub(super) fn open(request: &AudioRequest, options: AudioDecodeOptions) -> Result<Self> {
        match request.backend {
            AudioBackend::Native => {
                let reader = AudioReader::open(&request.asset.source, options)?;
                if request.semantics == AudioSemantics::Semantic
                    && reader.format_short_name() == "isomp4"
                {
                    return Err(Error::unsupported(
                        "native ISO MP4 audio does not establish edit-list/priming/trailing semantics; enable ffmpeg-audio and select .ffmpeg(), or explicitly choose .raw_decoded()",
                    ));
                }
                Ok(Self {
                    info: reader.info().clone(),
                    decoder: Decoder::Native(Box::new(reader)),
                    #[cfg(feature = "ffmpeg-audio")]
                    report: Default::default(),
                })
            }
            #[cfg(feature = "ffmpeg-audio")]
            AudioBackend::Ffmpeg => {
                use media_backend_ffmpeg::audio::AudioTrimPolicy;
                if request.options.corrupt == media_backend_native::audio::CorruptAudioPolicy::Skip
                {
                    return Err(Error::unsupported(
                        "FFmpeg audio requires strict corruption failure; native packet-skip recovery is unsupported",
                    ));
                }
                let mut configured = request.ffmpeg_options;
                configured.max_buffered_samples = options.max_buffered_samples;
                configured.max_packet_bytes = options.max_packet_bytes;
                configured.max_block_frames = options.max_block_frames;
                configured.probe.max_read_bytes = options.max_probe_bytes;
                configured.trim = match request.semantics {
                    AudioSemantics::Semantic => AudioTrimPolicy::Semantic,
                    AudioSemantics::RawDecoded => AudioTrimPolicy::RawDecoded,
                };
                let reader = media_backend_ffmpeg::audio::AudioReader::open(
                    &request.asset.source,
                    configured,
                )?;
                let discovered = reader.info();
                let info = AudioInfo {
                    track: discovered.track.0,
                    sample_rate: discovered.sample_rate,
                    layout: discovered.layout.clone(),
                    frames: discovered.frames,
                    start: discovered.start,
                    timestamp_quantum: discovered.timestamp_quantum,
                };
                Ok(Self {
                    info,
                    decoder: Decoder::Ffmpeg(Box::new(reader)),
                    report: Default::default(),
                })
            }
        }
    }
    pub(super) fn info(&self) -> &AudioInfo {
        &self.info
    }
    pub(super) fn next_block(&mut self, cancel: &CancellationToken) -> Result<Option<AudioBlock>> {
        match &mut self.decoder {
            Decoder::Native(reader) => {
                cancel.check()?;
                reader.next_block()
            }
            #[cfg(feature = "ffmpeg-audio")]
            Decoder::Ffmpeg(reader) => {
                let result = reader.next_block_checked(|| cancel.check());
                let report = reader.report();
                self.report = AudioDecodeReport {
                    decoded_packets: report.packets_sent,
                    skipped_packets: 0,
                    timestamp_adjustments: report.timestamp_adjustments,
                    max_timestamp_adjustment: report.max_timestamp_adjustment,
                    discontinuities: 0,
                    output_frames: report.output_frames,
                    probe_bytes: report.probe_bytes,
                    verified: None,
                    finished: report.finished,
                };
                result
            }
        }
    }
    pub(super) fn report(&self) -> &AudioDecodeReport {
        match &self.decoder {
            Decoder::Native(reader) => reader.report(),
            #[cfg(feature = "ffmpeg-audio")]
            Decoder::Ffmpeg(_) => &self.report,
        }
    }
    #[cfg(feature = "ffmpeg-audio")]
    pub(super) fn ffmpeg_report(&self) -> Option<&media_backend_ffmpeg::audio::AudioDecodeReport> {
        match &self.decoder {
            Decoder::Native(_) => None,
            Decoder::Ffmpeg(reader) => Some(reader.report()),
        }
    }
    #[cfg(feature = "ffmpeg-audio")]
    pub(super) fn ffmpeg_info(&self) -> Option<&media_backend_ffmpeg::audio::AudioInfo> {
        match &self.decoder {
            Decoder::Native(_) => None,
            Decoder::Ffmpeg(reader) => Some(reader.info()),
        }
    }
}
