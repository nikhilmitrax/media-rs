//! Bounded native demux probing, compressed packets, and explicit stream copy.
use std::collections::BTreeMap;
use std::fs::OpenOptions;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::{
    Arc, Mutex, OnceLock,
    atomic::{AtomicBool, AtomicU64, Ordering},
};

use crate::native;
use ffmpeg_next as ffmpeg;
use media_core::error::{Error, Result};
use media_core::media::{Packet, TrackId};
use media_core::time::{Time, TimeBase};
use media_io::source::{Source, SourceFingerprint};

/// Limits for native demuxer discovery, not an implicit full-file scan.
#[derive(Debug, Clone, Copy)]
pub struct ProbeOptions {
    /// Hard limit on actual source bytes read while opening/probing.
    pub max_read_bytes: u64,
    /// FFmpeg's analysis-duration hint in microseconds (not a hard time limit).
    pub analyze_duration_us: u64,
}
/// Cheap cooperative cancellation predicate checked between demux/native AVIO
/// reads. It cannot interrupt a caller's already-blocked Read implementation.
pub type ReadInterrupt = Arc<dyn Fn() -> bool + Send + Sync + 'static>;

pub(crate) fn indexed_seek_time(format: &str, pts: Time, packet_dts: Option<Time>) -> Time {
    // MPEG timestamp search compares packet decode timestamps. MP4/Matroska
    // indexes navigate presentation timestamps; passing DTS there can select
    // the preceding GOP unnecessarily. Neither value is guessed from FPS.
    if format
        .split(',')
        .any(|name| matches!(name, "mpegts" | "mpeg"))
    {
        packet_dts.unwrap_or(pts)
    } else {
        pts
    }
}
impl Default for ProbeOptions {
    fn default() -> Self {
        Self {
            max_read_bytes: 8 * 1024 * 1024,
            analyze_duration_us: 2_000_000,
        }
    }
}

/// Logical track role.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackKind {
    /// Moving or still video.
    Video,
    /// Audio samples.
    Audio,
    /// Subtitle stream.
    Subtitle,
    /// Data, attachment, or unknown stream.
    Other,
}

/// Stream metadata with original source timing and codec configuration.
#[derive(Debug, Clone)]
pub struct TrackInfo {
    /// Stable stream index within this asset.
    pub id: TrackId,
    /// Stream role.
    pub kind: TrackKind,
    /// FFmpeg codec name.
    pub codec: String,
    /// Original stream time base.
    pub time_base: TimeBase,
    /// Original stream starting presentation time when established.
    pub start: Option<Time>,
    /// Established stream duration, not a frame-rate estimate.
    pub duration: Option<Time>,
    /// Established frame count; absent when unavailable.
    pub frames: Option<u64>,
    /// Coded width for video.
    pub width: Option<u32>,
    /// Coded height for video.
    pub height: Option<u32>,
    /// Whether this video stream is attached cover art.
    pub cover_art: bool,
    /// Original display matrix, when present; not silently applied to pixels.
    pub display_matrix: Option<Vec<u8>>,
    /// Codec configuration bytes, separate from compressed packets.
    pub codec_config: Vec<u8>,
    /// Track metadata.
    pub metadata: BTreeMap<String, String>,
}

/// Asset discovery with a common timeline origin and preserved source offsets.
#[derive(Debug, Clone)]
pub struct AssetInfo {
    /// Demuxer format name.
    pub format: String,
    /// Container duration when established.
    pub duration: Option<Time>,
    /// Original source time mapped to asset time zero, if established.
    pub timeline_origin: Option<Time>,
    /// Selected source identity/version.
    pub source: SourceFingerprint,
    /// Tracks, including cover art and unsupported stream kinds.
    pub tracks: Vec<TrackInfo>,
    /// Actual bytes read during discovery.
    pub probe_bytes_read: u64,
    /// Container metadata.
    pub metadata: BTreeMap<String, String>,
}

/// Probe metadata within the configured source-read budget.
pub fn probe(source: &Source, options: ProbeOptions) -> Result<AssetInfo> {
    let opened = open_source(source, options)?;
    snapshot(
        &opened.context,
        source,
        opened.counter.total.load(Ordering::Relaxed),
    )
}

/// Probe container/bitstream headers while forbidding codec-based probing.
///
/// Some sources need a decoder to establish complete parameters; those can
/// return incomplete metadata or fail rather than decode pixels implicitly.
pub fn probe_headers(source: &Source, options: ProbeOptions) -> Result<AssetInfo> {
    let opened = open_source_without_decoding(source, options)?;
    snapshot(
        &opened.context,
        source,
        opened.counter.total.load(Ordering::Relaxed),
    )
}

pub(crate) struct OpenedInput {
    pub context: ffmpeg::format::context::Input,
    pub counter: Arc<ReadCounter>,
}
pub(crate) struct ReadCounter {
    pub total: AtomicU64,
    limit: AtomicU64,
    exhausted: AtomicBool,
    cancelled: AtomicBool,
    interrupt: Mutex<Option<ReadInterrupt>>,
}
impl ReadCounter {
    fn check(&self) -> Result<()> {
        let interrupt = self
            .interrupt
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .clone();
        if self.cancelled.load(Ordering::Relaxed)
            || interrupt.as_ref().is_some_and(|interrupt| interrupt())
        {
            self.cancelled.store(true, Ordering::Relaxed);
            return Err(Error::new(
                media_core::error::ErrorKind::Cancelled,
                "video native read cancelled",
            ));
        }
        if self.exhausted.load(Ordering::Relaxed) {
            return Err(Error::exhausted(
                "video execution exceeded its explicit source-read budget",
            )
            .with_constraint("video_source_bytes"));
        }
        Ok(())
    }
}
impl OpenedInput {
    pub(crate) fn set_interrupt(&self, interrupt: ReadInterrupt) {
        *self
            .counter
            .interrupt
            .lock()
            .unwrap_or_else(|poison| poison.into_inner()) = Some(interrupt);
    }
    pub fn allow_reading(&self) {
        self.counter.limit.store(u64::MAX, Ordering::Relaxed);
    }
    pub(crate) fn set_read_limit(&self, limit: u64) -> Result<()> {
        if limit == 0 || self.counter.total.load(Ordering::Relaxed) > limit {
            return Err(Error::exhausted(
                "video execution source-read budget is already exhausted",
            )
            .with_constraint("video_source_bytes"));
        }
        self.counter.limit.store(limit, Ordering::Relaxed);
        Ok(())
    }
    pub(crate) fn check_read_budget(&self) -> Result<()> {
        self.counter.check()
    }
}
pub(crate) fn initialize() -> Result<()> {
    static INITIALIZED: OnceLock<std::result::Result<(), String>> = OnceLock::new();
    INITIALIZED
        .get_or_init(|| ffmpeg::init().map_err(|e| e.to_string()))
        .as_ref()
        .map(|_| ())
        .map_err(|message| Error::backend("ffmpeg", message.clone()))
}
pub(crate) fn open_source(source: &Source, options: ProbeOptions) -> Result<OpenedInput> {
    open_source_impl(source, options, true, false, None)
}
pub(crate) fn open_source_without_decoding(
    source: &Source,
    options: ProbeOptions,
) -> Result<OpenedInput> {
    open_source_impl(source, options, false, false, None)
}
pub(crate) fn open_source_indexed(source: &Source, options: ProbeOptions) -> Result<OpenedInput> {
    open_source_impl(source, options, false, true, None)
}
pub(crate) fn open_source_controlled(
    source: &Source,
    options: ProbeOptions,
    indexed: bool,
    allow_probe_decoding: bool,
    interrupt: ReadInterrupt,
) -> Result<OpenedInput> {
    open_source_impl(
        source,
        options,
        allow_probe_decoding,
        indexed,
        Some(interrupt),
    )
}
fn open_source_impl(
    source: &Source,
    options: ProbeOptions,
    allow_probe_decoding: bool,
    headers_only: bool,
    interrupt: Option<ReadInterrupt>,
) -> Result<OpenedInput> {
    initialize()?;
    if options.max_read_bytes == 0 || options.analyze_duration_us > i64::MAX as u64 {
        return Err(Error::invalid(
            "FFmpeg probing needs a nonzero byte budget and valid analysis duration",
        ));
    }
    let counter = Arc::new(ReadCounter {
        total: AtomicU64::new(0),
        limit: AtomicU64::new(options.max_read_bytes),
        exhausted: AtomicBool::new(false),
        cancelled: AtomicBool::new(false),
        interrupt: Mutex::new(interrupt),
    });
    counter.check()?;
    let capabilities = source.capabilities()?;
    counter.check()?;
    let io = if capabilities.seekable {
        let reader = BudgetReader {
            reader: source.open()?,
            counter: Arc::clone(&counter),
            known_length: capabilities.known_length,
            position: 0,
        };
        ffmpeg::format::context::StreamIo::from_read_seek(reader)
    } else {
        let reader = BudgetReader {
            reader: source.open_reader()?,
            counter: Arc::clone(&counter),
            known_length: capabilities.known_length,
            position: 0,
        };
        ffmpeg::format::context::StreamIo::from_read(reader)
    }
    .map_err(backend_error)?;
    let mut dictionary = ffmpeg::Dictionary::new();
    dictionary.set(
        "probesize",
        &options
            .max_read_bytes
            .clamp(32, i64::MAX as u64)
            .to_string(),
    );
    dictionary.set("analyzeduration", &options.analyze_duration_us.to_string());
    if !allow_probe_decoding {
        // libavformat passes this whitelist to any codec it tries to open
        // during stream discovery. An impossible name rejects every decoder,
        // while compressed-header parsing and demux metadata remain available.
        dictionary.set("codec_whitelist", "__media_no_probe_decoders__");
        dictionary.set("fpsprobesize", "0");
    }
    // User-provided Read adapters own transport. Demuxers may not open network
    // subresources behind that adapter and bypass its capability/read budget.
    dictionary.set("protocol_whitelist", "file,pipe");
    dictionary.set(
        "format_whitelist",
        "mov,matroska,webm,avi,mpegts,mpeg,ogg,wav,flac,mp3,aac,aiff,asf",
    );
    let name = source.path().map(|p| p.to_string_lossy().into_owned());
    let mut context = if headers_only {
        native::input_from_headers(io, name.as_deref(), dictionary)
    } else {
        ffmpeg::format::input_from_stream(io, name.as_deref(), Some(dictionary))
            .map_err(backend_error)
    }
    .map_err(|e| {
        let error = if counter.cancelled.load(Ordering::Relaxed) {
            Error::new(
                media_core::error::ErrorKind::Cancelled,
                "video native open cancelled",
            )
            .with_cause(e)
        } else if counter.exhausted.load(Ordering::Relaxed) {
            Error::exhausted("FFmpeg probing exceeded the configured source-read budget")
                .with_constraint("max_read_bytes")
                .with_cause(e)
        } else {
            e
        };
        error.with_source(source.label()).with_operation("probe")
    })?;
    if headers_only {
        let mut incomplete = false;
        for stream in context.streams() {
            let parameters = stream.parameters();
            if parameters.medium() == ffmpeg::media::Type::Video {
                let (width, height, config) = native::parameter_description(&parameters)?;
                incomplete |= width.is_none()
                    || height.is_none()
                    || (matches!(
                        parameters.id(),
                        ffmpeg::codec::Id::H264 | ffmpeg::codec::Id::HEVC
                    ) && config.is_empty());
            }
        }
        if incomplete {
            let discovered = native::discover_compressed_headers(&mut context);
            counter.check()?;
            discovered?;
        }
    }
    if counter.exhausted.load(Ordering::Relaxed) {
        return Err(Error::exhausted(
            "FFmpeg probing reached the source-read budget before completing discovery",
        )
        .with_constraint("max_read_bytes")
        .with_source(source.label()));
    }
    counter.check()?;
    Ok(OpenedInput { context, counter })
}
pub(crate) fn snapshot(
    context: &ffmpeg::format::context::Input,
    source: &Source,
    probe_bytes_read: u64,
) -> Result<AssetInfo> {
    let mut tracks = Vec::new();
    for stream in context.streams() {
        let time_base = time_base(stream.time_base())?;
        let parameters = stream.parameters();
        let (width, height, codec_config) = native::parameter_description(&parameters)?;
        let kind = match parameters.medium() {
            ffmpeg::media::Type::Video => TrackKind::Video,
            ffmpeg::media::Type::Audio => TrackKind::Audio,
            ffmpeg::media::Type::Subtitle => TrackKind::Subtitle,
            _ => TrackKind::Other,
        };
        tracks.push(TrackInfo {
            id: TrackId(
                u32::try_from(stream.index())
                    .map_err(|_| Error::overflow("track index exceeds u32"))?,
            ),
            kind,
            codec: parameters.id().name().to_owned(),
            time_base,
            start: optional_ticks(stream.start_time(), time_base)?,
            duration: if stream.duration() > 0 {
                Some(time_base.time(stream.duration())?)
            } else {
                None
            },
            frames: if stream.frames() > 0 {
                Some(stream.frames() as u64)
            } else {
                None
            },
            width,
            height,
            cover_art: stream
                .disposition()
                .contains(ffmpeg::format::stream::Disposition::ATTACHED_PIC),
            display_matrix: stream
                .side_data()
                .find(|data| data.kind() == ffmpeg::codec::packet::side_data::Type::DisplayMatrix)
                .map(|data| data.data().to_vec()),
            codec_config,
            metadata: stream
                .metadata()
                .iter()
                .map(|(k, v)| (k.to_owned(), v.to_owned()))
                .collect(),
        });
    }
    let timeline_origin = optional_ticks(
        native::input_start_time(context),
        TimeBase::new(1, 1_000_000)?,
    )?
    .or_else(|| tracks.iter().filter_map(|t| t.start).min());
    Ok(AssetInfo {
        format: context.format().name().to_owned(),
        duration: if context.duration() > 0 {
            Some(Time::new(context.duration(), 1_000_000)?)
        } else {
            None
        },
        timeline_origin,
        source: source.fingerprint()?,
        tracks,
        probe_bytes_read,
        metadata: context
            .metadata()
            .iter()
            .map(|(k, v)| (k.to_owned(), v.to_owned()))
            .collect(),
    })
}
pub(crate) fn time_base(value: ffmpeg::Rational) -> Result<TimeBase> {
    let (n, d) = (value.numerator(), value.denominator());
    if n <= 0 || d <= 0 {
        return Err(Error::invalid("FFmpeg stream has an invalid time base"));
    }
    TimeBase::new(n as u32, d as u32)
}
pub(crate) fn optional_ticks(value: i64, base: TimeBase) -> Result<Option<Time>> {
    if value == ffmpeg::ffi::AV_NOPTS_VALUE {
        Ok(None)
    } else {
        Ok(Some(base.time(value)?))
    }
}
pub(crate) fn backend_error(error: ffmpeg::Error) -> Error {
    Error::backend("ffmpeg", error.to_string()).with_cause(error)
}

/// Incremental demuxer returning original compressed packets without decoding.
pub struct PacketReader {
    opened: OpenedInput,
    info: AssetInfo,
    finished: bool,
}

/// Native packet side data, retained independently of decoder state.
#[derive(Debug, Clone)]
pub struct PacketSideData {
    /// Native side-data kind, for example SkipSamples or NewExtraData.
    pub kind: String,
    /// Unmodified native side-data bytes.
    pub data: Arc<[u8]>,
}

/// Complete packet projection including native flags and side data.
#[derive(Debug, Clone)]
pub struct PacketWithSideData {
    /// Portable payload and timing.
    pub packet: Packet,
    /// Unmodified FFmpeg packet flag bits.
    pub flags: i32,
    /// Native configuration, trimming, and other per-packet data.
    pub side_data: Vec<PacketSideData>,
}
impl PacketReader {
    /// Open an independent source cursor, then allow incremental reads.
    pub fn open(source: &Source, options: ProbeOptions) -> Result<Self> {
        let opened = open_source(source, options)?;
        let info = snapshot(
            &opened.context,
            source,
            opened.counter.total.load(Ordering::Relaxed),
        )?;
        opened.allow_reading();
        Ok(Self {
            opened,
            info,
            finished: false,
        })
    }
    /// Open from container headers without opening CPU codec discovery workers.
    /// Incomplete headers remain an error/unknown instead of CPU video probing.
    pub fn open_headers(source: &Source, options: ProbeOptions) -> Result<Self> {
        let opened = open_source_without_decoding(source, options)?;
        let info = snapshot(
            &opened.context,
            source,
            opened.counter.total.load(Ordering::Relaxed),
        )?;
        opened.allow_reading();
        Ok(Self {
            opened,
            info,
            finished: false,
        })
    }
    /// Track codec configurations and source timing.
    pub fn info(&self) -> &AssetInfo {
        &self.info
    }
    /// Own an independent codec/extradata copy for an external decoder client.
    pub fn codec_parameters(
        &self,
        track: TrackId,
    ) -> Result<crate::packet_decode::CodecParameters> {
        let stream = self
            .opened
            .context
            .stream(track.0 as usize)
            .ok_or_else(|| Error::invalid("codec parameters requested for an absent track"))?;
        crate::packet_decode::clone_parameters(&stream.parameters(), 4 * 1024 * 1024)
    }
    /// Selected video configuration with explicit original time base and origin.
    pub fn video_decoder_config(
        &self,
        track: TrackId,
    ) -> Result<crate::packet_decode::VideoDecoderConfig> {
        let info = self
            .info
            .tracks
            .iter()
            .find(|info| info.id == track && info.kind == TrackKind::Video)
            .ok_or_else(|| {
                Error::invalid("decoder configuration requires an existing video track")
            })?;
        Ok(crate::packet_decode::VideoDecoderConfig {
            parameters: self.codec_parameters(track)?,
            track,
            time_base: info.time_base,
            timeline_origin: self.info.timeline_origin,
            orientation: info
                .display_matrix
                .as_deref()
                .map(native::display_orientation)
                .transpose()?
                .unwrap_or(media_core::media::Orientation::Normal),
            first_ordinal: None,
        })
    }
    /// Read a compressed packet; errors never become a successful EOF.
    ///
    /// Packet PTS/DTS retain the original source stream timeline. Container
    /// origin mapping is available through `info().timeline_origin`.
    /// Use `next_packet_with_side_data` when native flags/side data are needed.
    pub fn next_packet(&mut self) -> Result<Option<Packet>> {
        Ok(self.next_packet_with_side_data()?.map(|value| value.packet))
    }
    /// Read the full packet projection without losing codec/trim side data.
    pub fn next_packet_with_side_data(&mut self) -> Result<Option<PacketWithSideData>> {
        if self.finished {
            return Ok(None);
        }
        let mut packet = ffmpeg::Packet::empty();
        match packet.read(&mut self.opened.context) {
            Ok(()) => {}
            Err(ffmpeg::Error::Eof) => {
                self.finished = true;
                return Ok(None);
            }
            Err(error) => return Err(backend_error(error).with_operation("demux")),
        }
        let track = self
            .info
            .tracks
            .get(packet.stream())
            .ok_or_else(|| Error::invalid("packet refers to an undiscovered track"))?;
        let portable = Packet {
            track: track.id,
            data: Arc::from(packet.data().unwrap_or_default()),
            pts: packet.pts().map(|v| track.time_base.time(v)).transpose()?,
            dts: packet.dts().map(|v| track.time_base.time(v)).transpose()?,
            duration: if packet.duration() > 0 {
                Some(track.time_base.time(packet.duration())?)
            } else {
                None
            },
            keyframe: packet.is_key(),
        };
        Ok(Some(PacketWithSideData {
            packet: portable,
            flags: packet.flags().bits(),
            side_data: packet
                .side_data()
                .map(|data| PacketSideData {
                    kind: format!("{:?}", data.kind()),
                    data: Arc::from(data.data()),
                })
                .collect(),
        }))
    }
}

/// Completed compressed stream-copy result.
#[derive(Debug, Clone)]
pub struct RemuxReport {
    /// Finalized file location.
    pub destination: PathBuf,
    /// Input track IDs copied, in output stream order.
    pub tracks: Vec<TrackId>,
    /// Number of packets written.
    pub packets_written: u64,
    /// Source identity/version checked before opening and after finalization.
    pub source: SourceFingerprint,
    /// Actual encoded source reads, including the one input's probe/rereads.
    pub bytes_read: u64,
    /// Reads used to discover streams on that same input, never a second cursor.
    pub probe_bytes_read: u64,
}

/// Copy explicitly selected compressed streams and finalize the muxer.
///
/// No selected stream is silently dropped or re-encoded. Existing destinations
/// are not overwritten. Incompatibility is checked before creating the file;
/// a later error names the partial output, which may require removal by caller.
pub fn remux(source: &Source, destination: &Path, tracks: &[TrackId]) -> Result<RemuxReport> {
    remux_with_options(source, destination, Some(tracks), ProbeOptions::default())
}

/// Copy selected or all compressed streams using one independently opened input.
///
/// None resolves all logical tracks from the same context used for packet copy;
/// callers must not probe/reopen merely to discover that list. Some(empty),
/// duplicate/unknown IDs and incompatible output codecs fail before creating
/// output. Existing destinations are never overwritten. A later I/O/version
/// failure discloses the destination as possibly partial, just like remux().
pub fn remux_with_options(
    source: &Source,
    destination: &Path,
    tracks: Option<&[TrackId]>,
    options: ProbeOptions,
) -> Result<RemuxReport> {
    if tracks.is_some_and(|tracks| tracks.is_empty()) {
        return Err(Error::invalid(
            "remux needs at least one explicitly selected track",
        ));
    }
    if let Some(tracks) = tracks {
        let mut seen = std::collections::BTreeSet::new();
        if tracks.iter().any(|id| !seen.insert(id.0)) {
            return Err(Error::invalid("remux selection contains a duplicate track"));
        }
    }
    let name = destination
        .to_str()
        .ok_or_else(|| Error::invalid("FFmpeg destination path must be UTF-8"))?;
    let fingerprint = source.fingerprint()?;
    let mut opened = open_source(source, options)?;
    let probe_bytes_read = opened.counter.total.load(Ordering::Relaxed);
    if fingerprint.is_versioned() && source.fingerprint()? != fingerprint {
        return Err(Error::incompatible("source changed during remux discovery")
            .with_source(source.label()));
    }
    opened.allow_reading();
    let all_tracks = if tracks.is_none() {
        opened
            .context
            .streams()
            .map(|stream| {
                u32::try_from(stream.index())
                    .map(TrackId)
                    .map_err(|_| Error::overflow("remux logical stream index exceeds u32"))
            })
            .collect::<Result<Vec<_>>>()?
    } else {
        Vec::new()
    };
    let tracks = tracks.unwrap_or(&all_tracks);
    if tracks.is_empty() {
        return Err(Error::invalid("remux input has no streams"));
    }
    let mut chosen = Vec::new();
    for id in tracks {
        let stream = opened
            .context
            .stream(id.0 as usize)
            .ok_or_else(|| Error::invalid("remux selected an unknown track"))?;
        native::check_mux_codec(destination, stream.parameters().id())?;
        chosen.push(*id);
    }
    let file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .read(true)
        .open(destination)
        .map_err(Error::io)?;
    let sync = file.try_clone().map_err(Error::io)?;
    let output_io =
        ffmpeg::format::context::StreamIo::from_write_seek(file).map_err(backend_error)?;
    let mut output =
        ffmpeg::format::output_to_stream(output_io, Some(name), None).map_err(backend_error)?;
    let result = (|| -> Result<u64> {
        let mut mapping = BTreeMap::new();
        let mut input_bases = BTreeMap::new();
        for (index, id) in chosen.iter().enumerate() {
            let input = opened
                .context
                .stream(id.0 as usize)
                .ok_or_else(|| Error::invalid("selected track disappeared"))?;
            input_bases.insert(id.0 as usize, input.time_base());
            let mut stream = output
                .add_stream(ffmpeg::encoder::find(ffmpeg::codec::Id::None))
                .map_err(backend_error)?;
            stream.set_parameters(input.parameters());
            stream.set_time_base(input.time_base());
            stream.set_metadata(input.metadata().to_owned());
            native::clear_codec_tag(&mut stream);
            mapping.insert(id.0 as usize, index);
        }
        output.set_metadata(opened.context.metadata().to_owned());
        output.write_header().map_err(backend_error)?;
        let mut packets = 0u64;
        loop {
            let mut packet = ffmpeg::Packet::empty();
            match packet.read(&mut opened.context) {
                Ok(()) => {}
                Err(ffmpeg::Error::Eof) => break,
                Err(error) => return Err(backend_error(error)),
            }
            let Some(index) = mapping.get(&packet.stream()).copied() else {
                continue;
            };
            let time_base = output
                .stream(index)
                .ok_or_else(|| Error::invalid("output stream missing"))?
                .time_base();
            packet.rescale_ts(input_bases[&packet.stream()], time_base);
            packet.set_position(-1);
            packet.set_stream(index);
            packet
                .write_interleaved(&mut output)
                .map_err(backend_error)?;
            packets = packets
                .checked_add(1)
                .ok_or_else(|| Error::overflow("packet counter overflow"))?;
        }
        output.write_trailer().map_err(backend_error)?;
        native::flush_output(&mut output)?;
        Ok(packets)
    })();
    drop(output);
    let packets_written = result.map_err(|error| {
        error
            .with_source(destination.display().to_string())
            .with_operation("remux; partial destination may exist")
    })?;
    sync.sync_all().map_err(|error| {
        Error::io(error)
            .with_source(destination.display().to_string())
            .with_operation("finalize remux; destination may exist")
    })?;
    if fingerprint.is_versioned()
        && source.fingerprint().map_err(|error| {
            error
                .with_source(destination.display().to_string())
                .with_operation("remux source-version validation; destination may exist")
        })? != fingerprint
    {
        return Err(Error::incompatible(
            "source changed during remux; destination may contain a partial or inconsistent copy",
        )
        .with_source(destination.display().to_string())
        .with_operation("remux source-version validation"));
    }
    Ok(RemuxReport {
        destination: destination.to_owned(),
        tracks: chosen,
        packets_written,
        source: fingerprint,
        bytes_read: opened.counter.total.load(Ordering::Relaxed),
        probe_bytes_read,
    })
}

struct BudgetReader<T> {
    reader: T,
    counter: Arc<ReadCounter>,
    known_length: Option<u64>,
    position: u64,
}
impl<T: Read> Read for BudgetReader<T> {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        self.counter
            .check()
            .map_err(|error| io::Error::other(error.to_string()))?;
        if bytes.is_empty() {
            return Ok(0);
        }
        let used = self.counter.total.load(Ordering::Relaxed);
        let remaining = self
            .counter
            .limit
            .load(Ordering::Relaxed)
            .saturating_sub(used);
        if remaining == 0 {
            if self.known_length == Some(self.position) {
                return Ok(0);
            }
            self.counter.exhausted.store(true, Ordering::Relaxed);
            return Err(io::Error::other("FFmpeg probe source-read budget exceeded"));
        }
        let allowed = bytes
            .len()
            .min(usize::try_from(remaining).unwrap_or(usize::MAX));
        let count = self.reader.read(&mut bytes[..allowed])?;
        self.counter
            .total
            .fetch_add(count as u64, Ordering::Relaxed);
        self.position = self
            .position
            .checked_add(count as u64)
            .ok_or_else(|| io::Error::other("source position overflow"))?;
        Ok(count)
    }
}
impl<T: Read + Seek> Seek for BudgetReader<T> {
    fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
        self.counter
            .check()
            .map_err(|error| io::Error::other(error.to_string()))?;
        let current = self.reader.seek(position)?;
        self.position = current;
        if matches!(position, SeekFrom::End(0)) {
            self.known_length = Some(current);
        }
        Ok(current)
    }
}

#[cfg(test)]
mod read_budget_tests {
    use super::*;
    fn counter(limit: u64, interrupt: Option<ReadInterrupt>) -> Arc<ReadCounter> {
        Arc::new(ReadCounter {
            total: AtomicU64::new(0),
            limit: AtomicU64::new(limit),
            exhausted: AtomicBool::new(false),
            cancelled: AtomicBool::new(false),
            interrupt: Mutex::new(interrupt),
        })
    }
    #[test]
    fn exact_known_eof_is_not_a_budget_overrun_but_rereading_is() {
        let counter = counter(4, None);
        let mut reader = BudgetReader {
            reader: io::Cursor::new(vec![1, 2, 3, 4]),
            counter: counter.clone(),
            known_length: Some(4),
            position: 0,
        };
        reader.read_exact(&mut [0; 4]).unwrap();
        assert_eq!(reader.read(&mut [0; 1]).unwrap(), 0);
        assert_eq!(counter.total.load(Ordering::SeqCst), 4);
        assert!(!counter.exhausted.load(Ordering::SeqCst));
        reader.seek(SeekFrom::Start(0)).unwrap();
        assert!(reader.read(&mut [0; 1]).is_err());
        assert_eq!(counter.total.load(Ordering::SeqCst), 4);
        assert!(counter.exhausted.load(Ordering::SeqCst));
    }
    #[test]
    fn cancellation_is_checked_before_underlying_reads_or_seeks() {
        let cancelled = Arc::new(AtomicBool::new(false));
        let token = cancelled.clone();
        let counter = counter(4, Some(Arc::new(move || token.load(Ordering::SeqCst))));
        let mut reader = BudgetReader {
            reader: io::Cursor::new(vec![1, 2, 3, 4]),
            counter: counter.clone(),
            known_length: Some(4),
            position: 0,
        };
        cancelled.store(true, Ordering::SeqCst);
        assert!(reader.read(&mut [0; 1]).is_err());
        assert!(reader.seek(SeekFrom::End(0)).is_err());
        assert_eq!(reader.reader.position(), 0);
        assert_eq!(counter.total.load(Ordering::SeqCst), 0);
        assert_eq!(
            counter.check().unwrap_err().kind(),
            media_core::error::ErrorKind::Cancelled
        );
    }
}
