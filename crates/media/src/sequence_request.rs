//! Bounded animation canvases and independent TIFF pages with retained budgets.
//!
//! These are finite eager requests, distinct from a still image. Native codec
//! scratch limits are best effort; opaque codec/private heap use is unmeasured.
//! Encoded input, conservative allowed output, frame metadata, and retained
//! decoded surfaces are reserved against the shared session before decoding.

use crate::{
    image_request::{decode_limits, read_encoded},
    request::Asset,
};
use media_backend_native::{
    image::MetadataPolicy,
    image_sequence::{self, DisplayFrame, ImagePage},
    image_sequence_encode::{self as sequence_encode, EncodedSequence},
};
use media_core::{
    error::{Error, Result},
    media::Image,
    time::Time,
};
use media_io::sink::{FinishedSink, Sink};
use media_pipeline::{
    diagnostics::ExecutionReport,
    session::{BatchReservation, HostReservation},
};
use std::{io::Write, path::Path, sync::Arc};

pub use media_backend_native::image_sequence::{LoopCount, SequenceLimits};
pub use media_backend_native::image_sequence_encode::{
    AnimationEncodeOptions, AnimationFormat, SequenceEncodeLimits, TiffEncodeOptions,
};

/// Lazy GIF/APNG/animated WebP display-canvas extraction.
#[derive(Debug, Clone)]
pub struct AnimationRequest {
    asset: Asset,
    limits: SequenceLimits,
}
impl AnimationRequest {
    pub(crate) fn new(asset: Asset) -> Self {
        Self {
            asset,
            limits: Default::default(),
        }
    }
    /// Replace per-frame, retained-byte, metadata, and frame-count limits.
    /// Session bounds remain authoritative and may lower these maxima.
    pub fn limits(mut self, limits: SequenceLimits) -> Self {
        self.limits = limits;
        self
    }
    /// Decode every composited display canvas, retaining exact timing and loops.
    /// Still PNG/WebP objects and unsupported precision are errors.
    pub fn read(&self) -> Result<AnimationBatch> {
        let _cpu = self.asset.session.resources.reserve_cpu(1, 1)?;
        let source = self.asset.source.fingerprint()?;
        let batch = Arc::new(self.asset.session.resources.reserve_batch()?);
        let input = read_encoded(&self.asset.source, &self.asset.session.resources)?;
        let (limits, mut reservation) = reserve_output(
            &self.asset,
            self.limits,
            std::mem::size_of::<DisplayFrame>(),
        )?;
        let mut animation = image_sequence::decode_animation(&input.bytes, limits)?;
        let bytes_read = input.bytes.len() as u64;
        verify_source(&self.asset, &source)?;
        let metadata = Arc::new(
            reservation.split_off(
                animation
                    .frames
                    .capacity()
                    .checked_mul(std::mem::size_of::<DisplayFrame>())
                    .and_then(|bytes| bytes.checked_add(1024))
                    .ok_or_else(|| Error::overflow("animation result metadata"))?,
            )?,
        );
        for frame in &mut animation.frames {
            frame.image = retain_surface(&frame.image, &mut reservation, &batch, &metadata)?;
        }
        drop(reservation);
        drop(input);
        let count = animation.frames.len() as u64;
        Ok(AnimationBatch {
            source,
            frames: animation.frames,
            loop_count: animation.loop_count,
            total_duration: animation.total_duration,
            effective_limits: limits,
            report: report(&self.asset, bytes_read, count, Vec::new()),
            _batch: batch,
            _metadata: metadata,
        })
    }
    /// Encode every composited canvas with exact timing and explicit metadata policy.
    /// The returned bytes retain their session host reservation until dropped.
    pub fn encode(&self, mut options: AnimationEncodeOptions) -> Result<SequenceEncodedResult> {
        let mut batch = self.read()?;
        let _cpu = self.asset.session.resources.reserve_cpu(1, 1)?;
        let animation = image_sequence::Animation {
            frames: std::mem::take(&mut batch.frames),
            loop_count: batch.loop_count,
            total_duration: batch.total_duration,
        };
        let mut limits = options.limits;
        let reservation = reserve_encoding(&self.asset, &mut limits, |limits| {
            sequence_encode::animation_working_bytes(
                &animation,
                AnimationEncodeOptions { limits, ..options },
            )
        })?;
        options.limits = limits;
        let encoded = sequence_encode::encode_animation(&animation, options)?;
        encoded_result(&self.asset, encoded, batch.report, reservation)
    }
    /// Encode completely before opening the destination, then flush and finalize it.
    /// Preflight, decode, timing and output-size failures preserve existing files.
    pub fn write(
        &self,
        path: impl AsRef<Path>,
        options: AnimationEncodeOptions,
    ) -> Result<SequenceWriteResult> {
        write_encoded(self.encode(options)?, path.as_ref())
    }
}

/// Lazy TIFF page extraction; pages do not acquire invented animation timing.
#[derive(Debug, Clone)]
pub struct PagesRequest {
    asset: Asset,
    limits: SequenceLimits,
    metadata_policy: MetadataPolicy,
}
impl PagesRequest {
    pub(crate) fn new(asset: Asset) -> Self {
        Self {
            asset,
            limits: Default::default(),
            metadata_policy: MetadataPolicy::RequirePreservation,
        }
    }
    /// Replace per-page, retained-byte, metadata, and page-count limits.
    pub fn limits(mut self, limits: SequenceLimits) -> Self {
        self.limits = limits;
        self
    }
    /// Permit and report metadata which the TIFF page projection cannot retain.
    pub fn allow_metadata_loss(mut self) -> Self {
        self.metadata_policy = MetadataPolicy::AllowLoss;
        self
    }
    /// Decode independent native-precision pages with their own geometry/orientation.
    pub fn read(&self) -> Result<PageBatch> {
        let _cpu = self.asset.session.resources.reserve_cpu(1, 1)?;
        let source = self.asset.source.fingerprint()?;
        let batch = Arc::new(self.asset.session.resources.reserve_batch()?);
        let input = read_encoded(&self.asset.source, &self.asset.session.resources)?;
        let (limits, mut reservation) =
            reserve_output(&self.asset, self.limits, std::mem::size_of::<ImagePage>())?;
        let mut pages = image_sequence::decode_pages(&input.bytes, limits)?;
        if !pages.metadata_loss.is_empty()
            && self.metadata_policy == MetadataPolicy::RequirePreservation
        {
            return Err(Error::incompatible(format!(
                "TIFF page projection would lose metadata {:?}; explicitly allow metadata loss",
                pages.metadata_loss
            )));
        }
        verify_source(&self.asset, &source)?;
        let metadata_bytes = pages
            .pages
            .capacity()
            .checked_mul(std::mem::size_of::<ImagePage>())
            .and_then(|bytes| bytes.checked_add(1024))
            .and_then(|bytes| {
                pages
                    .metadata_loss
                    .iter()
                    .try_fold(bytes, |bytes, warning| {
                        bytes.checked_add(warning.capacity() + std::mem::size_of::<String>())
                    })
            })
            .ok_or_else(|| Error::overflow("page result metadata"))?;
        let metadata = Arc::new(reservation.split_off(metadata_bytes)?);
        for page in &mut pages.pages {
            page.image = retain_surface(&page.image, &mut reservation, &batch, &metadata)?;
        }
        let bytes_read = input.bytes.len() as u64;
        drop(reservation);
        drop(input);
        let count = pages.pages.len() as u64;
        Ok(PageBatch {
            source,
            pages: pages.pages,
            effective_limits: limits,
            report: report(&self.asset, bytes_read, count, pages.metadata_loss),
            _batch: batch,
            _metadata: metadata,
        })
    }
    /// Encode all native-precision pages; alpha association and orientation remain explicit.
    pub fn encode_tiff(&self, mut options: TiffEncodeOptions) -> Result<SequenceEncodedResult> {
        let batch = self.read()?;
        let _cpu = self.asset.session.resources.reserve_cpu(1, 1)?;
        let mut limits = options.limits;
        let reservation = reserve_encoding(&self.asset, &mut limits, |limits| {
            sequence_encode::tiff_working_bytes(
                &batch.pages,
                TiffEncodeOptions { limits, ..options },
            )
        })?;
        options.limits = limits;
        let encoded = sequence_encode::encode_tiff(&batch.pages, options)?;
        encoded_result(&self.asset, encoded, batch.report, reservation)
    }
    /// Encode every page before opening the destination, then explicitly finalize.
    /// Read projection and output metadata policies are separately enforced.
    pub fn write_tiff(
        &self,
        path: impl AsRef<Path>,
        options: TiffEncodeOptions,
    ) -> Result<SequenceWriteResult> {
        write_encoded(self.encode_tiff(options)?, path.as_ref())
    }
}

/// Complete encoded sequence bytes with a retained session payload reservation.
pub struct SequenceEncodedResult {
    bytes: Vec<u8>,
    /// Decode/encode counts and explicit metadata/allocation diagnostics.
    pub report: ExecutionReport,
    _reservation: HostReservation,
}
impl std::fmt::Debug for SequenceEncodedResult {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SequenceEncodedResult")
            .field("bytes", &self.bytes.len())
            .field("report", &self.report)
            .finish()
    }
}
impl SequenceEncodedResult {
    /// Borrow finalized bytes without releasing their session accounting.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}
/// Evidence that all frames/pages were encoded and the destination was flushed.
#[derive(Debug)]
pub struct SequenceWriteResult {
    /// Explicitly finalized file destination.
    pub output: FinishedSink,
    /// Decode/encode counts and metadata-loss diagnostics.
    pub report: ExecutionReport,
}

fn reserve_encoding(
    asset: &Asset,
    limits: &mut SequenceEncodeLimits,
    estimate: impl Fn(SequenceEncodeLimits) -> Result<usize>,
) -> Result<HostReservation> {
    let budget = asset.session.resources.host_budget();
    let available = budget.limit().saturating_sub(budget.current());
    let overhead = estimate(SequenceEncodeLimits {
        max_encoded_bytes: 1,
        max_working_bytes: usize::MAX,
        ..*limits
    })?
    .checked_sub(1)
    .ok_or_else(|| Error::overflow("sequence encoder overhead"))?;
    limits.max_working_bytes = limits.max_working_bytes.min(available);
    limits.max_encoded_bytes = limits
        .max_encoded_bytes
        .min(asset.session.limits().encoded_bytes)
        .min(limits.max_working_bytes.saturating_sub(overhead));
    if limits.max_encoded_bytes == 0 {
        return Err(Error::exhausted(
            "session has no remaining sequence encoding payload capacity",
        ));
    }
    budget.reserve(estimate(*limits)?)
}
fn encoded_result(
    asset: &Asset,
    encoded: EncodedSequence,
    mut report: ExecutionReport,
    reservation: HostReservation,
) -> Result<SequenceEncodedResult> {
    report.warnings.extend(encoded.metadata_loss);
    report.warnings.push("sequence encoder private compression tables/scratch heap are unmeasured; bounded host reservations cover allowed encoded extent and typed/indexed working payloads".into());
    report.output_count = encoded.count as u64;
    report.peak_host_bytes = asset.session.resources.host_budget().peak();
    let reservation = reservation.shrink_to(encoded.bytes.len())?;
    Ok(SequenceEncodedResult {
        bytes: encoded.bytes,
        report,
        _reservation: reservation,
    })
}
fn write_encoded(encoded: SequenceEncodedResult, path: &Path) -> Result<SequenceWriteResult> {
    let mut sink = Sink::file(path)?;
    sink.write_all(encoded.bytes()).map_err(Error::io)?;
    Ok(SequenceWriteResult {
        output: sink.finish()?,
        report: encoded.report,
    })
}

/// Finite composited animation; extracted images retain their pixel/metadata charges.
#[derive(Debug)]
pub struct AnimationBatch {
    /// Immutable source identity/version captured before reading this sequence.
    pub source: media_io::source::SourceFingerprint,
    /// Full display canvases, with exact original durations and frame indices.
    pub frames: Vec<DisplayFrame>,
    /// Total playback count, including the first play.
    pub loop_count: LoopCount,
    /// Exact sum of encoded durations for one play.
    pub total_duration: Time,
    /// Session-clamped maxima actually passed to the native decoder.
    pub effective_limits: SequenceLimits,
    /// Actual encoded input/output counts and explicit allocation limitations.
    pub report: ExecutionReport,
    _batch: Arc<BatchReservation>,
    _metadata: Arc<HostReservation>,
}
/// Finite independent TIFF pages with preserved supported numeric precision.
#[derive(Debug)]
pub struct PageBatch {
    /// Immutable source identity/version captured before reading these pages.
    pub source: media_io::source::SourceFingerprint,
    /// Independent page geometry, pixel format, orientation, and alpha association.
    pub pages: Vec<ImagePage>,
    /// Session-clamped maxima actually passed to the native decoder.
    pub effective_limits: SequenceLimits,
    /// Actual input/output counts and allowed metadata-loss diagnostics.
    pub report: ExecutionReport,
    _batch: Arc<BatchReservation>,
    _metadata: Arc<HostReservation>,
}

fn reserve_output(
    asset: &Asset,
    mut limits: SequenceLimits,
    structure_bytes: usize,
) -> Result<(SequenceLimits, HostReservation)> {
    if limits.max_frames == 0 || limits.max_total_decoded_bytes == 0 {
        return Err(Error::invalid("sequence bounds must be nonzero"));
    }
    let session = &asset.session.resources;
    let session_limits = decode_limits(session);
    limits.decode.max_encoded_bytes = limits
        .decode
        .max_encoded_bytes
        .min(session_limits.max_encoded_bytes);
    limits.decode.max_pixels = limits.decode.max_pixels.min(session_limits.max_pixels);
    let available = session
        .host_budget()
        .limit()
        .saturating_sub(session.host_budget().current());
    // Header/guard/diagnostic allowance is separate from native's pixel+ICC/EXIF
    // bound. Leave at least half of the remaining session for actual surfaces.
    let per_frame = structure_bytes
        .checked_mul(2)
        .and_then(|v| v.checked_add(1024))
        .ok_or_else(|| Error::overflow("sequence frame metadata"))?;
    limits.max_frames = limits
        .max_frames
        .min(available.saturating_sub(1024) / 2 / per_frame);
    if limits.max_frames == 0 {
        return Err(Error::exhausted(
            "session has insufficient remaining host capacity for a sequence result",
        ));
    }
    let headers = limits
        .max_frames
        .checked_mul(per_frame)
        .and_then(|v| v.checked_add(1024))
        .ok_or_else(|| Error::overflow("sequence metadata reservation"))?;
    limits.max_total_decoded_bytes = limits
        .max_total_decoded_bytes
        .min(available.saturating_sub(headers));
    if limits.max_total_decoded_bytes == 0 {
        return Err(Error::exhausted(
            "session has no remaining sequence pixel capacity",
        ));
    }
    limits.decode.max_decoded_bytes = limits
        .decode
        .max_decoded_bytes
        .min(session_limits.max_decoded_bytes)
        .min(limits.max_total_decoded_bytes as u64);
    limits.decode.max_metadata_bytes = limits
        .decode
        .max_metadata_bytes
        .min(limits.max_total_decoded_bytes);
    let reservation = session.host_budget().reserve(
        headers
            .checked_add(limits.max_total_decoded_bytes)
            .ok_or_else(|| Error::overflow("sequence output reservation"))?,
    )?;
    Ok((limits, reservation))
}
fn retain_surface(
    image: &Image,
    reservation: &mut HostReservation,
    batch: &Arc<BatchReservation>,
    metadata: &Arc<HostReservation>,
) -> Result<Image> {
    let bytes = image
        .storage()
        .byte_len()
        .checked_add(media_backend_native::image::metadata_payload_bytes(
            image.metadata(),
        )?)
        .ok_or_else(|| Error::overflow("sequence surface and metadata size"))?;
    let guard = reservation.split_off(bytes)?;
    Ok(image
        .clone()
        .retain(guard)
        .retain(batch.clone())
        .retain(metadata.clone()))
}
fn verify_source(asset: &Asset, source: &media_io::source::SourceFingerprint) -> Result<()> {
    if source.is_versioned() && source != &asset.source.fingerprint()? {
        return Err(Error::incompatible(
            "source changed during image sequence decoding",
        ));
    }
    Ok(())
}
fn report(asset: &Asset, bytes: u64, count: u64, mut warnings: Vec<String>) -> ExecutionReport {
    warnings.push("codec private scratch/heap allocations are limited best effort and unmeasured; host counters cover encoded input, result metadata, and decoded pixel/ICC/EXIF reservations".into());
    ExecutionReport {
        backend: "image-sequence".into(),
        bytes_read: bytes,
        decoded_count: count,
        output_count: count,
        peak_host_bytes: asset.session.resources.host_budget().peak(),
        warnings,
        ..Default::default()
    }
}
