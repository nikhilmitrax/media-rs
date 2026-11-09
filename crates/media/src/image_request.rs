//! Native still-image requests, reusable plans, and explicit completion.
use crate::request::Asset;
use media_backend_native::image as native;
use media_backend_native::image_color::{self, ColorConvertSpec, ResizeSpec};
use media_backend_native::image_pixels::{self, PadSpec, PixelConvertSpec};
use media_core::{
    device::Device,
    error::{Error, Result},
    media::Image,
};
use media_io::{
    sink::{FinishedSink, Sink},
    source::{Source, SourceFingerprint},
};
use media_pipeline::{
    diagnostics::{ExecutionReport, PlanStep},
    operation::{Filter, Fit, ImageOperation},
    plan::PlanDescription,
    session::{HostReservation, Session as Resources},
};
use std::{
    io::{Read, Write},
    path::Path,
};

/// Color-aware resize using the facade's shared geometry and filter types.
#[derive(Debug, Clone, Copy)]
pub struct ColorResizeSpec {
    /// Inside bounds or centered cover geometry.
    pub fit: Fit,
    /// Explicit reconstruction filter.
    pub filter: Filter,
    /// Interpolate stored components or linear light.
    pub domain: image_color::ResizeDomain,
    /// Color interpretation, alpha association and gamut behavior.
    pub color: ColorConvertSpec,
}
impl ColorResizeSpec {
    /// Lanczos3 reconstruction in linear light, with strict color metadata rules.
    pub fn linear_light(fit: Fit) -> Self {
        Self {
            fit,
            filter: Filter::Lanczos3,
            domain: image_color::ResizeDomain::LinearLight,
            color: ColorConvertSpec::default(),
        }
    }
}
impl From<ColorResizeSpec> for ResizeSpec {
    fn from(spec: ColorResizeSpec) -> Self {
        Self {
            fit: native_fit(spec.fit),
            filter: native_filter(spec.filter),
            domain: spec.domain,
            color: spec.color,
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum SpatialOperation {
    Basic(ImageOperation),
    Color(ColorConvertSpec),
    ColorResize(ResizeSpec),
    Pad(PadSpec),
    Pixels(PixelConvertSpec),
}

/// Lazy still-image request.
#[derive(Debug, Clone)]
pub struct ImageRequest {
    asset: Asset,
    operations: Vec<SpatialOperation>,
    metadata_policy: native::MetadataPolicy,
}
impl ImageRequest {
    /// Session whose budgets are shared by this request's outputs.
    pub fn session(&self) -> &crate::Session {
        &self.asset.session
    }
    pub(crate) fn new(asset: Asset) -> Self {
        Self {
            asset,
            operations: Vec::new(),
            metadata_policy: native::MetadataPolicy::RequirePreservation,
        }
    }
    /// Apply orientation to pixels, updating orientation metadata.
    pub fn apply_orientation(mut self) -> Self {
        self.operations
            .push(SpatialOperation::Basic(ImageOperation::ApplyOrientation));
        self
    }
    /// Resize stored components using Lanczos3. Color-managed filtering is separate.
    pub fn resize(self, fit: Fit) -> Self {
        self.resize_with_filter(fit, Filter::Lanczos3)
    }
    /// Resize using an explicit stored-component reconstruction filter.
    pub fn resize_with_filter(mut self, fit: Fit, filter: Filter) -> Self {
        self.operations
            .push(SpatialOperation::Basic(ImageOperation::Resize {
                fit,
                filter,
            }));
        self
    }
    /// Crop an exact in-bounds rectangle.
    pub fn crop(mut self, x: u32, y: u32, width: u32, height: u32) -> Self {
        self.operations
            .push(SpatialOperation::Basic(ImageOperation::Crop {
                x,
                y,
                width,
                height,
            }));
        self
    }
    /// Pad stored pixel coordinates using an exact source-format fill value.
    pub fn pad(mut self, spec: PadSpec) -> Self {
        self.operations.push(SpatialOperation::Pad(spec));
        self
    }
    /// Convert RGB/RGBA component storage with explicit precision/alpha rules.
    /// Premultiplied sources must first be unassociated through convert_color.
    pub fn convert_pixels(mut self, spec: PixelConvertSpec) -> Self {
        self.operations.push(SpatialOperation::Pixels(spec));
        self
    }
    /// Convert ICC/RGB color with explicit missing-color, gamut and alpha rules.
    pub fn convert_color(mut self, spec: ColorConvertSpec) -> Self {
        self.operations.push(SpatialOperation::Color(spec));
        self
    }
    /// Resize with explicit linear-light or stored-component color/alpha semantics.
    pub fn resize_color(mut self, spec: impl Into<ResizeSpec>) -> Self {
        self.operations
            .push(SpatialOperation::ColorResize(spec.into()));
        self
    }
    /// Explicitly allow reported metadata loss, such as stale EXIF removal.
    pub fn allow_metadata_loss(mut self) -> Self {
        self.metadata_policy = native::MetadataPolicy::AllowLoss;
        self
    }
    /// Explicitly convert the spatial result to model-input tensor storage.
    #[cfg(feature = "interop")]
    pub fn to_tensor(
        self,
        spec: media_interop::tensor::TensorSpec,
    ) -> crate::tensor_request::ImageTensorRequest {
        crate::tensor_request::ImageTensorRequest::new(self, spec)
    }
    /// Probe without decoding pixels.
    ///
    /// This convenience path materializes at most the session's encoded byte
    /// limit, then applies the backend's counted metadata-read limit.
    pub fn probe(&self) -> Result<native::ImageInfo> {
        let _cpu = self.asset.session.resources.reserve_cpu(1, 1)?;
        self.probe_admitted()
    }
    fn probe_admitted(&self) -> Result<native::ImageInfo> {
        let input = read_encoded(&self.asset.source, &self.asset.session.resources)?;
        native::probe(&input.bytes, decode_limits(&self.asset.session.resources)).map_err(|e| {
            e.with_source(self.asset.source.label())
                .with_operation("probe")
        })
    }
    /// Resolve backend/operation choices while retaining no decoder cursor.
    pub fn compile(&self) -> Result<ImagePlan> {
        let _cpu = self.asset.session.resources.reserve_cpu(1, 1)?;
        let fingerprint = self.asset.source.fingerprint()?;
        let info = self.probe_admitted()?;
        if self.asset.source.fingerprint()? != fingerprint {
            return Err(Error::incompatible("source changed during image probing"));
        }
        let mut description = PlanDescription::default();
        description.steps.push(PlanStep {
            operation: "decode still image".into(),
            backend: "image".into(),
            output: Device::Cpu,
            format: Some(info.pixel_format),
            materializes: true,
        });
        let mut format = info.pixel_format;
        for op in &self.operations {
            if let SpatialOperation::Pixels(spec) = op {
                format = spec.target;
            }
            description.steps.push(PlanStep {
                operation: format!("{op:?}"),
                backend: "image".into(),
                output: Device::Cpu,
                format: Some(format),
                materializes: true,
            });
        }
        description.unresolved.push(
            "codec/allocator internal memory is bounded best-effort; driver costs unknown".into(),
        );
        if !fingerprint.is_versioned() {
            description.unresolved.push(
                "unversioned source: dimensions and limits are revalidated on every execution"
                    .into(),
            );
        }
        Ok(ImagePlan {
            request: self.clone(),
            fingerprint,
            description,
        })
    }
    /// Materialize a finite image within the session limits.
    pub fn read(&self) -> Result<Image> {
        Ok(self.read_with_report()?.image)
    }
    /// Materialize with observed counters and metadata-loss diagnostics.
    pub fn read_with_report(&self) -> Result<ImageResult> {
        self.execute_image(None)
    }
    /// Encode based on a supported filename extension and explicitly finalize.
    pub fn write(&self, path: impl AsRef<Path>) -> Result<ImageWriteResult> {
        let path = path.as_ref();
        let extension = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        let format = match extension.as_str() {
            "png" => native::ImageFormat::Png,
            "jpg" | "jpeg" => native::ImageFormat::Jpeg,
            "webp" => native::ImageFormat::WebP,
            _ => {
                return Err(Error::unsupported(
                    "image output needs a .png, .jpg/.jpeg, or .webp extension",
                ));
            }
        };
        let mut options = native::EncodeOptions::new(format);
        options.metadata_policy = self.metadata_policy;
        let encoded = self.encode(options)?;
        encoded.write_to(Sink::file(path)?)
    }
    /// Encode into a byte sink and finish it after the encoder completes.
    pub fn write_with_options(
        &self,
        sink: Sink,
        options: native::EncodeOptions,
    ) -> Result<ImageWriteResult> {
        self.encode(options)?.write_to(sink)
    }
    /// Encode finite bytes while retaining their Session allocation charge.
    /// Native codec-private scratch remains outside the payload counter.
    pub fn encode(&self, mut options: native::EncodeOptions) -> Result<ImageEncodedResult> {
        let mut result = self.read_with_report()?;
        let _cpu = self.asset.session.resources.reserve_cpu(1, 1)?;
        options.max_output_bytes = options
            .max_output_bytes
            .min(self.asset.session.limits().encoded_bytes);
        options.max_output_bytes = options.max_output_bytes.min(
            self.asset
                .session
                .resources
                .host_budget()
                .limit()
                .saturating_sub(self.asset.session.resources.host_budget().current()),
        );
        let encoding = self
            .asset
            .session
            .resources
            .host_budget()
            .reserve(options.max_output_bytes)?;
        let encoded = native::encode(&result.image, options)?;
        result
            .report
            .warnings
            .extend(encoded.metadata_loss.iter().map(|v| format!("{v:?}")));
        result.report.peak_host_bytes = self.asset.session.resources.host_budget().peak();
        let reservation = encoding.shrink_to(encoded.bytes.capacity())?;
        Ok(ImageEncodedResult {
            bytes: encoded.bytes,
            report: result.report,
            _reservation: reservation,
        })
    }
}
/// Encoded bytes whose payload remains charged until the buffer is dropped.
pub struct ImageEncodedResult {
    bytes: Vec<u8>,
    /// Encoding diagnostics and metadata-loss decisions.
    pub report: ExecutionReport,
    // Drop bytes before returning their reservation.
    _reservation: HostReservation,
}
impl std::fmt::Debug for ImageEncodedResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ImageEncodedResult")
            .field("bytes", &self.bytes.len())
            .field("report", &self.report)
            .finish()
    }
}
impl ImageEncodedResult {
    /// Borrow encoded bytes without releasing their budget or allocating a copy.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
    /// Write and finalize a caller-provided destination while retaining input.
    /// Destination-owned memory, such as Sink::memory, is accounted separately
    /// from this Session's intermediate encoded buffer.
    pub fn write_to(self, mut sink: Sink) -> Result<ImageWriteResult> {
        sink.write_all(&self.bytes).map_err(Error::io)?;
        let output = sink.finish()?;
        Ok(ImageWriteResult {
            output,
            report: self.report,
        })
    }
}
/// Reusable image work; each execution reopens the source independently.
#[derive(Debug, Clone)]
pub struct ImagePlan {
    request: ImageRequest,
    fingerprint: SourceFingerprint,
    description: PlanDescription,
}
impl ImagePlan {
    /// Programmatically inspect resolved backend/materialization choices.
    pub fn description(&self) -> &PlanDescription {
        &self.description
    }
    /// Explain operation and allocation boundaries.
    pub fn explain(&self) -> String {
        self.description.explain()
    }
    /// Execute independently, rejecting stale versioned source assumptions.
    pub fn execute(&self) -> Result<ImageResult> {
        self.request.execute_image(Some(&self.fingerprint))
    }
}
impl ImageRequest {
    fn execute_image(&self, expected: Option<&SourceFingerprint>) -> Result<ImageResult> {
        let source = &self.asset.source;
        let session = &self.asset.session.resources;
        let _cpu = session.reserve_cpu(1, 1)?;
        let fingerprint = source.fingerprint()?;
        if expected.is_some_and(|expected| expected.is_versioned() && expected != &fingerprint) {
            return Err(Error::incompatible(
                "source changed after this plan was compiled; compile again",
            )
            .with_source(source.label())
            .with_operation("execute"));
        }
        let input = read_encoded(source, session)?;
        let limits = decode_limits(session);
        let decoder = native::prepare(&input.bytes, limits)?;
        let info = decoder.info();
        let output_bytes = image_bytes(
            info.width,
            info.height,
            info.pixel_format
                .bytes_per_pixel()
                .ok_or_else(|| Error::unsupported("native image output is not packed"))?,
        )?
        .checked_add(metadata_bytes(&info.metadata)?)
        .ok_or_else(|| Error::overflow("decoded image metadata charge"))?;
        let output_guard = session.host_budget().reserve(output_bytes)?;
        let mut image = decoder.decode()?.retain(output_guard);
        let bytes_read = input.bytes.len() as u64;
        drop(input);
        let mut warnings = Vec::new();
        for op in &self.operations {
            let charge = match *op {
                SpatialOperation::Basic(op) => transform_charge(&image, op)?,
                SpatialOperation::Color(spec) => image_color::working_bytes(&image, None)?
                    .checked_add(image_color::metadata_bytes(&image, &spec, false)?)
                    .ok_or_else(|| Error::overflow("color output metadata charge"))?,
                SpatialOperation::ColorResize(spec) => {
                    image_color::working_bytes(&image, Some(spec))?
                        .checked_add(image_color::metadata_bytes(&image, &spec.color, true)?)
                        .ok_or_else(|| Error::overflow("color resize output metadata charge"))?
                }
                SpatialOperation::Pad(spec) => image_pixels::pad_working_bytes(&image, &spec)?,
                SpatialOperation::Pixels(spec) => {
                    image_pixels::convert_working_bytes(&image, &spec)?
                }
            };
            let charge = if matches!(op, SpatialOperation::Pad(_) | SpatialOperation::Pixels(_)) {
                charge
            } else {
                charge
                    .checked_add(metadata_bytes(image.metadata())?)
                    .ok_or_else(|| Error::overflow("transform metadata charge"))?
            };
            let temporary = session.host_budget().reserve(charge)?;
            let output = match *op {
                SpatialOperation::Basic(ImageOperation::ApplyOrientation) => {
                    native::apply_orientation(&image)?
                }
                SpatialOperation::Basic(ImageOperation::Resize { fit, filter }) => {
                    native::resize(&image, native_fit(fit), native_filter(filter))?
                }
                SpatialOperation::Basic(ImageOperation::Crop {
                    x,
                    y,
                    width,
                    height,
                }) => native::crop(&image, x, y, width, height)?,
                SpatialOperation::Color(spec) => {
                    let output = image_color::convert(&image, spec)?;
                    warnings.extend(output.assumptions);
                    native::TransformOutput {
                        image: output.image,
                        metadata_loss: output.metadata_loss,
                    }
                }
                SpatialOperation::ColorResize(spec) => {
                    let output = image_color::resize(&image, spec)?;
                    warnings.extend(output.assumptions);
                    native::TransformOutput {
                        image: output.image,
                        metadata_loss: output.metadata_loss,
                    }
                }
                SpatialOperation::Pad(spec) => {
                    let output = image_pixels::pad(&image, spec)?;
                    warnings.extend(output.changes.iter().map(|change| format!("{change:?}")));
                    native::TransformOutput {
                        image: output.image,
                        metadata_loss: output.metadata_loss,
                    }
                }
                SpatialOperation::Pixels(spec) => {
                    let output = image_pixels::convert(&image, spec)?;
                    warnings.extend(output.changes.iter().map(|change| format!("{change:?}")));
                    native::TransformOutput {
                        image: output.image,
                        metadata_loss: output.metadata_loss,
                    }
                }
            };
            if !output.metadata_loss.is_empty()
                && self.metadata_policy == native::MetadataPolicy::RequirePreservation
            {
                return Err(Error::incompatible(format!(
                    "transform would lose metadata {:?}; explicitly allow metadata loss",
                    output.metadata_loss
                ))
                .with_operation("transform"));
            }
            warnings.extend(output.metadata_loss.iter().map(|v| format!("{v:?}")));
            if image.storage().shares_allocation(output.image.storage()) {
                image = output.image;
            } else {
                let final_bytes = output
                    .image
                    .storage()
                    .byte_len()
                    .checked_add(metadata_bytes(output.image.metadata())?)
                    .ok_or_else(|| Error::overflow("transformed image allocation charge"))?;
                let guard = temporary.shrink_to(final_bytes)?;
                image = output.image.retain(guard);
            }
        }
        if fingerprint.is_versioned() && source.fingerprint()? != fingerprint {
            return Err(Error::incompatible("source changed during image execution")
                .with_source(source.label()));
        }
        Ok(ImageResult {
            image,
            report: ExecutionReport {
                backend: "image".into(),
                bytes_read,
                decoded_count: 1,
                output_count: 1,
                peak_host_bytes: session.host_budget().peak(),
                warnings,
                ..ExecutionReport::default()
            },
        })
    }
}

/// Materialized image and execution evidence.
#[derive(Debug)]
pub struct ImageResult {
    /// Shared immutable output.
    pub image: Image,
    /// Observed library counters and explicit limitations.
    pub report: ExecutionReport,
}
/// Explicitly finalized encoded output.
#[derive(Debug)]
pub struct ImageWriteResult {
    /// Completed byte destination.
    pub output: FinishedSink,
    /// Execution and metadata diagnostics.
    pub report: ExecutionReport,
}
pub(crate) struct EncodedInput {
    pub(crate) bytes: Vec<u8>,
    _guards: Vec<HostReservation>,
}
pub(crate) fn read_encoded(source: &Source, session: &Resources) -> Result<EncodedInput> {
    let mut reader = source.open_reader()?;
    let mut bytes = Vec::new();
    let mut guards = Vec::new();
    let mut buffer = [0; 65536];
    let limit = session.limits().encoded_bytes;
    loop {
        let available = limit.saturating_sub(bytes.len());
        if available == 0 {
            let mut extra = [0];
            if read_retry(&mut reader, &mut extra)? != 0 {
                return Err(
                    Error::exhausted("encoded image exceeds session input limit")
                        .with_constraint("encoded_bytes"),
                );
            }
            break;
        }
        let n = read_retry(&mut reader, &mut buffer[..available.min(65536)])?;
        if n == 0 {
            break;
        }
        let guard = session.host_budget().reserve(n)?;
        bytes
            .try_reserve_exact(n)
            .map_err(|_| Error::exhausted("cannot allocate encoded image bytes"))?;
        bytes.extend_from_slice(&buffer[..n]);
        guards.push(guard);
    }
    Ok(EncodedInput {
        bytes,
        _guards: guards,
    })
}
fn read_retry(reader: &mut dyn Read, buffer: &mut [u8]) -> Result<usize> {
    loop {
        match reader.read(buffer) {
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            result => return result.map_err(Error::io),
        }
    }
}
pub(crate) fn decode_limits(session: &Resources) -> native::DecodeLimits {
    native::DecodeLimits {
        max_encoded_bytes: session.limits().encoded_bytes as u64,
        max_decoded_bytes: session.limits().host_memory as u64,
        max_pixels: session.limits().image_pixels,
        ..Default::default()
    }
}
fn image_bytes(width: u32, height: u32, bpp: usize) -> Result<usize> {
    (width as usize)
        .checked_mul(height as usize)
        .and_then(|v| v.checked_mul(bpp))
        .ok_or_else(|| Error::overflow("image allocation dimensions overflow"))
}
fn metadata_bytes(metadata: &media_core::media::ImageMetadata) -> Result<usize> {
    native::metadata_payload_bytes(metadata)
}
fn transform_charge(image: &Image, op: ImageOperation) -> Result<usize> {
    let bpp = image
        .pixel_format()
        .bytes_per_pixel()
        .ok_or_else(|| Error::unsupported("packed host transform required"))?;
    let (w, h) = match op {
        ImageOperation::ApplyOrientation => (image.width(), image.height()),
        ImageOperation::Crop { width, height, .. } => (width, height),
        ImageOperation::Resize {
            fit: Fit::Inside { width, height },
            ..
        } => (width, height),
        ImageOperation::Resize {
            fit: Fit::Cover { width, height },
            ..
        } => {
            if width == 0 || height == 0 {
                return Err(Error::invalid("resize dimensions must be nonzero"));
            }
            let (iw, ih) = (u64::from(image.width()), u64::from(image.height()));
            if u64::from(width) * ih >= u64::from(height) * iw {
                (
                    width,
                    u32::try_from((ih * u64::from(width)).div_ceil(iw))
                        .map_err(|_| Error::overflow("cover height overflow"))?,
                )
            } else {
                (
                    u32::try_from((iw * u64::from(height)).div_ceil(ih))
                        .map_err(|_| Error::overflow("cover width overflow"))?,
                    height,
                )
            }
        }
    };
    image
        .storage()
        .byte_len()
        .checked_add(
            image_bytes(w, h, bpp)?
                .checked_mul(3)
                .ok_or_else(|| Error::overflow("transform scratch estimate overflow"))?,
        )
        .ok_or_else(|| Error::overflow("transform allocation estimate overflow"))
}
fn native_fit(value: Fit) -> native::Fit {
    match value {
        Fit::Inside { width, height } => native::Fit::inside(width, height),
        Fit::Cover { width, height } => native::Fit::cover(width, height),
    }
}
fn native_filter(value: Filter) -> native::Filter {
    match value {
        Filter::Nearest => native::Filter::Nearest,
        Filter::Triangle => native::Filter::Triangle,
        Filter::CatmullRom => native::Filter::CatmullRom,
        Filter::Gaussian => native::Filter::Gaussian,
        Filter::Lanczos3 => native::Filter::Lanczos3,
    }
}
