//! Bounded still-image codecs and explicit spatial transforms.
//!
//! PNG, JPEG, WebP, and BMP decoding use native Rust codecs. Animated PNG/WebP
//! and GIF/TIFF are rejected rather than silently selecting a frame or page.
//! ICC profiles are transported here; `image_color` applies explicit color
//! conversions. Resize below filters stored components without linearization.

use std::cell::Cell;
use std::io::{self, BufReader, Cursor, Read, Seek, SeekFrom, Write};
use std::rc::Rc;

use image::{ColorType, DynamicImage, ImageBuffer, ImageDecoder, ImageEncoder};
use media_core::error::{Error, Result};
use media_core::media::{ColorDescription, Image, ImageMetadata, Orientation, PixelFormat};

/// Container format independent of the filename extension.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ImageFormat {
    /// Portable Network Graphics, including sixteen-bit components.
    Png,
    /// JPEG.
    Jpeg,
    /// WebP; this backend writes lossless eight-bit images.
    WebP,
    /// Windows bitmap, decode only.
    Bmp,
}

/// Limits checked before allocating decoded pixels.
///
/// The decoded byte limit bounds output storage. Codec allocations are also
/// limited, but the codec's allocation limit is documented as best effort.
#[derive(Clone, Copy, Debug)]
pub struct DecodeLimits {
    /// Maximum width.
    pub max_width: u32,
    /// Maximum height.
    pub max_height: u32,
    /// Maximum pixel count.
    pub max_pixels: u64,
    /// Maximum decoded output bytes.
    pub max_decoded_bytes: u64,
    /// Maximum encoded input bytes.
    pub max_encoded_bytes: u64,
    /// Maximum returned ICC and EXIF bytes combined.
    pub max_metadata_bytes: usize,
    /// Maximum input bytes read by a metadata-only probe, counting rereads.
    pub max_probe_bytes: usize,
}

impl Default for DecodeLimits {
    fn default() -> Self {
        Self {
            max_width: 32_768,
            max_height: 32_768,
            max_pixels: 64 * 1024 * 1024,
            max_decoded_bytes: 256 * 1024 * 1024,
            max_encoded_bytes: 256 * 1024 * 1024,
            max_metadata_bytes: 4 * 1024 * 1024,
            max_probe_bytes: 8 * 1024 * 1024,
        }
    }
}

impl DecodeLimits {
    pub(crate) fn codec(self) -> image::Limits {
        let mut limits = image::Limits::default();
        limits.max_image_width = Some(self.max_width);
        limits.max_image_height = Some(self.max_height);
        limits.max_alloc = Some(self.max_decoded_bytes);
        limits
    }

    pub(crate) fn check(self, width: u32, height: u32, bytes: u64) -> Result<()> {
        if width == 0 || height == 0 {
            return Err(Error::invalid("image dimensions must be nonzero"));
        }
        if width > self.max_width
            || height > self.max_height
            || u64::from(width) * u64::from(height) > self.max_pixels
            || bytes > self.max_decoded_bytes
        {
            return Err(Error::invalid(format!(
                "image {width}x{height} with {bytes} decoded bytes exceeds configured limits"
            )));
        }
        Ok(())
    }
}

/// Metadata discovered without pixel decompression.
#[derive(Clone, Debug)]
pub struct ImageInfo {
    /// Stored width, before display orientation.
    pub width: u32,
    /// Stored height.
    pub height: u32,
    /// Decoded representation.
    pub pixel_format: PixelFormat,
    /// Detected container.
    pub format: ImageFormat,
    /// Transported metadata; no inferred color conversion.
    pub metadata: ImageMetadata,
    /// Input bytes read while probing, including rereads.
    pub bytes_read: usize,
}

/// Aspect-ratio policy for resize.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Fit {
    /// Fit inside a rectangle while preserving aspect ratio.
    Inside {
        /// Bounding width.
        width: u32,
        /// Bounding height.
        height: u32,
    },
    /// Fill a rectangle, then center-crop.
    Cover {
        /// Output width.
        width: u32,
        /// Output height.
        height: u32,
    },
}

impl Fit {
    /// Fit inside the bounds.
    pub const fn inside(width: u32, height: u32) -> Self {
        Self::Inside { width, height }
    }
    /// Fill and center-crop to the dimensions.
    pub const fn cover(width: u32, height: u32) -> Self {
        Self::Cover { width, height }
    }
}

/// Reconstruction filter on stored component values.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Filter {
    /// Nearest neighbor, useful for labels and masks.
    Nearest,
    /// Bilinear interpolation.
    Triangle,
    /// Cubic Catmull–Rom.
    CatmullRom,
    /// Gaussian.
    Gaussian,
    /// Three-lobe Lanczos.
    Lanczos3,
}

impl From<Filter> for image::imageops::FilterType {
    fn from(value: Filter) -> Self {
        match value {
            Filter::Nearest => Self::Nearest,
            Filter::Triangle => Self::Triangle,
            Filter::CatmullRom => Self::CatmullRom,
            Filter::Gaussian => Self::Gaussian,
            Filter::Lanczos3 => Self::Lanczos3,
        }
    }
}

/// Metadata that could not be retained faithfully.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MetadataLoss {
    /// Opaque EXIF removed after geometry changed, avoiding stale tags.
    ExifAfterGeometry,
    /// Opaque EXIF removed after color changed, avoiding stale color-space tags.
    ExifAfterColorTransform,
    /// Opaque EXIF removed after component type/channel changes, avoiding stale tags.
    ExifAfterComponentTransform,
    /// Encoder could not carry the ICC profile.
    IccProfile,
    /// Encoder could not carry the EXIF block.
    Exif,
    /// Named color declarations could not be serialized without an ICC profile.
    ColorDescription,
}

/// Spatial operation result and metadata-loss report.
#[derive(Clone, Debug)]
pub struct TransformOutput {
    /// Transformed media.
    pub image: Image,
    /// Removed metadata.
    pub metadata_loss: Vec<MetadataLoss>,
}

/// Retained ICC, EXIF and color-description string payload bytes.
///
/// This counts owned payload lengths, not allocator bookkeeping or Rust headers.
pub fn metadata_payload_bytes(metadata: &ImageMetadata) -> Result<usize> {
    let mut bytes = metadata
        .icc_profile
        .as_ref()
        .map_or(0, Vec::len)
        .checked_add(metadata.exif.as_ref().map_or(0, Vec::len))
        .ok_or_else(|| Error::overflow("image metadata length overflow"))?;
    for value in [
        &metadata.color.primaries,
        &metadata.color.transfer,
        &metadata.color.matrix,
        &metadata.color.range,
        &metadata.color.chroma_siting,
    ]
    .into_iter()
    .flatten()
    {
        bytes = bytes
            .checked_add(value.len())
            .ok_or_else(|| Error::overflow("image color metadata length overflow"))?;
    }
    Ok(bytes)
}

/// Policy for metadata an encoder cannot represent.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MetadataPolicy {
    /// Fail if supplied metadata cannot be carried.
    RequirePreservation,
    /// Permit loss and report removed fields.
    AllowLoss,
}

/// Explicit encoding settings.
#[derive(Clone, Copy, Debug)]
pub struct EncodeOptions {
    /// Output container.
    pub format: ImageFormat,
    /// JPEG quality, 1–100.
    pub jpeg_quality: u8,
    /// Treatment of unsupported metadata.
    pub metadata_policy: MetadataPolicy,
    /// Maximum encoded output bytes.
    pub max_output_bytes: usize,
}

impl EncodeOptions {
    /// Defaults for an explicitly selected format.
    pub const fn new(format: ImageFormat) -> Self {
        Self {
            format,
            jpeg_quality: 90,
            metadata_policy: MetadataPolicy::RequirePreservation,
            max_output_bytes: 256 * 1024 * 1024,
        }
    }
}

/// Completed file bytes and any explicitly permitted metadata loss.
#[derive(Clone, Debug)]
pub struct EncodedImage {
    /// Complete encoded file.
    pub bytes: Vec<u8>,
    /// Metadata not retained.
    pub metadata_loss: Vec<MetadataLoss>,
}

/// Inspect dimensions and metadata within a counted read budget.
///
/// The caller supplies the complete encoded object, but probing never
/// decompresses pixels. Oversized header/metadata reads fail explicitly.
pub fn probe(bytes: &[u8], limits: DecodeLimits) -> Result<ImageInfo> {
    Ok(prepare(bytes, limits)?.into_info())
}

/// One execution-local still decoder whose inspected header and metadata are
/// reused for pixel decoding. It borrows immutable encoded input and retains no
/// source cursor; callers can reserve the exact output payload after info().
pub struct PreparedImage<'a> {
    decoder: Box<dyn ImageDecoder + 'a>,
    info: ImageInfo,
    read_budget: Rc<Cell<usize>>,
}
impl PreparedImage<'_> {
    /// Validated dimensions, representation and metadata without decoding pixels.
    pub fn info(&self) -> &ImageInfo {
        &self.info
    }
    /// Consume header-only work and release the decoder without decoding pixels.
    pub fn into_info(self) -> ImageInfo {
        self.info
    }
    /// Decode using the same inspected codec state. The metadata-read cap ends
    /// here; encoded/output size and codec allocation limits remain in force.
    pub fn decode(self) -> Result<Image> {
        self.read_budget.set(usize::MAX);
        let expected = (self.info.width, self.info.height, self.info.pixel_format);
        let image = from_dynamic(
            DynamicImage::from_decoder(self.decoder).map_err(codec_error)?,
            self.info.metadata,
        )?;
        if (image.width(), image.height(), image.pixel_format()) != expected {
            return Err(Error::incompatible(
                "decoded image changed its inspected geometry or representation",
            ));
        }
        Ok(image)
    }
}

/// Inspect bounded still-image metadata once, then allow explicit decoding with
/// the same codec state. No pixels are decompressed before decode() is called.
pub fn prepare(bytes: &[u8], limits: DecodeLimits) -> Result<PreparedImage<'_>> {
    let format = detect_still(bytes, limits)?;
    let count = Rc::new(Cell::new(0));
    let read_budget = Rc::new(Cell::new(limits.max_probe_bytes));
    let input = BudgetReader {
        cursor: Cursor::new(bytes),
        count: Rc::clone(&count),
        budget: Rc::clone(&read_budget),
    };
    let mut reader = image::ImageReader::with_format(BufReader::new(input), format.codec());
    reader.limits(limits.codec());
    let mut decoder = reader.into_decoder().map_err(codec_error)?;
    let mut info = inspect(&mut decoder, format, limits)?;
    let (color, header_reads) = encoded_color(bytes, format)?;
    info.metadata.color = color;
    info.bytes_read = count
        .get()
        .checked_add(header_reads)
        .ok_or_else(|| Error::overflow("probe byte counter overflow"))?;
    if info.bytes_read > limits.max_probe_bytes {
        return Err(Error::exhausted("color metadata probe exceeds read budget"));
    }
    Ok(PreparedImage {
        decoder: Box::new(decoder),
        info,
        read_budget,
    })
}

/// Decode a still image while preserving native bit depth.
///
/// Display orientation remains metadata until explicitly applied.
pub fn decode(bytes: &[u8], limits: DecodeLimits) -> Result<Image> {
    let format = detect_still(bytes, limits)?;
    let mut reader = image::ImageReader::with_format(Cursor::new(bytes), format.codec());
    reader.limits(limits.codec());
    let mut decoder = reader.into_decoder().map_err(codec_error)?;
    let mut info = inspect(&mut decoder, format, limits)?;
    info.metadata.color = encoded_color(bytes, format)?.0;
    from_dynamic(
        DynamicImage::from_decoder(decoder).map_err(codec_error)?,
        info.metadata,
    )
}

/// Apply display orientation to pixels and clear the orientation flag.
///
/// Opaque EXIF is removed and reported when pixels change, because its dimension
/// and thumbnail tags cannot safely be propagated unchanged.
pub fn apply_orientation(image: &Image) -> Result<TransformOutput> {
    if image.metadata().orientation == Orientation::Normal {
        return Ok(TransformOutput {
            image: image.clone(),
            metadata_loss: Vec::new(),
        });
    }
    let mut pixels = to_dynamic(image)?;
    pixels.apply_orientation(to_orientation(image.metadata().orientation));
    let (mut metadata, metadata_loss) = transformed_metadata(image);
    metadata.orientation = Orientation::Normal;
    Ok(TransformOutput {
        image: from_dynamic(pixels, metadata)?,
        metadata_loss,
    })
}

/// Resize stored component values with explicit geometry and filter.
///
/// Pixel format, ICC, and color description are retained. This method does not
/// apply profiles, linearize transfers, or premultiply alpha. It enforces the
/// default decoded size limit on output and the cover intermediate allocation.
pub fn resize(image: &Image, fit: Fit, filter: Filter) -> Result<TransformOutput> {
    let (width, height) = match fit {
        Fit::Inside { width, height } | Fit::Cover { width, height } => (width, height),
    };
    check_allocation(width, height, image.pixel_format())?;
    let pixels = to_dynamic(image)?;
    let output = match fit {
        Fit::Inside { .. } => pixels.resize(width, height, filter.into()),
        Fit::Cover { .. } => {
            let (w, h) = cover_dimensions(image.width(), image.height(), width, height)?;
            check_allocation(w, h, image.pixel_format())?;
            pixels.resize_exact(w, h, filter.into()).crop_imm(
                (w - width) / 2,
                (h - height) / 2,
                width,
                height,
            )
        }
    };
    let (metadata, metadata_loss) = transformed_metadata(image);
    Ok(TransformOutput {
        image: from_dynamic(output, metadata)?,
        metadata_loss,
    })
}

/// Crop a nonempty, in-bounds rectangle without resampling.
pub fn crop(image: &Image, x: u32, y: u32, width: u32, height: u32) -> Result<TransformOutput> {
    if width == 0
        || height == 0
        || x.checked_add(width).is_none_or(|end| end > image.width())
        || y.checked_add(height).is_none_or(|end| end > image.height())
    {
        return Err(Error::invalid(
            "crop rectangle must be nonempty and in bounds",
        ));
    }
    let output = to_dynamic(image)?.crop_imm(x, y, width, height);
    let (metadata, metadata_loss) = transformed_metadata(image);
    Ok(TransformOutput {
        image: from_dynamic(output, metadata)?,
        metadata_loss,
    })
}

/// Encode without implicit format conversion or alpha removal.
///
/// JPEG accepts Gray8/Rgb8, WebP accepts Rgb8/Rgba8, and PNG accepts eight- and
/// sixteen-bit packed formats. Float pixels require explicit conversion. Apply
/// orientation first so display flags and EXIF cannot conflict.
pub fn encode(image: &Image, options: EncodeOptions) -> Result<EncodedImage> {
    if image.metadata().orientation != Orientation::Normal {
        return Err(Error::unsupported(
            "apply image orientation before encoding",
        ));
    }
    if image
        .metadata()
        .exif
        .as_ref()
        .and_then(|exif| image::metadata::Orientation::from_exif_chunk(exif))
        .is_some_and(|o| o != image::metadata::Orientation::NoTransforms)
    {
        return Err(Error::invalid(
            "EXIF orientation conflicts with normal image orientation",
        ));
    }
    if !(1..=100).contains(&options.jpeg_quality) {
        return Err(Error::invalid("JPEG quality must be between 1 and 100"));
    }
    let pixels = to_dynamic(image)?;
    let color = pixels.color();
    let supported = match options.format {
        ImageFormat::Png => !matches!(color, ColorType::Rgb32F | ColorType::Rgba32F),
        ImageFormat::Jpeg => matches!(color, ColorType::L8 | ColorType::Rgb8),
        ImageFormat::WebP => matches!(color, ColorType::Rgb8 | ColorType::Rgba8),
        ImageFormat::Bmp => false,
    };
    if !supported {
        return Err(Error::unsupported(format!(
            "{:?} cannot encode {:?} without an explicit conversion",
            options.format,
            image.pixel_format()
        )));
    }
    let mut writer = LimitedWriter {
        bytes: Vec::new(),
        max: options.max_output_bytes,
    };
    let mut loss = Vec::new();
    match options.format {
        ImageFormat::Png => {
            let mut encoder = image::codecs::png::PngEncoder::new(&mut writer);
            set_metadata(
                &mut encoder,
                image.metadata(),
                options.metadata_policy,
                &mut loss,
            )?;
            encoder
                .write_image(
                    pixels.as_bytes(),
                    image.width(),
                    image.height(),
                    color.into(),
                )
                .map_err(codec_error)?;
        }
        ImageFormat::Jpeg => {
            let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(
                &mut writer,
                options.jpeg_quality,
            );
            set_metadata(
                &mut encoder,
                image.metadata(),
                options.metadata_policy,
                &mut loss,
            )?;
            encoder
                .write_image(
                    pixels.as_bytes(),
                    image.width(),
                    image.height(),
                    color.into(),
                )
                .map_err(codec_error)?;
        }
        ImageFormat::WebP => {
            let mut encoder = image::codecs::webp::WebPEncoder::new_lossless(&mut writer);
            set_metadata(
                &mut encoder,
                image.metadata(),
                options.metadata_policy,
                &mut loss,
            )?;
            encoder
                .write_image(
                    pixels.as_bytes(),
                    image.width(),
                    image.height(),
                    color.into(),
                )
                .map_err(codec_error)?;
        }
        ImageFormat::Bmp => return Err(Error::unsupported("BMP encoding is not provided")),
    }
    Ok(EncodedImage {
        bytes: writer.bytes,
        metadata_loss: loss,
    })
}

impl ImageFormat {
    fn codec(self) -> image::ImageFormat {
        match self {
            Self::Png => image::ImageFormat::Png,
            Self::Jpeg => image::ImageFormat::Jpeg,
            Self::WebP => image::ImageFormat::WebP,
            Self::Bmp => image::ImageFormat::Bmp,
        }
    }
}

fn detect_still(bytes: &[u8], limits: DecodeLimits) -> Result<ImageFormat> {
    if bytes.len() as u64 > limits.max_encoded_bytes {
        return Err(Error::invalid(
            "encoded image exceeds configured byte limit",
        ));
    }
    match image::guess_format(bytes).map_err(codec_error)? {
        image::ImageFormat::Png => {
            let mut offset = 8usize;
            while let Some(header) = bytes.get(offset..offset.saturating_add(8)) {
                let size = u32::from_be_bytes(header[..4].try_into().expect("four bytes")) as usize;
                if &header[4..] == b"acTL" {
                    return Err(Error::unsupported(
                        "animated PNG requires an animation-frame API",
                    ));
                }
                offset = offset
                    .checked_add(size)
                    .and_then(|n| n.checked_add(12))
                    .ok_or_else(|| Error::overflow("PNG chunk size overflow"))?;
                if &header[4..] == b"IEND" {
                    break;
                }
            }
            Ok(ImageFormat::Png)
        }
        image::ImageFormat::WebP => {
            let mut offset = 12usize;
            while let Some(header) = bytes.get(offset..offset.saturating_add(8)) {
                let size = u32::from_le_bytes(header[4..].try_into().expect("four bytes")) as usize;
                if &header[..4] == b"ANIM" || &header[..4] == b"ANMF" {
                    return Err(Error::unsupported(
                        "animated WebP requires an animation-frame API",
                    ));
                }
                offset = offset
                    .checked_add(size)
                    .and_then(|n| n.checked_add(8 + (size & 1)))
                    .ok_or_else(|| Error::overflow("WebP chunk size overflow"))?;
            }
            Ok(ImageFormat::WebP)
        }
        image::ImageFormat::Jpeg => Ok(ImageFormat::Jpeg),
        image::ImageFormat::Bmp => Ok(ImageFormat::Bmp),
        image::ImageFormat::Gif | image::ImageFormat::Tiff => Err(Error::unsupported(
            "GIF animation and TIFF page selection require an explicit frame/page API",
        )),
        other => Err(Error::unsupported(format!(
            "native image format {other:?} is not enabled"
        ))),
    }
}

fn inspect(
    decoder: &mut impl ImageDecoder,
    format: ImageFormat,
    limits: DecodeLimits,
) -> Result<ImageInfo> {
    let (width, height) = decoder.dimensions();
    limits.check(width, height, decoder.total_bytes())?;
    let orientation = decoder.orientation().map_err(codec_error)?;
    let icc_profile = decoder.icc_profile().map_err(codec_error)?;
    let exif = decoder.exif_metadata().map_err(codec_error)?;
    let metadata_bytes = icc_profile
        .as_ref()
        .map_or(0, Vec::len)
        .checked_add(exif.as_ref().map_or(0, Vec::len))
        .ok_or_else(|| Error::overflow("image metadata size overflow"))?;
    if metadata_bytes > limits.max_metadata_bytes {
        return Err(Error::invalid(
            "image metadata exceeds configured byte limit",
        ));
    }
    let metadata = ImageMetadata {
        orientation: from_orientation(orientation),
        icc_profile,
        exif,
        ..ImageMetadata::default()
    };
    Ok(ImageInfo {
        width,
        height,
        pixel_format: pixel_format(decoder.color_type())?,
        format,
        metadata,
        bytes_read: 0,
    })
}

// Color chunks must precede IDAT. Preserve established declarations without
// assuming sRGB for untagged PNG/JPEG/WebP/BMP. Unknown numeric values remain
// explicit strings so color-managed requests reject rather than guess.
pub(crate) fn encoded_color(
    bytes: &[u8],
    format: ImageFormat,
) -> Result<(ColorDescription, usize)> {
    if format != ImageFormat::Png {
        return Ok((ColorDescription::default(), 0));
    }
    let mut offset = 8usize;
    let mut reads = 0usize;
    let mut srgb = false;
    let mut has_icc = false;
    let mut cicp = None;
    let mut gamma = None;
    let mut chromaticities = None;
    while let Some(header) = bytes.get(offset..offset.saturating_add(8)) {
        reads = reads
            .checked_add(8)
            .ok_or_else(|| Error::overflow("PNG color header count overflow"))?;
        let size = u32::from_be_bytes(header[..4].try_into().expect("four bytes")) as usize;
        let kind = &header[4..];
        if kind == b"iCCP" {
            has_icc = true;
        }
        if kind == b"IDAT" || kind == b"IEND" {
            break;
        }
        let start = offset
            .checked_add(8)
            .ok_or_else(|| Error::overflow("PNG chunk overflow"))?;
        let end = start
            .checked_add(size)
            .ok_or_else(|| Error::overflow("PNG chunk overflow"))?;
        if matches!(kind, b"sRGB" | b"cICP" | b"gAMA" | b"cHRM") {
            let data = bytes
                .get(start..end)
                .ok_or_else(|| Error::invalid("truncated PNG color chunk"))?;
            reads = reads
                .checked_add(size)
                .ok_or_else(|| Error::overflow("PNG color metadata count overflow"))?;
            match kind {
                b"sRGB" => {
                    if size != 1 {
                        return Err(Error::invalid("invalid PNG sRGB chunk"));
                    }
                    srgb = true;
                }
                b"cICP" => {
                    if size != 4 || data[3] > 1 {
                        return Err(Error::invalid("invalid PNG cICP chunk"));
                    }
                    cicp = Some([data[0], data[1], data[2], data[3]]);
                }
                b"gAMA" => {
                    if size != 4 {
                        return Err(Error::invalid("invalid PNG gamma chunk"));
                    }
                    gamma = Some(u32::from_be_bytes(data.try_into().expect("four bytes")));
                }
                b"cHRM" => {
                    if size != 32 {
                        return Err(Error::invalid("invalid PNG chromaticity chunk"));
                    }
                    chromaticities = Some(
                        data.as_chunks::<4>()
                            .0
                            .iter()
                            .map(|v| u32::from_be_bytes(*v))
                            .collect::<Vec<_>>(),
                    );
                }
                _ => {}
            }
        }
        offset = end
            .checked_add(4)
            .ok_or_else(|| Error::overflow("PNG chunk overflow"))?;
    }
    let mut color = ColorDescription::default();
    if let Some(values) = cicp {
        color.primaries = match values[0] {
            2 => None,
            1 => Some("bt709".into()),
            9 => Some("bt2020".into()),
            12 => Some("display-p3".into()),
            v => Some(format!("cicp:{v}")),
        };
        color.transfer = match values[1] {
            2 => None,
            1 | 6 | 14 | 15 => Some("bt709".into()),
            8 => Some("linear".into()),
            13 => Some("srgb".into()),
            16 => Some("smpte2084".into()),
            18 => Some("hlg".into()),
            v => Some(format!("cicp:{v}")),
        };
        color.matrix = Some(if values[2] == 0 {
            "rgb".into()
        } else {
            format!("cicp:{}", values[2])
        });
        color.range = Some(if values[3] == 1 { "full" } else { "limited" }.into());
    }
    if srgb {
        if cicp.is_some()
            && (color.primaries.as_deref().is_some_and(|v| v != "bt709")
                || color.transfer.as_deref().is_some_and(|v| v != "srgb")
                || color.matrix.as_deref() != Some("rgb")
                || color.range.as_deref() != Some("full"))
        {
            return Err(Error::incompatible(
                "PNG sRGB and cICP chunks contradict each other",
            ));
        }
        color = ColorDescription {
            primaries: Some("bt709".into()),
            transfer: Some("srgb".into()),
            matrix: Some("rgb".into()),
            range: Some("full".into()),
            chroma_siting: None,
        };
    } else if cicp.is_none() && !has_icc {
        if let Some(gamma) = gamma {
            if gamma == 0 {
                return Err(Error::invalid("PNG gamma must be positive"));
            }
            color.transfer = Some(format!("png-gamma:{gamma}"));
        }
        if let Some(chrm) = chromaticities {
            let srgb = [31270, 32900, 64000, 33000, 30000, 60000, 15000, 6000];
            color.primaries = Some(if chrm.iter().zip(srgb).all(|(a, b)| a.abs_diff(b) <= 2) {
                "bt709".into()
            } else {
                format!("png-chromaticities:{chrm:?}")
            });
        }
        if gamma.is_some() || color.primaries.is_some() {
            color.matrix = Some("rgb".into());
            color.range = Some("full".into());
        }
    }
    Ok((color, reads))
}

fn transformed_metadata(image: &Image) -> (ImageMetadata, Vec<MetadataLoss>) {
    let mut metadata = image.metadata().clone();
    let loss = if metadata.exif.take().is_some() {
        vec![MetadataLoss::ExifAfterGeometry]
    } else {
        Vec::new()
    };
    (metadata, loss)
}

fn set_metadata(
    encoder: &mut impl ImageEncoder,
    metadata: &ImageMetadata,
    policy: MetadataPolicy,
    loss: &mut Vec<MetadataLoss>,
) -> Result<()> {
    if metadata.icc_profile.is_none() && metadata.color != ColorDescription::default() {
        if policy == MetadataPolicy::RequirePreservation {
            return Err(Error::unsupported(
                "this encoder cannot preserve named color declarations without an ICC profile; explicitly convert to a profiled color space or allow reported metadata loss",
            ));
        }
        loss.push(MetadataLoss::ColorDescription);
    }
    if let Some(profile) = &metadata.icc_profile
        && let Err(error) = encoder.set_icc_profile(profile.clone())
    {
        if policy == MetadataPolicy::RequirePreservation {
            return Err(Error::backend("image", error.to_string()));
        }
        loss.push(MetadataLoss::IccProfile);
    }
    if let Some(exif) = &metadata.exif
        && let Err(error) = encoder.set_exif_metadata(exif.clone())
    {
        if policy == MetadataPolicy::RequirePreservation {
            return Err(Error::backend("image", error.to_string()));
        }
        loss.push(MetadataLoss::Exif);
    }
    Ok(())
}

fn cover_dimensions(sw: u32, sh: u32, width: u32, height: u32) -> Result<(u32, u32)> {
    let wide = u64::from(width) * u64::from(sh);
    let tall = u64::from(height) * u64::from(sw);
    let (w, h) = if wide >= tall {
        (u64::from(width), wide.div_ceil(u64::from(sw)))
    } else {
        (tall.div_ceil(u64::from(sh)), u64::from(height))
    };
    Ok((
        u32::try_from(w).map_err(|_| Error::overflow("cover width overflow"))?,
        u32::try_from(h).map_err(|_| Error::overflow("cover height overflow"))?,
    ))
}

fn check_allocation(width: u32, height: u32, format: PixelFormat) -> Result<()> {
    let bytes = u64::from(width)
        .checked_mul(u64::from(height))
        .and_then(|n| bytes_per_pixel(format).ok().and_then(|b| n.checked_mul(b)))
        .ok_or_else(|| Error::overflow("image allocation size overflow"))?;
    DecodeLimits::default().check(width, height, bytes)
}

fn pixel_format(color: ColorType) -> Result<PixelFormat> {
    Ok(match color {
        ColorType::L8 => PixelFormat::Gray8,
        ColorType::La8 => PixelFormat::GrayAlpha8,
        ColorType::Rgb8 => PixelFormat::Rgb8,
        ColorType::Rgba8 => PixelFormat::Rgba8,
        ColorType::L16 => PixelFormat::Gray16,
        ColorType::La16 => PixelFormat::GrayAlpha16,
        ColorType::Rgb16 => PixelFormat::Rgb16,
        ColorType::Rgba16 => PixelFormat::Rgba16,
        ColorType::Rgb32F => PixelFormat::RgbF32,
        ColorType::Rgba32F => PixelFormat::RgbaF32,
        _ => {
            return Err(Error::unsupported(format!(
                "unsupported color type {color:?}"
            )));
        }
    })
}

fn bytes_per_pixel(format: PixelFormat) -> Result<u64> {
    Ok(match format {
        PixelFormat::Gray8 => 1,
        PixelFormat::GrayAlpha8 | PixelFormat::Gray16 => 2,
        PixelFormat::Rgb8 => 3,
        PixelFormat::Rgba8 | PixelFormat::GrayAlpha16 => 4,
        PixelFormat::Rgb16 => 6,
        PixelFormat::Rgba16 => 8,
        PixelFormat::RgbF32 => 12,
        PixelFormat::RgbaF32 => 16,
        _ => {
            return Err(Error::unsupported(
                "native operations require packed host pixels",
            ));
        }
    })
}

fn to_dynamic(image: &Image) -> Result<DynamicImage> {
    let (width, height) = (image.width(), image.height());
    let bytes = image.packed_bytes()?;
    macro_rules! buffer {
        ($variant:ident, $pixel:ty, $data:expr) => {
            DynamicImage::$variant(
                ImageBuffer::<$pixel, _>::from_raw(width, height, $data)
                    .ok_or_else(|| Error::invalid("invalid packed image buffer"))?,
            )
        };
    }
    let u16s = || {
        bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|v| u16::from_le_bytes([v[0], v[1]]))
            .collect::<Vec<_>>()
    };
    let f32s = || {
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| f32::from_le_bytes([v[0], v[1], v[2], v[3]]))
            .collect::<Vec<_>>()
    };
    Ok(match image.pixel_format() {
        PixelFormat::Gray8 => buffer!(ImageLuma8, image::Luma<u8>, bytes.to_vec()),
        PixelFormat::GrayAlpha8 => buffer!(ImageLumaA8, image::LumaA<u8>, bytes.to_vec()),
        PixelFormat::Rgb8 => buffer!(ImageRgb8, image::Rgb<u8>, bytes.to_vec()),
        PixelFormat::Rgba8 => buffer!(ImageRgba8, image::Rgba<u8>, bytes.to_vec()),
        PixelFormat::Gray16 => buffer!(ImageLuma16, image::Luma<u16>, u16s()),
        PixelFormat::GrayAlpha16 => buffer!(ImageLumaA16, image::LumaA<u16>, u16s()),
        PixelFormat::Rgb16 => buffer!(ImageRgb16, image::Rgb<u16>, u16s()),
        PixelFormat::Rgba16 => buffer!(ImageRgba16, image::Rgba<u16>, u16s()),
        PixelFormat::RgbF32 => buffer!(ImageRgb32F, image::Rgb<f32>, f32s()),
        PixelFormat::RgbaF32 => buffer!(ImageRgba32F, image::Rgba<f32>, f32s()),
        _ => {
            return Err(Error::unsupported(
                "native operations require packed host pixels",
            ));
        }
    })
}

fn from_dynamic(image: DynamicImage, metadata: ImageMetadata) -> Result<Image> {
    let (width, height) = (image.width(), image.height());
    let format = pixel_format(image.color())?;
    let bytes = match image {
        DynamicImage::ImageLuma8(b) => b.into_raw(),
        DynamicImage::ImageLumaA8(b) => b.into_raw(),
        DynamicImage::ImageRgb8(b) => b.into_raw(),
        DynamicImage::ImageRgba8(b) => b.into_raw(),
        DynamicImage::ImageLuma16(b) => b
            .into_raw()
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect(),
        DynamicImage::ImageLumaA16(b) => b
            .into_raw()
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect(),
        DynamicImage::ImageRgb16(b) => b
            .into_raw()
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect(),
        DynamicImage::ImageRgba16(b) => b
            .into_raw()
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect(),
        DynamicImage::ImageRgb32F(b) => b
            .into_raw()
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect(),
        DynamicImage::ImageRgba32F(b) => b
            .into_raw()
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect(),
        _ => {
            return Err(Error::unsupported(
                "unsupported dynamic image representation",
            ));
        }
    };
    Ok(Image::from_packed(width, height, format, bytes)?.with_metadata(metadata))
}

fn to_orientation(orientation: Orientation) -> image::metadata::Orientation {
    use image::metadata::Orientation as O;
    match orientation {
        Orientation::Normal => O::NoTransforms,
        Orientation::MirrorHorizontal => O::FlipHorizontal,
        Orientation::Rotate180 => O::Rotate180,
        Orientation::MirrorVertical => O::FlipVertical,
        Orientation::Transpose => O::Rotate90FlipH,
        Orientation::Rotate90 => O::Rotate90,
        Orientation::Transverse => O::Rotate270FlipH,
        Orientation::Rotate270 => O::Rotate270,
    }
}

fn from_orientation(orientation: image::metadata::Orientation) -> Orientation {
    use image::metadata::Orientation as O;
    match orientation {
        O::NoTransforms => Orientation::Normal,
        O::FlipHorizontal => Orientation::MirrorHorizontal,
        O::Rotate180 => Orientation::Rotate180,
        O::FlipVertical => Orientation::MirrorVertical,
        O::Rotate90FlipH => Orientation::Transpose,
        O::Rotate90 => Orientation::Rotate90,
        O::Rotate270FlipH => Orientation::Transverse,
        O::Rotate270 => Orientation::Rotate270,
    }
}

fn codec_error(error: image::ImageError) -> Error {
    Error::backend("image", error.to_string())
}

struct LimitedWriter {
    bytes: Vec<u8>,
    max: usize,
}
impl Write for LimitedWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.len() > self.max.saturating_sub(self.bytes.len()) {
            return Err(io::Error::other(
                "encoded output exceeds configured byte limit",
            ));
        }
        self.bytes
            .try_reserve_exact(bytes.len())
            .map_err(io::Error::other)?;
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

struct BudgetReader<'a> {
    cursor: Cursor<&'a [u8]>,
    count: Rc<Cell<usize>>,
    budget: Rc<Cell<usize>>,
}
impl Read for BudgetReader<'_> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        if buffer.is_empty() {
            return Ok(0);
        }
        let remaining = self.budget.get().saturating_sub(self.count.get());
        if remaining == 0 {
            return Err(io::Error::other("image probe read budget exceeded"));
        }
        let allowed = remaining.min(buffer.len());
        let read = self.cursor.read(&mut buffer[..allowed])?;
        self.count.set(self.count.get() + read);
        Ok(read)
    }
}
impl Seek for BudgetReader<'_> {
    fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
        self.cursor.seek(position)
    }
}
