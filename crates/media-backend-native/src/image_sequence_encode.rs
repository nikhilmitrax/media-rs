//! Finite lossless animation and native-precision multipage TIFF encoding.
//!
//! Inputs are full composited canvases, not changed rectangles. Timing and
//! palettes are exact: unsupported fractions, partial GIF alpha, and palette
//! quantization fail. Output and typed pixel copies have explicit payload bounds;
//! codec compression tables/private scratch allocations remain unmeasured.
use crate::{
    image::{MetadataPolicy, metadata_payload_bytes},
    image_color::AlphaMode,
    image_sequence::{Animation, ImagePage, LoopCount},
};
use media_core::{
    error::{Error, Result},
    media::{ColorDescription, Image, ImageMetadata, Orientation, PixelFormat},
    time::Time,
};
use std::{
    borrow::Cow,
    collections::HashMap,
    io::{self, Seek, SeekFrom, Write},
};

/// Animation container with exact supported pixels and timing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnimationFormat {
    /// Lossless eight-bit RGBA APNG.
    Apng,
    /// Exact per-frame palettes with binary alpha and centisecond timing.
    Gif,
}
/// Finite sequence input, output and explicit working-payload bounds.
#[derive(Debug, Clone, Copy)]
pub struct SequenceEncodeLimits {
    /// Maximum frames or pages.
    pub max_frames: usize,
    /// Maximum sum of input storage and owned image metadata payload.
    pub max_input_bytes: usize,
    /// Maximum encoded object extent, including seek-based TIFF headers.
    pub max_encoded_bytes: usize,
    /// Allowed output plus largest temporary typed/indexed pixel payload.
    pub max_working_bytes: usize,
}
impl Default for SequenceEncodeLimits {
    fn default() -> Self {
        Self {
            max_frames: 1024,
            max_input_bytes: 256 * 1024 * 1024,
            max_encoded_bytes: 256 * 1024 * 1024,
            max_working_bytes: 512 * 1024 * 1024,
        }
    }
}
/// Exact animation encoding; no implicit timing or color quantization policy.
#[derive(Debug, Clone, Copy)]
pub struct AnimationEncodeOptions {
    /// Explicit output container.
    pub format: AnimationFormat,
    /// Require faithful metadata carriage or report specific allowed loss.
    pub metadata_policy: MetadataPolicy,
    /// Finite input/output/working bounds.
    pub limits: SequenceEncodeLimits,
}
impl AnimationEncodeOptions {
    /// Exact pixels/timing with strict metadata preservation.
    pub fn new(format: AnimationFormat) -> Self {
        Self {
            format,
            metadata_policy: MetadataPolicy::RequirePreservation,
            limits: Default::default(),
        }
    }
}
/// Native-precision uncompressed TIFF encoding with explicit alpha association.
#[derive(Debug, Clone, Copy)]
pub struct TiffEncodeOptions {
    /// Opaque EXIF and unprofiled color declarations require explicit loss.
    pub metadata_policy: MetadataPolicy,
    /// Finite input/output/working bounds.
    pub limits: SequenceEncodeLimits,
}
impl Default for TiffEncodeOptions {
    fn default() -> Self {
        Self {
            metadata_policy: MetadataPolicy::RequirePreservation,
            limits: Default::default(),
        }
    }
}
/// Complete finalized encoded object; no bytes are returned after codec failure.
#[derive(Debug)]
pub struct EncodedSequence {
    /// Entire encoded object within max_encoded_bytes.
    pub bytes: Vec<u8>,
    /// Specific metadata that the selected container could not carry.
    pub metadata_loss: Vec<String>,
    /// Encoded display frames or image directories.
    pub count: usize,
}

/// Conservative explicit payload reservation, without codec/private scratch.
pub fn animation_working_bytes(
    animation: &Animation,
    options: AnimationEncodeOptions,
) -> Result<usize> {
    check_inputs(
        animation.frames.iter().map(|frame| &frame.image),
        options.limits,
    )?;
    let scratch = if options.format == AnimationFormat::Gif {
        animation.frames.iter().try_fold(0, |largest, frame| {
            Ok::<_, Error>(largest.max(frame.image.packed_bytes()?.len() / 4))
        })?
    } else {
        0
    };
    working(scratch, animation.frames.len(), options.limits)
}
/// Conservative output plus largest typed TIFF page copy payload reservation.
pub fn tiff_working_bytes(pages: &[ImagePage], options: TiffEncodeOptions) -> Result<usize> {
    check_inputs(pages.iter().map(|page| &page.image), options.limits)?;
    let scratch = pages
        .iter()
        .filter(|page| {
            !matches!(
                page.image.pixel_format(),
                PixelFormat::Gray8
                    | PixelFormat::GrayAlpha8
                    | PixelFormat::Rgb8
                    | PixelFormat::Rgba8
            )
        })
        .try_fold(0, |largest, page| {
            Ok::<_, Error>(largest.max(page.image.packed_bytes()?.len()))
        })?;
    working(scratch, pages.len(), options.limits)
}
/// Encode every full display frame, validating exact continuous timing first.
pub fn encode_animation(
    animation: &Animation,
    options: AnimationEncodeOptions,
) -> Result<EncodedSequence> {
    animation_working_bytes(animation, options)?;
    let first = animation
        .frames
        .first()
        .ok_or_else(|| Error::invalid("animation has no frames"))?;
    let (width, height) = (first.image.width(), first.image.height());
    let mut pts = Time::ZERO;
    let mut losses = Vec::new();
    loop_plays(animation.loop_count)?;
    for (index, frame) in animation.frames.iter().enumerate() {
        if frame.index as usize != index || frame.pts != pts || frame.duration < Time::ZERO {
            return Err(Error::invalid(
                "animation requires contiguous zero-based frame indices, PTS from zero, and nonnegative durations",
            ));
        }
        if frame.image.pixel_format() != PixelFormat::Rgba8
            || frame.image.width() != width
            || frame.image.height() != height
        {
            return Err(Error::incompatible(
                "animation encoding requires same-canvas composited straight RGBA8 frames",
            ));
        }
        if frame.image.metadata().orientation != Orientation::Normal {
            return Err(Error::unsupported(
                "apply animation frame orientation before encoding",
            ));
        }
        if frame
            .image
            .metadata()
            .exif
            .as_ref()
            .and_then(|exif| image::metadata::Orientation::from_exif_chunk(exif))
            .is_some_and(|orientation| orientation != image::metadata::Orientation::NoTransforms)
        {
            return Err(Error::incompatible(
                "animation EXIF orientation conflicts with normal display canvases",
            ));
        }
        match options.format {
            AnimationFormat::Apng => {
                apng_delay(frame.duration)?;
                if frame.image.metadata() != first.image.metadata() {
                    losses.push(format!(
                        "frame {index} distinct metadata cannot be carried by APNG global metadata"
                    ));
                }
            }
            AnimationFormat::Gif => {
                gif_delay(frame.duration)?;
                gif_palette(frame.image.packed_bytes()?, false)?;
            }
        }
        pts = pts.checked_add(frame.duration)?;
    }
    if pts != animation.total_duration {
        return Err(Error::invalid(
            "animation total_duration contradicts the exact sum of frame durations",
        ));
    }
    match options.format {
        AnimationFormat::Apng => {
            validate_profile(&first.image)?;
            if apng_color(&first.image.metadata().color).is_none() {
                metadata_losses(first.image.metadata(), 0, false, &mut losses);
            }
        }
        AnimationFormat::Gif => {
            for (index, frame) in animation.frames.iter().enumerate() {
                metadata_losses(frame.image.metadata(), index, true, &mut losses);
            }
        }
    }
    check_losses(&losses, options.metadata_policy)?;
    let mut output = BoundedBuffer::new(options.limits.max_encoded_bytes);
    match options.format {
        AnimationFormat::Apng => {
            let metadata = first.image.metadata();
            let mut info = png::Info::with_size(width, height);
            info.color_type = png::ColorType::Rgba;
            info.bit_depth = png::BitDepth::Eight;
            info.icc_profile = metadata
                .icc_profile
                .as_ref()
                .map(|bytes| Cow::Borrowed(bytes.as_slice()));
            info.exif_metadata = metadata
                .exif
                .as_ref()
                .map(|bytes| Cow::Borrowed(bytes.as_slice()));
            let mut encoder = png::Encoder::with_info(&mut output, info).map_err(png_error)?;
            encoder
                .set_animated(
                    animation.frames.len() as u32,
                    loop_plays(animation.loop_count)?,
                )
                .map_err(png_error)?;
            encoder
                .set_blend_op(png::BlendOp::Source)
                .map_err(png_error)?;
            encoder
                .set_dispose_op(png::DisposeOp::None)
                .map_err(png_error)?;
            encoder.validate_sequence(true);
            let mut writer = encoder.write_header().map_err(png_error)?;
            // png 0.18 exposes cICP in Info for decoding but does not serialize
            // that field in its encoder header. Emit the validated chunk before
            // the first IDAT, without inventing a color profile or altering pixels.
            if let Some(color) = apng_color(&metadata.color) {
                writer
                    .write_chunk(
                        png::chunk::cICP,
                        &[
                            color.color_primaries,
                            color.transfer_function,
                            color.matrix_coefficients,
                            u8::from(color.is_video_full_range_image),
                        ],
                    )
                    .map_err(png_error)?;
            }
            for frame in &animation.frames {
                let (numerator, denominator) = apng_delay(frame.duration)?;
                writer
                    .set_frame_delay(numerator, denominator)
                    .map_err(png_error)?;
                writer
                    .write_image_data(frame.image.packed_bytes()?)
                    .map_err(png_error)?;
            }
            writer.finish().map_err(png_error)?;
        }
        AnimationFormat::Gif => {
            let width = u16::try_from(width)
                .map_err(|_| Error::unsupported("GIF canvas width exceeds u16"))?;
            let height = u16::try_from(height)
                .map_err(|_| Error::unsupported("GIF canvas height exceeds u16"))?;
            let mut encoder =
                gif::Encoder::new(&mut output, width, height, &[]).map_err(gif_error)?;
            match animation.loop_count {
                LoopCount::Once => {}
                LoopCount::Infinite => encoder
                    .set_repeat(gif::Repeat::Infinite)
                    .map_err(gif_error)?,
                LoopCount::Finite(plays) => encoder
                    .set_repeat(gif::Repeat::Finite(
                        u16::try_from(plays - 1)
                            .map_err(|_| Error::unsupported("GIF total plays exceed 65,536"))?,
                    ))
                    .map_err(gif_error)?,
            }
            for frame in &animation.frames {
                let (palette, indices, transparent) =
                    gif_palette(frame.image.packed_bytes()?, true)?;
                let encoded = gif::Frame {
                    width,
                    height,
                    delay: gif_delay(frame.duration)?,
                    dispose: gif::DisposalMethod::Background,
                    transparent,
                    palette: Some(palette),
                    buffer: Cow::Owned(indices),
                    ..Default::default()
                };
                encoder.write_frame(&encoded).map_err(gif_error)?;
            }
            encoder.into_inner().map_err(gif_error)?;
        }
    }
    Ok(EncodedSequence {
        bytes: output.bytes,
        metadata_loss: losses,
        count: animation.frames.len(),
    })
}

/// Encode every TIFF directory, retaining native sample precision and orientation.
/// Opaque EXIF directory relocation is unsupported and must be explicitly allowed.
pub fn encode_tiff(pages: &[ImagePage], options: TiffEncodeOptions) -> Result<EncodedSequence> {
    tiff_working_bytes(pages, options)?;
    let mut losses = Vec::new();
    for (index, page) in pages.iter().enumerate() {
        if page.index as usize != index {
            return Err(Error::invalid(
                "TIFF pages require contiguous zero-based directory indices",
            ));
        }
        let alpha = matches!(
            page.image.pixel_format(),
            PixelFormat::GrayAlpha8
                | PixelFormat::GrayAlpha16
                | PixelFormat::Rgba8
                | PixelFormat::Rgba16
                | PixelFormat::RgbaF32
        );
        if alpha != page.alpha.is_some() {
            return Err(Error::invalid(
                "TIFF alpha association must be declared exactly for alpha-bearing pages",
            ));
        }
        match page.image.pixel_format() {
            PixelFormat::Gray8
            | PixelFormat::Gray16
            | PixelFormat::GrayAlpha8
            | PixelFormat::GrayAlpha16
            | PixelFormat::Rgb8
            | PixelFormat::Rgba8
            | PixelFormat::Rgb16
            | PixelFormat::Rgba16
            | PixelFormat::RgbF32
            | PixelFormat::RgbaF32 => {}
            _ => {
                return Err(Error::unsupported(
                    "TIFF output supports Gray8/16, GrayAlpha8/16 and RGB/RGBA8/16/f32",
                ));
            }
        }
        validate_profile(&page.image)?;
        metadata_losses(page.image.metadata(), index, false, &mut losses);
        if page.image.metadata().exif.is_some() {
            losses.push(format!(
                "page {index} opaque EXIF directory cannot be safely relocated into TIFF"
            ));
        }
    }
    check_losses(&losses, options.metadata_policy)?;
    let mut output = BoundedBuffer::new(options.limits.max_encoded_bytes);
    {
        let mut encoder = tiff::encoder::TiffEncoder::new(&mut output).map_err(tiff_error)?;
        use tiff::encoder::colortype as ct;
        for page in pages {
            let bytes = page.image.packed_bytes()?;
            match page.image.pixel_format() {
                PixelFormat::Gray8 | PixelFormat::GrayAlpha8 => {
                    tiff_page::<ct::Gray8>(&mut encoder, page, bytes)?
                }
                PixelFormat::Rgb8 | PixelFormat::Rgba8 => {
                    tiff_page::<ct::RGB8>(&mut encoder, page, bytes)?
                }
                PixelFormat::Gray16 | PixelFormat::GrayAlpha16 => {
                    let values = typed_samples(
                        bytes
                            .chunks_exact(2)
                            .map(|value| u16::from_le_bytes(value.try_into().expect("two bytes"))),
                        bytes.len() / 2,
                    )?;
                    tiff_page::<ct::Gray16>(&mut encoder, page, &values)?;
                }
                PixelFormat::Rgb16 | PixelFormat::Rgba16 => {
                    let values = typed_samples(
                        bytes
                            .chunks_exact(2)
                            .map(|value| u16::from_le_bytes(value.try_into().expect("two bytes"))),
                        bytes.len() / 2,
                    )?;
                    tiff_page::<ct::RGB16>(&mut encoder, page, &values)?;
                }
                PixelFormat::RgbF32 | PixelFormat::RgbaF32 => {
                    let values = typed_samples(
                        bytes
                            .chunks_exact(4)
                            .map(|value| f32::from_le_bytes(value.try_into().expect("four bytes"))),
                        bytes.len() / 4,
                    )?;
                    tiff_page::<ct::RGB32Float>(&mut encoder, page, &values)?;
                }
                _ => unreachable!("validated pixel format"),
            }
        }
    }
    Ok(EncodedSequence {
        bytes: output.bytes,
        metadata_loss: losses,
        count: pages.len(),
    })
}
fn typed_samples<T>(samples: impl Iterator<Item = T>, count: usize) -> Result<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| Error::exhausted("cannot allocate typed TIFF sample payload"))?;
    values.extend(samples);
    Ok(values)
}
fn tiff_page<C: tiff::encoder::colortype::ColorType>(
    encoder: &mut tiff::encoder::TiffEncoder<&mut BoundedBuffer>,
    page: &ImagePage,
    values: &[C::Inner],
) -> Result<()>
where
    [C::Inner]: tiff::encoder::TiffValue,
{
    use tiff::tags::{ExtraSamples, Tag};
    let mut encoded = encoder
        .new_image::<C>(page.image.width(), page.image.height())
        .map_err(tiff_error)?;
    let row_bytes = page.image.width() as usize
        * page
            .image
            .pixel_format()
            .bytes_per_pixel()
            .expect("validated packed page");
    let rows = (64 * 1024 / row_bytes)
        .max(1)
        .min(page.image.height() as usize);
    encoded.rows_per_strip(rows as u32).map_err(tiff_error)?;
    if let Some(alpha) = page.alpha {
        encoded
            .extra_samples(&[match alpha {
                AlphaMode::Straight => ExtraSamples::UnassociatedAlpha,
                AlphaMode::Premultiplied => ExtraSamples::AssociatedAlpha,
            }])
            .map_err(tiff_error)?;
    }
    encoded
        .encoder()
        .write_tag(
            Tag::Orientation,
            orientation(page.image.metadata().orientation),
        )
        .map_err(tiff_error)?;
    if let Some(profile) = &page.image.metadata().icc_profile {
        encoded
            .encoder()
            .write_tag(Tag::IccProfile, profile.as_slice())
            .map_err(tiff_error)?;
    }
    encoded.write_data(values).map_err(tiff_error)
}
fn orientation(value: Orientation) -> u16 {
    match value {
        Orientation::Normal => 1,
        Orientation::MirrorHorizontal => 2,
        Orientation::Rotate180 => 3,
        Orientation::MirrorVertical => 4,
        Orientation::Transpose => 5,
        Orientation::Rotate90 => 6,
        Orientation::Transverse => 7,
        Orientation::Rotate270 => 8,
    }
}
fn check_inputs<'a>(
    images: impl Iterator<Item = &'a Image>,
    limits: SequenceEncodeLimits,
) -> Result<()> {
    if limits.max_frames == 0
        || limits.max_input_bytes == 0
        || limits.max_encoded_bytes == 0
        || limits.max_working_bytes == 0
    {
        return Err(Error::invalid("sequence encoding limits must be nonzero"));
    }
    let mut count = 0usize;
    let mut bytes = 0usize;
    for image in images {
        count = count
            .checked_add(1)
            .ok_or_else(|| Error::overflow("sequence count overflow"))?;
        let metadata = metadata_payload_bytes(image.metadata())?;
        bytes = bytes
            .checked_add(image.storage().byte_len())
            .and_then(|bytes| bytes.checked_add(metadata))
            .ok_or_else(|| Error::overflow("sequence input payload overflow"))?;
        image.packed_bytes()?;
        if count > limits.max_frames || count > u32::MAX as usize {
            return Err(Error::exhausted(
                "sequence exceeds encode frame/page count bound",
            ));
        }
        if bytes > limits.max_input_bytes {
            return Err(Error::exhausted(
                "sequence exceeds encode input payload bound",
            ));
        }
    }
    if count == 0 {
        return Err(Error::invalid("cannot encode an empty sequence"));
    }
    Ok(())
}
fn working(scratch: usize, count: usize, limits: SequenceEncodeLimits) -> Result<usize> {
    let bytes = limits
        .max_encoded_bytes
        .checked_add(scratch)
        .and_then(|bytes| {
            count
                .checked_mul(1024)
                .and_then(|metadata| bytes.checked_add(metadata))
        })
        .and_then(|bytes| bytes.checked_add(4096))
        .ok_or_else(|| Error::overflow("sequence encoding working payload overflow"))?;
    if bytes > limits.max_working_bytes {
        return Err(Error::exhausted(
            "sequence encoding exceeds working payload bound",
        ));
    }
    Ok(bytes)
}
fn validate_profile(image: &Image) -> Result<()> {
    if let Some(bytes) = &image.metadata().icc_profile {
        let profile = moxcms::ColorProfile::new_from_slice(bytes)
            .map_err(|error| Error::backend("moxcms", error.to_string()).with_cause(error))?;
        let expected = if matches!(
            image.pixel_format(),
            PixelFormat::Gray8
                | PixelFormat::Gray16
                | PixelFormat::GrayAlpha8
                | PixelFormat::GrayAlpha16
        ) {
            moxcms::DataColorSpace::Gray
        } else {
            moxcms::DataColorSpace::Rgb
        };
        if profile.color_space != expected {
            return Err(Error::incompatible(
                "sequence pixel channels and ICC device color space disagree",
            ));
        }
    }
    Ok(())
}
fn metadata_losses(metadata: &ImageMetadata, index: usize, gif: bool, losses: &mut Vec<String>) {
    if gif && metadata.icc_profile.is_some() {
        losses.push(format!("frame {index} GIF encoder cannot carry ICC"));
    }
    if gif && metadata.exif.is_some() {
        losses.push(format!("frame {index} GIF encoder cannot carry EXIF"));
    }
    if metadata.color != ColorDescription::default() && (gif || metadata.icc_profile.is_none()) {
        losses.push(format!(
            "image {index} color description cannot be serialized without a supported ICC profile"
        ));
    }
}
fn apng_color(color: &ColorDescription) -> Option<png::CodingIndependentCodePoints> {
    if color.matrix.as_deref() != Some("rgb") || color.chroma_siting.is_some() {
        return None;
    }
    let full = match color.range.as_deref()? {
        "full" | "pc" => true,
        "limited" | "tv" => false,
        _ => return None,
    };
    let primaries = match color.primaries.as_deref() {
        None => 2,
        Some("bt709") => 1,
        Some("bt2020") => 9,
        Some("display-p3") => 12,
        Some(value) => value.strip_prefix("cicp:")?.parse().ok()?,
    };
    let transfer = match color.transfer.as_deref() {
        None => 2,
        Some("bt709") => 1,
        Some("linear") => 8,
        Some("srgb") => 13,
        Some("smpte2084") => 16,
        Some("hlg") => 18,
        Some(value) => value.strip_prefix("cicp:")?.parse().ok()?,
    };
    Some(png::CodingIndependentCodePoints {
        color_primaries: primaries,
        transfer_function: transfer,
        matrix_coefficients: 0,
        is_video_full_range_image: full,
    })
}
fn check_losses(losses: &[String], policy: MetadataPolicy) -> Result<()> {
    if !losses.is_empty() && policy == MetadataPolicy::RequirePreservation {
        return Err(Error::unsupported(format!(
            "sequence metadata preservation failed: {losses:?}; explicitly allow metadata loss"
        )));
    }
    Ok(())
}
fn loop_plays(value: LoopCount) -> Result<u32> {
    match value {
        LoopCount::Infinite => Ok(0),
        LoopCount::Once => Ok(1),
        LoopCount::Finite(value) if value >= 2 => Ok(value),
        _ => Err(Error::invalid(
            "finite animation loop count must be at least two total plays",
        )),
    }
}
fn apng_delay(value: Time) -> Result<(u16, u16)> {
    Ok((
        u16::try_from(value.numerator()).map_err(|_| {
            Error::unsupported("APNG duration numerator is not exactly representable as u16")
        })?,
        u16::try_from(value.denominator()).map_err(|_| {
            Error::unsupported("APNG duration denominator is not exactly representable as u16")
        })?,
    ))
}
fn gif_delay(value: Time) -> Result<u16> {
    let numerator = i128::from(value.numerator()) * 100;
    let denominator = i128::from(value.denominator());
    if numerator < 0 || numerator % denominator != 0 {
        return Err(Error::unsupported(
            "GIF duration must be exactly representable in centiseconds; timing quantization is not implicit",
        ));
    }
    u16::try_from(numerator / denominator)
        .map_err(|_| Error::unsupported("GIF duration exceeds 65,535 centiseconds"))
}
type GifPalette = (Vec<u8>, Vec<u8>, Option<u8>);
fn gif_palette(bytes: &[u8], indexed: bool) -> Result<GifPalette> {
    let transparent = bytes.chunks_exact(4).any(|pixel| pixel[3] == 0);
    let mut palette = if transparent {
        vec![0, 0, 0]
    } else {
        Vec::new()
    };
    let mut map = HashMap::new();
    let mut indices = Vec::new();
    if indexed {
        indices
            .try_reserve_exact(bytes.len() / 4)
            .map_err(|_| Error::exhausted("cannot allocate GIF indices"))?;
    }
    for pixel in bytes.chunks_exact(4) {
        let index = match pixel[3] {
            0 => {
                if pixel[..3] != [0, 0, 0] {
                    return Err(Error::unsupported(
                        "GIF cannot preserve hidden RGB of transparent pixels; explicitly normalize those pixels first",
                    ));
                }
                0
            }
            255 => {
                let color = [pixel[0], pixel[1], pixel[2]];
                if let Some(index) = map.get(&color) {
                    *index
                } else {
                    let index = palette.len() / 3;
                    if index >= 256 {
                        return Err(Error::unsupported(
                            "GIF requires more than 256 exact palette entries; color quantization must be explicit",
                        ));
                    }
                    palette.extend_from_slice(&color);
                    map.insert(color, index as u8);
                    index as u8
                }
            }
            _ => {
                return Err(Error::unsupported(
                    "GIF binary alpha cannot preserve partial transparency",
                ));
            }
        };
        if indexed {
            indices.push(index);
        }
    }
    Ok((palette, indices, transparent.then_some(0)))
}
fn png_error(error: png::EncodingError) -> Error {
    if matches!(&error, png::EncodingError::IoError(io) if is_output_bound(io)) {
        return Error::exhausted("encoded sequence exceeds output byte bound")
            .with_constraint("max_encoded_bytes")
            .with_cause(error);
    }
    Error::backend("png", error.to_string()).with_cause(error)
}
fn gif_error(error: gif::EncodingError) -> Error {
    if matches!(&error, gif::EncodingError::Io(io) if is_output_bound(io)) {
        return Error::exhausted("encoded sequence exceeds output byte bound")
            .with_constraint("max_encoded_bytes")
            .with_cause(error);
    }
    Error::backend("gif", error.to_string()).with_cause(error)
}
fn tiff_error(error: tiff::TiffError) -> Error {
    if matches!(&error, tiff::TiffError::IoError(io) if is_output_bound(io)) {
        return Error::exhausted("encoded sequence exceeds output byte bound")
            .with_constraint("max_encoded_bytes")
            .with_cause(error);
    }
    Error::backend("tiff", error.to_string()).with_cause(error)
}
#[derive(Debug)]
struct OutputBound;
impl std::fmt::Display for OutputBound {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("encoded sequence exceeds output byte bound")
    }
}
impl std::error::Error for OutputBound {}
fn is_output_bound(error: &io::Error) -> bool {
    error
        .get_ref()
        .is_some_and(|error| error.is::<OutputBound>())
}

/// Checked extent writer: seeks never grow or escape the allowed output bound.
struct BoundedBuffer {
    bytes: Vec<u8>,
    position: usize,
    limit: usize,
}
impl BoundedBuffer {
    fn new(limit: usize) -> Self {
        Self {
            bytes: Vec::new(),
            position: 0,
            limit,
        }
    }
}
impl Write for BoundedBuffer {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let end = self
            .position
            .checked_add(bytes.len())
            .filter(|end| *end <= self.limit)
            .ok_or_else(|| io::Error::other(OutputBound))?;
        if end > self.bytes.len() {
            self.bytes
                .try_reserve_exact(end - self.bytes.len())
                .map_err(|_| io::Error::other("cannot allocate bounded encoded sequence"))?;
            self.bytes.resize(end, 0);
        }
        self.bytes[self.position..end].copy_from_slice(bytes);
        self.position = end;
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
impl Seek for BoundedBuffer {
    fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
        let next = match position {
            SeekFrom::Start(value) => i128::from(value),
            SeekFrom::Current(value) => self.position as i128 + i128::from(value),
            SeekFrom::End(value) => self.bytes.len() as i128 + i128::from(value),
        };
        if next < 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "negative encoded sequence seek",
            ));
        }
        if next > self.limit as i128 {
            return Err(io::Error::other(OutputBound));
        }
        self.position = next as usize;
        Ok(self.position as u64)
    }
}
