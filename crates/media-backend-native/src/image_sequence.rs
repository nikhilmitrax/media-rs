//! Bounded composited animation frames and independent TIFF pages.
//!
//! Animation codecs apply disposal/blending to the stored canvas and return
//! straight RGBA8 display frames. Orientation remains explicit metadata. TIFF
//! pages preserve supported native integer/floating-point representations and
//! are independent images with no invented animation timing.
use crate::{
    image::{DecodeLimits, ImageFormat},
    image_color::AlphaMode,
};
use image::{AnimationDecoder, ImageDecoder};
use media_core::{
    error::{Error, Result},
    media::{Image, ImageMetadata, Orientation, PixelFormat},
    time::Time,
};
use std::io::{BufReader, Cursor};

/// Total animation playback count, normalized across container conventions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopCount {
    /// Play the frame sequence once.
    Once,
    /// Repeat forever.
    Infinite,
    /// Total complete plays, including the first; always at least two.
    Finite(u32),
}
/// Per-frame codec limits and total retained sequence bounds.
#[derive(Debug, Clone, Copy)]
pub struct SequenceLimits {
    /// Encoded object, per-frame geometry, metadata and codec limits.
    pub decode: DecodeLimits,
    /// Maximum returned display frames/pages, checked before pixel allocation.
    pub max_frames: usize,
    /// Maximum sum of retained pixel storage and duplicated ICC/EXIF metadata.
    pub max_total_decoded_bytes: usize,
}
impl Default for SequenceLimits {
    fn default() -> Self {
        Self {
            decode: DecodeLimits::default(),
            max_frames: 1024,
            max_total_decoded_bytes: 256 * 1024 * 1024,
        }
    }
}
/// A fully composited display canvas, not a raw changed rectangle.
#[derive(Debug, Clone)]
pub struct DisplayFrame {
    /// Straight RGBA8 stored canvas with retained metadata.
    pub image: Image,
    /// Exact presentation time from the beginning of the first loop.
    pub pts: Time,
    /// Exact encoded duration; zero remains zero without a browser delay clamp.
    pub duration: Time,
    /// Zero-based frame identity.
    pub index: u32,
}
/// Fully decoded bounded animation.
#[derive(Debug, Clone)]
pub struct Animation {
    /// Display frames in presentation order.
    pub frames: Vec<DisplayFrame>,
    /// Complete playback count.
    pub loop_count: LoopCount,
    /// Exact sum of encoded durations for one play.
    pub total_duration: Time,
}
/// Decode GIF, animated WebP, or APNG with disposal and blending applied.
///
/// Still PNG/WebP objects are rejected by this explicit animation API. APNG
/// sixteen-bit compositing is rejected instead of reducing precision to RGBA8.
pub fn decode_animation(bytes: &[u8], limits: SequenceLimits) -> Result<Animation> {
    validate_limits(bytes, limits)?;
    let cursor = || BufReader::new(Cursor::new(bytes));
    let format = image::guess_format(bytes).map_err(image_error)?;
    let (frames, loop_count, metadata, dimensions, expected) = match format {
        image::ImageFormat::Gif => {
            let (expected, loop_count) = gif_structure(bytes)?;
            check_count(expected, limits)?;
            let mut decoder = image::codecs::gif::GifDecoder::new(cursor()).map_err(image_error)?;
            decoder
                .set_limits(limits.decode.codec())
                .map_err(image_error)?;
            let dimensions = decoder.dimensions();
            let metadata = metadata(&mut decoder, limits.decode)?;
            (
                decoder.into_frames(),
                loop_count,
                metadata,
                dimensions,
                expected,
            )
        }
        image::ImageFormat::Png => {
            let (expected, loop_count) = png_structure(bytes)?;
            check_count(expected, limits)?;
            let mut decoder = image::codecs::png::PngDecoder::new(cursor()).map_err(image_error)?;
            if !decoder.is_apng().map_err(image_error)? {
                return Err(Error::invalid("PNG is not animated"));
            }
            if matches!(
                decoder.color_type(),
                image::ColorType::L16
                    | image::ColorType::La16
                    | image::ColorType::Rgb16
                    | image::ColorType::Rgba16
            ) {
                return Err(Error::unsupported(
                    "sixteen-bit APNG compositing cannot be represented faithfully by the RGBA8 animation codec",
                ));
            }
            decoder
                .set_limits(limits.decode.codec())
                .map_err(image_error)?;
            let dimensions = decoder.dimensions();
            let mut metadata = metadata(&mut decoder, limits.decode)?;
            metadata.color = crate::image::encoded_color(bytes, ImageFormat::Png)?.0;
            (
                decoder.apng().map_err(image_error)?.into_frames(),
                loop_count,
                metadata,
                dimensions,
                expected,
            )
        }
        image::ImageFormat::WebP => {
            let (expected, loop_count) = webp_structure(bytes)?;
            check_count(expected, limits)?;
            let mut decoder =
                image::codecs::webp::WebPDecoder::new(cursor()).map_err(image_error)?;
            if !decoder.has_animation() {
                return Err(Error::invalid("WebP is not animated"));
            }
            decoder
                .set_limits(limits.decode.codec())
                .map_err(image_error)?;
            let dimensions = decoder.dimensions();
            let metadata = metadata(&mut decoder, limits.decode)?;
            (
                webp_frames(bytes, limits, dimensions, expected)?,
                loop_count,
                metadata,
                dimensions,
                expected,
            )
        }
        _ => {
            return Err(Error::unsupported(
                "animation decoding supports GIF, animated WebP, and APNG",
            ));
        }
    };
    let frame_bytes = u64::from(dimensions.0)
        .checked_mul(u64::from(dimensions.1))
        .and_then(|v| v.checked_mul(4))
        .ok_or_else(|| Error::overflow("animation canvas overflow"))?;
    limits
        .decode
        .check(dimensions.0, dimensions.1, frame_bytes)?;
    let metadata_bytes = crate::image::metadata_payload_bytes(&metadata)?;
    let planned = frame_bytes
        .checked_add(metadata_bytes as u64)
        .ok_or_else(|| Error::overflow("animation retained length overflow"))?
        .checked_mul(expected as u64)
        .ok_or_else(|| Error::overflow("animation sequence size overflow"))?;
    if planned > limits.max_total_decoded_bytes as u64 {
        return Err(
            Error::exhausted("animation exceeds total retained pixel budget")
                .with_constraint("max_total_decoded_bytes"),
        );
    }
    let mut output = Vec::new();
    let mut pts = Time::ZERO;
    for frame in frames {
        if output.len() >= expected {
            return Err(Error::invalid(
                "animation decoder returned more frames than the validated container declares",
            ));
        }
        let frame = frame.map_err(image_error)?;
        if frame.left() != 0 || frame.top() != 0 || frame.buffer().dimensions() != dimensions {
            return Err(Error::backend(
                "image",
                "animation codec did not return a full composited canvas",
            ));
        }
        let (numerator, denominator) = frame.delay().numer_denom_ms();
        let duration = Time::new(i64::from(numerator), denominator)?.checked_mul_ratio(1, 1000)?;
        let index = u32::try_from(output.len())
            .map_err(|_| Error::overflow("animation frame index overflow"))?;
        let image = Image::from_packed(
            dimensions.0,
            dimensions.1,
            PixelFormat::Rgba8,
            frame.into_buffer().into_raw(),
        )?
        .with_metadata(metadata.clone());
        output.push(DisplayFrame {
            image,
            pts,
            duration,
            index,
        });
        pts = pts.checked_add(duration)?;
    }
    if output.len() != expected {
        return Err(Error::invalid(
            "animation ended before all declared frames were decoded",
        ));
    }
    Ok(Animation {
        frames: output,
        loop_count,
        total_duration: pts,
    })
}
fn webp_frames(
    bytes: &[u8],
    limits: SequenceLimits,
    dimensions: (u32, u32),
    expected: usize,
) -> Result<image::Frames<'_>> {
    // image's wrapper leaves the optional clear color unset, which skips
    // disposal-to-background. Supply the actual ANIM BGRA hint explicitly.
    let mut native =
        image_webp::WebPDecoder::new(BufReader::new(Cursor::new(bytes))).map_err(webp_error)?;
    native.set_memory_limit(usize::try_from(limits.decode.max_decoded_bytes).unwrap_or(usize::MAX));
    let [b, g, r, a] = native
        .background_color_hint()
        .ok_or_else(|| Error::invalid("WebP animation lacks background color"))?;
    if !native.has_alpha() && a != 255 {
        return Err(Error::unsupported(
            "WebP transparent animation background requires the alpha canvas flag",
        ));
    }
    native
        .set_background_color([r, g, b, a])
        .map_err(webp_error)?;
    let mut remaining = expected;
    Ok(image::Frames::new(Box::new(std::iter::from_fn(
        move || {
            if remaining == 0 {
                return None;
            }
            remaining -= 1;
            let mut pixels = vec![
                0u8;
                dimensions.0 as usize
                    * dimensions.1 as usize
                    * if native.has_alpha() { 4 } else { 3 }
            ];
            let delay = match native.read_frame(&mut pixels) {
                Ok(delay) => delay,
                Err(error) => {
                    remaining = 0;
                    return Some(Err(image::ImageError::IoError(std::io::Error::other(
                        error,
                    ))));
                }
            };
            let rgba = if native.has_alpha() {
                pixels
            } else {
                pixels
                    .chunks_exact(3)
                    .flat_map(|p| [p[0], p[1], p[2], 255])
                    .collect()
            };
            Some(Ok(image::Frame::from_parts(
                image::RgbaImage::from_raw(dimensions.0, dimensions.1, rgba)
                    .expect("checked full frame"),
                0,
                0,
                image::Delay::from_numer_denom_ms(delay, 1),
            )))
        },
    ))))
}
fn metadata(decoder: &mut impl ImageDecoder, limits: DecodeLimits) -> Result<ImageMetadata> {
    let icc_profile = decoder.icc_profile().map_err(image_error)?;
    let exif = decoder.exif_metadata().map_err(image_error)?;
    if icc_profile
        .as_ref()
        .map_or(0, Vec::len)
        .checked_add(exif.as_ref().map_or(0, Vec::len))
        .is_none_or(|v| v > limits.max_metadata_bytes)
    {
        return Err(Error::exhausted("animation metadata exceeds byte budget"));
    }
    let orientation = match decoder.orientation().map_err(image_error)? {
        image::metadata::Orientation::NoTransforms => Orientation::Normal,
        image::metadata::Orientation::FlipHorizontal => Orientation::MirrorHorizontal,
        image::metadata::Orientation::Rotate180 => Orientation::Rotate180,
        image::metadata::Orientation::FlipVertical => Orientation::MirrorVertical,
        image::metadata::Orientation::Rotate90FlipH => Orientation::Transpose,
        image::metadata::Orientation::Rotate90 => Orientation::Rotate90,
        image::metadata::Orientation::Rotate270FlipH => Orientation::Transverse,
        image::metadata::Orientation::Rotate270 => Orientation::Rotate270,
    };
    Ok(ImageMetadata {
        icc_profile,
        exif,
        orientation,
        ..ImageMetadata::default()
    })
}
fn validate_limits(bytes: &[u8], limits: SequenceLimits) -> Result<()> {
    if limits.max_frames == 0 || limits.max_total_decoded_bytes == 0 {
        return Err(Error::invalid("sequence bounds must be nonzero"));
    }
    if bytes.len() as u64 > limits.decode.max_encoded_bytes {
        return Err(Error::exhausted("encoded sequence exceeds byte budget"));
    }
    Ok(())
}
fn check_count(count: usize, limits: SequenceLimits) -> Result<()> {
    if count == 0 {
        return Err(Error::invalid("animation has no display frames"));
    }
    if count > limits.max_frames {
        return Err(
            Error::exhausted("sequence exceeds frame count budget").with_constraint("max_frames")
        );
    }
    Ok(())
}
fn total_plays(value: u32) -> LoopCount {
    match value {
        0 => LoopCount::Infinite,
        1 => LoopCount::Once,
        v => LoopCount::Finite(v),
    }
}
fn png_structure(bytes: &[u8]) -> Result<(usize, LoopCount)> {
    let mut offset = 8usize;
    let mut declared = None;
    let mut controls = 0;
    while let Some(header) = bytes.get(offset..offset.saturating_add(8)) {
        let size = u32::from_be_bytes(header[..4].try_into().expect("four bytes")) as usize;
        let start = offset
            .checked_add(8)
            .ok_or_else(|| Error::overflow("PNG chunk overflow"))?;
        let end = start
            .checked_add(size)
            .ok_or_else(|| Error::overflow("PNG chunk overflow"))?;
        let data = bytes
            .get(start..end)
            .ok_or_else(|| Error::invalid("truncated APNG chunk"))?;
        match &header[4..] {
            b"acTL" => {
                if size != 8 || declared.is_some() {
                    return Err(Error::invalid("invalid repeated APNG animation control"));
                }
                declared = Some((
                    u32::from_be_bytes(data[..4].try_into().expect("four bytes")) as usize,
                    total_plays(u32::from_be_bytes(
                        data[4..].try_into().expect("four bytes"),
                    )),
                ));
            }
            b"fcTL" => {
                controls += 1;
            }
            b"IEND" => break,
            _ => {}
        }
        offset = end
            .checked_add(4)
            .ok_or_else(|| Error::overflow("PNG chunk overflow"))?;
    }
    let declared = declared.ok_or_else(|| Error::invalid("PNG is not animated"))?;
    if declared.0 != controls {
        return Err(Error::invalid(
            "APNG frame-control count contradicts animation header",
        ));
    }
    Ok(declared)
}
fn webp_structure(bytes: &[u8]) -> Result<(usize, LoopCount)> {
    let mut offset = 12usize;
    let mut frames = 0;
    let mut plays = None;
    while let Some(header) = bytes.get(offset..offset.saturating_add(8)) {
        let size = u32::from_le_bytes(header[4..].try_into().expect("four bytes")) as usize;
        let start = offset
            .checked_add(8)
            .ok_or_else(|| Error::overflow("WebP chunk overflow"))?;
        let end = start
            .checked_add(size)
            .ok_or_else(|| Error::overflow("WebP chunk overflow"))?;
        let data = bytes
            .get(start..end)
            .ok_or_else(|| Error::invalid("truncated WebP animation chunk"))?;
        match &header[..4] {
            b"ANMF" => frames += 1,
            b"ANIM" => {
                if size != 6 || plays.is_some() {
                    return Err(Error::invalid("invalid WebP animation header"));
                }
                plays = Some(total_plays(u32::from(u16::from_le_bytes([
                    data[4], data[5],
                ]))));
            }
            _ => {}
        }
        offset = end
            .checked_add(size & 1)
            .ok_or_else(|| Error::overflow("WebP chunk overflow"))?;
    }
    Ok((
        frames,
        plays.ok_or_else(|| Error::invalid("WebP is not animated"))?,
    ))
}
fn gif_structure(bytes: &[u8]) -> Result<(usize, LoopCount)> {
    let packed = *bytes
        .get(10)
        .ok_or_else(|| Error::invalid("truncated GIF screen descriptor"))?;
    let mut offset = 13
        + if packed & 0x80 != 0 {
            3 * (1usize << ((packed & 7) + 1))
        } else {
            0
        };
    let mut frames = 0;
    let mut loop_count = LoopCount::Once;
    fn blocks<'a>(bytes: &'a [u8], offset: &mut usize) -> Result<Vec<&'a [u8]>> {
        let mut output = Vec::new();
        loop {
            let size = usize::from(
                *bytes
                    .get(*offset)
                    .ok_or_else(|| Error::invalid("truncated GIF subblock"))?,
            );
            *offset += 1;
            if size == 0 {
                break;
            }
            let end = offset
                .checked_add(size)
                .ok_or_else(|| Error::overflow("GIF block overflow"))?;
            let data = bytes
                .get(*offset..end)
                .ok_or_else(|| Error::invalid("truncated GIF data"))?;
            if output.len() < 2 {
                output.push(data);
            }
            *offset = end;
        }
        Ok(output)
    }
    loop {
        match bytes
            .get(offset)
            .copied()
            .ok_or_else(|| Error::invalid("GIF lacks trailer"))?
        {
            0x3b => break,
            0x21 => {
                let label = *bytes
                    .get(offset + 1)
                    .ok_or_else(|| Error::invalid("truncated GIF extension"))?;
                offset += 2;
                let chunks = blocks(bytes, &mut offset)?;
                if label == 0x01 {
                    return Err(Error::unsupported(
                        "GIF plain-text rendering is not representable as decoded bitmap frames",
                    ));
                }
                if label == 0xf9
                    && chunks
                        .first()
                        .is_some_and(|v| v.first().is_some_and(|packed| packed & 2 != 0))
                {
                    return Err(Error::unsupported(
                        "interactive GIF user-input timing requires an explicit playback policy",
                    ));
                }
                if label == 0xff
                    && chunks
                        .first()
                        .is_some_and(|v| *v == b"NETSCAPE2.0" || *v == b"ANIMEXTS1.0")
                {
                    let data = chunks
                        .get(1)
                        .filter(|v| v.len() == 3 && v[0] == 1)
                        .ok_or_else(|| Error::invalid("invalid GIF loop extension"))?;
                    let repeats = u16::from_le_bytes([data[1], data[2]]);
                    loop_count = if repeats == 0 {
                        LoopCount::Infinite
                    } else {
                        total_plays(u32::from(repeats) + 1)
                    };
                }
            }
            0x2c => {
                let packed = *bytes
                    .get(offset + 9)
                    .ok_or_else(|| Error::invalid("truncated GIF image descriptor"))?;
                offset += 10;
                if packed & 0x80 != 0 {
                    offset += 3 * (1usize << ((packed & 7) + 1));
                }
                offset += 1;
                blocks(bytes, &mut offset)?;
                frames += 1;
            }
            _ => return Err(Error::invalid("invalid GIF block introducer")),
        }
    }
    Ok((frames, loop_count))
}

/// One independent TIFF image directory, without synthetic timing.
#[derive(Debug, Clone)]
pub struct ImagePage {
    /// Pixels preserving native supported component precision.
    pub image: Image,
    /// Zero-based image-directory identity.
    pub index: u32,
    /// Established alpha association, absent when there is no alpha channel.
    pub alpha: Option<AlphaMode>,
}
/// Bounded multipage TIFF output.
#[derive(Debug, Clone)]
pub struct Pages {
    /// Independent pages, potentially with different geometry/pixel format.
    pub pages: Vec<ImagePage>,
    /// TIFF metadata not projected into the portable image metadata model.
    pub metadata_loss: Vec<String>,
}
/// Decode all supported TIFF directories without selecting a silent first page.
///
/// Chunky Gray/RGB/RGBA eight/sixteen-bit and RGB/RGBA f32 are supported.
/// Planar, palette, CMYK, signed, and other precision layouts fail explicitly.
pub fn decode_pages(bytes: &[u8], limits: SequenceLimits) -> Result<Pages> {
    use tiff::{
        ColorType,
        decoder::{Decoder, DecodingResult},
        tags::Tag,
    };
    validate_limits(bytes, limits)?;
    if image::guess_format(bytes).map_err(image_error)? != image::ImageFormat::Tiff {
        return Err(Error::invalid("page decoder requires TIFF"));
    }
    let mut decoder = Decoder::new(Cursor::new(bytes)).map_err(tiff_error)?;
    let mut native_limits = tiff::decoder::Limits::default();
    native_limits.decoding_buffer_size =
        usize::try_from(limits.decode.max_decoded_bytes).unwrap_or(usize::MAX);
    native_limits.intermediate_buffer_size = native_limits.decoding_buffer_size;
    native_limits.ifd_value_size = limits.decode.max_metadata_bytes;
    decoder = decoder.with_limits(native_limits);
    let mut pages = Vec::new();
    let mut retained = 0usize;
    let mut metadata_loss = Vec::new();
    loop {
        if pages.len() >= limits.max_frames {
            return Err(
                Error::exhausted("TIFF exceeds page count budget").with_constraint("max_frames")
            );
        }
        let (width, height) = decoder.dimensions().map_err(tiff_error)?;
        if decoder
            .find_tag_unsigned::<u16>(Tag::PlanarConfiguration)
            .map_err(tiff_error)?
            .unwrap_or(1)
            != 1
        {
            return Err(Error::unsupported(
                "planar TIFF needs explicit plane interleaving",
            ));
        }
        let color = decoder.colortype().map_err(tiff_error)?;
        // tiff 0.11 reports grayscale extra samples as Multiband, even when
        // ExtraSamples establishes exactly one associated/unassociated alpha.
        let gray_alpha = matches!(
            color,
            ColorType::Multiband {
                bit_depth: 8 | 16,
                num_samples: 2
            }
        ) && decoder
            .find_tag_unsigned::<u16>(Tag::PhotometricInterpretation)
            .map_err(tiff_error)?
            == Some(1)
            && matches!(
                decoder
                    .find_tag_unsigned_vec::<u16>(Tag::ExtraSamples)
                    .map_err(tiff_error)?
                    .as_deref(),
                Some([1] | [2])
            );
        let format = match color {
            ColorType::Gray(8) => PixelFormat::Gray8,
            ColorType::Gray(16) => PixelFormat::Gray16,
            ColorType::GrayA(8) => PixelFormat::GrayAlpha8,
            ColorType::GrayA(16) => PixelFormat::GrayAlpha16,
            ColorType::Multiband {
                bit_depth: 8,
                num_samples: 2,
            } if gray_alpha => PixelFormat::GrayAlpha8,
            ColorType::Multiband {
                bit_depth: 16,
                num_samples: 2,
            } if gray_alpha => PixelFormat::GrayAlpha16,
            ColorType::RGB(8) => PixelFormat::Rgb8,
            ColorType::RGBA(8) => PixelFormat::Rgba8,
            ColorType::RGB(16) => PixelFormat::Rgb16,
            ColorType::RGBA(16) => PixelFormat::Rgba16,
            ColorType::RGB(32) => PixelFormat::RgbF32,
            ColorType::RGBA(32) => PixelFormat::RgbaF32,
            _ => {
                return Err(Error::unsupported(format!(
                    "TIFF page color {color:?} needs an explicit representation conversion"
                )));
            }
        };
        let required_sample_format = if matches!(format, PixelFormat::RgbF32 | PixelFormat::RgbaF32)
        {
            3
        } else {
            1
        };
        if decoder
            .find_tag_unsigned_vec::<u16>(Tag::SampleFormat)
            .map_err(tiff_error)?
            .is_some_and(|values| values.iter().any(|v| *v != required_sample_format))
            || (required_sample_format == 3
                && decoder
                    .find_tag(Tag::SampleFormat)
                    .map_err(tiff_error)?
                    .is_none())
        {
            return Err(Error::unsupported(
                "TIFF numeric sample format is incompatible with the requested native precision",
            ));
        }
        let required = u64::from(width)
            .checked_mul(u64::from(height))
            .and_then(|v| v.checked_mul(format.bytes_per_pixel().expect("packed") as u64))
            .ok_or_else(|| Error::overflow("TIFF page size overflow"))?;
        limits.decode.check(width, height, required)?;
        retained = retained
            .checked_add(
                usize::try_from(required)
                    .map_err(|_| Error::overflow("TIFF page exceeds address space"))?,
            )
            .ok_or_else(|| Error::overflow("TIFF retained size overflow"))?;
        if retained > limits.max_total_decoded_bytes {
            return Err(Error::exhausted("TIFF exceeds retained pixel byte budget")
                .with_constraint("max_total_decoded_bytes"));
        }
        let alpha = if matches!(
            format,
            PixelFormat::Rgba8
                | PixelFormat::Rgba16
                | PixelFormat::RgbaF32
                | PixelFormat::GrayAlpha8
                | PixelFormat::GrayAlpha16
        ) {
            match decoder
                .find_tag_unsigned_vec::<u16>(Tag::ExtraSamples)
                .map_err(tiff_error)?
                .as_deref()
            {
                Some([1]) => Some(AlphaMode::Premultiplied),
                Some([2]) => Some(AlphaMode::Straight),
                _ => {
                    return Err(Error::unsupported(
                        "TIFF alpha association is unspecified or has unsupported extra samples",
                    ));
                }
            }
        } else {
            None
        };
        let icc_profile = decoder
            .find_tag(Tag::IccProfile)
            .map_err(tiff_error)?
            .map(|v| v.into_u8_vec())
            .transpose()
            .map_err(tiff_error)?;
        retained = retained
            .checked_add(icc_profile.as_ref().map_or(0, Vec::len))
            .ok_or_else(|| Error::overflow("TIFF retained metadata length overflow"))?;
        if retained > limits.max_total_decoded_bytes {
            return Err(
                Error::exhausted("TIFF metadata exceeds retained byte budget")
                    .with_constraint("max_total_decoded_bytes"),
            );
        }
        let orientation = match decoder
            .find_tag_unsigned::<u16>(Tag::Orientation)
            .map_err(tiff_error)?
            .unwrap_or(1)
        {
            1 => Orientation::Normal,
            2 => Orientation::MirrorHorizontal,
            3 => Orientation::Rotate180,
            4 => Orientation::MirrorVertical,
            5 => Orientation::Transpose,
            6 => Orientation::Rotate90,
            7 => Orientation::Transverse,
            8 => Orientation::Rotate270,
            _ => return Err(Error::invalid("invalid TIFF orientation")),
        };
        if decoder
            .find_tag(Tag::ExifDirectory)
            .map_err(tiff_error)?
            .is_some()
        {
            metadata_loss.push(format!(
                "page {} EXIF directory was not projected",
                pages.len()
            ));
        }
        for (tag, label) in [
            (Tag::GpsDirectory, "GPS directory"),
            (Tag::Unknown(700), "XMP"),
            (Tag::Unknown(33723), "IPTC"),
            (Tag::Unknown(301), "transfer-function tag"),
            (Tag::Unknown(318), "white-point tag"),
            (Tag::Unknown(319), "primary-chromaticity tag"),
        ] {
            if decoder.find_tag(tag).map_err(tiff_error)?.is_some() {
                metadata_loss.push(format!("page {} {label} was not projected", pages.len()));
            }
        }
        let decoded = decoder.read_image().map_err(tiff_error)?;
        let pixels = match (format, decoded) {
            (
                PixelFormat::Gray8
                | PixelFormat::GrayAlpha8
                | PixelFormat::Rgb8
                | PixelFormat::Rgba8,
                DecodingResult::U8(v),
            ) => v,
            (
                PixelFormat::Gray16
                | PixelFormat::GrayAlpha16
                | PixelFormat::Rgb16
                | PixelFormat::Rgba16,
                DecodingResult::U16(v),
            ) => v.into_iter().flat_map(u16::to_le_bytes).collect(),
            (PixelFormat::RgbF32 | PixelFormat::RgbaF32, DecodingResult::F32(v)) => {
                v.into_iter().flat_map(f32::to_le_bytes).collect()
            }
            _ => {
                return Err(Error::unsupported(
                    "TIFF sample numeric type contradicts supported output representation",
                ));
            }
        };
        let image =
            Image::from_packed(width, height, format, pixels)?.with_metadata(ImageMetadata {
                orientation,
                icc_profile,
                ..ImageMetadata::default()
            });
        pages.push(ImagePage {
            image,
            index: u32::try_from(pages.len())
                .map_err(|_| Error::overflow("TIFF page index overflow"))?,
            alpha,
        });
        if !decoder.more_images() {
            break;
        }
        decoder.next_image().map_err(tiff_error)?;
    }
    Ok(Pages {
        pages,
        metadata_loss,
    })
}
fn image_error(error: image::ImageError) -> Error {
    Error::backend("image", error.to_string()).with_cause(error)
}
fn tiff_error(error: tiff::TiffError) -> Error {
    Error::backend("tiff", error.to_string()).with_cause(error)
}
fn webp_error(error: image_webp::DecodingError) -> Error {
    Error::backend("image-webp", error.to_string()).with_cause(error)
}
