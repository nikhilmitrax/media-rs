//! Checked views over existing host pixels; conversions remain explicit.

use media_core::{
    device::Device,
    error::{Error, Result},
    media::{Image, PixelFormat},
    storage::MemoryKind,
};

/// Owned lease retaining an image allocation while exposing its visible byte rows.
///
/// Pitched rows are supported. No implicit download, repack, color conversion, or
/// orientation operation occurs. Each row excludes right-side allocation padding.
#[derive(Clone, Debug)]
pub struct HostImageView {
    image: Image,
    row_bytes: usize,
}
impl HostImageView {
    /// Check packed-pixel addressing and host placement, then retain the allocation.
    pub fn new(image: &Image) -> Result<Self> {
        if image.storage().device() != Device::Cpu
            || !matches!(
                image.storage().kind(),
                MemoryKind::Host | MemoryKind::PinnedHost
            )
        {
            return Err(Error::incompatible(
                "host image view requires existing CPU-addressable storage; transfer explicitly",
            ));
        }
        let bpp = image.pixel_format().bytes_per_pixel().ok_or_else(|| {
            Error::incompatible("multiplanar pixels need a plane-specific or converted view")
        })?;
        let row_bytes = (image.width() as usize)
            .checked_mul(bpp)
            .ok_or_else(|| Error::overflow("visible row bytes"))?;
        if image.layout().planes.len() != 1 {
            return Err(Error::incompatible(
                "packed-pixel image view requires one plane",
            ));
        }
        let plane = &image.layout().planes[0];
        if plane.row_bytes < row_bytes || plane.rows < image.height() as usize {
            return Err(Error::incompatible(
                "image plane cannot cover visible geometry",
            ));
        }
        // Wait on this producer dependency only; no device transfer is attempted.
        image.host_bytes()?;
        Ok(Self {
            image: image.clone(),
            row_bytes,
        })
    }
    /// Visible width.
    pub fn width(&self) -> u32 {
        self.image.width()
    }
    /// Visible height.
    pub fn height(&self) -> u32 {
        self.image.height()
    }
    /// Stored pixel representation.
    pub fn pixel_format(&self) -> PixelFormat {
        self.image.pixel_format()
    }
    /// Original color/orientation/profile metadata retained alongside the raw view.
    pub fn metadata(&self) -> &media_core::media::ImageMetadata {
        self.image.metadata()
    }
    /// A visible row, excluding pitch padding and retaining the original byte order.
    pub fn row(&self, index: usize) -> Result<&[u8]> {
        if index >= self.height() as usize {
            return Err(Error::invalid("image row index out of bounds"));
        }
        let plane = &self.image.layout().planes[0];
        let start = plane
            .row_stride
            .checked_mul(index)
            .and_then(|v| v.checked_add(plane.offset))
            .ok_or_else(|| Error::overflow("image row address"))?;
        let end = start
            .checked_add(self.row_bytes)
            .ok_or_else(|| Error::overflow("image row end"))?;
        self.image
            .host_bytes()?
            .get(start..end)
            .ok_or_else(|| Error::incompatible("visible image row exceeds allocation"))
    }
    /// Existing RGB8 image-crate view; padded rows require explicit repacking first.
    pub fn try_rgb8(&self) -> Result<::image::ImageBuffer<::image::Rgb<u8>, &[u8]>> {
        try_rgb8_view(&self.image)
    }
    /// Existing RGBA8 image-crate view; padded rows require explicit repacking first.
    pub fn try_rgba8(&self) -> Result<::image::ImageBuffer<::image::Rgba<u8>, &[u8]>> {
        try_rgba8_view(&self.image)
    }
}

/// Borrow tightly packed RGB8 values without copying pixels.
/// Primaries/transfer are unspecified in this raw view; retain original Image metadata.
pub fn try_rgb8_view(image: &Image) -> Result<::image::ImageBuffer<::image::Rgb<u8>, &[u8]>> {
    checked_packed(image, PixelFormat::Rgb8)?;
    let mut view =
        ::image::ImageBuffer::from_raw(image.width(), image.height(), visible_packed_bytes(image)?)
            .ok_or_else(|| {
                Error::incompatible("RGB8 view dimensions cannot cover the allocation")
            })?;
    view.set_rgb_primaries(::image::metadata::CicpColorPrimaries::Unspecified);
    view.set_transfer_function(::image::metadata::CicpTransferCharacteristics::Unspecified);
    Ok(view)
}
/// Borrow an existing tightly packed RGBA8 image without dropping alpha.
pub fn try_rgba8_view(image: &Image) -> Result<::image::ImageBuffer<::image::Rgba<u8>, &[u8]>> {
    checked_packed(image, PixelFormat::Rgba8)?;
    let mut view =
        ::image::ImageBuffer::from_raw(image.width(), image.height(), visible_packed_bytes(image)?)
            .ok_or_else(|| {
                Error::incompatible("RGBA8 view dimensions cannot cover the allocation")
            })?;
    view.set_rgb_primaries(::image::metadata::CicpColorPrimaries::Unspecified);
    view.set_transfer_function(::image::metadata::CicpTransferCharacteristics::Unspecified);
    Ok(view)
}
fn checked_packed(image: &Image, format: PixelFormat) -> Result<()> {
    if image.storage().device() != Device::Cpu
        || !matches!(
            image.storage().kind(),
            MemoryKind::Host | MemoryKind::PinnedHost
        )
        || image.pixel_format() != format
        || !image.layout().is_packed()
    {
        return Err(Error::incompatible(
            "image-crate view needs the exact packed host representation; convert explicitly",
        ));
    }
    Ok(())
}
fn visible_packed_bytes(image: &Image) -> Result<&[u8]> {
    let count = (image.width() as usize)
        .checked_mul(image.height() as usize)
        .and_then(|n| n.checked_mul(image.pixel_format().bytes_per_pixel()?))
        .ok_or_else(|| Error::overflow("packed image length"))?;
    image
        .packed_bytes()?
        .get(..count)
        .ok_or_else(|| Error::incompatible("packed image is truncated"))
}
