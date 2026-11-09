//! Validated media descriptions, timing, and retained immutable storage.
use crate::{
    error::{Error, Result},
    storage::{MemoryKind, StorageLease},
    time::Time,
};
use std::sync::Arc;
/// Pixel representation; multi-byte host scalars use little-endian storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PixelFormat {
    /// 8-bit luminance.
    Gray8,
    /// 8-bit luminance/alpha.
    GrayAlpha8,
    /// Packed 8-bit RGB.
    Rgb8,
    /// Packed 8-bit RGBA.
    Rgba8,
    /// 16-bit luminance.
    Gray16,
    /// 16-bit luminance/alpha.
    GrayAlpha16,
    /// Packed 16-bit RGB.
    Rgb16,
    /// Packed 16-bit RGBA.
    Rgba16,
    /// Packed f32 RGB.
    RgbF32,
    /// Packed f32 RGBA.
    RgbaF32,
    /// 8-bit Y and interleaved 4:2:0 UV.
    Nv12,
    /// 10 valid bits in 16-bit Y/UV storage.
    P010,
    /// Separate 8-bit 4:2:0 Y/U/V planes.
    Yuv420p,
}
impl PixelFormat {
    /// Packed bytes per pixel; absent for multiplanar formats.
    pub fn bytes_per_pixel(self) -> Option<usize> {
        match self {
            Self::Gray8 => Some(1),
            Self::GrayAlpha8 | Self::Gray16 => Some(2),
            Self::Rgb8 => Some(3),
            Self::Rgba8 | Self::GrayAlpha16 => Some(4),
            Self::Rgb16 => Some(6),
            Self::Rgba16 => Some(8),
            Self::RgbF32 => Some(12),
            Self::RgbaF32 => Some(16),
            Self::Nv12 | Self::P010 | Self::Yuv420p => None,
        }
    }
    /// Component count excluding subsampling geometry.
    pub fn channels(self) -> usize {
        match self {
            Self::Gray8 | Self::Gray16 => 1,
            Self::GrayAlpha8 | Self::GrayAlpha16 => 2,
            Self::Rgba8 | Self::Rgba16 | Self::RgbaF32 => 4,
            _ => 3,
        }
    }
}
/// Addressable plane; absent plane layouts describe an opaque surface.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlaneLayout {
    /// Byte offset.
    pub offset: usize,
    /// Bytes between row starts.
    pub row_stride: usize,
    /// Sample bytes per row, excluding padding.
    pub row_bytes: usize,
    /// Allocated rows.
    pub rows: usize,
}
impl PlaneLayout {
    /// Last addressed byte plus one, checked.
    pub fn extent(&self) -> Result<usize> {
        if self.rows == 0 || self.row_bytes == 0 || self.row_stride < self.row_bytes {
            return Err(Error::invalid(
                "plane needs nonzero rows/bytes and sufficient pitch",
            ));
        }
        self.row_stride
            .checked_mul(self.rows - 1)
            .and_then(|v| v.checked_add(self.row_bytes))
            .and_then(|v| v.checked_add(self.offset))
            .ok_or_else(|| Error::overflow("plane address overflow"))
    }
}
/// Visible geometry and native allocation layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageLayout {
    /// Visible width.
    pub width: u32,
    /// Visible height.
    pub height: u32,
    /// Allocation width including padding.
    pub allocated_width: u32,
    /// Allocation height including padding.
    pub allocated_height: u32,
    /// Pixel representation.
    pub format: PixelFormat,
    /// Addressable planes; empty only for opaque native surfaces.
    pub planes: Vec<PlaneLayout>,
}
impl ImageLayout {
    /// Tightly packed visible host layout.
    pub fn packed(width: u32, height: u32, format: PixelFormat) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(Error::invalid("image dimensions must be nonzero"));
        }
        let bpp = format
            .bytes_per_pixel()
            .ok_or_else(|| Error::invalid("multiplanar format needs explicit planes"))?;
        let row = (width as usize)
            .checked_mul(bpp)
            .ok_or_else(|| Error::overflow("image row overflow"))?;
        Ok(Self {
            width,
            height,
            allocated_width: width,
            allocated_height: height,
            format,
            planes: vec![PlaneLayout {
                offset: 0,
                row_stride: row,
                row_bytes: row,
                rows: height as usize,
            }],
        })
    }
    /// Whether visible bytes form exactly one contiguous packed image.
    pub fn is_packed(&self) -> bool {
        let Some(bpp) = self.format.bytes_per_pixel() else {
            return false;
        };
        let Some(row) = (self.width as usize).checked_mul(bpp) else {
            return false;
        };
        self.width == self.allocated_width
            && self.height == self.allocated_height
            && self.planes.len() == 1
            && self.planes[0]
                == PlaneLayout {
                    offset: 0,
                    row_stride: row,
                    row_bytes: row,
                    rows: self.height as usize,
                }
    }
    /// Validate all addressed bytes and minimum plane geometry.
    pub fn validate(&self, storage: &StorageLease) -> Result<()> {
        if self.width == 0
            || self.height == 0
            || self.allocated_width < self.width
            || self.allocated_height < self.height
        {
            return Err(Error::invalid("invalid visible/allocation dimensions"));
        }
        if storage.kind() == MemoryKind::Opaque {
            if !self.planes.is_empty() {
                return Err(Error::invalid(
                    "opaque surface cannot claim byte-addressed planes",
                ));
            }
            return Ok(());
        }
        let (w, h) = (
            self.allocated_width as usize,
            self.allocated_height as usize,
        );
        let mul = |a: usize, b: usize| {
            a.checked_mul(b)
                .ok_or_else(|| Error::overflow("plane dimensions overflow"))
        };
        let expected = match self.format.bytes_per_pixel() {
            Some(bpp) => vec![(mul(w, bpp)?, h)],
            None => {
                let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
                match self.format {
                    PixelFormat::Nv12 => vec![(w, h), (mul(cw, 2)?, ch)],
                    PixelFormat::P010 => vec![(mul(w, 2)?, h), (mul(cw, 4)?, ch)],
                    PixelFormat::Yuv420p => vec![(w, h), (cw, ch), (cw, ch)],
                    _ => return Err(Error::invalid("unknown plane format")),
                }
            }
        };
        if expected.len() != self.planes.len() {
            return Err(Error::invalid("plane count disagrees with pixel format"));
        }
        let mut previous_end = 0;
        for (plane, (row, rows)) in self.planes.iter().zip(expected) {
            let end = plane.extent()?;
            if plane.row_bytes < row
                || plane.rows < rows
                || plane.offset < previous_end
                || end > storage.byte_len()
            {
                return Err(Error::invalid(
                    "planes overlap, exceed storage, or cannot cover the image",
                ));
            }
            previous_end = end;
        }
        Ok(())
    }
}
/// EXIF display transform, separate from pixel mutation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Orientation {
    /// Identity.
    #[default]
    Normal,
    /// Left/right reflection.
    MirrorHorizontal,
    /// Half turn.
    Rotate180,
    /// Top/bottom reflection.
    MirrorVertical,
    /// Main-diagonal reflection.
    Transpose,
    /// Clockwise quarter turn.
    Rotate90,
    /// Opposite-diagonal reflection.
    Transverse,
    /// Clockwise three-quarter turn.
    Rotate270,
}
/// Color interpretation; missing metadata remains unknown.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ColorDescription {
    /// Color primaries.
    pub primaries: Option<String>,
    /// Transfer function.
    pub transfer: Option<String>,
    /// Matrix coefficients.
    pub matrix: Option<String>,
    /// Full/limited range.
    pub range: Option<String>,
    /// Chroma sample position.
    pub chroma_siting: Option<String>,
}
/// Metadata retained independently of the shared pixel allocation.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ImageMetadata {
    /// Display orientation still required by the pixels.
    pub orientation: Orientation,
    /// ICC profile.
    pub icc_profile: Option<Vec<u8>>,
    /// EXIF payload.
    pub exif: Option<Vec<u8>>,
    /// Established color description.
    pub color: ColorDescription,
}
/// Immutable image; cloning retains its allocation and readiness.
#[derive(Debug, Clone)]
pub struct Image {
    layout: ImageLayout,
    metadata: Arc<ImageMetadata>,
    // Release metadata before storage guards return combined payload charges.
    storage: StorageLease,
}
impl Image {
    /// Validate and retain native storage.
    pub fn new(
        layout: ImageLayout,
        storage: StorageLease,
        metadata: ImageMetadata,
    ) -> Result<Self> {
        layout.validate(&storage)?;
        Ok(Self {
            layout,
            storage,
            metadata: Arc::new(metadata),
        })
    }
    /// Own packed host bytes without a copy.
    pub fn from_packed(
        width: u32,
        height: u32,
        format: PixelFormat,
        bytes: Vec<u8>,
    ) -> Result<Self> {
        let layout = ImageLayout::packed(width, height, format)?;
        if bytes.len() != layout.planes[0].extent()? {
            return Err(Error::invalid(
                "packed image byte count disagrees with dimensions/format",
            ));
        }
        Self::new(
            layout,
            StorageLease::from_bytes(bytes),
            ImageMetadata::default(),
        )
    }
    /// Visible width.
    pub fn width(&self) -> u32 {
        self.layout.width
    }
    /// Visible height.
    pub fn height(&self) -> u32 {
        self.layout.height
    }
    /// Pixel representation.
    pub fn pixel_format(&self) -> PixelFormat {
        self.layout.format
    }
    /// Complete layout.
    pub fn layout(&self) -> &ImageLayout {
        &self.layout
    }
    /// Retained storage.
    pub fn storage(&self) -> &StorageLease {
        &self.storage
    }
    /// Metadata.
    pub fn metadata(&self) -> &ImageMetadata {
        &self.metadata
    }
    /// Replace metadata without mutating shared pixels.
    pub fn with_metadata(mut self, metadata: ImageMetadata) -> Self {
        self.metadata = Arc::new(metadata);
        self
    }
    /// Retain an allocation budget or external lifetime guard with this image.
    pub fn retain<T: std::fmt::Debug + Send + Sync + 'static>(mut self, guard: T) -> Self {
        self.storage = self.storage.retain(guard);
        self
    }
    /// Existing host allocation bytes, interpreted using the plane layout.
    pub fn host_bytes(&self) -> Result<&[u8]> {
        self.storage.host_bytes()
    }
    /// Compatible packed visible bytes, rejecting padded/opaque storage.
    pub fn packed_bytes(&self) -> Result<&[u8]> {
        if !self.layout.is_packed() {
            return Err(Error::incompatible(
                "image needs explicit repacking for a packed view",
            ));
        }
        self.storage.host_bytes()
    }
    /// Explicitly duplicate host pixels; immutable metadata remains shared.
    pub fn deep_copy(&self) -> Result<Self> {
        Ok(Self {
            layout: self.layout.clone(),
            storage: self.storage.deep_copy_host()?,
            metadata: self.metadata.clone(),
        })
    }
}
/// Logical stream identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TrackId(pub u32);
/// Presentation/source timing, independent of packet/decode order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FrameTiming {
    /// Asset presentation time.
    pub pts: Option<Time>,
    /// Established display duration.
    pub duration: Option<Time>,
    /// Backend-reported packet/codec duration hint. This does not by itself
    /// establish a presentation interval, especially for VFR and reordered video.
    pub native_duration: Option<Time>,
    /// Original source time.
    pub source_pts: Option<Time>,
    /// Established presentation ordinal.
    pub ordinal: Option<u64>,
}
/// Spatial image and video timing.
#[derive(Debug, Clone)]
pub struct VideoFrame {
    /// Retained image.
    pub image: Image,
    /// Logical stream.
    pub track: TrackId,
    /// Source/presentation timing.
    pub timing: FrameTiming,
}
/// Compressed packet with optional timing.
#[derive(Debug, Clone)]
pub struct Packet {
    /// Logical stream.
    pub track: TrackId,
    /// Shared compressed bytes.
    pub data: Arc<[u8]>,
    /// Presentation time.
    pub pts: Option<Time>,
    /// Decode time.
    pub dts: Option<Time>,
    /// Established duration.
    pub duration: Option<Time>,
    /// Suitable random-access point.
    pub keyframe: bool,
}
/// Speaker identity; channel count alone does not define a mix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChannelPosition {
    /// Mono.
    Mono,
    /// Front left.
    FrontLeft,
    /// Front right.
    FrontRight,
    /// Front center.
    FrontCenter,
    /// Low-frequency effects.
    Lfe,
    /// Back left.
    BackLeft,
    /// Back right.
    BackRight,
    /// Side left.
    SideLeft,
    /// Side right.
    SideRight,
    /// Explicit discrete channel without speaker assumptions.
    Discrete(u16),
}
/// Ordered channel positions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChannelLayout {
    positions: Vec<ChannelPosition>,
}
impl ChannelLayout {
    /// Validate a nonempty unique set of positions.
    pub fn new(positions: Vec<ChannelPosition>) -> Result<Self> {
        if positions.is_empty() || positions.len() > 64 {
            return Err(Error::invalid("channel layout needs 1..=64 channels"));
        }
        for (i, p) in positions.iter().enumerate() {
            if positions[..i].contains(p) {
                return Err(Error::invalid("duplicate channel position"));
            }
        }
        Ok(Self { positions })
    }
    /// Named mono layout.
    pub fn mono() -> Self {
        Self {
            positions: vec![ChannelPosition::Mono],
        }
    }
    /// Front-left/front-right stereo.
    pub fn stereo() -> Self {
        Self {
            positions: vec![ChannelPosition::FrontLeft, ChannelPosition::FrontRight],
        }
    }
    /// Channel count.
    pub fn channels(&self) -> usize {
        self.positions.len()
    }
    /// Ordered speaker positions.
    pub fn positions(&self) -> &[ChannelPosition] {
        &self.positions
    }
}
/// Interleaved f32 host audio anchored to an explicit sample grid.
#[derive(Debug, Clone)]
pub struct AudioBlock {
    samples: Arc<[f32]>,
    sample_rate: u32,
    layout: ChannelLayout,
    start: Time,
    source_start: Option<Time>,
    retained: Vec<Arc<dyn AudioGuard>>,
}
trait AudioGuard: std::fmt::Debug + Send + Sync {}
impl<T: std::fmt::Debug + Send + Sync> AudioGuard for T {}
impl AudioBlock {
    /// Validate rate and complete interleaved frames.
    pub fn new(
        samples: impl Into<Arc<[f32]>>,
        sample_rate: u32,
        layout: ChannelLayout,
        start: Time,
    ) -> Result<Self> {
        let samples = samples.into();
        if sample_rate == 0 || samples.len() % layout.channels() != 0 {
            return Err(Error::invalid(
                "audio needs nonzero rate and complete interleaved frames",
            ));
        }
        Ok(Self {
            samples,
            sample_rate,
            layout,
            start,
            source_start: None,
            retained: Vec::new(),
        })
    }
    /// Retain a budget or consumer lifetime guard with every clone of this block.
    pub fn retain<T: std::fmt::Debug + Send + Sync + 'static>(mut self, guard: T) -> Self {
        self.retained.push(Arc::new(guard));
        self
    }
    /// Interleaved scalar samples.
    pub fn samples(&self) -> &[f32] {
        &self.samples
    }
    /// Rate in Hz.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
    /// Channel positions.
    pub fn channel_layout(&self) -> &ChannelLayout {
        &self.layout
    }
    /// Sample frames per channel.
    pub fn frames(&self) -> usize {
        self.samples.len() / self.layout.channels()
    }
    /// First-sample time.
    pub fn start(&self) -> Time {
        self.start
    }
    /// Original source timestamp of the first sample, when preserved by the adapter.
    /// This can differ from the continuous sample grid after timestamp quantization.
    pub fn source_start(&self) -> Option<Time> {
        self.source_start
    }
    /// Attach original source timing without changing the output sample grid.
    pub fn with_source_start(mut self, time: Option<Time>) -> Self {
        self.source_start = time;
        self
    }
    /// Exclusive end on the same sample grid.
    pub fn end(&self) -> Result<Time> {
        self.start.checked_add(Time::from_samples(
            i64::try_from(self.frames())
                .map_err(|_| Error::overflow("audio sample count exceeds media time"))?,
            self.sample_rate,
        )?)
    }
}
