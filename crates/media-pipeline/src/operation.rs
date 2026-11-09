//! Backend-independent spatial request descriptions.
/// Aspect-ratio behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fit {
    /// Preserve ratio within bounds.
    Inside {
        /// Bounding width.
        width: u32,
        /// Bounding height.
        height: u32,
    },
    /// Fill bounds and center-crop.
    Cover {
        /// Output width.
        width: u32,
        /// Output height.
        height: u32,
    },
}
impl Fit {
    /// Preserve ratio inside bounds.
    pub const fn inside(width: u32, height: u32) -> Self {
        Self::Inside { width, height }
    }
    /// Fill bounds and center-crop.
    pub const fn cover(width: u32, height: u32) -> Self {
        Self::Cover { width, height }
    }
}
/// Filtering of stored component values; color-managed filtering is separate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Filter {
    /// Nearest-neighbor sample.
    Nearest,
    /// Bilinear interpolation.
    Triangle,
    /// Cubic Catmull-Rom.
    CatmullRom,
    /// Gaussian.
    Gaussian,
    /// Three-lobe Lanczos.
    #[default]
    Lanczos3,
}
/// Declarative spatial operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageOperation {
    /// Apply metadata orientation to pixels.
    ApplyOrientation,
    /// Resize with explicit geometry and stored-component filter.
    Resize {
        /// Geometry.
        fit: Fit,
        /// Filter.
        filter: Filter,
    },
    /// Exact in-bounds crop.
    Crop {
        /// Left coordinate.
        x: u32,
        /// Top coordinate.
        y: u32,
        /// Output width.
        width: u32,
        /// Output height.
        height: u32,
    },
}
