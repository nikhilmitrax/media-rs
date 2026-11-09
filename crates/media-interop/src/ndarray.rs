//! Optional checked ndarray views over existing aligned host tensor values.
//!
//! Enable the ndarray feature. Views borrow the Tensor, share its f32 storage,
//! and retain its physical shape/element strides without axis reinterpretation.

#[cfg(feature = "ndarray")]
use crate::tensor::Tensor;
#[cfg(feature = "ndarray")]
use media_core::error::{Error, Result};

/// Borrow an immutable ndarray view without copying numeric values.
#[cfg(feature = "ndarray")]
pub fn try_tensor_view(tensor: &Tensor) -> Result<::ndarray::ArrayViewD<'_, f32>> {
    use ::ndarray::{ArrayViewD, IxDyn, ShapeBuilder};
    ArrayViewD::from_shape(
        IxDyn(tensor.shape()).strides(IxDyn(tensor.strides())),
        tensor.data(),
    )
    .map_err(|error| Error::incompatible(format!("ndarray cannot view tensor layout: {error}")))
}

/// Borrow raw 8-bit packed pixels as height/width/channel ndarray values.
/// Pitched rows and plane offsets are preserved; color/orientation metadata stays
/// on the source Image. No repacking, orientation, or color conversion occurs.
#[cfg(feature = "ndarray")]
pub fn try_image_u8_view(
    image: &media_core::media::Image,
) -> Result<::ndarray::ArrayView3<'_, u8>> {
    use ::ndarray::{ArrayView3, ShapeBuilder};
    use media_core::media::PixelFormat;
    if !matches!(
        image.pixel_format(),
        PixelFormat::Gray8 | PixelFormat::GrayAlpha8 | PixelFormat::Rgb8 | PixelFormat::Rgba8
    ) {
        return Err(Error::incompatible(
            "u8 ndarray image view requires an existing 8-bit packed pixel format",
        ));
    }
    crate::image::HostImageView::new(image)?;
    let plane = &image.layout().planes[0];
    let channels = image.pixel_format().channels();
    let bytes = image
        .host_bytes()?
        .get(plane.offset..)
        .ok_or_else(|| Error::incompatible("image plane offset exceeds host storage"))?;
    ArrayView3::from_shape(
        (image.height() as usize, image.width() as usize, channels).strides((
            plane.row_stride,
            channels,
            1,
        )),
        bytes,
    )
    .map_err(|error| Error::incompatible(format!("ndarray cannot view image layout: {error}")))
}
