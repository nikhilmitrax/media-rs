//! Explicit host tensor materialization with shared allocation accounting.
#[cfg(feature = "native")]
use crate::image_request::ImageRequest;
use crate::request::Session;
use media_core::{
    error::{Error, Result},
    media::Image,
};
use media_interop::tensor::{Tensor, TensorSpec, images_to_tensor};

/// A lazy spatial request followed by explicit model-input packing.
#[derive(Debug, Clone)]
#[cfg(feature = "native")]
pub struct ImageTensorRequest {
    request: ImageRequest,
    spec: TensorSpec,
}
#[cfg(feature = "native")]
impl ImageTensorRequest {
    pub(crate) fn new(request: ImageRequest, spec: TensorSpec) -> Self {
        Self { request, spec }
    }
    /// Execute transforms, then pack/normalize a compatible f32 RGB tensor.
    pub fn read(&self) -> Result<Tensor> {
        let result = self.request.read_with_report()?;
        pack_images(&[result.image], self.request.session(), &self.spec)
    }
}
/// Explicitly prepare host model input within a session's shared output budget.
pub fn pack_images(images: &[Image], session: &Session, spec: &TensorSpec) -> Result<Tensor> {
    let _cpu = session.resources.reserve_cpu(0, 0)?;
    let batch = session.resources.reserve_batch()?;
    let first = images
        .first()
        .ok_or_else(|| Error::invalid("tensor batch needs at least one image"))?;
    let bytes = images
        .len()
        .checked_mul(first.width() as usize)
        .and_then(|v| v.checked_mul(first.height() as usize))
        .and_then(|v| v.checked_mul(12))
        .and_then(|v| v.checked_add(252))
        .ok_or_else(|| Error::overflow("tensor batch allocation dimensions overflow"))?;
    if bytes > spec.allocation_limit() {
        return Err(Error::exhausted(
            "tensor output exceeds its requested allocation limit",
        ));
    }
    let guard = session.resources.host_budget().reserve(bytes)?;
    let tensor = images_to_tensor(images, &spec.clone().max_bytes(bytes))?;
    if tensor.allocation_bytes() > bytes {
        return Err(Error::exhausted(
            "tensor allocation exceeded its reserved budget",
        ));
    }
    Ok(tensor.retain(guard).retain(batch))
}
