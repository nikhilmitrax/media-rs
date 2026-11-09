//! Structured descriptions and observed library-level execution costs.
use media_core::{device::Device, media::PixelFormat};
/// A resolved operation in a reusable plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanStep {
    /// Operation identity.
    pub operation: String,
    /// Backend identity.
    pub backend: String,
    /// Output memory/context.
    pub output: Device,
    /// Output representation, when known.
    pub format: Option<PixelFormat>,
    /// Whether this step requires new output storage.
    pub materializes: bool,
}
/// Observed counters, distinct from opaque native/driver costs.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExecutionReport {
    /// Selected backend.
    pub backend: String,
    /// Encoded input bytes read.
    pub bytes_read: u64,
    /// Decoded frames/blocks.
    pub decoded_count: u64,
    /// Returned frames/blocks.
    pub output_count: u64,
    /// Live library-accounted host allocation high-water mark.
    pub peak_host_bytes: usize,
    /// Actual hardware decoding, separate from final placement.
    pub hardware_decode: bool,
    /// Recorded decoded host transfer bytes.
    pub decoded_host_transfer_bytes: u64,
    /// Driver memory/copies remain unknown unless separately measured.
    pub opaque_driver_bytes: Option<u64>,
    /// Metadata/fidelity limitations exercised by this request.
    pub warnings: Vec<String>,
}
