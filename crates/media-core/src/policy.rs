//! Separate decoder, placement, transfer, fidelity, and indexing decisions.
use crate::device::Device;
/// Interpretation of backend duration hints, distinct from established display
/// intervals. Packet/codec durations can disagree with reordered presentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FrameDurationPolicy {
    /// Preserve the native hint separately; leave display duration unknown.
    #[default]
    Unknown,
    /// Explicitly interpret the backend hint as a display interval, accepting
    /// possible gaps/overlaps. Requests and reports disclose this assumption.
    TrustReported,
}
/// Decoder implementation policy, independent of output placement.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum DecodePolicy {
    /// Select a supported implementation.
    #[default]
    Automatic,
    /// Prefer hardware after hard constraints.
    PreferHardware,
    /// Fail without a hardware decoder.
    RequireHardware,
    /// Require this exact backend.
    RequireBackend(String),
}
/// Output placement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Placement {
    /// Predictable host output.
    #[default]
    Host,
    /// Keep native decoded placement.
    Preserve,
    /// Require the actual device/context.
    Require(Device),
}
/// Permission for implicit decoded host crossings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TransferPolicy {
    /// Only explicit transfer operations are allowed.
    #[default]
    ExplicitOnly,
    /// Forbid decoded pixels/samples crossing host memory.
    ForbidHost,
    /// Allow transfers with reporting.
    Allow,
}
/// Same-device materialization permission.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DeviceWorkPolicy {
    /// Allow conversion/repacking/detachment.
    #[default]
    Allow,
    /// Require an already compatible representation.
    ForbidMaterialization,
}
/// Numerical/color approximation permission.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Fidelity {
    /// Preserve declared operation semantics.
    #[default]
    Strict,
    /// Allow a reported approximation.
    AllowApproximation,
}
/// Missing coverage behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Boundary {
    /// Fail with missing-data context.
    #[default]
    Error,
    /// Keep established coverage only.
    Truncate,
    /// Pad missing audio with zero samples.
    Silence,
    /// Explicitly repeat video frames.
    RepeatFrame,
    /// Request cyclic coverage; adapters without defined wrap semantics reject it.
    Wrap,
}
/// Index construction policy, separate from temporal exactness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IndexPolicy {
    /// Build only needed information.
    #[default]
    Lazy,
    /// Reuse a validated cache or build missing information.
    Cached,
    /// Authorize a complete scan.
    Eager,
    /// Require a matching existing index.
    RequireExisting,
}
/// Checked view copy permission.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CopyPolicy {
    /// Share compatible storage or fail.
    #[default]
    Forbid,
    /// Explicitly permit a reported copy.
    Allow,
}
