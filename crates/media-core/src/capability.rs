//! Hard-policy matching for concrete supported routes.
use crate::{
    device::Device,
    error::{Error, Result},
    media::PixelFormat,
    policy::{DecodePolicy, DeviceWorkPolicy, Placement, TransferPolicy},
};
/// Decoder implementation category.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecoderKind {
    /// Software decoding.
    Software,
    /// Dedicated hardware decoding.
    Hardware,
}
/// One explicitly supported decode/transform/output combination.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapabilityRoute {
    /// Backend identity.
    pub backend: String,
    /// Decoder implementation.
    pub decoder: DecoderKind,
    /// Native decoded placement.
    pub decoded_device: Device,
    /// Final output placement.
    pub output_device: Device,
    /// Final representation.
    pub output_format: PixelFormat,
    /// Decoded bytes cross host/device memory.
    pub decoded_host_crossing: bool,
    /// Device-local allocation/repacking is required.
    pub device_materialization: bool,
}
/// Policies checked independently against each route.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RouteRequirements {
    /// Decoder selection.
    pub decode: DecodePolicy,
    /// Output placement.
    pub placement: Placement,
    /// Implicit transfer permission.
    pub transfers: TransferPolicy,
    /// Device-local materialization permission.
    pub device_work: DeviceWorkPolicy,
    /// Required representation.
    pub output_format: Option<PixelFormat>,
}
impl RouteRequirements {
    /// Validate one route, returning actionable incompatibility context.
    pub fn check(&self, r: &CapabilityRoute) -> Result<()> {
        let fail = |message: &str, constraint: &str| {
            Error::incompatible(message)
                .with_backend(&r.backend)
                .with_constraint(constraint)
        };
        match &self.decode {
            DecodePolicy::RequireHardware if r.decoder != DecoderKind::Hardware => {
                return Err(fail(
                    "required hardware decode is unavailable on this software route",
                    "decode",
                ));
            }
            DecodePolicy::RequireBackend(name) if name != &r.backend => {
                return Err(fail("route uses a different required backend", "decode"));
            }
            _ => {}
        }
        match self.placement {
            Placement::Host if !r.output_device.is_host() => {
                return Err(fail("CPU output was requested", "placement"));
            }
            Placement::Preserve if r.decoded_device != r.output_device => {
                return Err(fail("route changes native placement", "placement"));
            }
            Placement::Require(device) if device != r.output_device => {
                return Err(fail(
                    "route uses a different native device/context",
                    "placement",
                ));
            }
            _ => {}
        }
        if self.transfers != TransferPolicy::Allow
            && (r.decoded_host_crossing || r.decoded_device.is_host() != r.output_device.is_host())
        {
            return Err(fail(
                "decoded host transfer is forbidden but this route requires one",
                "decoded_transfers",
            ));
        }
        if self.device_work == DeviceWorkPolicy::ForbidMaterialization && r.device_materialization {
            return Err(fail(
                "device materialization is forbidden but required by this route",
                "device_work",
            ));
        }
        if self.output_format.is_some_and(|f| f != r.output_format) {
            return Err(fail(
                "route cannot produce the required format",
                "pixel_format",
            ));
        }
        Ok(())
    }
    /// Deterministic routing: hard constraints precede hardware preferences.
    pub fn select<'a>(&self, routes: &'a [CapabilityRoute]) -> Result<&'a CapabilityRoute> {
        let mut candidates: Vec<_> = routes.iter().filter(|r| self.check(r).is_ok()).collect();
        let rank = |r: &CapabilityRoute| {
            if self.decode == DecodePolicy::PreferHardware && r.decoder == DecoderKind::Hardware {
                0
            } else {
                1
            }
        };
        candidates.sort_by(|a, b| rank(a).cmp(&rank(b)).then(a.backend.cmp(&b.backend)));
        candidates.first().copied().ok_or_else(|| {
            let reasons = routes
                .iter()
                .filter_map(|r| self.check(r).err())
                .map(|e| e.to_string())
                .collect::<Vec<_>>()
                .join("; ");
            Error::unsupported(if reasons.is_empty() {
                "no backend routes registered".to_owned()
            } else {
                format!("no compatible route: {reasons}")
            })
            .with_operation("compile")
        })
    }
}
