//! Build diagnostics and explicit hardware capability boundaries.
use media_core::error::{Error, Result};

/// Whether this build links native FFmpeg libraries.
pub const fn is_linked() -> bool {
    cfg!(feature = "linked")
}

/// Runtime-linked library version, or an actionable build-capability error.
pub fn version() -> Result<String> {
    #[cfg(feature = "linked")]
    {
        crate::format::initialize()?;
        Ok(format!("libavformat {}", ffmpeg_next::format::version()))
    }
    #[cfg(not(feature = "linked"))]
    {
        Err(Error::unsupported(
            "FFmpeg backend is not linked; enable media-backend-ffmpeg/linked and install FFmpeg development libraries",
        ))
    }
}

/// Hardware device types compiled into the linked FFmpeg library.
///
/// This lists compiled support, not proof that a usable device or codec is
/// available. Actual hardware decode requires separate device/codec negotiation.
pub fn compiled_device_types() -> Result<Vec<String>> {
    #[cfg(feature = "linked")]
    {
        crate::format::initialize()?;
        Ok(crate::native::compiled_device_types())
    }
    #[cfg(not(feature = "linked"))]
    {
        Err(Error::unsupported(
            "FFmpeg hardware discovery requires the linked feature",
        ))
    }
}

/// Explain why hardware support cannot be established without a source/context.
///
/// No software decode followed by upload is substituted for hardware decoding.
pub fn require_hardware_decode() -> Result<()> {
    Err(Error::unsupported(
        "hardware decode requires opening cuda::CudaVideoReader for the actual source and native context with the cuda feature; software decoding or a host upload cannot satisfy hardware-required policy",
    ))
}
