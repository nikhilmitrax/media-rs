//! Ownership, representation validation, and hard policy contracts.
use media_core::{
    capability::{CapabilityRoute, DecoderKind, RouteRequirements},
    device::{ContextId, Device},
    error::{ErrorKind, Result},
    media::{
        AudioBlock, ChannelLayout, ChannelPosition, Image, ImageLayout, ImageMetadata, PixelFormat,
        PlaneLayout,
    },
    policy::{DecodePolicy, Placement, TransferPolicy},
    storage::{Completion, MemoryKind, StorageLease, StorageOwner},
    time::Time,
};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
#[test]
fn image_layout_rejects_short_overlapping_and_padded_packed_views() {
    assert!(Image::from_packed(2, 2, PixelFormat::Rgb8, vec![0; 11]).is_err());
    let layout = ImageLayout {
        width: 2,
        height: 2,
        allocated_width: 2,
        allocated_height: 2,
        format: PixelFormat::Rgb8,
        planes: vec![PlaneLayout {
            offset: 0,
            row_stride: 8,
            row_bytes: 6,
            rows: 2,
        }],
    };
    let image = Image::new(
        layout,
        StorageLease::from_bytes(vec![0; 14]),
        ImageMetadata::default(),
    )
    .unwrap();
    assert_eq!(image.host_bytes().unwrap().len(), 14);
    assert_eq!(
        image.packed_bytes().unwrap_err().kind(),
        ErrorKind::Incompatible
    );
    let invalid = ImageLayout {
        width: 2,
        height: 2,
        allocated_width: 2,
        allocated_height: 2,
        format: PixelFormat::Nv12,
        planes: vec![
            PlaneLayout {
                offset: 0,
                row_stride: 2,
                row_bytes: 2,
                rows: 2,
            },
            PlaneLayout {
                offset: 3,
                row_stride: 2,
                row_bytes: 2,
                rows: 1,
            },
        ],
    };
    assert!(
        Image::new(
            invalid,
            StorageLease::from_bytes(vec![0; 6]),
            ImageMetadata::default()
        )
        .is_err()
    );
}
#[test]
fn odd_multiplanar_geometry_and_checked_extent_are_validated() {
    let layout = ImageLayout {
        width: 3,
        height: 3,
        allocated_width: 3,
        allocated_height: 3,
        format: PixelFormat::Nv12,
        planes: vec![
            PlaneLayout {
                offset: 0,
                row_stride: 3,
                row_bytes: 3,
                rows: 3,
            },
            PlaneLayout {
                offset: 9,
                row_stride: 4,
                row_bytes: 4,
                rows: 2,
            },
        ],
    };
    assert!(
        Image::new(
            layout.clone(),
            StorageLease::from_bytes(vec![0; 17]),
            ImageMetadata::default()
        )
        .is_ok()
    );
    assert!(
        Image::new(
            layout,
            StorageLease::from_bytes(vec![0; 16]),
            ImageMetadata::default()
        )
        .is_err()
    );
    assert_eq!(
        PlaneLayout {
            offset: usize::MAX,
            row_stride: 2,
            row_bytes: 2,
            rows: 1
        }
        .extent()
        .unwrap_err()
        .kind(),
        ErrorKind::Overflow
    );
}
#[test]
fn aliases_prevent_mutation_and_explicit_copy_detaches() {
    let mut lease = StorageLease::from_bytes(vec![1, 2, 3]);
    let alias = lease.clone();
    assert!(lease.try_host_bytes_mut().is_err());
    let mut detached = lease.deep_copy_host().unwrap();
    detached.try_host_bytes_mut().unwrap()[0] = 9;
    assert_eq!(alias.host_bytes().unwrap(), [1, 2, 3]);
    drop(alias);
    lease.try_host_bytes_mut().unwrap()[0] = 4;
    assert_eq!(lease.host_bytes().unwrap(), [4, 2, 3]);
}
#[derive(Debug)]
struct WaitCounter(Arc<AtomicUsize>);
impl Completion for WaitCounter {
    fn is_complete(&self) -> Result<bool> {
        Ok(false)
    }
    fn wait(&self) -> Result<()> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}
#[derive(Debug)]
struct NativeReadOnly(Vec<u8>);
impl StorageOwner for NativeReadOnly {
    fn byte_len(&self) -> usize {
        self.0.len()
    }
    fn kind(&self) -> MemoryKind {
        MemoryKind::PinnedHost
    }
    fn device(&self) -> Device {
        Device::Cpu
    }
    fn host_bytes(&self) -> Option<&[u8]> {
        Some(&self.0)
    }
}
#[test]
fn host_view_establishes_producer_readiness_and_backend_exclusivity() {
    let waits = Arc::new(AtomicUsize::new(0));
    let mut lease = StorageLease::from_owner(
        Arc::new(NativeReadOnly(vec![1])),
        Arc::new(WaitCounter(waits.clone())),
    );
    assert_eq!(lease.host_bytes().unwrap(), [1]);
    assert_eq!(waits.load(Ordering::SeqCst), 1);
    assert!(
        lease.try_host_bytes_mut().is_err(),
        "backend can retain reference pictures despite unique application owner"
    );
}
#[derive(Debug)]
struct DropCounter(Arc<AtomicUsize>);
impl Drop for DropCounter {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}
#[test]
fn external_guards_survive_cloning_and_reader_wrapper_drop() {
    let dropped = Arc::new(AtomicUsize::new(0));
    let image = Image::from_packed(1, 1, PixelFormat::Rgb8, vec![1, 2, 3])
        .unwrap()
        .retain(DropCounter(dropped.clone()));
    let retained = image.clone();
    drop(image);
    assert_eq!(retained.packed_bytes().unwrap(), [1, 2, 3]);
    assert_eq!(dropped.load(Ordering::SeqCst), 0);
    drop(retained);
    assert_eq!(dropped.load(Ordering::SeqCst), 1);
}
fn route(
    backend: &str,
    decoder: DecoderKind,
    decoded_device: Device,
    output_device: Device,
) -> CapabilityRoute {
    CapabilityRoute {
        backend: backend.into(),
        decoder,
        decoded_device,
        output_device,
        output_format: PixelFormat::Rgb8,
        decoded_host_crossing: false,
        device_materialization: false,
    }
}
#[test]
fn gpu_output_does_not_satisfy_hardware_decode_or_transfer_policy() {
    let gpu = Device::Cuda {
        ordinal: 0,
        context: ContextId::new(1).unwrap(),
    };
    let software = route("upload", DecoderKind::Software, Device::Cpu, gpu);
    let mut required = RouteRequirements {
        placement: Placement::Require(gpu),
        decode: DecodePolicy::RequireHardware,
        transfers: TransferPolicy::Allow,
        ..Default::default()
    };
    assert_eq!(
        required
            .check(&software)
            .unwrap_err()
            .context()
            .constraint
            .as_deref(),
        Some("decode")
    );
    required.decode = DecodePolicy::Automatic;
    required.transfers = TransferPolicy::ForbidHost;
    assert_eq!(
        required
            .check(&software)
            .unwrap_err()
            .context()
            .constraint
            .as_deref(),
        Some("decoded_transfers")
    );
    required.transfers = TransferPolicy::Allow;
    assert!(required.check(&software).is_ok());
}
#[test]
fn native_context_and_hard_constraints_dominate_preferences() {
    let gpu = Device::Cuda {
        ordinal: 0,
        context: ContextId::new(1).unwrap(),
    };
    let other = Device::Cuda {
        ordinal: 0,
        context: ContextId::new(2).unwrap(),
    };
    let routes = [
        route("hw", DecoderKind::Hardware, gpu, gpu),
        route("cpu", DecoderKind::Software, Device::Cpu, Device::Cpu),
    ];
    let req = RouteRequirements {
        decode: DecodePolicy::PreferHardware,
        ..Default::default()
    };
    assert_eq!(req.select(&routes).unwrap().backend, "cpu");
    assert!(
        RouteRequirements {
            placement: Placement::Require(other),
            ..req
        }
        .select(&routes)
        .is_err()
    );
}
#[test]
fn audio_requires_named_distinct_layout_and_anchored_complete_frames() {
    assert!(
        ChannelLayout::new(vec![ChannelPosition::FrontLeft, ChannelPosition::FrontLeft]).is_err()
    );
    assert!(AudioBlock::new(vec![0.0; 3], 48000, ChannelLayout::stereo(), Time::ZERO).is_err());
    let block = AudioBlock::new(
        vec![0.0; 960],
        48000,
        ChannelLayout::stereo(),
        Time::from_millis(10),
    )
    .unwrap();
    assert_eq!(block.end().unwrap(), Time::from_millis(20));
}
