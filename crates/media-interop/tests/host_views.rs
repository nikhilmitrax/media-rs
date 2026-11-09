//! Checked host representation views and physically packed tensor values.

use media_core::{
    device::{ContextId, Device},
    media::{
        ColorDescription, Image, ImageLayout, ImageMetadata, Orientation, PixelFormat, PlaneLayout,
    },
    storage::{MemoryKind, Ready, StorageLease, StorageOwner},
};
use media_interop::{
    image::{HostImageView, try_rgb8_view, try_rgba8_view},
    tensor::{
        AlphaPolicy, ColorAssumption, NumericRange, Tensor, TensorLayout, TensorSpec, to_tensor,
    },
};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

fn srgb(image: Image) -> Image {
    image.with_metadata(ImageMetadata {
        color: ColorDescription {
            primaries: Some("bt709".into()),
            transfer: Some("srgb".into()),
            matrix: Some("identity".into()),
            range: Some("full".into()),
            chroma_siting: None,
        },
        ..Default::default()
    })
}
fn padded() -> Image {
    srgb(
        Image::new(
            ImageLayout {
                width: 2,
                height: 2,
                allocated_width: 2,
                allocated_height: 2,
                format: PixelFormat::Rgb8,
                planes: vec![PlaneLayout {
                    offset: 2,
                    row_stride: 8,
                    row_bytes: 6,
                    rows: 2,
                }],
            },
            StorageLease::from_bytes(vec![
                99, 99, 255, 0, 0, 0, 255, 0, 88, 88, 0, 0, 255, 255, 255, 255,
            ]),
            ImageMetadata::default(),
        )
        .unwrap(),
    )
}

#[test]
fn packed_image_views_share_storage_and_reject_reinterpretation() {
    let image = Image::from_packed(2, 1, PixelFormat::Rgb8, vec![1, 2, 3, 4, 5, 6]).unwrap();
    let view = try_rgb8_view(&image).unwrap();
    assert_eq!(view.as_raw().as_ptr(), image.host_bytes().unwrap().as_ptr());
    assert_eq!(view.get_pixel(1, 0).0, [4, 5, 6]);
    assert!(try_rgba8_view(&image).is_err());
    assert!(try_rgb8_view(&padded()).is_err());
    let view = HostImageView::new(&padded()).unwrap();
    assert_eq!(view.row(0).unwrap(), &[255, 0, 0, 0, 255, 0]);
    assert_eq!(view.row(1).unwrap(), &[0, 0, 255, 255, 255, 255]);
    assert!(view.row(2).is_err());
}

#[derive(Debug)]
struct TrackedBytes {
    bytes: Vec<u8>,
    dropped: Arc<AtomicUsize>,
    device: Device,
}
impl Drop for TrackedBytes {
    fn drop(&mut self) {
        self.dropped.fetch_add(1, Ordering::Relaxed);
    }
}
impl StorageOwner for TrackedBytes {
    fn byte_len(&self) -> usize {
        self.bytes.len()
    }
    fn kind(&self) -> MemoryKind {
        if self.device == Device::Cpu {
            MemoryKind::Host
        } else {
            MemoryKind::DeviceLinear
        }
    }
    fn device(&self) -> Device {
        self.device
    }
    fn host_bytes(&self) -> Option<&[u8]> {
        Some(&self.bytes)
    }
}

#[test]
fn owning_host_view_retains_storage_and_rejects_device_placement() {
    let dropped = Arc::new(AtomicUsize::new(0));
    let storage = StorageLease::from_owner(
        Arc::new(TrackedBytes {
            bytes: vec![1, 2, 3],
            dropped: dropped.clone(),
            device: Device::Cpu,
        }),
        Arc::new(Ready),
    );
    let image = Image::new(
        ImageLayout::packed(1, 1, PixelFormat::Rgb8).unwrap(),
        storage,
        ImageMetadata::default(),
    )
    .unwrap();
    let view = HostImageView::new(&image).unwrap();
    drop(image);
    assert_eq!(dropped.load(Ordering::Relaxed), 0);
    assert_eq!(view.row(0).unwrap(), &[1, 2, 3]);
    drop(view);
    assert_eq!(dropped.load(Ordering::Relaxed), 1);
    let storage = StorageLease::from_owner(
        Arc::new(TrackedBytes {
            bytes: vec![1, 2, 3],
            dropped,
            device: Device::Cuda {
                ordinal: 0,
                context: ContextId::new(1).unwrap(),
            },
        }),
        Arc::new(Ready),
    );
    let image = Image::new(
        ImageLayout::packed(1, 1, PixelFormat::Rgb8).unwrap(),
        storage,
        ImageMetadata::default(),
    )
    .unwrap();
    assert!(HostImageView::new(&image).is_err());
    assert!(try_rgb8_view(&image).is_err());
}

#[test]
fn pitched_pixels_pack_to_actual_nchw_or_nhwc_memory() {
    let image = padded();
    let tensor = to_tensor(&image, &TensorSpec::rgb_f32()).unwrap();
    assert_eq!(tensor.shape(), &[1, 3, 2, 2]);
    assert_eq!(tensor.strides(), &[12, 4, 2, 1]);
    assert_eq!(
        tensor.data(),
        &[1., 0., 0., 1., 0., 1., 0., 1., 0., 0., 1., 1.]
    );
    assert_eq!(tensor.data().as_ptr().align_offset(256), 0);
    let nhwc = to_tensor(&image, &TensorSpec::rgb_f32().layout(TensorLayout::Nhwc)).unwrap();
    assert_eq!(nhwc.shape(), &[1, 2, 2, 3]);
    assert_eq!(nhwc.strides(), &[12, 6, 3, 1]);
    assert_eq!(
        nhwc.data(),
        &[1., 0., 0., 0., 1., 0., 0., 0., 1., 1., 1., 1.]
    );
    assert_eq!(tensor.layout(), Some(TensorLayout::Nchw));
    assert!(tensor.assumptions().is_empty());
}

#[test]
fn normalization_numeric_range_and_alpha_are_explicit() {
    let image = padded();
    let tensor = to_tensor(
        &image,
        &TensorSpec::rgb_f32().normalize([0.5, 0.25, 0.0], [0.5, 0.25, 1.0]),
    )
    .unwrap();
    assert_eq!(
        tensor.data(),
        &[1., -1., -1., 1., -1., 3., -1., 3., 0., 0., 1., 1.]
    );
    let image = srgb(
        Image::from_packed(1, 1, PixelFormat::Gray16, 65535_u16.to_le_bytes().to_vec()).unwrap(),
    );
    assert_eq!(
        to_tensor(&image, &TensorSpec::rgb_f32().range(NumericRange::Byte))
            .unwrap()
            .data(),
        &[255.; 3]
    );
    let image = srgb(Image::from_packed(1, 1, PixelFormat::Rgba8, vec![255, 128, 0, 0]).unwrap());
    assert!(to_tensor(&image, &TensorSpec::rgb_f32()).is_err());
    let tensor = to_tensor(&image, &TensorSpec::rgb_f32().alpha(AlphaPolicy::Discard)).unwrap();
    assert_eq!(tensor.data(), &[1., 128.0 / 255.0, 0.]);
    assert!(to_tensor(&image, &TensorSpec::rgb_f32().normalize([0.; 3], [0.; 3])).is_err());
}

#[test]
fn unknown_and_contradictory_color_and_orientation_are_not_silently_accepted() {
    let image = Image::from_packed(1, 1, PixelFormat::Rgb8, vec![255, 0, 0]).unwrap();
    assert!(to_tensor(&image, &TensorSpec::rgb_f32()).is_err());
    let assume = TensorSpec::rgb_f32().color_assumption(ColorAssumption::AssumeSrgb);
    let tensor = to_tensor(&image, &assume).unwrap();
    assert_eq!(tensor.assumptions().len(), 4);
    let mut metadata = image.metadata().clone();
    metadata.color.transfer = Some("pq".into());
    assert!(to_tensor(&image.clone().with_metadata(metadata), &assume).is_err());
    let mut metadata = image.metadata().clone();
    metadata.icc_profile = Some(vec![1, 2, 3]);
    assert!(to_tensor(&image.clone().with_metadata(metadata), &assume).is_err());
    let mut metadata = image.metadata().clone();
    metadata.orientation = Orientation::Rotate90;
    assert!(to_tensor(&image.with_metadata(metadata), &assume).is_err());
}

#[test]
fn aligned_numeric_tensor_construction_checks_shapes_and_sharing() {
    let tensor = Tensor::from_vec(vec![1., 2., 3., 4.], vec![2, 2]).unwrap();
    assert_eq!(tensor.data(), &[1., 2., 3., 4.]);
    assert_eq!(tensor.data().as_ptr().align_offset(256), 0);
    let clone = tensor.clone();
    assert!(clone.shares_storage(&tensor));
    let detached = tensor.deep_copy().unwrap();
    assert!(!detached.shares_storage(&tensor));
    assert_eq!(detached.data(), tensor.data());
    assert!(Tensor::from_vec(vec![1.], vec![2]).is_err());
    assert!(Tensor::from_vec(vec![], vec![usize::MAX, 2]).is_err());
}

#[test]
fn batch_packing_materializes_matching_images_in_caller_order() {
    use media_interop::tensor::images_to_tensor;
    let red = srgb(Image::from_packed(1, 1, PixelFormat::Rgb8, vec![255, 0, 0]).unwrap());
    let blue = srgb(Image::from_packed(1, 1, PixelFormat::Rgb8, vec![0, 0, 255]).unwrap());
    let batch = images_to_tensor(&[red.clone(), blue], &TensorSpec::rgb_f32()).unwrap();
    assert_eq!(batch.shape(), &[2, 3, 1, 1]);
    assert_eq!(batch.strides(), &[3, 1, 1, 1]);
    assert_eq!(batch.data(), &[1., 0., 0., 0., 0., 1.]);
    let wide = srgb(Image::from_packed(2, 1, PixelFormat::Rgb8, vec![0; 6]).unwrap());
    assert!(images_to_tensor(&[red.clone(), wide], &TensorSpec::rgb_f32()).is_err());
    assert!(images_to_tensor(&[], &TensorSpec::rgb_f32()).is_err());
    assert!(to_tensor(&red, &TensorSpec::rgb_f32().max_bytes(4)).is_err());
    assert!(batch.allocation_bytes() >= batch.data().len() * 4);
    let unknown = Image::from_packed(1, 1, PixelFormat::Rgb8, vec![255, 0, 0]).unwrap();
    let assumed = images_to_tensor(
        &[unknown.clone(), unknown],
        &TensorSpec::rgb_f32().color_assumption(ColorAssumption::AssumeSrgb),
    )
    .unwrap();
    assert_eq!(assumed.assumptions().len(), 4);
    assert!(
        assumed
            .assumptions()
            .iter()
            .all(|value| value.ends_with("2/2 images"))
    );
}

#[test]
fn f32_input_preserves_its_declared_scale_and_rejects_nonfinite_values() {
    let values = [0.25_f32, 0.5, 2.0];
    let bytes = values.into_iter().flat_map(f32::to_le_bytes).collect();
    let image = srgb(Image::from_packed(1, 1, PixelFormat::RgbF32, bytes).unwrap());
    assert_eq!(
        to_tensor(&image, &TensorSpec::rgb_f32()).unwrap().data(),
        &values
    );
    let bytes = [f32::NAN, 0.0, 0.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    let invalid = srgb(Image::from_packed(1, 1, PixelFormat::RgbF32, bytes).unwrap());
    assert!(to_tensor(&invalid, &TensorSpec::rgb_f32()).is_err());
}

#[cfg(not(feature = "icc"))]
#[test]
fn icc_conversion_is_a_checked_optional_capability() {
    use media_interop::tensor::{IccGamut, IccPolicy};
    let image = Image::from_packed(1, 1, PixelFormat::Rgb8, vec![1, 2, 3])
        .unwrap()
        .with_metadata(ImageMetadata {
            icc_profile: Some(vec![1, 2, 3]),
            ..Default::default()
        });
    let error = to_tensor(
        &image,
        &TensorSpec::rgb_f32().icc_policy(IccPolicy::ConvertToSrgb {
            gamut: IccGamut::Reject,
        }),
    )
    .unwrap_err();
    assert!(error.message().contains("feature"));
}

#[cfg(feature = "ndarray")]
#[test]
fn ndarray_view_shares_typed_values_and_obeys_layout() {
    let tensor = to_tensor(&padded(), &TensorSpec::rgb_f32()).unwrap();
    let array = media_interop::ndarray::try_tensor_view(&tensor).unwrap();
    assert_eq!(array.as_ptr(), tensor.data().as_ptr());
    assert_eq!(array.shape(), &[1, 3, 2, 2]);
    assert_eq!(array[[0, 2, 1, 0]], 1.);
    assert_eq!(array[[0, 0, 1, 0]], 0.);
    let image = padded();
    let view = media_interop::ndarray::try_image_u8_view(&image).unwrap();
    assert_eq!(view.shape(), &[2, 2, 3]);
    assert_eq!(view.strides(), &[8, 3, 1]);
    assert_eq!(view.as_ptr(), image.host_bytes().unwrap()[2..].as_ptr());
    assert_eq!(view[[1, 0, 2]], 255);
}
