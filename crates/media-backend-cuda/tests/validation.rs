//! GPU-independent validation of CUDA kernel dimensions and resource contracts.

use media_backend_cuda::ResizeDomain;
use media_backend_cuda::{
    ChromaInterpolation, CropRect, CudaContext, CudaLimits, CudaYuvImage, Interpolation,
    Normalization, PreprocessSpec, TensorLayout,
};
use media_core::error::ErrorKind;
use media_core::media::{ImageLayout, PixelFormat, PlaneLayout};

fn nv12() -> ImageLayout {
    ImageLayout {
        width: 4,
        height: 4,
        allocated_width: 4,
        allocated_height: 4,
        format: PixelFormat::Nv12,
        planes: vec![
            PlaneLayout {
                offset: 0,
                row_stride: 8,
                row_bytes: 4,
                rows: 4,
            },
            PlaneLayout {
                offset: 32,
                row_stride: 8,
                row_bytes: 4,
                rows: 2,
            },
        ],
    }
}

#[test]
fn yuv_validation_checks_padding_pitch_overlap_and_allocation_extent() {
    let layout = nv12();
    CudaYuvImage::validate_layout(&layout, 44).unwrap();
    assert!(CudaYuvImage::validate_layout(&layout, 43).is_err());
    let mut overlap = layout.clone();
    overlap.planes[1].offset = 24;
    assert!(CudaYuvImage::validate_layout(&overlap, 44).is_err());
    let mut pitch = layout.clone();
    pitch.planes[0].row_stride = 3;
    assert!(CudaYuvImage::validate_layout(&pitch, 44).is_err());
    let mut missing = layout.clone();
    missing.planes.pop();
    assert!(CudaYuvImage::validate_layout(&missing, 44).is_err());
    let mut rgb = layout.clone();
    rgb.format = PixelFormat::Rgb8;
    assert_eq!(
        CudaYuvImage::validate_layout(&rgb, 44).unwrap_err().kind(),
        ErrorKind::Unsupported
    );
}

#[test]
fn p010_requires_high_bit_storage_alignment_and_complete_uv_pairs() {
    let mut layout = nv12();
    layout.format = PixelFormat::P010;
    layout.planes[0].row_bytes = 8;
    layout.planes[1].row_bytes = 8;
    CudaYuvImage::validate_layout(&layout, 48).unwrap();
    layout.planes[1].row_stride = 9;
    assert!(CudaYuvImage::validate_layout(&layout, 49).is_err());
    layout.planes[1].row_stride = 8;
    layout.planes[1].offset = 33;
    assert!(CudaYuvImage::validate_layout(&layout, 49).is_err());
    let mut odd = nv12();
    odd.width = 3;
    odd.allocated_width = 3;
    odd.planes[1].row_bytes = 3;
    assert!(CudaYuvImage::validate_layout(&odd, 44).is_err());
}

#[test]
fn tensor_shape_crop_and_float_range_are_explicitly_validated() {
    let layout = nv12();
    let mut spec = PreprocessSpec::new(2, 3).unwrap();
    assert_eq!(spec.chroma_interpolation, ChromaInterpolation::MatchSpatial);
    assert_eq!(spec.interpolation, Interpolation::Bilinear);
    spec.chroma_interpolation = ChromaInterpolation::NearestTiesEarlier;
    spec.validate(&layout).unwrap();
    assert_eq!(
        spec.interpolation,
        Interpolation::Bilinear,
        "UV reconstruction never changes spatial policy"
    );
    assert_eq!(spec.resize_domain, ResizeDomain::ConvertedRgb);
    assert_eq!(spec.shape(), [1, 3, 3, 2]);
    assert_eq!(spec.output_elements().unwrap(), 18);
    spec.layout = TensorLayout::Nhwc;
    assert_eq!(spec.shape(), [1, 3, 2, 3]);
    spec.crop = Some(CropRect {
        x: 1,
        y: 1,
        width: 3,
        height: 3,
    });
    spec.validate(&layout).unwrap();
    spec.crop = Some(CropRect {
        x: u32::MAX,
        y: 0,
        width: 2,
        height: 2,
    });
    assert!(spec.validate(&layout).is_err());
    spec.crop = Some(CropRect {
        x: 1,
        y: 1,
        width: 0,
        height: 2,
    });
    assert!(spec.validate(&layout).is_err());
    assert!(PreprocessSpec::new(0, 1).is_err());
    assert!(PreprocessSpec::new(65_536, 65_536).is_err());
    assert!(PreprocessSpec::new(16_777_216, 1).is_err());
    for norm in [
        Normalization {
            mean: [f32::NAN, 0.0, 0.0],
            std: [1.0; 3],
        },
        Normalization {
            mean: [0.0; 3],
            std: [0.0, 1.0, 1.0],
        },
        Normalization {
            mean: [0.0; 3],
            std: [f32::INFINITY; 3],
        },
        Normalization {
            mean: [0.0; 3],
            std: [f32::MIN_POSITIVE / 16.0; 3],
        },
    ] {
        assert!(norm.validate().is_err());
    }
}

#[test]
fn invalid_limits_and_ordinals_fail_before_driver_discovery() {
    assert!(
        CudaLimits {
            memory_bytes: 0,
            streams: 1
        }
        .validate()
        .is_err()
    );
    assert!(
        CudaLimits {
            memory_bytes: 1,
            streams: 0
        }
        .validate()
        .is_err()
    );
    assert_eq!(
        CudaContext::new(u32::MAX, CudaLimits::default())
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidInput
    );
}

#[test]
fn absent_driver_returns_an_actionable_error_instead_of_panicking() {
    if !CudaContext::driver_available() {
        let error = CudaContext::new(0, CudaLimits::default()).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert!(error.message().contains("driver library is unavailable"));
    }
}

#[test]
fn final_batch_shape_and_allocation_size_are_checked_without_a_gpu() {
    let mut spec = PreprocessSpec::new(2, 4).unwrap();
    assert_eq!(spec.batch_elements(3).unwrap(), 72);
    assert_eq!(spec.batch_shape(3).unwrap(), [3, 3, 4, 2]);
    spec.layout = TensorLayout::Nhwc;
    assert_eq!(spec.batch_shape(3).unwrap(), [3, 4, 2, 3]);
    assert!(spec.batch_elements(0).is_err());
    assert!(spec.batch_elements(usize::MAX).is_err());
}
