//! Real GPU checks. Explicit execution fails if GPU/driver/NVRTC is unavailable.
//!
//! Run: cargo test -p media-backend-cuda --test hardware -- --ignored --test-threads=1

use media_backend_cuda::ResizeDomain;
use media_backend_cuda::{
    ChromaInterpolation, ChromaSiting, CropRect, CudaContext, CudaLimits, CudaYuvImage,
    Interpolation, Normalization, PreprocessSpec, TensorLayout, YuvMatrix, YuvRange,
};
use media_core::error::ErrorKind;
use media_core::media::{ImageLayout, PixelFormat, PlaneLayout};
use media_core::storage::Completion;

use cudarc::driver::{CudaSlice, DevicePtr};
use media_core::device::Device;
use media_core::storage::{MemoryKind, StorageLease, StorageOwner};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

fn context() -> CudaContext {
    CudaContext::new(
        0,
        CudaLimits {
            memory_bytes: 16 * 1024 * 1024,
            streams: 8,
        },
    )
    .expect("actual NVIDIA GPU with compatible CUDA driver is required")
}

fn layout(format: PixelFormat) -> ImageLayout {
    let bytes = if format == PixelFormat::P010 { 2 } else { 1 };
    ImageLayout {
        width: 2,
        height: 2,
        allocated_width: 2,
        allocated_height: 2,
        format,
        planes: vec![
            PlaneLayout {
                offset: 0,
                row_stride: 2 * bytes,
                row_bytes: 2 * bytes,
                rows: 2,
            },
            PlaneLayout {
                offset: 4 * bytes,
                row_stride: 2 * bytes,
                row_bytes: 2 * bytes,
                rows: 1,
            },
        ],
    }
}

fn close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "element {i}: actual {actual}, expected {expected}"
        );
    }
}

// These oracle inputs assign nine asymmetric UV block colors, and distinct Y
// values per pixel. P010 retains extra low code bits beyond 8-bit precision.
fn block_y(ten_bit: bool, x: usize, y: usize) -> f64 {
    let code = 100 + 3 * x + 7 * y;
    if ten_bit {
        (4 * code + (x + 2 * y) % 4) as f64
    } else {
        code as f64
    }
}
fn block_uv(ten_bit: bool, x: usize, y: usize) -> [f64; 2] {
    let u = 88 + 18 * x + 7 * y;
    let v = 160 - 13 * x + 9 * y;
    if ten_bit {
        [
            (4 * u + (x + 2 * y + 1) % 4) as f64,
            (4 * v + (2 * x + y + 2) % 4) as f64,
        ]
    } else {
        [u as f64, v as f64]
    }
}
fn block_rgb(y: f64, uv: [f64; 2], ten_bit: bool, range: YuvRange) -> [f64; 3] {
    let (yb, ys, cb, cs) = match (ten_bit, range) {
        (false, YuvRange::Limited) => (16., 219., 128., 224.),
        (false, YuvRange::Full) => (0., 255., 128., 255.),
        (true, YuvRange::Limited) => (64., 876., 512., 896.),
        (true, YuvRange::Full) => (0., 1023., 512., 1023.),
    };
    let y = (y - yb) / ys;
    let u = (uv[0] - cb) / cs;
    let v = (uv[1] - cb) / cs;
    // Independent double-precision BT.709 inverse equations.
    let (kr, kb) = (0.2126, 0.0722);
    let kg = 1. - kr - kb;
    [
        (y + 2. * (1. - kr) * v).clamp(0., 1.),
        (y - 2. * kb * (1. - kb) / kg * u - 2. * kr * (1. - kr) / kg * v).clamp(0., 1.),
        (y + 2. * (1. - kb) * u).clamp(0., 1.),
    ]
}
fn block_pixels(ten_bit: bool, range: YuvRange) -> Vec<[f64; 3]> {
    (0..6)
        .flat_map(|y| {
            (0..6).map(move |x| {
                block_rgb(
                    block_y(ten_bit, x, y),
                    block_uv(ten_bit, x / 2, y / 2),
                    ten_bit,
                    range,
                )
            })
        })
        .collect()
}
fn pack_oracle(
    pixels: &[[f64; 3]],
    layout: TensorLayout,
    normalization: Normalization,
) -> Vec<f32> {
    let normalized = |value: f64, channel: usize| {
        ((value - f64::from(normalization.mean[channel])) / f64::from(normalization.std[channel]))
            as f32
    };
    match layout {
        TensorLayout::Nhwc => pixels
            .iter()
            .flat_map(|pixel| (0..3).map(move |channel| normalized(pixel[channel], channel)))
            .collect(),
        TensorLayout::Nchw => (0..3)
            .flat_map(|channel| {
                pixels
                    .iter()
                    .map(move |pixel| normalized(pixel[channel], channel))
            })
            .collect(),
    }
}
fn block_image(
    stream: &media_backend_cuda::CudaStream,
    ten_bit: bool,
    range: YuvRange,
    siting: ChromaSiting,
) -> CudaYuvImage {
    let unit = if ten_bit { 2 } else { 1 };
    let pitch = 8 * unit;
    let y_bytes = 6 * pitch;
    let mut bytes = vec![0xd7u8; y_bytes + 3 * pitch];
    let mut put = |offset: usize, code: f64| {
        if ten_bit {
            bytes[offset..offset + 2].copy_from_slice(&((code as u16) << 6).to_le_bytes());
        } else {
            bytes[offset] = code as u8;
        }
    };
    for y in 0..6 {
        for x in 0..6 {
            put(y * pitch + x * unit, block_y(ten_bit, x, y));
        }
    }
    for y in 0..3 {
        for x in 0..3 {
            let uv = block_uv(ten_bit, x, y);
            put(y_bytes + y * pitch + 2 * x * unit, uv[0]);
            put(y_bytes + y * pitch + (2 * x + 1) * unit, uv[1]);
        }
    }
    let layout = ImageLayout {
        width: 6,
        height: 6,
        allocated_width: 6,
        allocated_height: 6,
        format: if ten_bit {
            PixelFormat::P010
        } else {
            PixelFormat::Nv12
        },
        planes: vec![
            PlaneLayout {
                offset: 0,
                row_stride: pitch,
                row_bytes: 6 * unit,
                rows: 6,
            },
            PlaneLayout {
                offset: y_bytes,
                row_stride: pitch,
                row_bytes: 6 * unit,
                rows: 3,
            },
        ],
    };
    CudaYuvImage::new(
        stream.upload(&bytes).unwrap(),
        layout,
        YuvMatrix::Bt709,
        range,
        siting,
    )
    .unwrap()
}

#[test]
#[ignore = "requires real CUDA/NVRTC; independently verifies pitched NV12/P010 chroma reconstruction"]
fn explicit_chroma_nearest_ties_earlier_matches_block_oracle_and_preserves_legacy_filters() {
    let context = context();
    let stream = context.stream().unwrap();
    for ten_bit in [false, true] {
        for range in [YuvRange::Limited, YuvRange::Full] {
            let expected = block_pixels(ten_bit, range);
            for siting in [
                ChromaSiting::Left,
                ChromaSiting::TopLeft,
                ChromaSiting::Center,
            ] {
                let image = block_image(&stream, ten_bit, range, siting);
                for layout in [TensorLayout::Nchw, TensorLayout::Nhwc] {
                    let mut spec = PreprocessSpec::new(6, 6).unwrap();
                    spec.layout = layout;
                    spec.chroma_interpolation = ChromaInterpolation::NearestTiesEarlier;
                    let actual = image
                        .preprocess(&stream, spec)
                        .unwrap()
                        .download(&stream)
                        .unwrap();
                    close(
                        &actual,
                        &pack_oracle(&expected, layout, spec.normalization),
                        0.000_003,
                    );
                }
            }
            // Pin the existing Left-sited filters independently at x=1,y=0:
            // Nearest's upward tie chooses UV(1,0); bilinear takes their mean.
            let image = block_image(&stream, ten_bit, range, ChromaSiting::Left);
            let mut spec = PreprocessSpec::new(6, 6).unwrap();
            spec.layout = TensorLayout::Nhwc;
            let bilinear = image
                .preprocess(&stream, spec)
                .unwrap()
                .download(&stream)
                .unwrap();
            let a = block_uv(ten_bit, 0, 0);
            let b = block_uv(ten_bit, 1, 0);
            let mean = [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5];
            close(
                &bilinear[3..6],
                &block_rgb(block_y(ten_bit, 1, 0), mean, ten_bit, range).map(|v| v as f32),
                0.000_003,
            );
            spec.interpolation = Interpolation::Nearest;
            let nearest = image
                .preprocess(&stream, spec)
                .unwrap()
                .download(&stream)
                .unwrap();
            close(
                &nearest[3..6],
                &block_rgb(block_y(ten_bit, 1, 0), b, ten_bit, range).map(|v| v as f32),
                0.000_003,
            );
            assert!(
                (nearest[5] - expected[1][2] as f32).abs() > 0.05,
                "fixture must distinguish old upward UV ties from explicit earlier ties"
            );
        }
    }
}

#[test]
#[ignore = "requires real CUDA/NVRTC; verifies absolute crop phase, RGB resize and separate nearest tie policies"]
fn chroma_compatibility_preserves_odd_crop_rgb_resize_and_spatial_nearest_ties() {
    let context = context();
    let stream = context.stream().unwrap();
    for ten_bit in [false, true] {
        let range = YuvRange::Limited;
        let image = block_image(&stream, ten_bit, range, ChromaSiting::Left);
        let source = block_pixels(ten_bit, range);
        for layout in [TensorLayout::Nchw, TensorLayout::Nhwc] {
            let mut spec = PreprocessSpec::new(3, 5).unwrap();
            spec.layout = layout;
            spec.crop = Some(CropRect {
                x: 1,
                y: 1,
                width: 4,
                height: 4,
            });
            spec.chroma_interpolation = ChromaInterpolation::NearestTiesEarlier;
            spec.normalization = Normalization {
                mean: [0.1, 0.2, 0.3],
                std: [0.5, 0.25, 0.75],
            };
            // First reconstruct the complete RGB image by integer block lookup,
            // then crop/resize that image using a separate scalar RGB oracle.
            let mut resized = Vec::new();
            for y in 0..5 {
                for x in 0..3 {
                    let sx = (1. + (x as f64 + 0.5) * 4. / 3. - 0.5).clamp(1., 4.);
                    let sy = (1. + (y as f64 + 0.5) * 4. / 5. - 0.5).clamp(1., 4.);
                    let x0 = sx.floor() as usize;
                    let y0 = sy.floor() as usize;
                    let x1 = (x0 + 1).min(4);
                    let y1 = (y0 + 1).min(4);
                    let fx = sx - x0 as f64;
                    let fy = sy - y0 as f64;
                    resized.push(std::array::from_fn(|channel| {
                        let top = source[y0 * 6 + x0][channel] * (1. - fx)
                            + source[y0 * 6 + x1][channel] * fx;
                        let bottom = source[y1 * 6 + x0][channel] * (1. - fx)
                            + source[y1 * 6 + x1][channel] * fx;
                        top * (1. - fy) + bottom * fy
                    }));
                }
            }
            close(
                &image
                    .preprocess(&stream, spec)
                    .unwrap()
                    .download(&stream)
                    .unwrap(),
                &pack_oracle(&resized, layout, spec.normalization),
                0.000_006,
            );
            // Luma/RGB downsample ties remain upward: output centers .5,2.5,4.5
            // choose source columns 1,3,5 even though UV ties choose earlier.
            let mut nearest = PreprocessSpec::new(3, 6).unwrap();
            nearest.layout = layout;
            nearest.interpolation = Interpolation::Nearest;
            nearest.chroma_interpolation = ChromaInterpolation::NearestTiesEarlier;
            let expected = (0..6)
                .flat_map(|y| [1, 3, 5].map(|x| source[y * 6 + x]))
                .collect::<Vec<_>>();
            close(
                &image
                    .preprocess(&stream, nearest)
                    .unwrap()
                    .download(&stream)
                    .unwrap(),
                &pack_oracle(&expected, layout, nearest.normalization),
                0.000_003,
            );
        }
        // Explicit plane-domain resizing keeps linear Y but reconstructs UV
        // independently. At source(1.5,1.5), Left siting gives UV(.75,.5),
        // whose nearest-earlier sample is (1,0).
        let mut planes = PreprocessSpec::new(1, 1).unwrap();
        planes.crop = Some(CropRect {
            x: 1,
            y: 1,
            width: 2,
            height: 2,
        });
        planes.resize_domain = ResizeDomain::YuvPlanes;
        planes.chroma_interpolation = ChromaInterpolation::NearestTiesEarlier;
        let y = (block_y(ten_bit, 1, 1)
            + block_y(ten_bit, 2, 1)
            + block_y(ten_bit, 1, 2)
            + block_y(ten_bit, 2, 2))
            * 0.25;
        close(
            &image
                .preprocess(&stream, planes)
                .unwrap()
                .download(&stream)
                .unwrap(),
            &block_rgb(y, block_uv(ten_bit, 1, 0), ten_bit, range).map(|v| v as f32),
            0.000_003,
        );
    }
}

#[test]
#[ignore = "requires actual CUDA; imports retain external context/stream owners and never skip missing hardware"]
fn imported_native_context_and_stream_preserve_owners_quotas_and_queue_registration() {
    use cudarc::driver::CudaContext as NativeContext;
    CudaContext::driver_api_version().expect("CUDA driver is required");
    let native = NativeContext::new(0).expect("a real NVIDIA GPU is required");
    let native_stream = native.default_stream().fork().unwrap();
    let native_identity = native.cu_ctx();
    let stream_identity = native_stream.cu_stream();
    let weak_native = Arc::downgrade(&native);
    let context = CudaContext::from_native(
        native.clone(),
        CudaLimits {
            memory_bytes: 4096,
            streams: 1,
        },
    )
    .unwrap();
    assert!(Arc::ptr_eq(context.native_context(), &native));
    assert_eq!(context.native_context().cu_ctx(), native_identity);
    let stream = context.import_stream(native_stream.clone()).unwrap();
    assert!(Arc::ptr_eq(stream.native_stream(), &native_stream));
    assert_eq!(stream.native_stream().cu_stream(), stream_identity);
    assert_eq!(
        context.stream().unwrap_err().kind(),
        ErrorKind::ResourceExhausted
    );
    assert_eq!(
        context
            .import_stream(native_stream.clone())
            .unwrap_err()
            .kind(),
        ErrorKind::Incompatible
    );

    // Equal actual primary handles do not mean independent cudarc owners share
    // their tracking state. Import requires the same retained Arc owner.
    let separate = NativeContext::new(0).unwrap();
    assert_eq!(separate.cu_ctx(), native_identity);
    assert!(!Arc::ptr_eq(&native, &separate));
    let incompatible_stream = separate.default_stream().fork().unwrap();
    assert_eq!(
        context
            .import_stream(incompatible_stream.clone())
            .unwrap_err()
            .kind(),
        ErrorKind::Incompatible
    );
    drop(incompatible_stream);
    drop(separate);

    let bytes: Vec<u8> = (0..1024).map(|i| (i % 251) as u8).collect();
    let buffer = stream.upload(&bytes).unwrap();
    assert_eq!(context.memory_used(), bytes.len());
    drop(native_stream);
    drop(native);
    assert!(
        weak_native.upgrade().is_some(),
        "imported owners remain retained"
    );
    buffer.readiness().depend_on(stream.queue()).unwrap();
    assert_eq!(stream.download(&buffer).unwrap(), bytes);
    drop(buffer);
    drop(stream);
    assert_eq!(context.memory_used(), 0);

    // Context-default zero can be imported without a new queue or global sync.
    // Cross-stream reads still use explicit completion dependencies even when
    // the original cudarc owner currently has no owned non-default streams.
    let default = context
        .import_stream(context.native_context().default_stream())
        .unwrap();
    assert_eq!(default.queue().id, 0);
    let buffer = default.upload(&bytes).unwrap();
    let alias =
        CudaContext::from_native(context.native_context().clone(), context.limits()).unwrap();
    assert_eq!(
        alias
            .import_stream(alias.native_context().default_stream())
            .unwrap_err()
            .kind(),
        ErrorKind::Incompatible
    );
    buffer.readiness().depend_on(default.queue()).unwrap();
    let reader_context = CudaContext::new(0, CudaLimits::default()).unwrap();
    let reader = reader_context.stream().unwrap();
    assert_eq!(reader.download(&buffer).unwrap(), bytes);
    drop(buffer);
    assert_eq!(context.memory_used(), 0);
    drop(reader);
    drop(reader_context);
    drop(default);
    drop(alias);
    drop(context);
    assert!(
        weak_native.upgrade().is_none(),
        "all imported native ownership retires after final output/queue drop"
    );
}

#[test]
#[ignore = "requires a real NVIDIA GPU; explicit execution must not skip unavailable hardware"]
fn explicit_transfers_cross_stream_dependencies_and_retained_drop() {
    let context = context();
    let producer = context.stream().unwrap();
    let consumer = context.stream().unwrap();
    let bytes: Vec<u8> = (0..4096).map(|i| (i % 251) as u8).collect();
    let buffer = producer.upload(&bytes).unwrap();
    let storage = buffer.storage_lease();
    assert!(storage.host_bytes().is_err());
    assert!(storage.shares_allocation(&buffer.storage_lease()));
    buffer.readiness().depend_on(consumer.queue()).unwrap();
    let retained_again = CudaContext::new(0, CudaLimits::default()).unwrap();
    assert_eq!(retained_again.device(), context.device());
    let independently_wrapped_queue = retained_again.stream().unwrap();
    buffer
        .readiness()
        .depend_on(independently_wrapped_queue.queue())
        .unwrap();
    let detached = consumer.copy(&buffer).unwrap();
    let lease = buffer
        .retain_until(consumer.record_completion().unwrap())
        .unwrap();
    drop(buffer);
    drop(storage);
    drop(producer);
    drop(context);
    assert_eq!(consumer.download(&detached).unwrap(), bytes);
    lease.wait().unwrap();
    drop(lease);
    drop(detached);
    assert_eq!(consumer.context().memory_used(), 0);
    consumer.context().check_errors().unwrap();
}

#[test]
#[ignore = "requires a real NVIDIA GPU and NVRTC; explicit execution must fail when unavailable"]
fn nv12_preprocessing_survives_input_and_stream_wrapper_drop() {
    let context = context();
    let producer = context.stream().unwrap();
    let processing = context.stream().unwrap();
    let download = context.stream().unwrap();
    let bytes = [16_u8, 235, 81, 145, 128, 128];
    let image = CudaYuvImage::new(
        producer.upload(&bytes).unwrap(),
        layout(PixelFormat::Nv12),
        YuvMatrix::Bt709,
        YuvRange::Limited,
        ChromaSiting::Center,
    )
    .unwrap();
    let tensor = image
        .preprocess(&processing, PreprocessSpec::new(2, 2).unwrap())
        .unwrap();
    assert_eq!(tensor.shape(), [1, 3, 2, 2]);
    assert_eq!(tensor.strides(), [12, 4, 2, 1]);
    assert_eq!(tensor.device(), context.device());
    assert!(tensor.storage_lease().host_bytes().is_err());
    drop(image);
    drop(producer);
    drop(processing);
    drop(context);
    let channel = [0.0, 1.0, 65.0 / 219.0, 129.0 / 219.0];
    close(
        &tensor.download(&download).unwrap(),
        &channel.repeat(3),
        0.000_01,
    );
    drop(tensor);
    assert_eq!(download.context().memory_used(), 0);
}

#[test]
#[ignore = "requires a real NVIDIA GPU and NVRTC; explicit execution must fail when unavailable"]
fn p010_high_bits_and_color_normalization_are_exercised() {
    let context = context();
    let stream = context.stream().unwrap();
    let codes = [64_u16, 940, 502, 283, 512, 512];
    let bytes: Vec<u8> = codes
        .into_iter()
        .flat_map(|code| (code << 6).to_le_bytes())
        .collect();
    let image = CudaYuvImage::new(
        stream.upload(&bytes).unwrap(),
        layout(PixelFormat::P010),
        YuvMatrix::Bt2020,
        YuvRange::Limited,
        ChromaSiting::Left,
    )
    .unwrap();
    let tensor = image
        .preprocess(&stream, PreprocessSpec::new(2, 2).unwrap())
        .unwrap();
    close(
        &tensor.download(&stream).unwrap(),
        &[0.0, 1.0, 0.5, 0.25].repeat(3),
        0.000_01,
    );
    let red = CudaYuvImage::new(
        stream.upload(&[81_u8, 81, 81, 81, 90, 240]).unwrap(),
        layout(PixelFormat::Nv12),
        YuvMatrix::Bt601,
        YuvRange::Limited,
        ChromaSiting::TopLeft,
    )
    .unwrap();
    let mut spec = PreprocessSpec::new(2, 2).unwrap();
    spec.layout = TensorLayout::Nhwc;
    spec.normalization = Normalization {
        mean: [0.5, 0.25, 0.125],
        std: [0.5, 0.25, 0.125],
    };
    let tensor = red.preprocess(&stream, spec).unwrap();
    close(
        &tensor.download(&stream).unwrap(),
        &[0.995_607_3, -1.0, -1.0].repeat(4),
        0.000_02,
    );
}

#[test]
#[ignore = "requires a real NVIDIA GPU and NVRTC; explicit execution must fail when unavailable"]
fn pitched_crop_resize_and_both_tensor_layouts_execute_real_kernels() {
    let context = context();
    let stream = context.stream().unwrap();
    let mut bytes = vec![0xee_u8; 48];
    let y = [
        [16, 32, 48, 64],
        [48, 64, 96, 112],
        [96, 128, 160, 192],
        [192, 208, 224, 235],
    ];
    for row in 0..4 {
        bytes[row * 8..row * 8 + 4].copy_from_slice(&y[row]);
    }
    bytes[32..36].fill(128);
    bytes[40..44].fill(128);
    let layout = ImageLayout {
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
    };
    let image = CudaYuvImage::new(
        stream.upload(&bytes).unwrap(),
        layout,
        YuvMatrix::Bt709,
        YuvRange::Limited,
        ChromaSiting::Center,
    )
    .unwrap();
    let mut spec = PreprocessSpec::new(1, 1).unwrap();
    spec.crop = Some(CropRect {
        x: 1,
        y: 1,
        width: 2,
        height: 2,
    });
    spec.layout = TensorLayout::Nhwc;
    let tensor = image.preprocess(&stream, spec).unwrap();
    close(
        &tensor.download(&stream).unwrap(),
        &[96.0 / 219.0; 3],
        0.000_01,
    );
    spec.width = 2;
    spec.height = 2;
    spec.interpolation = Interpolation::Nearest;
    let tensor = image.preprocess(&stream, spec).unwrap();
    let expected: Vec<f32> = [48.0 / 219.0, 80.0 / 219.0, 112.0 / 219.0, 144.0 / 219.0]
        .into_iter()
        .flat_map(|value| [value; 3])
        .collect();
    close(&tensor.download(&stream).unwrap(), &expected, 0.000_01);
}

#[test]
#[ignore = "requires real CUDA contexts; explicit execution must not skip unavailable hardware"]
fn allocation_limits_stream_limits_and_same_ordinal_context_mismatch_are_enforced() {
    let context = CudaContext::new_owned(
        0,
        CudaLimits {
            memory_bytes: 24,
            streams: 1,
        },
    )
    .unwrap();
    let stream = context.stream().unwrap();
    assert_eq!(
        context.stream().unwrap_err().kind(),
        ErrorKind::ResourceExhausted
    );
    let buffer = stream.upload(&[1_u8; 16]).unwrap();
    assert_eq!(
        stream.upload(&[2_u8; 9]).unwrap_err().kind(),
        ErrorKind::ResourceExhausted
    );
    assert_eq!((context.memory_used(), context.memory_peak()), (16, 16));
    let other = CudaContext::new_owned(0, CudaLimits::default()).unwrap();
    assert_ne!(context.device(), other.device());
    let other_stream = other.stream().unwrap();
    assert_eq!(
        other_stream.download(&buffer).unwrap_err().kind(),
        ErrorKind::Incompatible
    );
    assert_eq!(
        buffer
            .readiness()
            .depend_on(other_stream.queue())
            .unwrap_err()
            .kind(),
        ErrorKind::Incompatible
    );
    drop(buffer);
    drop(stream);
    assert_eq!(context.memory_used(), 0);
    context.stream().unwrap();
}

#[derive(Debug)]
struct ExternalOwner {
    _allocation: CudaSlice<u8>,
    device: Device,
    released: Arc<AtomicBool>,
}
impl StorageOwner for ExternalOwner {
    fn byte_len(&self) -> usize {
        6
    }
    fn kind(&self) -> MemoryKind {
        MemoryKind::Opaque
    }
    fn device(&self) -> Device {
        self.device
    }
}
impl Drop for ExternalOwner {
    fn drop(&mut self) {
        self.released.store(true, Ordering::Release);
    }
}

#[test]
#[ignore = "requires real CUDA/NVRTC; exercises native plane import and asynchronous retirement"]
#[allow(unsafe_code)]
fn borrowed_native_planes_retire_after_consumption_while_output_remains_alive() {
    let context = context();
    let producer = context.stream().unwrap();
    let processing = context.stream().unwrap();
    let download = context.stream().unwrap();
    let allocation = producer
        .native_stream()
        .clone_htod(&[16_u8, 235, 81, 145, 128, 128])
        .unwrap();
    let base = {
        let (pointer, guard) = allocation.device_ptr(producer.native_stream());
        drop(guard);
        pointer
    };
    let released = Arc::new(AtomicBool::new(false));
    let owner = Arc::new(ExternalOwner {
        _allocation: allocation,
        device: context.device(),
        released: released.clone(),
    });
    // SAFETY: This is the retained producer's actual native stream and context;
    // it stays alive through event recording and the imported allocation lifetime.
    let completion = unsafe {
        context.record_external_completion(producer.native_stream().cu_stream() as usize as u64)
    }
    .unwrap();
    let lease = StorageLease::from_owner(owner.clone(), Arc::new(completion));
    // SAFETY: Both addresses describe the owned six-byte cudarc allocation in
    // the verified context, with valid NV12 plane sizes and producer ordering.
    let image = unsafe {
        CudaYuvImage::from_external_planes(
            lease,
            layout(PixelFormat::Nv12),
            [base, base + 4],
            YuvMatrix::Bt709,
            YuvRange::Limited,
            ChromaSiting::Center,
        )
    }
    .unwrap();
    assert!(image.buffer().is_none());
    let tensor = image
        .preprocess(&processing, PreprocessSpec::new(2, 2).unwrap())
        .unwrap();
    drop(owner);
    drop(image);
    drop(producer);
    drop(processing);
    drop(context);
    close(
        &tensor.download(&download).unwrap(),
        &[0.0, 1.0, 65.0 / 219.0, 129.0 / 219.0].repeat(3),
        0.000_01,
    );
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    while !released.load(Ordering::Acquire) && std::time::Instant::now() < deadline {
        std::thread::yield_now();
    }
    assert!(
        released.load(Ordering::Acquire),
        "input surface must retire independently of retained output tensor"
    );
    assert_eq!(tensor.shape(), [1, 3, 2, 2]);
    download.context().check_errors().unwrap();
}

#[test]
#[ignore = "requires real CUDA/NVRTC; verifies saturated-edge interpolation order"]
fn rgb_resize_does_not_silently_use_plane_domain_fusion() {
    let context = context();
    let stream = context.stream().unwrap();
    let image = CudaYuvImage::new(
        stream.upload(&[16_u8, 235, 16, 235, 90, 240]).unwrap(),
        layout(PixelFormat::Nv12),
        YuvMatrix::Bt601,
        YuvRange::Limited,
        ChromaSiting::TopLeft,
    )
    .unwrap();
    let mut spec = PreprocessSpec::new(1, 1).unwrap();
    let converted = image.preprocess(&stream, spec).unwrap();
    close(
        &converted.download(&stream).unwrap(),
        &[0.8505, 0.350_656_06, 0.349_696_43],
        0.000_02,
    );
    spec.resize_domain = ResizeDomain::YuvPlanes;
    let planes = image.preprocess(&stream, spec).unwrap();
    close(
        &planes.download(&stream).unwrap(),
        &[1.0, 0.201_312_12, 0.199_392_86],
        0.000_02,
    );
}

#[test]
#[ignore = "requires real CUDA/NVRTC; verifies overlapping owned-buffer reads retire safely"]
fn owned_buffer_reads_on_different_streams_retain_all_prior_work() {
    let context = context();
    let upload = context.stream().unwrap();
    let slow = context.stream().unwrap();
    let fast = context.stream().unwrap();
    let image = CudaYuvImage::new(
        upload.upload(&[81_u8, 81, 81, 81, 90, 240]).unwrap(),
        layout(PixelFormat::Nv12),
        YuvMatrix::Bt601,
        YuvRange::Limited,
        ChromaSiting::TopLeft,
    )
    .unwrap();
    let large = image
        .preprocess(&slow, PreprocessSpec::new(1024, 1024).unwrap())
        .unwrap();
    let small = image
        .preprocess(&fast, PreprocessSpec::new(1, 1).unwrap())
        .unwrap();
    drop(image);
    drop(upload);
    close(
        &small.download(&fast).unwrap(),
        &[0.997_803_6, 0.0, 0.0],
        0.000_02,
    );
    let values = large.download(&fast).unwrap();
    assert!(
        values[..1024 * 1024]
            .iter()
            .all(|&value| (value - 0.997_803_6).abs() < 0.000_02)
    );
    assert!(
        values[1024 * 1024..]
            .iter()
            .all(|&value| value.abs() < 0.000_02)
    );
}

#[test]
#[ignore = "requires real CUDA/NVRTC; verifies one final batched allocation with repeated mixed-format sources"]
fn direct_batch_output_preserves_slot_order_pitch_and_both_layouts() {
    let context = context();
    let upload = context.stream().unwrap();
    let processing = context.stream().unwrap();
    let consumer = context.stream().unwrap();
    let red = CudaYuvImage::new(
        upload.upload(&[81_u8, 81, 81, 81, 90, 240]).unwrap(),
        layout(PixelFormat::Nv12),
        YuvMatrix::Bt601,
        YuvRange::Limited,
        ChromaSiting::TopLeft,
    )
    .unwrap();
    let mut white_bytes = vec![0xee_u8; 24];
    for offset in [0, 2, 8, 10] {
        white_bytes[offset..offset + 2].copy_from_slice(&(940_u16 << 6).to_le_bytes());
    }
    for offset in [16, 18] {
        white_bytes[offset..offset + 2].copy_from_slice(&(512_u16 << 6).to_le_bytes());
    }
    let white_layout = ImageLayout {
        width: 2,
        height: 2,
        allocated_width: 2,
        allocated_height: 2,
        format: PixelFormat::P010,
        planes: vec![
            PlaneLayout {
                offset: 0,
                row_stride: 8,
                row_bytes: 4,
                rows: 2,
            },
            PlaneLayout {
                offset: 16,
                row_stride: 8,
                row_bytes: 4,
                rows: 1,
            },
        ],
    };
    let white = CudaYuvImage::new(
        upload.upload(&white_bytes).unwrap(),
        white_layout,
        YuvMatrix::Bt709,
        YuvRange::Limited,
        ChromaSiting::Center,
    )
    .unwrap();
    let images = [red.clone(), white, red];
    let mut spec = PreprocessSpec::new(2, 2).unwrap();
    let tensor = processing.preprocess_batch(&images, spec).unwrap();
    assert_eq!(tensor.shape(), [3, 3, 2, 2]);
    assert_eq!(tensor.strides(), [12, 4, 2, 1]);
    // 6-byte red input + 24-byte pitched P010 input + one 144-byte final batch.
    // No per-image float allocation or device-to-device concatenation is used.
    assert_eq!(context.memory_peak(), 174);
    let red_nchw = [vec![0.997_803_6; 4], vec![0.; 8]].concat();
    let expected = [red_nchw.clone(), vec![1.; 12], red_nchw].concat();
    close(&tensor.download(&consumer).unwrap(), &expected, 0.000_02);
    drop(tensor);
    assert_eq!(context.memory_used(), 30);
    spec.layout = TensorLayout::Nhwc;
    let tensor = processing.preprocess_batch(&images, spec).unwrap();
    assert_eq!(tensor.shape(), [3, 2, 2, 3]);
    assert_eq!(tensor.strides(), [12, 6, 3, 1]);
    let red_nhwc = [0.997_803_6, 0., 0.].repeat(4);
    let expected = [red_nhwc.clone(), vec![1.; 12], red_nhwc].concat();
    drop(images);
    drop(upload);
    drop(processing);
    drop(context);
    close(&tensor.download(&consumer).unwrap(), &expected, 0.000_02);
    assert_eq!(consumer.context().memory_used(), 144);
    drop(tensor);
    assert_eq!(consumer.context().memory_used(), 0);
}

#[test]
#[ignore = "requires real CUDA; validates full-batch budgets and context errors before NVRTC work"]
fn batch_rejects_invalid_context_and_full_allocation_before_launch() {
    let context = CudaContext::new_owned(
        0,
        CudaLimits {
            memory_bytes: 64,
            streams: 2,
        },
    )
    .unwrap();
    let stream = context.stream().unwrap();
    let image = CudaYuvImage::new(
        stream.upload(&[16_u8, 16, 16, 16, 128, 128]).unwrap(),
        layout(PixelFormat::Nv12),
        YuvMatrix::Bt709,
        YuvRange::Limited,
        ChromaSiting::Center,
    )
    .unwrap();
    let spec = PreprocessSpec::new(2, 2).unwrap();
    let images = [image.clone(), image];
    assert_eq!(
        stream.preprocess_batch(&images, spec).unwrap_err().kind(),
        ErrorKind::ResourceExhausted
    );
    assert_eq!((context.memory_used(), context.memory_peak()), (6, 6));
    let other = CudaContext::new_owned(0, CudaLimits::default()).unwrap();
    let wrong = other.stream().unwrap();
    assert_eq!(
        wrong.preprocess_batch(&images, spec).unwrap_err().kind(),
        ErrorKind::Incompatible
    );
    assert_eq!(other.memory_used(), 0);
    assert_eq!(
        stream.preprocess_batch(&[], spec).unwrap_err().kind(),
        ErrorKind::InvalidInput
    );
}
