//! Real CUDA DLPack consumer, exclusivity, ordering, and retirement checks.
//!
//! Run: cargo test -p media-interop --features cuda --test cuda_dlpack -- --ignored --test-threads=1
//! Explicit execution fails when a CUDA 12.8 driver, NVIDIA GPU, or NVRTC is absent.
#![cfg(feature = "cuda")]

use cudarc::{
    driver::{CudaFunction, CudaSlice, LaunchConfig, PushKernelArg},
    nvrtc::{CompileOptions, compile_ptx_with_opts},
};
use media_backend_cuda::{
    ChromaSiting, CudaContext, CudaLimits, CudaStream, CudaTensor, CudaYuvImage, PreprocessSpec,
    YuvMatrix, YuvRange,
};
use media_core::{
    device::Device,
    error::ErrorKind,
    media::{ImageLayout, PixelFormat, PlaneLayout},
    policy::CopyPolicy,
    storage::Completion,
};
use media_interop::{
    dlpack::{self, DLManagedTensor, DLManagedTensorVersioned, DLTensor},
    native::{NativeTensorDType, NativeTensorFormat, export_cuda_tensor},
};

fn context() -> CudaContext {
    CudaContext::new_owned(
        0,
        CudaLimits {
            memory_bytes: 1024 * 1024,
            streams: 8,
        },
    )
    .expect("an actual NVIDIA GPU with a compatible CUDA 12.8 driver is required")
}

fn tensor(producer: &CudaStream) -> CudaTensor {
    let image = CudaYuvImage::new(
        producer.upload(&[16_u8, 235, 81, 145, 128, 128]).unwrap(),
        ImageLayout {
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
                    offset: 4,
                    row_stride: 2,
                    row_bytes: 2,
                    rows: 1,
                },
            ],
        },
        YuvMatrix::Bt709,
        YuvRange::Limited,
        ChromaSiting::Center,
    )
    .unwrap();
    image
        .preprocess(producer, PreprocessSpec::new(2, 2).unwrap())
        .unwrap()
}

fn expected() -> Vec<f32> {
    [0.0, 1.0, 65.0 / 219.0, 129.0 / 219.0].repeat(3)
}

fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() < 0.000_02,
            "actual {actual}, expected {expected}"
        );
    }
}

/// The consumer is deliberately independent of CudaTensor and its storage APIs.
#[allow(unsafe_code)]
mod consumer {
    use super::*;

    // It reads and mutates the actual foreign device pointer on the consumer
    // stream. The short delay leaves queued work when the descriptor is released.
    const KERNEL: &str = r#"
extern "C" __global__ void consume(float* source, float* output, unsigned int n) {
    if (threadIdx.x == 0) {
        unsigned long long begin = clock64();
        while (clock64() - begin < 20000000ULL) {}
    }
    __syncthreads();
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float value = source[i];
        output[i] = value * 2.0f + 1.0f;
        source[i] = -value;
    }
}
"#;

    fn launch(view: &DLTensor, stream: &CudaStream) -> (CudaSlice<f32>, CudaFunction) {
        let Device::Cuda { ordinal, .. } = stream.context().device() else {
            unreachable!()
        };
        assert_eq!(view.device.device_type, 2);
        assert_eq!(view.device.device_id, ordinal as i32);
        assert_eq!(
            (view.dtype.code, view.dtype.bits, view.dtype.lanes),
            (2, 32, 1)
        );
        assert_eq!(view.ndim, 4);
        assert_eq!(view.byte_offset, 0);
        assert!(!view.data.is_null());
        assert_eq!(view.data as usize % 256, 0);
        // SAFETY: The original live producer descriptor owns four shape/stride
        // entries and its actual device allocation until the final deleter call.
        unsafe {
            assert_eq!(std::slice::from_raw_parts(view.shape, 4), &[1, 3, 2, 2]);
            assert_eq!(std::slice::from_raw_parts(view.strides, 4), &[12, 4, 2, 1]);
        }
        let ptx = compile_ptx_with_opts(
            KERNEL,
            CompileOptions {
                arch: Some("compute_75"),
                ..Default::default()
            },
        )
        .expect("NVRTC is required for the real foreign CUDA consumer");
        let module = stream.context().native_context().load_module(ptx).unwrap();
        let function = module.load_function("consume").unwrap();
        let mut output = stream.native_stream().alloc_zeros::<f32>(12).unwrap();
        let address = view.data as usize as u64;
        let count = 12_u32;
        let mut builder = stream.native_stream().launch_builder(&function);
        builder.arg(&address).arg(&mut output).arg(&count);
        // SAFETY: address is the checked, uniquely writable foreign f32 device
        // allocation in this exact context; both arrays contain twelve elements.
        // Export ordered readiness onto this stream, which is the only consumer.
        unsafe { builder.launch(LaunchConfig::for_num_elems(count)) }.unwrap();
        drop(builder);
        // Keep the loaded module alive while its kernel is pending. Unloading it
        // here could itself synchronize and hide a missing consumer-retirement wait.
        (output, function)
    }

    pub fn versioned(
        raw: *mut DLManagedTensorVersioned,
        stream: &CudaStream,
        flags: u64,
    ) -> Vec<f32> {
        // SAFETY: into_raw transferred the original live descriptor to this
        // consumer. It submits all uses on the selected stream before release,
        // preserves bookkeeping, and calls the matching deleter exactly once.
        unsafe {
            assert_eq!((*raw).version, dlpack::DLPackVersion { major: 1, minor: 3 });
            assert_eq!((*raw).flags, flags);
            let (output, _function) = launch(&(*raw).dl_tensor, stream);
            let completion = stream.record_completion().unwrap();
            // No consumer synchronize precedes deletion: the producer's deleter
            // must record/wait the consumer completion before freeing its bytes.
            (*raw).deleter.unwrap()(raw);
            assert!(
                completion.is_complete().unwrap(),
                "deleter must retire submitted foreign work"
            );
            stream.native_stream().clone_dtoh(&output).unwrap()
        }
    }

    pub fn legacy(raw: *mut DLManagedTensor, stream: &CudaStream) -> Vec<f32> {
        // SAFETY: Same original descriptor, exclusive access, selected stream,
        // final-submission, and exactly-once retirement contract as versioned.
        unsafe {
            let (output, _function) = launch(&(*raw).dl_tensor, stream);
            let completion = stream.record_completion().unwrap();
            (*raw).deleter.unwrap()(raw);
            assert!(
                completion.is_complete().unwrap(),
                "legacy deleter must retire submitted foreign work"
            );
            stream.native_stream().clone_dtoh(&output).unwrap()
        }
    }
}

#[test]
#[ignore = "requires actual CUDA/NVRTC; explicit execution fails if hardware is unavailable"]
fn real_foreign_cuda_kernel_owns_and_retires_legacy_and_versioned_descriptors() {
    for legacy in [false, true] {
        let context = context();
        let producer = context.stream().unwrap();
        let consumer = context.stream().unwrap();
        let tensor = tensor(&producer);
        assert_eq!(context.memory_used(), 48);
        if legacy {
            let raw = dlpack::export_cuda_legacy(tensor, &consumer, CopyPolicy::Forbid)
                .unwrap()
                .into_raw();
            drop(producer);
            drop(context);
            assert_eq!(consumer.context().memory_used(), 48);
            close(
                &consumer::legacy(raw, &consumer),
                &expected().iter().map(|v| v * 2.0 + 1.0).collect::<Vec<_>>(),
            );
        } else {
            let raw = dlpack::export_cuda_versioned(tensor, &consumer, CopyPolicy::Forbid)
                .unwrap()
                .into_raw();
            drop(producer);
            drop(context);
            assert_eq!(consumer.context().memory_used(), 48);
            close(
                &consumer::versioned(raw, &consumer, 0),
                &expected().iter().map(|v| v * 2.0 + 1.0).collect::<Vec<_>>(),
            );
        }
        assert_eq!(consumer.context().memory_used(), 0);
        consumer.context().check_errors().unwrap();
    }
}

#[test]
#[ignore = "requires actual CUDA/NVRTC; explicit execution fails if hardware is unavailable"]
fn every_shared_allocation_lease_rejects_writable_export_without_explicit_detachment() {
    let context = context();
    let producer = context.stream().unwrap();
    let consumer = context.stream().unwrap();

    let value = tensor(&producer);
    let alias = value.clone();
    assert!(dlpack::export_cuda_versioned(value, &consumer, CopyPolicy::Forbid).is_err());
    close(&alias.download(&consumer).unwrap(), &expected());
    drop(alias);

    let value = tensor(&producer);
    let buffer = value.buffer().clone();
    assert!(dlpack::export_cuda_legacy(value, &consumer, CopyPolicy::Forbid).is_err());
    close(&consumer.download(&buffer).unwrap(), &expected());
    drop(buffer);

    let value = tensor(&producer);
    let storage = value.storage_lease();
    assert!(dlpack::export_cuda_versioned(value, &consumer, CopyPolicy::Forbid).is_err());
    assert!(storage.host_bytes().is_err());
    assert_eq!(context.memory_used(), 48);
    drop(storage);
    assert_eq!(context.memory_used(), 0);

    let value = tensor(&producer);
    let alias = value.clone();
    let exported = dlpack::export_cuda_versioned(value, &consumer, CopyPolicy::Allow).unwrap();
    assert_eq!(context.memory_used(), 96);
    let output = consumer::versioned(exported.into_raw(), &consumer, dlpack::FLAG_IS_COPIED);
    close(
        &output,
        &expected().iter().map(|v| v * 2.0 + 1.0).collect::<Vec<_>>(),
    );
    close(&alias.download(&consumer).unwrap(), &expected());
    assert_eq!(context.memory_used(), 48);
    drop(alias);
    assert_eq!(context.memory_used(), 0);
    context.check_errors().unwrap();
}

#[test]
#[ignore = "requires actual CUDA/NVRTC; explicit execution fails if hardware is unavailable"]
fn native_descriptor_checks_real_context_and_owns_the_unique_allocation() {
    let context = context();
    let producer = context.stream().unwrap();
    let consumer = context.stream().unwrap();
    let other = CudaContext::new_owned(0, CudaLimits::default()).unwrap();
    let other_stream = other.stream().unwrap();
    assert_ne!(other.device(), context.device());
    assert_eq!(
        export_cuda_tensor(tensor(&producer), &other_stream, CopyPolicy::Allow)
            .unwrap_err()
            .kind(),
        ErrorKind::Incompatible
    );
    assert_eq!(context.memory_used(), 0);

    let exported = export_cuda_tensor(tensor(&producer), &consumer, CopyPolicy::Forbid).unwrap();
    let descriptor = *exported.descriptor();
    assert_eq!(descriptor.device(), context.device());
    assert_eq!(descriptor.format(), NativeTensorFormat::RgbF32);
    assert_eq!(descriptor.dtype(), NativeTensorDType::F32);
    assert_eq!(descriptor.shape(), [1, 3, 2, 2]);
    assert_eq!(descriptor.strides(), [12, 4, 2, 1]);
    assert_eq!(descriptor.byte_len(), 48);
    assert!(!descriptor.copied());
    assert_ne!(descriptor.address(), 0);
    assert_eq!(descriptor.address() % 256, 0);
    drop(context);
    drop(producer);
    assert_eq!(consumer.context().memory_used(), 48);
    drop(exported);
    assert_eq!(consumer.context().memory_used(), 0);

    let unconsumed =
        dlpack::export_cuda_versioned(tensor(&consumer), &consumer, CopyPolicy::Forbid).unwrap();
    assert_eq!(consumer.context().memory_used(), 48);
    drop(unconsumed);
    assert_eq!(consumer.context().memory_used(), 0);
    consumer.context().check_errors().unwrap();
}
