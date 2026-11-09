//! DLPack C ABI ownership, alignment, mutability, and eventual resource retirement.

use media_interop::{
    dlpack::{self, DLPackVersion},
    tensor::Tensor,
};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

#[derive(Debug)]
struct Retired(Arc<AtomicUsize>);
impl Drop for Retired {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
}

#[allow(unsafe_code)]
mod consumer {
    use media_interop::dlpack::{DLManagedTensor, DLManagedTensorVersioned};

    pub fn inspect_and_mutate(raw: *mut DLManagedTensorVersioned, expected_pointer: *const f32) {
        // SAFETY: Tests transfer one original live descriptor to this consumer,
        // use it exclusively, and release it exactly once after the final access.
        unsafe {
            let tensor = &mut *raw;
            assert_eq!(tensor.version.major, 1);
            assert_eq!(tensor.version.minor, 3);
            assert_eq!(tensor.flags, 0);
            let view = &mut tensor.dl_tensor;
            assert_eq!(view.device.device_type, 1);
            assert_eq!(view.device.device_id, 0);
            assert_eq!(view.ndim, 2);
            assert_eq!(view.dtype.code, 2);
            assert_eq!(view.dtype.bits, 32);
            assert_eq!(view.dtype.lanes, 1);
            assert_eq!(view.byte_offset, 0);
            assert_eq!(view.data.cast::<f32>(), expected_pointer.cast_mut());
            assert_eq!((view.data as usize) % 256, 0);
            assert_eq!(std::slice::from_raw_parts(view.shape, 2), &[2, 2]);
            assert_eq!(std::slice::from_raw_parts(view.strides, 2), &[2, 1]);
            let samples = std::slice::from_raw_parts_mut(view.data.cast::<f32>(), 4);
            assert_eq!(samples, &[1., 2., 3., 4.]);
            samples[0] = 42.;
            assert_eq!(samples[0], 42.);
            tensor.deleter.expect("producer deleter")(raw);
        }
    }
    pub fn release_versioned(raw: *mut DLManagedTensorVersioned) {
        // SAFETY: The unique descriptor comes from into_raw and has not been released.
        unsafe {
            (*raw).deleter.expect("producer deleter")(raw);
        }
    }
    pub fn legacy(raw: *mut DLManagedTensor) {
        // SAFETY: Original legacy descriptor is live, exclusively owned, and released once.
        unsafe {
            assert_eq!((*raw).dl_tensor.dtype.bits, 32);
            assert_eq!((*raw).dl_tensor.device.device_type, 1);
            assert_eq!(*(*raw).dl_tensor.data.cast::<f32>(), 7.);
            (*raw).deleter.expect("producer deleter")(raw);
        }
    }
    pub fn empty(raw: *mut DLManagedTensorVersioned) {
        // SAFETY: Live unique descriptor remains owned until its final deleter call.
        unsafe {
            assert!((*raw).dl_tensor.data.is_null());
            assert!(!(*raw).dl_tensor.strides.is_null());
            assert_eq!(*(*raw).dl_tensor.shape, 0);
            (*raw).deleter.expect("producer deleter")(raw);
        }
    }
    pub fn scalar(raw: *mut DLManagedTensorVersioned) {
        // SAFETY: Live scalar descriptor holds one aligned f32 until release.
        unsafe {
            assert_eq!((*raw).dl_tensor.ndim, 0);
            assert!((*raw).dl_tensor.shape.is_null());
            assert!((*raw).dl_tensor.strides.is_null());
            assert_eq!(*(*raw).dl_tensor.data.cast::<f32>(), 9.);
            (*raw).deleter.expect("producer deleter")(raw);
        }
    }
}

#[test]
fn versioned_foreign_consumer_can_write_and_controls_retirement() {
    let dropped = Arc::new(AtomicUsize::new(0));
    let tensor = Tensor::from_vec(vec![1., 2., 3., 4.], vec![2, 2])
        .unwrap()
        .retain(Retired(dropped.clone()));
    let pointer = tensor.data().as_ptr();
    let export = dlpack::export_versioned(tensor).unwrap();
    assert_eq!(dropped.load(Ordering::Relaxed), 0);
    let raw = export.into_raw();
    assert_eq!(dropped.load(Ordering::Relaxed), 0);
    consumer::inspect_and_mutate(raw, pointer);
    assert_eq!(dropped.load(Ordering::Relaxed), 1);
}

#[test]
fn an_unconsumed_export_is_released_by_raii() {
    let dropped = Arc::new(AtomicUsize::new(0));
    let tensor = Tensor::from_vec(vec![7.], vec![1])
        .unwrap()
        .retain(Retired(dropped.clone()));
    let export = dlpack::export_versioned(tensor).unwrap();
    assert!(!export.as_ptr().is_null());
    drop(export);
    assert_eq!(dropped.load(Ordering::Relaxed), 1);
}

#[test]
fn aliased_storage_is_rejected_and_an_explicit_independent_copy_exports() {
    let tensor = Tensor::from_vec(vec![7.], vec![1]).unwrap();
    let alias = tensor.clone();
    assert!(dlpack::export_versioned(tensor).is_err());
    assert_eq!(alias.data(), &[7.]);
    let detached = alias.deep_copy().unwrap();
    assert!(!alias.shares_storage(&detached));
    let export = dlpack::export_versioned(detached).unwrap();
    consumer::release_versioned(export.into_raw());
    assert_eq!(alias.data(), &[7.]);
    consumer::legacy(dlpack::export_legacy(alias).unwrap().into_raw());
}

#[test]
fn empty_and_scalar_tensors_use_their_required_c_descriptors() {
    consumer::empty(
        dlpack::export_versioned(Tensor::from_vec(vec![], vec![0, 3]).unwrap())
            .unwrap()
            .into_raw(),
    );
    consumer::scalar(
        dlpack::export_versioned(Tensor::from_vec(vec![9.], vec![]).unwrap())
            .unwrap()
            .into_raw(),
    );
}

#[cfg(target_pointer_width = "64")]
#[test]
fn descriptor_layout_matches_the_dlpack_header_on_64_bit_hosts() {
    use media_interop::dlpack::{
        DLDataType, DLDevice, DLManagedTensor, DLManagedTensorVersioned, DLTensor,
    };
    use std::mem::{offset_of, size_of};
    assert_eq!(size_of::<DLPackVersion>(), 8);
    assert_eq!(size_of::<DLDevice>(), 8);
    assert_eq!(size_of::<DLDataType>(), 4);
    assert_eq!(size_of::<DLTensor>(), 48);
    assert_eq!(size_of::<DLManagedTensor>(), 64);
    assert_eq!(size_of::<DLManagedTensorVersioned>(), 80);
    assert_eq!(offset_of!(DLManagedTensorVersioned, manager_ctx), 8);
    assert_eq!(offset_of!(DLManagedTensorVersioned, deleter), 16);
    assert_eq!(offset_of!(DLManagedTensorVersioned, flags), 24);
    assert_eq!(offset_of!(DLManagedTensorVersioned, dl_tensor), 32);
}
