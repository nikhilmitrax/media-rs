//! End-to-end pixel-to-tensor packing and output ownership through the facade.
#![cfg(all(feature = "native", feature = "interop"))]
use media::{backend::native::image as codec, io::source::Source, prelude::*};
fn asset(session: &Session) -> Asset {
    let image = Image::from_packed(2, 1, PixelFormat::Rgb8, vec![255, 0, 128, 0, 255, 64]).unwrap();
    let encoded = codec::encode(&image, EncodeOptions::new(ImageFormat::Png)).unwrap();
    session.from_source(Source::memory(encoded.bytes))
}
#[test]
fn explicit_color_assumption_and_normalization_produce_physical_nchw() {
    let session = Session::default();
    let request = asset(&session).image().unwrap();
    assert!(
        request
            .clone()
            .to_tensor(TensorSpec::rgb_f32())
            .read()
            .is_err()
    );
    let tensor = request
        .to_tensor(
            TensorSpec::rgb_f32()
                .color_assumption(ColorAssumption::AssumeSrgb)
                .normalize([0.5; 3], [0.5; 3]),
        )
        .read()
        .unwrap();
    assert_eq!(tensor.shape(), [1, 3, 1, 2]);
    assert_eq!(&tensor.data()[..4], &[1.0, -1.0, -1.0, 1.0]);
    assert!(!tensor.assumptions().is_empty());
    assert!(session.live_host_bytes() >= tensor.allocation_bytes());
    let clone = tensor.clone();
    drop(tensor);
    assert!(session.live_host_bytes() > 0);
    drop(clone);
    assert_eq!(session.live_host_bytes(), 0);
}
#[test]
fn output_limit_failure_releases_all_image_and_tensor_guards() {
    let session = Session::default();
    let spec = TensorSpec::rgb_f32()
        .color_assumption(ColorAssumption::AssumeSrgb)
        .max_bytes(4);
    assert!(
        asset(&session)
            .image()
            .unwrap()
            .to_tensor(spec)
            .read()
            .is_err()
    );
    assert_eq!(session.live_host_bytes(), 0);
}
#[test]
fn retained_tensors_and_host_foreign_exports_hold_the_batch_slot() {
    let session = Session::builder()
        .limits(Limits::new().in_flight_batches(1))
        .build()
        .unwrap();
    let request = asset(&session)
        .image()
        .unwrap()
        .to_tensor(TensorSpec::rgb_f32().color_assumption(ColorAssumption::AssumeSrgb));
    let tensor = request.read().unwrap();
    let alias = tensor.clone();
    drop(tensor);
    assert_eq!(session.live_batches(), 1);
    assert_eq!(
        request.read().unwrap_err().kind(),
        ErrorKind::ResourceExhausted
    );
    let exported = media::interop::dlpack::export_versioned(alias).unwrap();
    assert_eq!(session.live_batches(), 1);
    drop(exported);
    assert_eq!(session.live_batches(), 0);
    assert!(request.read().is_ok());
}

#[cfg(feature = "icc")]
#[test]
fn native_managed_srgb_output_packs_through_explicit_icc_conversion_and_retires_budgets() {
    let session = Session::builder()
        .limits(Limits::new().host_memory(4096))
        .build()
        .unwrap();
    let image =
        Image::from_packed(2, 1, PixelFormat::Rgb8, vec![30, 60, 90, 120, 150, 180]).unwrap();
    let bytes = codec::encode(&image, EncodeOptions::new(ImageFormat::Png))
        .unwrap()
        .bytes;
    let request = session
        .from_source(Source::memory(bytes))
        .image()
        .unwrap()
        .convert_color(ColorConvertSpec {
            target: Some(ColorSpace::Srgb),
            missing_color: Some(ColorSpace::Srgb),
            ..Default::default()
        });
    assert!(
        request
            .clone()
            .to_tensor(TensorSpec::rgb_f32())
            .read()
            .is_err()
    );
    assert_eq!(session.live_host_bytes(), 0);
    assert_eq!(session.live_batches(), 0);
    let spec = TensorSpec::rgb_f32().icc_policy(IccPolicy::ConvertToSrgb {
        gamut: IccGamut::Reject,
    });
    let tensor = request.clone().to_tensor(spec.clone()).read().unwrap();
    assert_eq!(tensor.shape(), &[1, 3, 1, 2]);
    let expected = [
        30.0 / 255.0,
        120.0 / 255.0,
        60.0 / 255.0,
        150.0 / 255.0,
        90.0 / 255.0,
        180.0 / 255.0,
    ];
    for (actual, expected) in tensor.data().iter().zip(expected) {
        assert!((actual - expected).abs() < 0.0001);
    }
    assert!(tensor.assumptions().is_empty());
    assert!(
        tensor
            .color_operations()
            .iter()
            .any(|v| v.contains("nonlinear sRGB"))
    );
    assert!(session.live_host_bytes() >= tensor.allocation_bytes());
    let retained = tensor.clone();
    drop(tensor);
    assert_eq!(session.live_batches(), 1);
    drop(retained);
    assert_eq!(session.live_host_bytes(), 0);
    assert_eq!(session.live_batches(), 0);
    let limited = spec.icc_limits(IccLimits {
        max_profile_bytes: 16,
        ..Default::default()
    });
    assert_eq!(
        request.to_tensor(limited).read().unwrap_err().kind(),
        ErrorKind::ResourceExhausted
    );
    assert_eq!(session.live_host_bytes(), 0);
    assert_eq!(session.live_batches(), 0);
}
