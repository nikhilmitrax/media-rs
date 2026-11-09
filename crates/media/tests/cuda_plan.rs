//! GPU plan descriptions remain lazy and inspectable without NVIDIA hardware.
#![cfg(all(feature = "ffmpeg", feature = "cuda"))]

use media::{
    Session,
    backend::cuda::{ChromaInterpolation, CudaLimits, Interpolation, PreprocessSpec},
    core::{
        error::ErrorKind,
        policy::{FrameDurationPolicy, TransferPolicy},
        time::Time,
    },
    cuda_session::CudaSessionLimits,
    io::source::Source,
};

#[test]
fn lazy_gpu_description_reports_known_policies_without_inventing_context_identity() {
    let session = Session::builder()
        .cuda_limits(CudaSessionLimits {
            max_codec_sessions: 3,
            max_native_bytes: 4096,
        })
        .build()
        .unwrap();
    let limits = CudaLimits {
        memory_bytes: 8192,
        streams: 2,
    };
    let times = [Time::from_millis(100), Time::ZERO, Time::from_millis(100)];
    let mut spec = PreprocessSpec::new(4, 2).unwrap();
    spec.chroma_interpolation = ChromaInterpolation::NearestTiesEarlier;
    let plan = session
        .from_source(Source::memory(Vec::new()))
        .video()
        .unwrap()
        .decode_options(media::backend::ffmpeg::codec::VideoDecodeOptions {
            duration_policy: FrameDurationPolicy::TrustReported,
            ..Default::default()
        })
        .cuda(3, limits)
        .sample(times)
        .to_tensor(spec)
        .compile()
        .unwrap();
    let description = plan.description();
    assert_eq!(description.ordinal, 3);
    assert!(
        description.context.is_none(),
        "compile never makes up an ordinal-only native context"
    );
    assert_eq!(description.context_limits, limits);
    assert_eq!(
        description.preprocessing.chroma_interpolation,
        ChromaInterpolation::NearestTiesEarlier
    );
    assert_eq!(
        description.preprocessing.interpolation,
        Interpolation::Bilinear
    );
    assert!(
        plan.explain()
            .contains("Chroma interpolation NearestTiesEarlier")
    );
    assert_eq!(description.output_shape, Some([3, 3, 2, 4]));
    assert_eq!(
        description.requested_times.as_deref(),
        Some(times.as_slice())
    );
    assert_eq!(description.session_resources.max_codec_sessions, 3);
    assert!(description.hardware_required);
    assert_eq!(
        description.decoder.duration_policy,
        FrameDurationPolicy::TrustReported
    );
    assert_eq!(description.decoded_transfers, TransferPolicy::ForbidHost);
    assert!(!description.unresolved.is_empty());
    assert_eq!(session.live_cuda_codec_sessions(), 0);
    assert_eq!(session.reserved_cuda_native_bytes(), 0);
}

#[test]
fn empty_gpu_tensor_schedule_fails_during_compile_before_native_bootstrap() {
    let request = Session::default()
        .from_source(Source::memory(Vec::new()))
        .video()
        .unwrap()
        .cuda(0, CudaLimits::default())
        .sample([])
        .to_tensor(PreprocessSpec::new(2, 2).unwrap());
    assert_eq!(
        request.compile().unwrap_err().kind(),
        ErrorKind::InvalidInput
    );
}
