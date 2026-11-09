//! Required hardware sampling, fused GPU preprocessing, and a retained DLPack export.
use media::backend::cuda::{CudaLimits, PreprocessSpec};
use media::prelude::*;

fn main() -> Result<()> {
    let mut arguments = std::env::args().skip(1);
    let input = arguments
        .next()
        .ok_or_else(|| Error::invalid("usage: gpu_video_tensor INPUT [SECONDS ...]"))?;
    let mut times = Vec::new();
    for argument in arguments {
        times.push(Time::from_seconds(argument.parse::<f64>().map_err(
            |_| Error::invalid("timestamps must be finite seconds"),
        )?)?);
    }
    if times.is_empty() {
        times = vec![
            Time::from_millis(200),
            Time::ZERO,
            Time::from_millis(200),
            Time::from_millis(100),
        ];
    }
    let plan = media::open(input)?
        .video()?
        .sample(times)
        .duration_inference(DurationInference::NextTimestamp)
        .cuda(0, CudaLimits::default())
        .to_tensor(PreprocessSpec::new(224, 224)?)
        .compile()?;
    println!("{}", plan.explain());
    let batch = plan.execute()?;
    println!(
        "shape: {:?}; report: {:?}",
        batch.tensor.shape(),
        batch.report
    );
    let consumer = batch.context().stream()?;
    let export =
        media::interop::dlpack::export_cuda_versioned(batch.tensor, &consumer, CopyPolicy::Forbid)?;
    println!("retained DLPack descriptor: {:p}", export.as_ptr());
    // A native consumer can take export.into_raw() and call its deleter once,
    // after its final submission on this same consumer stream. Keeping this
    // wrapper instead uses RAII and waits for consumer completion when dropped.
    drop(export);
    Ok(())
}
