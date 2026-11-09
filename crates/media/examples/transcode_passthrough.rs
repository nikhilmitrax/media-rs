//! Native NVDEC -> component resize -> NVENC, with compressed audio passthrough.
use media::backend::cuda::CudaLimits;
use media::prelude::*;

fn main() -> Result<()> {
    let mut arguments = std::env::args().skip(1);
    let input = arguments.next().ok_or_else(|| {
        Error::invalid("usage: transcode_passthrough INPUT NEW_OUTPUT [MAX_WIDTH MAX_HEIGHT]")
    })?;
    let output = arguments
        .next()
        .ok_or_else(|| Error::invalid("provide a new output path"))?;
    let width = arguments
        .next()
        .map(|value| value.parse::<u32>())
        .transpose()
        .map_err(|_| Error::invalid("width must be an unsigned integer"))?
        .unwrap_or(1280);
    let height = arguments
        .next()
        .map(|value| value.parse::<u32>())
        .transpose()
        .map_err(|_| Error::invalid("height must be an unsigned integer"))?
        .unwrap_or(720);
    let report = media::open(input)?
        .convert()
        .resize_components(Fit::inside(width, height))
        .bitrate(4_000_000)
        .cuda(0, CudaLimits::default())
        .write(output)?;
    println!("{report:?}");
    Ok(())
}
