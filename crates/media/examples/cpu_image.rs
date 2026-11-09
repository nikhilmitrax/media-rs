//! Apply orientation and resize a still image, with explicit metadata-loss reports.
use media::prelude::*;
fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let input = args
        .next()
        .ok_or_else(|| Error::invalid("usage: cpu_image INPUT OUTPUT"))?;
    let output = args
        .next()
        .ok_or_else(|| Error::invalid("usage: cpu_image INPUT OUTPUT"))?;
    let request = media::open(input)?
        .image()?
        .apply_orientation()
        .resize(Fit::inside(512, 512))
        .allow_metadata_loss();
    println!("{}", request.compile()?.explain());
    let result = request.write(output)?;
    println!("{:?}", result.report);
    Ok(())
}
