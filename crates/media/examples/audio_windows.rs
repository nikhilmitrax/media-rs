//! Decode, downmix, resample continuously, and form complete 20 ms windows.
use media::prelude::*;
fn main() -> Result<()> {
    let input = std::env::args()
        .nth(1)
        .ok_or_else(|| Error::invalid("usage: audio_windows INPUT"))?;
    let mut stream = media::open(input)?
        .audio()?
        .downmix(ChannelLayout::mono())
        .resample(16000)
        .windows(Window::samples(320).hop(160))
        .stream()?;
    let mut count = 0;
    while let Some(window) = stream.next_block()? {
        count += 1;
        println!(
            "window {count}: {} ({} frames)",
            window.start(),
            window.frames()
        );
    }
    println!(
        "discarded incomplete tail: {} frames",
        stream.discarded_window_frames()
    );
    Ok(())
}
