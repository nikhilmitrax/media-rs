//! Checked FFmpeg 5.1+ audio channel metadata unavailable in the safe wrapper.
use ffmpeg_next as ffmpeg;
use media_core::{
    error::{Error, Result},
    media::{ChannelLayout, ChannelPosition},
};

pub(crate) fn decoder_layout(decoder: &ffmpeg::decoder::Audio) -> Result<ChannelLayout> {
    // SAFETY: decoder owns a live AVCodecContext. This borrowed channel layout
    // remains valid throughout validation and no native pointer escapes.
    unsafe {
        layout_with_legacy(
            &(*decoder.as_ptr()).ch_layout,
            decoder.channels(),
            decoder.channel_layout().bits(),
        )
    }
}
pub(crate) fn frame_layout(frame: &ffmpeg::frame::Audio) -> Result<ChannelLayout> {
    // SAFETY: frame owns a live AVFrame and its channel layout for this borrow.
    unsafe {
        layout_with_legacy(
            &(*frame.as_ptr()).ch_layout,
            frame.channels(),
            frame.channel_layout().bits(),
        )
    }
}

pub(crate) fn establish_mpeg_channels(
    decoder: &mut ffmpeg::decoder::Audio,
    packet: &ffmpeg::Packet,
) -> Result<()> {
    let bytes = packet
        .data()
        .and_then(|bytes| bytes.get(..4))
        .ok_or_else(|| Error::invalid("short MPEG audio header"))?;
    let header = u32::from_be_bytes(bytes.try_into().expect("four-byte header"));
    let version = (header >> 19) & 3;
    let layer = (header >> 17) & 3;
    let sample_index = (header >> 10) & 3;
    if header >> 21 != 0x7ff || version == 1 || layer == 0 || sample_index == 3 {
        return Err(Error::invalid(
            "invalid MPEG audio header during bounded channel discovery",
        ));
    }
    let rate = [44100, 48000, 32000][sample_index as usize]
        / match version {
            3 => 1,
            2 => 2,
            _ => 4,
        };
    if decoder.rate() != rate {
        return Err(Error::incompatible(
            "MPEG audio header rate differs from stream metadata",
        ));
    }
    let channels = if (header >> 6) & 3 == 3 {
        ffmpeg::ChannelLayout::MONO
    } else {
        ffmpeg::ChannelLayout::STEREO
    };
    decoder.set_channel_layout(channels);
    // SAFETY: the decoder exclusively owns its live AVCodecContext. The actual
    // compressed MPEG header establishes mono/stereo. Set the modern field too
    // because ffmpeg-next's FFmpeg 6 compatibility setter updates only legacy
    // fields. Native mono/stereo masks own no dynamically allocated channel map.
    unsafe {
        let layout = &mut (*decoder.as_mut_ptr()).ch_layout;
        ffmpeg::ffi::av_channel_layout_uninit(layout);
        let status = ffmpeg::ffi::av_channel_layout_from_mask(layout, channels.bits());
        if status < 0 {
            return Err(crate::format::backend_error(ffmpeg::Error::from(status)));
        }
    }
    Ok(())
}
fn layout_with_legacy(
    current: &ffmpeg::ffi::AVChannelLayout,
    channels: u16,
    mask: u64,
) -> Result<ChannelLayout> {
    if current.nb_channels > 0 {
        return layout(current);
    }
    // FFmpeg 6 can populate only the deprecated count/mask before the first
    // decoded frame. The safe wrapper selects those legacy fields on that SDK.
    let legacy = ffmpeg::ffi::AVChannelLayout {
        order: if mask == 0 {
            ffmpeg::ffi::AVChannelOrder::AV_CHANNEL_ORDER_UNSPEC
        } else {
            ffmpeg::ffi::AVChannelOrder::AV_CHANNEL_ORDER_NATIVE
        },
        nb_channels: i32::from(channels),
        u: ffmpeg::ffi::AVChannelLayout__bindgen_ty_1 { mask },
        opaque: std::ptr::null_mut(),
    };
    layout(&legacy)
}
fn layout(layout: &ffmpeg::ffi::AVChannelLayout) -> Result<ChannelLayout> {
    use ffmpeg::ffi::{AVChannel, AVChannelOrder};
    if layout.nb_channels <= 0 || layout.nb_channels > 64 {
        return Err(Error::unsupported("FFmpeg audio needs 1..=64 channels"));
    }
    if layout.order == AVChannelOrder::AV_CHANNEL_ORDER_UNSPEC {
        // Preserve ordinary WAV and other unpositioned channels without
        // inventing speakers. Named downmixing still requires an explicit mix.
        return ChannelLayout::new(
            (0..layout.nb_channels as u16)
                .map(ChannelPosition::Discrete)
                .collect(),
        );
    }
    if layout.order != AVChannelOrder::AV_CHANNEL_ORDER_NATIVE {
        return Err(Error::unsupported(
            "FFmpeg custom/ambisonic speaker layouts are unsupported",
        ));
    }
    let mut positions = Vec::with_capacity(layout.nb_channels as usize);
    for index in 0..layout.nb_channels as u32 {
        // SAFETY: a validated live native layout is borrowed for the call; index
        // is below its positive bounded channel count and no pointer is retained.
        let channel = unsafe { ffmpeg::ffi::av_channel_layout_channel_from_index(layout, index) };
        positions.push(match channel {
            AVChannel::AV_CHAN_FRONT_CENTER if layout.nb_channels == 1 => ChannelPosition::Mono,
            AVChannel::AV_CHAN_FRONT_LEFT => ChannelPosition::FrontLeft,
            AVChannel::AV_CHAN_FRONT_RIGHT => ChannelPosition::FrontRight,
            AVChannel::AV_CHAN_FRONT_CENTER => ChannelPosition::FrontCenter,
            AVChannel::AV_CHAN_LOW_FREQUENCY => ChannelPosition::Lfe,
            AVChannel::AV_CHAN_BACK_LEFT => ChannelPosition::BackLeft,
            AVChannel::AV_CHAN_BACK_RIGHT => ChannelPosition::BackRight,
            AVChannel::AV_CHAN_SIDE_LEFT => ChannelPosition::SideLeft,
            AVChannel::AV_CHAN_SIDE_RIGHT => ChannelPosition::SideRight,
            other => {
                return Err(Error::unsupported(format!(
                    "FFmpeg speaker {other:?} has no established portable mapping"
                )));
            }
        });
    }
    ChannelLayout::new(positions)
}

/// Scalar conversion with identity channel order. No speaker layout is needed
/// because there is no mixing, filtering, delay or change in sample rate.
pub(crate) fn discrete_samples(frame: &ffmpeg::frame::Audio, channels: usize) -> Result<Vec<f32>> {
    use ffmpeg::format::Sample;
    let count = frame
        .samples()
        .checked_mul(channels)
        .ok_or_else(|| Error::overflow("discrete audio sample count"))?;
    let planar = frame.is_planar();
    let planes = if planar { channels } else { 1 };
    if frame.planes() != planes || frame.format().bytes() == 0 {
        return Err(Error::incompatible(
            "decoded discrete audio planes are inconsistent",
        ));
    }
    let per_plane = if planar { frame.samples() } else { count };
    let byte_count = per_plane
        .checked_mul(frame.format().bytes())
        .ok_or_else(|| Error::overflow("discrete audio plane size"))?;
    let mut output = vec![0.0; count];
    for plane in 0..planes {
        let bytes = frame
            .data(plane)
            .get(..byte_count)
            .ok_or_else(|| Error::incompatible("decoded discrete audio plane is short"))?;
        for (i, bytes) in bytes.chunks_exact(frame.format().bytes()).enumerate() {
            let sample = match frame.format() {
                Sample::U8(_) => (f32::from(bytes[0]) - 128.0) / 128.0,
                Sample::I16(_) => {
                    f32::from(i16::from_ne_bytes(
                        bytes.try_into().expect("two-byte sample"),
                    )) / 32768.0
                }
                Sample::I32(_) => {
                    i32::from_ne_bytes(bytes.try_into().expect("four-byte sample")) as f32
                        / 2147483648.0
                }
                Sample::I64(_) => {
                    (i64::from_ne_bytes(bytes.try_into().expect("eight-byte sample")) as f64
                        / 9223372036854775808.0) as f32
                }
                Sample::F32(_) => f32::from_ne_bytes(bytes.try_into().expect("four-byte sample")),
                Sample::F64(_) => {
                    f64::from_ne_bytes(bytes.try_into().expect("eight-byte sample")) as f32
                }
                Sample::None => {
                    return Err(Error::unsupported("decoded audio has no scalar format"));
                }
            };
            output[if planar { i * channels + plane } else { i }] = sample;
        }
    }
    Ok(output)
}
