# media-rs

Media-rs is a project to create a unified and simplified interface for efficient
ingestion, processing and output of images, audio and video.

It aims to simplify efficient data processing for ML applications such as
training, with zero-copy access and no host roundtrip where the backend supports
it. The goal is a performant, Rust-native alternative to TorchCodec.

It builds on existing battle-tested libraries (FFmpeg, image, Symphonia) and
wraps them in a simpler, unified interface.

The core workflows are implemented and usable. The API is still experimental,
and developer documentation is a work in progress. Start with the examples below
and the [runnable examples](crates/media/examples/README.md).

## AI Usage Disclosure
- Manually designed API with AI agents largely writing the code.
- Tested manually and via agents.

## Current Features

| Path | Current implementation |
| --- | --- |
| Still images | PNG/JPEG/WebP/BMP decode, native precision, orientation/crop/resize/pad, explicit pixel/ICC/color/alpha conversion, PNG/JPEG/WebP encode |
| Animation and pages | Composited GIF/APNG/WebP frames with rational timing/loops; independent supported TIFF pages; exact APNG/GIF/TIFF encoding |
| Audio | Symphonia decode, explicit channel matrices, continuous Rubato resampling, exact sample-grid slices/windows, typed PCM and finalized float WAV; optional FFmpeg AAC timing |
| Video | FFmpeg CPU decode, presentation-time batches, explicit reusable timing indexes and GOP-local seeks, packet codec state machines, incremental writer, remux and CPU transcode with audio passthrough |
| GPU | Required NVDEC NV12/P010, fused crop/resize/YUV-to-RGB/normalize/pack, direct batch allocation, explicit transfers and native/DLPack exports |
| Clips and datasets | Validated common-clock CPU or CUDA-video/CPU-audio clips, separate modality boundary policies, deterministic sharding/tail masks, bounded worker admission/prefetch |
| Sources | Files, immutable memory, reader factories, seekable factories, concurrent versioned ranges, optional HTTP/HTTPS strong-ETag ranges |

## Getting started

You need Rust 1.88 or newer (edition 2024). The default features cover native
image and audio processing and need no FFmpeg SDK or CUDA installation.

The crates are not published to crates.io yet. From a source checkout, try an
example with your own input:

```sh
cargo run -p media --example cpu_image -- photo.jpg thumbnail.webp
cargo run -p media --example audio_windows -- speech.wav
```

To use the library from a neighboring Rust project, add a path dependency:

```toml
[dependencies]
media = { path = "../media-rs/crates/media" }
```

Video needs the `ffmpeg` feature and system development libraries. For GPU video
and DLPack exports, enable `ffmpeg`, `cuda` and `interop` together:

```toml
media = { path = "../media-rs/crates/media", features = ["ffmpeg", "cuda", "interop"] }
```

See [building and testing](#building-and-testing) for system requirements and
the [feature table](#features) for smaller configurations.

## Examples

### Resize an image

Still-image geometry uses stored components by default. Color-managed resize is
explicit, including how absent color information and alpha are interpreted.

```rust
use media::prelude::*;

let output = media::open("photo.jpg")?
    .image()?
    .apply_orientation()
    .resize(Fit::inside(512, 512))
    .allow_metadata_loss()
    .write("thumbnail.webp")?;
println!("{:?}", output.report);
```

### Read audio windows

Continuous audio windows retain resampler history and the source sample-grid
origin. Named downmixing requires established speaker positions; use an explicit
matrix for unspecified layouts.

```rust
use media::prelude::*;

let mut reader = media::open("speech.wav")?
    .audio()?
    .downmix(ChannelLayout::mono())
    .resample(16_000)
    .windows(Window::samples(320).hop(160))
    .stream()?;
while let Some(window) = reader.next_block()? {
    process(window)?;
}
```

For semantic M4A/MP4 audio, enable `ffmpeg-audio` and select `.ffmpeg()` before
the audio terminal operation. Native MP4 reads require an explicit raw-decoded
policy because their edit/priming mapping is not established. FFmpeg audio uses
source presentation time by default; asset time is an explicit timeline choice.

Whole-stream audio reads expose actual decoded coverage, including a later first
valid sample when codec priming outlasts the header start. Genuine timestamp
gaps/overlaps fail under the normal clock policy. For sample processing of a file
with broken timestamps, `audio()?.decoded_sample_clock()` explicitly selects
FFmpeg, keeps all valid decoded samples on a continuous reconstructed clock, and
retains original timestamps in decoder blocks' `source_start()` before
resampling. Its report counts reconstructions; this mode does not promise video
synchronization or fill missing samples. Named downmixing still requires
established speakers or an explicit mix.

### Sample video into a CUDA tensor

The typed CUDA path requires actual hardware decode and forbids decoded host
crossings. RGB retains the source primaries and transfer function; tensor
normalization does not imply ICC conversion or HDR tone mapping.

```rust
use media::prelude::*;
use media::backend::cuda::{CudaLimits, PreprocessSpec};

let plan = media::open("movie.mp4")?.video()?
    .sample([Time::from_millis(900), Time::from_millis(100), Time::from_millis(900)])
    .duration_inference(DurationInference::NextTimestamp)
    .cuda(0, CudaLimits::default())
    .to_tensor(PreprocessSpec::new(224, 224)?)
    .compile()?;
println!("{}", plan.explain());
let batch = plan.execute()?;
// batch.tensor remains in CUDA memory. Downloading is a separate explicit call.
```

Runnable examples live in [crates/media/examples](crates/media/examples/README.md).
Backend escape hatches remain available through `media::backend`; shared media
objects, rational time, source adapters, and policy types are re-exported.

### Reuse a video index

For repeated timestamp queries, explicitly build and reuse a decoded timing
index. Construction performs a bounded full decode once; subsequent reads seek
to verified keyframes and decode nearby dependency frames. Default reads retain
their prefix behavior and never build a hidden index.

```rust
use media::{prelude::*, video_index::IndexBuildLimits};

let session = Session::default();
let video = session.open("movie.mp4")?.video()?;
let index = video.build_index(IndexBuildLimits::default())?;
index.save("movie.media-index")?; // New file only; trusted local cache.
let frames = video.with_index(index)
    .sample([Time::from_millis(9000), Time::from_millis(1000)])
    .duration_inference(DurationInference::NextTimestamp)
    .read()?;
println!("{} decoded, {} seeks", frames.report.decoded_frames, frames.report.seeks);
```

The CUDA request supports the same index API and builds its index through NVDEC
without downloading pixels. A file-backed request can use `load_index(path,
limits)` on later runs. Persistence supports versioned files/public custom
identities; private, range and memory identities remain in-process. Index files
are caller-trusted evidence, with checksums and source/track/policy validation.
Cache version 2 also records compressed key-packet DTS for MPEG navigation;
rebuild caches written by the earlier format. Presentation timestamps and global
frame identities remain separate from the timestamp used to navigate a demuxer.

## Building and testing

Rust 1.88 or newer is required (edition 2024). Default native operation needs no
FFmpeg SDK or CUDA installation. Fetch dependencies once before offline checks.

```sh
cargo test --workspace --exclude media-backend-ffmpeg --locked
cargo check -p media --no-default-features --locked
cargo test -p media --features interop,http-tls --locked
```

The FFmpeg feature links system FFmpeg libraries through pkg-config and bindgen.
Install Clang, pkg-config, FFmpeg, and development packages for libavcodec,
libavformat, libavutil, and libswscale. `cuda-filters` additionally needs
libavfilter development files; `ffmpeg-audio` needs libswresample development
files. The `ffmpeg` executable generates test fixtures;
set `MEDIA_REQUIRE_FFMPEG_TESTS=1` to make missing fixture support an error.

```sh
MEDIA_REQUIRE_FFMPEG_TESTS=1 cargo test --workspace --all-features --locked
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
RUSTDOCFLAGS='-D warnings' cargo doc --workspace --all-features --no-deps --locked
cargo fmt --all --check
```

On Homebrew installations, ensure `/opt/homebrew/bin` is on PATH and use
`PKG_CONFIG_PATH=/opt/homebrew/lib/pkgconfig` when discovery needs it.
CUDA builds load the driver and NVRTC dynamically; compiling does not require a
GPU. Execution needs compatible NVIDIA driver libraries, CUDA 12.8 NVRTC,
and, for video, libnvcuvid/libnvidia-encode exposure. GPU tests are ignored during
ordinary CPU CI and must be explicitly invoked on real hardware; explicit runs
fail when hardware is unavailable.

## Features

| Facade feature | Effect |
| --- | --- |
| `native` (default) | Rust image codecs, ICC transforms, animations/pages, audio decode/resampling |
| `ffmpeg` | Linked FFmpeg demux/mux, CPU video, packets and software transcode |
| `ffmpeg-audio` | Native audio processing with FFmpeg demux/decode for edit-aware AAC and container timing |
| `cuda` | Dynamic CUDA allocation, events, preprocessing; enables NVDEC/NVENC when FFmpeg is also selected |
| `cuda-filters` | FFmpeg/CUDA plus native `scale_cuda` component resizing when the linked library supports it |
| `interop` | Checked host tensors, image views, DLPack; CUDA exports when CUDA is also selected |
| `icc` | Interop plus explicit ICC-to-sRGB tensor conversion and gamut policy |
| `http` | Bounded synchronous HTTP range sources |
| `http-tls` | HTTP sources with rustls HTTPS support |

`media-interop/ndarray` enables ndarray views directly. A minimal facade has no
default features and exposes shared contracts and source/session APIs.

## Contracts and limits

- Time is signed rational; intervals are half-open. Exact selection, navigation,
  and index construction remain distinct. Repeated selections retain identities.
- `clone()` shares immutable storage and metadata. Foreign writable exports
  establish unique ownership, explicitly detach, or fail. Native decoder surfaces
  retain their owning context and producer/consumer completion.
- Session limits account for library-owned payloads and declared working
  reservations. Codec/driver-private heaps and reference pools can remain opaque;
  counters never claim whole-process or whole-driver memory accounting.
- Materialized batches acquire bounded slots without waiting on application-held
  output. Frame/tensor aliases and exported consumers keep those slots alive.
- CPU executions, codec instances and codec threads have shared admission limits.
  CUDA contexts, owned allocations, codec instances and known retained native
  frame bytes share Session budgets; application-held frames keep their charges.
- `write()` drains and finalizes. WAV output stages before replacement; image
  encoding finishes before opening its file destination. Native remux/transcode
  refuses overwrite and reports possible partial output on failure.
- HTTP ranges require authorized HEAD, known length, strong ETags, and exact 206
  responses. They pin versions per cursor; no implicit full download or spool is
  performed. Custom range providers can supply object-store signing semantics.

## Benchmarks

Recorded on November 3, 2026, before the initial release. These are historical
measurements of synthetic workflows; later compatibility fixes have not been
rebenchmarked. Times are milliseconds, lower is better.

### CPU workflows

Apple M4 Pro, macOS 27.2, Rust 1.97.1, FFmpeg 8.1; one codec worker. Medians
of six warm iterations after one initial run, including processing, the consumer,
output retirement and finalization. Direct calls use the same native backends.

| Workflow | media-rs | Direct backend |
| --- | ---: | ---: |
| Image decode, orientation, resize and PNG encode | 4.363 | 4.337 |
| 4 s audio: 48→16 kHz, overlapping windows | 2.242 | 2.017 |
| 6 s H.264 video: sequential RGB decode | 30.252 | 30.124 |
| 6 s H.264/AAC remux and finalize | 6.000 | 6.043 |
| 6 s H.264 resize/transcode, AAC passthrough | 68.006 | 67.328 |

Image input is 768×512; CPU video input is 320×180. These small workstation
measurements show the wrapper cost for those workflows, not a general speedup.

### GPU video sampling

4K H.264/HEVC, six-second 30 fps sources, centered 224×224 RGB f32 NCHW
crops and the same simple CUDA consumer. Sparse requests select four frames
with a repeat; dense requests select sixteen frames. Medians of five steady
iterations; encoded input is already in memory and validation downloads are
outside timing. Each complete comparison contains 384 numerically matched rows.

| GPU / reference | Source | Request | Prefix | Indexed | Reference |
| --- | --- | --- | ---: | ---: | ---: |
| A10G / TorchCodec 0.17 | H264 | sparse | 958.87 | 481.83 | 260.34 |
| A10G / TorchCodec 0.17 | H264 | dense | 1045.91 | 1177.46 | 737.37 |
| A10G / TorchCodec 0.17 | HEVC | sparse | 551.97 | 223.44 | 146.70 |
| A10G / TorchCodec 0.17 | HEVC | dense | 602.23 | 476.26 | 406.01 |
| T4 / DALI 2.3 | H264 | sparse | 889.87 | 503.82 | 1882.68 |
| T4 / DALI 2.3 | H264 | dense | 969.82 | 1133.73 | 1073.71 |
| T4 / DALI 2.3 | HEVC | sparse | 482.34 | 202.66 | 998.19 |
| T4 / DALI 2.3 | HEVC | dense | 564.29 | 421.01 | 527.76 |

The two GPU comparisons use different source color contracts: BT709/Left with
point chroma for TorchCodec, and BT601/Center with bilinear chroma for DALI.
Their numbers cannot rank TorchCodec against DALI. Indexed requests reuse the
timing index and context but open a fresh decoder; TorchCodec retains its decoder,
and DALI retains its Pipeline with private decoder reuse unknown. Initial index
construction is excluded here and can outweigh the benefit for a single request.

TorchCodec leads indexed media-rs in these cases. In the separate DALI task,
indexed sparse reads are faster, while DALI leads dense H.264. There is no
universal winner across codecs, request patterns and reuse scopes.

[Raw measurements and recorded environments](benches/measurements/2026-11-03/metadata.json),
[CPU methodology](benches/README.md) and
[GPU methodology and reproduction](benches/external-comparison.md) include the
1080p rows, initial/fresh timings, decoded work, validation and memory limits.

## Documentation and contributing

- [Runnable examples](crates/media/examples/README.md)
- [Testing](tests/README.md) and [benchmarks](benches/README.md)
- [Contributing](CONTRIBUTING.md)

Build the API documentation locally with `cargo doc -p media --no-deps --open`.
Issues and pull requests are welcome. For a bug report, include a small example,
the enabled features, and your platform and backend versions.

## License

Media-rs is licensed under the [MIT license](LICENSE). Dependencies keep their
own licenses; see [third-party licensing](THIRD_PARTY.md), especially when
distributing binaries that use FFmpeg or NVIDIA runtimes.
