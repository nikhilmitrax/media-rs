# Validation

Executable Rust integration tests live in each owning crate's tests directory.
The root is a virtual Cargo workspace. Fixtures are generated or committed with
provenance; no GPU is needed for ordinary native/contract checks.

| Area | Exercised contracts |
| --- | --- |
| Time/sampling | Rational overflow/rounding, half-open grids, VFR 0/40/100/140ms selection, ambiguity/gaps, deterministic ties, repeated caller order |
| Storage/policies | Pitched/opaque layouts, immutable aliases, explicit copies, independent decode/residency/transfer policies |
| Images | Native precision, single encoded read/inspection, source-version checks, all orientations, crop/resize/pad, exact/quantized pixel conversion, ICC/color/alpha/gamut semantics, metadata loss, animation disposal/blend/loops, APNG/GIF/TIFF encode |
| Audio | Continuous/chunked resampling equivalence, sample grids, quantized timestamps vs true gaps, windows, typed/dithered PCM, final drain, transactional output, optional FFmpeg AAC skip/span trim |
| Video/container | Generated H264/B-frames/VFR, exact-tick indexed seeks and ordinals, explicit index bounds/persistence/version checks, incremental decoder/writer drain, source/asset timing, duration provenance, native pixels, single-input remux, transcode/no-overwrite/finalization |
| Clips | Nonzero common origins and stream offsets, exact track proof, fractional adjacent clips, coverage and separate boundary masks, resource retirement/cancellation |
| Sources | Independent cursors, real capabilities, stale source/index rejection, concurrent versioned ranges, bounded loopback HTTP with exact ETag/206 semantics |
| Execution | Shared host/batch/CPU/codec/thread admission, CUDA context/codec/native ownership, nonblocking backpressure, cancellation, deterministic dataset keys/shards/tails, reversed worker completion |
| Interop | Physical shape/strides/alignment, image/ndarray views, real legacy/versioned DLPack C ABI, optional ICC-to-sRGB conversion |

## CPU checks

Default native builds do not link FFmpeg. CI is configured for stable
Linux/macOS/Windows and the Rust 1.88 floor. Recorded local runs cover macOS and
Linux/Rust 1.88. FFmpeg checks require Clang/pkg-config, native development
libraries, and the ffmpeg fixture generator.

```sh
cargo test --workspace --exclude media-backend-ffmpeg --locked
MEDIA_REQUIRE_FFMPEG_TESTS=1 cargo test --workspace --all-features --locked
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
RUSTDOCFLAGS='-D warnings' cargo doc --workspace --all-features --no-deps --locked
```

Loopback HTTP tests bind a local listener and use no external service. Restricted
local environments may require permission for that bind. The CI image allows it.
The FFmpeg Linux job installs libavfilter SDKs for CUDA filters and libswresample
SDKs for the optional audio path.

## GPU tests

GPU suites are explicitly ignored in ordinary CPU CI. Invoking them with
--ignored fails when required hardware is missing; it never reports a passing
hardware skip. The Modal harness builds on CPU first and records source,
GPU/driver identity, native libraries, results and durations. Suites cover NVRTC
kernels, pitched NV12/P010 layouts, indexed VFR seeking, NVDEC/NVENC, DLPack
ownership, cross-stream consumers and full facade workflows. See the
[GPU instructions](../infra/gpu/README.md) to run them and retain results locally.

On a machine with the NVIDIA runtime, run the CUDA backend's hardware suite with:

```sh
cargo test -p media-backend-cuda --test hardware --locked -- --ignored --test-threads=1
```

Other hardware suites live in the FFmpeg, interop and facade crates. Select the
owning package, features and test target when invoking them. A workspace-wide
`--ignored` run also includes the long-running and external-corpus tests below.

Passing a compile check, nvidia-smi or an advertised FFmpeg accelerator never
establishes a hardware workflow. Driver-private copies/allocations are not inferred
from library counters. Performance measurements are separate from correctness;
see [benchmark methodology](../benches/README.md).

## Adoption scenarios

`media/tests/adoption_workflows.rs` exercises common audio codecs/scalars,
MP4/MOV/WebM/transport streams, VFR/open GOPs, rotation, independent CLI pixel and
timing comparisons, gapless extent, decoder priming, explicit timestamp recovery,
wrong media and resource retirement. `media-core/tests/adoption_properties.rs`
adds 20,000 deterministic time/layout cases. These run in ordinary feature CI.

The longer checks are explicit and bounded:

```sh
MEDIA_REQUIRE_FFMPEG_TESTS=1 cargo test -p media --release --features ffmpeg-audio \
  --test adoption_workflows -- --ignored --nocapture --test-threads=1
```

Without CUDA enabled this runs the mixed-worker soak, an encoded one-hour CFR30
recording with index persistence/reuse, and media/index-cache mutations (including
recomputed cache checksums). `MEDIA_ADOPTION_ITERATIONS` controls soak work
(default 2,400, maximum 100,000); `MEDIA_ADOPTION_WORKERS` chooses 1–32 workers.
`MEDIA_ADOPTION_MUTATIONS` controls media mutation count (default 1,200).
The fixed cache mutation check uses another 1,200 cases.

The real-media corpus is downloaded only on explicit request, outside Git:

```sh
python3 infra/bench/adoption_corpus.py --fetch
MEDIA_ADOPTION_CORPUS="$PWD/artifacts/adoption-20261103/corpus" \
  cargo test -p media --release --features ffmpeg-audio --test adoption_corpus \
  -- --ignored --nocapture
```

It compares every decoded pixel/sample with the independent CLI, checks indexed
frame identities, and demonstrates explicit recovery for the camera footage's
verified duplicate audio timestamps. Source URLs, licenses and expected checksums
are pinned in the [corpus manifest](../infra/bench/adoption-corpus.json).
