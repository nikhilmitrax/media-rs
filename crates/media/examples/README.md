# Runnable examples

Run these from the workspace root with the matching features and real input.
The linked FFmpeg must support each requested codec/filter. Hardware examples
require actual NVIDIA video/compute libraries and return errors without them.

```sh
cargo run -p media --example cpu_image -- INPUT.jpg OUTPUT.webp
cargo run -p media --example audio_windows -- INPUT.wav
cargo run -p media --features ffmpeg,cuda,interop --example gpu_video_tensor -- INPUT.mp4 0.9 0.1 0.9 0.2
cargo run -p media --features cuda-filters --example transcode_passthrough -- INPUT.mp4 NEW_OUTPUT.mp4 1280 720
```

The image example applies orientation, uses stored-component resize, explicitly
permits reported metadata loss, and finalizes output. Use ColorResizeSpec and
ColorConvertSpec for color-managed/linear-light operations; those are separate
from stored-component geometry.

The audio example decodes/downmixes/resamples continuously before complete
320-sample windows with a160-sample hop. The source must establish speaker
positions for named downmixing. Its incomplete tail count is reported.

The GPU tensor example requires NVDEC, samples unsorted/repeated timestamps,
produces one contiguous NCHW f32 device allocation and prepares an owned DLPack
export on a compatible consumer stream. Source matrix/range/chroma/primaries/
transfer must be established or explicitly interpreted through the request API.
Normalization is not color-space conversion; no host download is performed.
A foreign consumer taking into_raw must call the matching deleter exactly once
after its final submission on that retained stream.

The native transcode example requires NVDEC, scale_cuda and NVENC. It resizes YUV
components inside bounds, copies selected compressed audio, drains delayed frames,
and finalizes a new output file. It does not silently insert a CPU fallback.
Available codec/profile/format combinations depend on the linked FFmpeg build
and GPU. See the [README](../../../README.md#contracts-and-limits) and
[GPU tests](../../../tests/README.md#gpu-tests).

workflow_bench is a release-mode measurement runner with generated fixtures,
direct-backend comparisons and output validation; see
[benchmark instructions](../../../benches/README.md).

`gpu_workflow_bench` measures required NVDEC → 224×224 NCHW f32 preprocessing →
an independent elementwise CUDA consumer through native and versioned DLPack
exports. It compares the facade with the same bounded native backend, validates
caller order and every output float, and keeps validation downloads outside its
GPU timing stage. Build and generate its synthetic fixture in a CPU image before
starting a GPU:

```sh
python3 infra/bench/generate_gpu_fixture.py --ffmpeg /usr/bin/ffmpeg
cargo build --release -p media --features ffmpeg,cuda,interop --example gpu_workflow_bench --locked
target/release/examples/gpu_workflow_bench artifacts/bench/gpu-fixtures/gpu6s.mp4 artifacts/bench/gpu-RUN 7
```

Seven iterations yield an initial use plus six steady measurements for sparse
repeated and dense sixteen-frame sampling, alternating comparison order. It
requires CUDA 12.8+ driver/NVRTC and linked FFmpeg NVDEC support. It provisions
no resources; raw timing, setup, environment, source and fixture provenance are
archived in the output directory. See the benchmark instructions for timing,
allocation-counter and synthetic-consumer limitations.

`matched_bench_bridge` is a benchmark-only cdylib, built with `--features ffmpeg,cuda,interop`. It exports native CPU/CUDA center crops through a narrow, single-use DLPack C ABI into the common Python comparator at `infra/bench/external_workflow_bench.py`. Its retained consumer stream and exactly-once descriptor ownership are documented in `benches/external-comparison.md`. It is not a production Python binding. The benchmark covers matching 1080p/4K H264/HEVC inputs, actual facade prefix/indexed controls, direct native FFmpeg and optional live facade cursors alongside TorchCodec/DALI.
