# Bounded Linux/GPU validation

The Modal harness builds native dependencies and all test executables on CPU
capacity before calling one short GPU Function. It creates no deployed service,
persistent Volume, or warm container. GPU allocation is limited to one container,
600 seconds, retries disabled, and a two-second scale-down window. Results and
logs are written to ignored `artifacts/gpu/`. Running the harness uses your Modal
account and incurs the provider's compute charges.

## Run an immutable source snapshot

Authenticate Modal through its normal CLI. Existing local credentials are used;
credentials, target artifacts and unrelated workspace files are never uploaded.

```sh
python3 infra/gpu/snapshot.py
# Use the printed absolute path in the next command.
MEDIA_GPU_SOURCE_SNAPSHOT=/absolute/path/to/snapshot modal run -m infra.gpu.modal_validation::main
```

Cargo manifests/lock, crate source and allowlisted `infra/bench` scripts are
copied into content-addressed snapshots; Git metadata and account configuration
are excluded. Each result records that source digest. The image uses CUDA12.8.1,
Ubuntu24.04, Rust1.88 and distribution FFmpeg development libraries. A separate
manifest-keyed CPU layer compiles dependencies using empty workspace libraries;
real source overlays those cache scaffolds before any correctness target builds.
Release dependencies have a separate manifest-keyed CPU layer before real source
overlays, so source-only edits reuse optimized third-party builds too. Correctness
and benchmark apps are separate: invoking a test never builds an unused benchmark
or downloads its framework wheels. These cache layers establish no behavior.
The actual test executables are built
from the recorded source snapshot before GPU allocation.

The Function allows L4/T4/A10 placement. Tests execute actual NVRTC kernels,
NVDEC, native NVENC/scale_cuda, cross-stream consumers and legacy/versioned DLPack.
Explicit hardware invocation fails when required support is missing. It never
turns unavailable hardware into a passing skip. T4 cannot establish AV1 behavior.

Logs and timestamped/hash-keyed structured results go to ignored artifacts/gpu.
The latest result is also written as modal-validation.json. Preserve failure
logs; do not replace proof for an old snapshot with claims about current source.
Observe the existing CLI process through completion, then verify its app is
stopped. Do not restart a still-running job because observation timed out.

## CPU-only Linux regression

Version-specific demux/mux behavior can be isolated without allocating a GPU:

```sh
MEDIA_GPU_SOURCE_SNAPSHOT=/absolute/path/to/snapshot modal run -m infra.gpu.modal_validation::cpu
```

The CPU Function executes the header-only AAC copy regression in the same image.
It has no GPU configuration. The container's CUDA base-image warning about an
absent NVIDIA driver is expected in this lane.

Run the complete Linux/Rust 1.88 workspace without allocating a GPU:

```sh
MEDIA_GPU_SOURCE_SNAPSHOT=/absolute/path/to/snapshot modal run -m infra.gpu.modal_validation::workspace
```

Run the separate CPU-only AddressSanitizer app against the same frozen source:

```sh
MEDIA_GPU_SOURCE_SNAPSHOT=/absolute/path/to/snapshot modal run -m infra.gpu.modal_sanitizer::main
```

It pins nightly-2025-12-05, instruments Rust std/crates/dependencies, enables
leak detection, and runs the ordinary adoption matrix plus bounded media/cache
mutations. System FFmpeg shared libraries are not themselves instrumented.
The CPU function is limited to 600 seconds with one container and no retries.
The complete hardware suite is run through `::main`; `::adoption` runs
only the adoption GPU cases. Each run records the source snapshot and outcomes
in its local result files.

## GPU workflow measurements

```sh
MEDIA_GPU_SOURCE_SNAPSHOT=/absolute/path/to/snapshot modal run -m infra.gpu.modal_benchmark::main
```

This lane generates its synthetic fixture and compiles the release executable
in CPU image layers, then runs seven bounded iterations on one GPU. It compares
the facade with direct calls into the same native backends and exact context.
Sparse/dense timestamp requests, native/DLPack exports and an independent CUDA
consumer must agree before timing rows are emitted. Validation downloads occur
after the GPU stage and have separate byte/time counters. Source, fixture,
environment, initial setup and raw measurement artifacts are exported to
`artifacts/bench/modal-gpu-SOURCEHASH-RUNID/`. See [measurement semantics](../../benches/README.md).

## External framework environment

The external framework environment can be built/import-checked on CPU first:

```sh
modal run -m infra.gpu.modal_frameworks::main
```

It pins Python 3.12, Torch/TorchCodec CUDA wheels, DALI and NumPy. This check
does not establish GPU availability or codec support. The separate comparison
app allocates a GPU only after its release bridge and synthetic fixtures build
on CPU. Package, fixture and exact source versions accompany the result. See
the [comparison methodology](../../benches/README.md).

The matched comparison currently requests one T4 with 2 CPU cores and 16 GiB
host memory. Use the paired profile/quality/framework flags in the methodology;
the unqualified default retains the original color stress comparison. Its work
deadline is at most 1,000 seconds and Function timeout 1,200 seconds, separate
from the correctness Function's 600-second bound. It creates no Volume/service.
Use `modal run --detach -m infra.gpu.modal_comparison::main ...` so a local
disconnect does not cancel the last submitted Function. A submission JSON records
the FunctionCall ID before waiting, allowing result recovery without duplicate
execution. Export results and verify the app is stopped. The successful detached
T4 run stopped automatically when its entrypoint completed.

## Local or alternative provider

GPU tests can also run on a compatible Linux machine:

```sh
cargo test -p media-backend-cuda --test hardware -- --ignored --test-threads=1
cargo test -p media-backend-ffmpeg --features cuda --test gpu_decode -- --ignored --test-threads=1
cargo test -p media-interop --features cuda --test cuda_dlpack -- --ignored --test-threads=1
cargo test -p media-backend-ffmpeg --features cuda-filters --test gpu_encode -- --ignored --test-threads=1
cargo test -p media --no-default-features --features ffmpeg,cuda-filters,interop --test cuda_video -- --ignored --test-threads=1
cargo test -p media --all-features --test cuda_session -- --ignored --test-threads=1
cargo test -p media --all-features --test cuda_clips -- --ignored --test-threads=1
cargo test -p media --all-features --test indexed_video -- --ignored --test-threads=1
cargo test -p media --all-features --test adoption_workflows indexed_cuda -- --ignored --test-threads=1
```

Install Clang, pkg-config, FFmpeg and libavcodec/libavformat/libavutil/libswscale
SDKs; the optional filter path also needs libavfilter, and FFmpeg audio needs
libswresample development files. Execution needs the
host libcuda driver, CUDA12.8 NVRTC/builtins, libnvcuvid and libnvidia-encode.
The linked FFmpeg must support the requested hardware codec/filter. In NVIDIA
Container Toolkit environments expose compute,video,utility driver capabilities;
installing a toolkit cannot supply missing host video-driver libraries. Do not
replace the cloud host driver inside the container.

For longer interactive sessions, retain Cargo/toolchain/fixture caches on an
explicit compatible volume and keep compiler intermediates on local scratch
when network metadata latency is high. Storage compatibility and GPU stock are
provider/location-specific. Export source and results outside spot instances.
The short Modal test workflow uses cached images and no separately billed
persistent storage.
