# Complete CPU workflow benchmarks

The `workflow_bench` example measures complete blocking workflows, including
source construction/opening, probing, decoding, processing, consumer hashing,
output retirement, and explicit encoder/mux finalization. It compares the facade
with direct calls into **the same linked native backends**, using the same
filters, formats, limits, stream selection, codec parameters and one codec worker.
It does not compare against an FFmpeg subprocess implementation.

```sh
python3 infra/bench/generate_fixtures.py
PATH=/opt/homebrew/bin:$PATH PKG_CONFIG_PATH=/opt/homebrew/lib/pkgconfig \
  cargo build --release -p media --example workflow_bench --features native,ffmpeg,interop
target/release/examples/workflow_bench \
  artifacts/bench/fixtures artifacts/bench/local-run 7 all \
  >artifacts/bench/local-run.stdout 2>artifacts/bench/local-run.stderr
```

On Linux, use the installed FFmpeg SDK's pkg-config path. `--ffmpeg PATH` selects
the executable used **only to generate the fixture**. The output directory must
not already contain `raw.csv`; a run never overwrites earlier measurements.
The final argument can select any individual mode instead of `all`.

The generator writes a 768×512 RGB PNG with EXIF orientation 6, a four-second
mono PCM16 WAV at 48 kHz, and a six-second CFR H.264/AAC MP4 at 320×180/30 fps.
The AAC track starts 250 ms later than video (before codec priming).
Video samples are explicitly converted from the lavfi BT.601 interpretation to
BT.709 limited range before encoding. `manifest.json` records SHA-256 digests,
sizes, generator parameters and tool versions. Exact encoded bytes may differ
across codec/zlib versions; the manifest identifies the files actually measured.

| Mode | Work performed | Validation |
|---|---|---|
| `image` | Decode, apply orientation, Lanczos3 inside 320×320, PNG encode into a finalized memory sink | Exact encoded-byte/diagnostics digest, 213×320 geometry and output count |
| `audio` | Continuous mono 48→16 kHz FFT resampling, 320-sample windows with 160-sample hop; chunk hint 1024 | Exact sample/timing digest, 399 windows and matching decoded-packet counts; incomplete tail included in digest |
| `video` | Sequential RGB8 decode to drained EOF | Exact pixels/source timing digest and 180 decoded frames |
| `sparse` | Displayed-at requests at 0.25, 2.75, 5.25, 2.75 seconds | Caller-ordered output validated against independently decoded real display intervals |
| `dense` | 60 displayed-at requests from 0 through 5.9 seconds | Same independent correctness oracle |
| `remux` | Explicitly copy both compressed tracks, finalize a new MP4 | Packet payload, codec configuration, PTS/DTS/duration and side-data digest |
| `transcode` | H.264 software resize to 160×90, bilinear stored YUV, 500 kb/s, veryfast, B=2/GOP=30, compressed AAC copy | Same packet/timing digest; 180 output/decoded frames and nonempty audio |
| `slowconsumer` | Sequential decode while retaining at most eight frames and sleeping 1 ms per output | Exact video digest/count; release all retained frames before stopping the clock |

Sparse/dense rows have **no direct performance comparison**: the independent
reference scans to EOF and does not implement the facade's candidate scheduler.
It runs after the first measured call, outside timing, and validates correctness
without warming that call first. This runner uses default prefix selection,
through the largest requested PTS plus one future frame. Explicit indexed
execution is measured separately in the linked 1080p/4K comparison.

For compared modes, facade/direct order alternates across iterations. Each pair
must have matching digests, output counts and decoded counts before its rows are
written. Digests must also remain stable across iterations. The digest is a
noncryptographic Rust `DefaultHasher` equality check for this toolchain/run, not
a durable cross-toolchain content identifier. Fixture SHA-256 is separate.
Container file bytes are deliberately not compared: incidental container fields
can vary while payload and timing remain equivalent. The packet validation read
after a finalized remux/transcode is outside timing and outside input-byte counts.

`initial_use` means the first measured call for a mode/variant in this process.
It is **not a cold OS-cache or fresh-process measurement**: earlier modes may
have initialized the shared backend, the first member of a pair may warm it for
the second, and fixture generation already touched the files. Warm medians should
exclude iteration zero. No reusable decoder/plan is retained between iterations.
Thus warm rows measure a fresh request with warm process/filesystem state.

`first_result_ns` is the first caller-visible frame/window, or the completed batch
for sampling. The image API exposes a completed memory file; remux/transcode
expose a finalized file, so their first-result latency is completion-bound rather
than the internal first encoded packet. `end_to_end_ns` includes digest consumers
and dropping their outputs/decoders/resamplers. Files are kept for review after
timing; removing files is not part of a conversion workflow.

`source_payload_bytes_read` counts bytes actually returned by every underlying
file read, including rereads and facade probing. A shared counting file adapter
is used on both paths; immutable fixture identity and known length are supplied
to that adapter. Read-system-call/seek counts, RSS, native codec/FFT scratch,
allocator internals, queue waits, GPU memory and transfer counts are **unmeasured**.
`peak_host_accounted_bytes` is the facade's reservation high-water mark, including
conservative work reservations, **not observed allocated RSS**. Direct native
peaks and unaccounted CPU codec/mux peaks are `unmeasured`, never inferred as zero.
The slow-consumer mode is a synchronous pull reader with retained outputs; it
does not model a concurrent producer queue. Its fixed sleep dominates throughput.

The image mode explicitly allows and compares reported stale EXIF/color-declaration removal and uses the legacy stored
component resize contract. It does not benchmark ICC conversion, linear-light
resize, or alpha semantics. The source sRGB declaration currently has no ICC
profile; the encoder reports `ColorDescription` loss rather than silently
claiming preservation. This is not a metadata-preservation benchmark.

Raw CSV, environment metadata, copied fixture manifests, finalized outputs and
logs are generated under ignored `artifacts/bench/`. Summarize a run with
`python3 infra/bench/summarize.py RUN_DIRECTORY`.

## Required CUDA complete workflows

The `gpu_workflow_bench` example compares the facade and the same direct native
NVDEC/CUDA backend, using one Session canonical CUDA context/allocator and equal
decoder limits. Generate the small fixture and compile in a CPU image before
allocating GPU time:

```sh
python3 infra/bench/generate_gpu_fixture.py --ffmpeg /usr/bin/ffmpeg
cargo build --release -p media --features ffmpeg,cuda,interop \
  --example gpu_workflow_bench --locked
```

On a machine with the linked FFmpeg NVDEC support and the existing CUDA 12.8+
driver/NVRTC runtime, run:

```sh
target/release/examples/gpu_workflow_bench \
  artifacts/bench/gpu-fixtures/gpu6s.mp4 artifacts/bench/gpu-RUN 7
```

The runner requires release mode and does not provision hardware. The generated
fixture is six seconds of CFR30 H.264 at 320×180, with real B frames and an
explicit BT.601-to-BT.709 YUV conversion before encoding, limited range and left
chroma siting. Its manifest records the exact command, FFmpeg version and SHA-256.
There are four requested times with one repetition in unsorted order, or sixteen
distinct unsorted times. Both paths use DisplayedAt with explicit NextTimestamp
inference and leave unverified native durations Unknown. Both produce contiguous
224×224 NCHW RGB f32 using the same bilinear/converted-RGB kernel and normalization.
Retained NVDEC output leases are bounded to twenty; this accommodates sixteen
selected frames plus previous/current candidates without retaining every decoded
surface. Global ordinals come from decoding the actual stream beginning. The
direct scheduler keeps only its local pair and selected candidates. Repetitions
share source leases and are restored to caller order in the final tensor.

Each path exports through both the native descriptor and versioned DLPack APIs
with copies forbidden. An independently compiled CUDA consumer reads the exported
address and writes its own allocation: `out[i] = in[i] * 0.5 + 0.25`. This is a
declared elementwise consumer workload, not a model inference benchmark. Its
descriptor checks, exactly-once matching DLPack deleter and same-stream retirement
are in one documented unsafe boundary. The consumer's native CUDA output allocation
uses the same actual context but is outside the library allocator counter; its
known byte size is reported separately.

`setup.csv` separates context bootstrap and consumer NVRTC/module setup. The first
producer row is labeled as including first preprocessing NVRTC compilation; other
iteration-zero rows share that module cache. The default seven iterations provide
six steady samples per path/mode/export, alternating facade/direct execution order.
No operating-system file-cache coldness is claimed. Both kernels and modules remain
loaded through their queued work.

`raw.csv` records wall time from source open through the tensor producer event,
independent consumer completion/deleter, and input surface/tensor retirement. These
waits target relevant events, without a device-wide synchronization. Only afterward
does validation download the independent consumer output. Its exact transfer bytes
and elapsed download time are separate columns. Caller order, exact source ordinals,
actual decoded-frame counts and every consumer-result float bit must match within
each pair before a timing row is emitted. A stable FNV-1a digest is archived after
that full comparison. Expected fixture ordinals and normalized output bounds are
also checked; these comparisons do not constitute an independent reference for the
shared CUDA preprocessing algorithm.

Counters distinguish actual encoded payload bytes, known retained native leases,
and context-owned allocation high-water marks. Context/Session peaks are cumulative
within the run; direct native lease peaks cover one execution. Direct host heap,
opaque codec/driver memory and driver-internal transfers remain unmeasured. Zero
decoded host crossings describes the exercised library GPU stage; explicit
validation downloads are excluded and separately counted. Host validation vectors
and the independent consumer's output are not silently charged to Session budgets.
The output folder preserves raw/setup CSVs, environment including `nvidia-smi`,
the compiled runner source/Cargo.lock and an actual fixture copy/manifest. Interpret
steady distributions together with decoded work and these measurement boundaries;
the runner establishes no universal facade speedup or overhead percentage.

## Recorded measurements

The [README benchmark tables](../README.md#benchmarks) summarize recorded CPU
and GPU runs. Their [raw rows and environments](measurements/2026-11-03/metadata.json)
are committed separately from generated media and run directories. These are
historical measurements, not a performance guarantee for later source changes.

The [matched 1080p/4K comparison](external-comparison.md) describes the separate
FFmpeg/TorchCodec/DALI workloads, color contracts, reuse scopes and reproduction.
