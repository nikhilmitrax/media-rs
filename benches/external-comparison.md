# Matched 1080p/4K workflow comparison

This runner compares the facade, direct native FFmpeg, TorchCodec and NVIDIA DALI in the same process on the same GPU. It also runs CPU facade/direct FFmpeg comparisons. Run results determine whether a comparison is numerically matched; having the same output dimensions is insufficient. The runner never falls back to host decoding for a CUDA case.

Recorded CPU/GPU rows and environments are committed in the
[measurement metadata](measurements/2026-11-03/metadata.json); the
[README](../README.md#benchmarks) summarizes selected steady results.

`media-prefix` uses
`.sample(...).selection(DisplayedAt).duration_inference(NextTimestamp).read()`
as the prefix-selection control for indexed seeking.

## Quality-qualified paired tasks

The default BT709/Left stress task can expose reconstruction differences. Quality is explicit in each CSV row, environment, setup record and diagnostic archive; results are grouped by profile and quality rather than presented as a universal ranking.

| Pair | Source profile | Native preprocessing | External path |
|---|---|---|---|
| TorchCodec | `bt709-left` | `--quality point-chroma`: RGB resize remains bilinear; only UV reconstruction uses `ChromaInterpolation::NearestTiesEarlier` | NVDEC RGB8 block-repeat, device crop and cropped float normalization |
| DALI | `bt601-center` | `--quality native-bilinear`: unchanged bilinear RGB/chroma behavior with genuinely centered source samples | Fixed limited-range BT601 converter with centered linear chroma, fused crop/layout/normalization |
| Fidelity stress control | `bt709-left` | `--quality native-bilinear`: original default | External differences remain measured and excluded when unmatched |

`--profile` must match the fixture manifest. The bridge receives both profile and quality explicitly and checks the actual decoded primaries/transfer/matrix/range/siting. It never relabels Left as Center. The default production reconstruction policy remains unchanged; the point-chroma mode is opt-in and separately tested.

The independently named BT601/Center corpus uses native RGB `testsrc` → explicit limited-range BT601 YUV420 downsampling with centered horizontal/vertical chroma positions, followed by the same frame barcode and explicit SMPTE170M primaries/transfer/matrix and Center tags. It differs from the retained original `testsrc2` corpus; comparisons across these profiles would conflate source complexity and reconstruction quality. Its manifest and suffixed filenames are separate, and center-crop chroma variation is checked to ensure actual colorful content. TorchCodec's RGB8 truncation versus native float conversion remains a disclosed intermediate difference and must still pass the unchanged final numeric tolerance. No successful paired result is assumed before GPU validation.

## Common task

Both profiles contain six-second, 30-fps, 1920×1080 and 3840×2160 H264/HEVC files, with real B-frames and 30-frame GOPs. The original profile converts its generated testsrc2 pattern from BT601 to BT709 before encoding and labels BT709 primaries/transfer/matrix, limited range, left chroma. The independently generated BT601/Center profile follows the RGB testsrc path described above. Both colorful and neutral-chroma fixtures are retained. Generation ends with explicit `setparams` and verifies all five decoded/container color fields; encoder CLI flags alone proved insufficient with the initial FFmpeg8 filter chain. `--patch-metadata` repairs existing generated files through `h264_metadata`/`hevc_metadata` bitstream filters and stream copy, preserving every decoded YUV frame/timestamp hash and sampled barcode. Original/new file hashes and the exact repair command are stored in the manifest; no lossy re-encode is used for repair. Every frame includes an eight-bit ordinal barcode inside the common crop; an independent software decode checks IDs 0, 1, 63, 141 and 179 during generation.

Requests are unsorted and can repeat: sparse `[141,9,141,63]`, or dense `[150,140,...,0]`. All producers select exactly these presentation frames, then take the centered 224×224 crop with **identity resize**, produce RGB float32 NCHW in [0,1], project values onto the eight-bit grid, and execute the independent common consumer `output = input * 0.5 + 0.25`. Grid projection and the consumer use the same Torch operations for every CUDA producer. CPU native comparisons use identical NumPy operations.

Identity resize avoids incorrectly equating different reconstruction filters: TorchCodec's resize includes bilinear antialiasing, while the native fused kernel has another sampling contract. This task measures selection, decode, color conversion, crop, layout/dtype conversion and a simple consumer; it does not represent an arbitrary model or matched downsampling benchmark.

TorchCodec CPU uses its native `CenterCrop((224,224))` transform with `output_dtype=torch.uint8`, followed by float32/255 conversion of only the 224×224 decoder output. CUDA uses standard NVDEC `output_dtype=torch.uint8`, an exact device-side tensor crop view, and float32/255 conversion of only that crop. The common task projects onto an eight-bit grid, so it does not require a full-frame float intermediate. TorchCodec0.17 explicitly rejects decoder transforms for CUDA; the benchmark does not claim a fused CUDA crop for it. The CPU intermediate matches native RGB24; requesting CPU float directly selects TorchCodec RGB48 and introduces a different reconstruction/quantization path. DALI uses `crop_mirror_normalize(crop=[224,224], crop_pos_x=0.5, crop_pos_y=0.5, rounding="truncate", output_layout="FCHW", dtype=FLOAT, mean=[0], std=[255])`. All fixture dimensions give exact integer crop coordinates, so no half-pixel rounding difference is introduced. The supported operator configuration is used for each device. TorchCodec CUDA explicitly produces the full frame before its crop; other private full-frame or fused work remains unmeasured. The setup record includes these configurations.

The native CUDA path converts NV12 to RGB using its declared matrix/range/chroma semantics. TorchCodec CUDA produces RGB8 and then crops/normalizes on the device; its CPU configuration produces cropped RGB8 before normalization. DALI converts decoder RGB8 through its fused crop/normalize/layout operator. The task retains the declared source signal encoding: BT709 for the original profile and SMPTE170M for the BT601/Center profile. It does not apply an EOTF or convert to linear light. A grid projection cannot erase matrix or chroma reconstruction differences. Validation downloads the consumer output only **after** the timed GPU stage, checks every barcode and compares every component with the same fixture's `media-prefix` reference. The default matching limits are maximum RGB error ≤2/255 and mean ≤0.5/255, expressed before the consumer's scaling. Since the declared task projects onto an eight-bit grid, validation recovers integer codes using float64 `rint((consumer-.25)*510)` and applies those unchanged limits in code units. Independently, consumer values must be finite float32 in [.25,.75] and within 1e-4 code units of the nearest code. This bounds float32 roundtrip residue without accepting a true extra code of error. Raw floating errors and grid residuals remain recorded. Larger differences set `matched=false`; those rows remain visible and must be excluded from speed comparisons. Gray results establish only the neutral-chroma case. A failed colorful comparison must not be generalized into color equivalence.

For CFR fixtures, frame indices and exact requests `index/30` are equivalent, and actual frame IDs independently verify selection. Native adapters use `FrameDurationPolicy::Unknown`, with explicit `NextTimestamp` inference in facade sampling. No native packet duration hint is asserted to be an established display interval. This benchmark does not establish general VFR equivalence for frameworks with different frame-index policies.

The Linux CPU preflight initially tested direct TorchCodec float output and found correct frame IDs but a maximum difference of 3/255 (mean approximately 0.373/255). TorchCodec0.17 `CpuDeviceInterface.cpp` selects RGB48 for float output, and `Frame.cpp` normalizes that uint16 intermediate by 65535. Decoder-side CenterCrop uses its filtergraph path. A local independent FFmpeg reference reproduces RGB48-versus-RGB24 projection differences while explicit BT709 leaves them unchanged; substituting BT601 causes much larger color errors. CPU RGB8 with the same decoder-side crop therefore preserves the task's RGB24 intermediate without relaxing the numeric threshold. GPU's separately implemented RGB8 conversion kernel must pass its own validation; its CUDA crop is a tensor view because the pinned native stream API only accepts transforms on CPU. The [native transform device check](https://github.com/meta-pytorch/torchcodec/blob/v0.17.0/src/torchcodec/_core/SingleStreamDecoder.cpp) establishes that limitation. See the pinned [CPU implementation](https://github.com/meta-pytorch/torchcodec/blob/v0.17.0/src/torchcodec/_core/CpuDeviceInterface.cpp), [dtype conversion](https://github.com/meta-pytorch/torchcodec/blob/v0.17.0/src/torchcodec/_core/Frame.cpp) and [scaler implementation](https://github.com/meta-pytorch/torchcodec/blob/v0.17.0/src/torchcodec/_core/SwScale.cpp).

## Reuse and setup scopes

All variants use preloaded encoded bytes. File reading, handle construction and visible index building are recorded separately in `setup.json`; this is not a cold-disk or fresh-process benchmark. The first execution is `initial-use`, including lazy CUDA module/kernel setup. Later executions are `steady`, and variant order rotates each iteration. Initial process/framework imports, CUDA initialization and per-object setup are retained in `environment.json`/`setup.json`, outside steady timing.

| Variant | Fresh scope | Warm scope |
|---|---|---|
| `media-prefix` | Fresh decoder each request; context and encoded input retained | Same behavior; retained context, no reusable decoder claim |
| `media-indexed` | Full decoded timing index built and charged on every request, followed by local GOP execution | Full index built in setup and retained; each request opens a **fresh decoder** using cached evidence |
| `ffmpeg` / `ffmpeg-seek` | Fresh direct libavformat/libavcodec reader; keyframe seek + decode to exact PTS per unique target | Same live native reader; the same keyframe seeks and exact-PTS decode |
| `ffmpeg-prefix` (optional) | Direct native prefix decode; fresh reader | Same live native reader, rewound to zero, prefix decoded again |
| `media-cursor` (optional) | Direct facade reader prefix decode; fresh reader | Same live facade reader and seeks; explicit true-reader reuse control |
| `torchcodec` | Fresh exact-mode VideoDecoder, NVDEC cache disabled | Same exact-mode VideoDecoder retained, NVDEC cache disabled |
| `dali` | Fresh Pipeline, index enabled, input fed for each request | Same Pipeline retained; internal decoder/index amortization is **unknown** |

Disabling TorchCodec's otherwise-default NVDEC cache makes fresh decoder rows honest. A future recommended-default-cache experiment must carry another label. DALI initialization can defer input-dependent index work until `run()`, so its first execution retains that cost. These are useful reusable-object comparisons, but the objects have different lifetimes; the `reuse` column is mandatory when interpreting results.

Known native limits match across facade and direct paths: 20 retained surfaces, 32 MiB per surface, 512 MiB retained native payload, CUDA owned allocation limit 512 MiB. This accommodates dense16 at 4K; opaque decoder reference pools remain unmeasured. Codec threads and DALI pipeline threads are one. The default direct FFmpeg baseline sorts/deduplicates targets, seeks near every unique timestamp using the native keyframe-seek API, decodes to its exact source PTS, and restores caller order/repeats. It does not use a deliberately prefix-only baseline. Its per-target seeks do not share all preroll within a GOP, unlike the indexed facade schedule, so seek/decode counts remain essential. The direct FFmpeg CUDA baseline uses native FFmpeg libraries and the same native preprocessing/DLPack bridge, **not** a CLI path involving hwdownload.

## Timings, counters and ownership

`ready_ns` runs from request submission through a targeted event confirming GPU producer completion. `consumer_ns` includes the common numeric projection and consumer, ending with completion of that consumer stream. `retire_ns` includes dropping imported input tensors, producer leases, fresh external decoder/pipeline objects, and waiting for native input/output event retirement. `total_ns` is their sum. No explicit whole-device synchronize is used. Validation transfer/digest/frame-ID checks have separate `validation_ns` and `validation_dtoh_bytes` columns. The consumer output remains alive through validation; its later disposal is outside producer retirement.

The benchmark-only C ABI owns the CUDA consumer stream. Python constructs `torch.cuda.ExternalStream` from that retained queue and both consumes the legacy DLPack capsule and submits the consumer on it. The native exporter places readiness dependencies on that exact queue; the native deleter records retirement on it. A raw legacy capsule cannot negotiate another foreign stream. Unconsumed capsules release their original descriptor exactly once; consumed capsules are renamed by Torch and belong exclusively to Torch's deleter. Capsule callbacks and the loaded library remain alive through interpreter lifetime. Error cleanup completes the specific consumer queue before dropping leases. This ABI is not a production binding.

DALI uses its `TensorGPU.__dlpack__` protocol, which receives Torch's actual current stream, and holds the pipeline/output lease through consumer completion. DALI affinity is explicitly disabled (`affine=False`) with one pipeline worker. For sandboxes without full NVML, the runner environment sets the documented `DALI_DISABLE_NVML=1` before import; this value is recorded in environment metadata. This changes management/affinity queries, and retains NVDEC decoding and GPU output validation. See [NVIDIA's environment-variable documentation](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/env_vars.html). TorchCodec's CPU fallback status is checked after decode and forbidden for CUDA; DALI output must identify itself as a CUDA device. Output tensors must reside on device zero.

Counters distinguish actual native decoded frames/read bytes/seeks, cold index work, selected-surface payload peaks, cumulative Session native reservations and CUDA owned allocator peaks. Encoded `source_bytes` measures payload reads including rereads/probes, not physical disk traffic. Torch allocated/reserved peaks exclude native FFmpeg/DALI heaps. Internal framework decoded counts/source-read counts and driver copy/allocation activity are unknown, left blank or identified in environment metadata. `known_host_crossings=0` describes the explicit decoded-pixel application path, not a measurement of private driver behavior. Validation downloads are always counted.

Each run writes raw rows, setup, environment, fixture manifest, runner/bridge sources, Cargo.lock and the bridge library digest to its output directory. Initial-use numeric mismatches also save actual/reference consumer-output arrays, requested/observed frame IDs and interpretation metadata as NPZ, after all timed work and validation transfers. Diagnostics are limited to the first scope per fixture/case/framework, at most 16 files, 8 MiB compressed per file and 64 MiB total; duplicate/limit skips are explicit in mismatch-diagnostics.json. Load with allow_pickle=False. The saved arrays are float32 NCHW consumer values; recover original RGB with `(value-.25)*2`. No extra GPU download is introduced. The Modal runner additionally records the immutable full source snapshot. Only `status=ok,matched=true` steady rows support matched timing comparisons. `validated_matches` and `unvalidated` counts distinguish successful execution from numeric comparison; a run without any validated reference match exits unsuccessfully. Mismatches, unsupported paths and bounded stops remain explicit; results from one GPU and synthetic workload do not imply universal speedups.

## Reproduction

Use Python 3.12, a CUDA13-compatible driver, Rust ≥1.88, and native FFmpeg SDK libraries. The CPU-built framework image pins Torch2.11.0+cu130, TorchCodec0.17.0+cu130, DALI2.3.0 and NumPy2.2.6. Install Torch/TorchCodec from the CUDA130 wheel index, then the other requirements from PyPI:

```sh
python -m pip install --index-url https://download.pytorch.org/whl/cu130 torch==2.11.0 torchcodec==0.17.0
python -m pip install nvidia-dali-cuda120==2.3.0 numpy==2.2.6
python infra/bench/external_workflow_bench.py --check-deps
python infra/bench/generate_external_fixtures.py --output artifacts/bench/external-fixtures
cargo build --release -p media --features ffmpeg,cuda,interop --example matched_bench_bridge --locked
python infra/bench/external_workflow_bench.py \
  --fixtures artifacts/bench/external-fixtures \
  --rust-lib target/release/examples/libmatched_bench_bridge.so \
  --output artifacts/bench/matched-gpu --iterations 6 --max-seconds 480
```

Fixture generation and release compilation must happen on CPU before allocating the GPU. `python infra/bench/external_workflow_bench.py --selfcheck` runs a NumPy-only validator regression: the historical float32 two-code boundary, two-ULP roundoff, a true three-code error, and invalid NaN/off-grid/range outputs. `--verify-only` checks existing fixture IDs without regenerating. Filter a GPU smoke run with `--resolutions 1080p --codecs h264 --patterns color --iterations 2`. A full bounded run retains completed rows if its deadline prevents later cases. Increase `--max-seconds` explicitly up to 1800 when the cloud budget permits. The macOS bridge suffix is `.dylib`.

CPU comparisons can run without Torch installed:

```sh
python infra/bench/external_workflow_bench.py --device cpu \
  --frameworks media-prefix,media-indexed,ffmpeg,media-cursor \
  --fixtures artifacts/bench/external-fixtures \
  --rust-lib target/release/examples/libmatched_bench_bridge.dylib \
  --output artifacts/bench/matched-cpu --iterations 6
```

DALI H264/HEVC CPU decoding is unsupported in DALI2.3; a CPU DALI request records an error rather than substituting a different codec/backend. TorchCodec CPU can be included when its pinned dependencies are available.

Cloud entry points are maintained separately: framework-only CPU import proof is `modal run -m infra.gpu.modal_frameworks::main`; the matched comparison is `modal run -m infra.gpu.modal_comparison::main`. Functional correctness uses `infra.gpu.modal_validation::main`/`::workspace`; the original small-workload GPU benchmark uses `infra.gpu.modal_benchmark::main`.


Generate the second colored corpus on CPU without overwriting the stress corpus:

```sh
python infra/bench/generate_external_fixtures.py --profile bt601-center --patterns color \
  --output artifacts/bench/external-fixtures-bt601-center
```

Run the pairs into separate output directories with the same fresh/warm scopes:

```sh
python infra/bench/external_workflow_bench.py --fixtures artifacts/bench/external-fixtures \
  --profile bt709-left --quality point-chroma \
  --frameworks media-prefix,media-indexed,ffmpeg,torchcodec \
  --rust-lib target/release/examples/libmatched_bench_bridge.so \
  --output artifacts/bench/paired-torchcodec --patterns color --iterations 6
python infra/bench/external_workflow_bench.py --fixtures artifacts/bench/external-fixtures-bt601-center \
  --profile bt601-center --quality native-bilinear \
  --frameworks media-prefix,media-indexed,ffmpeg,dali \
  --rust-lib target/release/examples/libmatched_bench_bridge.so \
  --output artifacts/bench/paired-dali --patterns color --iterations 6
```

The CPU preflight uses `--profile bt709-left --quality native-bilinear` and the original fixture directory independently of the requested GPU pair. The benchmark-only C ABI version is checked before construction so an older runner cannot silently supply the wrong quality/profile argument layout.

## Primary API and dependency references

- [TorchCodec0.17 release](https://github.com/meta-pytorch/torchcodec/releases/tag/v0.17.0), [CUDA installation](https://github.com/meta-pytorch/torchcodec#cuda-support), [VideoDecoder exact seek, device and fallback API](https://meta-pytorch.org/torchcodec/stable/generated/torchcodec.decoders.VideoDecoder.html), [NVDEC backend selection](https://meta-pytorch.org/torchcodec/stable/generated/torchcodec.decoders.set_cuda_backend.html), [NVDEC cache capacity](https://meta-pytorch.org/torchcodec/stable/generated/torchcodec.decoders.set_nvdec_cache_capacity.html), [CenterCrop](https://meta-pytorch.org/torchcodec/stable/generated/torchcodec.transforms.CenterCrop.html).
- [DALI2.3 package](https://pypi.org/project/nvidia-dali-cuda120/2.3.0/), [in-memory video decoder, arbitrary frames, index and codec support](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/operations/nvidia.dali.fn.decoders.video.html), [TensorGPU DLPack protocol](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/data_types.html#nvidia.dali.tensors.TensorGPU.__dlpack__), [fused crop/normalize/layout](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/operations/nvidia.dali.fn.crop_mirror_normalize.html).
- [Torch2.11 package](https://pypi.org/project/torch/2.11.0/), [NumPy2.2.6 package](https://pypi.org/project/numpy/2.2.6/).


## Pinned converter differences and original stress failures

The original L4 colored smoke executed all paths, but its external TorchCodec/DALI outputs did not meet the common numeric threshold. Those rows are excluded from matched timing claims. The following source-level differences are established. The separately declared paired contracts do not establish equivalence for the default709 stress task.

- TorchCodec0.17 at commit `d9dc54238e173000bc3969eb3959777026ff9ae8` [converts each NV12 2×2 block with one UV pair](https://github.com/meta-pytorch/torchcodec/blob/d9dc54238e173000bc3969eb3959777026ff9ae8/src/torchcodec/_core/color_conversion.cu), then clamps and casts to uint8. Its [matrix calculation](https://github.com/meta-pytorch/torchcodec/blob/d9dc54238e173000bc3969eb3959777026ff9ae8/src/torchcodec/_core/color_conversion.cpp) honors BT709. This block-repeat reconstruction differs from native bilinear chroma. For Left siting, native nearest with ties upward also differs at odd horizontal pixels; changing source siting metadata to hide that distinction would be incorrect.
- DALI2.3 at commit `68b40ed8bd2f77985c4cf114e50fcca065cb66d7` [uses the BT601 converter for limited-range video](https://github.com/NVIDIA/DALI/blob/68b40ed8bd2f77985c4cf114e50fcca065cb66d7/dali/operators/video/color_space.cu). Its UV coordinates and [linear sampler's half-pixel adjustment](https://github.com/NVIDIA/DALI/blob/68b40ed8bd2f77985c4cf114e50fcca065cb66d7/dali/kernels/imgproc/sampler.h) imply centered chroma. [Decoder initialization](https://github.com/NVIDIA/DALI/blob/68b40ed8bd2f77985c4cf114e50fcca065cb66d7/dali/operators/video/frames_decoder_gpu.cc) selects full/limited conversion without a matrix selector. The public [video operator schema](https://github.com/NVIDIA/DALI/blob/68b40ed8bd2f77985c4cf114e50fcca065cb66d7/dali/operators/video/decoder/video_decoder_cpu.cc) and [constructor dispatch](https://github.com/NVIDIA/DALI/blob/68b40ed8bd2f77985c4cf114e50fcca065cb66d7/dali/operators/video/decoder/video_decoder_base.h) do not expose a matrix or chroma override. This path cannot be labeled an equivalent BT709/Left converter; post-correcting clipped RGB does not recover lost values.

Neither these findings nor grayscale agreement justify loosening the threshold or claiming general color equivalence. A different explicitly selected reconstruction policy would require its own declared task, correctness checks and measurements.
