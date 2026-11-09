# Indexed reader measurements — November 9, 2026

Before/after runs on Tesla T4 GPUs with 2 CPU cores, Torch 2.11.0+cu130 and TorchCodec 0.17.0+cu130. Each run contains 96 numerically matched rows; the table uses five steady samples after one initial use. These are separate cloud runs on the same GPU class, not an isolated same-process A/B experiment.

The task is the same 4K H264/HEVC BT709/Left point-chroma center224 crop, RGB float32 NCHW, identity resize and common CUDA consumer documented in [the methodology](../../external-comparison.md). Setup/index construction and validation downloads are excluded; producer completion, the common consumer and input retirement are included. TorchCodec retains its decoder with its global NVDEC cache disabled in both runs.

| Codec | Requests | Indexed before (ms) | Indexed after (ms) | Reduction | TorchCodec after (ms) |
| --- | --- | ---: | ---: | ---: | ---: |
| H264 | sparse | 650.76 | 612.91 | 5.8% | 267.19 |
| H264 | dense | 1556.14 | 1145.51 | 26.4% | 808.74 |
| HEVC | sparse | 272.05 | 172.78 | 36.5% | 170.59 |
| HEVC | dense | 612.70 | 464.52 | 24.2% | 469.72 |

The final warm path uses `indexed_reader()` to retain the actual decoder. Previously, only the timing index and CUDA context were retained. Ordinary request/plan reads still open independent decoders. Work/read/seek reports and limits reset for every reusable-reader batch; source and decoded-index evidence are validated on every call.

Dense H264 seeks fell from six to two in every steady sample, with decoded work rising from 106 to 142 frames. This trades extra forward decode for fewer expensive restarts. HEVC retained its GOP-local seeks while benefiting from decoder reuse. HEVC is approximately level with TorchCodec in these cases; H264 remains slower. Sparse H264 has substantial run-to-run variation, so its small measured reduction is not strong evidence of a stable gain.

Deferred native-surface retirement was evaluated and removed. A preliminary deferred run measured sparse H264 indexed latency at 785.30 ms; a nine-steady-sample diagnostic restoring synchronous release measured 598.51 ms. The final full comparison above retains the established synchronous release. These experiments do not isolate the remaining H264 backend cost.

Validation: 342 local tests and 38 real GPU tests passed, including VFR/B-frame identities, repeated caller order, retained outputs, per-batch limits, cancellation, 4K/Main10 conformance and the 128-request concurrent soak. Formatting, Clippy and rustdoc checks passed.

Raw rows, source digests, setup costs, framework versions, GPU/driver details and verification commands are retained in [metadata.json](metadata.json), [before rows](torchcodec-t4-before.csv), and [after rows](torchcodec-t4-after.csv). Historical November 3 measurements are unchanged.
