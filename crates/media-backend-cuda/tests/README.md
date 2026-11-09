# CUDA validation

CPU-only checks run by default:

```sh
cargo test -p media-backend-cuda --offline
```

Actual hardware checks are ignored by default and must fail if their required
hardware or runtime is unavailable when explicitly selected:

```sh
cargo test -p media-backend-cuda --test hardware -- --ignored --test-threads=1
```

Runtime requirements: Linux/Windows NVIDIA GPU with compute capability 7.5 or
newer, a CUDA 12.8-compatible host driver, and NVRTC 12.8 development libraries.
On Linux the loader must find `libcuda.so.1`, `libnvrtc.so.12`, and NVRTC's matching
builtins library. A CUDA 12.8.1 development image supplies NVRTC; the host supplies
the driver. No CUDA installation is needed just to compile this Rust crate.

The tests execute real upload/download, cross-stream copies and events,
NV12/P010 conversion, RGB packing/normalization, crop/resize with padded pitch,
native context incompatibility, and allocation/stream limits. They do not prove
hardware video decoding. FFmpeg's CUDA integration tests exercise that separate
path and the borrowed native-plane adapter.

Color conversion produces clamped nonlinear RGB before normalization. RGB-domain
resize is the default: conversion/clamping precedes RGB interpolation. Explicit
`ResizeDomain::YuvPlanes` instead interpolates YUV planes before conversion and
can differ on saturated edges; it is not an equivalent RGB-resize optimization.
Matrix,
code range and chroma siting are explicit. No transfer linearization, ICC color
management, HDR tone mapping, or CPU fallback is implied.
