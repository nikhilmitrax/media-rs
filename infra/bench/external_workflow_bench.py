#!/usr/bin/env python3
"""Bounded matched same-process media/FFmpeg/TorchCodec/DALI comparison.

See benches/external-comparison.md for timing, color and ownership contracts.
CPU-only Rust comparisons use NumPy; CUDA requires the pinned framework image.
"""
from __future__ import annotations
import argparse
import contextlib
import ctypes as C
import csv
import gc
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import time


class Stats(C.Structure):
    _fields_ = [(name, C.c_uint64) for name in (
        "decoded_frames", "source_bytes", "seeks", "owned_peak_bytes", "setup_ns",
        "index_build_ns", "index_decoded_frames", "index_source_bytes", "index_cache_hits",
        "native_selected_peak_bytes", "session_native_peak_bytes", "host_peak_bytes")]


# The destructor receives a borrowed PyObject address, never a new Python reference.
# Converting it to py_object inside destruction can recursively destroy the capsule.
_CAP_NEW = C.pythonapi.PyCapsule_New
_CAP_NEW.argtypes = [C.c_void_p, C.c_char_p, C.c_void_p]
_CAP_NEW.restype = C.py_object
_CAP_VALID = C.pythonapi.PyCapsule_IsValid
_CAP_VALID.argtypes = [C.c_void_p, C.c_char_p]
_CAP_VALID.restype = C.c_int
_CAP_GET = C.pythonapi.PyCapsule_GetPointer
_CAP_GET.argtypes = [C.c_void_p, C.c_char_p]
_CAP_GET.restype = C.c_void_p
_DESTRUCTOR = C.CFUNCTYPE(None, C.c_void_p)
# Keep both the library and callbacks alive until process shutdown. A consumed
# capsule is renamed used_dltensor by its consumer and is never manually released.
_CAPSULE_KEEPALIVE = []


class LegacyArray:
    """One-use CPU DLPack provider for NumPy; capsule transfers unique ownership."""
    def __init__(self, capsule):
        self.capsule = capsule

    def __dlpack__(self, stream=None, **_kwargs):
        if stream is not None:
            raise ValueError("CPU provider does not accept a CUDA stream")
        capsule, self.capsule = self.capsule, None
        if capsule is None:
            raise RuntimeError("DLPack descriptor was already consumed")
        return capsule

    def __dlpack_device__(self):
        return (1, 0)


class Bridge:
    def __init__(self, library):
        self.lib = C.CDLL(str(library.resolve()))
        if not hasattr(self.lib, "mb_abi_version") or self.lib.mb_abi_version() != 2:
            raise RuntimeError("benchmark C ABI mismatch; rebuild matched_bench_bridge for this runner")
        self.lib.mb_last_error.restype = C.c_char_p
        self.lib.mb_open.argtypes = [C.c_char_p, C.c_uint32, C.c_uint32, C.c_uint32,
                                     C.c_uint32, C.c_uint32, C.c_uint32, C.c_uint32, C.POINTER(Stats)]
        self.lib.mb_open.restype = C.c_void_p
        self.lib.mb_run.argtypes = [C.c_void_p, C.POINTER(C.c_uint64), C.c_size_t, C.POINTER(Stats)]
        self.lib.mb_run.restype = C.c_void_p
        self.lib.mb_consumer_stream.argtypes = [C.c_void_p]
        self.lib.mb_consumer_stream.restype = C.c_uint64
        self.lib.mb_close.argtypes = [C.c_void_p]
        self.lib.mb_close.restype = None
        self.lib.mb_retire.argtypes = [C.c_void_p]
        self.lib.mb_retire.restype = C.c_int
        self.lib.mb_release.argtypes = [C.c_void_p]
        self.lib.mb_release.restype = None
        lib = self.lib

        @_DESTRUCTOR
        def release(capsule_address):
            if _CAP_VALID(capsule_address, b"dltensor"):
                lib.mb_release(_CAP_GET(capsule_address, b"dltensor"))
        self.release = release
        _CAPSULE_KEEPALIVE.append((lib, release))

    def error(self):
        return self.lib.mb_last_error().decode("utf-8", "replace")

    def capsule(self, pointer):
        try:
            return _CAP_NEW(pointer, b"dltensor", C.cast(self.release, C.c_void_p))
        except BaseException:
            self.lib.mb_release(pointer)
            raise


class RustRunner:
    def __init__(self, bridge, name, path, fixture, gpu, warm, torch, np, quality):
        self.bridge, self.torch, self.np, self.gpu = bridge, torch, np, gpu
        self.name, self.handle = name, None
        backend = {"media-prefix": 0, "ffmpeg-prefix": 1, "media-indexed": 2, "media-cursor": 3, "ffmpeg": 4, "ffmpeg-seek": 4}[name]
        self.setup = Stats()
        self.handle = bridge.lib.mb_open(os.fsencode(path), backend, gpu, warm,
                                         fixture["width"], fixture["height"], quality == "point-chroma",
                                         fixture["color_profile"] == "bt601-center", C.byref(self.setup))
        if not self.handle:
            raise RuntimeError(bridge.error())
        raw = bridge.lib.mb_consumer_stream(self.handle)
        self.stream = torch.cuda.ExternalStream(raw) if gpu else None
        self.reuse = ("index+context;fresh-decoder" if warm and name == "media-indexed" else
                      "context;fresh-decoder" if name in ("media-prefix", "media-indexed") or not warm else
                      "same-live-reader+context")

    def get(self, indices):
        stats = Stats()
        array = (C.c_uint64 * len(indices))(*indices)
        pointer = self.bridge.lib.mb_run(self.handle, array, len(indices), C.byref(stats))
        if not pointer:
            raise RuntimeError(self.bridge.error())
        capsule = self.bridge.capsule(pointer)
        if self.gpu:
            # Caller enters this exact external stream before descriptor consumption.
            tensor = self.torch.from_dlpack(capsule)
        else:
            tensor = self.np.from_dlpack(LegacyArray(capsule))
        return tensor, {name: getattr(stats, name) for name, _ in Stats._fields_}

    def retire(self):
        if self.bridge.lib.mb_retire(self.handle) != 0:
            raise RuntimeError(self.bridge.error())

    def close(self):
        if self.handle:
            self.bridge.lib.mb_close(self.handle)
            self.handle = None


class TorchCodecRunner:
    def __init__(self, encoded, gpu, warm, torch, fixture, indices):
        from torchcodec.decoders import VideoDecoder, set_cuda_backend
        from torchcodec.transforms import CenterCrop
        self.encoded, self.gpu, self.warm, self.torch = encoded, gpu, warm, torch
        self.fixture, self.stream = fixture, torch.cuda.Stream() if gpu else None
        self.decoder = None
        self.keepalive = None
        self.reuse = "same-VideoDecoder;NVDEC-cache=0" if warm else "fresh-VideoDecoder;NVDEC-cache=0"
        start = time.perf_counter_ns()
        self.construct = lambda: VideoDecoder(encoded, dimension_order="NCHW", num_ffmpeg_threads=1,
                                               device="cuda:0" if gpu else "cpu", seek_mode="exact",
                                               transforms=None if gpu else [CenterCrop((224, 224))],
                                               output_dtype=torch.uint8)
        # Backend selection is thread-local and read at construction.
        self.backend = lambda: set_cuda_backend("nvdec") if gpu else contextlib.nullcontext()
        if warm:
            with self.backend():
                self.decoder = self.construct()
        self.setup = {"setup_ns": time.perf_counter_ns() - start,
                      "transforms": ("NVDEC-uint8;device-crop-center224;crop-only-float32/255;NCHW" if gpu else
                                     "CenterCrop((224,224));decoder-uint8;crop-only-float32/255;NCHW")}

    def get(self, indices):
        with self.backend():
            decoder = self.decoder if self.warm else self.construct()
        batch = decoder.get_frames_at(indices=indices)
        if self.gpu and bool(decoder.cpu_fallback):
            raise RuntimeError(f"forbidden CPU fallback: {decoder.cpu_fallback}")
        tensor = batch.data
        if self.gpu and not tensor.is_cuda:
            raise RuntimeError("TorchCodec output is not CUDA resident")
        expected_dtype = self.torch.uint8
        expected_shape = ((len(indices), 3, self.fixture["height"], self.fixture["width"]) if self.gpu else
                          (len(indices), 3, 224, 224))
        if tuple(tensor.shape) != expected_shape or tensor.dtype != expected_dtype:
            raise RuntimeError(f"TorchCodec decoded output mismatch {tensor.shape}/{tensor.dtype}")
        if self.gpu:
            # Pinned TorchCodec0.17 rejects decoder transforms on CUDA. Use its
            # supported RGB8 output and an exact device-side tensor crop view.
            top, left = (self.fixture["height"] - 224) // 2, (self.fixture["width"] - 224) // 2
            tensor = tensor[:, :, top:top + 224, left:left + 224]
        # Cast only the crop, preserving the source's 8-bit task and avoiding a
        # forced full-frame float batch. CPU float decode would select RGB48.
        tensor = tensor.to(self.torch.float32).div(255)
        self.keepalive = (decoder, batch)
        return tensor, {}

    def close(self):
        self.keepalive, self.decoder = None, None


class DaliRunner:
    def __init__(self, encoded, gpu, warm, torch, fixture, indices):
        if not gpu:
            raise RuntimeError("DALI 2.3 CPU video decoder does not support H264/HEVC; mixed NVDEC required")
        from nvidia.dali import fn, pipeline_def, types
        import numpy as np
        self.encoded, self.warm, self.torch, self.fixture = np.frombuffer(encoded, dtype=np.uint8), warm, torch, fixture
        self.stream = torch.cuda.Stream()
        self.pipeline = None
        self.keepalive = None
        self.reuse = "same-Pipeline;private-decoder-reuse=unknown" if warm else "fresh-Pipeline;index-on-input"
        @pipeline_def(batch_size=1, num_threads=1, device_id=0, exec_async=False,
                      exec_pipelined=False, prefetch_queue_depth=1)
        def create():
            data = fn.external_source(name="encoded", device="cpu", dtype=types.UINT8, ndim=1)
            frames = fn.decoders.video(data, device="mixed", frames=indices, build_index=True, pad_mode="none", affine=False)
            return fn.crop_mirror_normalize(frames, device="gpu", crop=[224, 224],
                                            crop_pos_x=0.5, crop_pos_y=0.5, rounding="truncate",
                                            dtype=types.FLOAT, output_layout="FCHW", mean=[0.0], std=[255.0])
        self.construct = create
        start = time.perf_counter_ns()
        if warm:
            self.pipeline = self.construct()
            self.pipeline.build()
        self.setup = {"setup_ns": time.perf_counter_ns() - start,
                      "transforms": "video(affine=False);crop_mirror_normalize(center224,FCHW,float32,mean0,std255,truncate)"}

    def get(self, indices):
        pipeline = self.pipeline
        if pipeline is None:
            pipeline = self.construct()
            pipeline.build()
        pipeline.feed_input("encoded", [self.encoded])
        outputs = pipeline.run()
        tensor_list = outputs[0]
        native = tensor_list.as_tensor()
        if native.__dlpack_device__()[0] != 2:
            raise RuntimeError("DALI output is not CUDA resident")
        # The protocol passes Torch's actual current consumer stream to DALI.
        tensor = self.torch.from_dlpack(native)
        if tensor.ndim != 5 or tensor.shape[0] != 1:
            raise RuntimeError(f"unexpected DALI shape {tensor.shape}")
        tensor = tensor[0]
        if tuple(tensor.shape) != (len(indices), 3, 224, 224) or tensor.dtype != self.torch.float32:
            raise RuntimeError(f"DALI transformed output mismatch {tensor.shape}/{tensor.dtype}")
        # Retain outputs and fresh pipeline through consumer completion; no implicit
        # GPU memory access can outlive this lease, including error paths.
        self.keepalive = (pipeline, outputs, native)
        return tensor, {}

    def close(self):
        self.keepalive, self.pipeline = None, None


def package_versions():
    result = {}
    for name in ("torch", "torchcodec", "nvidia-dali-cuda120", "numpy"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not-installed"
    return result


def check_deps():
    import torch
    import torchcodec
    import numpy
    from nvidia.dali import fn
    from nvidia.dali.tensors import TensorGPU
    assert hasattr(fn.decoders, "video")
    assert hasattr(TensorGPU, "__dlpack__")
    from torchcodec.decoders import VideoDecoder, set_nvdec_cache_capacity
    from torchcodec.transforms import CenterCrop
    assert CenterCrop((224, 224)) is not None
    assert hasattr(fn, "crop_mirror_normalize")
    assert callable(VideoDecoder) and callable(set_nvdec_cache_capacity)
    print(json.dumps({"packages": package_versions(), "python": sys.version, "cpu_imports": "passed"}, indent=2))


def marker_ids(array, np):
    # Consumer output = original / 2 + 1/4. White/black marker centers are neutral.
    values = array[:, 0, 40, [40 + bit * 16 for bit in range(8)]]
    return ((values > 0.5).astype(np.uint16) * (1 << np.arange(8))).sum(axis=1).tolist()


def recover_rgb_grid(consumer, np):
    """Invert the float32 consumer, verifying tiny rounding residue independently."""
    if consumer.dtype != np.float32 or consumer.size == 0:
        raise ValueError("validation requires nonempty float32 consumer output")
    if not bool(np.isfinite(consumer).all()) or float(consumer.min()) < 0.25 or float(consumer.max()) > 0.75:
        raise ValueError("consumer output must be finite and within [.25,.75]")
    scaled = (consumer.astype(np.float64) - 0.25) * 510.0
    rounded = np.rint(scaled)
    residual = float(np.abs(scaled - rounded).max())
    if residual > 1e-4 or float(rounded.min()) < 0 or float(rounded.max()) > 255:
        raise ValueError(f"consumer output is not on the8-bit task grid: residual={residual} code units")
    return rounded.astype(np.int16), residual


def compare_rgb_grid(actual, reference, np, max_error=2 / 255, mean_error=0.5 / 255):
    """Compare the declared 8-bit task in integer codes, retaining raw float evidence."""
    if actual.shape != reference.shape:
        raise ValueError("validation actual/reference shapes differ")
    a, a_residual = recover_rgb_grid(actual, np)
    r, r_residual = recover_rgb_grid(reference, np)
    difference = np.abs(a - r)
    maximum, mean = int(difference.max()), float(difference.mean())
    raw = np.abs(actual.astype(np.float64) - reference.astype(np.float64)) * 2
    return {"max_rgb_error": maximum / 255, "mean_rgb_error": mean / 255,
            "max_code_error": maximum, "mean_code_error": mean,
            "raw_float_max_rgb_error": float(raw.max()), "raw_float_mean_rgb_error": float(raw.mean()),
            "actual_grid_residual_codes": a_residual, "reference_grid_residual_codes": r_residual,
            "matched": maximum <= max_error * 255 and mean <= mean_error * 255}


def validation_selfcheck():
    """NumPy-only regression for exact task thresholds and float32 consumer rounding."""
    import numpy as np
    codes = np.arange(256, dtype=np.float32)
    consumer = (codes * np.float32(1 / 255) * np.float32(.5) + np.float32(.25)).astype(np.float32)
    recovered, residual = recover_rgb_grid(consumer, np)
    assert np.array_equal(recovered, np.arange(256)) and residual < 1e-4
    # Find a real two-code difference that the old raw-float test rejected.
    errors = np.abs(consumer[2:].astype(np.float64)-consumer[:-2].astype(np.float64))*2
    index = int(errors.argmax())
    reference = consumer[index:index+1].copy()
    actual = consumer[index+2:index+3].copy()
    assert compare_rgb_grid(actual, reference, np, mean_error=2/255)["matched"]
    assert errors[index] > 2/255
    # Opposite float32 roundoff of two ULPs at interior values remains the same codes.
    reference = consumer[100:101].copy(); actual = consumer[102:103].copy()
    for _ in range(2):
        reference = np.nextafter(reference, np.float32(-np.inf))
        actual = np.nextafter(actual, np.float32(np.inf))
    reference_many = np.repeat(reference, 5)
    actual_many = reference_many.copy(); actual_many[0] = actual[0]
    result = compare_rgb_grid(actual_many, reference_many, np)
    assert result["matched"] and result["max_code_error"] == 2 and result["mean_code_error"] == .4
    reference_ten = np.repeat(consumer[100:101], 10)
    three_codes = reference_ten.copy(); three_codes[0] = consumer[103]
    result = compare_rgb_grid(three_codes, reference_ten, np)
    assert not result["matched"] and result["max_code_error"] == 3 and result["mean_code_error"] == .3
    for invalid in (np.array([np.nan], dtype=np.float32), np.array([.251], dtype=np.float32),
                    np.array([.2499], dtype=np.float32), np.array([np.inf], dtype=np.float32)):
        try: recover_rgb_grid(invalid, np)
        except ValueError: pass
        else: raise AssertionError("invalid offgrid/nonfinite/range output accepted")
    print(json.dumps({"validation_selfcheck": "passed", "max_float32_grid_residual_codes": residual,
                      "old_rejected_two_code_error": float(errors[index])}, indent=2))

class DiagnosticArchive:
    """Post-timing evidence, limited to 16 files/64MiB total/8MiB per file."""
    def __init__(self, directory, quality="native-bilinear"):
        self.directory, self.records, self.seen = directory, [], set()
        self.quality = quality
        self.bytes = 0

    def save(self, fixture, case, scope, framework, actual, reference, indices, np):
        key = (fixture["file"], case, framework)
        if key in self.seen:
            return "already-captured"
        self.seen.add(key)
        record = {"fixture": fixture["file"], "case": case, "scope": scope, "framework": framework,
                  "quality": self.quality, "color_profile": fixture.get("color_profile", "bt709-left")}
        if len([item for item in self.records if item.get("status") == "saved"]) >= 16:
            record["status"] = "file-limit"
            self.records.append(record)
            return "file-limit"
        remaining = min(8 * 1024 * 1024, 64 * 1024 * 1024 - self.bytes)
        if remaining <= 0:
            record["status"] = "total-byte-limit"
            self.records.append(record)
            return "total-byte-limit"
        class LimitedBytes(io.BytesIO):
            def write(self, data):
                if self.tell() + len(data) > remaining:
                    raise ValueError("diagnostic compressed byte limit exceeded")
                return super().write(data)
        buffer = LimitedBytes()
        metrics = compare_rgb_grid(actual, reference, np)
        metrics.pop("matched")  # Archive error evidence, not an implicit threshold policy.
        metadata = {**record, "fixture_sha256": fixture["sha256"],
                    "array_contract": "consumer output float32 NCHW; original RGB=(value-.25)*2",
                    **metrics}
        try:
            np.savez_compressed(buffer, actual=actual, reference=reference,
                                requested_ids=np.asarray(indices, dtype=np.int64),
                                actual_ids=np.asarray(marker_ids(actual, np), dtype=np.int64),
                                reference_ids=np.asarray(marker_ids(reference, np), dtype=np.int64),
                                metadata_utf8=np.frombuffer(json.dumps(metadata).encode(), dtype=np.uint8))
            payload = buffer.getvalue()
            name = f"mismatch-{Path(fixture['file']).stem}-{case}-{scope}-{framework}.npz"
            (self.directory / name).write_bytes(payload)
            self.bytes += len(payload)
            record.update(status="saved", file=name, bytes=len(payload))
            self.records.append(record)
            return name
        except ValueError as error:
            record.update(status="compressed-byte-limit", error=str(error))
            self.records.append(record)
            return "compressed-byte-limit"


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--fixtures", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rust-lib", type=Path)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--profile", choices=("bt709-left", "bt601-center"), default="bt709-left")
    parser.add_argument("--quality", choices=("native-bilinear", "point-chroma"), default="native-bilinear")
    parser.add_argument("--frameworks", default="media-prefix,media-indexed,ffmpeg,torchcodec,dali")
    parser.add_argument("--scopes", default="fresh,warm")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--resolutions", default="1080p,4k")
    parser.add_argument("--codecs", default="h264,hevc")
    parser.add_argument("--patterns", default="gray,color")
    parser.add_argument("--cases", default="sparse,dense")
    parser.add_argument("--max-seconds", type=int, default=480)
    parser.add_argument("--max-rgb-error", type=float, default=2 / 255)
    parser.add_argument("--max-rgb-mean-error", type=float, default=0.5 / 255)
    parser.add_argument("--check-deps", action="store_true")
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        validation_selfcheck()
        return
    if args.check_deps:
        check_deps()
        return
    if not args.fixtures or not args.output or not args.rust_lib:
        parser.error("--fixtures, --output and --rust-lib are required")
    if not 1 <= args.iterations <= 10 or not 1 <= args.max_seconds <= 1800:
        parser.error("iterations must be 1..10; max-seconds must be 1..1800")
    allowed_frameworks = {"media-prefix", "media-indexed", "media-cursor", "ffmpeg", "ffmpeg-seek", "ffmpeg-prefix", "torchcodec", "dali"}
    if not set(args.frameworks.split(",")) <= allowed_frameworks:
        parser.error("unknown framework")
    if not set(args.scopes.split(",")) <= {"fresh", "warm"}:
        parser.error("scope must be fresh or warm")
    if not set(args.cases.split(",")) <= {"sparse", "dense"}:
        parser.error("case must be sparse or dense")
    import numpy as np
    gpu = args.device == "cuda"
    if args.quality == "point-chroma" and not gpu:
        parser.error("point-chroma is an explicit CUDA reconstruction policy")
    frameworks = args.frameworks.split(",")
    torch = None
    if gpu or "torchcodec" in frameworks:
        import torch
        torch.set_num_threads(1)
        if gpu:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required; CPU fallback is forbidden")
            torch.cuda.init()
        if "torchcodec" in frameworks:
            from torchcodec.decoders import set_nvdec_cache_capacity
            # Fresh decoder really means fresh hardware decoder; reusable object
            # rows remain reusable. Global framework decoder-cache effects excluded.
            set_nvdec_cache_capacity(0)
    bridge = Bridge(args.rust_lib)
    manifest = json.loads((args.fixtures / "manifest.json").read_text())
    fixtures = [item for item in manifest["files"] if item["resolution"] in args.resolutions.split(",")
                and item["codec"] in args.codecs.split(",") and item["pattern"] in args.patterns.split(",")]
    fixtures = [{**item, "color_profile": item.get("color_profile", manifest.get("color_profile", "bt709-left"))} for item in fixtures]
    if any(item["color_profile"] != args.profile for item in fixtures):
        parser.error("fixture manifest color profile differs from the explicit --profile")
    if not fixtures:
        raise RuntimeError("no fixture matches filters")
    cases = {"sparse": [141, 9, 141, 63], "dense": list(reversed(range(0, 160, 10)))}
    args.output.mkdir(parents=True, exist_ok=True)
    environment = {"python": sys.version, "platform": platform.platform(), "packages": package_versions(),
                   "device": args.device, "profile": args.profile, "quality": args.quality, "color_profiles": sorted({item["color_profile"] for item in fixtures}), "argv": sys.argv, "pid": os.getpid(),
                   "runtime_environment": {key: os.environ.get(key) for key in ("DALI_DISABLE_NVML",)},
                   "comparison": "center224;identity-resize;RGB-unit-f32-NCHW;8bit-grid;consumer=x*.5+.25",
                   "cuda": {"name": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability(),
                            "torch_cuda": torch.version.cuda} if gpu else None,
                   "unknown": ["opaque codec/driver memory", "framework internal decoded-frame counts",
                               "driver-internal host/device crossings", "DALI internal decoder reuse/index amortization"]}
    try:
        environment["nvidia_smi"] = subprocess.check_output(["nvidia-smi", "-q"], text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        environment["nvidia_smi"] = None
    (args.output / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    shutil.copy2(args.fixtures / "manifest.json", args.output / "fixture-manifest.json")
    for path in (Path(__file__), Path("crates/media/examples/matched_bench_bridge.rs"), Path("Cargo.lock")):
        if path.exists():
            shutil.copy2(path, args.output / path.name)
    (args.output / "library-sha256.txt").write_text(hashlib.sha256(args.rust_lib.read_bytes()).hexdigest() + "\n")
    fields = ["fixture", "color_profile", "quality", "device", "case", "framework", "scope", "reuse", "iteration", "phase", "status", "error",
              "setup_ns", "ready_ns", "consumer_ns", "retire_ns", "total_ns", "output_frames", "digest", "frame_ids",
              "validation_dtoh_bytes", "validation_ns", "max_rgb_error", "mean_rgb_error", "max_code_error", "mean_code_error",
              "raw_float_max_rgb_error", "raw_float_mean_rgb_error", "actual_grid_residual_codes", "reference_grid_residual_codes", "matched",
              "decoded_frames", "source_bytes", "seeks", "owned_peak_bytes", "index_build_ns", "index_decoded_frames",
              "index_source_bytes", "index_cache_hits", "native_selected_peak_bytes", "session_native_peak_bytes",
              "host_peak_bytes", "known_host_crossings", "torch_allocated_peak_bytes", "torch_reserved_peak_bytes", "mismatch_diagnostic"]
    setup_records, rows, start_run = [], [], time.monotonic()
    diagnostics = DiagnosticArchive(args.output, args.quality)
    stopped = False
    with (args.output / "raw.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for fixture in fixtures:
            path = args.fixtures / fixture["file"]
            encoded = path.read_bytes()
            if hashlib.sha256(encoded).hexdigest() != fixture["sha256"]:
                raise RuntimeError(f"fixture hash mismatch: {path}")
            for case in args.cases.split(","):
                indices = cases[case]
                for scope in args.scopes.split(","):
                    runners, errors = {}, {}
                    for name in frameworks:
                        try:
                            if name.startswith("media-") or name in ("ffmpeg", "ffmpeg-seek", "ffmpeg-prefix"):
                                runner = RustRunner(bridge, name, path, fixture, gpu, scope == "warm", torch, np, args.quality)
                            elif name == "torchcodec":
                                runner = TorchCodecRunner(encoded, gpu, scope == "warm", torch, fixture, indices)
                            elif name == "dali":
                                runner = DaliRunner(encoded, gpu, scope == "warm", torch, fixture, indices)
                            else:
                                raise RuntimeError(f"unknown framework {name}")
                            runners[name] = runner
                            setup = runner.setup
                            setup = {name: getattr(setup, name) for name, _ in Stats._fields_} if isinstance(setup, Stats) else setup
                            setup_records.append({"fixture": fixture["file"], "case": case, "scope": scope,
                                                  "framework": name, "reuse": runner.reuse, "quality": args.quality,
                                                  "color_profile": fixture["color_profile"], **setup})
                        except Exception as error:
                            errors[name] = str(error)
                    reference = None
                    try:
                        for iteration in range(args.iterations):
                            # Rotate order each iteration, with baseline first on the
                            # validation iteration so no framework is always warmed first.
                            order = frameworks[iteration % len(frameworks):] + frameworks[:iteration % len(frameworks)]
                            for name in order:
                                if time.monotonic() - start_run >= args.max_seconds:
                                    stopped = True
                                    break
                                row = {field: "" for field in fields}
                                row.update(fixture=fixture["file"], color_profile=fixture["color_profile"], quality=args.quality, device=args.device, case=case, framework=name,
                                           scope=scope, iteration=iteration, phase="initial-use" if iteration == 0 else "steady",
                                           status="error", known_host_crossings=0 if gpu and name in runners else "")
                                runner = runners.get(name)
                                if runner is None:
                                    row["error"] = errors[name]
                                else:
                                    row["reuse"] = runner.reuse
                                    tensor = consumer = None
                                    stream = runner.stream
                                    completion = None
                                    try:
                                        if gpu:
                                            torch.cuda.reset_peak_memory_stats()
                                        start = time.perf_counter_ns()
                                        with torch.cuda.stream(stream) if gpu else contextlib.nullcontext():
                                            tensor, counters = runner.get(indices)
                                            if gpu:
                                                if not tensor.is_cuda or tensor.device.index != 0:
                                                    raise RuntimeError("forbidden host/foreign-device output")
                                                # Targeted event measures producer completion; never deviceSync.
                                                ready = torch.cuda.Event()
                                                ready.record(stream)
                                                ready.synchronize()
                                            ready_at = time.perf_counter_ns()
                                            if gpu:
                                                # Shared task includes projecting all RGB onto the same 8-bit grid.
                                                # Decode color algorithms can still differ and must pass validation.
                                                tensor = tensor.mul(255).round().clamp(0, 255).div(255)
                                                consumer = tensor.mul(0.5).add(0.25).contiguous()
                                                completion = torch.cuda.Event()
                                                completion.record(stream)
                                                completion.synchronize()
                                            else:
                                                if torch is not None and isinstance(tensor, torch.Tensor):
                                                    tensor = tensor.numpy()
                                                consumer = np.ascontiguousarray(np.round(tensor * 255).clip(0, 255) / 255 * 0.5 + 0.25)
                                            consumer_at = time.perf_counter_ns()
                                        row.update(counters)
                                        row.update(ready_ns=ready_at - start, consumer_ns=consumer_at - ready_at,
                                                   output_frames=len(indices), status="ok")
                                        # Independent validation happens after the timed GPU stage. Preserve
                                        # the consumer output until this transfer; input leases can retire now.
                                        del tensor
                                        tensor = None
                                        if isinstance(runner, RustRunner):
                                            runner.retire()
                                        if isinstance(runner, (DaliRunner, TorchCodecRunner)):
                                            runner.keepalive = None
                                        retired_at = time.perf_counter_ns()
                                        row["retire_ns"] = retired_at - consumer_at
                                        row["total_ns"] = retired_at - start
                                        validation_start = time.perf_counter_ns()
                                        array = consumer.cpu().numpy() if gpu else consumer
                                        if tuple(array.shape) != (len(indices), 3, 224, 224) or array.dtype != np.float32:
                                            raise RuntimeError(f"unexpected output contract {array.shape}/{array.dtype}")
                                        actual = marker_ids(array, np)
                                        if actual != indices:
                                            raise RuntimeError(f"actual frame IDs {actual} != {indices}")
                                        row.update(frame_ids=json.dumps(actual), digest=hashlib.sha256(array.tobytes()).hexdigest(),
                                                   validation_dtoh_bytes=array.nbytes if gpu else 0)
                                        if reference is None and name == "media-prefix":
                                            reference = array.copy()
                                        if reference is not None:
                                            row.update(compare_rgb_grid(array, reference, np,
                                                                        args.max_rgb_error, args.max_rgb_mean_error))
                                            if row["matched"] is False and iteration == 0:
                                                row["mismatch_diagnostic"] = diagnostics.save(
                                                    fixture, case, scope, name, array, reference, indices, np)
                                        else:
                                            _, row["actual_grid_residual_codes"] = recover_rgb_grid(array, np)
                                            row["matched"] = "unvalidated-no-media-prefix-reference"
                                        row["validation_ns"] = time.perf_counter_ns() - validation_start
                                        if gpu:
                                            row["torch_allocated_peak_bytes"] = torch.cuda.max_memory_allocated()
                                            row["torch_reserved_peak_bytes"] = torch.cuda.max_memory_reserved()
                                        consumer = None
                                        if isinstance(runner, (DaliRunner, TorchCodecRunner)):
                                            runner.keepalive = None
                                    except Exception as error:
                                        # If a post-submission error happens, retire this exact queue before
                                        # dropping producer leases. No global synchronization or host fallback.
                                        if gpu and stream is not None:
                                            stream.synchronize()
                                        row.update(status="error", error=str(error), matched=False)
                                        consumer = tensor = None
                                        if isinstance(runner, (DaliRunner, TorchCodecRunner)):
                                            runner.keepalive = None
                                writer.writerow(row)
                                file.flush()
                                rows.append(row)
                                detail = (f" error={str(row['error'])[:600]!r}" if row["error"] else
                                          f" max_rgb_error={row['max_rgb_error']} mean_rgb_error={row['mean_rgb_error']}")
                                print(f"{fixture['file']} {case} {scope} {iteration} {name}: {row['status']} matched={row['matched']}{detail}", flush=True)
                            if stopped:
                                break
                    finally:
                        for runner in runners.values():
                            runner.close()
                        gc.collect()
                    if stopped:
                        break
                if stopped:
                    break
            if stopped:
                break
    (args.output / "setup.json").write_text(json.dumps(setup_records, indent=2) + "\n")
    (args.output / "mismatch-diagnostics.json").write_text(json.dumps({"bytes": diagnostics.bytes, "records": diagnostics.records}, indent=2) + "\n")
    groups = {}
    for row in rows:
        if row["status"] != "ok" or row["matched"] is not True:
            continue
        key = (row["fixture"], row["case"], row["scope"], row["framework"], row["phase"])
        groups.setdefault(key, []).append(row)
    medians = []
    for (fixture, case, scope, framework, phase), observations in sorted(groups.items()):
        entry = dict(fixture=fixture, quality=args.quality, color_profile=next(item["color_profile"] for item in fixtures if item["file"] == fixture),
                     case=case, scope=scope, framework=framework, phase=phase,
                     samples=len(observations), reuse=observations[0]["reuse"])
        for metric in ("total_ns", "ready_ns", "consumer_ns", "retire_ns", "decoded_frames", "source_bytes", "seeks", "index_build_ns"):
            values = [float(row[metric]) for row in observations if row[metric] != ""]
            if values:
                entry[metric] = statistics.median(values)
        medians.append(entry)
    summary = {"profile": args.profile, "quality": args.quality, "rows": len(rows), "ok": sum(row["status"] == "ok" for row in rows),
               "errors": sum(row["status"] != "ok" for row in rows),
               "validated_matches": sum(row["status"] == "ok" and row["matched"] is True for row in rows),
               "unvalidated": sum(row["status"] == "ok" and isinstance(row["matched"], str) for row in rows),
               "mismatched": sum(row["status"] == "ok" and row["matched"] is False for row in rows),
               "bounded_stop": stopped, "elapsed_seconds": time.monotonic() - start_run,
               "matched_medians": medians,
               "interpretation": "Only matched=true rows share the validated numeric task; initial-use includes lazy setup. Reuse labels differ explicitly. No universal speedup claim."}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "matched_medians"}, indent=2))
    if not rows or summary["validated_matches"] == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
