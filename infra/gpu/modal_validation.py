"""Bounded correctness checks; compilation runs on CPU before GPU allocation.

Run `modal run -m infra.gpu.modal_validation::main` after normal authentication.
Benchmarks use the separate modal_benchmark app and never build implicitly here.
"""
import json
import os
from pathlib import Path

import modal

from infra.gpu.modal_image import ROOT, SOURCE, source_image

app = modal.App("media-rs-gpu-validation")
image = (
    source_image()
    .run_commands(
        "/root/.cargo/bin/cargo test -p media-backend-cuda --test hardware --locked --no-run",
        "/root/.cargo/bin/cargo test -p media-backend-ffmpeg --features cuda --test gpu_decode --locked --no-run",
        "/root/.cargo/bin/cargo test -p media-interop --features cuda --test cuda_dlpack --locked --no-run",
        "/root/.cargo/bin/cargo test -p media-backend-ffmpeg --features cuda-filters --test gpu_encode --locked --no-run",
        "/root/.cargo/bin/cargo test -p media --no-default-features --features ffmpeg,cuda-filters,interop --test cuda_video --locked --no-run",
        "/root/.cargo/bin/cargo test -p media --all-features --test cuda_session --locked --no-run",
        "/root/.cargo/bin/cargo test -p media --all-features --test cuda_clips --locked --no-run",
        "/root/.cargo/bin/cargo test -p media --all-features --test indexed_video --locked --no-run",
        "/root/.cargo/bin/cargo test -p media --all-features --test adoption_workflows --locked --no-run",
    )
    .add_local_python_source("infra.gpu")
)


def run_command(command, timeout):
    """Retain bounded partial diagnostics on subprocess timeout."""
    import subprocess
    import time

    start = time.monotonic()
    try:
        completed = subprocess.run(
            command, cwd="/workspace", capture_output=True, text=True,
            timeout=timeout, check=False,
            env={**os.environ, "MEDIA_REQUIRE_FFMPEG_TESTS": "1"},
        )
        result = {"command": command, "returncode": completed.returncode,
                  "stdout": completed.stdout, "stderr": completed.stderr}
    except subprocess.TimeoutExpired as error:
        def decoded(value):
            return value.decode(errors="replace") if isinstance(value, bytes) else value or ""
        result = {"command": command, "returncode": -1,
                  "stdout": decoded(error.stdout),
                  "stderr": decoded(error.stderr) + f"\n{timeout}-second timeout"}
    result["seconds"] = time.monotonic() - start
    return result


@app.function(
    image=image, gpu=["L4", "T4", "A10"], cpu=2, memory=8192, timeout=600,
    scaledown_window=2, max_containers=1, retries=0,
)
def validate(adoption_only: bool = False):
    import ctypes.util
    import os
    import subprocess
    import time

    start = time.monotonic()
    results = []
    commands = [
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"],
        ["/root/.cargo/bin/cargo", "test", "-p", "media-backend-cuda", "--test", "hardware",
         "--locked", "--", "--ignored", "--nocapture", "--test-threads=1"],
        ["/root/.cargo/bin/cargo", "test", "-p", "media-backend-ffmpeg", "--features", "cuda",
         "--test", "gpu_decode", "--locked", "--", "--ignored", "--nocapture", "--test-threads=1"],
        ["/root/.cargo/bin/cargo", "test", "-p", "media-interop", "--features", "cuda",
         "--test", "cuda_dlpack", "--locked", "--", "--ignored", "--nocapture", "--test-threads=1"],
        ["/root/.cargo/bin/cargo", "test", "-p", "media-backend-ffmpeg", "--features", "cuda-filters",
         "--test", "gpu_encode", "--locked", "--", "--ignored", "--nocapture", "--test-threads=1"],
        ["/root/.cargo/bin/cargo", "test", "-p", "media", "--no-default-features", "--features", "ffmpeg,cuda-filters,interop",
         "--test", "cuda_video", "--locked", "--", "--ignored", "--nocapture", "--test-threads=1"],
        ["/root/.cargo/bin/cargo", "test", "-p", "media", "--all-features",
         "--test", "cuda_session", "--locked", "--", "--ignored", "--nocapture", "--test-threads=1"],
        ["/root/.cargo/bin/cargo", "test", "-p", "media", "--all-features",
         "--test", "cuda_clips", "--locked", "--", "--ignored", "--nocapture", "--test-threads=1"],
        ["/root/.cargo/bin/cargo", "test", "-p", "media", "--all-features",
         "--test", "indexed_video", "--locked", "--", "--ignored", "--nocapture", "--test-threads=1"],
        ["/root/.cargo/bin/cargo", "test", "-p", "media", "--all-features",
         "--test", "adoption_workflows", "indexed_cuda", "--locked", "--", "--ignored", "--nocapture", "--test-threads=1"],
    ]
    if adoption_only:
        commands = [commands[0], commands[-1]]
    for command in commands:
        result = run_command(command, 180)
        results.append(result)
        print(json.dumps({"command": command, "returncode": result["returncode"],
                          "seconds": result["seconds"]}), flush=True)
    return {
        "elapsed_seconds": time.monotonic() - start,
        "video_libraries": {
            "nvcuvid": ctypes.util.find_library("nvcuvid"),
            "nvidia_encode": ctypes.util.find_library("nvidia-encode"),
        },
        "results": results,
        "cuda_filters": [line.strip() for line in subprocess.run(
            ["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True,
            timeout=10, check=False,
        ).stdout.splitlines() if "cuda" in line],
    }


@app.local_entrypoint()
def main():
    from datetime import datetime, timezone

    result = validate.remote()
    result["recorded_at"] = datetime.now(timezone.utc).isoformat()
    manifest = SOURCE / "source-manifest.json"
    if manifest.exists():
        metadata = json.loads(manifest.read_text())
        result["source_sha256"] = metadata["sha256"]
        result["source_file_count"] = len(metadata["files"])
    destination = ROOT / "artifacts" / "gpu"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "modal-validation.json").write_text(json.dumps(result, indent=2))
    if "source_sha256" in result:
        (destination / f"modal-validation-{result['source_sha256'][:16]}.json").write_text(
            json.dumps(result, indent=2)
        )
    print(json.dumps({"source_sha256": result.get("source_sha256"),
                      "elapsed_seconds": result["elapsed_seconds"],
                      "returncodes": [step["returncode"] for step in result["results"]]}))
    if any(step["returncode"] != 0 for step in result["results"]):
        raise SystemExit("Real GPU validation failed; see artifacts/gpu/modal-validation.json")


@app.local_entrypoint()
def adoption():
    result = validate.remote(True)
    metadata = json.loads((SOURCE / "source-manifest.json").read_text())
    result["source_sha256"] = metadata["sha256"]
    destination = ROOT / "artifacts" / "gpu" / f"modal-adoption-{metadata['sha256'][:16]}.json"
    destination.write_text(json.dumps(result, indent=2))
    print(json.dumps({"result": str(destination), "elapsed_seconds": result["elapsed_seconds"],
                      "returncodes": [step["returncode"] for step in result["results"]]}))
    if any(step["returncode"] != 0 for step in result["results"]):
        raise SystemExit("GPU adoption scenarios failed")


@app.function(image=image, cpu=2, memory=8192, timeout=180,
              scaledown_window=2, max_containers=1, retries=0)
def validate_cpu():
    """Isolate version-specific packet/mux behavior without allocating a GPU."""
    import subprocess
    command = ["/root/.cargo/bin/cargo", "test", "-p", "media-backend-ffmpeg",
               "--features", "cuda-filters", "--lib", "header_only_audio_copy",
               "--locked", "--", "--nocapture"]
    result = run_command(command, 150)
    version = subprocess.run(["ffmpeg", "-version"], capture_output=True,
                             text=True, timeout=10, check=False).stdout.splitlines()[0]
    result["ffmpeg"] = version
    return result


@app.local_entrypoint()
def cpu():
    result = validate_cpu.remote()
    manifest = json.loads((SOURCE / "source-manifest.json").read_text())
    result["source_sha256"] = manifest["sha256"]
    destination = ROOT / "artifacts" / "gpu" / f"modal-cpu-{manifest['sha256'][:16]}.json"
    destination.write_text(json.dumps(result, indent=2))
    print(json.dumps({"result": str(destination), "returncode": result["returncode"]}))
    if result["returncode"] != 0:
        raise SystemExit("CPU packet/mux regression failed")


@app.function(image=image, cpu=2, memory=8192, timeout=600,
              scaledown_window=2, max_containers=1, retries=0)
def validate_workspace():
    """Full Rust1.88/Linux feature tests; ignored GPU tests stay unexercised."""
    import subprocess
    command=["/root/.cargo/bin/cargo", "test", "--workspace", "--all-features", "--locked"]
    result = run_command(command, 540)
    rust=subprocess.run(["/root/.cargo/bin/rustc","--version"], capture_output=True,
        text=True, timeout=10, check=False).stdout.strip()
    result.update(rustc=rust, gpu_allocated=False)
    return result


@app.local_entrypoint()
def workspace():
    result=validate_workspace.remote()
    metadata=json.loads((SOURCE/"source-manifest.json").read_text())
    result["source_sha256"]=metadata["sha256"]
    destination=ROOT/"artifacts"/"gpu"/f"modal-workspace-{metadata['sha256'][:16]}.json"
    destination.write_text(json.dumps(result,indent=2))
    print(json.dumps({"result": str(destination), "returncode": result["returncode"], "rustc": result["rustc"]}))
    if result["returncode"]!=0:
        raise SystemExit("Rust1.88/Linux workspace tests failed")
