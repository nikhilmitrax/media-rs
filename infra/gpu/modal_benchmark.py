"""Release GPU workflow measurements; fixture generation/builds use CPU only."""
import json
import os
from pathlib import Path

import modal

from infra.gpu.modal_image import ROOT, SOURCE, source_image

app = modal.App("media-rs-gpu-benchmark")

# Fixture generation and release compilation remain CPU image-build work.
benchmark_image = (
    source_image(release=True)
    .add_local_file(str(SOURCE / "infra" / "bench" / "generate_gpu_fixture.py"
                        if modal.is_local() else Path("/opt/media-generate-gpu-fixture.py")),
                    "/opt/media-generate-gpu-fixture.py", copy=True)
    .run_commands(
        "python /opt/media-generate-gpu-fixture.py --output /workspace/gpu-fixtures --ffmpeg /usr/bin/ffmpeg",
        "/root/.cargo/bin/cargo build --release -p media --all-features --example gpu_workflow_bench --locked",
    )
    .add_local_python_source("infra.gpu")
)


@app.function(image=benchmark_image, gpu=["L4", "T4", "A10"], cpu=2,
              memory=8192, timeout=600, scaledown_window=2,
              max_containers=1, retries=0)
def benchmark_gpu():
    """Run only the prebuilt, bounded benchmark; export artifacts before shutdown."""
    import subprocess
    import time

    command = ["/workspace/target/release/examples/gpu_workflow_bench",
               "/workspace/gpu-fixtures/gpu6s.mp4", "/tmp/media-gpu-benchmark", "7"]
    start = time.monotonic()
    completed = subprocess.run(command, cwd="/workspace", capture_output=True,
                               text=True, timeout=540, check=False,
                               env={**os.environ, "PATH": "/root/.cargo/bin:" + os.environ["PATH"]})
    directory = Path("/tmp/media-gpu-benchmark")
    files = {path.name: path.read_bytes() for path in directory.glob("*") if path.is_file()}
    return {"command": command, "returncode": completed.returncode,
            "stdout": completed.stdout, "stderr": completed.stderr,
            "elapsed_seconds": time.monotonic() - start, "files": files}


@app.local_entrypoint()
def main():
    from datetime import datetime, timezone
    import uuid

    result = benchmark_gpu.remote()
    files = result.pop("files")
    metadata = json.loads((SOURCE / "source-manifest.json").read_text())
    result["source_sha256"] = metadata["sha256"]
    result["recorded_at"] = datetime.now(timezone.utc).isoformat()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    result["run_id"] = run_id
    directory = ROOT / "artifacts" / "bench" / f"modal-gpu-{metadata['sha256'][:16]}-{run_id}"
    directory.mkdir(parents=True, exist_ok=False)
    for name, content in files.items():
        # The remote collector only returns basenames, never arbitrary paths.
        if Path(name).name != name:
            raise ValueError("invalid remote artifact name")
        (directory / name).write_bytes(content)
    (directory / "result.json").write_text(json.dumps(result, indent=2))
    (directory / "source-manifest.json").write_text(json.dumps(metadata, indent=2))
    (directory / "modal-harness.py").write_bytes(Path(__file__).read_bytes())
    (directory / "fixture-generator.py").write_bytes(
        (SOURCE / "infra" / "bench" / "generate_gpu_fixture.py").read_bytes())
    print(json.dumps({"returncode":result["returncode"], "elapsed_seconds":result["elapsed_seconds"], "artifact_directory":str(directory)}, indent=2))
    if result["returncode"] != 0:
        raise SystemExit("GPU workflow benchmark failed; partial artifacts retained")
