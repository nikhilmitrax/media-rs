"""Bounded same-GPU 1080p/4K comparison using prebuilt code and CPU-made fixtures."""
import hashlib
import json
import os
from pathlib import Path

import modal

from infra.gpu.modal_frameworks import framework_base_image
from infra.gpu.modal_image import ROOT, SOURCE, dependency_snapshot

app = modal.App("media-rs-external-comparison")
FIXTURES = (Path(os.environ.get("MEDIA_COMPARISON_FIXTURES", str(ROOT / "artifacts" / "bench" / "external-fixtures")))
            if modal.is_local() else Path("/bench-fixtures"))
fixture_manifest = json.loads((FIXTURES / "manifest.json").read_text())
fixture_profile = fixture_manifest.get("color_profile", "bt709-left")
if fixture_profile not in ("bt709-left", "bt601-center"):
    raise ValueError("unsupported comparison fixture profile")
dependencies = dependency_snapshot(SOURCE) if modal.is_local() else SOURCE
comparison_image = (
    framework_base_image
    .add_local_dir(str(dependencies), "/workspace", copy=True)
    .workdir("/workspace")
    .run_commands("/root/.cargo/bin/cargo build --release -p media --lib --all-features --locked")
    .add_local_dir(str(SOURCE / "crates"), "/workspace/crates", copy=True)
    .run_commands("/root/.cargo/bin/cargo build --release -p media --all-features --example matched_bench_bridge --locked")
)
comparison_image = comparison_image.add_local_file(
    str(FIXTURES / "manifest.json"), "/bench-fixtures/manifest.json", copy=True
)
for fixture in fixture_manifest["files"]:
    name = fixture["file"]
    if Path(name).name != name or not name.endswith(".mp4"):
        raise ValueError("only generated MP4 basenames may enter the comparison image")
    path = FIXTURES / name
    if modal.is_local() and hashlib.sha256(path.read_bytes()).hexdigest() != fixture["sha256"]:
        raise ValueError(f"fixture hash mismatch: {name}")
    comparison_image = comparison_image.add_local_file(str(path), f"/bench-fixtures/{name}", copy=True)

# Keep the CPU ABI/transform check on its established BT709 corpus even when
# the GPU task intentionally uses DALI's BT601 conversion contract.
preflight = (ROOT / "artifacts" / "bench" / "external-fixtures"
             if modal.is_local() else Path("/bench-preflight-fixtures"))
if modal.is_local():
    preflight_manifest = json.loads((preflight / "manifest.json").read_text())
    preflight_fixture = next(item for item in preflight_manifest["files"]
                             if item["resolution"] == "1080p" and item["codec"] == "h264"
                             and item["pattern"] == "color")
    preflight_path = preflight / preflight_fixture["file"]
    if hashlib.sha256(preflight_path.read_bytes()).hexdigest() != preflight_fixture["sha256"]:
        raise ValueError("CPU preflight fixture hash mismatch")
else:
    preflight_fixture = {"file": "1080p-h264-color.mp4"}
    preflight_path = Path("/bench-preflight-fixtures") / preflight_fixture["file"]
comparison_image = (
    comparison_image
    .add_local_file(str(preflight / "manifest.json"), "/bench-preflight-fixtures/manifest.json", copy=True)
    .add_local_file(str(preflight_path), f"/bench-preflight-fixtures/{preflight_fixture['file']}", copy=True)
)

comparison_image = (
    comparison_image
    .add_local_dir(str(SOURCE / "infra" / "bench"), "/workspace/infra/bench", copy=True)
    .run_commands("python /workspace/infra/bench/external_workflow_bench.py --selfcheck")
    .run_commands("python /workspace/infra/bench/external_workflow_bench.py --check-deps")
    .run_commands(
        "python /workspace/infra/bench/external_workflow_bench.py --device cpu --frameworks media-prefix,media-indexed,ffmpeg,torchcodec --fixtures /bench-preflight-fixtures --profile bt709-left --quality native-bilinear --rust-lib /workspace/target/release/examples/libmatched_bench_bridge.so --output /tmp/comparison-cpu-preflight --resolutions 1080p --codecs h264 --patterns color --cases sparse --scopes fresh --iterations 1",
        "python -c \"import csv,json; rows=list(csv.DictReader(open('/tmp/comparison-cpu-preflight/raw.csv'))); print(json.dumps([{k:r[k] for k in ['framework','matched','max_rgb_error','mean_rgb_error','frame_ids']} for r in rows],indent=2)); r=json.load(open('/tmp/comparison-cpu-preflight/summary.json')); assert r['rows']==4 and r['errors']==0 and r['mismatched']==0, r\"",
    )
    .add_local_python_source("infra.gpu")
)


@app.function(image=comparison_image, gpu="T4", cpu=2, memory=16384,
              timeout=1200, scaledown_window=2, max_containers=1, retries=0)
def compare(resolutions, codecs, patterns, cases, scopes, frameworks, quality,
            profile, iterations, max_seconds):
    import subprocess
    import time
    import uuid

    start = time.monotonic()
    directory = Path("/tmp") / f"media-comparison-{uuid.uuid4().hex}"
    command = ["python", "/workspace/infra/bench/external_workflow_bench.py",
               "--fixtures", "/bench-fixtures", "--output", str(directory),
               "--rust-lib", "/workspace/target/release/examples/libmatched_bench_bridge.so",
               "--device", "cuda", "--resolutions", resolutions, "--codecs", codecs,
               "--patterns", patterns, "--cases", cases, "--scopes", scopes,
               "--frameworks", frameworks, "--quality", quality, "--profile", profile,
               "--iterations", str(iterations), "--max-seconds", str(max_seconds)]
    try:
        completed = subprocess.run(command, cwd="/workspace", capture_output=True,
                                   text=True, timeout=max_seconds + 120, check=False,
                                   env={**os.environ, "DALI_DISABLE_NVML": "1"})
        result = {"returncode": completed.returncode, "stdout": completed.stdout,
                  "stderr": completed.stderr}
    except subprocess.TimeoutExpired as error:
        def decoded(value):
            return value.decode(errors="replace") if isinstance(value, bytes) else value or ""
        result = {"returncode": -1, "stdout": decoded(error.stdout),
                  "stderr": decoded(error.stderr) + "\ncomparison subprocess timeout"}
    summary = directory / "summary.json"
    if summary.exists():
        import csv
        validation = json.loads(summary.read_text())
        with (directory / "raw.csv").open(newline="") as rows_file:
            rows = list(csv.DictReader(rows_file))
        validation["all_rows_matched"] = bool(rows) and all(
            row["status"] == "ok" and row["matched"] == "True" for row in rows
        )
        result["validation"] = validation
    result.update(command=command, elapsed_seconds=time.monotonic() - start,
                  fixture_profile=profile,
                  runtime_env={"DALI_DISABLE_NVML": "1"},
                  files={path.name: path.read_bytes() for path in directory.glob("*") if path.is_file()})
    return result


def choices(value, permitted):
    items = value.split(",")
    if not items or any(item not in permitted for item in items):
        raise ValueError(f"expected comma-separated values from {permitted}")
    return value


@app.local_entrypoint()
def main(resolutions: str = "1080p,4k", codecs: str = "h264,hevc",
         patterns: str = "gray,color", cases: str = "sparse,dense",
         scopes: str = "fresh,warm",
         frameworks: str = "media-prefix,media-indexed,ffmpeg,torchcodec,dali",
         quality: str = "native-bilinear", profile: str = "bt709-left",
         iterations: int = 6, max_seconds: int = 900):
    from datetime import datetime, timezone
    import uuid

    choices(resolutions, ("1080p", "4k"))
    choices(codecs, ("h264", "hevc"))
    choices(patterns, ("gray", "color"))
    choices(cases, ("sparse", "dense"))
    choices(scopes, ("fresh", "warm"))
    choices(frameworks, ("media-prefix", "media-indexed", "media-cursor", "ffmpeg",
                         "ffmpeg-seek", "ffmpeg-prefix", "torchcodec", "dali"))
    choices(quality, ("native-bilinear", "point-chroma"))
    choices(profile, ("bt709-left", "bt601-center"))
    if profile != fixture_profile:
        raise ValueError(f"requested {profile}, but fixture manifest is {fixture_profile}")
    if not 1 <= iterations <= 10 or not 1 <= max_seconds <= 1000:
        raise ValueError("iterations must be 1..10 and max-seconds 1..1000")
    metadata = json.loads((SOURCE / "source-manifest.json").read_text())
    call = compare.spawn(resolutions, codecs, patterns, cases, scopes, frameworks,
                         quality, profile, iterations, max_seconds)
    submission = {
        "function_call_id": call.object_id, "source_sha256": metadata["sha256"],
        "recorded_at": datetime.now(timezone.utc).isoformat(), "profile": profile,
        "quality": quality, "resolutions": resolutions, "codecs": codecs,
        "frameworks": frameworks, "max_seconds": max_seconds,
    }
    submission_path = ROOT / "artifacts" / "bench" / f"submission-{call.object_id}.json"
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission_path.write_text(json.dumps(submission, indent=2))
    print(json.dumps(submission), flush=True)
    result = call.get()
    files = result.pop("files")
    result.update(source_sha256=metadata["sha256"], recorded_at=datetime.now(timezone.utc).isoformat())
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    directory = ROOT / "artifacts" / "bench" / f"external-{metadata['sha256'][:16]}-{run_id}"
    directory.mkdir(parents=True, exist_ok=False)
    for name, content in files.items():
        if Path(name).name != name:
            raise ValueError("invalid remote artifact name")
        (directory / name).write_bytes(content)
    (directory / "result.json").write_text(json.dumps(result, indent=2))
    (directory / "source-manifest.json").write_text(json.dumps(metadata, indent=2))
    for path in (Path(__file__), ROOT / "infra" / "gpu" / "modal_image.py",
                 ROOT / "infra" / "gpu" / "modal_frameworks.py"):
        (directory / path.name).write_bytes(path.read_bytes())
    print(json.dumps({"returncode": result["returncode"], "elapsed_seconds": result["elapsed_seconds"],
                      "validation": {k:v for k,v in result.get("validation", {}).items() if k != "matched_medians"},
                      "artifact_directory": str(directory)}, indent=2))
    if (result["returncode"] != 0
            or not result.get("validation", {}).get("all_rows_matched", False)
            or result.get("validation", {}).get("bounded_stop", True)):
        raise SystemExit(f"GPU comparison failed; partial artifacts retained in {directory}")
