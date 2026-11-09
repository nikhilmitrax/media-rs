"""Separate CPU-only ASan app, with instrumented dependencies cached before source."""
import json
import os
import modal
from infra.gpu.modal_image import ROOT, SOURCE, dependency_image
from infra.gpu.modal_validation import run_command

app = modal.App("media-rs-adoption-sanitizer")

# AddressSanitizer has a startup recursion on this project's current macOS host.
# Run an instrumented Rust standard library and dependencies on Linux instead;
# distribution FFmpeg shared libraries are not themselves ASan-instrumented.
sanitizer_flags = "-Zsanitizer=address -Cdebuginfo=1"
sanitizer_image = (
    dependency_image()
    .run_commands(
        "/root/.cargo/bin/rustup toolchain install nightly-2025-12-05 --profile minimal --component rust-src",
    )
    .run_commands(
        "RUSTFLAGS='-Zsanitizer=address -Cdebuginfo=1' CARGO_TARGET_DIR=/workspace/target-adoption-asan "
        "/root/.cargo/bin/cargo +nightly-2025-12-05 -Zbuild-std test -p media --features ffmpeg-audio "
        "--target x86_64-unknown-linux-gnu --lib --locked --no-run",
    )
    .add_local_dir(str(SOURCE / "crates"), "/workspace/crates", copy=True)
    .run_commands(
        "RUSTFLAGS='-Zsanitizer=address -Cdebuginfo=1' CARGO_TARGET_DIR=/workspace/target-adoption-asan "
        "/root/.cargo/bin/cargo +nightly-2025-12-05 -Zbuild-std test -p media --features ffmpeg-audio "
        "--target x86_64-unknown-linux-gnu --test adoption_workflows --locked --no-run",
    )
    .add_local_python_source("infra.gpu")
)


@app.function(image=sanitizer_image, cpu=2, memory=8192, timeout=600,
              scaledown_window=2, max_containers=1, retries=0)
def validate_sanitizer():
    os.environ.update(RUSTFLAGS=sanitizer_flags,
                      CARGO_TARGET_DIR="/workspace/target-adoption-asan",
                      ASAN_OPTIONS="detect_leaks=1:halt_on_error=1")
    prefix = ["/root/.cargo/bin/cargo", "+nightly-2025-12-05", "-Zbuild-std", "test", "-p", "media",
              "--features", "ffmpeg-audio", "--target", "x86_64-unknown-linux-gnu",
              "--test", "adoption_workflows", "--locked"]
    return {"gpu_allocated": False, "instrumented": "Rust standard library, crates and dependencies; system FFmpeg uninstrumented",
            "results": [run_command(prefix + ["--", "--nocapture", "--test-threads=1"], 240),
                        run_command(prefix + ["mutated_media", "--", "--ignored", "--nocapture", "--test-threads=1"], 240)]}


@app.local_entrypoint()
def main():
    result = validate_sanitizer.remote()
    metadata = json.loads((SOURCE / "source-manifest.json").read_text())
    result["source_sha256"] = metadata["sha256"]
    destination = ROOT / "artifacts" / "gpu" / f"modal-sanitizer-{metadata['sha256'][:16]}.json"
    destination.write_text(json.dumps(result, indent=2))
    print(json.dumps({"result": str(destination), "returncodes": [step["returncode"] for step in result["results"]]}))
    if any(step["returncode"] != 0 for step in result["results"]):
        raise SystemExit("Linux AddressSanitizer validation failed")
