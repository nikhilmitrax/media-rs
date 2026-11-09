"""Pinned comparative-benchmark environment; preflight never allocates a GPU."""
import json
from pathlib import Path

import modal

from infra.gpu.modal_image import ROOT, toolchain_image

app = modal.App("media-rs-framework-preflight")
framework_base_image = (
    toolchain_image(python_version="3.12")
    .pip_install("torch==2.11.0", "torchcodec==0.17.0",
                 index_url="https://download.pytorch.org/whl/cu130")
    .pip_install("nvidia-dali-cuda120==2.3.0", "numpy==2.2.6")
    .run_commands(
        "python -c \"import importlib.metadata as m; import torch, torchcodec, nvidia.dali; print({p:m.version(p) for p in ['torch','torchcodec','nvidia-dali-cuda120','numpy']})\""
    )
)
framework_image = framework_base_image.add_local_python_source("infra.gpu")


@app.function(image=framework_image, cpu=2, memory=8192, timeout=120,
              scaledown_window=2, max_containers=1, retries=0)
def check_frameworks():
    import importlib.metadata
    import platform
    import subprocess
    import sys

    import torch
    from torchcodec.decoders import VideoDecoder
    import nvidia.dali

    # Import APIs without initializing CUDA. Real codec/driver proof belongs to
    # the explicitly requested GPU comparison app.
    return {"gpu_allocated": False, "python": sys.version,
            "platform": platform.platform(),
            "packages": {package: importlib.metadata.version(package) for package in
                         ("torch", "torchcodec", "nvidia-dali-cuda120", "numpy")},
            "torch_cuda_build": torch.version.cuda,
            "torchcodec_decoder_imported": VideoDecoder is not None,
            "dali_imported": nvidia.dali is not None,
            "ffmpeg": subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0]}


@app.local_entrypoint()
def main():
    from datetime import datetime, timezone

    result = check_frameworks.remote()
    result["recorded_at"] = datetime.now(timezone.utc).isoformat()
    directory = ROOT / "artifacts" / "bench"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "framework-preflight.json"
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
