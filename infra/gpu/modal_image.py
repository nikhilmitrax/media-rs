"""CPU-only image recipes shared by validation and performance runs.

Third-party debug/release builds precede the real source overlay. Empty crate
sources in the dependency layer are cache scaffolds, never behavioral evidence.
No credentials, Git metadata, build output, Volume, or GPU enters these recipes.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import tomllib

import modal

ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/workspace")
SOURCE = Path(os.environ.get("MEDIA_GPU_SOURCE_SNAPSHOT", str(ROOT))).resolve()


def dependency_snapshot(source):
    manifests = [source / "Cargo.toml", source / "Cargo.lock",
                 *sorted((source / "crates").glob("*/Cargo.toml"))]
    contents = {str(path.relative_to(source)): path.read_bytes() for path in manifests}
    digest = hashlib.sha256(b"dependency-skeleton-v2" + b"".join(
        name.encode() + b"\0" + contents[name] for name in sorted(contents)
    )).hexdigest()
    directory = ROOT / "artifacts" / "gpu" / "dependency-snapshots"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / digest
    if destination.exists():
        return destination
    temporary = Path(tempfile.mkdtemp(prefix="preparing-", dir=directory))
    try:
        for name, content in contents.items():
            output = temporary / name
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(content)
            if name.startswith("crates/"):
                manifest = tomllib.loads(content.decode())
                library = output.parent / manifest.get("lib", {}).get("path", "src/lib.rs")
                library.parent.mkdir(parents=True, exist_ok=True)
                library.write_text("//! Dependency cache only; replaced before validation.\n")
                for kind in ("example", "bench", "test", "bin"):
                    for target in manifest.get(kind, []):
                        folder = {"example": "examples", "bench": "benches", "test": "tests", "bin": "src/bin"}[kind]
                        placeholder = output.parent / target.get("path", f"{folder}/{target['name']}.rs")
                        placeholder.parent.mkdir(parents=True, exist_ok=True)
                        placeholder.write_text("fn main() {}\n")
        temporary.rename(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def toolchain_image(*, python_version="3.13"):
    return (
        modal.Image.from_registry(
            "nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python=python_version
        )
        .apt_install(
            "curl", "ca-certificates", "build-essential", "clang", "libclang-dev",
            "pkg-config", "ffmpeg", "libavcodec-dev", "libavformat-dev",
            "libavutil-dev", "libswscale-dev",
        )
        .run_commands(
            "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o /tmp/rustup-init.sh",
            "sh /tmp/rustup-init.sh -y --profile minimal --default-toolchain 1.88.0",
        )
        .env({"RUSTUP_TOOLCHAIN": "1.88.0", "CARGO_BUILD_JOBS": "2"})
        .apt_install("libavfilter-dev")
        .apt_install("libswresample-dev")
    )


def dependency_image(*, python_version="3.13", release=False):
    dependencies = dependency_snapshot(SOURCE) if modal.is_local() else SOURCE
    base = (
        toolchain_image(python_version=python_version)
        .add_local_dir(str(dependencies), "/workspace", copy=True)
        .workdir("/workspace")
        .run_commands("/root/.cargo/bin/cargo build -p media --lib --all-features --locked")
    )
    if release:
        base = base.run_commands(
            "/root/.cargo/bin/cargo build --release -p media --lib --all-features --locked"
        )
    return base


def source_image(*, python_version="3.13", release=False):
    return dependency_image(python_version=python_version, release=release).add_local_dir(
        str(SOURCE / "crates"), "/workspace/crates", copy=True
    )


def source_metadata():
    return json.loads((SOURCE / "source-manifest.json").read_text())
