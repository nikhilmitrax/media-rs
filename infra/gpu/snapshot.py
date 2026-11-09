"""Snapshot Cargo inputs and allowlisted benchmark sources, excluding build output."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[2]
destination = ROOT / "artifacts" / "gpu" / "snapshots"
destination.mkdir(parents=True, exist_ok=True)
temporary = Path(tempfile.mkdtemp(prefix="preparing-", dir=destination))
try:
    for name in ("Cargo.toml", "Cargo.lock"):
        shutil.copy2(ROOT / name, temporary / name)
    shutil.copytree(ROOT / "crates", temporary / "crates")
    shutil.copytree(ROOT / "infra" / "bench", temporary / "infra" / "bench",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    files = {}
    for path in sorted(temporary.rglob("*")):
        if path.is_file():
            files[str(path.relative_to(temporary))] = hashlib.sha256(path.read_bytes()).hexdigest()
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    (temporary / "source-manifest.json").write_text(json.dumps({"sha256": digest, "files": files}, indent=2))
    final = destination / digest
    if final.exists():
        shutil.rmtree(temporary)
    else:
        temporary.rename(final)
    print(final)
except BaseException:
    shutil.rmtree(temporary, ignore_errors=True)
    raise
