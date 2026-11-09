"""Verify pinned public fixtures; downloading requires the explicit --fetch flag."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path(__file__).with_name("adoption-corpus.json")


def verify(path, record):
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    if size != record["bytes"] or digest.hexdigest() != record["sha256"]:
        raise ValueError(f"fixture does not match the pinned manifest: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--directory", type=Path, default=ROOT / "artifacts/adoption-20261103/corpus")
    arguments = parser.parse_args()
    records = json.loads(MANIFEST.read_text())["files"]
    arguments.directory.mkdir(parents=True, exist_ok=True)
    for record in records:
        path = arguments.directory / record["file"]
        if not path.exists():
            if not arguments.fetch:
                raise SystemExit(f"missing fixture: {path}; use --fetch to download the pinned public corpus")
            temporary = tempfile.NamedTemporaryFile(dir=arguments.directory, delete=False)
            staging = Path(temporary.name)
            try:
                with temporary:
                    request = urllib.request.Request(record["url"], headers={"User-Agent": "media-rs-adoption-validation/1.0"})
                    with urllib.request.urlopen(request, timeout=45) as response:
                        size = 0
                        while chunk := response.read(1024 * 1024):
                            size += len(chunk)
                            if size > record["bytes"]:
                                raise ValueError("fixture exceeds its pinned byte bound")
                            temporary.write(chunk)
                verify(staging, record)
                staging.replace(path)
            finally:
                staging.unlink(missing_ok=True)
        verify(path, record)
        print(json.dumps({"file": str(path), "bytes": record["bytes"], "sha256": record["sha256"], "verified": True}))


if __name__ == "__main__":
    main()
