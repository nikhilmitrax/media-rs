#!/usr/bin/env python3
"""Generate synthetic CFR H264 outside GPU billing and archive its exact command."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/bench/gpu-fixtures"))
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "gpu6s.mp4"
    command = [args.ffmpeg, "-v", "error", "-nostdin", "-y", "-f", "lavfi", "-i",
               "testsrc2=size=320x180:rate=30:duration=6", "-vf",
               "scale=in_color_matrix=bt601:out_color_matrix=bt709:out_range=tv",
               "-an", "-c:v", "libx264", "-threads", "1", "-preset", "veryfast", "-crf", "22",
               "-bf", "2", "-g", "30", "-x264-params", "b-adapt=0", "-pix_fmt", "yuv420p",
               "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
               "-color_range", "tv", "-chroma_sample_location", "left", str(path)]
    subprocess.run(command, check=True, capture_output=True)
    metadata = {"generator": "infra/bench/generate_gpu_fixture.py", "command": command,
                "ffmpeg": subprocess.check_output([args.ffmpeg, "-version"], text=True).splitlines()[0],
                "file": path.name, "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "description": "6s CFR30 H264320x180 BT709 limited/left, forced B frames; synthetic testsrc2"}
    (args.output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
