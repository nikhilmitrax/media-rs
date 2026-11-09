#!/usr/bin/env python3
"""Generate local synthetic workflow fixtures and SHA-256/environment provenance."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import struct
import subprocess
import wave
import zlib


def chunk(kind, payload):
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/bench/fixtures"))
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    width, height = 768, 512
    rows = bytearray()
    for y in range(height):
        rows.append(0)
        for x in range(width):
            rows.extend(((7 * x + 3 * y) % 256, (x ^ y) & 255, (13 * x - 5 * y) % 256))
    # PNG eXIf stores a TIFF directory: orientation 6 rotates stored pixels 90deg.
    exif = b"II" + struct.pack("<HIH", 42, 8, 1) + struct.pack("<HHII", 0x112, 3, 1, 6) + struct.pack("<I", 0)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += chunk(b"sRGB", b"\x01") + chunk(b"eXIf", exif) + chunk(b"IDAT", zlib.compress(rows, 6)) + chunk(b"IEND", b"")
    (args.output / "oriented.png").write_bytes(png)
    with wave.open(str(args.output / "mono48k.wav"), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(48000)
        samples = (round(12000 * math.sin(2 * math.pi * 440 * i / 48000) + 3000 * math.sin(2 * math.pi * 1200 * i / 48000)) for i in range(4 * 48000))
        output.writeframes(b"".join(struct.pack("<h", sample) for sample in samples))
    command = [args.ffmpeg, "-v", "error", "-nostdin", "-y", "-f", "lavfi", "-i",
               "testsrc2=size=320x180:rate=30:duration=6", "-itsoffset", "0.25", "-f", "lavfi", "-i",
               "sine=frequency=440:sample_rate=48000:duration=6", "-vf",
               "scale=in_color_matrix=bt601:out_color_matrix=bt709:out_range=tv",
               "-c:v", "libx264", "-threads:v", "1", "-preset", "veryfast", "-crf", "22", "-bf", "2", "-g", "30",
               "-pix_fmt", "yuv420p", "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
               "-color_range", "tv", "-chroma_sample_location", "left", "-c:a", "aac", "-threads:a", "1", "-b:a", "96000",
               str(args.output / "av6s.mp4")]
    subprocess.run(command, check=True, capture_output=True)
    files = {}
    for name in ["oriented.png", "mono48k.wav", "av6s.mp4"]:
        data = (args.output / name).read_bytes()
        files[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    metadata = {"generator": "infra/bench/generate_fixtures.py", "python": platform.python_version(),
                "platform": platform.platform(), "zlib": zlib.ZLIB_RUNTIME_VERSION,
                "ffmpeg": subprocess.check_output([args.ffmpeg, "-version"], text=True).splitlines()[0],
                "video_command": command, "files": files,
                "description": {"image": "768x512 RGB8 PNG with sRGB and EXIF orientation6",
                                "audio": "4s mono PCM16 WAV at48000Hz, 440+1200Hz tones",
                                "video": "6s CFR30 H264320x180 BT709 limited + AAC48k delayed250ms"}}
    (args.output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(args.output / "manifest.json")


if __name__ == "__main__":
    main()
