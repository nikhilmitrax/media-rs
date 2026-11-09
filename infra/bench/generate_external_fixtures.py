#!/usr/bin/env python3
"""Generate real 1080p/4K H264/HEVC CFR fixtures with independently readable frame IDs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import uuid


PROFILES = {
    "bt709-left": {"signal": "bt709", "chroma": "left", "cicp": 1, "chroma_code": 0},
    "bt601-center": {"signal": "smpte170m", "chroma": "center", "cicp": 6, "chroma_code": 1},
}

def filters(pattern, profile="bt709-left"):
    # The 224x224 center crop contains an eight-bit ordinal barcode. Each bit is
    # a 16x16 neutral-chroma square; validation reads its center, far from edges.
    marker = "between(X,(W-224)/2+32,(W-224)/2+159)*between(Y,(H-224)/2+32,(H-224)/2+47)"
    bits = "if(bitand(N,pow(2,floor((X-((W-224)/2+32))/16))),235,16)"
    chroma_marker = "between(X,(W-112)/2+16,(W-112)/2+79)*between(Y,(H-112)/2+16,(H-112)/2+23)"
    background = "lum(X,Y)" if pattern == "color" else "16+mod(floor(X/16)+floor(Y/16)+N,220)"
    cb = "cb(X,Y)" if pattern == "color" else "128"
    cr = "cr(X,Y)" if pattern == "color" else "128"
    signal = PROFILES[profile]["signal"]
    conversion = ("scale=in_color_matrix=bt601:out_color_matrix=bt709:out_range=tv," if profile == "bt709-left" else
                  "scale=out_color_matrix=bt601:out_range=tv:out_h_chr_pos=128:out_v_chr_pos=128,format=yuv420p,")
    return (conversion +
            f"geq=lum='if({marker},{bits},{background})':"
            f"cb='if({chroma_marker},128,{cb})':cr='if({chroma_marker},128,{cr})',"
            f"setparams=range=limited:color_primaries={signal}:color_trc={signal}:colorspace={signal}")


def verify_fixture(ffmpeg, path):
    """Independent software decode proves barcode identity, including late frames."""
    selected = [0, 1, 63, 141, 179]
    expression = "+".join(f"eq(n,{index})" for index in selected)
    raw = subprocess.check_output([ffmpeg, "-v", "error", "-nostdin", "-i", str(path),
                                   "-vf", f"select='{expression}',crop=224:224", "-fps_mode", "passthrough",
                                   "-pix_fmt", "rgb24", "-f", "rawvideo", "-"])
    frame_bytes = 224 * 224 * 3
    if len(raw) != len(selected) * frame_bytes:
        raise RuntimeError(f"reference frame count mismatch: {path}")
    actual = []
    for frame in range(len(selected)):
        ordinal = 0
        for bit in range(8):
            position = frame * frame_bytes + (40 * 224 + 40 + 16 * bit) * 3
            ordinal |= int(raw[position] > 128) << bit
        actual.append(ordinal)
    if actual != selected:
        raise RuntimeError(f"reference barcode {actual} != {selected}: {path}")
    return actual

def color_metadata(ffprobe, path, decoded=False):
    level = "frame" if decoded else "stream"
    command = [ffprobe, "-v", "error", "-select_streams", "v:0"]
    if decoded:
        command += ["-read_intervals", "%+#1"]
    command += ["-show_entries", f"{level}=color_range,color_space,color_transfer,color_primaries,chroma_location",
                "-of", "json", str(path)]
    records = json.loads(subprocess.check_output(command))[level + "s"]
    if not records:
        raise RuntimeError(f"missing {level} metadata: {path}")
    return {key: records[0].get(key) for key in
            ("color_range", "color_space", "color_transfer", "color_primaries", "chroma_location")}


def verify_metadata(ffprobe, path, profile="bt709-left"):
    signal = PROFILES[profile]["signal"]
    expected = {"color_range": "tv", "color_space": signal, "color_transfer": signal,
                "color_primaries": signal, "chroma_location": PROFILES[profile]["chroma"]}
    result = {level: color_metadata(ffprobe, path, level == "frame") for level in ("stream", "frame")}
    if any(record != expected for record in result.values()):
        raise RuntimeError(f"source metadata {result} != {expected}: {path}")
    return result


def frame_fingerprints(ffmpeg, path):
    # The stream is already native YUV420P8. Hash every decoded plane byte and
    # display timestamp without RGB conversion, not just a few sampled markers.
    raw = subprocess.check_output([ffmpeg, "-v", "error", "-nostdin", "-threads", "1", "-i", str(path),
                                   "-map", "0:v:0", "-an", "-c:v", "rawvideo", "-threads", "1",
                                   "-pix_fmt", "yuv420p", "-f", "framemd5", "-hash", "sha256", "-"])
    records = [line for line in raw.decode().splitlines() if line and not line.startswith("#")]
    return {"frames": len(records), "sha256": hashlib.sha256("\n".join(records).encode()).hexdigest()}


def patch_metadata(args):
    manifest_path = args.output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    originals = args.output / "before-metadata-patch"
    originals.mkdir(exist_ok=True)
    for fixture in manifest["files"]:
        path = args.output / fixture["file"]
        original_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if original_sha != fixture["sha256"]:
            raise RuntimeError(f"manifest source changed before metadata patch: {path}")
        profile = fixture.get("color_profile", manifest.get("color_profile", "bt709-left"))
        spec = PROFILES[profile]
        try:
            verified = verify_metadata(args.ffprobe, path, profile)
            fixture["verified_color"] = verified
            print("metadata already verified", path.name, flush=True)
            continue
        except RuntimeError:
            pass
        before_metadata = {level: color_metadata(args.ffprobe, path, level == "frame") for level in ("stream", "frame")}
        before = frame_fingerprints(args.ffmpeg, path)
        temporary = args.output / f"metadata-{uuid.uuid4().hex}.mp4"
        code = spec["cicp"]
        bsf = fixture["codec"] + f"_metadata=colour_primaries={code}:transfer_characteristics={code}:matrix_coefficients={code}:video_full_range_flag=0:chroma_sample_loc_type={spec['chroma_code']}"
        command = [args.ffmpeg, "-v", "error", "-nostdin", "-n", "-i", str(path), "-map", "0:v:0", "-c", "copy",
                   "-bsf:v", bsf, "-color_primaries", spec["signal"], "-color_trc", spec["signal"], "-colorspace", spec["signal"],
                   "-color_range", "tv", "-movflags", "+write_colr", str(temporary)]
        try:
            subprocess.run(command, check=True, capture_output=True)
            after = frame_fingerprints(args.ffmpeg, temporary)
            if before != after or before["frames"] != fixture["frames"]:
                raise RuntimeError(f"metadata patch changed decoded pixels/count/timing: {path}: {before} != {after}")
            verified = verify_metadata(args.ffprobe, temporary, profile)
            ids = verify_fixture(args.ffmpeg, temporary)
            destination = originals / path.name
            if destination.exists():
                raise RuntimeError(f"original fixture backup already exists: {destination}")
            shutil.copy2(path, destination)
            new_sha = hashlib.sha256(temporary.read_bytes()).hexdigest()
            fixture["metadata_patch"] = {"original_sha256": original_sha, "patched_sha256": new_sha,
                                         "command": command, "before_color": before_metadata,
                                         "decoded_yuv_frames": before, "decoded_yuv_unchanged": True}
            temporary.replace(path)
            fixture.update(sha256=new_sha, bytes=path.stat().st_size, verified_color=verified, reference_frame_ids=ids)
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            print("patched and verified", path.name, before, flush=True)
        finally:
            temporary.unlink(missing_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--profile", choices=tuple(PROFILES), default="bt709-left")
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg")
    parser.add_argument("--resolutions", default="1080p,4k")
    parser.add_argument("--codecs", default="h264,hevc")
    parser.add_argument("--patterns", default="gray,color")
    parser.add_argument("--seconds", type=int, default=6)
    parser.add_argument("--ffprobe", default=shutil.which("ffprobe") or "/opt/homebrew/bin/ffprobe")
    parser.add_argument("--patch-metadata", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.seconds < 6 or args.seconds > 8:
        parser.error("benchmark frame IDs/requests require six to eight seconds")
    if args.output is None:
        args.output = Path("artifacts/bench/external-fixtures" if args.profile == "bt709-left" else "artifacts/bench/external-fixtures-bt601-center")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.patch_metadata:
        patch_metadata(args)
        return
    if args.verify_only:
        manifest_path = args.output / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for fixture in manifest["files"]:
            fixture["reference_frame_ids"] = verify_fixture(args.ffmpeg, args.output / fixture["file"])
            fixture["verified_color"] = verify_metadata(args.ffprobe, args.output / fixture["file"], fixture.get("color_profile", manifest.get("color_profile", "bt709-left")))
            print("verified", fixture["file"], fixture["reference_frame_ids"], flush=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        return
    manifest = {"generator": "infra/bench/generate_external_fixtures.py", "color_profile": args.profile, "fps": [30, 1],
                "seconds": args.seconds, "crop": [224, 224], "marker": {"x": 40, "y": 40, "bit_stride": 16},
                "ffmpeg": subprocess.check_output([args.ffmpeg, "-version"], text=True).splitlines()[0],
                "files": []}
    for resolution in args.resolutions.split(","):
        width, height = {"1080p": (1920, 1080), "4k": (3840, 2160)}[resolution]
        for codec in args.codecs.split(","):
            for pattern in args.patterns.split(","):
                suffix = "" if args.profile == "bt709-left" else "-bt601-center"
                path = args.output / f"{resolution}-{codec}-{pattern}{suffix}.mp4"
                spec = PROFILES[args.profile]
                source = "testsrc2" if args.profile == "bt709-left" else "testsrc"
                command = [args.ffmpeg, "-v", "error", "-nostdin", "-y", "-f", "lavfi", "-i",
                           f"{source}=size={width}x{height}:rate=30:duration={args.seconds}", "-vf", filters(pattern, args.profile),
                           "-an", "-c:v", "libx264" if codec == "h264" else "libx265", "-threads", "1",
                           "-preset", "veryfast", "-crf", "18", "-bf", "2", "-g", "30", "-pix_fmt", "yuv420p"]
                if codec == "h264":
                    command += ["-x264-params", "b-adapt=0:scenecut=0"]
                else:
                    command += ["-x265-params", "pools=none:frame-threads=1:wpp=0:b-adapt=0:scenecut=0:log-level=error"]
                command += ["-color_primaries", spec["signal"], "-color_trc", spec["signal"], "-colorspace", spec["signal"],
                            "-color_range", "tv", "-chroma_sample_location", spec["chroma"], str(path)]
                subprocess.run(command, check=True, capture_output=True)
                manifest["files"].append({"file": path.name, "resolution": resolution, "width": width, "height": height,
                                          "codec": codec, "pattern": pattern, "color_profile": args.profile, "frames": args.seconds * 30,
                                          "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                          "command": command, "reference_frame_ids": verify_fixture(args.ffmpeg, path),
                                          "verified_color": verify_metadata(args.ffprobe, path, args.profile)})
                (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
                print(path, flush=True)


if __name__ == "__main__":
    main()
