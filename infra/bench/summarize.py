#!/usr/bin/env python3
"""Summarize validated raw workflow rows without discarding their provenance."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    directory = args.directory
    with (directory / "raw.csv").open(newline="") as source:
        rows = list(csv.DictReader(source))
    if not rows or any(row["validation"] != "passed" for row in rows):
        raise SystemExit("only a nonempty, fully validated run can be summarized")
    gpu = "tensor_ready_ns" in rows[0]
    keys = ("sampling", "export", "variant") if gpu else ("mode", "variant")
    metrics = ("tensor_ready_ns", "consumer_complete_ns", "input_retired_ns", "validation_ns") if gpu else ("end_to_end_ns", "first_result_ns")
    groups = {}
    for row in rows:
        if row["phase"] not in ("warm", "steady"):
            continue
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    summaries = []
    for group, values in sorted(groups.items()):
        result = dict(zip(keys, group))
        result["n"] = len(values)
        for metric in metrics:
            numbers = [int(row[metric]) / 1_000_000 for row in values]
            result[metric.removesuffix("_ns") + "_ms"] = {
                "median": statistics.median(numbers), "min": min(numbers), "max": max(numbers)
            }
        for key in (("outputs", "decoded_frames", "source_payload_bytes", "validation_download_bytes", "decoded_host_crossings_GPU_stage") if gpu else ("outputs", "decoded_units", "source_payload_bytes_read", "peak_host_accounted_bytes")):
            result[key] = sorted({row[key] for row in values})
        summaries.append(result)
    digests = {}
    for name in ("runner.rs", "Cargo.lock", "fixture-manifest.json", "source-manifest.json"):
        path = directory / name
        if path.exists():
            digests[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = {"measurement_rows": len(rows), "validation_passed": True,
              "kind": "gpu" if gpu else "cpu", "warm_results": summaries,
              "artifact_sha256": digests}
    manifest = directory / "source-manifest.json"
    if manifest.exists():
        result["source_sha256"] = json.loads(manifest.read_text())["sha256"]
    (directory / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"rows": len(rows), "groups": len(summaries),
                      "summary": str(directory / "summary.json")}))


if __name__ == "__main__":
    main()
