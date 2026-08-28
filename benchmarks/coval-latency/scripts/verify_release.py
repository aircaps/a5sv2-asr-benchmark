#!/usr/bin/env python3
"""Offline integrity and reproducibility check for the staged release."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from analysis import analyze, load_rows
from score import score


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    result = root / "results/macbook-air-m4-coreml-cpu-ane"
    rows = load_rows(result / "raw-results.jsonl.gz")
    plan = json.loads((root / "plans/coval-full-947.json").read_text(encoding="utf-8"))
    if len(rows) != 947 or len(plan["items"]) != 947:
        raise RuntimeError("Expected 947 result and plan rows")
    if any(row["status"] != "ok" or row["error"] is not None for row in rows):
        raise RuntimeError("Release contains a failed row")
    row_keys = Counter((row["dataset_id"], row["sample_id"], row["audio_sha256"]) for row in rows)
    plan_keys = Counter((row["dataset_id"], row["sample_id"], row["sha256"]) for row in plan["items"])
    if row_keys != plan_keys:
        raise RuntimeError("Result identities/hashes differ from the plan")
    expected_counts = {"stt-v1": 50, "stt-v3": 897}
    actual_counts = Counter(str(row["dataset_id"]) for row in rows)
    if actual_counts != expected_counts:
        raise RuntimeError(f"Unexpected dataset counts: {actual_counts}")
    if analyze(rows) != json.loads((result / "latency-summary.json").read_text(encoding="utf-8")):
        raise RuntimeError("Latency summary is not reproducible")
    if score(rows) != json.loads((result / "wer-summary.json").read_text(encoding="utf-8")):
        raise RuntimeError("WER summary is not reproducible")
    for line in (result / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        if sha256(result / name) != expected:
            raise RuntimeError(f"Checksum mismatch: {name}")
    print("verified: 947/947 rows, 0 failures, exact plan match, summaries reproducible")


if __name__ == "__main__":
    main()
