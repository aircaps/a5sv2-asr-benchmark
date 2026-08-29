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

RESULT_NAMES = (
    "macbook-air-m4-coreml-cpu-ane",
    "iphone-16-coreml-cpu-ane",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    plan = json.loads((root / "plans/coval-full-947.json").read_text(encoding="utf-8"))
    plan_keys = Counter((row["dataset_id"], row["sample_id"], row["sha256"]) for row in plan["items"])
    if len(plan["items"]) != 947:
        raise RuntimeError("Expected 947 plan rows")
    expected_counts = {"stt-v1": 50, "stt-v3": 897}
    for result_name in RESULT_NAMES:
        result = root / "results" / result_name
        rows = load_rows(result / "raw-results.jsonl.gz")
        if len(rows) != 947:
            raise RuntimeError(f"{result_name}: expected 947 result rows")
        if any(row["status"] != "ok" or row["error"] is not None for row in rows):
            raise RuntimeError(f"{result_name}: release contains a failed row")
        row_keys = Counter(
            (row["dataset_id"], row["sample_id"], row["audio_sha256"]) for row in rows
        )
        if row_keys != plan_keys:
            raise RuntimeError(f"{result_name}: result identities/hashes differ from the plan")
        actual_counts = Counter(str(row["dataset_id"]) for row in rows)
        if actual_counts != expected_counts:
            raise RuntimeError(f"{result_name}: unexpected dataset counts: {actual_counts}")
        latency = json.loads((result / "latency-summary.json").read_text(encoding="utf-8"))
        if analyze(rows) != latency:
            raise RuntimeError(f"{result_name}: latency summary is not reproducible")
        wer = json.loads((result / "wer-summary.json").read_text(encoding="utf-8"))
        if score(rows) != wer:
            raise RuntimeError(f"{result_name}: WER summary is not reproducible")
        for line in (result / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
            expected, name = line.split("  ", 1)
            if sha256(result / name) != expected:
                raise RuntimeError(f"{result_name}: checksum mismatch: {name}")
        print(
            f"verified {result_name}: 947/947 rows, 0 failures, "
            "exact plan match, summaries reproducible"
        )


if __name__ == "__main__":
    main()
