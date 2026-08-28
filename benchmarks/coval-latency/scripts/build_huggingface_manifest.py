#!/usr/bin/env python3
"""Create the exact file mapping for the companion Hugging Face dataset."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET_PREFIX = "results/coval-latency-v1"
SOURCES = (
    "README.md",
    "PROTOCOL.md",
    "DATA_LICENSES.md",
    "methodology.json",
    "result.schema.json",
    "plans/coval-full-947.json",
    "results/macbook-air-m4-coreml-cpu-ane/raw-results.jsonl.gz",
    "results/macbook-air-m4-coreml-cpu-ane/latency-summary.json",
    "results/macbook-air-m4-coreml-cpu-ane/wer-summary.json",
    "results/macbook-air-m4-coreml-cpu-ane/run-metadata.json",
    "results/macbook-air-m4-coreml-cpu-ane/SHA256SUMS",
)


def main() -> None:
    files = []
    for relative in SOURCES:
        path = ROOT / relative
        files.append(
            {
                "source": relative,
                "target": f"{TARGET_PREFIX}/{relative}",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
        )
    payload = {
        "schema_version": "1.0",
        "dataset_repository": "AirCaps/a5sv2-asr-benchmark-dataset",
        "target_prefix": TARGET_PREFIX,
        "files": files,
    }
    output = ROOT / "huggingface/upload-manifest.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
