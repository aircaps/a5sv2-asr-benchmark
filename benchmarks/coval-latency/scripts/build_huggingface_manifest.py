#!/usr/bin/env python3
"""Create the exact file mapping for the companion Hugging Face dataset."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET_PREFIX = "results/coval-latency-v1"
COMMON_SOURCES = (
    "README.md",
    "PROTOCOL.md",
    "DATA_LICENSES.md",
    "methodology.json",
    "result.schema.json",
    "plans/coval-full-947.json",
)
RESULT_NAMES = (
    "macbook-air-m4-coreml-cpu-ane",
    "iphone-16-coreml-cpu-ane",
)
RESULT_FILES = (
    "raw-results.jsonl.gz",
    "latency-summary.json",
    "wer-summary.json",
    "run-metadata.json",
    "SHA256SUMS",
)


def main() -> None:
    files = []
    sources = list(COMMON_SOURCES)
    sources.extend(
        f"results/{result_name}/{name}"
        for result_name in RESULT_NAMES
        for name in RESULT_FILES
    )
    for relative in sources:
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
