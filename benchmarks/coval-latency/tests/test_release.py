from __future__ import annotations

import gzip
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/macbook-air-m4-coreml-cpu-ane"


def load_rows() -> list[dict[str, object]]:
    with gzip.open(RESULT / "raw-results.jsonl.gz", "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def test_complete_exact_plan() -> None:
    rows = load_rows()
    plan = json.loads((ROOT / "plans/coval-full-947.json").read_text(encoding="utf-8"))
    assert len(rows) == len(plan["items"]) == 947
    assert Counter(row["dataset_id"] for row in rows) == {"stt-v1": 50, "stt-v3": 897}
    assert all(row["status"] == "ok" and row["error"] is None for row in rows)
    result_keys = {
        (row["dataset_id"], row["sample_id"], row["audio_sha256"]) for row in rows
    }
    plan_keys = {(row["dataset_id"], row["sample_id"], row["sha256"]) for row in plan["items"]}
    assert result_keys == plan_keys


def test_primitive_timing_identity() -> None:
    for row in load_rows():
        expected = max(
            0.0,
            float(row["audio_to_final_seconds"])
            - float(row["speech_end_offset_ms"]) / 1000.0,
        )
        assert math.isclose(
            float(row["time_to_final_segment_seconds"]), expected, abs_tol=1e-12
        )


def test_pinned_plan_and_result_hashes() -> None:
    plan_hash = hashlib.sha256((ROOT / "plans/coval-full-947.json").read_bytes()).hexdigest()
    assert plan_hash == "439ef68e1fef427e346981dbec82a3da4183aeb560309e82f79bd5f5a886d333"
    metadata = json.loads((RESULT / "run-metadata.json").read_text(encoding="utf-8"))
    assert metadata["plan_sha256"] == plan_hash
    assert metadata["coverage"] == "947/947"


def test_headline_summary_values() -> None:
    latency = json.loads((RESULT / "latency-summary.json").read_text(encoding="utf-8"))
    combined = latency["by_dataset"]["all_item_weighted"]["metrics"]
    assert combined["time_to_first_token_seconds"]["p50"]["estimate_seconds"] == 1.649057542
    assert (
        combined["time_to_final_segment_seconds"]["p50"]["estimate_seconds"]
        == 0.045497124999999805
    )
    wer = json.loads((RESULT / "wer-summary.json").read_text(encoding="utf-8"))
    assert wer["successful_rows"] == 947
    assert wer["failed_rows"] == 0
