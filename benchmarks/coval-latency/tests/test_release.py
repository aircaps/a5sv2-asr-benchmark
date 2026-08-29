from __future__ import annotations

import gzip
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULT_NAMES = (
    "macbook-air-m4-coreml-cpu-ane",
    "iphone-16-coreml-cpu-ane",
)


def load_rows(result_name: str) -> list[dict[str, object]]:
    result = ROOT / "results" / result_name
    with gzip.open(result / "raw-results.jsonl.gz", "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


@pytest.mark.parametrize("result_name", RESULT_NAMES)
def test_complete_exact_plan(result_name: str) -> None:
    rows = load_rows(result_name)
    plan = json.loads((ROOT / "plans/coval-full-947.json").read_text(encoding="utf-8"))
    assert len(rows) == len(plan["items"]) == 947
    assert Counter(row["dataset_id"] for row in rows) == {"stt-v1": 50, "stt-v3": 897}
    assert all(row["status"] == "ok" and row["error"] is None for row in rows)
    result_keys = {
        (row["dataset_id"], row["sample_id"], row["audio_sha256"]) for row in rows
    }
    plan_keys = {(row["dataset_id"], row["sample_id"], row["sha256"]) for row in plan["items"]}
    assert result_keys == plan_keys


@pytest.mark.parametrize("result_name", RESULT_NAMES)
def test_primitive_timing_identity(result_name: str) -> None:
    for row in load_rows(result_name):
        expected = max(
            0.0,
            float(row["audio_to_final_seconds"])
            - float(row["speech_end_offset_ms"]) / 1000.0,
        )
        assert math.isclose(
            float(row["time_to_final_segment_seconds"]), expected, abs_tol=1e-12
        )


@pytest.mark.parametrize("result_name", RESULT_NAMES)
def test_pinned_plan_and_result_hashes(result_name: str) -> None:
    plan_hash = hashlib.sha256((ROOT / "plans/coval-full-947.json").read_bytes()).hexdigest()
    assert plan_hash == "439ef68e1fef427e346981dbec82a3da4183aeb560309e82f79bd5f5a886d333"
    result = ROOT / "results" / result_name
    metadata = json.loads((result / "run-metadata.json").read_text(encoding="utf-8"))
    assert metadata["plan_sha256"] == plan_hash
    assert metadata["coverage"] == "947/947"


@pytest.mark.parametrize(
    ("result_name", "expected_ttft_p50", "expected_final_p50"),
    (
        (
            "macbook-air-m4-coreml-cpu-ane",
            1.649057542,
            0.045497124999999805,
        ),
        (
            "iphone-16-coreml-cpu-ane",
            1.656328667,
            0.05001012499999957,
        ),
    ),
)
def test_headline_summary_values(
    result_name: str, expected_ttft_p50: float, expected_final_p50: float
) -> None:
    result = ROOT / "results" / result_name
    latency = json.loads((result / "latency-summary.json").read_text(encoding="utf-8"))
    combined = latency["by_dataset"]["all_item_weighted"]["metrics"]
    assert (
        combined["time_to_first_token_seconds"]["p50"]["estimate_seconds"]
        == expected_ttft_p50
    )
    assert combined["time_to_final_segment_seconds"]["p50"]["estimate_seconds"] == expected_final_p50
    wer = json.loads((result / "wer-summary.json").read_text(encoding="utf-8"))
    assert wer["successful_rows"] == 947
    assert wer["failed_rows"] == 0
