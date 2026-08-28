#!/usr/bin/env python3
"""Build a deterministic, publication-safe latency result bundle."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

from analysis import analyze
from score import score

LOGICAL_RUN_ID = "macbook-air-m4-coreml-cpu-ane-trial-1"
PUBLIC_VARIANT = "a5sv2-coreml-cpu-ane"
PUBLIC_FIELDS = (
    "schema_version",
    "method_id",
    "trial",
    "dataset_id",
    "dataset_version",
    "sample_id",
    "audio_path",
    "audio_sha256",
    "reference",
    "hypothesis",
    "speech_end_offset_ms",
    "effective_audio_seconds",
    "time_to_first_token_seconds",
    "audio_to_final_seconds",
    "time_to_final_segment_seconds",
    "started_at_utc",
    "status",
    "error",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_gzip_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
        with io.TextIOWrapper(compressed, encoding="utf-8") as text:
            for row in rows:
                text.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def session_metadata(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    sessions = []
    for session_id in sorted({str(row["execution_session_id"]) for row in rows}):
        group = [row for row in rows if row["execution_session_id"] == session_id]
        started = min(datetime.fromisoformat(str(row["started_at_utc"])) for row in group)
        ended = max(
            datetime.fromisoformat(str(row["started_at_utc"]))
            + timedelta(seconds=float(row["audio_to_final_seconds"]))
            for row in group
        )
        sessions.append(
            {
                "execution_session_id": session_id,
                "successful_rows": len(group),
                "started_at_utc": started.isoformat(),
                "ended_at_utc": ended.isoformat(),
            }
        )
    return sorted(sessions, key=lambda item: str(item["started_at_utc"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--private-run-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_rows = load_jsonl(args.input)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    plan_items = plan["items"]
    if len(source_rows) != 947 or len(plan_items) != 947:
        raise RuntimeError("Release requires exactly 947 source rows and 947 plan items")
    if any(row.get("status") != "ok" for row in source_rows):
        raise RuntimeError("Release contains a failed source row")
    plan_by_key = {(item["dataset_id"], item["sample_id"]): item for item in plan_items}
    if len(plan_by_key) != 947:
        raise RuntimeError("Plan contains duplicate identities")

    public_rows: list[dict[str, object]] = []
    for source in source_rows:
        key = (source["dataset_id"], source["sample_id"])
        item = plan_by_key.get(key)
        if item is None or item["sha256"] != source["audio_sha256"]:
            raise RuntimeError(f"Source row does not match plan: {key}")
        public = {field: source.get(field) for field in PUBLIC_FIELDS}
        public.update(
            {
                "schema_version": "1.1",
                "run_id": LOGICAL_RUN_ID,
                "execution_session_id": source["run_id"],
                "variant": PUBLIC_VARIANT,
            }
        )
        public_rows.append(public)
    if Counter((row["dataset_id"], row["sample_id"]) for row in public_rows) != Counter(
        plan_by_key.keys()
    ):
        raise RuntimeError("Released identities differ from the plan")

    args.output.mkdir(parents=True, exist_ok=True)
    raw_path = args.output / "raw-results.jsonl.gz"
    write_gzip_jsonl(raw_path, public_rows)
    write_json(args.output / "latency-summary.json", analyze(public_rows))
    write_json(args.output / "wer-summary.json", score(public_rows))

    private = json.loads(args.private_run_metadata.read_text(encoding="utf-8"))
    metadata = {
        "schema_version": "1.0",
        "result_id": LOGICAL_RUN_ID,
        "system": "A5Sv2",
        "variant": PUBLIC_VARIANT,
        "coverage": "947/947",
        "logical_trials": 1,
        "concurrency": 1,
        "inference_disclosure": (
            "Model weights and inference implementation are private and are not included. "
            "Raw predictions, primitive timings, artifact attestations, and scoring code are public."
        ),
        "compute_policy": (
            "Apple Core ML with CPU and Neural Engine eligible. Core ML controls actual operation placement."
        ),
        "environment": {
            **private["environment"],
            "power_state": "AC Power; battery 100%; charged",
            "computer": "MacBook Air (13-inch, M4, 2025)",
            "model_identifier": "Mac16,12",
            "part_number": "MW123LL/A",
            "cpu_cores": 10,
            "cpu_performance_cores": 4,
            "cpu_efficiency_cores": 6,
            "gpu_cores": 8,
            "neural_engine_cores": 16,
            "unified_memory_gb": 16,
            "internal_storage": "256 GB Apple SSD (APPLE SSD AP0256Z)",
            "os_build": "25B78",
            "year_introduced": 2025,
        },
        "plan_sha256": sha256(args.plan),
        "raw_results_sha256": sha256(raw_path),
        "artifact_attestations": {
            "model_bundle_sha256": private["private_artifacts"]["model_sha256"],
            "runner_sha256": private["private_artifacts"]["worker_sha256"],
        },
        "execution_sessions": session_metadata(public_rows),
        "interruption_disclosure": (
            "The logical trial was stopped at the user's request after 12 successful items and "
            "resumed with the same plan, hardware, model bundle, runner, compute policy, and output. "
            "Completed items were not rerun; every row retains its execution-session identifier."
        ),
        "controls": {
            "power": "AC power; battery reported 100% at preflight and completion",
            "thermal_telemetry": "not recorded continuously",
            "warmup": "one untimed warmup per execution session",
            "outlier_removal": "none",
            "failed_rows": 0,
        },
    }
    write_json(args.output / "run-metadata.json", metadata)

    checksum_paths = [
        raw_path,
        args.output / "latency-summary.json",
        args.output / "wer-summary.json",
        args.output / "run-metadata.json",
    ]
    (args.output / "SHA256SUMS").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in checksum_paths),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
