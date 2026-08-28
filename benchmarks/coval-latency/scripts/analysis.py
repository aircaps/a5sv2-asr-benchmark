#!/usr/bin/env python3
"""Deterministic latency summaries for released Coval-compatible rows."""
from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import statistics
from pathlib import Path
from typing import Callable

METRICS = (
    "time_to_first_token_seconds",
    "audio_to_final_seconds",
    "time_to_final_segment_seconds",
)
STATISTICS = ("mean", "p50", "p95")


def load_rows(path: Path) -> list[dict[str, object]]:
    opener = gzip.open if path.suffix == ".gz" else path.open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def estimator(name: str) -> Callable[[list[float]], float]:
    if name == "mean":
        return statistics.fmean
    probability = {"p50": 0.50, "p95": 0.95}[name]
    return lambda values: percentile(values, probability)


def item_bootstrap_interval(
    values: list[float], function: Callable[[list[float]], float], *, seed: str, iterations: int
) -> list[float]:
    rng = random.Random(seed)
    size = len(values)
    estimates = [
        function([values[index] for index in rng.choices(range(size), k=size)])
        for _ in range(iterations)
    ]
    return [percentile(estimates, 0.025), percentile(estimates, 0.975)]


def analyze(
    rows: list[dict[str, object]], *, iterations: int = 10_000, seed: int = 20260827
) -> dict[str, object]:
    successful = [row for row in rows if row.get("status") == "ok"]
    datasets = sorted({str(row["dataset_id"]) for row in successful})
    groups = {dataset: [row for row in successful if row["dataset_id"] == dataset] for dataset in datasets}
    groups["all_item_weighted"] = successful
    output: dict[str, object] = {}
    for group_name, group in groups.items():
        metrics: dict[str, object] = {}
        for metric in METRICS:
            values = [float(row[metric]) for row in group if row.get(metric) is not None]
            summaries: dict[str, object] = {"observations": len(values)}
            for statistic_name in STATISTICS:
                function = estimator(statistic_name)
                summaries[statistic_name] = {
                    "estimate_seconds": function(values),
                    "item_bootstrap_95ci_seconds": item_bootstrap_interval(
                        values,
                        function,
                        seed=f"{seed}:{group_name}:{metric}:{statistic_name}",
                        iterations=iterations,
                    ),
                }
            summaries["min_seconds"] = min(values)
            summaries["max_seconds"] = max(values)
            metrics[metric] = summaries
        output[group_name] = {
            "successful_rows": len(group),
            "failed_rows": sum(
                row.get("status") != "ok"
                and (group_name == "all_item_weighted" or row.get("dataset_id") == group_name)
                for row in rows
            ),
            "metrics": metrics,
        }
    return {
        "schema_version": "1.0",
        "logical_trials": len({row["run_id"] for row in rows}),
        "execution_sessions": len({row["execution_session_id"] for row in rows}),
        "bootstrap": {
            "method": "percentile bootstrap over items within each reported group",
            "iterations": iterations,
            "seed": seed,
            "confidence": 0.95,
            "limitation": (
                "Intervals quantify item-sampling uncertainty for this one logical device trial; "
                "they do not estimate run-to-run, thermal, or device-to-device variance."
            ),
        },
        "by_dataset": output,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()
    print(
        json.dumps(
            analyze(load_rows(args.results), iterations=args.iterations, seed=args.seed),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
