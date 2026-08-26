from __future__ import annotations

import argparse
import gzip
import json
import unicodedata
from pathlib import Path
from statistics import mean

import jiwer

from a5sv2_eval.dataset import REPORT_CONDITIONS

CONDITIONS = REPORT_CONDITIONS


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("’", "'").replace("`", "'").replace("'", "")
    text = "".join(char if char.isalnum() or char.isspace() else " " for char in text)
    return " ".join(text.split())


def corpus_counts(rows: list[dict]) -> dict:
    hits = substitutions = deletions = insertions = 0
    for row in rows:
        result = jiwer.process_words(
            normalize(row["reference_raw"]), normalize(row["prediction_raw"])
        )
        hits += result.hits
        substitutions += result.substitutions
        deletions += result.deletions
        insertions += result.insertions
    reference_words = hits + substitutions + deletions
    errors = substitutions + deletions + insertions
    return {
        "hits": hits,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "reference_words": reference_words,
        "errors": errors,
        "wer_pct": 100 * errors / reference_words,
    }


def load_rows(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def identity(row: dict) -> tuple:
    return row["dataset_revision"], row["id"], row["source_sha256"], row["reference_raw"]


row_identity = identity


def score_files(paths: list[Path]) -> list[dict]:
    systems: dict[str, dict[int, list[dict]]] = {}
    names = {}
    for path in paths:
        rows = load_rows(path)
        if not rows or any(row.get("status", "ok") != "ok" for row in rows):
            raise RuntimeError(f"Expected only successful rows in {path}")
        system_id, trial = rows[0]["system_id"], int(rows[0].get("trial", 1))
        if any(row["system_id"] != system_id or int(row.get("trial", 1)) != trial for row in rows):
            raise RuntimeError(f"Mixed system or trial in {path}")
        names[system_id] = rows[0]["system"]
        systems.setdefault(system_id, {}).setdefault(trial, []).extend(rows)

    output = []
    for system_id, trials in systems.items():
        for trial, rows in trials.items():
            keys = [identity(row) for row in rows]
            if len(keys) != len(set(keys)):
                raise RuntimeError(f"Duplicate rows for {system_id} trial {trial}")
        datasets = sorted({row["dataset_id"] for rows in trials.values() for row in rows})
        dataset_records = []
        for dataset in datasets:
            groups = {
                trial: [row for row in rows if row["dataset_id"] == dataset]
                for trial, rows in trials.items()
                if any(row["dataset_id"] == dataset for row in rows)
            }
            keys = [sorted(identity(row) for row in rows) for rows in groups.values()]
            if any(next_keys != keys[0] for next_keys in keys[1:]):
                raise RuntimeError(
                    f"Trials do not contain identical {dataset} rows for {system_id}"
                )
            conditions = sorted({row["condition"] for row in next(iter(groups.values()))})
            for condition in ["overall", *conditions] if len(conditions) > 1 else ["overall"]:
                subsets = [
                    rows
                    if condition == "overall"
                    else [r for r in rows if r["condition"] == condition]
                    for rows in groups.values()
                ]
                scored = [corpus_counts(rows) for rows in subsets]
                record = {
                    "system_id": system_id,
                    "system": names[system_id],
                    "dataset": dataset,
                    "condition": condition,
                    "trials": len(scored),
                    "samples_per_trial": len(subsets[0]),
                    "reference_words": scored[0]["reference_words"],
                    "mean_wer_pct": mean(score["wer_pct"] for score in scored),
                }
                output.append(record)
                if condition == "overall":
                    dataset_records.append(record)
        if len(dataset_records) > 1:
            trials_used = {row["trials"] for row in dataset_records}
            trials_value = next(iter(trials_used)) if len(trials_used) == 1 else "mixed"
            for condition, wer in (
                ("macro", mean(row["mean_wer_pct"] for row in dataset_records)),
                (
                    "pooled",
                    sum(row["mean_wer_pct"] * row["reference_words"] for row in dataset_records)
                    / sum(row["reference_words"] for row in dataset_records),
                ),
            ):
                output.append(
                    {
                        "system_id": system_id,
                        "system": names[system_id],
                        "dataset": "ALL",
                        "condition": condition,
                        "trials": trials_value,
                        "samples_per_trial": sum(
                            row["samples_per_trial"] for row in dataset_records
                        ),
                        "reference_words": sum(row["reference_words"] for row in dataset_records),
                        "mean_wer_pct": wer,
                    }
                )
    return output


def tsv(records: list[dict]) -> str:
    columns = [
        "system_id",
        "system",
        "dataset",
        "condition",
        "trials",
        "samples_per_trial",
        "reference_words",
        "mean_wer_pct",
    ]
    lines = ["\t".join(columns)]
    lines += [
        "\t".join(
            f"{row[column]:.6f}" if column == "mean_wer_pct" else str(row[column])
            for column in columns
        )
        for row in records
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Score saved transcript trials")
    parser.add_argument("paths", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = tsv(score_files(args.paths))
    if args.output:
        args.output.write_text(result, encoding="utf-8")
    else:
        print(result, end="")


if __name__ == "__main__":
    main()
