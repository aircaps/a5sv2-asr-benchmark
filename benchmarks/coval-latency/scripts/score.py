#!/usr/bin/env python3
"""Reproduce Coval WER and the parent repository's corpus-WER convention."""
from __future__ import annotations

import argparse
import json
import statistics
import unicodedata
from pathlib import Path

import jiwer
from whisper_normalizer.english import EnglishTextNormalizer

from analysis import load_rows

_coval_normalizer = EnglishTextNormalizer()


def a5sv2_normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("’", "'").replace("`", "'").replace("'", "")
    text = "".join(character if character.isalnum() or character.isspace() else " " for character in text)
    return " ".join(text.split())


def counts(rows: list[dict[str, object]]) -> dict[str, int | float]:
    hits = substitutions = deletions = insertions = 0
    for row in rows:
        result = jiwer.process_words(
            a5sv2_normalize(str(row["reference"])),
            a5sv2_normalize(str(row["hypothesis"])),
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


def score(rows: list[dict[str, object]]) -> dict[str, object]:
    successful = [row for row in rows if row.get("status") == "ok"]
    datasets = sorted({str(row["dataset_id"]) for row in successful})
    groups = {dataset: [row for row in successful if row["dataset_id"] == dataset] for dataset in datasets}

    coval: dict[str, object] = {}
    standard: dict[str, object] = {}
    for dataset, group in groups.items():
        item_wer = [
            100
            * jiwer.wer(
                _coval_normalizer(str(row["reference"])),
                _coval_normalizer(str(row["hypothesis"])),
            )
            for row in group
        ]
        coval[dataset] = {
            "rows": len(group),
            "mean_item_wer_pct": statistics.fmean(item_wer),
            "p50_item_wer_pct": statistics.median(item_wer),
        }
        standard[dataset] = {"rows": len(group), **counts(group)}

    all_item_wer = [
        100
        * jiwer.wer(
            _coval_normalizer(str(row["reference"])),
            _coval_normalizer(str(row["hypothesis"])),
        )
        for row in successful
    ]
    coval["all_item_weighted"] = {
        "rows": len(successful),
        "mean_item_wer_pct": statistics.fmean(all_item_wer),
        "p50_item_wer_pct": statistics.median(all_item_wer),
        "macro_mean_item_wer_pct": statistics.fmean(
            float(coval[dataset]["mean_item_wer_pct"]) for dataset in datasets
        ),
    }
    pooled = counts(successful)
    standard["all_pooled"] = {"rows": len(successful), **pooled}
    standard["all_macro"] = {
        "datasets": len(datasets),
        "wer_pct": statistics.fmean(float(standard[dataset]["wer_pct"]) for dataset in datasets),
    }
    return {
        "schema_version": "1.0",
        "successful_rows": len(successful),
        "failed_rows": len(rows) - len(successful),
        "coval_compatible": {
            "normalization": "whisper-normalizer EnglishTextNormalizer",
            "normalization_version": "2",
            "aggregation": "arithmetic mean of per-item WER percentages",
            "results": coval,
        },
        "a5sv2_repository_standard": {
            "normalization": (
                "Unicode NFKC; lowercase; remove apostrophes; replace other punctuation with "
                "spaces; collapse whitespace"
            ),
            "aggregation": "corpus edit counts; pooled and unweighted dataset macro reported",
            "results": standard,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    print(json.dumps(score(load_rows(args.results)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
