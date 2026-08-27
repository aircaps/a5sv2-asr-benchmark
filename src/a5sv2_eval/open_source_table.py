from __future__ import annotations

import argparse
import csv
from pathlib import Path

from a5sv2_eval.score import score_files


OPEN_SOURCE_SYSTEMS = {
    "a5sv2": "A5Sv2",
    "nvidia_nemotron_3_asr_streaming_0_6b": "NVIDIA Nemotron 3 ASR Streaming 0.6B",
    "mistral_voxtral_mini_transcribe_realtime_2602": (
        "Mistral Voxtral Mini Transcribe Realtime 2602"
    ),
    "kyutai_stt_2_6b_en": "Kyutai STT 2.6B English",
    "whisper_large_v3_ufal_streaming": "Whisper Large V3 (UFAL Whisper-Streaming)",
}
CORPUS_COLUMNS = {
    "AirCaps/mega-asr-noise-a5sv2": "mega_asr_wer_pct",
    "AMI": "ami_wer_pct",
    "DiPCo": "dipco_wer_pct",
    "NOTSOFAR": "notsofar_wer_pct",
}
MARKER_START = "<!-- OPEN_SOURCE_TABLE_START -->"
MARKER_END = "<!-- OPEN_SOURCE_TABLE_END -->"


def baseline(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    row = next(row for row in rows if row["system_id"] == "a5sv2")
    return {key: float(value) if key.endswith("wer_pct") else value for key, value in row.items()}


def build(paths: list[Path], baseline_csv: Path) -> list[dict]:
    rows = [baseline(baseline_csv)]
    records = score_files(paths)
    by_system: dict[str, list[dict]] = {}
    for record in records:
        by_system.setdefault(record["system_id"], []).append(record)
    expected = set(OPEN_SOURCE_SYSTEMS) - {"a5sv2"}
    if set(by_system) != expected:
        raise RuntimeError(f"Expected result systems {sorted(expected)}, found {sorted(by_system)}")
    for system_id, system_records in by_system.items():
        corpus = {
            record["dataset"]: record
            for record in system_records
            if record["condition"] == "overall"
        }
        if set(corpus) != set(CORPUS_COLUMNS):
            raise RuntimeError(f"{system_id} does not contain exactly the four benchmark corpora")
        aggregates = {
            record["condition"]: record for record in system_records if record["dataset"] == "ALL"
        }
        if set(aggregates) != {"macro", "pooled"}:
            raise RuntimeError(f"Missing macro/pooled records for {system_id}")
        if any(record["trials"] != 1 for record in corpus.values()):
            raise RuntimeError(f"Only saved trial 1 may be published for {system_id}")
        row = {
            "system_id": system_id,
            "system": OPEN_SOURCE_SYSTEMS[system_id],
            "trial": 1,
            "macro_wer_pct": aggregates["macro"]["mean_wer_pct"],
            "pooled_wer_pct": aggregates["pooled"]["mean_wer_pct"],
            "reference_words": aggregates["pooled"]["reference_words"],
            "coverage": f"{aggregates['pooled']['samples_per_trial']}/1285",
        }
        row.update(
            {column: corpus[dataset]["mean_wer_pct"] for dataset, column in CORPUS_COLUMNS.items()}
        )
        if row["coverage"] != "1285/1285" or row["reference_words"] != 133_033:
            raise RuntimeError(f"Incomplete benchmark coverage for {system_id}: {row['coverage']}")
        rows.append(row)
    return sorted(rows, key=lambda row: row["pooled_wer_pct"])


def markdown(rows: list[dict]) -> str:
    lines = [
        "| System | Mega-ASR | AMI | DiPCo | NOTSOFAR | Macro | Pooled |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        values = [
            row["system"],
            row["mega_asr_wer_pct"],
            row["ami_wer_pct"],
            row["dipco_wer_pct"],
            row["notsofar_wer_pct"],
            row["macro_wer_pct"],
            row["pooled_wer_pct"],
        ]
        lines.append(
            "| " + " | ".join([values[0], *(f"{value:.6f}" for value in values[1:])]) + " |"
        )
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "system_id",
        "system",
        "trial",
        "mega_asr_wer_pct",
        "ami_wer_pct",
        "dipco_wer_pct",
        "notsofar_wer_pct",
        "macro_wer_pct",
        "pooled_wer_pct",
        "reference_words",
        "coverage",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: f"{value:.6f}" if isinstance(value, float) else value
                    for key, value in row.items()
                }
            )


def update_readme(path: Path, table: str) -> None:
    text = path.read_text(encoding="utf-8")
    if text.count(MARKER_START) != 1 or text.count(MARKER_END) != 1:
        raise RuntimeError("README open-source table markers are missing or duplicated")
    before, rest = text.split(MARKER_START)
    _, after = rest.split(MARKER_END)
    path.write_text(
        before + MARKER_START + "\n" + table + "\n" + MARKER_END + after,
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the open-source-only result table")
    parser.add_argument("paths", type=Path, nargs="+")
    parser.add_argument("--baseline-csv", type=Path, default=Path("results/four-corpus.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/open-source-four-corpus.csv"))
    parser.add_argument("--readme", type=Path)
    args = parser.parse_args()
    rows = build(args.paths, args.baseline_csv)
    table = markdown(rows)
    write_csv(args.output, rows)
    if args.readme:
        update_readme(args.readme, table)
    print(table)


if __name__ == "__main__":
    main()
