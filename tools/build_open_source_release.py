from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from a5sv2_eval.release import public_prediction, sha256, write_csv, write_gzip
from a5sv2_eval.score import corpus_counts, load_rows


SYSTEMS = {
    "a5sv2": "A5Sv2",
    "nvidia_nemotron_3_asr_streaming_0_6b": "NVIDIA Nemotron 3 ASR Streaming 0.6B",
    "mistral_voxtral_mini_transcribe_realtime_2602": (
        "Mistral Voxtral Mini Transcribe Realtime 2602"
    ),
    "whisper_large_v3_ufal_streaming": "Whisper Large V3 (UFAL Whisper-Streaming)",
    "kyutai_stt_2_6b_en": "Kyutai STT 2.6B English",
}
INPUTS = {
    "nvidia_nemotron_3_asr_streaming_0_6b": (
        "api",
        (
            "nvidia_nemotron_3_asr_streaming_0_6b.jsonl",
            "together_nemotron_3_asr_streaming_0_6b.jsonl",
        ),
        {"together_nemotron_3_asr_streaming_0_6b", "nvidia_nemotron_3_asr_streaming_0_6b"},
    ),
    "mistral_voxtral_mini_transcribe_realtime_2602": (
        "api",
        ("mistral_voxtral_mini_transcribe_realtime_2602.jsonl",),
        {"mistral_voxtral_mini_transcribe_realtime_2602"},
    ),
    "whisper_large_v3_ufal_streaming": (
        "gpu",
        ("whisper_large_v3_ufal_streaming.jsonl",),
        {"whisper_large_v3_ufal_streaming"},
    ),
    "kyutai_stt_2_6b_en": (
        "gpu",
        ("kyutai_stt_2_6b_en.jsonl",),
        {"kyutai_stt_2_6b_en"},
    ),
}
CORPORA = ["Mega-ASR", "AMI", "DiPCo", "NOTSOFAR"]
REFERENCE_MATCH_FIELDS = (
    "condition",
    "reference_raw",
    "sample_rate",
    "num_samples",
    "duration_seconds",
    "source_sha256",
    "pcm_sha256",
)


def corpus_name(row: dict) -> str:
    if row["dataset_id"] in {
        "AirCaps/mega-asr-noise-a5sv2",
        "zhifeixie/Voices-in-the-Wild-2M",
    }:
        return "Mega-ASR"
    return row["dataset_id"]


def validate_against_release(
    path: Path,
    references: list[dict],
    public_system_id: str,
    accepted_input_ids: set[str],
) -> tuple[list[dict], list[dict]]:
    rows = load_rows(path)
    if len(rows) != len(references):
        raise RuntimeError(f"Expected {len(references)} rows in {path}, found {len(rows)}")
    if any(row.get("status", "ok") != "ok" for row in rows):
        raise RuntimeError(f"Non-success row in {path}")
    if any(
        row.get("system_id") not in accepted_input_ids or int(row.get("trial", 1)) != 1
        for row in rows
    ):
        raise RuntimeError(f"Mixed system or non-trial-1 row in {path}")

    references_by_id = {row["id"]: row for row in references}
    rows_by_id = {row["id"]: row for row in rows}
    if len(rows_by_id) != len(rows) or set(rows_by_id) != set(references_by_id):
        raise RuntimeError(f"Reference coverage mismatch or duplicate IDs in {path}")

    alias_counts: dict[tuple[str, str, str, str], int] = {}
    for key, row in rows_by_id.items():
        reference = references_by_id[key]
        for field in REFERENCE_MATCH_FIELDS:
            if row.get(field) != reference.get(field):
                raise RuntimeError(f"{field} mismatch for {key} in {path}")
        if (row.get("dataset_id"), row.get("dataset_revision")) != (
            reference.get("dataset_id"),
            reference.get("dataset_revision"),
        ):
            alias = (
                row.get("dataset_id", ""),
                row.get("dataset_revision", ""),
                reference.get("dataset_id", ""),
                reference.get("dataset_revision", ""),
            )
            alias_counts[alias] = alias_counts.get(alias, 0) + 1

    aliases = [
        {
            "system_id": public_system_id,
            "evaluated_dataset_id": source_id,
            "evaluated_revision": source_revision,
            "release_reference_dataset_id": release_id,
            "release_reference_revision": release_revision,
            "rows": count,
            "verification": (
                "IDs, references, conditions, sample counts, durations, source hashes, and "
                "PCM hashes are identical"
            ),
        }
        for (source_id, source_revision, release_id, release_revision), count in sorted(
            alias_counts.items()
        )
    ]
    return [public_prediction(row, public_system_id) for row in rows], aliases


def score_system(system_id: str, rows: list[dict]) -> list[dict]:
    output = []
    expected_counts = {"Mega-ASR": 1250, "AMI": 7, "DiPCo": 6, "NOTSOFAR": 22}
    for corpus in CORPORA:
        group = [row for row in rows if corpus_name(row) == corpus]
        if len(group) != expected_counts[corpus]:
            raise RuntimeError(
                f"Expected {expected_counts[corpus]} {corpus} rows for {system_id}, got {len(group)}"
            )
        output.append(
            {
                "system_id": system_id,
                "system": SYSTEMS[system_id],
                "trial": 1,
                "corpus": corpus,
                "samples": len(group),
                **corpus_counts(group),
            }
        )
    return output


def summarize(scores: list[dict]) -> list[dict]:
    summaries = []
    for system_id, system in SYSTEMS.items():
        rows = [row for row in scores if row["system_id"] == system_id]
        values = {row["corpus"]: row["wer_pct"] for row in rows}
        summaries.append(
            {
                "system_id": system_id,
                "system": system,
                "trial": 1,
                "mega_asr_wer_pct": values["Mega-ASR"],
                "ami_wer_pct": values["AMI"],
                "dipco_wer_pct": values["DiPCo"],
                "notsofar_wer_pct": values["NOTSOFAR"],
                "macro_wer_pct": sum(values.values()) / 4,
                "pooled_wer_pct": 100
                * sum(row["errors"] for row in rows)
                / sum(row["reference_words"] for row in rows),
                "reference_words": sum(row["reference_words"] for row in rows),
                "coverage": f"{sum(row['samples'] for row in rows)}/1285",
            }
        )
    return sorted(summaries, key=lambda row: row["pooled_wer_pct"])


def markdown(rows: list[dict]) -> str:
    keys = (
        "mega_asr_wer_pct",
        "ami_wer_pct",
        "dipco_wer_pct",
        "notsofar_wer_pct",
        "macro_wer_pct",
        "pooled_wer_pct",
    )
    best = {key: min(row[key] for row in rows) for key in keys}
    lines = [
        "| System | Mega-ASR | AMI | DiPCo | NOTSOFAR | Macro | Pooled |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        values = [
            f"**{row[key]:.6f}**" if row[key] == best[key] else f"{row[key]:.6f}"
            for key in keys
        ]
        lines.append("| " + " | ".join([row["system"], *values]) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the public A5Sv2/open-source result bundle from completed runs"
    )
    parser.add_argument("--base-release", type=Path, required=True)
    parser.add_argument("--api-mega-dir", type=Path, required=True)
    parser.add_argument("--api-meeting-dir", type=Path, required=True)
    parser.add_argument("--gpu-run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-revision", default="uncommitted-working-tree")
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"Output already exists: {args.output}")

    references = {
        "mega-asr": load_rows(args.base_release / "references/mega-asr.jsonl.gz"),
        "meetings": load_rows(args.base_release / "references/meetings.jsonl.gz"),
    }
    if len(references["mega-asr"]) != 1250 or len(references["meetings"]) != 35:
        raise RuntimeError("Base release does not contain the fixed 1,285 references")
    for group in references:
        destination = args.output / "references" / f"{group}.jsonl.gz"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.base_release / "references" / f"{group}.jsonl.gz", destination)

    public_rows: dict[str, list[dict]] = {"a5sv2": []}
    inputs = []
    aliases = []
    for group in references:
        source = args.base_release / "predictions/a5sv2" / f"{group}.jsonl.gz"
        destination = args.output / "predictions/a5sv2" / f"{group}.jsonl.gz"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        public_rows["a5sv2"].extend(load_rows(source))
        inputs.append({"system_id": "a5sv2", "corpus_group": group, "sha256": sha256(source)})

    for system_id, (source_type, filenames, accepted_ids) in INPUTS.items():
        combined = []
        for group, api_dir, gpu_subdir in (
            ("mega-asr", args.api_mega_dir, "mega-results"),
            ("meetings", args.api_meeting_dir, "meeting-results"),
        ):
            directory = api_dir if source_type == "api" else args.gpu_run_dir / gpu_subdir
            candidates = [directory / filename for filename in filenames]
            raw = next((path for path in candidates if path.exists()), None)
            if raw is None:
                raise RuntimeError(f"Missing result input; checked: {candidates}")
            rows, row_aliases = validate_against_release(
                raw, references[group], system_id, accepted_ids
            )
            write_gzip(args.output / "predictions" / system_id / f"{group}.jsonl.gz", rows)
            combined.extend(rows)
            aliases.extend(row_aliases)
            inputs.append(
                {"system_id": system_id, "corpus_group": group, "raw_sha256": sha256(raw)}
            )
        public_rows[system_id] = combined

    scores = []
    for system_id in SYSTEMS:
        scores.extend(score_system(system_id, public_rows[system_id]))
    summaries = summarize(scores)
    rank = {row["system_id"]: index for index, row in enumerate(summaries)}
    scores.sort(key=lambda row: (rank[row["system_id"]], CORPORA.index(row["corpus"])))
    write_csv(args.output / "scores/corpus-scores.csv", scores, list(scores[0]))
    write_csv(args.output / "scores/four-corpus.csv", summaries, list(summaries[0]))

    base_metadata = json.loads((args.base_release / "metadata.json").read_text(encoding="utf-8"))
    metadata = {
        "schema_version": "1.0",
        "release_status": "staged; uncommitted and unpublished",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_code_revision": args.code_revision,
        "base_release_evaluation_code_commit": base_metadata["evaluation_code_commit"],
        "reference_rows": 1285,
        "reference_words": 133033,
        "trials_per_system": 1,
        "systems": list(SYSTEMS),
        "scoring": base_metadata["scoring"],
        "uncertainty": base_metadata["uncertainty"],
        "mega_asr_disclosure": base_metadata["mega_asr_disclosure"],
        "dataset_revision_reconciliation": aliases,
        "input_artifacts": inputs,
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "README.md").write_text(
        "# A5Sv2 open-source four-corpus benchmark results\n\n"
        "WER (%), lower is better. Every value is saved trial 1 with complete "
        "1,285/1,285 coverage; no multi-trial averaging is used.\n\n"
        + markdown(summaries)
        + "\n\nRaw predictions and the exact fixed references are included for independent "
        "rescoring. Inference methods and pinned dependencies are documented in the linked "
        "evaluation repository. Verify this bundle with `sha256sum -c SHA256SUMS`.\n",
        encoding="utf-8",
    )
    files = sorted(path for path in args.output.rglob("*") if path.is_file())
    checksums = "".join(f"{sha256(path)}  {path.relative_to(args.output)}\n" for path in files)
    (args.output / "SHA256SUMS").write_text(checksums, encoding="utf-8")
    print(f"Saved {args.output} ({len(files) + 1} files)")


if __name__ == "__main__":
    main()
