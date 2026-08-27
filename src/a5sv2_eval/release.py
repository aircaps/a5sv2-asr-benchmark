from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path

from a5sv2_eval.score import corpus_counts, load_rows, row_identity

BASE_SYSTEMS = {
    "a5sv2": "A5Sv2",
    "assemblyai_universal_3_5_pro_realtime": "AssemblyAI Universal-3.5 Pro Realtime",
    "deepgram_nova_3_streaming": "Deepgram Nova-3 Streaming",
    "elevenlabs_scribe_v2_realtime": "ElevenLabs Scribe v2 Realtime",
    "google_chirp_3_streaming": "Google Chirp 3 Streaming",
    "openai_gpt_live_transcribe": "OpenAI GPT Live Transcribe",
}
OPEN_SOURCE_SYSTEMS = {
    "nvidia_nemotron_3_asr_streaming_0_6b": "NVIDIA Nemotron 3 ASR Streaming 0.6B",
    "mistral_voxtral_mini_transcribe_realtime_2602": (
        "Mistral Voxtral Mini Transcribe Realtime 2602"
    ),
    "kyutai_stt_2_6b_en": "Kyutai STT 2.6B English",
    "whisper_large_v3_ufal_streaming": "Whisper Large V3 (UFAL Whisper-Streaming)",
}
SYSTEMS = {**BASE_SYSTEMS, **OPEN_SOURCE_SYSTEMS}
API_SYSTEMS = set(BASE_SYSTEMS) - {"a5sv2"}
CORPORA = ["Mega-ASR", "AMI", "DiPCo", "NOTSOFAR"]
REFERENCE_KEYS = [
    "dataset_id",
    "dataset_revision",
    "id",
    "condition",
    "row_index",
    "meeting_id",
    "reference_raw",
    "sample_rate",
    "num_samples",
    "duration_seconds",
    "source_sha256",
    "pcm_sha256",
    "streaming_scope",
    "split",
    "device",
    "channel",
    "gain_db",
    "source_transport",
    "source_transport_revision",
    "transcript_sha256",
]
PREDICTION_KEYS = [
    *REFERENCE_KEYS,
    "trial",
    "system_id",
    "system",
    "model",
    "prediction_raw",
    "status",
    "attempts",
    "elapsed_seconds",
    "streaming_config",
    "run_config",
    "provider_metadata",
    "completed_at",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_gzip(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
        with io.TextIOWrapper(compressed, encoding="utf-8") as text:
            for row in rows:
                text.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def public_reference(row: dict) -> dict:
    return {key: row[key] for key in REFERENCE_KEYS if key in row}


def public_prediction(row: dict, system_id: str) -> dict:
    public = {key: row[key] for key in PREDICTION_KEYS if key in row}
    public.update(
        {
            "schema_version": "1.0",
            "trial": 1,
            "system_id": system_id,
            "system": SYSTEMS[system_id],
        }
    )
    if system_id == "a5sv2":
        model_hash = row.get("run_config", {}).get("model_sha256")
        public.update(
            {
                "model": "a5sv2",
                "streaming_config": {
                    "mode": "streaming",
                    "attention_context": [70, 6],
                    "compute_dtype": "float32",
                },
                "run_config": {"model_sha256": model_hash},
            }
        )
        public.pop("provider_metadata", None)
    return public


def validate_result(path: Path, manifest: list[dict], system_id: str) -> list[dict]:
    rows = load_rows(path)
    if len(rows) != len(manifest) or any(row.get("status", "ok") != "ok" for row in rows):
        raise RuntimeError(f"Expected {len(manifest)} successful rows in {path}")
    if any(int(row.get("trial", 1)) != 1 for row in rows):
        raise RuntimeError(f"Only saved trial 1 can be released: {path}")
    if system_id != "a5sv2" and any(row.get("system_id") != system_id for row in rows):
        raise RuntimeError(f"Mixed or incorrect system_id for {system_id}: {path}")
    if len({row_identity(row) for row in rows}) != len(rows):
        raise RuntimeError(f"Duplicate rows in {path}")
    expected = sorted(row_identity(row) for row in manifest)
    if sorted(row_identity(row) for row in rows) != expected:
        raise RuntimeError(f"Manifest mismatch in {path}")
    return [public_prediction(row, system_id) for row in rows]


def result_inputs(paths: list[Path], label: str, expected: set[str]) -> dict[str, Path]:
    output = {}
    for path in paths:
        rows = load_rows(path)
        if not rows or rows[0].get("system_id") not in expected:
            raise RuntimeError(f"Unapproved or empty {label} result: {path}")
        system_id = rows[0]["system_id"]
        if system_id in output:
            raise RuntimeError(f"Duplicate {label} result for {system_id}")
        output[system_id] = path
    if set(output) != expected:
        raise RuntimeError(f"{label} results must contain exactly: {sorted(expected)}")
    return output


def corpus_name(row: dict) -> str:
    name = row["dataset_id"]
    return "Mega-ASR" if name == "zhifeixie/Voices-in-the-Wild-2M" else name


def score(system_id: str, rows: list[dict]) -> list[dict]:
    output = []
    for corpus in CORPORA:
        group = [row for row in rows if corpus_name(row) == corpus]
        counts = corpus_counts(group)
        output.append(
            {
                "system_id": system_id,
                "system": SYSTEMS[system_id],
                "trial": 1,
                "corpus": corpus,
                "samples": len(group),
                **counts,
            }
        )
    return output


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the public four-corpus result bundle")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--mega-manifest", type=Path, required=True)
    parser.add_argument("--meetings-manifest", type=Path, required=True)
    parser.add_argument("--mega-results", type=Path, nargs="+", required=True)
    parser.add_argument("--meeting-results", type=Path, nargs="+", required=True)
    parser.add_argument("--open-source-mega-results", type=Path, nargs="*", default=[])
    parser.add_argument("--open-source-meeting-results", type=Path, nargs="*", default=[])
    parser.add_argument("--a5sv2-mega", type=Path, required=True)
    parser.add_argument("--a5sv2-meetings", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"Output already exists: {args.output}")

    mega_manifest = load_rows(args.mega_manifest)
    meeting_manifest = load_rows(args.meetings_manifest)
    if len(mega_manifest) != 1250 or len(meeting_manifest) != 35:
        raise RuntimeError("Expected 1,250 Mega-ASR rows and 35 meeting rows")
    write_gzip(
        args.output / "references/mega-asr.jsonl.gz", [public_reference(r) for r in mega_manifest]
    )
    write_gzip(
        args.output / "references/meetings.jsonl.gz",
        [public_reference(r) for r in meeting_manifest],
    )

    mega_inputs = result_inputs(args.mega_results, "Mega-ASR", API_SYSTEMS)
    meeting_inputs = result_inputs(args.meeting_results, "meeting", API_SYSTEMS)
    if bool(args.open_source_mega_results) != bool(args.open_source_meeting_results):
        raise RuntimeError("Open-source Mega-ASR and meeting results must be supplied together")
    included_systems = dict(BASE_SYSTEMS)
    if args.open_source_mega_results:
        expected_open = set(OPEN_SOURCE_SYSTEMS)
        mega_inputs.update(
            result_inputs(args.open_source_mega_results, "open-source Mega-ASR", expected_open)
        )
        meeting_inputs.update(
            result_inputs(args.open_source_meeting_results, "open-source meeting", expected_open)
        )
        included_systems.update(OPEN_SOURCE_SYSTEMS)
    inputs = []
    scores = []
    for system_id in included_systems:
        mega_path = args.a5sv2_mega if system_id == "a5sv2" else mega_inputs[system_id]
        meeting_path = args.a5sv2_meetings if system_id == "a5sv2" else meeting_inputs[system_id]
        mega_rows = validate_result(mega_path, mega_manifest, system_id)
        meeting_rows = validate_result(meeting_path, meeting_manifest, system_id)
        write_gzip(args.output / f"predictions/{system_id}/mega-asr.jsonl.gz", mega_rows)
        write_gzip(args.output / f"predictions/{system_id}/meetings.jsonl.gz", meeting_rows)
        scores.extend(score(system_id, mega_rows + meeting_rows))
        inputs.extend(
            [
                {"system_id": system_id, "corpus_group": "mega-asr", "sha256": sha256(mega_path)},
                {
                    "system_id": system_id,
                    "corpus_group": "meetings",
                    "sha256": sha256(meeting_path),
                },
            ]
        )

    score_fields = [
        "system_id",
        "system",
        "trial",
        "corpus",
        "samples",
        "hits",
        "substitutions",
        "deletions",
        "insertions",
        "reference_words",
        "errors",
        "wer_pct",
    ]
    summary = []
    for system_id, system in included_systems.items():
        rows = [row for row in scores if row["system_id"] == system_id]
        values = {row["corpus"]: row["wer_pct"] for row in rows}
        summary.append(
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
    summary.sort(key=lambda row: row["pooled_wer_pct"])
    rank = {row["system_id"]: index for index, row in enumerate(summary)}
    scores.sort(key=lambda row: (rank[row["system_id"]], CORPORA.index(row["corpus"])))
    write_csv(args.output / "scores/corpus-scores.csv", scores, score_fields)
    summary_fields = list(summary[0])
    write_csv(args.output / "scores/four-corpus.csv", summary, summary_fields)
    if args.open_source_mega_results:
        open_source_ids = {"a5sv2", *OPEN_SOURCE_SYSTEMS}
        open_source_summary = [row for row in summary if row["system_id"] in open_source_ids]
        write_csv(
            args.output / "scores/open-source-four-corpus.csv",
            open_source_summary,
            summary_fields,
        )

    metadata = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_code_commit": args.code_commit,
        "reference_rows": 1285,
        "reference_words": 133033,
        "trials_per_system": 1,
        "systems": list(included_systems),
        "scoring": {
            "metric": "corpus WER",
            "trial_aggregation": "none; release v1 reports trial 1",
            "macro": "unweighted arithmetic mean of four corpus WERs",
            "pooled": "total errors divided by total reference words",
            "normalization": "Unicode NFKC; lowercase; remove apostrophes; replace other punctuation with spaces; collapse whitespace",
        },
        "uncertainty": {"reported": False},
        "mega_asr_disclosure": {
            "source_dataset": "zhifeixie/Voices-in-the-Wild-2M",
            "source_revision": "a8a35d3319737190d6fd3d39157b258eaab35980",
            "selection_seed": 20260823,
            "selection": "Sampled from Mega-ASR-Train because the standard test set was not acoustically challenging enough to discriminate robust ASR systems",
            "a5sv2_exact_items_used_for_training_or_model_selection": False,
            "third_party_exact_item_or_distribution_overlap": "unknown",
            "distribution_level_independence_claimed": False,
        },
        "input_artifacts": inputs,
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    table = ["| System | Macro WER | Pooled WER |", "|---|---:|---:|"]
    table += [
        f"| {row['system']} | {row['macro_wer_pct']:.6f} | {row['pooled_wer_pct']:.6f} |"
        for row in summary
    ]
    (args.output / "README.md").write_text(
        "# A5Sv2 four-corpus benchmark results\n\n"
        "WER (%), lower is better. Every value is from saved trial 1; no multi-trial averaging is used. References and raw predictions are included for independent rescoring.\n\n"
        + "\n".join(table)
        + "\n\nVerify files with `sha256sum -c SHA256SUMS`. Audio is reconstructed from pinned public sources using the linked evaluation repository.\n",
        encoding="utf-8",
    )
    files = sorted(path for path in args.output.rglob("*") if path.is_file())
    checksums = "".join(f"{sha256(path)}  {path.relative_to(args.output)}\n" for path in files)
    (args.output / "SHA256SUMS").write_text(checksums, encoding="utf-8")
    print(f"Saved {args.output} ({len(files) + 1} files)")


if __name__ == "__main__":
    main()
