from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

from a5sv2_eval.open_source_table import MARKER_END, MARKER_START, markdown


APPROVED_IDS = {
    "a5sv2",
    "nvidia_nemotron_3_asr_streaming_0_6b",
    "mistral_voxtral_mini_transcribe_realtime_2602",
    "whisper_large_v3_ufal_streaming",
    "kyutai_stt_2_6b_en",
}


def read_scores(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if {row["system_id"] for row in rows} != APPROVED_IDS or len(rows) != 5:
        raise RuntimeError("Open-source score file must contain exactly the five approved systems")
    for row in rows:
        for key in (
            "mega_asr_wer_pct",
            "ami_wer_pct",
            "dipco_wer_pct",
            "notsofar_wer_pct",
            "macro_wer_pct",
            "pooled_wer_pct",
        ):
            row[key] = float(row[key])
    return rows


def update_markers(path: Path, table: str) -> None:
    text = path.read_text(encoding="utf-8")
    if text.count(MARKER_START) != 1 or text.count(MARKER_END) != 1:
        raise RuntimeError(f"Missing or duplicate open-source table markers in {path}")
    before, remainder = text.split(MARKER_START)
    _, after = remainder.split(MARKER_END)
    path.write_text(
        before + MARKER_START + "\n" + table + "\n" + MARKER_END + after,
        encoding="utf-8",
    )


def verify_checksums(bundle: Path) -> None:
    for line in (bundle / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        digest = hashlib.sha256((bundle / relative).read_bytes()).hexdigest()
        if digest != expected:
            raise RuntimeError(f"Checksum mismatch: {relative}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage the completed open-source release")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--github-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    args = parser.parse_args()

    verify_checksums(args.bundle)
    scores = read_scores(args.bundle / "scores/four-corpus.csv")
    table = markdown(scores)
    shutil.copyfile(
        args.bundle / "scores/four-corpus.csv",
        args.github_root / "results/open-source-four-corpus.csv",
    )
    shutil.copyfile(
        args.bundle / "scores/corpus-scores.csv",
        args.github_root / "results/open-source-corpus-scores.csv",
    )
    update_markers(args.github_root / "README.md", table)
    update_markers(args.dataset_root / "README.md", table)
    subprocess.run(
        [sys.executable, str(args.github_root / "tools/render_charts.py")],
        cwd=args.github_root,
        check=True,
    )

    forbidden = ("parakeet", "voxtral_local", "nemotron_nemo", "vast_native")
    tracked_candidates = [
        path
        for root in (args.github_root, args.bundle)
        for path in root.rglob("*")
        if path.is_file() and ".git" not in path.parts
    ]
    leaked = [
        str(path)
        for path in tracked_candidates
        if any(term in path.name.lower() for term in forbidden)
    ]
    if leaked:
        raise RuntimeError(f"Private/excluded implementation paths found: {leaked}")
    print(table)


if __name__ == "__main__":
    main()
