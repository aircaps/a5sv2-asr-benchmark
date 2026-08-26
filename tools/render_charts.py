from __future__ import annotations

import csv
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROWS = list(csv.DictReader((ROOT / "results/four-corpus.csv").open()))
LABELS = {
    "ElevenLabs Scribe v2 Realtime": ("ElevenLabs Scribe v2", "Realtime"),
    "A5Sv2": ("A5Sv2", ""),
    "AssemblyAI Universal-3.5 Pro Realtime": ("AssemblyAI Universal-3.5", "Pro Realtime"),
    "OpenAI GPT Live Transcribe": ("OpenAI GPT Live", "Transcribe"),
    "Deepgram Nova-3 Streaming": ("Deepgram Nova-3", "Streaming"),
    "Google Chirp 3 Streaming": ("Google Chirp 3", "Streaming"),
}


def frame(title: str, subtitle: str, width: int, height: int, body: str, desc: str) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
<title id="title">{escape(title)}</title><desc id="desc">{escape(desc)}</desc>
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="60" y="65" font-family="Georgia,serif" font-size="38" fill="#151515">{escape(title)}</text>
<text x="60" y="103" font-family="Arial,sans-serif" font-size="20" fill="#666666">{escape(subtitle)}</text>
{body}</svg>
'''


def pooled() -> str:
    rows = sorted(ROWS, key=lambda row: float(row["pooled_wer_pct"]))
    width, height, left, top, bottom, ymax = 1600, 760, 70, 160, 165, 55
    plot_h = height - top - bottom
    colors = ["#1b9e77", "#2d6cdf", "#7a4ea3", "#e66101", "#1f4e79", "#c51b7d"]
    body = []
    for tick in range(0, 56, 10):
        y = top + plot_h * (1 - tick / ymax)
        body.append(f'<line x1="{left}" x2="1540" y1="{y:.1f}" y2="{y:.1f}" stroke="#d0d0d0" stroke-dasharray="4 8"/>')
        body.append(f'<text x="55" y="{y + 7:.1f}" text-anchor="end" font-family="Arial,sans-serif" font-size="17" fill="#777">{tick}</text>')
    slot = (1540 - left) / len(rows)
    bar_w = 138
    for index, (row, color) in enumerate(zip(rows, colors, strict=True)):
        value = float(row["pooled_wer_pct"])
        x = left + index * slot + (slot - bar_w) / 2
        y = top + plot_h * (1 - value / ymax)
        body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{top + plot_h - y:.1f}" rx="8" fill="{color}"/>')
        body.append(f'<text x="{x + bar_w / 2:.1f}" y="{y - 12:.1f}" text-anchor="middle" font-family="Arial,sans-serif" font-size="21" font-weight="700" fill="#111">{value:.1f}%</text>')
        first, second = LABELS[row["system"]]
        body.append(f'<text x="{x + bar_w / 2:.1f}" y="{height - 115}" text-anchor="middle" font-family="Arial,sans-serif" font-size="17" fill="#222">{escape(first)}</text>')
        if second:
            body.append(f'<text x="{x + bar_w / 2:.1f}" y="{height - 91}" text-anchor="middle" font-family="Arial,sans-serif" font-size="17" fill="#222">{escape(second)}</text>')
    return frame(
        "A5Sv2 ASR Benchmark — Pooled WER",
        "% of words transcribed incorrectly · lower is better · 133,033 reference words",
        width,
        height,
        "\n".join(body),
        "Pooled WER ranked from lowest to highest for six streaming ASR systems.",
    )


def by_corpus() -> str:
    width, height, left, right, top, bottom, ymax = 1800, 900, 70, 35, 180, 185, 85
    plot_h = height - top - bottom
    corpora = [
        ("Mega-ASR", "mega_asr_wer_pct", "#2d6cdf"),
        ("AMI", "ami_wer_pct", "#16827a"),
        ("DiPCo", "dipco_wer_pct", "#d95f02"),
        ("NOTSOFAR", "notsofar_wer_pct", "#6f4c9b"),
    ]
    body = []
    for tick in range(0, 81, 20):
        y = top + plot_h * (1 - tick / ymax)
        body.append(f'<line x1="{left}" x2="{width-right}" y1="{y:.1f}" y2="{y:.1f}" stroke="#d0d0d0" stroke-dasharray="4 8"/>')
        body.append(f'<text x="55" y="{y + 7:.1f}" text-anchor="end" font-family="Arial,sans-serif" font-size="17" fill="#777">{tick}</text>')
    legend_x = 850
    for index, (corpus, _, color) in enumerate(corpora):
        x = legend_x + index * 210
        body.append(f'<rect x="{x}" y="125" width="19" height="19" rx="3" fill="{color}"/>')
        body.append(f'<text x="{x + 28}" y="141" font-family="Arial,sans-serif" font-size="18" fill="#333">{corpus}</text>')
    slot = (width - left - right) / len(ROWS)
    bar_w, gap = 48, 7
    group_w = len(corpora) * bar_w + (len(corpora) - 1) * gap
    for index, row in enumerate(ROWS):
        group_x = left + index * slot + (slot - group_w) / 2
        for offset, (_, key, color) in enumerate(corpora):
            value = float(row[key])
            x = group_x + offset * (bar_w + gap)
            y = top + plot_h * (1 - value / ymax)
            body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{top + plot_h - y:.1f}" rx="5" fill="{color}"/>')
            body.append(f'<text x="{x + bar_w / 2:.1f}" y="{y - 7:.1f}" text-anchor="middle" font-family="Arial,sans-serif" font-size="13" font-weight="700" fill="#222">{value:.1f}</text>')
        center = group_x + group_w / 2
        first, second = LABELS[row["system"]]
        body.append(f'<text x="{center:.1f}" y="{height - 125}" text-anchor="middle" font-family="Arial,sans-serif" font-size="16" fill="#222">{escape(first)}</text>')
        if second:
            body.append(f'<text x="{center:.1f}" y="{height - 102}" text-anchor="middle" font-family="Arial,sans-serif" font-size="16" fill="#222">{escape(second)}</text>')
    return frame(
        "A5Sv2 ASR Benchmark — WER by Corpus",
        "% of words transcribed incorrectly · lower is better · continuous full-meeting streaming",
        width,
        height,
        "\n".join(body),
        "Grouped bars compare WER on Mega-ASR, AMI, DiPCo, and NOTSOFAR for six streaming ASR systems.",
    )


if __name__ == "__main__":
    (ROOT / "docs/wer-pooled.svg").write_text(pooled(), encoding="utf-8")
    (ROOT / "docs/wer-by-corpus.svg").write_text(by_corpus(), encoding="utf-8")
