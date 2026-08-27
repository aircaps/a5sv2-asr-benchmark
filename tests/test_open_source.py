from __future__ import annotations

import csv
from types import SimpleNamespace

from a5sv2_eval import open_source_table
from a5sv2_eval.open_source_table import OPEN_SOURCE_SYSTEMS, build, markdown
from a5sv2_eval.pcm import resample_pcm16
from a5sv2_eval.providers.kyutai import KyutaiSTT26B
from a5sv2_eval.providers.mistral import MistralVoxtralRealtime
from a5sv2_eval.providers.nvidia import NvidiaNemotron3
from a5sv2_eval.providers.whisper_streaming import WhisperLargeV3Streaming


def test_api_open_source_configs_use_requested_chunks():
    row = {"sample_rate": 48_000}
    nvidia = object.__new__(NvidiaNemotron3).streaming_config(row)
    mistral = object.__new__(MistralVoxtralRealtime).streaming_config(row)

    assert nvidia["chunk_ms"] == 560
    assert nvidia["sample_rate"] == 16_000
    assert nvidia["turn_detection"] == "none"
    assert nvidia["pacing"] == "realtime"
    assert mistral["chunk_ms"] == 560
    assert mistral["sample_rate"] == 48_000
    assert mistral["target_streaming_delay_ms"] == 480
    assert mistral["pacing"] == "realtime"


def test_local_open_source_configs_record_method_and_acceleration():
    row = {"sample_rate": 16_000}
    kyutai = object.__new__(KyutaiSTT26B)
    kyutai.mimi = SimpleNamespace(sample_rate=24_000, frame_rate=12.5)
    kyutai.prefix_seconds = 1.0
    kyutai.delay_seconds = 2.5

    assert kyutai.streaming_config(row)["codec_frame_ms"] == 80
    assert "accelerated" in kyutai.streaming_config(row)["pacing"]
    whisper = object.__new__(WhisperLargeV3Streaming).streaming_config(row)
    assert whisper["method"].startswith("UFAL LocalAgreement-2")
    assert whisper["chunk_ms"] == 560


def test_pcm_resampling_is_deterministic():
    pcm = b"\0\0" * 48_000
    first = resample_pcm16(pcm, 48_000, 16_000)
    assert len(first) == 16_000 * 2
    assert resample_pcm16(pcm, 48_000, 16_000) == first
    assert resample_pcm16(pcm, 48_000, 48_000) is pcm


def test_open_source_table_requires_and_formats_all_systems(tmp_path, monkeypatch):
    baseline_path = tmp_path / "baseline.csv"
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
    with baseline_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "system_id": "a5sv2",
                "system": "A5Sv2",
                "trial": 1,
                "mega_asr_wer_pct": 1,
                "ami_wer_pct": 2,
                "dipco_wer_pct": 3,
                "notsofar_wer_pct": 4,
                "macro_wer_pct": 2.5,
                "pooled_wer_pct": 2.6,
                "reference_words": 133033,
                "coverage": "1285/1285",
            }
        )

    records = []
    corpus = {
        "AirCaps/mega-asr-noise-a5sv2": (1250, 32928),
        "AMI": (7, 32928),
        "DiPCo": (6, 33681),
        "NOTSOFAR": (22, 33496),
    }
    for system_id in set(OPEN_SOURCE_SYSTEMS) - {"a5sv2"}:
        for dataset, (samples, words) in corpus.items():
            records.append(
                {
                    "system_id": system_id,
                    "dataset": dataset,
                    "condition": "overall",
                    "trials": 1,
                    "samples_per_trial": samples,
                    "reference_words": words,
                    "mean_wer_pct": 10.0,
                }
            )
        for condition in ["macro", "pooled"]:
            records.append(
                {
                    "system_id": system_id,
                    "dataset": "ALL",
                    "condition": condition,
                    "trials": 1,
                    "samples_per_trial": 1285,
                    "reference_words": 133033,
                    "mean_wer_pct": 10.0,
                }
            )
    monkeypatch.setattr(open_source_table, "score_files", lambda paths: records)
    rows = build([tmp_path / "unused.jsonl"], baseline_path)
    table = markdown(rows)
    assert len(rows) == 5
    assert table.count("\n") == 6
    assert "A5Sv2" in table
