from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from a5sv2_eval.common import parser, run_from_args
from a5sv2_eval.pcm import pcm16_float32, resample_pcm16


class WhisperLargeV3Streaming:
    system_id = "whisper_large_v3_ufal_streaming"
    system = "Whisper Large V3 (UFAL Whisper-Streaming)"
    model = "openai/whisper-large-v3"
    converted_model_revision = "edaa852ec7e145841d8ffdb056a99866b5f0a478"
    upstream_revision = "6da90b44b7e50d79695e68166d2a2c7609c75abb"
    max_concurrency = 1

    def __init__(self) -> None:
        root = Path(os.environ["WHISPER_STREAMING_ROOT"]).resolve()
        model_dir = Path(os.environ["WHISPER_LARGE_V3_MODEL_DIR"]).resolve()
        if not (root / "whisper_online.py").is_file():
            raise RuntimeError(f"Invalid WHISPER_STREAMING_ROOT: {root}")
        if not (model_dir / "model.bin").is_file():
            raise RuntimeError(f"Invalid WHISPER_LARGE_V3_MODEL_DIR: {model_dir}")
        sys.path.insert(0, str(root))
        from whisper_online import FasterWhisperASR, OnlineASRProcessor

        self.asr = FasterWhisperASR("en", model_dir=str(model_dir))
        self.online = OnlineASRProcessor(self.asr, tokenizer=None, buffer_trimming=("segment", 15))

    def streaming_config(self, row: dict) -> dict:
        return {
            "model": self.model,
            "converted_checkpoint": "Systran/faster-whisper-large-v3",
            "converted_model_revision": self.converted_model_revision,
            "upstream_inference_revision": self.upstream_revision,
            "method": "UFAL LocalAgreement-2 with self-adaptive segment trimming",
            "backend": "faster-whisper",
            "source_sample_rate": row["sample_rate"],
            "sample_rate": 16_000,
            "chunk_ms": 560,
            "beam_size": 5,
            "condition_on_previous_text": True,
            "buffer_trimming": ["segment", 15],
            "compute_type": "float16",
            "pacing": "accelerated_fixed_chunk_sequence",
        }

    def capacity_units(self, row: dict) -> int:
        return 1

    async def transcribe(self, pcm: bytes, config: dict) -> dict:
        audio = pcm16_float32(
            resample_pcm16(pcm, config["source_sample_rate"], config["sample_rate"])
        )
        chunk_samples = config["sample_rate"] * config["chunk_ms"] // 1000
        self.online.init()
        committed: list[str] = []
        updates = 0
        for offset in range(0, len(audio), chunk_samples):
            self.online.insert_audio_chunk(np.asarray(audio[offset : offset + chunk_samples]))
            result = self.online.process_iter()
            updates += 1
            if result[2]:
                committed.append(result[2])
        final = self.online.finish()
        if final[2]:
            committed.append(final[2])
        return {
            "text": self.asr.sep.join(committed).strip(),
            "streaming_updates": updates,
            "upstream_inference_revision": self.upstream_revision,
            "converted_model_revision": self.converted_model_revision,
        }


def main() -> None:
    args = parser(
        "Run Whisper Large V3 with UFAL Whisper-Streaming",
        "whisper_large_v3_ufal_streaming.jsonl",
        concurrency=1,
    ).parse_args()
    run_from_args(WhisperLargeV3Streaming, args)


if __name__ == "__main__":
    main()
