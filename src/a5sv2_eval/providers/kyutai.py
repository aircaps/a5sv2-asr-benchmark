from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np

from a5sv2_eval.common import parser, run_from_args


class KyutaiSTT26B:
    system_id = "kyutai_stt_2_6b_en"
    system = "Kyutai STT 2.6B English"
    model = "kyutai/stt-2.6b-en"
    model_revision = "a07aec56d22be5589cd0bc8709c75b6cf3e3039d"
    upstream_revision = "4c4f65e147df056adf3346290d64c7b9649b18c9"
    max_concurrency = 1

    def __init__(self) -> None:
        import moshi.models
        import torch
        from huggingface_hub import snapshot_download

        self.torch = torch
        snapshot = Path(
            snapshot_download(repo_id=self.model, revision=self.model_revision)
        ).resolve()
        self.info = moshi.models.loaders.CheckpointInfo.from_hf_repo(
            self.model,
            moshi_weights=snapshot / "model.safetensors",
            mimi_weights=snapshot / "mimi-pytorch-e351c8d8@125.safetensors",
            tokenizer=snapshot / "tokenizer_en_audio_4000.model",
            config_path=snapshot / "config.json",
        )
        self.mimi = self.info.get_mimi(device="cuda")
        self.tokenizer = self.info.get_text_tokenizer()
        self.lm = self.info.get_moshi(device="cuda", dtype=torch.bfloat16)
        self.lm_gen = moshi.models.LMGen(self.lm, temp=0, temp_text=0.0)
        self.prefix_seconds = self.info.stt_config.get("audio_silence_prefix_seconds", 1.0)
        self.delay_seconds = self.info.stt_config.get("audio_delay_seconds", 5.0)
        self.padding_token_id = self.info.raw_config.get("text_padding_token_id", 3)

    def streaming_config(self, row: dict) -> dict:
        return {
            "model": self.model,
            "model_revision": self.model_revision,
            "upstream_inference_revision": self.upstream_revision,
            "source_sample_rate": row["sample_rate"],
            "sample_rate": self.mimi.sample_rate,
            "codec_frame_ms": 1000 / self.mimi.frame_rate,
            "model_text_delay_seconds": self.delay_seconds,
            "audio_silence_prefix_seconds": self.prefix_seconds,
            "temperature": 0,
            "dtype": "bfloat16",
            "pacing": "accelerated_fixed_chunk_sequence",
        }

    def capacity_units(self, row: dict) -> int:
        return 1

    async def transcribe(self, pcm: bytes, config: dict) -> dict:
        torch = self.torch
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32_768.0
        audio = torch.from_numpy(samples).to("cuda")[None, None]

        import julius

        audio = julius.resample_frac(audio, config["source_sample_rate"], self.mimi.sample_rate)
        if audio.shape[-1] % self.mimi.frame_size:
            padding = self.mimi.frame_size - audio.shape[-1] % self.mimi.frame_size
            audio = torch.nn.functional.pad(audio, (0, padding))

        prefix_chunks = math.ceil(self.prefix_seconds * self.mimi.frame_rate)
        suffix_chunks = math.ceil(self.delay_seconds * self.mimi.frame_rate)
        silence = torch.zeros((1, 1, self.mimi.frame_size), device="cuda")
        chunks = itertools.chain(
            itertools.repeat(silence, prefix_chunks),
            torch.split(audio, self.mimi.frame_size, dim=-1),
            itertools.repeat(silence, suffix_chunks),
        )
        pieces: list[str] = []
        chunk_count = 0
        with torch.inference_mode(), self.mimi.streaming(1), self.lm_gen.streaming(1):
            for chunk in chunks:
                chunk_count += 1
                audio_tokens = self.mimi.encode(chunk)
                text_tokens = self.lm_gen.step(audio_tokens)
                token = int(text_tokens[0, 0, 0].cpu())
                if token not in (0, self.padding_token_id):
                    pieces.append(self.tokenizer.id_to_piece(token).replace("▁", " "))
        return {
            "text": "".join(pieces).strip(),
            "codec_chunks": chunk_count,
            "model_revision": self.model_revision,
            "upstream_inference_revision": self.upstream_revision,
        }


def main() -> None:
    args = parser(
        "Run the official Kyutai STT 2.6B streaming inference stack",
        "kyutai_stt_2_6b_en.jsonl",
        concurrency=1,
    ).parse_args()
    run_from_args(KyutaiSTT26B, args)


if __name__ == "__main__":
    main()
