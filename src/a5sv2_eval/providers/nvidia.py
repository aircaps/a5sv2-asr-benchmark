from __future__ import annotations

import hashlib
import os

from together import AsyncTogether

from a5sv2_eval.common import parser, realtime_chunks, run_from_args
from a5sv2_eval.pcm import resample_pcm16


class NvidiaNemotron3:
    system_id = "nvidia_nemotron_3_asr_streaming_0_6b"
    system = "NVIDIA Nemotron 3 ASR Streaming 0.6B"
    model = "nvidia/nemotron-3-asr-streaming-0.6b"

    def __init__(self) -> None:
        self.client = AsyncTogether(api_key=os.environ["TOGETHER_API_KEY"])

    def streaming_config(self, row: dict) -> dict:
        return {
            "model": self.model,
            "language": "en",
            "audio_format": "pcm_s16le_16000",
            "source_sample_rate": row["sample_rate"],
            "sample_rate": 16_000,
            "chunk_ms": 560,
            "turn_detection": "none",
            "resampler": "scipy.signal.resample_poly",
            "pacing": "realtime",
        }

    def capacity_units(self, row: dict) -> int:
        return 1

    async def transcribe(self, pcm: bytes, config: dict) -> dict:
        audio = resample_pcm16(pcm, config["source_sample_rate"], config["sample_rate"])
        chunk_bytes = config["sample_rate"] * 2 * config["chunk_ms"] // 1000
        async with self.client.beta.realtime.transcription(
            model=self.model,
            input_audio_format=config["audio_format"],
            sample_rate=config["sample_rate"],
            language=config["language"],
            turn_detection={"type": "none"},
            max_chunk_ms=config["chunk_ms"],
        ) as session:
            async for chunk in realtime_chunks(audio, config["sample_rate"], chunk_bytes):
                await session.append(chunk)
            return {
                "text": await session.flush(),
                "streamed_pcm_sha256": hashlib.sha256(audio).hexdigest(),
            }

    async def close(self) -> None:
        await self.client.close()


def main() -> None:
    args = parser(
        "Stream the benchmark through NVIDIA Nemotron 3 ASR Streaming 0.6B",
        "nvidia_nemotron_3_asr_streaming_0_6b.jsonl",
        concurrency=1,
    ).parse_args()
    run_from_args(NvidiaNemotron3, args)


if __name__ == "__main__":
    main()
