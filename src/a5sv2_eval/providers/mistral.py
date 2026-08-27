from __future__ import annotations

import os

from mistralai.client import Mistral
from mistralai.client.models import AudioFormat, RealtimeTranscriptionError, TranscriptionStreamDone

from a5sv2_eval.common import parser, realtime_chunks, run_from_args


class MistralVoxtralRealtime:
    system_id = "mistral_voxtral_mini_transcribe_realtime_2602"
    system = "Mistral Voxtral Mini Transcribe Realtime 2602"
    model = "voxtral-mini-transcribe-realtime-2602"

    def __init__(self) -> None:
        self.client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

    def streaming_config(self, row: dict) -> dict:
        return {
            "model": self.model,
            "audio_format": "pcm_s16le",
            "sample_rate": row["sample_rate"],
            "channels": 1,
            "chunk_ms": 560,
            "target_streaming_delay_ms": 480,
            "pacing": "realtime",
        }

    def capacity_units(self, row: dict) -> int:
        return 1

    async def transcribe(self, pcm: bytes, config: dict) -> dict:
        chunk_bytes = config["sample_rate"] * 2 * config["chunk_ms"] // 1000

        async def audio():
            async for chunk in realtime_chunks(pcm, config["sample_rate"], chunk_bytes):
                yield chunk

        async for event in self.client.audio.realtime.transcribe_stream(
            audio_stream=audio(),
            model=self.model,
            audio_format=AudioFormat(
                encoding=config["audio_format"], sample_rate=config["sample_rate"]
            ),
            target_streaming_delay_ms=config["target_streaming_delay_ms"],
        ):
            if isinstance(event, TranscriptionStreamDone):
                return {"text": event.text}
            if isinstance(event, RealtimeTranscriptionError):
                raise RuntimeError(str(event.error))
        raise RuntimeError("Stream ended without transcription.done")


def main() -> None:
    args = parser(
        "Stream the benchmark through Mistral Voxtral Mini Transcribe Realtime 2602",
        "mistral_voxtral_mini_transcribe_realtime_2602.jsonl",
        concurrency=1,
    ).parse_args()
    run_from_args(MistralVoxtralRealtime, args)


if __name__ == "__main__":
    main()
