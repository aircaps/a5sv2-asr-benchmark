from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os

import numpy as np
from scipy.signal import resample_poly
from websockets.asyncio.client import connect

from a5sv2_eval.common import parser, realtime_chunks, run_from_args


TARGET_SAMPLE_RATE = 24_000


def resample(pcm: bytes, source_rate: int) -> bytes:
    """Convert native mono PCM16 to the API-required 24 kHz mono PCM16."""
    if source_rate == TARGET_SAMPLE_RATE:
        return bytes(pcm)
    samples = np.frombuffer(pcm, dtype="<i2")
    audio = resample_poly(samples, TARGET_SAMPLE_RATE, source_rate)
    return np.rint(audio).clip(-32_768, 32_767).astype("<i2").tobytes()


class OpenAI:
    system_id = "openai_gpt_live_transcribe"
    system = "OpenAI GPT Live Transcribe"
    model = "gpt-live-transcribe"

    def __init__(self) -> None:
        self.api_key = os.environ["OPENAI_API_KEY"]
        self.url = "wss://api.openai.com/v1/realtime?intent=transcription"

    def streaming_config(self, row: dict) -> dict:
        return {
            "model": self.model,
            "audio_format": "audio/pcm",
            "source_sample_rate": row["sample_rate"],
            "sample_rate": TARGET_SAMPLE_RATE,
            "channels": 1,
            "languages": ["en"],
            "delay": "xhigh",
            "turn_detection": None,
            "chunk_ms": 100,
            "resampler": "scipy.signal.resample_poly",
            "completion_timeout_seconds": 600,
        }

    def capacity_units(self, row: dict) -> int:
        return 1

    async def transcribe(self, pcm: bytes, config: dict) -> dict:
        audio = resample(pcm, config["source_sample_rate"])
        ready = asyncio.Event()
        finished = asyncio.Event()
        errors: list[str] = []
        transcript = ""
        item_id = None
        usage = None

        async with connect(
            self.url,
            additional_headers={"Authorization": f"Bearer {self.api_key}"},
            open_timeout=20,
            close_timeout=10,
            ping_interval=None,
            max_size=None,
        ) as websocket:

            async def receive() -> None:
                nonlocal item_id, transcript, usage
                async for raw in websocket:
                    message = json.loads(raw)
                    event_type = message.get("type")
                    if event_type == "session.updated":
                        ready.set()
                    elif event_type == "conversation.item.input_audio_transcription.completed":
                        item_id = message.get("item_id")
                        transcript = message.get("transcript", "")
                        usage = message.get("usage")
                        finished.set()
                    elif event_type == "conversation.item.input_audio_transcription.failed":
                        errors.append(json.dumps(message.get("error", message)))
                        finished.set()
                    elif event_type == "error":
                        errors.append(json.dumps(message.get("error", message)))
                        ready.set()
                        finished.set()

            receiver = asyncio.create_task(receive())
            try:
                await websocket.send(
                    json.dumps(
                        {
                            "type": "session.update",
                            "session": {
                                "type": "transcription",
                                "audio": {
                                    "input": {
                                        "format": {
                                            "type": config["audio_format"],
                                            "rate": config["sample_rate"],
                                        },
                                        "transcription": {
                                            "model": config["model"],
                                            "languages": config["languages"],
                                            "delay": config["delay"],
                                        },
                                        "turn_detection": config["turn_detection"],
                                    }
                                },
                            },
                        }
                    )
                )
                await asyncio.wait_for(ready.wait(), 20)
                if errors:
                    raise RuntimeError(errors[0])

                chunk_bytes = config["sample_rate"] * 2 * config["chunk_ms"] // 1000
                async for chunk in realtime_chunks(audio, config["sample_rate"], chunk_bytes):
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "input_audio_buffer.append",
                                "audio": base64.b64encode(chunk).decode("ascii"),
                            }
                        )
                    )
                await websocket.send(json.dumps({"type": "input_audio_buffer.commit"}))
                await asyncio.wait_for(finished.wait(), config["completion_timeout_seconds"])
                if errors:
                    raise RuntimeError(errors[0])
            finally:
                if not receiver.done():
                    receiver.cancel()
                await asyncio.gather(receiver, return_exceptions=True)

        return {
            "text": transcript,
            "item_id": item_id,
            "usage": usage,
            "streamed_pcm_sha256": hashlib.sha256(audio).hexdigest(),
            "streamed_audio_seconds": len(audio) / (TARGET_SAMPLE_RATE * 2),
        }


def main() -> None:
    args = parser(
        "Stream the benchmark through OpenAI GPT Live Transcribe",
        "openai_gpt_live_transcribe.jsonl",
        concurrency=8,
    ).parse_args()
    run_from_args(OpenAI, args)


if __name__ == "__main__":
    main()
