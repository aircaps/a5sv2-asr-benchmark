from __future__ import annotations

import asyncio
import base64
import json
import os
from urllib.parse import urlencode

from websockets.asyncio.client import connect

from a5sv2_eval.common import parser, realtime_chunks, run_from_args


class ElevenLabs:
    system_id = "elevenlabs_scribe_v2_realtime"
    system = "ElevenLabs Scribe v2 Realtime"
    model = "scribe_v2_realtime"

    def __init__(self) -> None:
        self.api_key = os.environ["ELEVENLABS_API_KEY"]

    def streaming_config(self, row: dict) -> dict:
        return {
            "model_id": self.model,
            "audio_format": f"pcm_{row['sample_rate']}",
            "language_code": "en",
            "commit_strategy": "manual",
            "sample_rate": row["sample_rate"],
            "chunk_ms": 100,
        }

    def capacity_units(self, row: dict) -> int:
        return 1

    async def transcribe(self, pcm: bytes, config: dict) -> dict:
        query = {
            key: value for key, value in config.items() if key not in {"sample_rate", "chunk_ms"}
        }
        parts, metadata, errors = [], {}, []
        committed = asyncio.Event()
        async with connect(
            f"wss://api.elevenlabs.io/v1/speech-to-text/realtime?{urlencode(query)}",
            additional_headers={"xi-api-key": self.api_key},
            open_timeout=20,
            close_timeout=10,
            ping_interval=None,
            max_size=None,
        ) as websocket:

            async def receive() -> None:
                async for raw in websocket:
                    message = json.loads(raw)
                    kind = message.get("message_type")
                    if kind == "session_started":
                        metadata["session_id"] = message.get("session_id")
                    elif kind == "committed_transcript":
                        parts.append(message.get("text", ""))
                        committed.set()
                    elif "error" in message:
                        errors.append(f"{kind}: {message['error']}")
                        committed.set()

            receiver = asyncio.create_task(receive())
            try:
                chunk_bytes = config["sample_rate"] * 2 * config["chunk_ms"] // 1000
                sent = 0
                async for chunk in realtime_chunks(pcm, config["sample_rate"], chunk_bytes):
                    sent += len(chunk)
                    await websocket.send(
                        json.dumps(
                            {
                                "message_type": "input_audio_chunk",
                                "audio_base_64": base64.b64encode(chunk).decode(),
                                "commit": sent == len(pcm),
                                "sample_rate": config["sample_rate"],
                            }
                        )
                    )
                await asyncio.wait_for(committed.wait(), 30)
                if errors:
                    raise RuntimeError(errors[0])
            finally:
                if not receiver.done():
                    receiver.cancel()
                await asyncio.gather(receiver, return_exceptions=True)
        return {"text": "".join(parts), **metadata}


def main() -> None:
    args = parser(
        "Stream the benchmark through ElevenLabs Scribe v2 Realtime",
        "elevenlabs_scribe_v2_realtime.jsonl",
        concurrency=40,
    ).parse_args()
    run_from_args(ElevenLabs, args)


if __name__ == "__main__":
    main()
