from __future__ import annotations

import asyncio
import json
import os
from urllib.parse import urlencode

from websockets.asyncio.client import connect

from a5sv2_eval.common import parser, realtime_chunks, run_from_args


class Deepgram:
    system_id = "deepgram_nova_3_streaming"
    system = "Deepgram Nova-3 Streaming"
    model = "nova-3"

    def __init__(self) -> None:
        self.api_key = os.environ["DEEPGRAM_API_KEY"]

    def streaming_config(self, row: dict) -> dict:
        return {
            "model": self.model,
            "language": "en-US",
            "encoding": "linear16",
            "sample_rate": row["sample_rate"],
            "channels": 1,
            "smart_format": True,
            "interim_results": False,
            "chunk_ms": 100,
        }

    def capacity_units(self, row: dict) -> int:
        return 1

    async def transcribe(self, pcm: bytes, config: dict) -> dict:
        parts, metadata = [], {}
        closed = asyncio.Event()
        query = {key: str(value).lower() for key, value in config.items() if key != "chunk_ms"}
        async with connect(
            f"wss://api.deepgram.com/v1/listen?{urlencode(query)}",
            additional_headers={"Authorization": f"Token {self.api_key}"},
            open_timeout=20,
            close_timeout=10,
            ping_interval=None,
            max_size=None,
        ) as websocket:

            async def receive() -> None:
                async for raw in websocket:
                    message = json.loads(raw)
                    if message.get("type") == "Results":
                        transcript = message["channel"]["alternatives"][0]["transcript"].strip()
                        if message.get("is_final") and transcript:
                            parts.append(transcript)
                        response_metadata = message.get("metadata", {})
                        metadata["request_id"] = response_metadata.get("request_id")
                        metadata["model_info"] = response_metadata.get("model_info")
                    elif message.get("type") == "Metadata":
                        metadata["request_id"] = message.get("request_id")
                        closed.set()

            receiver = asyncio.create_task(receive())
            try:
                chunk_bytes = config["sample_rate"] * 2 * config["chunk_ms"] // 1000
                async for chunk in realtime_chunks(pcm, config["sample_rate"], chunk_bytes):
                    await websocket.send(chunk)
                await websocket.send(json.dumps({"type": "CloseStream"}))
                await asyncio.wait_for(closed.wait(), 30)
            finally:
                if not receiver.done():
                    receiver.cancel()
                await asyncio.gather(receiver, return_exceptions=True)
        return {"text": " ".join(parts), **metadata}


def main() -> None:
    args = parser(
        "Stream the benchmark through Deepgram Nova-3",
        "deepgram_nova_3.jsonl",
        concurrency=150,
    ).parse_args()
    run_from_args(Deepgram, args)


if __name__ == "__main__":
    main()
