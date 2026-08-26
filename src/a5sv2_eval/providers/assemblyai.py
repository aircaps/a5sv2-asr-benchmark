from __future__ import annotations

import asyncio
import json
import os
from urllib.parse import urlencode

from websockets.asyncio.client import connect

from a5sv2_eval.common import parser, realtime_chunks, run_from_args


class AssemblyAI:
    system_id = "assemblyai_universal_3_5_pro_realtime"
    system = "AssemblyAI Universal-3.5 Pro Realtime"
    model = "universal-3-5-pro"

    def __init__(self) -> None:
        self.api_key = os.environ["ASSEMBLYAI_API_KEY"]

    def streaming_config(self, row: dict) -> dict:
        return {
            "speech_model": self.model,
            "mode": "balanced",
            "sample_rate": row["sample_rate"],
            "encoding": "pcm_s16le",
            "chunk_ms": 100,
        }

    def capacity_units(self, row: dict) -> int:
        return 1

    async def transcribe(self, pcm: bytes, config: dict) -> dict:
        query = {key: value for key, value in config.items() if key not in {"encoding", "chunk_ms"}}
        parts, metadata = [], {}
        terminated = asyncio.Event()
        async with connect(
            f"wss://streaming.assemblyai.com/v3/ws?{urlencode(query)}",
            additional_headers={"Authorization": self.api_key},
            open_timeout=20,
            close_timeout=10,
            ping_interval=None,
            max_size=None,
        ) as websocket:

            async def receive() -> None:
                async for raw in websocket:
                    message = json.loads(raw)
                    if message.get("type") == "Begin":
                        metadata["session_id"] = message.get("id")
                    elif message.get("type") == "Turn" and message.get("end_of_turn"):
                        transcript = message.get("transcript", "").strip()
                        if transcript:
                            parts.append(transcript)
                    elif message.get("type") == "Termination":
                        metadata["audio_duration_seconds"] = message.get("audio_duration_seconds")
                        terminated.set()

            receiver = asyncio.create_task(receive())
            try:
                chunk_bytes = config["sample_rate"] * 2 * config["chunk_ms"] // 1000
                async for chunk in realtime_chunks(pcm, config["sample_rate"], chunk_bytes):
                    await websocket.send(chunk)
                await websocket.send(json.dumps({"type": "Terminate"}))
                await asyncio.wait_for(terminated.wait(), 20)
            finally:
                if not receiver.done():
                    receiver.cancel()
                await asyncio.gather(receiver, return_exceptions=True)
        return {"text": " ".join(parts), **metadata}


def main() -> None:
    args = parser(
        "Stream the benchmark through AssemblyAI Universal-3.5 Pro",
        "assemblyai_universal_3_5_pro.jsonl",
        concurrency=100,
        starts=100,
    ).parse_args()
    run_from_args(AssemblyAI, args)


if __name__ == "__main__":
    main()
