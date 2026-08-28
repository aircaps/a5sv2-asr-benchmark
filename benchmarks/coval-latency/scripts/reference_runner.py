#!/usr/bin/env python3
"""Public timing harness contract; provide a separate backend implementation."""
from __future__ import annotations

import json
import sys
import time
import wave
from array import array
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

SAMPLE_RATE = 16_000
SAMPLE_WIDTH = 2
CHUNK_FRAMES = 1_600


class Session(Protocol):
    @property
    def complete_transcript(self) -> str: ...

    def feed(self, samples: array[float]) -> str: ...

    def finalize(self) -> str: ...

    def close(self) -> None: ...


class Backend(Protocol):
    """The experiment is public; model-specific implementations may remain external."""

    variant: str

    def warmup(self) -> None: ...

    def begin(self) -> Session: ...


@dataclass(frozen=True)
class Item:
    dataset_id: str
    dataset_version: str
    duration_sec: float
    path: str
    sample_id: str
    sha256: str
    speech_end_offset_ms: float
    transcript: str


def read_and_trim(path: Path, item: Item) -> array[float]:
    with wave.open(str(path), "rb") as source:
        actual = (source.getframerate(), source.getnchannels(), source.getsampwidth())
        if actual != (SAMPLE_RATE, 1, SAMPLE_WIDTH):
            raise ValueError(f"expected 16 kHz mono PCM16, got {actual}")
        pcm = source.readframes(source.getnframes())
    trailing_ms = max(0.0, item.duration_sec * 1000.0 - item.speech_end_offset_ms)
    tail_bytes = int(round(trailing_ms / 1000.0 * SAMPLE_RATE)) * SAMPLE_WIDTH
    if 0 < tail_bytes < len(pcm):
        pcm = pcm[:-tail_bytes]
    values = array("h")
    values.frombytes(pcm)
    if sys.byteorder != "little":
        values.byteswap()
    return array("f", (sample / 32768.0 for sample in values))


def sleep_until(deadline_ns: int) -> None:
    remaining = deadline_ns - time.monotonic_ns()
    if remaining > 0:
        time.sleep(remaining / 1_000_000_000)


def run_item(backend: Backend, item: Item, audio_path: Path, run_id: str) -> dict[str, object]:
    """Run one item using the released pacing and timing definitions."""
    samples = read_and_trim(audio_path, item)
    session = backend.begin()
    started_at = datetime.now(tz=UTC).isoformat()
    start_ns = time.monotonic_ns()
    first_token_seconds: float | None = None
    sent_frames = 0
    try:
        for offset in range(0, len(samples), CHUNK_FRAMES):
            chunk = samples[offset : offset + CHUNK_FRAMES]
            snapshot = session.feed(chunk)
            if first_token_seconds is None and snapshot.strip():
                first_token_seconds = (time.monotonic_ns() - start_ns) / 1e9
            sent_frames += len(chunk)
            sleep_until(start_ns + round(sent_frames / SAMPLE_RATE * 1e9))
        snapshot = session.finalize()
        if first_token_seconds is None and snapshot.strip():
            first_token_seconds = (time.monotonic_ns() - start_ns) / 1e9
        final_seconds = (time.monotonic_ns() - start_ns) / 1e9
        return {
            "schema_version": "1.0",
            "method_id": "coval-stt-oracle@52d72516b5158c693d57eef43bd9044a7cd0a28d",
            "run_id": run_id,
            "trial": 1,
            "variant": backend.variant,
            "dataset_id": item.dataset_id,
            "dataset_version": item.dataset_version,
            "sample_id": item.sample_id,
            "audio_path": item.path,
            "audio_sha256": item.sha256,
            "reference": item.transcript,
            "hypothesis": session.complete_transcript.strip(),
            "speech_end_offset_ms": item.speech_end_offset_ms,
            "effective_audio_seconds": len(samples) / SAMPLE_RATE,
            "time_to_first_token_seconds": first_token_seconds,
            "audio_to_final_seconds": final_seconds,
            "time_to_final_segment_seconds": max(
                0.0, final_seconds - item.speech_end_offset_ms / 1000.0
            ),
            "started_at_utc": started_at,
            "status": "ok",
            "error": None,
        }
    finally:
        session.close()


def append_result(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def load_plan(path: Path) -> list[Item]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [Item(**item) for item in payload["items"]]


def row_json(row: dict[str, object]) -> str:
    return json.dumps(asdict(row) if hasattr(row, "__dataclass_fields__") else row, sort_keys=True)
