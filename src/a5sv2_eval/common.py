from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import mmap
import os
import time
import unicodedata
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_MANIFEST = Path("benchmark_data/mega/manifest.jsonl")


def row_key(row: dict) -> tuple[str, str, str]:
    return row["dataset_revision"], row["id"], row["source_sha256"]


def _normalized_tokens(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("'", "").replace("’", "").replace("`", "")
    text = "".join(
        character if character.isalnum() or character.isspace() else " " for character in text
    )
    return text.split()


def _right_token_ends(words: list[str]) -> tuple[list[str], list[int]]:
    tokens = []
    raw_word_ends = []
    for index, word in enumerate(words, 1):
        normalized = _normalized_tokens(word)
        tokens.extend(normalized)
        raw_word_ends.extend([index] * len(normalized))
    return tokens, raw_word_ends


def stitch_boundary(existing: str, continuation: str, maximum_words: int = 80) -> tuple[str, int]:
    """Join overlapped sessions without changing non-duplicate provider text."""
    if not existing:
        return continuation.strip(), 0
    if not continuation:
        return existing.strip(), 0
    left = existing.split()
    right = continuation.split()
    left_keys = _normalized_tokens(existing)
    right_keys, right_word_ends = _right_token_ends(right)
    maximum = min(maximum_words, len(left_keys), len(right_keys))
    overlap = next(
        (count for count in range(maximum, 0, -1) if left_keys[-count:] == right_keys[:count]),
        0,
    )
    removed_words = right_word_ends[overlap - 1] if overlap else 0
    return " ".join([*left, *right[removed_words:]]), removed_words


async def realtime_chunks(pcm: bytes, sample_rate: int, chunk_bytes: int) -> AsyncIterator[bytes]:
    """Yield PCM chunks no faster than real time, without catch-up bursts."""
    clock = asyncio.get_running_loop()
    deadline = clock.time()
    for offset in range(0, len(pcm), chunk_bytes):
        chunk = pcm[offset : offset + chunk_bytes]
        yield chunk
        deadline = max(deadline + len(chunk) / (sample_rate * 2), clock.time())
        await asyncio.sleep(max(0, deadline - clock.time()))


class Capacity:
    def __init__(self, units: int):
        self.total = self.available = units
        self.condition = asyncio.Condition()

    @asynccontextmanager
    async def use(self, units: int):
        if units > self.total:
            raise ValueError(f"Session needs {units} units; only {self.total} configured")
        async with self.condition:
            await self.condition.wait_for(lambda: self.available >= units)
            self.available -= units
        try:
            yield
        finally:
            async with self.condition:
                self.available += units
                self.condition.notify_all()


class StartLimiter:
    def __init__(self, starts_per_minute: int):
        self.interval = 60 / starts_per_minute if starts_per_minute else 0
        self.next_start = 0.0
        self.lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self.lock:
            now = asyncio.get_running_loop().time()
            start = max(now, self.next_start)
            self.next_start = start + self.interval
        await asyncio.sleep(max(0, start - now))


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


async def run_benchmark(
    provider: Any,
    manifest_path: Path,
    output_path: Path,
    concurrency: int,
    starts_per_minute: int = 0,
    retries: int = 2,
    limit: int | None = None,
    trial: int = 1,
    trials: int = 1,
    close_provider: bool = True,
    shard_count: int = 1,
    shard_index: int = 0,
    pre_sharded: bool = False,
) -> int:
    maximum_concurrency = getattr(provider, "max_concurrency", None)
    if maximum_concurrency and concurrency > maximum_concurrency:
        raise ValueError(
            f"{provider.system} supports at most {maximum_concurrency} stream(s) per process; "
            "use deterministic GPU shards for parallel local inference"
        )
    manifest = read_jsonl(manifest_path)
    if limit is not None:
        manifest = manifest[:limit]
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("Require shard_count >= 1 and 0 <= shard_index < shard_count")
    if not pre_sharded:
        manifest = [row for index, row in enumerate(manifest) if index % shard_count == shard_index]
    if not manifest or len({row_key(row) for row in manifest}) != len(manifest):
        raise RuntimeError("Manifest is empty or contains duplicate utterance keys")
    maximum = getattr(provider, "max_audio_seconds", None)
    if maximum and any(row["duration_seconds"] > maximum for row in manifest):
        raise RuntimeError(f"{provider.system} cannot stream recordings longer than {maximum}s")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    journal_path = output_path.with_suffix(".progress.jsonl")
    existing = read_jsonl(output_path)
    if limit is not None and len(existing) > len(manifest):
        raise RuntimeError("Refusing to truncate an existing output; choose another --output")
    saved = existing + read_jsonl(journal_path)
    results = {row_key(row): row for row in saved if row.get("system_id") == provider.system_id}
    pending = [
        row
        for row in manifest
        if results.get(row_key(row), {}).get("status") != "ok"
        or results[row_key(row)].get("streaming_config") != provider.streaming_config(row)
    ]
    pending.sort(key=lambda row: row.get("duration_seconds", 0), reverse=True)

    capacity = Capacity(concurrency)
    slots = asyncio.Semaphore(concurrency)
    limiter = StartLimiter(starts_per_minute)
    write_lock = asyncio.Lock()
    completed = len(manifest) - len(pending)
    print(f"{provider.system} trial {trial}/{trials}: {completed}/{len(manifest)}", flush=True)

    async def save(result: dict) -> None:
        nonlocal completed
        async with write_lock:
            results[row_key(result)] = result
            with journal_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            completed += 1
            step = 1 if len(manifest) < 100 else 25
            if completed % step == 0 or completed == len(manifest):
                print(
                    f"{provider.system} trial {trial}/{trials}: {completed}/{len(manifest)}",
                    flush=True,
                )

    async def process(row: dict) -> None:
        async with slots:
            pcm_path = manifest_path.parent / row["pcm_path"]
            with (
                pcm_path.open("rb") as handle,
                mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as pcm,
            ):
                if hashlib.sha256(pcm).hexdigest() != row["pcm_sha256"]:
                    raise RuntimeError(f"PCM hash mismatch: {pcm_path}")
                config = provider.streaming_config(row)
                last_error = ""
                for attempt in range(1, retries + 2):
                    try:
                        async with capacity.use(provider.capacity_units(row)):
                            await limiter.wait()
                            started = time.monotonic()
                            response = await provider.transcribe(pcm, config)
                        await save(
                            {
                                **row,
                                "schema_version": "1.0",
                                "trial": trial,
                                "system_id": provider.system_id,
                                "system": provider.system,
                                "model": provider.model,
                                "prediction_raw": response.pop("text"),
                                "status": "ok",
                                "attempts": attempt,
                                "elapsed_seconds": time.monotonic() - started,
                                "streaming_config": config,
                                "run_config": {
                                    "concurrency": concurrency,
                                    "starts_per_minute": starts_per_minute,
                                    "retries": retries,
                                    "trials": trials,
                                    "shard_count": shard_count,
                                    "shard_index": shard_index,
                                },
                                "provider_metadata": response,
                                "completed_at": datetime.now(timezone.utc).isoformat(),
                            }
                        )
                        return
                    except Exception as error:
                        last_error = f"{type(error).__name__}: {error}"
                        print(
                            f"{provider.system} {row['id']} attempt {attempt}: {last_error}",
                            flush=True,
                        )
                        if attempt <= retries:
                            await asyncio.sleep(2**attempt)

            await save(
                {
                    **row,
                    "schema_version": "1.0",
                    "trial": trial,
                    "system_id": provider.system_id,
                    "system": provider.system,
                    "model": provider.model,
                    "prediction_raw": "",
                    "status": "error",
                    "attempts": retries + 1,
                    "error": last_error,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                }
            )

    try:
        await asyncio.gather(*(process(row) for row in pending))
    finally:
        if close_provider and hasattr(provider, "close"):
            await provider.close()

    ordered = [results[row_key(row)] for row in manifest]
    write_jsonl(output_path, ordered)
    journal_path.unlink(missing_ok=True)
    failures = sum(row["status"] != "ok" for row in ordered)
    print(f"Saved {output_path} ({failures} failures)")
    return failures


def trial_output(path: Path, trial: int) -> Path:
    return path if trial == 1 else path.with_name(f"{path.stem}.trial_{trial}{path.suffix}")


async def run_trials(provider_factory, args) -> int:
    if args.trials < 1:
        raise ValueError("--trials must be at least 1")
    failures = 0
    provider = provider_factory()
    try:
        for trial in range(1, args.trials + 1):
            failures += await run_benchmark(
                provider,
                args.manifest,
                trial_output(args.output, trial),
                args.concurrency,
                args.starts_per_minute,
                args.retries,
                args.limit,
                trial,
                args.trials,
                close_provider=False,
                shard_count=getattr(args, "shard_count", 1),
                shard_index=getattr(args, "shard_index", 0),
                pre_sharded=getattr(args, "pre_sharded", False),
            )
    finally:
        if hasattr(provider, "close"):
            await provider.close()
    return failures


def run_from_args(provider_factory, args) -> None:
    raise SystemExit(bool(asyncio.run(run_trials(provider_factory, args))))


def parser(description: str, output_name: str, concurrency: int, starts: int = 0):
    result = argparse.ArgumentParser(description=description)
    result.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    result.add_argument("--output", type=Path, default=Path("benchmark_data/results") / output_name)
    result.add_argument("--concurrency", type=int, default=concurrency)
    result.add_argument("--starts-per-minute", type=int, default=starts)
    result.add_argument("--retries", type=int, default=2)
    result.add_argument("--limit", type=int)
    result.add_argument("--trials", type=int, default=5)
    result.add_argument("--shard-count", type=int, default=1)
    result.add_argument("--shard-index", type=int, default=0)
    result.add_argument(
        "--pre-sharded",
        action="store_true",
        help="Manifest is already a complete shard; retain shard metadata without modulo filtering",
    )
    return result
