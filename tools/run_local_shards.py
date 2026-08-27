from __future__ import annotations

import argparse
import asyncio
import heapq
import os
import sys
from pathlib import Path

from a5sv2_eval.common import read_jsonl, write_jsonl
from merge_shards import merge


MODULES = {
    "kyutai": "a5sv2_eval.providers.kyutai",
    "whisper": "a5sv2_eval.providers.whisper_streaming",
}


def balanced_manifests(manifest: Path, model: str, output: Path, count: int) -> list[Path]:
    """Assign longest recordings first to the currently lightest deterministic shard."""
    rows = read_jsonl(manifest)
    if not rows:
        raise ValueError(f"Manifest is empty: {manifest}")
    assignments: list[list[dict]] = [[] for _ in range(count)]
    queue = [(0.0, index) for index in range(count)]
    heapq.heapify(queue)
    ordered = sorted(
        enumerate(rows),
        key=lambda item: (-float(item[1].get("duration_seconds", 0)), item[0]),
    )
    for _, row in ordered:
        load, shard_index = heapq.heappop(queue)
        assignments[shard_index].append(row)
        heapq.heappush(queue, (load + float(row.get("duration_seconds", 0)), shard_index))

    paths = [
        manifest.with_name(
            f".{manifest.stem}.{model}.{output.stem}.balanced_{index}_of_{count}.jsonl"
        )
        for index in range(count)
    ]
    for path, assignment in zip(paths, assignments, strict=True):
        write_jsonl(path, assignment)
    return paths


async def run(args) -> int:
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one CUDA device index")
    shard_paths = [
        args.output.with_name(f"{args.output.stem}.shard_{index}_of_{len(gpus)}.jsonl")
        for index in range(len(gpus))
    ]
    shard_manifests = balanced_manifests(args.manifest, args.model, args.output, len(gpus))
    processes = []
    for index, (gpu, shard_path, shard_manifest) in enumerate(
        zip(gpus, shard_paths, shard_manifests, strict=True)
    ):
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = gpu
        if args.endpoints:
            endpoints = [value.strip() for value in args.endpoints.split(",") if value.strip()]
            if len(endpoints) != len(gpus):
                raise ValueError("--endpoints must contain one URL per GPU/shard")
            environment["VLLM_REALTIME_URL"] = endpoints[index]
        command = [
            sys.executable,
            "-m",
            MODULES[args.model],
            "--manifest",
            str(shard_manifest),
            "--output",
            str(shard_path),
            "--trials",
            "1",
            "--shard-count",
            str(len(gpus)),
            "--shard-index",
            str(index),
            "--pre-sharded",
            "--concurrency",
            str(args.concurrency_per_shard),
        ]
        processes.append(await asyncio.create_subprocess_exec(*command, env=environment))
    codes = await asyncio.gather(*(process.wait() for process in processes))
    if any(codes):
        return 1
    merge(args.manifest, shard_paths, args.output)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Shard one local model across Runpod GPUs")
    parser.add_argument("model", choices=MODULES)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency-per-shard", type=int, default=1)
    parser.add_argument(
        "--endpoints",
        help="Comma-separated realtime WebSocket URLs, one per GPU/shard",
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
