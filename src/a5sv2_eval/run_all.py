from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

PROVIDERS = {
    "deepgram": ("a5sv2_eval.providers.deepgram", "DEEPGRAM_API_KEY", "deepgram_nova_3.jsonl"),
    "assemblyai": (
        "a5sv2_eval.providers.assemblyai",
        "ASSEMBLYAI_API_KEY",
        "assemblyai_universal_3_5_pro.jsonl",
    ),
    "google": ("a5sv2_eval.providers.google", "GOOGLE_CLOUD_PROJECT", "google_chirp_3.jsonl"),
    "openai": (
        "a5sv2_eval.providers.openai",
        "OPENAI_API_KEY",
        "openai_gpt_live_transcribe.jsonl",
    ),
    "elevenlabs": (
        "a5sv2_eval.providers.elevenlabs",
        "ELEVENLABS_API_KEY",
        "elevenlabs_scribe_v2_realtime.jsonl",
    ),
    "nvidia": (
        "a5sv2_eval.providers.nvidia",
        "TOGETHER_API_KEY",
        "nvidia_nemotron_3_asr_streaming_0_6b.jsonl",
    ),
    "mistral": (
        "a5sv2_eval.providers.mistral",
        "MISTRAL_API_KEY",
        "mistral_voxtral_mini_transcribe_realtime_2602.jsonl",
    ),
    "kyutai": ("a5sv2_eval.providers.kyutai", None, "kyutai_stt_2_6b_en.jsonl"),
    "whisper": (
        "a5sv2_eval.providers.whisper_streaming",
        None,
        "whisper_large_v3_ufal_streaming.jsonl",
    ),
}
DEFAULT_PROVIDERS = ["deepgram", "assemblyai", "google", "openai", "elevenlabs"]


async def run(
    names: list[str], trials: int, manifest: Path, output_dir: Path, concurrency: int | None
) -> int:
    missing = [
        PROVIDERS[name][1]
        for name in names
        if PROVIDERS[name][1] and not os.getenv(PROVIDERS[name][1])
    ]
    if missing:
        raise RuntimeError(f"Missing environment variables: {', '.join(missing)}")
    processes = []
    try:
        commands = [
            [
                sys.executable,
                "-m",
                PROVIDERS[name][0],
                "--trials",
                str(trials),
                "--manifest",
                str(manifest),
                "--output",
                str(output_dir / PROVIDERS[name][2]),
            ]
            for name in names
        ]
        if concurrency:
            for command in commands:
                command.extend(["--concurrency", str(concurrency)])
        processes = [await asyncio.create_subprocess_exec(*command) for command in commands]
        return int(any(await asyncio.gather(*(process.wait() for process in processes))))
    finally:
        for process in processes:
            if process.returncode is None:
                process.terminate()
        await asyncio.gather(*(process.wait() for process in processes), return_exceptions=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run streaming ASR providers in parallel")
    parser.add_argument("providers", nargs="*", choices=PROVIDERS, default=DEFAULT_PROVIDERS)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--manifest", type=Path, default=Path("benchmark_data/mega/manifest.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_data/results"))
    args = parser.parse_args()
    raise SystemExit(
        asyncio.run(
            run(args.providers, args.trials, args.manifest, args.output_dir, args.concurrency)
        )
    )


if __name__ == "__main__":
    main()
