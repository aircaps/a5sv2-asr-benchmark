import asyncio
import hashlib
import json
from types import SimpleNamespace

from a5sv2_eval.common import realtime_chunks, run_benchmark, run_trials, write_jsonl


class FakeProvider:
    system_id = "fake"
    system = "Fake"
    model = "fake-1"

    def __init__(self):
        self.calls = []

    def streaming_config(self, row):
        return {"sample_rate": row["sample_rate"]}

    def capacity_units(self, row):
        return 1

    async def transcribe(self, pcm, config):
        self.calls.append((pcm, config))
        return {"text": "hello", "request_id": "test"}


def manifest(tmp_path):
    pcm = b"\0\0" * 10
    (tmp_path / "clip.pcm").write_bytes(pcm)
    path = tmp_path / "manifest.jsonl"
    write_jsonl(
        path,
        [
            {
                "dataset_revision": "revision",
                "id": "clip",
                "source_sha256": "source",
                "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
                "pcm_path": "clip.pcm",
                "sample_rate": 16000,
                "condition": "noise",
                "reference_raw": "hello",
            }
        ],
    )
    return path


async def test_runner_journals_resumes_and_separates_trials(tmp_path):
    path = manifest(tmp_path)
    args = SimpleNamespace(
        manifest=path,
        output=tmp_path / "result.jsonl",
        concurrency=1,
        starts_per_minute=0,
        retries=0,
        limit=None,
        trials=2,
    )
    provider = FakeProvider()
    assert await run_trials(lambda: provider, args) == 0
    assert len(provider.calls) == 2
    assert json.loads(args.output.read_text())["trial"] == 1
    assert json.loads((tmp_path / "result.trial_2.jsonl").read_text())["trial"] == 2

    resumed = FakeProvider()
    assert await run_benchmark(resumed, path, args.output, 1, retries=0) == 0
    assert resumed.calls == []


async def test_realtime_chunks_are_not_faster_than_audio():
    started = asyncio.get_running_loop().time()
    chunks = [chunk async for chunk in realtime_chunks(bytes(640), 16000, 320)]
    assert chunks == [bytes(320), bytes(320)]
    assert asyncio.get_running_loop().time() - started >= 0.019
