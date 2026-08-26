import json

import a5sv2_eval.providers.assemblyai as assembly_module
import a5sv2_eval.providers.deepgram as deepgram_module
import a5sv2_eval.providers.elevenlabs as elevenlabs_module
import a5sv2_eval.providers.openai as openai_module
from a5sv2_eval.providers.assemblyai import AssemblyAI
from a5sv2_eval.providers.deepgram import Deepgram
from a5sv2_eval.providers.elevenlabs import ElevenLabs
from a5sv2_eval.providers.google import Google, stitch_boundary
from a5sv2_eval.providers.openai import OpenAI


class FakeWebSocket:
    def __init__(self, responses):
        self.responses = responses
        self.sent = []

    async def send(self, message):
        self.sent.append(message)

    async def __aiter__(self):
        for response in self.responses:
            yield json.dumps(response)


class FakeConnection:
    def __init__(self, websocket):
        self.websocket = websocket

    async def __aenter__(self):
        return self.websocket

    async def __aexit__(self, *args):
        return False


async def one_chunk(pcm, sample_rate, chunk_bytes):
    yield pcm


def test_provider_configs_preserve_native_rate(monkeypatch):
    for name in [
        "DEEPGRAM_API_KEY",
        "ASSEMBLYAI_API_KEY",
        "ELEVENLABS_API_KEY",
        "OPENAI_API_KEY",
    ]:
        monkeypatch.setenv(name, "test")
    row = {"sample_rate": 48000}

    assert Deepgram().streaming_config(row)["sample_rate"] == 48000
    assert AssemblyAI().streaming_config(row) == {
        "speech_model": "universal-3-5-pro",
        "mode": "balanced",
        "sample_rate": 48000,
        "encoding": "pcm_s16le",
        "chunk_ms": 100,
    }
    google = object.__new__(Google)
    google.region = "us"
    assert google.streaming_config(row)["sample_rate"] == 48000
    assert google.capacity_units(row) == 6
    assert google.streaming_config(row)["session_audio_seconds"] == 285
    assert ElevenLabs().streaming_config(row)["audio_format"] == "pcm_48000"
    assert OpenAI().streaming_config(row)["sample_rate"] == 24000


def test_google_stitch_boundary_only_removes_exact_normalized_overlap():
    stitched, removed = stitch_boundary(
        "One sentence ends with Hello, WORLD!", "hello world and this continues"
    )
    assert stitched == "One sentence ends with Hello, WORLD! and this continues"
    assert removed == 2

    stitched, removed = stitch_boundary("we really agree", "we agree on this")
    assert stitched == "we really agree we agree on this"
    assert removed == 0

    stitched, removed = stitch_boundary("we heard one two", "one—two more")
    assert stitched == "we heard one two more"
    assert removed == 1


async def test_google_rolls_over_filters_replay_and_records_audit_metadata():
    provider = object.__new__(Google)
    provider.region = "us"
    calls = []
    responses = [
        [{"text": "start shared words", "result_end_seconds": 285.0}],
        [
            {"text": "shared words", "result_end_seconds": 4.0},
            {"text": "shared words finish", "result_end_seconds": 20.0},
        ],
    ]

    async def transcribe_session(pcm, config):
        calls.append(len(pcm))
        return responses[len(calls) - 1]

    provider._transcribe_session = transcribe_session
    config = provider.streaming_config({"sample_rate": 10})
    result = await provider.transcribe(bytes(300 * 10 * 2), config)

    assert calls == [285 * 10 * 2, 20 * 10 * 2]
    assert result["text"] == "start shared words finish"
    assert result["session_count"] == 2
    assert result["streamed_audio_seconds"] == 305
    assert result["segments"][1]["source_start_seconds"] == 280
    assert result["segments"][1]["results_discarded_in_replay"] == 1


async def test_openai_commits_one_manual_turn(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    websocket = FakeWebSocket(
        [
            {"type": "session.updated"},
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": "item",
                "transcript": "hello",
                "usage": {"type": "duration", "seconds": 0.1},
            },
        ]
    )
    monkeypatch.setattr(openai_module, "connect", lambda *a, **k: FakeConnection(websocket))
    monkeypatch.setattr(openai_module, "realtime_chunks", one_chunk)
    provider = OpenAI()
    result = await provider.transcribe(
        bytes(4800), provider.streaming_config({"sample_rate": 24000})
    )

    assert json.loads(websocket.sent[-1]) == {"type": "input_audio_buffer.commit"}
    assert json.loads(websocket.sent[0])["session"]["audio"]["input"]["transcription"] == {
        "model": "gpt-live-transcribe",
        "languages": ["en"],
        "delay": "xhigh",
    }
    assert result["text"] == "hello"


async def test_deepgram_closes_stream(monkeypatch):
    monkeypatch.setenv("DEEPGRAM_API_KEY", "test")
    websocket = FakeWebSocket(
        [
            {
                "type": "Results",
                "is_final": True,
                "channel": {"alternatives": [{"transcript": "hello"}]},
                "metadata": {"request_id": "request"},
            },
            {"type": "Metadata", "request_id": "request"},
        ]
    )
    monkeypatch.setattr(deepgram_module, "connect", lambda *a, **k: FakeConnection(websocket))
    monkeypatch.setattr(deepgram_module, "realtime_chunks", one_chunk)
    provider = Deepgram()
    result = await provider.transcribe(b"\0\0", provider.streaming_config({"sample_rate": 16000}))

    assert websocket.sent == [b"\0\0", json.dumps({"type": "CloseStream"})]
    assert result["text"] == "hello"


async def test_assemblyai_terminates_stream(monkeypatch):
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "test")
    websocket = FakeWebSocket(
        [
            {"type": "Begin", "id": "session"},
            {"type": "Turn", "end_of_turn": True, "transcript": "hello"},
            {"type": "Termination", "audio_duration_seconds": 1},
        ]
    )
    monkeypatch.setattr(assembly_module, "connect", lambda *a, **k: FakeConnection(websocket))
    monkeypatch.setattr(assembly_module, "realtime_chunks", one_chunk)
    provider = AssemblyAI()
    result = await provider.transcribe(b"\0\0", provider.streaming_config({"sample_rate": 16000}))

    assert websocket.sent == [b"\0\0", json.dumps({"type": "Terminate"})]
    assert result["text"] == "hello"


async def test_elevenlabs_commits_last_chunk(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test")
    websocket = FakeWebSocket(
        [
            {"message_type": "session_started", "session_id": "session"},
            {"message_type": "committed_transcript", "text": "hello"},
        ]
    )
    monkeypatch.setattr(elevenlabs_module, "connect", lambda *a, **k: FakeConnection(websocket))
    monkeypatch.setattr(elevenlabs_module, "realtime_chunks", one_chunk)
    provider = ElevenLabs()
    result = await provider.transcribe(b"\0\0", provider.streaming_config({"sample_rate": 16000}))

    sent = json.loads(websocket.sent[0])
    assert sent["commit"] is True
    assert sent["sample_rate"] == 16000
    assert result["text"] == "hello"
