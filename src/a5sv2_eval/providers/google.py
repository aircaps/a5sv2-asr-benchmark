from __future__ import annotations

import math
import os

from google.api_core.client_options import ClientOptions
from google.cloud import speech_v2

from a5sv2_eval.common import parser, realtime_chunks, run_from_args, stitch_boundary


class Google:
    system_id = "google_chirp_3_streaming"
    system = "Google Chirp 3 Streaming"
    model = "chirp_3"
    session_audio_seconds = 285
    overlap_seconds = 5

    def __init__(self, project: str, region: str) -> None:
        self.region = region
        self.recognizer = f"projects/{project}/locations/{region}/recognizers/_"
        self.client = speech_v2.SpeechAsyncClient(
            client_options=ClientOptions(
                api_endpoint=f"{region}-speech.googleapis.com",
                quota_project_id=project,
            )
        )

    def streaming_config(self, row: dict) -> dict:
        return {
            "model": self.model,
            "language": "en-US",
            "encoding": "LINEAR16",
            "sample_rate": row["sample_rate"],
            "channels": 1,
            "automatic_punctuation": True,
            "region": self.region,
            "chunk_bytes": 14_400,
            "session_audio_seconds": self.session_audio_seconds,
            "overlap_seconds": self.overlap_seconds,
            "stitching": "result_end_offset_then_exact_normalized_token_overlap_v1",
        }

    def capacity_units(self, row: dict) -> int:
        return math.ceil(row["sample_rate"] / 8000)

    async def _transcribe_session(self, pcm: bytes, config: dict) -> list[dict]:
        recognition_config = speech_v2.RecognitionConfig(
            explicit_decoding_config=speech_v2.ExplicitDecodingConfig(
                encoding=speech_v2.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=config["sample_rate"],
                audio_channel_count=1,
            ),
            language_codes=["en-US"],
            model=self.model,
            features=speech_v2.RecognitionFeatures(enable_automatic_punctuation=True),
        )
        first = speech_v2.StreamingRecognizeRequest(
            recognizer=self.recognizer,
            streaming_config=speech_v2.StreamingRecognitionConfig(
                config=recognition_config,
                streaming_features=speech_v2.StreamingRecognitionFeatures(interim_results=False),
            ),
        )

        async def requests():
            yield first
            async for chunk in realtime_chunks(pcm, config["sample_rate"], config["chunk_bytes"]):
                yield speech_v2.StreamingRecognizeRequest(audio=chunk)

        parts = []
        responses = await self.client.streaming_recognize(requests=requests())
        async for response in responses:
            for result in response.results:
                if result.is_final and result.alternatives:
                    text = result.alternatives[0].transcript.strip()
                    if text:
                        parts.append(
                            {
                                "text": text,
                                "result_end_seconds": _duration_seconds(result.result_end_offset),
                            }
                        )
        return parts

    async def transcribe(self, pcm: bytes, config: dict) -> dict:
        sample_rate = config["sample_rate"]
        bytes_per_second = sample_rate * 2
        session_bytes = int(config["session_audio_seconds"] * bytes_per_second)
        overlap_bytes = int(config["overlap_seconds"] * bytes_per_second)
        step_bytes = session_bytes - overlap_bytes
        if step_bytes <= 0:
            raise ValueError("Google session overlap must be shorter than the session")

        stitched = ""
        segments = []
        start = 0
        while start < len(pcm):
            end = min(start + session_bytes, len(pcm))
            session_results = await self._transcribe_session(pcm[start:end], config)
            replay_seconds = (
                0 if start == 0 else min(config["overlap_seconds"], start / bytes_per_second)
            )
            kept = [
                result
                for result in session_results
                if result["result_end_seconds"] > replay_seconds
            ]
            session_text = " ".join(result["text"] for result in kept)
            stitched, removed_words = stitch_boundary(stitched, session_text)
            segments.append(
                {
                    "index": len(segments),
                    "source_start_seconds": start / bytes_per_second,
                    "source_end_seconds": end / bytes_per_second,
                    "audio_seconds": (end - start) / bytes_per_second,
                    "replayed_seconds": replay_seconds,
                    "final_results": len(session_results),
                    "results_discarded_in_replay": len(session_results) - len(kept),
                    "stitch_overlap_words": removed_words,
                    "transcript_raw": " ".join(result["text"] for result in session_results),
                    "transcript_after_time_filter": session_text,
                }
            )
            if end == len(pcm):
                break
            start += step_bytes

        return {
            "text": stitched,
            "session_count": len(segments),
            "streamed_audio_seconds": sum(segment["audio_seconds"] for segment in segments),
            "segments": segments,
        }

    async def close(self) -> None:
        await self.client.transport.close()


def _duration_seconds(value) -> float:
    if hasattr(value, "total_seconds"):
        return value.total_seconds()
    return float(value.seconds) + float(getattr(value, "microseconds", 0)) / 1_000_000


def main() -> None:
    args_parser = parser(
        "Stream the benchmark through Google Chirp 3",
        "google_chirp_3.jsonl",
        concurrency=42,
    )
    args_parser.add_argument("--project", default=os.getenv("GOOGLE_CLOUD_PROJECT"))
    args_parser.add_argument("--region", default=os.getenv("GOOGLE_LOCATION") or "us")
    args = args_parser.parse_args()
    if not args.project:
        args_parser.error("set GOOGLE_CLOUD_PROJECT or pass --project")
    run_from_args(lambda: Google(args.project, args.region), args)


if __name__ == "__main__":
    main()
