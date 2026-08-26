from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import wave
import zipfile
from pathlib import Path
from urllib.request import urlopen
from xml.etree import ElementTree as ET

from datasets import Audio, load_dataset
from huggingface_hub import hf_hub_download
import numpy as np
from pyarrow import parquet

from a5sv2_eval.common import write_jsonl
from a5sv2_eval.dataset import AMI, DIPCO, MEGA, NOTSOFAR
from a5sv2_eval.score import normalize


def file_hash(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, path: Path) -> Path:
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with urlopen(url) as source, temporary.open("wb") as target:
        copied = 0
        while chunk := source.read(8 * 1024 * 1024):
            target.write(chunk)
            copied += len(chunk)
            if copied // (256 << 20) != (copied - len(chunk)) // (256 << 20):
                print(f"Downloaded {path.name}: {copied / (1 << 30):.1f} GiB", flush=True)
    temporary.replace(path)
    return path


def read_wav(path: Path) -> tuple[bytes, int, int]:
    with wave.open(str(path)) as audio:
        if (audio.getnchannels(), audio.getsampwidth(), audio.getcomptype()) != (1, 2, "NONE"):
            raise RuntimeError(f"Expected mono PCM16 WAV: {path}")
        rate, frames = audio.getframerate(), audio.getnframes()
        return audio.readframes(frames), rate, frames


def apply_gain(pcm: bytes, gain_db: float) -> bytes:
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float64)
    samples = np.rint(samples * 10 ** (gain_db / 20))
    if np.max(np.abs(samples), initial=0) > 32767:
        raise RuntimeError(f"{gain_db:+g} dB clips")
    return samples.astype("<i2").tobytes()


def save_meeting(
    root: Path, dataset: str, revision: str, meeting: str, wav: Path, reference: str, **metadata
) -> dict:
    pcm, rate, frames = read_wav(wav)
    gain_db = metadata.get("gain_db", 0.0)
    if gain_db:
        pcm = apply_gain(pcm, gain_db)
    path = root / "pcm" / dataset / f"{meeting}.pcm"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pcm)
    return {
        "dataset_id": dataset,
        "dataset_revision": revision,
        "id": meeting,
        "condition": dataset.lower(),
        "meeting_id": meeting,
        "reference_raw": " ".join(reference.split()),
        "sample_rate": rate,
        "num_samples": frames,
        "duration_seconds": frames / rate,
        "pcm_path": str(path.relative_to(root)),
        "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
        "source_sha256": file_hash(wav),
        "streaming_scope": "full_recording",
        **metadata,
    }


def prepare_mega(root: Path) -> list[dict]:
    rows, pcm_dir = [], root / "pcm"
    pcm_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(MEGA["id"], revision=MEGA["revision"])
    for condition in MEGA["conditions"]:
        for row in dataset[condition].cast_column("audio", Audio(decode=False)):
            payload = row["audio"]
            wav = payload["bytes"] or Path(payload["path"]).read_bytes()
            if hashlib.sha256(wav).hexdigest() != row["sha256"]:
                raise RuntimeError(f"Source hash mismatch: {row['id']}")
            with wave.open(io.BytesIO(wav)) as audio:
                rate, frames = audio.getframerate(), audio.getnframes()
                pcm = audio.readframes(frames)
            path = pcm_dir / f"{row['id']}.pcm"
            path.write_bytes(pcm)
            rows.append(
                {
                    "dataset_id": MEGA["id"],
                    "dataset_revision": MEGA["revision"],
                    "id": row["id"],
                    "condition": condition,
                    "row_index": int(row["row_index"]),
                    "reference_raw": row["text"],
                    "sample_rate": rate,
                    "num_samples": frames,
                    "duration_seconds": frames / rate,
                    "pcm_path": str(path.relative_to(root)),
                    "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
                    "source_sha256": row["sha256"],
                    "streaming_scope": "utterance",
                }
            )
            if len(rows) % 100 == 0:
                print(f"Mega-ASR: {len(rows)}/1250", flush=True)
    return rows


def dipco_audio(source: Path, split: str, meeting: str) -> tuple[Path, str]:
    name = f"{meeting}_{DIPCO['device']}.{DIPCO['channel']}.wav"
    if meeting not in DIPCO["mirror_shards"]:
        return (
            Path(
                hf_hub_download(
                    DIPCO["repo"],
                    f"audio/{split}/{name}",
                    repo_type="dataset",
                    revision=DIPCO["revision"],
                )
            ),
            DIPCO["repo"],
        )
    shard = DIPCO["mirror_shards"][meeting]
    path = source / "dipco" / name
    if not path.is_file():
        parquet_path = hf_hub_download(
            DIPCO["eval_mirror"],
            f"data/eval-{shard:05d}-of-00024.parquet",
            repo_type="dataset",
            revision=DIPCO["eval_mirror_revision"],
        )
        rows = parquet.read_table(parquet_path, columns=["audio", "session_id"]).to_pylist()
        matches = [row for row in rows if row["session_id"] == name.removesuffix(".wav")]
        if len(matches) != 1 or not matches[0]["audio"]["bytes"]:
            raise RuntimeError(f"Missing {name} from pinned DiPCo eval mirror")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(matches[0]["audio"]["bytes"])
    return path, DIPCO["eval_mirror"]


def prepare_dipco(root: Path, source: Path) -> list[dict]:
    rows = []
    for split, meetings in DIPCO["meetings"].items():
        for meeting in meetings:
            transcript = Path(
                hf_hub_download(
                    DIPCO["repo"],
                    f"transcriptions/{split}/{meeting}.json",
                    repo_type="dataset",
                    revision=DIPCO["revision"],
                )
            )
            wav, transport = dipco_audio(source, split, meeting)
            annotations = json.loads(transcript.read_text())
            tags = set(re.findall(r"\[[^]]+]", " ".join(row["words"] for row in annotations)))
            if tags - {"[noise]", "[unintelligible]", "[laugh]"}:
                raise RuntimeError(f"Unknown DiPCo tags: {sorted(tags)}")

            def seconds(value: str) -> float:
                hours, minutes, seconds_ = value.split(":")
                return int(hours) * 3600 + int(minutes) * 60 + float(seconds_)

            annotations.sort(
                key=lambda row: (seconds(row["start_time"][DIPCO["device"]]), row["speaker_id"])
            )
            reference = " ".join(
                re.sub(r"\[(noise|unintelligible|laugh)]", " ", row["words"]) for row in annotations
            )
            rows.append(
                save_meeting(
                    root,
                    "DiPCo",
                    DIPCO["revision"],
                    meeting,
                    wav,
                    reference,
                    split=split,
                    device=DIPCO["device"],
                    channel=DIPCO["channel"],
                    gain_db=DIPCO["gain_db"],
                    source_transport=transport,
                    source_transport_revision=(
                        DIPCO["eval_mirror_revision"]
                        if transport == DIPCO["eval_mirror"]
                        else DIPCO["revision"]
                    ),
                    transcript_sha256=file_hash(transcript),
                )
            )
            print(f"DiPCo: {meeting}", flush=True)
    return rows


def ami_reference(archive: zipfile.ZipFile, meeting: str) -> str:
    words = []
    for name in sorted(
        name
        for name in archive.namelist()
        if re.search(rf"(^|/){meeting}\.[^.]+\.words\.xml$", name)
    ):
        speaker = Path(name).name.split(".")[1]
        for node in ET.fromstring(archive.read(name)):
            if node.tag.rsplit("}", 1)[-1] != "w" or not (node.text or "").strip():
                continue
            try:
                words.append((float(node.attrib["starttime"]), speaker, node.text.strip()))
            except (KeyError, ValueError):
                pass
    return " ".join(word for _, _, word in sorted(words))


def prepare_ami(root: Path, source: Path) -> list[dict]:
    annotations = download(
        "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip",
        source / "ami_public_manual_1.6.2.zip",
    )
    rows = []
    with zipfile.ZipFile(annotations) as archive:
        for meeting in AMI["meetings"]:
            name = f"{meeting}.{AMI['channel']}.wav"
            wav = download(
                f"https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/{meeting}/audio/{name}",
                source / "ami" / name,
            )
            rows.append(
                save_meeting(
                    root,
                    "AMI",
                    AMI["revision"],
                    meeting,
                    wav,
                    ami_reference(archive, meeting),
                    split="scenario-only unseen evaluation",
                    channel=AMI["channel"],
                    gain_db=0.0,
                    transcript_sha256=file_hash(annotations),
                )
            )
            print(f"AMI: {meeting}", flush=True)
    return rows


def prepare_notsofar(root: Path) -> list[dict]:
    rows = []
    prefix = f"benchmark-datasets/eval_set/{NOTSOFAR['version']}/MTG"
    for meeting in NOTSOFAR["meetings"]:
        files = {
            name: Path(
                hf_hub_download(
                    "microsoft/NOTSOFAR",
                    f"{prefix}/{meeting}/{name}",
                    repo_type="dataset",
                    revision=NOTSOFAR["revision"],
                )
            )
            for name in (NOTSOFAR["audio"], "devices.json", "gt_transcription.json")
        }
        devices = json.loads(files["devices.json"].read_text())
        valid = [
            device
            for device in devices
            if not device["is_mc"]
            and not device["is_close_talk"]
            and device["wav_file_names"] == NOTSOFAR["audio"]
        ]
        if len(valid) != 1:
            raise RuntimeError(f"Expected one {NOTSOFAR['audio']} stream for {meeting}")
        transcript = json.loads(files["gt_transcription.json"].read_text())
        transcript.sort(key=lambda row: (row["start_time"], row["speaker_id"], row["end_time"]))
        reference = " ".join(re.sub(r"<[^>]*>", " ", row.get("text", "")) for row in transcript)
        rows.append(
            save_meeting(
                root,
                "NOTSOFAR",
                NOTSOFAR["revision"],
                meeting,
                files[NOTSOFAR["audio"]],
                reference,
                split=f"eval-small {NOTSOFAR['version']}",
                channel=NOTSOFAR["audio"],
                gain_db=0.0,
                transcript_sha256=file_hash(files["gt_transcription.json"]),
            )
        )
        print(f"NOTSOFAR: {meeting}", flush=True)
    return rows


def validate(rows: list[dict], expected_words: int) -> None:
    words = sum(len(normalize(row["reference_raw"]).split()) for row in rows)
    if words != expected_words or len(
        {(row["dataset_revision"], row["id"]) for row in rows}
    ) != len(rows):
        raise RuntimeError(f"Expected {expected_words} words and unique rows; found {words}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the fixed benchmark corpora")
    parser.add_argument("corpora", nargs="*", choices=["mega", "meetings"], default=["mega"])
    parser.add_argument("--output", type=Path, default=Path("benchmark_data"))
    args = parser.parse_args()
    source = args.output / "source"
    if "mega" in args.corpora:
        rows = prepare_mega(args.output / "mega")
        validate(rows, 32_928)
        write_jsonl(args.output / "mega" / "manifest.jsonl", rows)
    if "meetings" in args.corpora:
        root = args.output / "meetings"
        groups = [
            prepare_dipco(root, source),
            prepare_ami(root, source),
            prepare_notsofar(root),
        ]
        for rows, words in zip(groups, (33_681, 32_928, 33_496), strict=True):
            validate(rows, words)
        write_jsonl(root / "manifest.jsonl", [row for rows in groups for row in rows])
    print("Preparation complete", flush=True)


if __name__ == "__main__":
    main()
