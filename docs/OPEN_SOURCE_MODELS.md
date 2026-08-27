# Open-source model evaluation

This comparison evaluates four open-weight ASR models against A5Sv2 on the same fixed 1,285-item,
four-corpus benchmark. A result is publishable only when saved trial 1 has all 1,285 successful
rows, the pinned references, and the frozen model, decoder, and streaming configuration below.

## Frozen systems

| System | Implementation | Streaming method | Input schedule |
|---|---|---|---|
| NVIDIA Nemotron 3 ASR Streaming 0.6B | NVIDIA model `nvidia/nemotron-3-asr-streaming-0.6b` exposed through a realtime transcription endpoint | Native cache-aware streaming | 560 ms PCM chunks, real-time paced |
| Mistral Voxtral Mini Transcribe Realtime 2602 | Mistral API model `voxtral-mini-transcribe-realtime-2602` | Native causal streaming with the model's 480 ms transcription delay | 560 ms PCM chunks, real-time paced |
| Kyutai STT 2.6B English | Official `moshi==0.2.11` PyTorch inference and `kyutai/stt-2.6b-en` revision `a07aec56d22be5589cd0bc8709c75b6cf3e3039d` | Native decoder-only streaming through Mimi frames; temperature 0; official prefix and delayed-output flush | 80 ms model frames, computationally unaware |
| OpenAI Whisper Large V3 | UFAL Whisper-Streaming revision `6da90b44b7e50d79695e68166d2a2c7609c75abb`, `faster-whisper` backend, converted checkpoint revision `edaa852ec7e145841d8ffdb056a99866b5f0a478` | Published LocalAgreement-2 policy, self-adaptive segment trimming, beam 5, 15 s buffer threshold | 560 ms chunks, computationally unaware |

Primary references:

- NVIDIA's [Nemotron model page](https://www.together.ai/models/nemotron-3-asr-streaming-0-6b).
- Mistral's [realtime transcription guide](https://docs.mistral.ai/studio/audio/speech_to_text/realtime_transcription)
  and [Voxtral Realtime model page](https://docs.mistral.ai/models/voxtral-mini-transcribe-realtime-26-02).
- Kyutai's [model card](https://huggingface.co/kyutai/stt-2.6b-en) and
  [official delayed-streams inference code](https://github.com/kyutai-labs/delayed-streams-modeling).
- The peer-reviewed [Whisper-Streaming paper](https://aclanthology.org/2023.ijcnlp-demo.3/)
  and [UFAL implementation](https://github.com/ufal/whisper_streaming).

The NVIDIA run used Together AI's hosted realtime endpoint and `together[realtime]==2.31.0`; the
request model identifier and realtime wire protocol are preserved in the runner and result
metadata so the run can be replicated. This serving detail does not change the displayed model
identity. The account's observed maximum of five concurrent sessions was used. The Voxtral run
used `mistralai[realtime]==2.9.4` and the account's observed concurrency limit.

## Pacing and fairness

Both hosted APIs receive audio against a monotonic clock. Sending audio faster could alter
provider buffering, backpressure, or server scheduling, so API speedup comes only from independent
concurrent sessions. Each request records the 560 ms schedule and effective concurrency.

Kyutai and Whisper use computationally unaware streaming: the exact ordered chunk sequence is
presented without sleeping between chunks. Removing wall-clock sleeps does not change the audio,
chunk boundaries, state transitions, or decoding decisions. This is the distinction exposed by
UFAL's `--comp_unaware` mode and Kyutai's official file-client RTF control.

Whisper Large V3 is an offline checkpoint. Its result therefore measures the checkpoint together
with the cited Whisper-Streaming conversion, a known limitation relative to native streaming
models. No runner uses a prompt, reference-aware stitching, denoising, beamforming, silence
removal, or post-ASR correction.

## API reproduction

Install the pinned API dependencies, provide credentials through environment variables, and run a
small paid smoke test before removing `--limit 2`. Never commit credential files.

```bash
python -m pip install -e '.[api]'

python -m a5sv2_eval.providers.nvidia \
  --manifest benchmark_data/mega/manifest.jsonl \
  --output benchmark_data/results/nvidia_nemotron_3_asr_streaming_0_6b.jsonl \
  --trials 1 --limit 2 --concurrency 1

python -m a5sv2_eval.providers.mistral \
  --manifest benchmark_data/mega/manifest.jsonl \
  --output benchmark_data/results/mistral_voxtral_mini_transcribe_realtime_2602.jsonl \
  --trials 1 --limit 2 --concurrency 1
```

For production, remove `--limit 2`, set `--concurrency` to the current account maximum, and repeat
both commands with `benchmark_data/meetings/manifest.jsonl`. The runners are append-only and
resume completed rows safely.

## Local GPU reproduction

Kyutai uses the official checkpoint and inference package:

```bash
python -m pip install -e '.[kyutai]'
python tools/run_local_shards.py kyutai --gpus 0,1,2,3,4,5,6,7 \
  --manifest benchmark_data/mega/manifest.jsonl \
  --output benchmark_data/results/kyutai_stt_2_6b_en.jsonl
```

Whisper uses the pinned community implementation and model conversion:

```bash
git clone https://github.com/ufal/whisper_streaming.git external/whisper_streaming
git -C external/whisper_streaming checkout 6da90b44b7e50d79695e68166d2a2c7609c75abb
python -m pip install -e '.[whisper]'
huggingface-cli download Systran/faster-whisper-large-v3 \
  --revision edaa852ec7e145841d8ffdb056a99866b5f0a478 \
  --local-dir models/faster-whisper-large-v3
export WHISPER_STREAMING_ROOT=$PWD/external/whisper_streaming
export WHISPER_LARGE_V3_MODEL_DIR=$PWD/models/faster-whisper-large-v3
python tools/run_local_shards.py whisper --gpus 0,1,2,3,4,5,6,7 \
  --manifest benchmark_data/meetings/manifest.jsonl \
  --output benchmark_data/meetings/results/whisper_large_v3_ufal_streaming.jsonl
```

The production local run used one Vast.ai machine with eight NVIDIA GeForce RTX 5090 GPUs and
eight deterministic duration-balanced shards per model/corpus wave. The sharder assigns longest
recordings first to the currently lightest shard and merges rows back into manifest order. Every
published Kyutai and Whisper result contains complete 1,250-row Mega-ASR and 35-row meeting output
with `status=ok`.

## Validation and release bundle

`tools/build_open_source_release.py` accepts the existing public four-corpus release, the two
completed API result directories, and the validated local GPU run directory. It rejects failures,
duplicate IDs, missing rows, non-trial-1 data, reference/hash mismatches, or any unapproved model.
It independently recomputes edit counts and WER, then writes only A5Sv2 and the four frozen systems
to `results/open-source-four-corpus-v1/` with deterministic gzip files and SHA-256 checksums.

```bash
python tools/build_open_source_release.py \
  --base-release ../a5sv2-asr-benchmark-dataset/results/four-corpus-v1 \
  --api-mega-dir benchmark_data/results \
  --api-meeting-dir benchmark_data/meetings/results \
  --gpu-run-dir /path/to/validated-gpu-run \
  --output ../a5sv2-asr-benchmark-dataset/results/open-source-four-corpus-v1
```

Prediction rows retain the dataset revision actually evaluated. If a historical Mega-ASR
materialization name is encountered, packaging accepts it only after IDs, references, conditions,
sample counts, durations, source hashes, and PCM hashes all match the current release; the mapping
is recorded in `metadata.json`.
