# Hugging Face release

Publish references, raw predictions, scores, metadata, and checksums—not redistributed corpus
audio. `python -m a5sv2_eval.prepare mega meetings` reconstructs audio from pinned public sources
and verifies the hashes recorded in each manifest row.

Recommended dataset-repository layout:

```text
results/four-corpus-v1/
  README.md
  metadata.json
  SHA256SUMS
  references/
    mega-asr.jsonl.gz
    meetings.jsonl.gz
  predictions/
    <system_id>/mega-asr.jsonl.gz
    <system_id>/meetings.jsonl.gz
  scores/
    corpus-scores.csv
    four-corpus.csv
```

Build the bundle from the two manifests and saved trial 1 for every corpus and system. Release v1
does not average multiple inference trials:

```bash
python -m a5sv2_eval.release \
  --output hf_upload/results/four-corpus-v1 \
  --code-commit "$(git rev-parse HEAD)" \
  --mega-manifest benchmark_data/mega/manifest.jsonl \
  --meetings-manifest benchmark_data/meetings/manifest.jsonl \
  --mega-results /path/to/api/mega/assemblyai_universal_3_5_pro.jsonl \
    /path/to/api/mega/deepgram_nova_3.jsonl \
    /path/to/api/mega/elevenlabs_scribe_v2_realtime.jsonl \
    /path/to/api/mega/google_chirp_3.jsonl \
    /path/to/api/mega/openai_gpt_live_transcribe.jsonl \
  --meeting-results benchmark_data/meetings/results/assemblyai_universal_3_5_pro.jsonl \
    benchmark_data/meetings/results/deepgram_nova_3.jsonl \
    benchmark_data/meetings/results/elevenlabs_scribe_v2_realtime.jsonl \
    benchmark_data/meetings/results/google_chirp_3.jsonl \
    benchmark_data/meetings/results/openai_gpt_live_transcribe.jsonl \
  --a5sv2-mega /path/to/a5sv2/mega.jsonl \
  --a5sv2-meetings /path/to/a5sv2/meetings.jsonl
```

The builder uses an explicit public-system allowlist and rejects missing systems, failures,
duplicates, or manifest mismatches. It removes local paths, writes deterministic gzip files, and
hashes every published artifact. Review the bundle, verify `sha256sum -c SHA256SUMS`, then upload:
The first release does not include confidence intervals; the planned method is the paired cluster
bootstrap specified in `PROTOCOL.md`.

```bash
hf upload AirCaps/a5sv2-asr-benchmark-dataset hf_upload/results/four-corpus-v1 \
  results/four-corpus-v1 --repo-type dataset
```

Keep the existing dataset-card configuration unchanged. Link the GitHub repository for executable
evaluation code and label `results/four-corpus-v1/` as the canonical four-corpus release.
