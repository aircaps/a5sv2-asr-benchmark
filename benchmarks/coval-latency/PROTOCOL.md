# Coval-compatible device latency protocol

## Frozen inputs

- Coval repository commit: `52d72516b5158c693d57eef43bd9044a7cd0a28d`
- `stt-v1` manifest SHA-256:
  `8c35e1d080e5482bb64eb8b9f39fd870ca334f9f5ca6d188680750715b6605e3`
- `stt-v3` manifest SHA-256:
  `286de7c17410237c026662dfb41d42e94eeb835258d0dd2c4ffaab04a55d898b`
- Full 947-item plan SHA-256:
  `439ef68e1fef427e346981dbec82a3da4183aeb560309e82f79bd5f5a886d333`
- Audio: 16 kHz, mono, PCM16; every file is verified against its manifest SHA-256.

The plan contains all 50 `stt-v1` and all 897 `stt-v3` items. Selection and order were fixed
before inference. No item was selected, removed, or replaced based on output.

## Exact oracle endpoint

`speech_end_offset_ms` is copied from Coval's pinned manifests. Coval produced it offline as the
end of the last Silero VAD speech segment using `min_silence_duration_ms=300` and
`speech_pad_ms=100`, clamped to clip duration and rounded to 0.1 ms. The exact upstream script is
vendored unmodified at `vendor/coval/precompute_vad_offsets.py`.

For each clip, remove only Coval's duration-derived tail:

```text
trailing_ms = max(0, duration_sec * 1000 - speech_end_offset_ms)
tail_frames = round(trailing_ms / 1000 * 16000)
effective_pcm = pcm_without_last(tail_frames)
```

No fixed 200 ms, 300 ms, or other post-speech silence is added. End-of-stream is requested
immediately after the final paced audio chunk.

## Streaming and timing

1. Load the private model and perform one untimed warmup per execution session.
2. Parse and validate WAV, trim to the oracle boundary, and convert samples before the timer.
3. Create a fresh logical inference session for each clip.
4. Set `t0` immediately before the first 100 ms chunk is submitted.
5. Submit chunks sequentially. Pace each chunk against the absolute deadline
   `t0 + cumulative_frames / 16000`; do not accumulate relative sleeps.
6. Record first token at the first non-empty transcript snapshot.
7. After the final paced chunk, immediately finalize and record final completion.
8. Run at concurrency one. Retain every success, failure, and outlier.

Primitive metrics:

```text
TTFT = first_nonempty_transcript_time - t0
audio_to_final = final_completion_time - t0
oracle_EOS_to_final = max(0, audio_to_final - speech_end_offset_ms / 1000)
```

Model loading, connection/session preparation, audio download, hash validation, WAV parsing,
sample conversion, and warmup are excluded. The release records primitive seconds in every row;
display conversions to milliseconds happen only in documentation.

## Execution controls

Common controls:

- Compute policy: Core ML, CPU and Neural Engine eligible; actual placement is controlled by Core
  ML and was not introspected per operation.
- Concurrency: one.
- Trial: one complete ordered pass per published device result.
- Warmup: one per process launch.
- Outlier removal: none.

Recorded Mac environment:

- Device: MacBook Air (13-inch, M4, 2025), model identifier `Mac16,12`, part number
  `MW123LL/A`.
- Chip: Apple M4 with a 10-core CPU (4 performance and 6 efficiency cores), 8-core GPU, and
  16-core Neural Engine.
- Memory and storage: 16 GB unified memory and 256 GB Apple SSD (`APPLE SSD AP0256Z`).
- OS: macOS 26.1 build `25B78`, arm64.
- Power: AC, battery 100% at checked boundaries.
- Thermal telemetry: not recorded continuously.

The Mac trial was deliberately stopped after row 12 and resumed. Completed identities were read
from the append-only result file and skipped; none was rerun. This yields one logical trial and two
execution sessions, both retained in the release.

Recorded iPhone environment:

- Device: physical iPhone 16 (6.1-inch, 2024), model identifier `iPhone17,3`, hardware model
  `D47AP`, 128 GB storage. Physical memory was not captured by the run harness.
- Chip: Apple A18 with a 6-core CPU (2 performance and 4 efficiency cores), 5-core GPU, and
  16-core Neural Engine.
- OS: iOS 26.6.1 beta build `23G83`, arm64e.
- Power: wired and charging at 100% initially; wired and full at completion. This differs from the
  preregistered unplugged-phone control.
- Thermal state: nominal before and after; Low Power Mode disabled before and after.

The iPhone trial completed all 947 items in one uninterrupted execution session. A replication
under the preregistered unplugged control remains required before treating it as the final
power-controlled phone result.

## Aggregation and uncertainty

Report `stt-v1` and `stt-v3` separately. Means, p50, and linearly interpolated p95 are calculated
from raw request values with no filtering. The combined group is item-weighted and secondary.

The deterministic 10,000-iteration percentile bootstrap resamples items within each reported
group using seed `20260827`. Its intervals describe item-sampling uncertainty conditional on this
single logical trial. They do not include independent run, thermal, or device variance.

## WER

The release reports both conventions:

1. **Coval-compatible:** `whisper-normalizer==0.1.12` `EnglishTextNormalizer`, normalization
   version 2, `jiwer==4.0.0`, arithmetic mean of item WER percentages.
2. **Parent-repository standard:** Unicode NFKC, lowercase, remove apostrophes, replace other
   punctuation with spaces, collapse whitespace, then sum `jiwer` hits/substitutions/deletions/
   insertions into corpus WER. Pooled and unweighted dataset-macro values are reported.

The two WER values are not directly interchangeable.

## Comparability

The corpus, oracle endpoint, realtime pacing, pre-t0 convention, TTFT, and finalization metric are
compatible with the pinned Coval implementation. Coval's scheduled headline repeatedly samples
10 items per dataset and its hosted execution environment differs. A direct provider ranking must
also match item identities, time window, model revision, concurrency class, and execution cohort.
