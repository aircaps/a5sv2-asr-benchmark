# A5Sv2 Coval-compatible device latency benchmark

This directory publishes the methodology, exact inputs, raw predictions, primitive timings,
scoring code, integrity checks, and complete Mac and iPhone results for A5Sv2. Model weights and
inference implementation are private and are not distributed.

The run follows Coval Benchmarks commit
[`52d72516b5158c693d57eef43bd9044a7cd0a28d`](https://github.com/coval-ai/benchmarks/tree/52d72516b5158c693d57eef43bd9044a7cd0a28d):
the same 947 clips, manifest-provided oracle endpoints, 100 ms realtime pacing, and latency
definitions. No arbitrary post-speech silence is appended.

## Result summary

### Benchmark devices

#### MacBook Air

| Specification | Recorded value |
|---|---|
| Computer | MacBook Air (13-inch, M4, 2025) |
| Model identifier | `Mac16,12` |
| Part number | `MW123LL/A` |
| Chip | Apple M4 |
| CPU | 10 cores: 4 performance + 6 efficiency |
| GPU | 8 cores |
| Neural Engine | 16 cores |
| Memory | 16 GB unified memory |
| Internal storage | 256 GB Apple SSD (`APPLE SSD AP0256Z`) |
| Operating system | macOS 26.1, build `25B78`, arm64 |
| Power during run | AC power; battery reported 100% at checked boundaries |
| Benchmark concurrency | 1 |

The marketing model and year are corroborated by Apple's
[model-identification page](https://support.apple.com/en-gb/102869) and
[2025 M4 MacBook Air specifications](https://support.apple.com/en-us/122209). The Core ML compute
policy allowed CPU and Neural Engine execution; Core ML controls the actual operation placement.

#### iPhone

| Specification | Recorded value |
|---|---|
| Device | iPhone 16 (6.1-inch, 2024), physical device |
| Model identifier | `iPhone17,3` |
| Hardware model | `D47AP` |
| Chip | Apple A18 |
| CPU | 6 cores: 2 performance + 4 efficiency |
| GPU | 5 cores |
| Neural Engine | 16 cores |
| Memory | Not captured by the run harness |
| Internal storage | 128 GB |
| Operating system | iOS 26.6.1 beta, build `23G83`, arm64e |
| Power during run | Wired; charging at 100% initially and full at completion |
| Thermal state | Nominal before and after |
| Low Power Mode | Disabled before and after |
| Benchmark concurrency | 1 |

The model year, chip, CPU, GPU, Neural Engine, and available capacity tiers are corroborated by
Apple's [iPhone 16 technical specifications](https://support.apple.com/en-ca/121029). The tested
unit's 128 GB capacity, identifiers, OS/build, power, and thermal state were read from the physical
device. The run did not capture physical memory, so no unverified RAM value is asserted. The Core
ML compute policy allowed CPU and Neural Engine execution; Core ML controls actual operation
placement.

### Latency summary

All latency values are milliseconds. Lower is better.

| Device | Dataset | Rows | TTFT mean | TTFT p50 | TTFT p95 | Oracle EOS→final mean | EOS→final p50 | EOS→final p95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| MacBook Air M4 | `stt-v1` | 50 | 1,062.99 | 1,148.55 | 1,649.80 | 45.37 | 47.51 | 51.92 |
| MacBook Air M4 | `stt-v3` | 897 | 1,672.31 | 1,649.60 | 2,743.84 | 43.91 | 45.35 | 53.33 |
| MacBook Air M4 | All items | 947 | 1,640.14 | 1,649.06 | 2,736.37 | 43.99 | 45.50 | 53.28 |
| iPhone 16 | `stt-v1` | 50 | 1,074.42 | 1,155.44 | 1,660.00 | 49.66 | 49.18 | 54.36 |
| iPhone 16 | `stt-v3` | 897 | 1,680.78 | 1,656.61 | 2,754.92 | 50.43 | 50.06 | 57.26 |
| iPhone 16 | All items | 947 | 1,648.77 | 1,656.33 | 2,753.96 | 50.39 | 50.01 | 57.23 |

Coverage is 947/947 for each device with zero failures and no outlier removal. `All items` is
item-weighted and is therefore dominated by the 897-item `stt-v3` set. The per-dataset rows are
the primary result.

### Accuracy

Two WER conventions are included because Coval and the parent A5Sv2 benchmark intentionally use
different normalizers and aggregations.

| Device | Dataset | Coval mean item WER | A5Sv2-standard corpus WER |
|---|---|---:|---:|
| MacBook Air M4 | `stt-v1` | 2.7794% | 2.6012% |
| MacBook Air M4 | `stt-v3` | 4.2010% | 4.7103% |
| MacBook Air M4 | Combined | 4.1259% item-weighted | 4.6081% pooled |
| iPhone 16 | `stt-v1` | 2.7794% | 2.6012% |
| iPhone 16 | `stt-v3` | 4.1869% | 4.7005% |
| iPhone 16 | Combined | 4.1126% item-weighted | 4.5987% pooled |

Coval-compatible WER uses `EnglishTextNormalizer` and averages per-item WER percentages. The
A5Sv2 repository standard applies its documented NFKC/punctuation normalization and sums corpus
edit counts. These values answer different questions and must not be interchanged.

## Interpretation and limitations

- The Mac result is one complete logical trial. It was stopped after 12 successful items and resumed with
  unchanged artifacts and controls; the remaining 935 items completed in a second execution
  session. Every raw row retains its execution-session ID.
- The iPhone result is one uninterrupted complete trial. Its phone remained wired and charging,
  which differs from the preregistered unplugged-phone control. The deviation is retained in the
  public metadata rather than silently normalized away.
- Item-bootstrap intervals in `latency-summary.json` quantify uncertainty over the tested clips.
  They do not measure run-to-run, thermal, or device-to-device variance. Each device currently has
  one complete trial; three independent full trials per device are still required for a strong
  configuration-comparison claim.
- Coval's scheduled leaderboard repeatedly samples 10 items per dataset and may interleave hosted
  providers. This full, isolated device run is method-compatible but does not recreate a specific
  historical draw or hosted-provider contention.
- `stt-v1` is LibriSpeech `test-clean`. `stt-v3` contains conversational clips with
  model-generated references. See `DATA_LICENSES.md` before redistribution.
- TTFT starts with the first audio chunk. Model loading, audio download/verification, conversion,
  and one warmup per execution session are outside the timed interval, matching Coval's pre-t0
  convention.
- Oracle EOS→final is finalization latency after an offline endpoint. It is not live endpoint
  detection latency or total user-perceived turn latency.

## Reproduce the published scores

From the repository root:

```bash
python -m venv .latency-venv
source .latency-venv/bin/activate
python -m pip install -r benchmarks/coval-latency/requirements.txt
python benchmarks/coval-latency/scripts/verify_release.py
```

Regenerate either summary directly from the raw rows:

```bash
python benchmarks/coval-latency/scripts/analysis.py \
  benchmarks/coval-latency/results/macbook-air-m4-coreml-cpu-ane/raw-results.jsonl.gz
python benchmarks/coval-latency/scripts/score.py \
  benchmarks/coval-latency/results/macbook-air-m4-coreml-cpu-ane/raw-results.jsonl.gz
python benchmarks/coval-latency/scripts/analysis.py \
  benchmarks/coval-latency/results/iphone-16-coreml-cpu-ane/raw-results.jsonl.gz
python benchmarks/coval-latency/scripts/score.py \
  benchmarks/coval-latency/results/iphone-16-coreml-cpu-ane/raw-results.jsonl.gz
```

The inference run itself requires an implementation of the public backend contract in
`scripts/reference_runner.py`. The released package deliberately cannot reconstruct private
weights or inference code. Artifact hashes allow the publisher to attest whether future runs used
the identical private model bundle and runner.

## Contents

- `PROTOCOL.md` — exact experiment and timing contract.
- `methodology.json` — machine-readable pins and definitions.
- `plans/coval-full-947.json` — exact ordered plan and audio hashes.
- `results/macbook-air-m4-coreml-cpu-ane/` — compressed raw rows, summaries, metadata, checksums.
- `results/iphone-16-coreml-cpu-ane/` — compressed raw rows, summaries, metadata, checksums.
- `scripts/` — deterministic release builder, analysis, scoring, verification, and public runner
  contract.
- `vendor/coval/` — Coval's unmodified oracle-endpoint precomputation script and license.
- `huggingface/` — prepared upload mapping and dataset-card section for the companion dataset.

## Licensing

Code in this directory is Apache-2.0 under the repository license. Upstream data and transcripts
retain their own terms; see `DATA_LICENSES.md` and `THIRD_PARTY_NOTICES.md`.
