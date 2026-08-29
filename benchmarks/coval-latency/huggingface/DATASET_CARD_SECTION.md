## A5Sv2 Coval-compatible device latency v1

This bundle contains complete 947-item A5Sv2 latency trials on a MacBook Air (13-inch, M4, 2025)
and a physical iPhone 16 (6.1-inch, A18, 2024). The Mac is model identifier `Mac16,12`, with a
10-core CPU, 8-core GPU, 16-core Neural Engine, 16 GB unified memory, and 256 GB SSD. The iPhone is
model identifier `iPhone17,3`, hardware model `D47AP`, with a 6-core CPU, 5-core GPU, 16-core
Neural Engine, and 128 GB storage; physical memory was not captured by the run harness. Both use
Coval's pinned `stt-v1` and `stt-v3` manifests, oracle endpoints, 100 ms realtime pacing, and
metric definitions. The bundle includes raw predictions and primitive timings, exact input
identities and hashes, latency and WER summaries, run metadata, scoring code references, and
integrity checks. It does not include model weights, inference code, or audio files.

Mac headline item-weighted results are TTFT mean/p50/p95 of 1640.14/1649.06/2736.37 ms and oracle
EOS-to-final mean/p50/p95 of 43.99/45.50/53.28 ms. iPhone headline item-weighted results are TTFT
mean/p50/p95 of 1648.77/1656.33/2753.96 ms and oracle EOS-to-final mean/p50/p95 of
50.39/50.01/57.23 ms. Per-dataset values are primary and are recorded in
`results/coval-latency-v1/README.md`.

Each trial completed 947/947 items with zero failures. The Mac trial was paused after 12 rows and
resumed with unchanged artifacts and controls; session IDs are retained. The iPhone trial was
uninterrupted, remained wired and charging/full, and therefore differs from the preregistered
unplugged-phone control. Item-bootstrap intervals do not represent independent run-to-run or
device-to-device uncertainty.

Licensing is mixed: `stt-v1` is derived from LibriSpeech `test-clean` under CC BY 4.0; Coval's
`stt-v3` manifest records that no source-data license was published by `pipecat-ai`. Review the
included `DATA_LICENSES.md` before redistribution.
