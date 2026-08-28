## A5Sv2 Coval-compatible device latency v1

This bundle contains one complete 947-item A5Sv2 latency trial on a MacBook Air (13-inch, M4,
2025), model identifier `Mac16,12`, with a 10-core CPU, 8-core GPU, 16-core Neural Engine, 16 GB
unified memory, and 256 GB SSD. It uses Coval's pinned `stt-v1` and `stt-v3` manifests, oracle
endpoints, 100 ms realtime pacing, and metric definitions. It includes raw predictions and
primitive timings, exact input identities and hashes, latency and WER summaries, run metadata,
scoring code references, and integrity checks. It does not include model weights, inference code,
or audio files.

Headline item-weighted results are TTFT mean/p50/p95 of 1640.14/1649.06/2736.37 ms and oracle
EOS-to-final mean/p50/p95 of 43.99/45.50/53.28 ms. Per-dataset values are primary and are recorded
in `results/coval-latency-v1/README.md`.

The trial completed 947/947 items with zero failures. It was paused after 12 rows and resumed with
unchanged artifacts and controls; session IDs are retained. Item-bootstrap intervals do not
represent independent run-to-run or device-to-device uncertainty.

Licensing is mixed: `stt-v1` is derived from LibriSpeech `test-clean` under CC BY 4.0; Coval's
`stt-v3` manifest records that no source-data license was published by `pipecat-ai`. Review the
included `DATA_LICENSES.md` before redistribution.
