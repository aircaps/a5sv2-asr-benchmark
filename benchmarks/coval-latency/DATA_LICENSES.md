# Dataset licensing and provenance

This release contains references and predictions but no audio files. Audio is retrieved from
Coval's public dataset bucket using the paths and SHA-256 hashes in the pinned plan.

| Dataset | Source | Upstream license disclosure |
|---|---|---|
| `stt-v1` | LibriSpeech `test-clean` / OpenSLR-12 | CC BY 4.0 |
| `stt-v3` | `pipecat-ai/stt-benchmark-data` train split | No data license published by the source, as recorded in Coval's manifest |

The Apache-2.0 license for this repository's code does not relicense upstream audio or transcript
content. The `stt-v3` licensing status should be reviewed before public redistribution. The plan,
references, and raw result bundle are staged for transparency but are not a representation that
the underlying `stt-v3` data carries an Apache-2.0 license.

Dataset limitations documented by Coval also apply: `stt-v1` is an easy read-speech set that may
overlap common training data, while `stt-v3` contains model-generated reference transcripts.
