# Evaluation protocol

## Fixed data

Selections were frozen from references and metadata only; no model output was used.

- **Mega-ASR:** 1,250 utterances sampled from revision
  `a8a35d3319737190d6fd3d39157b258eaab35980` of the training splits of
  [Voices-in-the-Wild-2M](https://huggingface.co/datasets/zhifeixie/Voices-in-the-Wild-2M),
  introduced by [Mega-ASR](https://xzf-thu.github.io/Mega-ASR/)
  ([paper](https://arxiv.org/abs/2605.19833)). The fixed selection contains 250 English
  utterances from each of `far_field`, `far_field_noise`, `noise`, `obstructed_noise`, and
  `recording_noise`. Selection seed `20260823`, shuffle buffer `512`, source-shard IDs, source-row
  identities, and audio hashes are retained in the code and release artifacts.
- **AMI:** `ES2004d ES2014a IS1009a IS1009b TS3003b TS3003c TS3007b` from the official
  scenario-only unseen-evaluation pool. Among subsets covering all five meeting series and all
  four scenario phases, this is the lexicographically first smallest subset minimizing distance
  from 32,928 normalized words. Annotations are version 1.6.2.
- **DiPCo:** eval `S01 S03 S06 S07 S08`, plus `S02`, the first official dev session. The entire
  eval split has 28,813 normalized words; the fixed addition brings it to 33,681. DiPCo has no
  training split. Transcripts and four WAVs use `huckiyang/DiPCo` revision
  `e2b29d3d0d88692c744feb15e290f7316b68014e`; two missing WAVs use
  `vidalfernando/dipco_eval` revision `9eaeeb8264fd655401ae77531784e4aad7fe6611`.
- **NOTSOFAR:** 22 meetings from eval-small `240629.1_eval_small_with_GT`, revision
  `ba8fd0f034ce185fe4d24f47e53b4b8194795f07`:

  ```text
  MTG_32102 MTG_32007 MTG_32068 MTG_32022 MTG_32048 MTG_32069 MTG_32080 MTG_32107
  MTG_32072 MTG_32088 MTG_32178 MTG_32108 MTG_32105 MTG_32082 MTG_32003 MTG_32071
  MTG_32087 MTG_32000 MTG_32106 MTG_32026 MTG_32322 MTG_32004
  ```

  Candidate meetings contain `sc_meetup_0/ch0.wav`. Sort candidates by
  `SHA256("notsofar-production-v1:" + meeting_id)` and take the prefix whose cumulative
  normalized word count is closest to 32,928. The selection spans eight rooms and is disjoint
  from NOTSOFAR training and development speakers and rooms.

## Audio and references

Each meeting is one complete, untrimmed 16 kHz mono PCM16 recording. There is no segmentation,
padding, denoising, beamforming, silence removal, or prompting.

- AMI uses the conventional single distant-microphone channel `Array1-01`, unchanged.
- DiPCo uses `U01.CH1`, the nearest array device on average. A fixed +5.1 dB gain is applied and
  preparation aborts if any sample clips.
- NOTSOFAR uses the source dataset's single-channel conference device
  `sc_meetup_0/ch0.wav`, unchanged. No close-talk audio is used.

DiPCo's `[noise]`, `[laugh]`, and `[unintelligible]` markers and NOTSOFAR XML tags are excluded
from lexical references; text enclosed by tags is retained. AMI uses only lexical `<w>` elements.
DiPCo and NOTSOFAR turns are ordered by annotated start time. AMI words are ordered by start time,
with speaker ID as the deterministic tie-break. `prepare.py` records SHA-256 hashes for source
audio, prepared PCM, and transcripts.

This suite measures speaker-agnostic plain WER, not diarization. Overlap is serialized into one
reference; NOTSOFAR results are therefore not official tcpWER/tcORC-WER.

## Streaming

Audio is sent in source order, no faster than real time, against a monotonic clock. Each utterance
or full meeting is a fresh logical run. Concurrent runs change wall time only. Final transcripts
alone are scored; transient failures retry the complete logical run. A publishable run contains
no failed rows. Requests, attempts, timing, concurrency, provider identifiers, and audit metadata
are retained in result rows.

| System | Request configuration | Account limit used |
|---|---|---:|
| Deepgram Nova-3 | `nova-3`, `en-US`, smart formatting, 100 ms PCM | 150 |
| AssemblyAI Universal-3.5 Pro | balanced, 100 ms PCM | 100; 100 starts/min |
| Google Chirp 3 | `chirp_3`, `en-US`, punctuation, at most 14.4 kB PCM | 42 weighted units |
| OpenAI GPT Live Transcribe | `gpt-live-transcribe`, English, `xhigh`, manual commit, 100 ms PCM | 8 |
| ElevenLabs Scribe v2 Realtime | English, manual commit, 100 ms PCM | 40 |

Google permits streaming sessions up to five minutes. One logical meeting therefore uses
285-second sessions with five seconds of source audio replayed at each reconnect. In every
noninitial session, finalized results ending at or before five seconds are discarded. The rest is
joined after removing only the longest exact normalized suffix/prefix token match, capped at 80
words. There is no fuzzy or reference-aware edit. Raw per-session text, offsets, discarded result
counts, and removed-token counts are retained. See [`docs/CHIRP_3.md`](docs/CHIRP_3.md).

GPT Live Transcribe accepts 24 kHz mono PCM16. Native PCM is converted deterministically with
`scipy.signal.resample_poly`, rounded once to signed 16-bit PCM, and hashed. The session uses
English, `xhigh` delay, no VAD, prompt, or noise reduction, and one manual commit after the final
paced audio byte. Other systems receive native PCM.

Provider references: [Deepgram](https://developers.deepgram.com/reference/speech-to-text/listen-streaming),
[AssemblyAI](https://www.assemblyai.com/docs/streaming/api-spec/streaming-websocket),
[Google](https://cloud.google.com/speech-to-text/quotas),
[OpenAI](https://developers.openai.com/api/docs/models/gpt-live-transcribe), and
[ElevenLabs](https://elevenlabs.io/docs/api-reference/speech-to-text/v-1-speech-to-text-realtime).

## Scoring

Reference and hypothesis receive the same transform:

1. Unicode NFKC and lowercase.
2. Remove straight, curly, and backtick apostrophes.
3. Replace every other non-alphanumeric, non-whitespace character with a space.
4. Collapse whitespace and split on spaces.

No number, abbreviation, spelling, or filler rewriting is performed. Corpus WER is
`(substitutions + deletions + insertions) / (hits + substitutions + deletions)`. It is never an
average of utterance WERs. Release v1 reports saved trial 1 for every system and corpus; there is
no multi-trial averaging. The primary four-corpus aggregate is the arithmetic mean of the four
trial-1 corpus WERs; pooled WER is total trial-1 errors divided by total reference words.

Release v1 reports point estimates only; confidence intervals are not reported.

## Mega-ASR disclosure

Mega-ASR is not speaker-disjoint held-out data and must not be used for training when reporting
this benchmark. We sampled from **Mega-ASR-Train**, rather than the standard Mega-ASR test set,
because in our experiments the standard test set was not acoustically challenging enough to
clearly discriminate among robust ASR systems. This is a derived evaluation set; it should not be
mixed into training when reporting results on it.

A5Sv2 was not trained or fine-tuned on the 1,250 selected recordings or transcripts, and they were
not used for checkpoint selection, hyperparameter tuning, prompt tuning, or any other
model-selection decision. Equivalent information is unavailable for third-party systems. This
exact-item statement does not establish distribution-level independence from the public source
corpus or its data-generation process.
