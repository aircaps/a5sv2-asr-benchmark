# Google Chirp 3 long-recording methodology

## Why rollover is required

Google Cloud Speech-to-Text V2 limits a `StreamingRecognize` stream to five minutes and requires
audio to arrive at approximately real-time speed. Google directs longer-lived clients to its
endless-streaming pattern: close before the ceiling, open a new stream, replay the unfinalized
tail, and correct the new stream's local timestamps.

Chirp 3 supports streaming recognition and automatically returns utterance-level result-end
offsets. It does not support word-level timestamps in streaming mode. That rules out exact
word-timestamp ownership at a reconnect.

## Fixed production policy

The runner applies the same model and recognition configuration in every session:

- `chirp_3`, `en-US`, mono LINEAR16 at the stored native rate;
- automatic punctuation enabled, no prompt, adaptation, denoising, or diarization;
- audio requests no larger than 14,400 bytes, paced no faster than real time;
- 285 seconds maximum source audio per session, 15 seconds below the service ceiling;
- five seconds of deterministic audio replay at each boundary.

The 285-second window minimizes context resets while keeping a practical safety margin. The
five-second replay gives the next session complete audio for speech cut by the previous close.
Unlike silence-aware splitting, its boundaries do not inspect or transform the test audio.

## Stitching

Each session returns only finalized results. For every session after the first:

1. Discard finalized results whose utterance end offset is at or before the five-second replay
   boundary.
2. Concatenate the remaining finalized results in response order.
3. Compare at most 80 whitespace-delimited words at the previous suffix and new prefix after NFKC
   normalization, lowercase, apostrophe removal, and replacement of other punctuation with spaces.
4. Remove only the longest exact suffix/prefix match. If there is no exact match, retain both
   strings unchanged.

This is deliberately conservative. Fuzzy alignment could lower WER by choosing among conflicting
hypotheses, but would introduce provider-specific transcript rewriting and difficult-to-audit
thresholds. The implementation never consults the reference and never changes a nonduplicate
provider token. The common scorer then applies exactly the same normalization to every provider's
stitched hypothesis and reference.

Every output row stores session source intervals, replay duration, raw session transcripts,
post-time-filter transcripts, finalized/discarded result counts, deduplicated word counts, total
streamed duration, and the complete streaming configuration. These fields make every boundary
decision independently inspectable.

## Comparability caveats

- A Google meeting is one logical benchmark recording and one scored hypothesis, but not one
  uninterrupted provider session. Model context resets at each rollover.
- Five seconds is retransmitted and may be billed again at every boundary. `streamed_audio_seconds`
  therefore exceeds source duration for long recordings.
- An utterance spanning the five-second ownership cutoff may contain both replay and new speech.
  Exact overlap removal handles verbatim duplication; changed hypotheses are retained rather than
  silently reconciled.
- This protocol optimizes boundary integrity, not the score: its constants are fixed before full
  evaluation and are never selected from reference WER.

Primary references: [Google Cloud quotas and limits](https://cloud.google.com/speech-to-text/quotas),
[Chirp 3 model and feature support](https://cloud.google.com/speech-to-text/v2/docs/models/chirp-3),
and Google's [Python endless-streaming sample](https://github.com/GoogleCloudPlatform/python-docs-samples/blob/main/speech/microphone/transcribe_streaming_infinite.py).
