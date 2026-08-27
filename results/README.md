# Results

- `four-corpus.csv`: headline corpus, macro, and pooled WER.
- `corpus-scores.csv`: exact edit counts and corpus WER used for the headline table.
- `mega-asr-conditions.csv`: Mega-ASR acoustic-condition breakdown.
- `open-source-four-corpus.csv`: separate A5Sv2/open-source headline table. Every listed system
  has complete four-corpus coverage; partial and smoke-test runs are rejected.
- `open-source-corpus-scores.csv`: exact edit counts behind the open-source headline rows.

All values come from saved transcript trial 1 for every system and corpus. No CSV contains a
multi-trial mean. Scores use `src/a5sv2_eval/score.py`; regenerate the SVGs with
`python tools/render_charts.py`. Confidence intervals are not reported.
