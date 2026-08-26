import json

from a5sv2_eval.dataset import DATASET_ID, DATASET_REVISION
from a5sv2_eval.score import corpus_counts, normalize, score_files


def test_normalize():
    assert normalize("Don’t, DON'T, dont!") == "dont dont dont"
    assert normalize("one—two / three") == "one two three"
    assert normalize("ＡＢＣ  １２３") == "abc 123"


def test_corpus_counts():
    counts = corpus_counts(
        [{"reference_raw": "one two three", "prediction_raw": "one four three five"}]
    )
    assert counts == {
        "hits": 2,
        "substitutions": 1,
        "deletions": 0,
        "insertions": 1,
        "reference_words": 3,
        "errors": 2,
        "wer_pct": 200 / 3,
    }


def test_corpus_wer_and_trial_mean(tmp_path):
    base = [
        {
            "system_id": "test",
            "system": "Test",
            "dataset_id": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "id": f"{condition}-{index}",
            "source_sha256": f"{condition}-{index}",
            "condition": condition,
            "sample_rate": 16000,
            "num_samples": 16000,
            "duration_seconds": 1.0,
            "reference_raw": "Hello world.",
            "prediction_raw": "hello world",
            "status": "ok",
        }
        for condition in [
            "far_field",
            "far_field_noise",
            "noise",
            "obstructed_noise",
            "recording_noise",
        ]
        for index in range(250)
    ]
    trials = [base, [{**row, "trial": 2, "prediction_raw": "hello"} for row in base]]
    paths = []
    for number, rows in enumerate(trials, 1):
        path = tmp_path / f"trial_{number}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        paths.append(path)

    records = score_files(paths)
    overall = next(row for row in records if row["condition"] == "overall")
    assert overall["trials"] == 2
    assert overall["mean_wer_pct"] == 25
