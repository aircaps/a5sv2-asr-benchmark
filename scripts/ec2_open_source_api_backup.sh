#!/usr/bin/env bash
set -euo pipefail

: "${RUN_BUCKET:?RUN_BUCKET is required}"
: "${RUN_PREFIX:?RUN_PREFIX is required}"
cd /opt/a5sv2

state=benchmark_data/open_source_api/state
logs=benchmark_data/open_source_api/logs
mkdir -p "$state" "$logs" benchmark_data/results

sync_outputs() {
  aws s3 sync benchmark_data/results "s3://${RUN_BUCKET}/${RUN_PREFIX}/results/mega" \
    --exclude '*' --include 'nvidia_*.jsonl' --include 'mistral_*.jsonl' \
    --sse AES256 --only-show-errors
  aws s3 sync benchmark_data/meetings/results \
    "s3://${RUN_BUCKET}/${RUN_PREFIX}/results/meetings" \
    --exclude '*' --include 'nvidia_*.jsonl' --include 'mistral_*.jsonl' \
    --sse AES256 --only-show-errors
  aws s3 sync "$logs" "s3://${RUN_BUCKET}/${RUN_PREFIX}/logs" \
    --sse AES256 --only-show-errors
  aws s3 sync "$state" "s3://${RUN_BUCKET}/${RUN_PREFIX}/state" \
    --sse AES256 --only-show-errors
}

while true; do
  .venv/bin/python - <<'PY' > "$state/status.json"
import json
from datetime import datetime, timezone
from pathlib import Path

root = Path("benchmark_data")
systems = {
    "nvidia": "nvidia_nemotron_3_asr_streaming_0_6b.jsonl",
    "mistral": "mistral_voxtral_mini_transcribe_realtime_2602.jsonl",
}
status = {"updated_at": datetime.now(timezone.utc).isoformat(), "runs": {}}
for provider, filename in systems.items():
    for corpus, directory, expected in (
        ("mega", root / "results", 1250),
        ("meetings", root / "meetings" / "results", 35),
    ):
        final = directory / filename
        progress = final.with_suffix(".progress.jsonl")
        path = final if final.exists() else progress
        rows = []
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rows.append(json.loads(line))
        status["runs"][f"{provider}.{corpus}"] = {
            "completed_rows": len(rows),
            "expected_rows": expected,
            "failed_rows": sum(row.get("status") != "ok" for row in rows),
            "finalized": final.exists(),
        }
print(json.dumps(status, indent=2, sort_keys=True))
PY
  sync_outputs
  aws s3 cp "$state/status.json" "s3://${RUN_BUCKET}/${RUN_PREFIX}/status.json" \
    --sse AES256 --only-show-errors

  if [[ -f "$state/nvidia.mega.done" \
        && -f "$state/nvidia.meetings.done" \
        && -f "$state/mistral.mega.done" \
        && -f "$state/mistral.meetings.done" ]]; then
    .venv/bin/python - <<'PY' > "$state/completion.json"
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

files = {
    "nvidia.mega": (Path("benchmark_data/results/nvidia_nemotron_3_asr_streaming_0_6b.jsonl"), 1250),
    "nvidia.meetings": (Path("benchmark_data/meetings/results/nvidia_nemotron_3_asr_streaming_0_6b.jsonl"), 35),
    "mistral.mega": (Path("benchmark_data/results/mistral_voxtral_mini_transcribe_realtime_2602.jsonl"), 1250),
    "mistral.meetings": (Path("benchmark_data/meetings/results/mistral_voxtral_mini_transcribe_realtime_2602.jsonl"), 35),
}
report = {"completed_at": datetime.now(timezone.utc).isoformat(), "results": {}}
for name, (path, expected) in files.items():
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != expected or any(row.get("status") != "ok" for row in rows):
        raise SystemExit(f"validation failed for {name}")
    keys = [(row["dataset_revision"], row["id"], row["source_sha256"]) for row in rows]
    if len(set(keys)) != expected:
        raise SystemExit(f"duplicate keys in {name}")
    report["results"][name] = {
        "rows": len(rows),
        "successful_rows": len(rows),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
print(json.dumps(report, indent=2, sort_keys=True))
PY
    sync_outputs
    aws s3 cp "$state/completion.json" "s3://${RUN_BUCKET}/${RUN_PREFIX}/completion.json" \
      --sse AES256 --only-show-errors
    if [[ -x /opt/a5sv2-release/finalize.sh ]]; then
      /opt/a5sv2-release/finalize.sh
    fi
    sleep 30
    shutdown -h now
    exit 0
  fi
  sleep 60
done
