#!/usr/bin/env bash
set -uo pipefail

provider="${1:?provider is required}"
: "${RUN_BUCKET:?RUN_BUCKET is required}"
: "${RUN_PREFIX:?RUN_PREFIX is required}"

cd /opt/a5sv2
set -a
source /opt/a5sv2-secrets/.env.txt
set +a
export PYTHONPATH=src

case "$provider" in
  nvidia)
    module=a5sv2_eval.providers.nvidia
    filename=nvidia_nemotron_3_asr_streaming_0_6b.jsonl
    concurrency="${TOGETHER_CONCURRENCY:-5}"
    starts_per_minute="${TOGETHER_STARTS_PER_MINUTE:-30}"
    ;;
  mistral)
    module=a5sv2_eval.providers.mistral
    filename=mistral_voxtral_mini_transcribe_realtime_2602.jsonl
    concurrency="${MISTRAL_CONCURRENCY:-5}"
    starts_per_minute="${MISTRAL_STARTS_PER_MINUTE:-60}"
    ;;
  *)
    echo "Unknown provider: $provider" >&2
    exit 2
    ;;
esac

mkdir -p benchmark_data/results benchmark_data/open_source_api/logs \
  benchmark_data/open_source_api/state
log="benchmark_data/open_source_api/logs/${provider}.log"

backup_corpus() {
  local corpus="$1"
  local result_dir="$2"
  aws s3 sync "$result_dir" "s3://${RUN_BUCKET}/${RUN_PREFIX}/results/${corpus}" \
    --exclude '*' --include "${filename%.jsonl}*.jsonl" \
    --sse AES256 --only-show-errors
  aws s3 cp "$log" "s3://${RUN_BUCKET}/${RUN_PREFIX}/logs/${provider}.log" \
    --sse AES256 --only-show-errors
}

for corpus in mega meetings; do
  if [[ "$corpus" == mega ]]; then
    manifest=benchmark_data/mega/manifest.jsonl
    result_dir=benchmark_data/results
  else
    manifest=benchmark_data/meetings/manifest.jsonl
    result_dir=benchmark_data/meetings/results
  fi
  output="${result_dir}/${filename}"
  marker="benchmark_data/open_source_api/state/${provider}.${corpus}.done"
  attempt=0
  while [[ ! -f "$marker" ]]; do
    attempt=$((attempt + 1))
    echo "$(date -u +%FT%TZ) starting ${provider}/${corpus} resume cycle ${attempt}; concurrency=${concurrency}; starts_per_minute=${starts_per_minute}" | tee -a "$log"
    set +e
    .venv/bin/python -u -m "$module" \
      --manifest "$manifest" --output "$output" --trials 1 \
      --concurrency "$concurrency" --starts-per-minute "$starts_per_minute" \
      --retries 5 2>&1 | tee -a "$log"
    status=${PIPESTATUS[0]}
    set -e
    backup_corpus "$corpus" "$result_dir"
    if [[ "$status" -eq 0 ]]; then
      date -u +%FT%TZ > "$marker"
      aws s3 cp "$marker" "s3://${RUN_BUCKET}/${RUN_PREFIX}/state/${provider}.${corpus}.done" \
        --sse AES256 --only-show-errors
      break
    fi
    echo "$(date -u +%FT%TZ) ${provider}/${corpus} exited ${status}; resuming in 60 seconds" | tee -a "$log"
    sleep 60
  done
done
