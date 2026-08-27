#!/usr/bin/env bash
set -euo pipefail

: "${RUN_BUCKET:?RUN_BUCKET is required}"
: "${RUN_PREFIX:?RUN_PREFIX is required}"

release_root=/opt/a5sv2-release
github_root="$release_root/github"
dataset_root="$release_root/dataset"
gpu_run="$release_root/gpu-run"
bundle="$dataset_root/results/open-source-four-corpus-v1"
artifact_root="$release_root/artifacts"
staging_prefix="${RUN_PREFIX}/release-staging"

if [[ -f "$release_root/finalized.done" ]]; then
  exit 0
fi

exec 9>"$release_root/finalize.lock"
flock -n 9 || exit 0

export PYTHONPATH="$github_root/src"
cd /opt/a5sv2
if [[ ! -e "$bundle" ]]; then
  build_parent=$(mktemp -d "$release_root/bundle-build.XXXXXX")
  build_output="$build_parent/open-source-four-corpus-v1"
  .venv/bin/python "$github_root/tools/build_open_source_release.py" \
    --base-release "$dataset_root/results/four-corpus-v1" \
    --api-mega-dir /opt/a5sv2/benchmark_data/results \
    --api-meeting-dir /opt/a5sv2/benchmark_data/meetings/results \
    --gpu-run-dir "$gpu_run" \
    --output "$build_output" \
    --code-revision uncommitted-working-tree
  mv "$build_output" "$bundle"
  rmdir "$build_parent"
fi

.venv/bin/python "$github_root/tools/stage_open_source_release.py" \
  --bundle "$bundle" \
  --github-root "$github_root" \
  --dataset-root "$dataset_root"

cd "$github_root"
git add -A
git diff --cached --check
if git diff --cached --name-only | grep -Eiq 'parakeet|voxtral_local|nemotron_nemo|vast_native'; then
  echo "Excluded private implementation found in GitHub staging" >&2
  exit 1
fi
git diff --cached --binary > "$release_root/github-staged.patch"
git status --short > "$release_root/github-status.txt"

cd "$dataset_root"
git add -A
git diff --cached --check
if git diff --cached --name-only | grep -Eiq 'parakeet|voxtral_local|nemotron_nemo|vast_native'; then
  echo "Excluded private implementation found in dataset staging" >&2
  exit 1
fi
git diff --cached --binary > "$release_root/huggingface-staged.patch"
git status --short > "$release_root/huggingface-status.txt"

mkdir -p "$artifact_root"
tar --exclude=.git -C "$github_root" -czf "$artifact_root/github-worktree.tar.gz" .
tar --exclude=.git -C "$dataset_root" -czf "$artifact_root/huggingface-worktree.tar.gz" .
cp "$release_root/github-staged.patch" "$artifact_root/"
cp "$release_root/huggingface-staged.patch" "$artifact_root/"
cp "$release_root/github-status.txt" "$artifact_root/"
cp "$release_root/huggingface-status.txt" "$artifact_root/"
cd "$artifact_root"
sha256sum github-worktree.tar.gz huggingface-worktree.tar.gz \
  github-staged.patch huggingface-staged.patch > SHA256SUMS

aws s3 sync "$artifact_root" "s3://${RUN_BUCKET}/${staging_prefix}" \
  --sse AES256 --only-show-errors
date -u +%FT%TZ > "$release_root/finalized.done"
aws s3 cp "$release_root/finalized.done" \
  "s3://${RUN_BUCKET}/${staging_prefix}/finalized.done" \
  --sse AES256 --only-show-errors
