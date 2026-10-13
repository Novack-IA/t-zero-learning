#!/usr/bin/env bash
# Run a batch of configurations in parallel. Each line of the file given as
# $1 is "tag|config|overrides"; tags whose CSV already exists are skipped.
# Usage:  bash scripts/ppo_assignment/run_batch.sh batches/<file>.txt [jobs]
set -u
cd "$(dirname "$0")/../.."
BATCH="$1"
JOBS="${2:-7}"
mkdir -p runs/ppo_assignment/logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

run_one() {
  local line="$1" tag config ov rest
  tag="${line%%|*}"; rest="${line#*|}"
  config="${rest%%|*}"; ov="${rest#*|}"
  if [ -f "runs/ppo_assignment/$tag.csv" ]; then
    echo "skip $tag"; return 0
  fi
  .venv/bin/python scripts/ppo_assignment/run_one.py --config "$config" --tag "$tag" --override $ov \
    > "runs/ppo_assignment/logs/$tag.log" 2>&1
  echo "finished $tag (exit $?)"
}
export -f run_one

grep -v '^[[:space:]]*#' "$BATCH" | grep -v '^[[:space:]]*$' | \
  xargs -P "$JOBS" -I LINE bash -c 'run_one "$0"' LINE
