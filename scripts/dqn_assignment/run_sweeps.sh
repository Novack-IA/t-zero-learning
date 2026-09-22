#!/usr/bin/env bash
# Drive every sweep configuration of the DQN assignment in parallel.
# Usage (from the repo root):  bash scripts/dqn_assignment/run_sweeps.sh [jobs]
# Configurations whose CSV already exists are skipped, so the script is resumable.
set -u
cd "$(dirname "$0")/../.."
JOBS="${1:-7}"
mkdir -p runs/dqn_assignment/logs
# one thread per process: parallelism comes from running configurations side by side
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

run_one() {
  local line="$1" tag ov
  tag="${line%%|*}"
  ov="${line#*|}"
  if [ -f "runs/dqn_assignment/$tag.csv" ]; then
    echo "skip $tag (already done)"
    return 0
  fi
  .venv/bin/python scripts/dqn_assignment/run_one.py --tag "$tag" --override $ov \
    > "runs/dqn_assignment/logs/$tag.log" 2>&1
  echo "finished $tag (exit $?)"
}
export -f run_one

grep -v '^[[:space:]]*#' scripts/dqn_assignment/sweeps.txt | grep -v '^[[:space:]]*$' | \
  xargs -P "$JOBS" -I LINE bash -c 'run_one "$0"' LINE
