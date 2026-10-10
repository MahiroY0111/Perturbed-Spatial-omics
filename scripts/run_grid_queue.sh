#!/usr/bin/env bash
# The whole hyperparameter comparison: data of seeds 11-13, then the unpaired_group chains, then the heterogeneous
# response and modality shift chains (the latter reuse the unpaired_group base setting), K chains at a time.
# Usage:  scripts/run_grid_queue.sh [K=3]          (logs: results/logs/grid/)
set -uo pipefail
K=${1:-3}
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
L=$REPO/results/logs/grid; mkdir -p "$L"
TH=6 bash "$REPO/scripts/generate_data.sh" selection 11 12 13
run_one() {
  local n=$1 s=$2
  bash "$REPO/scripts/run_grid.sh" "$n" "$s" 6 > "$REPO/results/logs/grid/${s}_$n.log" 2>&1 \
    && echo "[$(date +%H:%M:%S)] OK   $n $s" || echo "[$(date +%H:%M:%S)] FAIL $n $s"
}
export -f run_one; export REPO
printf '%s\n' "11 unpaired_group" "12 unpaired_group" "13 unpaired_group" | xargs -P "$K" -L 1 bash -c 'run_one "$0" "$1"'
printf '%s\n' "11 unpaired_group_heterogeneous" "12 unpaired_group_heterogeneous" "13 unpaired_group_heterogeneous" \
              "11 unpaired_group_modality" "12 unpaired_group_modality" "13 unpaired_group_modality" | xargs -P "$K" -L 1 bash -c 'run_one "$0" "$1"'
echo "[$(date +%H:%M:%S)] grid finished"
