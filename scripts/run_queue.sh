#!/usr/bin/env bash
# Run a list of scripts/run_condition.sh jobs (one job per line, arguments only) with K jobs at a time.
# Usage:  scripts/run_queue.sh <list file> <K>          (logs: results/logs/conditions/)
set -uo pipefail
LIST=$1; K=$2
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$REPO/results/logs/conditions"
run_one() {
  local line="$*"; local name; name=$(echo "$line" | tr ' :.' '___')
  bash "$REPO/scripts/run_condition.sh" $line > "$REPO/results/logs/conditions/$name.log" 2>&1 \
    && echo "[$(date +%H:%M:%S)] OK   $line" || echo "[$(date +%H:%M:%S)] FAIL $line"
}
export -f run_one; export REPO
grep -v '^\s*#' "$LIST" | grep -v '^\s*$' | xargs -P "$K" -I{} bash -c 'run_one {}'
echo "[$(date +%H:%M:%S)] queue finished: $LIST"
