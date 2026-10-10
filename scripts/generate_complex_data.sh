#!/usr/bin/env bash
# Generate the data of the complex spatial structure scenario (no model training). It is derived from the modality
# shift scenario of the same data seed, so run scripts/generate_data.sh evaluation <seed> first.
# Usage:  scripts/generate_complex_data.sh <seed> [<seed> ...]
set -uo pipefail
TH=${TH:-2}
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
for N in "$@"; do
  ROOT=results/complex_scenario/seed_$N
  if have "$ROOT/generated_data_manifest.sha256"; then echo "[$(date +%H:%M:%S)] complex data seed $N exists"; continue; fi
  echo "[$(date +%H:%M:%S)] complex data seed $N"
  dc py python simulation/complex_scenario/generate.py --project-root /workspace --seed "$N" --output-root "$ROOT" > /dev/null 2>&1 \
     || echo "[$(date +%H:%M:%S)] complex data seed $N FAILED"
done
