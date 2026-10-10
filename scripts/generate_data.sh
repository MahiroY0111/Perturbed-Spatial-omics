#!/usr/bin/env bash
# Generate the simulated data of one or more data seeds (no model training):
#   generation settings -> Splatter cell pool (R) -> scenarios (Python), written to results/simulation/data_seed_<N>/.
# Seeds 21-30 are the final evaluation; seeds 11-13 are used for the hyperparameter comparison (scripts/run_grid.sh).
# Each step is skipped when its output exists.
# Usage:  scripts/generate_data.sh <evaluation|selection> <seed> [<seed> ...]     (threads: TH, default 4)
set -uo pipefail
SET=$1; shift
case $SET in
  evaluation) GEN=simulation/generate_evaluation_seeds.py ;;
  selection)  GEN=simulation/generate_selection_seeds.py ;;
  *) echo "first argument must be evaluation or selection"; exit 2 ;;
esac
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
mkdir -p "$REPO/results/logs"
for N in "$@"; do
(
set -euo pipefail
say() { echo "[$(date +%H:%M:%S)] data seed $N: $*"; }
DATA=results/simulation/data_seed_$N
if ! have "$DATA/generated_data_manifest.sha256"; then
  if ! have "$DATA/generation_config_input.json"; then
    say "write generation settings"
    dc py python - "$N" <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, "simulation")
import generate_data_seed as base
n = int(sys.argv[1])
template = base.read_json(Path("simulation/generation_config.json"))
config = base.build_config(template, n)
root = Path(f"results/simulation/data_seed_{n}")
root.mkdir(parents=True, exist_ok=True)
(root / "generation_config_input.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
  fi
  if ! have "$DATA/common/splatter_pool_counts.h5"; then
    say "Splatter cell pool (R)"
    dc r Rscript simulation/create_pool.R "$DATA/generation_config_input.json" "$DATA/common"
  fi
  say "scenarios (Python)"
  dc py python "$GEN" --data-seed "$N" --template-config simulation/generation_config.json \
     --output-root "$DATA" --project-root /workspace --resume > /dev/null
fi
say "done"
) > "$REPO/results/logs/data_seed_$N.log" 2>&1
# (the subshell is not tested with || so that set -e stays in force inside it)
[ $? -eq 0 ] || echo "data seed $N FAILED (see results/logs/data_seed_$N.log)"
done
