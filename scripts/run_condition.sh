#!/usr/bin/env bash
# Run one simulated condition end to end inside the containers:
#   derive the condition data -> SIMO mapping (with and without cell reuse) -> spot model -> CellOT and predictions
#   -> composition diagnostics -> evaluation (results/conditions/<family>/seed_<N>/<name>/eval.csv)
# A stopped run resumes with the same command. Data come from scripts/generate_data.sh (and generate_complex_data.sh).
#
# Usage:
#   scripts/run_condition.sh capture  <seed> <main|complex> <none|orig|beta:A:B|const:P|betaconst:A:B:P> [threads=2]
#       main: modality shift scenario with the ST capture efficiency replaced (none = no capture model,
#       orig = as generated, beta:A:B = gene-wise Beta(A, B), const:P = the same capture probability P for all genes).
#       complex: complex spatial structure scenario (orig) or the same scenario without capture efficiency (none).
#   scripts/run_condition.sh hetero   <seed> [threads=2]                  heterogeneous response scenario
#   scripts/run_condition.sh refctrl  <seed> <factor> [threads=2]
#       control reference replaced by cells not used in the ST (1,000 per group); <factor> scales the responding
#       group in both references (1.0 = unchanged).
#   scripts/run_condition.sh refshift <seed> <factor> [threads=2]
#       both references replaced by cells not used in the ST (500 per group per condition).
#   scripts/run_condition.sh refsub   <seed> <factor> [threads=2]
#       control reference of the same size as refctrl, drawn from the cells that make up the ST.
#   scripts/run_condition.sh realistic <seed> <factor> <capture spec> [threads=2]
#       refctrl reference combined with an ST capture model (e.g. betaconst:8:2:0.03).
# Environment: PYSVC=py (CPU, default) or pygpu (GPU); OUT=results/conditions (output root).
set -euo pipefail
KIND=$1; N=$2
OUT=${OUT:-results/conditions}
SFX="modelseed$N"
REUSE=base

case $KIND in
  capture)
    BASE=$3; SPEC=$4; TH=${5:-2}
    NAME="${BASE}_cap_$(echo "$SPEC" | tr ':.' '__')"
    ROOT=$OUT/capture/seed_$N/$NAME
    if [ "$BASE" = main ]; then
      SRC=results/simulation/data_seed_$N/unpaired_group_modality; TRUTH=group; BASENAME=main_cap_none
    else
      SRC=results/complex_scenario/seed_$N/unpaired_group_mapping_ambiguity_spatial_context; TRUTH=complex; BASENAME=complex_cap_orig
    fi
    if [ "$SPEC" = orig ]; then SCEN=$SRC; else SCEN=$ROOT/scenario; fi
    [ "$NAME" = "$BASENAME" ] && REUSE=none
    TAG="$BASE;$SPEC" ;;
  hetero)
    TH=${3:-2}; NAME=hetero_main; ROOT=$OUT/hetero/seed_$N/$NAME
    SCEN=results/simulation/data_seed_$N/unpaired_group_heterogeneous; TRUTH=group; TAG="hetero;orig"; REUSE=none ;;
  refshift|refctrl|refsub)
    F=$3; TH=${4:-2}
    case $KIND in refshift) P=ref ;; refctrl) P=refctrl ;; refsub) P=refsub ;; esac
    NAME="${P}_f$(echo "$F" | tr '.' 'p')"; ROOT=$OUT/$KIND/seed_$N/$NAME
    SCEN=$ROOT/scenario; TRUTH=group; TAG="$KIND;$F"; REUSE=none ;;
  realistic)
    F=$3; SPEC=$4; TH=${5:-2}
    RNAME="refctrl_f$(echo "$F" | tr '.' 'p')"
    NAME="real_$(echo "${F}_${SPEC}" | tr ':.' '__')"; ROOT=$OUT/realistic/seed_$N/$NAME
    SRC=$OUT/refctrl/seed_$N/$RNAME/scenario
    SCEN=$ROOT/scenario; TRUTH=group; TAG="realistic;$F;$SPEC"; REUSE=refctrl ;;
  *) echo "unknown condition: $KIND"; exit 2 ;;
esac
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
say() { echo "[$(date +%H:%M:%S)] $KIND seed $N $NAME: $*"; }
mkdir -p "$REPO/$ROOT/simo" "$REPO/$ROOT/model/models"

# 1. condition data
if [ "$KIND" = capture ] && [ "$SPEC" != orig ]; then
  have "$SCEN/variant_config.json" || { say "derive data"; dc py python pipeline/make_conditions.py capture --source "$SRC" --out "$SCEN" --capture "$SPEC" --seed "$N" > /dev/null; }
elif [ "$KIND" = realistic ]; then
  have "$SRC/variant_config.json" || { say "run 'refctrl $N $F' first: $SRC is missing"; exit 1; }
  have "$SCEN/variant_config.json" || { say "derive data"; dc py python pipeline/make_conditions.py capture --source "$SRC" --out "$SCEN" --capture "$SPEC" --seed "$N" > /dev/null; }
elif [ "$KIND" = refsub ]; then
  have "$SCEN/variant_config.json" || { say "derive data"; dc py python pipeline/make_conditions.py refctrl --from-st-cells --data-root "results/simulation/data_seed_$N" --out "$SCEN" --target-factor "$F" > /dev/null; }
elif [ "$KIND" = refshift ] || [ "$KIND" = refctrl ]; then
  have "$SCEN/variant_config.json" || { say "derive data"; dc py python pipeline/make_conditions.py "$KIND" --data-root "results/simulation/data_seed_$N" --out "$SCEN" --target-factor "$F" > /dev/null; }
fi

# 2. SIMO mapping (CPU, in the background)
SIMO_PID=""
if ! have "$ROOT/simo/map.json"; then
  say "SIMO mapping (background)"
  ( dc simo python pipeline/simo_mapping.py --scenario-dir "$SCEN" --output-prefix "$ROOT/simo/map" --model-seed "$N" > "$REPO/$ROOT/simo/log.txt" 2>&1 ) &
  SIMO_PID=$!
fi

# 3. autoencoder and CellOT reused from the base condition of the same seed (same scRNA-seq reference)
if [ "$REUSE" != none ]; then
  if [ "$REUSE" = base ]; then BM=$OUT/capture/seed_$N/$BASENAME/model/models; BS=$BASENAME;
  else BM=$OUT/refctrl/seed_$N/$RNAME/model/models; BS=$RNAME; fi
  for k in ae cellot; do
    [ -e "$REPO/$BM/${BS}_${k}_G8000_$SFX.pt" ] || { say "run the base condition first: $BM/${BS}_${k}_G8000_$SFX.pt is missing"; exit 1; }
    [ -e "$REPO/$ROOT/model/models/${NAME}_${k}_G8000_$SFX.pt" ] || cp "$REPO/$BM/${BS}_${k}_G8000_$SFX.pt" "$REPO/$ROOT/model/models/${NAME}_${k}_G8000_$SFX.pt"
  done
fi

# 4. training, predictions, diagnostics, evaluation
C=(--scenario-dir "$SCEN" --output-root "$ROOT/model" --name "$NAME" --seed "$N")
have "$ROOT/model/predictions/${NAME}_proposed_control_raw_G8000_$SFX.h5ad" || { say "spot model ($PYSVC)"; dc "$PYSVC" python pipeline/run_condition.py "${C[@]}" --stage stmix > "$REPO/$ROOT/stmix_log.txt" 2>&1; }
if [ -n "$SIMO_PID" ]; then say "wait for SIMO"; wait "$SIMO_PID" || { say "SIMO mapping FAILED"; exit 1; }; fi
have "$ROOT/model/predictions/${NAME}_proposed_stim_raw_G8000_$SFX.h5ad" || { say "CellOT and predictions ($PYSVC)"; dc "$PYSVC" python pipeline/run_condition.py "${C[@]}" --stage full \
    --mappings "norepeat=$ROOT/simo/map_norepeat.h5ad" "repeat=$ROOT/simo/map_repeat.h5ad" > "$REPO/$ROOT/full_log.txt" 2>&1; }
have "$ROOT/model/diagnostics/${NAME}_composition_$SFX.npz" || { say "diagnostics"; dc "$PYSVC" python pipeline/run_condition.py "${C[@]}" --stage diag > /dev/null 2>&1 || say "diagnostics failed (continuing)"; }
say "evaluate"
dc py python pipeline/reference_rows.py --scenario-dir "$SCEN" --arm-dir "$ROOT/model" --name "$NAME" --seed "$N" --truth-kind "$TRUTH" --tag "$TAG" --out "$ROOT/eval.csv" > "$REPO/$ROOT/eval_log.txt" 2>&1
say "done"
