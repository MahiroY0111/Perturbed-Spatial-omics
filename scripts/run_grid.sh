#!/usr/bin/env bash
# Hyperparameter comparison of the spatial penalty weight (lambda_sp_gamma in {0.01, 0.1, 1, 5}) and the weight of the
# spot/type offsets (lambda_gamma in {0.01, 0.1, 1}): 12 settings for one data seed and one scenario, all other
# settings as in pipeline/standard_settings.json. The base setting (0.1, 1.0) runs first; the other 11 reuse its
# autoencoder and CellOT, and the modality shift scenario reuses those of unpaired_group (same scRNA-seq), so run the
# unpaired_group chain of a seed before its modality shift chain. SIMO+CellOT does not depend on these weights and is not run.
# Data: scripts/generate_data.sh selection <seed>. Summary: python tables/selection_grid_summary.py
# Usage:  scripts/run_grid.sh <seed> <unpaired_group|unpaired_group_heterogeneous|unpaired_group_modality> [threads=6]
set -euo pipefail
N=$1; SCN=$2; TH=${3:-6}
OUT=${OUT:-results/selection_grid}
SFX="modelseed$N"
DATA=results/simulation/data_seed_$N
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
say() { echo "[$(date +%H:%M:%S)] grid seed $N $SCN: $*"; }
case $SCN in unpaired_group) S=grp ;; unpaired_group_heterogeneous) S=het ;; unpaired_group_modality) S=mod ;; *) echo "unknown scenario $SCN"; exit 2 ;; esac
SCEN=$DATA/$SCN
have "$SCEN/st_ctrl_raw.h5ad" || { say "scenario data missing: $SCEN"; exit 1; }
lab() { echo "$1" | tr '.' 'p'; }
cell_root() { echo "$OUT/seed_$N/$SCN/sp$(lab "$1")_g$(lab "$2")"; }
cell_name() { echo "${S}_sp$(lab "$1")_g$(lab "$2")"; }
run_cell() {  # $1 = lambda_sp_gamma, $2 = lambda_gamma, $3 = models directory to copy the autoencoder and CellOT from (or none), $4 = file prefix there
  local SP=$1 G=$2 FROM=$3 FROMNAME=$4
  local ROOT NAME; ROOT=$(cell_root "$SP" "$G"); NAME=$(cell_name "$SP" "$G")
  local OV=(--override "lambda_sp_gamma=$SP" "lambda_gamma=$G")
  local C=(--scenario-dir "$SCEN" --output-root "$ROOT/model" --name "$NAME" --seed "$N")
  mkdir -p "$REPO/$ROOT/model/models"
  if [ "$FROM" != none ]; then
    for k in ae cellot; do
      [ -e "$REPO/$ROOT/model/models/${NAME}_${k}_G8000_$SFX.pt" ] || cp "$REPO/$FROM/${FROMNAME}_${k}_G8000_$SFX.pt" "$REPO/$ROOT/model/models/${NAME}_${k}_G8000_$SFX.pt"
    done
  fi
  have "$ROOT/model/predictions/${NAME}_proposed_control_raw_G8000_$SFX.h5ad" || { say "$NAME spot model"; dc "$PYSVC" python pipeline/run_condition.py "${C[@]}" --stage stmix "${OV[@]}" > "$REPO/$ROOT/stmix_log.txt" 2>&1; }
  have "$ROOT/model/predictions/${NAME}_proposed_stim_raw_G8000_$SFX.h5ad" || { say "$NAME predictions"; dc "$PYSVC" python pipeline/run_condition.py "${C[@]}" --stage full "${OV[@]}" > "$REPO/$ROOT/full_log.txt" 2>&1; }
  have "$ROOT/model/diagnostics/${NAME}_composition_$SFX.npz" || { say "$NAME diagnostics"; dc "$PYSVC" python pipeline/run_condition.py "${C[@]}" --stage diag "${OV[@]}" > "$REPO/$ROOT/diag_log.txt" 2>&1 || say "$NAME diagnostics failed (continuing)"; }
  have "$ROOT/eval.csv" || { say "$NAME evaluate"; dc py python pipeline/reference_rows.py --scenario-dir "$SCEN" --arm-dir "$ROOT/model" --name "$NAME" --seed "$N" \
      --truth-kind group --tag "grid;$SCN;$SP;$G" --out "$ROOT/eval.csv" > "$REPO/$ROOT/eval_log.txt" 2>&1; }
  say "$NAME done"
}
B_SP=0.1; B_G=1.0
if [ "$SCN" = unpaired_group_modality ]; then
  GB="$(cell_root $B_SP $B_G | sed "s#/$SCN/#/unpaired_group/#")/model/models"; GBN="grp_sp$(lab $B_SP)_g$(lab $B_G)"
  [ -e "$REPO/$GB/${GBN}_cellot_G8000_$SFX.pt" ] || { say "run the unpaired_group chain of seed $N first ($GB is missing)"; exit 1; }
  for f in sc_ctrl_raw.h5ad sc_stim_raw.h5ad; do
    cmp -s "$REPO/$DATA/unpaired_group/$f" "$REPO/$SCEN/$f" || { say "modality $f differs from unpaired_group"; exit 1; }
  done
  run_cell $B_SP $B_G "$GB" "$GBN"
else
  run_cell $B_SP $B_G none none
fi
BM="$(cell_root $B_SP $B_G)/model/models"; BN=$(cell_name $B_SP $B_G)
for SP in 0.01 0.1 1.0 5.0; do
  for G in 0.01 0.1 1.0; do
    [ "$SP" = "$B_SP" ] && [ "$G" = "$B_G" ] && continue
    run_cell "$SP" "$G" "$BM" "$BN"
  done
done
say "all settings done"
