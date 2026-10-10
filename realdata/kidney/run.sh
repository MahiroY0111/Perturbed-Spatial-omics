#!/usr/bin/env bash
# Mouse kidney ischemia-reperfusion injury: build the inputs, run the method (standard settings with clustering seeds
# 1 and 2, and lambda_mix_gene = 1.0), compare predicted and observed region-level changes in the 12 h section,
# attribute the predicted change to cell types and summarize (results/kidney_iri/). Data: realdata/kidney/fetch.sh.
# Usage:  realdata/kidney/run.sh [threads=6]          Environment: PYSVC=py (CPU, default) or pygpu
set -uo pipefail
TH=${1:-6}
source "$(dirname "${BASH_SOURCE[0]}")/../../scripts/common.sh"
DATA=data/kidney_iri; OUT=results/kidney_iri; S=scenario_12h; NAME=kidney
say() { echo "[$(date +%H:%M:%S)] kidney: $*"; }
have "$OUT/$S/scenario_config.json" || { say "build inputs"; dc py python realdata/kidney/make_scenario.py --root $DATA --stim 12hours --out $OUT/$S; }
if [ ! -e "$REPO/$OUT/$S/simo/map.json" ]; then
  mkdir -p "$REPO/$OUT/$S/simo"; say "SIMO mapping"
  dc simo python pipeline/simo_mapping.py --scenario-dir $OUT/$S --output-prefix $OUT/$S/simo/map --model-seed 1 --skip-qc > "$REPO/$OUT/$S/simo/log.txt" 2>&1 || say "SIMO FAILED"
fi
run_model() {  # $1 = tag (model or model_mix1), $2 = seed
  local TAG=$1 N=$2 OV=() MAPS=()
  [ $TAG = model_mix1 ] && OV=(--override lambda_mix_gene=1.0)
  [ $TAG = model ] && MAPS=(--mappings "norepeat=$OUT/$S/simo/map_norepeat.h5ad" "repeat=$OUT/$S/simo/map_repeat.h5ad")
  local C=(--scenario-dir $OUT/$S --output-root $OUT/$S/$TAG --name $NAME --seed $N --auto-dims --skip-qc "${OV[@]}")
  mkdir -p "$REPO/$OUT/$S/$TAG"
  local P="$REPO/$OUT/$S/$TAG/predictions/${NAME}_proposed"
  [ -e "${P}_control_raw_G8000_modelseed$N.h5ad" ] || { say "$TAG seed $N: spot model"; dc "$PYSVC" python pipeline/run_condition.py "${C[@]}" --stage stmix > "$REPO/$OUT/$S/$TAG/stmix_seed$N.log" 2>&1 || say "FAILED"; }
  [ -e "${P}_stim_raw_G8000_modelseed$N.h5ad" ] || { say "$TAG seed $N: CellOT and predictions"; dc "$PYSVC" python pipeline/run_condition.py "${C[@]}" --stage full "${MAPS[@]}" > "$REPO/$OUT/$S/$TAG/full_seed$N.log" 2>&1 || say "FAILED"; }
  [ -e "${P}_noT_raw_G8000_modelseed$N.h5ad" ] || { say "$TAG seed $N: diagnostics"; dc "$PYSVC" python pipeline/run_condition.py "${C[@]}" --stage diag > "$REPO/$OUT/$S/$TAG/diag_seed$N.log" 2>&1 || say "FAILED"; }
  say "$TAG seed $N: cell-type attribution"; dc "$PYSVC" python pipeline/run_condition.py "${C[@]}" --stage attrib > "$REPO/$OUT/$S/$TAG/attrib_seed$N.log" 2>&1 || say "FAILED"
}
run_model model 1; run_model model 2; run_model model_mix1 1
say "region evaluation"
dc py python realdata/kidney/region_eval.py --root $OUT --data-root $DATA --scenario $S --stim-section hour12 --seed 1 --out $OUT/$S/eval.csv > /dev/null 2>&1 || say "FAILED"
dc py python realdata/kidney/region_eval.py --root $OUT --data-root $DATA --scenario $S --stim-section hour12 --seed 2 --out $OUT/$S/eval_seed2.csv > /dev/null 2>&1 || say "FAILED"
dc py python realdata/kidney/region_eval.py --root $OUT --data-root $DATA --scenario $S --stim-section hour12 --seed 1 --model-dir model_mix1 --out $OUT/$S/eval_mix1.csv > /dev/null 2>&1 || say "FAILED"
say "cell-type attribution by region"
dc py python realdata/kidney/type_attribution.py --scenario $OUT/$S --model-dir model_mix1 --regions $OUT/$S/regions_spots.csv --ctrl-section sham \
   --markers Havcr1 Lcn2 Spp1 Clu Krt8 Fgb Hspa1a Slc34a1 Slc12a1 --out $OUT/$S/attrib_mix1 > /dev/null 2>&1 || say "FAILED"
say "summary"; dc py python realdata/kidney/summarize.py --ws $OUT --data-root $DATA --figure
say "done"
