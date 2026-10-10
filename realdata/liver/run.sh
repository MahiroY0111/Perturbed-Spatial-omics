#!/usr/bin/env bash
# Mouse APAP liver injury: build the inputs, run the method and the comparison methods, compare predicted and observed
# zone-level changes, and summarize (results/liver_apap/). Data: realdata/liver/fetch.sh.
#   scenario        reference snRNA-seq: healthy nuclei (control) and 18-30 h after injury (stimulated);
#                   control ST: healthy Visium section hea02; evaluation: the two 24 h sections
#   scenario_hea03  the same with the other healthy section (hea03) as control ST
# Each scenario is run with the standard settings (model/) and with lambda_mix_gene = 1.0 (model_mix1/); the standard
# scenario is also run with clustering seeds 2 and 3 and compared with the 36 h and 48 h sections.
# Usage:  realdata/liver/run.sh [threads=4]          Environment: PYSVC=py (CPU, default) or pygpu
set -uo pipefail
TH=${1:-4}
source "$(dirname "${BASH_SOURCE[0]}")/../../scripts/common.sh"
DATA=data/liver_apap; OUT=results/liver_apap; NAME=liver
say() { echo "[$(date +%H:%M:%S)] liver: $*"; }
for S in scenario scenario_hea03; do
  CTRL=hea02; [ $S = scenario_hea03 ] && CTRL=hea03
  have "$OUT/$S/scenario_config.json" || { say "$S: build inputs"; dc py python realdata/liver/make_scenario.py --root $DATA \
      --stim 018h 024h 030h --min-umi 2000 --min-genes 500 --ctrl-section $CTRL --out $OUT/$S; }
done
run_model() {  # $1 = scenario, $2 = tag (model or model_mix1), $3 = seed, $4 = control section
  local S=$1 TAG=$2 N=$3 CTRL=$4 OV=() MAPS=()
  [ $TAG = model_mix1 ] && OV=(--override lambda_mix_gene=1.0)
  local C=(--scenario-dir $OUT/$S --output-root $OUT/$S/$TAG --name $NAME --seed $N --auto-dims --skip-qc "${OV[@]}")
  [ -e "$REPO/$OUT/$S/simo/map_norepeat.h5ad" ] && MAPS=(--mappings "norepeat=$OUT/$S/simo/map_norepeat.h5ad" "repeat=$OUT/$S/simo/map_repeat.h5ad")
  mkdir -p "$REPO/$OUT/$S/$TAG"
  local P="$REPO/$OUT/$S/$TAG/predictions/${NAME}_proposed"
  [ -e "${P}_control_raw_G8000_modelseed$N.h5ad" ] || { say "$S $TAG seed $N: spot model"; dc "$PYSVC" python pipeline/run_condition.py "${C[@]}" --stage stmix > "$REPO/$OUT/$S/$TAG/stmix_seed$N.log" 2>&1 || say "FAILED"; }
  [ -e "${P}_stim_raw_G8000_modelseed$N.h5ad" ] || { say "$S $TAG seed $N: CellOT and predictions"; dc "$PYSVC" python pipeline/run_condition.py "${C[@]}" --stage full "${MAPS[@]}" > "$REPO/$OUT/$S/$TAG/full_seed$N.log" 2>&1 || say "FAILED"; }
  [ -e "${P}_noT_raw_G8000_modelseed$N.h5ad" ] || { say "$S $TAG seed $N: diagnostics"; dc "$PYSVC" python pipeline/run_condition.py "${C[@]}" --stage diag > "$REPO/$OUT/$S/$TAG/diag_seed$N.log" 2>&1 || say "FAILED"; }
  [ "$N" = 1 ] && { say "$S $TAG: cell-type attribution"; dc "$PYSVC" python pipeline/run_condition.py "${C[@]}" --stage attrib > "$REPO/$OUT/$S/$TAG/attrib_seed$N.log" 2>&1 || say "FAILED"; }
  return 0
}
zone_eval() {  # $1 = scenario, $2 = tag, $3 = seed, $4 = control section, $5 = output name, $6... = observed sections
  local S=$1 TAG=$2 N=$3 CTRL=$4 NAME_OUT=$5; shift 5
  dc py python realdata/liver/zone_eval.py --scenario $OUT/$S --model-dir $TAG --seed $N --sections-dir $DATA/sections \
     --ctrl-section $CTRL --stim-sections "$@" --out $OUT/$S/$NAME_OUT > /dev/null 2>&1 || say "zone evaluation $S $NAME_OUT FAILED"
}
for S in scenario scenario_hea03; do
  CTRL=hea02; [ $S = scenario_hea03 ] && CTRL=hea03
  if [ ! -e "$REPO/$OUT/$S/simo/map.json" ]; then
    mkdir -p "$REPO/$OUT/$S/simo"; say "$S: SIMO mapping"
    dc simo python pipeline/simo_mapping.py --scenario-dir $OUT/$S --output-prefix $OUT/$S/simo/map --model-seed 1 --skip-qc > "$REPO/$OUT/$S/simo/log.txt" 2>&1 || say "SIMO FAILED"
  fi
  for TAG in model model_mix1; do
    run_model $S $TAG 1 $CTRL
    zone_eval $S $TAG 1 $CTRL zone_eval_$TAG.csv pod24A pod24B
  done
done
for N in 2 3; do
  run_model scenario model $N hea02
  zone_eval scenario model $N hea02 zone_eval_model_seed$N.csv pod24A pod24B
done
for TAG in model model_mix1; do
  zone_eval scenario $TAG 1 hea02 zone_eval_${TAG}_vs_pod36.csv pod36A pod36B
  zone_eval scenario $TAG 1 hea02 zone_eval_${TAG}_vs_pod48.csv pod48A pod48B
done
say "depth check"; dc py python realdata/liver/depth_check.py --ws $OUT
say "summary"; dc py python realdata/liver/summarize.py --ws $OUT --data-root $DATA --figure
say "done"
