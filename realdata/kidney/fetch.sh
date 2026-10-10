#!/usr/bin/env bash
# Download the mouse kidney ischemia-reperfusion injury data from GEO (snRNA-seq GSE139107, Visium GSE269622)
# into data/kidney_iri/download/ and unpack the Space Ranger outputs into data/kidney_iri/visium/<section>/.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
R=$REPO/data/kidney_iri; mkdir -p "$R/download"; cd "$R/download"
B=https://ftp.ncbi.nlm.nih.gov/geo
for f in GSE139107_MouseIRI.metadata.txt.gz GSE139107_MouseIRI_control.dge.txt.gz GSE139107_MouseIRI_4hours.dge.txt.gz \
         GSE139107_MouseIRI_12hours.dge.txt.gz GSE139107_MouseIRI_2days.dge.txt.gz; do
  [ -s $f ] || curl -sS -L -o $f "$B/series/GSE139nnn/GSE139107/suppl/$f"
done
for s in GSM8323120_visium_sham_male_spaceranger GSM8323121_visium_hour4_male_spaceranger \
         GSM8323122_visium_hour12_male_spaceranger GSM8323123_visium_day2_male_spaceranger; do
  g=${s%%_*}; sec=$(echo "$s" | cut -d_ -f3)
  [ -s $s.tar.gz ] || curl -sS -L -o $s.tar.gz "$B/samples/GSM8323nnn/$g/suppl/$s.tar.gz"
  mkdir -p "$R/visium/$sec"; [ -n "$(ls "$R/visium/$sec")" ] || tar -xzf $s.tar.gz -C "$R/visium/$sec"
done
echo "kidney data ready in $R"
