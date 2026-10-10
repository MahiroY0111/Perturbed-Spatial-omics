#!/usr/bin/env bash
# Download the mouse acetaminophen (APAP) liver injury time course from GEO (snRNA-seq GSE223558, Visium GSE223560)
# into data/liver_apap/download/ and unpack the per-sample files into download/sn/ and download/st/.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
D=$REPO/data/liver_apap/download; mkdir -p "$D/sn" "$D/st"; cd "$D"
B=https://ftp.ncbi.nlm.nih.gov/geo/series/GSE223nnn
[ -s GSE223558_RAW.tar ] || curl -sS -L -o GSE223558_RAW.tar "$B/GSE223558/suppl/GSE223558_RAW.tar"
[ -s GSE223560_RAW.tar ] || curl -sS -L -o GSE223560_RAW.tar "$B/GSE223560/suppl/GSE223560_RAW.tar"
[ -n "$(ls sn)" ] || tar -xf GSE223558_RAW.tar -C sn
[ -n "$(ls st)" ] || tar -xf GSE223560_RAW.tar -C st
echo "liver data ready in $D"
