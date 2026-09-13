#!/bin/sh
# Public BANC tables (no login). The wiring is NOT here: see docs/banc.md.
set -eu
dir="${1:-data/banc}"
base="https://raw.githubusercontent.com/htem/BANC-project/main/data"
mkdir -p "$dir"
echo "fetching BANC metadata into $dir (about 68 MB)"
curl -fL --progress-bar -o "$dir/banc_888_meta.parquet" "$base/meta/banc_888_meta_20260521.parquet"
curl -fL --progress-bar -o "$dir/banc_neck_functional_classes.csv" \
    "$base/banc_annotations/v888/banc_neck_functional_classes.csv"
curl -fL --progress-bar -o "$dir/banc_neck_functional_classes_by_neuron.csv" \
    "$base/banc_annotations/v888/banc_neck_functional_classes_by_neuron.csv"
ls -la "$dir"
