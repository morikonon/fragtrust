#!/usr/bin/env bash
# Whole pipeline on synthetic data in a few minutes; run before the real data (also on the server).
source "$(dirname "$0")/_common.sh"
r=(--results results_smoke)

ft synth --config configs/data/synthetic.yaml
ft tune --config configs/smoke/lgbm.yaml "${r[@]}"
ft train --config configs/smoke/lgbm.yaml --seed 0 1 "${r[@]}"
ft train --config configs/smoke/cnn.yaml --seed 0 1 "${r[@]}"
ft train --config configs/smoke/transformer.yaml --seed 0 "${r[@]}"
ft ensemble --name smoke_cnn_ens results_smoke/runs/smoke_cnn_s0 results_smoke/runs/smoke_cnn_s1 "${r[@]}"
ft analyze "${r[@]}"
ft summarize "${r[@]}"
ft status "${r[@]}"
