#!/usr/bin/env bash
# Base plan (CONTEXT.md §10.1). Every step skips finished work and resumes unfinished runs,
# so after any crash simply run this script again.
source "$(dirname "$0")/_common.sh"
mkdir -p results/logs

ft env
ft prepare --config configs/data/fft75_4096_s1_v1.yaml

# LightGBM uses the CPUs while the GPU trains the networks
(
    ft tune --config configs/models/lgbm.yaml
    ft train --config configs/models/lgbm.yaml --seed 0 1 2 3 4
) > results/logs/lgbm.out 2>&1 &
lgbm_pid=$!

ft train --config configs/models/cnn_m.yaml --seed 0 1 2
ft train --config configs/models/cnn_s.yaml --seed 0
ft train --config configs/models/cnn_l.yaml --seed 0
ft train --config configs/models/transformer.yaml --seed 0 1 2

wait "$lgbm_pid"
ft ensemble --name cnn_m_ens results/runs/cnn_m_s0 results/runs/cnn_m_s1 results/runs/cnn_m_s2
ft analyze
ft summarize
