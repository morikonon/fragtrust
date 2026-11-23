#!/usr/bin/env bash
# Every step of the pipeline on synthetic data in a few minutes, on the laptop or the server.
# Finished steps are skipped on a rerun; delete results_smoke/ for a clean run.
source "$(dirname "$0")/_common.sh"
r=(--results results_smoke)
runs=results_smoke/runs
image=data/case_study/images/smoke_case

step "synthetic data"
ft synth --config configs/data/synthetic.yaml

step "training"
ft tune --config configs/smoke/lgbm.yaml "${r[@]}"
ft train --config configs/smoke/lgbm.yaml --seed 0 1 "${r[@]}"
ft train --config configs/smoke/cnn.yaml --seed 0 1 "${r[@]}"
ft train --config configs/smoke/transformer.yaml --seed 0 "${r[@]}"
ft ensemble --name smoke_cnn_ens "$runs/smoke_cnn_s0" "$runs/smoke_cnn_s1" "${r[@]}"

step "trust analysis"
ft analyze "${r[@]}"
ft summarize "${r[@]}"

step "SHAP (LightGBM)"
ft explain "$runs/smoke_lgbm_s0" "${r[@]}"

step "case study on a simulated image (the paper uses a real FAT32 image, see data/case_study/README.md)"
ft case-simulate --data data/processed/synthetic --out "$image" --name smoke_case
ft case-scan "$runs/smoke_lgbm_s0" "$runs/smoke_cnn_s0" --image "$image" "${r[@]}"
ft case-eval "$runs/smoke_lgbm_s0" "$runs/smoke_cnn_s0" --image "$image" "${r[@]}"

step "CPU benchmark"
backend=torch
if python -c "import onnx, onnxruntime" 2>/dev/null; then
    backend=onnx
else
    echo "onnx/onnxruntime not installed: benchmarking the networks with PyTorch instead"
fi
ft benchmark "$runs/smoke_lgbm_s0" "$runs/smoke_cnn_s0" "$runs/smoke_transformer_s0" \
    --n-blocks 1000 --threads 1 2 --backend "$backend" "${r[@]}"

step "paper figures and tables"
ft report --config configs/smoke/paper.yaml

step "status"
ft status "${r[@]}"
echo
echo "smoke run finished: results_smoke/ (figures and tables in results_smoke/paper/)"
