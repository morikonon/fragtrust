#!/usr/bin/env bash
# Base plan on the GPU server (README, section 2). Every step skips finished work and resumes
# unfinished runs, so after any crash simply run this script again. The case study runs afterwards
# with scripts/run_case_study.sh, once the evidence image exists.
source "$(dirname "$0")/_common.sh"
mkdir -p results/logs
exec > >(tee -a results/logs/base_plan.log) 2>&1

step "preflight"
python - <<'EOF'
import importlib.util
import shutil

import torch

problems = []
if not torch.cuda.is_available():
    problems.append("PyTorch sees no CUDA GPU")
for module in ("onnx", "onnxruntime", "matplotlib"):
    if importlib.util.find_spec(module) is None:
        problems.append(f"{module} is missing: run scripts/setup_server.sh")
free_gb = shutil.disk_usage(".").free / 2**30
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'} | free disk: {free_gb:.0f} GB")
if free_gb < 25:
    print("WARNING: less than 25 GB free; the processed subset, feature cache and runs need about 20 GB")
if problems:
    raise SystemExit("preflight failed:\n  " + "\n  ".join(problems))
EOF
ft env

step "data"
ft prepare --config configs/data/fft75_4096_s1_v1.yaml

step "training: LightGBM on the CPUs in the background (results/logs/lgbm.out), networks on the GPU"
(
    ft tune --config configs/models/lgbm.yaml
    ft train --config configs/models/lgbm.yaml --seed 0 1 2 3 4
) > results/logs/lgbm.out 2>&1 &
lgbm_pid=$!

# A rerun must not find a LightGBM job left over from a crashed run writing into the same run directory.
stop_lgbm() {
    if kill -0 "$lgbm_pid" 2>/dev/null; then
        echo "stopping the background LightGBM job (pid $lgbm_pid)" >&2
        pkill -P "$lgbm_pid" 2>/dev/null || true
        kill "$lgbm_pid" 2>/dev/null || true
    fi
}
trap stop_lgbm EXIT

ft train --config configs/models/cnn_m.yaml --seed 0 1 2
ft train --config configs/models/cnn_s.yaml --seed 0
ft train --config configs/models/cnn_l.yaml --seed 0
ft train --config configs/models/transformer.yaml --seed 0 1 2

step "waiting for LightGBM"
if ! wait "$lgbm_pid"; then
    echo "LightGBM failed: see results/logs/lgbm.out" >&2
    exit 1
fi

step "ensemble and trust analysis"
ft ensemble --name cnn_m_ens results/runs/cnn_m_s0 results/runs/cnn_m_s1 results/runs/cnn_m_s2
ft analyze
ft summarize

step "SHAP (LightGBM)"
ft explain results/runs/lgbm_s0

step "CPU benchmark: 1 thread and every available CPU, ONNX Runtime for the networks"
ft benchmark results/runs/lgbm_s0 results/runs/cnn_s_s0 results/runs/cnn_m_s0 results/runs/cnn_l_s0 results/runs/transformer_s0

step "paper figures and tables (paper/)"
ft report

step "status"
ft status
echo
echo "base plan finished; next: scripts/run_case_study.sh once the evidence image is built"
