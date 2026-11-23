#!/usr/bin/env bash
# One-time setup on the Linux GPU server; safe to rerun. Reuses the image's CUDA build of PyTorch.
source "$(dirname "$0")/_common.sh"

as_root() {
    [[ $EUID -eq 0 ]] && command -v apt-get >/dev/null
}

apt_install() {
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "$@"
}

step "system packages"
missing=()
command -v mkfs.fat >/dev/null || command -v mkfs.vfat >/dev/null || missing+=(dosfstools)
command -v mcopy >/dev/null || missing+=(mtools)
command -v tmux >/dev/null || missing+=(tmux)
command -v wget >/dev/null || missing+=(wget)
if ((${#missing[@]} == 0)); then
    echo "dosfstools, mtools, tmux, wget: present"
elif as_root; then
    apt_install "${missing[@]}"
else
    echo "WARNING: missing ${missing[*]}; install them as root (dosfstools and mtools are only needed to build the case-study image)" >&2
fi

step "python environment (.venv, with the system site packages)"
if [[ ! -x .venv/bin/python ]]; then
    rm -rf .venv
    if ! python3 -m venv --system-site-packages .venv; then
        rm -rf .venv
        as_root || { echo "python3 -m venv failed: install python3-venv as root and rerun" >&2; exit 1; }
        apt_install python3-venv
        python3 -m venv --system-site-packages .venv
    fi
fi
activate_venv
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -e ".[dev,bench]"

step "PyTorch and GPU"
python - <<'EOF'
import torch
print(f"torch {torch.__version__} | CUDA build {torch.version.cuda} | GPU visible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("WARNING: PyTorch sees no CUDA GPU; run_base_plan.sh will refuse to start. Use a CUDA image.")
EOF

step "tests (on Linux with the bench extra nothing should be skipped)"
python -m pytest -rs

step "hardware record"
ft env
echo
echo "setup finished"
