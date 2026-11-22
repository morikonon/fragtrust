#!/usr/bin/env bash
# One-time setup on the GPU server. Reuses the image's CUDA build of PyTorch.
source "$(dirname "$0")/_common.sh"

python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest
ft env
python -c "import torch; print('torch', torch.__version__, '| CUDA', torch.cuda.is_available())"
