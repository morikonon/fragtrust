"""Logits for arbitrary blocks from a finished run: network, LightGBM or deep ensemble."""

from __future__ import annotations

import numpy as np
from scipy.special import log_softmax, logsumexp

from .config import RunDir
from .data.store import FragmentStore


def load_net(run: RunDir, device="cpu"):
    import torch

    from .models import build_net

    cfg = run.config()
    data = FragmentStore(cfg["data"]["dir"])
    model = build_net(cfg["model"], data.n_classes, data.block_size)
    model.load_state_dict(torch.load(run.path / "best.pt", map_location="cpu", weights_only=True))
    return model.to(device).eval()


def load_booster(run: RunDir):
    import lightgbm as lgb

    return lgb.Booster(model_file=str(run.path / "model.txt"))


def predict_blocks(run: RunDir, x: np.ndarray, device: str = "auto", chunk_rows: int = 16384) -> np.ndarray:
    """Logits (networks) or log-probabilities (LightGBM, ensembles) for every row of `x`."""
    cfg = run.config()
    if "members" in cfg:
        parts = [log_softmax(predict_blocks(RunDir(run.path.parent / m), x, device, chunk_rows), axis=1) for m in cfg["members"]]
        return (logsumexp(np.stack(parts), axis=0) - np.log(len(parts))).astype(np.float32)
    if cfg["model"]["type"] == "lightgbm":
        from .features import extract
        from .models.boosting import log_proba

        booster = load_booster(run)
        return np.concatenate([log_proba(booster, extract(np.asarray(x[a : a + chunk_rows]))) for a in range(0, len(x), chunk_rows)])

    from .runtime import resolve_device
    from .training.trainer import predict_logits

    torch_device = resolve_device(device)
    return predict_logits(load_net(run, torch_device), x, torch_device)
