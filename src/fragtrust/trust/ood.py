"""Scores for separating unseen data (encrypted, random) from in-distribution blocks."""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp
from sklearn.metrics import roc_auc_score

from .calibration import probs


def anomaly_score(logits: np.ndarray, temperature: float, kind: str) -> np.ndarray:
    """Higher = more likely out of distribution."""
    if kind == "msp":
        return -probs(logits, temperature).max(axis=1)
    if kind == "entropy":
        p = probs(logits, temperature)
        return -(p * np.log(np.clip(p, 1e-30, None))).sum(axis=1)
    if kind == "energy":
        return -temperature * logsumexp(logits / temperature, axis=1)
    raise ValueError(f"unknown score {kind!r}")


def auroc(in_dist: np.ndarray, out_dist: np.ndarray) -> float:
    labels = np.r_[np.zeros(len(in_dist)), np.ones(len(out_dist))]
    return float(roc_auc_score(labels, np.r_[in_dist, out_dist]))
