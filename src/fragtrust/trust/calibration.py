"""Calibration metrics and temperature scaling (Guo et al., ICML 2017)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax


def probs(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    return softmax(logits / temperature, axis=1)


def nll(logits: np.ndarray, y: np.ndarray, temperature: float = 1.0) -> float:
    return float(-log_softmax(logits / temperature, axis=1)[np.arange(len(y)), y].mean())


def brier(p: np.ndarray, y: np.ndarray) -> float:
    """Multiclass Brier score: mean squared distance to the one-hot label, in [0, 2]."""
    return float(((p**2).sum(axis=1) - 2 * p[np.arange(len(y)), y] + 1).mean())


def reliability(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15, adaptive: bool = False) -> pd.DataFrame:
    """Per-bin count, mean confidence and accuracy; equal-width (lo, hi] bins or equal-mass bins."""
    if adaptive:
        bins = np.empty(len(conf), dtype=np.int64)
        bins[np.argsort(conf, kind="stable")] = np.arange(len(conf)) * n_bins // len(conf)
    else:
        bins = np.clip(np.ceil(conf * n_bins).astype(np.int64) - 1, 0, n_bins - 1)
    count = np.bincount(bins, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        confidence = np.bincount(bins, conf, n_bins) / count
        accuracy = np.bincount(bins, correct.astype(np.float64), n_bins) / count
    return pd.DataFrame({"bin": np.arange(n_bins), "count": count, "confidence": confidence, "accuracy": accuracy})


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15, adaptive: bool = False) -> float:
    table = reliability(conf, correct, n_bins, adaptive)
    filled = table[table["count"] > 0]
    return float((filled["count"] * (filled["accuracy"] - filled["confidence"]).abs()).sum() / len(conf))


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Temperature minimising NLL, searched over log T in [-3, 3]."""
    result = minimize_scalar(
        lambda log_t: nll(logits, y, float(np.exp(log_t))), bounds=(-3.0, 3.0), method="bounded", options={"xatol": 1e-4}
    )
    return float(np.exp(result.x))


def calibration_report(logits: np.ndarray, y: np.ndarray, temperature: float, n_bins: int = 15) -> dict[str, float]:
    p = probs(logits, temperature)
    conf = p.max(axis=1)
    correct = p.argmax(axis=1) == y
    return {
        "temperature": temperature,
        "accuracy": float(correct.mean()),
        "mean_confidence": float(conf.mean()),
        "nll": nll(logits, y, temperature),
        "brier": brier(p, y),
        "ece": ece(conf, correct, n_bins),
        "ece_adaptive": ece(conf, correct, n_bins, adaptive=True),
    }
