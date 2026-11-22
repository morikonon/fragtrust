"""Split conformal prediction sets.

lac:  score 1 - p_y (Sadinle et al., 2019); smallest sets on average.
aps:  randomised adaptive prediction sets (Romano et al., 2020).
raps: aps plus a penalty on large sets (Angelopoulos et al., ICLR 2021).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

METHODS = ("lac", "aps", "raps")


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample corrected (1 - alpha) quantile; inf when the calibration set is too small."""
    n = len(scores)
    level = np.ceil((n + 1) * (1 - alpha)) / n
    return float("inf") if level > 1 else float(np.quantile(scores, level, method="higher"))


def _ranked(p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(-p, axis=1, kind="stable")
    sorted_p = np.take_along_axis(p, order, axis=1)
    return order, sorted_p, np.cumsum(sorted_p, axis=1)


@dataclass
class ConformalPredictor:
    method: str
    alpha: float
    lam: float = 0.01
    k_reg: int = 2
    q: float | None = None

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"unknown conformal method {self.method!r}")

    def _penalty(self, n_classes: int) -> np.ndarray:
        if self.method != "raps":
            return np.zeros(n_classes)
        return self.lam * np.maximum(0, np.arange(1, n_classes + 1) - self.k_reg)

    def _rank_scores(self, p: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Score of the class at every rank; non-decreasing along each row."""
        order, sorted_p, cum = _ranked(p)
        return order, cum - u[:, None] * sorted_p + self._penalty(p.shape[1])

    def fit(self, p: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> ConformalPredictor:
        if self.method == "lac":
            scores = 1.0 - p[np.arange(len(y)), y]
        else:
            order, rank_scores = self._rank_scores(p, rng.random(len(y)))
            rank_of_y = np.argmax(order == y[:, None], axis=1)
            scores = rank_scores[np.arange(len(y)), rank_of_y]
        self.q = conformal_quantile(scores, self.alpha)
        return self

    def predict(self, p: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Boolean (n, n_classes) membership matrix."""
        if self.q is None:
            raise RuntimeError("call fit() first")
        if self.method == "lac":
            return p >= 1.0 - self.q
        order, rank_scores = self._rank_scores(p, rng.random(len(p)))
        sets = np.zeros(p.shape, dtype=bool)
        np.put_along_axis(sets, order, rank_scores <= self.q, axis=1)
        return sets

    @property
    def name(self) -> str:
        return f"{self.method}@{1 - self.alpha:.2f}"


def set_metrics(
    sets: np.ndarray, y: np.ndarray | None = None, target: float | None = None
) -> tuple[dict[str, Any], pd.Series | None]:
    """Set-size statistics; with labels also marginal and per-class coverage."""
    sizes = sets.sum(axis=1)
    metrics: dict[str, Any] = {
        "mean_size": float(sizes.mean()),
        "median_size": float(np.median(sizes)),
        "singleton_frac": float((sizes == 1).mean()),
        "empty_frac": float((sizes == 0).mean()),
    }
    if y is None:
        return metrics, None
    covered = sets[np.arange(len(y)), y]
    class_coverage = pd.Series(covered).groupby(y).mean()
    metrics["coverage"] = float(covered.mean())
    metrics["worst_class_coverage"] = float(class_coverage.min())
    if target is not None:
        metrics["classes_below_target"] = int((class_coverage < target).sum())
    return metrics, class_coverage
