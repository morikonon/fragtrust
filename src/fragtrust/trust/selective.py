"""Selective classification: risk-coverage curves, AURC and accuracy-targeted abstention thresholds."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import binom


def risk_coverage(conf: np.ndarray, correct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coverage and error rate when accepting the k most confident samples, k = 1..n."""
    order = np.argsort(-conf, kind="stable")
    accepted = np.arange(1, len(conf) + 1)
    return accepted / len(conf), np.cumsum(~correct[order]) / accepted


def aurc(conf: np.ndarray, correct: np.ndarray) -> float:
    return float(risk_coverage(conf, correct)[1].mean())


def ltt_min_size(target_acc: float, delta: float) -> int:
    """Smallest accepted set that can pass the binomial test at all (only with zero errors)."""
    return int(np.ceil(np.log(delta) / np.log1p(-(1.0 - target_acc))))


def meets_target(selective_accuracy: float | None, target_acc: float) -> bool:
    """Abstaining on everything answers nothing, so it does not meet a target."""
    return selective_accuracy is not None and selective_accuracy >= target_acc


def fit_threshold(
    conf: np.ndarray, correct: np.ndarray, target_acc: float, method: str = "ltt", delta: float = 0.05, grid: int = 100
) -> float | None:
    """Lowest confidence threshold whose selective accuracy on calibration data reaches `target_acc`.

    empirical: largest coverage whose empirical accuracy >= target (no finite-sample guarantee).
    ltt: accepted-set sizes are tested in increasing order with a one-sided binomial test and the
    scan stops at the first failure (fixed-sequence Learn-then-Test, Angelopoulos et al. 2021), so
    the selective error is <= 1 - target with probability >= 1 - delta under exchangeability.
    None means no coverage qualifies: abstain on everything.
    """
    if not 0 < target_acc < 1:
        raise ValueError("target_acc must be in (0, 1)")
    order = np.argsort(-conf, kind="stable")
    sorted_conf = conf[order]
    errors = np.cumsum(~correct[order])
    n, max_risk = len(conf), 1.0 - target_acc
    if method == "empirical":
        ok = np.flatnonzero(errors / np.arange(1, n + 1) <= max_risk)
        return float(sorted_conf[ok[-1]]) if len(ok) else None
    if method != "ltt":
        raise ValueError(f"unknown method {method!r}")
    # smaller sets fail the test even with zero errors and would end the sequence at once;
    # m_min depends only on (target, delta), so skipping them keeps the order data-independent
    m_min = ltt_min_size(target_acc, delta)
    if m_min > n:
        return None
    sizes = np.unique(np.r_[m_min, np.ceil(np.arange(1, grid + 1) * n / grid).astype(np.int64)])
    threshold = None
    for m in sizes[sizes >= m_min]:
        if binom.cdf(errors[m - 1], m, max_risk) > delta:
            break
        threshold = float(sorted_conf[m - 1])
    return threshold


def apply_threshold(conf: np.ndarray, correct: np.ndarray, threshold: float | None) -> dict[str, Any]:
    accepted = conf >= threshold if threshold is not None else np.zeros(len(conf), dtype=bool)
    n_accepted = int(accepted.sum())
    return {
        "threshold": threshold,
        "answered": n_accepted,
        "coverage": n_accepted / len(conf),
        "selective_accuracy": float(correct[accepted].mean()) if n_accepted else None,
    }


def selective_report(
    cal_conf: np.ndarray,
    cal_correct: np.ndarray,
    test_conf: np.ndarray,
    test_correct: np.ndarray,
    targets: tuple[float, ...],
    delta: float,
) -> dict[str, Any]:
    """Thresholds fitted on calibration data, evaluated once on test."""
    report: dict[str, Any] = {
        "aurc": aurc(test_conf, test_correct),
        "aurc_oracle": aurc(test_correct.astype(np.float64), test_correct),
        "n_cal": len(cal_conf),
        "targets": {},
    }
    for target in targets:
        entry = {}
        for method in ("empirical", "ltt"):
            result = apply_threshold(test_conf, test_correct, fit_threshold(cal_conf, cal_correct, target, method, delta))
            entry[method] = {**result, "meets_target": meets_target(result["selective_accuracy"], target)}
        entry["ltt"]["m_min"] = ltt_min_size(target, delta)
        report["targets"][f"{target:.2f}"] = entry
    return report
