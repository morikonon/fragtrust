import numpy as np
import pytest
from helpers import calibrated_sample
from scipy.special import softmax

from fragtrust.trust.calibration import brier, ece, fit_temperature, reliability
from fragtrust.trust.conformal import METHODS, ConformalPredictor, conformal_quantile, set_metrics
from fragtrust.trust.ood import anomaly_score, auroc
from fragtrust.trust.selective import apply_threshold, aurc, fit_threshold


def _bernoulli_confidence(n, rng):
    conf = rng.uniform(0.3, 1.0, n)
    return conf, rng.random(n) < conf


def test_ece_is_small_when_calibrated_and_large_when_overconfident():
    rng = np.random.default_rng(0)
    conf, correct = _bernoulli_confidence(200_000, rng)
    assert ece(conf, correct) < 0.01
    assert ece(conf, correct, adaptive=True) < 0.01
    overconfident = np.full(100_000, 0.9)
    assert ece(overconfident, rng.random(100_000) < 0.6) == pytest.approx(0.3, abs=0.01)


def test_adaptive_bins_have_equal_mass():
    conf = np.random.default_rng(1).beta(5, 1, 1500)
    counts = reliability(conf, conf > 0.5, n_bins=15, adaptive=True)["count"]
    assert counts.min() == counts.max() == 100


def test_temperature_scaling_recovers_true_temperature():
    rng = np.random.default_rng(2)
    logits = rng.normal(0, 3, (50_000, 10))
    p = softmax(logits / 2.0, axis=1)
    y = (p.cumsum(axis=1) > rng.random((len(p), 1))).argmax(axis=1)
    assert fit_temperature(logits, y) == pytest.approx(2.0, rel=0.05)


def test_brier_bounds():
    y = np.array([0, 1, 2])
    assert brier(np.eye(3), y) == pytest.approx(0.0)
    assert brier(np.full((3, 4), 0.25), y) == pytest.approx(0.75)


def test_aurc_is_minimal_for_perfect_ranking():
    rng = np.random.default_rng(3)
    correct = rng.random(10_000) < 0.8
    assert aurc(correct.astype(float), correct) < aurc(rng.random(10_000), correct)


def test_ltt_threshold_meets_target_on_fresh_data():
    rng = np.random.default_rng(4)
    cal_conf, cal_correct = _bernoulli_confidence(20_000, rng)
    test_conf, test_correct = _bernoulli_confidence(200_000, rng)
    tau_ltt = fit_threshold(cal_conf, cal_correct, 0.95, "ltt")
    tau_emp = fit_threshold(cal_conf, cal_correct, 0.95, "empirical")
    assert tau_ltt is not None and tau_ltt >= tau_emp
    result = apply_threshold(test_conf, test_correct, tau_ltt)
    assert result["selective_accuracy"] >= 0.95
    assert 0 < result["coverage"] < 1


def test_threshold_is_none_when_target_is_unreachable():
    conf = np.linspace(0, 1, 1000)
    wrong = np.zeros(1000, dtype=bool)
    assert fit_threshold(conf, wrong, 0.9, "ltt") is None
    assert fit_threshold(conf, wrong, 0.9, "empirical") is None
    assert apply_threshold(conf, wrong, None)["coverage"] == 0


def test_conformal_sets_reach_nominal_coverage():
    rng = np.random.default_rng(5)
    cal_p, cal_y = calibrated_sample(5_000, 10, rng)
    test_p, test_y = calibrated_sample(40_000, 10, rng)
    for method in METHODS:
        cp = ConformalPredictor(method, alpha=0.1).fit(cal_p, cal_y, rng)
        metrics, per_class = set_metrics(cp.predict(test_p, rng), test_y, 0.9)
        assert 0.885 <= metrics["coverage"] <= 0.92, (method, metrics)
        assert len(per_class) == 10


def test_conformal_quantile_is_infinite_for_tiny_calibration_sets():
    assert conformal_quantile(np.array([0.1, 0.2, 0.3]), alpha=0.05) == float("inf")


def test_ood_scores_separate_confident_from_flat_logits():
    rng = np.random.default_rng(6)
    confident = rng.normal(0, 1, (500, 5))
    confident[:, 0] += 8
    flat = rng.normal(0, 0.1, (500, 5))
    for kind in ("msp", "entropy", "energy"):
        assert auroc(anomaly_score(confident, 1.0, kind), anomaly_score(flat, 1.0, kind)) > 0.99
