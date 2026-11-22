"""Post-hoc trust analysis of one run from its saved logits (CPU only).

Protocol (no test-set tuning):
  val_a -> temperature (and, during training, early stopping / LightGBM grid)
  val_b -> abstention thresholds and conformal quantiles
  test  -> every reported number; probes -> out-of-distribution behaviour
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import RunDir
from ..data.store import EVAL_SPLITS, PROBES, FragmentStore
from ..runtime import setup_logging, write_json
from .calibration import calibration_report, fit_temperature, probs, reliability
from .conformal import ConformalPredictor, set_metrics
from .ood import anomaly_score, auroc
from .selective import risk_coverage, selective_report

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnalysisOptions:
    ece_bins: int = 15
    targets: tuple[float, ...] = (0.95, 0.99)
    delta: float = 0.05
    alphas: tuple[float, ...] = (0.10, 0.05)
    conformal: tuple[str, ...] = ("lac", "aps", "raps")
    main_conformal: str = "aps@0.95"
    entropy_edges: tuple[float, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 7.5, 7.9, 7.95, 8.001)
    curve_points: int = 2000
    seed: int = 0


def _entropy(p: np.ndarray) -> np.ndarray:
    return -(p * np.log(np.clip(p, 1e-30, None))).sum(axis=1)


def _ltt_threshold(selective: dict[str, Any], target: float) -> float | None:
    return selective["msp"]["targets"][f"{target:.2f}"]["ltt"]["threshold"]


def _abstained(conf: np.ndarray, threshold: float | None) -> float:
    return 1.0 if threshold is None else float((conf < threshold).mean())


def analyze(run_path: Path, opts: AnalysisOptions = AnalysisOptions()) -> dict[str, Any]:
    run = RunDir(run_path)
    out = run.path / "analysis"
    setup_logging(out / "analyze.log")
    data = FragmentStore(run.config()["data"]["dir"])
    has_logits = run.meta().get("logits_kind") == "logits"
    rng = np.random.default_rng(opts.seed)

    y = {s: data.y(s) for s in ("val_a", "val_b", "test")}
    logits = {s: run.logits(s) for s in EVAL_SPLITS}
    temperature = fit_temperature(logits["val_a"], y["val_a"])
    log.info("%s: temperature %.4f fitted on val_a", run.id, temperature)
    p = {s: probs(v, temperature) for s, v in logits.items()}
    conf = {s: v.max(axis=1) for s, v in p.items()}
    correct = {s: p[s].argmax(axis=1) == y[s] for s in y}
    report: dict[str, Any] = {"run": run.id, "temperature": temperature, "n_test": len(y["test"])}

    raw_test = probs(logits["test"])
    report["calibration"] = {
        "uncalibrated": calibration_report(logits["test"], y["test"], 1.0, opts.ece_bins),
        "temperature_scaled": calibration_report(logits["test"], y["test"], temperature, opts.ece_bins),
    }
    for name, pt in (("uncalibrated", raw_test), ("temperature_scaled", p["test"])):
        reliability(pt.max(axis=1), pt.argmax(axis=1) == y["test"], opts.ece_bins).to_csv(
            out / f"reliability_{name}.csv", index=False
        )
    del raw_test

    scores = {"msp": conf, "neg_entropy": {s: -_entropy(p[s]) for s in ("val_b", "test")}}
    report["selective"] = {
        name: selective_report(sc["val_b"], correct["val_b"], sc["test"], correct["test"], opts.targets, opts.delta)
        for name, sc in scores.items()
    }
    coverage, risk = risk_coverage(conf["test"], correct["test"])
    keep = np.unique(np.linspace(0, len(coverage) - 1, opts.curve_points).astype(np.int64))
    pd.DataFrame({"coverage": coverage[keep], "risk": risk[keep]}).to_csv(out / "risk_coverage.csv", index=False)

    predictors = [
        ConformalPredictor(method, alpha).fit(p["val_b"], y["val_b"], rng)
        for method in opts.conformal
        for alpha in opts.alphas
    ]
    report["conformal"], class_coverage, test_sets = {}, {}, {}
    for cp in predictors:
        test_sets[cp.name] = cp.predict(p["test"], rng)
        metrics, class_coverage[cp.name] = set_metrics(test_sets[cp.name], y["test"], 1 - cp.alpha)
        report["conformal"][cp.name] = {"q": cp.q, **metrics}
    coverage_table = pd.DataFrame(class_coverage)
    coverage_table.insert(0, "class", [data.class_names[i] for i in coverage_table.index])
    coverage_table.to_csv(out / "conformal_class_coverage.csv", index=False)
    main_cp = next(cp for cp in predictors if cp.name == opts.main_conformal)
    main_sizes = test_sets[opts.main_conformal].sum(axis=1)
    del test_sets

    tau99 = _ltt_threshold(report["selective"], max(opts.targets))
    entropy_bins = (
        pd.DataFrame(
            {
                "bin": pd.cut(data.test_entropy(), list(opts.entropy_edges), right=False),
                "correct": correct["test"],
                "confidence": conf["test"],
                "accepted": conf["test"] >= tau99 if tau99 is not None else False,
                "set_size": main_sizes,
            }
        )
        .groupby("bin", observed=False)
        .agg(
            n=("correct", "size"),
            accuracy=("correct", "mean"),
            mean_confidence=("confidence", "mean"),
            coverage_at_max_target=("accepted", "mean"),
            mean_set_size=("set_size", "mean"),
        )
        .reset_index()
    )
    entropy_bins["bin"] = entropy_bins["bin"].astype(str)
    entropy_bins.to_csv(out / "entropy_bins.csv", index=False)

    kinds = ("msp", "entropy", "energy") if has_logits else ("msp", "entropy")
    id_scores = {k: anomaly_score(logits["test"], temperature, k) for k in kinds}
    report["ood"] = {}
    for probe in PROBES:
        predicted = pd.Series(p[probe].argmax(axis=1)).map(dict(enumerate(data.class_names)))
        sets, _ = set_metrics(main_cp.predict(p[probe], rng))
        report["ood"][probe] = {
            "n": len(p[probe]),
            "mean_confidence": float(conf[probe].mean()),
            "auroc": {k: auroc(id_scores[k], anomaly_score(logits[probe], temperature, k)) for k in kinds},
            "abstained": {
                f"{t:.2f}": _abstained(conf[probe], _ltt_threshold(report["selective"], t)) for t in opts.targets
            },
            f"{opts.main_conformal}_sets": sets,
            "top_predicted": predicted.value_counts(normalize=True).head(5).round(4).to_dict(),
        }

    write_json(out / "trust.json", report)
    cal = report["calibration"]
    sel = report["selective"]["msp"]["targets"][f"{max(opts.targets):.2f}"]["ltt"]
    selective_acc = "n/a" if sel["selective_accuracy"] is None else f"{sel['selective_accuracy']:.4f}"
    log.info(
        "%s | ECE %.4f -> %.4f | coverage at %.0f%% target %.3f (accuracy %s) | %s coverage %.4f, mean size %.2f",
        run.id, cal["uncalibrated"]["ece"], cal["temperature_scaled"]["ece"], 100 * max(opts.targets),
        sel["coverage"], selective_acc, opts.main_conformal,
        report["conformal"][opts.main_conformal]["coverage"], report["conformal"][opts.main_conformal]["mean_size"],
    )
    return report
