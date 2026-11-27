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
from sklearn.metrics import f1_score

from ..config import RunDir
from ..data.store import DUPLICATES_FILE, EVAL_SPLITS, PROBES, FragmentStore
from ..features import block_entropy
from ..runtime import setup_logging, write_json
from .calibration import calibration_report, ece, fit_temperature, probs, reliability
from .conformal import ConformalPredictor, set_metrics
from .ood import anomaly_score, auroc
from .selective import risk_coverage, selective_report

log = logging.getLogger(__name__)

GROUPS = ("all", "duplicated", "without")


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


def _dec4(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _abstained(conf: np.ndarray, threshold: float | None) -> float:
    return 1.0 if threshold is None else float((conf < threshold).mean())


def group_metrics(y: np.ndarray, pred: np.ndarray, conf: np.ndarray, n_bins: int) -> dict[str, Any]:
    """Accuracy, macro-F1 over the classes present and ECE of a group of test blocks."""
    if not len(y):
        return {"n": 0, "accuracy": None, "macro_f1": None, "ece": None}
    correct = pred == y
    return {
        "n": len(y),
        "accuracy": float(correct.mean()),
        "macro_f1": float(f1_score(y, pred, labels=np.unique(y), average="macro", zero_division=0)),
        "ece": ece(conf, correct, n_bins),
    }


def duplicate_effect(
    y: np.ndarray, pred: np.ndarray, conf: np.ndarray, masks: dict[str, np.ndarray], n_bins: int
) -> dict[str, Any]:
    """Per train side: metrics on all test blocks, on those duplicated in train, and on the rest."""
    every = group_metrics(y, pred, conf, n_bins)
    report = {}
    for side, mask in masks.items():
        if len(mask) != len(y):
            raise ValueError(f"{side} duplicate mask has {len(mask)} entries for {len(y)} test blocks")
        without = group_metrics(y[~mask], pred[~mask], conf[~mask], n_bins)
        report[side] = {
            "all": every,
            "duplicated": group_metrics(y[mask], pred[mask], conf[mask], n_bins),
            "without": without,
            "all_minus_without": {
                k: None if without[k] is None else every[k] - without[k] for k in ("accuracy", "macro_f1", "ece")
            },
        }
    return report


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

    masks = data.test_duplicates()
    if masks is None:
        log.info("%s: no %s in %s (prepared before the masks existed, or synthetic data): duplicate breakdown "
                 "skipped; `fragtrust duplicates` adds it", run.id, DUPLICATES_FILE, data.root)
    else:
        report["duplicates"] = duplicate_effect(y["test"], p["test"].argmax(axis=1), conf["test"], masks, opts.ece_bins)
        for side, d in report["duplicates"].items():
            log.info("%s: %s duplicates %s | accuracy all / duplicated / without %s | ECE (TS) %s", run.id, side,
                     f"{d['duplicated']['n']:,}", *(" / ".join(_dec4(d[g][k]) for g in GROUPS) for k in ("accuracy", "ece")))

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

    test_entropy = data.test_entropy()
    tau99 = _ltt_threshold(report["selective"], max(opts.targets))
    entropy_bins = (
        pd.DataFrame(
            {
                "bin": pd.cut(test_entropy, list(opts.entropy_edges), right=False),
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
        model_auroc = {k: auroc(id_scores[k], anomaly_score(logits[probe], temperature, k)) for k in kinds}
        # model-free baseline: does a plain entropy threshold already separate the probe?
        entropy_auroc = auroc(test_entropy, block_entropy(data.x(probe)))
        log.info("%s: %s AUROC %s, byte entropy %.4f", run.id, probe,
                 ", ".join(f"{k} {v:.4f}" for k, v in model_auroc.items()), entropy_auroc)
        report["ood"][probe] = {
            "n": len(p[probe]),
            "mean_confidence": float(conf[probe].mean()),
            "auroc": {**model_auroc, "byte_entropy": entropy_auroc},
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
