"""TreeSHAP attributions for the LightGBM model (LightGBM's built-in `pred_contrib`, exact for trees)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import RunDir
from .data.store import FragmentStore, stratified_indices
from .features import FEATURE_NAMES, N_FEATURES
from .inference import load_booster
from .runtime import setup_logging, write_json

log = logging.getLogger(__name__)


def explain(run_path: Path, per_class: int = 20, top_k: int = 5, seed: int = 0) -> dict[str, Any]:
    run = RunDir(run_path)
    cfg = run.config()
    if cfg["model"]["type"] != "lightgbm":
        raise ValueError("SHAP attributions are implemented for the LightGBM model only")
    out = run.path / "analysis"
    setup_logging(out / "explain.log")
    data = FragmentStore(cfg["data"]["dir"])
    y_all = data.y("test")
    per_class = min(per_class, int(np.bincount(y_all).min()))
    idx = stratified_indices(y_all, per_class, np.random.default_rng(seed))
    x, y = data.features("test")[idx], y_all[idx]

    booster = load_booster(run)
    contrib = booster.predict(x, pred_contrib=True).reshape(len(y), data.n_classes, N_FEATURES + 1)[:, :, :N_FEATURES]
    mean_abs = np.abs(contrib).mean(axis=(0, 1))
    gain = booster.feature_importance("gain")
    rho = float(spearmanr(mean_abs, gain).statistic)

    global_table = pd.DataFrame({"feature": FEATURE_NAMES, "mean_abs_shap": mean_abs, "gain": gain})
    global_table = global_table.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    global_table.insert(0, "rank", np.arange(1, N_FEATURES + 1))
    global_table.to_csv(out / "shap_global.csv", index=False)

    toward_true = contrib[np.arange(len(y)), y]
    rows = []
    for k, name in enumerate(data.class_names):
        mean = toward_true[y == k].mean(axis=0)
        for rank, j in enumerate(np.argsort(-mean)[:top_k], 1):
            rows.append({"class": name, "rank": rank, "feature": FEATURE_NAMES[j], "mean_shap": float(mean[j])})
    pd.DataFrame(rows).to_csv(out / "shap_per_class.csv", index=False)

    report = {
        "run": run.id,
        "blocks": int(len(y)),
        "per_class": per_class,
        "spearman_shap_vs_gain": rho,
        "top_features": global_table.head(10)[["feature", "mean_abs_shap"]].to_dict("records"),
    }
    write_json(out / "shap.json", report)
    log.info("%s: SHAP on %d test blocks; top features %s; Spearman(SHAP, gain) = %.3f",
             run.id, len(y), ", ".join(global_table["feature"].head(5)), rho)
    return report
