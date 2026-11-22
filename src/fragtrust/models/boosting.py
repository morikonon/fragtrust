"""Byte-histogram features + LightGBM (model M1)."""

from __future__ import annotations

import itertools
import logging
from typing import Any

import lightgbm as lgb
import numpy as np

from ..features import FEATURE_NAMES


def base_params(n_classes: int, seed: int, threads: int) -> dict[str, Any]:
    return {
        "objective": "multiclass",
        "num_class": n_classes,
        "metric": ["multi_logloss", "multi_error"],
        "seed": seed,
        "num_threads": threads,
        "deterministic": True,
        "force_row_wise": True,
        "verbosity": -1,
    }


def grid_points(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    return [dict(zip(grid, values)) for values in itertools.product(*grid.values())]


def fit(
    params: dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    num_boost_round: int,
    early_stopping: int,
    log_every: int,
) -> lgb.Booster:
    """Train with early stopping on the validation log-loss; progress goes to the fragtrust log."""
    lgb.register_logger(logging.getLogger("fragtrust.lightgbm"))
    train_set = lgb.Dataset(x_train, y_train, feature_name=FEATURE_NAMES)
    val_set = lgb.Dataset(x_val, y_val, reference=train_set)
    return lgb.train(
        params,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=[val_set],
        valid_names=["val_a"],
        callbacks=[
            lgb.early_stopping(early_stopping, first_metric_only=True, verbose=True),
            lgb.log_evaluation(log_every),
        ],
    )


def log_proba(booster: lgb.Booster, x: np.ndarray) -> np.ndarray:
    """Log-probabilities; they act as logits for temperature scaling and conformal scores."""
    p = booster.predict(x, num_iteration=booster.best_iteration)
    return np.log(np.clip(p, 1e-12, None)).astype(np.float32)
