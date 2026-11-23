from __future__ import annotations

from pathlib import Path

import numpy as np

from fragtrust.data.store import FragmentStore
from fragtrust.data.synthetic import make_synthetic

CNN = {"type": "cnn", "emb_dim": 8, "channels": [16, 16], "kernel_size": 5, "pool": 8}
TRANSFORMER = {"type": "transformer", "patch_size": 64, "emb_dim": 4, "d_model": 32, "n_layers": 1, "n_heads": 2}


def make_store(root: Path, block_size: int = 512, per_class: tuple[int, int, int, int] = (60, 30, 30, 30), seed: int = 0) -> FragmentStore:
    make_synthetic(
        {
            "name": "tiny",
            "out_dir": str(root),
            "seed": seed,
            "block_size": block_size,
            "per_class": dict(zip(("train", "val_a", "val_b", "test"), per_class)),
            "probes": {"n_blocks": 20},
        }
    )
    return FragmentStore(root)


def calibrated_sample(n: int, n_classes: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Probabilities and labels drawn from them, i.e. a perfectly calibrated classifier."""
    p = rng.dirichlet(np.full(n_classes, 0.3), size=n)
    y = (p.cumsum(axis=1) > rng.random((n, 1))).argmax(axis=1)
    return p, y


def cnn_config(data_dir: Path, name: str = "cnn", model: dict = CNN, epochs: int = 2) -> dict:
    return {
        "name": name,
        "device": "cpu",
        "data": {"dir": str(data_dir)},
        "model": model,
        "train": {"epochs": epochs, "batch_size": 32, "lr": 3e-3, "warmup_epochs": 0.5, "eval_batch_size": 128, "log_every": 1000},
    }


def transformer_config(data_dir: Path, name: str = "transformer") -> dict:
    return cnn_config(data_dir, name, TRANSFORMER, epochs=1)


def lgbm_config(data_dir: Path, name: str = "lgbm") -> dict:
    return {
        "name": name,
        "data": {"dir": str(data_dir)},
        "model": {"type": "lightgbm"},
        "params": {"num_leaves": 7, "learning_rate": 0.2, "min_data_in_leaf": 5},
        "boost": {"num_boost_round": 30, "early_stopping": 10, "log_every": 100},
        "threads": 1,
    }
