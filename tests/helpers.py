from __future__ import annotations

from pathlib import Path

import numpy as np

from fragtrust.data.store import FragmentStore
from fragtrust.data.synthetic import make_synthetic


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
