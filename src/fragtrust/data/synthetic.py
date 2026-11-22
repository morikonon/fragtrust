"""Small synthetic dataset in the processed format, for smoke tests without FFT-75.

Classes have different byte statistics; `uniform` and `near_uniform` overlap on purpose so the
models make errors and the calibration/abstention code has something to measure.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from ..runtime import utc_now, write_json
from .prepare import class_balance, write_derived
from .store import write_split

Generator = Callable[[np.random.Generator, int, int], np.ndarray]

_TEXT = np.frombuffer(b"etaoin shrdlu cmfwyp ETAOIN\n.,;0123456789", dtype=np.uint8)
_NEAR_UNIFORM_P = np.full(256, 1.0)
_NEAR_UNIFORM_P[:16] = 1.15
_NEAR_UNIFORM_P /= _NEAR_UNIFORM_P.sum()


def _pattern(rng: np.random.Generator, n: int, length: int) -> np.ndarray:
    period = rng.integers(0, 256, size=(n, 16), dtype=np.uint8)
    blocks = np.tile(period, length // 16)
    noise = rng.random((n, length)) < 0.05
    blocks[noise] = rng.integers(0, 256, size=noise.sum(), dtype=np.uint8)
    return blocks


GENERATORS: dict[str, Generator] = {
    "text": lambda rng, n, length: rng.choice(_TEXT, size=(n, length)),
    "sparse": lambda rng, n, length: ((rng.random((n, length)) < 0.05) * rng.integers(1, 256, (n, length))).astype(np.uint8),
    "geometric": lambda rng, n, length: np.minimum(rng.geometric(0.03, (n, length)), 255).astype(np.uint8),
    "nibble": lambda rng, n, length: rng.integers(0, 16, (n, length), dtype=np.uint8),
    "pattern": _pattern,
    "uniform": lambda rng, n, length: rng.integers(0, 256, (n, length), dtype=np.uint8),
    "near_uniform": lambda rng, n, length: rng.choice(256, size=(n, length), p=_NEAR_UNIFORM_P).astype(np.uint8),
}


def make_synthetic(cfg: dict[str, Any]) -> Path:
    out = Path(cfg["out_dir"])
    rng = np.random.default_rng(cfg["seed"])
    names = list(GENERATORS)
    for split, n in cfg["per_class"].items():
        x = np.concatenate([GENERATORS[name](rng, n, cfg["block_size"]) for name in names])
        y = np.repeat(np.arange(len(names)), n)
        order = rng.permutation(len(y))
        write_split(out, split, x[order], y[order])
    probes = write_derived(out, cfg["probes"]["n_blocks"], cfg["seed"])
    class_balance(out, names).to_csv(out / "class_balance.csv", index=False)
    write_json(
        out / "manifest.json",
        {
            "name": cfg["name"],
            "created": utc_now(),
            "synthetic": True,
            "seed": cfg["seed"],
            "block_size": cfg["block_size"],
            "class_names": names,
            "class_names_source": "synthetic generators",
            "probes": probes,
        },
    )
    return out
