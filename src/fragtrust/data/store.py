"""Processed dataset layout.

<root>/<split>_x.npy      uint8 (n, block_size), memory-mappable
<root>/<split>_y.npy      int16 (n,)
<root>/test_entropy.npy   float32 Shannon entropy of each test block, bits per byte
<root>/features/*.npy     cached hand-crafted features (LightGBM)
<root>/manifest.json      written last: its presence marks a complete dataset
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ..features import extract_file
from ..runtime import read_json

SPLITS = ("train", "val_a", "val_b", "test")
PROBES = ("ood_aes_ctr", "ood_random")
EVAL_SPLITS = ("val_a", "val_b", "test", *PROBES)


class FragmentStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        manifest = self.root / "manifest.json"
        if not manifest.exists():
            raise FileNotFoundError(f"{manifest} not found: run `fragtrust prepare` (or `synth`) first")
        self.manifest = read_json(manifest)

    @property
    def name(self) -> str:
        return self.manifest["name"]

    @property
    def class_names(self) -> list[str]:
        return self.manifest["class_names"]

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    @property
    def block_size(self) -> int:
        return self.manifest["block_size"]

    def x(self, split: str) -> np.ndarray:
        return np.load(self.root / f"{split}_x.npy", mmap_mode="r")

    def y(self, split: str) -> np.ndarray:
        return np.load(self.root / f"{split}_y.npy").astype(np.int64)

    def test_entropy(self) -> np.ndarray:
        return np.load(self.root / "test_entropy.npy")

    def features(self, split: str, workers: int | None = None) -> np.ndarray:
        path = self.root / "features" / f"{split}.npy"
        if not path.exists():
            save_atomic(path, extract_file(self.root / f"{split}_x.npy", workers))
        return np.load(path)


def save_atomic(path: Path, array: np.ndarray) -> None:
    """Write via a temp file so concurrent jobs never read a half-written array."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as f:
        np.save(f, array)
    tmp.replace(path)


def write_split(root: Path, split: str, x: np.ndarray, y: np.ndarray) -> None:
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / f"{split}_x.npy", np.asarray(x, dtype=np.uint8))
    np.save(root / f"{split}_y.npy", np.asarray(y, dtype=np.int16))


def stratified_indices(y: np.ndarray, per_class: int | None, rng: np.random.Generator) -> np.ndarray:
    """Sorted indices with `per_class` random rows of every class (all rows if None)."""
    if per_class is None:
        return np.arange(len(y))
    picked = []
    for c in np.unique(y):
        members = np.flatnonzero(y == c)
        if per_class > len(members):
            raise ValueError(f"class {c}: requested {per_class} rows, only {len(members)} available")
        picked.append(rng.choice(members, per_class, replace=False))
    return np.sort(np.concatenate(picked))


def split_in_halves(idx: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Per-class random halves of `idx`; both returned sorted."""
    first, second = [], []
    labels = y[idx]
    for c in np.unique(labels):
        members = rng.permutation(idx[labels == c])
        half = len(members) // 2
        first.append(members[:half])
        second.append(members[half:])
    return np.sort(np.concatenate(first)), np.sort(np.concatenate(second))
