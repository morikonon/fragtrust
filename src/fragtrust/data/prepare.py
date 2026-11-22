"""Build the processed FFT-75 subset from the official .npz files.

Stratified, fixed-seed sampling; one streaming pass per official file copies the selected rows,
hashes every block (for the train/test duplicate check) and flags constant blocks.
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap

from ..features import block_entropy
from ..runtime import git_commit, read_json, utc_now, write_json
from .leakage import block_hashes, duplicate_report
from .npz import NpzSource
from .probes import PROBE_DESCRIPTIONS, make_probes
from .store import SPLITS, split_in_halves, stratified_indices

log = logging.getLogger(__name__)


def load_class_names(path: str | None, n_classes: int) -> tuple[list[str], str]:
    if path is None:
        return [f"class_{i:02d}" for i in range(n_classes)], "generic placeholder names"
    spec = read_json(Path(path))
    names = spec["classes"]
    if len(names) != n_classes:
        raise ValueError(f"{path}: {len(names)} class names, expected {n_classes}")
    return names, spec.get("source", path)


def sha256_file(path: Path, chunk_bytes: int = 1 << 24) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def copy_rows(
    src: NpzSource, outputs: dict[str, np.ndarray], out: Path, chunk_rows: int
) -> tuple[np.ndarray, np.ndarray]:
    """Copy the rows `outputs[split]` (sorted indices) of `src` into <split>_x.npy in one pass.

    Returns a 64-bit hash and a constant-block flag for every row of the source.
    """
    targets = {
        split: open_memmap(out / f"{split}_x.npy", mode="w+", dtype=np.uint8, shape=(len(idx), src.block_size))
        for split, idx in outputs.items()
    }
    for split, idx in outputs.items():
        np.save(out / f"{split}_y.npy", src.y[idx].astype(np.int16))
    hashes = np.empty(src.n, dtype=np.uint64)
    constant = np.empty(src.n, dtype=bool)
    t0, next_report = time.perf_counter(), 0.1
    for start, chunk in src.iter_chunks(chunk_rows):
        stop = start + len(chunk)
        for split, idx in outputs.items():
            lo, hi = np.searchsorted(idx, (start, stop))
            targets[split][lo:hi] = chunk[idx[lo:hi] - start]
        hashes[start:stop] = block_hashes(chunk)
        constant[start:stop] = chunk.min(axis=1) == chunk.max(axis=1)
        if stop / src.n >= next_report or stop == src.n:
            elapsed = time.perf_counter() - t0
            log.info(
                "%s: %5.1f%% (%s rows, %.0f rows/s, eta %.1f min)",
                src.path.name, 100 * stop / src.n, f"{stop:,}", stop / elapsed, (src.n - stop) / (stop / elapsed) / 60,
            )
            next_report += 0.1
    for target in targets.values():
        target.flush()
    return hashes, constant


def write_derived(root: Path, n_probe: int, seed: int) -> dict[str, Any]:
    """Test-block entropy and OOD probes, derived from <root>/test_x.npy."""
    test_x = np.load(root / "test_x.npy", mmap_mode="r")
    np.save(root / "test_entropy.npy", block_entropy(test_x))
    info = {}
    for name, blocks in make_probes(test_x, n_probe, seed).items():
        np.save(root / f"{name}_x.npy", blocks)
        info[name] = {"rows": len(blocks), "description": PROBE_DESCRIPTIONS[name]}
    return info


def class_balance(root: Path, class_names: list[str]) -> pd.DataFrame:
    table = pd.DataFrame({"class": class_names})
    for split in SPLITS:
        table[split] = np.bincount(np.load(root / f"{split}_y.npy"), minlength=len(class_names))
    return table


def prepare(cfg: dict[str, Any], results: Path, force: bool = False) -> Path:
    out = Path(cfg["out_dir"])
    if (out / "manifest.json").exists() and not force:
        log.info("%s already prepared, skipping (--force to rebuild)", out)
        return out
    out.mkdir(parents=True, exist_ok=True)
    (out / "indices").mkdir(exist_ok=True)
    report_dir = results / "data" / cfg["name"]
    report_dir.mkdir(parents=True, exist_ok=True)
    names, names_source = load_class_names(cfg.get("class_names"), cfg["n_classes"])
    rng = np.random.default_rng(cfg["seed"])
    t0 = time.perf_counter()

    sources = {key: NpzSource(path, cfg["block_size"]) for key, path in cfg["raw"].items()}
    for key, src in sources.items():
        labels = np.unique(src.y)
        if labels.min() < 0 or labels.max() >= len(names):
            raise ValueError(f"{src.path}: labels {labels.min()}..{labels.max()} outside 0..{len(names) - 1}")
        log.info("%s: %s rows, %d classes, keys x=%r y=%r", src.path, f"{src.n:,}", len(labels), src.x_key, src.y_key)

    sub = cfg["subset"]
    train_idx = stratified_indices(sources["train"].y, sub["train_per_class"], rng)
    val_a, val_b = split_in_halves(stratified_indices(sources["val"].y, sub["val_per_class"], rng), sources["val"].y, rng)
    test_idx = stratified_indices(sources["test"].y, sub["test_per_class"], rng)
    plan = {"train": {"train": train_idx}, "val": {"val_a": val_a, "val_b": val_b}, "test": {"test": test_idx}}

    row_stats = {}
    for key, outputs in plan.items():
        row_stats[key] = copy_rows(sources[key], outputs, out, cfg["chunk_rows"])
        for split, idx in outputs.items():
            np.save(out / "indices" / f"{split}.npy", idx)

    (train_hash, _), (test_hash, test_const) = row_stats["train"], row_stats["test"]
    train_y, test_y = sources["train"].y, sources["test"].y
    official, per_class = duplicate_report(train_hash, train_y, test_hash, test_y, test_const)
    subset, _ = duplicate_report(
        train_hash[train_idx], train_y[train_idx], test_hash[test_idx], test_y[test_idx], test_const[test_idx]
    )
    per_class.insert(1, "class", [names[i] for i in per_class["label"]])
    write_json(report_dir / "leakage.json", {"official_train_vs_test": official, "subset_train_vs_test": subset})
    per_class.to_csv(report_dir / "leakage_per_class.csv", index=False)
    log.info("duplicates: %d of %d official test blocks also occur in official train", official["test_in_train"], official["n_test"])

    probes = write_derived(out, cfg["probes"]["n_blocks"], cfg["seed"])
    class_balance(out, names).to_csv(report_dir / "class_balance.csv", index=False)

    source_of = {split: key for key, outputs in plan.items() for split in outputs}
    manifest = {
        "name": cfg["name"],
        "created": utc_now(),
        "git_commit": git_commit(),
        "seed": cfg["seed"],
        "block_size": cfg["block_size"],
        "class_names": names,
        "class_names_source": names_source,
        "subset": sub,
        "sources": {
            key: {
                "path": str(src.path),
                "bytes": src.path.stat().st_size,
                "sha256": sha256_file(src.path) if cfg.get("hash_raw", True) else None,
                "x_key": src.x_key,
                "y_key": src.y_key,
                "rows": src.n,
            }
            for key, src in sources.items()
        },
        "splits": {
            split: {
                "rows": len(np.load(out / f"{split}_y.npy")),
                "source": source_of[split],
                "sha256": sha256_file(out / f"{split}_x.npy"),
            }
            for split in SPLITS
        },
        "probes": probes,
    }
    write_json(out / "manifest.json", manifest)
    write_json(report_dir / "manifest.json", manifest)
    log.info("dataset %s ready in %.1f min: %s", cfg["name"], (time.perf_counter() - t0) / 60, out)
    return out
