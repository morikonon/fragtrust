"""Classify every data cluster of the formatted image, without file-system metadata."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..config import RunDir
from ..data.store import FragmentStore
from ..inference import predict_blocks
from .build import load_build
from .signatures import match_headers, propagate

log = logging.getLogger(__name__)

HEAD_BYTES = 96


def image_blocks(image_dir: Path) -> np.ndarray:
    """Data area of the image as a (n_clusters, cluster_size) memory map; the geometry is taken from
    the build record because the examiner-facing image has been reformatted."""
    g = load_build(image_dir)["geometry"]
    return np.memmap(Path(image_dir) / "image.img", dtype=np.uint8, mode="r", offset=g["data_offset"], shape=(g["n_clusters"], g["cluster_size"]))


def case_dir(results: Path, image_dir: Path) -> Path:
    return results / "case_study" / load_build(image_dir)["name"]


def scan(run_path: Path, image_dir: Path, results: Path, device: str = "auto", force: bool = False) -> Path:
    run = RunDir(run_path)
    out = case_dir(results, image_dir) / "scans" / f"{run.id}.npy"
    if out.exists() and not force:
        log.info("%s: scan exists, skipping", run.id)
        return out
    blocks = image_blocks(image_dir)
    block_size = FragmentStore(run.config()["data"]["dir"]).block_size
    if blocks.shape[1] != block_size:
        raise ValueError(f"cluster size {blocks.shape[1]} differs from the model block size {block_size}")
    t0 = time.perf_counter()
    logits = predict_blocks(run, blocks, device)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, logits.astype(np.float32))
    seconds = time.perf_counter() - t0
    log.info("%s: %s clusters scanned in %.1fs (%.1f MB/s)", run.id, f"{len(blocks):,}", seconds, blocks.nbytes / seconds / 1e6)
    return out


def signature_table(image_dir: Path, results: Path, chunk_rows: int = 16384) -> dict[str, Any]:
    """Header matches, contiguous propagation and empty-cluster flags for every cluster (cached)."""
    path = case_dir(results, image_dir) / "signatures.npz"
    if path.exists():
        return dict(np.load(path))
    blocks = image_blocks(image_dir)
    heads = np.empty((len(blocks), HEAD_BYTES), dtype=np.uint8)
    empty = np.empty(len(blocks), dtype=bool)
    for a in range(0, len(blocks), chunk_rows):
        chunk = np.asarray(blocks[a : a + chunk_rows])
        heads[a : a + len(chunk)] = chunk[:, :HEAD_BYTES]
        empty[a : a + len(chunk)] = ~chunk.any(axis=1)
    header = match_headers(heads)
    table = {"header": header, "propagated": propagate(header, empty), "empty": empty}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **table)
    return table
