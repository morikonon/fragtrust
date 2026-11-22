"""Hand-crafted block features for the gradient-boosting model, vectorised over rows."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .runtime import cpu_threads

log = logging.getLogger(__name__)

STAT_NAMES = ("entropy", "mean", "std", "longest_run", "bigram_entropy", "printable")
FEATURE_NAMES = [f"byte_{b:02x}" for b in range(256)] + list(STAT_NAMES)
N_FEATURES = len(FEATURE_NAMES)

_PRINTABLE = np.zeros(256, dtype=bool)
_PRINTABLE[0x20:0x7F] = True
_PRINTABLE[[0x09, 0x0A, 0x0D]] = True
_BYTE_VALUES = np.arange(256, dtype=np.float64)


def byte_histograms(x: np.ndarray) -> np.ndarray:
    """Normalised 256-bin histogram per row."""
    n, length = x.shape
    offsets = (np.arange(n, dtype=np.int64) * 256)[:, None]
    counts = np.bincount((x + offsets).ravel(), minlength=n * 256).reshape(n, 256)
    return counts / length


def entropy_bits(p: np.ndarray) -> np.ndarray:
    """Shannon entropy in bits of each row of a probability matrix."""
    safe = np.where(p > 0, p, 1.0)
    return -(p * np.log2(safe)).sum(axis=1)


def longest_run(x: np.ndarray) -> np.ndarray:
    """Longest run of one repeated byte, as a fraction of the block length."""
    n, length = x.shape
    starts_mask = np.ones((n, length), dtype=bool)
    starts_mask[:, 1:] = x[:, 1:] != x[:, :-1]
    starts = np.flatnonzero(starts_mask)
    lengths = np.diff(np.append(starts, n * length))
    first_run_of_row = np.searchsorted(starts, np.arange(n) * length)
    return np.maximum.reduceat(lengths, first_run_of_row) / length


def bigram_entropy(x: np.ndarray) -> np.ndarray:
    """Entropy in bits of the distribution of consecutive byte pairs."""
    n, length = x.shape
    codes = (x[:, :-1].astype(np.int64) << 8) | x[:, 1:]
    codes |= (np.arange(n, dtype=np.int64) << 16)[:, None]
    # the row id in the high bits keeps rows contiguous after a per-row sort
    codes = np.sort(codes, axis=1).ravel()
    starts = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
    p = np.diff(np.append(starts, codes.size)) / (length - 1)
    return np.bincount(codes[starts] >> 16, weights=-p * np.log2(p), minlength=n)


def extract(x: np.ndarray) -> np.ndarray:
    hist = byte_histograms(x)
    mean = hist @ _BYTE_VALUES
    std = np.sqrt(np.maximum(hist @ _BYTE_VALUES**2 - mean**2, 0.0))
    stats = np.column_stack(
        [entropy_bits(hist), mean, std, longest_run(x), bigram_entropy(x), hist[:, _PRINTABLE].sum(axis=1)]
    )
    return np.hstack([hist, stats]).astype(np.float32)


def block_entropy(x: np.ndarray, chunk_rows: int = 8192) -> np.ndarray:
    out = np.empty(len(x), dtype=np.float32)
    for a in range(0, len(x), chunk_rows):
        out[a : a + chunk_rows] = entropy_bits(byte_histograms(np.asarray(x[a : a + chunk_rows])))
    return out


def _extract_rows(path: str, start: int, stop: int) -> np.ndarray:
    return extract(np.asarray(np.load(path, mmap_mode="r")[start:stop]))


def extract_file(npy_path: Path, workers: int | None = None, chunk_rows: int = 2048) -> np.ndarray:
    """Features of every row of a uint8 .npy block array, computed in parallel chunks."""
    n = len(np.load(npy_path, mmap_mode="r"))
    workers = cpu_threads(workers)
    starts = list(range(0, n, chunk_rows))
    stops = [min(s + chunk_rows, n) for s in starts]
    out = np.empty((n, N_FEATURES), dtype=np.float32)
    t0 = time.perf_counter()
    if workers == 1 or len(starts) == 1:
        for a, b in zip(starts, stops):
            out[a:b] = _extract_rows(str(npy_path), a, b)
    else:
        with ProcessPoolExecutor(workers) as pool:
            parts = pool.map(_extract_rows, [str(npy_path)] * len(starts), starts, stops, chunksize=4)
            for a, b, part in zip(starts, stops, parts):
                out[a:b] = part
    log.info("features: %s (%s rows) in %.1fs with %d workers", npy_path.name, f"{n:,}", time.perf_counter() - t0, workers)
    return out
