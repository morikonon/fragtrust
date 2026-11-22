"""Exact-duplicate blocks between training and test data.

FiFTy splits FFT-75 at block level, so identical blocks (e.g. zero padding, repeated headers)
can sit in both train and test. This module measures how often that happens.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


def block_hashes(blocks: np.ndarray) -> np.ndarray:
    """64-bit BLAKE2b digest of every row."""
    digests = b"".join(hashlib.blake2b(row, digest_size=8).digest() for row in blocks)
    return np.frombuffer(digests, dtype="<u8").copy()


def duplicate_report(
    train_hash: np.ndarray,
    train_y: np.ndarray,
    test_hash: np.ndarray,
    test_y: np.ndarray,
    test_constant: np.ndarray,
) -> tuple[dict, pd.DataFrame]:
    seen = np.isin(test_hash, train_hash)
    train_pairs = pd.MultiIndex.from_arrays([train_hash, train_y]).unique()
    same_label = pd.MultiIndex.from_arrays([test_hash, test_y]).isin(train_pairs)
    n = len(test_hash)
    summary = {
        "n_train": len(train_hash),
        "n_test": n,
        "test_in_train": int(seen.sum()),
        "test_in_train_frac": float(seen.mean()),
        "test_in_train_same_label": int(same_label.sum()),
        "test_in_train_other_label_only": int((seen & ~same_label).sum()),
        "test_in_train_constant_blocks": int((seen & test_constant).sum()),
        "test_unique_frac": float(len(np.unique(test_hash)) / n),
    }
    per_class = (
        pd.DataFrame({"label": test_y, "in_train": seen})
        .groupby("label")
        .agg(n_test=("in_train", "size"), test_in_train=("in_train", "sum"))
        .reset_index()
    )
    per_class["test_in_train_frac"] = per_class["test_in_train"] / per_class["n_test"]
    return summary, per_class
