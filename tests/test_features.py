from collections import Counter

import numpy as np
import pytest

from fragtrust.features import (
    FEATURE_NAMES,
    bigram_entropy,
    byte_histograms,
    entropy_bits,
    extract,
    extract_file,
    longest_run,
)


def _naive_bigram_entropy(row: np.ndarray) -> float:
    counts = np.array(list(Counter(zip(row[:-1], row[1:])).values()), dtype=float)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def _naive_longest_run(row: np.ndarray) -> int:
    best = run = 1
    for a, b in zip(row[:-1], row[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return best


def test_uniform_and_constant_blocks():
    x = np.stack([np.tile(np.arange(256, dtype=np.uint8), 16), np.full(4096, 7, np.uint8)])
    hist = byte_histograms(x)
    assert hist.sum(axis=1) == pytest.approx([1.0, 1.0])
    assert entropy_bits(hist) == pytest.approx([8.0, 0.0])
    assert longest_run(x) == pytest.approx([1 / 4096, 1.0])
    assert bigram_entropy(x) == pytest.approx([8.0, 0.0], abs=1e-3)


def test_vectorised_statistics_match_naive_loops():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 4, (20, 300), dtype=np.uint8)
    x[3, 100:180] = 2
    assert bigram_entropy(x) == pytest.approx([_naive_bigram_entropy(r) for r in x])
    assert longest_run(x) * 300 == pytest.approx([_naive_longest_run(r) for r in x])


def test_printable_share_of_text():
    text = np.frombuffer(b"Hello, world!\n" * 30, dtype=np.uint8)[None, :]
    assert extract(text)[0, FEATURE_NAMES.index("printable")] == pytest.approx(1.0)


def test_parallel_extraction_equals_serial(tmp_path):
    x = np.random.default_rng(1).integers(0, 256, (50, 512), dtype=np.uint8)
    np.save(tmp_path / "x.npy", x)
    parallel = extract_file(tmp_path / "x.npy", workers=2, chunk_rows=16)
    np.testing.assert_allclose(parallel, extract(x))
    assert parallel.shape == (50, len(FEATURE_NAMES))
