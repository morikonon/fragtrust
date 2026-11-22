import numpy as np
import pytest

from fragtrust.data.npz import NpzSource, detect_keys, read_headers


def _stream(src: NpzSource, chunk_rows: int) -> np.ndarray:
    return np.concatenate([chunk for _, chunk in src.iter_chunks(chunk_rows)])


def test_stream_equals_full_array(tmp_path):
    rng = np.random.default_rng(0)
    x = rng.integers(0, 256, (37, 64), dtype=np.uint8)
    y = rng.integers(0, 5, 37).astype(np.uint8)
    for save in (np.savez, np.savez_compressed):
        path = tmp_path / f"{save.__name__}.npz"
        save(path, x=x, y=y)
        src = NpzSource(path, 64)
        assert (src.x_key, src.y_key, src.n) == ("x", "y", 37)
        np.testing.assert_array_equal(_stream(src, 10), x)
        np.testing.assert_array_equal(src.y, y)


def test_keys_detected_by_shape(tmp_path):
    path = tmp_path / "a.npz"
    np.savez(path, blocks=np.zeros((4, 32), np.uint8), labels=np.arange(4), extra=np.zeros(3))
    assert detect_keys(read_headers(path), 32) == ("blocks", "labels")


def test_wider_integer_blocks_are_cast(tmp_path):
    path = tmp_path / "a.npz"
    x = np.arange(64, dtype=np.int16).reshape(2, 32)
    np.savez(path, x=x, y=np.array([0, 1]))
    out = _stream(NpzSource(path, 32), 1)
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, x)


def test_wrong_block_size_is_rejected(tmp_path):
    path = tmp_path / "a.npz"
    np.savez(path, x=np.zeros((4, 32), np.uint8), y=np.arange(4))
    with pytest.raises(ValueError):
        NpzSource(path, 64)
