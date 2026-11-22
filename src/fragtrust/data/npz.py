"""Streaming reader for arrays inside .npz archives.

FFT-75 ships each split as a compressed .npz (x: blocks, y: labels; ~25 GB for 4096-B scenario 1),
so blocks are decompressed chunk by chunk instead of being loaded at once.
"""

from __future__ import annotations

import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import IO

import numpy as np

Header = tuple[tuple[int, ...], np.dtype]


def _read_header(f: IO[bytes]) -> Header:
    version = np.lib.format.read_magic(f)
    if version == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
    elif version == (2, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
    else:
        raise ValueError(f"unsupported .npy format version {version}")
    if fortran:
        raise ValueError("Fortran-ordered arrays are not supported")
    return shape, dtype


def read_headers(path: Path) -> dict[str, Header]:
    headers = {}
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            if name.endswith(".npy"):
                with zf.open(name) as f:
                    headers[name[: -len(".npy")]] = _read_header(f)
    return headers


def _pick(candidates: list[str], preferred: str, what: str, available: list[str]) -> str:
    if preferred in candidates:
        return preferred
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(f"cannot identify the {what} array among {available}")


def detect_keys(headers: dict[str, Header], block_size: int) -> tuple[str, str]:
    """Find the (n, block_size) integer block array and the matching (n,) label array."""
    names = list(headers)
    xs = [k for k, (shape, dt) in headers.items() if len(shape) == 2 and shape[1] == block_size and dt.kind in "ui"]
    x_key = _pick(xs, "x", f"(n, {block_size}) block", names)
    n = headers[x_key][0][0]
    ys = [k for k, (shape, dt) in headers.items() if shape == (n,) and dt.kind in "ui"]
    return x_key, _pick(ys, "y", "label", names)


class NpzSource:
    """Labels held in memory, blocks streamed in chunks of rows."""

    def __init__(self, path: str | Path, block_size: int):
        self.path = Path(path)
        self.block_size = block_size
        headers = read_headers(self.path)
        self.x_key, self.y_key = detect_keys(headers, block_size)
        shape, self.x_dtype = headers[self.x_key]
        self.n = shape[0]
        with np.load(self.path) as archive:
            self.y = archive[self.y_key].astype(np.int64)

    def iter_chunks(self, chunk_rows: int) -> Iterator[tuple[int, np.ndarray]]:
        row_bytes = self.block_size * self.x_dtype.itemsize
        with zipfile.ZipFile(self.path) as zf, zf.open(f"{self.x_key}.npy") as f:
            _read_header(f)
            for start in range(0, self.n, chunk_rows):
                rows = min(chunk_rows, self.n - start)
                buf = f.read(rows * row_bytes)
                if len(buf) != rows * row_bytes:
                    raise EOFError(f"{self.path}: archive truncated at row {start}")
                yield start, _as_uint8(np.frombuffer(buf, self.x_dtype).reshape(rows, self.block_size))


def _as_uint8(chunk: np.ndarray) -> np.ndarray:
    if chunk.dtype == np.uint8:
        return chunk
    if chunk.min() < 0 or chunk.max() > 255:
        raise ValueError("block values outside 0..255")
    return chunk.astype(np.uint8)
