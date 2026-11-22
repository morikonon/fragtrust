"""Mini-batches over uint8 block arrays.

With `resident=True` the whole array is copied to the GPU once (the 1.5M x 4096 B training subset
is ~6 GB), so batching is a device-side gather and the GPU never waits on a data loader.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np
import torch

Batch = tuple[torch.Tensor, torch.Tensor]


def fits_on_device(device: torch.device, nbytes: int, max_share: float = 0.5) -> bool:
    if device.type != "cuda":
        return False
    free, _ = torch.cuda.mem_get_info(device)
    return nbytes < max_share * free


def to_device(x: np.ndarray, device: torch.device, chunk_rows: int = 65536) -> torch.Tensor:
    out = torch.empty(x.shape, dtype=torch.uint8, device=device)
    for a in range(0, len(x), chunk_rows):
        out[a : a + chunk_rows] = torch.from_numpy(np.array(x[a : a + chunk_rows])).to(device)
    return out


class BlockBatches:
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        device: torch.device,
        *,
        shuffle: bool = False,
        seed: int = 0,
        resident: bool = False,
        drop_last: bool = False,
    ):
        self.n = len(y)
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.resident = resident
        self.generator = torch.Generator().manual_seed(seed)
        if resident:
            self.x = to_device(x, device)
            self.y = torch.as_tensor(y, dtype=torch.long, device=device)
        else:
            self.x = x
            self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return self.n // self.batch_size if self.drop_last else math.ceil(self.n / self.batch_size)

    def __iter__(self) -> Iterator[Batch]:
        order = torch.randperm(self.n, generator=self.generator) if self.shuffle else None
        for i in range(len(self)):
            a, b = i * self.batch_size, min((i + 1) * self.batch_size, self.n)
            yield self._gather(order[a:b]) if order is not None else self._slice(a, b)

    def _gather(self, idx: torch.Tensor) -> Batch:
        if self.resident:
            idx = idx.to(self.device)
            return self.x[idx], self.y[idx]
        idx, _ = idx.sort()
        return self._upload(torch.from_numpy(self.x[idx.numpy()])), self._upload(self.y[idx])

    def _slice(self, a: int, b: int) -> Batch:
        if self.resident:
            return self.x[a:b], self.y[a:b]
        return self._upload(torch.from_numpy(np.array(self.x[a:b]))), self._upload(self.y[a:b])

    def _upload(self, t: torch.Tensor) -> torch.Tensor:
        if self.device.type == "cuda":
            return t.pin_memory().to(self.device, non_blocking=True)
        return t.to(self.device)
