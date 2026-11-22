from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class ByteCNN(nn.Module):
    """Byte embedding -> [Conv1d, BatchNorm, GELU, MaxPool] blocks -> mean+max pooling -> linear head."""

    def __init__(
        self,
        n_classes: int,
        emb_dim: int = 32,
        channels: Sequence[int] = (64, 128, 128, 256),
        kernel_size: int = 9,
        pool: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed = nn.Embedding(256, emb_dim)
        layers: list[nn.Module] = []
        c_in = emb_dim
        for c_out in channels:
            layers += [
                nn.Conv1d(c_in, c_out, kernel_size, padding=kernel_size // 2, bias=False),
                nn.BatchNorm1d(c_out),
                nn.GELU(),
                nn.MaxPool1d(pool),
            ]
            c_in = c_out
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * c_in, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(self.embed(x.long()).transpose(1, 2))
        return self.head(torch.cat([h.mean(dim=-1), h.amax(dim=-1)], dim=1))
