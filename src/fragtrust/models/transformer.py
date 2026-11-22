from __future__ import annotations

import torch
from torch import nn


class ByteTransformer(nn.Module):
    """Byte embedding -> non-overlapping patches -> pre-norm Transformer encoder -> [CLS] head."""

    def __init__(
        self,
        n_classes: int,
        block_size: int,
        patch_size: int = 16,
        emb_dim: int = 32,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if block_size % patch_size:
            raise ValueError(f"block_size {block_size} is not divisible by patch_size {patch_size}")
        self.patch_size = patch_size
        self.embed = nn.Embedding(256, emb_dim)
        self.patch = nn.Linear(patch_size * emb_dim, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, block_size // patch_size + 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, ff_mult * d_model, dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, length = x.shape
        tokens = self.patch(self.embed(x.long()).reshape(b, length // self.patch_size, -1))
        h = torch.cat([self.cls.expand(b, -1, -1), tokens], dim=1) + self.pos
        return self.head(self.norm(self.encoder(h)[:, 0]))
