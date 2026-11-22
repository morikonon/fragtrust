from __future__ import annotations

from typing import Any

from torch import nn

from .cnn import ByteCNN
from .transformer import ByteTransformer

NETS: dict[str, type[nn.Module]] = {"cnn": ByteCNN, "transformer": ByteTransformer}


def build_net(spec: dict[str, Any], n_classes: int, block_size: int) -> nn.Module:
    kind = spec["type"]
    if kind not in NETS:
        raise ValueError(f"unknown network type {kind!r}; expected one of {sorted(NETS)}")
    kwargs = {k: v for k, v in spec.items() if k != "type"}
    if kind == "transformer":
        kwargs["block_size"] = block_size
    return NETS[kind](n_classes=n_classes, **kwargs)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
