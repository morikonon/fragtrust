"""Every number in the draft comes from a results file.

The draft writes numbers as tags, resolved and rendered by `fragtrust numbers`:

    {{results/summary/by_model.csv:cnn_m/accuracy|pct2}}
    {{results/runs/cnn_m_s0/analysis/trust.json:calibration/temperature_scaled/ece|.3f}}

path:key|format. JSON keys are separated by '/'; for CSV the key is <value of the first column>/<column>.
Formats: any Python format spec (.3f, ,d), or pctN (x100 with N decimals); .Nf and pctN round half
up. Numbers typed by hand outside tags are listed as warnings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

import pandas as pd

from ..runtime import read_json

TAG = re.compile(r"\{\{\s*([^:{}]+?)\s*:\s*([^|{}]+?)\s*(?:\|\s*([^{}]+?)\s*)?\}\}")
BARE_NUMBER = re.compile(r"(?<![\w.\-\[])(\d+\.\d+|\d+(?:\.\d+)?\s?%)(?![\w.])")
FIXED = re.compile(r"\.(\d+)f")
IGNORED = re.compile(r"`[^`]*`|\[\d+(?:[,–-]\s*\d+)*\]|\bRQ\d\b|\b(?:19|20)\d\d\b")


@dataclass
class Check:
    rendered: str
    resolved: int = 0
    errors: list[str] = field(default_factory=list)
    untagged: list[str] = field(default_factory=list)


def resolve(root: Path, path: str, key: str) -> Any:
    file = root / path
    if not file.exists():
        raise KeyError(f"file not found: {path}")
    parts = key.split("/")
    if file.suffix == ".json":
        value: Any = read_json(file)
        for part in parts:
            if isinstance(value, list):
                value = value[int(part)]
            elif isinstance(value, dict) and part in value:
                value = value[part]
            else:
                raise KeyError(f"{path}: key {key!r} not found at {part!r}")
        return value
    if file.suffix == ".csv":
        if len(parts) != 2:
            raise KeyError(f"{path}: CSV keys are <row>/<column>, got {key!r}")
        table = pd.read_csv(file)
        rows = table[table.iloc[:, 0].astype(str) == parts[0]]
        if rows.empty or parts[1] not in table.columns:
            raise KeyError(f"{path}: row {parts[0]!r} or column {parts[1]!r} not found")
        return rows.iloc[0][parts[1]]
    raise KeyError(f"{path}: only .json and .csv are supported")


def render(value: Any, spec: str | None) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        raise ValueError("value is empty")
    if not spec:
        return str(value)
    if spec.startswith("pct"):
        return rounded(value, int(spec[3:] or 1), scale=100)
    if fixed := FIXED.fullmatch(spec):
        return rounded(value, int(fixed.group(1)))
    return format(value, spec)


def rounded(value: float, digits: int, scale: float = 1) -> str:
    """Round half up on the decimal value as written in the results file, so 0.8765 in percent
    with one digit is 87.7 (binary floating point gives 87.6). Tables use the same rounding."""
    exact = Decimal(str(value)) * Decimal(str(scale))
    return str(exact.quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP))


def check(draft: Path, root: Path) -> Check:
    text = Path(draft).read_text()
    result = Check(rendered=text)

    def substitute(match: re.Match) -> str:
        path, key, spec = match.groups()
        try:
            out = render(resolve(root, path, key), spec)
        except (KeyError, ValueError, IndexError) as error:
            line = text.count("\n", 0, match.start()) + 1
            result.errors.append(f"line {line}: {match.group(0)} -> {error}")
            return match.group(0)
        result.resolved += 1
        return out

    result.rendered = TAG.sub(substitute, text)
    for number, line in enumerate(text.splitlines(), 1):
        stripped = IGNORED.sub("", TAG.sub("", line))
        result.untagged += [f"line {number}: {m.group(1)}" for m in BARE_NUMBER.finditer(stripped)]
    return result
