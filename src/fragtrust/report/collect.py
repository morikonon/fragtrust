"""Read finished runs, analyses, benchmarks and case-study outputs for figures and tables."""

from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..config import RunDir, done_runs
from ..runtime import read_json


def lookup(doc: Any, path: tuple[str, ...]) -> Any:
    for key in path:
        if not isinstance(doc, dict) or key not in doc:
            return None
        doc = doc[key]
    return doc


def mean_std(values: list[float | None]) -> tuple[float | None, float | None]:
    clean = [v for v in values if v is not None and not np.isnan(v)]
    if not clean:
        return None, None
    return float(np.mean(clean)), float(np.std(clean, ddof=1)) if len(clean) > 1 else None


class Results:
    def __init__(self, root: Path):
        self.root = Path(root)

    @cached_property
    def _runs(self) -> list[tuple[RunDir, dict[str, Any]]]:
        return [(run, run.meta()) for run in done_runs(self.root)]

    def runs(self, model: str) -> list[RunDir]:
        chosen = [(meta.get("seed") or 0, run) for run, meta in self._runs if meta.get("model") == model]
        return [run for _, run in sorted(chosen, key=lambda item: item[0])]

    def first(self, model: str) -> RunDir | None:
        runs = [r for r in self.runs(model) if (r.path / "analysis" / "trust.json").exists()]
        return runs[0] if runs else None

    def doc(self, run: RunDir, source: str) -> dict[str, Any]:
        path = {"metrics": run.path / "metrics.json", "meta": run.path / "meta.json", "trust": run.path / "analysis" / "trust.json"}[source]
        return read_json(path) if path.exists() else {}

    def values(self, model: str, source: str, path: tuple[str, ...]) -> list[Any]:
        return [lookup(self.doc(run, source), path) for run in self.runs(model)]

    def stat(self, model: str, source: str, path: tuple[str, ...]) -> tuple[float | None, float | None]:
        return mean_std(self.values(model, source, path))

    def benchmark(self, run_id: str, host: str | None = None, backends: tuple[str, ...] = ("onnx", "lightgbm", "torch")) -> dict[str, Any] | None:
        """First available result in backend order; ONNX Runtime is the deployment runtime."""
        hosts = [self.root / "benchmarks" / host] if host else sorted((self.root / "benchmarks").glob("*"))
        for backend in backends:
            for directory in hosts:
                path = directory / f"{run_id}.{backend}.json"
                if path.exists():
                    return read_json(path)
        return None


def load_groups(path: str | None, class_names: list[str]) -> list[tuple[str, list[str]]]:
    """Ordered (group, classes) pairs; classes missing from the file go to a trailing 'other' group."""
    groups: list[tuple[str, list[str]]] = []
    if path and Path(path).exists():
        spec = yaml.safe_load(Path(path).read_text())
        groups = [(g, [c for c in members if c in class_names]) for g, members in spec.items()]
        groups = [(g, members) for g, members in groups if members]
    covered = {c for _, members in groups for c in members}
    rest = [c for c in class_names if c not in covered]
    if rest:
        groups.append(("unassigned" if groups else "classes", rest))
    return groups
