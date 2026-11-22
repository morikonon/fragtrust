"""Experiment configs (YAML) and run directories."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .runtime import read_json, utc_now, write_json


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    cfg.setdefault("name", path.stem)
    return cfg


def save_yaml(obj: Any, path: Path) -> None:
    path.write_text(yaml.safe_dump(obj, sort_keys=False))


class RunDir:
    """One trained model or ensemble: config, logs, weights, logits and analysis in one folder."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    @property
    def id(self) -> str:
        return self.path.name

    @property
    def done(self) -> bool:
        return (self.path / "DONE").exists()

    def mark_done(self) -> None:
        (self.path / "DONE").write_text(utc_now() + "\n")

    def config(self) -> dict[str, Any]:
        return yaml.safe_load((self.path / "config.yaml").read_text())

    def meta(self) -> dict[str, Any]:
        path = self.path / "meta.json"
        return read_json(path) if path.exists() else {}

    def update_meta(self, **fields: Any) -> None:
        write_json(self.path / "meta.json", {**self.meta(), **fields})

    def save_logits(self, split: str, logits: np.ndarray) -> None:
        np.save(self.path / f"logits_{split}.npy", np.asarray(logits, dtype=np.float32))

    def logits(self, split: str) -> np.ndarray:
        return np.load(self.path / f"logits_{split}.npy")


def done_runs(results: Path) -> list[RunDir]:
    return [run for run in map(RunDir, sorted((results / "runs").glob("*"))) if run.done]
