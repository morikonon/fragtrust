"""One table per run and one per model (mean ± std over seeds) from metrics.json, meta.json and trust.json."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from .config import done_runs
from .runtime import read_json

log = logging.getLogger(__name__)

# column -> (source file, path inside it, scale)
COLUMNS: dict[str, tuple[str, tuple[str, ...], float]] = {
    "accuracy": ("metrics", ("accuracy",), 1),
    "macro_f1": ("metrics", ("macro_f1",), 1),
    "ece": ("trust", ("calibration", "uncalibrated", "ece"), 1),
    "ece_ts": ("trust", ("calibration", "temperature_scaled", "ece"), 1),
    "nll_ts": ("trust", ("calibration", "temperature_scaled", "nll"), 1),
    "aurc": ("trust", ("selective", "msp", "aurc"), 1),
    "coverage_at_95": ("trust", ("selective", "msp", "targets", "0.95", "ltt", "coverage"), 1),
    "coverage_at_99": ("trust", ("selective", "msp", "targets", "0.99", "ltt", "coverage"), 1),
    "aps95_coverage": ("trust", ("conformal", "aps@0.95", "coverage"), 1),
    "aps95_set_size": ("trust", ("conformal", "aps@0.95", "mean_size"), 1),
    "auroc_aes": ("trust", ("ood", "ood_aes_ctr", "auroc", "msp"), 1),
    "auroc_random": ("trust", ("ood", "ood_random", "auroc", "msp"), 1),
    "params": ("meta", ("params",), 1),
    "size_mb": ("meta", ("model_bytes",), 1e-6),
    "train_min": ("meta", ("train_seconds",), 1 / 60),
}


def _lookup(doc: dict[str, Any], path: tuple[str, ...]) -> Any:
    for key in path:
        if not isinstance(doc, dict) or key not in doc:
            return None
        doc = doc[key]
    return doc


def summarize(results: Path) -> pd.DataFrame:
    rows = []
    for run in done_runs(results):
        trust_file = run.path / "analysis" / "trust.json"
        sources = {
            "metrics": read_json(run.path / "metrics.json"),
            "meta": run.meta(),
            "trust": read_json(trust_file) if trust_file.exists() else {},
        }
        row: dict[str, Any] = {"run": run.id, "model": sources["meta"].get("model", run.id), "seed": sources["meta"].get("seed")}
        for column, (source, path, scale) in COLUMNS.items():
            value = _lookup(sources[source], path)
            row[column] = value * scale if isinstance(value, (int, float)) else None
        rows.append(row)
    if not rows:
        raise RuntimeError(f"no finished runs under {results / 'runs'}")

    out = results / "summary"
    out.mkdir(parents=True, exist_ok=True)
    runs = pd.DataFrame(rows)
    runs.to_csv(out / "runs.csv", index=False)
    metrics = list(COLUMNS)
    grouped = runs.groupby("model", sort=False)[metrics]
    by_model = grouped.mean().join(grouped.std(), rsuffix="_std").join(runs.groupby("model").size().rename("n_runs"))
    by_model.to_csv(out / "by_model.csv")
    (out / "by_model.md").write_text(_markdown(by_model, metrics))
    log.info("summary of %d runs written to %s", len(runs), out)
    return by_model


def _cell(mean: Any, std: Any, column: str) -> str:
    if pd.isna(mean):
        return "–"
    digits = 0 if column == "params" else 1 if column in ("size_mb", "train_min") else 4
    text = f"{mean:,.{digits}f}"
    return f"{text} ± {std:.{digits}f}" if not pd.isna(std) and column not in ("params", "size_mb") else text


def _markdown(by_model: pd.DataFrame, metrics: list[str]) -> str:
    header = ["model", "runs", *metrics]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for model, row in by_model.iterrows():
        cells = [str(model), str(row["n_runs"])] + [_cell(row[m], row[f"{m}_std"], m) for m in metrics]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
