"""All paper figures and tables with one command; assets.json records the files each one was built from."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from ..runtime import git_commit, utc_now, write_json

log = logging.getLogger(__name__)


def _class_names(results, cfg: dict[str, Any]) -> list[str]:
    from ..data.store import FragmentStore

    for model in cfg["models"]:
        runs = results.runs(model)
        if runs:
            return FragmentStore(runs[0].config()["data"]["dir"]).class_names
    return []


def build_assets(config: Path) -> dict[str, Any]:
    from .collect import Results
    from .figures import FIGURES, Context
    from .style import apply_style
    from .tables import TABLES, write_table

    cfg = yaml.safe_load(Path(config).read_text())
    results = Results(Path(cfg.get("results", "results")))
    ctx = Context(results, cfg, Path(cfg.get("out_dir", "paper")), _class_names(results, cfg))
    apply_style()
    built: dict[str, Any] = {}
    for figure in FIGURES:
        made = figure(ctx)
        if made is None:
            log.info("figure %-14s skipped (inputs not available yet)", figure.__name__)
            continue
        files, sources = made
        built[f"figure:{figure.__name__}"] = {"files": files, "sources": sorted(set(sources))}
        log.info("figure %-14s -> %s", figure.__name__, files[0])
    for name, table in TABLES.items():
        made = table(ctx)
        if made is None:
            log.info("table  %-14s skipped (inputs not available yet)", name)
            continue
        df, sources = made
        files = write_table(df, ctx.out / "tables", name)
        built[f"table:{name}"] = {"files": files, "sources": sorted(set(sources))}
        log.info("table  %-14s -> %s", name, files[0])
    write_json(ctx.out / "assets.json", {"created": utc_now(), "git_commit": git_commit(), "config": str(config), "assets": built})
    return built
