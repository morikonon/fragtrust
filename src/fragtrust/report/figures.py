"""Paper figures, each built only from files under results/. A figure whose inputs do not exist yet
is skipped, so the command can run at any stage of the project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, NullFormatter
from matplotlib.transforms import blended_transform_factory

from ..case_study.evaluate import ABSTAINED, WRONG
from .collect import Results, load_groups
from .style import AXIS, CATEGORICAL, EMPTY, INK_2, MARKERS, MUTED, SERIES, TEXT_WIDTH, model_styles, save


@dataclass
class Context:
    results: Results
    cfg: dict[str, Any]
    out: Path
    class_names: list[str]

    @property
    def models(self) -> dict[str, str]:
        return self.cfg["models"]

    @property
    def base_models(self) -> dict[str, str]:
        return {m: label for m, label in self.models.items() if not m.endswith("_ens")}

    def styles(self, models) -> dict[str, dict]:
        return model_styles(list(models))


Figure = tuple[list[str], list[str]]


def _legend_on_top(fig, handles, ncol: int) -> None:
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=ncol, handlelength=2.2)


def reliability(ctx: Context) -> Figure | None:
    runs = [(m, label, ctx.results.first(m)) for m, label in ctx.models.items()]
    runs = [r for r in runs if r[2] is not None]
    if not runs:
        return None
    styles = ctx.styles(m for m, _, _ in runs)
    ncols = 2 if len(runs) > 2 else len(runs)
    nrows = -(-len(runs) // ncols)
    side = TEXT_WIDTH / ncols * 0.78
    fig, axes = plt.subplots(nrows, ncols, figsize=(TEXT_WIDTH * 0.8, side * nrows + 0.5), sharex=True, sharey=True,
                             squeeze=False, layout="constrained")
    sources = []
    for ax, (model, label, run) in zip(axes.flat, runs):
        trust = ctx.results.doc(run, "trust")
        for kind, style in (
            ("uncalibrated", {"color": MUTED, "marker": "o", "markerfacecolor": "white", "linestyle": "--"}),
            ("temperature_scaled", {k: styles[model][k] for k in ("color", "marker", "markerfacecolor")} | {"linestyle": "-"}),
        ):
            path = run.path / "analysis" / f"reliability_{kind}.csv"
            table = pd.read_csv(path)
            table = table[table["count"] > 0]
            ax.plot(table["confidence"], table["accuracy"], **style)
            sources.append(str(path))
        ax.plot([0, 1], [0, 1], color=AXIS, linewidth=0.8, zorder=0)
        before, after = (trust["calibration"][k]["ece"] for k in ("uncalibrated", "temperature_scaled"))
        ax.set_title(label, loc="left", pad=11)
        ax.text(0, 1.02, f"ECE {before:.3f} → {after:.3f}, T = {trust['temperature']:.2f}", transform=ax.transAxes,
                fontsize=6.5, color=INK_2, va="bottom")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
    for ax in list(axes.flat)[len(runs):]:
        ax.set_visible(False)
    fig.supxlabel("Confidence (max. softmax probability)", fontsize=8)
    fig.supylabel("Accuracy", fontsize=8)
    handles = [
        Line2D([], [], color=MUTED, marker="o", markerfacecolor="white", linestyle="--", label="Uncalibrated"),
        Line2D([], [], color=INK_2, marker="o", linestyle="-", label="Temperature-scaled (model colour)"),
        Line2D([], [], color=AXIS, linestyle="-", label="Perfect calibration"),
    ]
    _legend_on_top(fig, handles, 3)
    return save(fig, ctx.out / "figures", "reliability"), sources


def risk_coverage(ctx: Context) -> Figure | None:
    runs = [(m, label, ctx.results.first(m)) for m, label in ctx.models.items()]
    runs = [r for r in runs if r[2] is not None]
    if not runs:
        return None
    styles = ctx.styles(m for m, _, _ in runs)
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH * 0.62, 2.3))
    sources = []
    for model, label, run in runs:
        path = run.path / "analysis" / "risk_coverage.csv"
        table = pd.read_csv(path)
        ax.plot(table["coverage"], 100 * table["risk"], label=label, markevery=max(1, len(table) // 8), **styles[model])
        sources.append(str(path))
    transform = blended_transform_factory(ax.transAxes, ax.transData)
    for target in (95, 99):
        ax.axhline(100 - target, color=MUTED, linewidth=0.6, linestyle=(0, (3, 2)), zorder=1)
        ax.text(1.01, 100 - target, f"{target}% accuracy", transform=transform, fontsize=6.5, color=INK_2, va="center")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Coverage (share of blocks labelled automatically)")
    ax.set_ylabel("Error rate on labelled blocks (%)")
    ax.legend(loc="upper left")
    return save(fig, ctx.out / "figures", "risk_coverage"), sources


def per_class_f1(ctx: Context) -> Figure | None:
    models = [m for m in ctx.base_models if ctx.results.runs(m)]
    if not models:
        return None
    styles = ctx.styles(models)
    groups = load_groups(ctx.cfg.get("groups"), ctx.class_names)
    order = [c for _, members in groups for c in members]
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 2.5))
    sources = []
    x = np.arange(len(order))
    for model in models:
        tables = []
        for run in ctx.results.runs(model):
            path = run.path / "per_class_test.csv"
            tables.append(pd.read_csv(path).set_index("class")["f1"])
            sources.append(str(path))
        f1 = pd.concat(tables, axis=1).mean(axis=1).reindex(order)
        ax.plot(x, f1.to_numpy(), linestyle="none", label=ctx.models[model], markersize=3,
                **{k: styles[model][k] for k in ("color", "marker", "markerfacecolor")})
    transform = blended_transform_factory(ax.transData, ax.transAxes)
    start = 0
    for name, members in groups:
        if start:
            ax.axvline(start - 0.5, color=AXIS, linewidth=0.6)
        ax.text(start + (len(members) - 1) / 2, 1.02, name, transform=transform, ha="center", fontsize=6.5, color=INK_2)
        start += len(members)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=90, fontsize=5.5)
    ax.grid(axis="x", visible=False)
    ax.set_xlim(-0.7, len(order) - 0.3)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("F1 on test (mean over seeds)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.08), ncol=len(models))
    return save(fig, ctx.out / "figures", "per_class_f1"), sources


def _entropy_label(interval: str) -> str:
    lo, hi = interval.strip("[]()").split(",")
    return f"{float(lo):g}–{min(float(hi), 8.0):g}"


def entropy(ctx: Context) -> Figure | None:
    models = [m for m in ctx.models if ctx.results.first(m)]
    if not models:
        return None
    styles = ctx.styles(models)
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.4), sharex=True, layout="constrained")
    sources, labels = [], None
    for model in models:
        tables = []
        for run in ctx.results.runs(model):
            path = run.path / "analysis" / "entropy_bins.csv"
            if path.exists():
                tables.append(pd.read_csv(path))
                sources.append(str(path))
        table = pd.concat(tables).groupby("bin", sort=False)[["accuracy", "coverage_at_max_target"]].mean()
        labels = [_entropy_label(b) for b in table.index]
        x = np.arange(len(table))
        for ax, column in zip(axes, ("accuracy", "coverage_at_max_target")):
            ax.plot(x, 100 * table[column].to_numpy(), label=ctx.models[model], **styles[model])
    for ax, (title, ylabel) in zip(axes, (("(a)", "Accuracy (%)"), ("(b)", "Coverage at 99% target (%)"))):
        ax.set_title(title, loc="left")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 102)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=6)
        ax.set_xlabel("Block entropy (bits per byte)")
    _legend_on_top(fig, axes[0].get_legend_handles_labels()[0], len(models))
    return save(fig, ctx.out / "figures", "entropy"), sources


def capacity(ctx: Context) -> Figure | None:
    models = {m: label for m, label in ctx.cfg.get("capacity", {}).items() if ctx.results.runs(m)}
    if len(models) < 2:
        return None
    params = [ctx.results.doc(ctx.results.runs(m)[0], "meta").get("params") for m in models]
    stats = [ctx.results.stat(m, "metrics", ("accuracy",)) for m in models]
    benches = [ctx.results.benchmark(ctx.results.runs(m)[0].id, ctx.cfg.get("benchmark_host")) for m in models]
    has_bench = all(benches)
    fig, axes = plt.subplots(1, 2 if has_bench else 1, figsize=(TEXT_WIDTH if has_bench else TEXT_WIDTH * 0.5, 2.2),
                             squeeze=False, layout="constrained")
    ax = axes[0, 0]
    means = [100 * s[0] for s in stats]
    errors = [100 * (s[1] or 0) for s in stats]
    ax.errorbar(params, means, yerr=errors, color=SERIES[0], marker=MARKERS[0], capsize=2, linewidth=1.2)
    for p, m, label in zip(params, means, models.values()):
        ax.annotate(label, (p, m), textcoords="offset points", xytext=(4, 4), ha="left", va="bottom", fontsize=6.5, color=INK_2)
    _params_axis(ax, params)
    ax.margins(y=0.25)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("(a)", loc="left")
    sources = [str(r.path / "metrics.json") for m in models for r in ctx.results.runs(m)]
    if has_bench:
        ax = axes[0, 1]
        thread_counts = sorted({row["threads"] for b in benches for row in b["results"]})
        for i, threads in enumerate((thread_counts[0], thread_counts[-1]) if len(thread_counts) > 1 else thread_counts):
            speed = [next(row["mb_per_s"] for row in b["results"] if row["threads"] == threads) for b in benches]
            ax.plot(params, speed, color=SERIES[i], marker=MARKERS[i], label=f"{threads} thread{'s' if threads > 1 else ''}")
        _params_axis(ax, params)
        ax.set_ylabel("CPU throughput (MB/s)")
        ax.set_ylim(bottom=0)
        ax.set_title(f"(b) {benches[0]['hardware']['cpu']}", loc="left", fontsize=7)
        ax.legend(loc="best")
    return save(fig, ctx.out / "figures", "capacity"), sources


def _params_axis(ax, params: list[int]) -> None:
    ax.set_xscale("log")
    ax.set_xticks(params)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1e6:.2g}M" if v >= 1e5 else f"{v / 1e3:.3g}k"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(min(params) / 1.6, max(params) * 1.6)
    ax.set_xlabel("Parameters (log scale)")


def _feature_label(name: str) -> str:
    return f"byte 0x{name[5:]}" if name.startswith("byte_") else name.replace("_", " ")


def shap(ctx: Context) -> Figure | None:
    run_id = ctx.cfg.get("shap_run")
    path = ctx.results.root / "runs" / str(run_id) / "analysis" / "shap_global.csv"
    if not run_id or not path.exists():
        return None
    top = pd.read_csv(path).head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH * 0.55, 2.8))
    ax.barh([_feature_label(f) for f in top["feature"]], top["mean_abs_shap"], color=SERIES[0], height=0.6)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Mean |SHAP value| (log-odds)")
    ax.tick_params(axis="y", labelsize=6.5, length=0)
    return save(fig, ctx.out / "figures", "shap_top_features"), [str(path)]


def _binned(values: np.ndarray, mask: np.ndarray, edges: np.ndarray) -> np.ndarray:
    out = np.full(len(edges) - 1, np.nan)
    for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        if mask[a:b].any():
            out[i] = values[a:b][mask[a:b]].mean()
    return out


def disk_map(ctx: Context) -> Figure | None:
    spec = ctx.cfg.get("case_study") or {}
    path = ctx.results.root / "case_study" / str(spec.get("image")) / str(spec.get("run")) / "disk_map.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=False)
    label, model, signature = data["label"], data["model"], data["signature"]
    names = [str(n) for n in data["class_names"]]
    groups = load_groups(ctx.cfg.get("groups"), names)
    group_of = {c: i for i, (_, members) in enumerate(groups) for c in members}
    gidx = np.array([group_of[names[k]] if k >= 0 else (len(groups) if k == -2 else -1) for k in label])
    evidence = label != -1
    edges = np.linspace(0, len(label), min(1000, len(label)) + 1).astype(int)

    majority = np.full(len(edges) - 1, np.nan)
    for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        g = gidx[a:b][evidence[a:b]]
        if len(g):
            majority[i] = np.bincount(g).argmax()
    rows = [
        ("Model: wrong", _binned(model == WRONG, evidence, edges)),
        ("Model: abstained", _binned(model == ABSTAINED, evidence, edges)),
        ("Carver: wrong", _binned(signature == WRONG, evidence, edges)),
        ("Carver: no answer", _binned(signature == ABSTAINED, evidence, edges)),
    ]
    n_bytes = len(label) * float(data["cluster_size"])
    scale, unit = (1e9, "GB") if n_bytes >= 2e9 else (1e6, "MB")
    extent = (0, n_bytes / scale, 0, 1)

    fig, axes = plt.subplots(1 + len(rows), 1, figsize=(TEXT_WIDTH, 2.0), sharex=True, layout="constrained")
    fig.get_layout_engine().set(h_pad=0.01, hspace=0.04)
    type_cmap = ListedColormap(list(CATEGORICAL[: len(groups)]) + [INK_2])
    type_cmap.set_bad(EMPTY)
    share_cmap = plt.get_cmap("Greys").copy()
    share_cmap.set_bad(EMPTY)
    strips = [("File type", majority, type_cmap, -0.5, len(groups) + 0.5)] + [(n, v, share_cmap, 0, 1) for n, v in rows]
    for ax, (name, values, cmap, lo, hi) in zip(axes, strips):
        ax.imshow(np.ma.masked_invalid(values)[None, :], aspect="auto", cmap=cmap, vmin=lo, vmax=hi, extent=extent,
                  interpolation="nearest")
        ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=6.5)
        ax.set_yticks([])
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
    for ax in axes[:-1]:
        ax.tick_params(axis="x", bottom=False)
    axes[-1].set_xlabel(f"Position in data area ({unit})")

    types = [Patch(color=CATEGORICAL[i], label=name) for i, (name, _) in enumerate(groups)]
    types += [Patch(color=INK_2, label="unknown / encrypted"), Patch(facecolor=EMPTY, edgecolor=AXIS, linewidth=0.4, label="free space")]
    fig.legend(handles=types, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=5, fontsize=6.5, handlelength=1.2,
               columnspacing=1.2, borderaxespad=0.1)
    shares = [Patch(facecolor=share_cmap(v), edgecolor=AXIS, linewidth=0.4, label=f"{v:.0%}") for v in (0.0, 0.5, 1.0)]
    fig.legend(handles=shares, loc="upper right", bbox_to_anchor=(1.0, 0.0), ncol=3, fontsize=6.5, handlelength=1.2,
               columnspacing=0.8, title=f"Share of evidence blocks per bin, {float(data['target']):.0%} target",
               title_fontsize=6.5, borderaxespad=0.1)
    return save(fig, ctx.out / "figures", "disk_map"), [str(path)]


FIGURES = (reliability, risk_coverage, per_class_f1, entropy, capacity, shap, disk_map)
