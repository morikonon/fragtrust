"""Paper tables (markdown + CSV), mean ± std over seeds, built only from files under results/."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd

from ..runtime import read_json
from .collect import lookup, mean_std
from .figures import Context
from .numbers import rounded

Table = tuple[pd.DataFrame, list[str]]

PCT, DEC3, DEC2, INT, MB = "pct", "dec3", "dec2", "int", "mb"


def fmt(mean: float | None, std: float | None = None, kind: str = PCT) -> str:
    if mean is None:
        return "–"
    if kind == INT:
        return f"{mean:,.0f}"
    scale, digits = {PCT: (100, 2), DEC3: (1, 3), DEC2: (1, 2), MB: (1e-6, 2)}[kind]
    text = rounded(mean, digits, scale)
    return text if std is None else f"{text} ± {rounded(std, digits, scale)}"


def to_markdown(df: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def _models(ctx: Context) -> dict[str, str]:
    models = {m: label for m, label in ctx.models.items() if ctx.results.runs(m)}
    models |= {m: label for m, label in ctx.cfg.get("capacity", {}).items() if ctx.results.runs(m) and m not in models}
    return models


def _trust_sources(ctx: Context, models) -> list[str]:
    return [str(r.path / "analysis" / "trust.json") for m in models for r in ctx.results.runs(m)]


def _cell(ctx: Context, model: str, source: str, path: tuple[str, ...], kind: str = PCT) -> str:
    return fmt(*ctx.results.stat(model, source, path), kind=kind)


def dataset(ctx: Context) -> Table | None:
    base = ctx.results.root / "data" / str(ctx.cfg.get("dataset"))
    if not (base / "manifest.json").exists():
        return None
    manifest = read_json(base / "manifest.json")
    balance = pd.read_csv(base / "class_balance.csv")
    use = {"train": "fitting", "val_a": "early stopping, grid, temperature", "val_b": "thresholds, conformal quantiles", "test": "all reported numbers"}
    rows = [{"Split": s, "Blocks": f"{manifest['splits'][s]['rows']:,}", "Per class": f"{balance[s].min():,}–{balance[s].max():,}", "Used for": use[s]} for s in use]
    rows += [{"Split": name, "Blocks": f"{info['rows']:,}", "Per class": "–", "Used for": info["description"]} for name, info in manifest["probes"].items()]
    return pd.DataFrame(rows), [str(base / "manifest.json"), str(base / "class_balance.csv")]


def leakage(ctx: Context) -> Table | None:
    path = ctx.results.root / "data" / str(ctx.cfg.get("dataset")) / "leakage.json"
    if not path.exists():
        return None
    report = read_json(path)
    rows = []
    for name, label in (("official_train_vs_test", "official train vs test"), ("subset_train_vs_test", "our train subset vs test")):
        r = report[name]
        rows.append({
            "Comparison": label,
            "Test blocks": f"{r['n_test']:,}",
            "Also in train": f"{r['test_in_train']:,} ({fmt(r['test_in_train_frac'])} %)",
            "Same label": f"{r['test_in_train_same_label']:,}",
            "Other label only": f"{r['test_in_train_other_label_only']:,}",
            "Constant blocks": f"{r['test_in_train_constant_blocks']:,}",
        })
    return pd.DataFrame(rows), [str(path)]


def main_results(ctx: Context) -> Table | None:
    models = _models(ctx)
    if not models:
        return None
    rows = []
    for model, label in models.items():
        bench = ctx.results.benchmark(ctx.results.runs(model)[0].id, ctx.cfg.get("benchmark_host"))
        speeds = sorted(bench["results"], key=lambda r: r["threads"]) if bench else []
        rows.append({
            "Model": label,
            "Seeds": len(ctx.results.runs(model)),
            "Accuracy (%)": _cell(ctx, model, "metrics", ("accuracy",)),
            "Macro-F1 (%)": _cell(ctx, model, "metrics", ("macro_f1",)),
            "Params": _cell(ctx, model, "meta", ("params",), INT),
            "Size (MB)": fmt(mean_std(ctx.results.values(model, "meta", ("model_bytes",)))[0], kind=MB),
            "Train (min)": fmt(mean_std([v / 60 if v else None for v in ctx.results.values(model, "meta", ("train_seconds",))])[0], kind=DEC2),
            "CPU MB/s, 1 thread": f"{speeds[0]['mb_per_s']:.1f}" if speeds else "–",
            "CPU MB/s, all threads": f"{speeds[-1]['mb_per_s']:.1f} ({speeds[-1]['threads']})" if speeds else "–",
        })
    sources = [str(r.path / f) for m in models for r in ctx.results.runs(m) for f in ("metrics.json", "meta.json")]
    return pd.DataFrame(rows), sources


def calibration(ctx: Context) -> Table | None:
    models = {m: l for m, l in ctx.models.items() if ctx.results.first(m)}
    if not models:
        return None
    rows = []
    for model, label in models.items():
        c = ("calibration",)
        rows.append({
            "Model": label,
            "T": _cell(ctx, model, "trust", ("temperature",), DEC2),
            "ECE": _cell(ctx, model, "trust", c + ("uncalibrated", "ece"), DEC3),
            "ECE (TS)": _cell(ctx, model, "trust", c + ("temperature_scaled", "ece"), DEC3),
            "Adaptive ECE (TS)": _cell(ctx, model, "trust", c + ("temperature_scaled", "ece_adaptive"), DEC3),
            "NLL": _cell(ctx, model, "trust", c + ("uncalibrated", "nll"), DEC3),
            "NLL (TS)": _cell(ctx, model, "trust", c + ("temperature_scaled", "nll"), DEC3),
            "Brier (TS)": _cell(ctx, model, "trust", c + ("temperature_scaled", "brier"), DEC3),
        })
    return pd.DataFrame(rows), _trust_sources(ctx, models)


def selective(ctx: Context) -> Table | None:
    models = {m: l for m, l in ctx.models.items() if ctx.results.first(m)}
    if not models:
        return None
    rows = []
    for model, label in models.items():
        row = {"Model": label, "AURC": _cell(ctx, model, "trust", ("selective", "msp", "aurc"), DEC3)}
        for target in ("0.95", "0.99"):
            base = ("selective", "msp", "targets", target)
            met = ctx.results.values(model, "trust", base + ("ltt", "meets_target"))
            pct = target[2:]
            row |= {
                f"Coverage {pct}% (LTT)": _cell(ctx, model, "trust", base + ("ltt", "coverage")),
                f"Accuracy {pct}% (LTT)": _cell(ctx, model, "trust", base + ("ltt", "selective_accuracy")),
                f"Met {pct}%": f"{sum(bool(v) for v in met)}/{len(met)}",
                f"Coverage {pct}% (empirical)": _cell(ctx, model, "trust", base + ("empirical", "coverage")),
            }
        rows.append(row)
    return pd.DataFrame(rows), _trust_sources(ctx, models)


def conformal(ctx: Context) -> Table | None:
    models = {m: l for m, l in ctx.models.items() if ctx.results.first(m)}
    if not models:
        return None
    first = ctx.results.doc(ctx.results.first(next(iter(models))), "trust")["conformal"]
    methods = sorted({name.split("@")[0] for name in first}, key=["lac", "aps", "raps"].index)
    levels = sorted({name.split("@")[1] for name in first})
    rows = []
    for model, label in models.items():
        for method in methods:
            row = {"Model": label, "Method": method.upper()}
            for level in levels:
                path = ("conformal", f"{method}@{level}")
                pct = f"{float(level):.0%}"
                row |= {
                    f"Coverage {pct}": _cell(ctx, model, "trust", path + ("coverage",)),
                    f"Set size {pct}": _cell(ctx, model, "trust", path + ("mean_size",), DEC2),
                    f"Worst class {pct}": _cell(ctx, model, "trust", path + ("worst_class_coverage",)),
                }
            rows.append(row)
    return pd.DataFrame(rows), _trust_sources(ctx, models)


def ood(ctx: Context) -> Table | None:
    models = {m: l for m, l in ctx.models.items() if ctx.results.first(m)}
    if not models:
        return None
    names = {"ood_aes_ctr": "AES-CTR", "ood_random": "random"}
    rows = []
    for model, label in models.items():
        for probe, probe_label in names.items():
            base = ("ood", probe)
            top = lookup(ctx.results.doc(ctx.results.first(model), "trust"), base + ("top_predicted",)) or {}
            top_class = next(iter(top), None)
            rows.append({
                "Model": label,
                "Probe": probe_label,
                "AUROC (MSP)": _cell(ctx, model, "trust", base + ("auroc", "msp"), DEC3),
                "AUROC (entropy)": _cell(ctx, model, "trust", base + ("auroc", "entropy"), DEC3),
                "AUROC (energy)": _cell(ctx, model, "trust", base + ("auroc", "energy"), DEC3),
                "Abstained at 99% (%)": _cell(ctx, model, "trust", base + ("abstained", "0.99")),
                "APS 95% set size": _cell(ctx, model, "trust", base + ("aps@0.95_sets", "mean_size"), DEC2),
                "Most frequent prediction": f"{top_class} ({fmt(top[top_class])} %)" if top_class else "–",
            })
    return pd.DataFrame(rows), _trust_sources(ctx, models)


def _case_reports(ctx: Context) -> list[tuple[str, dict, str]]:
    spec = ctx.cfg.get("case_study") or {}
    base = ctx.results.root / "case_study" / str(spec.get("image"))
    out = []
    for run_id, label in (spec.get("runs") or {}).items():
        path = base / run_id / "case_study.json"
        if path.exists():
            out.append((label, read_json(path), str(path)))
    return out


def case_study(ctx: Context) -> Table | None:
    reports = _case_reports(ctx)
    if not reports:
        return None
    rows = []
    for label, r, _ in reports:
        t = r["transferred"]
        rows.append({
            "Method": label,
            "Accuracy, all blocks (%)": fmt(t["all"]["accuracy"]),
            "Answered (%)": fmt(t["all"]["coverage@0.99"]),
            "Accuracy of answers (%)": fmt(t["all"]["selective_accuracy@0.99"]),
            "Answered, middle blocks (%)": fmt(lookup(t, ("middle", "coverage@0.99"))),
            "Accuracy, fragmented files (%)": fmt(lookup(t, ("fragmented", "accuracy"))),
            "APS 95% coverage (%)": fmt(t["all"]["aps@0.95_coverage"]),
            "APS 95% set size": fmt(t["all"]["aps@0.95_size"], kind=DEC2),
        })
    sig = reports[0][1]["signatures"]
    for variant, name in (("header", "Signatures, header only"), ("propagated", "Signatures + contiguous carving")):
        s = sig[variant]
        rows.append({
            "Method": name,
            "Accuracy, all blocks (%)": fmt(s["all"]["correct_overall"]),
            "Answered (%)": fmt(s["all"]["coverage"]),
            "Accuracy of answers (%)": fmt(s["all"]["accuracy_answered"]),
            "Answered, middle blocks (%)": fmt(lookup(s, ("middle", "coverage"))),
            "Accuracy, fragmented files (%)": fmt(lookup(s, ("fragmented", "correct_overall"))),
            "APS 95% coverage (%)": "–",
            "APS 95% set size": "–",
        })
    return pd.DataFrame(rows), [src for _, _, src in reports]


def _split_stat(summary: dict, key: str, kind: str = PCT) -> str:
    stat = summary[key]
    return fmt(stat["mean"], stat["std"], kind)


def recalibration(ctx: Context) -> Table | None:
    reports = [(label, r, src) for label, r, src in _case_reports(ctx) if "transferred" in r["recalibration"]]
    if not reports:
        return None
    rows = []
    for label, r, _ in reports:
        rec = r["recalibration"]
        for params in ("transferred", "recalibrated"):
            m = rec[params]
            rows.append({
                "Model": label,
                "Fitted on": "FFT-75 (transferred)" if params == "transferred" else f"{rec['cal_fraction']:.0%} of image files",
                "T": _split_stat(m, "temperature", DEC2),
                "Answered at 99% (%)": _split_stat(m, "coverage@0.99"),
                "Accuracy of answers (%)": _split_stat(m, "selective_accuracy@0.99"),
                "Splits meeting 99% (%)": fmt(m["meets@0.99"]["mean"]),
                "APS 95% coverage (%)": _split_stat(m, "aps@0.95_coverage"),
                "APS 95% set size": _split_stat(m, "aps@0.95_size", DEC2),
            })
    return pd.DataFrame(rows), [src for _, _, src in reports]


TABLES: dict[str, Callable[[Context], Table | None]] = {
    "dataset": dataset,
    "leakage": leakage,
    "main_results": main_results,
    "calibration": calibration,
    "selective": selective,
    "conformal": conformal,
    "ood": ood,
    "case_study": case_study,
    "recalibration": recalibration,
}


def write_table(df: pd.DataFrame, out_dir: Path, name: str) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{name}.csv", index=False)
    (out_dir / f"{name}.md").write_text(to_markdown(df))
    return [str(out_dir / f"{name}.md"), str(out_dir / f"{name}.csv")]
