"""Command-line entry point. Run from the repository root: `fragtrust <command> --help`."""

from __future__ import annotations

import argparse
from pathlib import Path

from .runtime import setup_logging


def _env(args: argparse.Namespace) -> None:
    from .runtime import hardware_info, installed_packages, write_json

    setup_logging()
    info = hardware_info()
    write_json(args.results / "hardware.json", info)
    (args.results / "environment.txt").write_text("\n".join(installed_packages()) + "\n")
    gpus = ", ".join(g["name"] for g in info["gpus"]) or "none"
    print(f"{info['cpu']} | {info['cpus_available']} of {info['cpu_count']} cores available | {info['ram_gb']} GB RAM | GPU: {gpus}")
    print(f"written: {args.results / 'hardware.json'}, {args.results / 'environment.txt'}")


def _prepare(args: argparse.Namespace) -> None:
    from .config import load_config
    from .data.prepare import prepare

    cfg = load_config(args.config)
    setup_logging(args.results / "data" / cfg["name"] / "prepare.log")
    prepare(cfg, args.results, args.force)


def _synth(args: argparse.Namespace) -> None:
    from .config import load_config
    from .data.synthetic import make_synthetic

    setup_logging()
    print(f"synthetic dataset written to {make_synthetic(load_config(args.config))}")


def _tune(args: argparse.Namespace) -> None:
    from .config import load_config
    from .experiment import tune

    setup_logging()
    tune(load_config(args.config), args.results, args.force)


def _train(args: argparse.Namespace) -> None:
    from .config import load_config
    from .experiment import train

    setup_logging()
    cfg = load_config(args.config)
    for seed in args.seed:
        train(cfg, seed, args.results, args.force)


def _ensemble(args: argparse.Namespace) -> None:
    from .experiment import ensemble

    setup_logging()
    ensemble(args.runs, args.name, args.results, args.force)


def _analyze(args: argparse.Namespace) -> None:
    from .config import RunDir, done_runs
    from .trust.analyze import analyze

    setup_logging()
    runs = [RunDir(p) for p in args.runs] if args.runs else done_runs(args.results)
    for run in runs:
        if (run.path / "analysis" / "trust.json").exists() and not args.force:
            print(f"{run.id}: already analysed (use --force)")
            continue
        analyze(run.path)


def _summarize(args: argparse.Namespace) -> None:
    from .summarize import summarize

    setup_logging(args.results / "summary" / "summarize.log")
    summarize(args.results)
    print((args.results / "summary" / "by_model.md").read_text())


def _status(args: argparse.Namespace) -> None:
    import pandas as pd

    from .config import RunDir
    from .runtime import read_json

    rows = []
    for run in map(RunDir, sorted((args.results / "runs").glob("*"))):
        history = run.path / "history.csv"
        last = pd.read_csv(history).iloc[-1] if history.exists() else None
        metrics = run.path / "metrics.json"
        rows.append(
            {
                "run": run.id,
                "state": "done" if run.done else "running",
                "epoch": int(last["epoch"]) if last is not None else None,
                "val_a_acc": last["val_a_acc"] if last is not None else None,
                "epoch_s": last["epoch_seconds"] if last is not None else None,
                "test_acc": read_json(metrics)["accuracy"] if metrics.exists() else None,
                "analysed": (run.path / "analysis" / "trust.json").exists(),
            }
        )
    if not rows:
        print(f"no runs under {args.results / 'runs'}")
        return
    print(pd.DataFrame(rows).astype({"epoch": "Int64"}).to_string(index=False))


def _benchmark(args: argparse.Namespace) -> None:
    from .benchmark import benchmark
    from .runtime import available_cpus

    setup_logging()
    threads = sorted(set(args.threads or [1, available_cpus()]))
    for run in args.runs:
        benchmark(run, args.results, args.n_blocks, threads, args.backend, args.force)


def _explain(args: argparse.Namespace) -> None:
    from .explain import explain

    setup_logging()
    for run in args.runs:
        if (run / "analysis" / "shap.json").exists() and not args.force:
            print(f"{run.name}: SHAP already computed (use --force)")
            continue
        explain(run, args.per_class)


def _case_build(args: argparse.Namespace) -> None:
    from .case_study.build import build_image
    from .config import load_config
    from .data.prepare import load_class_names

    cfg = load_config(args.config)
    setup_logging(Path(cfg["out_dir"]) / "build.log")
    build_image(cfg, load_class_names(cfg["class_names"], cfg["n_classes"])[0])


def _case_simulate(args: argparse.Namespace) -> None:
    from .case_study.simulate import simulate_image

    setup_logging()
    print(f"simulated image written to {simulate_image(args.data, args.out, args.name, seed=args.seed)}")


def _case_scan(args: argparse.Namespace) -> None:
    from .case_study.scan import scan
    from .runtime import in_fresh_process

    setup_logging()
    for run in args.runs:
        in_fresh_process(scan, run, args.image, args.results, args.device, args.force)


def _case_eval(args: argparse.Namespace) -> None:
    from .case_study.evaluate import evaluate_case

    setup_logging()
    for run in args.runs:
        evaluate_case(run, args.image, args.results, args.cal_fraction, args.splits, args.seed)


def _report(args: argparse.Namespace) -> None:
    from .report import build_assets

    setup_logging()
    build_assets(args.config)


def _numbers(args: argparse.Namespace) -> None:
    from .report.numbers import check

    result = check(args.draft, args.root)
    out = args.out or args.draft.with_suffix(".rendered.md")
    out.write_text(result.rendered)
    print(f"{result.resolved} tags resolved -> {out}")
    for warning in result.untagged:
        print(f"untagged number, check by hand: {warning}")
    for error in result.errors:
        print(f"ERROR {error}")
    if result.errors:
        raise SystemExit(1)


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--results", type=Path, default=Path("results"), help="results root (default: results)")
    force = argparse.ArgumentParser(add_help=False)
    force.add_argument("--force", action="store_true", help="redo even if the output already exists")
    config = argparse.ArgumentParser(add_help=False)
    config.add_argument("--config", type=Path, required=True)

    parser = argparse.ArgumentParser(prog="fragtrust", description="Trustworthy file fragment classification experiments.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("env", parents=[common], help="record hardware and package versions").set_defaults(func=_env)
    sub.add_parser("prepare", parents=[common, config, force], help="build the processed FFT-75 subset").set_defaults(func=_prepare)
    sub.add_parser("synth", parents=[common, config], help="write a synthetic dataset for smoke tests").set_defaults(func=_synth)
    sub.add_parser("tune", parents=[common, config, force], help="LightGBM grid search on val_a").set_defaults(func=_tune)

    p = sub.add_parser("train", parents=[common, config, force], help="train one model per seed")
    p.add_argument("--seed", type=int, nargs="+", default=[0])
    p.set_defaults(func=_train)

    p = sub.add_parser("ensemble", parents=[common, force], help="average finished runs into a deep ensemble")
    p.add_argument("--name", required=True)
    p.add_argument("runs", type=Path, nargs="+")
    p.set_defaults(func=_ensemble)

    p = sub.add_parser("analyze", parents=[common, force], help="calibration, abstention, conformal sets, OOD")
    p.add_argument("runs", type=Path, nargs="*", help="run directories (default: every finished run)")
    p.set_defaults(func=_analyze)

    sub.add_parser("summarize", parents=[common], help="tables over all runs").set_defaults(func=_summarize)
    sub.add_parser("status", parents=[common], help="progress of every run").set_defaults(func=_status)

    p = sub.add_parser("benchmark", parents=[common, force], help="CPU inference throughput")
    p.add_argument("runs", type=Path, nargs="+")
    p.add_argument("--n-blocks", type=int, default=20000)
    p.add_argument("--threads", type=int, nargs="+", help="thread counts (default: 1 and every available CPU)")
    p.add_argument("--backend", choices=("onnx", "torch"), default="onnx", help="for networks; LightGBM always uses its own")
    p.set_defaults(func=_benchmark)

    p = sub.add_parser("explain", parents=[common, force], help="TreeSHAP attributions of a LightGBM run")
    p.add_argument("runs", type=Path, nargs="+")
    p.add_argument("--per-class", type=int, default=20)
    p.set_defaults(func=_explain)

    image = argparse.ArgumentParser(add_help=False)
    image.add_argument("--image", type=Path, required=True, help="image directory (image.img, ground_truth.csv, build.json)")
    sub.add_parser("case-build", parents=[config], help="build the formatted FAT32 evidence image (Linux)").set_defaults(func=_case_build)

    p = sub.add_parser("case-simulate", help="simulated image from a processed dataset, for smoke tests")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--name", default="simulated")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=_case_simulate)

    p = sub.add_parser("case-scan", parents=[common, image, force], help="classify every cluster of the image")
    p.add_argument("runs", type=Path, nargs="+")
    p.add_argument("--device", default="auto")
    p.set_defaults(func=_case_scan)

    p = sub.add_parser("case-eval", parents=[common, image], help="guarantees on the image, carver baseline, recalibration")
    p.add_argument("runs", type=Path, nargs="+")
    p.add_argument("--cal-fraction", type=float, default=0.2)
    p.add_argument("--splits", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=_case_eval)

    p = sub.add_parser("report", help="all paper figures and tables")
    p.add_argument("--config", type=Path, default=Path("configs/paper.yaml"))
    p.set_defaults(func=_report)

    p = sub.add_parser("numbers", help="resolve {{file:key|format}} tags in the draft and list untagged numbers")
    p.add_argument("draft", type=Path)
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--out", type=Path)
    p.set_defaults(func=_numbers)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
