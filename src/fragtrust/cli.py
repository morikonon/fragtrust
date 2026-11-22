"""Command-line entry point. Run from the repository root: `fragtrust <command> --help`."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .runtime import setup_logging


def _env(args: argparse.Namespace) -> None:
    from .runtime import hardware_info, installed_packages, write_json

    setup_logging()
    info = hardware_info()
    write_json(args.results / "hardware.json", info)
    (args.results / "environment.txt").write_text("\n".join(installed_packages()) + "\n")
    gpus = ", ".join(g["name"] for g in info["gpus"]) or "none"
    print(f"{info['cpu']} | {info['cpu_count']} cores | {info['ram_gb']} GB RAM | GPU: {gpus}")
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

    setup_logging()
    for run in args.runs:
        benchmark(run, args.results, args.n_blocks, args.threads)


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

    p = sub.add_parser("benchmark", parents=[common], help="CPU inference throughput")
    p.add_argument("runs", type=Path, nargs="+")
    p.add_argument("--n-blocks", type=int, default=20000)
    p.add_argument("--threads", type=int, nargs="+", default=[1, os.cpu_count() or 1])
    p.set_defaults(func=_benchmark)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
