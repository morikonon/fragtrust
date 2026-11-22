"""Steps that create run directories: tune (LightGBM grid), train, ensemble.

Every run lives in results/runs/<config name>_s<seed>/ and is idempotent: a finished run (DONE file)
is skipped, an unfinished one resumes from its last checkpoint.
"""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import log_softmax, logsumexp

from .config import RunDir, save_yaml
from .data.store import EVAL_SPLITS, FragmentStore, stratified_indices
from .runtime import cpu_threads, git_commit, hardware_info, read_json, seed_everything, setup_logging, utc_now, write_json
from .trust.metrics import classification_summary

log = logging.getLogger(__name__)


def _train_subset(cfg: dict[str, Any], y: np.ndarray) -> np.ndarray | None:
    """Optional smaller stratified training set, identical across seeds (learning curves, quick runs)."""
    per_class = cfg["data"].get("max_train_per_class")
    if per_class is None:
        return None
    return stratified_indices(y, per_class, np.random.default_rng(cfg["data"].get("subset_seed", 0)))


def _write_test_metrics(run: RunDir, data: FragmentStore) -> dict[str, float]:
    summary, per_class, confusion = classification_summary(data.y("test"), run.logits("test").argmax(axis=1), data.class_names)
    write_json(run.path / "metrics.json", summary)
    per_class.to_csv(run.path / "per_class_test.csv", index=False)
    np.save(run.path / "confusion_test.npy", confusion)
    return summary


def _open_run(cfg: dict[str, Any], seed: int, results: Path, force: bool) -> RunDir | None:
    run = RunDir(results / "runs" / f"{cfg['name']}_s{seed}")
    if run.done and not force:
        log.info("%s is done, skipping (use --force to retrain)", run.id)
        return None
    if force and run.path.exists():
        shutil.rmtree(run.path)
    if (run.path / "config.yaml").exists() and run.config() != cfg:
        raise RuntimeError(f"{run.path} holds an unfinished run with a different config; rerun with --force")
    run.path.mkdir(parents=True, exist_ok=True)
    save_yaml(cfg, run.path / "config.yaml")
    return run


def train(cfg: dict[str, Any], seed: int, results: Path, force: bool = False) -> None:
    run = _open_run(cfg, seed, results, force)
    if run is None:
        return
    setup_logging(run.path / "train.log")
    data = FragmentStore(cfg["data"]["dir"])
    commit = git_commit()
    log.info("run %s | dataset %s (%d classes) | git %s", run.id, data.name, data.n_classes, commit)
    run.update_meta(
        run_id=run.id,
        model=cfg["name"],
        seed=seed,
        dataset=data.name,
        started=run.meta().get("started", utc_now()),
        git_commit=commit,
        hardware=hardware_info(),
    )
    seed_everything(seed, cfg.get("deterministic", False))
    t0 = time.perf_counter()
    if cfg["model"]["type"] == "lightgbm":
        info = _train_lightgbm(cfg, seed, data, run, results)
    else:
        info = _train_net(cfg, seed, data, run)
    summary = _write_test_metrics(run, data)
    run.update_meta(finished=utc_now(), wall_seconds=round(time.perf_counter() - t0, 1), **info)
    run.mark_done()
    (run.path / "last.pt").unlink(missing_ok=True)
    log.info("%s done | test accuracy %.4f | macro-F1 %.4f", run.id, summary["accuracy"], summary["macro_f1"])


def _train_net(cfg: dict[str, Any], seed: int, data: FragmentStore, run: RunDir) -> dict[str, Any]:
    import torch

    from .models import build_net, count_params
    from .runtime import resolve_device
    from .training.batches import BlockBatches, fits_on_device
    from .training.trainer import TrainConfig, fit, predict_logits

    tc = TrainConfig.from_dict(cfg.get("train", {}))
    device = resolve_device(cfg.get("device", "auto"))
    model = build_net(cfg["model"], data.n_classes, data.block_size).to(device)
    n_params = count_params(model)
    log.info("%s on %s: %s parameters", cfg["model"]["type"], device, f"{n_params:,}")

    x_train, y_train = data.x("train"), data.y("train")
    subset = _train_subset(cfg, y_train)
    if subset is not None:
        x_train, y_train = x_train[subset], y_train[subset]
    x_val, y_val = data.x("val_a"), data.y("val_a")
    nbytes = x_train.nbytes + x_val.nbytes
    resident = fits_on_device(device, nbytes) if tc.resident == "auto" else bool(tc.resident)
    log.info(
        "train %s blocks | val_a %s blocks | %.1f GB kept on %s",
        f"{len(y_train):,}", f"{len(y_val):,}", nbytes / 2**30, device if resident else "host",
    )
    train_batches = BlockBatches(x_train, y_train, tc.batch_size, device, shuffle=True, seed=seed, resident=resident, drop_last=True)
    val_batches = BlockBatches(x_val, y_val, tc.eval_batch_size, device, resident=resident)
    state = fit(model, train_batches, val_batches, tc, device, run.path)
    del train_batches, val_batches
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model.load_state_dict(torch.load(run.path / "best.pt", map_location=device, weights_only=True))
    for split in EVAL_SPLITS:
        run.save_logits(split, predict_logits(model, data.x(split), device, tc.eval_batch_size))
    log.info("saved logits for %s", ", ".join(EVAL_SPLITS))
    return {
        "logits_kind": "logits",
        "params": n_params,
        "model_bytes": (run.path / "best.pt").stat().st_size,
        "train_blocks": len(y_train),
        **state,
    }


def _tuned_params(cfg: dict[str, Any], results: Path) -> dict[str, Any]:
    if "grid" not in cfg:
        return {}
    best = results / "tuning" / cfg["name"] / "best.json"
    if not best.exists():
        raise FileNotFoundError(f"{best} not found: run `fragtrust tune --config <this config>` first")
    return read_json(best)["params"]


def _train_lightgbm(cfg: dict[str, Any], seed: int, data: FragmentStore, run: RunDir, results: Path) -> dict[str, Any]:
    from .models import boosting

    threads = cpu_threads(cfg.get("threads"))
    params = {**boosting.base_params(data.n_classes, seed, threads), **cfg.get("params", {}), **_tuned_params(cfg, results)}
    log.info("LightGBM params: %s", params)
    x_train, y_train = data.features("train", threads), data.y("train")
    subset = _train_subset(cfg, y_train)
    if subset is not None:
        x_train, y_train = x_train[subset], y_train[subset]
    log.info("train %s blocks x %d features", f"{len(y_train):,}", x_train.shape[1])
    t0 = time.perf_counter()
    booster = boosting.fit(params, x_train, y_train, data.features("val_a", threads), data.y("val_a"), **cfg["boost"])
    train_seconds = time.perf_counter() - t0
    del x_train
    model_file = run.path / "model.txt"
    booster.save_model(str(model_file), num_iteration=booster.best_iteration)
    for split in EVAL_SPLITS:
        run.save_logits(split, boosting.log_proba(booster, data.features(split, threads)))
    log.info("best iteration %d, %d trees, %.1f min", booster.best_iteration, booster.num_trees(), train_seconds / 60)
    return {
        "logits_kind": "log_probs",
        "best_iteration": booster.best_iteration,
        "model_bytes": model_file.stat().st_size,
        "train_blocks": len(y_train),
        "train_seconds": round(train_seconds, 1),
        "lightgbm_params": params,
    }


def tune(cfg: dict[str, Any], results: Path, force: bool = False) -> None:
    """LightGBM grid on a stratified training subsample, selected by val_a log-loss."""
    from .models import boosting

    out = results / "tuning" / cfg["name"]
    if (out / "best.json").exists() and not force:
        log.info("%s already tuned, skipping (use --force)", cfg["name"])
        return
    setup_logging(out / "tune.log")
    data = FragmentStore(cfg["data"]["dir"])
    threads = cpu_threads(cfg.get("threads"))
    per_class = cfg["tune"]["train_per_class"]
    y_all = data.y("train")
    idx = stratified_indices(y_all, per_class, np.random.default_rng(cfg["tune"].get("seed", 0)))
    x_train, y_train = data.features("train", threads)[idx], y_all[idx]
    x_val, y_val = data.features("val_a", threads), data.y("val_a")
    rows = []
    for point in boosting.grid_points(cfg["grid"]):
        params = {**boosting.base_params(data.n_classes, 0, threads), **cfg.get("params", {}), **point}
        log.info("grid point %s", point)
        t0 = time.perf_counter()
        booster = boosting.fit(params, x_train, y_train, x_val, y_val, **cfg["boost"])
        logp = boosting.log_proba(booster, x_val)
        rows.append(
            {
                **point,
                "best_iteration": booster.best_iteration,
                "val_a_logloss": float(-logp[np.arange(len(y_val)), y_val].mean()),
                "val_a_acc": float((logp.argmax(axis=1) == y_val).mean()),
                "seconds": round(time.perf_counter() - t0, 1),
            }
        )
        log.info("-> %s", rows[-1])
        pd.DataFrame(rows).to_csv(out / "grid.csv", index=False)
    best = min(rows, key=lambda r: r["val_a_logloss"])
    write_json(
        out / "best.json",
        {"params": {k: best[k] for k in cfg["grid"]}, "selected_by": "val_a_logloss", "train_per_class": per_class, "result": best},
    )
    log.info("best grid point: %s", best)


def ensemble(run_paths: list[Path], name: str, results: Path, force: bool = False) -> None:
    """Deep ensemble: average of member probabilities, stored as log-probabilities."""
    members = [RunDir(p) for p in run_paths]
    unfinished = [m.id for m in members if not m.done]
    if unfinished:
        raise RuntimeError(f"members not finished: {unfinished}")
    data_dirs = {m.config()["data"]["dir"] for m in members}
    if len(data_dirs) != 1:
        raise RuntimeError(f"members use different datasets: {sorted(data_dirs)}")
    out = RunDir(results / "runs" / name)
    if out.done and not force:
        log.info("%s is done, skipping (use --force)", name)
        return
    out.path.mkdir(parents=True, exist_ok=True)
    setup_logging(out.path / "train.log")
    cfg = {**members[0].config(), "name": name, "members": [m.id for m in members]}
    save_yaml(cfg, out.path / "config.yaml")
    for split in EVAL_SPLITS:
        stacked = np.stack([log_softmax(m.logits(split), axis=1) for m in members])
        out.save_logits(split, logsumexp(stacked, axis=0) - np.log(len(members)))
    summary = _write_test_metrics(out, FragmentStore(cfg["data"]["dir"]))
    metas = [m.meta() for m in members]
    out.update_meta(
        run_id=name,
        model=name,
        members=cfg["members"],
        logits_kind="log_probs",
        params=sum(m.get("params") or 0 for m in metas) or None,
        model_bytes=sum(m.get("model_bytes") or 0 for m in metas),
        train_seconds=sum(m.get("train_seconds") or 0 for m in metas),
        finished=utc_now(),
        git_commit=git_commit(),
    )
    out.mark_done()
    log.info("%s (%d members) | test accuracy %.4f | macro-F1 %.4f", name, len(members), summary["accuracy"], summary["macro_f1"])
