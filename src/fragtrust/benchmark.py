"""CPU inference throughput of a trained run ("can an examiner scan a drive on a laptop?")."""

from __future__ import annotations

import logging
import resource
import socket
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from .config import RunDir
from .data.store import FragmentStore
from .runtime import hardware_info, setup_logging, write_json

log = logging.getLogger(__name__)

Predictor = Callable[[np.ndarray, int], np.ndarray]


def _lightgbm_predictor(run: RunDir) -> Predictor:
    import lightgbm as lgb

    from .features import extract

    booster = lgb.Booster(model_file=str(run.path / "model.txt"))
    return lambda x, threads: booster.predict(extract(x), num_threads=threads)


def _net_predictor(run: RunDir, data: FragmentStore) -> Predictor:
    import torch

    from .models import build_net
    from .training.trainer import predict_logits

    model = build_net(run.config()["model"], data.n_classes, data.block_size)
    model.load_state_dict(torch.load(run.path / "best.pt", map_location="cpu", weights_only=True))
    cpu = torch.device("cpu")

    def predict(x: np.ndarray, threads: int) -> np.ndarray:
        torch.set_num_threads(threads)
        return predict_logits(model, x, cpu, batch_size=256)

    return predict


def _peak_rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**20 if sys.platform == "darwin" else peak / 2**10


def benchmark(run_path: Path, results: Path, n_blocks: int, threads: list[int]) -> dict[str, Any]:
    run = RunDir(run_path)
    cfg = run.config()
    if "members" in cfg:
        raise ValueError("benchmark single models, not ensembles")
    out = results / "benchmarks" / socket.gethostname()
    setup_logging(out / "benchmark.log")
    data = FragmentStore(cfg["data"]["dir"])
    x = np.array(data.x("test")[:n_blocks])
    predict = _lightgbm_predictor(run) if cfg["model"]["type"] == "lightgbm" else _net_predictor(run, data)
    predict(x[:256], threads[0])

    rows = []
    for t in threads:
        t0 = time.perf_counter()
        predict(x, t)
        seconds = time.perf_counter() - t0
        bytes_per_s = len(x) * data.block_size / seconds
        rows.append(
            {
                "threads": t,
                "blocks_per_s": round(len(x) / seconds, 1),
                "mb_per_s": round(bytes_per_s / 1e6, 2),
                "us_per_block": round(1e6 * seconds / len(x), 1),
                "minutes_per_32gb": round(32e9 / bytes_per_s / 60, 1),
                "minutes_per_256gb": round(256e9 / bytes_per_s / 60, 1),
            }
        )
        log.info("%s | %d threads | %.1f MB/s | 32 GB in %.1f min", run.id, t, rows[-1]["mb_per_s"], rows[-1]["minutes_per_32gb"])
    result = {
        "run": run.id,
        "n_blocks": len(x),
        "block_size": data.block_size,
        "note": "LightGBM time includes single-process feature extraction" if cfg["model"]["type"] == "lightgbm" else None,
        "results": rows,
        "peak_rss_mb": round(_peak_rss_mb(), 1),
        "hardware": hardware_info(),
    }
    write_json(out / f"{run.id}.json", result)
    return result
