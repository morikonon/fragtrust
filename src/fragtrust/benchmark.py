"""CPU inference throughput of a trained run ("can an examiner scan a drive on a laptop?").

Backends: `onnx` (ONNX Runtime, the deployment runtime; needs the `bench` extra), `torch` (PyTorch
eager, depends on how PyTorch was built for the machine), and LightGBM for the boosting model.
Feature extraction is included in the LightGBM time. Results go to
results/benchmarks/<host>/<run>.<backend>.json.
"""

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
from .inference import load_booster, load_net
from .runtime import hardware_info, setup_logging, write_json

log = logging.getLogger(__name__)

Runner = Callable[[np.ndarray], np.ndarray]
BATCH = 256


def export_onnx(run: RunDir, block_size: int) -> Path:
    import torch

    path = run.path / "model.onnx"
    if not path.exists():
        torch.onnx.export(
            load_net(run),
            torch.zeros(1, block_size, dtype=torch.uint8),
            str(path),
            input_names=["blocks"],
            output_names=["logits"],
            dynamic_axes={"blocks": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=17,
        )
    return path


def _torch_runner(run: RunDir, block_size: int) -> Callable[[int], Runner]:
    import torch

    from .training.trainer import predict_logits

    model, cpu = load_net(run), torch.device("cpu")

    def make(threads: int) -> Runner:
        torch.set_num_threads(threads)
        return lambda x: predict_logits(model, x, cpu, BATCH)

    return make


def _onnx_runner(run: RunDir, block_size: int) -> Callable[[int], Runner]:
    import onnxruntime as ort

    path = export_onnx(run, block_size)

    def make(threads: int) -> Runner:
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        return lambda x: np.concatenate([session.run(None, {"blocks": x[a : a + BATCH]})[0] for a in range(0, len(x), BATCH)])

    return make


def _lightgbm_runner(run: RunDir, block_size: int) -> Callable[[int], Runner]:
    from .features import extract

    booster = load_booster(run)
    return lambda threads: (lambda x: booster.predict(extract(x), num_threads=threads))


def _peak_rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**20 if sys.platform == "darwin" else peak / 2**10


def benchmark(run_path: Path, results: Path, n_blocks: int, threads: list[int], backend: str = "onnx") -> dict[str, Any]:
    run = RunDir(run_path)
    cfg = run.config()
    if "members" in cfg:
        raise ValueError("benchmark single models, not ensembles")
    if cfg["model"]["type"] == "lightgbm":
        backend = "lightgbm"
    out = results / "benchmarks" / socket.gethostname()
    setup_logging(out / "benchmark.log")
    data = FragmentStore(cfg["data"]["dir"])
    x = np.array(data.x("test")[:n_blocks])
    factory = {"torch": _torch_runner, "onnx": _onnx_runner, "lightgbm": _lightgbm_runner}[backend](run, data.block_size)

    reference_gap = None
    if backend == "onnx":
        reference = _torch_runner(run, data.block_size)(threads[0])(x[:BATCH])
        reference_gap = float(np.abs(factory(threads[0])(x[:BATCH]) - reference).max())
        log.info("ONNX vs PyTorch logits: max abs difference %.2e", reference_gap)

    rows = []
    for t in threads:
        predict = factory(t)
        predict(x[:BATCH])
        t0 = time.perf_counter()
        predict(x)
        seconds = time.perf_counter() - t0
        bytes_per_s = len(x) * data.block_size / seconds
        rows.append({
            "threads": t,
            "blocks_per_s": round(len(x) / seconds, 1),
            "mb_per_s": round(bytes_per_s / 1e6, 2),
            "us_per_block": round(1e6 * seconds / len(x), 1),
            "minutes_per_32gb": round(32e9 / bytes_per_s / 60, 1),
            "minutes_per_256gb": round(256e9 / bytes_per_s / 60, 1),
        })
        log.info("%s [%s] | %d threads | %.1f MB/s | 32 GB in %.1f min", run.id, backend, t, rows[-1]["mb_per_s"], rows[-1]["minutes_per_32gb"])
    result = {
        "run": run.id,
        "backend": backend,
        "n_blocks": len(x),
        "block_size": data.block_size,
        "batch_size": BATCH,
        "onnx_vs_torch_max_abs_diff": reference_gap,
        "results": rows,
        "peak_rss_mb": round(_peak_rss_mb(), 1),
        "hardware": hardware_info(),
    }
    write_json(out / f"{run.id}.{backend}.json", result)
    return result
