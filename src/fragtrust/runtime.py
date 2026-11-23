"""Logging, seeding, device selection and environment capture shared by all commands."""

from __future__ import annotations

import json
import logging
import os
import platform
import random
import socket
import subprocess
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
T = TypeVar("T")
KEY_PACKAGES = (
    "numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "torch", "cryptography", "pyyaml",
    "matplotlib", "onnx", "onnxruntime",
)


def setup_logging(log_file: Path | None = None) -> logging.Logger:
    """Route all `fragtrust.*` loggers to stdout and, if given, append them to `log_file`."""
    logger = logging.getLogger("fragtrust")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    formatter = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S")
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def _logged_call(log_file: Path | None, fn: Callable[..., T], args: tuple) -> T:
    setup_logging(log_file)
    return fn(*args)


def in_fresh_process(fn: Callable[..., T], *args: Any, log_file: Path | None = None) -> T:
    """fn(*args) in a new interpreter, logging to stdout and `log_file` there too. LightGBM and PyTorch
    each bring an OpenMP runtime, and using both in one process can deadlock (seen on macOS); a fresh
    process also makes the peak memory that of this call alone."""
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context

    with ProcessPoolExecutor(1, mp_context=get_context("spawn")) as pool:
        return pool.submit(_logged_call, log_file, fn, args).result()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _to_builtin(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=_to_builtin) + "\n")
    tmp.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def seed_everything(seed: int, deterministic: bool = False) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = not deterministic
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(name: str = "auto"):
    import torch

    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cgroup_cpu_quota(root: Path = Path("/sys/fs/cgroup")) -> float | None:
    """CPU limit of the container in CPUs (cgroup v2 or v1), None when unlimited or not in a cgroup."""
    v2 = root / "cpu.max"
    v1 = root / "cpu" / "cpu.cfs_quota_us", root / "cpu" / "cpu.cfs_period_us"
    try:
        if v2.exists():
            quota, period = v2.read_text().split()[:2]
            return None if quota == "max" else int(quota) / int(period)
        if v1[0].exists():
            quota, period = int(v1[0].read_text()), int(v1[1].read_text())
            return quota / period if quota > 0 else None
    except (OSError, ValueError):
        pass
    return None


def available_cpus() -> int:
    """CPUs this process may use. Inside a container (RunPod) os.cpu_count() reports every core of the
    host, so the affinity mask is capped by the cgroup CPU quota; oversubscribed threads run slower."""
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:
        n = os.cpu_count() or 1
    quota = cgroup_cpu_quota()
    return max(1, min(n, int(quota))) if quota else n


def cpu_threads(requested: int | None = None) -> int:
    return requested or max(1, available_cpus() - 1)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def git_commit() -> str | None:
    """HEAD of this project's own repository (suffix -dirty if tracked files changed), else None."""
    try:
        if Path(_git("rev-parse", "--show-toplevel")).resolve() != PROJECT_ROOT:
            return None
        commit = _git("rev-parse", "HEAD")
        dirty = _git("status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.CalledProcessError):
        return None
    return f"{commit}-dirty" if dirty else commit


def _cpu_model() -> str:
    if sys.platform == "darwin":
        try:
            return subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=True
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def key_versions() -> dict[str, str | None]:
    versions = {}
    for name in KEY_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def installed_packages() -> list[str]:
    return sorted({f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions()}, key=str.lower)


def hardware_info() -> dict[str, Any]:
    import torch

    info: dict[str, Any] = {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "cpus_available": available_cpus(),
        "ram_gb": round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1),
        "cuda": torch.version.cuda,
        "mps": torch.backends.mps.is_available(),
        "gpus": [],
        "packages": key_versions(),
    }
    if torch.cuda.is_available():
        props = [torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())]
        info["gpus"] = [{"name": p.name, "memory_gb": round(p.total_memory / 2**30, 1)} for p in props]
        info["cudnn"] = torch.backends.cudnn.version()
    return info
