"""Block-level ground truth: data cluster -> evidence file -> FFT-75 type."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

from .fat32 import Fat32, FatFile

EXTENSION_ALIASES = {"jpeg": "jpg", "jpe": "jpg", "tif": "tiff", "htm": "html", "markdown": "md", "tgz": "gz", "text": "txt"}
NOT_EVIDENCE = "not_evidence"
UNKNOWN = "unknown"


def read_overrides(files_dir: Path) -> dict[str, str]:
    """Optional labels.csv (columns: path,label) for files whose extension does not tell the type,
    e.g. an encrypted container labelled 'encrypted'."""
    path = files_dir / "labels.csv"
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = [[cell.strip() for cell in row] for row in csv.reader(f) if any(cell.strip() for cell in row)]
    if not rows or rows[0] != ["path", "label"]:
        raise ValueError(f"{path}: the first line must be exactly 'path,label' (comma-separated)")
    bad = [i for i, row in enumerate(rows[1:], 2) if len(row) != 2 or not all(row)]
    if bad:
        raise ValueError(f"{path}: lines {bad} must hold a path and a label separated by one comma")
    return dict(rows[1:])


def label_for(relative_path: str, class_names: set[str], overrides: dict[str, str]) -> str:
    if relative_path in overrides:
        return overrides[relative_path]
    ext = Path(relative_path).suffix.lower().lstrip(".")
    ext = EXTENSION_ALIASES.get(ext, ext)
    return ext if ext in class_names else UNKNOWN


def cluster_table(fs: Fat32, evidence: dict[str, tuple[int, str]]) -> pd.DataFrame:
    """One row per data cluster. `evidence` maps image path -> (file_id, label); other clusters are
    labelled not_evidence (free space, deleted or live filler files)."""
    g = fs.geometry
    n = g.n_clusters
    file_id = np.full(n, -1, dtype=np.int32)
    label = np.full(n, NOT_EVIDENCE, dtype=object)
    index_in_file = np.full(n, -1, dtype=np.int32)
    position = np.full(n, "", dtype=object)
    used_bytes = np.zeros(n, dtype=np.int32)
    fragmented = np.zeros(n, dtype=bool)
    for f in fs.files():
        if f.path not in evidence or not f.clusters:
            continue
        fid, lab = evidence[f.path]
        idx = np.asarray(f.clusters) - 2
        k = np.arange(len(idx))
        file_id[idx] = fid
        label[idx] = lab
        index_in_file[idx] = k
        position[idx] = _positions(len(idx))
        used_bytes[idx] = np.minimum(g.cluster_size, f.size - k * g.cluster_size)
        fragmented[idx] = f.fragments > 1
    return pd.DataFrame(
        {
            "cluster": np.arange(2, n + 2),
            "offset": g.data_offset + np.arange(n, dtype=np.int64) * g.cluster_size,
            "file_id": file_id,
            "label": label,
            "index_in_file": index_in_file,
            "position": position,
            "used_bytes": used_bytes,
            "fragmented": fragmented,
        }
    )


def _positions(n: int) -> np.ndarray:
    if n == 1:
        return np.array(["single"], dtype=object)
    out = np.full(n, "middle", dtype=object)
    out[0], out[-1] = "header", "tail"
    return out


def fragmentation_stats(files: list[FatFile]) -> dict[str, float]:
    fragments = np.array([f.fragments for f in files if f.clusters])
    return {
        "files": int(len(fragments)),
        "fragmented_files": int((fragments > 1).sum()),
        "fragments_mean": float(fragments.mean()) if len(fragments) else 0.0,
        "fragments_max": int(fragments.max()) if len(fragments) else 0,
    }
