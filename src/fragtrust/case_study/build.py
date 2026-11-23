"""Build a quick-formatted FAT32 evidence image with known block-level ground truth.

Linux tools only (dosfstools + mtools, no root, no mounting):
  1. mkfs.fat creates an empty FAT32 image with 4096-byte clusters;
  2. zero-filled filler files fill the volume except a small tail;
  3. every second filler (in allocation order) is deleted, leaving holes;
  4. the evidence files are copied in random order, so large files are split across holes;
  5. ground truth is read from the file system and verified by reassembling every file (sha256);
  6. mkfs.fat runs again on the same image (quick format) and the evidence clusters are compared
     with the pre-format copy to show what survived.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..runtime import read_json, utc_now, write_json
from .fat32 import Fat32
from .ground_truth import cluster_table, fragmentation_stats, label_for, read_overrides

log = logging.getLogger(__name__)

EXCLUDED = {"SOURCES.md", "labels.csv"}
BATCH = 200


def _tool(*names: str) -> str:
    for name in names:
        if path := shutil.which(name):
            return path
    raise RuntimeError(f"{' / '.join(names)} not found: install dosfstools and mtools (apt-get install dosfstools mtools)")


def _run(cmd: list[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "MTOOLS_SKIP_CHECK": "1"})
    if result.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd[:3])} ... failed: {result.stderr.strip() or result.stdout.strip()}")
    return result.stdout + result.stderr


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            digest.update(chunk)
    return digest.hexdigest()


def _mkfs(mkfs: str, image: Path, cfg: dict[str, Any], create: bool) -> str:
    cmd = [mkfs, "-F", "32", "-S", "512", "-s", str(cfg["cluster_size"] // 512), "-n", cfg["volume_label"]]
    cmd += ["-C", str(image), str(cfg["size_mb"] * 1024)] if create else [str(image)]
    return _run(cmd)


def _copy_batches(mcopy: str, image: Path, sources: list[Path], target_dir: str) -> None:
    for a in range(0, len(sources), BATCH):
        _run([mcopy, "-i", str(image), "-o", *map(str, sources[a : a + BATCH]), f"::/{target_dir}/"])


def _write_fillers(tools: dict[str, str], image: Path, cfg: dict[str, Any], rng: np.random.Generator) -> int:
    cluster = cfg["cluster_size"]
    budget = Fat32(image).free_clusters() * cluster - cfg["tail_mb"] * 2**20
    low, high = (kb * 1024 // cluster for kb in cfg["filler_kb"])
    sizes = []
    while budget > high * cluster:
        sizes.append(int(rng.integers(low, high + 1)) * cluster)
        budget -= sizes[-1] + cluster
    _run([tools["mmd"], "-i", str(image), "::/FILL"])
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for i, size in enumerate(sizes):
            path = Path(tmp) / f"F{i:05d}.BIN"
            with open(path, "wb") as f:
                f.truncate(size)
            paths.append(path)
        _copy_batches(tools["mcopy"], image, paths, "FILL")
    return len(sizes)


def _delete_alternate_fillers(mdel: str, image: Path) -> int:
    fillers = sorted((f for f in Fat32(image).files() if f.path.startswith("FILL/")), key=lambda f: f.clusters[0])
    victims = [f"::/{f.path}" for f in fillers[::2]]
    for a in range(0, len(victims), BATCH):
        _run([mdel, "-i", str(image), *victims[a : a + BATCH]])
    return len(victims)


def build_image(cfg: dict[str, Any], class_names: list[str]) -> Path:
    t0 = time.perf_counter()
    tools = {"mkfs": _tool("mkfs.fat", "mkfs.vfat"), **{t: _tool(t) for t in ("mcopy", "mdel", "mmd")}}
    files_dir, out = Path(cfg["files_dir"]), Path(cfg["out_dir"])
    image = out / "image.img"
    if image.exists():
        raise FileExistsError(f"{image} exists; delete {out} to rebuild")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg["seed"])

    sources = sorted(p for p in files_dir.rglob("*") if p.is_file() and p.name not in EXCLUDED and not p.name.startswith("."))
    if not sources:
        raise FileNotFoundError(f"no evidence files in {files_dir}")
    overrides = read_overrides(files_dir)
    stray = sorted(set(overrides) - {p.relative_to(files_dir).as_posix() for p in sources})
    if stray:
        raise ValueError(f"labels.csv names {len(stray)} file(s) not found under {files_dir}: {stray[:5]}")
    names = set(class_names)
    evidence = []
    for i, j in enumerate(rng.permutation(len(sources))):
        src = sources[j]
        rel = src.relative_to(files_dir).as_posix()
        evidence.append(
            {
                "file_id": i,
                "image_path": f"CASE/F{i:04d}{src.suffix.lower()}",
                "source_path": rel,
                "label": label_for(rel, names, overrides),
                "size": src.stat().st_size,
                "sha256": _sha256(src),
            }
        )
    total = sum(e["size"] for e in evidence)
    labels = pd.Series([e["label"] for e in evidence]).value_counts()
    log.info("%d evidence files, %.1f MB; labels: %s", len(evidence), total / 2**20, ", ".join(f"{k} {v}" for k, v in labels.items()))

    mkfs_out = _mkfs(tools["mkfs"], image, cfg, create=True)
    n_fillers = _write_fillers(tools, image, cfg, rng)
    n_deleted = _delete_alternate_fillers(tools["mdel"], image)
    free = Fat32(image).free_clusters() * cfg["cluster_size"]
    log.info("%d fillers written, %d deleted; %.1f MB free for %.1f MB of evidence", n_fillers, n_deleted, free / 2**20, total / 2**20)
    if free < total * 1.02 + len(evidence) * cfg["cluster_size"]:
        raise RuntimeError("not enough free space after deletions: increase size_mb or tail_mb, or use fewer files")

    _run([tools["mmd"], "-i", str(image), "::/CASE"])
    with tempfile.TemporaryDirectory() as tmp:
        for e in evidence:
            staged = Path(tmp) / Path(e["image_path"]).name
            shutil.copyfile(files_dir / e["source_path"], staged)
            _run([tools["mcopy"], "-i", str(image), "-o", str(staged), f"::/{e['image_path']}"])
            staged.unlink()

    fs = Fat32(image)
    by_path = {f.path: f for f in fs.files()}
    for e in evidence:
        f = by_path[e["image_path"]]
        if hashlib.sha256(fs.read_file(f)).hexdigest() != e["sha256"]:
            raise RuntimeError(f"ground truth check failed: {e['image_path']} does not reassemble to its source")
        e |= {"clusters": len(f.clusters), "fragments": f.fragments}
    table = cluster_table(fs, {e["image_path"]: (e["file_id"], e["label"]) for e in evidence})
    table.to_csv(out / "ground_truth.csv", index=False)
    pd.DataFrame(evidence).to_csv(out / "files.csv", index=False)
    geometry = fs.geometry
    preformat = out / "image_preformat.img"
    shutil.copyfile(image, preformat)

    format_out = _mkfs(tools["mkfs"], image, cfg, create=False)
    after = Fat32(image)
    if after.geometry != geometry:
        raise RuntimeError(f"quick format changed the geometry: {geometry} -> {after.geometry}")
    changed = _changed_evidence_clusters(preformat, image, table, geometry.cluster_size)
    if not cfg.get("keep_preformat", True):
        preformat.unlink()

    report = {
        "name": cfg["name"],
        "created": utc_now(),
        "config": cfg,
        "tools": {"mkfs": mkfs_out.strip().splitlines()[0] if mkfs_out.strip() else tools["mkfs"], "mtools": _run([tools["mcopy"], "--version"]).splitlines()[0]},
        "geometry": {**geometry.__dict__, "cluster_size": geometry.cluster_size, "data_offset": geometry.data_offset, "n_clusters": geometry.n_clusters},
        "evidence": {"bytes": total, "clusters": int((table["file_id"] >= 0).sum()), **fragmentation_stats([by_path[e["image_path"]] for e in evidence])},
        "labels": table.loc[table["file_id"] >= 0, "label"].value_counts().to_dict(),
        "fillers": {"written": n_fillers, "deleted": n_deleted},
        "verification": {"files_reassembled": len(evidence), "evidence_clusters_changed_by_format": changed, "files_after_format": len(after.files())},
        "format_output": format_out.strip(),
        "sha256": {"image": _sha256(image), "image_preformat": _sha256(preformat) if preformat.exists() else None},
        "seconds": round(time.perf_counter() - t0, 1),
    }
    write_json(out / "build.json", report)
    log.info("image ready: %s (%d evidence clusters, %d fragmented files, %d clusters changed by format)",
             image, report["evidence"]["clusters"], report["evidence"]["fragmented_files"], changed)
    return out


def _changed_evidence_clusters(before: Path, after: Path, table: pd.DataFrame, cluster_size: int) -> int:
    changed = 0
    with open(before, "rb") as a, open(after, "rb") as b:
        for offset in table.loc[table["file_id"] >= 0, "offset"]:
            a.seek(offset)
            b.seek(offset)
            changed += a.read(cluster_size) != b.read(cluster_size)
    return changed


def load_build(image_dir: Path) -> dict[str, Any]:
    return read_json(Path(image_dir) / "build.json")
