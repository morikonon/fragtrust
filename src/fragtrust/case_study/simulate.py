"""Simulated evidence image for smoke tests where dosfstools/mtools are unavailable. Not for the paper.

Writes the same files as `build.py` (image.img holding only the data area, ground_truth.csv, files.csv,
build.json): test blocks of a processed dataset grouped into "files", split into fragments, shuffled
and separated by empty clusters, plus one random-byte file labelled 'encrypted'.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.store import FragmentStore
from ..runtime import utc_now, write_json
from .ground_truth import NOT_EVIDENCE


def simulate_image(store_dir: Path, out_dir: Path, name: str = "simulated", n_files: int = 40, seed: int = 0) -> Path:
    store = FragmentStore(store_dir)
    x, y, names, block = store.x("test"), store.y("test"), store.class_names, store.block_size
    rng = np.random.default_rng(seed)
    files = []
    for _ in range(n_files):
        c = int(rng.integers(len(names)))
        members = np.flatnonzero(y == c)
        files.append((names[c], np.asarray(x[rng.choice(members, min(len(members), int(rng.integers(3, 20))), replace=False)])))
    files.append(("encrypted", rng.integers(0, 256, (15, block), dtype=np.uint8)))

    pieces = []
    for fid, (_, blocks) in enumerate(files):
        n_frag = int(rng.integers(1, 4)) if len(blocks) >= 6 else 1
        cuts = np.sort(rng.choice(np.arange(1, len(blocks)), n_frag - 1, replace=False))
        pieces += [(fid, part) for part in np.split(np.arange(len(blocks)), cuts)]

    n_pieces = Counter(fid for fid, _ in pieces)
    clusters, rows = [], []
    for i in rng.permutation(len(pieces)):
        fid, part = pieces[i]
        for _ in range(int(rng.integers(0, 4))):
            clusters.append(np.zeros(block, dtype=np.uint8))
            rows.append((-1, NOT_EVIDENCE, -1, "", 0, False))
        label, blocks = files[fid]
        fragmented = n_pieces[fid] > 1
        for k in part:
            position = "single" if len(blocks) == 1 else "header" if k == 0 else "tail" if k == len(blocks) - 1 else "middle"
            clusters.append(blocks[k])
            rows.append((fid, label, int(k), position, block, fragmented))

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.stack(clusters).tofile(out / "image.img")
    gt = pd.DataFrame(rows, columns=["file_id", "label", "index_in_file", "position", "used_bytes", "fragmented"])
    gt.insert(0, "cluster", np.arange(2, len(gt) + 2))
    gt.insert(1, "offset", np.arange(len(gt), dtype=np.int64) * block)
    gt.to_csv(out / "ground_truth.csv", index=False)
    pd.DataFrame(
        [{"file_id": fid, "label": label, "clusters": len(blocks)} for fid, (label, blocks) in enumerate(files)]
    ).to_csv(out / "files.csv", index=False)
    write_json(
        out / "build.json",
        {
            "name": name,
            "created": utc_now(),
            "simulated": True,
            "geometry": {"data_offset": 0, "cluster_size": block, "n_clusters": len(gt)},
        },
    )
    return out
