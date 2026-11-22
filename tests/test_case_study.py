"""FAT32 parsing, ground truth, the signature baseline, image scanning and the case-study evaluation."""

import gzip
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fragtrust.case_study.build import build_image
from fragtrust.case_study.evaluate import NONE
from fragtrust.case_study.fat32 import Fat32
from fragtrust.case_study.ground_truth import NOT_EVIDENCE, cluster_table
from fragtrust.case_study.scan import image_blocks
from fragtrust.case_study.signatures import SIGNATURES, candidate_sets, match_headers, propagate
from fragtrust.data.prepare import load_class_names
from fragtrust.data.store import FragmentStore

FIXTURES = Path(__file__).parent / "fixtures"
PROJECT = Path(__file__).parents[1]


@pytest.fixture(scope="module")
def fat_image(tmp_path_factory) -> Fat32:
    path = tmp_path_factory.mktemp("fat") / "image.img"
    with gzip.open(FIXTURES / "fat32_macos_512.img.gz") as src, open(path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    return Fat32(path)


def test_fat32_files_reassemble_to_source_hashes(fat_image):
    manifest = json.loads((FIXTURES / "fat32_macos_512.manifest.json").read_text())
    files = {f.path: f for f in fat_image.files()}
    others = set(files) - set(manifest)
    assert set(manifest) <= set(files)
    assert all(path.startswith("FILL/") or Path(path).name.startswith("._") for path in others)
    for path, expected in manifest.items():
        data = fat_image.read_file(files[path])
        assert len(data) == expected["size"], path
        assert hashlib.sha256(data).hexdigest() == expected["sha256"], path
    assert files["SUB/DEEP/empty.log"].clusters == ()
    assert max(f.fragments for f in files.values()) >= 4


def test_cluster_table_follows_cluster_chains(fat_image):
    files = [f for f in fat_image.files() if f.clusters]
    table = cluster_table(fat_image, {f.path: (i, "txt") for i, f in enumerate(files)})
    g = fat_image.geometry
    assert len(table) == g.n_clusters
    assert (table["offset"] == [g.cluster_offset(c) for c in table["cluster"]]).all()
    for i, f in enumerate(files):
        rows = table[table["file_id"] == i].sort_values("index_in_file")
        assert rows["cluster"].tolist() == list(f.clusters)
        assert rows["used_bytes"].sum() == f.size
        assert rows["fragmented"].all() == (f.fragments > 1)
        assert rows["position"].iloc[0] == ("single" if len(rows) == 1 else "header")
    assert (table.loc[table["file_id"] < 0, "label"] == NOT_EVIDENCE).all()


def _block(head: bytes, size: int = 512) -> np.ndarray:
    block = np.full(size, ord("A"), dtype=np.uint8)
    block[: len(head)] = np.frombuffer(head, dtype=np.uint8)
    return block


def test_signatures_prefer_the_most_specific_match():
    names = ["jpg", "epub", "docx", "zip", "cr2", "tiff", "dng", "heic", "mp4", "mov", "mobi", "txt"]
    cases = [
        (b"\xff\xd8\xff\xe0", {"jpg"}),
        (b"PK\x03\x04" + bytes(26) + b"mimetypeapplication/epub+zip", {"epub"}),
        (b"PK\x03\x04" + bytes(26) + b"[Content_Types].xml", {"docx", "zip"}),
        (b"II*\x00\x10\x00\x00\x00CR", {"cr2"}),
        (b"II*\x00\x08\x00\x00\x00", {"tiff", "dng"}),
        (b"\x00\x00\x00\x18ftypheic", {"heic"}),
        (b"\x00\x00\x00\x18ftypisom", {"mp4", "mov", "heic"}),
        (bytes(60) + b"BOOKMOBI", {"mobi"}),
        (b"plain text, no magic number", set()),
    ]
    candidates = candidate_sets(match_headers(np.stack([_block(head) for head, _ in cases])), names)
    for (head, expected), row in zip(cases, candidates):
        assert {names[j] for j in np.flatnonzero(row)} == expected, head[:12]


def test_signature_classes_are_fft75_types():
    fft75, _ = load_class_names(PROJECT / "configs" / "data" / "fft75_s1_classes.json", 75)
    assert {c for sig in SIGNATURES for c in sig.classes} <= set(fft75)


def test_propagation_stops_at_next_header_or_empty_cluster():
    headers = np.array([-1, 3, -1, -1, 7, -1, -1, -1])
    empty = np.array([0, 0, 0, 1, 0, 0, 1, 0], dtype=bool)
    assert propagate(headers, empty).tolist() == [-1, 3, 3, -1, 7, 7, -1, -1]


def test_scan_and_evaluation_on_simulated_image(case):
    gt = pd.read_csv(case.image_dir / "ground_truth.csv")
    names = FragmentStore(case.trained.data).class_names
    known = gt["label"].isin(names).to_numpy()
    assert image_blocks(case.image_dir).shape[0] == len(gt)

    for run_id, report in case.reports.items():
        logits = np.load(case.trained.results / "case_study" / "sim" / "scans" / f"{run_id}.npy")
        assert logits.shape == (len(gt), len(names))
        y = gt.loc[known, "label"].map({n: i for i, n in enumerate(names)}).to_numpy()
        accuracy = float((logits[known].argmax(axis=1) == y).mean())
        assert report["transferred"]["all"]["accuracy"] == pytest.approx(accuracy)
        assert report["clusters"] == len(gt)
        assert report["evidence_clusters_known_type"] == int(known.sum())
        assert set(report["unseen_types"]) == {"encrypted"}
        assert report["not_evidence"]["empty_share"] == 1.0
        assert report["signatures"]["propagated"]["all"]["coverage"] == 0.0
        assert set(report["recalibration"]) >= {"transferred", "recalibrated"}

        out = case.out(run_id)
        assert len(pd.read_csv(out / "recalibration_splits.csv")) == 2 * report["recalibration"]["n_splits"]
        per_type = pd.read_csv(out / "per_type.csv")
        assert per_type["clusters"].sum() == known.sum()
        disk = np.load(out / "disk_map.npz")
        assert len(disk["label"]) == len(gt)
        assert (disk["model"][~known] == NONE).all()
        assert (disk["label"][(gt["label"] == "encrypted").to_numpy()] == -2).all()


def _have_fat_tools() -> bool:
    return bool(shutil.which("mkfs.fat") or shutil.which("mkfs.vfat")) and all(shutil.which(t) for t in ("mcopy", "mdel", "mmd"))


@pytest.mark.skipif(not _have_fat_tools(), reason="needs dosfstools and mtools (Linux)")
def test_build_image_fragments_and_survives_quick_format(tmp_path):
    files = tmp_path / "files"
    rng = np.random.default_rng(0)
    for i, (ext, kb) in enumerate([("jpg", 300), ("pdf", 200), ("txt", 40), ("bin", 120), ("png", 500)] * 2):
        (files / f"d{i % 3}").mkdir(parents=True, exist_ok=True)
        (files / f"d{i % 3}" / f"file{i}.{ext}").write_bytes(rng.integers(1, 256, kb * 1024, dtype=np.uint8).tobytes())
    (files / "labels.csv").write_text("path,label\nd0/file3.bin,encrypted\n")
    cfg = {"name": "tiny", "files_dir": str(files), "out_dir": str(tmp_path / "image"), "size_mb": 40, "cluster_size": 512,
           "volume_label": "TEST", "seed": 0, "filler_kb": [4, 64], "tail_mb": 2, "keep_preformat": True}
    out = build_image(cfg, ["jpg", "pdf", "txt", "png"])

    report = json.loads((out / "build.json").read_text())
    assert report["verification"]["files_reassembled"] == 10
    assert report["verification"]["evidence_clusters_changed_by_format"] == 0
    assert report["verification"]["files_after_format"] == 0
    assert report["evidence"]["fragmented_files"] > 0
    gt = pd.read_csv(out / "ground_truth.csv")
    assert set(gt["label"]) == {"jpg", "pdf", "txt", "png", "encrypted", "unknown", NOT_EVIDENCE}

    files_table = pd.read_csv(out / "files.csv")
    blocks = image_blocks(out)
    for row in files_table.itertuples():
        clusters = gt[gt["file_id"] == row.file_id].sort_values("index_in_file")["cluster"].to_numpy()
        data = blocks[clusters - 2].tobytes()[: row.size]
        assert hashlib.sha256(data).hexdigest() == row.sha256, row.source_path
