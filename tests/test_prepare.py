import numpy as np
import pytest

from fragtrust.data.prepare import add_duplicates, prepare
from fragtrust.data.store import DUPLICATES_FILE, FragmentStore
from fragtrust.runtime import read_json, write_json

BLOCK = 64
COUNTS = {"train": 40, "val": 20, "test": 20}


def _write_raw(root):
    rng = np.random.default_rng(1)
    raw = {}
    for split, per_class in COUNTS.items():
        y = np.repeat(np.arange(3), per_class).astype(np.uint8)
        x = rng.integers(0, 256, (len(y), BLOCK), dtype=np.uint8)
        raw[split] = (x, y)
    raw["test"][0][0] = raw["train"][0][2]
    raw["test"][0][1] = raw["train"][0][5]
    for split, (x, y) in raw.items():
        np.savez_compressed(root / f"{split}.npz", x=x, y=y)
    return raw


def _config(tmp_path):
    return {
        "name": "tiny",
        "out_dir": str(tmp_path / "processed"),
        "seed": 7,
        "block_size": BLOCK,
        "n_classes": 3,
        "class_names": None,
        "raw": {split: str(tmp_path / f"{split}.npz") for split in COUNTS},
        "subset": {"train_per_class": 10, "val_per_class": 8, "test_per_class": None},
        "probes": {"n_blocks": 5},
        "chunk_rows": 7,
    }


def test_prepare_builds_consistent_subset(tmp_path):
    raw = _write_raw(tmp_path)
    cfg = _config(tmp_path)
    prepare(cfg, tmp_path / "results")
    store = FragmentStore(cfg["out_dir"])
    out = tmp_path / "processed"

    assert np.bincount(store.y("train")).tolist() == [10, 10, 10]
    assert np.bincount(store.y("val_a")).tolist() == [4, 4, 4]
    assert np.bincount(store.y("val_b")).tolist() == [4, 4, 4]
    assert len(store.y("test")) == 60

    for split, source in (("train", "train"), ("val_a", "val"), ("val_b", "val"), ("test", "test")):
        idx = np.load(out / "indices" / f"{split}.npy")
        np.testing.assert_array_equal(np.asarray(store.x(split)), raw[source][0][idx])
        np.testing.assert_array_equal(store.y(split), raw[source][1][idx])

    val_a, val_b = (np.load(out / "indices" / f"{s}.npy") for s in ("val_a", "val_b"))
    assert not set(val_a) & set(val_b)

    leakage = read_json(tmp_path / "results" / "data" / "tiny" / "leakage.json")
    assert leakage["official_train_vs_test"]["test_in_train"] == 2
    assert leakage["official_train_vs_test"]["test_in_train_same_label"] == 2
    assert store.x("ood_aes_ctr").shape == (5, BLOCK)
    assert store.test_entropy().shape == (60,)


def test_prepare_is_reproducible_and_idempotent(tmp_path):
    _write_raw(tmp_path)
    cfg = _config(tmp_path)
    prepare(cfg, tmp_path / "results")
    first = np.load(tmp_path / "processed" / "indices" / "train.npy")
    manifest = (tmp_path / "processed" / "manifest.json").read_text()
    prepare(cfg, tmp_path / "results")
    assert (tmp_path / "processed" / "manifest.json").read_text() == manifest
    prepare(cfg, tmp_path / "results", force=True)
    np.testing.assert_array_equal(np.load(tmp_path / "processed" / "indices" / "train.npy"), first)


def test_duplicate_masks_mark_test_blocks_in_official_train_and_in_our_subset(tmp_path):
    _write_raw(tmp_path)
    cfg = _config(tmp_path)
    prepare(cfg, tmp_path / "results")
    out = tmp_path / "processed"
    masks = FragmentStore(out).test_duplicates()

    train_idx = np.load(out / "indices" / "train.npy")
    assert 2 in train_idx and 5 not in train_idx
    assert np.flatnonzero(masks["official_train"]).tolist() == [0, 1]
    assert np.flatnonzero(masks["subset_train"]).tolist() == [0]
    subset = read_json(tmp_path / "results" / "data" / "tiny" / "leakage.json")["subset_train_vs_test"]
    assert masks["subset_train"].sum() == subset["test_in_train"]


def test_add_duplicates_reproduces_the_masks_of_prepare(tmp_path):
    _write_raw(tmp_path)
    cfg = _config(tmp_path)
    results = tmp_path / "results"
    prepare(cfg, results)
    store = FragmentStore(cfg["out_dir"])
    expected = store.test_duplicates()
    (store.root / DUPLICATES_FILE).unlink()
    assert store.test_duplicates() is None

    add_duplicates(cfg, results)
    for side, mask in store.test_duplicates().items():
        np.testing.assert_array_equal(mask, expected[side])

    leakage_path = results / "data" / "tiny" / "leakage.json"
    leakage = read_json(leakage_path)
    leakage["official_train_vs_test"]["test_in_train"] += 1
    write_json(leakage_path, leakage)
    add_duplicates(cfg, results)
    with pytest.raises(ValueError, match="official_train"):
        add_duplicates(cfg, results, force=True)
