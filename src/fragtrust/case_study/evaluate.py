"""RQ4: do the guarantees fitted on FFT-75 hold on a formatted FAT32 image, and does a small
labelled sample from the image restore them?

Transferred = temperature, abstention thresholds and conformal quantiles fitted on FFT-75 (val_a/val_b).
Recalibrated = the same three refitted on a random share of the image's evidence files, evaluated on
the remaining files (grouped by file, so clusters of one file never sit on both sides).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from ..config import RunDir
from ..data.store import FragmentStore
from ..runtime import read_json, setup_logging, write_json
from ..trust.calibration import ece, fit_temperature, probs
from ..trust.conformal import ConformalPredictor
from ..trust.selective import apply_threshold, fit_threshold, meets_target
from .build import load_build
from .ground_truth import NOT_EVIDENCE
from .scan import case_dir, signature_table
from .signatures import candidate_sets

log = logging.getLogger(__name__)

TARGETS = (0.95, 0.99)
ALPHA = 0.05
DELTA = 0.05
CONFORMAL = ("lac", "aps")
POSITIONS = ("header", "middle", "tail", "single")

# disk-map status codes
NONE, CORRECT, WRONG, ABSTAINED = 0, 1, 2, 3


@dataclass
class TrustParams:
    temperature: float
    thresholds: dict[float, float | None]
    conformal: dict[str, ConformalPredictor]


def transferred_params(trust: dict[str, Any]) -> TrustParams:
    thresholds = {t: trust["selective"]["msp"]["targets"][f"{t:.2f}"]["ltt"]["threshold"] for t in TARGETS}
    conformal = {
        m: ConformalPredictor(m, ALPHA, q=trust["conformal"][f"{m}@{1 - ALPHA:.2f}"]["q"]) for m in CONFORMAL
    }
    return TrustParams(trust["temperature"], thresholds, conformal)


def fit_params(logits: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> TrustParams:
    temperature = fit_temperature(logits, y)
    p = probs(logits, temperature)
    conf, correct = p.max(axis=1), p.argmax(axis=1) == y
    return TrustParams(
        temperature,
        {t: fit_threshold(conf, correct, t, "ltt", DELTA) for t in TARGETS},
        {m: ConformalPredictor(m, ALPHA).fit(p, y, rng) for m in CONFORMAL},
    )


def trust_metrics(params: TrustParams, logits: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> dict[str, Any]:
    if len(y) == 0:
        return {"n": 0}
    p = probs(logits, params.temperature)
    conf, correct = p.max(axis=1), p.argmax(axis=1) == y
    out: dict[str, Any] = {"n": int(len(y)), "accuracy": float(correct.mean()), "ece": ece(conf, correct)}
    for t, tau in params.thresholds.items():
        result = apply_threshold(conf, correct, tau)
        acc = result["selective_accuracy"]
        out |= {
            f"coverage@{t:.2f}": result["coverage"],
            f"answered@{t:.2f}": result["answered"],
            f"no_answers@{t:.2f}": result["answered"] == 0,
            f"selective_accuracy@{t:.2f}": acc,
            f"meets@{t:.2f}": meets_target(acc, t),
        }
    for cp in params.conformal.values():
        sets = cp.predict(p, rng)
        out |= {f"{cp.name}_coverage": float(sets[np.arange(len(y)), y].mean()), f"{cp.name}_size": float(sets.sum(axis=1).mean())}
    return out


def abstention_metrics(params: TrustParams, logits: np.ndarray, class_names: list[str], rng: np.random.Generator) -> dict[str, Any]:
    """Behaviour on clusters whose true type the model cannot know (unknown, encrypted, free space)."""
    if len(logits) == 0:
        return {"n": 0}
    p = probs(logits, params.temperature)
    conf = p.max(axis=1)
    top = pd.Series(np.asarray(class_names)[p.argmax(axis=1)]).value_counts(normalize=True).head(5).round(4)
    out: dict[str, Any] = {"n": int(len(p)), "mean_confidence": float(conf.mean()), "top_predicted": top.to_dict()}
    for t, tau in params.thresholds.items():
        out[f"abstained@{t:.2f}"] = 1.0 if tau is None else float((conf < tau).mean())
    for cp in params.conformal.values():
        out[f"{cp.name}_size"] = float(cp.predict(p, rng).sum(axis=1).mean())
    return out


def signature_metrics(candidates: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    if len(y) == 0:
        return {"n": 0}
    answered = candidates.any(axis=1)
    correct = candidates[np.arange(len(y)), y]
    return {
        "n": int(len(y)),
        "coverage": float(answered.mean()),
        "accuracy_answered": float(correct[answered].mean()) if answered.any() else None,
        "mean_set_size_answered": float(candidates[answered].sum(axis=1).mean()) if answered.any() else None,
        "correct_overall": float(correct.mean()),
    }


def recalibration(
    logits: np.ndarray, y: np.ndarray, file_ids: np.ndarray, transferred: TrustParams,
    cal_fraction: float, n_splits: int, rng: np.random.Generator,
) -> tuple[dict[str, Any], pd.DataFrame]:
    files = np.unique(file_ids)
    if len(files) < 5:
        return {"skipped": f"only {len(files)} labelled files"}, pd.DataFrame()
    n_cal = max(1, round(cal_fraction * len(files)))
    rows = []
    for split in range(n_splits):
        cal = np.isin(file_ids, rng.permutation(files)[:n_cal])
        local = fit_params(logits[cal], y[cal], rng)
        for name, params in (("transferred", transferred), ("recalibrated", local)):
            rows.append({"split": split, "params": name, "cal_files": n_cal, "cal_clusters": int(cal.sum()),
                         "temperature": params.temperature, **trust_metrics(params, logits[~cal], y[~cal], rng)})
    table = pd.DataFrame(rows)
    metric_columns = [c for c in table.columns if c not in ("split", "params")]
    table[metric_columns] = table[metric_columns].astype(float)
    numeric = table.drop(columns=["split"]).groupby("params").agg(["mean", "std"])
    summary = {
        params: {col: {"mean": float(numeric.loc[params, (col, "mean")]), "std": float(numeric.loc[params, (col, "std")])}
                 for col in numeric.columns.levels[0]}
        for params in numeric.index
    }
    return {"cal_fraction": cal_fraction, "n_splits": n_splits, "files": int(len(files)), **summary}, table


def _groups(gt: pd.DataFrame) -> dict[str, np.ndarray]:
    groups = {"all": np.ones(len(gt), dtype=bool), "fragmented": gt["fragmented"].to_numpy(), "contiguous": ~gt["fragmented"].to_numpy()}
    groups |= {pos: (gt["position"] == pos).to_numpy() for pos in POSITIONS}
    return {name: mask for name, mask in groups.items() if mask.any()}


def evaluate_case(
    run_path: Path, image_dir: Path, results: Path, cal_fraction: float = 0.2, n_splits: int = 20, seed: int = 0
) -> dict[str, Any]:
    run = RunDir(run_path)
    base = case_dir(results, image_dir)
    out = base / run.id
    setup_logging(out / "case_study.log")
    trust_file = run.path / "analysis" / "trust.json"
    scan_file = base / "scans" / f"{run.id}.npy"
    for required, command in ((trust_file, "analyze"), (scan_file, "case-scan")):
        if not required.exists():
            raise FileNotFoundError(f"{required} missing: run `fragtrust {command}` first")

    names = FragmentStore(run.config()["data"]["dir"]).class_names
    index = {n: i for i, n in enumerate(names)}
    gt = pd.read_csv(Path(image_dir) / "ground_truth.csv")
    logits = np.load(scan_file)
    sig = signature_table(Path(image_dir), results)
    params = transferred_params(read_json(trust_file))
    rng = np.random.default_rng(seed)

    known = gt["label"].isin(index).to_numpy()
    gk = gt[known].reset_index(drop=True)
    y = gk["label"].map(index).to_numpy()
    lk = logits[known]
    groups = _groups(gk)
    pred = probs(lk, params.temperature).argmax(axis=1)

    report: dict[str, Any] = {
        "run": run.id,
        "image": load_build(image_dir)["name"],
        "clusters": int(len(gt)),
        "evidence_clusters_known_type": int(known.sum()),
        "types_present": sorted(gk["label"].unique().tolist()),
        "macro_f1": float(f1_score(y, pred, labels=np.unique(y), average="macro")),
        "transferred": {g: trust_metrics(params, lk[m], y[m], rng) for g, m in groups.items()},
        "signatures": {},
    }
    for variant in ("header", "propagated"):
        candidates = candidate_sets(sig[variant][known], names)
        report["signatures"][variant] = {g: signature_metrics(candidates[m], y[m]) for g, m in groups.items()}

    unseen = sorted(set(gt["label"]) - set(names) - {NOT_EVIDENCE})
    report["unseen_types"] = {lab: abstention_metrics(params, logits[(gt["label"] == lab).to_numpy()], names, rng) for lab in unseen}
    free = (gt["label"] == NOT_EVIDENCE).to_numpy()
    report["not_evidence"] = {
        "empty_share": float(sig["empty"][free].mean()) if free.any() else None,
        **abstention_metrics(params, logits[free], names, rng),
    }

    summary, table = recalibration(lk, y, gk["file_id"].to_numpy(), params, cal_fraction, n_splits, rng)
    report["recalibration"] = summary
    if len(table):
        table.to_csv(out / "recalibration_splits.csv", index=False)

    _per_type(gk, y, lk, params, sig["header"][known], names).to_csv(out / "per_type.csv", index=False)
    cluster_size = load_build(image_dir)["geometry"]["cluster_size"]
    _save_disk_map(out / "disk_map.npz", gt, known, y, lk, params, sig["propagated"], names, cluster_size)
    write_json(out / "case_study.json", report)

    t = report["transferred"]["all"]
    s = report["signatures"]["propagated"]["all"]
    answered = "n/a" if t["selective_accuracy@0.99"] is None else f"{t['selective_accuracy@0.99']:.4f}"
    log.info(
        "%s on %s | accuracy %.4f | coverage@0.99 %.3f (accuracy %s) | aps@0.95 coverage %.3f | signatures: coverage %.3f",
        run.id, report["image"], t["accuracy"], t["coverage@0.99"], answered, t["aps@0.95_coverage"], s["coverage"],
    )
    return report


def _per_type(gk: pd.DataFrame, y: np.ndarray, logits: np.ndarray, params: TrustParams, header_ids: np.ndarray, names: list[str]) -> pd.DataFrame:
    p = probs(logits, params.temperature)
    correct = p.argmax(axis=1) == y
    tau = params.thresholds[max(TARGETS)]
    accepted = p.max(axis=1) >= tau if tau is not None else np.zeros(len(y), dtype=bool)
    header_hit = candidate_sets(header_ids, names)[np.arange(len(y)), y]
    first = gk["position"].isin(["header", "single"]).to_numpy()
    frame = pd.DataFrame({"type": gk["label"], "file_id": gk["file_id"], "correct": correct, "accepted": accepted,
                          "header_hit": np.where(first, header_hit, np.nan), "fragmented": gk["fragmented"]})
    return frame.groupby("type").agg(
        files=("file_id", "nunique"), clusters=("correct", "size"), accuracy=("correct", "mean"),
        coverage_at_max_target=("accepted", "mean"), signature_header_detection=("header_hit", "mean"),
        fragmented_share=("fragmented", "mean"),
    ).reset_index()


def _save_disk_map(path: Path, gt: pd.DataFrame, known: np.ndarray, y: np.ndarray, logits: np.ndarray,
                   params: TrustParams, propagated: np.ndarray, names: list[str], cluster_size: int) -> None:
    label = np.full(len(gt), -1, dtype=np.int16)
    label[known] = y
    label[(~known) & (gt["label"] != NOT_EVIDENCE).to_numpy()] = -2
    p = probs(logits, params.temperature)
    correct = p.argmax(axis=1) == y
    tau = params.thresholds[max(TARGETS)]
    abstain = p.max(axis=1) < tau if tau is not None else np.ones(len(y), dtype=bool)
    model = np.full(len(gt), NONE, dtype=np.int8)
    model[known] = np.where(abstain, ABSTAINED, np.where(correct, CORRECT, WRONG))
    candidates = candidate_sets(propagated[known], names)
    signature = np.full(len(gt), NONE, dtype=np.int8)
    signature[known] = np.where(~candidates.any(axis=1), ABSTAINED, np.where(candidates[np.arange(len(y)), y], CORRECT, WRONG))
    np.savez_compressed(path, label=label, model=model, signature=signature, class_names=np.asarray(names),
                        target=max(TARGETS), cluster_size=cluster_size)
