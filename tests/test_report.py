"""SHAP attributions, CPU benchmark backends, the paper assets build and draft number tags."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from helpers import transformer_config

from fragtrust.benchmark import benchmark
from fragtrust.config import RunDir
from fragtrust.data.store import FragmentStore
from fragtrust.experiment import train
from fragtrust.features import N_FEATURES
from fragtrust.inference import load_booster
from fragtrust.report import build_assets, tables
from fragtrust.report.collect import Results
from fragtrust.report.figures import Context
from fragtrust.report.numbers import check
from fragtrust.runtime import read_json, write_json


def test_shap_contributions_add_up_to_raw_scores(trained):
    run = RunDir(trained.run("lgbm_s0"))
    store = FragmentStore(trained.data)
    x = store.features("test")[:50]
    booster = load_booster(run)
    contrib = booster.predict(x, pred_contrib=True).reshape(len(x), store.n_classes, N_FEATURES + 1)
    np.testing.assert_allclose(contrib.sum(axis=2), booster.predict(x, raw_score=True), atol=1e-6)

    table = pd.read_csv(run.path / "analysis" / "shap_global.csv")
    assert len(table) == N_FEATURES
    assert table["mean_abs_shap"].is_monotonic_decreasing
    report = read_json(run.path / "analysis" / "shap.json")
    assert -1 <= report["spearman_shap_vs_gain"] <= 1
    assert len(pd.read_csv(run.path / "analysis" / "shap_per_class.csv")) == 5 * store.n_classes


@pytest.mark.parametrize(("run_id", "backend"), [("cnn_s0", "torch"), ("lgbm_s0", "lightgbm")])
def test_benchmark_writes_throughput(trained, tmp_path, run_id, backend):
    result = benchmark(trained.run(run_id), tmp_path, n_blocks=64, threads=[1, 2], backend=backend)
    assert result["backend"] == backend
    assert [row["threads"] for row in result["results"]] == [1, 2]
    assert all(row["mb_per_s"] > 0 for row in result["results"])
    assert len(list((tmp_path / "benchmarks").glob(f"*/{run_id}.{backend}.json"))) == 1


@pytest.mark.parametrize("model", ["cnn", "transformer"])
def test_onnx_export_matches_pytorch(trained, tmp_path, model):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    if model == "transformer":
        train(transformer_config(trained.data), 0, tmp_path)
    run = trained.run("cnn_s0") if model == "cnn" else tmp_path / "runs" / "transformer_s0"
    result = benchmark(run, tmp_path, n_blocks=64, threads=[1], backend="onnx")
    assert result["onnx_vs_torch_max_abs_diff"] < 1e-3


def test_report_builds_every_available_asset(case, tmp_path):
    results = case.trained.results
    config = tmp_path / "paper.yaml"
    config.write_text(yaml.safe_dump({
        "results": str(results),
        "out_dir": str(tmp_path / "paper"),
        "models": {"lgbm": "LightGBM", "cnn": "CNN"},
        "shap_run": "lgbm_s0",
        "case_study": {"image": "sim", "run": "cnn_s0", "runs": {"lgbm_s0": "LightGBM", "cnn_s0": "CNN"}},
    }))
    built = build_assets(config)

    figures = {"reliability", "risk_coverage", "per_class_f1", "entropy", "shap", "disk_map"}
    tables = {"main_results", "calibration", "selective", "conformal", "ood", "case_study", "recalibration"}
    assert {f"figure:{f}" for f in figures} | {f"table:{t}" for t in tables} <= set(built)
    assert "figure:capacity" not in built
    for name, asset in built.items():
        assert asset["sources"], name
        assert all(Path(path).exists() for path in asset["files"]), name
    assets = json.loads((tmp_path / "paper" / "assets.json").read_text())
    assert set(assets["assets"]) == set(built)
    ood = pd.read_csv(tmp_path / "paper" / "tables" / "ood.csv")
    assert ood["AUROC (byte entropy)"].between(0, 1).all()


def _ltt(answered: int, accuracy: float | None, met: bool) -> dict:
    return {"ltt": {"answered": answered, "coverage": answered / 100, "selective_accuracy": accuracy, "meets_target": met},
            "empirical": {"coverage": 0.5}}


def _split_summary(coverage: float, accuracy: float, met: float, no_answers: float) -> dict:
    means = {"temperature": 1.0, "coverage@0.99": coverage, "selective_accuracy@0.99": accuracy, "meets@0.99": met,
             "no_answers@0.99": no_answers, "aps@0.95_coverage": 0.95, "aps@0.95_size": 1.2}
    return {key: {"mean": mean, "std": float("nan")} for key, mean in means.items()}


def test_tables_show_no_answers_instead_of_a_pass(tmp_path):
    results = tmp_path / "results"
    for seed, target99 in ((0, _ltt(40, 0.995, True)), (1, _ltt(0, None, False))):
        run = results / "runs" / f"m_s{seed}"
        write_json(run / "meta.json", {"model": "m", "seed": seed})
        write_json(run / "analysis" / "trust.json", {"selective": {"msp": {"aurc": 0.01, "targets": {"0.95": _ltt(90, 0.96, True), "0.99": target99}}}})
        (run / "DONE").write_text("done\n")
    write_json(results / "case_study" / "img" / "m_s0" / "case_study.json", {"recalibration": {
        "cal_fraction": 0.2,
        "transferred": _split_summary(0.6, 0.995, 0.75, 0.0),
        "recalibrated": _split_summary(0.0, float("nan"), 0.0, 1.0),
    }})
    ctx = Context(Results(results), {"models": {"m": "M"}, "case_study": {"image": "img", "runs": {"m_s0": "M"}}}, tmp_path, [])

    selective = tables.selective(ctx)[0].iloc[0]
    assert selective["Met 95%"] == "2/2"
    assert selective["Met 99%"] == "1/2 (1 no answers)"
    recal = tables.recalibration(ctx)[0].set_index("Fitted on")
    assert recal.loc["20% of image files", "Accuracy of answers (%)"] == "no answers"
    assert recal.loc["20% of image files", "Splits meeting 99% (%)"] == "0.00"
    assert recal.loc["20% of image files", "Splits with no answers at 99% (%)"] == "100.00"
    assert recal.loc["FFT-75 (transferred)", "Accuracy of answers (%)"] == "99.50"


def _groups(n_dup: int, acc_dup: float | None) -> dict:
    return {
        "all": {"n": 100, "accuracy": 0.8, "macro_f1": 0.79, "ece": 0.02},
        "duplicated": {"n": n_dup, "accuracy": acc_dup, "macro_f1": acc_dup, "ece": None if acc_dup is None else 0.01},
        "without": {"n": 100 - n_dup, "accuracy": 0.79, "macro_f1": 0.78, "ece": 0.021},
        "all_minus_without": {"accuracy": 0.01, "macro_f1": 0.01, "ece": -0.001},
    }


def test_duplicates_table_lists_both_train_sides_and_skips_runs_without_masks(tmp_path):
    results = tmp_path / "results"
    docs = {"m": {"duplicates": {"official_train": _groups(10, 0.99), "subset_train": _groups(0, None)}}, "old": {}}
    for model, doc in docs.items():
        run = results / "runs" / f"{model}_s0"
        write_json(run / "meta.json", {"model": model, "seed": 0})
        write_json(run / "analysis" / "trust.json", doc)
        (run / "DONE").write_text("done\n")
    ctx = Context(Results(results), {"models": {"m": "M", "old": "Old"}}, tmp_path, [])

    table = tables.duplicates(ctx)[0].set_index("Duplicated in")
    assert table["Model"].tolist() == ["M", "M"]
    assert table.loc["official train", "Accuracy duplicated (%)"] == "99.00"
    assert table.loc["official train", "All − without (pp)"] == "1.00"
    assert table.loc["our train subset", "Duplicated test blocks"] == "0"
    assert table.loc["our train subset", "Accuracy duplicated (%)"] == "–"
    ctx = Context(Results(results), {"models": {"old": "Old"}}, tmp_path, [])
    assert tables.duplicates(ctx) is None


def test_numbers_resolve_tags_and_flag_typed_numbers(tmp_path):
    write_json(tmp_path / "trust.json", {"calibration": {"ece": 0.01234}, "sizes": [1.5, 2.5]})
    pd.DataFrame({"model": ["cnn", "lgbm"], "accuracy": [0.91234, 0.8765]}).to_csv(tmp_path / "by_model.csv", index=False)
    draft = tmp_path / "draft.md"
    draft.write_text(
        "ECE {{trust.json:calibration/ece|.3f}}, accuracy {{by_model.csv:lgbm/accuracy|pct1}}%.\n"
        "Second set size {{trust.json:sizes/1}}; missing {{trust.json:calibration/nll|.3f}}.\n"
        "Typed by hand: 0.93 and 12%, but RQ2, [3, 4], 2026 and `0.5` are fine.\n"
    )
    result = check(draft, tmp_path)
    assert result.resolved == 3
    assert result.rendered.splitlines()[0] == "ECE 0.012, accuracy 87.7%."
    assert "Second set size 2.5;" in result.rendered
    assert len(result.errors) == 1 and result.errors[0].startswith("line 2:")
    assert result.untagged == ["line 3: 0.93", "line 3: 12%"]
