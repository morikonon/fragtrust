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
from fragtrust.report import build_assets
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
