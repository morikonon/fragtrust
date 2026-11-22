"""Session fixtures: a tiny trained and analysed LightGBM + CNN, and a simulated evidence image scanned by both."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from helpers import cnn_config, lgbm_config, make_store

from fragtrust.case_study.evaluate import evaluate_case
from fragtrust.case_study.scan import scan
from fragtrust.case_study.simulate import simulate_image
from fragtrust.experiment import train
from fragtrust.explain import explain
from fragtrust.trust.analyze import analyze

RUNS = ("lgbm_s0", "cnn_s0")


@dataclass(frozen=True)
class Trained:
    data: Path
    results: Path

    def run(self, run_id: str) -> Path:
        return self.results / "runs" / run_id


@dataclass(frozen=True)
class Case:
    trained: Trained
    image_dir: Path
    reports: dict[str, dict]

    def out(self, run_id: str) -> Path:
        return self.trained.results / "case_study" / "sim" / run_id


@pytest.fixture(scope="session")
def trained(tmp_path_factory) -> Trained:
    root = tmp_path_factory.mktemp("trained")
    make_store(root / "data")
    out = Trained(root / "data", root / "results")
    for cfg in (lgbm_config(out.data), cnn_config(out.data)):
        train(cfg, 0, out.results)
        analyze(out.run(f"{cfg['name']}_s0"))
    explain(out.run("lgbm_s0"), per_class=10)
    return out


@pytest.fixture(scope="session")
def case(trained, tmp_path_factory) -> Case:
    image_dir = simulate_image(trained.data, tmp_path_factory.mktemp("image"), name="sim", n_files=40, seed=0)
    reports = {}
    for run_id in RUNS:
        scan(trained.run(run_id), image_dir, trained.results, device="cpu")
        reports[run_id] = evaluate_case(trained.run(run_id), image_dir, trained.results, n_splits=5)
    return Case(trained, image_dir, reports)
