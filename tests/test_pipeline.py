"""End to end on a tiny synthetic dataset: train, resume, ensemble, analyze, summarize."""

import numpy as np
import pandas as pd
import torch
from helpers import make_store

from fragtrust.config import RunDir
from fragtrust.experiment import ensemble, train
from fragtrust.models import build_net
from fragtrust.runtime import read_json, seed_everything
from fragtrust.summarize import summarize
from fragtrust.training import trainer
from fragtrust.training.batches import BlockBatches
from fragtrust.training.trainer import TrainConfig, fit
from fragtrust.trust.analyze import analyze

CNN = {"type": "cnn", "emb_dim": 8, "channels": [16, 16], "kernel_size": 5, "pool": 8}


def _cnn_config(data_dir):
    return {
        "name": "cnn",
        "device": "cpu",
        "data": {"dir": str(data_dir)},
        "model": CNN,
        "train": {"epochs": 2, "batch_size": 32, "lr": 3e-3, "warmup_epochs": 0.5, "eval_batch_size": 128, "log_every": 1000},
    }


def _lgbm_config(data_dir):
    return {
        "name": "lgbm",
        "data": {"dir": str(data_dir)},
        "model": {"type": "lightgbm"},
        "params": {"num_leaves": 7, "learning_rate": 0.2, "min_data_in_leaf": 5},
        "boost": {"num_boost_round": 30, "early_stopping": 10, "log_every": 100},
        "threads": 1,
    }


def test_full_pipeline(tmp_path):
    data_dir = tmp_path / "data"
    make_store(data_dir)
    results = tmp_path / "results"
    for seed in (0, 1):
        train(_cnn_config(data_dir), seed, results)
    train(_lgbm_config(data_dir), 0, results)
    ensemble([results / "runs" / "cnn_s0", results / "runs" / "cnn_s1"], "cnn_ens", results)

    for run_id in ("cnn_s0", "cnn_s1", "lgbm_s0", "cnn_ens"):
        run = RunDir(results / "runs" / run_id)
        assert run.done
        assert not (run.path / "last.pt").exists()
        report = analyze(run.path)
        assert set(report) >= {"calibration", "selective", "conformal", "ood"}
        assert run.logits("test").shape == (210, 7)
    assert "energy" not in read_json(results / "runs" / "lgbm_s0" / "analysis" / "trust.json")["ood"]["ood_random"]["auroc"]

    by_model = summarize(results)
    assert by_model.loc["cnn", "n_runs"] == 2
    assert pd.read_csv(results / "summary" / "runs.csv").shape[0] == 4

    train(_cnn_config(data_dir), 0, results)
    assert len(pd.read_csv(results / "runs" / "cnn_s0" / "history.csv")) == 2


def test_interrupted_training_resumes_to_identical_weights(tmp_path):
    store = make_store(tmp_path / "data", per_class=(40, 20, 20, 20))
    cfg = TrainConfig(epochs=3, batch_size=32, lr=3e-3, warmup_epochs=0.5, eval_batch_size=128, log_every=1000, patience=10)
    cpu = torch.device("cpu")

    def run(out_dir, interrupt_at_eval=None):
        out_dir.mkdir()
        seed_everything(0)
        model = build_net(CNN, store.n_classes, store.block_size)
        batches = lambda: (
            BlockBatches(store.x("train"), store.y("train"), cfg.batch_size, cpu, shuffle=True, seed=0, drop_last=True),
            BlockBatches(store.x("val_a"), store.y("val_a"), cfg.eval_batch_size, cpu),
        )
        if interrupt_at_eval is not None:
            original, calls = trainer.evaluate, []

            def failing_evaluate(*args):
                calls.append(1)
                if len(calls) == interrupt_at_eval:
                    raise KeyboardInterrupt
                return original(*args)

            trainer.evaluate = failing_evaluate
            try:
                fit(model, *batches(), cfg, cpu, out_dir)
            except KeyboardInterrupt:
                pass
            finally:
                trainer.evaluate = original
            torch.manual_seed(123)
        fit(model, *batches(), cfg, cpu, out_dir)
        return model.state_dict()

    straight = run(tmp_path / "straight")
    resumed = run(tmp_path / "resumed", interrupt_at_eval=2)
    for key in straight:
        np.testing.assert_allclose(straight[key].numpy(), resumed[key].numpy(), atol=1e-6, err_msg=key)
    assert len(pd.read_csv(tmp_path / "resumed" / "history.csv")) == 3
