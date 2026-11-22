from pathlib import Path

import numpy as np
import pytest
import torch

from fragtrust.config import load_config
from fragtrust.models import build_net, count_params
from fragtrust.training.batches import BlockBatches
from fragtrust.training.trainer import TrainConfig

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "models"
EXPECTED_PARAMS = {
    "cnn_s": (0.05e6, 0.15e6),
    "cnn_m": (0.4e6, 0.8e6),
    "cnn_l": (2.5e6, 4.0e6),
    "transformer": (0.7e6, 1.2e6),
}


def test_paper_configs_have_intended_sizes_and_shapes():
    x = torch.randint(0, 256, (2, 4096), dtype=torch.uint8)
    for name, (low, high) in EXPECTED_PARAMS.items():
        model = build_net(load_config(CONFIGS / f"{name}.yaml")["model"], n_classes=75, block_size=4096).eval()
        assert low <= count_params(model) <= high, name
        with torch.no_grad():
            assert model(x).shape == (2, 75), name


def test_train_config_casts_yaml_strings_and_rejects_typos():
    assert TrainConfig.from_dict({"lr": "2e-3", "epochs": 3}).lr == pytest.approx(2e-3)
    with pytest.raises(ValueError):
        TrainConfig.from_dict({"learning_rate": 1e-3})


def test_shuffled_batches_visit_every_row_once():
    x = np.arange(103 * 8, dtype=np.uint8).reshape(103, 8)
    y = np.arange(103)
    batches = BlockBatches(x, y, 10, torch.device("cpu"), shuffle=True, seed=0)
    seen = torch.cat([yb for _, yb in batches]).numpy()
    assert sorted(seen) == list(range(103))
    first, again = [yb for _, yb in batches][0], [yb for _, yb in batches][0]
    assert not torch.equal(first, again)
    assert len(BlockBatches(x, y, 10, torch.device("cpu"), drop_last=True)) == 10
