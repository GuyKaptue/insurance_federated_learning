"""Unit tests for ``src/insurance_fl/fl/task.py`` (Layer 1 / Task 0)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("flwr")

from insurance_fl.fl import task as fl_task  # noqa: E402


@pytest.fixture(scope="module")
def dataset_dir(tmp_path_factory: pytest.TempPathFactory):
    """Generate the smoke dataset once for this module (~2 s)."""
    return fl_task.prepare_dataset(tmp_path_factory.mktemp("fl_task0"), 512, 42)


def test_prepare_is_idempotent_and_loads(dataset_dir) -> None:
    assert fl_task.prepare_dataset(dataset_dir, 512, 42) == dataset_dir
    for client_id in fl_task.CLIENT_IDS:
        x, y = fl_task.load_partition(dataset_dir, client_id, split="train")
        assert x.dtype == np.float32 and x.shape[0] > 100
        assert set(y.keys()) == {
            "claim_risk_score",
            "customer_lifetime_value",
            "will_renew",
            "is_high_risk",
            "customer_segment",
        }
        assert len(y["claim_risk_score"]) == x.shape[0]
        assert not np.isnan(x).any()


def test_load_partition_missing_data_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="--prepare"):
        fl_task.load_partition(tmp_path, "germany_allianz")


def test_train_and_evaluate_smoke(dataset_dir) -> None:
    x, y = fl_task.load_partition(dataset_dir, "germany_allianz")
    x, y = x[:64], {k: v[:64] for k, v in y.items()}
    net = fl_task.build_model(input_dim=x.shape[1])
    loss = fl_task.train(net, x, y, epochs=1, batch_size=16, lr=1e-3)
    assert np.isfinite(loss)
    metrics = fl_task.evaluate(net, x, y)
    assert np.isfinite(metrics["loss"])
    assert "claim_risk_score_rmse" in metrics
    assert "customer_segment_acc" in metrics
