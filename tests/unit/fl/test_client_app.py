"""Unit tests for ``src/insurance_fl/fl/client_app.py`` (Layer 1 / Task 0)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("flwr")

from insurance_fl.fl import task as fl_task  # noqa: E402
from insurance_fl.fl.client_app import InsuranceNNClient  # noqa: E402
from insurance_fl.fl.client_app import app as client_app  # noqa: E402


@pytest.fixture(scope="module")
def dataset_dir(tmp_path_factory: pytest.TempPathFactory):
    return fl_task.prepare_dataset(tmp_path_factory.mktemp("fl_ca"), 512, 42)


def test_client_app_is_flower_clientapp() -> None:
    from flwr.client import ClientApp as FlowerClientApp

    assert isinstance(client_app, FlowerClientApp)
    assert len(fl_task.CLIENT_IDS) == 4


def test_numpyclient_fit_evaluate_roundtrip(dataset_dir) -> None:
    x, y = fl_task.load_partition(dataset_dir, "france_axa")
    x, y = x[:64], {k: v[:64] for k, v in y.items()}
    net = fl_task.build_model(input_dim=x.shape[1])
    client = InsuranceNNClient(net, x, y, epochs=1, batch_size=16, lr=1e-3)
    init = client.get_parameters(config={})
    assert len(init) == len(net.get_weights())
    weights, n, fit_metrics = client.fit(init, {"local-epochs": 1})
    assert n == 64 and np.isfinite(fit_metrics["train_loss"])
    assert len(weights) == len(init)
    loss, n_eval, eval_metrics = client.evaluate(weights, {})
    assert n_eval == 64 and np.isfinite(loss)
    assert "customer_lifetime_value_rmse" in eval_metrics
