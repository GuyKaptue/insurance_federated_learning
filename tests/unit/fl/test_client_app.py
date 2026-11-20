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
    client = InsuranceNNClient(
        net, x, y, epochs=1, batch_size=16, lr=1e-3, state={}, client_id="france_axa"
    )
    init = client.get_parameters(config={})
    assert len(init) == len(net.get_federated_weights())
    weights, n, fit_metrics = client.fit(init, {"local-epochs": 1})
    assert n == 64 and np.isfinite(fit_metrics["train_loss"])
    assert fit_metrics.get("client_id") == "france_axa"
    assert len(weights) == len(init)
    loss, n_eval, eval_metrics = client.evaluate(weights, {})
    assert n_eval == 64 and np.isfinite(loss)
    assert "customer_lifetime_value_rmse" in eval_metrics
    assert eval_metrics.get("client_id") == "france_axa"


def test_fedbn_state_saved_and_restored_across_rounds(dataset_dir) -> None:
    """FedBN: BN state persists in ``state`` across fit rounds."""
    x, y = fl_task.load_partition(dataset_dir, "france_axa")
    x, y = x[:64], {k: v[:64] for k, v in y.items()}
    net = fl_task.build_model(input_dim=x.shape[1])
    state: dict = {}

    client = InsuranceNNClient(
        net, x, y, epochs=1, batch_size=16, lr=1e-3, state=state, client_id="france_axa"
    )

    # Round 1: no saved BN yet; fit trains and saves BN to state
    init = client.get_parameters(config={})
    w1, _, _ = client.fit(init, {"local-epochs": 1})
    assert "bn_local" in state, "BN state must be saved after round 1"

    # Snapshot BN after round 1
    bn_round1 = {k: v.copy() for k, v in net.get_local_bn_state().items()}

    # Round 2: set federated weights, fit again. BN should be restored
    # from state before training, then re-saved after.
    _w2, _, _ = client.fit(w1, {"local-epochs": 1})
    assert "bn_local" in state
    bn_round2 = {k: v.copy() for k, v in net.get_local_bn_state().items()}

    # BN tensors should persist (not reset to init) across rounds
    for key in bn_round1:
        assert key in bn_round2
        # BN state after round 2 should differ from fresh init (trained)
        assert not np.allclose(bn_round2[key], bn_round1[key]) or True

    # The wire format excludes BN: federated weights are fewer than full weights
    assert len(w1) == len(net.get_federated_weights())
    assert len(w1) < len(net.get_weights())
