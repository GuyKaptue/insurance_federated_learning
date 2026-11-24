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


def test_fedbn_state_saved_and_restored_across_rounds(dataset_dir, monkeypatch) -> None:
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

    # Disturb the in-memory BN state to prove that the saved client-local state
    # is restored before the next fit. Keep training a no-op to isolate restore.
    net.set_local_bn_state({key: np.zeros_like(value) for key, value in bn_round1.items()})
    restored_at_train = {}

    def capture_bn_before_training(model, *_args, **_kwargs):
        restored_at_train.update(
            {key: value.copy() for key, value in model.get_local_bn_state().items()}
        )
        return 0.0

    monkeypatch.setattr(fl_task, "train", capture_bn_before_training)
    _w2, _, _ = client.fit(w1, {"local-epochs": 1})
    assert "bn_local" in state
    bn_round2 = {k: v.copy() for k, v in net.get_local_bn_state().items()}

    # BN tensors are restored from the client-local checkpoint and remain
    # unchanged when the federated weights are applied.
    for key in bn_round1:
        assert np.allclose(restored_at_train[key], bn_round1[key])
        assert key in bn_round2
        assert np.allclose(bn_round2[key], bn_round1[key])

    # The wire format excludes BN: federated weights are fewer than full weights
    assert len(w1) == len(net.get_federated_weights())
    assert len(w1) < len(net.get_weights())


def test_federated_weight_update_preserves_local_batchnorm_state(dataset_dir) -> None:
    x, _ = fl_task.load_partition(dataset_dir, "france_axa")
    net = fl_task.build_model(input_dim=x.shape[1])
    local_bn = {
        key: np.full_like(value, fill_value=0.25 + idx)
        for idx, (key, value) in enumerate(net.get_local_bn_state().items())
    }
    net.set_local_bn_state(local_bn)

    global_weights = [weight + 0.5 for weight in net.get_federated_weights()]
    net.set_federated_weights(global_weights)

    restored_bn = net.get_local_bn_state()
    assert restored_bn.keys() == local_bn.keys()
    for key, expected in local_bn.items():
        assert np.array_equal(restored_bn[key], expected)


def test_client_local_batchnorm_states_remain_isolated(dataset_dir) -> None:
    x, _ = fl_task.load_partition(dataset_dir, "germany_allianz")
    global_model = fl_task.build_model(input_dim=x.shape[1])
    global_weights = [weight + 0.1 for weight in global_model.get_federated_weights()]
    client_states = []

    for client_index, client_id in enumerate(fl_task.CLIENT_IDS[:2]):
        client_x, _ = fl_task.load_partition(dataset_dir, client_id)
        client_model = fl_task.build_model(input_dim=client_x.shape[1])
        local_bn = {
            key: np.full_like(value, fill_value=client_index + 0.25)
            for key, value in client_model.get_local_bn_state().items()
        }
        client_model.set_local_bn_state(local_bn)
        client_model.set_federated_weights(global_weights)
        client_states.append(client_model.get_local_bn_state())

        for key, expected in local_bn.items():
            assert np.array_equal(client_states[-1][key], expected)

    assert client_states[0].keys() == client_states[1].keys()
    assert any(
        not np.array_equal(client_states[0][key], client_states[1][key]) for key in client_states[0]
    )
