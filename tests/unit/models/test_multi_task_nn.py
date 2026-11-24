"""Tests for insurance_fl.models.multi_task_nn (weights interface, losses)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from insurance_fl.fl.task import build_model, train
from insurance_fl.models.multi_task_nn import DEFAULT_HIDDEN_DIMS, DEFAULT_TASK_TYPES, MultiTaskNN

BATCH = 16
INPUT_DIM = 12


@pytest.fixture
def model() -> MultiTaskNN:
    torch.manual_seed(0)
    return MultiTaskNN(input_dim=INPUT_DIM, hidden_dims=[32, 16], dropout_rate=0.1)


@pytest.fixture
def inputs() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(BATCH, INPUT_DIM)


def _targets() -> dict:
    torch.manual_seed(2)
    return {
        "claim_risk_score": torch.randn(BATCH) * 20 + 50,
        "customer_lifetime_value": torch.rand(BATCH) * 100000 + 1000,
        "will_renew": torch.randint(0, 2, (BATCH,)).float(),
        "is_high_risk": torch.randint(0, 2, (BATCH,)).float(),
        "customer_segment": torch.randint(0, 3, (BATCH,)),
    }


# --- structure ------------------------------------------------------------
def test_default_five_tasks(model: MultiTaskNN) -> None:
    assert set(model.task_types) == set(DEFAULT_TASK_TYPES)
    assert len(model.heads) == 5


def test_forward_returns_raw_logits_shapes(model: MultiTaskNN, inputs: torch.Tensor) -> None:
    out = model(inputs)
    assert out["claim_risk_score"].shape == (BATCH,)
    assert out["customer_lifetime_value"].shape == (BATCH,)
    assert out["will_renew"].shape == (BATCH,)
    assert out["is_high_risk"].shape == (BATCH,)
    assert out["customer_segment"].shape == (BATCH, 3)


def test_predict_applies_activations(model: MultiTaskNN, inputs: torch.Tensor) -> None:
    preds = model.predict(inputs)
    assert torch.all((preds["will_renew"] >= 0) & (preds["will_renew"] <= 1))
    assert torch.all((preds["is_high_risk"] >= 0) & (preds["is_high_risk"] <= 1))
    probs = preds["customer_segment"]
    torch.testing.assert_close(probs.sum(dim=-1), torch.ones(BATCH))


def test_invalid_constructor_args() -> None:
    with pytest.raises(ValueError):
        MultiTaskNN(input_dim=0)
    with pytest.raises(ValueError):
        MultiTaskNN(input_dim=4, dropout_rate=1.5)
    with pytest.raises(ValueError):
        MultiTaskNN(input_dim=4, task_types={"x": "clustering"})


# --- losses ---------------------------------------------------------------
def test_losses_finite_and_backward(model: MultiTaskNN, inputs: torch.Tensor) -> None:
    logits = model(inputs)
    losses = model.get_losses(logits, _targets())
    assert set(losses) == set(DEFAULT_TASK_TYPES)
    for name, val in losses.items():
        assert torch.isfinite(val), name
    total = model.get_total_loss(losses)
    assert torch.isfinite(total)
    total.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "backward produced no gradients"


def test_total_loss_respects_weights(model: MultiTaskNN) -> None:
    losses = {k: torch.tensor(1.0) for k in model.task_types}
    total = model.get_total_loss(losses, dict.fromkeys(model.task_types, 2.0))
    assert float(total) == pytest.approx(2.0 * len(losses))


def test_get_losses_skips_missing_targets(model: MultiTaskNN, inputs: torch.Tensor) -> None:
    logits = model(inputs)
    losses = model.get_losses(logits, {"will_renew": torch.zeros(BATCH)})
    assert set(losses) == {"will_renew"}


# --- FL weight interface ---------------------------------------------------
def test_get_set_weights_roundtrip(model: MultiTaskNN) -> None:
    w1 = model.get_weights()
    clone = MultiTaskNN(input_dim=INPUT_DIM, hidden_dims=[32, 16], dropout_rate=0.1)
    clone.set_weights(w1)
    w2 = clone.get_weights()
    assert len(w1) == len(w2)
    for a, b in zip(w1, w2):
        np.testing.assert_array_equal(a, b)


def test_set_weights_count_and_shape_mismatch_raise(model: MultiTaskNN) -> None:
    with pytest.raises(ValueError):
        model.set_weights(model.get_weights()[:-1])
    bad = list(model.get_weights())
    bad[0] = np.zeros((3, 3))
    with pytest.raises(ValueError):
        model.set_weights(bad)


def test_weights_are_numpy_copies(model: MultiTaskNN) -> None:
    w = model.get_weights()
    assert all(isinstance(x, np.ndarray) for x in w)
    w[0][:] = 0.0  # mutating must not corrupt model
    assert not torch.all(model.shared_backbone[0].weight == 0)


# --- persistence -----------------------------------------------------------
def test_save_load_roundtrip(model: MultiTaskNN, inputs: torch.Tensor, tmp_path) -> None:
    model.eval()
    before = model(inputs)
    model.save(tmp_path / "m")
    loaded = MultiTaskNN.load(tmp_path / "m")
    loaded.eval()
    after = loaded(inputs)
    for k in before:
        torch.testing.assert_close(before[k], after[k])


# --- determinism -----------------------------------------------------------
def test_same_seed_same_weights() -> None:
    torch.manual_seed(123)
    m1 = MultiTaskNN(input_dim=8, hidden_dims=[16])
    torch.manual_seed(123)
    m2 = MultiTaskNN(input_dim=8, hidden_dims=[16])
    for a, b in zip(m1.get_weights(), m2.get_weights()):
        np.testing.assert_array_equal(a, b)


# --- spec alignment: 256 -> 128 -> 64 encoder ------------------------------
def test_spec_hidden_dims_constant() -> None:
    assert DEFAULT_HIDDEN_DIMS == [256, 128, 64]


def test_spec_encoder_33_features_five_outputs() -> None:
    """Spec: 33 input features -> 5 task outputs (2 regression + 2 binary + 1 multi)."""
    model = MultiTaskNN(input_dim=33, hidden_dims=DEFAULT_HIDDEN_DIMS)
    x = torch.randn(8, 33)
    out = model(x)
    assert out["claim_risk_score"].shape == (8,)
    assert out["customer_lifetime_value"].shape == (8,)
    assert out["will_renew"].shape == (8,)
    assert out["is_high_risk"].shape == (8,)
    assert out["customer_segment"].shape == (8, 3)


# --- Kendall uncertainty weighting -----------------------------------------
def test_kendall_zero_init_equals_equal_weight_sum(
    model: MultiTaskNN, inputs: torch.Tensor
) -> None:
    """Zero-init log_vars => Kendall loss == equal-weight sum at start."""
    torch.manual_seed(0)
    logits = model(inputs)
    losses = model.get_losses(logits, _targets())
    kendall = model.get_kendall_total_loss(losses)
    equal = model.get_total_loss(losses)
    assert float(kendall) == pytest.approx(float(equal))


def test_kendall_loss_changes_with_log_vars(model: MultiTaskNN, inputs: torch.Tensor) -> None:
    """Non-zero log_vars change the weighted total loss."""
    torch.manual_seed(0)
    logits = model(inputs)
    losses = model.get_losses(logits, _targets())
    base = model.get_kendall_total_loss(losses)
    with torch.no_grad():
        model.log_vars.fill_(1.0)
    shifted = model.get_kendall_total_loss(losses)
    assert float(shifted) != pytest.approx(float(base))


def test_kendall_uncertainty_weights(model: MultiTaskNN) -> None:
    """uncertainty_weights returns exp(-log_var) per task (1.0 at zero-init)."""
    weights = model.uncertainty_weights()
    assert set(weights) == set(model.task_types)
    assert len(weights) == 5
    # Zero-init log_vars => exp(-0) == 1.0 for every task
    for w in weights.values():
        assert w == pytest.approx(1.0)


# --- FedBN ----------------------------------------------------------------
def test_federated_weights_exclude_batchnorm(model: MultiTaskNN) -> None:
    """get_federated_weights returns only non-BN tensors."""
    bn_keys = model.local_bn_keys()
    assert bn_keys, "model should have BatchNorm layers"
    fed = model.get_federated_weights()
    all_w = model.get_weights()
    # Federated subset is strictly smaller (BN excluded)
    assert len(fed) < len(all_w)
    # Every federated key is non-BN
    fed_keys = model.federated_keys()
    assert not any("bn" in k.lower() for k in fed_keys)


def test_set_federated_weights_leaves_bn_untouched(model: MultiTaskNN) -> None:
    """set_federated_weights updates non-BN keys only; BN keys stay as-is."""
    bn_before = model.get_local_bn_state()
    fed = model.get_federated_weights()
    # Perturb federated weights slightly
    perturbed = [w + 0.01 for w in fed]
    model.set_federated_weights(perturbed)
    bn_after = model.get_local_bn_state()
    for key in bn_before:
        np.testing.assert_array_equal(bn_before[key], bn_after[key])


def test_local_bn_state_roundtrip(model: MultiTaskNN) -> None:
    """get_local_bn_state / set_local_bn_state preserve BN tensors exactly."""
    bn_before = model.get_local_bn_state()
    assert bn_before, "model should have BatchNorm state"
    # Mutate BN tensors to zero
    state = model.state_dict()
    for key in bn_before:
        state[key] = torch.zeros_like(state[key])
    model.load_state_dict(state, strict=False)
    for key in bn_before:
        assert torch.all(model.state_dict()[key] == 0)
    # Restore
    model.set_local_bn_state(bn_before)
    for key in bn_before:
        torch.testing.assert_close(model.state_dict()[key], torch.as_tensor(bn_before[key]))


def test_nn_checkpoint_roundtrip() -> None:
    net = build_model(input_dim=5)
    x = np.random.randn(20, 5).astype(np.float32)
    y = {"claim_risk_score": np.random.randn(20).astype(np.float32)}
    # Add minimal tasks
    for t, tt in DEFAULT_TASK_TYPES.items():
        if t not in y:
            if tt == "regression":
                y[t] = np.random.randn(20).astype(np.float32)
            elif tt == "binary":
                y[t] = np.random.randint(0, 2, size=20).astype(np.float32)
            else:
                y[t] = np.random.randint(0, 3, size=20).astype(np.float32)

    train(net, x, y, epochs=2, batch_size=8, lr=0.001)
    net.eval()

    with torch.no_grad():
        pred_before = net.forward(torch.as_tensor(x[:5].astype(np.float32)))
        pred_before = {k: v.detach().numpy().copy() for k, v in pred_before.items()}  # type: ignore[misc]

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "nn_checkpoint"
        net.save(path)

        # Reload
        loaded = type(net).load(path)
        loaded.eval()

        with torch.no_grad():
            pred_after = loaded.forward(torch.as_tensor(x[:5].astype(np.float32)))
            pred_after = {k: v.detach().numpy() for k, v in pred_after.items()}  # type: ignore[misc]

        for task in pred_before:
            np.testing.assert_allclose(pred_before[task], pred_after[task], rtol=1e-5)


def test_nn_checkpoint_missing_fails() -> None:
    from insurance_fl.models.multi_task_nn import MultiTaskNN

    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(FileNotFoundError):
            MultiTaskNN.load(Path(tmpdir) / "nonexistent")


def test_nn_checkpoint_metadata() -> None:
    net = build_model(input_dim=5)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "nn_checkpoint"
        net.save(path)
        assert (path / "model.pt").exists()
        assert (path / "metadata.json").exists()
        loaded = type(net).load(path)
        assert loaded.input_dim == 5
