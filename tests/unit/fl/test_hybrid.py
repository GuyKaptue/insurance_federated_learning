"""Unit tests for ``src/insurance_fl/fl/hybrid.py``."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from insurance_fl.fl.hybrid import (
    load_beta_diagnostics,
    load_beta_weights,
    optimize_beta,
    predict_hybrid,
    save_beta_weights,
)


def test_optimize_beta_bounds() -> None:
    rng = np.random.RandomState(42)
    nn = rng.randn(50)
    booster = rng.randn(50)
    y = rng.randn(50)
    beta, _diag = optimize_beta(nn, booster, y)
    if beta is not None:
        assert 0.0 <= beta <= 1.0


def test_optimize_beta_prefers_better_model() -> None:
    # NN is perfect, booster is noise -> beta should be high
    y = np.arange(50, dtype=float)
    nn = y.copy()
    booster = np.zeros(50)
    beta, _diag = optimize_beta(nn, booster, y)
    assert beta > 0.5


def test_predict_hybrid() -> None:
    nn = np.array([1.0, 2.0, 3.0])
    booster = np.array([0.0, 0.0, 0.0])
    result = predict_hybrid(nn, booster, 0.5)
    np.testing.assert_allclose(result, [0.5, 1.0, 1.5])


def test_save_load_beta() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "beta.json"
        weights = {"task_a": 0.7, "task_b": 0.3}
        save_beta_weights(path, weights, metadata={"version": "1.0"})
        loaded = load_beta_weights(path)
        assert loaded == weights


def test_optimize_beta_deterministic() -> None:
    rng = np.random.RandomState(42)
    nn = rng.randn(30)
    booster = rng.randn(30)
    y = rng.randn(30)
    b1, _ = optimize_beta(nn, booster, y)
    b2, _ = optimize_beta(nn, booster, y)
    assert b1 == b2


def test_optimize_beta_boundary_zero() -> None:
    # Booster is perfect, NN is noise -> beta should be near 0
    y = np.arange(30, dtype=float)
    nn = np.zeros(30)
    booster = y.copy()
    beta, diag = optimize_beta(nn, booster, y)
    assert beta <= 0.1
    assert diag["boundary_optimum"] is True


def test_optimize_beta_boundary_one() -> None:
    # NN is perfect, booster is noise -> beta should be near 1
    y = np.arange(30, dtype=float)
    nn = y.copy()
    booster = np.zeros(30)
    beta, diag = optimize_beta(nn, booster, y)
    assert beta >= 0.9
    assert diag["boundary_optimum"] is True


def test_optimize_beta_rejects_mismatched() -> None:
    with pytest.raises(ValueError):
        optimize_beta(np.array([1.0, 2.0]), np.array([1.0]), np.array([1.0, 2.0]))


def test_optimize_beta_rejects_nonfinite() -> None:
    with pytest.raises(ValueError):
        optimize_beta(np.array([1.0, np.nan]), np.array([1.0, 2.0]), np.array([1.0, 2.0]))


def test_save_load_beta_diagnostics() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "beta.json"
        weights = {"task_a": 0.7}
        diag = {"beta": 0.7, "loss_after_fusion": 0.1, "boundary_optimum": False}
        save_beta_weights(path, weights, diagnostics=diag)
        loaded_diag = load_beta_diagnostics(path)
        assert loaded_diag["beta"] == 0.7
        assert loaded_diag["loss_after_fusion"] == 0.1


def test_validate_beta_valid_interior() -> None:
    from insurance_fl.fl.hybrid import validate_beta

    assert validate_beta(0.6) == 0.6


def test_validate_beta_boundary_zero() -> None:
    from insurance_fl.fl.hybrid import validate_beta

    assert validate_beta(0.0) == 0.0


def test_validate_beta_boundary_one() -> None:
    from insurance_fl.fl.hybrid import validate_beta

    assert validate_beta(1.0) == 1.0


def test_validate_beta_rejects_negative() -> None:
    from insurance_fl.fl.hybrid import validate_beta

    with pytest.raises(ValueError):
        validate_beta(-0.01)


def test_validate_beta_rejects_above_one() -> None:
    from insurance_fl.fl.hybrid import validate_beta

    with pytest.raises(ValueError):
        validate_beta(1.01)


def test_validate_beta_rejects_nan() -> None:
    from insurance_fl.fl.hybrid import validate_beta

    with pytest.raises(ValueError):
        validate_beta(float("nan"))


def test_validate_beta_rejects_inf() -> None:
    from insurance_fl.fl.hybrid import validate_beta

    with pytest.raises(ValueError):
        validate_beta(float("inf"))


def test_validate_beta_rejects_bool() -> None:
    from insurance_fl.fl.hybrid import validate_beta

    with pytest.raises(ValueError):
        validate_beta(True)


def test_multiclass_stacking_concept() -> None:
    """Document and test multiclass stacking: per-class NNLS, row normalization."""
    import numpy as np
    from scipy.optimize import nnls

    rng = np.random.RandomState(42)
    n = 100
    K = 3
    # OOF predictions from 3 boosters, shape (n, K) each
    oof1 = rng.rand(n, K)
    oof2 = rng.rand(n, K)
    oof3 = rng.rand(n, K)
    # Normalize to probabilities
    for arr in (oof1, oof2, oof3):
        arr[:] = arr / arr.sum(axis=1, keepdims=True)
    y_true = rng.randint(0, K, size=n)

    # Per-class NNLS
    alphas = np.zeros((3, K))  # 3 boosters, K classes
    for c in range(K):
        A = np.column_stack([oof1[:, c], oof2[:, c], oof3[:, c]])
        b = (y_true == c).astype(float)
        w, _ = nnls(A, b)
        alphas[:, c] = w

    # Stack
    stack = np.zeros((n, K))
    for c in range(K):
        stack[:, c] = (
            alphas[0, c] * oof1[:, c] + alphas[1, c] * oof2[:, c] + alphas[2, c] * oof3[:, c]
        )
    # Normalize rows
    stack = stack / stack.sum(axis=1, keepdims=True)

    # Verify row sums
    np.testing.assert_allclose(stack.sum(axis=1), 1.0, atol=1e-10)
    # Verify nonnegative
    assert np.all(stack >= 0)
    # Verify finite
    assert np.all(np.isfinite(stack))


def test_beta_uses_real_booster_predictions() -> None:
    """Regression test: beta optimization uses booster predictions, not NN proxy."""
    import numpy as np

    from insurance_fl.fl.hybrid import optimize_beta

    # NN is worse than booster
    y = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0] * 10, dtype=float)
    nn_pred = np.full(60, 0.5)  # constant (bad)
    booster_pred = y.copy()  # perfect (good)
    beta, _diag = optimize_beta(nn_pred, booster_pred, y, task_type="binary")
    assert beta <= 0.1, f"Expected beta near 0 (booster better), got {beta}"
    assert _diag["loss_before_booster"] < _diag["loss_before_nn"]


def test_beta_nn_better() -> None:
    """Regression test: when NN is better, beta near 1."""
    import numpy as np

    from insurance_fl.fl.hybrid import optimize_beta

    y = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0] * 10, dtype=float)
    nn_pred = y.copy()  # perfect (good)
    booster_pred = np.full(60, 0.5)  # constant (bad)
    beta, _diag = optimize_beta(nn_pred, booster_pred, y, task_type="binary")
    assert beta >= 0.9, f"Expected beta near 1 (NN better), got {beta}"
    assert _diag["loss_before_nn"] < _diag["loss_before_booster"]


def test_beta_interior() -> None:
    """Regression test: interior beta when both contribute."""
    import numpy as np

    from insurance_fl.fl.hybrid import optimize_beta

    rng = np.random.RandomState(42)
    y = rng.randint(0, 2, size=100).astype(float)
    nn_pred = np.clip(y + rng.randn(100) * 0.3, 0, 1)
    booster_pred = np.clip(y + rng.randn(100) * 0.3, 0, 1)
    beta, _diag = optimize_beta(nn_pred, booster_pred, y, task_type="binary")
    assert 0.0 <= beta <= 1.0
    assert np.isfinite(beta)


def test_beta_uses_task_loss_not_mse() -> None:
    """Regression test: binary beta uses log-loss, differs from MSE."""
    import numpy as np

    from insurance_fl.fl.hybrid import optimize_beta

    # Crafted case where MSE and log-loss disagree
    y = np.array([0.0, 1.0] * 25, dtype=float)
    nn_pred = np.array([0.9, 0.1] * 25, dtype=float)  # wrong (reversed)
    booster_pred = np.array([0.1, 0.9] * 25, dtype=float)  # correct
    beta, _diag = optimize_beta(nn_pred, booster_pred, y, task_type="binary")
    # Booster is better, so beta should be near 0
    assert beta < 0.5, f"Expected beta < 0.5 (booster better), got {beta}"
    assert _diag["loss_function"] == "log_loss"


def test_uninformative_task_gets_no_beta_claim() -> None:
    import numpy as np

    from insurance_fl.fl.hybrid import STATUS_UNINFORMATIVE, optimize_beta, predict_hybrid

    rng = np.random.default_rng(0)
    n = 200
    y = rng.integers(0, 2, size=n)
    prior = float(y.mean())
    nn = np.full(n, prior)
    booster = np.full(n, prior)
    beta_claim, diag = optimize_beta(
        nn, booster, y, task_type="binary", task_name="toy_uninformative"
    )
    assert diag["task_status"] == STATUS_UNINFORMATIVE
    assert diag["beats_baseline"] is False
    assert diag["usable_beta"] is False
    assert beta_claim is None
    assert diag["beta_claim"] is None
    assert 0.0 <= diag["beta"] <= 1.0
    with pytest.raises(ValueError):
        predict_hybrid(nn, booster, beta_claim, task_status=diag["task_status"])


def test_informative_task_keeps_beta_claim() -> None:
    import numpy as np

    from insurance_fl.fl.hybrid import STATUS_INFORMATIVE, optimize_beta, validate_beta

    rng = np.random.default_rng(1)
    n = 400
    x = rng.normal(size=n)
    y = (x + 0.5 * rng.normal(size=n) > 0).astype(int)
    nn = np.clip(1 / (1 + np.exp(-2.0 * x)), 0.01, 0.99)
    booster = np.clip(1 / (1 + np.exp(-0.5 * x)), 0.01, 0.99)
    beta_claim, diag = optimize_beta(
        nn, booster, y, task_type="binary", task_name="toy_informative"
    )
    assert diag["task_status"] == STATUS_INFORMATIVE
    assert diag["beats_baseline"] is True
    assert beta_claim is not None
    assert validate_beta(beta_claim) == beta_claim


def test_fallback_guard() -> None:
    import numpy as np

    from insurance_fl.fl.stacking import fit_stacking_meta, validate_alpha

    # Forced bad stack: one booster is perfect, others are noise
    rng = np.random.RandomState(42)
    n = 50
    y = rng.randn(n)
    oof_good = y + rng.randn(n) * 0.01
    oof_bad1 = rng.randn(n)
    oof_bad2 = rng.randn(n)
    alpha = fit_stacking_meta([oof_good, oof_bad1, oof_bad2], y)
    validated = validate_alpha(alpha)
    # The good booster should have the highest weight
    assert np.argmax(validated) == 0, f"Expected booster 0 to have highest weight, got {validated}"
    assert validated[0] > 0
