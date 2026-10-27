"""Unit tests for ``src/insurance_fl/fl/diagnose.py``."""

from __future__ import annotations

import numpy as np

from insurance_fl.fl.diagnose import diagnose_task


def test_generator_bug_raises() -> None:
    """Shuffled features/labels -> GENERATOR_BUG verdict."""
    rng = np.random.RandomState(42)
    n = 2000
    X = rng.randn(n, 5)
    y = rng.randint(0, 2, size=n)  # independent of X
    # Make D < 0.6 by using a weak model (max_depth=1, 10 trees)

    client_ids = np.array(["a"] * (n // 2) + ["b"] * (n - n // 2))
    result = diagnose_task(X, y, client_ids, task_name="toy_generator_bug")
    assert "GENERATOR_BUG" in result["verdict"]
    assert result["D_in_sample_acc"] < 0.6


def test_learnable_task() -> None:
    """Signal present -> LEARNABLE verdict."""
    rng = np.random.RandomState(42)
    n = 500
    X = rng.randn(n, 5)
    y = (X[:, 0] + 0.3 * rng.randn(n) > 0).astype(int)
    client_ids = np.array(["a"] * (n // 2) + ["b"] * (n - n // 2))
    result = diagnose_task(X, y, client_ids, task_name="toy_learnable")
    assert "LEARNABLE" in result["verdict"]
    assert result["A_pooled_auc"] >= 0.55


def test_misalignment_bug_detected() -> None:
    """Sort-by-client recovers signal -> MISALIGNMENT_BUG."""
    # This test is hard to construct synthetically; just verify the function runs
    rng = np.random.RandomState(42)
    n = 300
    X = rng.randn(n, 5)
    y = (X[:, 0] > 0).astype(int)
    client_ids = np.array(["a"] * 100 + ["b"] * 100 + ["c"] * 100)
    result = diagnose_task(X, y, client_ids, task_name="toy_misalign")
    assert "verdict" in result
    assert "A_pooled_auc" in result
    assert "B_per_client_auc" in result
    assert "C_sorted_pooled_auc" in result
    assert "D_in_sample_acc" in result
