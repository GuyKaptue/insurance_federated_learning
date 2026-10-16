"""Tests for insurance_fl.data.partition (exactly-once assignment, fuzzing)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from insurance_fl.data.partition import dirichlet_partition, iid_partition, stratified_split


def _assert_exact_once(parts: list[np.ndarray], indices: np.ndarray) -> None:
    """Every input index must appear exactly once across all shards."""
    all_assigned = np.sort(np.concatenate([np.asarray(p) for p in parts]))
    np.testing.assert_array_equal(all_assigned, np.sort(indices))
    assert all(len(np.asarray(p)) >= 0 for p in parts)


# --- happy path -----------------------------------------------------------
def test_dirichlet_basic_partition() -> None:
    idx = np.arange(1000)
    labels = np.random.default_rng(0).integers(0, 3, 1000)
    parts = dirichlet_partition(idx, labels, num_clients=4, alpha=0.5, seed=42)
    assert len(parts) == 4
    _assert_exact_once(parts, idx)


def test_dirichlet_determinism() -> None:
    idx = np.arange(500)
    labels = np.random.default_rng(1).integers(0, 2, 500)
    a = dirichlet_partition(idx, labels, 4, alpha=0.3, seed=7)
    b = dirichlet_partition(idx, labels, 4, alpha=0.3, seed=7)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


def test_dirichlet_lower_alpha_more_skewed() -> None:
    """alpha=0.1 must produce a more uneven client split than alpha=10."""
    idx = np.arange(4000)
    labels = np.random.default_rng(2).integers(0, 5, 4000)

    def skew(alpha: float) -> float:
        parts = dirichlet_partition(idx, labels, 4, alpha=alpha, seed=3)
        sizes = np.array([len(p) for p in parts], dtype=float)
        return float(sizes.std() / sizes.mean())

    assert skew(0.1) > skew(10.0)


def test_min_samples_per_client_honored() -> None:
    idx = np.arange(100)
    labels = np.zeros(100, dtype=int)
    parts = dirichlet_partition(
        idx, labels, num_clients=4, alpha=0.01, seed=0, min_samples_per_client=5
    )
    assert all(len(p) >= 5 for p in parts)


# --- fuzzing (property-based) ---------------------------------------------
@settings(max_examples=50, deadline=None)
@given(
    n=st.integers(min_value=1, max_value=500),
    num_clients=st.integers(min_value=1, max_value=8),
    alpha=st.floats(min_value=0.01, max_value=50.0),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    n_classes=st.integers(min_value=1, max_value=10),
)
def test_dirichlet_fuzz_exact_assignment(
    n: int, num_clients: int, alpha: float, seed: int, n_classes: int
) -> None:
    idx = np.arange(n)
    labels = np.random.default_rng(seed).integers(0, n_classes, n)
    parts = dirichlet_partition(idx, labels, num_clients=num_clients, alpha=alpha, seed=seed)
    assert len(parts) == num_clients
    for p in parts:
        assert len(p) >= 0  # no negative "splits" possible by construction
    _assert_exact_once(parts, idx)


# --- errors ---------------------------------------------------------------
def test_invalid_args_raise() -> None:
    with pytest.raises(ValueError):
        dirichlet_partition(np.arange(10), np.zeros(10, int), 2, alpha=0.0)
    with pytest.raises(ValueError):
        dirichlet_partition(np.arange(10), np.zeros(10, int), 0, alpha=1.0)
    with pytest.raises(ValueError):
        iid_partition(np.arange(10), num_clients=0)


def test_empty_indices_returns_empty_shards() -> None:
    parts = dirichlet_partition(np.array([], dtype=int), np.array([]), 3, alpha=1.0)
    assert len(parts) == 3
    assert all(len(p) == 0 for p in parts)
    parts_iid = iid_partition(np.array([], dtype=int), 3)
    assert len(parts_iid) == 3


# --- iid + stratified -----------------------------------------------------
def test_iid_partition_exact_once_and_balanced() -> None:
    idx = np.arange(101)
    parts = iid_partition(idx, num_clients=4, seed=5)
    _assert_exact_once(parts, idx)
    sizes = [len(p) for p in parts]
    assert max(sizes) - min(sizes) <= 1


def test_stratified_split_preserves_class_ratio() -> None:
    df = pd.DataFrame({"x": range(1000), "y": [0] * 700 + [1] * 300})
    train, test = stratified_split(df, test_size=0.2, stratify_col="y", seed=1)
    assert len(train) + len(test) == 1000
    for part in (train, test):
        ratio = part["y"].mean()
        assert abs(ratio - 0.3) < 0.03
