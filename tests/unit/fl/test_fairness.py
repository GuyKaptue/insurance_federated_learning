"""Unit tests for ``src/insurance_fl/fl/fairness.py``."""

from __future__ import annotations

import pytest

from insurance_fl.fl.fairness import CLIENT_COUNTRY_MAP, compute_disparity, stratified_sample


def test_imports() -> None:
    assert CLIENT_COUNTRY_MAP is not None


def test_stratified_sample_basic() -> None:
    clients = ["germany_allianz", "france_axa", "belgium_ageas", "switzerland_helvetia"]
    selected, uncovered = stratified_sample(clients, num_clients=4, seed=42)
    assert len(selected) == 4
    assert len(uncovered) == 0


def test_compute_disparity_basic() -> None:
    result = compute_disparity({"A": 0.8, "B": 0.7})
    assert result["disparity"] == pytest.approx(0.1)
