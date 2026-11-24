"""Integration tests for fairness (stratified sampling + disparity)."""

from __future__ import annotations

import pytest

from insurance_fl.fl.fairness import (
    CLIENT_COUNTRY_MAP,
    compute_disparity,
    group_clients_by_country,
    stratified_sample,
)


def test_group_clients_by_country() -> None:
    clients = ["germany_allianz", "france_axa", "belgium_ageas", "switzerland_helvetia"]
    groups = group_clients_by_country(clients)
    assert len(groups) == 4
    for c in clients:
        assert c in CLIENT_COUNTRY_MAP


def test_stratified_sample_covers_all_countries() -> None:
    clients = ["germany_allianz", "france_axa", "belgium_ageas", "switzerland_helvetia"]
    selected, uncovered = stratified_sample(clients, num_clients=4, seed=42)
    assert len(selected) == 4
    assert len(uncovered) == 0
    # All countries represented
    countries = {CLIENT_COUNTRY_MAP[c] for c in selected}
    assert len(countries) == 4


def test_stratified_sample_no_duplicates() -> None:
    clients = ["germany_allianz", "france_axa", "belgium_ageas", "switzerland_helvetia"]
    selected, _ = stratified_sample(clients, num_clients=4, seed=42)
    assert len(selected) == len(set(selected))


def test_stratified_sample_reproducible() -> None:
    clients = ["germany_allianz", "france_axa", "belgium_ageas", "switzerland_helvetia"]
    s1, _ = stratified_sample(clients, num_clients=4, seed=42)
    s2, _ = stratified_sample(clients, num_clients=4, seed=42)
    assert s1 == s2


def test_stratified_sample_insufficient_budget() -> None:
    clients = ["germany_allianz", "france_axa", "belgium_ageas", "switzerland_helvetia"]
    # Only 2 slots but 4 countries -> some uncovered
    selected, uncovered = stratified_sample(clients, num_clients=2, seed=42)
    assert len(selected) == 2
    assert len(uncovered) > 0


def test_stratified_sample_handles_unavailable() -> None:
    clients = ["germany_allianz", "france_axa", "belgium_ageas", "switzerland_helvetia"]
    available = {"germany_allianz", "france_axa"}
    selected, uncovered = stratified_sample(clients, num_clients=2, seed=42, available=available)
    assert all(c in available for c in selected)
    # Belgium and Switzerland uncovered
    assert "Belgium" in uncovered or "Switzerland" in uncovered


def test_stratified_sample_empty_group() -> None:
    clients = ["germany_allianz"]
    selected, _uncovered = stratified_sample(clients, num_clients=1, seed=42)
    assert len(selected) == 1
    assert "germany_allianz" in selected


def test_disparity_calculation() -> None:
    metrics = {"Germany": 0.8, "France": 0.7, "Belgium": 0.75, "Switzerland": 0.85}
    result = compute_disparity(metrics)
    assert result["disparity"] == pytest.approx(0.15)
    assert result["max_country"] == "Switzerland"
    assert result["min_country"] == "France"


def test_disparity_missing_metrics() -> None:
    metrics = {"Germany": 0.8, "France": None, "Belgium": 0.75, "Switzerland": 0.85}
    result = compute_disparity(metrics)
    assert "France" in result["missing_countries"]
    assert result["disparity"] is not None


def test_disparity_empty() -> None:
    result = compute_disparity({})
    assert result["disparity"] is None
    assert result["meets_threshold"] is None


def test_disparity_zero_mean() -> None:
    metrics = {"Germany": 0.0, "France": 0.0}
    result = compute_disparity(metrics)
    assert result["disparity"] == 0.0
    assert result["threshold"] is None
    assert result["meets_threshold"] is None


def test_disparity_threshold() -> None:
    metrics = {"Germany": 0.80, "France": 0.79}
    result = compute_disparity(metrics, threshold_pct=0.05)
    assert result["disparity"] == pytest.approx(0.01)
    assert result["meets_threshold"] is True
