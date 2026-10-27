"""Unit tests for ``src/insurance_fl/fl/strategy.py`` (Layer 1 / Task 0)."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("flwr")

from insurance_fl.fl.strategy import build_strategy


def test_build_strategy_fedavg() -> None:
    from flwr.server.strategy import FedAvg

    strategy = build_strategy(
        "fedavg",
        fraction_fit=1.0,
        fraction_evaluate=0.5,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=4,
    )
    assert isinstance(strategy, FedAvg)
    assert strategy.min_available_clients == 4


def test_strategy_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Task 4"):
        build_strategy("nonexistent_strategy")
