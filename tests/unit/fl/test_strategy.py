"""Unit tests for ``src/insurance_fl/fl/strategy.py`` (Layer 1 / Task 0)."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("flwr")

from insurance_fl.fl import strategy as strategy_module
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
    assert isinstance(strategy.inner, FedAvg)
    assert strategy.min_available_clients == 4


def test_strategy_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Task 4"):
        build_strategy("nonexistent_strategy")


def test_mlflow_enabled_requires_tracking_extra(monkeypatch) -> None:
    monkeypatch.setattr(strategy_module, "_MLFLOW_AVAILABLE", False)
    monkeypatch.setattr(strategy_module, "mlflow", None)
    with pytest.raises(
        RuntimeError,
        match="mlflow_enabled=True but mlflow is not installed; install insurance-federated-learning\\[tracking\\]",
    ):
        strategy_module.HybridAggregation(mlflow_enabled=True)


def test_build_strategy_hybrid_mlflow_missing_raises(monkeypatch) -> None:
    monkeypatch.setattr(strategy_module, "_MLFLOW_AVAILABLE", False)
    monkeypatch.setattr(strategy_module, "mlflow", None)
    with pytest.raises(RuntimeError, match="install insurance-federated-learning\\[tracking\\]"):
        build_strategy("hybrid", mlflow_enabled=True)
