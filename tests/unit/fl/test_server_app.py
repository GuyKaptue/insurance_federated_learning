"""Unit tests for ``src/insurance_fl/fl/server_app.py`` (Layer 1 / Task 0)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("flwr")

from insurance_fl.fl.server_app import app as server_app  # noqa: E402
from insurance_fl.fl.server_app import server_fn  # noqa: E402


def test_server_app_is_flower_serverapp() -> None:
    from flwr.server import ServerApp as FlowerServerApp

    assert isinstance(server_app, FlowerServerApp)


def test_server_fn_builds_fedavg() -> None:
    ctx = SimpleNamespace(
        run_config={
            "strategy-type": "fedavg",
            "fraction-fit": 1.0,
            "fraction-evaluate": 0.5,
            "min-fit-clients": 2,
            "min-evaluate-clients": 2,
            "min-available-clients": 4,
            "num-server-rounds": 2,
        }
    )
    components = server_fn(ctx)
    from flwr.server.strategy import FedAvg

    assert isinstance(components.strategy, FedAvg)
    assert components.config.num_rounds == 2
