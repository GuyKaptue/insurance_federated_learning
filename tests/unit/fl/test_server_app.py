"""Unit tests for ``src/insurance_fl/fl/server_app.py`` (Layer 1 / Task 0)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("flwr")

from insurance_fl.fl.server_app import app as server_app
from insurance_fl.fl.server_app import server_fn


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


def test_server_app_main_secagg_enabled_fails_closed(monkeypatch):
    """SecAgg+ runtime path must fail explicitly (no hang, no plaintext fallback)."""
    from types import SimpleNamespace

    from flwr.common import Context

    from insurance_fl.fl import server_app

    # minimal context mock
    rc = {
        "dp-enabled": False,
        "dp-noise-multiplier": 0.1,
        "dp-max-grad-norm": 1.0,
        "audit-path": "logs/audit.jsonl",
        "run-id": "test-secagg-fail-closed",
        "strategy-type": "fedavg",
        "fraction-fit": 1.0,
        "fraction-evaluate": 0.0,
        "min-fit-clients": 4,
        "min-evaluate-clients": 4,
        "min-available-clients": 4,
        "num-server-rounds": 1,
        "secagg-enabled": "true",
        "require-all-clients": "false",
    }
    context = Context(run_id=1, node_id=0, node_config={}, state={}, run_config=rc)
    grid = SimpleNamespace()

    with pytest.raises(NotImplementedError, match=r"SecAgg\+ runtime orchestration"):
        server_app.main(grid, context)
