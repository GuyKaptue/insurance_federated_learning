"""Unit tests for ``src/insurance_fl/fl/server_app.py`` (Layer 1 / Task 0)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("flwr")

from insurance_fl.fl.privacy import RdpAccountantWrapper, account_round
from insurance_fl.fl.server_app import app as server_app
from insurance_fl.fl.server_app import privacy_config_from_run_config, server_fn


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

    assert isinstance(components.strategy.inner, FedAvg)
    assert components.config.num_rounds == 2


def test_server_fn_resumes_privacy_accounting_state(tmp_path) -> None:
    state_path = tmp_path / "privacy-state.json"
    audit_path = tmp_path / "audit.jsonl"
    run_config = {
        "strategy-type": "fedavg",
        "fraction-fit": 1.0,
        "fraction-evaluate": 0.5,
        "min-fit-clients": 2,
        "min-evaluate-clients": 2,
        "min-available-clients": 4,
        "num-server-rounds": 5,
        "privacy-enabled": "true",
        "privacy-local-dp-clip-norm": 1.0,
        "privacy-local-dp-noise-std": 1.0,
        "privacy-delta": 1e-5,
        "privacy-max-epsilon": 8.0,
        "privacy-state-path": str(state_path),
        "privacy-resume": "false",
        "audit-path": str(audit_path),
    }
    accountant = RdpAccountantWrapper(privacy_config_from_run_config(run_config))
    account_round(accountant, state_path, server_round=1)
    run_config["privacy-resume"] = "true"

    components = server_fn(SimpleNamespace(run_config=run_config))

    assert components.strategy.accountant.rounds == 1
    assert components.strategy.accountant.resumed_from_round == 1
    assert '"accounting_status": "resumed"' in audit_path.read_text()


def test_privacy_resume_requires_state_path() -> None:
    with pytest.raises(ValueError, match="privacy-state-path"):
        server_fn(SimpleNamespace(run_config={"privacy-resume": "true"}))


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
