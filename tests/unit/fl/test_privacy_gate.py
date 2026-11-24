"""Unit tests for the privacy pre-release budget gate (Stage C).

Covers:
  * budget refusal at dispatch (configure_fit returns [] before any release);
  * strategy-level invocation of the gate (account_round called before dispatch);
  * no unauthorized aggregation after refusal (accountant rounds do not increase);
  * crash between dispatch and aggregation (resume preserves rounds_spent);
  * privacy_budget_exhausted emitted once; resumed_from_round recorded;
  * corrupt / truncated / tampered state fail closed;
  * fresh run onto existing state without resume=true raises;
  * config mismatch on resume raises;
  * interrupted-and-resumed accounting equals uninterrupted accounting;
  * atomic state write (old state intact if replace fails).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest

pytest.importorskip("torch")
pytest.importorskip("flwr")


from flwr.common import Parameters

from insurance_fl.fl.privacy import (
    PrivacyConfig,
    PrivacyStateError,
    RdpAccountantWrapper,
    account_round,
)
from insurance_fl.fl.strategy import build_strategy


def _make_accountant(max_epsilon: float, noise_multiplier: float = 1.0) -> RdpAccountantWrapper:
    cfg = PrivacyConfig(
        max_epsilon=max_epsilon,
        target_delta=1e-5,
        noise_multiplier=noise_multiplier,
        sampling_rate=1.0,
    )
    return RdpAccountantWrapper(cfg)


def _make_client_manager(n: int = 1) -> SimpleNamespace:
    nodes = [SimpleNamespace(cid=f"node-{i}") for i in range(n)]
    return SimpleNamespace(
        all=lambda: {f"node-{i}": n for i, n in enumerate(nodes)},
        num_available=lambda: n,
        sample=lambda num_clients, min_num_clients=None: nodes[:num_clients],
    )


class _AuditSpy:
    """Collects audit logger log_event calls."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def log_event(self, event_type: str, payload: dict) -> None:
        self.events.append((event_type, dict(payload)))

    def log_privacy_event(self, round: int | None = None, **kwargs: Any) -> None:
        self.events.append(("privacy_budget_exhausted", dict(kwargs, round=round)))


# ---------------------------------------------------------------------------
# C5-1: budget refusal at dispatch
# ---------------------------------------------------------------------------


def test_budget_refusal_at_first_round() -> None:
    # A budget so small that even the first release would exceed it.
    max_epsilon = 0.05
    acc = _make_accountant(max_epsilon)
    strategy = build_strategy(
        "hybrid",
        accountant=acc,
        max_epsilon=max_epsilon,
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=1,
    )
    result = strategy.configure_fit(1, None, None)
    assert result == []
    assert strategy.budget_exhausted is True
    # Final epsilon (zero rounds spent) is at/below the budget.
    assert acc.epsilon()["epsilon"] <= max_epsilon
    # One more release would still exceed the budget.
    assert acc.project_epsilon(additional_releases=1) > max_epsilon


def test_refusal_after_allowed_rounds() -> None:
    # Budget large enough for two rounds, refused on the third.
    max_epsilon = 10.0
    acc = _make_accountant(max_epsilon, noise_multiplier=1.0)
    strategy = build_strategy(
        "hybrid",
        accountant=acc,
        max_epsilon=max_epsilon,
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=1,
    )
    results = []
    for server_round in range(1, 5):
        result = strategy.configure_fit(server_round, None, _make_client_manager(1))
        results.append((server_round, bool(result)))
        if result is None:
            break
    # Some rounds allowed, then refusal.
    allowed = [r for r, ok in results if ok]
    refused = [r for r, ok in results if not ok]
    assert len(allowed) >= 1
    assert refused and refused[0] == allowed[-1] + 1
    assert strategy.budget_exhausted is True
    assert acc.epsilon()["epsilon"] <= max_epsilon + 1e-9
    assert acc.project_epsilon(additional_releases=1) > max_epsilon


# ---------------------------------------------------------------------------
# C5-2: strategy-level invocation (the gate is actually called)
# ---------------------------------------------------------------------------


def test_gate_invoked_before_dispatch_saves_state(tmp_path: Path, monkeypatch) -> None:
    state_path = tmp_path / "privacy_state.json"
    max_epsilon = 100.0  # large: round allowed
    acc = _make_accountant(max_epsilon)
    strategy = build_strategy(
        "hybrid",
        fraction_fit=1.0,
        accountant=acc,
        max_epsilon=max_epsilon,
        privacy_state_path=str(state_path),
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=1,
    )
    original_configure_fit = strategy.inner.configure_fit
    dispatched = []

    def check_state_before_dispatch(server_round, parameters, client_manager):
        dispatch_state = json.loads(state_path.read_text())
        assert dispatch_state["rounds_spent"] == 1
        dispatched.append(server_round)
        return original_configure_fit(server_round, parameters, client_manager)

    monkeypatch.setattr(strategy.inner, "configure_fit", check_state_before_dispatch)
    result = strategy.configure_fit(1, None, _make_client_manager(1))
    assert result is not None  # dispatch happened
    assert dispatched == [1]
    assert acc.rounds == 1  # account_round stepped before dispatch
    assert state_path.exists()  # state persisted before dispatch
    state = json.loads(state_path.read_text())
    assert state["rounds_spent"] == 1


def test_aggregate_fit_does_not_account_a_second_release(tmp_path: Path) -> None:
    acc = _make_accountant(max_epsilon=100.0)
    strategy = build_strategy(
        "fedavg",
        accountant=acc,
        max_epsilon=100.0,
        privacy_state_path=str(tmp_path / "privacy-state.json"),
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=1,
    )
    strategy.configure_fit(1, None, _make_client_manager(1))
    assert acc.rounds == 1

    strategy.aggregate_fit(1, [], [])

    assert acc.rounds == 1


# ---------------------------------------------------------------------------
# C5-3: no unauthorized aggregation after refusal
# ---------------------------------------------------------------------------


def test_no_accounting_after_refusal() -> None:
    max_epsilon = 0.05
    acc = _make_accountant(max_epsilon)
    strategy = build_strategy(
        "hybrid",
        accountant=acc,
        max_epsilon=max_epsilon,
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=1,
    )
    strategy.configure_fit(1, None, None)  # refused
    rounds_before = acc.rounds
    strategy.configure_fit(2, None, None)  # still refused, no step
    assert acc.rounds == rounds_before == 0
    assert strategy.budget_exhausted is True


# ---------------------------------------------------------------------------
# C5-4: crash between dispatch and aggregation (resume preserves rounds_spent)
# ---------------------------------------------------------------------------


def test_resume_preserves_rounds_spent(tmp_path: Path) -> None:
    state_path = tmp_path / "privacy_state.json"
    max_epsilon = 100.0
    acc = _make_accountant(max_epsilon)
    # Dispatch round 1: account + persist BEFORE dispatch.
    account_round(acc, str(state_path), server_round=1)
    assert acc.rounds == 1
    # Simulate a crash: a fresh accountant loads the persisted state.
    resumed = RdpAccountantWrapper.load_state(
        str(state_path), cfg=PrivacyConfig(max_epsilon=max_epsilon), resume=True
    )
    assert resumed.rounds == 1  # round 1 was spent and survives the crash
    assert resumed.resumed_from_round == 1


# ---------------------------------------------------------------------------
# C5-5: privacy_budget_exhausted emitted once; resumed_from_round recorded
# ---------------------------------------------------------------------------


def test_privacy_budget_exhausted_emitted_once(tmp_path: Path) -> None:
    state_path = tmp_path / "privacy_state.json"
    max_epsilon = 0.05
    acc = _make_accountant(max_epsilon)
    audit = _AuditSpy()
    strategy = build_strategy(
        "hybrid",
        fraction_fit=1.0,
        accountant=acc,
        max_epsilon=max_epsilon,
        privacy_state_path=str(state_path),
        audit_logger=audit,
        run_id="run-1",
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=1,
    )
    strategy.configure_fit(1, None, None)  # refusal -> emit
    strategy.configure_fit(2, None, None)  # still refused -> no second emit
    events = [e for e, _ in audit.events if e == "privacy_budget_exhausted"]
    assert len(events) == 1  # emitted exactly once
    payload = audit.events[0][1]
    assert payload["round"] == 1
    assert payload["max_epsilon"] == max_epsilon
    assert payload["projected"] > max_epsilon


# ---------------------------------------------------------------------------
# C5-6: corrupt / truncated / tampered state fail closed
# ---------------------------------------------------------------------------


def test_corrupt_state_fails_closed(tmp_path: Path) -> None:
    state_path = tmp_path / "privacy_state.json"
    state_path.write_text("{not valid json")
    with pytest.raises(PrivacyStateError):
        RdpAccountantWrapper.load_state(
            str(state_path),
            cfg=PrivacyConfig(max_epsilon=10.0),
            resume=True,
        )


def test_truncated_state_fails_closed(tmp_path: Path) -> None:
    state_path = tmp_path / "privacy_state.json"
    state_path.write_text('{"version":')
    with pytest.raises(PrivacyStateError):
        RdpAccountantWrapper.load_state(
            str(state_path),
            cfg=PrivacyConfig(max_epsilon=10.0),
            resume=True,
        )


def test_tampered_state_fails_closed(tmp_path: Path) -> None:
    state_path = tmp_path / "privacy_state.json"
    acc = _make_accountant(10.0)
    account_round(acc, str(state_path), server_round=1)
    data = json.loads(state_path.read_text())
    data["rounds_spent"] = 999  # tamper without recomputing sha256
    state_path.write_text(json.dumps(data))
    with pytest.raises(PrivacyStateError):
        RdpAccountantWrapper.load_state(
            str(state_path),
            cfg=PrivacyConfig(max_epsilon=10.0),
            resume=True,
        )


def test_missing_required_keys_fail(tmp_path: Path) -> None:
    state_path = tmp_path / "privacy_state.json"
    state_path.write_text(json.dumps({"version": 1}))
    with pytest.raises(PrivacyStateError):
        RdpAccountantWrapper.load_state(
            str(state_path),
            cfg=PrivacyConfig(max_epsilon=10.0),
            resume=True,
        )


def test_version_mismatch_fail(tmp_path: Path) -> None:
    state_path = tmp_path / "privacy_state.json"
    acc = _make_accountant(10.0)
    account_round(acc, str(state_path), server_round=1)
    data = json.loads(state_path.read_text())
    data["version"] = 999
    state_path.write_text(json.dumps(data))
    with pytest.raises(PrivacyStateError):
        RdpAccountantWrapper.load_state(
            str(state_path),
            cfg=PrivacyConfig(max_epsilon=10.0),
            resume=True,
        )


# ---------------------------------------------------------------------------
# C5-7: fresh run onto existing state raises; config mismatch on resume raises
# ---------------------------------------------------------------------------


def test_fresh_run_onto_existing_state_raises(tmp_path: Path) -> None:
    state_path = tmp_path / "privacy_state.json"
    acc = _make_accountant(10.0)
    account_round(acc, str(state_path), server_round=1)
    # A fresh run (resume=False) must not silently reset the budget.
    with pytest.raises(PrivacyStateError):
        RdpAccountantWrapper.load_state(
            str(state_path), cfg=PrivacyConfig(max_epsilon=10.0), resume=False
        )


def test_config_mismatch_on_resume_raises(tmp_path: Path) -> None:
    state_path = tmp_path / "privacy_state.json"
    acc = _make_accountant(10.0, noise_multiplier=1.0)
    account_round(acc, str(state_path), server_round=1)
    # Resume with a different noise_multiplier -> fail closed.
    with pytest.raises(PrivacyStateError):
        RdpAccountantWrapper.load_state(
            str(state_path),
            cfg=PrivacyConfig(max_epsilon=10.0, noise_multiplier=2.0),
            resume=True,
        )


# ---------------------------------------------------------------------------
# C5-8: interrupted-and-resumed accounting equals uninterrupted
# ---------------------------------------------------------------------------


def test_interrupted_resumed_equals_uninterrupted(tmp_path: Path) -> None:
    state_path = tmp_path / "privacy_state.json"
    cfg = PrivacyConfig(max_epsilon=100.0, noise_multiplier=1.0)

    # Uninterrupted: 3 rounds.
    acc_un = RdpAccountantWrapper(cfg)
    for _ in range(3):
        acc_un.step()
    eps_uninterrupted = acc_un.epsilon()["epsilon"]

    # Interrupted: 2 rounds, persist, crash, resume, 1 more round.
    acc_in = RdpAccountantWrapper(cfg)
    for _ in range(2):
        acc_in.step()
    acc_in.save_state(str(state_path))  # crash after dispatch of round 2
    resumed = RdpAccountantWrapper.load_state(str(state_path), cfg=cfg, resume=True)
    assert resumed.rounds == 2  # round 1 and 2 survived the crash
    resumed.step()  # round 3 after resume
    eps_resumed = resumed.epsilon()["epsilon"]

    assert abs(eps_resumed - eps_uninterrupted) < 1e-9
    assert resumed.rounds == 3


# ---------------------------------------------------------------------------
# C5-9: atomic state write (old state intact if replace fails)
# ---------------------------------------------------------------------------


def test_atomic_state_write_preserves_old_on_replace_failure(tmp_path: Path) -> None:
    state_path = tmp_path / "privacy_state.json"
    acc = _make_accountant(100.0)
    account_round(acc, str(state_path), server_round=1)
    old_content = state_path.read_text()

    # Force os.replace to fail during the second save.
    with mock.patch("os.replace", side_effect=OSError("disk")):
        acc.step()  # rounds -> 2
        with pytest.raises(OSError):
            acc.save_state(str(state_path))
    # Old state file must still be intact (rounds_spent == 1).
    state = json.loads(state_path.read_text())
    assert state["rounds_spent"] == 1
    assert state_path.read_text() == old_content


# ---------------------------------------------------------------------------
# Misconfiguration fail-closed at start-up
# ---------------------------------------------------------------------------


def test_max_epsilon_without_accountant_raises() -> None:
    with pytest.raises(ValueError, match="no privacy accountant"):
        build_strategy(
            "hybrid",
            fraction_fit=1.0,
            accountant=None,
            max_epsilon=10.0,
            min_fit_clients=1,
            min_evaluate_clients=1,
            min_available_clients=1,
        )


def test_accountant_without_max_epsilon_raises() -> None:
    acc = _make_accountant(10.0)
    with pytest.raises(ValueError, match="max_epsilon is required"):
        build_strategy(
            "hybrid",
            fraction_fit=1.0,
            accountant=acc,
            max_epsilon=None,
            min_fit_clients=1,
            min_evaluate_clients=1,
            min_available_clients=1,
        )


def test_privacy_enabled_unsupported_strategy_fails_closed() -> None:
    accountant = _make_accountant(max_epsilon=10.0)
    with pytest.raises(ValueError, match="Unknown strategy"):
        build_strategy(
            "ungatable",
            accountant=accountant,
            max_epsilon=10.0,
        )


def test_no_accountant_no_max_epsilon_is_ok() -> None:
    # No DP configured: no gate, no error.
    strategy = build_strategy(
        "hybrid",
        fraction_fit=1.0,
        accountant=None,
        max_epsilon=None,
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=1,
    )
    assert strategy.accountant is None
    assert strategy.max_epsilon is None
    assert strategy.configure_fit(1, None, _make_client_manager(1)) is not None


# ---------------------------------------------------------------------------
# C5-14: gate applies to every strategy type
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["fedavg", "fedprox", "qfedavg", "hybrid"])
def test_every_strategy_type_is_gated(name: str, tmp_path: Path) -> None:
    # Budget = 8.0 allows exactly 2 rounds (round 3 projected ~8.04 > 8.0).
    state_path = tmp_path / f"privacy_{name}.json"
    max_epsilon = 8.0
    acc = _make_accountant(max_epsilon)
    strategy = build_strategy(
        name,
        accountant=acc,
        max_epsilon=max_epsilon,
        privacy_state_path=str(state_path),
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=4,
    )
    params = Parameters(tensors=[], tensor_type="numpy")
    results = [
        bool(strategy.configure_fit(r, params, _make_client_manager(4))) for r in range(1, 6)
    ]
    # Rounds 1 and 2 dispatch; rounds 3, 4, 5 refused.
    assert results == [True, True, False, False, False]
    assert strategy.accountant.rounds == 2
    assert strategy.budget_exhausted is True
    assert strategy.stopped_for_budget is True
    assert strategy.rounds_trained == 2
    assert strategy.rounds_skipped == 3


@pytest.mark.parametrize("name", ["fedavg", "fedprox", "qfedavg", "hybrid"])
def test_privacy_disabled_never_refuses(name: str) -> None:
    # No accountant: no gate, all rounds dispatch.
    strategy = build_strategy(
        name,
        accountant=None,
        max_epsilon=None,
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=4,
    )
    params = Parameters(tensors=[], tensor_type="numpy")
    results = [
        bool(strategy.configure_fit(r, params, _make_client_manager(4))) for r in range(1, 6)
    ]
    assert results == [True, True, True, True, True]
    assert strategy.budget_exhausted is False
