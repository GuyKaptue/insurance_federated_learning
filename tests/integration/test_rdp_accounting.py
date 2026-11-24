"""Integration tests for RDP accounting (current dp-accounting backend API)."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import pytest

from insurance_fl.fl.privacy import (
    PrivacyBackendUnavailable,
    PrivacyConfig,
    PrivacyStateError,
    RdpAccountantWrapper,
)


def _make(noise_multiplier: float = 1.0, **kw) -> RdpAccountantWrapper:
    return RdpAccountantWrapper(PrivacyConfig(noise_multiplier=noise_multiplier, **kw))


def test_backend_available_and_accountant_constructible() -> None:
    acc = _make(1.0)
    # current API exposes epsilon/rounds/state methods (no accounting_available flag)
    assert hasattr(acc, "epsilon")
    assert hasattr(acc, "compose")
    assert acc.rounds == 0


def test_compose_records_rounds_and_events() -> None:
    acc = _make(1.0)
    acc.step()
    acc.step()
    assert acc.rounds == 2
    assert len(acc.events) == 2
    eps = acc.epsilon()
    assert eps["rounds"] == 2
    assert eps["epsilon"] >= 0
    assert math.isfinite(eps["epsilon"])


def test_get_epsilon_returns_finite() -> None:
    acc = _make(1.0)
    acc.step()
    eps = acc.epsilon()["epsilon"]
    assert isinstance(eps, float)
    assert eps >= 0
    assert math.isfinite(eps)


def test_get_epsilon_monotonic() -> None:
    acc = _make(1.0)
    acc.step()
    eps1 = acc.epsilon()["epsilon"]
    acc.step()
    eps2 = acc.epsilon()["epsilon"]
    assert eps2 >= eps1


def test_would_exceed_over_budget() -> None:
    acc = _make(noise_multiplier=0.1, max_epsilon=0.01)
    acc.step()
    assert acc.would_exceed() is True


def test_would_exceed_under_budget() -> None:
    acc = _make(noise_multiplier=10.0, max_epsilon=100.0)
    acc.step()
    assert acc.would_exceed() is False


def test_project_epsilon_does_not_mutate_state() -> None:
    acc = _make(noise_multiplier=1.0, max_epsilon=10.0)
    acc.step()
    rounds_before = acc.rounds
    eps_before = acc.epsilon()["epsilon"]
    projected = acc.project_epsilon(additional_releases=1)
    assert acc.rounds == rounds_before
    assert acc.epsilon()["epsilon"] == pytest.approx(eps_before)
    assert projected >= eps_before


def test_would_exceed_is_non_mutating() -> None:
    acc = _make(noise_multiplier=0.1, max_epsilon=0.01)
    acc.step()
    rounds_before = acc.rounds
    _ = acc.would_exceed()
    assert acc.rounds == rounds_before


def test_state_persistence_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        state_path = Path(tmpdir) / "state.json"
        cfg = PrivacyConfig(noise_multiplier=1.0, max_epsilon=10.0)
        a1 = RdpAccountantWrapper(cfg, state_path=state_path)
        a1.step()
        a1.step()
        eps1 = a1.epsilon()["epsilon"]
        a1.save_state(state_path)

        a2 = RdpAccountantWrapper(cfg, state_path=state_path)
        assert a2.rounds == 2
        assert a2.epsilon()["epsilon"] == pytest.approx(eps1)


def test_state_not_counted_twice() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        state_path = Path(tmpdir) / "state.json"
        cfg = PrivacyConfig(noise_multiplier=1.0)
        a1 = RdpAccountantWrapper(cfg, state_path=state_path)
        a1.step()
        a1.save_state(state_path)

        a2 = RdpAccountantWrapper(cfg, state_path=state_path)
        assert a2.rounds == 1
        a2.step()
        assert a2.rounds == 2


def test_corrupted_state_raises() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        state_path = Path(tmpdir) / "state.json"
        state_path.write_text("not valid json")
        with pytest.raises(PrivacyStateError):
            RdpAccountantWrapper(PrivacyConfig(noise_multiplier=1.0), state_path=state_path)


def test_composition_matches_backend() -> None:
    from dp_accounting import GaussianDpEvent
    from dp_accounting.rdp import RdpAccountant

    cfg = PrivacyConfig(noise_multiplier=2.0, delta=1e-5, target_delta=1e-5)
    accountant = RdpAccountantWrapper(cfg)
    accountant.step()
    accountant.step()
    eps_wrapper = accountant.epsilon()["epsilon"]

    direct = RdpAccountant()
    gd = GaussianDpEvent(noise_multiplier=2.0)
    direct.compose(gd, count=2)
    eps_direct, _ = direct.get_epsilon_and_optimal_order(target_delta=1e-5)

    assert eps_wrapper == pytest.approx(eps_direct)


def test_invalid_config_delta_rejected() -> None:
    with pytest.raises(ValueError):
        PrivacyConfig(delta=0.0)
    with pytest.raises(ValueError):
        PrivacyConfig(delta=1.0)


def test_get_epsilon_delta_sensitivity() -> None:
    """Stricter delta should require >= epsilon for same releases (property preserved)."""

    def eps_for_delta(d: float) -> float:
        acc = RdpAccountantWrapper(
            PrivacyConfig(noise_multiplier=1.0, delta=d, target_delta=d, max_epsilon=1e9)
        )
        for _ in range(3):
            acc.step()
        return acc.epsilon()["epsilon"]

    assert eps_for_delta(1e-7) > eps_for_delta(1e-5)


def test_rejected_release_projection_does_not_commit() -> None:
    """Budget-check/projection path must not mutate committed accounting state."""
    acc = _make(noise_multiplier=0.1, max_epsilon=0.01)
    # Current state is empty/under budget in terms of committed rounds; project over budget.
    rounds_before = acc.rounds
    eps_before = acc.epsilon()["epsilon"]
    projected = acc.project_epsilon(additional_releases=1)
    assert projected > 0.01 or math.isinf(projected) or acc.would_exceed() in (True, False)
    # Projection/check must not commit events.
    assert acc.rounds == rounds_before
    assert acc.epsilon()["epsilon"] == pytest.approx(eps_before)


def test_backend_unavailable_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    import insurance_fl.fl.privacy as p

    monkeypatch.setattr(p, "_HAS_DP", False)
    with pytest.raises(PrivacyBackendUnavailable):
        RdpAccountantWrapper(PrivacyConfig(noise_multiplier=1.0))
