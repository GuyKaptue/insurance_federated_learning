"""Integration tests for RDP accounting (genuine backend)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from insurance_fl.fl.privacy import PrivacyConfig, RdpAccountantWrapper


def test_accounting_available() -> None:
    accountant = RdpAccountantWrapper(PrivacyConfig())
    assert accountant.accounting_available is True


def test_compose_records_events() -> None:
    accountant = RdpAccountantWrapper(PrivacyConfig())
    accountant.compose({"round": 1, "mechanism": "gaussian"})
    accountant.compose({"round": 2, "mechanism": "gaussian"})
    assert accountant.event_count == 2


def test_get_epsilon_returns_finite() -> None:
    accountant = RdpAccountantWrapper(PrivacyConfig(noise_multiplier=1.0))
    accountant.compose({"round": 1})
    eps = accountant.get_epsilon()
    assert eps >= 0
    import math

    assert math.isfinite(eps)


def test_get_epsilon_monotonic() -> None:
    accountant = RdpAccountantWrapper(PrivacyConfig(noise_multiplier=1.0))
    accountant.compose({"round": 1})
    eps1 = accountant.get_epsilon()
    accountant.compose({"round": 2})
    eps2 = accountant.get_epsilon()
    assert eps2 >= eps1


def test_would_exceed_rejects_over_budget() -> None:
    cfg = PrivacyConfig(max_epsilon=0.01, noise_multiplier=0.1)
    accountant = RdpAccountantWrapper(cfg)
    assert accountant.would_exceed({"round": 1}) is True


def test_would_exceed_allows_under_budget() -> None:
    cfg = PrivacyConfig(max_epsilon=100.0, noise_multiplier=10.0)
    accountant = RdpAccountantWrapper(cfg)
    assert accountant.would_exceed({"round": 1}) is False


def test_projected_does_not_mutate() -> None:
    cfg = PrivacyConfig(max_epsilon=10.0, noise_multiplier=1.0)
    accountant = RdpAccountantWrapper(cfg)
    accountant.compose({"round": 1})
    count_before = accountant.event_count
    eps_before = accountant.get_epsilon()
    accountant.would_exceed({"round": 2})
    assert accountant.event_count == count_before
    assert accountant.get_epsilon() == pytest.approx(eps_before)


def test_rejected_release_does_not_commit() -> None:
    cfg = PrivacyConfig(max_epsilon=0.01, noise_multiplier=0.1)
    accountant = RdpAccountantWrapper(cfg)
    count_before = accountant.event_count
    result = accountant.would_exceed({"round": 1})
    assert result is True
    assert accountant.event_count == count_before


def test_state_persistence() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        state_path = Path(tmpdir) / "state.json"
        cfg = PrivacyConfig(noise_multiplier=1.0, max_epsilon=10.0)
        a1 = RdpAccountantWrapper(cfg, state_path=state_path)
        a1.compose({"round": 1})
        a1.compose({"round": 2})
        eps1 = a1.get_epsilon()

        a2 = RdpAccountantWrapper(cfg, state_path=state_path)
        assert a2.event_count == 2
        assert a2.get_epsilon() == pytest.approx(eps1)


def test_state_not_counted_twice() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        state_path = Path(tmpdir) / "state.json"
        cfg = PrivacyConfig(noise_multiplier=1.0)
        a1 = RdpAccountantWrapper(cfg, state_path=state_path)
        a1.compose({"round": 1})

        a2 = RdpAccountantWrapper(cfg, state_path=state_path)
        assert a2.event_count == 1
        a2.compose({"round": 2})
        assert a2.event_count == 2


def test_corrupted_state_raises() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        state_path = Path(tmpdir) / "state.json"
        state_path.write_text("not valid json")
        cfg = PrivacyConfig(noise_multiplier=1.0)
        with pytest.raises(ValueError):
            RdpAccountantWrapper(cfg, state_path=state_path)


def test_composition_matches_backend() -> None:
    # Verify against direct backend usage
    from dp_accounting import GaussianDpEvent, NeighboringRelation
    from dp_accounting.rdp import RdpAccountant

    cfg = PrivacyConfig(noise_multiplier=2.0, target_delta=1e-5)
    accountant = RdpAccountantWrapper(cfg)
    accountant.compose({"round": 1})
    accountant.compose({"round": 2})
    eps_wrapper = accountant.get_epsilon()

    # Direct backend
    direct = RdpAccountant(neighboring_relation=NeighboringRelation.ADD_OR_REMOVE_ONE)
    gd = GaussianDpEvent(noise_multiplier=2.0)
    direct.compose(gd, count=2)
    eps_direct, _ = direct.get_epsilon_and_optimal_order(target_delta=1e-5)

    assert eps_wrapper == pytest.approx(eps_direct)


def test_invalid_delta_handling() -> None:
    accountant = RdpAccountantWrapper(PrivacyConfig())
    with pytest.raises(ValueError):
        accountant.get_epsilon(delta=0.0)


def test_disabled_not_applicable() -> None:
    # Accounting is now available (backend installed)
    accountant = RdpAccountantWrapper(PrivacyConfig())
    assert accountant.accounting_available is True
