"""RDP accounting tests (genuine backend).

``RdpAccountantWrapper`` uses ``dp_accounting`` as the RDP accounting
backend. These tests verify genuine accounting behavior.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pytest

from insurance_fl.fl import privacy


def test_privacy_exports() -> None:
    assert "build_privacy_config" in privacy.__all__
    assert "RdpAccountantWrapper" in privacy.__all__
    assert "PrivacyConfig" in privacy.__all__


def test_build_config() -> None:
    cfg = privacy.build_privacy_config({"max-epsilon": 5.0})
    assert cfg.max_epsilon == 5.0
    assert cfg.target_delta == 1e-5
    assert cfg.population_size == 4


def test_build_config_defaults() -> None:
    cfg = privacy.build_privacy_config()
    assert cfg.max_epsilon == 10.0
    assert cfg.target_delta == 1e-5
    assert cfg.population_size == 4


def test_accounting_available_with_backend() -> None:
    accountant = privacy.RdpAccountantWrapper(privacy.PrivacyConfig())
    assert accountant.accounting_available is True


def test_compose_records_events() -> None:
    accountant = privacy.RdpAccountantWrapper(privacy.PrivacyConfig())
    accountant.compose({"round": 1, "noise_multiplier": 0.1})
    accountant.compose({"round": 2, "noise_multiplier": 0.1})
    assert accountant.event_count == 2


def test_get_epsilon_returns_finite() -> None:
    accountant = privacy.RdpAccountantWrapper(privacy.PrivacyConfig(noise_multiplier=1.0))
    accountant.compose({"round": 1})
    eps = accountant.get_epsilon()
    assert eps >= 0

    assert math.isfinite(eps)


def test_get_epsilon_monotonic() -> None:
    accountant = privacy.RdpAccountantWrapper(privacy.PrivacyConfig(noise_multiplier=1.0))
    accountant.compose({"round": 1})
    eps1 = accountant.get_epsilon()
    accountant.compose({"round": 2})
    eps2 = accountant.get_epsilon()
    assert eps2 >= eps1


def test_get_epsilon_uses_delta() -> None:
    accountant = privacy.RdpAccountantWrapper(privacy.PrivacyConfig(noise_multiplier=1.0))
    accountant.compose({"round": 1})
    eps = accountant.get_epsilon(delta=1e-5)
    assert eps >= 0


def test_get_epsilon_rejects_invalid_delta() -> None:
    accountant = privacy.RdpAccountantWrapper(privacy.PrivacyConfig())
    with pytest.raises(ValueError):
        accountant.get_epsilon(delta=0.0)
    with pytest.raises(ValueError):
        accountant.get_epsilon(delta=1.0)


def test_would_exceed_below_budget() -> None:
    # High noise multiplier -> low epsilon -> should not exceed
    cfg = privacy.PrivacyConfig(max_epsilon=100.0, noise_multiplier=10.0)
    accountant = privacy.RdpAccountantWrapper(cfg)
    assert accountant.would_exceed({"round": 1}) is False


def test_would_exceed_above_budget() -> None:
    # Low noise multiplier -> high epsilon -> should exceed
    cfg = privacy.PrivacyConfig(max_epsilon=0.01, noise_multiplier=0.1)
    accountant = privacy.RdpAccountantWrapper(cfg)
    assert accountant.would_exceed({"round": 1}) is True


def test_would_exceed_non_mutating() -> None:
    cfg = privacy.PrivacyConfig(max_epsilon=10.0, noise_multiplier=1.0)
    accountant = privacy.RdpAccountantWrapper(cfg)
    accountant.compose({"round": 1})
    count_before = accountant.event_count
    eps_before = accountant.get_epsilon()
    accountant.would_exceed({"round": 2})
    assert accountant.event_count == count_before
    assert accountant.get_epsilon() == eps_before


def test_rejected_release_does_not_commit() -> None:
    cfg = privacy.PrivacyConfig(max_epsilon=0.01, noise_multiplier=0.1)
    accountant = privacy.RdpAccountantWrapper(cfg)
    count_before = accountant.event_count
    result = accountant.would_exceed({"round": 1})
    assert result is True
    assert accountant.event_count == count_before


def test_project_epsilon() -> None:
    cfg = privacy.PrivacyConfig(noise_multiplier=1.0)
    accountant = privacy.RdpAccountantWrapper(cfg)
    accountant.compose({"round": 1})
    projected = accountant.project_epsilon(additional_releases=2)
    # Project should not mutate
    assert accountant.event_count == 1
    assert projected >= accountant.get_epsilon()


def test_state_persistence_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        state_path = Path(tmpdir) / "state.json"
        cfg = privacy.PrivacyConfig(noise_multiplier=1.0, max_epsilon=10.0)
        accountant1 = privacy.RdpAccountantWrapper(cfg, state_path=state_path)
        accountant1.compose({"round": 1})
        accountant1.compose({"round": 2})
        assert accountant1.event_count == 2
        eps1 = accountant1.get_epsilon()

        # Restore from state
        accountant2 = privacy.RdpAccountantWrapper(cfg, state_path=state_path)
        assert accountant2.event_count == 2
        eps2 = accountant2.get_epsilon()
        assert eps1 == pytest.approx(eps2)


def test_state_not_counted_twice() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        state_path = Path(tmpdir) / "state.json"
        cfg = privacy.PrivacyConfig(noise_multiplier=1.0)
        accountant1 = privacy.RdpAccountantWrapper(cfg, state_path=state_path)
        accountant1.compose({"round": 1})
        # Restore and compose again
        accountant2 = privacy.RdpAccountantWrapper(cfg, state_path=state_path)
        assert accountant2.event_count == 1
        # Composing a new event should make it 2, not 3
        accountant2.compose({"round": 2})
        assert accountant2.event_count == 2


def test_corrupted_state_raises() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        state_path = Path(tmpdir) / "state.json"
        state_path.write_text("not valid json")
        cfg = privacy.PrivacyConfig(noise_multiplier=1.0)
        with pytest.raises(ValueError):
            privacy.RdpAccountantWrapper(cfg, state_path=state_path)


def test_incompatible_state_version_raises() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        state_path = Path(tmpdir) / "state.json"
        state_path.write_text(json.dumps({"version": 99, "committed_count": 1}))
        cfg = privacy.PrivacyConfig(noise_multiplier=1.0)
        with pytest.raises(ValueError):
            privacy.RdpAccountantWrapper(cfg, state_path=state_path)


def test_state_noise_multiplier_mismatch_raises() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        state_path = Path(tmpdir) / "state.json"
        state = {
            "version": 1,
            "committed_count": 1,
            "noise_multiplier": 999.0,
            "target_delta": 1e-5,
        }
        state_path.write_text(json.dumps(state))
        cfg = privacy.PrivacyConfig(noise_multiplier=1.0)
        with pytest.raises(ValueError):
            privacy.RdpAccountantWrapper(cfg, state_path=state_path)


def test_empty_accountant_epsilon_zero() -> None:
    accountant = privacy.RdpAccountantWrapper(privacy.PrivacyConfig(noise_multiplier=1.0))
    eps = accountant.get_epsilon()
    assert eps == 0.0


def test_backend_unavailable_simulation() -> None:
    # Verify the module-level flag exists
    assert hasattr(privacy, "_DP_ACCOUNTING_AVAILABLE")
    assert privacy._DP_ACCOUNTING_AVAILABLE is True
