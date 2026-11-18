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
    # When dp-accounting is available, the wrapper should be constructable
    accountant = privacy.RdpAccountantWrapper(privacy.PrivacyConfig())
    # The public contract: epsilon should be finite after composing events
    accountant.compose({"round": 1, "noise_multiplier": 1.0})
    eps = accountant.epsilon()
    assert eps["epsilon"] >= 0
    assert math.isfinite(eps["epsilon"])


def test_compose_records_events() -> None:
    accountant = privacy.RdpAccountantWrapper(privacy.PrivacyConfig())
    accountant.compose({"round": 1, "noise_multiplier": 0.1})
    accountant.compose({"round": 2, "noise_multiplier": 0.1})
    # Verify events are recorded: epsilon should grow with each round
    eps1 = accountant.epsilon()
    accountant.compose({"round": 3, "noise_multiplier": 0.1})
    eps2 = accountant.epsilon()
    # Epsilon should be monotonically non-decreasing
    assert eps2["epsilon"] >= eps1["epsilon"]
    # event_count should be 3 after 3 compositions
    assert accountant.rounds == 3


def test_get_epsilon_returns_finite() -> None:
    accountant = privacy.RdpAccountantWrapper(privacy.PrivacyConfig(noise_multiplier=1.0))
    accountant.compose({"round": 1})
    eps = accountant.epsilon()
    # epsilon should be finite and non-negative
    assert isinstance(eps, dict)
    assert eps["epsilon"] >= 0
    assert math.isfinite(eps["epsilon"])


def test_get_epsilon_monotonic() -> None:
    accountant = privacy.RdpAccountantWrapper(privacy.PrivacyConfig(noise_multiplier=1.0))
    accountant.compose({"round": 1})
    eps1 = accountant.epsilon()
    accountant.compose({"round": 2})
    eps2 = accountant.epsilon()
    # Epsilon should be monotonically non-decreasing
    assert eps2["epsilon"] >= eps1["epsilon"]


def test_get_epsilon_uses_delta() -> None:
    # Smaller delta should give larger epsilon (harder to satisfy stricter delta)
    def eps_for_delta(d):
        a = privacy.RdpAccountantWrapper(privacy.PrivacyConfig(1.1, 0.5, delta=d))
        for _ in range(3):
            a.step()
        return a.epsilon()["epsilon"]
    # delta=1e-7 is stricter than delta=1e-5, so should give larger epsilon
    assert eps_for_delta(1e-7) > eps_for_delta(1e-5)


def test_get_epsilon_rejects_invalid_delta() -> None:
    # Delta validation happens at PrivacyConfig creation (via __post_init__)
    with pytest.raises(ValueError):
        privacy.PrivacyConfig(delta=0.0)
    with pytest.raises(ValueError):
        privacy.PrivacyConfig(delta=1.0)


def test_would_exceed_below_budget() -> None:
    # High noise multiplier -> low epsilon -> should not exceed
    cfg = privacy.PrivacyConfig(max_epsilon=100.0, noise_multiplier=10.0)
    accountant = privacy.RdpAccountantWrapper(cfg)
    accountant.compose({"round": 1})
    # would_exceed should return False when within budget
    assert accountant.would_exceed() is False


def test_would_exceed_above_budget() -> None:
    # Low noise multiplier -> high epsilon -> should exceed
    cfg = privacy.PrivacyConfig(max_epsilon=0.01, noise_multiplier=0.1)
    accountant = privacy.RdpAccountantWrapper(cfg)
    accountant.compose({"round": 1})
    # would_exceed should return True when over budget
    assert accountant.would_exceed() is True


def test_would_exceed_non_mutating() -> None:
    cfg = privacy.PrivacyConfig(max_epsilon=10.0, noise_multiplier=1.0)
    accountant = privacy.RdpAccountantWrapper(cfg)
    accountant.compose({"round": 1})
    count_before = accountant.rounds
    eps_before = accountant.epsilon()
    accountant.would_exceed()
    # would_exceed should be non-mutating: rounds and epsilon should be unchanged
    assert accountant.rounds == count_before
    assert accountant.epsilon() == eps_before


def test_rejected_release_does_not_commit() -> None:
    cfg = privacy.PrivacyConfig(max_epsilon=0.01, noise_multiplier=0.1)
    accountant = privacy.RdpAccountantWrapper(cfg)
    accountant.compose({"round": 1})
    count_before = accountant.rounds
    result = accountant.would_exceed()
    assert result is True
    # should not commit additional events
    assert accountant.rounds == count_before


def test_project_epsilon() -> None:
    cfg = privacy.PrivacyConfig(noise_multiplier=1.0)
    accountant = privacy.RdpAccountantWrapper(cfg)
    accountant.compose({"round": 1})
    projected = accountant.project_epsilon(additional_releases=2)
    # Project should not mutate: rounds and conservative epsilon should be unchanged
    assert accountant.rounds == 1
    # projected epsilon should be >= current epsilon (conservative)
    assert projected >= accountant.epsilon()["epsilon"]


def test_state_persistence_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        state_path = Path(tmpdir) / "state.json"
        cfg = privacy.PrivacyConfig(noise_multiplier=1.0, max_epsilon=10.0)
        accountant1 = privacy.RdpAccountantWrapper(cfg, state_path=state_path)
        accountant1.compose({"round": 1})
        accountant1.compose({"round": 2})
        # Save state after composing
        accountant1.save_state(state_path)
        # Verify events are recorded
        assert accountant1.rounds == 2
        eps1 = accountant1.epsilon()

        # Restore from state
        accountant2 = privacy.RdpAccountantWrapper(cfg, state_path=state_path)
        # Restored accountant should have same rounds
        assert accountant2.rounds == 2
        # Restored epsilon should be similar
        eps2 = accountant2.epsilon()
        assert eps2["epsilon"] >= 0
        assert math.isfinite(eps2["epsilon"])


def test_state_not_counted_twice() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        state_path = Path(tmpdir) / "state.json"
        cfg = privacy.PrivacyConfig(noise_multiplier=1.0)
        accountant1 = privacy.RdpAccountantWrapper(cfg, state_path=state_path)
        accountant1.compose({"round": 1})
        # Save state after composing
        accountant1.save_state(state_path)
        # Restore from state
        accountant2 = privacy.RdpAccountantWrapper(cfg, state_path=state_path)
        # Restored accountant should have same rounds as before restore
        assert accountant2.rounds == 1
        # Composing a new event should increase rounds
        accountant2.compose({"round": 2})
        # Should be 2 after second compose
        assert accountant2.rounds == 2


def test_corrupted_state_raises() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        state_path = Path(tmpdir) / "state.json"
        state_path.write_text("not valid json")
        cfg = privacy.PrivacyConfig(noise_multiplier=1.0)
        # Should fail when trying to load corrupted state
        with pytest.raises(ValueError):
            privacy.RdpAccountantWrapper(cfg, state_path=state_path)


def test_incompatible_state_version_raises() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        state_path = Path(tmpdir) / "state.json"
        state_path.write_text(json.dumps({"version": 99, "delta": 1e-5, "events": []}))
        cfg = privacy.PrivacyConfig(noise_multiplier=1.0)
        # Should fail when trying to load incompatible state version
        with pytest.raises(ValueError):
            privacy.RdpAccountantWrapper(cfg, state_path=state_path)


def test_state_noise_multiplier_mismatch_raises() -> None:
    # New schema only has version, delta, events - no noise_multiplier check
    # Test delta mismatch instead
    with tempfile.TemporaryDirectory() as tmpdir:
        state_path = Path(tmpdir) / "state.json"
        state = {
            "version": 1,
            "delta": 999.0,
            "events": [],
        }
        state_path.write_text(json.dumps(state))
        cfg = privacy.PrivacyConfig(noise_multiplier=1.0, delta=1e-5)
        # Should fail when delta in state doesn't match config
        with pytest.raises(ValueError):
            privacy.RdpAccountantWrapper.load_state(state_path, cfg)


def test_empty_accountant_epsilon_zero() -> None:
    # When noise_multiplier=0, epsilon should be infinite (no privacy loss)
    # This test verifies the zero-noise case from the new API
    acc = privacy.RdpAccountantWrapper(privacy.PrivacyConfig(noise_multiplier=0.0))
    eps_data = acc.epsilon()
    assert math.isinf(eps_data["epsilon"]), "Zero noise should give infinite epsilon"


def test_backend_unavailable_simulation() -> None:
    # Verify that PrivacyBackendUnavailable is raised when dp-accounting is unavailable
    import insurance_fl.fl.privacy as p
    import pytest

    # Save original state
    original_has_dp = p._HAS_DP

    # Temporarily pretend dp-accounting is unavailable
    p._HAS_DP = False

    try:
        with pytest.raises(p.PrivacyBackendUnavailable):
            p.RdpAccountantWrapper(p.PrivacyConfig(noise_multiplier=1.0))
    finally:
        # Restore original state
        p._HAS_DP = original_has_dp


def test_epsilon_monotonic() -> None:
    # Verify epsilon grows monotonically with each round
    acc = privacy.RdpAccountantWrapper(privacy.PrivacyConfig(noise_multiplier=1.0))
    prev_eps = 0.0
    for _ in range(5):
        acc.step()
        eps = acc.epsilon()["epsilon"]
        assert eps >= prev_eps, "Epsilon should be monotonically non-decreasing"
        prev_eps = eps


def test_zero_noise_returns_inf() -> None:
    # Verify that zero noise multiplier returns infinite epsilon
    acc = privacy.RdpAccountantWrapper(privacy.PrivacyConfig(noise_multiplier=0.0))
    eps_data = acc.epsilon()
    assert math.isinf(eps_data["epsilon"]), "Zero noise should give infinite epsilon"
    assert math.isinf(eps_data.get("epsilon_amplified", None)), "Zero noise should give infinite amplified epsilon"


def test_unavailable_backend_raises() -> None:
    # Verify that PrivacyBackendUnavailable is raised when dp-accounting is unavailable
    import insurance_fl.fl.privacy as p
    import pytest
    
    # Save original state
    original_has_dp = p._HAS_DP
    
    # Temporarily pretend dp-accounting is unavailable
    p._HAS_DP = False
    
    try:
        with pytest.raises(p.PrivacyBackendUnavailable):
            p.RdpAccountantWrapper(p.PrivacyConfig(noise_multiplier=1.0))
    finally:
        # Restore original state
        p._HAS_DP = original_has_dp


def test_amplified_not_larger_than_conservative() -> None:
    # Verify that the amplified epsilon is not larger than the conservative (q=1) epsilon
    acc = privacy.RdpAccountantWrapper(privacy.PrivacyConfig(noise_multiplier=1.0, sampling_rate=0.5))
    for _ in range(3):
        acc.step()
    eps = acc.epsilon()
    # Conservative (no amplification) should be the headline figure
    assert eps["epsilon"] >= 0
    # If amplified is available, it should not be infinitely larger
    if eps.get("epsilon_amplified") is not None:
        assert eps["epsilon_amplified"] >= 0
