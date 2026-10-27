"""Unit tests for Local DP configuration and builder.

Verifies:
- Correct configuration-to-constructor mapping.
- Rejection of invalid privacy parameters.
- Explicit behavior when Local DP is disabled.
- No false claims that the presence of a modifier establishes an
  end-to-end privacy guarantee.
"""

from __future__ import annotations

from flwr.client.mod.localdp_mod import LocalDpMod

from insurance_fl.fl.dp import LocalDPConfig, build_local_dp_mod


def test_disabled_returns_none() -> None:
    """When Local DP is disabled, no modifier is produced."""
    cfg = LocalDPConfig(enabled=False)
    assert build_local_dp_mod(cfg) is None


def test_enabled_returns_local_dp_mod() -> None:
    """When enabled, a LocalDpMod is returned with correct params."""
    cfg = LocalDPConfig(
        enabled=True,
        clipping_norm=2.0,
        sensitivity=0.5,
        epsilon=4.0,
        delta=1e-5,
    )
    mod = build_local_dp_mod(cfg)
    assert isinstance(mod, LocalDpMod)
    assert mod.clipping_norm == 2.0
    assert mod.sensitivity == 0.5
    assert mod.epsilon == 4.0
    assert mod.delta == 1e-5


def test_rejects_nonpositive_epsilon() -> None:
    cfg = LocalDPConfig(enabled=True, epsilon=0.0)
    try:
        build_local_dp_mod(cfg)
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "epsilon" in str(exc).lower()


def test_rejects_negative_epsilon() -> None:
    cfg = LocalDPConfig(enabled=True, epsilon=-1.0)
    try:
        build_local_dp_mod(cfg)
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "epsilon" in str(exc).lower()


def test_rejects_invalid_delta_zero() -> None:
    cfg = LocalDPConfig(enabled=True, delta=0.0)
    try:
        build_local_dp_mod(cfg)
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "delta" in str(exc).lower()


def test_rejects_invalid_delta_one() -> None:
    cfg = LocalDPConfig(enabled=True, delta=1.0)
    try:
        build_local_dp_mod(cfg)
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "delta" in str(exc).lower()


def test_rejects_nonpositive_clipping_norm() -> None:
    cfg = LocalDPConfig(enabled=True, clipping_norm=0.0)
    try:
        build_local_dp_mod(cfg)
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "clipping_norm" in str(exc).lower()


def test_rejects_nonpositive_sensitivity() -> None:
    cfg = LocalDPConfig(enabled=True, sensitivity=0.0)
    try:
        build_local_dp_mod(cfg)
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "sensitivity" in str(exc).lower()


def test_disabled_config_not_false_claim() -> None:
    """Disabled Local DP means no Local-DP guarantee from this mechanism."""
    cfg = LocalDPConfig(enabled=False)
    assert cfg.enabled is False
    assert build_local_dp_mod(cfg) is None


def test_default_values() -> None:
    """Default config has sane values."""
    cfg = LocalDPConfig()
    assert cfg.enabled is False
    assert cfg.clipping_norm == 1.0
    assert cfg.sensitivity == 1.0
    assert cfg.epsilon == 2.0
    assert cfg.delta == 1e-5
