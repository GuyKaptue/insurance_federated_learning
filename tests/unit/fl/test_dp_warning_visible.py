"""Regression test: LocalDP disabled state must surface a UserWarning."""

from __future__ import annotations

import warnings

import pytest

from insurance_fl.fl.dp_warning_visible import LocalDPConfig, build_local_dp_mod


def test_local_dp_disabled_emits_warning() -> None:
    cfg = LocalDPConfig(enabled=False)
    with pytest.warns(UserWarning, match="Local DP"):
        assert build_local_dp_mod(cfg) is None


def test_local_dp_enabled_emits_no_warning() -> None:
    cfg = LocalDPConfig(enabled=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build_local_dp_mod(cfg)
        local_dp_warnings = [
            w
            for w in caught
            if issubclass(w.category, UserWarning) and "Local DP" in str(w.message)
        ]
        assert local_dp_warnings == []
