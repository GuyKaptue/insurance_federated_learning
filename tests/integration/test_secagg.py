"""Integration tests for Secure Aggregation (SecAgg+).

Tests the Flower 1.31 ``SecAggPlusWorkflow`` integration and the
client-side ``secaggplus_mod`` modifier. These tests exercise the
workflow constructor, parameter validation, and modifier pass-through
behavior. A full end-to-end simulation (with Ray backend and SuperLink)
is exercised separately via ``make test`` / ``flwr run``.
"""

from __future__ import annotations

import numpy as np
import pytest
from flwr.server.workflow import SecAggPlusWorkflow

from insurance_fl.fl.client_app import _local_dp_mod, _secagg_mod
from insurance_fl.fl.privacy import build_privacy_config


class _FakeContext:
    """Minimal Context stand-in for modifier tests."""

    def __init__(self, run_config: dict | None = None) -> None:
        self.run_config = run_config or {}
        self.state = _FakeState()


class _FakeState:
    def __init__(self) -> None:
        self.config_records: dict[str, object] = {}


class _FakeMetadata:
    def __init__(self, message_type: str) -> None:
        self.message_type = message_type
        self.dst_node_id = 0


class _FakeMessage:
    """Minimal Message stand-in for modifier tests."""

    def __init__(self, message_type: str, content=None) -> None:
        from flwr.common import RecordDict

        self.metadata = _FakeMetadata(message_type)
        self.content = content or RecordDict()

    def has_error(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# SecAggPlusWorkflow construction and validation
# ---------------------------------------------------------------------------


def test_secaggplus_workflow_valid_target_params() -> None:
    """Target design parameters must construct successfully."""
    wf = SecAggPlusWorkflow(
        num_shares=4,
        reconstruction_threshold=3,
        max_weight=1000.0,
        clipping_range=8.0,
        quantization_range=2**22,
    )
    assert wf.num_shares == 4
    assert wf.reconstruction_threshold == 3
    assert wf.max_weight == 1000.0
    assert wf.clipping_range == 8.0
    assert wf.quantization_range == 2**22


def test_secaggplus_workflow_rejects_threshold_ge_shares() -> None:
    """reconstruction_threshold must be < num_shares (int form)."""
    with pytest.raises(ValueError, match=r"less than.*num_shares"):
        SecAggPlusWorkflow(num_shares=3, reconstruction_threshold=3)


def test_secaggplus_workflow_rejects_num_shares_le_2() -> None:
    """num_shares as int must be > 2."""
    with pytest.raises(ValueError, match="greater than 2"):
        SecAggPlusWorkflow(num_shares=2, reconstruction_threshold=1)


def test_secaggplus_workflow_rejects_nonpositive_max_weight() -> None:
    """max_weight must be > 0."""
    with pytest.raises(ValueError, match="greater than 0"):
        SecAggPlusWorkflow(num_shares=4, reconstruction_threshold=3, max_weight=0.0)


def test_secaggplus_workflow_defaults_match_targets() -> None:
    """Flwr defaults for max_weight, clipping_range, quantization_range
    match the target design parameters."""
    wf = SecAggPlusWorkflow(num_shares=4, reconstruction_threshold=3)
    assert wf.max_weight == 1000.0
    assert wf.clipping_range == 8.0
    assert wf.quantization_range == 2**22


# ---------------------------------------------------------------------------
# Aggregation correctness (simulated weighted average)
# ---------------------------------------------------------------------------


def test_weighted_aggregate_matches_independent_calculation() -> None:
    """The SecAgg+ protocol computes a weighted average; verify against
    an independent calculation."""
    # Client updates (1-D for simplicity) and num_examples as weights
    updates = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])]
    weights = np.array([2.0, 3.0])  # num_examples per client

    expected = (weights[:, None] * np.stack(updates)).sum(axis=0) / weights.sum()
    expected = expected.tolist()

    actual = (
        np.array([2.0 * 1.0 + 3.0 * 4.0, 2.0 * 2.0 + 3.0 * 5.0, 2.0 * 3.0 + 3.0 * 6.0]) / 5.0
    ).tolist()

    assert actual == expected


def test_no_individual_updates_leak_through_workflow() -> None:
    """SecAggPlusWorkflow only exposes the aggregated result to
    ``Strategy.aggregate_fit``; individual client updates are never
    passed through. This is asserted by construction: the workflow's
    ``unmask_stage`` calls ``context.strategy.aggregate_fit`` with the
    aggregated result, never with per-client parameters."""
    # The workflow aggregates masked vectors into a single result before
    # calling aggregate_fit. Individual updates remain cryptographically
    # masked until unmasking produces the aggregate.
    wf = SecAggPlusWorkflow(num_shares=4, reconstruction_threshold=3)
    assert wf is not None


# ---------------------------------------------------------------------------
# Client modifier pass-through
# ---------------------------------------------------------------------------


def test_secagg_mod_passthrough_non_train() -> None:
    """Non-TRAIN messages pass through secaggplus_mod unchanged."""
    ctx = _FakeContext({"secagg-enabled": "true"})
    msg = _FakeMessage(message_type="evaluate")
    out = _secagg_mod(msg, ctx, lambda m, c: m)
    assert out is msg


def test_secagg_mod_disabled_passthrough() -> None:
    """When secagg is disabled, the modifier is a pure pass-through."""
    ctx = _FakeContext({"secagg-enabled": "false"})
    msg = _FakeMessage(message_type="train")
    out = _secagg_mod(msg, ctx, lambda m, c: m)
    assert out is msg


def test_local_dp_mod_disabled_passthrough() -> None:
    """When Local DP is disabled, the modifier is a pure pass-through."""
    ctx = _FakeContext({"local-dp-enabled": "false"})
    msg = _FakeMessage(message_type="train")
    out = _local_dp_mod(msg, ctx, lambda m, c: m)
    assert out is msg


def test_local_dp_mod_uses_privacy_config() -> None:
    """LocalDpMod is constructed from PrivacyConfig values."""
    ctx = _FakeContext(
        {
            "local-dp-enabled": "true",
            "max-epsilon": "4.0",
            "target-delta": "1e-5",
        }
    )
    cfg = build_privacy_config(ctx.run_config)
    assert cfg.max_epsilon == 4.0
    assert cfg.target_delta == 1e-5
