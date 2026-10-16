"""Dummy aggregate protocol test (O-6).

A non-weight client (per-feature means) must satisfy ClientProtocol and
round-trip through RoundResult — proving the FL glue is not weights-only.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from insurance_fl.fl.protocol import ClientProtocol, RoundResult


class MeanAggregateClient:
    """Toy client returning per-feature means (no model weights)."""

    def __init__(self, client_id: str, data: np.ndarray) -> None:
        self.client_id = client_id
        self._data = data

    def local_round(self, instruction: dict[str, Any]) -> RoundResult:
        scale = float(instruction.get("scale", 1.0))
        means = self._data.mean(axis=0) * scale
        return RoundResult(
            arrays={"feature_means": means, "counts": np.array([len(self._data)])},
            metadata={"client_id": self.client_id, "kind": "mean_aggregate"},
        )


def _assert_is_protocol(client: ClientProtocol) -> None:
    assert isinstance(client.local_round({}), RoundResult)


def test_dummy_aggregate_protocol_round_trip() -> None:
    rng = np.random.default_rng(7)
    data = rng.normal(size=(50, 4))
    client = MeanAggregateClient("dummy-1", data)
    _assert_is_protocol(client)
    result = client.local_round({"scale": 2.0})
    assert set(result.arrays) == {"feature_means", "counts"}
    assert result.metadata["client_id"] == "dummy-1"
    np.testing.assert_allclose(result.arrays["feature_means"], data.mean(axis=0) * 2.0)
    assert result.arrays["counts"].tolist() == [50]
