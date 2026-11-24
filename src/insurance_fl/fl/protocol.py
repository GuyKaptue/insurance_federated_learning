"""Generic federated-learning client/server protocol (O-6).

A round returns a dict of named numpy arrays plus metadata — NOT only
model weights — so P3 aggregate protocols (means, quantiles, gradients)
plug in without rewriting the FL glue.

Wave 1 scope: interface + dummy aggregate protocol test only. No Flower
wiring, no training loop; those arrive in later waves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass
class RoundResult:
    """Payload returned by one federated round for one client."""

    arrays: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class ClientProtocol(Protocol):
    """Minimal interface every FL client must implement."""

    client_id: str

    def local_round(self, instruction: dict[str, Any]) -> RoundResult:
        """Run one local round; return named arrays + metadata."""
        ...  # pragma: no cover - protocol stub
