"""Local Differential Privacy configuration and client-modifier builder.

Wraps Flower's ``LocalDpMod`` with explicit validation. Local DP is
applied on the client before Secure Aggregation masking: clip the model
update, add Gaussian noise, then cryptographically mask.

Local DP alone provides a per-client differential privacy guarantee; it
is orthogonal to Secure Aggregation (which provides confidentiality of
individual updates, not DP) and to RDP accounting (which tracks privacy
loss).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LocalDPConfig:
    """Local DP configuration.

    Attributes:
        enabled: Whether Local DP is active. When False, no Local DP
            guarantee is provided by this mechanism.
        clipping_norm: Maximum L2 norm of the client model update before
            noise is added. Must be positive.
        sensitivity: L2 sensitivity of the update. Must be positive.
        epsilon: Privacy budget (epsilon). Must be positive.
        delta: Privacy failure probability (delta). Must be in (0, 1).
    """

    enabled: bool = False
    clipping_norm: float = 1.0
    sensitivity: float = 1.0
    epsilon: float = 2.0
    delta: float = 1e-5


def build_local_dp_mod(config: LocalDPConfig):
    """Build a Flower ``LocalDpMod`` client modifier from config.

    Returns ``None`` when Local DP is disabled (no modifier is applied).
    Raises ``ValueError`` for invalid parameters — no silent substitution
    of defaults that would materially alter the privacy policy.
    """
    if not config.enabled:
        return None

    _validate(config)

    from flwr.client.mod.localdp_mod import LocalDpMod

    return LocalDpMod(
        clipping_norm=config.clipping_norm,
        sensitivity=config.sensitivity,
        epsilon=config.epsilon,
        delta=config.delta,
    )


def _validate(config: LocalDPConfig) -> None:
    if config.epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {config.epsilon}")
    if not 0 < config.delta < 1:
        raise ValueError(f"delta must be in (0, 1), got {config.delta}")
    if config.clipping_norm <= 0:
        raise ValueError(f"clipping_norm must be positive, got {config.clipping_norm}")
    if config.sensitivity <= 0:
        raise ValueError(f"sensitivity must be positive, got {config.sensitivity}")


__all__ = ["LocalDPConfig", "build_local_dp_mod"]
