"""Privacy layer — RDP accountant wrapper + privacy config.

Uses ``dp_accounting`` (Google) as the RDP accounting backend when available.
Accounting is genuine: ``get_epsilon()`` returns real RDP epsilon values
composed from GaussianDpEvent releases. ``would_exceed()`` projects
epsilon without mutating committed state.

When ``dp_accounting`` is unavailable, all operations fail closed:
``get_epsilon()`` raises ``NotImplementedError``, ``would_exceed()``
returns ``True``.

Local DP mechanism: Gaussian noise with noise_multiplier = sigma/sensitivity.
Each federated round is one privacy-relevant release. Sampling is not
amplified (local DP protects each client independently).
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Try to import dp_accounting backend
try:
    from dp_accounting import GaussianDpEvent, NeighboringRelation
    from dp_accounting.rdp import RdpAccountant

    _DP_ACCOUNTING_AVAILABLE = True
except ImportError:
    _DP_ACCOUNTING_AVAILABLE = False
    GaussianDpEvent = None  # type: ignore[assignment,misc]
    NeighboringRelation = None  # type: ignore[assignment,misc]
    RdpAccountant = None  # type: ignore[assignment,misc]


@dataclass
class PrivacyConfig:
    max_epsilon: float = 10.0
    target_delta: float = 1e-5
    population_size: int = 4
    noise_multiplier: float = 1.0
    neighboring_relation: str = "add_or_remove_one"


def build_privacy_config(run_config: dict[str, Any] | None = None) -> PrivacyConfig:
    rc = run_config or {}
    noise_mult = float(rc.get("noise-multiplier", 0.0))
    if noise_mult <= 0:
        dp_epsilon = float(rc.get("dp-epsilon", rc.get("epsilon", 0.0)))
        dp_delta = float(rc.get("dp-delta", rc.get("delta", 0.0)))
        if dp_epsilon > 0 and 0 < dp_delta < 1.25:
            noise_mult = math.sqrt(2.0 * math.log(1.25 / dp_delta)) / dp_epsilon
        else:
            noise_mult = float(rc.get("noise-multiplier-default", 1.0))
    return PrivacyConfig(
        max_epsilon=float(rc.get("max-epsilon", 10.0)),
        target_delta=float(rc.get("target-delta", 1e-5)),
        population_size=int(rc.get("population-size", 4)),
        noise_multiplier=noise_mult,
        neighboring_relation=str(rc.get("neighboring-relation", "add_or_remove_one")),
    )


_RELATION_MAP: dict[str, Any] = {}


def _get_relation(name: str) -> Any:
    if not _DP_ACCOUNTING_AVAILABLE:
        return None
    if not _RELATION_MAP:
        _RELATION_MAP["add_or_remove_one"] = NeighboringRelation.ADD_OR_REMOVE_ONE
        _RELATION_MAP["replace_one"] = NeighboringRelation.REPLACE_ONE
    return _RELATION_MAP.get(name, NeighboringRelation.ADD_OR_REMOVE_ONE)


class RdpAccountantWrapper:
    """RDP accountant wrapper using ``dp_accounting`` backend."""

    def __init__(
        self,
        config: PrivacyConfig,
        state_path: str | Path | None = None,
    ):
        self.config = config
        self._state_path = Path(state_path) if state_path else None
        self._committed_count: int = 0
        self._events_ledger: list[dict[str, Any]] = []
        self._accountant: Any = None

        if _DP_ACCOUNTING_AVAILABLE:
            relation = _get_relation(config.neighboring_relation)
            self._accountant = RdpAccountant(
                neighboring_relation=relation,
            )
            self._load_state()

    @property
    def accounting_available(self) -> bool:
        return _DP_ACCOUNTING_AVAILABLE and self._accountant is not None

    @property
    def event_count(self) -> int:
        return self._committed_count

    def compose(self, event: dict[str, Any]) -> None:
        """Commit a privacy event (GaussianDpEvent release)."""
        self._events_ledger.append(dict(event))
        if self.accounting_available:
            gd_event = GaussianDpEvent(noise_multiplier=self.config.noise_multiplier)
            self._accountant.compose(gd_event, count=1)
            self._committed_count += 1
            self._save_state()

    def get_epsilon(self, delta: float | None = None) -> float:
        """Return cumulative RDP epsilon at ``delta``."""
        delta = self.config.target_delta if delta is None else float(delta)
        if delta <= 0 or delta >= 1:
            raise ValueError(f"delta must be in (0, 1), got {delta}")
        if not self.accounting_available:
            raise NotImplementedError(
                "RDP accounting backend not available; privacy loss is "
                "not formally accounted for. Committed events: "
                f"{self._committed_count}; configured delta={delta}."
            )
        eps, _ = self._accountant.get_epsilon_and_optimal_order(target_delta=delta)
        return float(eps)

    def would_exceed(self, event: dict[str, Any], max_epsilon: float | None = None) -> bool:
        """Projected budget check (non-mutating)."""
        max_epsilon = self.config.max_epsilon if max_epsilon is None else float(max_epsilon)
        if not self.accounting_available:
            return True  # fail closed
        projected_count = self._committed_count + 1
        temp_accountant = RdpAccountant(
            neighboring_relation=_get_relation(self.config.neighboring_relation),
        )
        gd_event = GaussianDpEvent(noise_multiplier=self.config.noise_multiplier)
        temp_accountant.compose(gd_event, count=projected_count)
        projected_eps, _ = temp_accountant.get_epsilon_and_optimal_order(
            target_delta=self.config.target_delta
        )
        return float(projected_eps) > max_epsilon

    def project_epsilon(self, additional_releases: int = 1) -> float:
        """Project cumulative epsilon with additional releases (non-mutating)."""
        if not self.accounting_available:
            raise NotImplementedError("Accounting backend not available")
        if additional_releases < 0:
            raise ValueError("additional_releases must be non-negative")
        projected_count = self._committed_count + additional_releases
        temp_accountant = RdpAccountant(
            neighboring_relation=_get_relation(self.config.neighboring_relation),
        )
        gd_event = GaussianDpEvent(noise_multiplier=self.config.noise_multiplier)
        if projected_count > 0:
            temp_accountant.compose(gd_event, count=projected_count)
        eps, _ = temp_accountant.get_epsilon_and_optimal_order(
            target_delta=self.config.target_delta
        )
        return float(eps)

    def _save_state(self) -> None:
        if self._state_path is None:
            return
        state = {
            "version": 1,
            "committed_count": self._committed_count,
            "noise_multiplier": self.config.noise_multiplier,
            "target_delta": self.config.target_delta,
            "max_epsilon": self.config.max_epsilon,
            "neighboring_relation": self.config.neighboring_relation,
        }
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._state_path.write_text(json.dumps(state, indent=2))

    def _load_state(self) -> None:
        if self._state_path is None or not self._state_path.exists():
            return
        try:
            state = json.loads(self._state_path.read_text())
        except (json.JSONDecodeError, OSError):
            raise ValueError("Corrupted accounting state file") from None
        if state.get("version") != 1:
            raise ValueError(f"Incompatible state version: {state.get('version')}, expected 1")
        if state.get("noise_multiplier") != self.config.noise_multiplier:
            raise ValueError("State noise_multiplier mismatch")
        if state.get("target_delta") != self.config.target_delta:
            raise ValueError("State target_delta mismatch")
        count = int(state.get("committed_count", 0))
        if count < 0:
            raise ValueError("Invalid committed_count in state")
        gd_event = GaussianDpEvent(noise_multiplier=self.config.noise_multiplier)
        if count > 0:
            self._accountant.compose(gd_event, count=count)
        self._committed_count = count


if not _DP_ACCOUNTING_AVAILABLE:
    import warnings

    warnings.warn(
        "insurance_fl.fl.privacy: dp_accounting backend not available. "
        "RDP accounting is unavailable; get_epsilon() raises "
        "NotImplementedError and would_exceed() fails closed.",
        UserWarning,
        stacklevel=2,
    )

__all__ = ["PrivacyConfig", "RdpAccountantWrapper", "build_privacy_config"]
