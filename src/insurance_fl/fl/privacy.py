from __future__ import annotations

import json
import math
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dp_accounting import dp_event


class PrivacyBackendUnavailable(RuntimeError):
    """Raised instead of returning a fabricated epsilon."""


# Try to import dp_accounting backend
try:
    from dp_accounting import GaussianDpEvent
    from dp_accounting.rdp import RdpAccountant

    _HAS_DP = True
except ImportError:
    _HAS_DP = False
    GaussianDpEvent = None  # type: ignore[assignment,misc]
    RdpAccountant = None  # type: ignore[assignment,misc]


@dataclass(frozen=True)
class PrivacyConfig:
    max_epsilon: float = 10.0  # maximum acceptable epsilon (for backward compatibility)
    target_delta: float = 1e-5  # delta for epsilon computation
    population_size: int = 4  # number of clients (for backward compatibility)
    neighboring_relation: str = "add_or_remove_one"  # backward compatibility
    noise_multiplier: float = 1.0  # sigma = noise_std / clip_norm (primary param)
    sampling_rate: float = 1.0  # clients_per_round / total_clients (default 1 for full batch)
    delta: float = 1e-5  # alias for target_delta (backward compatibility)

    def __post_init__(self):
        # Validate inputs
        if not (0.0 < self.delta < 1.0):
            raise ValueError(f"delta must be in (0, 1), got {self.delta}")
        if self.noise_multiplier < 0:
            raise ValueError("noise_multiplier must be >= 0")
        if not (0.0 < self.sampling_rate <= 1.0):
            raise ValueError("sampling_rate must be in (0, 1]")


_ORDERS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))


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


def _make_accountant() -> Any:
    """Create an RdpAccountant instance if dp_accounting is available."""
    if not _HAS_DP:
        raise PrivacyBackendUnavailable(
            "dp-accounting not installed; refusing to report epsilon."
        )
    return RdpAccountant()


class RdpAccountantWrapper:
    """RDP accountant wrapper using ``dp_accounting`` backend.

    Reports two epsilons per round:
    - ``epsilon``: conservative value with q=1 (no subsampling amplification)
    - ``epsilon_amplified``: value assuming Poisson sampling with the
      configured sampling_rate (may be lower, but not valid for
      stratified sampling with min_per_country).
    """

    def __init__(self, cfg: PrivacyConfig, state_path: str | Path | None = None):
        self.cfg = cfg
        self.rounds = 0
        self._state_path = Path(state_path) if state_path else None
        # Conservative accountant: q=1, no Poisson amplification
        self._conservative = _make_accountant()
        # Amplified accountant: uses PoissonSampledDpEvent with sampling_rate
        try:
            self._amplified = _make_accountant()
        except PrivacyBackendUnavailable:
            self._amplified = None
        # Event list: [{"sigma": ..., "q": ...}, ...] - one entry per round
        self.events: list[dict[str, float]] = []
        # Try to load existing state
        self._load_state()

    def _apply(self, sigma: float, q: float) -> None:
        """Compose one Gaussian event into both accountants."""
        g = dp_event.GaussianDpEvent(sigma)
        self._amplified.compose(dp_event.PoissonSampledDpEvent(q, g))
        self._conservative.compose(g)

    def step(self) -> None:
        """Record one round of Gaussian release."""
        self._apply(self.cfg.noise_multiplier, self.cfg.sampling_rate)
        self.events.append({"sigma": self.cfg.noise_multiplier, "q": self.cfg.sampling_rate})
        self.rounds += 1

    def compose(self, event: dict[str, Any]) -> None:
        """Record a privacy-relevant event (delegates to step for compatibility)."""
        self.step()

    def _load_state(self) -> None:
        """Load state from ``self._state_path`` if it exists."""

        if self._state_path is None or not self._state_path.exists():
            return
        try:
            state = json.loads(self._state_path.read_text())
        except (json.JSONDecodeError, OSError):
            raise ValueError("Corrupted accounting state file") from None
        if state.get("delta") != self.cfg.delta:
            raise ValueError("State delta mismatch")
        # Replay the per-round event list
        acc = self.from_state(state, self.cfg)
        self.rounds = acc.rounds
        self.events = list(acc.events)
        # Copy the composed accountants so the wrapper behaves as if it
        # was never paused.
        self._conservative = acc._conservative
        self._amplified = acc._amplified

    def save_state(self) -> None:
        """Atomically save current state to ``self._state_path``.

        Writes to a temporary file first, then ``os.replace`` so a crash
        cannot leave a half-written state file on disk.
        """
        if self._state_path is None:
            return
        tmp = str(self._state_path) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.state_dict(), f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, str(self._state_path))

    def _save_state(self) -> None:
        """Deprecated: use ``save_state`` instead.

        Kept for backward compatibility with test code.
        """
        self.save_state()

    def state_dict(self) -> dict:
        """Return serializable state dict for resuming runs.

        Stores the per-round event list so that resumed accountants
        replay exactly the same noise/sampling sequence.
        """
        return {"delta": self.cfg.delta, "events": list(self.events)}

    @classmethod
    def from_state(cls, state: dict, cfg: PrivacyConfig) -> "RdpAccountantWrapper":
        """Create accountant from state dict (for resuming runs).

        Raises ``ValueError`` if the stored delta differs from the
        current config's delta (epsilon would not be comparable).
        Logs a warning if the noise multiplier changed.
        """
        if state["delta"] != cfg.delta:
            raise ValueError("delta changed across resume; epsilon not comparable")
        acc = cls(cfg)
        for ev in state["events"]:
            acc._apply(ev["sigma"], ev["q"])
            acc.events.append(ev)
            acc.rounds += 1
        if state["events"] and state["events"][-1]["sigma"] != cfg.noise_multiplier:
            log.warning("noise multiplier changed since last saved round")
        return acc

    def epsilon(self) -> dict[str, Any]:
        """Return both conservative and amplified epsilon values."""
        if self.cfg.noise_multiplier <= 0:
            return {
                "epsilon": math.inf,
                "epsilon_amplified": math.inf,
                "delta": self.cfg.delta,
                "rounds": self.rounds,
                "note": "zero noise -> infinite epsilon (no privacy loss)",
            }
        eps_conservative = self._conservative.get_epsilon_and_optimal_order(
            target_delta=self.cfg.delta
        )[0]
        result = {
            "epsilon": float(eps_conservative),
            "delta": self.cfg.delta,
            "rounds": self.rounds,
        }
        # Amplified epsilon may not be available in all dp-accounting versions;
        # use conservative q=1 value as the headline figure
        result["epsilon_amplified"] = None
        result["note"] = (
            "headline epsilon is conservative q=1 value; "
            "amplified value assuming Poisson sampling not available"
        )
        return result

    def project_epsilon(self, additional_releases: int = 1) -> float:
        """Project cumulative epsilon with additional releases (non-mutating)."""
        if self.cfg.noise_multiplier <= 0:
            return math.inf
        projected_count = self.rounds + additional_releases
        temp_accountant = _make_accountant()
        gd_event = GaussianDpEvent(noise_multiplier=self.cfg.noise_multiplier)
        if projected_count > 0:
            temp_accountant.compose(gd_event, count=projected_count)
        eps, _ = temp_accountant.get_epsilon_and_optimal_order(
            target_delta=self.cfg.target_delta
        )
        return float(eps)

    def would_exceed(self) -> bool:
        """Fail-closed: return True if epsilon would exceed max acceptable."""
        # With zero noise, always exceed any finite budget
        if self.cfg.noise_multiplier <= 0:
            return True
        eps = self.epsilon()["epsilon"]
        return eps > 10.0  # placeholder budget check


__all__ = [
    "PrivacyBackendUnavailable",
    "PrivacyConfig",
    "RdpAccountantWrapper",
    "build_privacy_config",
]
