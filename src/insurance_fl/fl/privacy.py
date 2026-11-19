from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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
        sampling_rate=float(rc.get("sampling-rate", 1.0)),
        delta=float(rc.get("target-delta", 1e-5)),
    )


def _make_accountant() -> Any:
    """Create an RdpAccountant instance if dp_accounting is available."""
    if not _HAS_DP:
        raise PrivacyBackendUnavailable("dp-accounting not installed; refusing to report epsilon.")
    return RdpAccountant()


class RdpAccountantWrapper:
    """RDP accountant wrapper using ``dp_accounting`` backend.

    Reports two epsilons per round:
    - ``epsilon``: conservative value with q=1 (no subsampling amplification)
    - ``epsilon_amplified``: value assuming Poisson sampling with the
      configured sampling_rate (may be lower, but not valid for
      stratified sampling with min_per_country).
    """

    STATE_VERSION = 1
    _REQUIRED = {"version", "delta", "events"}

    def __init__(self, cfg: PrivacyConfig, state_path: str | Path | None = None):
        self.cfg = cfg
        self.rounds = 0
        self._state_path = Path(state_path) if state_path is not None else None
        # Conservative accountant: q=1, no Poisson amplification
        self._conservative = _make_accountant()
        # Amplified accountant placeholder (not all dp-accounting versions support sampling)
        self._amplified = None
        # Track per-round events for round-trip
        self.events: list[dict[str, Any]] = []
        # Load state if state_path is provided
        if self._state_path is not None and self._state_path.exists():
            self._load_state()

    def _compose_event(self, ev: dict[str, Any]) -> None:
        """Compose one recorded event into internal accountants without mutating history."""
        sigma = float(ev.get("sigma", self.cfg.noise_multiplier))
        g = GaussianDpEvent(noise_multiplier=sigma)
        self._conservative.compose(g)
        self.rounds += 1

    def _load_state(self) -> None:
        """Load state from state_path (called from __init__)."""
        if self._state_path is None or not self._state_path.exists():
            return
        try:
            state = json.loads(self._state_path.read_text())
        except (json.JSONDecodeError, OSError):
            raise ValueError("Corrupted accounting state file") from None
        self._restore_state(state)

    def _restore_state(self, state: dict[str, Any]) -> None:
        missing = self._REQUIRED - set(state.keys())
        if missing:
            raise ValueError(f"privacy state missing keys: {sorted(missing)}")
        if state.get("version") != self.STATE_VERSION:
            raise ValueError(
                f"Incompatible state version: {state.get('version')}, expected {self.STATE_VERSION}"
            )
        if state.get("delta") != self.cfg.delta:
            raise ValueError("delta changed across resume; epsilon not comparable")
        events = list(state.get("events", []))
        self.events = events
        self.rounds = 0
        self._conservative = _make_accountant()
        self._amplified = None
        for ev in self.events:
            self._compose_event(ev)

    @classmethod
    def from_state(cls, cfg: PrivacyConfig, state: dict[str, Any]) -> RdpAccountantWrapper:
        """Create accountant from state dict (for resuming runs)."""
        acc = cls.__new__(cls)
        acc.cfg = cfg
        acc.rounds = 0
        acc._state_path = None
        acc._conservative = _make_accountant()
        acc._amplified = None
        acc.events = []
        acc._restore_state(state)
        return acc

    def save_state(self, path: str | Path | None = None) -> None:
        """Save accountant state to disk (atomic via temp+rename)."""
        if path is not None:
            self._state_path = Path(path)
        if self._state_path is None:
            return
        state = self.state_dict()
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = str(self._state_path) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, str(self._state_path))

    @classmethod
    def load_state(
        cls, path: str | Path | None = None, cfg: PrivacyConfig | None = None
    ) -> RdpAccountantWrapper:
        """Load accountant state from disk."""
        cfg = cfg or PrivacyConfig()
        if path is None:
            return cls(cfg)
        state_path = Path(path)
        if not state_path.exists():
            return cls(cfg, state_path=state_path)
        with open(state_path, encoding="utf-8") as f:
            state = json.load(f)
        acc = cls.from_state(cfg, state)
        acc._state_path = state_path
        return acc

    def state_dict(self) -> dict[str, Any]:
        """Return serializable state dict for resuming runs."""
        return {
            "version": self.STATE_VERSION,
            "delta": self.cfg.delta,
            "events": list(self.events),
        }

    def step(self) -> None:
        """Record one round of Gaussian release."""
        ev = {"sigma": self.cfg.noise_multiplier, "q": 1.0}
        self.events.append(ev)
        self._compose_event(ev)
        # Keep events list bounded (max 100 rounds)
        if len(self.events) > 100:
            self.events = self.events[-100:]

    def compose(self, event: dict[str, Any]) -> None:
        """Record a privacy-relevant event (delegates to step for compatibility)."""
        self.step()

    def get_epsilon(self) -> float:
        """Return conservative (q=1) epsilon for current state."""
        return float(self.epsilon()["epsilon"])

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
        result: dict[str, Any] = {
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
        projected_count = self.rounds + int(additional_releases)
        temp_accountant = _make_accountant()
        gd_event = GaussianDpEvent(noise_multiplier=self.cfg.noise_multiplier)
        if projected_count > 0:
            temp_accountant.compose(gd_event, count=projected_count)
        eps, _ = temp_accountant.get_epsilon_and_optimal_order(target_delta=self.cfg.delta)
        return float(eps)

    def would_exceed(self) -> bool:
        """Fail-closed: return True if epsilon would exceed max acceptable."""
        # With zero noise, always exceed any finite budget
        if self.cfg.noise_multiplier <= 0:
            return True
        eps = self.epsilon()["epsilon"]
        return bool(eps > float(self.cfg.max_epsilon))


__all__ = [
    "PrivacyBackendUnavailable",
    "PrivacyConfig",
    "RdpAccountantWrapper",
    "build_privacy_config",
]
