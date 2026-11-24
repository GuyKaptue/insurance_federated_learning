"""Structured, append-only JSONL audit trail logger.

Produces one JSON object per line. Event types:
- fl_round_start
- fl_round_end
- privacy_event
- fairness_metric
- model_export
- config_snapshot

Security: redacts secret-bearing config keys (case-insensitive: secret,
token, password, key). Never logs raw data, PII, individual model
updates, or credentials. Append-only application behavior is not
tamper-proof; deployment-level controls are required.

Retention: AUDIT_RETENTION_DAYS = 180 (intended six-month policy, not
proof of legal compliance). Automatic deletion is NOT implemented;
enforcement belongs to deployment and governance operations.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

AUDIT_RETENTION_DAYS = 180

_SECRET_KEYS = {"secret", "token", "password", "key"}

_REQUIRED_FIELDS: dict[str, set[str]] = {
    "fl_round_start": {"run_id", "round", "timestamp_utc"},
    "fl_round_end": {"run_id", "round", "timestamp_utc"},
    "privacy_event": {"run_id", "timestamp_utc"},
    "fairness_metric": {"run_id", "round", "timestamp_utc"},
    "model_export": {"run_id", "timestamp_utc"},
    "config_snapshot": {"run_id", "timestamp_utc"},
}

_lock = threading.Lock()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _redact(obj: Any, depth: int = 0) -> Any:
    """Recursively redact secret-bearing keys (case-insensitive)."""
    if depth > 20:
        return "<max-depth>"
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            if any(s in str(k).lower() for s in _SECRET_KEYS):
                result[k] = "<REDACTED>"
            else:
                result[k] = _redact(v, depth + 1)
        return result
    if isinstance(obj, (list, tuple)):
        return [_redact(item, depth + 1) for item in obj]
    return obj


def _config_hash(config: dict[str, Any]) -> str:
    """Deterministic hash of a config dict."""
    serialized = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()


def _event_hash(event: dict[str, Any]) -> str:
    """Deterministic hash of an event (excluding event_hash itself)."""
    clean = {k: v for k, v in event.items() if k != "event_hash"}
    serialized = json.dumps(clean, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()


@dataclass
class AuditLogger:
    """Append-only JSONL audit logger."""

    path: Path
    run_id: str
    _events: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, event: dict[str, Any]) -> None:
        """Append a single event as a JSON line. Thread-safe."""
        # Validate required fields
        event_type = event.get("event_type")
        if event_type in _REQUIRED_FIELDS:
            missing = _REQUIRED_FIELDS[event_type] - set(event.keys())
            if missing:
                raise ValueError(f"Event {event_type!r} missing required fields: {sorted(missing)}")

        # Add timestamp if missing
        if "timestamp_utc" not in event:
            event["timestamp_utc"] = _utc_now()

        # Add event hash
        event["event_hash"] = _event_hash(event)

        # Redact secrets
        event = _redact(event)

        line = json.dumps(event, default=str, ensure_ascii=False) + "\n"

        with _lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())

        self._events.append(event)

    def log_fl_round_start(
        self,
        round: int,
        participating_clients: int | None = None,
        strategy: str | None = None,
        data_version: str | None = None,
        **extra: Any,
    ) -> None:
        event: dict[str, Any] = {
            "event_type": "fl_round_start",
            "run_id": self.run_id,
            "round": round,
            "timestamp_utc": _utc_now(),
        }
        if participating_clients is not None:
            event["participating_clients"] = participating_clients
        if strategy is not None:
            event["strategy"] = strategy
        if data_version is not None:
            event["data_version"] = data_version
        event.update(extra)
        self._write(event)

    def log_fl_round_end(
        self,
        round: int,
        aggregated_metric: float | None = None,
        cumulative_epsilon: float | str | None = None,
        wall_time_s: float | None = None,
        **extra: Any,
    ) -> None:
        event: dict[str, Any] = {
            "event_type": "fl_round_end",
            "run_id": self.run_id,
            "round": round,
            "timestamp_utc": _utc_now(),
        }
        if aggregated_metric is not None:
            event["aggregated_metric"] = aggregated_metric
        if cumulative_epsilon is not None:
            event["cumulative_epsilon"] = cumulative_epsilon
        if wall_time_s is not None:
            event["wall_time_s"] = wall_time_s
        event.update(extra)
        self._write(event)

    def log_privacy_event(
        self,
        round: int | None = None,
        noise_multiplier: float | None = None,
        epsilon: float | str | None = None,
        mechanism: str | None = None,
        accounting_status: str | None = None,
        **extra: Any,
    ) -> None:
        event: dict[str, Any] = {
            "event_type": "privacy_event",
            "run_id": self.run_id,
            "timestamp_utc": _utc_now(),
        }
        if round is not None:
            event["round"] = round
        if noise_multiplier is not None:
            event["noise_multiplier"] = noise_multiplier
        if epsilon is not None:
            event["epsilon"] = epsilon
        if mechanism is not None:
            event["mechanism"] = mechanism
        if accounting_status is not None:
            event["accounting_status"] = accounting_status
        event.update(extra)
        self._write(event)

    def log_fairness_metric(
        self,
        round: int,
        per_country_metric: dict[str, Any] | None = None,
        disparity: float | None = None,
        **extra: Any,
    ) -> None:
        event: dict[str, Any] = {
            "event_type": "fairness_metric",
            "run_id": self.run_id,
            "round": round,
            "timestamp_utc": _utc_now(),
        }
        if per_country_metric is not None:
            event["per_country_metric"] = per_country_metric
        if disparity is not None:
            event["disparity"] = disparity
        event.update(extra)
        self._write(event)

    def log_model_export(
        self,
        task: str,
        model_hash: str,
        beta_weights: list[float] | None = None,
        nn_param_hash: str | None = None,
        **extra: Any,
    ) -> None:
        event: dict[str, Any] = {
            "event_type": "model_export",
            "run_id": self.run_id,
            "task": task,
            "model_hash": model_hash,
            "timestamp_utc": _utc_now(),
        }
        if beta_weights is not None:
            event["beta_weights"] = beta_weights
        if nn_param_hash is not None:
            event["nn_param_hash"] = nn_param_hash
        event.update(extra)
        self._write(event)

    def log_config_snapshot(
        self,
        config: dict[str, Any],
        **extra: Any,
    ) -> None:
        event: dict[str, Any] = {
            "event_type": "config_snapshot",
            "run_id": self.run_id,
            "config_hash": _config_hash(config),
            "sanitized_config": _redact(config),
            "timestamp_utc": _utc_now(),
        }
        event.update(extra)
        self._write(event)

    @property
    def event_count(self) -> int:
        return len(self._events)

    def read_events(self) -> list[dict[str, Any]]:
        """Read all events from the JSONL file."""
        if not self.path.exists():
            return []
        events = []
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
        return events


__all__ = ["AUDIT_RETENTION_DAYS", "AuditLogger"]
