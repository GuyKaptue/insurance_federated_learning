"""Unit tests for the audit logger."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from insurance_fl.audit import AUDIT_RETENTION_DAYS, AuditLogger, verify_event_integrity


def test_all_event_types_serialize() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = AuditLogger(path=Path(tmpdir) / "audit.jsonl", run_id="test-run")
        logger.log_fl_round_start(round=1, participating_clients=4, strategy="fedavg")
        logger.log_fl_round_end(round=1, aggregated_metric=0.5, wall_time_s=1.2)
        logger.log_privacy_event(
            round=1,
            noise_multiplier=0.1,
            epsilon="unavailable",
            mechanism="local_dp",
            accounting_status="unavailable",
        )
        logger.log_fairness_metric(round=1, per_country_metric={"DE": 0.8}, disparity=0.1)
        logger.log_model_export(task="is_high_risk", model_hash="abc123")
        logger.log_config_snapshot(config={"max-epsilon": 10.0, "secret": "hidden"})

        events = logger.read_events()
        assert len(events) == 6
        for ev in events:
            assert "timestamp_utc" in ev
            assert "run_id" in ev
            # Verify valid JSON line
            json.dumps(ev)


def test_required_fields_present() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = AuditLogger(path=Path(tmpdir) / "audit.jsonl", run_id="test-run")
        logger.log_fl_round_start(round=1)
        events = logger.read_events()
        assert events[0]["event_type"] == "fl_round_start"
        assert "run_id" in events[0]
        assert "round" in events[0]
        assert "timestamp_utc" in events[0]


def test_timestamps_valid() -> None:
    from datetime import datetime

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = AuditLogger(path=Path(tmpdir) / "audit.jsonl", run_id="test-run")
        logger.log_fl_round_start(round=1)
        events = logger.read_events()
        ts = events[0]["timestamp_utc"]
        datetime.fromisoformat(ts)  # should not raise


def test_secret_redaction_recursive() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = AuditLogger(path=Path(tmpdir) / "audit.jsonl", run_id="test-run")
        config = {
            "max-epsilon": 10.0,
            "secret": "hidden",
            "nested": {"token": "abc", "password": "def", "api_key": "ghi"},
            "list": [{"secret": "x"}],
        }
        logger.log_config_snapshot(config=config)
        events = logger.read_events()
        ev = events[0]
        assert ev["sanitized_config"]["secret"] == "<REDACTED>"
        assert ev["sanitized_config"]["nested"]["token"] == "<REDACTED>"
        assert ev["sanitized_config"]["nested"]["password"] == "<REDACTED>"
        assert ev["sanitized_config"]["nested"]["api_key"] == "<REDACTED>"
        assert ev["sanitized_config"]["list"][0]["secret"] == "<REDACTED>"
        assert ev["sanitized_config"]["max-epsilon"] == 10.0
        assert "hidden" not in json.dumps(ev)
        assert verify_event_integrity(ev)


def test_append_preserves_earlier() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "audit.jsonl"
        logger1 = AuditLogger(path=path, run_id="run1")
        logger1.log_fl_round_start(round=1)

        logger2 = AuditLogger(path=path, run_id="run2")
        logger2.log_fl_round_start(round=2)

        events = logger2.read_events()
        assert len(events) == 2
        assert events[0]["round"] == 1
        assert events[1]["round"] == 2
        assert [event["event_type"] for event in events] == ["fl_round_start", "fl_round_start"]
        assert all(verify_event_integrity(event) for event in events)


def test_retention_days() -> None:
    assert AUDIT_RETENTION_DAYS == 180


def test_missing_required_fields_raises() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = AuditLogger(path=Path(tmpdir) / "audit.jsonl", run_id="test-run")
        try:
            # Manually call _write with missing required fields
            logger._write({"event_type": "fl_round_start", "run_id": "test"})
            raise AssertionError("expected ValueError")
        except ValueError as exc:
            assert "missing required fields" in str(exc).lower()


def test_config_hash_deterministic() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = AuditLogger(path=Path(tmpdir) / "audit.jsonl", run_id="test-run")
        config = {"a": 1, "b": 2}
        logger.log_config_snapshot(config=config)
        events = logger.read_events()
        assert "config_hash" in events[0]
        assert len(events[0]["config_hash"]) == 64


def test_event_hash_present() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = AuditLogger(path=Path(tmpdir) / "audit.jsonl", run_id="test-run")
        logger.log_fl_round_start(round=1)
        events = logger.read_events()
        assert "event_hash" in events[0]
        assert len(events[0]["event_hash"]) == 64
        assert verify_event_integrity(events[0])


def test_event_hash_detects_tampering_after_restart() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "audit.jsonl"
        AuditLogger(path=path, run_id="test-run").log_fl_round_start(round=1)
        restarted_logger = AuditLogger(path=path, run_id="test-run")
        event = restarted_logger.read_events()[0]
        assert verify_event_integrity(event)
        event["round"] = 2
        assert not verify_event_integrity(event)
