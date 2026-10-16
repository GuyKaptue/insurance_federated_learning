"""Tests for insurance_fl.config.loader (happy path, validation, errors)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from insurance_fl.config.loader import (
    DPConfig,
    FederatedConfig,
    FullConfig,
    TasksConfig,
    load_config,
)

CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "federated_config.yaml"


def _write_yaml(tmp_path: Path, payload: dict) -> Path:
    p = tmp_path / "federated_config.yaml"
    p.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return p


# --- happy path -----------------------------------------------------------
def test_load_real_config_file() -> None:
    """The repo's actual config/federated_config.yaml must validate."""
    cfg = load_config(CONFIG_PATH)
    assert isinstance(cfg, FullConfig)
    assert cfg.federated.num_rounds >= 1
    assert cfg.federated.strategy_type in {"fedavg", "fedprox", "fedadam", "fedyogi"}
    assert len(cfg.tasks) >= 1
    assert sum(t.type == "regression" for t in cfg.tasks) == 2
    assert sum(t.type == "binary_classification" for t in cfg.tasks) == 2
    assert sum(t.type == "multi_classification" for t in cfg.tasks) == 1


def test_defaults_when_yaml_empty(tmp_path: Path) -> None:
    cfg = load_config(_write_yaml(tmp_path, {}))
    assert cfg.federated.num_rounds == 10
    assert cfg.dp.enabled is False


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


# --- validation -----------------------------------------------------------
def test_bad_strategy_rejected() -> None:
    with pytest.raises(ValidationError):
        FederatedConfig(strategy_type="fedsumo")


def test_bad_fraction_rejected() -> None:
    with pytest.raises(ValidationError):
        FederatedConfig(fraction_fit=0.0)
    with pytest.raises(ValidationError):
        FederatedConfig(fraction_fit=1.5)


def test_bad_dp_type_rejected() -> None:
    with pytest.raises(ValidationError):
        DPConfig(dp_type="central")


def test_splits_must_sum_to_one(tmp_path: Path) -> None:
    payload = {"dataset": {"splits": {"train": 0.9, "val": 0.9}}}
    with pytest.raises(ValidationError):
        load_config(_write_yaml(tmp_path, payload))


def test_malformed_yaml_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("federated: [unclosed", encoding="utf-8")
    with pytest.raises(yaml.YAMLError):
        load_config(p)


def test_tasks_config_wraps_list() -> None:
    tc = TasksConfig.from_list(
        [
            {"name": "will_renew", "type": "binary_classification"},
            {"name": "customer_segment", "type": "multi_classification", "classes": [0, 1, 2]},
        ]
    )
    assert len(tc.tasks) == 2
    with pytest.raises(ValidationError):
        TasksConfig.from_list([{"name": "x", "type": "clustering"}])
