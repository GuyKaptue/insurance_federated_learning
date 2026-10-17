"""
Configuration loader for federated_config.yaml.

Loads and validates the YAML configuration file using pydantic for
schema enforcement. Malformed configuration fails loudly with clear
error messages.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator


class DatasetConfig(BaseModel):
    """Dataset generation configuration."""

    name: str = "european_insurance_federated"
    path: str = "data/processed/european_insurance_federated"
    version: str = "v1.0"
    random_seed: int = 42
    base_samples_per_client: int = 5000
    client_sizes: dict = Field(default_factory=dict)
    site_data_size: float = 0.05
    splits: dict = Field(
        default_factory=lambda: {
            "train": 0.60,
            "val": 0.20,
            "test": 0.15,
            "site": 0.05,
        }
    )

    @field_validator("site_data_size")
    @classmethod
    def _validate_site_size(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError("site_data_size must be between 0 and 1")
        return v

    @field_validator("splits")
    @classmethod
    def _validate_splits(cls, v: dict) -> dict:
        total = sum(v.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"split proportions must sum to 1.0, got {total}")
        return v


class FederatedConfig(BaseModel):
    """Federated learning configuration."""

    num_rounds: int = 10
    num_clients: int = 4
    clients_per_round: int = 4
    fraction_fit: float = 1.0
    fraction_evaluate: float = 0.5
    min_fit_clients: int = 2
    min_evaluate_clients: int = 2
    min_available_clients: int = 2
    local_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.001
    strategy_type: str = "fedavg"
    aggregation_type: str = "weighted"
    proximal_mu: float = 0.1
    server_learning_rate: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    early_stopping_patience: int = 20
    validation_split: float = 0.2
    warmup_strategy: str = "none"
    warmup_rounds: int = 3

    @field_validator("strategy_type")
    @classmethod
    def _validate_strategy(cls, v: str) -> str:
        allowed = {"fedavg", "fedprox", "fedadam", "fedyogi"}
        if v.lower() not in allowed:
            raise ValueError(f"strategy_type must be one of {allowed}, got {v}")
        return v.lower()

    @field_validator("fraction_fit", "fraction_evaluate")
    @classmethod
    def _validate_fraction(cls, v: float) -> float:
        if not 0 < v <= 1:
            raise ValueError(f"fraction must be in (0, 1], got {v}")
        return v


class DPConfig(BaseModel):
    """Differential privacy configuration."""

    enabled: bool = False
    dp_type: str = "none"
    epsilon: float = 2.0
    delta: float = 1e-5
    clipping_norm: float = 1.0
    noise_multiplier: float = 0.1

    @field_validator("dp_type")
    @classmethod
    def _validate_dp_type(cls, v: str) -> str:
        allowed = {"none", "local_client", "central_server"}
        if v not in allowed:
            raise ValueError(f"dp_type must be one of {allowed}, got {v}")
        return v


class ModelConfig(BaseModel):
    """Model configuration."""

    ensemble_alpha: float = 0.5
    optimize_alpha: bool = True
    alpha_range: list[float] = [0.01, 0.99]
    alpha_search_steps: int = 11
    nn_hidden_dims: list[int] = [256, 128, 64]
    nn_dropout: float = 0.3
    nn_weight_decay: float = 1e-4
    nn_learning_rate: float = 0.001
    nn_epochs: int = 100
    nn_batch_size: int = 32
    use_ensemble_features: bool = True
    use_batch_norm: bool = True

    @field_validator("nn_dropout")
    @classmethod
    def _validate_dropout(cls, v: float) -> float:
        if not 0 <= v < 1:
            raise ValueError(f"nn_dropout must be in [0, 1), got {v}")
        return v


class TaskConfig(BaseModel):
    """Configuration for a single ML task."""

    name: str
    type: str = Field(..., description="regression, binary_classification, multi_classification")
    range: list[float] | None = None
    classes: list[int] | None = None

    @field_validator("type")
    @classmethod
    def _validate_task_type(cls, v: str) -> str:
        allowed = {"regression", "binary_classification", "multi_classification"}
        if v not in allowed:
            raise ValueError(f"task type must be one of {allowed}, got {v}")
        return v


class TasksConfig(BaseModel):
    """List of ML task configurations."""

    tasks: list[TaskConfig] = Field(default_factory=list)

    @classmethod
    def from_list(cls, items: list[TaskConfig]) -> TasksConfig:
        """Wrap a plain list of task configs."""
        return cls(tasks=items)


class ExplainabilityConfig(BaseModel):
    """Explainability configuration."""

    shap_sample_size: int = 1000
    shap_background_size: int = 100
    enable_shap: bool = True
    enable_audit: bool = True
    enable_personas: bool = True


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = "INFO"
    log_dir: str = "logs"


class FullConfig(BaseModel):
    """Complete configuration container."""

    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    federated: FederatedConfig = Field(default_factory=FederatedConfig)
    dp: DPConfig = Field(default_factory=DPConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    tasks: list[TaskConfig] = Field(default_factory=list)
    explainability: ExplainabilityConfig = Field(default_factory=ExplainabilityConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def load_config(path: str | Path | None = None) -> FullConfig:
    """
    Load and validate configuration from a YAML file.

    Parameters
    ----------
    path : Optional[str | Path]
        Path to federated_config.yaml. If None, uses the default
        config/federated_config.yaml relative to the project root.

    Returns
    -------
    FullConfig
        Validated configuration object.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    yaml.YAMLError
        If the YAML is malformed.
    pydantic.ValidationError
        If the config fails schema validation.
    """
    if path is None:
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        path = project_root / "config" / "federated_config.yaml"

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raw = {}

    return FullConfig(**raw)
