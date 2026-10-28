"""Re-export shim for causal truth leakage regression tests."""

from __future__ import annotations

from insurance_fl.causal.truth import (
    EFFECT_COLUMN,
    OUTCOME_TRUE,
    inject_truth,
    ml_feature_columns,
    ml_target_columns,
)

__all__ = [
    "EFFECT_COLUMN",
    "OUTCOME_TRUE",
    "inject_truth",
    "ml_feature_columns",
    "ml_target_columns",
]
