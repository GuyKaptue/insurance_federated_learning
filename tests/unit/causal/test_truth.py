"""Tests for src/insurance_fl/causal/truth.py."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from insurance_fl.causal.truth import (
    EFFECT_COLUMN,
    OUTCOME_TRUE,
    TREATMENT,
    TRUTH_COLUMNS,
    ground_truth_ate,
    inject_truth,
    ml_feature_columns,
    ml_target_columns,
    write_ground_truth,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "country": ["Germany", "France", "Belgium", "Switzerland"],
            "number_of_claims": [0, 1, 2, 0],
            "claim_risk_score": [1.0, 2.0, 3.0, 4.0],
            "annual_income": [10.0, 20.0, 30.0, 40.0],
        }
    )


def test_inject_truth_adds_columns() -> None:
    df = _sample_df()
    out = inject_truth(df, seed=7, base_effect=5.0)
    assert "claim_risk_score_true" in out.columns
    assert "effect_has_claims" in out.columns
    assert "has_claims" in out.columns
    assert out["claim_risk_score_true"].tolist() == [1.0, 7.5, 7.5, 4.0]


def test_inject_truth_deterministic() -> None:
    df = _sample_df()
    a = inject_truth(df, seed=7, base_effect=5.0)
    b = inject_truth(df, seed=7, base_effect=5.0)
    pd.testing.assert_frame_equal(a, b)


def test_ground_truth_ate() -> None:
    df = inject_truth(_sample_df(), seed=7, base_effect=5.0)
    assert ground_truth_ate(df) == 5.0 * 2 / 4


def test_ml_columns_exclude_truth() -> None:
    df = inject_truth(_sample_df(), seed=7, base_effect=5.0)
    assert "claim_risk_score_true" not in ml_target_columns()
    assert "claim_risk_score_true" not in ml_feature_columns(df)
    assert "effect_has_claims" not in ml_feature_columns(df)


def test_write_ground_truth(tmp_path: Path) -> None:
    df = inject_truth(_sample_df(), seed=7, base_effect=5.0)
    path = write_ground_truth(tmp_path, df, seed=7, base_effect=5.0)
    payload = json.loads(path.read_text())
    assert payload["ate"] == ground_truth_ate(df)
    assert payload["treatment"] == TREATMENT
    assert payload["outcome"] == OUTCOME_TRUE


def test_truth_constants() -> None:
    assert TRUTH_COLUMNS == ("claim_risk_score_true", "effect_has_claims")
    assert TREATMENT == "has_claims"
    assert OUTCOME_TRUE == "claim_risk_score_true"
    assert EFFECT_COLUMN == "effect_has_claims"
