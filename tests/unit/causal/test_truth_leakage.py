"""Leakage tests for semi-synthetic causal truth columns (DQ-I04)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from insurance_fl.causal.truth import (
    EFFECT_COLUMN,
    OUTCOME_TRUE,
    inject_truth,
    ml_feature_columns,
    ml_target_columns,
)


def _sample_df(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "country": rng.choice(["Germany", "France", "Belgium", "Switzerland"], size=n),
            "number_of_claims": rng.poisson(1.0, size=n),
            "claim_risk_score": rng.normal(10.0, 2.0, size=n),
            "annual_income": rng.normal(50000.0, 10000.0, size=n),
            "age": rng.integers(18, 80, size=n),
            "savings_balance": rng.normal(10000.0, 2000.0, size=n),
            "credit_score": rng.integers(300, 850, size=n),
            "debt_to_income_ratio": rng.random(size=n),
            "customer_lifetime_value": rng.normal(1000.0, 200.0, size=n),
            "will_renew": rng.integers(0, 2, size=n),
            "is_high_risk": rng.integers(0, 2, size=n),
            "customer_segment": rng.integers(0, 3, size=n),
        }
    )


def test_truth_not_in_ml_targets() -> None:
    assert OUTCOME_TRUE not in ml_target_columns()
    assert EFFECT_COLUMN not in ml_target_columns()


def test_truth_not_in_ml_features() -> None:
    df = inject_truth(_sample_df(), seed=7, base_effect=5.0)
    features = ml_feature_columns(df)
    assert OUTCOME_TRUE not in features
    assert EFFECT_COLUMN not in features


def test_truth_not_predictable_beyond_chance() -> None:
    df = inject_truth(_sample_df(), seed=7, base_effect=5.0)
    features = ml_feature_columns(df)
    X = df[features].fillna(0.0)
    y = df[OUTCOME_TRUE]
    model = LinearRegression().fit(X, y)
    r2 = r2_score(y, model.predict(X))
    assert r2 <= 0.99


def test_targets_do_not_perfectly_predict_truth() -> None:
    df = inject_truth(_sample_df(), seed=7, base_effect=5.0)
    targets = [
        "claim_risk_score",
        "customer_lifetime_value",
        "will_renew",
        "is_high_risk",
        "customer_segment",
    ]
    X = df[targets].fillna(0.0)
    y = df[OUTCOME_TRUE]
    model = LinearRegression().fit(X, y)
    r2 = r2_score(y, model.predict(X))
    assert r2 <= 0.99
