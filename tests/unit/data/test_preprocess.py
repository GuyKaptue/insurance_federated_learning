"""Tests for insurance_fl.data.preprocess (fit-on-train, leakage-safe)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from insurance_fl.data.preprocess import DataPreprocessor, PreprocessorConfig


@pytest.fixture
def train_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 200
    return pd.DataFrame(
        {
            "age": rng.integers(18, 80, n),
            "annual_income": rng.normal(50000, 15000, n),
            "credit_score": rng.integers(300, 850, n),
            "city": rng.choice(["Berlin", "Paris", "Brussels", "Zurich"], n),
            "tier": rng.choice(["Basic", "Premium"], n),
            "claim_risk_score": rng.uniform(0, 100, n),  # target
            "customer_id": [f"C{i:05d}" for i in range(n)],  # id
            "country": rng.choice(["Germany", "France"], n),  # metadata
        }
    )


TARGETS = ["claim_risk_score"]


def _cfg(**kw) -> PreprocessorConfig:
    kw.setdefault("target_columns", TARGETS)
    return PreprocessorConfig(**kw)


# --- happy path -----------------------------------------------------------
def test_fit_transform_shapes(train_df: pd.DataFrame) -> None:
    pp = DataPreprocessor(_cfg())
    out = pp.fit_transform(train_df, targets=train_df[TARGETS])
    # 3 numeric + onehot(city) + onehot(tier) columns, plus target
    assert "claim_risk_score" in out.columns
    assert len(out) == len(train_df)
    assert not out.columns.duplicated().any()
    feature_cols = [c for c in out.columns if c not in TARGETS]
    assert any(c.startswith("city_") for c in feature_cols)


def test_scaler_fits_on_train_only(train_df: pd.DataFrame) -> None:
    """Val transform must NOT change train statistics (leakage guard)."""
    pp = DataPreprocessor(_cfg())
    pp.fit(train_df)
    train_out = pp.transform(train_df)
    # scaled train numerics: mean ~0, std ~1 for standard scaler
    age_col = [c for c in train_out.columns if c == "age"][0]
    assert abs(train_out[age_col].mean()) < 1e-6
    assert abs(train_out[age_col].std(ddof=0) - 1.0) < 1e-6

    val = train_df.iloc[:50].copy()
    val["age"] = val["age"] + 500  # shift val far away
    val_out = pp.transform(val)
    # scaler parameters must be untouched by transform
    assert abs(train_out[age_col].mean()) < 1e-6
    assert len(val_out) == 50


def test_targets_attached_separately(train_df: pd.DataFrame) -> None:
    pp = DataPreprocessor(_cfg())
    out = pp.fit_transform(train_df, targets=train_df[TARGETS])
    np.testing.assert_allclose(
        out["claim_risk_score"].to_numpy(), train_df["claim_risk_score"].to_numpy()
    )


def test_id_and_metadata_dropped(train_df: pd.DataFrame) -> None:
    pp = DataPreprocessor(_cfg())
    out = pp.fit_transform(train_df)
    for col in ("customer_id", "country"):
        assert col not in out.columns


# --- errors ---------------------------------------------------------------
def test_transform_before_fit_raises(train_df: pd.DataFrame) -> None:
    with pytest.raises(RuntimeError):
        DataPreprocessor(_cfg()).transform(train_df)


def test_transform_missing_columns_raises(train_df: pd.DataFrame) -> None:
    pp = DataPreprocessor(_cfg()).fit(train_df)
    with pytest.raises(ValueError):
        pp.transform(train_df.drop(columns=["age"]))


def test_fit_empty_or_wrong_type_raises() -> None:
    with pytest.raises(ValueError):
        DataPreprocessor(_cfg()).fit(pd.DataFrame())
    with pytest.raises(TypeError):
        DataPreprocessor(_cfg()).fit([1, 2, 3])  # type: ignore[arg-type]


def test_bad_scaler_or_encoder_rejected(train_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        DataPreprocessor(_cfg(scaling_method="magic")).fit(train_df)
    with pytest.raises(ValueError):
        DataPreprocessor(_cfg(encoding_method="magic")).fit(train_df)


# --- edge cases -----------------------------------------------------------
def test_unseen_category_encoded_safely(train_df: pd.DataFrame) -> None:
    pp = DataPreprocessor(_cfg(encoding_method="label")).fit(train_df)
    val = train_df.copy()
    val.loc[val.index[:5], "city"] = "NewCity"
    out = pp.transform(val)
    city_col = [c for c in out.columns if c == "city"][0]
    assert (out[city_col].iloc[:5] == -1).all()  # documented unseen -> -1


def test_nan_imputation_leaves_no_nans(train_df: pd.DataFrame) -> None:
    df = train_df.copy()
    df.loc[df.index[:10], "annual_income"] = np.nan
    pp = DataPreprocessor(_cfg())
    out = pp.fit_transform(df)
    assert int(out.isna().sum().sum()) == 0


def test_minmax_scaler_range(train_df: pd.DataFrame) -> None:
    pp = DataPreprocessor(_cfg(scaling_method="minmax", encoding_method="label"))
    out = pp.fit_transform(train_df)
    assert out["age"].min() >= 0.0 and out["age"].max() <= 1.0
