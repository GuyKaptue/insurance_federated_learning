"""Unit tests for the data readiness gate (src/insurance_fl/data/readiness.py).

Tests the post-preprocess contract that a client partition must satisfy
before it is consumed by training or evaluation: no missing values, no
non-finite values, sufficient samples, unique customer IDs, and no
zero-variance features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from insurance_fl.data.readiness import DataNotReadyError, ReadinessReport, check_data_readiness


def _make_clean_df() -> pd.DataFrame:
    """Generate a healthy partition DataFrame with no quality issues."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame(
        {
            "age": np.random.uniform(18, 90, n),
            "annual_income": np.random.uniform(20000, 150000, n),
            "years_as_customer": np.random.uniform(0, 60, n),
            "claim_risk_score": np.random.uniform(0, 100, n),
            "customer_lifetime_value": np.random.uniform(1000, 250000, n),
            "will_renew": np.random.randint(0, 2, n),
            "is_high_risk": np.random.randint(0, 2, n),
            "customer_segment": np.random.randint(0, 3, n),
            "customer_id": [f"cust_{i}" for i in range(n)],
        }
    )


@pytest.fixture
def clean_df() -> pd.DataFrame:
    """A healthy partition with no quality issues."""
    return _make_clean_df()


@pytest.fixture
def dirty_df() -> pd.DataFrame:
    """A partition with a missing value in a feature column."""
    df = _make_clean_df()
    df.loc[0, "annual_income"] = np.nan
    return df


@pytest.fixture
def zero_variance_df() -> pd.DataFrame:
    """A partition where a numeric feature column is constant."""
    df = _make_clean_df()
    df["annual_income"] = 999.99  # zero variance
    return df


@pytest.fixture
def duplicate_id_df() -> pd.DataFrame:
    """A partition with duplicate customer_id values."""
    df = _make_clean_df()
    # Make the first two rows share the same customer_id
    df.loc[1, "customer_id"] = df.loc[0, "customer_id"]
    return df


@pytest.fixture
def non_finite_df() -> pd.DataFrame:
    """A partition with a +inf value."""
    df = _make_clean_df()
    df.loc[2, "age"] = np.inf
    return df


@pytest.fixture
def sample_data_dir(tmp_path) -> str:
    """Create a minimal federated dataset layout with one client."""
    data_dir = tmp_path / "data"
    client_dir = data_dir / "federated" / "germany_allianz"
    client_dir.mkdir(parents=True)
    (client_dir / "train.parquet").write_bytes(b"dummy")
    return str(data_dir)


class TestReadinessReport:
    """Tests for the ReadinessReport dataclass."""

    def test_ready_report_is_ready(self) -> None:
        report = ReadinessReport(
            client_id="test",
            split="train",
            n_samples=100,
            n_features=5,
            missing_values=0,
            non_finite_count=0,
            duplicates=0,
            feature_variance_min=1.0,
            feature_variance_max=5.0,
            is_ready=True,
            failures=[],
        )
        assert report.is_ready is True

    def test_unready_report_is_not_ready(self) -> None:
        report = ReadinessReport(
            client_id="test",
            split="train",
            n_samples=100,
            n_features=5,
            missing_values=1,
            non_finite_count=0,
            duplicates=0,
            feature_variance_min=0.0,
            feature_variance_max=5.0,
            is_ready=False,
            failures=["missing values"],
        )
        assert report.is_ready is False
        assert report.failures == ["missing values"]


class TestCheckDataReadiness:
    """Tests for check_data_readiness()."""

    def test_clean_partition_is_ready(self, clean_df: pd.DataFrame, sample_data_dir: str) -> None:
        """A clean partition passes the gate."""
        # Write the clean DataFrame to the expected location
        clean_df.to_parquet(
            sample_data_dir + "/federated/germany_allianz/train.parquet", index=False
        )

        report = check_data_readiness(sample_data_dir, "germany_allianz", "train")
        assert report.is_ready is True
        assert report.n_samples == 200
        assert report.n_features == 8  # all numeric columns except excluded
        assert report.missing_values == 0
        assert report.non_finite_count == 0
        assert report.duplicates == 0
        assert report.feature_variance_min > 0

    def test_missing_file_raises_file_not_found(self, sample_data_dir: str) -> None:
        """Missing partition file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            check_data_readiness(sample_data_dir, "nonexistent_client", "train")

    def test_missing_values_are_reported(
        self, dirty_df: pd.DataFrame, sample_data_dir: str
    ) -> None:
        """A partition with missing values is reported as not ready."""
        dirty_df.to_parquet(
            sample_data_dir + "/federated/germany_allianz/train.parquet", index=False
        )

        report = check_data_readiness(sample_data_dir, "germany_allianz", "train")
        assert report.is_ready is False
        assert report.missing_values == 1
        assert len(report.failures) > 0
        assert any("missing" in f.lower() for f in report.failures)

    def test_zero_variance_columns_are_reported(
        self, zero_variance_df: pd.DataFrame, sample_data_dir: str
    ) -> None:
        """A partition with zero-variance columns is reported as not ready."""
        zero_variance_df.to_parquet(
            sample_data_dir + "/federated/germany_allianz/train.parquet", index=False
        )

        report = check_data_readiness(sample_data_dir, "germany_allianz", "train")
        assert report.is_ready is False
        assert any("zero variance" in f.lower() for f in report.failures)

    def test_duplicate_ids_are_reported(
        self, duplicate_id_df: pd.DataFrame, sample_data_dir: str
    ) -> None:
        """A partition with duplicate customer IDs is reported as not ready."""
        duplicate_id_df.to_parquet(
            sample_data_dir + "/federated/germany_allianz/train.parquet", index=False
        )

        report = check_data_readiness(sample_data_dir, "germany_allianz", "train")
        assert report.is_ready is False
        assert report.duplicates == 1
        assert any("duplicate" in f.lower() for f in report.failures)

    def test_non_finite_values_are_reported(
        self, non_finite_df: pd.DataFrame, sample_data_dir: str
    ) -> None:
        """A partition with non-finite values is reported as not ready."""
        non_finite_df.to_parquet(
            sample_data_dir + "/federated/germany_allianz/train.parquet", index=False
        )

        report = check_data_readiness(sample_data_dir, "germany_allianz", "train")
        assert report.is_ready is False
        assert report.non_finite_count == 1
        assert any("non-finite" in f.lower() for f in report.failures)

    def test_batch_norm_guard_requires_min_samples(self, sample_data_dir: str) -> None:
        """A partition with fewer than 2 samples is reported as not ready."""
        df = pd.DataFrame(
            {
                "age": [30],
                "annual_income": [50000],
                "claim_risk_score": [50.0],
                "customer_id": ["cust_0"],
            }
        )
        df.to_parquet(sample_data_dir + "/federated/germany_allianz/train.parquet", index=False)

        report = check_data_readiness(sample_data_dir, "germany_allianz", "train")
        assert report.is_ready is False
        assert any("BatchNorm" in f for f in report.failures)

    def test_excluded_columns_not_checked(
        self, clean_df: pd.DataFrame, sample_data_dir: str
    ) -> None:
        """Target columns and customer_id are excluded from feature checks."""
        # Make target columns have zero variance (should not be checked)
        clean_df["claim_risk_score"] = 0.0
        clean_df.to_parquet(
            sample_data_dir + "/federated/germany_allianz/train.parquet", index=False
        )

        report = check_data_readiness(sample_data_dir, "germany_allianz", "train")
        # Should be ready because excluded columns are ignored
        assert report.is_ready is True
