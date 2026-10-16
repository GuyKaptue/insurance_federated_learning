"""Tests for insurance_fl.data.quality (metrics + score bounds)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from insurance_fl.data.quality import DataQualityAnalyzer, QualityMetrics


@pytest.fixture
def df_with_issues() -> pd.DataFrame:
    # Rows 1 and 4 are exact duplicates (same values in all columns);
    # row 3 contains NaN in "a"; row 5's c=1000 is a clear outlier.
    return pd.DataFrame(
        {
            "a": [1.0, 2.0, np.nan, 2.0, 5.0],
            "b": [10, 20, 30, 20, 40],
            "c": [0.1, 0.2, 0.3, 0.2, 1000.0],
            "d": ["x", "y", "x", "y", "x"],
        }
    )


def test_analyze_returns_metrics(df_with_issues: pd.DataFrame) -> None:
    m = DataQualityAnalyzer().analyze(df_with_issues)
    assert isinstance(m, QualityMetrics)
    assert m.total_rows == 5
    assert m.total_columns == 4
    assert m.missing_values["a"] == pytest.approx(0.2)
    assert m.missing_values["b"] == 0.0
    assert m.duplicate_rows == 1  # rows (40, 1000-outlier not dup) -> one exact dup
    # pandas 3.0 reports string columns as "str"; older pandas as "object"
    assert m.column_types["d"] in ("object", "string", "str")
    assert m.correlation_matrix is not None
    assert "mean" in m.summary_stats["c"]


def test_quality_score_bounds(df_with_issues: pd.DataFrame) -> None:
    az = DataQualityAnalyzer()
    assert az.overall_quality_score() == 0.0  # not yet analyzed -> 0.0 documented
    az.analyze(df_with_issues)
    score = az.overall_quality_score()
    assert 0.0 <= score <= 100.0
    assert score < 100.0  # missing + duplicates must deduct


def test_clean_data_scores_high() -> None:
    clean = pd.DataFrame({"x": np.arange(50.0), "y": np.arange(50.0)})
    az = DataQualityAnalyzer()
    az.analyze(clean)
    assert az.overall_quality_score() == pytest.approx(100.0, abs=1e-6)


def test_empty_dataframe_no_division_error() -> None:
    empty = pd.DataFrame({"a": []})
    m = DataQualityAnalyzer().analyze(empty)
    assert m.total_rows == 0
    assert m.missing_values == {"a": 0.0}
    assert m.duplicate_percentage == 0.0
