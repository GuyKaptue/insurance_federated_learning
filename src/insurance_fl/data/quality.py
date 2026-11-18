"""
Data quality analysis for the Insurance Federated Learning Platform.
Computes missing values, outliers, duplicates, and overall quality scores.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class QualityMetrics:
    """Container for data quality metrics."""

    total_rows: int
    total_columns: int
    missing_values: Dict[str, float]
    duplicate_rows: int
    duplicate_percentage: float
    outlier_counts: Dict[str, int]
    outlier_percentages: Dict[str, float]
    unique_values: Dict[str, int]
    column_types: Dict[str, str]
    correlation_matrix: Optional[pd.DataFrame] = None
    summary_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class DataQualityAnalyzer:
    """Comprehensive data quality analyzer."""

    def __init__(self) -> None:
        self.quality_metrics: Optional[QualityMetrics] = None

    def analyze(self, df: pd.DataFrame) -> QualityMetrics:
        """Perform comprehensive data quality analysis."""
        total_rows = len(df)
        total_columns = len(df.columns)

        # Missing values
        missing_values = {}
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_values[col] = float(missing_count) / total_rows if total_rows > 0 else 0.0

        # Duplicates (robust to unhashable/object values such as numpy arrays)
        try:
            dup_mask = df.duplicated()
        except TypeError:
            # Fallback: hash stable string representations per cell/row
            safe = df.map(
                lambda x: x.tobytes()
                if isinstance(x, np.ndarray)
                else (tuple(x) if isinstance(x, list) else x)
            )
            try:
                dup_mask = safe.duplicated()
            except TypeError:
                dup_mask = safe.astype(str).duplicated()
        duplicate_rows = int(dup_mask.sum())
        duplicate_percentage = float(duplicate_rows) / total_rows if total_rows > 0 else 0.0

        # Outliers (IQR method for numerical columns)
        outlier_counts = {}
        outlier_percentages = {}
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                q1 = col_data.quantile(0.25)
                q3 = col_data.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                outliers = ((col_data < lower) | (col_data > upper)).sum()
                outlier_counts[col] = int(outliers)
                outlier_percentages[col] = (
                    float(outliers) / len(col_data) if len(col_data) > 0 else 0.0
                )

        # Unique values (robust to unhashable values)
        unique_values = {}
        for col in df.columns:
            try:
                unique_values[col] = int(df[col].nunique(dropna=True))
            except TypeError:
                s = df[col].dropna()
                try:
                    unique_values[col] = int(
                        s.map(
                            lambda x: x.tobytes()
                            if isinstance(x, np.ndarray)
                            else (tuple(x) if isinstance(x, list) else x)
                        ).nunique()
                    )
                except Exception:
                    unique_values[col] = int(s.astype(str).nunique())

        # Column types
        column_types = {col: str(dtype) for col, dtype in df.dtypes.items()}

        # Correlation matrix for numerical columns
        correlation_matrix = None
        if len(numerical_cols) > 1:
            correlation_matrix = df[numerical_cols].corr()

        # Summary statistics
        summary_stats = {}
        for col in numerical_cols:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                summary_stats[col] = {
                    "mean": float(col_data.mean()),
                    "std": float(col_data.std()),
                    "min": float(col_data.min()),
                    "max": float(col_data.max()),
                    "median": float(col_data.median()),
                }

        self.quality_metrics = QualityMetrics(
            total_rows=total_rows,
            total_columns=total_columns,
            missing_values=missing_values,
            duplicate_rows=duplicate_rows,
            duplicate_percentage=duplicate_percentage,
            outlier_counts=outlier_counts,
            outlier_percentages=outlier_percentages,
            unique_values=unique_values,
            column_types=column_types,
            correlation_matrix=correlation_matrix,
            summary_stats=summary_stats,
        )

        return self.quality_metrics

    def overall_quality_score(self) -> float:
        """Compute overall quality score (0-100)."""
        if self.quality_metrics is None:
            return 0.0

        m = self.quality_metrics
        score = 100.0

        # Deduct for missing values
        if m.missing_values:
            avg_missing = sum(m.missing_values.values()) / len(m.missing_values)
            score -= avg_missing * 30

        # Deduct for duplicates
        score -= m.duplicate_percentage * 10

        # Deduct for outliers
        if m.outlier_counts:
            avg_outlier_pct = sum(m.outlier_percentages.values()) / len(m.outlier_percentages)
            score -= avg_outlier_pct * 20

        return max(0.0, min(100.0, score))
