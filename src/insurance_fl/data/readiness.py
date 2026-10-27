"""Data readiness gate for the Insurance Federated Learning Platform.

Validates that a client's pre-loaded partition (features + targets) satisfies
the contracts required by the training and evaluation pipeline before it is
consumed. This is the *post-preprocess* contract: if preprocess.py has not
run (or quality issues were injected), the gate fails closed and the caller
aborts, so that no downstream stage trains on dirty data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


class DataNotReadyError(ValueError):
    """Raised when a data partition fails the readiness gate."""


@dataclass
class ReadinessReport:
    """Outcome of a single readiness check."""

    client_id: str
    split: str
    n_samples: int
    n_features: int
    missing_values: int
    non_finite_count: int
    duplicates: int
    feature_variance_min: float
    feature_variance_max: float
    is_ready: bool
    failures: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.is_ready = not bool(self.failures)


def _check_missing(df: pd.DataFrame) -> list[str]:
    """Check for missing values in numeric feature columns."""
    issues: list[str] = []
    num_cols = df.select_dtypes(include=[np.number]).columns
    missing = int(df[num_cols].isna().sum().sum())
    if missing > 0:
        issues.append(f"{missing} missing value(s) in {len(num_cols)} numeric feature column(s)")
    return issues


def _check_non_finite(df: pd.DataFrame) -> list[str]:
    """Check for non-finite values (NaN, +inf, -inf) in numeric feature columns."""
    issues: list[str] = []
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) == 0:
        return issues
    arr = df[num_cols].to_numpy(dtype=np.float32, na_value=np.nan)
    non_finite = int((~np.isfinite(arr)).sum())
    if non_finite > 0:
        issues.append(f"{non_finite} non-finite value(s) in numeric feature column(s)")
    return issues


def _check_sufficient_samples(df: pd.DataFrame) -> list[str]:
    """Check at least one sample present (and >= 2 for BatchNorm training)."""
    issues: list[str] = []
    if len(df) < 2:
        issues.append(f"only {len(df)} sample(s); need >= 2 for BatchNorm training")
    return issues


def _check_duplicate_ids(df: pd.DataFrame, id_col: str = "customer_id") -> list[str]:
    """Check that the partition does not contain duplicate client identifiers."""
    issues: list[str] = []
    if id_col not in df.columns:
        return issues
    n_dup = int(df[id_col].duplicated().sum())
    if n_dup > 0:
        issues.append(f"{n_dup} duplicate {id_col} value(s)")
    return issues


def _check_zero_variance(df: pd.DataFrame) -> list[str]:
    """Check that no numeric feature column is constant (zero variance)."""
    issues: list[str] = []
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) == 0:
        return issues
    arr = df[num_cols].to_numpy(dtype=np.float32, na_value=np.nan)
    for i, col in enumerate(num_cols):
        col_vals = arr[:, i]
        valid = col_vals[~np.isnan(col_vals)]
        if len(valid) < 2:
            issues.append(f"column '{col}' has < 2 non-null entries")
            continue
        var = float(np.var(valid))
        if var <= 1e-6:
            issues.append(f"column '{col}' has zero variance")
    return issues


def check_data_readiness(data_dir: Path, client_id: str, split: str = "train") -> ReadinessReport:
    """Validate a client partition against the post-preprocess contract.

    Parameters
    ----------
    data_dir:
        Directory that contains federated/<client_id>/<split>.parquet.
    client_id:
        Identifier of the client whose partition to validate.
    split:
        Dataset split name.

    Returns
    -------
    ReadinessReport
        Report describing the outcome. is_ready is False and failures is
        non-empty if any check failed.

    Raises
    ------
    FileNotFoundError
        If the partition file does not exist.
    """
    data_dir = Path(data_dir)
    path = data_dir / "federated" / client_id / f"{split}.parquet"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing partition {path}. Run first: " "python -m insurance_fl.fl.task --prepare"
        )

    df = pd.read_parquet(path)

    all_num_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [c for c in all_num_cols if c not in _EXCLUDED_FEATURES]
    features = df[feature_cols] if feature_cols else df.iloc[:, :0]

    failures = []
    failures += _check_missing(features)
    failures += _check_non_finite(features)
    failures += _check_sufficient_samples(df)
    failures += _check_duplicate_ids(df)
    failures += _check_zero_variance(features)

    n_samples = int(len(df))
    n_features = int(len(all_num_cols))
    missing = int(features.isna().sum().sum())
    arr = features.to_numpy(dtype=np.float32, na_value=np.nan)
    non_finite = int((~np.isfinite(arr)).sum())

    variance_min = float("nan")
    variance_max = float("nan")
    if n_features > 0:
        valid = ~np.isnan(arr).any(axis=1)
        if valid.any():
            col_var = np.var(arr[valid], axis=0, ddof=1)
            ctrl = col_var[col_var > 0.0]
            variance_min = float(ctrl.min()) if len(ctrl) else 0.0
            variance_max = float(ctrl.max()) if len(ctrl) else 0.0
        else:
            variance_min = 0.0
            variance_max = 0.0

    report = ReadinessReport(
        client_id=client_id,
        split=split,
        n_samples=n_samples,
        n_features=n_features,
        missing_values=missing,
        non_finite_count=non_finite,
        duplicates=0,
        feature_variance_min=variance_min,
        feature_variance_max=variance_max,
        is_ready=False if failures else True,
        failures=failures,
    )

    # Duplicate-row check.
    if n_samples > 1:
        report.duplicates = int(df["customer_id"].duplicated().sum())
        if report.duplicates > 0:
            report.failures.append(f"{report.duplicates} duplicate customer_id value(s)")

    return report


_EXCLUDED_FEATURES = set(
    [
        "claim_risk_score",
        "customer_lifetime_value",
        "will_renew",
        "is_high_risk",
        "customer_segment",
        "customer_id",
    ]
)


def main() -> int:
    """CLI entry point: validate readiness for a single client partition."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate client data readiness.")
    parser.add_argument("data_dir", type=Path, help="Root of the processed dataset")
    parser.add_argument("client_id", help="Client identifier")
    parser.add_argument("--split", default="train", help="Dataset split (default: train)")
    args = parser.parse_args()

    report = check_data_readiness(args.data_dir, args.client_id, args.split)
    print(f"client={args.client_id} split={args.split}")
    print(f"  samples={report.n_samples} features={report.n_features}")
    print(f"  missing_values={report.missing_values} non_finite={report.non_finite_count}")
    print(f"  duplicates={report.duplicates}")
    print(f"  variance=[{report.feature_variance_min:.4f}, {report.feature_variance_max:.4f}]")
    print(f"  is_ready={report.is_ready}")
    if report.failures:
        print("  FAILURES:")
        for f in report.failures:
            print(f"    - {f}")
        return 1
    print("  OK: data passes readiness gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
