"""Transportability layer for the Insurance Federated Learning Platform.

Implements covariate-shift measurement (SMD), outcome-model transport,
and causal-effect transport. Governed by stated source/target populations
and numeric thresholds.

Public API:
  @dataclass TransportReport
  def measure_covariate_shift(source_df, target_df) -> dict
  def transport_outcome_model(source_df, target_df, task) -> TransportReport
  def transport_causal_effect(estimand, source_df, target_df, dag) -> dict
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import log_loss, mean_squared_error

from insurance_fl.causal.preflight import require_causal_deps

require_causal_deps()


@dataclass
class TransportReport:
    """Result of a transportability assessment for one source→target pair."""

    source_clients: tuple[str, ...]
    target_client: str
    smd_worst: dict[str, float]
    source_loss: float
    target_loss: float
    loss_ratio: float
    transportable_metric: bool


def _client_df(data_dir: Path, client_id: str, split: str = "train") -> pd.DataFrame:
    path = Path(data_dir) / "federated" / client_id / f"{split}.parquet"
    return pd.read_parquet(path)


def measure_covariate_shift(source_df: pd.DataFrame, target_df: pd.DataFrame) -> dict[str, float]:
    numeric_cols = [
        c
        for c in source_df.select_dtypes(include=[np.number]).columns
        if c
        not in {
            "claim_risk_score",
            "customer_lifetime_value",
            "will_renew",
            "is_high_risk",
            "customer_segment",
            "customer_id",
        }
    ]
    smds: dict[str, float] = {}
    for col in numeric_cols:
        s_vals = source_df[col].dropna().values.astype(float)
        t_vals = target_df[col].dropna().values.astype(float)
        if len(s_vals) < 2 or len(t_vals) < 2:
            continue
        s_mean = float(np.mean(s_vals))
        t_mean = float(np.mean(t_vals))
        s_var = float(np.var(s_vals, ddof=1))
        t_var = float(np.var(t_vals, ddof=1))
        n_s = len(s_vals)
        n_t = len(t_vals)
        pooled_var = ((n_s - 1) * s_var + (n_t - 1) * t_var) / (n_s + n_t - 2)
        pooled_std = np.sqrt(pooled_var) if pooled_var > 0 else 1e-12
        smds[col] = float(abs(s_mean - t_mean) / pooled_std)
    return smds


def _worst_5(smds: dict[str, float]) -> dict[str, float]:
    sorted_items = sorted(smds.items(), key=lambda x: x[1], reverse=True)
    return dict(sorted_items[:5])


def transport_outcome_model(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    task: str,
    smd_threshold: float = 0.25,
    loss_multiplier: float = 1.5,
) -> TransportReport:
    feature_cols = [
        c
        for c in source_df.select_dtypes(include=[np.number]).columns
        if c
        not in {
            "claim_risk_score",
            "customer_lifetime_value",
            "will_renew",
            "is_high_risk",
            "customer_segment",
            "customer_id",
        }
    ]
    smds = measure_covariate_shift(source_df, target_df)
    smd_worst = _worst_5(smds)

    Xs = source_df[feature_cols].fillna(source_df[feature_cols].median())
    Xt = target_df[feature_cols].fillna(target_df[feature_cols].median())
    ys = source_df[task]
    yt = target_df[task]

    binary = set(ys.dropna().unique()).issubset({0, 1}) and len(set(ys.dropna().unique())) == 2

    if binary:
        model = LogisticRegression(max_iter=1000, solver="lbfgs")
        model.fit(Xs, ys)
        source_loss = float(log_loss(ys, model.predict_proba(Xs)))
        target_loss = float(log_loss(yt, model.predict_proba(Xt)))
    else:
        model = LinearRegression()
        model.fit(Xs, ys)
        source_loss = float(mean_squared_error(ys, model.predict(Xs)))
        target_loss = float(mean_squared_error(yt, model.predict(Xt)))

    loss_ratio = float(target_loss / source_loss) if source_loss > 0 else float("inf")
    transportable_metric = bool(loss_ratio <= loss_multiplier)

    source_clients = tuple(
        sorted(set(source_df.get("country", pd.Series(["unknown"])).unique().tolist()))
    )
    target_client = str(
        target_df.get("country", pd.Series(["unknown"])).iloc[0]
        if len(target_df) > 0
        else "unknown"
    )

    return TransportReport(
        source_clients=source_clients,
        target_client=target_client,
        smd_worst=smd_worst,
        source_loss=source_loss,
        target_loss=target_loss,
        loss_ratio=loss_ratio,
        transportable_metric=transportable_metric,
    )


def transport_causal_effect(
    estimand: Any,
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    dag: Any,
) -> dict[str, Any]:
    source_countries = set(source_df.get("country", pd.Series([])).unique())
    target_country = str(
        target_df.get("country", pd.Series(["unknown"])).iloc[0]
        if len(target_df) > 0
        else "unknown"
    )

    if target_country in source_countries:
        return {
            "transportable": True,
            "reason": (
                f"Target country {target_country} is in source population "
                f"{sorted(source_countries)}; no country-parameter shift."
            ),
        }

    return {
        "transportable": False,
        "reason": (
            f"Non-transportable effect: target country '{target_country}' not in "
            f"source population {sorted(source_countries)}. The causal effect of "
            f"{estimand.treatment} on {estimand.outcome} is identified via backdoor "
            f"adjustment on {list(getattr(estimand, 'adjustment_set', []))}, but "
            f"country parameters shift both treatment and outcome. Transport would "
            f"require re-estimating the estimand on the target client."
        ),
    }
