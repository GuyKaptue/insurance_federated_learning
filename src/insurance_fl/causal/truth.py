"""Semi-synthetic causal ground truth for the insurance data.

The generator is not modified. This module reads a generated client
DataFrame and adds derived truth columns used for causal evaluation only.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from insurance_fl.data.generator import TARGET_COLUMNS

TRUTH_COLUMNS = ("claim_risk_score_true", "effect_has_claims")
TREATMENT = "has_claims"
OUTCOME_TRUE = "claim_risk_score_true"
EFFECT_COLUMN = "effect_has_claims"

COUNTRY_EFFECT_MODIFIER = {
    "Germany": 1.0,
    "France": 1.1,
    "Belgium": 0.9,
    "Switzerland": 1.2,
}


def _has_claims(df: pd.DataFrame) -> pd.Series:
    if TREATMENT in df.columns:
        return df[TREATMENT].astype(int)
    if "number_of_claims" in df.columns:
        return (df["number_of_claims"] > 0).astype(int)
    raise ValueError("treatment column or number_of_claims is required")


def inject_truth(
    df: pd.DataFrame,
    *,
    seed: int = 42,
    base_effect: float = 5.0,
    country_modifiers: dict[str, float] | None = None,
    treatment: str = TREATMENT,
    outcome_true: str = OUTCOME_TRUE,
    effect_column: str = EFFECT_COLUMN,
) -> pd.DataFrame:
    """Return a copy of df with semi-synthetic truth columns.

    The five ML targets are unchanged. The injected effect is deterministic
    for a fixed seed and country modifiers.
    """
    if country_modifiers is None:
        country_modifiers = dict(COUNTRY_EFFECT_MODIFIER)
    out = df.copy()
    has_claims = _has_claims(out)
    if treatment not in out.columns:
        out[treatment] = has_claims
    if "country" in out.columns:
        modifier = out["country"].map(country_modifiers).fillna(1.0)
    else:
        modifier = pd.Series(1.0, index=out.index)
    effect = (base_effect * modifier * has_claims).astype(float)
    out[effect_column] = effect
    out[outcome_true] = out["claim_risk_score"] + effect
    out.attrs["truth_seed"] = seed
    out.attrs["truth_base_effect"] = base_effect
    return out


def ground_truth_ate(
    df: pd.DataFrame,
    *,
    treatment: str = TREATMENT,
    effect_column: str = EFFECT_COLUMN,
) -> float:
    """Return the injected average treatment effect."""
    if effect_column in df.columns:
        return float(df[effect_column].mean())
    if treatment in df.columns and OUTCOME_TRUE in df.columns:
        treated = df.loc[df[treatment] == 1, OUTCOME_TRUE]
        untreated = df.loc[df[treatment] == 0, OUTCOME_TRUE]
        return float(treated.mean() - untreated.mean())
    raise ValueError("truth columns are missing")


def ml_target_columns() -> list[str]:
    """Return the five ML target columns."""
    return list(TARGET_COLUMNS)


def ml_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric ML feature columns, excluding targets and truth."""
    excluded = set(TARGET_COLUMNS) | set(TRUTH_COLUMNS) | {"customer_id"}
    return [c for c in df.select_dtypes(include=["number"]).columns if c not in excluded]


def write_ground_truth(
    output_dir: str | Path,
    df: pd.DataFrame,
    *,
    seed: int = 42,
    base_effect: float = 5.0,
) -> Path:
    """Write the ground-truth ATE beside generated outputs."""
    path = Path(output_dir) / "ground_truth_ate.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "treatment": TREATMENT,
        "outcome": OUTCOME_TRUE,
        "base_effect": base_effect,
        "seed": seed,
        "ate": ground_truth_ate(df),
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


__all__ = [
    "EFFECT_COLUMN",
    "OUTCOME_TRUE",
    "TREATMENT",
    "TRUTH_COLUMNS",
    "ground_truth_ate",
    "inject_truth",
    "ml_feature_columns",
    "ml_target_columns",
    "write_ground_truth",
]
