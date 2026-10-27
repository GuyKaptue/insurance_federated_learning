"""Refutation tests for causal estimands.

Public API:
  def refute_placebo(df, estimand, n_perm=50) -> dict
  def refute_random_cause(df, estimand, n_sims=50) -> dict
  def refute_subset(df, estimand, fractions=(0.5, 0.8)) -> dict

Each returns {"passed": bool, "details": {...}}.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from insurance_fl.causal.estimands import CausalEstimand, estimate_backdoor, estimate_doubly_robust
from insurance_fl.causal.preflight import require_causal_deps

require_causal_deps()


def _ate_pair(df: pd.DataFrame, estimand: CausalEstimand) -> tuple[float, float]:
    bd = estimate_backdoor(df, estimand)
    dr = estimate_doubly_robust(df, estimand)
    return bd["ate"], dr["ate"]


def refute_placebo(df: pd.DataFrame, estimand: CausalEstimand, n_perm: int = 50) -> dict[str, Any]:
    """Shuffle treatment; ATE must collapse to ~0 (|ATE| < 0.1 * original)."""

    orig_ate, _ = _ate_pair(df, estimand)
    threshold = 0.1 * abs(orig_ate) if orig_ate != 0 else 0.1

    rng = np.random.RandomState(42)
    placebo_ates = []
    for _ in range(n_perm):
        df_perm = df.copy()
        df_perm[estimand.treatment] = rng.permutation(df_perm[estimand.treatment].values)
        try:
            ate, _ = _ate_pair(df_perm, estimand)
        except Exception:
            ate = 0.0
        placebo_ates.append(ate)

    max_placebo = float(np.max(np.abs(placebo_ates))) if placebo_ates else 0.0
    passed = max_placebo < threshold

    return {
        "passed": bool(passed),
        "details": {
            "test": "placebo",
            "original_ate": float(orig_ate),
            "threshold": float(threshold),
            "max_placebo_ate": max_placebo,
            "n_perm": int(n_perm),
            "placebo_ates": [float(a) for a in placebo_ates],
        },
    }


def refute_random_cause(
    df: pd.DataFrame, estimand: CausalEstimand, n_sims: int = 50
) -> dict[str, Any]:
    """Add a random confounder; ATE must stay within ±20% of original."""

    orig_ate, _ = _ate_pair(df, estimand)
    rng = np.random.RandomState(7)

    max_deviation = 0.0
    for _ in range(n_sims):
        df_rc = df.copy()
        df_rc["_random_cause"] = rng.normal(size=len(df_rc))
        # Add the random cause as an extra adjustment variable
        estimand_rc = CausalEstimand(
            name=estimand.name + "_random_cause",
            treatment=estimand.treatment,
            outcome=estimand.outcome,
            adjustment_set=(*list(estimand.adjustment_set), "_random_cause"),
            dag_path=estimand.dag_path,
        )
        try:
            ate, _ = _ate_pair(df_rc, estimand_rc)
        except Exception:
            ate = 0.0
        deviation = abs(ate - orig_ate) / (abs(orig_ate) + 1e-12)
        max_deviation = max(max_deviation, deviation)

    passed = max_deviation <= 0.20

    return {
        "passed": bool(passed),
        "details": {
            "test": "random_cause",
            "original_ate": float(orig_ate),
            "max_relative_deviation": float(max_deviation),
            "n_sims": int(n_sims),
            "tolerance": 0.20,
        },
    }


def refute_subset(
    df: pd.DataFrame, estimand: CausalEstimand, fractions: tuple[float, ...] = (0.5, 0.8)
) -> dict[str, Any]:
    """Estimate on subsets; ATE must stay within ±20% of full-sample ATE."""
    orig_ate, _ = _ate_pair(df, estimand)
    rng = np.random.RandomState(11)

    max_deviation = 0.0
    subset_results = []
    for frac in fractions:
        n_sub = max(2, int(len(df) * frac))
        idx = rng.choice(len(df), size=n_sub, replace=False)
        df_sub = df.iloc[idx].reset_index(drop=True)
        try:
            ate, _ = _ate_pair(df_sub, estimand)
        except Exception:
            ate = 0.0
        deviation = abs(ate - orig_ate) / (abs(orig_ate) + 1e-12)
        max_deviation = max(max_deviation, deviation)
        subset_results.append(
            {
                "fraction": float(frac),
                "n": int(n_sub),
                "ate": float(ate),
                "relative_deviation": float(deviation),
            }
        )

    passed = max_deviation <= 0.20

    return {
        "passed": bool(passed),
        "details": {
            "test": "subset",
            "original_ate": float(orig_ate),
            "max_relative_deviation": float(max_deviation),
            "fractions": [float(f) for f in fractions],
            "tolerance": 0.20,
            "subsets": subset_results,
        },
    }


def run_all_refutations(df: pd.DataFrame, estimand: CausalEstimand) -> dict[str, dict[str, Any]]:
    """Run all three refutation tests and return results keyed by test name."""
    return {
        "placebo": refute_placebo(df, estimand),
        "random_cause": refute_random_cause(df, estimand),
        "subset": refute_subset(df, estimand),
    }
