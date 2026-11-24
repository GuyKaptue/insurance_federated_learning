"""Causal estimands: backdoor adjustment and doubly-robust (AIPW) estimation.

Public API:
  @dataclass CausalEstimand
  def estimate_backdoor(df, estimand) -> dict
  def estimate_doubly_robust(df, estimand) -> dict
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.api import OLS, Logit, add_constant

from insurance_fl.causal.preflight import require_causal_deps

require_causal_deps()


@dataclass
class CausalEstimand:
    """A stated causal estimand with adjustment set and DAG citation."""

    name: str
    treatment: str
    outcome: str
    adjustment_set: tuple[str, ...]
    dag_path: Path

    @classmethod
    def from_dag(
        cls,
        name: str,
        treatment: str,
        outcome: str,
        adjustment_set: tuple[str, ...],
        dag_path: Path | str = Path("docs/causal/dag.yaml"),
    ) -> CausalEstimand:
        return cls(
            name=name,
            treatment=treatment,
            outcome=outcome,
            adjustment_set=adjustment_set,
            dag_path=Path(dag_path),
        )


def _is_binary(series: pd.Series) -> bool:
    vals = set(series.dropna().unique())
    return vals.issubset({0, 1}) and len(vals) == 2


def _fit_outcome_model(y: pd.Series, X: pd.DataFrame, binary: bool) -> Any:
    Xc = add_constant(X, has_constant="add")
    if binary:
        return Logit(y.values, Xc.values).fit(disp=0)
    return OLS(y.values, Xc.values).fit()


def _predict_outcome(model: Any, X: pd.DataFrame, binary: bool) -> np.ndarray:
    Xc = add_constant(X, has_constant="add")
    return np.asarray(model.predict(Xc.values), dtype=float)  # type: ignore[no-any-return]


def estimate_backdoor(df: pd.DataFrame, estimand: CausalEstimand) -> dict[str, Any]:
    """Backdoor adjustment estimator using GLM with the stated adjustment set.

    For binary outcome: logistic regression of Y on T + Z; ATE = mean over
    samples of (predict(T=1, Z) - predict(T=0, Z)).
    For continuous outcome: OLS of Y on T + Z; ATE = coefficient on T;
    CI from model.conf_int() on the treatment coefficient.

    Returns {"ate", "ci_low", "ci_high", "n", "method", "adjustment_set"}.
    """
    T = estimand.treatment
    Y = estimand.outcome
    Z = list(estimand.adjustment_set)
    binary = _is_binary(df[Y])

    data = df[[T, Y, *Z]].dropna()
    n = len(data)

    X = data[[*Z, T]].copy()
    y = data[Y]

    model = _fit_outcome_model(y, X, binary)

    data_t1 = data.copy()
    data_t1[T] = 1.0
    data_t0 = data.copy()
    data_t0[T] = 0.0

    X1 = data_t1[[*Z, T]]
    X0 = data_t0[[*Z, T]]

    p1 = _predict_outcome(model, X1, binary)
    p0 = _predict_outcome(model, X0, binary)

    ite = p1 - p0
    ate = float(np.mean(ite))

    if binary:
        # For logistic regression, ATE = mean of individual treatment effects.
        # SE via standard error of the mean of ITE.
        se = float(np.std(ite, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    else:
        # For OLS, ATE = coefficient on T. Use model standard error.
        t_idx = list(X.columns).index(T)
        # add_constant prepends a column, so shift index by 1
        se = float(model.bse[t_idx + 1])

    ci_low = float(ate - 1.96 * se)
    ci_high = float(ate + 1.96 * se)

    return {
        "ate": ate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n": n,
        "method": "backdoor_glm",
        "adjustment_set": list(estimand.adjustment_set),
    }


def estimate_doubly_robust(df: pd.DataFrame, estimand: CausalEstimand) -> dict[str, Any]:
    """Doubly-robust (AIPW) estimator with cross-fitted nuisance models.

    1. Propensity model: P(T=1 | Z) via logistic regression.
    2. Outcome model: E[Y | T, Z] via GLM.
    3. 5-fold cross-fitting to avoid overfitting nuisance models.
    4. AIPW ATE = mean of influence-function-style residuals.

    Returns same shape as estimate_backdoor with method="aipw".
    """
    from sklearn.model_selection import KFold

    T = estimand.treatment
    Y = estimand.outcome
    Z = list(estimand.adjustment_set)
    binary = _is_binary(df[Y])

    data = df[[T, Y, *Z]].dropna().reset_index(drop=True)
    n = len(data)

    t = data[T].values.astype(float)
    y = data[Y].values.astype(float)
    Zmat = data[Z].values

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    e_hat = np.zeros(n)
    mu_hat_0 = np.zeros(n)
    mu_hat_1 = np.zeros(n)

    for train_idx, test_idx in kf.split(data):
        Ztr = Zmat[train_idx]
        ttr = t[train_idx]
        ytr = y[train_idx]

        Xtr_p = add_constant(pd.DataFrame(Ztr), has_constant="add")
        prop_model = Logit(ttr, Xtr_p.values).fit(disp=0)
        Xte_p = add_constant(pd.DataFrame(Zmat[test_idx]), has_constant="add")
        e_hat[test_idx] = prop_model.predict(Xte_p.values)
        e_hat[test_idx] = np.clip(e_hat[test_idx], 0.01, 0.99)

        Xtr_o = add_constant(pd.DataFrame(np.column_stack([Ztr, ttr])), has_constant="add")
        out_model = _fit_outcome_model(pd.Series(ytr), pd.DataFrame(Xtr_o), binary)

        Zte = Zmat[test_idx]
        X0 = add_constant(
            pd.DataFrame(np.column_stack([Zte, np.zeros(len(Zte))])),
            has_constant="add",
        )
        X1 = add_constant(
            pd.DataFrame(np.column_stack([Zte, np.ones(len(Zte))])),
            has_constant="add",
        )
        mu_hat_0[test_idx] = _predict_outcome(out_model, pd.DataFrame(X0), binary)
        mu_hat_1[test_idx] = _predict_outcome(out_model, pd.DataFrame(X1), binary)

    e = np.clip(e_hat, 0.01, 0.99)
    mu_obs = np.where(t == 1, mu_hat_1, mu_hat_0)
    aipw_terms = (t - e) / (e * (1 - e)) * (y - mu_obs) + mu_hat_1 - mu_hat_0
    ate = float(np.mean(aipw_terms))
    se = float(np.std(aipw_terms, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    ci_low = float(ate - 1.96 * se)
    ci_high = float(ate + 1.96 * se)

    return {
        "ate": ate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n": n,
        "method": "aipw",
        "adjustment_set": list(estimand.adjustment_set),
    }
