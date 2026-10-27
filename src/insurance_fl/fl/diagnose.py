"""Diagnostic module for task learnability (§is_high_risk verdict).

Computes a verdict from four numbers so the report cannot narrate away
a sub-chance pooled AUC.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None  # type: ignore[assignment,misc]


def diagnose_task(
    X: np.ndarray,
    y: np.ndarray,
    client_ids: np.ndarray,
    *,
    task_name: str,
    seed: int = 0,
) -> dict[str, Any]:
    """Return a verdict dict that a report must quote verbatim.

    Proves 'learnable' vs 'generator/misalignment bug' with four numbers.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    client_ids = np.asarray(client_ids)

    if XGBClassifier is None:
        raise RuntimeError("xgboost not available")

    # A: pooled holdout AUC
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
    m = XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.1,
        early_stopping_rounds=20,
        eval_metric="auc",
    )
    m.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)
    auc_A = float(roc_auc_score(yte, m.predict_proba(Xte)[:, 1]))

    # B: per-client within-client AUC
    auc_B: dict[str, float | None] = {}
    for cid in np.unique(client_ids):
        mask = client_ids == cid
        Xc, yc = X[mask], y[mask]
        if yc.min() == yc.max():
            auc_B[str(cid)] = None
            continue
        Xa, Xb, ya, yb = train_test_split(Xc, yc, test_size=0.5, stratify=yc, random_state=seed)
        m2 = XGBClassifier(n_estimators=100, max_depth=3, eval_metric="auc")
        m2.fit(Xa, ya)
        auc_B[str(cid)] = float(roc_auc_score(yb, m2.predict_proba(Xb)[:, 1]))

    # C: sort-by-client pooled AUC (detects row/join misalignment)
    order = np.argsort(client_ids)
    Xs, ys = X[order], y[order]
    Xtr2, Xte2, ytr2, yte2 = train_test_split(Xs, ys, test_size=0.3, stratify=ys, random_state=seed)
    m3 = XGBClassifier(n_estimators=200, max_depth=3, eval_metric="auc")
    m3.fit(Xtr2, ytr2)
    auc_C = float(roc_auc_score(yte2, m3.predict_proba(Xte2)[:, 1]))

    # D: in-sample fit (detects label-unrelated-to-features vs split bug)
    m4 = XGBClassifier(n_estimators=10, max_depth=1, eval_metric="logloss")
    m4.fit(X, y)
    acc_D = float((m4.predict(X) == y).mean())

    # Verdict: computed, not narrated.
    if acc_D < 0.6:
        verdict = "GENERATOR_BUG: label not a function of features (D<0.6)"
    elif auc_A < 0.5 and auc_C >= 0.55:
        verdict = "MISALIGNMENT_BUG: sort-by-client recovers signal (A<0.5, C>=0.55)"
    elif auc_A < 0.55 and acc_D >= 0.7:
        verdict = "GENERALIZATION_BUG: fits in-sample but not out (A<0.55, D>=0.7)"
    elif auc_A >= 0.55:
        verdict = "LEARNABLE: signal present (A>=0.55); underfit if boosters underperform"
    else:
        verdict = "AMBIGUOUS: see A/B/C/D"

    return {
        "task": task_name,
        "A_pooled_auc": auc_A,
        "B_per_client_auc": auc_B,
        "C_sorted_pooled_auc": auc_C,
        "D_in_sample_acc": acc_D,
        "verdict": verdict,
    }


__all__ = ["diagnose_task"]
