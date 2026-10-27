"""Stacking layer — NNLS meta-learner over booster predictions."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import nnls


def validate_alpha(
    alpha: Any,
    *,
    atol: float = 1e-10,
) -> np.ndarray:
    """Validate NNLS stacking coefficients.

    Raises ValueError if:
    - Not a non-empty 1-D vector
    - Contains NaN or infinity
    - Any coefficient < -atol (negative)
    - All coefficients are <= atol (degenerate)
    """
    arr = np.asarray(alpha, dtype=float)

    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("alpha must be a non-empty 1-D vector")

    if not np.isfinite(arr).all():
        raise ValueError("alpha contains NaN or infinite values")

    if np.any(arr < -atol):
        raise ValueError("NNLS coefficients must be nonnegative")

    # Normalize tiny negative floating-point noise
    arr = np.maximum(arr, 0.0)

    if not np.any(arr > atol):
        raise ValueError("NNLS produced an all-zero or degenerate stacking solution")

    return np.asarray(arr, dtype=float)  # type: ignore[no-any-return]


def fit_stacking_meta(
    oof_predictions: Sequence[np.ndarray],
    y_true: np.ndarray,
    *,
    fallback: bool = True,
) -> np.ndarray:
    A = np.column_stack([np.asarray(p, dtype=float).ravel() for p in oof_predictions])
    b = np.asarray(y_true, dtype=float).ravel()
    weights, _ = nnls(A, b)
    # Validate (allow degenerate to be surfaced)
    if not np.any(weights > 0) and not fallback:
        raise ValueError("NNLS produced an all-zero or degenerate stacking solution")
    if fallback and not np.any(weights > 0):
        best = int(np.argmin([np.mean((np.asarray(p).ravel() - b) ** 2) for p in oof_predictions]))
        weights = np.zeros(len(oof_predictions))
        weights[best] = 1.0
    return np.asarray(weights, dtype=float)  # type: ignore[no-any-return]


def predict_stacked(
    predictions: Sequence[np.ndarray],
    weights: np.ndarray,
) -> np.ndarray:
    out = np.zeros_like(np.asarray(predictions[0], dtype=float))
    for w, p in zip(weights, predictions):
        out = out + float(w) * np.asarray(p, dtype=float)
    return np.asarray(out, dtype=float)  # type: ignore[no-any-return]


def generate_oof_predictions(
    model_fn: Callable[[], Any],
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_splits: int = 5,
    seed: int = 42,
) -> np.ndarray:
    from sklearn.model_selection import KFold

    oof = np.zeros(len(y), dtype=float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, va in kf.split(X):
        m = model_fn()
        m.fit(X[tr], y[tr])
        pred = m.predict(X[va])
        oof[va] = np.asarray(pred).ravel()
    return np.asarray(oof, dtype=float)  # type: ignore[no-any-return]


def save_stacking_meta(
    path: Path | str,
    weights: np.ndarray,
    oof_loss: float,
    fallback: bool = False,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    w = np.asarray(weights).ravel()
    payload = {
        "alpha_xgb": float(w[0]) if len(w) > 0 else 0.0,
        "alpha_lgbm": float(w[1]) if len(w) > 1 else 0.0,
        "alpha_cat": float(w[2]) if len(w) > 2 else 0.0,
        "oof_loss": float(oof_loss),
        "fallback": bool(fallback),
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


__all__ = [
    "fit_stacking_meta",
    "generate_oof_predictions",
    "load_stacking_meta",
    "predict_stacked",
    "save_stacking_meta",
    "validate_alpha",
]


def load_stacking_meta(path: Path | str) -> dict[str, Any]:
    """Load stacking meta from JSON file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Stacking meta not found: {path}")
    data = json.loads(path.read_text())
    result = dict(data)
    if "alpha_xgb" in result or "alpha_lgbm" in result or "alpha_cat" in result:
        alpha = [
            result.get("alpha_xgb", 0.0),
            result.get("alpha_lgbm", 0.0),
            result.get("alpha_cat", 0.0),
        ]
        validate_alpha(alpha)
    return result
