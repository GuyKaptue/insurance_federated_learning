"""Level-2 hybrid fusion and beta optimization (§1.5).

Combines NN predictions with booster-stacking predictions using a
learned beta coefficient per task, optimized on a global holdout set.

Change log vs previous version:
- UNINFORMATIVE rule: a task whose hybrid does not beat the constant
  baseline is flagged UNINFORMATIVE and emits NO beta claim. The beta
  value in diagnostics is still the raw argmin (for inspection), but
  `beta_claim` is None and `usable_beta` is False.
- diagnostics now carries `task_status` ("INFORMATIVE" | "UNINFORMATIVE"),
  `beta_claim` (float | None), `usable_beta` (bool), `beats_baseline` (bool).
- predict_hybrid refuses to fuse when the task is UNINFORMATIVE unless the
  caller explicitly passes `allow_uninformative=True` (for diagnostics only),
  so an uninformative beta can never leak into a hybrid prediction by accident.
- The argmin over {0, 1, interior} is unchanged.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar

# Tasks whose hybrid does not beat the constant baseline are reported with
# this status and get no beta claim.
STATUS_INFORMATIVE = "INFORMATIVE"
STATUS_UNINFORMATIVE = "UNINFORMATIVE"


def validate_beta(beta: Any) -> float:
    """Validate a beta fusion coefficient.

    Raises ValueError if:
    - Not a numeric scalar (bool rejected)
    - NaN or infinity
    - Outside [0, 1]
    """
    if isinstance(beta, bool):
        raise ValueError("beta must be a numeric scalar, not a boolean")

    try:
        value = float(beta)
    except (TypeError, ValueError) as exc:
        raise ValueError("beta must be a numeric scalar") from exc

    if not math.isfinite(value):
        raise ValueError("beta must be finite")

    if not 0.0 <= value <= 1.0:
        raise ValueError("beta must lie in the closed interval [0, 1]")

    return value


def _task_loss(preds: np.ndarray, y: np.ndarray, task_type: str) -> float:
    """Compute task-appropriate loss."""
    preds = np.asarray(preds, dtype=float)
    y = np.asarray(y, dtype=float)

    if task_type == "regression":
        return float(np.mean((preds - y) ** 2))

    eps = 1e-12
    if task_type == "binary":
        p = np.clip(preds, eps, 1.0 - eps)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    if task_type == "multi":
        p = np.clip(preds, eps, None)
        p = p / p.sum(axis=1, keepdims=True)
        return float(-np.mean(np.log(p[np.arange(len(y)), y.astype(int)] + eps)))

    raise ValueError(f"unknown task_type {task_type}")


def _baseline_loss(y_true: np.ndarray, task_type: str, n_classes: int | None = None) -> float:
    """Loss of the constant prior predictor for the given task type."""
    y_true = np.asarray(y_true)

    if task_type == "regression":
        return float(np.mean((y_true.astype(float) - float(np.mean(y_true.astype(float)))) ** 2))

    if task_type == "binary":
        prior = float(np.mean(y_true.astype(float)))
        eps = 1e-12
        p = np.clip(prior, eps, 1.0 - eps)
        return float(
            -np.mean(y_true.astype(float) * np.log(p) + (1 - y_true.astype(float)) * np.log(1 - p))
        )

    if task_type == "multi":
        if n_classes is None:
            n_classes = int(np.max(y_true.astype(int))) + 1
        counts = np.bincount(y_true.astype(int), minlength=n_classes)
        prior = counts / counts.sum()
        prior_arr = np.asarray(prior, dtype=float)
        return float(-np.mean(np.log(prior_arr[y_true.astype(int)] + 1e-12)))

    raise ValueError(f"unknown task_type {task_type}")


def optimize_beta(
    nn_preds: np.ndarray,
    booster_preds: np.ndarray,
    y_true: np.ndarray,
    *,
    task_type: str = "regression",
    task_name: str | None = None,
) -> tuple[float | None, dict[str, Any]]:
    """Optimize beta in [0, 1] for: hybrid = beta*nn + (1-beta)*booster.

    Returns (beta_claim, diagnostics).

    beta_claim is the beta to USE in the pipeline. It is None when the task
    is UNINFORMATIVE (hybrid does not beat the constant baseline). The raw
    argmin is always available in diagnostics["beta"] for inspection, but it
    is NOT a claim and must not be applied downstream.

    The raw argmin is taken over {0.0, 1.0, interior_optimum}, unchanged.
    """
    # ---------------------------------------------------------------- validate
    nn_preds = np.asarray(nn_preds, dtype=float)
    booster_preds = np.asarray(booster_preds, dtype=float)
    y_true = np.asarray(y_true)

    if task_type == "multi":
        if nn_preds.ndim != 2 or booster_preds.ndim != 2:
            raise ValueError("multi task requires 2D predictions (n, K)")
        if nn_preds.shape != booster_preds.shape:
            raise ValueError("nn and booster predictions must have same shape")
        if y_true.ndim != 1 or y_true.shape[0] != nn_preds.shape[0]:
            raise ValueError("y_true must be 1D with n samples")
    else:
        nn_preds = np.asarray(nn_preds, dtype=float).ravel()
        booster_preds = np.asarray(booster_preds, dtype=float).ravel()
        y_true = np.asarray(y_true, dtype=float).ravel()
        if (
            len(nn_preds) == 0
            or len(nn_preds) != len(booster_preds)
            or len(nn_preds) != len(y_true)
        ):
            raise ValueError("Prediction and target arrays must have equal non-zero length")

    if (
        not np.all(np.isfinite(nn_preds))
        or not np.all(np.isfinite(booster_preds))
        or not np.all(np.isfinite(y_true))
    ):
        raise ValueError("Inputs must be finite")

    if task_type == "binary":
        if not np.all((nn_preds >= -1e-9) & (nn_preds <= 1.0 + 1e-9)):
            raise ValueError("nn_preds must be in [0, 1] for binary classification")
        if not np.all((booster_preds >= -1e-9) & (booster_preds <= 1.0 + 1e-9)):
            raise ValueError("booster_preds must be in [0, 1] for binary classification")
        if not np.all(np.isin(y_true, [0, 1])):
            raise ValueError("y_true must be binary (0 or 1)")
        nn_preds = np.clip(nn_preds, 0.0, 1.0)
        booster_preds = np.clip(booster_preds, 0.0, 1.0)

    if task_type == "multi":
        if np.any(nn_preds < -1e-9) or np.any(booster_preds < -1e-9):
            raise ValueError("predictions must be non-negative for multi-class")
        nn_preds = nn_preds / nn_preds.sum(axis=1, keepdims=True)
        booster_preds = booster_preds / booster_preds.sum(axis=1, keepdims=True)
        if not np.all(np.isin(y_true.astype(int), np.arange(nn_preds.shape[1]))):
            raise ValueError("y_true labels out of range")

    # ---------------------------------------------------------------- baseline
    baseline_loss = _baseline_loss(
        y_true,
        task_type,
        n_classes=(nn_preds.shape[1] if task_type == "multi" else None),
    )

    loss_nn = _task_loss(nn_preds, y_true, task_type)
    loss_booster = _task_loss(booster_preds, y_true, task_type)

    # ---------------------------------------------------------------- optimizer
    def objective(beta: float) -> float:
        hybrid = beta * nn_preds + (1.0 - beta) * booster_preds
        return _task_loss(hybrid, y_true, task_type)

    result = minimize_scalar(objective, bounds=(0.0, 1.0), method="bounded")
    # UNCHANGED: argmin over endpoints + interior.
    candidates = [0.0, 1.0, float(result.x)]
    raw_beta = min(candidates, key=objective)
    raw_beta = validate_beta(raw_beta)

    hybrid_final = raw_beta * nn_preds + (1.0 - raw_beta) * booster_preds
    loss_hybrid = _task_loss(hybrid_final, y_true, task_type)

    # ---------------------------------------------------------------- UNINFORMATIVE rule
    # A task whose hybrid does not beat the constant baseline is UNINFORMATIVE
    # and gets NO beta claim. Compare with a tiny tolerance to avoid flagging a
    # task that ties the baseline as informative.
    tol = 1e-12
    beats_baseline = loss_hybrid < baseline_loss - tol
    if beats_baseline:
        task_status = STATUS_INFORMATIVE
        beta_claim: float | None = raw_beta
    else:
        task_status = STATUS_UNINFORMATIVE
        beta_claim = None

    boundary = raw_beta <= 1e-5 or raw_beta >= 1.0 - 1e-5

    diagnostics: dict[str, Any] = {
        "task_name": task_name,
        "task_type": task_type,
        "task_status": task_status,
        "beats_baseline": bool(beats_baseline),
        "usable_beta": beta_claim is not None,
        "beta_claim": beta_claim,  # None if UNINFORMATIVE
        "beta": raw_beta,  # raw argmin, inspection only
        "loss_before_nn": loss_nn,
        "loss_before_booster": loss_booster,
        "loss_after_fusion": loss_hybrid,
        "loss_baseline": baseline_loss,
        "optimizer_status": int(result.status) if hasattr(result, "status") else 0,
        "optimizer_message": str(result.message) if hasattr(result, "message") else "",
        "boundary_optimum": boundary,
        "n_samples": len(y_true),
        "loss_function": (
            "mse"
            if task_type == "regression"
            else ("log_loss" if task_type == "binary" else "cross_entropy")
        ),
    }
    return beta_claim, diagnostics


def predict_hybrid(
    nn_preds: np.ndarray,
    booster_preds: np.ndarray,
    beta: float | None,
    *,
    task_status: str | None = None,
    allow_uninformative: bool = False,
) -> np.ndarray:
    """Compute hybrid prediction: beta*nn + (1-beta)*booster.

    If task_status == UNINFORMATIVE and allow_uninformative is False, this
    raises instead of fusing, so an uninformative beta cannot leak into a
    prediction by accident. Use allow_uninformative=True only for diagnostics.
    """
    nn_preds = np.asarray(nn_preds, dtype=float)
    booster_preds = np.asarray(booster_preds, dtype=float)

    if task_status == STATUS_UNINFORMATIVE and not allow_uninformative:
        raise ValueError(
            "task is UNINFORMATIVE (hybrid does not beat the constant baseline); "
            "refusing to fuse. Pass allow_uninformative=True only for diagnostics."
        )

    if beta is None:
        if task_status is None or task_status == STATUS_UNINFORMATIVE:
            raise ValueError(
                "beta is None (no claim). For an INFORMATIVE task, pass the "
                "beta_claim from optimize_beta."
            )
        raise ValueError("beta is None but task is not UNINFORMATIVE; caller error")

    beta = validate_beta(beta)
    return np.asarray(beta * nn_preds + (1.0 - beta) * booster_preds, dtype=float)  # type: ignore[no-any-return]


def save_beta_weights(
    path: Path | str,
    beta_weights: dict[str, float | None],
    diagnostics: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Persist beta weights with diagnostics and metadata.

    A None value means the task is UNINFORMATIVE and carries no beta claim.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"beta_weights": beta_weights}
    if diagnostics:
        payload["diagnostics"] = diagnostics
    if metadata:
        payload["metadata"] = metadata
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def load_beta_weights(path: Path | str) -> dict[str, float | None]:
    """Load beta weights. None values are preserved (UNINFORMATIVE tasks)."""
    path = Path(path)
    data = json.loads(path.read_text())
    raw = dict(data.get("beta_weights", {}))
    validated: dict[str, float | None] = {}
    for task, val in raw.items():
        validated[task] = None if val is None else validate_beta(val)
    return validated


def load_beta_diagnostics(path: Path | str) -> dict[str, Any]:
    """Load beta diagnostics from file."""
    path = Path(path)
    data = json.loads(path.read_text())
    return dict(data.get("diagnostics", {}))


__all__ = [
    "STATUS_INFORMATIVE",
    "STATUS_UNINFORMATIVE",
    "load_beta_diagnostics",
    "load_beta_weights",
    "optimize_beta",
    "predict_hybrid",
    "save_beta_weights",
    "validate_beta",
]
