"""Membership Inference Attack (MIA) evaluation.

Black-box MIA using shadow models. Measures empirical membership
leakage — NOT a replacement for formal privacy accounting.

Shadow-model protocol:
1. Define a target model and task.
2. Create disjoint member and non-member evaluation samples.
3. Train shadow models using disjoint data partitions or synthetic data.
4. Extract the same features from shadow and target models.
5. Train an attack classifier using shadow-model membership labels.
6. Evaluate the attack on a held-out target-model evaluation split.
7. Report aggregate results and per-client results when data support it.

Candidate features: max predicted class probability, prediction entropy,
per-sample loss. Consistent feature definitions; task-appropriate handling
of binary, multiclass, and multilabel outputs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve


def compute_features(
    model,
    x: np.ndarray,
    y: np.ndarray,
    task_type: str = "binary",
) -> np.ndarray:
    """Extract MIA features from model predictions.

    Features per sample:
        0: max predicted probability
        1: prediction entropy
        2: per-sample loss (cross-entropy for classification, MSE for regression)

    Returns an (n_samples, 3) array.
    """
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y)

    # Get model predictions
    import torch

    model.eval()
    with torch.no_grad():
        xt = torch.as_tensor(x)
        logits = model.forward(xt)

    # Handle single-task: logits may be a dict or a tensor
    if isinstance(logits, dict):
        # Take the first task
        task_name = next(iter(logits.keys()))
        out = logits[task_name].detach().numpy()
    else:
        out = logits.detach().numpy()

    out = np.asarray(out)

    features = np.zeros((len(x), 3), dtype=np.float64)

    if task_type in ("binary", "multi_classification"):
        # Convert logits to probabilities
        if out.ndim == 1 or (out.ndim == 2 and out.shape[1] == 1):
            out = out.ravel()
            probs = 1.0 / (1.0 + np.exp(-out))
            probs = np.column_stack([1 - probs, probs])
        else:
            # Softmax
            exp_out = np.exp(out - np.max(out, axis=1, keepdims=True))
            probs = exp_out / exp_out.sum(axis=1, keepdims=True)

        # Max predicted probability
        features[:, 0] = probs.max(axis=1)

        # Entropy
        eps = 1e-12
        features[:, 1] = -np.sum(probs * np.log(probs + eps), axis=1)

        # Per-sample cross-entropy loss
        if out.ndim == 1 or (out.ndim == 2 and out.shape[1] == 1):
            y_flat = y.ravel().astype(int)
            p_true = np.where(y_flat == 1, probs[:, 1], probs[:, 0])
            features[:, 2] = -np.log(p_true + eps)
        else:
            y_flat = y.ravel().astype(int)
            n = len(y_flat)
            p_true = probs[np.arange(n), y_flat]
            features[:, 2] = -np.log(p_true + eps)
    else:
        # Regression: use MSE as loss, max prob = 1, entropy = 0
        if out.ndim == 2:
            out = out[:, 0]
        out = out.ravel()
        y_flat = y.ravel().astype(float)
        features[:, 0] = 1.0  # no probability for regression
        features[:, 1] = 0.0
        features[:, 2] = (out - y_flat) ** 2

    return np.asarray(features, dtype=float)  # type: ignore[no-any-return]


def tpr_at_fpr(y_true: np.ndarray, y_scores: np.ndarray, target_fpr: float = 0.01) -> float:
    """Compute TPR at a given FPR threshold."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    # Find the TPR at the FPR closest to target_fpr
    idx = int(np.searchsorted(fpr, target_fpr))
    if idx >= len(tpr):
        idx = len(tpr) - 1
    if idx == 0:
        return float(tpr[0])
    # Interpolate
    if fpr[idx] == fpr[idx - 1]:
        return float(tpr[idx])
    frac = (target_fpr - fpr[idx - 1]) / (fpr[idx] - fpr[idx - 1])
    return float(tpr[idx - 1] + frac * (tpr[idx] - tpr[idx - 1]))


def run_mia(
    target_model,
    shadow_models: list,
    x_member: np.ndarray,
    y_member: np.ndarray,
    x_nonmember: np.ndarray,
    y_nonmember: np.ndarray,
    x_shadow_member: np.ndarray,
    y_shadow_member: np.ndarray,
    x_shadow_nonmember: np.ndarray,
    y_shadow_nonmember: np.ndarray,
    task_type: str = "binary",
    task_name: str = "unknown",
    client_id: str = "unknown",
    dp_config: dict[str, Any] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Run the full MIA evaluation.

    Returns a dict with metrics and metadata.
    """
    np.random.seed(seed)

    # Extract features from shadow models
    shadow_features_list = []
    shadow_labels_list = []

    for sm in shadow_models:
        # Member features (label=1)
        f_mem = compute_features(sm, x_shadow_member, y_shadow_member, task_type)
        shadow_features_list.append(f_mem)
        shadow_labels_list.append(np.ones(len(f_mem)))

        # Non-member features (label=0)
        f_nonmem = compute_features(sm, x_shadow_nonmember, y_shadow_nonmember, task_type)
        shadow_features_list.append(f_nonmem)
        shadow_labels_list.append(np.zeros(len(f_nonmem)))

    X_shadow = np.vstack(shadow_features_list)
    y_shadow = np.concatenate(shadow_labels_list)

    # Train attack classifier
    attack_clf = RandomForestClassifier(n_estimators=100, random_state=seed)
    attack_clf.fit(X_shadow, y_shadow)

    # Evaluate on target model
    f_target_member = compute_features(target_model, x_member, y_member, task_type)
    f_target_nonmember = compute_features(target_model, x_nonmember, y_nonmember, task_type)

    X_target = np.vstack([f_target_member, f_target_nonmember])
    y_target = np.concatenate([np.ones(len(f_target_member)), np.zeros(len(f_target_nonmember))])

    y_scores = attack_clf.predict_proba(X_target)[:, 1]

    # Compute metrics
    try:
        auc = float(roc_auc_score(y_target, y_scores))
    except ValueError:
        auc = None

    tpr_1 = tpr_at_fpr(y_target, y_scores, 0.01)

    result = {
        "task": task_name,
        "client": client_id,
        "roc_auc": auc,
        "tpr_at_1pct_fpr": tpr_1,
        "member_count": len(x_member),
        "nonmember_count": len(x_nonmember),
        "shadow_model_count": len(shadow_models),
        "dp_config": dp_config or {},
        "accounting_status": "unavailable",
        "seed": seed,
        "limitations": [
            "Empirical attack result, not a formal privacy guarantee.",
            "Depends on attack strength, sample size, model access, and data distribution.",
            "AUC near 0.5 does not prove zero leakage.",
        ],
    }

    return result


__all__ = ["compute_features", "run_mia", "tpr_at_fpr"]
