"""Unit tests for MIA feature extraction and metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from insurance_fl.evaluation.mia import compute_features, tpr_at_fpr


def _make_model():
    """Create a simple model-like object for testing."""
    import torch.nn as nn

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 2)

        def forward(self, x):
            return self.fc(x)

    model = SimpleModel()
    model.eval()
    return model


def test_compute_features_shape() -> None:
    model = _make_model()
    x = np.random.randn(20, 4).astype(np.float32)
    y = np.random.randint(0, 2, size=20)
    features = compute_features(model, x, y, task_type="binary")
    assert features.shape == (20, 3)
    assert np.all(features[:, 0] >= 0)
    assert np.all(features[:, 0] <= 1)


def test_compute_features_entropy_nonnegative() -> None:
    model = _make_model()
    x = np.random.randn(20, 4).astype(np.float32)
    y = np.random.randint(0, 2, size=20)
    features = compute_features(model, x, y, task_type="binary")
    assert np.all(features[:, 1] >= 0)


def test_compute_features_loss_nonnegative() -> None:
    model = _make_model()
    x = np.random.randn(20, 4).astype(np.float32)
    y = np.random.randint(0, 2, size=20)
    features = compute_features(model, x, y, task_type="binary")
    assert np.all(features[:, 2] >= 0)


def test_tpr_at_fpr_basic() -> None:
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    tpr = tpr_at_fpr(y_true, y_scores, 0.01)
    assert 0.0 <= tpr <= 1.0


def test_tpr_at_fpr_perfect() -> None:
    y_true = np.array([0] * 50 + [1] * 50)
    y_scores = np.array([0.0] * 50 + [1.0] * 50)
    tpr = tpr_at_fpr(y_true, y_scores, 0.01)
    assert tpr >= 0.5


def test_synthetic_signal_attack_learns() -> None:
    """Attack classifier can learn a known membership signal."""
    from sklearn.ensemble import RandomForestClassifier

    # Create a clear signal: members have high max-prob, non-members have low
    np.random.seed(42)
    n = 100
    member_features = np.column_stack(
        [
            np.random.uniform(0.8, 1.0, n),  # high max prob
            np.random.uniform(0.0, 0.3, n),  # low entropy
            np.random.uniform(0.0, 0.5, n),  # low loss
        ]
    )
    nonmember_features = np.column_stack(
        [
            np.random.uniform(0.3, 0.6, n),  # low max prob
            np.random.uniform(0.5, 1.0, n),  # high entropy
            np.random.uniform(1.0, 3.0, n),  # high loss
        ]
    )

    X = np.vstack([member_features, nonmember_features])
    y = np.concatenate([np.ones(n), np.zeros(n)])

    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X, y)

    # Test on held-out (same distribution)
    member_test = np.column_stack(
        [
            np.random.uniform(0.8, 1.0, 20),
            np.random.uniform(0.0, 0.3, 20),
            np.random.uniform(0.0, 0.5, 20),
        ]
    )
    nonmember_test = np.column_stack(
        [
            np.random.uniform(0.3, 0.6, 20),
            np.random.uniform(0.5, 1.0, 20),
            np.random.uniform(1.0, 3.0, 20),
        ]
    )
    X_test = np.vstack([member_test, nonmember_test])
    y_test = np.concatenate([np.ones(20), np.zeros(20)])

    auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    assert auc > 0.8, f"Expected AUC > 0.8, got {auc}"


def test_chance_level_random_labels() -> None:
    """With random labels, AUC should be near 0.5."""
    from sklearn.ensemble import RandomForestClassifier

    np.random.seed(42)
    X = np.random.randn(200, 3)
    y = np.random.randint(0, 2, size=200)

    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X, y)

    X_test = np.random.randn(50, 3)
    y_test = np.random.randint(0, 2, size=50)
    scores = clf.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, scores)
    # Should be roughly 0.5 (within reasonable range)
    assert 0.3 < auc < 0.7, f"Expected near-chance AUC, got {auc}"


def test_empty_and_small_samples() -> None:
    """Edge cases: small sample counts."""
    model = _make_model()
    x = np.random.randn(2, 4).astype(np.float32)
    y = np.array([0, 1])
    features = compute_features(model, x, y, task_type="binary")
    assert features.shape == (2, 3)


def test_regression_features() -> None:
    model = _make_model()
    x = np.random.randn(10, 4).astype(np.float32)
    y = np.random.randn(10)
    features = compute_features(model, x, y, task_type="regression")
    assert features.shape == (10, 3)
    assert np.all(features[:, 0] == 1.0)  # max prob = 1 for regression
    assert np.all(features[:, 1] == 0.0)  # entropy = 0 for regression
