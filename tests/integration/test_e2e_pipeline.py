"""Integration tests: generate -> preprocess -> train -> evaluate.

Covers the full data/model pipeline that is implemented today. FL-simulation
coverage will be added in the fl/ package phase (marked integration).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from insurance_fl.data.generator import TARGET_COLUMNS, EuropeanInsuranceFederatedDataset
from insurance_fl.data.preprocess import DataPreprocessor, PreprocessorConfig
from insurance_fl.models.multi_task_nn import MultiTaskNN

pytestmark = pytest.mark.integration

TASK_TYPES = {
    "claim_risk_score": "regression",
    "customer_lifetime_value": "regression",
    "will_renew": "binary",
    "is_high_risk": "binary",
    "customer_segment": "multi",
}


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, kind: str) -> dict:
    if kind == "regression":
        err = y_pred - y_true
        ss_res = float((err**2).sum())
        ss_tot = float(((y_true - y_true.mean()) ** 2).sum()) or 1e-12
        return {
            "mae": float(np.abs(err).mean()),
            "rmse": float(np.sqrt((err**2).mean())),
            "r2": 1.0 - ss_res / ss_tot,
        }
    y_hat = (y_pred >= 0.5).astype(int)
    tp = int(((y_hat == 1) & (y_true == 1)).sum())
    fp = int(((y_hat == 1) & (y_true == 0)).sum())
    fn = int(((y_hat == 0) & (y_true == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"accuracy": float((y_hat == y_true).mean()), "f1": f1}


def test_end_to_end_generate_preprocess_train_evaluate(tmp_path) -> None:
    # 1) generate (small, quality injection on)
    gen = EuropeanInsuranceFederatedDataset(
        samples_per_client=300, random_seed=11, inject_quality_issues=True
    )
    out = str(tmp_path)
    gen.generate_dataset(out)

    # 2) load + preprocess (fit on train only)
    train = gen.load_dataset(out, "train", "germany_allianz")
    val = gen.load_dataset(out, "val", "germany_allianz")
    cfg = PreprocessorConfig(target_columns=TARGET_COLUMNS)
    pp = DataPreprocessor(cfg)
    X_train = pp.fit_transform(train, targets=train[TARGET_COLUMNS])
    X_val = pp.transform(val, targets=val[TARGET_COLUMNS])

    feature_cols = [c for c in X_train.columns if c not in TARGET_COLUMNS]
    assert feature_cols, "no feature columns produced"

    x_tr = torch.tensor(X_train[feature_cols].to_numpy(dtype=np.float32))
    x_va = torch.tensor(X_val[feature_cols].to_numpy(dtype=np.float32))

    # 3) train a few epochs on all 5 tasks
    torch.manual_seed(0)
    model = MultiTaskNN(
        input_dim=x_tr.shape[1],
        task_types=TASK_TYPES,
        hidden_dims=[64, 32],
        dropout_rate=0.1,
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    y_tr = {t: torch.tensor(X_train[t].to_numpy()) for t in TARGET_COLUMNS}
    model.train()
    losses = []
    for _ in range(5):
        opt.zero_grad()
        logits = model(x_tr)
        total = model.get_total_loss(model.get_losses(logits, y_tr))
        total.backward()
        opt.step()
        losses.append(total.detach().item())

    assert np.isfinite(losses).all()
    assert losses[-1] <= losses[0] * 1.5, f"loss diverged: {losses}"

    # 4) evaluate on val — metrics must be finite and sane
    model.eval()
    with torch.no_grad():
        preds = model.predict(x_va)

    reg = _metrics(
        X_val["claim_risk_score"].to_numpy(),
        preds["claim_risk_score"].numpy(),
        "regression",
    )
    assert np.isfinite(list(reg.values())).all()
    assert reg["rmse"] < 100.0  # target is bounded [0, 100]

    bin_m = _metrics(
        X_val["will_renew"].to_numpy(),
        preds["will_renew"].numpy(),
        "binary",
    )
    assert 0.0 <= bin_m["accuracy"] <= 1.0
    assert 0.0 <= bin_m["f1"] <= 1.0

    seg_pred = preds["customer_segment"].argmax(dim=-1).numpy()
    seg_true = X_val["customer_segment"].to_numpy().astype(int)
    assert set(np.unique(seg_pred)) <= {0, 1, 2}
    assert 0.0 <= float((seg_pred == seg_true).mean()) <= 1.0


def test_deterministic_pipeline_same_seed(tmp_path) -> None:
    """Same seed => identical preprocessed tensors (reproducibility gate)."""

    def run(root: str) -> torch.Tensor:
        gen = EuropeanInsuranceFederatedDataset(samples_per_client=200, random_seed=5)
        gen.generate_dataset(root)
        train = gen.load_dataset(root, "train", "france_axa")
        cfg = PreprocessorConfig(target_columns=TARGET_COLUMNS)
        X = DataPreprocessor(cfg).fit_transform(train, targets=train[TARGET_COLUMNS])
        feats = [c for c in X.columns if c not in TARGET_COLUMNS]
        return torch.tensor(X[feats].to_numpy(dtype=np.float32))

    import tempfile

    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        t1, t2 = run(d1), run(d2)
    assert t1.shape == t2.shape
    assert torch.equal(t1, t2)
