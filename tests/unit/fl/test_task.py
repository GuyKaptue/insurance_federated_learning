"""Unit tests for ``src/insurance_fl/fl/task.py`` (Layer 1 / Task 0)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("flwr")

from insurance_fl.fl import task as fl_task  # noqa: E402


@pytest.fixture(scope="module")
def dataset_dir(tmp_path_factory: pytest.TempPathFactory):
    """Generate the smoke dataset once for this module (~2 s)."""
    fl_task.clear_missing_policy_cache()
    path = fl_task.prepare_dataset(tmp_path_factory.mktemp("fl_task0"), 512, 42)
    fl_task.clear_missing_policy_cache()
    return path


def test_prepare_is_idempotent_and_loads(dataset_dir) -> None:
    fl_task.clear_missing_policy_cache()
    assert fl_task.prepare_dataset(dataset_dir, 512, 42) == dataset_dir
    for client_id in fl_task.CLIENT_IDS:
        x, y = fl_task.load_partition(dataset_dir, client_id, split="train")
        assert x.dtype == np.float32 and x.shape[0] > 100
        assert set(y.keys()) == {
            "claim_risk_score",
            "customer_lifetime_value",
            "will_renew",
            "is_high_risk",
            "customer_segment",
        }
        assert len(y["claim_risk_score"]) == x.shape[0]
        assert np.isfinite(x).all()


def test_load_partition_missing_data_raises(tmp_path) -> None:
    fl_task.clear_missing_policy_cache()
    with pytest.raises(FileNotFoundError, match="--prepare"):
        fl_task.load_partition(tmp_path, "germany_allianz")


def test_train_and_evaluate_smoke(dataset_dir) -> None:
    fl_task.clear_missing_policy_cache()
    x, y = fl_task.load_partition(dataset_dir, "germany_allianz")
    x, y = x[:64], {k: v[:64] for k, v in y.items()}
    net = fl_task.build_model(input_dim=x.shape[1])
    loss = fl_task.train(net, x, y, epochs=1, batch_size=16, lr=1e-3)
    assert np.isfinite(loss)
    metrics = fl_task.evaluate(net, x, y)
    assert np.isfinite(metrics["loss"])
    assert "claim_risk_score_rmse" in metrics
    assert "customer_segment_acc" in metrics


def test_feature_width_is_constant_across_clients(tmp_path) -> None:
    """All clients must have identical feature width (fixed missing indicators)."""
    fl_task.clear_missing_policy_cache()
    # Build minimal per-client parquet data with differing NaN patterns.
    for cid in fl_task.CLIENT_IDS:
        d = tmp_path / "federated" / cid
        d.mkdir(parents=True, exist_ok=True)
        for split, n in [("train", 10), ("val", 4), ("test", 4)]:
            df = pd.DataFrame(
                {
                    "annual_income": [50000.0] * n,
                    "savings_balance": [10000.0] * n,
                    "medical_expenses": [1000.0] * n,
                    "age": [40.0] * n,
                    "claim_risk_score": [1.0] * n,
                    "customer_lifetime_value": [2000.0] * n,
                    "will_renew": [1] * n,
                    "is_high_risk": [0] * n,
                    "customer_segment": [0] * n,
                }
            )
            # Inject NaNs in different columns per client (Belgium-like: none)
            if cid == "germany_allianz":
                df.loc[0, "annual_income"] = np.nan
                df.loc[1, "savings_balance"] = np.nan
            elif cid == "france_axa":
                df.loc[0, "medical_expenses"] = np.nan
            elif cid == "switzerland_helvetia":
                df.loc[2, "medical_expenses"] = np.nan
            df.to_parquet(d / f"{split}.parquet", index=False)

    widths = {}
    for cid in fl_task.CLIENT_IDS:
        x, _ = fl_task.load_partition(tmp_path, cid, split="train")
        widths[cid] = x.shape[1]
    assert len(set(widths.values())) == 1, f"feature widths differ across clients: {widths}"
    fl_task.clear_missing_policy_cache()


def test_policy_is_fit_on_train_only(tmp_path) -> None:
    """Val/test medians must not influence imputation; train median is used."""
    fl_task.clear_missing_policy_cache()
    cid = "germany_allianz"
    d = tmp_path / "federated" / cid
    d.mkdir(parents=True, exist_ok=True)
    # Train: annual_income [1,2,3,nan] -> median 2.0
    train = pd.DataFrame(
        {
            "annual_income": [1.0, 2.0, 3.0, np.nan],
            "savings_balance": [10.0, 10.0, 10.0, 10.0],
            "medical_expenses": [5.0, 5.0, 5.0, 5.0],
            "claim_risk_score": [0.0] * 4,
            "customer_lifetime_value": [1.0] * 4,
            "will_renew": [1] * 4,
            "is_high_risk": [0] * 4,
            "customer_segment": [0] * 4,
        }
    )
    # Val has NaN but a very different distribution (would give median 100 if fit on val)
    val = pd.DataFrame(
        {
            "annual_income": [100.0, 100.0, np.nan, 100.0],
            "savings_balance": [10.0, 10.0, 10.0, 10.0],
            "medical_expenses": [5.0, 5.0, 5.0, 5.0],
            "claim_risk_score": [0.0] * 4,
            "customer_lifetime_value": [1.0] * 4,
            "will_renew": [1] * 4,
            "is_high_risk": [0] * 4,
            "customer_segment": [0] * 4,
        }
    )
    train.to_parquet(d / "train.parquet", index=False)
    val.to_parquet(d / "val.parquet", index=False)
    x_val, _ = fl_task.load_partition(tmp_path, cid, split="val")
    # feature_columns order is from train df numeric non-target/excluded sorted by select_dtypes?
    # Easier: check that the imputed value equals train median (2.0), not val median (100.0)
    # annual_income is first feature column in this small df
    cols, _ = fl_task._fit_missing_policy(tmp_path, cid)
    ai_idx = cols.index("annual_income")
    assert np.isclose(x_val[2, ai_idx], 2.0), f"expected train median 2.0, got {x_val[2, ai_idx]}"
    fl_task.clear_missing_policy_cache()


def test_load_partition_is_finite_for_every_client() -> None:
    """load_partition must return finite features for all clients/splits (full data)."""
    from pathlib import Path

    data_dir = Path("data/processed/european_insurance_federated")
    if not data_dir.exists():
        pytest.skip("full federated data not available")
    fl_task.clear_missing_policy_cache()
    for cid in fl_task.CLIENT_IDS:
        for split in ("train", "val", "test"):
            try:
                x, _ = fl_task.load_partition(data_dir, cid, split=split)
            except FileNotFoundError:
                continue
            assert np.isfinite(x).all(), f"non-finite values for {cid}/{split}"
    fl_task.clear_missing_policy_cache()
