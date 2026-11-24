"""Unit tests for ``src/insurance_fl/fl/boosters.py``."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("xgboost")
pytest.importorskip("lightgbm")
pytest.importorskip("catboost")

from insurance_fl.fl.boosters import train_boosters_for_client


def test_boosters_train_and_save() -> None:
    rng = np.random.RandomState(42)
    x = rng.randn(100, 5).astype(np.float32)
    y_reg = rng.randn(100).astype(np.float32)
    (rng.rand(100) > 0.5).astype(int)
    rng.randint(0, 3, size=100)

    # Use only regression task for speed

    with tempfile.TemporaryDirectory() as tmpdir:
        result = train_boosters_for_client(
            client_id="test_client",
            x_train=x[:80],
            y_train={"claim_risk_score": y_reg[:80]},
            x_val=x[80:],
            y_val={"claim_risk_score": y_reg[80:]},
            output_dir=Path(tmpdir),
            tasks=["claim_risk_score"],
            n_folds=3,
            seed=42,
        )
        assert "claim_risk_score" in result
        meta = result["claim_risk_score"]
        assert "alpha_xgb" in meta
        assert "alpha_lgbm" in meta
        assert "alpha_cat" in meta
        # Artifacts exist
        assert (Path(tmpdir) / "xgboost_claim_risk_score.json").exists()
        assert (Path(tmpdir) / "lightgbm_claim_risk_score.txt").exists()
        assert (Path(tmpdir) / "catboost_claim_risk_score.cbm").exists()
        assert (Path(tmpdir) / "stacking_meta_claim_risk_score.json").exists()
