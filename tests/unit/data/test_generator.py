"""Tests for insurance_fl.data.generator (schema, determinism, no-leakage)."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from insurance_fl.data.generator import (
    SPLIT_NAMES,
    TARGET_COLUMNS,
    EuropeanInsuranceFederatedDataset,
)

N = 400  # per client — small for fast tests


def _small_gen(**kw) -> EuropeanInsuranceFederatedDataset:
    kw.setdefault("samples_per_client", N)
    kw.setdefault("random_seed", 42)
    return EuropeanInsuranceFederatedDataset(**kw)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory) -> dict:
    """Generate one small dataset per module (shared across tests)."""
    out = tmp_path_factory.mktemp("gen")
    gen = _small_gen()
    gen.generate_dataset(str(out))
    return {"dir": out, "gen": gen}


# --- schema ---------------------------------------------------------------
def test_all_five_targets_present_and_correctly_typed(tiny) -> None:
    df = tiny["gen"].load_dataset(str(tiny["dir"]), "train", "germany_allianz")
    for col in TARGET_COLUMNS:
        assert col in df.columns, col
    assert pd.api.types.is_float_dtype(df["claim_risk_score"])
    assert pd.api.types.is_float_dtype(df["customer_lifetime_value"])
    for col in ("will_renew", "is_high_risk", "customer_segment"):
        assert set(df[col].unique()) <= {0, 1, 2}, col
        assert df[col].dtype.kind in "iu", col


def test_target_ranges(tiny) -> None:
    df = tiny["gen"].load_dataset(str(tiny["dir"]), "train", "france_axa")
    assert df["claim_risk_score"].between(0, 100).all()
    assert df["customer_lifetime_value"].between(1000, 250000).all()
    assert df["customer_segment"].isin([0, 1, 2]).all()


def test_no_label_leakage_high_risk_vs_claim_risk(tiny) -> None:
    """is_high_risk must be generated WITHOUT claim_risk_score.

    If it were thresholded from claim_risk_score, then within a fixed
    claim_risk_score band the label would be constant. We check overlap at
    both the top and bottom deciles: labels must vary within each band.
    """
    df = pd.concat(
        [
            tiny["gen"].load_dataset(str(tiny["dir"]), split, client)
            for client in ["germany_allianz", "france_axa"]
            for split in ("train", "val")
        ],
        ignore_index=True,
    )
    hi = df[df["claim_risk_score"] > df["claim_risk_score"].quantile(0.7)]
    lo = df[df["claim_risk_score"] < df["claim_risk_score"].quantile(0.3)]
    assert hi["is_high_risk"].nunique() == 2, "claim_risk_score leaks into is_high_risk"
    assert lo["is_high_risk"].nunique() == 2, "claim_risk_score leaks into is_high_risk"


def test_currency_column_per_country(tiny) -> None:
    ch = tiny["gen"].load_dataset(str(tiny["dir"]), "train", "switzerland_helvetia")
    de = tiny["gen"].load_dataset(str(tiny["dir"]), "train", "germany_allianz")
    assert (ch["currency"] == "CHF").all()
    assert (de["currency"] == "EUR").all()


def test_unique_customer_ids(tiny) -> None:
    df = tiny["gen"].load_dataset(str(tiny["dir"]), "train", "belgium_ageas")
    assert df["customer_id"].is_unique


def test_splits_exist_for_every_client(tiny) -> None:
    base = Path(tiny["dir"])
    for client in tiny["gen"].client_ids:
        for split in SPLIT_NAMES:
            assert (base / "federated" / client / f"{split}.parquet").exists()
    assert (base / "metadata.json").exists()
    assert (base / "dataset_info.json").exists()
    assert (base / "centralized" / "all_clients.parquet").exists()


# --- determinism (suspect #4 guard) ---------------------------------------
def test_same_seed_same_output_same_process() -> None:
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        g1, g2 = _small_gen(), _small_gen()
        g1.generate_dataset(d1)
        g2.generate_dataset(d2)
        for client in g1.client_ids:
            for split in ("train", "val", "test", "site"):
                df1 = g1.load_dataset(d1, split, client)
                df2 = g2.load_dataset(d2, split, client)
                pd.testing.assert_frame_equal(
                    df1.drop(columns=["customer_id"]),
                    df2.drop(columns=["customer_id"]),
                )


@pytest.mark.slow
def test_determinism_across_processes() -> None:
    """Two separate processes with same seed produce identical data.

    Guards against PYTHONHASHSEED-salted hash() seeding (suspect #4).
    """
    script = (
        "import hashlib, tempfile;"
        "from insurance_fl.data.generator import EuropeanInsuranceFederatedDataset as G;"
        "d = tempfile.mkdtemp();"
        "g = G(samples_per_client=300, random_seed=7);"
        "g.generate_dataset(d);"
        "df = g.load_dataset(d, 'train', 'germany_allianz');"
        "print(hashlib.sha256(df.to_csv(index=False).encode()).hexdigest())"
    )
    import os

    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    hashes = set()
    for _ in range(2):
        r = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )
        hashes.add(r.stdout.strip().splitlines()[-1])
    assert len(hashes) == 1, f"Non-deterministic generation: {hashes}"


def test_different_seed_different_output() -> None:
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        _small_gen(random_seed=1).generate_dataset(d1)
        _small_gen(random_seed=2).generate_dataset(d2)
        g = _small_gen()
        df1 = g.load_dataset(d1, "train", "germany_allianz")
        df2 = g.load_dataset(d2, "train", "germany_allianz")
        assert not df1["annual_income"].equals(df2["annual_income"])


# --- quality injection flag (suspect #8) ----------------------------------
def test_quality_injection_flag_off_leaves_no_nans() -> None:
    with tempfile.TemporaryDirectory() as d:
        gen = _small_gen(inject_quality_issues=False)
        gen.generate_dataset(d)
        df = gen.load_dataset(d, "train", "germany_allianz")
        assert int(df.isna().sum().sum()) == 0


def test_quality_injection_rates_and_dtype_safety() -> None:
    """Injected NaNs: int cols upcast to float, ~3% missing, invariants hold.

    Guards suspect #7 (NaN into int columns, poisoned outlier max,
    tenure forced beyond age).
    """
    with tempfile.TemporaryDirectory() as d:
        gen = _small_gen()
        gen.generate_dataset(d)
        df = gen.load_dataset(d, "train", "germany_allianz")

        # 1) dtype safety: columns containing NaN must be float, not int-with-NaN
        nan_cols = df.columns[df.isna().any()]
        for col in nan_cols:
            assert pd.api.types.is_float_dtype(df[col]), f"{col} has NaN but {df[col].dtype}"

        # 2) missing rate close to the configured 3% (tolerance: binomial noise)
        numeric_cols = [c for c in df.select_dtypes("number").columns if c not in TARGET_COLUMNS]
        total_cells = len(df) * len(numeric_cols)
        rate = float(df[numeric_cols].isna().sum().sum()) / total_cells
        assert 0.005 < rate < 0.06, f"unexpected missing rate {rate}"

        # 3) invariants: tenure never exceeds age - 18; ids unique.
        #    Rows where either value is NaN are unknown -> excluded
        #    (NaN comparisons are False and would mask real violations).
        known = df[["years_as_customer", "age"]].dropna()
        assert (known["years_as_customer"] <= known["age"] - 18).all()
        assert df["customer_id"].is_unique


def test_tenure_injection_with_nan_age_does_not_crash() -> None:
    """Guards IntCastingNaNError: tenure injection must tolerate NaN ages.

    Regression: .astype("int64") on cap (age - 18) raised when age contained
    injected NaNs; only triggered when the 1% inconsistency mask overlapped a
    NaN-age row (probabilistic at small N, certain at 5000).
    """
    rng = np.random.default_rng(0)
    n = 3000
    age = rng.integers(18, 90, n).astype("float64")
    # tenure must respect age - 18 BEFORE injection (injection only touches ~1%)
    tenure = np.minimum(rng.integers(0, 40, n), age - 18).astype("float64")
    base = pd.DataFrame(
        {
            "age": age,
            "years_as_customer": tenure,
            "annual_income": rng.normal(50000, 15000, n),
        }
    )
    base.loc[::10, "age"] = np.nan  # ensure overlap with the 1% bad-mask

    gen = _small_gen()
    out = gen._add_data_quality_issues(base, np.random.default_rng(1))

    assert not out["years_as_customer"].isna().all()
    known = out[["years_as_customer", "age"]].dropna()
    assert (known["years_as_customer"] <= known["age"] - 18 + 1e-9).all()
