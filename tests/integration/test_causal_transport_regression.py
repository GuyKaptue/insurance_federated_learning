"""Regression tests for causal inference and transportability (CI-enforced)."""

from __future__ import annotations

import pytest

from pathlib import Path

import pandas as pd

from insurance_fl.causal.dag_check import load_dag, validate_dag
from insurance_fl.causal.estimands import CausalEstimand, estimate_backdoor, estimate_doubly_robust
from insurance_fl.transport.transport import transport_causal_effect, transport_outcome_model

DATA_DIR = Path("data/processed/european_insurance_federated")
DAG_PATH = Path("docs/causal/dag.yaml")

ADJUSTMENT_SET = (
    "age",
    "annual_income",
    "savings_balance",
    "credit_score",
    "debt_to_income_ratio",
)


def _require_data() -> None:
    sample = DATA_DIR / "federated" / "germany_allianz" / "train.parquet"
    if not sample.exists():
        pytest.skip(
            f"Canonical dataset missing at {sample}. "
            "Generate with: make data (or insurance-fl-generate --samples 5000 "
            "--output data/processed/european_insurance_federated)"
        )


def _load_all_train() -> pd.DataFrame:
    _require_data()
    client_ids = [
        "germany_allianz",
        "france_axa",
        "belgium_ageas",
        "switzerland_helvetia",
    ]
    dfs = [pd.read_parquet(DATA_DIR / "federated" / cid / "train.parquet") for cid in client_ids]
    data = pd.concat(dfs, ignore_index=True)
    data["has_claims"] = (data["number_of_claims"] > 0).astype(int)
    return data


def _load_train(client_id: str) -> pd.DataFrame:
    _require_data()
    return pd.read_parquet(DATA_DIR / "federated" / client_id / "train.parquet")


# ---------------------------------------------------------------------------
# DAG
# ---------------------------------------------------------------------------


def test_dag_passes_validation() -> None:
    spec = load_dag(DAG_PATH)
    assert validate_dag(spec) == []


def test_dag_fails_on_unknown_node_edge() -> None:
    import copy

    spec = load_dag(DAG_PATH)
    bad = copy.deepcopy(spec)
    bad["edges"].append({"source": "nonexistent_node", "target": "claim_risk_score"})
    errors = validate_dag(bad)
    assert errors, "mutated DAG should fail validation"


def test_dag_fails_on_bad_adjustment_set() -> None:
    import copy

    spec = load_dag(DAG_PATH)
    bad = copy.deepcopy(spec)
    adjustment_sets = bad.get("adjustment_sets")
    if adjustment_sets:
        key = next(iter(adjustment_sets))
        adjustment_sets["badkey_no_separator"] = adjustment_sets.pop(key)
    errors = validate_dag(bad)
    assert errors, "mutated adjustment set should fail validation"


# ---------------------------------------------------------------------------
# ATE agreement
# ---------------------------------------------------------------------------


def test_ate_estimators_agree() -> None:
    data = _load_all_train()
    estimand = CausalEstimand.from_dag(
        name="has_claims__claim_risk_score",
        treatment="has_claims",
        outcome="claim_risk_score",
        adjustment_set=ADJUSTMENT_SET,
        dag_path=DAG_PATH,
    )
    bd = estimate_backdoor(data, estimand)
    dr = estimate_doubly_robust(data, estimand)
    rel_diff = abs(bd["ate"] - dr["ate"]) / (abs(bd["ate"]) + 1e-12)
    assert rel_diff < 0.10, f"estimators disagree: rel_diff={rel_diff}"


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------


def test_transport_verdict_de_fr_to_ch() -> None:
    source_df = pd.concat(
        [_load_train("germany_allianz"), _load_train("france_axa")], ignore_index=True
    )
    target_df = _load_train("switzerland_helvetia")
    report = transport_outcome_model(source_df, target_df, "claim_risk_score")
    assert report.transportable_metric is True
    assert 1.0 < report.loss_ratio < 1.2

    source_df["has_claims"] = (source_df["number_of_claims"] > 0).astype(int)
    target_df["has_claims"] = (target_df["number_of_claims"] > 0).astype(int)
    estimand = CausalEstimand.from_dag(
        name="has_claims__claim_risk_score",
        treatment="has_claims",
        outcome="claim_risk_score",
        adjustment_set=ADJUSTMENT_SET,
        dag_path=DAG_PATH,
    )
    te = transport_causal_effect(estimand, source_df, target_df, None)
    assert te["transportable"] is False
