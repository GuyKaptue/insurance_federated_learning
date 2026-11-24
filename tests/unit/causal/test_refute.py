# Test for causal refute module

from pathlib import Path

import pandas as pd
import pytest

from insurance_fl.causal.estimands import CausalEstimand
from insurance_fl.causal.refute import run_all_refutations


def test_refute_import():
    """Verify refute module is importable and run_all_refutations is callable."""
    assert callable(run_all_refutations)


def test_refute_with_real_data():
    """Test refutation with real federated data (full dataset)."""
    path = Path(
        "data/processed/european_insurance_federated/federated/germany_allianz/train.parquet"
    )
    if not path.exists():
        pytest.skip(
            f"Canonical dataset missing at {path}. "
            "Generate with: make data (or insurance-fl-generate --samples 5000 "
            "--output data/processed/european_insurance_federated)"
        )
    data = pd.read_parquet(path)
    data["has_claims"] = (data["number_of_claims"] > 0).astype(int)

    estimand = CausalEstimand.from_dag(
        name="has_claims__claim_risk_score",
        treatment="has_claims",
        outcome="claim_risk_score",
        adjustment_set=(
            "age",
            "annual_income",
            "savings_balance",
            "credit_score",
            "debt_to_income_ratio",
        ),
        dag_path="docs/causal/dag.yaml",
    )

    results = run_all_refutations(data, estimand)
    # At minimum, the function should return a dict with 'passed' keys
    assert isinstance(results, dict)
    for _name, result in results.items():
        assert "passed" in result
