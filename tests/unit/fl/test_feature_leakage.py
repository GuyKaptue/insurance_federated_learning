from __future__ import annotations

from pathlib import Path

import pytest

from insurance_fl.fl.task import CLIENT_IDS, MISSING_INDICATOR_COLS, get_feature_names

DATA_DIR = Path("data/processed/european_insurance_federated")
FORBIDDEN = {
    "claim_risk_score",
    "customer_lifetime_value",
    "will_renew",
    "is_high_risk",
    "customer_segment",
    "segment_probabilities",
    "risk_probability",
    "renewal_probability",
    "customer_id",
}


@pytest.mark.skipif(
    not (DATA_DIR / "federated" / CLIENT_IDS[0] / "train.parquet").exists(),
    reason="Canonical dataset missing; run make data.",
)
def test_features_exclude_targets_and_identifiers() -> None:
    names = get_feature_names(DATA_DIR, CLIENT_IDS[0], "train")
    assert names, "expected non-empty feature list"
    bad = set(names) & FORBIDDEN
    assert not bad, f"forbidden columns in features: {sorted(bad)}"
    # 21 base numeric features + 3 fixed missingness indicators = 24 model inputs.
    assert len(names) == 24, f"expected 24 features (21 base + 3 indicators), got {len(names)}: {names}"
    expected_indicators = [f"{c}__missing" for c in MISSING_INDICATOR_COLS]
    for c in expected_indicators:
        assert c in names
    assert names[-3:] == expected_indicators
    assert all(not n.endswith("__missing") for n in names[:-3])
