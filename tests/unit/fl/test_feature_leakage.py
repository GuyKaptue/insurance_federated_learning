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
EXPECTED_INDICATORS = ("annual_income__missing", "savings_balance__missing", "medical_expenses__missing")


@pytest.mark.skipif(
    not (DATA_DIR / "federated" / CLIENT_IDS[0] / "train.parquet").exists(),
    reason="Canonical dataset missing; run make data.",
)
def test_feature_surface_is_identical_ordered_and_clean() -> None:
    """All clients expose the same ordered 24-feature model input surface.

    24 = 21 base numeric features + 3 fixed missingness indicators
    (annual_income__missing, savings_balance__missing, medical_expenses__missing).
    Earlier references to 26 were incorrect/outdated; the canonical load_partition
    output width is 24 with no target/identifier/probability leakage.
    """
    lists = {cid: get_feature_names(DATA_DIR, cid, "train") for cid in CLIENT_IDS}
    first = lists[CLIENT_IDS[0]]
    assert all(v == first for v in lists.values()), f"feature lists differ across clients: { {k: len(v) for k,v in lists.items()} }"
    assert len(first) == 24, f"expected 24 features (21 base + 3 indicators), got {len(first)}: {first}"
    bad = set(first) & FORBIDDEN
    assert not bad, f"forbidden columns in features: {sorted(bad)}"
    assert tuple(f"{c}__missing" for c in MISSING_INDICATOR_COLS) == EXPECTED_INDICATORS
    assert first[-3:] == list(EXPECTED_INDICATORS)
    assert set(EXPECTED_INDICATORS) <= set(first)
    assert all(not n.endswith("__missing") for n in first[:-3])

