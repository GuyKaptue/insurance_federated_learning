"""Tests for insurance_fl.data.country_profiles (schema fails loudly)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from insurance_fl.data.country_profiles import COUNTRY_PROFILES, CountryProfile

REQUIRED = dict(
    income_mean=50000.0,
    income_std=15000.0,
    savings_rate=0.1,
    savings_mean=3000.0,
    claim_frequency=0.04,
    life_expectancy=81,
    avg_age=46.0,
    population=1000000,
    gdp_per_capita=45000.0,
    car_premium=800.0,
    health_spending=6000.0,
    risk_appetite=0.7,
    digital_adoption=0.8,
    market_share=0.2,
    currency="EUR",
    regulatory_framework="TestReg",
)


def test_all_four_markets_present() -> None:
    assert set(COUNTRY_PROFILES) == {"Germany", "France", "Belgium", "Switzerland"}
    assert COUNTRY_PROFILES["Germany"].company == "Allianz"
    assert COUNTRY_PROFILES["Switzerland"].company == "Helvetia"


def test_switzerland_uses_chf_and_others_eur() -> None:
    assert COUNTRY_PROFILES["Switzerland"].currency == "CHF"
    for c in ("Germany", "France", "Belgium"):
        assert COUNTRY_PROFILES[c].currency == "EUR"


def test_valid_profile_constructs() -> None:
    p = CountryProfile(country="Testland", company="TestCo", **REQUIRED)
    assert p.country == "Testland"


def test_income_std_ge_mean_rejected() -> None:
    bad = dict(REQUIRED, income_std=60000.0)  # >= income_mean
    with pytest.raises(ValidationError):
        CountryProfile(country="X", company="Y", **bad)


def test_out_of_range_fields_rejected() -> None:
    with pytest.raises(ValidationError):
        CountryProfile(country="X", company="Y", **dict(REQUIRED, savings_rate=1.5))
    with pytest.raises(ValidationError):
        CountryProfile(country="X", company="Y", **dict(REQUIRED, life_expectancy=200))
    with pytest.raises(ValidationError):
        CountryProfile(country="X", company="Y", **dict(REQUIRED, claim_frequency=-0.1))


def test_missing_required_field_rejected() -> None:
    incomplete = dict(REQUIRED)
    del incomplete["gdp_per_capita"]
    with pytest.raises(ValidationError):
        CountryProfile(country="X", company="Y", **incomplete)
