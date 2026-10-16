"""
Country profiles for the European Insurance Federated Learning Platform.

Defines validated, real-world statistics for each participating country.
Uses pydantic for schema enforcement — malformed profiles fail loudly.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class CountryProfile(BaseModel):
    """Real-world statistics for a single country's insurance market."""

    country: str
    company: str
    income_mean: float = Field(..., gt=0)
    income_std: float = Field(..., gt=0)
    savings_rate: float = Field(..., ge=0, le=1)
    savings_mean: float = Field(..., gt=0)
    claim_frequency: float = Field(..., ge=0, le=1)
    life_expectancy: int = Field(..., gt=0, le=120)
    avg_age: float = Field(..., gt=0, le=100)
    population: int = Field(..., gt=0)
    gdp_per_capita: float = Field(..., gt=0)
    car_premium: float = Field(..., gt=0)
    health_spending: float = Field(..., gt=0)
    risk_appetite: float = Field(..., ge=0, le=1)
    digital_adoption: float = Field(..., ge=0, le=1)
    market_share: float = Field(..., ge=0, le=1)
    currency: str = Field(..., min_length=1, max_length=10)
    regulatory_framework: str = Field(..., min_length=1)

    @field_validator("income_std")
    @classmethod
    def _income_std_less_than_mean(cls, v: float, info) -> float:
        if "income_mean" in info.data and v >= info.data["income_mean"]:
            raise ValueError("income_std must be less than income_mean")
        return v


# Validated country profiles — real-world statistics for each market
COUNTRY_PROFILES: dict[str, CountryProfile] = {
    "Germany": CountryProfile(
        country="Germany",
        company="Allianz",
        income_mean=50000.0,
        income_std=15000.0,
        savings_rate=0.107,
        savings_mean=3568.0,
        claim_frequency=0.044,
        life_expectancy=81,
        avg_age=46.5,
        population=83200000,
        gdp_per_capita=48000.0,
        car_premium=800.0,
        health_spending=6000.0,
        risk_appetite=0.7,
        digital_adoption=0.85,
        market_share=0.25,
        currency="EUR",
        regulatory_framework="BaFin",
    ),
    "France": CountryProfile(
        country="France",
        company="AXA",
        income_mean=45000.0,
        income_std=14000.0,
        savings_rate=0.184,
        savings_mean=2876.0,
        claim_frequency=0.055,
        life_expectancy=82,
        avg_age=42.3,
        population=67750000,
        gdp_per_capita=43000.0,
        car_premium=900.0,
        health_spending=5500.0,
        risk_appetite=0.65,
        digital_adoption=0.80,
        market_share=0.22,
        currency="EUR",
        regulatory_framework="ACPR",
    ),
    "Belgium": CountryProfile(
        country="Belgium",
        company="Ageas",
        income_mean=42000.0,
        income_std=13000.0,
        savings_rate=0.112,
        savings_mean=2600.0,
        claim_frequency=0.050,
        life_expectancy=81,
        avg_age=43.0,
        population=11500000,
        gdp_per_capita=46000.0,
        car_premium=750.0,
        health_spending=5200.0,
        risk_appetite=0.6,
        digital_adoption=0.75,
        market_share=0.15,
        currency="EUR",
        regulatory_framework="NBB/FSMA",
    ),
    "Switzerland": CountryProfile(
        country="Switzerland",
        company="Helvetia",
        income_mean=65000.0,
        income_std=20000.0,
        savings_rate=0.195,
        savings_mean=14768.0,
        claim_frequency=0.038,
        life_expectancy=83,
        avg_age=47.0,
        population=8700000,
        gdp_per_capita=82000.0,
        car_premium=1200.0,
        health_spending=7500.0,
        risk_appetite=0.55,
        digital_adoption=0.90,
        market_share=0.18,
        currency="CHF",
        regulatory_framework="FINMA",
    ),
}
