"""Country-stratified client sampling and fairness reporting.

Provides a testable component for stratified client sampling by country,
guaranteeing at least one selected client per eligible country (subject
to availability and budget constraints), and computing per-country
fairness disparity.

Fairness disparity:
    D = max_c(m_c) - min_c(m_c)
where m_c is the chosen country-level metric. Higher or lower values
depend on the metric: for accuracy/AUC, higher is better (smaller D is
better). For loss, lower is better (smaller D is better). The threshold
of 5% of the mean is a configurable hypothesis, not a regulatory standard.
When the mean is zero or close to zero, division-based thresholds are
not defined; the raw disparity is reported instead.
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

CLIENT_COUNTRY_MAP: dict[str, str] = {
    "germany_allianz": "Germany",
    "france_axa": "France",
    "belgium_ageas": "Belgium",
    "switzerland_helvetia": "Switzerland",
}


@dataclass
class FairnessConfig:
    """Fairness configuration.

    Attributes:
        strategy: Fairness strategy identifier.
        min_clients_per_country: Minimum number of clients to select per
            country (when available).
        disparity_threshold_pct: Configurable hypothesis threshold for
            disparity as a fraction of the mean (default 0.05 = 5%).
    """

    strategy: str = "fedprox_stratified"
    min_clients_per_country: int = 1
    disparity_threshold_pct: float = 0.05


def group_clients_by_country(
    client_ids: list[str],
    country_map: dict[str, str] | None = None,
) -> dict[str, list[str]]:
    """Group client IDs by their country.

    Returns a dict mapping country -> list of client IDs.
    """
    cmap = country_map or CLIENT_COUNTRY_MAP
    groups: dict[str, list[str]] = {}
    for cid in client_ids:
        country = cmap.get(cid, cid)
        groups.setdefault(country, []).append(cid)
    return groups


def stratified_sample(
    client_ids: list[str],
    num_clients: int,
    *,
    country_map: dict[str, str] | None = None,
    min_per_country: int = 1,
    seed: int | None = None,
    available: set[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Select a stratified sample of clients by country.

    Guarantees at least ``min_per_country`` selected client per eligible
    country when sufficient clients and protocol constraints permit.
    Avoids duplicate selections. Handles empty country groups, unavailable
    clients, and insufficient sampling budgets explicitly.

    Parameters:
        client_ids: All eligible client IDs.
        num_clients: Total number of clients to select.
        country_map: Mapping from client ID to country.
        min_per_country: Minimum clients per country.
        seed: Random seed for reproducibility.
        available: Set of available client IDs. If None, all are available.

    Returns:
        (selected_client_ids, uncovered_countries) where
        uncovered_countries lists countries that could not be represented.
    """
    if available is None:
        available = set(client_ids)
    avail_clients = [c for c in client_ids if c in available]

    groups = group_clients_by_country(avail_clients, country_map)
    countries = sorted(groups.keys())

    selected: list[str] = []
    selected_set: set[str] = set()

    # Phase 1: guarantee at least min_per_country per country
    rng = random.Random(seed) if seed is not None else random.Random()
    for country in countries:
        candidates = [c for c in groups[country] if c not in selected_set]
        n_to_pick = min(min_per_country, len(candidates))
        # Shuffle candidates deterministically
        rng.shuffle(candidates)
        for c in candidates[:n_to_pick]:
            if len(selected) < num_clients:
                selected.append(c)
                selected_set.add(c)

    # Phase 2: fill remaining budget from all countries round-robin
    remaining = num_clients - len(selected)
    if remaining > 0:
        all_remaining = []
        for country in countries:
            for c in groups[country]:
                if c not in selected_set:
                    all_remaining.append(c)
        rng.shuffle(all_remaining)
        for c in all_remaining[:remaining]:
            selected.append(c)
            selected_set.add(c)

    # Determine uncovered countries (from full client list)
    all_groups = group_clients_by_country(client_ids, country_map)
    uncovered = []
    for country in all_groups:
        if not any(c in selected_set for c in all_groups[country]):
            uncovered.append(country)

    return selected, uncovered


def compute_disparity(
    country_metrics: Mapping[str, float | None],
    *,
    threshold_pct: float = 0.05,
) -> dict[str, Any]:
    """Compute fairness disparity D = max(m_c) - min(m_c).

    Returns a dict with:
        - disparity: D value (or None if no metrics)
        - max_country: country with highest metric
        - min_country: country with lowest metric
        - mean: mean of country metrics
        - threshold: threshold_pct * mean (or None if mean is zero)
        - meets_threshold: whether disparity < threshold (None if undefined)
        - missing_countries: countries with no metric
    """
    if not country_metrics:
        return {
            "disparity": None,
            "max_country": None,
            "min_country": None,
            "mean": None,
            "threshold": None,
            "meets_threshold": None,
            "missing_countries": [],
        }

    values = {k: v for k, v in country_metrics.items() if v is not None}
    missing = [k for k, v in country_metrics.items() if v is None]

    if not values:
        return {
            "disparity": None,
            "max_country": None,
            "min_country": None,
            "mean": None,
            "threshold": None,
            "meets_threshold": None,
            "missing_countries": missing,
        }

    max_country = max(values, key=lambda k: values[k])
    min_country = min(values, key=lambda k: values[k])
    disparity = values[max_country] - values[min_country]
    mean = sum(values.values()) / len(values)

    if abs(mean) < 1e-12:
        threshold = None
        meets = None
    else:
        threshold = threshold_pct * abs(mean)
        meets = disparity < threshold

    return {
        "disparity": disparity,
        "max_country": max_country,
        "min_country": min_country,
        "mean": mean,
        "threshold": threshold,
        "meets_threshold": meets,
        "missing_countries": missing,
    }


__all__ = [
    "CLIENT_COUNTRY_MAP",
    "FairnessConfig",
    "compute_disparity",
    "group_clients_by_country",
    "stratified_sample",
]
