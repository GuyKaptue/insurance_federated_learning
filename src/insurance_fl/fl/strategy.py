"""FL strategy factory (Layer 1 baseline; grown by Tasks 4-5 / Layer 4).

Task 0 provides only the FedAvg baseline wired through
``flwr.server.strategy.FedAvg``. ``HybridAggregation`` (federate NN
layers only, per-task booster trees kept local), FedProx proximal term
wiring, QFedAvg and the FedProx/QFedAvg mutual-exclusion rule (Option B)
arrive with Task 4. This module exists now so ``server_app.py`` has a
single, stable seam for those changes.
"""

from __future__ import annotations

from flwr.server.strategy import FedAvg

_ALLOWED = {"fedavg"}


def build_strategy(
    strategy_name: str = "fedavg",
    *,
    fraction_fit: float = 1.0,
    fraction_evaluate: float = 0.5,
    min_fit_clients: int = 2,
    min_evaluate_clients: int = 2,
    min_available_clients: int = 4,
    **kwargs: object,
) -> FedAvg:
    """Return the configured Flower strategy (Task 0: FedAvg only).

    Raises ``ValueError`` for anything else — no silent fallbacks, so the
    Task 4 extension must consciously add its strategies here.
    """
    if strategy_name != "fedavg":
        raise ValueError(
            f"Unknown strategy {strategy_name!r}; allowed: {sorted(_ALLOWED)}. "
            "HybridAggregation/QFedAvg/FedProx arrive with Task 4."
        )
    return FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
        **kwargs,  # type: ignore[arg-type]
    )
