"""Flower ServerApp (Layer 1 / Task 0): FedAvg over the 4 insurance clients.

Baseline aggregation only. ``HybridAggregation`` (federate NN only,
FedProx/QFedAvg interplay) is Task 4 / Layer 4 — strategy knobs for it
(``proximal_mu`` etc.) live in ``config/federated_config.yaml`` already
but are intentionally NOT wired here yet.

flwr 1.29 API note: ``server_fn`` returns ``ServerAppComponents`` (the
``ServerAppObjects`` name exists only in newer Flower versions), and
``ServerApp`` takes ``server_fn`` as a keyword argument.
"""

from __future__ import annotations

from flwr.common import Context
from flwr.server import ServerApp, ServerConfig
from flwr.server.serverapp_components import ServerAppComponents

from insurance_fl.fl.strategy import build_strategy


def server_fn(context: Context) -> ServerAppComponents:
    """Configure strategy + round budget from ``[tool.flwr.app.config]``."""
    rc = context.run_config
    strategy = build_strategy(
        strategy_name=str(rc.get("strategy-type", "fedavg")),
        fraction_fit=float(rc["fraction-fit"]),
        fraction_evaluate=float(rc["fraction-evaluate"]),
        min_fit_clients=int(rc["min-fit-clients"]),
        min_evaluate_clients=int(rc["min-evaluate-clients"]),
        min_available_clients=int(rc["min-available-clients"]),
    )
    config = ServerConfig(num_rounds=int(rc["num-server-rounds"]))
    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)
