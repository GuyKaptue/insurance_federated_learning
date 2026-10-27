"""Flower ServerApp (Layer 1 / Task 0): FedAvg over the 4 insurance clients.

Baseline aggregation only. When Secure Aggregation is enabled, the
``SecAggPlusWorkflow`` protects client updates via cryptographic masking
before they reach the strategy's ``aggregate_fit``.

flwr 1.31 API note: ``ServerApp`` supports two execution modes:
- Compatibility mode (``server_fn`` → ``ServerAppComponents`` → ``start_grid``)
- New execution mode (``@app.main()`` decorator receiving ``(grid, context)``)

``SecAggPlusWorkflow`` requires the new execution mode because it needs a
``LegacyContext`` and direct ``Grid`` access for the key-exchange /
masking / unmasking protocol stages.

SecAgg+ design parameters (flwr 1.31 defaults where applicable):

- ``num_shares=4`` — one share per expected client (4 participating clients).
- ``reconstruction_threshold=3`` — minimum shares to reconstruct a dropped
  client's key; tolerates up to 1 dropout with 4 clients.
- ``max_weight=1000.0`` — maximum weight assigned to any single client's
  update during weighted averaging.
- ``clipping_range=8.0`` — parameters are clipped to [-8.0, 8.0] before
  quantization.
- ``quantization_range=2**22`` — maps each parameter to an integer in
  [0, 2^22 - 1] for cryptographic operations.

These parameters are configured via ``run_config`` keys prefixed with
``secagg-``. Secure Aggregation alone is NOT a differential privacy
guarantee; it is orthogonal to Local DP and RDP accounting.
"""

from __future__ import annotations

from flwr.common import Context
from flwr.server import Server, ServerApp, ServerConfig
from flwr.server.client_manager import SimpleClientManager
from flwr.server.compat import start_grid
from flwr.server.compat.legacy_context import LegacyContext
from flwr.server.serverapp_components import ServerAppComponents
from flwr.server.workflow import SecAggPlusWorkflow

try:
    from flwr.serverapp.grid import Grid  # type: ignore[no-redef]
except ImportError:
    from flwr.server.grid import Grid  # type: ignore[no-redef]

from insurance_fl.audit import AuditLogger
from insurance_fl.fl.privacy import RdpAccountantWrapper, build_privacy_config
from insurance_fl.fl.strategy import build_strategy

app = ServerApp()


def build_components(context: Context) -> ServerAppComponents:
    """Build strategy + server config from run_config."""
    rc = context.run_config
    privacy_config = build_privacy_config(dict(rc))
    accountant = RdpAccountantWrapper(privacy_config)
    audit_logger = AuditLogger(
        path=__import__("pathlib").Path(str(rc.get("audit-path", "logs/audit.jsonl"))),
        run_id=str(rc.get("run-id", "fl-run")),
    )
    audit_logger.log_config_snapshot(config=dict(rc))

    strategy = build_strategy(
        strategy_name=str(rc.get("strategy-type", "fedavg")),
        fraction_fit=float(rc["fraction-fit"]),
        fraction_evaluate=float(rc["fraction-evaluate"]),
        min_fit_clients=int(rc["min-fit-clients"]),
        min_evaluate_clients=int(rc["min-evaluate-clients"]),
        min_available_clients=int(rc["min-available-clients"]),
        accountant=accountant,
        audit_logger=audit_logger,
        run_id=str(rc.get("run-id", "fl-run")),
    )
    config = ServerConfig(num_rounds=int(rc["num-server-rounds"]))
    return ServerAppComponents(strategy=strategy, config=config)


def server_fn(context: Context) -> ServerAppComponents:
    """Compatibility-mode entry point (used by tests and legacy runs)."""
    return build_components(context)


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Run federated training, optionally with Secure Aggregation.

    When ``secagg-enabled`` is true in ``run_config``, the
    ``SecAggPlusWorkflow`` protects client updates. Otherwise, the
    baseline ``start_grid`` / FedAvg path is used.
    """
    components = build_components(context)
    strategy = components.strategy
    config = components.config

    client_manager = SimpleClientManager()
    legacy_ctx = LegacyContext(
        context, config=config, strategy=strategy, client_manager=client_manager
    )

    rc = context.run_config
    secagg_enabled = str(rc.get("secagg-enabled", "false")).lower() == "true"

    if secagg_enabled:
        workflow = SecAggPlusWorkflow(
            num_shares=int(rc.get("secagg-num-shares", 4)),
            reconstruction_threshold=int(rc.get("secagg-reconstruction-threshold", 3)),
            max_weight=float(rc.get("secagg-max-weight", 1000.0)),
            clipping_range=float(rc.get("secagg-clipping-range", 8.0)),
            quantization_range=int(rc.get("secagg-quantization-range", 2**22)),
        )
        workflow(grid, legacy_ctx)
    else:
        server = Server(client_manager=client_manager, strategy=strategy)
        start_grid(
            server=server,
            config=config,
            strategy=strategy,
            client_manager=client_manager,
            grid=grid,
        )
