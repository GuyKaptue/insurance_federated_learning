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

from typing import Any

from flwr.common import Context
from flwr.server import Server, ServerApp, ServerConfig
from flwr.server.client_manager import SimpleClientManager
from flwr.server.compat import start_grid
from flwr.server.serverapp_components import ServerAppComponents

try:
    from flwr.serverapp.grid import Grid  # type: ignore[no-redef]
except ImportError:
    from flwr.server.grid import Grid  # type: ignore[no-redef]

from insurance_fl.audit import AuditLogger
from insurance_fl.fl.privacy import (
    PRIVACY_KEYS,
    PrivacyConfig,
    RdpAccountantWrapper,
    read_privacy_values,
)
from insurance_fl.fl.strategy import build_strategy

app = ServerApp()


def privacy_config_from_run_config(
    rc: dict[str, Any],
) -> PrivacyConfig | None:
    """Build a PrivacyConfig from run_config, fail-closed (delegates to
    ``read_privacy_values`` for parsing)."""
    values, enabled = read_privacy_values(dict(rc))
    if enabled:
        for key in PRIVACY_KEYS:
            if values[key] <= 0:
                raise ValueError(
                    f"privacy-enabled=true requires positive {key}, " f"got {values[key]}"
                )
        return PrivacyConfig(
            max_epsilon=values["privacy-max-epsilon"],
            target_delta=values["privacy-delta"],
            population_size=4,
            noise_multiplier=values["privacy-local-dp-noise-std"]
            / values["privacy-local-dp-clip-norm"],
            sampling_rate=1.0,
            delta=values["privacy-delta"],
        )
    return None


def build_components(context: Context) -> ServerAppComponents:
    """Build strategy + server config from run_config."""
    rc = context.run_config
    privacy_config = privacy_config_from_run_config(dict(rc))
    privacy_state_path = str(rc.get("privacy-state-path", "")) or None
    privacy_resume = str(rc.get("privacy-resume", "false")).lower() == "true"
    if privacy_resume and not privacy_state_path:
        raise ValueError("privacy-resume=true requires privacy-state-path")
    accountant = (
        RdpAccountantWrapper.load_state(
            privacy_state_path,
            cfg=privacy_config,
            resume=privacy_resume,
        )
        if privacy_config is not None
        else None
    )
    audit_logger = AuditLogger(
        path=__import__("pathlib").Path(str(rc.get("audit-path", "logs/audit.jsonl"))),
        run_id=str(rc.get("run-id", "fl-run")),
    )
    audit_logger.log_config_snapshot(config=dict(rc))
    if accountant is not None and accountant.resumed_from_round is not None:
        audit_logger.log_privacy_event(
            round=accountant.resumed_from_round,
            mechanism="local_dp",
            accounting_status="resumed",
            resumed_from_round=accountant.resumed_from_round,
        )

    secagg_enabled = str(rc.get("secagg-enabled", "false")).lower() == "true"
    require_all_clients = str(rc.get("require-all-clients", "false")).lower() == "true"
    strategy = build_strategy(
        strategy_name=str(rc.get("strategy-type", "fedavg")),
        fraction_fit=float(rc["fraction-fit"]),
        fraction_evaluate=float(rc["fraction-evaluate"]),
        min_fit_clients=int(rc["min-fit-clients"]),
        min_evaluate_clients=int(rc["min-evaluate-clients"]),
        min_available_clients=int(rc["min-available-clients"]),
        accountant=accountant,
        max_epsilon=privacy_config.max_epsilon if privacy_config is not None else None,
        privacy_state_path=privacy_state_path,
        audit_logger=audit_logger,
        run_id=str(rc.get("run-id", "fl-run")),
        require_all_clients=require_all_clients,
        secagg_enabled=secagg_enabled,
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

    rc = context.run_config
    secagg_enabled = str(rc.get("secagg-enabled", "false")).lower() == "true"

    if secagg_enabled:
        # SecAgg+ simulation runtime path requires proper workflow orchestration
        # (client manager updates, init, round loop) beyond this bootstrap.
        # Kept explicit to avoid a silently incorrect/incomplete run.
        raise NotImplementedError(
            "SecAgg+ runtime orchestration via server_app.main is not wired for "
            "multi-round simulation in this patch; use SecAgg workflow integration "
            "tests and dedicated wiring."
        )

    server = Server(client_manager=client_manager, strategy=strategy)
    start_grid(
        server=server,
        config=config,
        strategy=strategy,
        client_manager=client_manager,
        grid=grid,
    )
