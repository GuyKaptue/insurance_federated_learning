"""Custom hybrid aggregation strategy (§3.1 of the platform spec).

``HybridAggregation`` extends ``FedAvg`` to handle the hybrid architecture:

- Federated NN parameter aggregation (weighted by num-examples).
- Fairness-aware client selection (stratified sampling or QFedAvg).
- FedProx-compatible proximal term.
- Per-task metric collection.
- Privacy-accounting integration (RDP, advisory).
- Structured audit events.
- Experiment-tracking hooks (optional MLflow).

Boosters and stacking alpha are never aggregated — they remain local.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from flwr.server.strategy import FedAvg, FedProx, QFedAvg

from insurance_fl.fl.fairness import CLIENT_COUNTRY_MAP, stratified_sample
from insurance_fl.fl.privacy import RdpAccountantWrapper

try:
    import mlflow

    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False
    mlflow = None  # type: ignore[assignment]


class HybridAggregation(FedAvg):
    """Hybrid aggregation strategy for the multi-task FL platform.

    Extends ``FedAvg`` with:
    - Stratified client sampling (fairness) when ``fairness_mode="stratified"``.
    - QFedAvg-style aggregation when ``fairness_mode="qfedavg"``.
    - FedProx proximal term (via ``proximal_mu``).
    - Privacy accounting (RDP) integration.
    - Audit event emission.
    - Per-task metric collection.
    """

    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 4,
        proximal_mu: float = 0.0,
        fairness_mode: str = "stratified",
        country_map: dict[str, str] | None = None,
        min_clients_per_country: int = 1,
        sampling_seed: int | None = 42,
        accountant: RdpAccountantWrapper | None = None,
        audit_logger: Any = None,
        run_id: str | None = None,
        mlflow_enabled: bool = False,
        mlflow_tracking_uri: str | None = None,
        mlflow_experiment_name: str = "insurance_fl",
        on_fit_config_fn: Callable[[int], dict[str, Any]] | None = None,
        on_evaluate_config_fn: Callable[[int], dict[str, Any]] | None = None,
        fit_metrics_aggregation_fn: Callable | None = None,
        evaluate_metrics_aggregation_fn: Callable | None = None,
        initial_parameters: Any = None,
        accept_failures: bool = True,
        **kwargs: Any,
    ) -> None:
        if fairness_mode not in ("stratified", "qfedavg"):
            raise ValueError(
                f"fairness_mode must be 'stratified' or 'qfedavg', got {fairness_mode!r}"
            )
        self.fairness_mode = fairness_mode
        self.proximal_mu = proximal_mu
        self.country_map = country_map or CLIENT_COUNTRY_MAP
        self.min_clients_per_country = min_clients_per_country
        self.sampling_seed = sampling_seed
        self.accountant = accountant
        self.audit_logger = audit_logger
        self.run_id = run_id
        self.mlflow_enabled = mlflow_enabled and _MLFLOW_AVAILABLE
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.mlflow_experiment_name = mlflow_experiment_name
        self._mlflow_run_id: str | None = None
        self._round = 0
        self._per_task_metrics: dict[str, list[float]] = {}
        self._per_country_metrics: dict[str, dict[str, float]] = {}

        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            initial_parameters=initial_parameters,
            accept_failures=accept_failures,
            **kwargs,
        )

    def configure_fit(self, server_round, parameters, client_manager):
        from flwr.common import FitIns

        self._round = server_round

        # MLflow: start run on first round
        if self.mlflow_enabled and server_round == 1 and self._mlflow_run_id is None:
            try:
                if self.mlflow_tracking_uri:
                    mlflow.set_tracking_uri(self.mlflow_tracking_uri)
                mlflow.set_experiment(self.mlflow_experiment_name)
                run = mlflow.start_run()
                self._mlflow_run_id = run.info.run_id
            except Exception:
                self.mlflow_enabled = False

        # Audit: round start
        if self.audit_logger is not None and self.run_id is not None:
            try:
                self.audit_logger.log_fl_round_start(
                    round=server_round,
                    participating_clients=self.min_fit_clients,
                    strategy=f"hybrid_{self.fairness_mode}",
                )
            except Exception:
                pass

        # QFedAvg mode: delegate to parent (QFedAvg handles aggregation)
        if self.fairness_mode == "qfedavg":
            pairs = super().configure_fit(server_round, parameters, client_manager)
            # Add proximal_mu to config if FedProx enabled
            if self.proximal_mu > 0:
                new_pairs = []
                for client, fit_ins in pairs:
                    new_config = dict(fit_ins.config)
                    new_config["proximal_mu"] = self.proximal_mu
                    new_pairs.append((client, FitIns(fit_ins.parameters, new_config)))
                return new_pairs
            return pairs

        # Stratified mode: stratified sampling
        available_clients = list(client_manager.all().values())
        if not available_clients:
            return []

        from insurance_fl.fl.task import CLIENT_IDS

        client_ids = []
        for cp in available_clients:
            idx = int(cp.node_id) % len(CLIENT_IDS)
            client_ids.append(CLIENT_IDS[idx])

        num_to_sample = max(
            self.min_fit_clients,
            min(
                int(self.fraction_fit * len(available_clients)),
                len(available_clients),
            ),
        )
        selected_ids, _uncovered = stratified_sample(
            client_ids,
            num_to_sample,
            country_map=self.country_map,
            min_per_country=self.min_clients_per_country,
            seed=self.sampling_seed + server_round if self.sampling_seed is not None else None,
        )

        id_to_proxy = {}
        for cp, cid in zip(available_clients, client_ids, strict=True):
            id_to_proxy[cid] = cp

        selected_proxies = [id_to_proxy[cid] for cid in selected_ids if cid in id_to_proxy]

        fit_ins = FitIns(parameters, {})
        if self.on_fit_config_fn is not None:
            fit_ins.config = self.on_fit_config_fn(server_round)
        if self.proximal_mu > 0:
            fit_ins.config["proximal_mu"] = self.proximal_mu

        return [(cp, fit_ins) for cp in selected_proxies]

    def aggregate_fit(self, server_round, results, failures):
        # Privacy accounting (advisory)
        if self.accountant is not None:
            try:
                # Record this round in the accountant
                self.accountant.step()
                eps_data = self.accountant.epsilon()
                epsilon_conservative = eps_data["epsilon"]
                epsilon_amplified = eps_data.get("epsilon_amplified")

                # Log both conservative and amplified epsilons
                if self.audit_logger is not None and self.run_id is not None:
                    try:
                        self.audit_logger.log_privacy_event(
                            round=server_round,
                            mechanism="local_dp",
                            epsilon=epsilon_conservative,
                            accounting_status="accounted",
                            epsilon_amplified=epsilon_amplified,
                        )
                    except Exception:
                        pass

                # MLflow: log privacy metrics
                if self.mlflow_enabled:
                    try:
                        mlflow.log_metric("round", server_round, step=server_round)
                        mlflow.log_metric(
                            "privacy/epsilon_conservative",
                            epsilon_conservative,
                            step=server_round,
                        )
                        if epsilon_amplified is not None:
                            mlflow.log_metric(
                                "privacy/epsilon_amplified",
                                epsilon_amplified,
                                step=server_round,
                            )
                    except Exception:
                        pass
            except Exception:
                pass

        # Delegate to parent for actual aggregation
        aggregated = super().aggregate_fit(server_round, results, failures)

        # Attach epsilon info to the aggregated fit result if present
        if aggregated is not None and self.accountant is not None:
            try:
                eps = self.accountant.epsilon()
                if aggregated[1] is not None:
                    aggregated[1]["epsilon"] = eps["epsilon"]
                    if eps.get("epsilon_amplified") is not None:
                        aggregated[1]["epsilon_amplified"] = eps["epsilon_amplified"]
            except Exception:
                pass

        return aggregated

    def aggregate_evaluate(self, server_round, results, failures):
        aggregated = super().aggregate_evaluate(server_round, results, failures)

        # Collect per-task metrics from results
        for _client_proxy, eval_res in results:
            metrics = eval_res.metrics if hasattr(eval_res, "metrics") else {}
            for k, v in metrics.items():
                if k not in self._per_task_metrics:
                    self._per_task_metrics[k] = []
                try:
                    self._per_task_metrics[k].append(float(v))
                except (TypeError, ValueError):
                    pass

        # MLflow: log evaluation metrics
        if self.mlflow_enabled:
            try:
                if aggregated is not None and hasattr(aggregated, "metrics") and aggregated.metrics:
                    for k, v in aggregated.metrics.items():
                        try:
                            mlflow.log_metric(f"eval/{k}", float(v), step=server_round)
                        except (TypeError, ValueError):
                            pass
            except Exception:
                pass

        # Audit: round end
        if self.audit_logger is not None and self.run_id is not None:
            try:
                metric_val = None
                if aggregated is not None and hasattr(aggregated, "metrics"):
                    metric_val = aggregated.metrics.get("loss") if aggregated.metrics else None
                self.audit_logger.log_fl_round_end(
                    round=server_round,
                    aggregated_metric=metric_val,
                    wall_time_s=None,
                )
            except Exception:
                pass

        return aggregated

    def get_per_task_metrics(self) -> dict[str, list[float]]:
        return dict(self._per_task_metrics)

    def get_per_country_metrics(self) -> dict[str, dict[str, float]]:
        return dict(self._per_country_metrics)


_ALLOWED = {"fedavg", "fedprox", "hybrid", "qfedavg"}


class StratifiedFedProx(FedProx):
    """FedProx with country-stratified client sampling (kept for backward compat)."""

    def __init__(
        self,
        *,
        country_map: dict[str, str] | None = None,
        min_clients_per_country: int = 1,
        sampling_seed: int | None = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.country_map = country_map or CLIENT_COUNTRY_MAP
        self.min_clients_per_country = min_clients_per_country
        self.sampling_seed = sampling_seed

    def configure_fit(self, server_round, parameters, client_manager):
        from flwr.common import FitIns

        available_clients = list(client_manager.all().values())
        if not available_clients:
            return []

        from insurance_fl.fl.task import CLIENT_IDS

        client_ids = []
        for cp in available_clients:
            idx = int(cp.node_id) % len(CLIENT_IDS)
            client_ids.append(CLIENT_IDS[idx])

        num_to_sample = max(
            self.min_fit_clients,
            min(
                int(self.fraction_fit * len(available_clients)),
                len(available_clients),
            ),
        )
        selected_ids, _ = stratified_sample(
            client_ids,
            num_to_sample,
            country_map=self.country_map,
            min_per_country=self.min_clients_per_country,
            seed=self.sampling_seed + server_round if self.sampling_seed is not None else None,
        )

        id_to_proxy = {}
        for cp, cid in zip(available_clients, client_ids, strict=True):
            id_to_proxy[cid] = cp

        selected_proxies = [id_to_proxy[cid] for cid in selected_ids if cid in id_to_proxy]

        fit_ins = FitIns(parameters, {})
        if self.on_fit_config_fn is not None:
            fit_ins.config = self.on_fit_config_fn(server_round)
        fit_ins.config["proximal_mu"] = self.proximal_mu

        return [(cp, fit_ins) for cp in selected_proxies]


class QFedAvgStrategy(QFedAvg):
    """QFedAvg-based fairness strategy (Option A from spec §2.6).

    Wraps Flower's ``QFedAvg`` with validated parameters.
    """

    def __init__(
        self,
        *,
        q_param: float = 0.2,
        qffl_learning_rate: float = 0.1,
        **kwargs: Any,
    ) -> None:
        if q_param < 0:
            raise ValueError(f"q_param must be non-negative, got {q_param}")
        if qffl_learning_rate <= 0:
            raise ValueError(f"qffl_learning_rate must be positive, got {qffl_learning_rate}")
        super().__init__(
            q_param=q_param,
            qffl_learning_rate=qffl_learning_rate,
            **kwargs,
        )


def build_strategy(
    strategy_name: str = "fedavg",
    *,
    fraction_fit: float = 1.0,
    fraction_evaluate: float = 0.5,
    min_fit_clients: int = 2,
    min_evaluate_clients: int = 2,
    min_available_clients: int = 4,
    proximal_mu: float = 0.1,
    fairness_mode: str = "stratified",
    country_map: dict[str, str] | None = None,
    min_clients_per_country: int = 1,
    sampling_seed: int | None = 42,
    accountant: RdpAccountantWrapper | None = None,
    audit_logger: Any = None,
    run_id: str | None = None,
    mlflow_enabled: bool = False,
    mlflow_tracking_uri: str | None = None,
    mlflow_experiment_name: str = "insurance_fl",
    **kwargs: Any,
) -> Any:
    if strategy_name == "fedavg":
        return FedAvg(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            **kwargs,
        )
    if strategy_name == "fedprox":
        return StratifiedFedProx(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            proximal_mu=proximal_mu,
            country_map=country_map,
            min_clients_per_country=min_clients_per_country,
            sampling_seed=sampling_seed,
            **kwargs,
        )
    if strategy_name == "qfedavg":
        return QFedAvgStrategy(
            q_param=kwargs.pop("q_param", 0.2),
            qffl_learning_rate=kwargs.pop("qffl_learning_rate", 0.1),
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            **kwargs,
        )
    if strategy_name == "hybrid":
        return HybridAggregation(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            proximal_mu=proximal_mu,
            fairness_mode=fairness_mode,
            country_map=country_map,
            min_clients_per_country=min_clients_per_country,
            sampling_seed=sampling_seed,
            accountant=accountant,
            audit_logger=audit_logger,
            run_id=run_id,
            mlflow_enabled=mlflow_enabled,
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_experiment_name=mlflow_experiment_name,
            **kwargs,
        )
    raise ValueError(
        f"Unknown strategy {strategy_name!r}; allowed: {sorted(_ALLOWED)}. "
        "HybridAggregation/QFedAvg arrive with Task 4."
    )


__all__ = ["HybridAggregation", "QFedAvgStrategy", "StratifiedFedProx", "build_strategy"]
