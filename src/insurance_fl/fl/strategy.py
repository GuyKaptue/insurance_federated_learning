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

import logging
from collections.abc import Callable
from typing import Any

from flwr.common import EvaluateIns, FitIns
from flwr.server.strategy import FedAvg, FedProx, QFedAvg

from insurance_fl.fl.fairness import CLIENT_COUNTRY_MAP, stratified_sample
from insurance_fl.fl.privacy import RdpAccountantWrapper

log = logging.getLogger(__name__)

# Audit event type for conflicting/unknown identity claims (best-effort).
_CLIENT_ID_CONFLICT_EVENT = "client_identity_conflict"


def _audit_identity_conflict(audit_logger: Any, run_id: str | None, **details: Any) -> None:
    if audit_logger is None or run_id is None:
        return
    payload = {"run_id": run_id, **details}
    try:
        if hasattr(audit_logger, "log_event"):
            audit_logger.log_event(_CLIENT_ID_CONFLICT_EVENT, payload)
            return
    except Exception:
        pass
    try:
        log_fn = getattr(audit_logger, "log", None)
        if callable(log_fn):
            log_fn(_CLIENT_ID_CONFLICT_EVENT, **payload)
            return
    except Exception:
        pass
    try:
        getattr(audit_logger, "logger", None)  # keep audit logger referenced
        log_fn = getattr(getattr(audit_logger, "logger", None), "warning", None)
        if callable(log_fn):
            log_fn("audit_event=%s payload=%s", _CLIENT_ID_CONFLICT_EVENT, payload)
    except Exception:
        pass


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
        self.require_all_clients = bool(kwargs.pop("require_all_clients", False))
        self.secagg_enabled = bool(kwargs.pop("secagg_enabled", False))
        self.identity_learning_enabled = not self.secagg_enabled
        self._identity_stratification_logged = False
        self._mlflow_run_id: str | None = None
        self._round = 0
        self._per_task_metrics: dict[str, list[float]] = {}
        self._per_country_metrics: dict[str, dict[str, float]] = {}
        self._node_to_client_id: dict[int, str] = {}
        self._client_identity_conflict = False

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

    def _register_client_id_from_result(
        self,
        node_id: int,
        metrics: dict[str, Any],
        phase: str,
        observed_round: set[str] | None = None,
    ) -> str | None:
        raw = None
        if metrics is not None:
            try:
                raw = metrics.get("client_id")  # type: ignore[attr-defined]
            except Exception:
                raw = None
            if raw is None:
                try:
                    raw = metrics["client_id"]  # type: ignore[index]
                except Exception:
                    raw = None
        if raw is None or raw == "":
            self._client_identity_conflict = True
            _audit_identity_conflict(
                self.audit_logger,
                self.run_id,
                reason="missing_client_id",
                phase=phase,
                node_id=node_id,
                server_round=self._round,
            )
            raise ValueError(f"Missing client_id in {phase} metrics from node_id={node_id}")
        client_id = str(raw)
        from insurance_fl.fl.task import CLIENT_IDS

        if client_id not in CLIENT_IDS:
            self._client_identity_conflict = True
            _audit_identity_conflict(
                self.audit_logger,
                self.run_id,
                reason="unknown_client_id",
                phase=phase,
                node_id=node_id,
                client_id=client_id,
                server_round=self._round,
            )
            raise ValueError(f"Unknown client_id {client_id!r} from node_id={node_id}")
        known = self._node_to_client_id.get(node_id)
        if known is not None and known != client_id:
            self._client_identity_conflict = True
            _audit_identity_conflict(
                self.audit_logger,
                self.run_id,
                reason="conflicting_client_id",
                phase=phase,
                node_id=node_id,
                previous_client_id=known,
                client_id=client_id,
                server_round=self._round,
            )
            raise ValueError(
                f"Conflicting client_id for node_id={node_id}: {known!r} vs {client_id!r}"
            )
        if observed_round is not None and client_id in observed_round:
            self._client_identity_conflict = True
            _audit_identity_conflict(
                self.audit_logger,
                self.run_id,
                reason="duplicate_client_id",
                phase=phase,
                node_id=node_id,
                client_id=client_id,
                server_round=self._round,
            )
            raise ValueError(f"Duplicate client_id {client_id!r} in {phase} results")
        self._node_to_client_id[node_id] = client_id
        if observed_round is not None:
            observed_round.add(client_id)
        return client_id

    def configure_fit(self, server_round, parameters, client_manager):

        self._round = server_round
        self._client_identity_conflict = False

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

        # Stratified mode: select by client identity learned from replies.
        available_clients = list(client_manager.all().values())
        if not available_clients:
            return []

        from insurance_fl.fl.task import CLIENT_IDS

        num_to_sample = max(
            self.min_fit_clients,
            min(int(self.fraction_fit * len(available_clients)), len(available_clients)),
        )

        if not self.identity_learning_enabled:
            if not self._identity_stratification_logged:
                log.info(
                    "SecAgg+ mode: identity map inactive, selecting all nodes; "
                    "stratified selection is off"
                )
                self._identity_stratification_logged = True
            selected_proxies = available_clients
        else:
            unknown = [cp for cp in available_clients if cp.cid not in self._node_to_client_id]
            mapped_all = len(self._node_to_client_id) >= len(CLIENT_IDS) and set(
                self._node_to_client_id.values()
            ) == set(CLIENT_IDS)
            # Bootstrap: sample all available once to learn node_id->client_id mapping.
            if unknown or not mapped_all:
                selected_proxies = available_clients
            else:
                id_to_proxy = {
                    self._node_to_client_id[cp.cid]: cp
                    for cp in available_clients
                    if cp.cid in self._node_to_client_id
                }
                client_ids = list(id_to_proxy.keys())
                selected_ids, _uncovered = stratified_sample(
                    client_ids,
                    min(num_to_sample, len(client_ids)),
                    country_map=self.country_map,
                    min_per_country=self.min_clients_per_country,
                    seed=(
                        self.sampling_seed + server_round
                        if self.sampling_seed is not None
                        else None
                    ),
                )
                selected_proxies = [id_to_proxy[cid] for cid in selected_ids if cid in id_to_proxy]

        fit_ins = FitIns(parameters, {})
        if self.on_fit_config_fn is not None:
            fit_ins.config = self.on_fit_config_fn(server_round)
        if self.proximal_mu > 0:
            fit_ins.config["proximal_mu"] = self.proximal_mu

        return [(cp, fit_ins) for cp in selected_proxies]

    def aggregate_fit(self, server_round, results, failures):
        if self._client_identity_conflict:
            raise ValueError("Client identity conflict detected before aggregation")
        observed_ids: set[str] = set()
        if self.identity_learning_enabled:
            for client_proxy, fit_res in results:
                self._register_client_id_from_result(
                    client_proxy.cid, fit_res.metrics, "fit", observed_round=observed_ids
                )
            from insurance_fl.fl.task import CLIENT_IDS

            if self.require_all_clients and observed_ids != set(CLIENT_IDS):
                raise ValueError(
                    f"Expected fit identities {sorted(CLIENT_IDS)}, observed {sorted(observed_ids)}"
                )
            if observed_ids and not observed_ids.issubset(set(CLIENT_IDS)):
                raise ValueError(
                    f"Observed unknown fit identities: {sorted(observed_ids - set(CLIENT_IDS))}"
                )
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

    def configure_evaluate(self, server_round, parameters, client_manager):
        try:
            pairs = super().configure_evaluate(server_round, parameters, client_manager)
        except Exception:
            pairs = []
        available = list(client_manager.all().values())
        if not available:
            return pairs
        from insurance_fl.fl.task import CLIENT_IDS

        if not self.identity_learning_enabled:
            return pairs

        mapped = [cp for cp in available if cp.cid in self._node_to_client_id]
        mapped_ids = {self._node_to_client_id[cp.cid] for cp in mapped}
        if self.require_all_clients and mapped and mapped_ids == set(CLIENT_IDS):
            if pairs:
                evaluate_ins = pairs[0][1]
            else:
                evaluate_ins = EvaluateIns(parameters, {})
                if self.on_evaluate_config_fn is not None:
                    evaluate_ins.config = self.on_evaluate_config_fn(server_round)
            return [(cp, evaluate_ins) for cp in mapped]
        return pairs

    def aggregate_evaluate(self, server_round, results, failures):
        if self._client_identity_conflict:
            raise ValueError("Client identity conflict detected before evaluation aggregation")
        observed_ids: set[str] = set()
        if self.identity_learning_enabled:
            for client_proxy, eval_res in results:
                metrics = eval_res.metrics if hasattr(eval_res, "metrics") else {}
                self._register_client_id_from_result(
                    client_proxy.cid, metrics, "evaluate", observed_round=observed_ids
                )
            from insurance_fl.fl.task import CLIENT_IDS

            if self.require_all_clients and observed_ids != set(CLIENT_IDS):
                raise ValueError(
                    f"Expected evaluate identities {sorted(CLIENT_IDS)}, observed {sorted(observed_ids)}"
                )
            if observed_ids and not observed_ids.issubset(set(CLIENT_IDS)):
                raise ValueError(
                    f"Observed unknown evaluate identities: {sorted(observed_ids - set(CLIENT_IDS))}"
                )

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
        require_all_clients: bool = False,
        secagg_enabled: bool = False,
        audit_logger: Any = None,
        run_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.country_map = country_map or CLIENT_COUNTRY_MAP
        self.min_clients_per_country = min_clients_per_country
        self.sampling_seed = sampling_seed
        self._node_to_client_id: dict[int, str] = {}
        self._client_identity_conflict = False
        self.audit_logger = audit_logger
        self.run_id = run_id
        self._round = 0
        self.require_all_clients = bool(require_all_clients)
        self.secagg_enabled = bool(secagg_enabled)
        self.identity_learning_enabled = not self.secagg_enabled
        self._identity_stratification_logged = False

    def _register_client_id_from_result(
        self,
        node_id: int,
        metrics: dict[str, Any],
        phase: str,
        observed_round: set[str] | None = None,
    ) -> str | None:
        raw = None
        if metrics is not None:
            try:
                raw = metrics.get("client_id")  # type: ignore[attr-defined]
            except Exception:
                raw = None
            if raw is None:
                try:
                    raw = metrics["client_id"]  # type: ignore[index]
                except Exception:
                    raw = None
        if raw is None or raw == "":
            self._client_identity_conflict = True
            _audit_identity_conflict(
                self.audit_logger,
                getattr(self, "run_id", None),
                reason="missing_client_id",
                phase=phase,
                node_id=node_id,
                server_round=getattr(self, "_round", 0),
            )
            raise ValueError(f"Missing client_id in {phase} metrics from node_id={node_id}")
        client_id = str(raw)
        from insurance_fl.fl.task import CLIENT_IDS

        if client_id not in CLIENT_IDS:
            self._client_identity_conflict = True
            _audit_identity_conflict(
                self.audit_logger,
                getattr(self, "run_id", None),
                reason="unknown_client_id",
                phase=phase,
                node_id=node_id,
                client_id=client_id,
                server_round=getattr(self, "_round", 0),
            )
            raise ValueError(f"Unknown client_id {client_id!r} from node_id={node_id}")
        known = self._node_to_client_id.get(node_id)
        if known is not None and known != client_id:
            self._client_identity_conflict = True
            _audit_identity_conflict(
                self.audit_logger,
                getattr(self, "run_id", None),
                reason="conflicting_client_id",
                phase=phase,
                node_id=node_id,
                previous_client_id=known,
                client_id=client_id,
                server_round=getattr(self, "_round", 0),
            )
            raise ValueError(
                f"Conflicting client_id for node_id={node_id}: {known!r} vs {client_id!r}"
            )
        if observed_round is not None and client_id in observed_round:
            self._client_identity_conflict = True
            _audit_identity_conflict(
                self.audit_logger,
                getattr(self, "run_id", None),
                reason="duplicate_client_id",
                phase=phase,
                node_id=node_id,
                client_id=client_id,
                server_round=getattr(self, "_round", 0),
            )
            raise ValueError(f"Duplicate client_id {client_id!r} in {phase} results")
        self._node_to_client_id[node_id] = client_id
        if observed_round is not None:
            observed_round.add(client_id)
        return client_id

    def configure_fit(self, server_round, parameters, client_manager):

        self._round = server_round
        self._client_identity_conflict = False
        available_clients = list(client_manager.all().values())
        if not available_clients:
            return []

        from insurance_fl.fl.task import CLIENT_IDS

        num_to_sample = max(
            self.min_fit_clients,
            min(int(self.fraction_fit * len(available_clients)), len(available_clients)),
        )
        if not self.identity_learning_enabled:
            if not self._identity_stratification_logged:
                log.info(
                    "SecAgg+ mode: identity map inactive, selecting all nodes; stratified selection is off"
                )
                self._identity_stratification_logged = True
            selected_proxies = available_clients
        else:
            unknown = [cp for cp in available_clients if cp.cid not in self._node_to_client_id]
            mapped_all = len(self._node_to_client_id) >= len(CLIENT_IDS) and set(
                self._node_to_client_id.values()
            ) == set(CLIENT_IDS)
            if unknown or not mapped_all:
                selected_proxies = available_clients
            else:
                id_to_proxy = {
                    self._node_to_client_id[cp.cid]: cp
                    for cp in available_clients
                    if cp.cid in self._node_to_client_id
                }
                client_ids = list(id_to_proxy.keys())
                selected_ids, _ = stratified_sample(
                    client_ids,
                    min(num_to_sample, len(client_ids)),
                    country_map=self.country_map,
                    min_per_country=self.min_clients_per_country,
                    seed=(
                        self.sampling_seed + server_round
                        if self.sampling_seed is not None
                        else None
                    ),
                )
                selected_proxies = [id_to_proxy[cid] for cid in selected_ids if cid in id_to_proxy]

        fit_ins = FitIns(parameters, {})
        if self.on_fit_config_fn is not None:
            fit_ins.config = self.on_fit_config_fn(server_round)
        fit_ins.config["proximal_mu"] = self.proximal_mu
        return [(cp, fit_ins) for cp in selected_proxies]

    def aggregate_fit(self, server_round, results, failures):
        if self._client_identity_conflict:
            raise ValueError("Client identity conflict detected before aggregation")
        observed_ids: set[str] = set()
        if self.identity_learning_enabled:
            for client_proxy, fit_res in results:
                self._register_client_id_from_result(
                    client_proxy.cid, fit_res.metrics, "fit", observed_round=observed_ids
                )
            from insurance_fl.fl.task import CLIENT_IDS

            if self.require_all_clients and observed_ids != set(CLIENT_IDS):
                raise ValueError(
                    f"Expected fit identities {sorted(CLIENT_IDS)}, observed {sorted(observed_ids)}"
                )
        return super().aggregate_fit(server_round, results, failures)

    def configure_evaluate(self, server_round, parameters, client_manager):
        pairs = super().configure_evaluate(server_round, parameters, client_manager)
        available = list(client_manager.all().values())
        from insurance_fl.fl.task import CLIENT_IDS

        if not self.identity_learning_enabled:
            return pairs
        mapped = [cp for cp in available if cp.cid in self._node_to_client_id]
        if (
            self.require_all_clients
            and mapped
            and len({self._node_to_client_id[cp.cid] for cp in mapped}) == len(CLIENT_IDS)
        ):
            if pairs:
                evaluate_ins = pairs[0][1]
            else:
                evaluate_ins = EvaluateIns(parameters, {})
                if self.on_evaluate_config_fn is not None:
                    evaluate_ins.config = self.on_evaluate_config_fn(server_round)
            return [(cp, evaluate_ins) for cp in mapped]
        return pairs

    def aggregate_evaluate(self, server_round, results, failures):
        if self._client_identity_conflict:
            raise ValueError("Client identity conflict detected before evaluation aggregation")
        observed_ids: set[str] = set()
        if self.identity_learning_enabled:
            for client_proxy, eval_res in results:
                self._register_client_id_from_result(
                    client_proxy.cid, eval_res.metrics, "evaluate", observed_round=observed_ids
                )
            from insurance_fl.fl.task import CLIENT_IDS

            if self.require_all_clients and observed_ids != set(CLIENT_IDS):
                raise ValueError(
                    f"Expected evaluate identities {sorted(CLIENT_IDS)}, observed {sorted(observed_ids)}"
                )
        return super().aggregate_evaluate(server_round, results, failures)


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
    require_all_clients: bool = False,
    secagg_enabled: bool = False,
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
            require_all_clients=require_all_clients,
            secagg_enabled=secagg_enabled,
            audit_logger=audit_logger,
            run_id=run_id,
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
            require_all_clients=require_all_clients,
            secagg_enabled=secagg_enabled,
            **kwargs,
        )
    raise ValueError(
        f"Unknown strategy {strategy_name!r}; allowed: {sorted(_ALLOWED)}. "
        "HybridAggregation/QFedAvg arrive with Task 4."
    )


__all__ = ["HybridAggregation", "QFedAvgStrategy", "StratifiedFedProx", "build_strategy"]
