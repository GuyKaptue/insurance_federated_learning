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

import numpy as np
from flwr.common import EvaluateIns, FitIns, Parameters, parameters_to_ndarrays
from flwr.server.client_manager import ClientManager
from flwr.server.strategy import FedAvg, FedProx, QFedAvg, Strategy

from insurance_fl.fl.fairness import CLIENT_COUNTRY_MAP, stratified_sample
from insurance_fl.fl.privacy import RdpAccountantWrapper, account_round

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


def _validate_fit_updates(results: list) -> None:
    """Reject malformed or non-finite client tensors before strategy aggregation."""
    expected_shapes: list[tuple[int, ...]] | None = None
    for client_proxy, fit_res in results:
        if fit_res.num_examples <= 0:
            raise ValueError(
                f"Invalid num_examples={fit_res.num_examples} from node_id={client_proxy.cid}"
            )
        try:
            arrays = parameters_to_ndarrays(fit_res.parameters)
        except Exception as exc:
            raise ValueError(f"Malformed parameters from node_id={client_proxy.cid}") from exc
        shapes = [np.asarray(array).shape for array in arrays]
        if expected_shapes is None:
            expected_shapes = shapes
        elif shapes != expected_shapes:
            raise ValueError(
                f"Mismatched parameter shapes from node_id={client_proxy.cid}: "
                f"expected {expected_shapes}, received {shapes}"
            )
        for array in arrays:
            value = np.asarray(array)
            if not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
                raise ValueError(
                    f"Client update contains non-finite or non-numeric values "
                    f"from node_id={client_proxy.cid}"
                )


class _ClientIdentityMixin:
    """Shared validation for client identities learned from Flower replies."""

    _node_to_client_id: dict[int, str]
    _client_identity_conflict: bool

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
                raw = metrics.get("client_id")
            except Exception:
                raw = None
            if raw is None:
                try:
                    raw = metrics["client_id"]
                except Exception:
                    raw = None
        server_round = getattr(self, "_round", 0)
        audit_logger = getattr(self, "audit_logger", None)
        run_id = getattr(self, "run_id", None)
        if raw is None or raw == "":
            self._client_identity_conflict = True
            _audit_identity_conflict(
                audit_logger,
                run_id,
                reason="missing_client_id",
                phase=phase,
                node_id=node_id,
                server_round=server_round,
            )
            raise ValueError(f"Missing client_id in {phase} metrics from node_id={node_id}")
        client_id = str(raw)
        from insurance_fl.fl.task import CLIENT_IDS

        if client_id not in CLIENT_IDS:
            self._client_identity_conflict = True
            _audit_identity_conflict(
                audit_logger,
                run_id,
                reason="unknown_client_id",
                phase=phase,
                node_id=node_id,
                client_id=client_id,
                server_round=server_round,
            )
            raise ValueError(f"Unknown client_id {client_id!r} from node_id={node_id}")
        known = self._node_to_client_id.get(node_id)
        if known is not None and known != client_id:
            self._client_identity_conflict = True
            _audit_identity_conflict(
                audit_logger,
                run_id,
                reason="conflicting_client_id",
                phase=phase,
                node_id=node_id,
                previous_client_id=known,
                client_id=client_id,
                server_round=server_round,
            )
            raise ValueError(
                f"Conflicting client_id for node_id={node_id}: {known!r} vs {client_id!r}"
            )
        if observed_round is not None and client_id in observed_round:
            self._client_identity_conflict = True
            _audit_identity_conflict(
                audit_logger,
                run_id,
                reason="duplicate_client_id",
                phase=phase,
                node_id=node_id,
                client_id=client_id,
                server_round=server_round,
            )
            raise ValueError(f"Duplicate client_id {client_id!r} in {phase} results")
        self._node_to_client_id[node_id] = client_id
        if observed_round is not None:
            observed_round.add(client_id)
        return client_id


try:
    import mlflow

    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False
    mlflow = None  # type: ignore[assignment]


class HybridAggregation(_ClientIdentityMixin, FedAvg):
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
        if mlflow_enabled and not _MLFLOW_AVAILABLE:
            raise RuntimeError(
                "mlflow_enabled=True but mlflow is not installed; install "
                "insurance-federated-learning[tracking]"
            )
        self.mlflow_enabled = mlflow_enabled
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
        _validate_fit_updates(results)
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


class GatedStrategy(Strategy):
    """Composition wrapper that enforces the privacy budget before dispatch."""

    def __init__(
        self,
        inner: Strategy,
        *,
        accountant: RdpAccountantWrapper | None,
        state_path: str | None,
        max_epsilon: float | None,
        audit_logger: Any = None,
        run_id: str | None = None,
    ) -> None:
        if (accountant is None) != (max_epsilon is None):
            raise ValueError("GatedStrategy needs both accountant and max_epsilon")
        self.inner = inner
        self.accountant = accountant
        self.state_path = state_path
        self.max_epsilon = float(max_epsilon) if max_epsilon is not None else None
        self.audit_logger = audit_logger
        self.run_id = run_id
        self.budget_exhausted = False
        self.stopped_for_budget = False
        self.rounds_trained = 0
        self.rounds_skipped = 0

    def __getattr__(self, name: str) -> Any:
        if name == "inner":
            raise AttributeError(name)
        return getattr(self.inner, name)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ):
        if self.accountant is None or self.max_epsilon is None:
            return self.inner.configure_fit(server_round, parameters, client_manager)
        projected = self.accountant.project_epsilon(additional_releases=1)
        if self.budget_exhausted or projected > self.max_epsilon:
            self.rounds_skipped += 1
            self.stopped_for_budget = True
            if not self.budget_exhausted:
                self.budget_exhausted = True
                spent = self.accountant.project_epsilon(additional_releases=0)
                if self.audit_logger is not None:
                    try:
                        self.audit_logger.log_privacy_event(
                            round=server_round,
                            event="privacy_budget_exhausted",
                            epsilon=spent,
                            projected=projected,
                            max_epsilon=self.max_epsilon,
                        )
                    except Exception:
                        try:
                            self.audit_logger.log_event(
                                "privacy_budget_exhausted",
                                {
                                    "round": server_round,
                                    "epsilon": spent,
                                    "projected": projected,
                                    "max_epsilon": self.max_epsilon,
                                },
                            )
                        except Exception:
                            pass
            return []
        account_round(self.accountant, self.state_path, server_round)
        self.rounds_trained += 1
        return self.inner.configure_fit(server_round, parameters, client_manager)

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ):
        return self.inner.configure_evaluate(server_round, parameters, client_manager)

    def initialize_parameters(self, client_manager: ClientManager):
        return self.inner.initialize_parameters(client_manager)

    def aggregate_fit(self, server_round: int, results: list, failures: list):
        return self.inner.aggregate_fit(server_round, results, failures)

    def aggregate_evaluate(self, server_round: int, results: list, failures: list):
        return self.inner.aggregate_evaluate(server_round, results, failures)

    def evaluate(self, server_round: int, parameters: Parameters):
        return self.inner.evaluate(server_round, parameters)


class StratifiedFedProx(_ClientIdentityMixin, FedProx):
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
        _validate_fit_updates(results)
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
    max_epsilon: float | None = None,
    privacy_state_path: str | None = None,
    **kwargs: Any,
) -> Any:
    # Gate-only arguments are consumed here and never reach Flower constructors.
    if (accountant is None) != (max_epsilon is None):
        if max_epsilon is not None:
            raise ValueError("no privacy accountant configured")
        raise ValueError("max_epsilon is required when accountant is configured")
    gate_accountant = accountant
    gate_max_epsilon = max_epsilon
    if strategy_name not in _ALLOWED:
        raise ValueError(
            f"Unknown strategy {strategy_name!r}; allowed: {sorted(_ALLOWED)}. "
            "HybridAggregation/QFedAvg arrive with Task 4."
        )
    if strategy_name == "fedavg":
        strategy = FedAvg(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            **kwargs,
        )
        return GatedStrategy(
            strategy,
            accountant=gate_accountant,
            state_path=privacy_state_path,
            max_epsilon=gate_max_epsilon,
            audit_logger=audit_logger,
            run_id=run_id,
        )
    if strategy_name == "fedprox":
        strategy = StratifiedFedProx(
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
        return GatedStrategy(
            strategy,
            accountant=gate_accountant,
            state_path=privacy_state_path,
            max_epsilon=gate_max_epsilon,
            audit_logger=audit_logger,
            run_id=run_id,
        )
    if strategy_name == "qfedavg":
        strategy = QFedAvgStrategy(
            q_param=kwargs.pop("q_param", 0.2),
            qffl_learning_rate=kwargs.pop("qffl_learning_rate", 0.1),
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            **kwargs,
        )
        return GatedStrategy(
            strategy,
            accountant=gate_accountant,
            state_path=privacy_state_path,
            max_epsilon=gate_max_epsilon,
            audit_logger=audit_logger,
            run_id=run_id,
        )
    if strategy_name == "hybrid":
        strategy = HybridAggregation(
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
            accountant=None,
            audit_logger=audit_logger,
            run_id=run_id,
            mlflow_enabled=mlflow_enabled,
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_experiment_name=mlflow_experiment_name,
            require_all_clients=require_all_clients,
            secagg_enabled=secagg_enabled,
            **kwargs,
        )
        return GatedStrategy(
            strategy,
            accountant=gate_accountant,
            state_path=privacy_state_path,
            max_epsilon=gate_max_epsilon,
            audit_logger=audit_logger,
            run_id=run_id,
        )
    raise ValueError(
        f"Unknown strategy {strategy_name!r}; allowed: {sorted(_ALLOWED)}. "
        "HybridAggregation/QFedAvg arrive with Task 4."
    )


__all__ = [
    "GatedStrategy",
    "HybridAggregation",
    "QFedAvgStrategy",
    "StratifiedFedProx",
    "build_strategy",
]
