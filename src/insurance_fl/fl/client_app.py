"""Flower ClientApp (Layer 2 / Task 1): local training on one insurance client.

One simulated supernode maps to one of the 4 real client identities
(``node_id % 4`` — stable within a ``flwr run`` simulation session;
Task 11 revisits partitioning for the integration test).

Each client is READ-ONLY against the pre-generated smoke dataset
(``python -m insurance_fl.fl.task --prepare`` runs first).

FedBN (Task 1): BatchNorm tensors never cross the wire. The client
exchanges only non-BN parameters via :meth:`MultiTaskNN.get_federated_weights`
/ :meth:`MultiTaskNN.set_federated_weights`. Local BN state is persisted in
``context.state`` (a ``RecordDict`` keyed ``"bn_local"``) across rounds:
saved after :meth:`fit`, restored before :meth:`fit` and :meth:`evaluate`.

Secure Aggregation (Layer 3): when ``secagg-enabled`` is true in
``run_config``, the ``secaggplus_mod`` client modifier cryptographically
masks the client update before transmission. Local DP (Layer 4), when
enabled, is applied before SecAgg masking via the ``LocalDpMod`` modifier.
Modifier ordering (outermost first): SecAgg → Local DP → client fit, so
that on the outbound path Local DP noise is added to the raw update and
SecAgg then masks the protected update. Secure Aggregation alone is NOT a
differential privacy guarantee.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from flwr.client import Client, ClientApp, NumPyClient
from flwr.client.mod.localdp_mod import LocalDpMod
from flwr.client.mod.secure_aggregation.secaggplus_mod import secaggplus_mod
from flwr.common import ArrayRecord, Context, Scalar

from insurance_fl.fl import task as fl_task
from insurance_fl.fl.privacy import build_privacy_config
from insurance_fl.models.multi_task_nn import MultiTaskNN

log = logging.getLogger(__name__)


class InsuranceNNClient(NumPyClient):
    """Trains the MultiTaskNN on one client's local partition."""

    def __init__(
        self,
        net: MultiTaskNN,
        x: np.ndarray,
        y: dict[str, np.ndarray],
        *,
        epochs: int,
        batch_size: int,
        lr: float,
        state,
    ) -> None:
        super().__init__()
        self.net = net
        self.x = x
        self.y = y
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.state = state  # RecordDict: persists across rounds on this node

    def get_parameters(self, config: dict[str, Scalar]) -> list[np.ndarray]:
        # FedBN wire format: non-BN parameters only
        return self.net.get_federated_weights()

    def fit(
        self,
        parameters: list[np.ndarray],
        config: dict[str, Scalar],
    ) -> tuple[list[np.ndarray], int, dict[str, Scalar]]:
        # Restore local BN state saved from the previous round (if any)
        if "bn_local" in self.state:
            bn_record = self.state["bn_local"]
            bn_state = {k: arr.numpy() for k, arr in bn_record.items()}
            self.net.set_local_bn_state(bn_state)
            log.info("FedBN: restored local BN state from previous round")

        # Set federated (non-BN) weights; local BN is left untouched
        self.net.set_federated_weights(parameters)

        epochs = int(config.get("local-epochs", self.epochs))
        loss = fl_task.train(
            self.net,
            self.x,
            self.y,
            epochs=epochs,
            batch_size=self.batch_size,
            lr=self.lr,
        )

        # Persist updated local BN state for the next round
        bn_state = self.net.get_local_bn_state()
        torch_state = {k: torch.as_tensor(v) for k, v in bn_state.items()}
        self.state["bn_local"] = ArrayRecord(torch_state_dict=torch_state)
        log.info("FedBN: saved local BN state to context.state for next round")

        return self.net.get_federated_weights(), len(self.x), {"train_loss": loss}

    def evaluate(
        self,
        parameters: list[np.ndarray],
        config: dict[str, Scalar],
    ) -> tuple[float, int, dict[str, Scalar]]:
        # Restore local BN state saved from the previous round (if any)
        if "bn_local" in self.state:
            bn_record = self.state["bn_local"]
            bn_state = {k: arr.numpy() for k, arr in bn_record.items()}
            self.net.set_local_bn_state(bn_state)

        # Set federated (non-BN) weights; local BN is left untouched
        self.net.set_federated_weights(parameters)

        metrics = fl_task.evaluate(self.net, self.x, self.y)
        return float(metrics["loss"]), len(self.x), metrics


def client_fn(context: Context) -> Client:
    """Build the client bound to this supernode's insurance identity."""
    rc = context.run_config
    client_id = fl_task.CLIENT_IDS[int(context.node_id) % len(fl_task.CLIENT_IDS)]
    data_dir = Path(str(rc["data-dir"]))  # run_config values are Scalars
    x, y = fl_task.load_partition(data_dir, client_id, split="train")
    net = fl_task.build_model(input_dim=int(x.shape[1]))
    log.info(
        "node %s -> client %s (%d train rows, %d features)",
        context.node_id,
        client_id,
        len(x),
        x.shape[1],
    )
    return InsuranceNNClient(
        net,
        x,
        y,
        epochs=int(rc["local-epochs"]),
        batch_size=int(rc["batch-size"]),
        lr=float(rc["learning-rate"]),
        state=context.state,
    ).to_client()


def _is_enabled(context: Context, key: str) -> bool:
    """Read a boolean flag from ``run_config``."""
    return str(context.run_config.get(key, "false")).lower() == "true"


def _secagg_mod(msg, context, call_next):
    """Conditionally apply the SecAgg+ client modifier."""
    if _is_enabled(context, "secagg-enabled"):
        return secaggplus_mod(msg, context, call_next)
    return call_next(msg, context)


_local_dp_mod_instance: LocalDpMod | None = None


def _local_dp_mod(msg, context, call_next):
    """Conditionally apply Local DP before SecAgg masking."""
    global _local_dp_mod_instance
    if not _is_enabled(context, "local-dp-enabled"):
        return call_next(msg, context)
    if _local_dp_mod_instance is None:
        cfg = build_privacy_config(dict(context.run_config))
        _local_dp_mod_instance = LocalDpMod(
            clipping_norm=float(context.run_config.get("local-dp-clipping-norm", 1.0)),
            sensitivity=float(context.run_config.get("local-dp-sensitivity", 1.0)),
            epsilon=float(cfg.max_epsilon),
            delta=float(cfg.target_delta),
        )
    return _local_dp_mod_instance(msg, context, call_next)


# Modifier chain (outermost first): SecAgg → Local DP → client fit.
# On the outbound (client→server) path, Local DP noise is added to the raw
# update first, then SecAgg masks the protected update.
app = ClientApp(
    client_fn=client_fn,
    mods=[_secagg_mod, _local_dp_mod],
)


def hybrid_evaluate(
    net,
    x: np.ndarray,
    y: dict[str, np.ndarray],
    artifact_dir: Path | None = None,
    beta_weights: dict[str, float] | None = None,
    task_list: list[str] | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    """Evaluate using NN + booster + stacking + beta fusion.

    Falls back to NN-only if booster artifacts are unavailable (unless strict).
    In strict mode, missing artifacts raise ValueError.
    """
    from pathlib import Path as _Path

    metrics: dict[str, Any] = {}
    nn_metrics = fl_task.evaluate(net, x, y)
    for k, v in nn_metrics.items():
        metrics[f"nn_{k}"] = v

    if task_list is None:
        task_list = list(nn_metrics.keys()) if nn_metrics else []

    # Determine if hybrid is possible
    if artifact_dir is None or beta_weights is None:
        if strict:
            raise ValueError("Hybrid evaluation requires artifact_dir and beta_weights")
        for k, v in nn_metrics.items():
            metrics[k] = v
        metrics["evaluation_mode"] = "nn_only_fallback"
        metrics["fallback_reason"] = "no_artifact_dir_or_beta"
        return metrics

    adir = _Path(artifact_dir)
    from insurance_fl.fl.boosters import artifacts_available, load_booster_artifacts

    # Try hybrid fusion per task
    hybrid_preds: dict[str, np.ndarray] = {}
    fallback_reasons: list[str] = []
    loaded_components: dict[str, list[str]] = {}

    for task in task_list or []:
        if not artifacts_available(adir, task):
            fallback_reasons.append(f"missing_artifacts:{task}")
            continue

        from insurance_fl.fl.boosters import load_stacking_meta_for_task

        meta = load_stacking_meta_for_task(adir, task)
        if meta is None:
            fallback_reasons.append(f"invalid_meta:{task}")
            continue

        from insurance_fl.fl.stacking import validate_alpha

        alpha = validate_alpha(
            [
                meta.get("alpha_xgb", 1.0),
                meta.get("alpha_lgbm", 0.0),
                meta.get("alpha_cat", 0.0),
            ]
        )
        from insurance_fl.fl.hybrid import validate_beta

        beta = validate_beta(beta_weights.get(task, 0.5) if beta_weights else 0.5)

        # Load booster models and predict
        arts = load_booster_artifacts(adir, task)
        try:
            import lightgbm as lgb
            import xgboost as xgb
            from catboost import CatBoostClassifier, CatBoostRegressor

            task_type = meta.get("task_type", "regression")

            # Load XGBoost
            if task_type == "regression":
                xgb_model = xgb.XGBRegressor()
            else:
                xgb_model = xgb.XGBClassifier()
            xgb_model.load_model(str(arts["xgb"]))
            xgb_pred = np.asarray(xgb_model.predict(x), dtype=float).ravel()

            # Load LightGBM
            lgb_model = lgb.Booster(model_file=str(arts["lgbm"]))
            lgb_pred = np.asarray(lgb_model.predict(x), dtype=float).ravel()

            # Load CatBoost
            if task_type == "regression":
                cat_model = CatBoostRegressor()
            else:
                cat_model = CatBoostClassifier()
            cat_model.load_model(str(arts["cat"]))
            cat_pred = np.asarray(cat_model.predict(x), dtype=float).ravel()

            loaded_components[task] = ["xgb", "lgbm", "cat", "stacking", "beta"]

        except Exception as e:
            fallback_reasons.append(f"load_error:{task}:{e}")
            continue

        # NN prediction for this task (from trained model)
        import torch

        net.eval()
        with torch.no_grad():
            logits = net.forward(torch.as_tensor(x.astype(np.float32)))
        if isinstance(logits, dict):
            nn_out = logits[task].detach().cpu().numpy()
        else:
            nn_out = logits.detach().cpu().numpy()
        nn_pred = np.asarray(nn_out, dtype=float).ravel()
        if nn_pred.ndim == 0:
            nn_pred = np.full(len(x), float(nn_pred))

        # Convert raw logits to probabilities for classification tasks
        from insurance_fl.fl.task import DEFAULT_TASK_TYPES

        task_type = (
            meta.get("task_type", DEFAULT_TASK_TYPES.get(task, "regression"))
            if "meta" in dir()
            else DEFAULT_TASK_TYPES.get(task, "regression")
        )
        # Determine task type from DEFAULT_TASK_TYPES
        raw_type = DEFAULT_TASK_TYPES.get(task, "regression")
        if "binary" in raw_type:
            nn_pred = 1.0 / (1.0 + np.exp(-np.clip(nn_pred, -50, 50)))
        elif "multi" in raw_type:
            nn_out_2d = np.asarray(nn_out, dtype=float)
            if nn_out_2d.ndim == 1:
                nn_out_2d = nn_out_2d.reshape(-1, 1)
            exp_out = np.exp(nn_out_2d - np.max(nn_out_2d, axis=1, keepdims=True))
            nn_pred = (exp_out / exp_out.sum(axis=1, keepdims=True)).ravel()
            if nn_pred.shape[0] > len(x):
                nn_pred = nn_pred[: len(x)]

        # Row alignment check
        assert len(x) == len(y.get(task, [])) if task in y else True, f"{task}: x/y length mismatch"

        # Apply stacking
        from insurance_fl.fl.stacking import predict_stacked

        stacked = predict_stacked([xgb_pred, lgb_pred, cat_pred], np.asarray(alpha, dtype=float))

        # Apply beta fusion
        from insurance_fl.fl.hybrid import predict_hybrid

        hybrid = predict_hybrid(nn_pred, stacked, beta)
        hybrid_preds[task] = hybrid

    if strict and (fallback_reasons or not hybrid_preds):
        raise ValueError(f"Hybrid evaluation failed: {';'.join(fallback_reasons)}")

    if fallback_reasons and not hybrid_preds:
        hybrid_mode = "nn_only_fallback"
        for k, v in nn_metrics.items():
            metrics[k] = v
        metrics["evaluation_mode"] = hybrid_mode
        metrics["fallback_reason"] = ";".join(fallback_reasons)
        return metrics

    # Hybrid succeeded
    for k, v in nn_metrics.items():
        metrics[k] = v
    metrics["evaluation_mode"] = "hybrid"
    metrics["loaded_components"] = loaded_components
    if fallback_reasons:
        metrics["partial_fallback"] = ";".join(fallback_reasons)
    return metrics
