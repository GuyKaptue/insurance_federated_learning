"""Flower App task layer (Layer 1 / Task 0): data, model factory, train/eval.

Scope per ``Task_Specification.md`` Task 0 (canonical prompt
``prompts/31_TASK_SPECIFICATION.md``):

* ``prepare_dataset()`` — generates the smoke dataset ONCE **before**
  federation starts. Client apps are strictly read-only against it:
  concurrent generation from 4 simulated supernodes would race, and real
  FL sites hold their data locally beforehand anyway. CLI:

      python -m insurance_fl.fl.task --prepare

* Features: **numeric-only** columns (targets + ``customer_id`` excluded).
  One-hot encoding via ``DataPreprocessor`` needs cross-client column
  alignment (fit on global data) — deferred to P1 phase 7 / Task 11.

* Loss: **Kendall uncertainty-weighted** sum of the 5 task losses
  (:meth:`MultiTaskNN.get_kendall_total_loss`), implemented per Task 1
  (Layer 2). Layer 1 equal-weight fallback remains available via
  :meth:`MultiTaskNN.get_total_loss`.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import torch
from flwr.common import Scalar

from insurance_fl.data.generator import TARGET_COLUMNS, EuropeanInsuranceFederatedDataset
from insurance_fl.models.multi_task_nn import DEFAULT_TASK_TYPES, MultiTaskNN

log = logging.getLogger(__name__)

CLIENT_IDS: list[str] = [
    "germany_allianz",
    "france_axa",
    "belgium_ageas",
    "switzerland_helvetia",
]


def resolve_client_id(context) -> str:
    """Resolve canonical client identity from Flower simulation partition config.

    Prefers ``context.node_config`` keys ``partition-id`` and ``num-partitions``
    (Flower 1.31 simulation). The mapping is asserted to be a 1:1 assignment
    over ``CLIENT_IDS`` and never derived from ``context.node_id`` arithmetic.
    """
    node_config = getattr(context, "node_config", None) or {}
    if "partition-id" in node_config:
        partition_id = int(node_config["partition-id"])
        num_partitions = int(node_config.get("num-partitions", len(CLIENT_IDS)))
        if num_partitions != len(CLIENT_IDS):
            raise ValueError(
                f"num-partitions={num_partitions} does not match expected "
                f"{len(CLIENT_IDS)} clients ({CLIENT_IDS})"
            )
        if not (0 <= partition_id < len(CLIENT_IDS)):
            raise ValueError(
                f"partition-id={partition_id} out of range for {len(CLIENT_IDS)} clients"
            )
        client_id = CLIENT_IDS[partition_id]
        node_id = getattr(context, "node_id", None)
        log.info(
            "client_mapping node_id=%s partition_id=%s num_partitions=%s client_id=%s",
            node_id,
            partition_id,
            num_partitions,
            client_id,
        )
        print(
            f"CLIENT_MAPPING node_id={node_id} partition_id={partition_id} "
            f"num_partitions={num_partitions} client_id={client_id}",
            flush=True,
        )
        return client_id
    raise ValueError(
        "Missing partition-id in context.node_config; refusing node_id-based "
        "client identity mapping"
    )


DEFAULT_DATA_DIR = Path("data/processed/fl_task0_smoke")
DEFAULT_SAMPLES_PER_CLIENT = 512  # smallest size that satisfies stratified splits
DEFAULT_SEED = 42
DEFAULT_HIDDEN_DIMS: list[int] = [256, 128, 64]

_EXCLUDED_FEATURES = set(TARGET_COLUMNS) | {"customer_id"}
# Fixed, client-independent missingness indicator columns to ensure constant
# federated feature width (FedAvg requires identical shapes across clients).
MISSING_INDICATOR_COLS: tuple[str, ...] = (
    "annual_income",
    "savings_balance",
    "medical_expenses",
)
# Cache: (data_dir, client_id) -> (feature_cols, medians)
_MISSING_POLICY_CACHE: dict[tuple[str, str], tuple[list[str], dict[str, float]]] = {}


def prepare_dataset(
    output_dir: Path = DEFAULT_DATA_DIR,
    samples_per_client: int = DEFAULT_SAMPLES_PER_CLIENT,
    seed: int = DEFAULT_SEED,
) -> Path:
    """Generate the smoke dataset idempotently; return its directory.

    Uses the real ``EuropeanInsuranceFederatedDataset`` with quality-issue
    injection OFF so the baseline is not confounded by suspect #8
    (injection is owned by the generator's flag, audited under B2).
    """
    output_dir = Path(output_dir)
    marker = output_dir / "dataset_info.json"
    if marker.exists():
        log.info("Smoke dataset already present: %s", output_dir)
        return output_dir
    EuropeanInsuranceFederatedDataset(
        samples_per_client=samples_per_client,
        random_seed=seed,
        inject_quality_issues=False,
    ).generate_dataset(output_dir=str(output_dir))
    return output_dir


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Numeric feature columns: targets and identifiers excluded."""
    return [c for c in df.select_dtypes(include=["number"]).columns if c not in _EXCLUDED_FEATURES]


def _partition_path(data_dir: Path, client_id: str, split: str) -> Path:
    path = Path(data_dir) / "federated" / client_id / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing partition {path}. Run first: " "python -m insurance_fl.fl.task --prepare"
        )
    return path


def _fit_missing_policy(data_dir: Path, client_id: str) -> tuple[list[str], dict[str, float]]:
    """Fit median imputation on the TRAIN split only (per client).

    Returns (feature_cols, medians). Indicator columns are fixed globally
    via MISSING_INDICATOR_COLS so all clients share the same output width.
    """
    cache_key = (str(Path(data_dir).resolve()), client_id)
    if cache_key in _MISSING_POLICY_CACHE:
        return _MISSING_POLICY_CACHE[cache_key]
    train_df = pd.read_parquet(_partition_path(Path(data_dir), client_id, "train"))
    cols = feature_columns(train_df)
    med = train_df[cols].median(numeric_only=True)
    # If a column is entirely NaN in train, fall back to 0.0 deterministically.
    medians = {c: (0.0 if pd.isna(med[c]) else float(med[c])) for c in cols}
    _MISSING_POLICY_CACHE[cache_key] = (cols, medians)
    return cols, medians


def clear_missing_policy_cache() -> None:
    """Clear cached train-fit imputation policies (useful for tests)."""
    _MISSING_POLICY_CACHE.clear()


def _apply_missing_policy(
    df: pd.DataFrame, feature_cols: list[str], medians: dict[str, float]
) -> np.ndarray:
    xdf = df[feature_cols].copy()
    # Fixed indicators (client-independent width), added in declared order.
    indicators = {}
    for col in MISSING_INDICATOR_COLS:
        if col in xdf.columns:
            indicators[f"{col}__missing"] = xdf[col].isna().astype(np.float32)
        else:
            indicators[f"{col}__missing"] = np.zeros(len(xdf), dtype=np.float32)
    ind_df = pd.DataFrame(indicators, index=xdf.index)
    # Median imputation (train-fitted medians)
    for c in feature_cols:
        if xdf[c].isna().any():
            xdf[c] = xdf[c].fillna(medians.get(c, 0.0))
    xdf = pd.concat([xdf, ind_df], axis=1)
    arr = np.asarray(xdf.to_numpy(dtype=np.float32, copy=True), dtype=np.float32)
    if not np.isfinite(arr).all():
        # Final safety net for any remaining non-finite values.
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    return cast(np.ndarray, np.array(arr, dtype=np.float32, copy=False))


def get_feature_names(data_dir: Path, client_id: str, split: str = "train") -> list[str]:
    """Return ordered feature names produced by load_partition for a client/split."""
    path = _partition_path(Path(data_dir), client_id, split)
    df = pd.read_parquet(path)
    cols = feature_columns(df)
    return cols + [f"{c}__missing" for c in MISSING_INDICATOR_COLS]


def load_partition(
    data_dir: Path,
    client_id: str,
    split: str = "train",
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load one client's partition: (float32 X, {task: y}).

    Median imputation is fit on the client's TRAIN split only and applied to
    the requested split; fixed missingness indicators (MISSING_INDICATOR_COLS)
    are appended so all clients share identical feature width.

    Raises ``FileNotFoundError`` pointing at ``--prepare`` when data is
    missing — no hidden generation side effect inside clients.
    """
    data_dir = Path(data_dir)
    path = _partition_path(data_dir, client_id, split)
    df = pd.read_parquet(path)
    cols, medians = _fit_missing_policy(data_dir, client_id)
    x = _apply_missing_policy(df, cols, medians)
    y = {task: df[task].to_numpy(copy=True) for task in DEFAULT_TASK_TYPES}
    return x, y


def build_model(input_dim: int, hidden_dims: list[int] | None = None) -> MultiTaskNN:
    """Model factory shared by client and tests (same construction both sides)."""
    return MultiTaskNN(
        input_dim=input_dim,
        hidden_dims=list(hidden_dims) if hidden_dims else list(DEFAULT_HIDDEN_DIMS),
    )


def train(
    model: MultiTaskNN,
    x: np.ndarray,
    y: dict[str, np.ndarray],
    *,
    epochs: int = 1,
    batch_size: int = 32,
    lr: float = 1e-3,
) -> float:
    """Minibatch Adam training with Kendall uncertainty weights; return final loss.

    Singleton batches are skipped: ``BatchNorm1d`` needs >= 2 samples per
    batch in training mode.
    """
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    xt = torch.as_tensor(x)
    yt = {k: torch.as_tensor(v) for k, v in y.items()}
    n = int(xt.shape[0])
    last = float("nan")
    for _ in range(max(1, epochs)):
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            if idx.numel() < 2:
                continue
            opt.zero_grad()
            logits = model.forward(xt[idx])
            losses = model.get_losses(logits, {k: v[idx] for k, v in yt.items()})
            loss = model.get_kendall_total_loss(losses)
            loss.backward()
            opt.step()
            last = float(loss.detach().item())
    return last


def evaluate(
    model: MultiTaskNN,
    x: np.ndarray,
    y: dict[str, np.ndarray],
) -> dict[str, Scalar]:
    """Overall loss + per-task metrics (rmse for regression, acc otherwise).

    Values are ``Scalar`` (flwr's wire type) so metrics flow into Flower's
    fit/evaluate result dicts without conversion.
    """
    model.eval()
    metrics: dict[str, Scalar] = {}
    with torch.no_grad():
        xt = torch.as_tensor(x)
        yt = {k: torch.as_tensor(v) for k, v in y.items()}
        logits = model.forward(xt)
        losses = model.get_losses(logits, yt)
        metrics["loss"] = float(model.get_kendall_total_loss(losses).item())
        for task, task_type in DEFAULT_TASK_TYPES.items():
            out, target = logits[task], yt[task]
            if task_type == "regression":
                diff = out - target.float()
                metrics[f"{task}_rmse"] = float(torch.sqrt(torch.mean(diff**2)).item())
            elif task_type == "binary":
                pred = (torch.sigmoid(out) >= 0.5).to(target.dtype)
                metrics[f"{task}_acc"] = float((pred == target).float().mean().item())
            else:
                pred = out.argmax(dim=-1)
                metrics[f"{task}_acc"] = float((pred == target).float().mean().item())
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="insurance_fl.fl.task",
        description="Prepare the Layer 1 smoke dataset for the Flower App.",
    )
    parser.add_argument("--prepare", action="store_true", help="generate the dataset")
    parser.add_argument("--output-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--samples-per-client", type=int, default=DEFAULT_SAMPLES_PER_CLIENT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    if not args.prepare:
        parser.error("only --prepare is supported (see module docstring)")
    out = prepare_dataset(Path(args.output_dir), args.samples_per_client, args.seed)
    print(f"Dataset ready: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
