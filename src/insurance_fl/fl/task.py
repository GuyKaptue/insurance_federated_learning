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

* Loss: **equal-weight** sum of the 5 task losses. Kendall uncertainty
  weighting is Task 1 (Layer 2) — deliberately NOT implemented here.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from flwr.common import Scalar

from insurance_fl.data.generator import (
    TARGET_COLUMNS,
    EuropeanInsuranceFederatedDataset,
)
from insurance_fl.models.multi_task_nn import DEFAULT_TASK_TYPES, MultiTaskNN

log = logging.getLogger(__name__)

CLIENT_IDS: list[str] = [
    "germany_allianz",
    "france_axa",
    "belgium_ageas",
    "switzerland_helvetia",
]

DEFAULT_DATA_DIR = Path("data/processed/fl_task0_smoke")
DEFAULT_SAMPLES_PER_CLIENT = 512  # smallest size that satisfies stratified splits
DEFAULT_SEED = 42
DEFAULT_HIDDEN_DIMS: list[int] = [128, 64, 32]

_EXCLUDED_FEATURES = set(TARGET_COLUMNS) | {"customer_id"}


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


def load_partition(
    data_dir: Path,
    client_id: str,
    split: str = "train",
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load one client's partition: (float32 X, {task: y}).

    Raises ``FileNotFoundError`` pointing at ``--prepare`` when data is
    missing — no hidden generation side effect inside clients.
    """
    path = Path(data_dir) / "federated" / client_id / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing partition {path}. Run first: " "python -m insurance_fl.fl.task --prepare"
        )
    df = pd.read_parquet(path)
    # copy=True: parquet/pandas may yield read-only views; torch needs
    # writable arrays (UserWarning "not writable" otherwise).
    x = df[feature_columns(df)].to_numpy(dtype=np.float32, copy=True)
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
    """Minibatch Adam training with equal task weights; return final loss.

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
            loss = model.get_total_loss(losses)  # equal weights: Task 1 adds Kendall
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
        metrics["loss"] = float(model.get_total_loss(losses).item())
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
