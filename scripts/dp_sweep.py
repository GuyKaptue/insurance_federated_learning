"""Local DP utility sweep.

Evaluates the utility-privacy tradeoff of Local DP by training a
MultiTaskNN on a client partition, applying Local DP (clip + Gaussian
noise) to the trained model update, and measuring validation loss
degradation. Results are persisted as JSON.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any


def _git_state() -> dict[str, Any]:
    state: dict[str, Any] = {"git_sha": None, "dirty": None}
    try:
        state["git_sha"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return state
    try:
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
        )
        state["dirty"] = bool(dirty.strip())
    except Exception:
        state["dirty"] = None
    return state


def _dataset_provenance():
    root = Path(__file__).resolve().parents[1]
    mp = root / "docs" / "evidence" / "dataset_manifest.json"
    if not mp.exists():
        return {"dataset_version": None, "config_hash": None}
    try:
        m = json.loads(mp.read_text())
    except Exception:
        return {"dataset_version": None, "config_hash": None}
    return {
        "dataset_version": m.get("dataset_version"),
        "config_hash": m.get("config_hash"),
        "dataset_content_hash": m.get("files_content_sha256", {}).get("overall"),
    }


def _epsilon_for_noise_multiplier(
    noise_multiplier: float, sensitivity: float, clipping_norm: float, delta: float
) -> float:
    """Translate a target noise multiplier to Flower LocalDpMod's epsilon API."""
    if noise_multiplier <= 0 or sensitivity <= 0 or clipping_norm <= 0:
        raise ValueError("noise multiplier, sensitivity, and clipping norm must be positive")
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1)")
    return (
        sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / (clipping_norm * noise_multiplier)
    )


def _load_accountant_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Accountant state not found: {path}")
    state = json.loads(path.read_text(encoding="utf-8"))
    required = {"noise_multiplier", "delta", "rounds_spent", "config_fingerprint"}
    missing = sorted(required - set(state))
    if missing:
        raise ValueError(f"Accountant state missing fields: {missing}")
    return {
        "noise_multiplier": float(state["noise_multiplier"]),
        "delta": float(state["delta"]),
        "rounds_spent": int(state["rounds_spent"]),
        "config_fingerprint": state["config_fingerprint"],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="reports/dp_utility_curve.json", help="Output JSON path")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--client", default="germany_allianz")
    p.add_argument("--data-dir", default="data/processed/fl_task0_smoke")
    p.add_argument("--samples-per-client", type=int, default=512)
    p.add_argument(
        "--noise-multipliers",
        default="0.5,1.0,2.0",
        help="Comma-separated Gaussian noise multipliers relative to the clipping norm",
    )
    p.add_argument(
        "--epsilons",
        default=None,
        help="Deprecated compatibility input; converted to noise multipliers before applying the mechanism",
    )
    p.add_argument("--delta", type=float, default=None)
    p.add_argument("--clipping-norm", type=float, default=1.0)
    p.add_argument("--sensitivity", type=float, default=1.0)
    p.add_argument(
        "--accountant-state",
        default="docs/evidence/c6_run_2026-11-21/privacy_state.json",
        help="Persisted RDP accountant state whose multiplier/delta are reported for comparison",
    )
    p.add_argument("--epochs", type=int, default=5, help="Baseline/local train epochs")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--quiet", action="store_true", help="Suppress summary table output")
    return p.parse_args(argv)


def _train_local(
    data_dir: Path, client_id: str, seed: int, epochs: int, batch_size: int, lr: float
):
    import numpy as np
    import torch

    from insurance_fl.fl.task import build_model, load_partition, train

    torch.manual_seed(seed)
    np.random.seed(seed)

    x_train, y_train = load_partition(data_dir, client_id, split="train")
    x_val, y_val = load_partition(data_dir, client_id, split="val")

    model = build_model(input_dim=int(x_train.shape[1]))
    initial_params = [p.detach().numpy().copy() for p in model.parameters()]
    loss = train(model, x_train, y_train, epochs=epochs, batch_size=batch_size, lr=lr)
    return model, x_train, y_train, x_val, y_val, loss, initial_params


def _apply_local_dp(
    model,
    initial_ndarrays,
    input_dim: int,
    clipping_norm: float,
    sensitivity: float,
    epsilon: float,
    delta: float,
):
    import torch
    from flwr.client.mod.localdp_mod import (
        add_localdp_gaussian_noise_to_params,
        compute_clip_model_update,
    )
    from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

    from insurance_fl.fl.task import build_model

    trained_ndarrays = [p.detach().numpy().copy() for p in model.parameters()]
    compute_clip_model_update(trained_ndarrays, initial_ndarrays, clipping_norm)
    params_as_parameters = ndarrays_to_parameters(trained_ndarrays)
    noised_params = add_localdp_gaussian_noise_to_params(
        params_as_parameters, sensitivity, epsilon, delta
    )
    noised_ndarrays = parameters_to_ndarrays(noised_params)

    noised_model = build_model(input_dim=input_dim)
    noised_params_iter = iter(noised_ndarrays)
    with torch.no_grad():
        for param in noised_model.parameters():
            try:
                next_val = next(noised_params_iter)
                param.copy_(torch.as_tensor(next_val))
            except StopIteration:
                break
    return noised_model


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    accountant = _load_accountant_state(Path(args.accountant_state))
    delta = float(accountant["delta"] if args.delta is None else args.delta)
    if args.epsilons:
        epsilons = [float(x.strip()) for x in args.epsilons.split(",") if x.strip()]
        noise_multipliers = [
            args.sensitivity
            * math.sqrt(2.0 * math.log(1.25 / delta))
            / (args.clipping_norm * epsilon)
            for epsilon in epsilons
        ]
    else:
        noise_multipliers = [
            float(x.strip()) for x in args.noise_multipliers.split(",") if x.strip()
        ]
        if not any(
            math.isclose(value, accountant["noise_multiplier"], rel_tol=0, abs_tol=1e-12)
            for value in noise_multipliers
        ):
            noise_multipliers.append(float(accountant["noise_multiplier"]))
        epsilons = [
            _epsilon_for_noise_multiplier(value, args.sensitivity, args.clipping_norm, delta)
            for value in noise_multipliers
        ]

    # Heavy imports after argument parsing
    from insurance_fl.fl.task import DEFAULT_DATA_DIR, evaluate, prepare_dataset

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        data_dir = prepare_dataset(
            output_dir=DEFAULT_DATA_DIR if str(data_dir) == str(DEFAULT_DATA_DIR) else data_dir,
            samples_per_client=args.samples_per_client,
            seed=args.seed,
        )
    else:
        data_dir = Path(data_dir)

    (
        model,
        x_train,
        y_train,
        x_val,
        y_val,
        baseline_loss,
        initial_ndarrays,
    ) = _train_local(data_dir, args.client, args.seed, args.epochs, args.batch_size, args.lr)
    input_dim = int(x_train.shape[1])
    baseline_metrics = evaluate(model, x_val, y_val)
    git = _git_state()
    prov = _dataset_provenance()

    results: dict[str, Any] = {
        "dataset_version": prov.get("dataset_version"),
        "config_hash": prov.get("config_hash"),
        "dataset_content_hash": prov.get("dataset_content_hash"),
        "delta": delta,
        "clipping_norm": args.clipping_norm,
        "sensitivity": args.sensitivity,
        "mechanism": "Flower LocalDpMod Gaussian update clipping/noise helpers",
        "accountant": accountant,
        "accountant_noise_multiplier": accountant["noise_multiplier"],
        "noise_multipliers": noise_multipliers,
        "seed": args.seed,
        "client": args.client,
        "data_dir": str(data_dir),
        "n_train": len(x_train),
        "n_val": len(x_val),
        "epochs": args.epochs,
        "post_dp_epochs": 0,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "mechanism_epsilons": epsilons,
        "git_sha": git["git_sha"],
        "git_dirty": git["dirty"],
        "note": "Local DP utility sweep (model-update clip + Gaussian noise). Empirical utility measurements, not a formal DP guarantee report.",
        "baseline": {
            "epsilon": None,
            "train_loss": float(baseline_loss),
            "val_metrics": baseline_metrics,
        },
        "configurations": [],
    }

    if not args.quiet:
        print("\nLocal DP Utility Sweep")
        print("=" * 66)
        print(
            f"  delta={delta}, clipping_norm={args.clipping_norm}, "
            f"sensitivity={args.sensitivity}, accountant_noise_multiplier="
            f"{accountant['noise_multiplier']}"
        )
        print(f"  seed={args.seed}, client={args.client}, data_dir={data_dir}")
        print()
        print(f"  {'sigma':>8} | {'train_loss':>12} | {'val_loss':>12} | {'will_renew_acc':>14}")
        sep = "-" * 8 + "-+-" + "-" * 12 + "-+-" + "-" * 12 + "-+-" + "-" * 14
        print(f"  {sep}")
        print(
            f"  {'baseline':>8} | {baseline_loss:>12.6f} | "
            f"{float(baseline_metrics.get('loss', float('nan'))):>12.6f} | "
            f"{float(baseline_metrics.get('will_renew_acc', float('nan'))):>14.4f}"
        )

    for noise_multiplier, eps in zip(noise_multipliers, epsilons):
        try:
            noised_model = _apply_local_dp(
                model,
                initial_ndarrays,
                input_dim,
                args.clipping_norm,
                args.sensitivity,
                eps,
                delta,
            )
            train_metrics = evaluate(noised_model, x_train, y_train)
            metrics = evaluate(noised_model, x_val, y_val)
            results["configurations"].append(
                {
                    "noise_multiplier": float(noise_multiplier),
                    "epsilon": float(eps),
                    "train_loss": float(train_metrics.get("loss", float("nan"))),
                    "val_metrics": metrics,
                }
            )
            if not args.quiet:
                print(
                    f"  {noise_multiplier:>8.2f} | {float(train_metrics.get('loss', float('nan'))):>12.6f} | "
                    f"{float(metrics.get('loss', float('nan'))):>12.6f} | "
                    f"{float(metrics.get('will_renew_acc', float('nan'))):>14.4f}"
                )
        except Exception as exc:
            if not args.quiet:
                print(f"  {noise_multiplier:>8.2f} | FAILED: {exc}")
            results["configurations"].append(
                {
                    "noise_multiplier": float(noise_multiplier),
                    "epsilon": float(eps),
                    "error": str(exc),
                }
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, default=str))
    if not args.quiet:
        print()
        print(f"Results written to {out_path}")
        print(f"  git_dirty: {git['dirty']}")
        print("\n## Local DP Utility Summary\n")
        print("| noise multiplier | delta | val_loss | will_renew_acc | claim_risk_score_rmse |")
        print("|------------------|-------|----------|----------------|-----------------------|")
        for cfg in results["configurations"]:
            if "error" in cfg:
                print(f"| {cfg['noise_multiplier']} | {delta} | ERROR | ERROR | ERROR |")
                continue
            m = cfg["val_metrics"]
            print(
                f"| {cfg['noise_multiplier']} | {delta} | "
                f"{m.get('loss', 'N/A')} | {m.get('will_renew_acc', 'N/A')} | "
                f"{m.get('claim_risk_score_rmse', 'N/A')} |"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
