#!/usr/bin/env python
"""End-to-end hybrid federated learning pipeline (§Task 1).

Usage:
    python scripts/run_e2e.py [--rounds N] [--run-id ID] [--output-dir runs]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run E2E hybrid FL pipeline.")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="runs")
    parser.add_argument("--data-dir", type=str, default="data/processed/fl_task0_smoke")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _ensure_data(data_dir: Path) -> None:
    if not data_dir.exists():
        from insurance_fl.fl.task import prepare_dataset

        prepare_dataset(output_dir=data_dir, samples_per_client=256, seed=42)


def _train_boosters(data_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Stage 3: Train boosters per client."""
    from insurance_fl.fl.boosters import train_boosters_for_client
    from insurance_fl.fl.task import load_partition

    client_ids = ["germany_allianz", "france_axa", "belgium_ageas", "switzerland_helvetia"]
    tasks = [
        "claim_risk_score",
        "customer_lifetime_value",
        "will_renew",
        "is_high_risk",
        "customer_segment",
    ]
    results: dict[str, Any] = {}

    for cid in client_ids:
        x_train, y_train = load_partition(data_dir, cid, split="train")
        x_val, y_val = load_partition(data_dir, cid, split="val")
        adir = output_dir / "boosters" / cid
        r = train_boosters_for_client(
            client_id=cid,
            x_train=x_train,
            y_train={t: y_train[t] for t in tasks},
            x_val=x_val,
            y_val={t: y_val[t] for t in tasks},
            output_dir=adir,
            tasks=tasks,
            n_folds=3,
            seed=42,
        )
        results[cid] = r
    return results


def _run_fl_training(rounds: int, output_dir: Path) -> int:
    """Stage 4: Run Flower FL training."""
    log_path = output_dir / "fl.log"
    cmd = ["flwr", "run", ".", "--stream"]
    # Use simulation mode (no TLS) for E2E
    with open(log_path, "w") as f:
        proc = subprocess.run(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            timeout=300,
            cwd=str(Path.cwd()),
        )
    return proc.returncode


def _optimize_beta(output_dir: Path) -> dict[str, float | None]:
    """Stage 6: Optimize beta per task on validation data."""
    import torch

    from insurance_fl.fl.hybrid import optimize_beta, save_beta_weights
    from insurance_fl.fl.task import build_model, load_partition

    data_dir = Path("data/processed/fl_task0_smoke")
    client_id = "germany_allianz"
    x_val, y_val = load_partition(data_dir, client_id, split="val")

    # Load NN model
    nn_path = output_dir / "nn_checkpoint"
    if nn_path.exists():
        from insurance_fl.models.multi_task_nn import MultiTaskNN

        net = MultiTaskNN.load(nn_path)
    else:
        net = build_model(input_dim=int(x_val.shape[1]))

    net.eval()
    beta_weights: dict[str, float] = {}
    diagnostics: dict[str, Any] = {}

    tasks_to_optimize = [
        "claim_risk_score",
        "customer_lifetime_value",
        "will_renew",
        "is_high_risk",
        "customer_segment",
    ]
    for task in tasks_to_optimize:
        # NN predictions
        with torch.no_grad():
            logits = net.forward(torch.as_tensor(x_val.astype(np.float32)))
            nn_out = logits[task].detach().numpy()
            if nn_out.ndim > 1:
                nn_pred = (
                    nn_out[:, 0] if nn_out.shape[1] == 1 else nn_out.argmax(axis=1).astype(float)
                )
            else:
                nn_pred = nn_out.ravel()

        # Determine task type first
        from insurance_fl.fl.task import DEFAULT_TASK_TYPES

        raw_type = DEFAULT_TASK_TYPES.get(task, "regression")
        if "regression" in raw_type:
            tt = "regression"
        elif "binary" in raw_type:
            tt = "binary"
        else:
            tt = "multi"

        if tt == "multi":
            print(f"    {task}: multi-class beta optimization skipped (documented limitation)")
            continue

        # Booster predictions (from artifacts)
        adir = output_dir / "boosters" / client_id
        from insurance_fl.fl.boosters import load_booster_artifacts, load_stacking_meta_for_task
        from insurance_fl.fl.stacking import predict_stacked

        meta = load_stacking_meta_for_task(adir, task)
        alpha = (
            [meta["alpha_xgb"], meta["alpha_lgbm"], meta["alpha_cat"]] if meta else [1.0, 0.0, 0.0]
        )

        arts = load_booster_artifacts(adir, task)
        # Load boosters and predict
        import lightgbm as lgb
        import xgboost as xgb

        if tt == "regression":
            xgb_m = xgb.XGBRegressor()
        else:
            xgb_m = xgb.XGBClassifier()
        xgb_m.load_model(str(arts["xgb"]))
        if tt == "regression":
            xgb_raw = np.asarray(xgb_m.predict(x_val), dtype=float)
        else:
            xgb_raw = np.asarray(xgb_m.predict_proba(x_val), dtype=float)
            if xgb_raw.ndim > 1:
                xgb_raw = xgb_raw[:, 1]
        xgb_pred = np.asarray(xgb_raw, dtype=float).ravel()

        lgb_m = lgb.Booster(model_file=str(arts["lgbm"]))
        lgb_raw = np.asarray(lgb_m.predict(x_val), dtype=float)
        if lgb_raw.ndim > 1:
            lgb_raw = lgb_raw[:, 1]
        # LightGBM Booster.predict returns raw scores; apply sigmoid for binary
        if tt != "regression":
            lgb_raw = 1.0 / (1.0 + np.exp(-np.clip(lgb_raw, -50, 50)))
        lgb_pred = np.asarray(lgb_raw, dtype=float).ravel()

        from catboost import CatBoostClassifier, CatBoostRegressor

        if tt == "regression":
            cat_m = CatBoostRegressor()
        else:
            cat_m = CatBoostClassifier()
        cat_m.load_model(str(arts["cat"]))
        if tt == "regression":
            cat_raw = np.asarray(cat_m.predict(x_val), dtype=float)
        else:
            cat_raw = np.asarray(cat_m.predict_proba(x_val), dtype=float)
            if cat_raw.ndim > 1:
                cat_raw = cat_raw[:, 1]
        cat_pred = np.asarray(cat_raw, dtype=float).ravel()

        booster_pred = predict_stacked(
            [xgb_pred, lgb_pred, cat_pred], np.asarray(alpha, dtype=float)
        )
        if tt == "multi" and booster_pred.ndim == 1 and booster_pred.size > len(x_val):
            booster_pred = booster_pred.reshape(len(x_val), -1)

        # For classification, use sigmoid of NN output as probability
        if tt == "binary":
            nn_pred = 1.0 / (1.0 + np.exp(-np.clip(nn_pred, -50, 50)))
            booster_pred = 1.0 / (1.0 + np.exp(-np.clip(booster_pred, -50, 50)))
        elif tt == "multi":
            # For multi-class, use softmax
            def _softmax(a):
                e = (
                    np.exp(a - np.max(a, axis=-1, keepdims=True))
                    if a.ndim > 1
                    else np.exp(a - np.max(a))
                )
                return e / e.sum(axis=-1, keepdims=True) if a.ndim > 1 else e / e.sum()

            nn_raw_2d = logits[task].detach().numpy()
            if nn_raw_2d.ndim == 1:
                nn_raw_2d = nn_raw_2d.reshape(-1, 1)
            nn_pred = _softmax(nn_raw_2d)
            # Booster: use predict_proba
            booster_pred = (
                booster_pred.reshape(-1, 3) if booster_pred.size == len(x_val) * 3 else booster_pred
            )

        beta_claim, diag = optimize_beta(
            nn_pred, booster_pred, np.asarray(y_val[task]).ravel().astype(float), task_type=tt
        )
        beta_weights[task] = beta_claim
        diagnostics[task] = diag
        boundary = diag["boundary_optimum"]
        from sklearn.metrics import roc_auc_score

        try:
            auc_p1 = roc_auc_score(y_val[task], booster_pred)
            auc_1mp1 = roc_auc_score(y_val[task], 1.0 - booster_pred)
        except ValueError:
            auc_p1 = float("nan")
            auc_1mp1 = float("nan")
        try:
            from sklearn.utils import shuffle

            auc_shuf = roc_auc_score(shuffle(y_val[task], random_state=0), booster_pred)
        except ValueError:
            auc_shuf = float("nan")
        print(
            f"    [{task}] booster_AUC(p1)={auc_p1:.4f} AUC(1-p1)={auc_1mp1:.4f} AUC(shuffled_y)={auc_shuf:.4f} p1.mean()={booster_pred.mean():.4f} y.mean()={np.asarray(y_val[task]).mean():.4f}"
        )

        try:
            booster_auc = roc_auc_score(y_val[task], booster_pred)
        except ValueError:
            booster_auc = float("nan")
        print(
            "    "
            + task
            + ": status="
            + str(diag["task_status"])
            + " beta_claim="
            + str(beta_claim)
            + " loss_nn="
            + f"{diag['loss_before_nn']:.4f}"
            + " loss_booster="
            + f"{diag['loss_before_booster']:.4f}"
            + " loss_hybrid="
            + f"{diag['loss_after_fusion']:.4f}"
            + " baseline="
            + f"{diag['loss_baseline']:.4f}"
            + " boundary="
            + str(boundary)
            + " booster_AUC="
            + f"{booster_auc:.4f}"
        )

    beta_path = output_dir / "beta_weights.json"
    save_beta_weights(beta_path, beta_weights, diagnostics=diagnostics)
    return beta_weights


def _hybrid_eval(output_dir: Path) -> dict[str, Any]:
    """Stage 7: Hybrid evaluation."""

    from insurance_fl.fl.client_app import hybrid_evaluate
    from insurance_fl.fl.task import build_model, load_partition

    data_dir = Path("data/processed/fl_task0_smoke")
    client_id = "germany_allianz"
    x_test, y_test = load_partition(data_dir, client_id, split="test")

    nn_path = output_dir / "nn_checkpoint"
    if nn_path.exists():
        from insurance_fl.models.multi_task_nn import MultiTaskNN

        net = MultiTaskNN.load(nn_path)
    else:
        net = build_model(input_dim=int(x_test.shape[1]))

    net.eval()
    adir = output_dir / "boosters" / client_id
    beta_path = output_dir / "beta_weights.json"
    import json

    beta_weights = {}
    if beta_path.exists():
        beta_weights = json.loads(beta_path.read_text()).get("beta_weights", {})

    metrics = hybrid_evaluate(
        net,
        x_test[:20],
        {k: v[:20] for k, v in y_test.items()},
        artifact_dir=adir,
        beta_weights=beta_weights,
        task_list=["claim_risk_score"],
    )
    return metrics


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_id = args.run_id or f"run_{int(time.time())}"
    output_dir = Path(args.output_dir) / run_id
    data_dir = Path(args.data_dir)

    print(f"=== E2E Pipeline: {run_id} ===")
    print(f"rounds={args.rounds} output={output_dir}")

    if args.dry_run:
        print("Dry run: validating config only")
        return 0

    # Stage 1: Certs (skip for simulation mode)
    print("\n[1/8] Certs: skipped (simulation mode)")

    # Stage 2: Data preparation
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"ERROR: data_dir not found: {data_dir}", file=__import__("sys").stderr)
        return 1
    print("[2/8] Data preparation...")
    _ensure_data(data_dir)
    print(f"  data ready: {data_dir}")

    # Stage 3: Boosters
    print("[3/8] Training boosters...")
    try:
        booster_results = _train_boosters(data_dir, output_dir)
    except Exception as exc:
        print(f"  ERROR: booster training failed: {exc}", file=__import__("sys").stderr)
        return 1
    print(f"  boosters trained for {len(booster_results)} clients")

    # Stage 4: FL training
    print(f"[4/8] Running Flower FL ({args.rounds} rounds)...")
    rc = _run_fl_training(args.rounds, output_dir)
    if rc != 0:
        print(f"  ERROR: FL training failed with exit code {rc}", file=__import__("sys").stderr)
        return 1
    print("  FL training complete")

    # Stage 5: Checkpoint (save a model for evaluation)
    print("[5/8] Saving NN checkpoint...")
    from insurance_fl.fl.task import build_model, load_partition

    x_train, y_train = load_partition(data_dir, "germany_allianz", split="train")
    net = build_model(input_dim=int(x_train.shape[1]))
    from insurance_fl.fl.task import train

    train(net, x_train, y_train, epochs=30, batch_size=32, lr=0.001)
    ckpt_dir = output_dir / "nn_checkpoint"
    net.save(ckpt_dir)
    # Reload to verify
    from insurance_fl.models.multi_task_nn import MultiTaskNN

    MultiTaskNN.load(ckpt_dir)
    print(f"  checkpoint saved and reloaded: {ckpt_dir}")

    # Stage 6: Beta optimization
    print("[6/8] Optimizing beta...")
    beta_weights = _optimize_beta(output_dir)
    print(f"  beta: {beta_weights}")

    # Stage 7: Hybrid eval
    print("[7/8] Hybrid evaluation...")
    metrics = _hybrid_eval(output_dir)
    print(f"  eval mode: {metrics.get('evaluation_mode')}")
    print(f"  metrics keys: {list(metrics.keys())[:5]}")

    # Stage 8: Report
    print("[8/8] Writing report...")
    report = {
        "run_id": run_id,
        "rounds": args.rounds,
        "booster_clients": list(booster_results.keys()),
        "beta_weights": beta_weights,
        "eval_mode": metrics.get("evaluation_mode"),
        "metrics": {
            k: (v if not hasattr(v, "item") else v.item())
            for k, v in metrics.items()
            if not isinstance(v, (list, dict))
        },
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"  report written: {output_dir / 'report.json'}")

    print(f"\n=== Pipeline complete: {run_id} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
