#!/usr/bin/env python3
"""Evaluate is_high_risk shipped-model gate per client on v2 canonical data.

Produces/updates reports/is_high_risk_auc.json incrementally.
Selection/early stopping use validation only; test evaluated once per reported run.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from insurance_fl.fl.task import CLIENT_IDS, load_partition  # noqa: E402
from insurance_fl.models.multi_task_nn import DEFAULT_TASK_TYPES, MultiTaskNN  # noqa: E402

MANIFEST = ROOT / "docs" / "evidence" / "dataset_manifest.json"
DEFAULT_OUT = ROOT / "reports" / "is_high_risk_auc.json"
DATA_DIR = ROOT / "data" / "processed" / "european_insurance_federated"
TASK = "is_high_risk"


def git_state(cwd: Path = ROOT) -> dict[str, Any]:
    state = {"git_sha": None, "git_dirty": None}
    try:
        state["git_sha"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(cwd), text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        pass
    try:
        dirty = subprocess.check_output(
            [
                "git",
                "status",
                "--porcelain",
                "-uno",
                "--",
                ".",
                ":(exclude)reports",
                ":(exclude)artifacts",
                ":(exclude)docs/evidence",
            ],
            cwd=str(cwd),
            text=True,
            stderr=subprocess.DEVNULL,
        )
        state["git_dirty"] = bool(dirty.strip())
    except Exception:
        pass
    return state


def provenance() -> dict[str, Any]:
    if MANIFEST.exists():
        m = json.loads(MANIFEST.read_text())
        return {
            "dataset_version": m.get("dataset_version"),
            "config_hash": m.get("config_hash"),
            "manifest_seed": m.get("seed"),
            "manifest_samples_per_client": m.get("samples_per_client"),
        }
    return {"dataset_version": None, "config_hash": None}


def gate(auc: float, ci_low: float, target: float = 0.80) -> dict[str, Any]:
    auc_f = float(auc)
    ci_low_f = float(ci_low) if ci_low is not None and np.isfinite(ci_low) else float("nan")
    status = "below" if auc_f < target else ("supported" if np.isfinite(ci_low_f) and ci_low_f >= target else "marginal")
    return {
        "auc": auc_f,
        "ci_low": ci_low_f,
        "meets_target": auc_f >= target,
        "status": status,
    }


def bootstrap_auc_ci(y_true: np.ndarray, y_score: np.ndarray, seed: int, n_boot: int = 1000) -> tuple[float, float, float]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    auc = float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else float("nan")
    if len(np.unique(y_true)) < 2:
        return auc, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    aucs = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy = y_true[idx]
        if len(np.unique(yy)) < 2:
            continue
        try:
            aucs.append(float(roc_auc_score(yy, y_score[idx])))
        except Exception:
            continue
    if not aucs:
        return auc, float("nan"), float("nan")
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return auc, float(lo), float(hi)


def load_client_splits(client_id: str, data_dir: Path = DATA_DIR):
    x_tr, y_tr = load_partition(data_dir, client_id, split="train")
    x_va, y_va = load_partition(data_dir, client_id, split="val")
    x_te, y_te = load_partition(data_dir, client_id, split="test")
    return (x_tr, y_tr), (x_va, y_va), (x_te, y_te)


def majority_baseline(train_y: np.ndarray, test_y: np.ndarray, seed: int) -> dict[str, Any]:
    p = np.full(len(test_y), float(np.mean(train_y)), dtype=float)
    auc, lo, hi = bootstrap_auc_ci(test_y, p, seed=seed)
    maj = int(np.mean(train_y) >= 0.5)
    acc = float(np.mean((p >= 0.5).astype(int) == test_y.astype(int)))
    return {
        "model": "majority_baseline",
        "auc": auc,
        "ci95": [lo, hi],
        "gate": gate(auc, lo),
        "majority_class_train": maj,
        "accuracy_constant_prob": acc,
        "note": "constant-probability predictor; AUC expected ~0.5",
    }


def gbm_baseline(train_xy, val_xy, test_xy, seed: int) -> dict[str, Any]:
    (x_tr, y_tr), (x_va, y_va), (x_te, y_te) = train_xy, val_xy, test_xy
    clf = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_depth=None,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=seed,
    )
    clf.fit(x_tr, y_tr.astype(int))
    val_p = clf.predict_proba(x_va)[:, 1]
    test_p = clf.predict_proba(x_te)[:, 1]
    val_auc = float(roc_auc_score(y_va, val_p))
    auc, lo, hi = bootstrap_auc_ci(y_te, test_p, seed=seed + 1000)
    return {
        "model": "gbm_hist",
        "auc": auc,
        "ci95": [lo, hi],
        "gate": gate(auc, lo),
        "val_auc": val_auc,
        "n_iter": int(getattr(clf, "n_iter_", 0)),
        "params": {
            "max_iter": 300,
            "learning_rate": 0.05,
            "early_stopping": True,
            "validation_fraction": 0.1,
            "random_state": seed,
        },
        "selection": "validation_only (HGB internal early stopping on train holdout); test evaluated once",
    }


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def predict_is_high_risk_proba(model: MultiTaskNN, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.as_tensor(x, dtype=torch.float32))
        out = logits[TASK]
        return sigmoid_np(out.detach().cpu().numpy().reshape(-1))


def standardize_fit(train_x: np.ndarray):
    mean = train_x.mean(axis=0).astype(np.float32)
    std = train_x.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
    return mean, std


def standardize_apply(x: np.ndarray, mean, std):
    return ((x - mean) / std).astype(np.float32)


def nn_val_auc(model: MultiTaskNN, x_va: np.ndarray, y_va: np.ndarray) -> float:
    p = predict_is_high_risk_proba(model, x_va)
    if len(np.unique(y_va)) < 2:
        return float("nan")
    return float(roc_auc_score(y_va, p))


def train_nn(
    x_tr, y_tr_dict, x_va, y_va_binary, *, seed: int, hidden_dims, dropout, lr, weight_decay,
    max_epochs: int, patience: int, batch_size: int, single_task: bool,
) -> tuple[MultiTaskNN, dict[str, Any]]:
    set_seeds(seed)
    input_dim = int(x_tr.shape[1])
    if single_task:
        task_types = {TASK: "binary"}
        y_tr_use = {TASK: y_tr_dict[TASK]}
    else:
        task_types = dict(DEFAULT_TASK_TYPES)
        y_tr_use = y_tr_dict
    model = MultiTaskNN(
        input_dim=input_dim,
        task_types=task_types,
        hidden_dims=list(hidden_dims),
        dropout_rate=float(dropout),
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    xt = torch.as_tensor(x_tr, dtype=torch.float32)
    yt = {k: torch.as_tensor(v) for k, v in y_tr_use.items()}
    n = int(xt.shape[0])
    best_auc = -1.0
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    bad = 0
    history = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(n)
        last = float("nan")
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            if idx.numel() < 2:
                continue
            opt.zero_grad(set_to_none=True)
            logits = model(xt[idx])
            losses = model.get_losses(logits, {k: v[idx] for k, v in yt.items()})
            if single_task:
                loss = model.get_total_loss(losses)
            else:
                loss = model.get_kendall_total_loss(losses)
            loss.backward()
            opt.step()
            last = float(loss.detach().item())
        va = nn_val_auc(model, x_va, y_va_binary)
        history.append({"epoch": epoch, "train_loss": last, "val_auc": va})
        score = va if np.isfinite(va) else -1.0
        if score > best_auc + 1e-6:
            best_auc = score
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            bad = 0
        else:
            bad += 1
        if bad >= patience:
            break
    model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch,
        "best_val_auc": float(best_auc),
        "epochs_run": len(history),
    }


def load_report(out_path: Path) -> dict[str, Any]:
    if out_path.exists():
        return json.loads(out_path.read_text())
    return {"task": TASK, "data_dir": str(DATA_DIR), **provenance(), **git_state(), "clients": {}, "notes": []}


def save_report(report: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str))


def run_a1(out_path: Path = DEFAULT_OUT) -> dict[str, Any]:
    report = load_report(out_path)
    report.update({"task": TASK, "data_dir": str(DATA_DIR), **provenance(), **git_state()})
    report.setdefault("clients", {})
    report.setdefault("notes", [])
    for cid in CLIENT_IDS:
        (x_tr, y_trd), (x_va, y_vad), (x_te, y_ted) = load_client_splits(cid)
        y_tr, y_va, y_te = y_trd[TASK], y_vad[TASK], y_ted[TASK]
        client = report["clients"].get(cid, {})
        client.update({
            "splits": {"train": int(len(y_tr)), "val": int(len(y_va)), "test": int(len(y_te))},
            "pos_rate": {
                "train": float(np.mean(y_tr)),
                "val": float(np.mean(y_va)),
                "test": float(np.mean(y_te)),
            },
            "baselines": {
                "majority": majority_baseline(y_tr, y_te, seed=42),
                "gbm": gbm_baseline((x_tr, y_tr), (x_va, y_va), (x_te, y_te), seed=42),
            },
        })
        report["clients"][cid] = client
        print(f"A1 {cid}: majority_auc={client['baselines']['majority']['auc']:.4f} gbm_auc={client['baselines']['gbm']['auc']:.4f} ci={client['baselines']['gbm']['ci95']}", flush=True)
    note = "A1: majority baseline and GBM (HGB early stopping on train holdout; val/test AUC reported; test evaluated once)."
    if note not in report["notes"]:
        report["notes"].append(note)
    save_report(report, out_path)
    return report


def nn_grid() -> list[dict[str, Any]]:
    grid = []
    for head in ["single", "multi"]:
        for hidden in [[128, 64], [64, 32]]:
            for lr in [1e-3, 5e-4]:
                for dropout in [0.2, 0.3]:
                    grid.append({
                        "head": head,
                        "hidden_dims": hidden,
                        "lr": lr,
                        "dropout": dropout,
                        "weight_decay": 0.0,
                        "max_epochs": 60,
                        "patience": 10,
                        "batch_size": 64,
                    })
    return grid


def run_a2(out_path: Path = DEFAULT_OUT) -> dict[str, Any]:
    report = load_report(out_path)
    report.update({**provenance(), **git_state()})
    report.setdefault("clients", {})
    report.setdefault("notes", [])
    grid = nn_grid()
    seeds = [42, 43, 44]

    # Preload all client data
    data = {}
    for cid in CLIENT_IDS:
        (x_tr, y_trd), (x_va, y_vad), (x_te, y_ted) = load_client_splits(cid)
        mu, sd = standardize_fit(x_tr)
        data[cid] = {
            "x_tr": standardize_apply(x_tr, mu, sd),
            "y_trd": y_trd,
            "x_va": standardize_apply(x_va, mu, sd),
            "y_vad": y_vad,
            "x_te": standardize_apply(x_te, mu, sd),
            "y_ted": y_ted,
        }

    # Local-only per client
    for cid in CLIENT_IDS:
        d = data[cid]
        grid_log = []
        best_cfg = None
        best_val = -1.0
        for i, cfg in enumerate(grid):
            model, info = train_nn(
                d["x_tr"], d["y_trd"], d["x_va"], d["y_vad"][TASK],
                seed=42,
                hidden_dims=cfg["hidden_dims"], dropout=cfg["dropout"], lr=cfg["lr"],
                weight_decay=cfg["weight_decay"], max_epochs=cfg["max_epochs"],
                patience=cfg["patience"], batch_size=cfg["batch_size"], single_task=(cfg["head"]=="single"),
            )
            val_auc = float(info["best_val_auc"])
            entry = {"cfg_id": i, **cfg, "selection_seed": 42, "val_auc": val_auc, **info}
            grid_log.append(entry)
            if np.isfinite(val_auc) and val_auc > best_val:
                best_val = val_auc
                best_cfg = cfg
        # evaluate best with 3 seeds
        runs = {}
        preds_by_seed = {}
        for s in seeds:
            model, info = train_nn(
                d["x_tr"], d["y_trd"], d["x_va"], d["y_vad"][TASK],
                seed=s,
                hidden_dims=best_cfg["hidden_dims"], dropout=best_cfg["dropout"], lr=best_cfg["lr"],
                weight_decay=best_cfg["weight_decay"], max_epochs=best_cfg["max_epochs"],
                patience=best_cfg["patience"], batch_size=best_cfg["batch_size"], single_task=(best_cfg["head"]=="single"),
            )
            p = predict_is_high_risk_proba(model, d["x_te"])
            auc, lo, hi = bootstrap_auc_ci(d["y_ted"][TASK], p, seed=2000 + s)
            runs[str(s)] = {"test_auc": auc, "ci95": [lo, hi], "gate": gate(auc, lo), "best_val_auc": info["best_val_auc"], "best_epoch": info["best_epoch"]}
            preds_by_seed[str(s)] = p.tolist()
        median_seed = sorted(seeds, key=lambda s: runs[str(s)]["test_auc"])[1]
        med = runs[str(median_seed)]
        client = report["clients"].setdefault(cid, {})
        client.setdefault("neural", {})
        client["neural"]["local_only"] = {
            "best_config": best_cfg,
            "selection": "grid search on validation AUC (seed 42); test evaluated once per seed",
            "grid": grid_log,
            "seeds": seeds,
            "runs": runs,
            "median_seed": median_seed,
            "median_test_auc": med["test_auc"],
            "median_ci95": med["ci95"],
            "median_gate": med["gate"],
        }
        print(f"A2 local {cid}: best_val={best_val:.4f} cfg={best_cfg} median_test={med['test_auc']:.4f}", flush=True)

    # Pooled NN
    x_tr_all = np.vstack([data[c]["x_tr"] for c in CLIENT_IDS])
    y_trd_all = {t: np.concatenate([data[c]["y_trd"][t] for c in CLIENT_IDS]) for t in DEFAULT_TASK_TYPES}
    x_va_all = np.vstack([data[c]["x_va"] for c in CLIENT_IDS])
    y_va_all = np.concatenate([data[c]["y_vad"][TASK] for c in CLIENT_IDS])
    grid_log_p = []
    best_cfg_p = None
    best_val_p = -1.0
    for i, cfg in enumerate(grid):
        model, info = train_nn(
            x_tr_all, y_trd_all, x_va_all, y_va_all,
            seed=42,
            hidden_dims=cfg["hidden_dims"], dropout=cfg["dropout"], lr=cfg["lr"],
            weight_decay=cfg["weight_decay"], max_epochs=cfg["max_epochs"],
            patience=cfg["patience"], batch_size=cfg["batch_size"], single_task=(cfg["head"]=="single"),
        )
        val_auc = float(info["best_val_auc"])
        entry = {"cfg_id": i, **cfg, "selection_seed": 42, "val_auc_pooled": val_auc, **info}
        grid_log_p.append(entry)
        if np.isfinite(val_auc) and val_auc > best_val_p:
            best_val_p = val_auc
            best_cfg_p = cfg
    for cid in CLIENT_IDS:
        d = data[cid]
        runs = {}
        for s in seeds:
            model, info = train_nn(
                x_tr_all, y_trd_all, x_va_all, y_va_all,
                seed=s,
                hidden_dims=best_cfg_p["hidden_dims"], dropout=best_cfg_p["dropout"], lr=best_cfg_p["lr"],
                weight_decay=best_cfg_p["weight_decay"], max_epochs=best_cfg_p["max_epochs"],
                patience=best_cfg_p["patience"], batch_size=best_cfg_p["batch_size"], single_task=(best_cfg_p["head"]=="single"),
            )
            p = predict_is_high_risk_proba(model, d["x_te"])
            auc, lo, hi = bootstrap_auc_ci(d["y_ted"][TASK], p, seed=3000 + s)
            runs[str(s)] = {"test_auc": auc, "ci95": [lo, hi], "gate": gate(auc, lo), "best_val_auc_pooled": info["best_val_auc"], "best_epoch": info["best_epoch"]}
        median_seed = sorted(seeds, key=lambda s: runs[str(s)]["test_auc"])[1]
        med = runs[str(median_seed)]
        client = report["clients"].setdefault(cid, {})
        client.setdefault("neural", {})
        client["neural"]["pooled"] = {
            "best_config": best_cfg_p,
            "selection": "grid search on pooled validation AUC (seed 42); evaluated on each client's test split once per seed",
            "grid": grid_log_p,
            "seeds": seeds,
            "runs": runs,
            "median_seed": median_seed,
            "median_test_auc": med["test_auc"],
            "median_ci95": med["ci95"],
            "median_gate": med["gate"],
            "standardization": "per-client train statistics (same as local inputs in this run)",
        }
        print(f"A2 pooled {cid}: best_val_pooled={best_val_p:.4f} median_test={med['test_auc']:.4f}", flush=True)

    note = "A2: local-only and pooled NN (train-std, val-AUC early stopping, single vs multi-task head grid, 3 seeds 42/43/44; class weighting skipped ~0.52 balance)."
    if note not in report["notes"]:
        report["notes"].append(note)
    save_report(report, out_path)
    return report



def fedavg_fedbn_reference(
    data: dict[str, dict], *, rounds: int, local_epochs: int, lr: float, seed: int,
    hidden_dims, dropout: float, single_task: bool, batch_size: int = 64,
) -> tuple[dict[str, MultiTaskNN], dict[str, Any]]:
    set_seeds(seed)
    # initialize per-client models with same federated params, local BN kept per client
    models: dict[str, MultiTaskNN] = {}
    for cid in CLIENT_IDS:
        input_dim = int(data[cid]["x_tr"].shape[1])
        task_types = {TASK: "binary"} if single_task else dict(DEFAULT_TASK_TYPES)
        models[cid] = MultiTaskNN(
            input_dim=input_dim, task_types=task_types, hidden_dims=list(hidden_dims), dropout_rate=float(dropout)
        )
    global_w = [np.array(w, copy=True) for w in models[CLIENT_IDS[0]].get_federated_weights()]
    for cid in CLIENT_IDS:
        models[cid].set_federated_weights(global_w)

    curve = []
    for r in range(1, rounds + 1):
        updates, sizes = [], []
        for cid in CLIENT_IDS:
            d = data[cid]
            m = models[cid]
            m.set_federated_weights(global_w)  # local BN untouched
            # local train (few epochs) - simple SGD/Adam step loop
            y_use = {TASK: d["y_trd"][TASK]} if single_task else d["y_trd"]
            _train_nn_epochs_inplace(
                m, d["x_tr"], y_use, epochs=local_epochs, lr=lr, batch_size=batch_size,
                single_task=single_task, seed=seed * 1000 + r * 10 + CLIENT_IDS.index(cid),
            )
            updates.append([np.array(w, copy=True) for w in m.get_federated_weights()])
            sizes.append(int(len(d["x_tr"])))
        total = float(sum(sizes))
        weights = [s / total for s in sizes]
        new_global = []
        for k in range(len(global_w)):
            acc = np.zeros_like(updates[0][k], dtype=np.float64)
            for wi, u in zip(weights, updates):
                acc += wi * u[k].astype(np.float64)
            new_global.append(acc.astype(updates[0][k].dtype))
        global_w = new_global
        # set global to all for evaluation; each keeps own BN
        per_val = {}
        for cid in CLIENT_IDS:
            models[cid].set_federated_weights(global_w)
            va = nn_val_auc(models[cid], data[cid]["x_va"], data[cid]["y_vad"][TASK])
            per_val[cid] = float(va) if np.isfinite(va) else None
        mean_val = float(np.nanmean([v for v in per_val.values() if v is not None]))
        curve.append({"round": r, "mean_val_auc": mean_val, "per_client_val_auc": per_val})
        if r == 1 or r % 5 == 0 or r == rounds:
            print(f"  fed round {r}/{rounds} mean_val_auc={mean_val:.4f}", flush=True)
    for cid in CLIENT_IDS:
        models[cid].set_federated_weights(global_w)
    info = {
        "rounds": rounds,
        "local_epochs": local_epochs,
        "lr": lr,
        "batch_size": batch_size,
        "hidden_dims": list(hidden_dims),
        "dropout": float(dropout),
        "single_task": single_task,
        "seed": seed,
        "validation_curve": curve,
        "final_mean_val_auc": curve[-1]["mean_val_auc"] if curve else None,
    }
    return models, info


def _train_nn_epochs_inplace(model, x_tr, y_tr_dict, *, epochs: int, lr: float, batch_size: int, single_task: bool, seed: int):
    set_seeds(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    xt = torch.as_tensor(x_tr, dtype=torch.float32)
    yt = {k: torch.as_tensor(v) for k, v in y_tr_dict.items()}
    n = int(xt.shape[0])
    for _ in range(max(1, int(epochs))):
        model.train()
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            if idx.numel() < 2:
                continue
            opt.zero_grad(set_to_none=True)
            logits = model(xt[idx])
            losses = model.get_losses(logits, {k: v[idx] for k, v in yt.items()})
            loss = model.get_total_loss(losses) if single_task else model.get_kendall_total_loss(losses)
            loss.backward()
            opt.step()


def run_a3(out_path: Path = DEFAULT_OUT) -> dict[str, Any]:
    report = load_report(out_path)
    report.update({**provenance(), **git_state()})
    report.setdefault("clients", {})
    report.setdefault("notes", [])

    seeds = [42, 43, 44]
    rounds = 20
    # Config chosen on validation: small grid on seed 42 using pooled-best architecture family
    # (single-task head performed best in A2; [128,64], dropout 0.3). Tune local_epochs/lr on mean val AUC.
    base_hidden = [128, 64]
    base_dropout = 0.3
    single_task = True
    fed_grid = [
        {"local_epochs": 1, "lr": 1e-3},
        {"local_epochs": 2, "lr": 1e-3},
        {"local_epochs": 1, "lr": 5e-4},
        {"local_epochs": 2, "lr": 5e-4},
    ]

    data = {}
    for cid in CLIENT_IDS:
        (x_tr, y_trd), (x_va, y_vad), (x_te, y_ted) = load_client_splits(cid)
        mu, sd = standardize_fit(x_tr)
        data[cid] = {
            "x_tr": standardize_apply(x_tr, mu, sd),
            "y_trd": y_trd,
            "x_va": standardize_apply(x_va, mu, sd),
            "y_vad": y_vad,
            "x_te": standardize_apply(x_te, mu, sd),
            "y_ted": y_ted,
        }

    fed_grid_log = []
    best_fed = None
    best_fed_val = -1.0
    for i, cfg in enumerate(fed_grid):
        print(f"A3 fed grid {i+1}/{len(fed_grid)} cfg={cfg}", flush=True)
        _, info = fedavg_fedbn_reference(
            data, rounds=rounds, local_epochs=cfg["local_epochs"], lr=cfg["lr"], seed=42,
            hidden_dims=base_hidden, dropout=base_dropout, single_task=single_task,
        )
        val = float(info["final_mean_val_auc"] or float("nan"))
        entry = {"cfg_id": i, **cfg, "rounds": rounds, "hidden_dims": base_hidden, "dropout": base_dropout, "single_task": single_task, "selection_seed": 42, "final_mean_val_auc": val}
        fed_grid_log.append(entry)
        if np.isfinite(val) and val > best_fed_val:
            best_fed_val = val
            best_fed = cfg
    print(f"A3 best fed cfg={best_fed} final_mean_val_auc={best_fed_val:.4f}", flush=True)

    for cid in CLIENT_IDS:
        client = report["clients"].setdefault(cid, {})
        client.setdefault("neural", {})
        client["neural"].setdefault("federated_reference", {})

    # run best config for 3 seeds and collect per-client results
    per_seed_models = {}
    per_seed_info = {}
    for s in seeds:
        print(f"A3 federated run seed={s}", flush=True)
        models, info = fedavg_fedbn_reference(
            data, rounds=rounds, local_epochs=best_fed["local_epochs"], lr=best_fed["lr"], seed=s,
            hidden_dims=base_hidden, dropout=base_dropout, single_task=single_task,
        )
        per_seed_models[s] = models
        per_seed_info[s] = info

    for cid in CLIENT_IDS:
        runs = {}
        for s in seeds:
            m = per_seed_models[s][cid]
            d = data[cid]
            p = predict_is_high_risk_proba(m, d["x_te"])
            auc, lo, hi = bootstrap_auc_ci(d["y_ted"][TASK], p, seed=4000 + s)
            runs[str(s)] = {
                "test_auc": auc,
                "ci95": [lo, hi],
                "gate": gate(auc, lo),
                "final_mean_val_auc": per_seed_info[s]["final_mean_val_auc"],
            }
        median_seed = sorted(seeds, key=lambda s: runs[str(s)]["test_auc"])[1]
        med = runs[str(median_seed)]
        report["clients"][cid]["neural"]["federated_reference"] = {
            "method": "in-process FedAvg + FedBN (non-BN federated weights averaged; local BN state preserved per client)",
            "best_config": {**best_fed, "rounds": rounds, "hidden_dims": base_hidden, "dropout": base_dropout, "single_task": single_task, "batch_size": 64},
            "selection": "small grid on mean client validation AUC at final round (seed 42); test evaluated once per seed",
            "grid": fed_grid_log,
            "seeds": seeds,
            "runs": runs,
            "median_seed": median_seed,
            "median_test_auc": med["test_auc"],
            "median_ci95": med["ci95"],
            "median_gate": med["gate"],
            "validation_curves_by_seed": {str(s): per_seed_info[s]["validation_curve"] for s in seeds},
            "standardization": "per-client train statistics (fit on each client's train split)",
        }
        print(f"A3 fed {cid}: median_test={med['test_auc']:.4f} ci={med['ci95']}", flush=True)

    note = "A3: in-process FedAvg+FedBN reference (~20 rounds), local epochs/lr chosen on validation grid (seed 42), 3 seeds 42/43/44, per-client local BN eval on own test split; validation curves recorded."
    if note not in report["notes"]:
        report["notes"].append(note)
    save_report(report, out_path)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="a1", choices=["a1", "a2", "a3"])
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args(argv)
    out = Path(args.out)
    if args.stage == "a1":
        run_a1(out)
    elif args.stage == "a2":
        run_a2(out)
    elif args.stage == "a3":
        run_a3(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
