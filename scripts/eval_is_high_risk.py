#!/usr/bin/env python3
"""Evaluate is_high_risk shipped-model gate per client on v2 canonical data.

Produces/updates reports/is_high_risk_auc.json incrementally (A1 baselines, etc.).
Selection/early stopping use validation only; test evaluated once per reported run.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
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

from insurance_fl.fl.task import CLIENT_IDS, build_model, evaluate, load_partition, train  # noqa: E402
from insurance_fl.models.multi_task_nn import MultiTaskNN  # noqa: E402

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
            ["git", "status", "--porcelain", "-uno", "--", ".", ":(exclude)reports", ":(exclude)artifacts"],
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
    ci_low_f = float(ci_low)
    status = "below" if auc_f < target else ("supported" if ci_low_f >= target else "marginal")
    return {
        "auc": auc_f,
        "ci_low": ci_low_f,
        "meets_target": auc_f >= target,
        "status": status,
    }


def bootstrap_auc_ci(y_true: np.ndarray, y_score: np.ndarray, seed: int, n_boot: int = 1000) -> tuple[float, float, float]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    auc = float(roc_auc_score(y_true, y_score))
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
    return (x_tr, y_tr[TASK]), (x_va, y_va[TASK]), (x_te, y_te[TASK])


def majority_baseline(train_y: np.ndarray, test_y: np.ndarray, seed: int) -> dict[str, Any]:
    # Constant-score predictor => AUC 0.5 (uninformative ranking). Report majority class for reference.
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
    # Selection/early stopping on validation only: use early_stopping with internal split on train
    # (does not touch test). Report val AUC for transparency.
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
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def predict_is_high_risk_proba(model: MultiTaskNN, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.as_tensor(x, dtype=torch.float32))
        out = logits[TASK]
        return sigmoid(out.detach().cpu().numpy().reshape(-1))


def standardize_train(train_x, other_xs):
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    def tr(x):
        return ((x - mean) / std).astype(np.float32)
    return tr(train_x), [tr(x) for x in other_xs], {"mean": mean.tolist(), "std": std.tolist()}


def nn_val_auc(model: MultiTaskNN, x_va: np.ndarray, y_va: np.ndarray) -> float:
    p = predict_is_high_risk_proba(model, x_va)
    if len(np.unique(y_va)) < 2:
        return float("nan")
    return float(roc_auc_score(y_va, p))


def train_nn_local(
    x_tr, y_tr_dict, x_va, y_va, *, seed: int, hidden_dims, dropout, lr, weight_decay,
    max_epochs: int, patience: int, batch_size: int, single_task: bool,
) -> tuple[MultiTaskNN, dict[str, Any]]:
    set_seeds(seed)
    input_dim = int(x_tr.shape[1])
    if single_task:
        task_types = {TASK: "binary"}
        y_tr_use = {TASK: y_tr_dict[TASK]}
    else:
        task_types = None  # default 5 tasks
        y_tr_use = y_tr_dict
    model = MultiTaskNN(input_dim=input_dim, task_types=task_types, hidden_dims=list(hidden_dims), dropout_rate=dropout)
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
            opt.zero_grad()
            logits = model(xt[idx])
            losses = model.get_losses(logits, {k: v[idx] for k, v in yt.items()})
            loss = model.get_kendall_total_loss(losses) if not single_task else model.get_total_loss(losses)
            loss.backward()
            opt.step()
            last = float(loss.detach().item())
        va = nn_val_auc(model, x_va, y_va)
        history.append({"epoch": epoch, "train_loss": last, "val_auc": va})
        if np.isfinite(va) and va > best_auc + 1e-6:
            best_auc = va
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            bad = 0
        else:
            bad += 1
        if bad >= patience:
            break
    model.load_state_dict(best_state)
    info = {
        "best_epoch": best_epoch,
        "best_val_auc": float(best_auc),
        "epochs_run": len(history),
        "history_tail": history[-5:],
    }
    return model, info


def run_a1(out_path: Path = DEFAULT_OUT) -> dict[str, Any]:
    report = {
        "task": TASK,
        "data_dir": str(DATA_DIR),
        **provenance(),
        **git_state(),
        "clients": {},
        "notes": [],
    }
    for cid in CLIENT_IDS:
        (x_tr, y_tr), (x_va, y_va), (x_te, y_te) = load_client_splits(cid)
        client = {
            "splits": {"train": len(y_tr), "val": len(y_va), "test": len(y_te)},
            "pos_rate": {
                "train": float(np.mean(y_tr)),
                "val": float(np.mean(y_va)),
                "test": float(np.mean(y_te)),
            },
            "baselines": {
                "majority": majority_baseline(y_tr, y_te, seed=42),
                "gbm": gbm_baseline((x_tr, y_tr), (x_va, y_va), (x_te, y_te), seed=42),
            },
        }
        report["clients"][cid] = client
        print(f"A1 {cid}: majority_auc={client['baselines']['majority']['auc']:.4f} gbm_auc={client['baselines']['gbm']['auc']:.4f} ci={client['baselines']['gbm']['ci95']}", flush=True)
    report["notes"].append("A1: majority baseline and GBM (HGB early stopping on train holdout; val/test AUC reported; test evaluated once).")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    return report


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="a1", choices=["a1"])
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args(argv)
    if args.stage == "a1":
        run_a1(Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
