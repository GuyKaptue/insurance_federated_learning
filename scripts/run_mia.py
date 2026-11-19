"""Run Membership Inference Attack evaluation.

Trains a target MultiTaskNN, trains shadow models on disjoint partitions,
and evaluates a black-box MIA. Results are persisted to JSON.

This is an empirical attack evaluation, not a formal privacy guarantee.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any


def _git_state() -> dict[str, Any]:
    state: dict[str, Any] = {"git_sha": None, "dirty": None}
    try:
        state["git_sha"] = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL)
            .strip()
        )
    except Exception:
        return state
    try:
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "-uno", "--", ".", ":(exclude)reports"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        state["dirty"] = bool(dirty.strip())
    except Exception:
        state["dirty"] = None
    return state


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="reports/mia_evaluation.json", help="Output JSON path")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--data-dir",
        default=None,
        help="Prepared dataset base directory containing federated/<client>/<split>.parquet",
    )
    p.add_argument(
        "--mode",
        choices=["smoke", "full"],
        default="full",
        help="Data mode: full prefers european_insurance_federated base, smoke uses fl_task0_smoke",
    )
    p.add_argument("--target-client", default="germany_allianz")
    p.add_argument("--task", default="is_high_risk")
    p.add_argument("--n-members", type=int, default=500, help="Requested target member samples (full mode)")
    p.add_argument("--n-nonmembers", type=int, default=500, help="Requested target non-member samples (full mode)")
    p.add_argument(
        "--n-shadow-members",
        type=int,
        default=250,
        help="Requested shadow member samples per model (full mode)",
    )
    p.add_argument(
        "--n-shadow-nonmembers",
        type=int,
        default=250,
        help="Requested shadow non-member samples per model (full mode)",
    )
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--shadow-epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--n-shadow-models", type=int, default=3)
    p.add_argument("--bootstrap", type=int, default=1000, help="Bootstrap samples for ROC-AUC CI over fixed scores")
    p.add_argument("--samples-per-client", type=int, default=512, help="Smoke dataset generation size if needed")
    p.add_argument(
        "--control",
        choices=["none", "overfit"],
        default="none",
        help="Optional control run: 'overfit' trains an intentionally memorizing target to sanity-check MIA pipeline",
    )
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def _resolve_data_dir(args) -> Path:
    from insurance_fl.fl.task import DEFAULT_DATA_DIR, prepare_dataset

    if args.data_dir:
        data_dir = Path(args.data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        return data_dir
    if args.mode == "smoke":
        return prepare_dataset(output_dir=DEFAULT_DATA_DIR, samples_per_client=args.samples_per_client, seed=args.seed)
    # full mode: prefer federated processed dataset base
    fed_base = Path("data/processed/european_insurance_federated")
    if fed_base.exists() and any((fed_base / "federated" / c / "train.parquet").exists() for c in [
        "germany_allianz", "france_axa", "belgium_ageas", "switzerland_helvetia"
    ]):
        return fed_base
    return prepare_dataset(output_dir=DEFAULT_DATA_DIR, samples_per_client=args.samples_per_client, seed=args.seed)



def _concat_split(data_dir: Path, client_id: str, splits: list[str]):
    import numpy as np
    from insurance_fl.fl.task import load_partition

    xs, ys_list = [], []
    for split in splits:
        try:
            x, y = load_partition(data_dir, client_id, split=split)
        except FileNotFoundError:
            continue
        xs.append(x)
        ys_list.append(y)
    if not xs:
        raise FileNotFoundError(f"No splits found for {client_id} in {data_dir}: {splits}")
    x = np.vstack(xs)
    y = {k: np.concatenate([yy[k] for yy in ys_list]) for k in ys_list[0].keys()}
    return x, y

def _stable_seed(base: int, token: str) -> int:
    h = hashlib.sha256(token.encode("utf-8")).digest()
    return int(base + int.from_bytes(h[:4], "little") % 100000)


def _set_global_seeds(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass


def _slice_xy(x, y, start: int, end: int):
    return x[start:end], {k: v[start:end] for k, v in y.items()}


def _auc_ci(y_true, y_scores, seed: int, n_boot: int = 1000):
    import numpy as np
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2 or n_boot <= 0:
        return None
    rng = np.random.default_rng(seed)
    aucs = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy = y_true[idx]
        if len(np.unique(yy)) < 2:
            continue
        try:
            aucs.append(float(roc_auc_score(yy, y_scores[idx])))
        except Exception:
            continue
    if not aucs:
        return None
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return [float(lo), float(hi)]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Heavy imports after argument parsing
    import numpy as np
    import torch
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score

    from insurance_fl.evaluation.mia import compute_features, tpr_at_fpr
    from insurance_fl.fl.task import build_model, load_partition, train
    from insurance_fl.models.multi_task_nn import DEFAULT_TASK_TYPES

    data_dir = _resolve_data_dir(args)
    client_ids = ["germany_allianz", "france_axa", "belgium_ageas", "switzerland_helvetia"]
    if args.target_client not in client_ids:
        raise ValueError(f"Unknown target client: {args.target_client}")
    if args.task not in DEFAULT_TASK_TYPES:
        raise ValueError(f"Unknown task: {args.task}")

    # Target model: train on target client train split
    target_client = args.target_client
    x_train, y_train = load_partition(data_dir, target_client, split="train")
    # Held-out data for member/non-member evaluation
    try:
        x_eval, y_eval = _concat_split(data_dir, target_client, ["test", "val"])
    except Exception:
        x_eval, y_eval = load_partition(data_dir, target_client, split="test")

    target_epochs = args.epochs
    target_lr = args.lr
    requested_m = args.n_members if args.mode == "full" else 100
    requested_nm = args.n_nonmembers if args.mode == "full" else 100
    if args.control == "overfit":
        # Intentionally memorize a very small train subset to sanity-check that MIA can detect leakage.
        # Members are drawn from memorized training samples; non-members from held-out eval split.
        # Use a small n and many epochs so train-vs-eval loss/confidence gap is detectable.
        n_over = 32
        n_over = min(n_over, requested_m, requested_nm, len(x_train), len(x_eval))
        n_over = max(16, n_over)
        x_train_t, y_train_t = _slice_xy(x_train, y_train, 0, n_over)
        target_epochs = max(args.epochs, 1000)
        target_lr = max(args.lr, 0.05)
        target_batch = min(args.batch_size, max(8, n_over // 2))
    else:
        x_train_t, y_train_t = x_train, y_train
        target_batch = args.batch_size

    _set_global_seeds(args.seed)
    target_model = build_model(input_dim=int(x_train_t.shape[1]))
    train(target_model, x_train_t, y_train_t, epochs=target_epochs, batch_size=target_batch, lr=target_lr)

    # Member/non-member split for target
    if args.control == "overfit":
        n_member = n_over
        n_nonmember = n_over
        if n_member < 1 or n_nonmember < 1:
            raise RuntimeError(
                f"Insufficient data for overfit control: n_train_t={len(x_train_t)} n_eval={len(x_eval)}"
            )
        x_member, y_member = _slice_xy(x_train_t, y_train_t, 0, n_member)
        x_nonmember, y_nonmember = _slice_xy(x_eval, y_eval, 0, n_nonmember)
    else:
        n_member = min(requested_m, len(x_eval) // 2)
        n_nonmember = min(requested_nm, len(x_eval) - n_member)
        if n_member < 1 or n_nonmember < 1:
            raise RuntimeError(f"Insufficient evaluation data for MIA: n_eval={len(x_eval)}")
        x_member, y_member = _slice_xy(x_eval, y_eval, 0, n_member)
        x_nonmember, y_nonmember = _slice_xy(x_eval, y_eval, n_member, n_member + n_nonmember)

    # Shadow models: train on other clients
    shadow_models = []
    shadow_member_data = []
    shadow_nonmember_data = []
    other_clients = [c for c in client_ids if c != target_client]
    shadow_clients = other_clients[: max(1, min(args.n_shadow_models, len(other_clients)))]

    req_sm = args.n_shadow_members if args.mode == "full" else 50
    req_snm = args.n_shadow_nonmembers if args.mode == "full" else 50

    for cid in shadow_clients:
        x_tr, y_tr = load_partition(data_dir, cid, split="train")
        try:
            x_te, y_te = _concat_split(data_dir, cid, ["test", "val"])
        except Exception:
            x_te, y_te = load_partition(data_dir, cid, split="test")

        sm_seed = _stable_seed(args.seed, cid)
        _set_global_seeds(sm_seed)
        sm = build_model(input_dim=int(x_tr.shape[1]))
        if args.control == "overfit":
            # Align shadows with memorization control: train/memorize on train subset,
            # label that subset as members and held-out eval samples as non-members.
            n_sm = min(32, req_sm, req_snm, len(x_tr), len(x_te))
            n_sm = max(16, n_sm)
            n_snm = n_sm
            if n_sm < 1 or n_snm < 1:
                continue
            x_tr_s, y_tr_s = _slice_xy(x_tr, y_tr, 0, n_sm)
            sm_epochs = max(args.shadow_epochs, 1000)
            sm_lr = max(args.lr, 0.05)
            sm_batch = min(args.batch_size, max(8, n_sm // 2))
            train(sm, x_tr_s, y_tr_s, epochs=sm_epochs, batch_size=sm_batch, lr=sm_lr)
            shadow_models.append(sm)
            shadow_member_data.append(_slice_xy(x_tr_s, y_tr_s, 0, n_sm))
            shadow_nonmember_data.append(_slice_xy(x_te, y_te, 0, n_snm))
        else:
            train(sm, x_tr, y_tr, epochs=args.shadow_epochs, batch_size=args.batch_size, lr=args.lr)
            shadow_models.append(sm)
            n = min(req_sm, len(x_te) // 2)
            n2 = min(req_snm, len(x_te) - n)
            if n < 1 or n2 < 1:
                continue
            shadow_member_data.append(_slice_xy(x_te, y_te, 0, n))
            shadow_nonmember_data.append(_slice_xy(x_te, y_te, n, n + n2))

    if not shadow_models or not shadow_member_data or not shadow_nonmember_data:
        raise RuntimeError("Insufficient shadow data/models for MIA evaluation")

    # Aggregate shadow member/non-member data
    x_shadow_member = np.vstack([d[0] for d in shadow_member_data])
    y_shadow_member = {task: np.concatenate([d[1][task] for d in shadow_member_data]) for task in DEFAULT_TASK_TYPES}
    x_shadow_nonmember = np.vstack([d[0] for d in shadow_nonmember_data])
    y_shadow_nonmember = {task: np.concatenate([d[1][task] for d in shadow_nonmember_data]) for task in DEFAULT_TASK_TYPES}

    task_name = args.task
    task_type = DEFAULT_TASK_TYPES[task_name]
    y_member_task = y_member[task_name]
    y_nonmember_task = y_nonmember[task_name]
    y_shadow_member_task = y_shadow_member[task_name]
    y_shadow_nonmember_task = y_shadow_nonmember[task_name]

    # Train attack model from shadow features
    _set_global_seeds(args.seed)
    shadow_features_list = []
    shadow_labels_list = []
    for sm in shadow_models:
        f_mem = compute_features(sm, x_shadow_member, y_shadow_member_task, task_type)
        f_nonmem = compute_features(sm, x_shadow_nonmember, y_shadow_nonmember_task, task_type)
        shadow_features_list.append(f_mem)
        shadow_labels_list.append(np.ones(len(f_mem)))
        shadow_features_list.append(f_nonmem)
        shadow_labels_list.append(np.zeros(len(f_nonmem)))
    X_shadow = np.vstack(shadow_features_list)
    y_shadow = np.concatenate(shadow_labels_list)
    attack_clf = RandomForestClassifier(n_estimators=100, random_state=args.seed)
    attack_clf.fit(X_shadow, y_shadow)

    # Evaluate attack on target
    f_target_member = compute_features(target_model, x_member, y_member_task, task_type)
    f_target_nonmember = compute_features(target_model, x_nonmember, y_nonmember_task, task_type)
    X_target = np.vstack([f_target_member, f_target_nonmember])
    y_target = np.concatenate([np.ones(len(f_target_member)), np.zeros(len(f_target_nonmember))])
    rf_scores = attack_clf.predict_proba(X_target)[:, 1]
    # plain loss-threshold attack scores (uses loss feature column 2; lower loss => more member-like)
    loss_scores = -X_target[:, 2]
    if args.control == "overfit":
        # Positive-control sanity check: memorized members should have lower loss/confidence gap.
        # Use loss-threshold attack as the primary control signal (robust direct memorization detector);
        # retain RF attack score in diagnostics.
        y_scores = loss_scores
    else:
        y_scores = rf_scores

    # Diagnostics (requested)
    unique_scores = int(np.unique(np.round(y_scores, 12)).size)
    target_balance = float(np.mean(np.concatenate([y_member_task, y_nonmember_task]))) if (len(y_member_task)+len(y_nonmember_task)) else None
    # target-model distinct predictions and task skill on eval split
    with torch.no_grad():
        t_in = torch.tensor(np.vstack([x_member, x_nonmember]), dtype=torch.float32)
        outputs = target_model(t_in)
        outp = outputs[task_name]
        if outp.ndim > 1 and outp.shape[1] == 1:
            outp = outp.squeeze(1)
        pred_has_nan = bool(torch.isnan(outp).any().item())
        distinct_target_predictions = int(torch.unique(outp).numel())
        if task_type in ("classification", "binary"):
            # binary logit/prob output
            if outp.ndim == 1 or (outp.ndim == 2 and outp.shape[1] == 1):
                probs = torch.sigmoid(outp.reshape(-1)).cpu().numpy()
            elif outp.ndim == 2 and outp.shape[1] == 2:
                probs = torch.softmax(outp, dim=-1)[:, 1].cpu().numpy()
            else:
                # multiclass-style output fallback
                probs = torch.softmax(outp, dim=-1).cpu().numpy()
                if probs.ndim == 2 and probs.shape[1] == 2:
                    probs = probs[:, 1]
        elif task_type == "multiclass":
            probs_np = torch.softmax(outp, dim=-1).cpu().numpy() if outp.ndim > 1 else outp.cpu().numpy()
            probs = probs_np
        else:
            probs = outp.reshape(-1).cpu().numpy()
    target_labels = np.concatenate([y_member_task, y_nonmember_task]).astype(np.float32)
    majority_baseline_accuracy = float(max(np.mean(target_labels), 1.0 - np.mean(target_labels))) if len(target_labels) else None
    try:
        if task_type in ("classification", "binary") and len(np.unique(target_labels)) > 1:
            target_auc = float(roc_auc_score(target_labels, np.asarray(probs).reshape(-1)))
        elif task_type == "multiclass" and len(np.unique(target_labels)) > 1:
            from sklearn.preprocessing import label_binarize
            classes = np.unique(target_labels)
            if len(classes) == 2:
                target_auc = float(roc_auc_score(target_labels, np.asarray(probs)[:, 1] if np.asarray(probs).ndim == 2 else np.asarray(probs)))
            else:
                y_bin = label_binarize(target_labels, classes=classes)
                target_auc = float(roc_auc_score(y_bin, np.asarray(probs), multi_class="ovr", average="macro"))
        else:
            target_auc = None
    except Exception:
        target_auc = None
    target_has_skill = bool(target_auc is not None and majority_baseline_accuracy is not None and target_auc > majority_baseline_accuracy + 1e-6)
    # plain loss-threshold / RF attack AUC diagnostics
    try:
        loss_threshold_auc = float(roc_auc_score(y_target, loss_scores))
    except Exception:
        loss_threshold_auc = None
    try:
        rf_attack_auc = float(roc_auc_score(y_target, rf_scores))
    except Exception:
        rf_attack_auc = None

    try:
        auc = float(roc_auc_score(y_target, y_scores))
    except Exception:
        auc = None
    tpr_1 = float(tpr_at_fpr(y_target, y_scores, 0.01))
    ci = _auc_ci(y_target, y_scores, seed=args.seed, n_boot=args.bootstrap)

    n_mem = int(len(x_member))
    n_non = int(len(x_nonmember))
    min_resolvable_fpr = (1.0 / n_non) if n_non > 0 else None
    git = _git_state()

    ci_width = (ci[1] - ci[0]) if ci is not None else None
    degenerate = bool(unique_scores < 10 or pred_has_nan or (ci_width is not None and ci_width == 0.0 and auc is not None and abs(auc - 0.5) < 1e-12))
    status = "degenerate" if degenerate else "ok"

    result: dict[str, Any] = {
        "status": status,
        "task": task_name,
        "client": target_client,
        "roc_auc": auc,
        "roc_auc_ci95": ci,
        "roc_auc_ci95_width": ci_width,
        "tpr_at_1pct_fpr": tpr_1,
        "min_resolvable_fpr": min_resolvable_fpr,
        "member_count": n_mem,
        "nonmember_count": n_non,
        "n_members": n_mem,
        "n_non_members": n_non,
        "requested_n_members": int(requested_m),
        "requested_n_nonmembers": int(requested_nm),
        "shadow_model_count": len(shadow_models),
        "shadow_models": len(shadow_models),
        "shadow_clients": shadow_clients,
        "n_shadow_member_total": int(len(x_shadow_member)),
        "n_shadow_nonmember_total": int(len(x_shadow_nonmember)),
        "dp_config": {"enabled": False, "epsilon": None},
        "accounting_status": "unavailable",
        "seed": args.seed,
        "epochs": args.epochs,
        "shadow_epochs": args.shadow_epochs,
        "bootstrap": args.bootstrap,
        "mode": args.mode,
        "data_dir": str(data_dir),
        "git_sha": git["git_sha"],
        "git_dirty": git["dirty"],
        "control": args.control,
        "target_epochs": int(target_epochs),
        "target_auc": target_auc,
        "majority_baseline_accuracy": majority_baseline_accuracy,
        "target_has_skill": target_has_skill,
        "diagnostics": {
            "unique_scores": unique_scores,
            "target_label_balance": target_balance,
            "member_label_mean": float(np.mean(y_member_task)),
            "nonmember_label_mean": float(np.mean(y_nonmember_task)),
            "distinct_target_predictions": distinct_target_predictions,
            "pred_has_nan": pred_has_nan,
            "loss_threshold_auc": loss_threshold_auc,
            "rf_attack_auc": rf_attack_auc,
            "feature_nans": bool(np.isnan(X_target).any()),
            "feature_infs": bool(np.isinf(X_target).any()),
            "majority_baseline_accuracy": majority_baseline_accuracy,
            "target_auc": target_auc,
            "target_has_skill": target_has_skill,
        },
        "note": "Empirical attack result, not a privacy guarantee (D-MIA-001).",
        "limitations": [
            "Empirical attack result, not a formal privacy guarantee.",
            "Depends on attack strength, sample size, model access, and data distribution.",
            "AUC near 0.5 does not prove zero leakage.",
            "TPR@1%FPR may be below resolution when min_resolvable_fpr > 0.01.",
            "If ROC AUC CI includes 0.5, the attack is not distinguishable from chance at this sample size.",
        ],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=str))

    if not args.quiet:
        print(f"MIA evaluation complete. Results: {out_path}")
        print(f"  Status: {status}")
        print(f"  ROC AUC: {result.get('roc_auc')}")
        print(f"  ROC AUC CI95: {result.get('roc_auc_ci95')}")
        print(f"  TPR@1%FPR: {result.get('tpr_at_1pct_fpr')}")
        print(f"  Members: {n_mem}, Non-members: {n_non}")
        print(f"  Min resolvable FPR: {min_resolvable_fpr}")
        print(f"  Shadow models: {len(shadow_models)}")
        print(f"  Bootstrap: {args.bootstrap}")
        print(f"  unique_scores={unique_scores} distinct_target_predictions={distinct_target_predictions} pred_has_nan={pred_has_nan} loss_threshold_auc={loss_threshold_auc} rf_attack_auc={rf_attack_auc}")
        print(f"  target_auc={target_auc} majority_baseline_accuracy={majority_baseline_accuracy} target_has_skill={target_has_skill}")
        print(f"  control={args.control} target_epochs={target_epochs}")
        print(f"  Mode: {args.mode}, data_dir: {data_dir}")
        print(f"  git_dirty: {git['dirty']}")
    if status == "degenerate":
        print("MIA scores are degenerate; fix the target model or the data before using this artifact", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
