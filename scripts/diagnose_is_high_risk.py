"""Diagnose is_high_risk learnability on pooled/client splits.

Loads prepared partition data, trains quick XGBoost probes, and prints
pooled/per-client AUCs plus basic label diagnostics. This is an ad-hoc
analysis helper, not a formal evaluation artifact.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="data/processed/fl_task0_smoke")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--clients",
        default="germany_allianz,france_axa,belgium_ageas,switzerland_helvetia",
        help="Comma-separated client IDs",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Heavy imports after argument parsing
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    import xgboost as xgb

    from insurance_fl.fl.task import load_partition

    client_ids = [c.strip() for c in args.clients.split(",") if c.strip()]
    frames = []
    for cid in client_ids:
        x, y = load_partition(args.data_dir, cid, split="train")
        df = pd.DataFrame(x)
        df["is_high_risk"] = y["is_high_risk"]
        df["client_id"] = cid
        frames.append(df)

    pooled = pd.concat(frames, ignore_index=True)
    feats = [c for c in pooled.columns if c not in ("is_high_risk", "client_id")]
    y = pooled["is_high_risk"].to_numpy()
    X = pooled[feats].to_numpy()

    # (A) pooled holdout AUC
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=args.seed
    )
    m = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.1,
        early_stopping_rounds=20,
        eval_metric="auc",
    )
    m.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)
    auc_pooled = roc_auc_score(yte, m.predict_proba(Xte)[:, 1])
    print("A. pooled holdout AUC          :", round(float(auc_pooled), 4))

    # (B) per-client within-client AUC
    for cid, g in pooled.groupby("client_id"):
        idx = g.index.to_numpy()
        Xc = X[idx]
        yc = y[idx]
        if yc.min() == yc.max():
            print(f"B. {cid:24s} within-client AUC = n/a (single class)")
            continue
        Xa, Xb, ya, yb = train_test_split(
            Xc, yc, test_size=0.5, stratify=yc, random_state=args.seed
        )
        m2 = xgb.XGBClassifier(n_estimators=100, max_depth=3, eval_metric="auc")
        m2.fit(Xa, ya)
        auc_c = roc_auc_score(yb, m2.predict_proba(Xb)[:, 1])
        print(f"B. {cid:24s} within-client AUC = {auc_c:.4f}")

    # (C) shuffle/sort-by-client control
    pooled_sorted = pooled.sort_values("client_id").reset_index(drop=True)
    y2 = pooled_sorted["is_high_risk"].to_numpy()
    X2 = pooled_sorted[feats].to_numpy()
    Xtr2, Xte2, ytr2, yte2 = train_test_split(
        X2, y2, test_size=0.3, stratify=y2, random_state=args.seed
    )
    m3 = xgb.XGBClassifier(n_estimators=200, max_depth=3, eval_metric="auc")
    m3.fit(Xtr2, ytr2)
    auc_sorted = roc_auc_score(yte2, m3.predict_proba(Xte2)[:, 1])
    print("C. pooled AUC after sort-by-client:", round(float(auc_sorted), 4))

    # (D) in-sample fit diagnostics
    m4 = xgb.XGBClassifier(n_estimators=50, max_depth=6, eval_metric="logloss")
    m4.fit(X, y)
    print("D. in-sample accuracy (pooled) :", round(float((m4.predict(X) == y).mean()), 4))
    print("D. class balance               :", round(float(y.mean()), 4))
    print("D. feature count               :", int(X.shape[1]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
