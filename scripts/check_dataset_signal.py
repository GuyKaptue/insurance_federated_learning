#!/usr/bin/env python3
"""Dataset label-signal sanity check.

Computes Bayes-optimal AUC for the canonical `is_high_risk` logit per client
and fails if signal is implausibly low (indicating a broken/shifted label).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Ensure local src is importable when run as a script.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from insurance_fl.data.generator import high_risk_logit  # noqa: E402

CLIENTS = [
    "germany_allianz",
    "france_axa",
    "belgium_ageas",
    "switzerland_helvetia",
]
SPLITS = ("train", "val", "test", "site")


def _load_client_all(data_dir: Path, client: str) -> pd.DataFrame:
    base = data_dir / "federated" / client
    dfs = []
    for split in SPLITS:
        p = base / f"{split}.parquet"
        if p.exists():
            dfs.append(pd.read_parquet(p))
    if not dfs:
        raise FileNotFoundError(f"No parquet splits found for {client} under {base}")
    return pd.concat(dfs, ignore_index=True)


def check_dataset_signal(data_dir: str | Path, threshold: float = 0.70) -> dict:
    data_dir = Path(data_dir)
    results = {}
    ok = True
    for client in CLIENTS:
        try:
            df = _load_client_all(data_dir, client)
        except FileNotFoundError as e:
            results[client] = {"status": "missing", "error": str(e)}
            ok = False
            continue
        y = df["is_high_risk"].to_numpy()
        s = high_risk_logit(df)
        mask = np.isfinite(s) & np.isfinite(y.astype(float))
        auc = float(roc_auc_score(y[mask], s[mask])) if mask.sum() > 0 else float("nan")
        pos = float(np.mean(y))
        n = int(len(df))
        status = "pass" if np.isfinite(auc) and auc >= threshold else "fail"
        if status != "pass":
            ok = False
        results[client] = {"n": n, "positive_rate": pos, "bayes_auc_logit": auc, "status": status}
    return {"data_dir": str(data_dir), "threshold": threshold, "ok": ok, "clients": results}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-dir",
        default="data/processed/european_insurance_federated",
        help="Path to federated dataset root",
    )
    p.add_argument("--threshold", type=float, default=0.70)
    p.add_argument("--json", action="store_true", help="Print JSON only")
    args = p.parse_args(argv)
    try:
        report = check_dataset_signal(args.data_dir, args.threshold)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Dataset signal check: {report['data_dir']} (threshold AUC >= {report['threshold']})")
        for c, r in report["clients"].items():
            if r.get("status") == "missing":
                print(f"  - {c}: MISSING ({r.get('error')})")
            else:
                print(
                    f"  - {c}: bayes_auc={r['bayes_auc_logit']:.4f} pos={r['positive_rate']:.4f} "
                    f"n={r['n']} [{r['status']}]"
                )
        print("RESULT:", "PASS" if report["ok"] else "FAIL")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
