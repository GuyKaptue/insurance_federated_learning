"""CLI: estimate the default causal estimand (backdoor + doubly-robust).

Writes results to runs/causal/<estimand>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="data/processed/european_insurance_federated")
    p.add_argument("--dag-path", default="docs/causal/dag.yaml")
    p.add_argument("--estimand", default="has_claims__claim_risk_score")
    p.add_argument("--treatment", default="has_claims")
    p.add_argument("--outcome", default="claim_risk_score")
    p.add_argument(
        "--adjustment-set",
        default="age,annual_income,savings_balance,credit_score,debt_to_income_ratio",
        help="Comma-separated adjustment variables",
    )
    p.add_argument("--out-dir", default="runs/causal")
    p.add_argument("--tolerance", type=float, default=0.10)
    return p.parse_args(argv)


def _load_all_train(data_dir: Path):
    import pandas as pd

    client_ids = [
        "germany_allianz",
        "france_axa",
        "belgium_ageas",
        "switzerland_helvetia",
    ]
    dfs = []
    for cid in client_ids:
        df = pd.read_parquet(data_dir / "federated" / cid / "train.parquet")
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)
    data["has_claims"] = (data["number_of_claims"] > 0).astype(int)
    return data


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Lazy causal imports/deps after argument parsing so --help works without pgmpy
    from insurance_fl.causal.preflight import require_causal_deps

    require_causal_deps()
    from insurance_fl.causal.estimands import CausalEstimand, estimate_backdoor, estimate_doubly_robust

    data_dir = Path(args.data_dir)
    data = _load_all_train(data_dir)

    estimand = CausalEstimand.from_dag(
        name=args.estimand,
        treatment=args.treatment,
        outcome=args.outcome,
        adjustment_set=tuple(s.strip() for s in args.adjustment_set.split(",") if s.strip()),
        dag_path=Path(args.dag_path),
    )

    bd = estimate_backdoor(data, estimand)
    dr = estimate_doubly_robust(data, estimand)

    rel_diff = abs(bd["ate"] - dr["ate"]) / (abs(bd["ate"]) + 1e-12)
    agree = rel_diff < args.tolerance

    output = {
        "estimand": estimand.name,
        "treatment": estimand.treatment,
        "outcome": estimand.outcome,
        "adjustment_set": list(estimand.adjustment_set),
        "dag_path": str(estimand.dag_path),
        "backdoor": bd,
        "doubly_robust": dr,
        "relative_difference": rel_diff,
        "agree": bool(agree),
        "tolerance": args.tolerance,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{estimand.name}.json"
    with out_path.open("w") as f:
        json.dump(output, f, indent=2)

    print(f"Backdoor ATE: {bd['ate']:.6f} (95% CI [{bd['ci_low']:.6f}, {bd['ci_high']:.6f}])")
    print(f"Doubly-robust ATE: {dr['ate']:.6f} (95% CI [{dr['ci_low']:.6f}, {dr['ci_high']:.6f}])")
    print(f"Relative difference: {rel_diff:.6f} (tolerance {args.tolerance})")
    print(f"Estimators agree: {agree}")
    print(f"Wrote {out_path}")
    return 0 if agree else 1


if __name__ == "__main__":
    sys.exit(main())
