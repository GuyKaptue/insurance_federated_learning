"""CLI: run refutation tests for a causal estimand."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--estimand", default="has_claims__claim_risk_score")
    p.add_argument("--data-dir", default="data/processed/european_insurance_federated")
    p.add_argument("--dag-path", default="docs/causal/dag.yaml")
    p.add_argument("--treatment", default="has_claims")
    p.add_argument("--outcome", default="claim_risk_score")
    p.add_argument(
        "--adjustment-set",
        default="age,annual_income,savings_balance,credit_score,debt_to_income_ratio",
        help="Comma-separated adjustment variables",
    )
    p.add_argument("--out-dir", default="runs/causal")
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
    from insurance_fl.causal.estimands import CausalEstimand
    from insurance_fl.causal.refute import run_all_refutations

    data_dir = Path(args.data_dir)
    data = _load_all_train(data_dir)

    estimand = CausalEstimand.from_dag(
        name=args.estimand,
        treatment=args.treatment,
        outcome=args.outcome,
        adjustment_set=tuple(s.strip() for s in args.adjustment_set.split(",") if s.strip()),
        dag_path=Path(args.dag_path),
    )

    results = run_all_refutations(data, estimand)
    all_passed = all(r["passed"] for r in results.values())

    output = {
        "estimand": args.estimand,
        "all_passed": bool(all_passed),
        "results": results,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.estimand}_refutations.json"
    with out_path.open("w") as f:
        json.dump(output, f, indent=2, default=str)

    for test_name, result in results.items():
        status = "PASS" if result["passed"] else "FAIL"
        print(f"{test_name}: {status}")
        for k, v in result["details"].items():
            if k != "placebo_ates":
                print(f"  {k}: {v}")

    print(f"Wrote {out_path}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
