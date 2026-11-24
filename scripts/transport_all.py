"""CLI: run all-pairs transport matrix (4 clients x 3 targets = 12 pairs per task).

Each pair is single-source -> single-target.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CLIENT_MAP = {
    "de": "germany_allianz",
    "fr": "france_axa",
    "be": "belgium_ageas",
    "ch": "switzerland_helvetia",
}

CLIENT_COUNTRY = {
    "germany_allianz": "Germany",
    "france_axa": "France",
    "belgium_ageas": "Belgium",
    "switzerland_helvetia": "Switzerland",
}

CLIENT_KEYS = list(CLIENT_MAP.values())
CLIENT_SHORT = {v: k for k, v in CLIENT_MAP.items()}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="data/processed/european_insurance_federated")
    p.add_argument("--dag-path", default="docs/causal/dag.yaml")
    p.add_argument("--out-dir", default="runs/causal")
    p.add_argument("--tasks", default="claim_risk_score,will_renew,is_high_risk",
                   help="Comma-separated task/target columns")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def _load_train(data_dir: Path, client_id: str):
    import pandas as pd

    return pd.read_parquet(data_dir / "federated" / client_id / "train.parquet")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Lazy causal/transport imports/deps after argument parsing so --help works without pgmpy
    from insurance_fl.causal.preflight import require_causal_deps

    require_causal_deps()
    from insurance_fl.causal.estimands import CausalEstimand
    from insurance_fl.transport.transport import transport_causal_effect, transport_outcome_model

    data_dir = Path(args.data_dir)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    dfs = {k: _load_train(data_dir, k) for k in CLIENT_KEYS}
    for k in CLIENT_KEYS:
        dfs[k]["has_claims"] = (dfs[k]["number_of_claims"] > 0).astype(int)

    estimand = CausalEstimand.from_dag(
        name="has_claims__claim_risk_score",
        treatment="has_claims",
        outcome="claim_risk_score",
        adjustment_set=(
            "age",
            "annual_income",
            "savings_balance",
            "credit_score",
            "debt_to_income_ratio",
        ),
        dag_path=Path(args.dag_path),
    )

    results = []
    for target_key in CLIENT_KEYS:
        target_df = dfs[target_key]
        for source_key in CLIENT_KEYS:
            if source_key == target_key:
                continue
            source_df = dfs[source_key]

            for task in tasks:
                report = transport_outcome_model(source_df, target_df, task)
                te_result = transport_causal_effect(estimand, source_df, target_df, None)
                results.append({
                    "source": CLIENT_COUNTRY[source_key],
                    "target": CLIENT_COUNTRY[target_key],
                    "source_short": CLIENT_SHORT[source_key],
                    "target_short": CLIENT_SHORT[target_key],
                    "task": task,
                    "smd_worst": report.smd_worst,
                    "source_loss": report.source_loss,
                    "target_loss": report.target_loss,
                    "loss_ratio": report.loss_ratio,
                    "transportable_metric": report.transportable_metric,
                    "causal_effect_transport": te_result,
                })

    all_transportable = all(r["transportable_metric"] for r in results) if results else False

    output = {
        "all_transportable": bool(all_transportable),
        "n_pairs": len(results),
        "results": results,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "transport_all.json"
    with out_path.open("w") as f:
        json.dump(output, f, indent=2, default=str)

    if not args.quiet:
        print(f"{'Source':<10} {'Target':<10} {'Task':<22} {'Worst SMD':<12} {'Src Loss':<10} {'Tgt Loss':<10} {'Loss Ratio':<12} {'Transportable'}")
        print("-" * 100)
        for r in results:
            worst_smd = max(r["smd_worst"].values()) if r["smd_worst"] else 0.0
            worst_feat = max(r["smd_worst"], key=r["smd_worst"].get) if r["smd_worst"] else "-"
            print(f"{r['source_short']:<10} {r['target_short']:<10} {r['task']:<22} "
                  f"{worst_smd:.4f}({worst_feat:<12}) {r['source_loss']:<10.4f} {r['target_loss']:<10.4f} "
                  f"{r['loss_ratio']:<12.4f} {r['transportable_metric']}")
        print()
        print(f"All transportable: {all_transportable}")
        print(f"Wrote {out_path}")
    return 0 if all_transportable else 1


if __name__ == "__main__":
    sys.exit(main())
