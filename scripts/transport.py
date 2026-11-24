"""CLI: transport outcome model from source clients to target client.

Example:
  python scripts/transport.py --source-clients de,fr --target-client ch --task claim_risk_score
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-clients", required=True, help="Comma-separated client ids (de,fr,...)")
    p.add_argument("--target-client", required=True, help="Target client id (de/fr/be/ch)")
    p.add_argument("--task", required=True, help="Target column name")
    p.add_argument("--data-dir", default="data/processed/european_insurance_federated")
    p.add_argument("--dag-path", default="docs/causal/dag.yaml")
    p.add_argument("--out-dir", default="runs/causal")
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

    source_ids = [c.strip() for c in args.source_clients.split(",") if c.strip()]
    target_id = args.target_client.strip()

    source_client_keys = [CLIENT_MAP[c] for c in source_ids]
    target_client_key = CLIENT_MAP[target_id]

    data_dir = Path(args.data_dir)
    import pandas as pd

    source_dfs = [_load_train(data_dir, k) for k in source_client_keys]
    source_df = pd.concat(source_dfs, ignore_index=True)
    target_df = _load_train(data_dir, target_client_key)

    report = transport_outcome_model(source_df, target_df, args.task)

    # T4: causal effect transport (only if estimand passed C1-C4 on source)
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
    # Add has_claims column to both dataframes
    source_df["has_claims"] = (source_df["number_of_claims"] > 0).astype(int)
    target_df["has_claims"] = (target_df["number_of_claims"] > 0).astype(int)
    te_result = transport_causal_effect(estimand, source_df, target_df, None)

    output = {
        "source_clients": [CLIENT_COUNTRY[k] for k in source_client_keys],
        "target_client": CLIENT_COUNTRY[target_client_key],
        "task": args.task,
        "smd_worst": report.smd_worst,
        "source_loss": report.source_loss,
        "target_loss": report.target_loss,
        "loss_ratio": report.loss_ratio,
        "transportable_metric": report.transportable_metric,
        "causal_effect_transport": te_result,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "_".join(source_ids) + "_to_" + target_id
    out_path = out_dir / f"transport_{tag}_{args.task}.json"
    with out_path.open("w") as f:
        json.dump(output, f, indent=2)

    print(f"Source: {output['source_clients']} -> Target: {output['target_client']}")
    print(f"Task: {args.task}")
    print(f"SMD worst 5: {report.smd_worst}")
    print(f"Source loss: {report.source_loss:.6f}")
    print(f"Target loss: {report.target_loss:.6f}")
    print(f"Loss ratio: {report.loss_ratio:.6f}")
    print(f"Transportable (T3): {report.transportable_metric}")
    print(f"Causal effect transport: {te_result}")
    print(f"Wrote {out_path}")
    return 0 if report.transportable_metric else 1


if __name__ == "__main__":
    sys.exit(main())
