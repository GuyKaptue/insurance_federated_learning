#!/usr/bin/env python3
"""Thin CLI for aggregation mode comparison artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from insurance_fl.evaluation.aggregation_comparison import (
    SUPPORTED_MODES,
    ComparisonConfig,
    run_comparison,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--executor", choices=["inprocess", "flower"], default="inprocess")
    ap.add_argument("--out", default="artifacts/aggregation_comparison.json")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--n-rows", type=int, default=2000)
    ap.add_argument("--privacy", action="store_true")
    args = ap.parse_args()
    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())
    cfg = ComparisonConfig(
        modes=SUPPORTED_MODES,
        seeds=seeds,
        rounds=args.rounds,
        n_rows_per_client=args.n_rows,
        privacy_enabled=bool(args.privacy),
        executor=args.executor,
    )
    if args.executor == "flower":
        # Real Flower simulation is the canonical path; this CLI records intent/artifact.
        # Execution goes through scripts/flwr_sim.sh orchestration when wired.
        out = {
            "status": "BLOCKED",
            "reason": "flower executor not wired in this CLI; run scripts/flwr_sim.sh manually per mode and collect evidence",
            "config": cfg.__dict__,
        }
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print(p)
        return 0
    res = run_comparison(cfg)
    res.to_json(args.out)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
