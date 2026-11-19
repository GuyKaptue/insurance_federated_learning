"""Dataset-generation entry point (``insurance-fl-generate``)."""

from __future__ import annotations

import argparse
from pathlib import Path

from insurance_fl.data.generator import EuropeanInsuranceFederatedDataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the European insurance federated dataset."
    )
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--version", type=str, default="v1.0")
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/european_insurance_federated",
    )
    parser.add_argument("--format", type=str, choices=["parquet", "npy"], default="parquet")
    parser.add_argument(
        "--no-centralized",
        action="store_true",
        help="Do not create/save the pooled centralized dataset.",
    )
    parser.add_argument(
        "--no-quality-issues",
        action="store_true",
        help="Disable synthetic missing/outlier/inconsistency injection.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output directory if it already exists and is not empty.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = Path(args.output)
    if out.exists() and any(out.iterdir()) and not args.force:
        raise SystemExit(
            f"{out} is not empty. Pass --force to overwrite or choose another --output."
        )
    generator = EuropeanInsuranceFederatedDataset(
        samples_per_client=args.samples,
        random_seed=args.seed,
        dataset_version=args.version,
        inject_quality_issues=not args.no_quality_issues,
    )
    generator.generate_dataset(
        output_dir=str(out),
        save_format=args.format,
        include_centralized=not args.no_centralized,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
