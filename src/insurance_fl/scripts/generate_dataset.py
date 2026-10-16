"""Dataset-generation entry point (``insurance-fl-generate``)."""

from __future__ import annotations

import argparse

from insurance_fl.data.generator import EuropeanInsuranceFederatedDataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the European insurance federated dataset."
    )
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--version", type=str, default="v1.0")
    parser.add_argument("--output", type=str, default="data/processed/european_insurance_federated")
    parser.add_argument("--format", type=str, choices=["parquet", "npy"], default="parquet")
    parser.add_argument("--no-centralized", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    generator = EuropeanInsuranceFederatedDataset(
        samples_per_client=args.samples,
        random_seed=args.seed,
        dataset_version=args.version,
    )
    generator.generate_dataset(
        output_dir=args.output,
        save_format=args.format,
    )
    return 0
