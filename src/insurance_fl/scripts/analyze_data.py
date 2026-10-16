"""Analysis entry point (``insurance-fl-analyze``). Wave 1: args only."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyse generated datasets.")
    parser.add_argument("--input", type=str, default="data/processed/european_insurance_federated")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(f"input={args.input} (analysis: later wave)")
    return 0
