"""Evaluation entry point (``insurance-fl-evaluate``). Wave 1: args only."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate trained models.")
    parser.add_argument("--reports-dir", type=str, default="outputs/reports")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(f"reports-dir={args.reports_dir} (evaluation: later wave)")
    return 0
