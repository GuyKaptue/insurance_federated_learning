"""Federated-training entry point (``insurance-fl-train``).

Wave 1: parses args and echoes the plan. The training loop arrives with
the Flower wiring in a later wave.
"""

from __future__ import annotations

import argparse

from insurance_fl.config.loader import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run federated training.")
    parser.add_argument("--config", type=str, default="config/federated_config.yaml")
    parser.add_argument("--rounds", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    rounds = args.rounds or cfg.federated.num_rounds
    print(f"strategy={cfg.federated.strategy_type} rounds={rounds} (training: later wave)")
    return 0
