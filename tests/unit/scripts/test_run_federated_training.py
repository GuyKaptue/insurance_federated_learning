"""Mirror test: insurance_fl.scripts.run_federated_training (args + config echo)."""

from __future__ import annotations

from insurance_fl.scripts.run_federated_training import build_parser, main


def test_parser_defaults() -> None:
    args = build_parser().parse_args([])
    assert args.config == "config/federated_config.yaml"
    assert args.rounds is None


def test_main_echoes_strategy_and_rounds(capsys) -> None:
    # --dry-run avoids data-dir dependency and heavy booster pre-training in unit tests.
    assert main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "strategy=fedavg" in out
    assert "rounds=10" in out  # from config/federated_config.yaml


def test_main_rounds_override(capsys) -> None:
    assert main(["--dry-run", "--rounds", "3"]) == 0
    out = capsys.readouterr().out
    assert "rounds=3" in out
