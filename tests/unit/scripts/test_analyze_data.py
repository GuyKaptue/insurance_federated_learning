"""Mirror test: insurance_fl.scripts.analyze_data (args contract + exit code)."""

from __future__ import annotations

from insurance_fl.scripts.analyze_data import build_parser, main


def test_parser_defaults() -> None:
    args = build_parser().parse_args([])
    assert args.input == "data/processed/european_insurance_federated"


def test_main_returns_zero(capsys) -> None:
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "analysis: later wave" in out
