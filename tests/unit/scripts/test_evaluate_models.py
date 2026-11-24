"""Mirror test: insurance_fl.scripts.evaluate_models (args contract + exit code)."""

from __future__ import annotations

from insurance_fl.scripts.evaluate_models import build_parser, main


def test_parser_defaults() -> None:
    args = build_parser().parse_args([])
    assert args.reports_dir == "outputs/reports"


def test_main_returns_zero(capsys) -> None:
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "evaluation: later wave" in out
