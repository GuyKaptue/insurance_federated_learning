"""Mirror test: insurance_fl.scripts.deploy_api (args contract + exit code)."""

from __future__ import annotations

from insurance_fl.scripts.deploy_api import build_parser, main


def test_parser_defaults() -> None:
    args = build_parser().parse_args([])
    assert args.host == "127.0.0.1"
    assert args.port == 8000


def test_main_returns_zero(capsys) -> None:
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "serving: later wave" in out
