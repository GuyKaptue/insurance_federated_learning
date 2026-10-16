"""Mirror test: insurance_fl.scripts.generate_dataset (args contract only).

The heavy generate path is covered by tests/unit/data/test_generator.py;
here we pin the CLI surface: defaults and overrides must parse.
"""

from __future__ import annotations

from insurance_fl.scripts.generate_dataset import build_parser


def test_parser_defaults() -> None:
    args = build_parser().parse_args([])
    assert args.samples == 5000
    assert args.seed == 42
    assert args.version == "v1.0"
    assert args.format == "parquet"
    assert args.no_centralized is False


def test_parser_overrides() -> None:
    args = build_parser().parse_args(
        ["--samples", "100", "--seed", "7", "--format", "npy", "--no-centralized"]
    )
    assert args.samples == 100
    assert args.seed == 7
    assert args.format == "npy"
    assert args.no_centralized is True
