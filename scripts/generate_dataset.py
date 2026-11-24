#!/usr/bin/env python3
"""Deprecated wrapper: delegates to the official dataset generator CLI.

Prefer:
  insurance-fl-generate --samples 5000 --seed 42 --output data/processed/european_insurance_federated
  # or: python -m insurance_fl.scripts.generate_dataset ...
"""

from __future__ import annotations

import sys
import warnings

from insurance_fl.scripts.generate_dataset import main


if __name__ == "__main__":
    warnings.warn(
        "scripts/generate_dataset.py is deprecated; use `insurance-fl-generate` "
        "(insurance_fl.scripts.generate_dataset) instead.",
        DeprecationWarning,
        stacklevel=1,
    )
    raise SystemExit(main(sys.argv[1:]))
