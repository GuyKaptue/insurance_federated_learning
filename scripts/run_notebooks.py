"""Execute all notebooks in notebooks/ with nbconvert."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
TIMEOUT = 300


def main() -> int:
    notebooks = sorted(NB_DIR.glob("*.ipynb"))
    if not notebooks:
        print("No notebooks found")
        return 1
    ep = ExecutePreprocessor(timeout=TIMEOUT, kernel_name="python3")
    for nb_path in notebooks:
        print(f"Executing {nb_path.name} ...")
        nb = nbformat.read(nb_path, as_version=4)
        try:
            ep.preprocess(nb, {"metadata": {"path": str(NB_DIR)}})
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED {nb_path.name}: {exc}")
            return 1
        with nb_path.open("w") as f:
            nbformat.write(nb, f)
        print(f"OK {nb_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
EOF
