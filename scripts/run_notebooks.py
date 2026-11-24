"""Execute all notebooks in notebooks/ (validation run, no in-place writes)."""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
TIMEOUT = 300
KERNEL_NAME = "python3"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kernel", default=KERNEL_NAME, help="Jupyter kernel name")
    p.add_argument("--timeout", type=int, default=TIMEOUT)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    notebooks = sorted(NB_DIR.glob("*.ipynb"))
    if not notebooks:
        print("No notebooks found")
        return 1
    for nb_path in notebooks:
        print(f"Executing {nb_path.name} ...")
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / nb_path.name
            shutil.copy2(nb_path, work)
            nb = nbformat.read(work, as_version=4)
            try:
                client = NotebookClient(
                    nb,
                    timeout=args.timeout,
                    kernel_name=args.kernel,
                    resources={"metadata": {"path": str(NB_DIR)}},
                    allow_errors=False,
                )
                client.execute()
            except Exception as exc:  # noqa: BLE001
                print(f"FAILED {nb_path.name}: {exc}")
                return 1
        print(f"OK {nb_path.name}")
    # Guard: execution must not modify tracked notebook sources
    import subprocess

    try:
        r = subprocess.run(
            ["git", "diff", "--quiet", "--", "notebooks/"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if r.returncode != 0:
            print("ERROR: notebooks/ modified during execution run", file=sys.stderr)
            return 1
    except Exception as e:
        print(f"ERROR: could not verify notebooks unchanged: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
