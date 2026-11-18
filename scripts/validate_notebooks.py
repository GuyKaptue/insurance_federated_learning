"""Structural + execution validation. Exit code != 0 on any failure."""
import json
import os
import re
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
BAD = re.compile(r"TODO|FIXME|/Users/|/tmp/|\bXXX\b")
OUTPUT_PATH = ROOT / "reports" / "notebook_validation.json"
os.environ.setdefault("NB_MODE", "light")


def main() -> int:
    results = []
    notebooks = sorted(NB_DIR.glob("*.ipynb"))
    if not notebooks:
        print("No notebooks found")
        return 1
    for p in notebooks:
        row = {
            "notebook": p.name,
            "structure": "ok",
            "execution": "not executed",
            "notes": "",
            "seconds": None,
        }
        try:
            nb = nbformat.read(p, as_version=4)
        except Exception as e:
            row["structure"] = f"invalid: {e}"
            row["execution"] = "FAILED"
            results.append(row)
            continue
        try:
            nbformat.validate(nb)
        except Exception as e:
            row["structure"] = f"invalid: {e}"
        text = "\n".join("".join(c.get("source", [])) for c in nb.cells)
        bad_hits = BAD.findall(text)
        if bad_hits:
            row["structure"] += f" | placeholder/hardcoded path: {sorted(set(bad_hits))[:5]}"
        if not any(c.cell_type == "markdown" for c in nb.cells):
            row["structure"] += " | no markdown"
        if "_nb_common" not in text:
            row["structure"] += " | missing _nb_common"

        t0 = time.time()
        try:
            client = NotebookClient(
                nb,
                timeout=600,
                kernel_name="python3",
                resources={"metadata": {"path": str(NB_DIR)}},
                allow_errors=False,
            )
            client.execute()
            dt = time.time() - t0
            row["seconds"] = round(dt, 1)
            row["execution"] = f"passed ({dt:.0f}s)"
            # clear outputs for reproducible committed state? keep outputs minimal by not forcing;
            # but we do not rewrite notebook here.
        except Exception as e:
            dt = time.time() - t0
            row["seconds"] = round(dt, 1)
            row["execution"] = "FAILED"
            row["notes"] = str(e).splitlines()[-1][:400]
        results.append(row)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2))
    for r in results:
        print(json.dumps(r, ensure_ascii=False))

    failed = any(
        r["execution"] == "FAILED" or str(r.get("structure", "")).startswith("invalid")
        for r in results
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
