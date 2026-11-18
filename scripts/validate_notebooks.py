"""Structural + execution validation. Exit code != 0 on any failure."""
import json
import os
import re
import subprocess
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
KERNEL_NAME = os.environ.get("NB_KERNEL", "insurance-fl")


def _kernel_executable(kernel_name: str) -> str | None:
    try:
        out = subprocess.check_output(
            ["jupyter", "kernelspec", "list", "--json"], text=True, stderr=subprocess.DEVNULL
        )
        data = json.loads(out)
        spec = data.get("kernelspecs", {}).get(kernel_name, {}).get("spec", {})
        argv = spec.get("argv") or []
        if argv:
            return str(argv[0])
    except Exception:
        pass
    # fallback: read kernel.json directly
    try:
        base = Path.home() / "Library" / "Jupyter" / "kernels" / kernel_name / "kernel.json"
        if base.exists():
            kj = json.loads(base.read_text())
            argv = kj.get("argv") or []
            if argv:
                return str(argv[0])
    except Exception:
        pass
    return None


def _validator_env() -> dict:
    info = {
        "validator_python": sys.version.split()[0],
        "validator_executable": sys.executable,
        "kernel_name": KERNEL_NAME,
        "kernel_executable": _kernel_executable(KERNEL_NAME),
        "nb_mode": os.environ.get("NB_MODE", "light"),
    }
    try:
        import flwr
        info["validator_flwr"] = flwr.__version__
    except Exception:
        info["validator_flwr"] = None
    return info


def _assert_venv_env(info: dict) -> list[str]:
    errs = []
    venv_marker = str((ROOT / "venv").resolve())
    vexe = info.get("validator_executable") or ""
    kexe = info.get("kernel_executable")
    # Require validator and kernel to use repo venv python
    if "/venv/" not in vexe.replace(venv_marker, "/venv/") and not vexe.startswith(venv_marker):
        errs.append(f"validator executable not in repo venv: {vexe}")
    if not kexe:
        errs.append(
            f"Jupyter kernel '{KERNEL_NAME}' not found. Register it with: "
            f"make kernel (or ./venv/bin/python -m ipykernel install --user --name {KERNEL_NAME})"
        )
    elif "/venv/" not in kexe.replace(venv_marker, "/venv/") and not kexe.startswith(venv_marker):
        errs.append(f"kernel executable not in repo venv: {kexe}")
    return errs


def _extract_notebook_exec(nb) -> dict | None:
    # best-effort: find printed env_report dict in outputs
    for cell in nb.cells:
        for out in cell.get("outputs", []) or []:
            text = ""
            if out.get("output_type") == "stream":
                src = out.get("text", "")
                text = "".join(src) if isinstance(src, list) else str(src)
            elif out.get("output_type") in {"execute_result", "display_data"}:
                data = out.get("data", {}) or {}
                if "text/plain" in data:
                    src = data["text/plain"]
                    text = "".join(src) if isinstance(src, list) else str(src)
            if '"executable"' in text and '"flwr"' in text:
                # try to parse JSON object from text
                s = text.strip()
                try:
                    obj = json.loads(s)
                    if isinstance(obj, dict) and "executable" in obj:
                        return obj
                except Exception:
                    # try to locate first {...}
                    i = s.find("{")
                    j = s.rfind("}")
                    if i != -1 and j > i:
                        try:
                            obj = json.loads(s[i:j+1])
                            if isinstance(obj, dict) and "executable" in obj:
                                return obj
                        except Exception:
                            pass
    return None


def main() -> int:
    env = _validator_env()
    env_errs = _assert_venv_env(env)
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
            "kernel_name": KERNEL_NAME,
            "notebook_executable": None,
            "notebook_flwr": None,
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
            if env_errs:
                raise RuntimeError("; ".join(env_errs))
            client = NotebookClient(
                nb,
                timeout=600,
                kernel_name=KERNEL_NAME,
                resources={"metadata": {"path": str(NB_DIR)}},
                allow_errors=False,
            )
            client.execute()
            dt = time.time() - t0
            row["seconds"] = round(dt, 1)
            row["execution"] = f"passed ({dt:.0f}s)"
            nb_env = _extract_notebook_exec(nb)
            if nb_env:
                row["notebook_executable"] = nb_env.get("executable")
                row["notebook_flwr"] = nb_env.get("flwr")
                exe = str(nb_env.get("executable") or "")
                if "/venv/" not in exe and not exe.startswith(str((ROOT / "venv").resolve())):
                    row["execution"] = "FAILED"
                    row["notes"] = f"wrong interpreter: {exe}"
        except Exception as e:
            dt = time.time() - t0
            row["seconds"] = round(dt, 1)
            row["execution"] = "FAILED"
            row["notes"] = str(e).splitlines()[-1][:400]
        results.append(row)

    # Wrap results with env metadata (do not change legacy schema consumers silently? keep top-level list for now)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2))
    for r in results:
        print(json.dumps(r, ensure_ascii=False))

    failed = bool(env_errs) or any(
        r["execution"] == "FAILED" or str(r.get("structure", "")).startswith("invalid")
        for r in results
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
