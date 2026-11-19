"""Structural + execution validation for notebooks.

Executes notebooks in-memory with nbclient (no in-place writes) using a
venv-registered Jupyter kernel and captures notebook runtime environment via
the NB_ENV_JSON marker printed by notebooks/_nb_common.env_report().
Exit code != 0 on any structural/execution/environment failure.
"""

from __future__ import annotations

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


def _in_repo_venv(exe: str | None) -> bool:
    if not exe:
        return False
    exe = str(Path(exe).resolve())
    venv_bin = str((ROOT / "venv" / "bin").resolve())
    venv_root = str((ROOT / "venv").resolve())
    if exe.startswith(venv_bin + "/") or exe.startswith(venv_root + "/"):
        return True
    # CI/controlled env: allow current interpreter outside repo venv.
    if os.environ.get("CI") == "true" or os.environ.get("GITHUB_ACTIONS") == "true":
        try:
            return Path(exe).resolve() == Path(sys.executable).resolve()
        except Exception:
            return False
    return False


def _assert_venv_env(info: dict) -> list[str]:
    errs = []
    vexe = info.get("validator_executable")
    kexe = info.get("kernel_executable")
    ci = os.environ.get("CI") == "true" or os.environ.get("GITHUB_ACTIONS") == "true"
    if not _in_repo_venv(vexe):
        errs.append(f"validator executable not in repo venv: {vexe}")
    if not kexe:
        errs.append(
            f"Jupyter kernel '{KERNEL_NAME}' not found. Register it with: "
            f"make kernel (or ./venv/bin/python -m ipykernel install --user --name {KERNEL_NAME})"
        )
    elif not _in_repo_venv(kexe):
        if ci:
            # In CI the kernel must run the same interpreter/environment as the validator.
            try:
                if Path(str(kexe)).resolve() != Path(sys.executable).resolve():
                    errs.append(f"kernel executable not matching CI interpreter: {kexe}")
            except Exception:
                errs.append(f"kernel executable not matching CI interpreter: {kexe}")
        else:
            errs.append(f"kernel executable not in repo venv: {kexe}")
    return errs


def _extract_notebook_exec(nb) -> dict | None:
    """Extract NB_ENV_JSON marker from executed notebook outputs (best-effort)."""
    candidates: list[dict] = []
    for cell in nb.cells:
        for out in cell.get("outputs", []) or []:
            texts: list[str] = []
            ot = out.get("output_type")
            if ot == "stream":
                src = out.get("text", "")
                texts.append("".join(src) if isinstance(src, list) else str(src))
            elif ot in {"execute_result", "display_data"}:
                data = out.get("data", {}) or {}
                if "text/plain" in data:
                    src = data["text/plain"]
                    texts.append("".join(src) if isinstance(src, list) else str(src))
                if "application/json" in data and isinstance(data["application/json"], dict):
                    obj = data["application/json"]
                    if "executable" in obj:
                        candidates.append(obj)
            elif ot == "error":
                # ignore
                pass
            for text in texts:
                for line in text.splitlines():
                    line = line.strip()
                    if line.startswith("NB_ENV_JSON:"):
                        payload = line[len("NB_ENV_JSON:") :].strip()
                        try:
                            obj = json.loads(payload)
                            if isinstance(obj, dict):
                                candidates.append(obj)
                        except Exception:
                            pass
                # fallback: parse any JSON object containing executable/flwr
                if '"executable"' in text and not any(line.strip().startswith("NB_ENV_JSON:") for line in text.splitlines()):
                    s = text.strip()
                    try:
                        obj = json.loads(s)
                        if isinstance(obj, dict) and "executable" in obj:
                            candidates.append(obj)
                    except Exception:
                        i = s.find("{")
                        j = s.rfind("}")
                        if i != -1 and j > i:
                            try:
                                obj = json.loads(s[i : j + 1])
                                if isinstance(obj, dict) and "executable" in obj:
                                    candidates.append(obj)
                            except Exception:
                                pass
    return candidates[-1] if candidates else None


def main() -> int:
    env = _validator_env()
    env_errs = _assert_venv_env(env)
    results = []
    failed = False
    notebooks = sorted(NB_DIR.glob("*.ipynb"))
    if not notebooks:
        print("No notebooks found")
        return 1

    for p in notebooks:
        row = {
            "notebook": p.name,
            "status": "failed",
            "structure": "ok",
            "execution": "not executed",
            "notes": "",
            "seconds": None,
            "kernel_name": KERNEL_NAME,
            "notebook_executable": None,
            "notebook_flwr": None,
            "nb_env_captured": False,
        }
        try:
            nb = nbformat.read(p, as_version=4)
        except Exception as e:
            row["structure"] = f"invalid: {e}"
            row["execution"] = "FAILED"
            row["notes"] = f"invalid notebook: {e}"
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

        structure_ok = row["structure"] == "ok"
        t0 = time.time()
        exec_ok = False
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
            exec_ok = True

            nb_env = _extract_notebook_exec(nb)
            if nb_env:
                row["nb_env_captured"] = True
                row["notebook_executable"] = nb_env.get("executable")
                # support both old 'flwr' and direct version keys
                row["notebook_flwr"] = nb_env.get("flwr")
                exe = str(nb_env.get("executable") or "")
                if not _in_repo_venv(exe):
                    exec_ok = False
                    row["execution"] = "FAILED"
                    row["notes"] = f"wrong interpreter: {exe}"
            else:
                # env_report is expected in modernized notebooks; treat missing capture as failure for honesty
                exec_ok = False
                row["execution"] = "FAILED"
                row["notes"] = "missing NB_ENV_JSON notebook environment capture"
        except Exception as e:
            dt = time.time() - t0
            row["seconds"] = round(dt, 1)
            row["execution"] = "FAILED"
            if not row["notes"]:
                _lines = str(e).splitlines()
                row["notes"] = (_lines[-1] if _lines else str(e) or repr(e))[:400]

        row["status"] = "passed" if (structure_ok and exec_ok) else "failed"
        results.append(row)

    payload = {
        "validator_env": env,
        "env_errors": env_errs,
        "summary": {
            "total": len(results),
            "passed": sum(1 for r in results if r.get("status") == "passed"),
            "failed": sum(1 for r in results if r.get("status") != "passed"),
        },
        "results": results,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2))

    # human-readable summary
    for r in results:
        print(
            f"{r['status'].upper():6s} {r['notebook']}  exec={r['execution']}  "
            f"exe={r.get('notebook_executable')}  flwr={r.get('notebook_flwr')}  {r.get('notes','')}"
        )
    print(json.dumps(payload["summary"]))
    if env_errs:
        for e in env_errs:
            print(f"ENV_ERROR: {e}", file=sys.stderr)


    # Final guard: validation must never modify tracked notebook sources
    try:
        r = subprocess.run(
            ["git", "diff", "--quiet", "--", "notebooks/"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if r.returncode != 0:
            print("ERROR: notebooks/ modified during validation", file=sys.stderr)
            failed = True
    except Exception as e:
        print(f"ERROR: could not verify notebooks unchanged: {e}", file=sys.stderr)
        failed = True

    failed = failed or bool(env_errs) or payload["summary"]["failed"] > 0
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
