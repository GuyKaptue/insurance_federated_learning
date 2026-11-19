import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "run_mia.py"
PY = str(ROOT / "venv" / "bin" / "python")


def _run(tmp_path, name, *extra, mode="smoke"):
    out = tmp_path / f"{name}.json"
    cmd = [
        PY,
        str(SCRIPT),
        "--mode",
        mode,
        "--n-members",
        "100",
        "--n-nonmembers",
        "100",
        "--bootstrap",
        "50",
        "--out",
        str(out),
        "--quiet",
        *extra,
    ]
    subprocess.run(cmd, check=True, cwd=ROOT)
    return json.loads(out.read_text())


@pytest.mark.slow
def test_run_mia_is_deterministic(tmp_path):
    a = _run(tmp_path, "a")
    b = _run(tmp_path, "b")
    assert a["roc_auc"] == b["roc_auc"]


@pytest.mark.slow
def test_overfit_control_is_detected(tmp_path):
    r = _run(tmp_path, "c", "--control", "overfit")
    assert r["status"] == "ok"
    assert r["roc_auc"] > 0.6, f"positive control failed: roc_auc={r['roc_auc']}"


@pytest.mark.slow
def test_output_schema(tmp_path):
    r = _run(tmp_path, "d")
    for k in (
        "roc_auc",
        "roc_auc_ci95",
        "target_auc",
        "majority_baseline_accuracy",
        "target_has_skill",
        "status",
        "git_sha",
        "git_dirty",
        "seed",
    ):
        assert k in r
