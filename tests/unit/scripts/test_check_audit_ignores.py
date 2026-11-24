from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

CHECKER = Path(__file__).resolve().parents[3] / "scripts" / "ci" / "check_audit_ignores.py"


def _write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _runtime_report(tmp_path: Path, ids: list[str]) -> Path:
    content = {
        "dependencies": [
            {"name": "pkg-a", "version": "1.0", "vulns": [{"id": i, "fix_versions": ["2.0"]}]}
            for i in ids
        ]
    }
    p = tmp_path / "runtime.json"
    _write(p, json.dumps(content))
    return p


def _full_report(tmp_path: Path, ids: list[str]) -> Path:
    content = {
        "dependencies": [{"name": "pkg-a", "version": "1.0", "vulns": [{"id": i}]} for i in ids]
    }
    p = tmp_path / "full.json"
    _write(p, json.dumps(content))
    return p


def _ignores(tmp_path: Path, entries: list[dict]) -> Path:
    blocks = []
    for e in entries:
        blocks.append("[[ignore]]")
        for k, v in e.items():
            blocks.append(f'{k} = "{v}"')
        blocks.append("")
    p = tmp_path / "ignores.toml"
    _write(p, "\n".join(blocks))
    return p


def _run(
    tmp_path: Path, ignores: Path, runtime: Path, full: Path, today: str = "2026-11-20"
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(CHECKER),
            "--ignores",
            str(ignores),
            "--runtime-report",
            str(runtime),
            "--full-report",
            str(full),
            "--today",
            today,
        ],
        capture_output=True,
        text=True,
    )


def _base_entries() -> list[dict]:
    return [
        {
            "id": "VULN-001",
            "package": "pkg-a",
            "reason": "r",
            "exploitability_here": "e",
            "added": "2026-11-01",
            "expires": "2026-12-30",
        },
        {
            "id": "VULN-002",
            "package": "pkg-a",
            "reason": "r",
            "exploitability_here": "e",
            "added": "2026-11-01",
            "expires": "2026-12-30",
        },
    ]


def test_valid_config_passes(tmp_path: Path) -> None:
    runtime = _runtime_report(tmp_path, ["VULN-001", "VULN-002"])
    full = _full_report(tmp_path, ["VULN-001", "VULN-002"])
    ignores = _ignores(tmp_path, _base_entries())
    res = _run(tmp_path, ignores, runtime, full)
    assert res.returncode == 0, res.stderr + res.stdout
    assert "OK" in res.stdout


def test_unignored_runtime_finding_fails(tmp_path: Path) -> None:
    runtime = _runtime_report(tmp_path, ["VULN-001", "VULN-003"])
    full = _full_report(tmp_path, ["VULN-001", "VULN-003"])
    ignores = _ignores(tmp_path, [e for e in _base_entries() if e["id"] != "VULN-002"])
    res = _run(tmp_path, ignores, runtime, full)
    assert res.returncode != 0
    assert "VULN-003" in res.stderr


def test_missing_field_fails(tmp_path: Path) -> None:
    runtime = _runtime_report(tmp_path, ["VULN-001"])
    full = _full_report(tmp_path, ["VULN-001"])
    entries = [
        {
            "id": "VULN-001",
            "package": "pkg-a",
            "reason": "r",
            "added": "2026-11-01",
            "expires": "2026-12-30",
        }
    ]
    ignores = _ignores(tmp_path, entries)
    res = _run(tmp_path, ignores, runtime, full)
    assert res.returncode != 0
    assert "missing fields" in res.stderr


def test_expired_entry_fails(tmp_path: Path) -> None:
    runtime = _runtime_report(tmp_path, ["VULN-001"])
    full = _full_report(tmp_path, ["VULN-001"])
    entries = [dict(_base_entries()[0], expires="2026-11-19")]
    ignores = _ignores(tmp_path, entries)
    res = _run(tmp_path, ignores, runtime, full, today="2026-11-20")
    assert res.returncode != 0
    assert "expired" in res.stderr


def test_over_sixty_days_fails(tmp_path: Path) -> None:
    runtime = _runtime_report(tmp_path, ["VULN-001"])
    full = _full_report(tmp_path, ["VULN-001"])
    entries = [dict(_base_entries()[0], added="2026-01-01", expires="2027-03-01")]
    ignores = _ignores(tmp_path, entries)
    res = _run(tmp_path, ignores, runtime, full, today="2026-11-20")
    assert res.returncode != 0
    assert "60 days" in res.stderr


def test_stale_entry_fails(tmp_path: Path) -> None:
    runtime = _runtime_report(tmp_path, ["VULN-001"])
    full = _full_report(tmp_path, ["VULN-001"])
    entries = [
        *_base_entries(),
        {
            "id": "VULN-999",
            "package": "pkg-z",
            "reason": "r",
            "exploitability_here": "e",
            "added": "2026-11-01",
            "expires": "2026-12-30",
        },
    ]
    ignores = _ignores(tmp_path, entries)
    res = _run(tmp_path, ignores, runtime, full)
    assert res.returncode != 0
    assert "stale" in res.stderr
