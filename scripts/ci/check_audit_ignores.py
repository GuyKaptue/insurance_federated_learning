#!/usr/bin/env python3
"""Validate security/pip-audit-ignores.toml against pip-audit JSON reports.

Checks:
  * required fields present (id, package, reason, exploitability_here, added, expires)
  * expiry window <= 60 days and not expired
  * no stale ignore IDs (an ignore id must appear in at least one report)
  * no unignored findings in the RUNTIME report (blocking: every runtime
    advisory id must be covered by a justified, time-boxed ignore)
Full-lock unignored findings are reported as warnings only (report-only).

Exit non-zero if: invalid entries, expired/stale entries, or any runtime
finding is not covered by an ignore entry.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

try:
    import tomllib  # py311+
except Exception:  # pragma: no cover
    import tomli as tomllib  # type: ignore

REQUIRED = {"id", "package", "reason", "exploitability_here", "added", "expires"}
MAX_DAYS = 60


def load_ids(report_path: Path) -> set[str]:
    data = json.loads(report_path.read_text(encoding="utf-8"))
    ids: set[str] = set()
    for dep in data.get("dependencies", []):
        for v in dep.get("vulns", []):
            vid = v.get("id")
            if vid:
                ids.add(str(vid))
    return ids


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ignores", required=True)
    p.add_argument("--runtime-report", required=True)
    p.add_argument("--full-report", required=True)
    p.add_argument("--today", default=None, help="YYYY-MM-DD override for tests")
    args = p.parse_args(argv)

    ignores_path = Path(args.ignores)
    if not ignores_path.exists():
        print(f"ERROR: ignores file missing: {ignores_path}", file=sys.stderr)
        return 2
    try:
        cfg = tomllib.loads(ignores_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"ERROR: could not parse {ignores_path}: {e}", file=sys.stderr)
        return 2
    entries = cfg.get("ignore", [])
    if not isinstance(entries, list):
        print("ERROR: [ignore] entries must be a list of tables (use [[ignore]])", file=sys.stderr)
        return 2

    today = dt.date.fromisoformat(args.today) if args.today else dt.date.today()
    runtime_ids = load_ids(Path(args.runtime_report))
    full_ids = load_ids(Path(args.full_report))
    all_ids = runtime_ids | full_ids

    errors: list[str] = []
    warnings: list[str] = []
    seen: set[tuple[str, str]] = set()
    ignored_ids: set[str] = set()
    for i, e in enumerate(entries):
        if not isinstance(e, dict):
            errors.append(f"entry[{i}] not a table")
            continue
        missing = sorted(REQUIRED - set(e.keys()))
        if missing:
            errors.append(f"entry[{i}] missing fields: {missing}")
            continue
        vid = str(e["id"]).strip()
        pkg = str(e["package"]).strip()
        if (vid, pkg) in seen:
            errors.append(f"duplicate ignore entry: {(vid, pkg)}")
        seen.add((vid, pkg))
        ignored_ids.add(vid)
        try:
            added = dt.date.fromisoformat(str(e["added"]))
            expires = dt.date.fromisoformat(str(e["expires"]))
        except Exception as ex:
            errors.append(f"entry[{i}] {vid}/{pkg} invalid date: {ex}")
            continue
        if expires < today:
            errors.append(f"entry[{i}] {vid}/{pkg} expired on {expires}")
        if expires > added + dt.timedelta(days=MAX_DAYS):
            errors.append(
                f"entry[{i}] {vid}/{pkg} expiry window > {MAX_DAYS} days ({added}..{expires})"
            )
        if vid not in all_ids:
            errors.append(f"entry[{i}] {vid}/{pkg} stale (not present in audit reports)")

    # Blocking: every runtime advisory id must be covered by an ignore entry.
    unignored_runtime = sorted(runtime_ids - ignored_ids)
    if unignored_runtime:
        errors.append(
            f"RUNTIME findings not covered by ignores ({len(unignored_runtime)}): "
            + ", ".join(unignored_runtime)
        )

    # Report-only: full-lock findings not covered by ignores.
    unignored_full = sorted(full_ids - ignored_ids)
    if unignored_full:
        warnings.append(
            f"FULL-LOCK findings not covered by ignores ({len(unignored_full)}): "
            + ", ".join(unignored_full[:20])
            + (f" ... ({len(unignored_full)-20} more)" if len(unignored_full) > 20 else "")
        )

    if errors:
        print("pip-audit ignores: FAIL", file=sys.stderr)
        for err in errors:
            print(f"- {err}", file=sys.stderr)
        if warnings:
            print("warnings:", file=sys.stderr)
            for w in warnings:
                print(f"  - {w}", file=sys.stderr)
        return 1
    print(
        f"pip-audit ignores: OK ({len(entries)} entries covering all {len(runtime_ids)} runtime findings)"
    )
    if warnings:
        for w in warnings:
            print(f"  warn: {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
