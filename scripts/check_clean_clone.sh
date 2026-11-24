#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/venv/bin/python}"
TMP="$(mktemp -d "${TMPDIR:-/tmp}/insurance-fl-clean.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT

mkdir -p "$TMP/source"
tar -C "$ROOT" \
  --exclude='./.git' \
  --exclude='./venv' \
  --exclude='./.venv' \
  --exclude='./__pycache__' \
  --exclude='./.pytest_cache' \
  --exclude='./.mypy_cache' \
  --exclude='./.ruff_cache' \
  -cf - . | tar -C "$TMP/source" -xf -

cd "$TMP/source"
PYTHONPATH="$TMP/source/src:$TMP/source/notebooks${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON" -m pytest -q tests/unit/test_audit.py tests/unit/test_artifact_provenance.py
PYTHONPATH="$TMP/source/src:$TMP/source/notebooks${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON" -c 'import insurance_fl.audit; print("clean source snapshot import: OK")'
