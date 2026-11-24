#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON:-}"
if [ -z "$PYTHON_BIN" ]; then
  if [ -x venv/bin/python ]; then
    PYTHON_BIN="venv/bin/python"
  elif [ -x .venv/bin/python ]; then
    PYTHON_BIN=".venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

FLWR_BIN="${FLWR:-$("$PYTHON_BIN" -c 'import os,shutil,sys; p=shutil.which("flwr", path=os.path.dirname(sys.executable) or None); print(p or "")')}"
if [ -z "$FLWR_BIN" ]; then
  echo "ERROR: flwr CLI not found next to $PYTHON_BIN" >&2
  exit 1
fi

N_CLIENTS="${N_CLIENTS:-$(PYTHONPATH=src "$PYTHON_BIN" -c 'from insurance_fl.fl.task import CLIENT_IDS; print(len(CLIENT_IDS))')}"

# Isolate Flower user config/state from real ~/.flwr (Flower 1.31 honors FLWR_HOME).
TMP_FLWR_HOME="${FLWR_HOME:-$(mktemp -d "${TMPDIR:-/tmp}/flwr-home.XXXXXX")}"
CREATED_TMP_FLWR_HOME=0
if [ "${FLWR_HOME:-}" = "" ]; then
  CREATED_TMP_FLWR_HOME=1
fi
cleanup() {
  if [ "$CREATED_TMP_FLWR_HOME" -eq 1 ] && [ -n "${TMP_FLWR_HOME:-}" ] && [ -d "$TMP_FLWR_HOME" ]; then
    rm -rf "$TMP_FLWR_HOME"
  fi
}
trap cleanup EXIT INT TERM

export FLWR_HOME="$TMP_FLWR_HOME"
export HOME="${FLWR_SIM_HOME:-$TMP_FLWR_HOME}"
export PATH="$(dirname "$FLWR_BIN"):$PATH"
export PYTHONPATH="${PYTHONPATH:-src}"
export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export FLWR_TELEMETRY_ENABLED="${FLWR_TELEMETRY_ENABLED:-0}"

"$FLWR_BIN" run . --stream --federation-config "num-supernodes=$N_CLIENTS" "$@"
