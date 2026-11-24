#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export FLWR_HOME="${FLWR_HOME:-$ROOT/.flwr-compose}"
mkdir -p "$FLWR_HOME"
cp "$ROOT/docker/flwr-compose-config.toml" "$FLWR_HOME/config.toml"
exec "$ROOT/venv/bin/flwr" run "$ROOT" local-deployment --stream \
  --run-config "data-dir=/app/data/processed/fl_task0_smoke num-server-rounds=1 fraction-fit=1.0 fraction-evaluate=1.0 min-fit-clients=4 min-evaluate-clients=4 min-available-clients=4"
