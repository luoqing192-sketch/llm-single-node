#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${1:-outputs/cpu-smoke/merged}"
MODEL_NAME="${MODEL_NAME:-local-warehouse-llm}"
PORT="${PORT:-8000}"
VENV_DIR="${VENV_DIR:-$HOME/.venvs/llm-single-node}"
cd "$ROOT"

exec "$VENV_DIR/bin/python" -m single_node_llm.serve \
  --model "$MODEL_PATH" \
  --served-model-name "$MODEL_NAME" \
  --port "$PORT"
