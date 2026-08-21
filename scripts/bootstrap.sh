#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$HOME/.venvs/llm-single-node}"
cd "$ROOT"

if [[ ! -x "$VENV_DIR/bin/python" ]] || ! "$VENV_DIR/bin/python" -m pip --version >/dev/null 2>&1; then
  if ! python3 -m venv --clear "$VENV_DIR"; then
    sudo apt-get update
    sudo apt-get install -y python3-venv python3-pip
    python3 -m venv --clear "$VENV_DIR"
  fi
fi

"$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
if [[ "${USE_CUDA:-0}" == "1" ]]; then
  "$VENV_DIR/bin/python" -m pip install torch
else
  "$VENV_DIR/bin/python" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
fi
"$VENV_DIR/bin/python" -m pip install -e .

echo "Environment ready: $VENV_DIR"
"$VENV_DIR/bin/python" -m single_node_llm.preflight
