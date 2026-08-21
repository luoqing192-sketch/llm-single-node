#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-configs/cpu-smoke.yaml}"
VENV_DIR="${VENV_DIR:-$HOME/.venvs/llm-single-node}"
cd "$ROOT"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TOKENIZERS_PARALLELISM=false

for stage in pt sft dpo grpo; do
  echo "===== stage: $stage ====="
  "$VENV_DIR/bin/python" -m single_node_llm.train --config "$CONFIG" --stage "$stage"
done

"$VENV_DIR/bin/python" -m single_node_llm.merge --config "$CONFIG"
echo "Pipeline complete for $CONFIG"
