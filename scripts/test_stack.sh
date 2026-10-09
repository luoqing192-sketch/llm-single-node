#!/usr/bin/env bash
set -euo pipefail
GW="${GATEWAY_URL:-http://127.0.0.1:8080}"
AG="${AGENT_URL:-http://127.0.0.1:8091}"

echo "== gateway health =="
curl -fsS "$GW/health"
echo
echo "== models =="
curl -fsS "$GW/v1/models"
echo
echo "== echo chat =="
curl -fsS "$GW/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"echo-demo","messages":[{"role":"user","content":"ping"}]}'
echo
echo "== agent run =="
curl -fsS "$AG/v1/agent/run" \
  -H "Content-Type: application/json" \
  -d '{"message":"只回答：沙箱已连通","model":"echo-demo","max_steps":1}'
echo
