#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
docker compose up --build -d
docker compose ps
echo
echo "Gateway: http://127.0.0.1:8080/v1"
echo "Agent:   http://127.0.0.1:8091/v1/agent/run"
