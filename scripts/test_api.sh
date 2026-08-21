#!/usr/bin/env bash
set -euo pipefail

curl --fail --silent http://127.0.0.1:8000/health
echo
curl --fail --silent http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local-warehouse-llm","messages":[{"role":"user","content":"只回答 ok"}],"temperature":0,"max_tokens":8}'
echo

curl --fail --silent -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local-warehouse-llm","messages":[{"role":"user","content":"Answer ok"}],"stream":true,"temperature":0,"max_tokens":4}'
echo
