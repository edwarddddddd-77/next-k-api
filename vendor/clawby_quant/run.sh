#!/usr/bin/env bash
# Start the quant bot (backend serves API + built dashboard on one port).
set -euo pipefail
cd "$(dirname "$0")"

PORT="${1:-8899}"

if [ ! -d .venv ]; then
  python3 -m venv .venv
  ./.venv/bin/pip install -q -r requirements.txt
fi
if [ ! -d frontend/dist ]; then
  echo ">>> building frontend (first run)"
  source ~/.nvm/nvm.sh 2>/dev/null && nvm use 24 >/dev/null 2>&1 || true
  (cd frontend && npm install && npm run build)
fi

echo ">>> quant-bot on http://127.0.0.1:${PORT}  (mode defaults to PAPER)"
exec ./.venv/bin/uvicorn backend.main:app --port "$PORT"
