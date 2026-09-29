#!/usr/bin/env bash
# Run the API and the web server in one container.
#
# `set -e` plus the wait/kill trap matters: Cloud Run treats the container as
# healthy as long as PID 1 lives, so if the API died silently the service would
# keep serving a frontend whose every request 502s. Here, either process exiting
# takes the container down, which is what makes Cloud Run restart it.
set -euo pipefail

API_PORT=8000
WEB_PORT="${PORT:-8080}"

uvicorn services.api.main:app --host 127.0.0.1 --port "$API_PORT" --log-level info &
API_PID=$!

# Fail fast and loudly if the API cannot come up, rather than serving a broken UI.
for i in $(seq 1 60); do
  if curl -sf "http://127.0.0.1:${API_PORT}/api/v1/health" >/dev/null 2>&1; then
    echo "[start] API healthy"
    break
  fi
  if ! kill -0 "$API_PID" 2>/dev/null; then
    echo "[start] API process died during startup" >&2
    exit 1
  fi
  sleep 1
done

cd /app/web
NEXT_PUBLIC_API_URL="http://127.0.0.1:${API_PORT}" \
  npx next start -p "$WEB_PORT" -H 0.0.0.0 &
WEB_PID=$!

# If either half exits, bring the whole container down so Cloud Run replaces it.
trap 'kill -TERM $API_PID $WEB_PID 2>/dev/null || true' TERM INT
wait -n "$API_PID" "$WEB_PID"
echo "[start] a process exited — shutting down" >&2
kill -TERM "$API_PID" "$WEB_PID" 2>/dev/null || true
exit 1
