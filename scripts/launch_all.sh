#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"
START_INDEXER="${START_INDEXER:-1}"

if ! command -v docker >/dev/null 2>&1; then
  echo "Error: docker is required but not found." >&2
  exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
  echo "Error: docker compose is required but not available." >&2
  exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "Error: npm is required but not found." >&2
  exit 1
fi

PYTHON_BIN=""
if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
else
  echo "Error: python3 is required but not found." >&2
  exit 1
fi

cd "$ROOT_DIR"

echo "[1/4] Starting Qdrant..."
docker compose up -d qdrant

if [[ "$START_INDEXER" == "1" ]]; then
  echo "[2/4] Running indexing job (set START_INDEXER=0 to skip)..."
  docker compose run --rm rag-indexer
else
  echo "[2/4] Skipping indexing job (START_INDEXER=$START_INDEXER)."
fi

echo "[3/4] Starting backend on port $BACKEND_PORT..."
"$PYTHON_BIN" -m uvicorn backend.main:app --host 0.0.0.0 --port "$BACKEND_PORT" &
BACKEND_PID=$!

echo "[4/4] Starting frontend on port $FRONTEND_PORT..."
(
  cd "$ROOT_DIR/frontend"
  npm run dev -- --host 0.0.0.0 --port "$FRONTEND_PORT"
) &
FRONTEND_PID=$!

cleanup() {
  echo
  echo "Shutting down app processes..."
  if kill -0 "$BACKEND_PID" >/dev/null 2>&1; then
    kill "$BACKEND_PID" >/dev/null 2>&1 || true
  fi
  if kill -0 "$FRONTEND_PID" >/dev/null 2>&1; then
    kill "$FRONTEND_PID" >/dev/null 2>&1 || true
  fi
}

trap cleanup EXIT INT TERM

echo ""
echo "Semantic Atlas is running:"
echo "- Backend:  http://localhost:$BACKEND_PORT"
echo "- Frontend: http://localhost:$FRONTEND_PORT"
echo ""
echo "Press Ctrl+C to stop backend/frontend. Qdrant keeps running in Docker."

wait "$BACKEND_PID" "$FRONTEND_PID"
