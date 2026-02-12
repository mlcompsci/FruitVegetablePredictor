#!/usr/bin/env bash
# Quick-start: launch the FastAPI dev server.
# Usage:  bash scripts/run_dev.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "──────────────────────────────────────────"
echo "  Produce Identifier – Development Server"
echo "──────────────────────────────────────────"

# Activate venv if present
if [ -d "venv" ]; then
    source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null || true
fi

echo "[INFO] Starting Uvicorn on http://localhost:8000"
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
