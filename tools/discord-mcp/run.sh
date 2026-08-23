#!/usr/bin/env bash
# Stdio launcher for Cursor Cloud Agents (and local MCP hosts).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
if ! python3 -c "import mcp, httpx, dotenv, pydantic" 2>/dev/null; then
  python3 -m pip install --user -q -r "${ROOT}/requirements.txt"
fi
exec python3 -m discord_mcp.server
