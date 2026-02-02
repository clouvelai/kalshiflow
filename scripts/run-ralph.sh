#!/bin/bash
# =============================================================================
# RALPH - Self-Healing Agent Launcher
# =============================================================================
# Runs the RALPH agent which monitors the V3 trader, detects errors,
# and spawns Claude Code CLI sessions to fix bugs autonomously.
#
# Usage:
#   ./scripts/run-ralph.sh                # Default: monitor port 8005, paper env
#   ./scripts/run-ralph.sh --port 8005    # Custom port
#   ./scripts/run-ralph.sh --env paper    # Explicit environment
#   ./scripts/run-ralph.sh --log-level DEBUG  # Verbose logging
#
# Prerequisites:
#   - V3 trader running (./scripts/run-v3.sh)
#   - Claude Code CLI installed and on PATH
#   - aiohttp, websockets installed (part of backend deps)
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BACKEND_DIR="$PROJECT_ROOT/backend"

echo "============================================"
echo "  RALPH - Self-Healing Agent"
echo "  Monitoring V3 Trader"
echo "============================================"

# Check that claude CLI is available
if ! command -v claude &> /dev/null; then
    echo "ERROR: 'claude' CLI not found on PATH"
    echo "Install Claude Code: https://claude.ai/code"
    exit 1
fi

echo "Claude Code CLI: $(which claude)"
echo "Backend dir: $BACKEND_DIR"
echo "Args: $@"
echo "============================================"

cd "$BACKEND_DIR"

exec uv run python -m kalshiflow_rl.traderv3.ralph.agent "$@"
