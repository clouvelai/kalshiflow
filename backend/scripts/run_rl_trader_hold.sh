#!/bin/bash

# Run RL Trading Backend with HOLD-only Action Selector
# This script runs the actor service in paper trading mode with a HOLD-only strategy
# for safe testing of the e2e trading pipeline

echo "=========================================="
echo "RL Trading Backend - HOLD Strategy"
echo "=========================================="
echo "Environment: Paper Trading"
echo "Port: 8001"
echo "Strategy: HOLD-only (no trades)"
echo "=========================================="

# Navigate to backend directory
cd "$(dirname "$0")/.." || exit 1

# Export environment variables for paper trading with HOLD strategy
export ENVIRONMENT=paper
export RL_ACTOR_ENABLED=true
export RL_ACTOR_STRATEGY=hardcoded
export RL_ACTOR_MODEL_PATH=""
export RL_ACTOR_THROTTLE_MS=1000
export RL_MARKET_MODE=config
export RL_MARKET_TICKERS="INXD-25JAN03,DOGE-25JAN17"
export RL_LOG_LEVEL=INFO

# Override the default port to 8001
export UVICORN_PORT=8001

echo ""
echo "Starting RL Trading Backend with configuration:"
echo "  - ActorService: ENABLED"
echo "  - Strategy: hardcoded (HOLD-only)"
echo "  - Markets: $RL_MARKET_TICKERS"
echo "  - Throttle: ${RL_ACTOR_THROTTLE_MS}ms between actions"
echo ""
echo "Endpoints available:"
echo "  - Health: http://localhost:8001/rl/health"
echo "  - Status: http://localhost:8001/rl/status"
echo "  - WebSocket: ws://localhost:8001/rl/ws"
echo ""
echo "Press Ctrl+C to stop"
echo "------------------------------------------"

# Run the RL backend service on port 8001
uv run uvicorn kalshiflow_rl.app:app \
    --host 0.0.0.0 \
    --port 8001 \
    --log-level info \
    --reload