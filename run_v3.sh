#!/bin/bash

# V3 Trader launcher script
# Simple, direct execution with clean process management

set -e

# Kill any existing process on port 8005
echo "Checking for existing process on port 8005..."
if lsof -Pi :8005 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "Killing existing process on port 8005..."
    lsof -Pi :8005 -sTCP:LISTEN -t | xargs kill -9
    sleep 1
fi

# Set V3 specific environment
export TRADER_VERSION=v3
export BACKEND_PORT=8005
export ENVIRONMENT=${ENVIRONMENT:-paper}
export RL_MODE=discovery
export RL_ORDERBOOK_MARKET_LIMIT=100

echo "Starting V3 Trader..."
echo "  Version: $TRADER_VERSION"
echo "  Port: $BACKEND_PORT"
echo "  Environment: $ENVIRONMENT"
echo "  Mode: $RL_MODE"
echo "  Market Limit: $RL_ORDERBOOK_MARKET_LIMIT"

# Navigate to backend directory
cd backend

# Start V3 trader with uvicorn
uv run uvicorn src.kalshiflow_rl.traderv3.app:app \
    --host 0.0.0.0 \
    --port $BACKEND_PORT \
    --reload \
    --log-level info