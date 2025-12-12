#!/bin/bash
# Local testing script for RL Orderbook Collector Service

echo "🧪 Starting RL Orderbook Collector Service (Test Mode)..."

# Set test environment
export ENVIRONMENT="local"
export DEBUG="true"

# Set test markets
export RL_MARKET_TICKERS="KXCABOUT-29,KXFEDDECISION-25DEC,KXLLM1-25DEC31"

# Set test configuration
export RL_ORDERBOOK_BATCH_SIZE="10"
export RL_ORDERBOOK_FLUSH_INTERVAL="2.0"
export RL_ORDERBOOK_SAMPLE_RATE="1"

# Log configuration
echo "📋 Test Configuration:"
echo "  - Markets: ${RL_MARKET_TICKERS}"
echo "  - Environment: ${ENVIRONMENT}"
echo "  - Debug: ${DEBUG}"
echo "  - Batch Size: ${RL_ORDERBOOK_BATCH_SIZE}"
echo "  - Flush Interval: ${RL_ORDERBOOK_FLUSH_INTERVAL}s"

# Check for required environment variables
if [ -z "$DATABASE_URL" ]; then
    echo "⚠️  WARNING: DATABASE_URL not set, using test database"
    export DATABASE_URL="postgresql://test:test@localhost:5432/test"
fi

if [ -z "$KALSHI_API_KEY_ID" ] || [ -z "$KALSHI_PRIVATE_KEY_PATH" ]; then
    echo "❌ ERROR: Kalshi credentials not configured"
    echo "Please set KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH"
    exit 1
fi

# Change to backend directory
cd "$(dirname "$0")/.." || exit 1

echo "🚀 Starting service on http://localhost:8002"
echo "📊 Health endpoint: http://localhost:8002/rl/health"
echo "🔌 WebSocket endpoint: ws://localhost:8002/rl/ws"
echo ""
echo "Press Ctrl+C to stop..."

# Start the service
uv run uvicorn kalshiflow_rl.app:app \
    --host 127.0.0.1 \
    --port 8002 \
    --reload \
    --log-level info