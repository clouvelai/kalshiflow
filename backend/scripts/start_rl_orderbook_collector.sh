#!/bin/bash
# Startup script for RL Orderbook Collector Service

echo "🚀 Starting RL Orderbook Collector Service..."

# Set default environment variables if not already set
export PYTHONPATH="${PYTHONPATH}:/app/backend/src"
export UVICORN_HOST="${UVICORN_HOST:-0.0.0.0}"
export UVICORN_PORT="${UVICORN_PORT:-${PORT:-8000}}"

# Set default RL configuration if not provided
export RL_ORDERBOOK_BATCH_SIZE="${RL_ORDERBOOK_BATCH_SIZE:-100}"
export RL_ORDERBOOK_FLUSH_INTERVAL="${RL_ORDERBOOK_FLUSH_INTERVAL:-1.0}"
export RL_ORDERBOOK_SAMPLE_RATE="${RL_ORDERBOOK_SAMPLE_RATE:-1}"

# Log configuration
echo "📋 Configuration:"
echo "  - Markets: ${RL_MARKET_TICKERS}"
echo "  - Port: ${UVICORN_PORT}"
echo "  - Batch Size: ${RL_ORDERBOOK_BATCH_SIZE}"
echo "  - Flush Interval: ${RL_ORDERBOOK_FLUSH_INTERVAL}s"
echo "  - Sample Rate: 1/${RL_ORDERBOOK_SAMPLE_RATE}"

# Check for required environment variables
if [ -z "$DATABASE_URL" ]; then
    echo "❌ ERROR: DATABASE_URL is not set"
    exit 1
fi

if [ -z "$KALSHI_API_KEY_ID" ]; then
    echo "❌ ERROR: KALSHI_API_KEY_ID is not set"
    exit 1
fi

if [ -z "$KALSHI_PRIVATE_KEY_CONTENT" ]; then
    echo "❌ ERROR: KALSHI_PRIVATE_KEY_CONTENT is not set"
    exit 1
fi

if [ -z "$RL_MARKET_TICKERS" ]; then
    echo "⚠️  WARNING: RL_MARKET_TICKERS is not set, using default markets"
fi

# Change to backend directory
cd /app/backend || cd backend || {
    echo "❌ ERROR: Could not find backend directory"
    exit 1
}

# Start the service
echo "🔧 Starting Uvicorn server..."
exec uvicorn kalshiflow_rl.app:app \
    --host "$UVICORN_HOST" \
    --port "$UVICORN_PORT" \
    --log-level info \
    --access-log