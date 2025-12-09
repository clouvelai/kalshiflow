#!/bin/bash
# Run RL Orderbook Collector E2E Test

echo "🧪 Running RL Orderbook Collector E2E Test..."
echo "============================================"

# Set test environment
export ENVIRONMENT="test"
export RL_MARKET_TICKERS="KXCABOUT-29,KXFEDDECISION-25DEC,KXLLM1-25DEC31"

# Check for Kalshi credentials
if [ -z "$KALSHI_API_KEY_ID" ] || [ -z "$KALSHI_PRIVATE_KEY_PATH" ]; then
    echo "⚠️  WARNING: Kalshi credentials not set, some tests may fail"
fi

# Change to backend directory
cd "$(dirname "$0")/.." || exit 1

echo ""
echo "📋 Configuration:"
echo "  - Markets: ${RL_MARKET_TICKERS}"
echo "  - Environment: ${ENVIRONMENT}"
echo ""

# Run the E2E test
echo "🚀 Starting E2E test..."
uv run pytest tests/test_rl_orderbook_e2e.py -v -s --log-cli-level=INFO

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ E2E TEST PASSED"
    echo "============================================"
    echo "The RL Orderbook Collector service is ready for deployment!"
    exit 0
else
    echo ""
    echo "❌ E2E TEST FAILED"
    echo "============================================"
    echo "Please review the test output above for details."
    exit 1
fi