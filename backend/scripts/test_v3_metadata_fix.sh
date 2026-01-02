#!/bin/bash
# Test script for V3 state metadata fix

echo "=================================="
echo "Testing V3 State Metadata Fix"
echo "=================================="

# Kill any existing V3 trader
echo "Stopping any existing V3 trader..."
pkill -f "traderv3.app" 2>/dev/null
sleep 2

# Start V3 trader with trading enabled
echo "Starting V3 trader with trading enabled..."
cd backend

# Set environment for paper trading with trading client
export ENVIRONMENT=paper
export V3_ENABLE_TRADING_CLIENT=true
export V3_TRADING_MODE=paper
export V3_LOG_LEVEL=DEBUG  # Enable debug logging to see our validation messages
export V3_MAX_MARKETS=5  # Just a few markets for testing

# Start in background
echo "Starting trader on port 8005..."
uv run python src/kalshiflow_rl/traderv3/app.py &
TRADER_PID=$!

echo "Waiting for trader to start..."
sleep 5

# Check if trader is running
if ! ps -p $TRADER_PID > /dev/null; then
    echo "❌ Trader failed to start!"
    exit 1
fi

echo "Trader started with PID $TRADER_PID"

# Run the metadata validation test
echo ""
echo "Running metadata validation test..."
echo "=================================="
uv run python src/kalshiflow_rl/traderv3/tests/test_state_metadata.py

TEST_RESULT=$?

# Kill the trader
echo ""
echo "Stopping trader..."
kill $TRADER_PID 2>/dev/null
wait $TRADER_PID 2>/dev/null

echo ""
echo "=================================="
if [ $TEST_RESULT -eq 0 ]; then
    echo "✅ STATE METADATA FIX VERIFIED!"
    echo "Each state now has clean, appropriate metadata."
else
    echo "❌ STATE METADATA ISSUES DETECTED"
    echo "Check the validation output above for details."
fi
echo "=================================="

exit $TEST_RESULT