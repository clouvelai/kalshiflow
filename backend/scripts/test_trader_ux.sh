#!/bin/bash

# Test script for Trader UX MVP
# This script starts the RL backend service and verifies the trader dashboard works

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting Trader UX MVP Test...${NC}"
echo "========================================="

# Check if port 8002 is available
if lsof -Pi :8002 -sTCP:LISTEN -t >/dev/null ; then
    echo -e "${RED}Port 8002 is already in use. Please stop the service using it.${NC}"
    exit 1
fi

# Check if port 8000 is being used (main kalshiflow backend)
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null ; then
    echo -e "${YELLOW}Note: Main Kalshiflow backend is running on port 8000${NC}"
fi

# Start the RL backend service
echo -e "${GREEN}Starting RL Backend Service on port 8002...${NC}"
cd backend

# Export test market tickers for the trader
export RL_MARKET_TICKERS="INXD-25JAN03"
export RL_LOG_LEVEL="INFO"
export ENVIRONMENT=paper  # Use paper trading for testing

# Start the service in the background
uv run uvicorn kalshiflow_rl.app:app --port 8002 --reload &
RL_PID=$!

echo "Waiting for RL backend to start..."
sleep 5

# Check if the service started successfully
if ! ps -p $RL_PID > /dev/null; then
    echo -e "${RED}Failed to start RL backend service${NC}"
    exit 1
fi

echo -e "${GREEN}RL Backend started with PID: $RL_PID${NC}"

# Test the health endpoint
echo "Testing health endpoint..."
HEALTH_RESPONSE=$(curl -s http://localhost:8002/rl/health || echo "FAILED")
if [[ "$HEALTH_RESPONSE" == "FAILED" ]]; then
    echo -e "${RED}Health check failed${NC}"
    kill $RL_PID 2>/dev/null || true
    exit 1
fi

echo -e "${GREEN}Health check passed:${NC} $HEALTH_RESPONSE"

# Instructions for manual testing
echo ""
echo "========================================="
echo -e "${GREEN}RL Backend Service is running!${NC}"
echo ""
echo "To test the Trader Dashboard:"
echo "1. Open a new terminal"
echo "2. Navigate to the frontend directory: cd frontend"
echo "3. Start the frontend: npm run dev"
echo "4. Open browser to: http://localhost:5173/trader"
echo ""
echo "The dashboard should show:"
echo "- Live connection status"
echo "- Trader state panel (portfolio, positions, orders)"
echo "- Action feed (trading decisions)"
echo ""
echo "WebSocket endpoint: ws://localhost:8002/rl/ws"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop the service${NC}"
echo "========================================="

# Keep the service running
trap "echo 'Stopping RL backend...'; kill $RL_PID 2>/dev/null || true; exit" INT TERM
wait $RL_PID