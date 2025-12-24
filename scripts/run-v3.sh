#!/bin/bash

# TRADER V3 Runner Script
# Simple, clean launcher for V3 trader

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}       TRADER V3 - CLEAN ARCHITECTURE${NC}"
echo -e "${BLUE}============================================${NC}"

# Navigate to backend directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/../backend"
cd "$BACKEND_DIR"

# Parse arguments
ENVIRONMENT="${1:-paper}"
MODE="${2:-discovery}"  # Use discovery mode by default
MARKET_LIMIT="${3:-10}"  # Limit to 10 markets
PORT="${4:-8005}"

# Show configuration
echo -e "${YELLOW}Configuration:${NC}"
echo -e "  Environment: ${GREEN}$ENVIRONMENT${NC}"
echo -e "  Market Mode: ${GREEN}$MODE${NC}"
echo -e "  Market Limit: ${GREEN}$MARKET_LIMIT${NC}"
echo -e "  Port: ${GREEN}$PORT${NC}"
echo ""

# Set environment variables
export ENVIRONMENT="$ENVIRONMENT"
export RL_MODE="$MODE"  # Use same variable as RL trader
export RL_ORDERBOOK_MARKET_LIMIT="$MARKET_LIMIT"  # Use same variable as RL trader
export V3_PORT="$PORT"

# V3-specific configuration
export V3_LOG_LEVEL="INFO"
export V3_CALIBRATION_DURATION="10.0"
export V3_HEALTH_CHECK_INTERVAL="5.0"
export V3_WS_RECONNECT_INTERVAL="5.0"

# Trading client configuration (enabled for paper environment)
if [ "$ENVIRONMENT" = "paper" ]; then
    export V3_ENABLE_TRADING_CLIENT="true"
    export V3_TRADING_MAX_ORDERS="10"
    export V3_TRADING_MAX_POSITION_SIZE="100"
    echo -e "${GREEN}✓ Trading client enabled (paper mode)${NC}"
else
    export V3_ENABLE_TRADING_CLIENT="false"
    echo -e "${YELLOW}⚠ Trading client disabled (production safety)${NC}"
fi

# Check if .env file exists
ENV_FILE=".env.$ENVIRONMENT"
if [ ! -f "$ENV_FILE" ]; then
    echo -e "${RED}❌ Environment file not found: $ENV_FILE${NC}"
    echo -e "${YELLOW}Please create $ENV_FILE with your Kalshi credentials${NC}"
    exit 1
fi

# Source environment file
echo -e "${GREEN}✓ Loading environment from $ENV_FILE${NC}"
set -a
source "$ENV_FILE"
set +a

# Validate required variables
if [ -z "$KALSHI_API_KEY_ID" ] || [ -z "$KALSHI_PRIVATE_KEY_CONTENT" ]; then
    echo -e "${RED}❌ Missing required Kalshi credentials in $ENV_FILE${NC}"
    exit 1
fi

# Determine API URLs based on environment
if [ "$ENVIRONMENT" = "paper" ]; then
    export KALSHI_API_URL="${KALSHI_API_URL:-https://demo-api.kalshi.co/trade-api/v2}"
    export KALSHI_WS_URL="${KALSHI_WS_URL:-wss://demo-api.kalshi.co/trade-api/ws/v2}"
    echo -e "${YELLOW}📝 Using DEMO environment (paper trading)${NC}"
else
    export KALSHI_API_URL="${KALSHI_API_URL:-https://api.elections.kalshi.com/trade-api/v2}"
    export KALSHI_WS_URL="${KALSHI_WS_URL:-wss://api.elections.kalshi.com/trade-api/ws/v2}"
    echo -e "${YELLOW}💰 Using PRODUCTION environment${NC}"
fi

echo ""
echo -e "${BLUE}Starting TRADER V3...${NC}"
echo ""

# Run the V3 app directly
uv run python -m kalshiflow_rl.traderv3.app

echo ""
echo -e "${GREEN}✅ TRADER V3 stopped${NC}"