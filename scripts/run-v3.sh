#!/bin/bash
# =============================================================================
# TRADER V3 Launcher
# =============================================================================
# Runs the V3 Trader with deep_agent strategy (only remaining strategy)
#
# Usage:
#   ./scripts/run-v3.sh                    # Default: paper trading, lifecycle mode
#   ./scripts/run-v3.sh paper              # Paper trading (demo API)
#   ./scripts/run-v3.sh production         # Production (live API) - BE CAREFUL!
#
# Environment variables (optional overrides):
#   V3_ENABLE_TRADING_CLIENT=true          # Enable trading (default: true)
#   RL_MODE=lifecycle                      # Market discovery mode (default: lifecycle)
#   LIFECYCLE_MAX_MARKETS=1000             # Max tracked markets (default: 1000)
#
# The deep_agent strategy is configured via:
#   backend/src/kalshiflow_rl/traderv3/strategies/config/deep_agent.yaml
# =============================================================================

set -e

# Default to paper trading if not specified
ENVIRONMENT="${1:-paper}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}"
echo "============================================================"
echo "  TRADER V3 - Deep Agent Strategy"
echo "============================================================"
echo -e "${NC}"

# Switch to the appropriate environment
echo -e "${YELLOW}Environment: ${ENVIRONMENT}${NC}"

# Load the environment file
ENV_FILE=".env.${ENVIRONMENT}"
if [ ! -f "$ENV_FILE" ]; then
    echo -e "${RED}Error: Environment file not found: ${ENV_FILE}${NC}"
    echo "Available environments:"
    ls -1 .env.* 2>/dev/null || echo "  No .env files found"
    exit 1
fi

# Source the environment file
set -a
source "$ENV_FILE"
set +a

# Set default V3 configuration
export ENVIRONMENT="${ENVIRONMENT}"
export V3_ENABLE_TRADING_CLIENT="${V3_ENABLE_TRADING_CLIENT:-true}"
export RL_MODE="${RL_MODE:-lifecycle}"
export LIFECYCLE_MAX_MARKETS="${LIFECYCLE_MAX_MARKETS:-1000}"

# Enable entity system for Deep Agent (Reddit → Entities → Price Impacts)
export V3_ENTITY_SYSTEM_ENABLED="${V3_ENTITY_SYSTEM_ENABLED:-true}"

# Validate we're not accidentally using production API for paper mode
if [ "$ENVIRONMENT" = "paper" ]; then
    if [[ "$KALSHI_API_URL" == *"api.elections.kalshi.com"* ]]; then
        echo -e "${RED}ERROR: Paper mode but using production API URL!${NC}"
        echo "       Please check your .env.paper file"
        exit 1
    fi
    echo -e "${GREEN}✓ Using demo API (safe for paper trading)${NC}"
fi

if [ "$ENVIRONMENT" = "production" ]; then
    echo -e "${RED}"
    echo "============================================================"
    echo "  WARNING: PRODUCTION MODE - REAL MONEY AT RISK!"
    echo "============================================================"
    echo -e "${NC}"
    read -p "Are you sure you want to continue? (yes/no): " confirm
    if [ "$confirm" != "yes" ]; then
        echo "Aborted."
        exit 1
    fi
fi

# Display configuration
echo ""
echo -e "${CYAN}Configuration:${NC}"
echo "  - Trading Client: ${V3_ENABLE_TRADING_CLIENT}"
echo "  - Market Mode: ${RL_MODE}"
echo "  - Max Markets: ${LIFECYCLE_MAX_MARKETS}"
echo "  - Entity System: ${V3_ENTITY_SYSTEM_ENABLED} (Reddit → Entities → Price Impacts)"
echo "  - Strategy: deep_agent (self-improving agent)"
echo "  - Anthropic API: $([ -n \"$ANTHROPIC_API_KEY\" ] && echo 'configured' || echo 'NOT SET!')"
echo ""

# Check for required API credentials
if [ -z "$KALSHI_API_KEY_ID" ]; then
    echo -e "${RED}Error: Missing KALSHI_API_KEY_ID${NC}"
    echo "Please ensure KALSHI_API_KEY_ID is set in ${ENV_FILE}"
    exit 1
fi

# Check for key - either KALSHI_PRIVATE_KEY_CONTENT or KALSHI_PRIVATE_KEY_PATH
if [ -z "$KALSHI_PRIVATE_KEY_CONTENT" ] && [ -z "$KALSHI_PRIVATE_KEY_PATH" ]; then
    echo -e "${RED}Error: Missing required API key${NC}"
    echo "Please ensure KALSHI_PRIVATE_KEY_CONTENT or KALSHI_PRIVATE_KEY_PATH is set in ${ENV_FILE}"
    exit 1
fi

# If we have a path but not content, read the content from the file
if [ -z "$KALSHI_PRIVATE_KEY_CONTENT" ] && [ -n "$KALSHI_PRIVATE_KEY_PATH" ]; then
    if [ -f "$KALSHI_PRIVATE_KEY_PATH" ]; then
        export KALSHI_PRIVATE_KEY_CONTENT=$(cat "$KALSHI_PRIVATE_KEY_PATH")
        echo -e "${GREEN}✓ Read private key from ${KALSHI_PRIVATE_KEY_PATH}${NC}"
    else
        echo -e "${RED}Error: Private key file not found: ${KALSHI_PRIVATE_KEY_PATH}${NC}"
        exit 1
    fi
fi

# Change to backend directory
cd backend

echo -e "${CYAN}Starting V3 Trader...${NC}"
echo "  - API URL: ${KALSHI_API_URL}"
echo "  - WS URL: ${KALSHI_WS_URL}"
echo ""

# Run the V3 trader
exec uv run uvicorn src.kalshiflow_rl.traderv3.app:app \
    --host 0.0.0.0 \
    --port 8005 \
    --reload \
    --log-level info
