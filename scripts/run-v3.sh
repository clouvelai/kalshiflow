#!/bin/bash
# =============================================================================
# TRADER V3 Launcher
# =============================================================================
# Runs the V3 Trader with Kalshi-Polymarket arbitrage strategy.
# Lifecycle discovery is the default; set V3_MARKET_TICKERS to target specifics.
#
# Usage:
#   ./scripts/run-v3.sh                    # Default: paper trading
#   ./scripts/run-v3.sh paper              # Paper trading (demo API)
#   ./scripts/run-v3.sh production         # Production (live API) - BE CAREFUL!
#
# Arb system enabled via V3_ARB_ENABLED=true and V3_POLYMARKET_ENABLED=true
# =============================================================================

set -e

ENVIRONMENT="${1:-paper}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}"
echo "============================================================"
echo "  TRADER V3 - Kalshi-Polymarket Arbitrage"
echo "============================================================"
echo -e "${NC}"

# Load environment file
ENV_FILE=".env.${ENVIRONMENT}"
if [ ! -f "$ENV_FILE" ]; then
    echo -e "${RED}Error: Environment file not found: ${ENV_FILE}${NC}"
    ls -1 .env.* 2>/dev/null || echo "  No .env files found"
    exit 1
fi

set -a
source "$ENV_FILE"
set +a
export ENVIRONMENT="${ENVIRONMENT}"

# Safety: paper mode must use demo API
if [ "$ENVIRONMENT" = "paper" ]; then
    if [[ "$KALSHI_API_URL" == *"api.elections.kalshi.com"* ]]; then
        echo -e "${RED}ERROR: Paper mode but using production API URL!${NC}"
        exit 1
    fi
    echo -e "${GREEN}Using demo API (safe for paper trading)${NC}"
fi

# Safety: production confirmation
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

# Required credentials
if [ -z "$KALSHI_API_KEY_ID" ]; then
    echo -e "${RED}Error: Missing KALSHI_API_KEY_ID in ${ENV_FILE}${NC}"
    exit 1
fi

if [ -z "$KALSHI_PRIVATE_KEY_CONTENT" ] && [ -z "$KALSHI_PRIVATE_KEY_PATH" ]; then
    echo -e "${RED}Error: Missing KALSHI_PRIVATE_KEY_CONTENT or KALSHI_PRIVATE_KEY_PATH in ${ENV_FILE}${NC}"
    exit 1
fi

# Read key from file if needed
if [ -z "$KALSHI_PRIVATE_KEY_CONTENT" ] && [ -n "$KALSHI_PRIVATE_KEY_PATH" ]; then
    if [ -f "$KALSHI_PRIVATE_KEY_PATH" ]; then
        export KALSHI_PRIVATE_KEY_CONTENT=$(cat "$KALSHI_PRIVATE_KEY_PATH")
        echo -e "${GREEN}Read private key from ${KALSHI_PRIVATE_KEY_PATH}${NC}"
    else
        echo -e "${RED}Error: Private key file not found: ${KALSHI_PRIVATE_KEY_PATH}${NC}"
        exit 1
    fi
fi

if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo -e "${RED}Error: Missing ANTHROPIC_API_KEY in ${ENV_FILE}${NC}"
    exit 1
fi

if [ -z "$SUPABASE_URL" ]; then
    echo -e "${RED}Error: Missing SUPABASE_URL in ${ENV_FILE}${NC}"
    exit 1
fi

if [ -z "$SUPABASE_KEY" ] && [ -z "$SUPABASE_ANON_KEY" ]; then
    echo -e "${RED}Error: Missing SUPABASE_KEY or SUPABASE_ANON_KEY in ${ENV_FILE}${NC}"
    exit 1
fi

# Non-critical warnings
[ -z "$OPENAI_API_KEY" ] && echo -e "${YELLOW}Warning: OPENAI_API_KEY not set (needed for embeddings)${NC}"

# Display config
echo ""
echo -e "${CYAN}Configuration:${NC}"
echo "  - Environment: ${ENVIRONMENT}"
echo "  - API URL: ${KALSHI_API_URL}"
echo "  - Target tickers: ${V3_MARKET_TICKERS:-<lifecycle discovery>}"
echo ""

cd backend

echo -e "${CYAN}Starting V3 Trader...${NC}"

exec uv run uvicorn src.kalshiflow_rl.traderv3.app:app \
    --host 0.0.0.0 \
    --port 8005 \
    --reload \
    --log-level info
