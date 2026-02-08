#!/bin/bash
# =============================================================================
# TRADER V3 Launcher
# =============================================================================
# Runs the V3 Trader with single-event arbitrage and Captain agent.
# Lifecycle discovery is the default; set V3_MARKET_TICKERS to target specifics.
#
# Usage:
#   ./scripts/run-v3.sh                    # Default: paper trading
#   ./scripts/run-v3.sh paper              # Paper trading (demo API)
#   ./scripts/run-v3.sh production         # Production (live API) - BE CAREFUL!
#
# =============================================================================

set -e

ENVIRONMENT="${1:-paper}"
PORT=8005

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

# Spinner characters
SPINNER="⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

echo -e "${CYAN}${BOLD}"
echo "============================================================"
echo "  TRADER V3 - Single-Event Arbitrage + Captain Agent"
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
echo "  Environment: ${ENVIRONMENT}"
echo "  API URL: ${KALSHI_API_URL}"
echo "  Single-Arb: ${V3_SINGLE_ARB_ENABLED:-true}"
echo "  Captain: ${V3_SINGLE_ARB_CAPTAIN_ENABLED:-true}"
echo ""

# Kill any existing process on the port
if lsof -ti:$PORT >/dev/null 2>&1; then
    echo -e "${YELLOW}Stopping existing process on port $PORT...${NC}"
    lsof -ti:$PORT | xargs kill -9 2>/dev/null || true
    sleep 1
fi

cd backend

# Ensure logs directory exists
mkdir -p logs

# Start uvicorn in background, redirect output to log file
LOG_FILE="logs/v3-trader.log"
echo -e "${CYAN}Starting uvicorn (logs: $LOG_FILE)...${NC}"
uv run uvicorn src.kalshiflow_rl.traderv3.app:app \
    --host 0.0.0.0 \
    --port $PORT \
    --reload \
    --log-level warning \
    >> "$LOG_FILE" 2>&1 &
UVICORN_PID=$!

# Cleanup on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down...${NC}"
    kill $UVICORN_PID 2>/dev/null || true
    exit 0
}
trap cleanup SIGINT SIGTERM

# Wait for server to be reachable (up to 60 seconds)
echo -e "${DIM}Waiting for server...${NC}"
ATTEMPTS=0
MAX_ATTEMPTS=60
while ! curl -s --max-time 2 "http://localhost:$PORT/v3/health" >/dev/null 2>&1; do
    if ! kill -0 $UVICORN_PID 2>/dev/null; then
        echo -e "${RED}Server process died!${NC}"
        exit 1
    fi
    ATTEMPTS=$((ATTEMPTS + 1))
    if [ $ATTEMPTS -ge $MAX_ATTEMPTS ]; then
        echo -e "${RED}Timeout waiting for server${NC}"
        exit 1
    fi
    sleep 1
done
echo -e "${GREEN}Server listening on port $PORT${NC}"

# Poll status until ready (max 3 minutes)
echo ""
echo -e "${CYAN}${BOLD}Startup Progress:${NC}"
LAST_STATE=""
LAST_MARKETS=0
LAST_ORDERBOOK=0
LAST_EVENTS=0
SPINNER_IDX=0
START_TIME=$(date +%s)
MAX_STARTUP_SECONDS=180

while true; do
    # Check timeout
    ELAPSED=$(($(date +%s) - START_TIME))
    if [ $ELAPSED -ge $MAX_STARTUP_SECONDS ]; then
        echo ""
        echo -e "${YELLOW}Startup timeout after ${ELAPSED}s - Captain may still be initializing${NC}"
        echo -e "${DIM}Check logs: tail -f backend/logs/v3-trader.log${NC}"
        break
    fi
    # Get status
    STATUS=$(curl -s --max-time 2 "http://localhost:$PORT/v3/status" 2>/dev/null || echo "{}")

    STATE=$(echo "$STATUS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('state','unknown'))" 2>/dev/null || echo "unknown")
    MARKETS=$(echo "$STATUS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('components',{}).get('tracked_markets_state',{}).get('tracked',0))" 2>/dev/null || echo "0")
    ORDERBOOK=$(echo "$STATUS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('components',{}).get('orderbook_integration',{}).get('markets_connected',0))" 2>/dev/null || echo "0")
    EVENTS=$(echo "$STATUS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('components',{}).get('single_arb_coordinator',{}).get('events',0))" 2>/dev/null || echo "0")
    CAPTAIN_CYCLE=$(echo "$STATUS" | python3 -c "import sys,json; d=json.load(sys.stdin); c=d.get('components',{}).get('single_arb_coordinator',{}).get('captain',{}); print(c.get('cycle_count',0))" 2>/dev/null || echo "0")

    # Show progress when things change
    if [ "$STATE" != "$LAST_STATE" ]; then
        case "$STATE" in
            "initializing")
                echo -e "  ${YELLOW}Initializing...${NC}"
                ;;
            "connecting")
                echo -e "  ${YELLOW}Connecting to Kalshi API...${NC}"
                ;;
            "calibrating")
                echo -e "  ${YELLOW}Calibrating trading client...${NC}"
                ;;
            "ready")
                echo -e "  ${GREEN}State: READY${NC}"
                ;;
        esac
        LAST_STATE="$STATE"
    fi

    if [ "$MARKETS" != "$LAST_MARKETS" ] && [ "$MARKETS" != "0" ]; then
        echo -e "  ${GREEN}Markets discovered: $MARKETS${NC}"
        LAST_MARKETS="$MARKETS"
    fi

    if [ "$ORDERBOOK" != "$LAST_ORDERBOOK" ] && [ "$ORDERBOOK" != "0" ]; then
        echo -e "  ${GREEN}Orderbook connected: $ORDERBOOK markets${NC}"
        LAST_ORDERBOOK="$ORDERBOOK"
    fi

    if [ "$EVENTS" != "$LAST_EVENTS" ] && [ "$EVENTS" != "0" ]; then
        echo -e "  ${GREEN}Event index: $EVENTS events${NC}"
        LAST_EVENTS="$EVENTS"
    fi

    # Check if ready to start
    if [ "$STATE" = "ready" ] && [ "$MARKETS" -gt 0 ] && [ "$ORDERBOOK" -gt 0 ]; then
        # Check if Captain is running (look for running=True in captain stats)
        CAPTAIN_RUNNING=$(echo "$STATUS" | python3 -c "import sys,json; d=json.load(sys.stdin); c=d.get('components',{}).get('single_arb_coordinator',{}).get('captain',{}); print(c.get('running', False))" 2>/dev/null || echo "False")
        if [ "$CAPTAIN_RUNNING" = "True" ]; then
            CYCLE=$(echo "$STATUS" | python3 -c "import sys,json; d=json.load(sys.stdin); c=d.get('components',{}).get('single_arb_coordinator',{}).get('captain',{}); print(c.get('cycle_count',0))" 2>/dev/null || echo "0")
            echo ""
            echo -e "${GREEN}${BOLD}Captain is running! (cycle $CYCLE)${NC}"
            break
        fi
    fi

    # Spinner
    CHAR="${SPINNER:$SPINNER_IDX:1}"
    SPINNER_IDX=$(( (SPINNER_IDX + 1) % ${#SPINNER} ))
    printf "\r  ${DIM}%s Waiting for Captain to start...${NC}  " "$CHAR"

    sleep 1
done

# Show final status
echo ""
echo -e "${CYAN}============================================================${NC}"
echo -e "${GREEN}${BOLD}V3 Trader Ready${NC}"
echo -e "${CYAN}============================================================${NC}"
echo ""
echo -e "  Health:    http://localhost:$PORT/v3/health"
echo -e "  Status:    http://localhost:$PORT/v3/status"
echo -e "  Dashboard: http://localhost:5173/arb"
echo ""
echo -e "${DIM}Press Ctrl+C to stop${NC}"
echo ""

# Now tail the log file for ongoing output
LOG_FILE="logs/v3-trader.log"
if [ -f "$LOG_FILE" ]; then
    tail -f "$LOG_FILE" | grep --line-buffered -E "CAPTAIN|CYCLE|ARB_OPPORTUNITY|ERROR|trade_commando|mentions" &
    TAIL_PID=$!

    # Update cleanup to also kill tail
    cleanup() {
        echo ""
        echo -e "${YELLOW}Shutting down...${NC}"
        kill $TAIL_PID 2>/dev/null || true
        kill $UVICORN_PID 2>/dev/null || true
        exit 0
    }
    trap cleanup SIGINT SIGTERM
fi

# Wait for uvicorn to exit
wait $UVICORN_PID
