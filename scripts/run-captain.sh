#!/bin/bash
# =============================================================================
# Captain Quick Launcher
# =============================================================================
# Streamlined launcher for Captain agent development.
# Faster than run-v3.sh: no --reload, relaxed env validation, shorter timeouts.
#
# Usage:
#   ./scripts/run-captain.sh              # Default: paper trading
#   ./scripts/run-captain.sh paper        # Paper trading (demo API)
#
# =============================================================================

set -e

# Always run from project root (where .env files live)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

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

ts() { date "+%H:%M:%S"; }

echo -e "${CYAN}${BOLD}[$(ts)] Captain Quick Start (${ENVIRONMENT})${NC}"

# Load environment file
ENV_FILE=".env.${ENVIRONMENT}"
if [ ! -f "$ENV_FILE" ]; then
    echo -e "${RED}[$(ts)] Error: ${ENV_FILE} not found${NC}"
    exit 1
fi

set -a
source "$ENV_FILE"
set +a
export ENVIRONMENT="${ENVIRONMENT}"

# Force-enable Captain + Sniper
export V3_SINGLE_ARB_ENABLED=true
export V3_SINGLE_ARB_CAPTAIN_ENABLED=true
export V3_SNIPER_ENABLED=true

# Safety: paper mode must use demo API
if [ "$ENVIRONMENT" = "paper" ]; then
    if [[ "$KALSHI_API_URL" == *"api.elections.kalshi.com"* ]]; then
        echo -e "${RED}[$(ts)] ERROR: Paper mode but using production API URL!${NC}"
        exit 1
    fi
fi

# Critical env vars (hard fail)
if [ -z "$KALSHI_API_KEY_ID" ]; then
    echo -e "${RED}[$(ts)] Error: Missing KALSHI_API_KEY_ID${NC}"
    exit 1
fi

if [ -z "$KALSHI_PRIVATE_KEY_CONTENT" ] && [ -z "$KALSHI_PRIVATE_KEY_PATH" ]; then
    echo -e "${RED}[$(ts)] Error: Missing KALSHI_PRIVATE_KEY_CONTENT or KALSHI_PRIVATE_KEY_PATH${NC}"
    exit 1
fi

# Read key from file if needed
if [ -z "$KALSHI_PRIVATE_KEY_CONTENT" ] && [ -n "$KALSHI_PRIVATE_KEY_PATH" ]; then
    if [ -f "$KALSHI_PRIVATE_KEY_PATH" ]; then
        export KALSHI_PRIVATE_KEY_CONTENT=$(cat "$KALSHI_PRIVATE_KEY_PATH")
    else
        echo -e "${RED}[$(ts)] Error: Key file not found: ${KALSHI_PRIVATE_KEY_PATH}${NC}"
        exit 1
    fi
fi

if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo -e "${RED}[$(ts)] Error: Missing ANTHROPIC_API_KEY${NC}"
    exit 1
fi

if [ -z "$GOOGLE_API_KEY" ]; then
    echo -e "${RED}[$(ts)] Error: Missing GOOGLE_API_KEY (required for mentions rule parsing)${NC}"
    exit 1
fi

# Non-critical warnings (don't exit)
[ -z "$SUPABASE_URL" ] && echo -e "${YELLOW}[$(ts)] Warning: SUPABASE_URL not set (DB features disabled)${NC}"
[ -z "$OPENAI_API_KEY" ] && echo -e "${YELLOW}[$(ts)] Warning: OPENAI_API_KEY not set (embeddings disabled)${NC}"
[ -z "$TAVILY_API_KEY" ] && echo -e "${YELLOW}[$(ts)] Warning: TAVILY_API_KEY not set (news search disabled)${NC}"

# Kill any existing process on the port
if lsof -ti:$PORT >/dev/null 2>&1; then
    echo -e "${YELLOW}[$(ts)] Stopping existing process on port $PORT...${NC}"
    lsof -ti:$PORT | xargs kill -9 2>/dev/null || true
    sleep 1
fi

# Start frontend dev server if not already running
if ! lsof -ti:5173 >/dev/null 2>&1; then
    echo -e "${CYAN}[$(ts)] Starting frontend dev server...${NC}"
    cd frontend && npm run dev > /dev/null 2>&1 &
    FRONTEND_PID=$!
    cd ..
    echo -e "${GREEN}[$(ts)] Frontend starting on http://localhost:5173${NC}"
else
    echo -e "${DIM}[$(ts)] Frontend already running on port 5173${NC}"
    FRONTEND_PID=""
fi

cd backend
mkdir -p logs

# Start uvicorn WITHOUT --reload (faster startup, no file watcher)
LOG_FILE="logs/v3-trader.log"
echo -e "${CYAN}[$(ts)] Starting uvicorn (no reload, logs: $LOG_FILE)...${NC}"
# App writes to LOG_FILE via its own RotatingFileHandler.
# Only redirect stderr here for uvicorn startup errors.
uv run uvicorn src.kalshiflow_rl.traderv3.app:app \
    --host 0.0.0.0 \
    --port $PORT \
    --log-level warning \
    2>> "$LOG_FILE" &
UVICORN_PID=$!

cleanup() {
    echo ""
    echo -e "${YELLOW}[$(ts)] Shutting down...${NC}"
    kill $TAIL_PID 2>/dev/null || true
    kill $UVICORN_PID 2>/dev/null || true
    [ -n "$FRONTEND_PID" ] && kill $FRONTEND_PID 2>/dev/null || true
    exit 0
}
trap cleanup SIGINT SIGTERM

# Wait for health endpoint (timeout 30s)
echo -e "${DIM}[$(ts)] Waiting for server...${NC}"
for i in $(seq 1 30); do
    if curl -sf --max-time 2 "http://localhost:$PORT/v3/health" >/dev/null 2>&1; then
        echo -e "${GREEN}[$(ts)] Server listening on port $PORT${NC}"
        break
    fi
    if ! kill -0 $UVICORN_PID 2>/dev/null; then
        echo -e "${RED}[$(ts)] Server process died!${NC}"
        tail -20 "$LOG_FILE"
        exit 1
    fi
    sleep 1
done

# Wait for Captain to be running (timeout 90s)
echo -e "${DIM}[$(ts)] Waiting for Captain to start...${NC}"
for i in $(seq 1 45); do
    STATUS=$(curl -sf --max-time 2 "http://localhost:$PORT/v3/status" 2>/dev/null || echo "{}")
    CAPTAIN_RUNNING=$(echo "$STATUS" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    c = d.get('components', {}).get('single_arb_coordinator', {}).get('captain', {})
    print(c.get('running', False))
except: print('False')
" 2>/dev/null || echo "False")

    if [ "$CAPTAIN_RUNNING" = "True" ]; then
        EVENTS=$(echo "$STATUS" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d.get('components', {}).get('single_arb_coordinator', {}).get('events', 0))
except: print(0)
" 2>/dev/null || echo "0")
        echo -e "${GREEN}${BOLD}[$(ts)] Captain running! (events=$EVENTS)${NC}"
        break
    fi

    # Check for startup errors
    if grep -q "Failed to start\|CRITICAL" "$LOG_FILE" 2>/dev/null; then
        echo -e "${RED}[$(ts)] Startup error detected!${NC}"
        tail -20 "$LOG_FILE"
        break
    fi

    sleep 2
done

echo ""
echo -e "${GREEN}${BOLD}[$(ts)] Ready${NC}"
echo -e "  Health:    http://localhost:$PORT/v3/health"
echo -e "  Status:    http://localhost:$PORT/v3/status"
echo -e "  Control:   curl -X POST -H 'Content-Type: application/json' -d '{\"type\":\"captain_pause\"}' http://localhost:$PORT/v3/captain/control"
echo -e "  Dashboard: http://localhost:5173/arb"
echo -e "${DIM}Press Ctrl+C to stop${NC}"
echo ""

# Tail Captain-relevant logs
TAIL_PID=""
if [ -f "$LOG_FILE" ]; then
    tail -f "$LOG_FILE" | grep --line-buffered -E "CAPTAIN|CYCLE|DEFERRED_INIT|ARB_OPPORTUNITY|ERROR|trade_commando|mentions|SNIPER" &
    TAIL_PID=$!
fi

wait $UVICORN_PID
