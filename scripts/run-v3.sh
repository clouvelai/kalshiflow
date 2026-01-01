#!/bin/bash

# TRADER V3 Runner Script
# Unified launcher for V3 trader with frontend management
# This is the SINGLE source of truth for running V3

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

# Function to kill processes on a port
kill_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo -e "${YELLOW}Cleaning up existing process on port $port...${NC}"
        lsof -Pi :$port -sTCP:LISTEN -t | xargs kill -9 2>/dev/null || true
        sleep 1
    fi
}

# Parse arguments
ENVIRONMENT="${1:-paper}"
MODE="${2:-lifecycle}"  # Use lifecycle mode by default (market discovery via lifecycle events)
MARKET_LIMIT="${3:-1000}"  # Orderbook WS limit is 1000 markets

# Port configuration - SINGLE SOURCE OF TRUTH
BACKEND_PORT=8005  # V3 always uses 8005
FRONTEND_PORT=5173  # Standard Vite port

# Frontend configuration
FRONTEND_DIR="$SCRIPT_DIR/../frontend"

# Clean up any existing processes BEFORE showing config
echo -e "${YELLOW}Preparing clean environment...${NC}"
kill_port $BACKEND_PORT

# Show configuration
echo ""
echo -e "${YELLOW}Configuration:${NC}"
echo -e "  Environment: ${GREEN}$ENVIRONMENT${NC}"
echo -e "  Market Mode: ${GREEN}$MODE${NC}"
echo -e "  Market Limit: ${GREEN}$MARKET_LIMIT${NC}"
echo -e "  Backend Port: ${GREEN}$BACKEND_PORT${NC}"
echo -e "  Frontend Port: ${GREEN}$FRONTEND_PORT${NC}"
echo ""

# Set environment variables
export ENVIRONMENT="$ENVIRONMENT"
export RL_MODE="$MODE"  # Use same variable as RL trader
export RL_ORDERBOOK_MARKET_LIMIT="$MARKET_LIMIT"  # Use same variable as RL trader
export V3_PORT="$BACKEND_PORT"
export BACKEND_PORT="$BACKEND_PORT"  # For consistency

# V3-specific configuration
export V3_LOG_LEVEL="INFO"
export V3_CALIBRATION_DURATION="10.0"
export V3_HEALTH_CHECK_INTERVAL="5.0"
export V3_WS_RECONNECT_INTERVAL="5.0"

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

# Trading client configuration (AFTER env file loaded to override any defaults)
# This ensures our explicit settings take precedence
if [ "$ENVIRONMENT" = "paper" ]; then
    # Trading client settings
    export V3_ENABLE_TRADING_CLIENT="true"
    export V3_TRADING_MAX_ORDERS="100"
    export V3_TRADING_MAX_POSITION_SIZE="1000"

    # RLM (Reverse Line Movement) strategy - validated edge
    # ALWAYS use RLM for paper trading - this is the primary strategy
    export V3_TRADING_STRATEGY="rlm_no"
    export V3_ENABLE_WHALE_DETECTION="false"  # Disabled - RLM is primary strategy

    # RLM High Reliability Configuration (2.2% false positive rate)
    # See RLM_IMPROVEMENTS.md Section 10 for full reliability analysis
    export RLM_YES_THRESHOLD="0.70"      # >70% YES trades triggers signal (was 0.65)
    export RLM_MIN_TRADES="25"           # Minimum trades before evaluation (was 15)
    export RLM_MIN_PRICE_DROP="2"        # Minimum price drop in cents
    export RLM_CONTRACTS="20"            # Position size per signal (~$10-14 per trade)

    # Rate limiting for RLM execution
    export RLM_MAX_SIGNALS_PER_MINUTE="10"
    export RLM_TOKEN_REFILL_SECONDS="6"  # 60/10 = 6 seconds per token

    # Allow multiple positions/orders per market (for testing)
    export V3_ALLOW_MULTIPLE_POSITIONS="true"
    export V3_ALLOW_MULTIPLE_ORDERS="true"

    echo -e "${GREEN}✓ Trading client enabled (paper mode)${NC}"
    echo -e "${GREEN}✓ RLM strategy active (High Reliability: 70%/25, 2.2% FP rate)${NC}"
    echo -e "${GREEN}✓ Lifecycle mode for market discovery${NC}"
else
    export V3_ENABLE_TRADING_CLIENT="false"
    export V3_TRADING_STRATEGY="hold"
    echo -e "${YELLOW}⚠ Trading client disabled (production safety)${NC}"
fi

# Function to check if frontend is running
check_frontend_running() {
    if lsof -Pi :$FRONTEND_PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0  # Frontend is running
    else
        return 1  # Frontend is not running
    fi
}

# Function to start frontend
start_frontend() {
    if check_frontend_running; then
        echo -e "${GREEN}✓ Frontend already running on port $FRONTEND_PORT${NC}"
        echo -e "${BLUE}Access the V3 Trader Console at: ${GREEN}http://localhost:$FRONTEND_PORT/v3-trader${NC}"
    else
        echo -e "${BLUE}Starting frontend on port $FRONTEND_PORT...${NC}"
        
        # Check if frontend directory exists
        if [ ! -d "$FRONTEND_DIR" ]; then
            echo -e "${RED}❌ Frontend directory not found: $FRONTEND_DIR${NC}"
            return 1
        fi
        
        # Check if npm dependencies are installed
        if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
            echo -e "${YELLOW}Installing frontend dependencies...${NC}"
            (cd "$FRONTEND_DIR" && npm install)
        fi
        
        # Start frontend in background with proper backend port
        (
            cd "$FRONTEND_DIR"
            # Export VITE_BACKEND_PORT for frontend to know where backend is running
            export VITE_BACKEND_PORT=$BACKEND_PORT
            export VITE_V3_BACKEND_PORT=$BACKEND_PORT  # Explicit V3 port
            npm run dev > /dev/null 2>&1 &
        )
        
        # Wait for frontend to start
        echo -e "${YELLOW}Waiting for frontend to start...${NC}"
        sleep 4
        
        if check_frontend_running; then
            echo -e "${GREEN}✓ Frontend started successfully${NC}"
            echo -e "${BLUE}Access the V3 Trader Console at: ${GREEN}http://localhost:$FRONTEND_PORT/v3-trader${NC}"
        else
            echo -e "${RED}❌ Frontend failed to start${NC}"
            echo -e "${YELLOW}Try running 'cd frontend && npm run dev' manually to see errors${NC}"
            return 1
        fi
    fi
    echo ""
}

# Function to cleanup on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down...${NC}"
    
    # Kill the backend Python process if it's running
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null || true
    fi
    
    echo -e "${GREEN}✅ TRADER V3 stopped${NC}"
    echo -e "${YELLOW}Note: Frontend may still be running on port $FRONTEND_PORT${NC}"
}

# Setup trap for cleanup
trap cleanup EXIT INT TERM

echo ""
echo -e "${BLUE}Starting TRADER V3...${NC}"
echo ""

# Start frontend first if not running
start_frontend
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}Continuing without frontend...${NC}"
fi

echo -e "${BLUE}Starting V3 backend on port $BACKEND_PORT...${NC}"

# Run the V3 app directly and capture its PID
uv run python -m kalshiflow_rl.traderv3.app &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 3

# Check if backend started successfully
if kill -0 $BACKEND_PID 2>/dev/null; then
    echo -e "${GREEN}✓ V3 backend started successfully${NC}"
    echo ""
    echo -e "${GREEN}============================================${NC}"
    echo -e "${GREEN}     TRADER V3 RUNNING SUCCESSFULLY${NC}"
    echo -e "${GREEN}============================================${NC}"
    echo -e "${BLUE}Backend API: ${GREEN}http://localhost:$BACKEND_PORT${NC}"
    echo -e "${BLUE}V3 Console: ${GREEN}http://localhost:$FRONTEND_PORT/v3-trader${NC}"
    echo -e "${BLUE}Health Check: ${GREEN}http://localhost:$BACKEND_PORT/v3/health${NC}"
    echo -e "${GREEN}============================================${NC}"
    echo ""
    echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
    echo ""
else
    echo -e "${RED}❌ Backend failed to start${NC}"
    exit 1
fi

# Wait for the backend process
wait $BACKEND_PID