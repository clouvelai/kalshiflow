#!/bin/bash

# Run Kalshi RL Actor/Trader Backend Service
# 
# This script provides a consistent way to run the RL trading actor locally
# for end-to-end testing with configurable parameters.
#
# Default configuration:
# - Paper trading mode (demo-api.kalshi.co)
# - Discovery mode with 100 markets
# - Actor service enabled
# - Port 8002

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
DEFAULT_PORT=8002
DEFAULT_MARKET_LIMIT=100
DEFAULT_MODE="discovery"
DEFAULT_ENV="paper"

# Parse command line arguments
PORT=$DEFAULT_PORT
MARKET_LIMIT=$DEFAULT_MARKET_LIMIT
MODE=$DEFAULT_MODE
ENVIRONMENT=$DEFAULT_ENV
HELP=false
ACTOR_ENABLED=true

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Run Kalshi RL Actor/Trader backend service for local testing"
    echo ""
    echo "OPTIONS:"
    echo "  -p, --port PORT               Port to run on (default: $DEFAULT_PORT)"
    echo "  -m, --markets LIMIT           Orderbook market limit (default: $DEFAULT_MARKET_LIMIT)"
    echo "  -e, --env ENVIRONMENT         Environment: paper|production (default: $DEFAULT_ENV)"
    echo "  --mode MODE                   Market mode: discovery|config (default: $DEFAULT_MODE)"
    echo "  --no-actor                    Disable actor service (data collection only)"
    echo "  -h, --help                    Show this help message"
    echo ""
    echo "EXAMPLES:"
    echo "  $0                            # Run with defaults (paper, 100 markets, port 8002)"
    echo "  $0 -p 8003 -m 500            # Run on port 8003 with 500 markets"
    echo "  $0 -e production --no-actor   # Production mode, no trading, data collection only"
    echo "  $0 -m 50                      # Paper mode with just 50 markets for testing"
    echo ""
    echo "FRONTEND ACCESS:"
    echo "  Once running, access the RL Trader Dashboard at:"
    echo "  http://localhost:5173/rl-trader"
    echo ""
    echo "API ENDPOINTS:"
    echo "  Health:    http://localhost:PORT/rl/health"
    echo "  Status:    http://localhost:PORT/rl/status"
    echo "  WebSocket: ws://localhost:PORT/rl/ws"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -m|--markets)
            MARKET_LIMIT="$2"
            shift 2
            ;;
        -e|--env)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --no-actor)
            ACTOR_ENABLED=false
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            exit 1
            ;;
    esac
done

# Validate arguments
if ! [[ "$PORT" =~ ^[0-9]+$ ]] || [ "$PORT" -lt 1024 ] || [ "$PORT" -gt 65535 ]; then
    echo -e "${RED}Error: Port must be a number between 1024-65535${NC}"
    exit 1
fi

if ! [[ "$MARKET_LIMIT" =~ ^[0-9]+$ ]] || [ "$MARKET_LIMIT" -lt 1 ]; then
    echo -e "${RED}Error: Market limit must be a positive number${NC}"
    exit 1
fi

if [[ "$ENVIRONMENT" != "paper" && "$ENVIRONMENT" != "production" ]]; then
    echo -e "${RED}Error: Environment must be 'paper' or 'production'${NC}"
    exit 1
fi

if [[ "$MODE" != "discovery" && "$MODE" != "config" ]]; then
    echo -e "${RED}Error: Mode must be 'discovery' or 'config'${NC}"
    exit 1
fi

# Header
echo -e "${BLUE}=========================================="
echo -e "🤖 Kalshi RL Actor/Trader Service"
echo -e "==========================================${NC}"

# Show configuration
echo -e "${GREEN}Configuration:${NC}"
echo "  Environment:      $ENVIRONMENT"
echo "  Market Mode:      $MODE"
echo "  Market Limit:     $MARKET_LIMIT"
echo "  Port:             $PORT"
echo "  Actor Enabled:    $ACTOR_ENABLED"
echo ""

# Warn about production
if [[ "$ENVIRONMENT" == "production" ]]; then
    echo -e "${YELLOW}⚠️  WARNING: Running in PRODUCTION mode!${NC}"
    echo -e "${YELLOW}   This will use real Kalshi API and potentially real money.${NC}"
    echo ""
    read -p "Are you sure you want to continue? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        exit 1
    fi
fi

# Check if port is available
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${RED}Error: Port $PORT is already in use.${NC}"
    echo "Please stop the service using it or choose a different port with -p"
    exit 1
fi

# Navigate to backend directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/../backend"

if [ ! -d "$BACKEND_DIR" ]; then
    echo -e "${RED}Error: Backend directory not found at $BACKEND_DIR${NC}"
    exit 1
fi

cd "$BACKEND_DIR"

# Set up environment variables
echo -e "${GREEN}Setting up environment...${NC}"

export ENVIRONMENT="$ENVIRONMENT"
export RL_MODE="$MODE"
export RL_ORDERBOOK_MARKET_LIMIT="$MARKET_LIMIT"
export RL_ACTOR_ENABLED="$ACTOR_ENABLED"

# Additional settings for local testing
export RL_LOG_LEVEL="INFO"
if [[ "$MODE" == "config" ]]; then
    # Use a small set of test markets for config mode
    export RL_MARKET_TICKERS="INXD-25JAN03,KXFEDCHAIRNOM-29-CWAL"
fi

echo -e "${GREEN}Environment variables set:${NC}"
echo "  ENVIRONMENT=$ENVIRONMENT"
echo "  RL_MODE=$RL_MODE" 
echo "  RL_ORDERBOOK_MARKET_LIMIT=$RL_ORDERBOOK_MARKET_LIMIT"
echo "  RL_ACTOR_ENABLED=$RL_ACTOR_ENABLED"
echo ""

# Start the service
echo -e "${GREEN}Starting RL Backend Service...${NC}"
echo ""

# Show endpoints that will be available
echo -e "${BLUE}Endpoints:${NC}"
echo "  📊 Health:    http://localhost:$PORT/rl/health"
echo "  📈 Status:    http://localhost:$PORT/rl/status"
echo "  🔌 WebSocket: ws://localhost:$PORT/rl/ws"
echo ""
echo -e "${BLUE}Frontend Dashboard:${NC}"
echo "  🌐 RL Trader: http://localhost:5173/rl-trader"
echo "     (requires frontend: cd frontend && npm run dev)"
echo ""

echo -e "${YELLOW}Press Ctrl+C to stop the service${NC}"
echo "----------------------------------------"

# Run the service
exec uv run uvicorn kalshiflow_rl.app:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --log-level info \
    --reload