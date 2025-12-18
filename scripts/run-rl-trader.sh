#!/bin/bash

# Run Kalshi RL Trader Service
# 
# This script provides RL trading with actor service for paper trading.
# Focuses on safe trading decisions using demo account.
#
# Default configuration:
# - Paper trading mode (demo-api.kalshi.co)
# - Actor service enabled
# - Port 8003
# - Strategy selection (hardcoded HOLD or RL model)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
DEFAULT_PORT=8003
DEFAULT_MARKET_LIMIT=100
DEFAULT_MODE="discovery"
DEFAULT_ENV="paper"
DEFAULT_STRATEGY="hardcoded"
DEFAULT_CLEANUP="true"

# Parse command line arguments
PORT=$DEFAULT_PORT
MARKET_LIMIT=$DEFAULT_MARKET_LIMIT
MODE=$DEFAULT_MODE
ENVIRONMENT=$DEFAULT_ENV
STRATEGY=$DEFAULT_STRATEGY
CLEANUP=$DEFAULT_CLEANUP
HELP=false

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Run Kalshi RL Trader service for paper trading"
    echo ""
    echo "OPTIONS:"
    echo "  -p, --port PORT               Port to run on (default: $DEFAULT_PORT)"
    echo "  -m, --markets LIMIT           Orderbook market limit (default: $DEFAULT_MARKET_LIMIT)"
    echo "  -e, --env ENVIRONMENT         Environment: paper|local|production (default: $DEFAULT_ENV)"
    echo "  --mode MODE                   Market mode: discovery|config (default: $DEFAULT_MODE)"
    echo "  -s, --strategy STRATEGY       Action strategy: hardcoded|rl_model|quant_hardcoded|position_aware_quant_hardcoded (default: $DEFAULT_STRATEGY)"
    echo "  --no-cleanup                  Disable order/position cleanup on startup (default: cleanup enabled)"
    echo "  -h, --help                    Show this help message"
    echo ""
    echo "EXAMPLES:"
    echo "  $0                            # Run with defaults (hardcoded HOLD strategy, cleanup enabled)"
    echo "  $0 -s rl_model               # Use trained RL model for trading decisions"
    echo "  $0 -s rl_model -m 25         # RL model with 25 markets for testing"
    echo "  $0 --no-cleanup              # Skip order/position cleanup on startup"
    echo "  $0 -p 8004 -m 50             # Custom port and market limit"
    echo ""
    echo "PURPOSE:"
    echo "  - Paper trading with RL actor service"
    echo "  - Safe trading decisions using demo account"
    echo "  - Real-time trading strategy testing"
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
        -s|--strategy)
            STRATEGY="$2"
            shift 2
            ;;
        --no-cleanup)
            CLEANUP="false"
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

if [[ "$ENVIRONMENT" != "paper" && "$ENVIRONMENT" != "local" && "$ENVIRONMENT" != "production" ]]; then
    echo -e "${RED}Error: Environment must be 'paper', 'local', or 'production'${NC}"
    exit 1
fi

if [[ "$MODE" != "discovery" && "$MODE" != "config" ]]; then
    echo -e "${RED}Error: Mode must be 'discovery' or 'config'${NC}"
    exit 1
fi

# Validate strategy (allow position_aware variants)
VALID_STRATEGIES=("hardcoded" "rl_model" "quant_hardcoded" "position_aware" "position_aware_quant_hardcoded" "position_aware_hardcoded")
if [[ ! " ${VALID_STRATEGIES[@]} " =~ " ${STRATEGY} " ]]; then
    echo -e "${RED}Error: Strategy must be one of: ${VALID_STRATEGIES[*]}${NC}"
    exit 1
fi

# Function to get current model path from CURRENT_MODEL.json
get_current_model_path() {
    local script_dir="$1"
    local current_model_json="$script_dir/../backend/src/kalshiflow_rl/BEST_MODEL/CURRENT_MODEL.json"
    
    if [ ! -f "$current_model_json" ]; then
        echo -e "${RED}Error: CURRENT_MODEL.json not found at $current_model_json${NC}"
        exit 1
    fi
    
    # Extract model path using jq if available, otherwise use basic sed/grep
    if command -v jq >/dev/null 2>&1; then
        local model_path=$(jq -r '.current_model.full_path' "$current_model_json" 2>/dev/null)
    else
        # Fallback: use grep and sed to extract the model path
        local model_path=$(grep '"full_path"' "$current_model_json" | sed 's/.*"full_path": "\([^"]*\)".*/\1/')
    fi
    
    if [ -z "$model_path" ] || [ "$model_path" = "null" ]; then
        echo -e "${RED}Error: Could not extract model path from $current_model_json${NC}"
        exit 1
    fi
    
    echo "$model_path"
}

# Header
echo -e "${BLUE}=========================================="
echo -e "🤖 Kalshi RL Trader Service"
echo -e "==========================================${NC}"

# Resolve model path if using rl_model strategy
if [[ "$STRATEGY" == "rl_model" ]]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    MODEL_PATH=$(get_current_model_path "$SCRIPT_DIR")
    echo -e "${GREEN}🧠 RL Model Strategy Selected${NC}"
    echo -e "  Model: $(basename "$MODEL_PATH")"
    echo -e "  Path:  $MODEL_PATH"
    echo ""
fi

# Show configuration
echo -e "${GREEN}Configuration:${NC}"
echo "  Environment:      $ENVIRONMENT"
echo "  Market Mode:      $MODE"
echo "  Market Limit:     $MARKET_LIMIT"
echo "  Port:             $PORT"
echo "  Strategy:         $STRATEGY"
echo "  Actor Enabled:    true (trading enabled)"
echo "  Cleanup on Start: $CLEANUP"
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
elif [[ "$ENVIRONMENT" == "local" ]]; then
    echo -e "${YELLOW}⚠️  NOTE: Using LOCAL environment with real Kalshi API.${NC}"
    echo -e "${YELLOW}   This connects to production API but should be for data only.${NC}"
    echo ""
fi

# Paper trading safety message
if [[ "$ENVIRONMENT" == "paper" ]]; then
    echo -e "${GREEN}✅ SAFE: Using paper trading environment (demo account)${NC}"
    echo ""
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
export RL_ACTOR_ENABLED="true"  # ENABLED - trading enabled
export RL_CLEANUP_ON_START="$CLEANUP"

# Set up actor strategy and model path
# Pass strategy directly to allow position_aware variants
if [[ "$STRATEGY" == "rl_model" ]]; then
    export RL_ACTOR_STRATEGY="rl_model"
    export RL_ACTOR_MODEL_PATH="$MODEL_PATH"
else
    export RL_ACTOR_STRATEGY="$STRATEGY"
fi

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
echo "  RL_ACTOR_STRATEGY=$RL_ACTOR_STRATEGY"
echo "  RL_CLEANUP_ON_START=$RL_CLEANUP_ON_START"
if [[ "$STRATEGY" == "rl_model" ]]; then
    echo "  RL_ACTOR_MODEL_PATH=$RL_ACTOR_MODEL_PATH"
fi
echo ""

# Start the service
echo -e "${GREEN}Starting RL Trader Service...${NC}"
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

if [[ "$STRATEGY" == "rl_model" ]]; then
    echo -e "${YELLOW}Purpose: AI-powered trading using trained RL model${NC}"
else
    echo -e "${YELLOW}Purpose: Safe HOLD strategy trading for testing${NC}"
fi
echo ""

echo -e "${YELLOW}Press Ctrl+C to stop the service${NC}"
echo "----------------------------------------"

# Run the service
exec uv run uvicorn kalshiflow_rl.app:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --log-level info \
    --reload