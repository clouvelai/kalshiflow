#!/bin/bash

# Reset Captain - Clear tracked markets and restart
# Use this when changing discovery filters or to start fresh

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/../backend"

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}       CAPTAIN - RESET & RESTART${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Parse arguments (pass through to run-captain.sh)
ENVIRONMENT="${1:-paper}"
MODE="${2:-lifecycle}"
MARKET_LIMIT="${3:-1000}"

# Load environment
ENV_FILE="$BACKEND_DIR/.env.$ENVIRONMENT"
if [ -f "$ENV_FILE" ]; then
    set -a
    source "$ENV_FILE"
    set +a
fi

# Stop any running Captain
echo -e "${YELLOW}Stopping any running Captain...${NC}"
pkill -f "uvicorn.*8005" 2>/dev/null || true
pkill -f "kalshiflow_rl.traderv3" 2>/dev/null || true
sleep 2

# Clear tracked markets from database
echo -e "${YELLOW}Clearing tracked markets from database...${NC}"
cd "$BACKEND_DIR"

uv run python -c "
import asyncio
import sys
sys.path.insert(0, 'src')

async def clear():
    from kalshiflow_rl.data.database import rl_db
    await rl_db.initialize()

    # Count before
    count_before = await rl_db.count_tracked_markets(status=None)
    print(f'  Found {count_before} tracked markets')

    if count_before > 0:
        deleted = await rl_db.clear_tracked_markets()
        print(f'  Cleared {deleted} tracked markets')
    else:
        print(f'  No tracked markets to clear')

    await rl_db.close()

asyncio.run(clear())
"

echo -e "${GREEN}✓ Database cleared${NC}"
echo ""

# Start Captain with fresh state
echo -e "${BLUE}Starting Captain with fresh discovery...${NC}"
echo ""

exec "$SCRIPT_DIR/run-captain.sh" "$ENVIRONMENT" "$MODE" "$MARKET_LIMIT"
