#!/bin/bash
# Launch Admiral Market Maker (paper trading)
#
# Usage:
#   ./scripts/run-admiral.sh                    # Default: paper, event from .env
#   ./scripts/run-admiral.sh KXEVENT-ABC        # Specify event ticker
#   ./scripts/run-admiral.sh KXEVENT-A,KXEVENT-B  # Multiple events

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Load environment: backend/.env.paper (has inline key content),
# then root .env.paper (has V3_* config like V3_SUBACCOUNT).
# Later source wins, so root overrides backend where both define a var.
for ENV_FILE in "${PROJECT_DIR}/backend/.env.paper" "${PROJECT_DIR}/.env.paper"; do
    if [ -f "$ENV_FILE" ]; then
        set -a
        source "$ENV_FILE"
        set +a
    fi
done

# Override: enable MM, disable single_arb, disable hybrid data mode
export V3_MM_ENABLED=true
export V3_SINGLE_ARB_ENABLED=false
export V3_HYBRID_DATA_MODE=true
export V3_CLEANUP_ON_STARTUP=false

# Helper: resolve relative path and read key file
resolve_key() {
    local path="$1"
    if [[ "$path" == ./* ]]; then
        path="${PROJECT_DIR}/${path#./}"
    fi
    if [ -f "$path" ]; then
        cat "$path"
    else
        echo ""
    fi
}

# Convert demo key path to content if needed
if [ -z "${KALSHI_PRIVATE_KEY_CONTENT:-}" ] && [ -n "${KALSHI_PRIVATE_KEY_PATH:-}" ]; then
    KALSHI_PRIVATE_KEY_CONTENT=$(resolve_key "$KALSHI_PRIVATE_KEY_PATH")
    if [ -z "$KALSHI_PRIVATE_KEY_CONTENT" ]; then
        echo "ERROR: Demo key file not found: ${KALSHI_PRIVATE_KEY_PATH}"
        exit 1
    fi
    export KALSHI_PRIVATE_KEY_CONTENT
fi

# Convert prod key path to content if needed (for MM gateway)
if [ -z "${V3_PROD_PRIVATE_KEY_CONTENT:-}" ] && [ -n "${V3_PROD_PRIVATE_KEY_PATH:-}" ]; then
    V3_PROD_PRIVATE_KEY_CONTENT=$(resolve_key "$V3_PROD_PRIVATE_KEY_PATH")
    if [ -n "$V3_PROD_PRIVATE_KEY_CONTENT" ]; then
        export V3_PROD_PRIVATE_KEY_CONTENT
        echo "Production API credentials loaded (MM gateway will use production)"
    fi
fi

# Event tickers from CLI arg or env
if [ -n "${1:-}" ]; then
    export V3_MM_EVENT_TICKERS="$1"
fi

if [ -z "${V3_MM_EVENT_TICKERS:-}" ]; then
    echo "ERROR: No event tickers. Pass as argument or set V3_MM_EVENT_TICKERS in .env"
    echo "Usage: ./scripts/run-admiral.sh KXEVENT-ABC"
    exit 1
fi

echo "=== Admiral Market Maker ==="
echo "Events: ${V3_MM_EVENT_TICKERS}"
echo "Spread: ${V3_MM_BASE_SPREAD:-4}c"
echo "Size: ${V3_MM_QUOTE_SIZE:-10} contracts"
echo "Max position: ${V3_MM_MAX_POSITION:-100}"
echo "Port: ${V3_PORT:-8005}"
echo "========================="

cd "$PROJECT_DIR/backend"
exec uv run uvicorn kalshiflow_rl.traderv3.app:app \
    --host "${V3_HOST:-0.0.0.0}" \
    --port "${V3_PORT:-8005}"
