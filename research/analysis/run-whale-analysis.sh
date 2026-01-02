#!/bin/bash
# Run Whale Following Strategy Analysis Pipeline
#
# This script runs the complete analysis pipeline:
# 1. Fetch market outcomes (if not already done)
# 2. Analyze trade outcomes
# 3. Detect informed trading patterns
# 4. Backtest whale following strategies
#
# Usage:
#   ./scripts/run-whale-analysis.sh              # Full pipeline
#   ./scripts/run-whale-analysis.sh --skip-fetch # Skip API fetch (use existing outcomes)
#   ./scripts/run-whale-analysis.sh --quick      # Quick analysis (limit 1000 tickers)

set -e

cd "$(dirname "$0")/.."
echo "Working directory: $(pwd)"

SKIP_FETCH=false
QUICK_MODE=false
RATE_LIMIT=5

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-fetch)
            SKIP_FETCH=true
            shift
            ;;
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --rate-limit)
            RATE_LIMIT=$2
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--skip-fetch] [--quick] [--rate-limit N]"
            exit 1
            ;;
    esac
done

# Set paths
TRADES_FILE="training/reports/historical_trades_full.csv"
OUTCOMES_FILE="training/reports/market_outcomes.csv"
ENRICHED_FILE="training/reports/enriched_trades.csv"
SCRIPTS_DIR="src/kalshiflow_rl/scripts"

echo ""
echo "============================================"
echo "WHALE FOLLOWING STRATEGY ANALYSIS PIPELINE"
echo "============================================"
echo ""

# Check if trades file exists
if [ ! -f "$TRADES_FILE" ]; then
    echo "ERROR: Trades file not found: $TRADES_FILE"
    echo "Please run: uv run python src/kalshiflow_rl/scripts/analyze_historical_trades.py --export $TRADES_FILE"
    exit 1
fi

TRADE_COUNT=$(wc -l < "$TRADES_FILE" | tr -d ' ')
echo "Found trades file: $TRADES_FILE ($TRADE_COUNT lines)"

# Step 1: Fetch market outcomes
if [ "$SKIP_FETCH" = false ]; then
    echo ""
    echo "--------------------------------------------"
    echo "Step 1: Fetching Market Outcomes from Kalshi"
    echo "--------------------------------------------"

    FETCH_ARGS="--from-csv $TRADES_FILE --output $OUTCOMES_FILE --rate-limit $RATE_LIMIT --batch-save 500"

    if [ "$QUICK_MODE" = true ]; then
        FETCH_ARGS="$FETCH_ARGS --limit 1000"
        echo "Quick mode: Limiting to 1000 tickers"
    fi

    # Check if we should resume
    if [ -f "$OUTCOMES_FILE" ]; then
        EXISTING_COUNT=$(wc -l < "$OUTCOMES_FILE" | tr -d ' ')
        echo "Found existing outcomes file with $EXISTING_COUNT entries"
        read -p "Resume from existing data? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            FETCH_ARGS="$FETCH_ARGS --resume"
        fi
    fi

    echo "Running: uv run python $SCRIPTS_DIR/fetch_market_outcomes.py $FETCH_ARGS"
    ENVIRONMENT=local uv run python $SCRIPTS_DIR/fetch_market_outcomes.py $FETCH_ARGS
else
    echo ""
    echo "Step 1: Skipping market outcomes fetch (--skip-fetch)"

    if [ ! -f "$OUTCOMES_FILE" ]; then
        echo "ERROR: Outcomes file not found: $OUTCOMES_FILE"
        echo "Remove --skip-fetch to fetch outcomes"
        exit 1
    fi
fi

OUTCOME_COUNT=$(wc -l < "$OUTCOMES_FILE" | tr -d ' ')
echo "Market outcomes available: $OUTCOME_COUNT"

# Step 2: Analyze trade outcomes
echo ""
echo "--------------------------------------------"
echo "Step 2: Analyzing Trade Outcomes"
echo "--------------------------------------------"

echo "Running: uv run python $SCRIPTS_DIR/analyze_trade_outcomes.py --trades $TRADES_FILE --outcomes $OUTCOMES_FILE --report --export $ENRICHED_FILE"
uv run python $SCRIPTS_DIR/analyze_trade_outcomes.py \
    --trades $TRADES_FILE \
    --outcomes $OUTCOMES_FILE \
    --report \
    --export $ENRICHED_FILE

echo "Report saved to: training/reports/trade_outcome_analysis.txt"

# Step 3: Detect informed trading
echo ""
echo "--------------------------------------------"
echo "Step 3: Detecting Informed Trading Patterns"
echo "--------------------------------------------"

echo "Running: uv run python $SCRIPTS_DIR/detect_informed_trading.py --enriched $ENRICHED_FILE --report"
uv run python $SCRIPTS_DIR/detect_informed_trading.py \
    --enriched $ENRICHED_FILE \
    --report \
    --json training/reports/informed_trading_results.json

echo "Report saved to: training/reports/informed_trading_analysis.txt"

# Step 4: Backtest strategies
echo ""
echo "--------------------------------------------"
echo "Step 4: Backtesting Whale Following Strategies"
echo "--------------------------------------------"

echo "Running: uv run python $SCRIPTS_DIR/backtest_whale_following.py --enriched $ENRICHED_FILE --report --position-size 100"
uv run python $SCRIPTS_DIR/backtest_whale_following.py \
    --enriched $ENRICHED_FILE \
    --report \
    --position-size 100 \
    --json training/reports/backtest_results.json

echo "Report saved to: training/reports/backtest_report.txt"

# Summary
echo ""
echo "============================================"
echo "ANALYSIS COMPLETE"
echo "============================================"
echo ""
echo "Generated Files:"
echo "  - training/reports/market_outcomes.csv"
echo "  - training/reports/enriched_trades.csv"
echo "  - training/reports/trade_outcome_analysis.txt"
echo "  - training/reports/informed_trading_analysis.txt"
echo "  - training/reports/backtest_report.txt"
echo "  - training/reports/informed_trading_results.json"
echo "  - training/reports/backtest_results.json"
echo ""
echo "Documentation:"
echo "  - src/kalshiflow_rl/rl-assessment/whale-following-analysis.md"
echo ""
