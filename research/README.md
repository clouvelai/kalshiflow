# Kalshi Flow Research

Quantitative research and strategy development for the Kalshi Flow trading system.

## Directory Structure

```
research/
├── analysis/                    # Reusable analysis scripts
│   ├── public_trade_feed_analysis.py   # Main strategy analysis tool
│   ├── exhaustive_strategy_search.py   # Exhaustive strategy validation
│   └── ...                             # Other analysis scripts
│
├── data/                        # Historical data (gitignored)
│   ├── trades/                  # Trade history CSVs
│   │   ├── historical_trades_ALL.csv       # ~1.7M trades
│   │   └── enriched_trades_resolved_ALL.csv # Trades with outcomes
│   │
│   └── markets/                 # Market metadata
│       ├── settled_markets_ALL.json        # ~78k settled markets (4.5GB)
│       ├── market_outcomes_ALL.csv         # Settlement outcomes
│       └── settled_markets_NEEDED.json     # Subset for active analysis
│
├── reports/                     # Analysis output (JSON, TXT)
│   ├── all_trades_patterns.json
│   ├── exhaustive_search_results.json
│   └── ...
│
├── strategies/                  # Strategy documentation
│   ├── MVP_STRATEGY_IDEAS.md    # Main strategy analysis document
│   │
│   ├── validated/               # Proven strategies with edge
│   │   ├── MVP_BEST_STRATEGY.md          # YES at 80-90c (+5.1% edge)
│   │   ├── YES_80_90_STATUS.md           # Implementation status
│   │   └── FINAL_EVIDENCE_BASED_STRATEGY.md
│   │
│   ├── experimental/            # Strategies under testing
│   │   ├── ALL_TRADES_STRATEGY.md
│   │   ├── ADDITIONAL_STRATEGY_ANALYSIS.md
│   │   └── ...
│   │
│   └── rejected/                # Strategies that didn't work
│       └── whale-following-analysis.md   # Whale following has issues
│
└── README.md                    # This file
```

## Key Findings

### Validated Strategies (MVP-Ready)

| Strategy | Markets | Edge | Total Profit | Status |
|----------|---------|------|--------------|--------|
| **YES at 80-90c** | 2,110 | **+5.1%** | $1.6M | ✅ Implemented |
| NO at 80-90c | 2,808 | +3.3% | $708k | Consider adding |
| NO at 90-100c | 4,741 | +1.2% | $463k | Lower priority |

### Rejected Strategies

| Strategy | Reason |
|----------|--------|
| Whale-following (30-70c) | >30% concentration in single markets |
| Whale consensus following | Only 27.6% win rate when 100% agree |
| Time-of-day patterns | Insufficient statistical validation |

## Running Analysis

All analysis scripts are in `research/analysis/`. Run from the backend directory:

```bash
cd backend

# Main strategy analysis
uv run python ../research/analysis/public_trade_feed_analysis.py

# Exhaustive strategy search
uv run python ../research/analysis/exhaustive_strategy_search.py

# Fetch market outcomes for new markets
uv run python ../research/analysis/fetch_market_outcomes.py
```

## Data Management

Large data files are gitignored. To refresh data:

```bash
# Fetch all settled markets (warning: ~4.5GB)
uv run python ../research/analysis/fetch_full_settled_markets.py

# Enrich trades with outcomes
uv run python ../research/analysis/fetch_market_outcomes.py
```

## Adding New Analysis

1. Create script in `research/analysis/`
2. Output results to `research/reports/`
3. Document findings in `research/strategies/experimental/`
4. If validated, move to `research/strategies/validated/`
