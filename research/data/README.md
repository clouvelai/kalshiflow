# Research Data

This directory contains historical trade and market data for strategy analysis.

**Note:** Large files are gitignored. Run fetch scripts to populate locally.

## Data Files

### trades/
| File | Description | Size |
|------|-------------|------|
| `historical_trades_ALL.csv` | All ~1.7M public trades | ~200MB |
| `enriched_trades_ALL.csv` | Trades with market metadata | ~226MB |
| `enriched_trades_resolved_ALL.csv` | Trades with settlement outcomes | ~217MB |

### markets/
| File | Description | Size |
|------|-------------|------|
| `settled_markets_ALL.json` | Full market data for ~78k settled markets | ~4.5GB |
| `settled_markets_NEEDED.json` | Subset for current analysis | ~8MB |
| `market_outcomes_ALL.csv` | Settlement outcomes (ticker, result) | ~37MB |

## Refreshing Data

```bash
cd backend

# Fetch market outcomes
uv run python ../research/analysis/fetch_market_outcomes.py

# Fetch full settled markets (slow, 4.5GB)
uv run python ../research/analysis/fetch_full_settled_markets.py
```

## Data Dictionary

### Trade Fields
- `trade_id`: Unique trade identifier
- `ticker`: Market ticker
- `count`: Number of contracts
- `yes_price`: Price in cents (1-99)
- `taker_side`: "yes" or "no"
- `created_time`: Trade timestamp

### Market Outcome Fields
- `ticker`: Market ticker
- `result`: "yes", "no", or "void"
- `title`: Market title
- `category`: Market category
