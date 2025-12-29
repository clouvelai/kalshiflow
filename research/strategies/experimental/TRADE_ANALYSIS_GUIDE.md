# Trade Analysis Guide

**Purpose**: Step-by-step instructions for analyzing Kalshi trade data to discover profitable trading strategies.

**Last Updated**: 2025-12-28
**Author**: Quant Agent

---

## Overview

This guide documents the complete workflow for:
1. Extracting historical trade data from production
2. Fetching market outcomes from Kalshi API
3. Joining trades with outcomes to calculate actual P/L
4. Running pattern analysis to discover profitable strategies
5. Creating sport/category-specific deep dives

---

## Prerequisites

### Environment Setup

```bash
# Navigate to backend directory
cd /Users/samuelclark/Desktop/kalshiflow/backend

# Ensure dependencies are installed
uv sync

# Set environment to access production database (read-only)
export ENVIRONMENT=local
```

### Required Files

| File | Purpose |
|------|---------|
| `.env` or `.env.local` | Database credentials, Kalshi API keys |
| `KALSHI_API_KEY_ID` | API key for Kalshi REST API |
| `KALSHI_PRIVATE_KEY_CONTENT` | RSA private key for authentication |
| `DATABASE_URL` | PostgreSQL connection string |

---

## Step 1: Export Historical Trades from Production

**Script**: `src/kalshiflow_rl/scripts/analyze_historical_trades.py`

**Purpose**: Safely read trades from production database and export to CSV for analysis.

### Commands

```bash
# View database statistics
uv run python src/kalshiflow_rl/scripts/analyze_historical_trades.py --stats

# Export ALL trades to CSV (recommended for full analysis)
uv run python src/kalshiflow_rl/scripts/analyze_historical_trades.py --export training/reports/historical_trades_full.csv

# Export with date range filter
uv run python src/kalshiflow_rl/scripts/analyze_historical_trades.py --export trades.csv --start-date 2025-12-01 --end-date 2025-12-28

# Find top trades by cost
uv run python src/kalshiflow_rl/scripts/analyze_historical_trades.py --top-trades 100

# Find high leverage outliers
uv run python src/kalshiflow_rl/scripts/analyze_historical_trades.py --leverage-outliers
```

### Output Fields

| Field | Description |
|-------|-------------|
| `market_ticker` | Market identifier (e.g., KXNFLGAME-XXXX) |
| `yes_price` | YES side price in cents (0-100) |
| `no_price` | NO side price in cents (0-100) |
| `count` | Number of contracts |
| `taker_side` | Side taken: 'yes' or 'no' |
| `ts` | Timestamp of trade |
| `cost_dollars` | Computed: count * price / 100 |
| `potential_profit` | Computed: count * (100 - price) / 100 |
| `leverage_ratio` | Computed: potential_profit / cost_dollars |

### Safety Features

- Read-only database access
- Minimal connection pool (prevents load on production)
- 30-second timeout
- No write operations

---

## Step 2: Fetch Market Outcomes from Kalshi API

**Script**: `src/kalshiflow_rl/scripts/fetch_market_outcomes.py`

**Purpose**: Get settlement data (result: yes/no) for resolved markets.

### Commands

```bash
# Fetch outcomes for all unique tickers in trades CSV
uv run python src/kalshiflow_rl/scripts/fetch_market_outcomes.py \
    --trades training/reports/historical_trades_full.csv \
    --output training/reports/market_outcomes.csv

# Resume interrupted fetch (uses existing output file)
uv run python src/kalshiflow_rl/scripts/fetch_market_outcomes.py \
    --trades training/reports/historical_trades_full.csv \
    --output training/reports/market_outcomes.csv \
    --resume
```

### Rate Limiting

- 5 requests per second (Kalshi API limit)
- Automatic retry with exponential backoff
- Resume capability for interrupted fetches

### Output Fields

| Field | Description |
|-------|-------------|
| `ticker` | Market ticker |
| `result` | Settlement result: 'yes', 'no', or null |
| `status` | Market status: 'settled', 'open', etc. |
| `title` | Human-readable market title |
| `close_time` | When market closed |

---

## Step 3: Join Trades with Outcomes

**Script**: `src/kalshiflow_rl/scripts/analyze_trade_outcomes.py`

**Purpose**: Merge trades with settlement outcomes to calculate actual profit/loss.

### Commands

```bash
# Create enriched trades file with P/L calculations
uv run python src/kalshiflow_rl/scripts/analyze_trade_outcomes.py \
    --trades training/reports/historical_trades_full.csv \
    --outcomes training/reports/market_outcomes.csv \
    --output training/reports/enriched_trades_final.csv
```

### P/L Calculation Logic

```python
# Determine if trade won
if taker_side == market_result:
    is_winner = True
    actual_profit = max_payout - cost  # (count * $1) - (count * price/100)
else:
    is_winner = False
    actual_profit = -cost  # Lost entire cost basis
```

### Output Fields (Added)

| Field | Description |
|-------|-------------|
| `market_result` | Joined from outcomes: 'yes' or 'no' |
| `is_winner` | Boolean: did this trade win? |
| `actual_profit_dollars` | Realized P/L in dollars |
| `trade_price` | Price paid (yes_price or 100-no_price) |
| `breakeven_rate` | Price/100 (win rate needed to break even) |

---

## Step 4: Run Pattern Analysis

### 4a. Whale Following Backtest

**Script**: `src/kalshiflow_rl/scripts/backtest_whale_following.py`

```bash
uv run python src/kalshiflow_rl/scripts/backtest_whale_following.py \
    --input training/reports/enriched_trades_final.csv \
    --output training/reports/backtest_report.txt
```

**Pre-built Strategies Tested**:
- Whale moderate (>=100 contracts @ 30-70c)
- Whale longshot (>=100 contracts @ <=15c)
- Whale favorite (>=100 contracts @ >=85c)
- Mega whale (>=500 contracts)
- And more...

### 4b. Advanced Pattern Analysis

**Script**: `src/kalshiflow_rl/scripts/advanced_pattern_analysis.py`

```bash
uv run python src/kalshiflow_rl/scripts/advanced_pattern_analysis.py \
    --input training/reports/enriched_trades_final.csv \
    --output training/reports/advanced_pattern_analysis.txt
```

**Patterns Analyzed**:
- Timing patterns (time before market close)
- Contrarian patterns (betting against consensus)
- Category patterns (KXNFL, KXNBA, KXEP, etc.)
- Price-side interaction (YES vs NO at different prices)

### 4c. All-Trades Pattern Analysis

**Script**: `src/kalshiflow_rl/scripts/analyze_all_trades_patterns.py`

```bash
uv run python src/kalshiflow_rl/scripts/analyze_all_trades_patterns.py \
    --input training/reports/enriched_trades_final.csv \
    --output training/reports/all_trades_patterns.json
```

**Patterns Analyzed**:
- Size thresholds (5 to 1000+ contracts)
- Time of day / day of week
- Category + side combinations
- Volume clustering

### 4d. Unconventional Pattern Analysis

**Script**: `src/kalshiflow_rl/scripts/additional_pattern_analysis.py`

```bash
uv run python src/kalshiflow_rl/scripts/additional_pattern_analysis.py \
    --input training/reports/enriched_trades_final.csv \
    --output training/reports/additional_patterns.json
```

**Patterns Analyzed**:
- Minute-of-hour timing
- Consecutive trade patterns
- Psychological price levels (25c, 50c, 75c)
- Whale momentum (follow vs fade)
- Price movement sequences

---

## Step 5: Sport/Category Deep Dives

### NBA Deep Dive Example

**Script**: `src/kalshiflow_rl/scripts/nba_deep_dive_analysis.py`

```bash
uv run python src/kalshiflow_rl/scripts/nba_deep_dive_analysis.py \
    --input training/reports/enriched_trades_final.csv
```

**Analysis Includes**:
- YES vs NO side performance by price range
- Team-level alpha (which teams to bet on/against)
- Whale behavior in NBA specifically
- Leverage bucket analysis
- Comparison to NFL patterns

### Creating a New Sport/Category Deep Dive

To analyze a new category (e.g., soccer, crypto), follow this template:

```python
# 1. Filter to category
df_category = df[df['market_ticker'].str.startswith('KXSOCCER')]

# 2. Analyze YES vs NO at different price ranges
for side in ['yes', 'no']:
    for low, high in [(15, 25), (25, 35), (35, 45), (45, 55)]:
        subset = df_category[
            (df_category['taker_side'] == side) &
            (df_category['trade_price'] >= low) &
            (df_category['trade_price'] <= high)
        ]
        roi = subset['actual_profit_dollars'].sum() / subset['cost_dollars'].sum() * 100
        print(f"{side} @ {low}-{high}c: {roi:.1f}% ROI")

# 3. Compare to baseline (NFL/NBA patterns)
# 4. Look for category-specific edges
```

---

## Key Metrics to Calculate

### For Any Strategy

| Metric | Formula | What It Means |
|--------|---------|---------------|
| **Win Rate** | wins / total_trades | % of trades that won |
| **Breakeven Rate** | avg_price / 100 | Win rate needed to break even |
| **Edge** | win_rate - breakeven_rate | Advantage over random |
| **ROI** | total_profit / total_cost * 100 | Return on investment |
| **Profit Factor** | gross_wins / gross_losses | How much you win per dollar lost |

### Statistical Significance

- **Minimum sample size**: 50 trades (100+ preferred)
- **Edge threshold**: > 2% for significance
- **Confidence**: Verify pattern holds across multiple sub-segments

---

## Data File Locations

### Input Data

| File | Location | Description |
|------|----------|-------------|
| Raw trades | `training/reports/historical_trades_full.csv` | Direct DB export |
| Market outcomes | `training/reports/market_outcomes.csv` | Kalshi API settlements |
| Enriched trades | `training/reports/enriched_trades_final.csv` | Trades + outcomes + P/L |

### Analysis Outputs

| File | Location | Description |
|------|----------|-------------|
| Backtest report | `training/reports/backtest_report.txt` | Strategy backtest results |
| Pattern analysis | `training/reports/advanced_pattern_analysis.txt` | Pattern findings |
| All trades JSON | `training/reports/all_trades_patterns.json` | Machine-readable patterns |
| Additional patterns | `training/reports/additional_patterns.json` | Unconventional patterns |

### Strategy Documents

| File | Location | Description |
|------|----------|-------------|
| Core strategy | `traderv3/planning/FINAL_EVIDENCE_BASED_STRATEGY.md` | Whale NO at 30-50c |
| All trades | `traderv3/planning/ALL_TRADES_STRATEGY.md` | Night, retail fade, etc. |
| Additional | `traderv3/planning/ADDITIONAL_STRATEGY_ANALYSIS.md` | Timing, momentum, NBA |

---

## Refresh Workflow

### When to Refresh Data

1. **Weekly**: Re-run analysis with new trades
2. **After major events**: New season, rule changes
3. **When strategies underperform**: Check if patterns still hold

### Full Refresh Commands

```bash
cd /Users/samuelclark/Desktop/kalshiflow/backend

# Step 1: Export fresh trades
uv run python src/kalshiflow_rl/scripts/analyze_historical_trades.py \
    --export training/reports/historical_trades_$(date +%Y%m%d).csv

# Step 2: Fetch new market outcomes
uv run python src/kalshiflow_rl/scripts/fetch_market_outcomes.py \
    --trades training/reports/historical_trades_$(date +%Y%m%d).csv \
    --output training/reports/market_outcomes_$(date +%Y%m%d).csv

# Step 3: Create enriched file
uv run python src/kalshiflow_rl/scripts/analyze_trade_outcomes.py \
    --trades training/reports/historical_trades_$(date +%Y%m%d).csv \
    --outcomes training/reports/market_outcomes_$(date +%Y%m%d).csv \
    --output training/reports/enriched_trades_$(date +%Y%m%d).csv

# Step 4: Run pattern analysis
uv run python src/kalshiflow_rl/scripts/advanced_pattern_analysis.py \
    --input training/reports/enriched_trades_$(date +%Y%m%d).csv \
    --output training/reports/patterns_$(date +%Y%m%d).txt

# Step 5: Sport-specific deep dives as needed
uv run python src/kalshiflow_rl/scripts/nba_deep_dive_analysis.py \
    --input training/reports/enriched_trades_$(date +%Y%m%d).csv
```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Database connection timeout | Check `DATABASE_URL` in `.env`, ensure VPN if needed |
| Kalshi API 401 | Verify `KALSHI_API_KEY_ID` and `KALSHI_PRIVATE_KEY_CONTENT` |
| Kalshi API 429 | Rate limited - wait or reduce request frequency |
| Empty outcomes file | Markets may not be settled yet - check status |
| Low sample size warning | Need more trades - expand date range |

### Validation Checks

```python
# Check enriched data quality
import pandas as pd

df = pd.read_csv('training/reports/enriched_trades_final.csv')

# Should have market_result for most rows
print(f"Resolution rate: {df['market_result'].notna().mean():.1%}")

# P/L should sum to reasonable number
print(f"Total P/L: ${df['actual_profit_dollars'].sum():,.0f}")

# Win rate should be between 30-70% for most strategies
print(f"Overall win rate: {df['is_winner'].mean():.1%}")
```

---

## Analysis Philosophy

### Evidence Over Theory

- **Never** assume a pattern works without data
- **Always** calculate actual P/L, not theoretical edge
- **Validate** across multiple time periods and sub-segments
- **Be skeptical** of small sample sizes (<100 trades)

### Key Lessons Learned

1. **Partial data is misleading**: 27k trades suggested "Whale YES at 50-70c". 90k trades proved the opposite.

2. **Sports have different dynamics**: NFL NO strategy inverts to NBA YES strategy.

3. **Whales aren't always right**: Whale longshots lose 72.7%. Size alone doesn't indicate edge.

4. **Time patterns exist**: Minute 38-42 dramatically outperforms minute 05-15.

5. **Side matters enormously**: NO side overall: +20.5% ROI. YES side: -6.5% ROI.

---

## Script Reference

| Script | Purpose | Key Flags |
|--------|---------|-----------|
| `analyze_historical_trades.py` | Export trades from DB | `--export`, `--stats`, `--top-trades` |
| `fetch_market_outcomes.py` | Get settlements from API | `--trades`, `--output`, `--resume` |
| `analyze_trade_outcomes.py` | Join trades + outcomes | `--trades`, `--outcomes`, `--output` |
| `backtest_whale_following.py` | Strategy backtesting | `--input`, `--output` |
| `advanced_pattern_analysis.py` | Timing/category patterns | `--input`, `--output` |
| `analyze_all_trades_patterns.py` | All-trades analysis | `--input`, `--output` |
| `additional_pattern_analysis.py` | Unconventional patterns | `--input`, `--output` |
| `nba_deep_dive_analysis.py` | NBA-specific analysis | `--input` |

---

## Next Steps for Future Analysis

1. **Expand to other sports**: Soccer (EPL, La Liga), UFC, Golf
2. **Test seasonal patterns**: Do strategies work differently in playoffs?
3. **Add real-time validation**: Compare backtest to live trading results
4. **Explore inter-market correlations**: When one market moves, do related markets follow?
5. **Build automated refresh pipeline**: Daily/weekly data refresh with alerts
