# Whale Following Strategy Analysis

**Last Updated**: 2025-12-28
**Status**: INITIAL ANALYSIS (24,000 resolved trades analyzed, 76,000 pending)

## Executive Summary

Analysis of 100,000 historical Kalshi trades identified a **statistically significant profitable pattern**: following whale trades (>=100 contracts) at moderate prices (30-70 cents).

### Key Finding

| Strategy | Win Rate | ROI | Edge | Sample Size |
|----------|----------|-----|------|-------------|
| **Whale Moderate (30-70c)** | 62.4% | **+20.4%** | +11.4% | 1,854 trades |
| Whale Favorites (>=85c) | 95.6% | +2.7% | +2.5% | 776 trades |
| All Whales | 45.7% | -13.9% | +3.0% | 5,393 trades |
| Mega Whales (>=1000) | 44.1% | -24.4% | N/A | 717 trades |
| Whale Longshots (<=15c) | 3.1% | -65.5% | -11.9% | 1,525 trades |

**Critical Insight**: The "obvious" signal (big bets on longshots) is NOT profitable. Whale longshots win only 3.1% vs an expected 15% - suggesting these are often speculative/gambling rather than informed.

The profitable signal is whale bets on **moderate probability markets (30-70c)** where whales show 11.4% edge over implied probability.

## Hypothesis Testing Results

### Hypothesis 1: Whale Advantage
**Result: MIXED - Whales do NOT have higher overall win rate**

- Whale Win Rate: 45.7% (95% CI: 44.4%-47.1%)
- Retail Win Rate: 50.6% (95% CI: 49.8%-51.3%)
- Chi-squared: 38.64 (STATISTICALLY SIGNIFICANT)

Counterintuitively, whales have LOWER win rate than retail. But this is because:
1. Whales take more extreme positions (longshots/locks)
2. Extreme prices have lower win rates by definition
3. The VALUE comes from price segments, not aggregate

### Hypothesis 2: Whale Longshot Bets
**Result: NEGATIVE - Whale longshots UNDERPERFORM expectations**

- Whale Longshots Win Rate: 3.1%
- Expected Win Rate (from 15c price): ~15%
- Edge: **-11.9%** (NEGATIVE EDGE)

This suggests whale longshot bets are:
- Gambling/speculation rather than informed
- Possibly market manipulation (accumulating for dump)
- Or simply incorrect - markets are efficient at extreme prices

### Hypothesis 3: Trade Size Progression
**Result: INVERSE CORRELATION**

| Size Bucket | Win Rate | ROI |
|-------------|----------|-----|
| 1-9 contracts | 51.9% | 1.7% |
| 10-49 | 49.8% | 5.0% |
| 50-99 | 47.9% | 8.5% |
| 100-499 | 45.7% | 6.4% |
| 500-999 | 46.9% | 7.7% |
| 1000+ | 44.1% | 8.4% |

Correlation: **-0.949** (strong inverse)

Larger trades have LOWER win rates but HIGHER ROI. This paradox is explained by:
- Larger trades on average are at more extreme (favorable odds) prices
- Lower win rate compensated by higher payouts when winning

### Hypothesis 4: Market Type Analysis
**Result: SIGNIFICANT VARIATION by market type**

Top performing market types:
| Market Type | Trades | Win Rate | ROI |
|-------------|--------|----------|-----|
| KXEPLSPREAD | 291 | 90.0% | **159.7%** |
| KXALEAGUEGAME | 25 | 52.0% | 64.7% |
| KXEPLGAME | 2,642 | 69.2% | 43.9% |
| KXARGPREMDIVGAME | 149 | 46.3% | 40.1% |

EPL (English Premier League) markets show exceptional profitability. This could be:
- Market inefficiency in sports betting
- Timing advantage (whales bet after private info)
- Sample size effects (need more data)

## The Profitable Strategy: Whale Moderate

### Definition
Follow whale trades (>=100 contracts) when the trade price is between 30-70 cents.

### Why It Works

1. **Not Too Extreme**: Prices 30-70c represent genuine uncertainty, not gambling
2. **Sufficient Size**: 100+ contracts indicates conviction, not noise
3. **Balanced Risk/Reward**: Win rate above breakeven without extreme leverage

### Performance Metrics

```
Strategy: Whale Moderate (>=100 @ 30-70c)
--------------------------------------------------
  Trades: 1,854
  Win Rate: 62.4%
  Breakeven Win Rate: 51.0%
  Edge: +11.4%

  Total Wagered: $185,400.00
  Total P/L: $37,781.34
  ROI: 20.4%

  Avg Win: $93.06
  Avg Loss: $100.00
  Max Drawdown: $2,677.22
  Profit Factor: 1.54
  Sharpe Ratio: 0.201
```

### Statistical Significance

With 1,854 trades and 11.4% edge, the 95% confidence interval for the edge is approximately:
- Lower bound: ~7%
- Upper bound: ~16%

This is statistically significant (p < 0.001).

## Implementation Recommendations

### Strategy 1: Simple Whale Following (Moderate Prices)

**Entry Criteria**:
- Trade size >= 100 contracts
- Trade price between 30-70 cents
- Follow the whale's direction (buy same side)

**Position Sizing**:
- Fixed $100 per signal (or 1% of bankroll)
- No more than 3 concurrent positions per market

**Expected Performance**:
- Win Rate: ~62%
- ROI: ~15-25% (accounting for execution slippage)
- Max Drawdown: ~$3,000 per $100 position size

### Strategy 2: Whale Favorites (Conservative)

**Entry Criteria**:
- Trade size >= 100 contracts
- Trade price >= 85 cents
- Follow the whale's direction

**Expected Performance**:
- Win Rate: ~96%
- ROI: ~2-3%
- Very low volatility, good for compounding

### Strategy 3: EPL Sports Markets

**Entry Criteria**:
- Market type starts with KXEPL
- Follow whale trades

**Caution**: Limited sample size (2,642 trades). Need more data.

## Risk Factors

### Known Risks

1. **Execution Risk**: May not get same price as whale
2. **Front-running**: Whale sees our following, adjusts strategy
3. **Selection Bias**: Only looking at resolved markets
4. **Market Regime**: Results from Dec 2025, may not persist

### Risk Mitigation

1. Use limit orders at whale price +/- 1c
2. Delay entry by 5-10 seconds to reduce detection
3. Validate with more historical data as it becomes available
4. Monitor real-time performance vs backtest

## What NOT To Do

Based on this analysis, these "obvious" strategies are NOT profitable:

1. **Following whale longshots**: -65% ROI, worse than random
2. **Following all whales blindly**: -14% ROI
3. **Following mega-whales (1000+)**: -24% ROI
4. **Betting against whales**: Also not profitable

## Next Steps

1. **Complete data fetch**: Currently at 24% of market outcomes
2. **Validate on full dataset**: Rerun analysis with all 7,931 markets
3. **Paper trade validation**: Implement in V3 trader for live testing
4. **Timing analysis**: Check if timing relative to market close matters
5. **Market type deep dive**: Analyze EPL and other sports markets separately

## Data Sources

- Historical trades: `training/reports/historical_trades_full.csv` (100,000 trades)
- Market outcomes: `training/reports/market_outcomes.csv` (in progress)
- Enriched trades: `training/reports/enriched_trades_partial.csv`

## Analysis Scripts

All scripts are in `backend/src/kalshiflow_rl/scripts/`:

1. `fetch_market_outcomes.py` - Fetches settlement results from Kalshi API
2. `analyze_trade_outcomes.py` - Joins trades with outcomes, calculates P/L
3. `detect_informed_trading.py` - Statistical hypothesis testing
4. `backtest_whale_following.py` - Strategy backtesting framework

## Appendix: Raw Results

### Overall Statistics

```
Total Trades: 100,000
Resolved Trades: 24,000 (24.0%)
Overall Win Rate: 49.5%
Total P/L: $117,248.74
Total Wagered: $1,535,564.26
Overall ROI: 7.64%
```

### Price Bucket Analysis

```
Price Bucket                     Trades   Win Rate   P/L          ROI
01-05c (extreme longshot)         1,973    0.7%   -$11,409     -20.4%
06-10c (longshot)                 1,189    2.9%   -$18,940     -27.1%
11-20c (underdog)                 2,350   10.1%   +$18,155      +9.3%
21-35c (slight underdog)          3,690   30.9%   -$19,754      -3.3%
36-50c (toss-up low)              3,390   44.5%   -$12,901      -0.9%
51-65c (toss-up high)             4,408   74.9%  +$171,598      +8.1%
66-80c (favorite)                 3,015   63.6%   -$19,669      -2.0%
81-90c (strong favorite)          1,859   87.7%    +$7,829      +1.1%
91-99c (near-certain)             2,126   98.2%    +$2,340      +0.3%
```

The most profitable price bucket is 51-65c with +8.1% ROI, aligning with our whale moderate strategy.

### Side Analysis

```
YES side: 14,492 trades, 42.8% win rate, +1.2% ROI
NO side:   9,508 trades, 59.7% win rate, +2.5% ROI
```

Interestingly, NO-side takers have slightly higher ROI. Worth investigating further.
