# MVP Best Strategy: YES at 80-90c

## Status: Ready for Implementation

---

## Executive Summary

Based on exhaustive analysis of **1,619,902 resolved trades across 72,791 unique markets**, the best validated strategy is:

| Metric | Value |
|--------|-------|
| **Strategy** | YES at 80-90c |
| **Unique Markets** | 2,110 |
| **Win Rate** | 88.9% |
| **Breakeven** | 83.9% |
| **Edge** | **+5.1%** |
| **Total Profit** | $1,615,185 |
| **P-Value** | < 0.0001 |
| **Max Concentration** | 19.3% |

This document defines the MVP implementation for the V3 Trader system.

---

## Part 1: Strategy Mechanics

### 1.1 Signal Detection

**Trigger Conditions (ALL must be true):**
```
1. Best YES ask price >= $0.80 AND <= $0.90
2. Best YES ask size >= 10 contracts (liquidity filter)
3. Bid-ask spread <= $0.05 (execution quality)
4. Market NOT already in portfolio
5. Under max concurrent positions limit
```

**MVP Decision: Pure price-based signals (no whale following required)**

The +5.1% edge exists in the price range itself. Whale confirmation adds complexity without proven additional value.

### 1.2 Entry Mechanics

**Order Type: Limit Order**
- Place at midpoint biased toward bid
- 60-second timeout
- Cancel if price moves outside 80-90c range

**Entry Price Calculation:**
```python
def calculate_entry_price(best_ask, best_bid):
    spread = best_ask - best_bid
    if spread <= 0.02:      # Tight spread
        return best_ask - 0.01
    elif spread <= 0.04:    # Normal spread
        return (best_ask + best_bid) / 2
    else:                   # Wide spread
        return best_bid + 0.01
```

### 1.3 Position Sizing (Aggressive Paper Trading)

**Bankroll Context:** ~$5,000 paper cash, can add $2,500/day

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Contracts per trade | 100 | ~$85 cost at 85c average |
| Max concurrent positions | **100** | Stress test low-money state |
| Bankroll fraction | ~2% per trade | Will deplete quickly - intentional |

**Note:** 100 max positions with ~$85/trade = $8,500 max exposure. This WILL exceed $5k bankroll and test how trader handles insufficient funds.

**Per-Trade Economics:**
- Entry cost: 100 × $0.85 = $85.00
- Win profit: 100 × $0.15 = $15.00 (17.6% return)
- Loss: -$85.00 (100% of position)

**Scaling by Signal Tier:**
| Tier | Price Range | Contracts | Cost |
|------|-------------|-----------|------|
| A (Best) | 83-87c | 150 | ~$128 |
| B (Good) | 80-83c, 87-90c | 100 | ~$85 |

### 1.4 Exit Mechanics

**MVP: Hold to Settlement**
- No early exits in v1
- Binary markets settle cleanly
- The +5.1% edge is measured on settled outcomes

### 1.5 Risk Management (Aggressive Paper Trading)

```python
RISK_LIMITS = {
    # Per-trade (aggressive for paper)
    "max_cost_per_trade": 150.00,      # Up to 150 contracts at 100c

    # Daily (aggressive - can replenish $2500/day)
    "max_new_positions_per_day": 50,    # Very aggressive for paper
    "daily_loss_limit": 2000.00,        # $2k/day limit

    # Portfolio (stress test mode)
    "max_concurrent_positions": 100,    # 100 positions - will exceed bankroll
    "max_total_exposure": 10000.00,     # Allow exceeding bankroll

    # Weekly
    "weekly_loss_limit": 5000.00,       # Match bankroll - full wipeout allowed
}
```

**Circuit Breaker:**
```python
def should_pause_trading():
    return any([
        daily_pnl < -2000,              # $2k daily stop
        weekly_pnl < -5000,             # $5k weekly stop (full bankroll)
        open_positions >= 100,          # Max positions
        api_errors_last_hour > 10,      # API issues
        available_balance < 50,         # Insufficient funds detection
    ])
```

**Why Aggressive + Stress Test:**
- 100 max positions WILL exceed $5k bankroll
- Tests how trader handles "insufficient funds" rejections
- Validates error handling and recovery
- Paper losses are free learning
- Can replenish $2,500/day if depleted

---

## Part 2: V3 Trader Integration

### 2.1 Architecture Fit

The V3 trader already has:
- EventBus for orderbook events (`ORDERBOOK_SNAPSHOT`, `ORDERBOOK_DELTA`)
- `TradingDecisionService` for order execution
- `StateContainer` for position tracking
- Rate limiting and deduplication patterns (from WhaleExecutionService)

**New Component: `Yes8090Service`**
- Subscribes to orderbook events
- Detects 80-90c YES opportunities
- Creates `TradingDecision` objects
- Calls `TradingDecisionService.execute_decision()`

### 2.2 Files to Modify

| File | Change | Lines |
|------|--------|-------|
| `services/trading_decision_service.py` | Add `YES_80_90` to `TradingStrategy` enum | ~5 |
| `services/yes_80_90_service.py` | **NEW** - Signal detection + execution | ~200 |
| `core/coordinator.py` | Wire in `Yes8090Service` lifecycle | ~20 |
| `config/environment.py` | Add config variables | ~5 |

**Total new code: ~230 lines**

### 2.3 Data Flow

```
┌─────────────────┐
│ OrderbookClient │
│ (existing)      │
└────────┬────────┘
         │ ORDERBOOK_SNAPSHOT/DELTA events
         ▼
┌─────────────────┐
│ Yes8090Service  │ ← NEW
│ (signal detect) │
└────────┬────────┘
         │ TradingDecision
         ▼
┌─────────────────────────┐
│ TradingDecisionService  │
│ (existing)              │
└────────┬────────────────┘
         │ place_order()
         ▼
┌─────────────────────────────┐
│ V3TradingClientIntegration  │
│ (existing)                  │
└─────────────────────────────┘
```

### 2.4 Key Code Structure

**Yes8090Service (new file):**
```python
class Yes8090Service:
    def __init__(self, event_bus, decision_service, state_container, config):
        self.event_bus = event_bus
        self.decision_service = decision_service
        self.state = state_container
        self.config = config
        self.processed_markets = set()  # Deduplication

    async def start(self):
        self.event_bus.subscribe(EventType.ORDERBOOK_SNAPSHOT, self._on_orderbook)
        self.event_bus.subscribe(EventType.ORDERBOOK_DELTA, self._on_orderbook)

    async def _on_orderbook(self, event):
        ticker = event.data.market_ticker
        orderbook = event.data

        # Check signal conditions
        signal = self._detect_signal(ticker, orderbook)
        if signal:
            await self._execute_entry(signal)

    def _detect_signal(self, ticker, orderbook) -> Optional[Signal]:
        # Skip if already processed or have position
        if ticker in self.processed_markets:
            return None
        if self.state.has_position(ticker):
            return None
        if self.state.position_count >= self.config.max_concurrent:
            return None

        # Check price range
        yes_ask = orderbook.best_yes_ask
        if not (0.80 <= yes_ask <= 0.90):
            return None

        # Check liquidity and spread
        if orderbook.yes_ask_size < 10:
            return None
        if orderbook.spread > 0.05:
            return None

        return Signal(ticker=ticker, yes_ask=yes_ask, ...)

    async def _execute_entry(self, signal):
        self.processed_markets.add(signal.ticker)

        decision = TradingDecision(
            action=TradingAction.BUY,
            market_ticker=signal.ticker,
            side="yes",
            quantity=self.config.contracts_per_trade,
            price=self._calculate_entry_price(signal),
            reason=f"YES 80-90c signal at {signal.yes_ask}c",
            strategy=TradingStrategy.YES_80_90,
        )

        await self.decision_service.execute_decision(decision)
```

---

## Part 3: MVP Simplifications

### What We're SKIPPING in v1

| Feature | Why Skip | Add in v2? |
|---------|----------|------------|
| Whale signal confirmation | Edge exists without it | Yes |
| Early exit logic | Settlement is cleaner | Yes |
| Variable position sizing | Fixed size simpler | Yes |
| Market category filtering | Learn which work first | Yes |
| Order timeout/resubmit | Simple cancel on miss | Maybe |

### Absolute Minimum for MVP

**Must Have:**
- [ ] Price detection in 80-90c range
- [ ] Limit order placement at calculated price
- [ ] Position tracking (open/settled)
- [ ] Basic P/L reporting
- [ ] Circuit breakers (daily/weekly loss limits)

**Nice to Have:**
- [ ] Auto-cancel on price move
- [ ] Dashboard metrics
- [ ] Daily summary reporting

---

## Part 4: Expected Performance (Aggressive Paper)

### Opportunity Frequency

- 2,110 markets in dataset over ~12 months
- **Target: 5-10 trades per day** (aggressive pursuit)

### P/L Model (100 contracts/trade)

```
Win: +$15.00 (100 contracts × $0.15)
Loss: -$85.00 (100 contracts × $0.85)

Expected value per trade:
  = (0.889 × $15.00) + (0.111 × -$85.00)
  = $13.34 - $9.44
  = +$3.90 per trade
```

| Timeframe | Trades | Expected P/L | 95% Range |
|-----------|--------|--------------|-----------|
| Day | 8 | +$31.20 | -$100 to +$150 |
| Week | 56 | +$218.40 | -$300 to +$700 |
| Month | 240 | +$936.00 | +$200 to +$1,600 |

### Aggressive Validation Timeline

| Phase | Duration | Trades | Goal |
|-------|----------|--------|------|
| Paper MVP | Days 1-3 | 20-30 | Verify execution works |
| Paper scale | Days 4-7 | 50-80 | Validate signal detection |
| Paper validation | Week 2 | 100+ | Statistical confidence |
| Live decision | Week 3 | Review | Go/no-go on live trading |

**Goal: 100+ paper trades in 2 weeks for statistical validation**

---

## Part 5: Future Enhancements (v2+)

### Phase 2: Refinements
- Add whale signal confirmation as boost
- Early exit at 95c+ (lock in gains)
- Market category filtering (sports vs crypto vs esports)
- Variable position sizing based on signal strength

### Phase 3: Multi-Strategy
- Add NO at 80-90c (+3.3% edge, $708k profit)
- Add YES at 90-100c (+1.1% edge, $665k profit)
- Portfolio-level correlation management

### Phase 4: Advanced
- ML-based entry timing optimization
- Cross-market arbitrage detection
- Dynamic risk adjustment

---

## Part 6: Implementation Checklist

### Pre-Implementation
- [x] Create `traderv3/planning/MVP_BEST_STRATEGY.md` (this document)
- [ ] Review with quant committee
- [x] Finalize parameter choices

### Implementation
- [ ] Add `YES_80_90` to `TradingStrategy` enum
- [ ] Create `Yes8090Service` class
- [ ] Wire into `V3Coordinator` lifecycle
- [ ] Add config variables
- [ ] Add logging/metrics

### Testing
- [ ] Unit tests for signal detection
- [ ] Integration test with mock orderbook
- [ ] Paper trading validation (30+ trades)

### Go-Live
- [ ] Paper trading win rate > 80%
- [ ] Execution error rate < 5%
- [ ] Circuit breakers tested
- [ ] Bankroll funded
- [ ] Monitoring process established

---

## Summary

**The simplest viable strategy:**

1. **Entry**: Buy YES when price is 80-90c with good liquidity
2. **Size**: 100 contracts per trade (~$85), 150 for Tier A signals
3. **Exit**: Hold to settlement
4. **Risk**: Max 100 positions (stress test mode)

**Expected outcome (aggressive paper)**:
- +$936/month on ~240 trades
- 100+ trades in 2 weeks for validation
- +$3.90 expected value per trade
- **Will intentionally hit insufficient funds** - tests error handling

**Implementation effort**: ~230 lines of new code, leveraging existing V3 infrastructure.

**Paper Trading Parameters (Stress Test Mode):**
- Bankroll: ~$5,000 (can add $2,500/day)
- Contracts: 100-150 per trade
- **Max positions: 100** (will exceed bankroll)
- Daily loss limit: $2,000
- Weekly loss limit: $5,000 (full wipeout allowed)
