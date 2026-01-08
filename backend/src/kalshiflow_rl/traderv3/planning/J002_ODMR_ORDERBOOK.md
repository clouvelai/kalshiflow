# J-002 Orderbook-Driven Mean Reversion (ODMR) Implementation Plan

> Refined implementation plan incorporating feedback from quant, trader-specialist, and websocket-engineer reviews.
> Status: **PROPOSED** | Created: 2026-01-07 | Updated: 2026-01-07
> Strategy Type: **Journey Trading** (intra-market price movement, not settlement)

---

## Executive Summary

Implement J-002 ODMR strategy to capitalize on mean reversion opportunities with orderbook-filtered entry timing. This strategy addresses a critical gap in J-001 Dip Buyer: **95.7% of dips recover, but only 41.3% hit the 5c target**, indicating entry timing/quality issues.

**Core Innovation**: Use orderbook microstructure signals (spread compression, bid/ask imbalance) to improve entry timing beyond price-based signals alone.

**Refined Success Targets** (per quant review - 65% was unrealistic):
- Target hit rate: 41.3% -> **50%** (minimum), 55% (target)
- Avg P&L per journey: +1.34c -> **+2.0c** (minimum)
- Win rate: 60.7% -> **65%** (minimum)
- Profit factor: 1.39 -> **1.5** (minimum)

**Architecture Approach**: No core V3 changes required. ODMR follows the existing J001 pattern - journey tracking via per-market state, existing `signal_id` conventions, and standard `TradingDecisionService.execute_decision()` for both entry and exit orders.

---

## 1. Research Foundation

### 1.1 Journey Trading Viability (JOURNEY-001)

**Validation**: 75.77% of liquid markets (5+ trades) have 10c+ price range
- Data: 10.18M trades across 384,933 markets
- Target threshold: >=50% (exceeded)
- **Conclusion**: Journey trading is viable for liquid markets

### 1.2 Mean Reversion Strength (JOURNEY-002)

**Validation**: 95.7% of 10c dips recover 5c+ before settlement
- Total dips: 104,264
- Recovered dips: 99,755
- Recovery rate: **95.7%**
- Median recovery time: **1.4 minutes**

**Key Insight**: Mean reversion is extremely strong, but J-001 only captures 41.3% of recoveries -> **entry quality issue**

### 1.3 J-001 Dip Buyer Current Performance

| Metric | Value |
|--------|-------|
| Total Journeys | 2,689 |
| Target Hit Rate | **41.3%** (1,111/2,689) |
| Timeout Exits | 36.7% (988) |
| Stop Loss Exits | 19.1% (515) |
| Win Rate | 60.7% |
| Avg P&L/Journey | **+1.34c** |
| Profit Factor | 1.39 |

**Problem**: Theoretical 95.7% recovery rate vs actual 41.3% target hit -> **54.4% gap**

---

## 2. Simplified Filter Design

### 2.1 Quant Review: Filter Simplification

**Original Plan Issues**:
1. Too many filters at once (whale + velocity + escalation + direction mix)
2. Size escalation logic was flawed (strict ordering unrealistic)
3. Whale direction wasn't considered
4. 65% target was unrealistic

**Revised Approach: ONE FILTER AT A TIME**

Start with the single most promising filter, validate it, then add incrementally.

### 2.2 Tier 1 Filter: Whale YES Trade Detection (SINGLE FILTER)

**Hypothesis**: Large YES trades into dips indicate informed buying, improving recovery odds.

**Logic**:
```python
def _check_whale_yes_filter(self, trade: Trade, market_state: MarketJourneyState) -> bool:
    """
    Single filter: Recent whale YES trade into dip.

    Rationale: Whale buys YES when price is low = informed recovery bet.
    Whale direction matters - whale NO trades are not bullish.
    """
    # Only trigger on dip detection (base conditions already met)

    # Look for whale YES trade in recent history (last 5 trades)
    recent_trades = list(market_state.recent_trades)[-5:]

    # Calculate whale threshold: 2x average trade size
    if len(market_state.trade_size_history) < 10:
        return False
    avg_size = sum(market_state.trade_size_history) / len(market_state.trade_size_history)
    whale_threshold = avg_size * 2.0

    # Check for whale YES trade (direction matters!)
    for t in recent_trades:
        if t.taker_side == "yes" and t.count >= whale_threshold:
            return True

    return False
```

**Why This Filter First**:
1. Strongest signal from external research (whale detection is well-studied)
2. Simple to implement and backtest
3. Direction-aware (fixes original flaw)
4. Clear threshold (2x average size)

### 2.3 Future Filters (Phase 2+ Only)

Add incrementally ONLY if Tier 1 validates:

1. **Size Escalation (Fixed)**: Use regression slope, not strict ordering
   ```python
   # WRONG: Strict increasing
   if not all(sizes[i] <= sizes[i+1] for i in range(len(sizes)-1)):
       return False

   # CORRECT: Positive regression slope
   slope = calculate_slope(sizes)
   if slope <= 0:
       return False
   ```

2. **Trade Velocity**: Compare trade frequency windows
3. **Direction Mix**: Require both YES and NO trades (healthy two-way flow)

### 2.4 Tier 2: Orderbook Enhancement (Live-Only)

**Simplified Orderbook Filtering**:
```python
def _check_orderbook_entry(self, market_ticker: str) -> bool:
    """
    Simple orderbook filter: spread tight + bid-heavy.
    """
    ob_context = self._get_orderbook_context(market_ticker)

    if ob_context is None or ob_context.is_stale:
        return False

    # Single check: spread <= 2c AND bid > ask
    if ob_context.no_spread is None or ob_context.no_spread > 2:
        return False

    if ob_context.no_bid_total_volume <= ob_context.no_ask_total_volume:
        return False

    return True
```

---

## 3. Architecture: Following J001 Pattern (No Core Changes)

### 3.1 Key Insight: J001 Already Proves Journey Trading Works

After reviewing J001 Dip Buyer implementation, we confirmed that **no V3 core changes are needed**. The existing architecture fully supports journey trading:

| Concern | Resolution |
|---------|------------|
| Journey tracking | Per-market state (same as J001's `DipMarketState`) |
| Entry/exit linking | Implicit via market state, not explicit `journey_id` |
| Exit orders | Existing `execute_decision(action="sell")` works |
| TradingState | Existing `POSITION_OPEN` covers journey positions |

### 3.2 signal_id Convention (Existing Pattern)

ODMR uses the same `signal_id` pattern as RLM_NO and J001:

```python
# Entry signal_id
signal_id = f"{ticker}:{int(time.time() * 1000)}"
# Example: "KXBTC-25JAN10-YES:1736280000000"

# Exit signal_id
signal_id = f"{ticker}:exit:{int(time.time() * 1000)}"
# Example: "KXBTC-25JAN10-YES:exit:1736280500000"
```

**No explicit journey_id needed** - the link between entry and exit is implicit via per-market state tracking (same as J001).

### 3.3 Per-Market State (Same as J001)

```python
@dataclass
class MarketJourneyState:
    """Per-market state for ODMR tracking (mirrors J001's DipMarketState)."""
    market_ticker: str

    # Rolling high tracking
    rolling_high_price: Optional[int] = None
    rolling_high_time: Optional[float] = None

    # Trade history for filters
    recent_trades: deque = field(default_factory=lambda: deque(maxlen=20))
    trade_size_history: deque = field(default_factory=lambda: deque(maxlen=50))
    trade_count: int = 0

    # Position tracking (same pattern as J001)
    position_open: bool = False
    entry_price: Optional[int] = None
    entry_time: Optional[float] = None
    entry_trade_index: int = 0
    target_price: Optional[int] = None      # entry + 5c
    stop_loss_price: Optional[int] = None   # entry - 10c

    # Optional: spread tracking for future filter
    spread_history: deque = field(default_factory=lambda: deque(maxlen=10))
```

### 3.4 Exit Order Placement (Existing Pattern)

J001 already places exit orders using the standard `TradingDecisionService`:

```python
# From j001_dip_buyer.py - this pattern works, no changes needed
decision = TradingDecision(
    action="sell",
    market=ticker,
    side="yes",
    quantity=position_contracts,
    price=exit_price,
    reason=f"ODMR exit ({exit_type}): {ticker}",
    strategy_id="odmr",
    signal_params={
        "exit_type": exit_type,  # "target", "stop_loss", "timeout"
        "entry_price": state.entry_price,
        "exit_price": exit_price,
        "pnl_cents": exit_price - state.entry_price,
    }
)
await self._context.trading_service.execute_decision(decision)
```

### 3.5 ODMR and RLM_NO Coexistence

ODMR and RLM_NO can run simultaneously without conflict:

| Aspect | ODMR | RLM_NO |
|--------|------|--------|
| Trade Side | Buys **YES** on dips | Buys **NO** on flow imbalance |
| Exit Model | Price target/stop/timeout | Hold to settlement |
| Position Type | Short-term journey | Settlement bet |
| Conflict? | **None** - different sides, different exits |

A market can have both ODMR and RLM_NO positions simultaneously.

---

## 4. Event Subscriptions

### 4.1 Correct Event Pattern

**Subscribe to** (same as J001):
- `PUBLIC_TRADE_RECEIVED` - Primary trigger for dip detection and exit monitoring
- `ORDER_FILL` - Track entry/exit fill events
- `MARKET_DETERMINED` - Cleanup on settlement

**Do NOT subscribe to** (use on-demand queries instead):
- `ORDERBOOK_SNAPSHOT`
- `ORDERBOOK_DELTA`

```python
@StrategyRegistry.register("odmr")
class ODMRStrategy:
    name: str = "odmr"
    display_name: str = "Orderbook-Driven Mean Reversion"
    subscribed_events: Set[EventType] = {
        EventType.PUBLIC_TRADE_RECEIVED,
        EventType.ORDER_FILL,
        EventType.MARKET_DETERMINED,
    }
```

### 4.2 On-Demand Orderbook Access

Use existing `OrderbookContext` class (don't reimplement):

```python
async def _get_orderbook_context(self, market_ticker: str) -> Optional[OrderbookContext]:
    """Get orderbook context on-demand."""
    try:
        orderbook_state = await asyncio.wait_for(
            get_shared_orderbook_state(market_ticker),
            timeout=self._orderbook_timeout
        )
        snapshot = await orderbook_state.get_snapshot()

        return OrderbookContext.from_orderbook_snapshot(
            snapshot,
            stale_threshold_seconds=self._orderbook_stale_threshold,
            tight_spread=self._tight_spread,
            normal_spread=self._normal_spread,
        )
    except (asyncio.TimeoutError, Exception) as e:
        logger.debug(f"Orderbook unavailable for {market_ticker}: {e}")
        return None
```

### 4.3 Processing Lock

Add asyncio.Lock() to prevent concurrent processing (matching RLM/J001 pattern):

```python
def __init__(self):
    # ... other init ...
    self._processing_lock = asyncio.Lock()

async def _handle_public_trade(self, trade_event: PublicTradeEvent) -> None:
    if not self._running or not self._context:
        return

    async with self._processing_lock:
        await self._process_trade(trade_event)
```

### 4.4 MARKET_DETERMINED Handler for Cleanup

```python
async def _handle_market_determined(self, event: MarketDeterminedEvent) -> None:
    """Handle market determined events for cleanup."""
    ticker = event.market_ticker
    state = self._market_states.get(ticker)

    if state and state.position_open:
        # Position settled before we could exit - record as settlement exit
        self._stats["settlement_exits"] += 1
        logger.info(f"ODMR position settled: {ticker}")

    # Clean up market state
    if ticker in self._market_states:
        del self._market_states[ticker]

    logger.debug(f"ODMR cleanup for determined market: {ticker}")
```

---

## 5. Implementation Design

### 5.1 V3 Plugin Structure

**File**: `backend/src/kalshiflow_rl/traderv3/strategies/plugins/odmr.py`

```python
"""
ODMR Strategy Plugin - Orderbook-Driven Mean Reversion for TRADER V3.

This plugin implements journey trading with orderbook-filtered entries:
Buy YES dips when whale YES activity + tight spread + bid-heavy orderbook.

Architecture:
    - Follows J001 Dip Buyer pattern exactly
    - Per-market state tracking (MarketJourneyState)
    - Standard signal_id convention (no journey_id)
    - Uses existing TradingDecisionService for entry AND exit orders
    - No V3 core changes required
"""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Set

from ..protocol import Strategy, StrategyContext
from ..registry import StrategyRegistry
from ...core.events import EventType
from ...core.event_bus import PublicTradeEvent, MarketDeterminedEvent, OrderFillEvent
from ...core.state_machine import TraderState
from ....data.orderbook_state import get_shared_orderbook_state
from ...state.order_context import OrderbookContext
from ...services.trading_decision_service import TradingDecision, TradingStrategy

logger = logging.getLogger("kalshiflow_rl.traderv3.strategies.plugins.odmr")


@dataclass
class Trade:
    """Trade record for journey tracking."""
    market_ticker: str
    timestamp: float
    yes_price: int
    taker_side: str  # "yes" or "no"
    count: int
    trade_id: str


@dataclass
class MarketJourneyState:
    """Per-market state for ODMR tracking (same pattern as J001)."""
    market_ticker: str

    # Rolling high tracking
    rolling_high_price: Optional[int] = None
    rolling_high_time: Optional[float] = None

    # Trade history
    recent_trades: deque = field(default_factory=lambda: deque(maxlen=20))
    trade_size_history: deque = field(default_factory=lambda: deque(maxlen=50))
    trade_count: int = 0

    # Position tracking (matches J001 pattern)
    position_open: bool = False
    entry_price: Optional[int] = None
    entry_time: Optional[float] = None
    entry_trade_index: int = 0
    target_price: Optional[int] = None
    stop_loss_price: Optional[int] = None


@StrategyRegistry.register("odmr")
class ODMRStrategy:
    """
    ODMR Strategy Plugin - Orderbook-Driven Mean Reversion.

    Journey trading: buy YES dips, exit on price recovery.
    """

    name: str = "odmr"
    display_name: str = "Orderbook-Driven Mean Reversion"
    subscribed_events: Set[EventType] = {
        EventType.PUBLIC_TRADE_RECEIVED,
        EventType.ORDER_FILL,
        EventType.MARKET_DETERMINED,
    }

    def __init__(self):
        self._context: Optional[StrategyContext] = None
        self._running: bool = False
        self._started_at: Optional[float] = None

        # Base parameters
        self._dip_threshold_cents: int = 10
        self._recovery_target_cents: int = 5
        self._stop_loss_cents: int = 10
        self._timeout_trades: int = 50
        self._price_floor_cents: int = 15
        self._price_ceiling_cents: int = 85
        self._min_trades_before_dip: int = 5
        self._contracts_per_trade: int = 100

        # Whale filter (Tier 1 - single filter)
        self._whale_multiplier: float = 2.0
        self._whale_lookback: int = 5
        self._min_history_for_whale: int = 10

        # Orderbook filters (Tier 2 - live only)
        self._use_orderbook_filtering: bool = True
        self._max_entry_spread: int = 2
        self._orderbook_timeout: float = 2.0
        self._orderbook_stale_threshold: float = 5.0
        self._tight_spread: int = 2
        self._normal_spread: int = 4

        # State tracking
        self._market_states: Dict[str, MarketJourneyState] = {}
        self._open_positions: Set[str] = set()  # Tickers with open journey

        # Processing lock
        self._processing_lock = asyncio.Lock()

        # Statistics
        self._stats = {
            "trades_processed": 0,
            "dips_detected": 0,
            "whale_filter_passed": 0,
            "whale_filter_rejected": 0,
            "orderbook_filter_passed": 0,
            "orderbook_filter_rejected": 0,
            "entries_executed": 0,
            "target_exits": 0,
            "stop_loss_exits": 0,
            "timeout_exits": 0,
            "settlement_exits": 0,
        }

        # P&L tracking
        self._session_pnl_cents: int = 0

        logger.debug("ODMRStrategy initialized")

    async def start(self, context: StrategyContext) -> None:
        """Start the strategy."""
        self._context = context
        self._running = True
        self._started_at = time.time()

        # Subscribe to events
        await context.event_bus.subscribe_to_public_trade(self._handle_public_trade)
        await context.event_bus.subscribe_to_order_fill(self._handle_order_fill)
        await context.event_bus.subscribe_to_market_determined(self._handle_market_determined)

        logger.info("ODMR strategy started")

    async def stop(self) -> None:
        """Stop the strategy."""
        self._running = False
        logger.info(f"ODMR strategy stopped. Session P&L: {self._session_pnl_cents}c")

    def is_healthy(self) -> bool:
        """Check if strategy is healthy."""
        return self._running and self._context is not None

    def get_stats(self) -> Dict[str, Any]:
        """Get strategy statistics."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "running": self._running,
            "started_at": self._started_at,
            "session_pnl_cents": self._session_pnl_cents,
            "open_positions": len(self._open_positions),
            **self._stats,
        }

    # Event handlers and trade processing follow J001 pattern...
```

### 5.2 Configuration

**File**: `backend/src/kalshiflow_rl/traderv3/strategies/config/odmr.yaml`

```yaml
name: odmr
enabled: true
display_name: "Orderbook-Driven Mean Reversion"
max_positions: 30

params:
  # Base journey parameters
  dip_threshold_cents: 10
  recovery_target_cents: 5
  stop_loss_cents: 10
  timeout_trades: 50
  price_floor_cents: 15
  price_ceiling_cents: 85
  min_trades_before_dip: 5
  contracts_per_trade: 100

  # Tier 1: Whale filter (single filter - start simple)
  whale_multiplier: 2.0        # 2x average = whale
  whale_lookback: 5            # Check last 5 trades
  min_history_for_whale: 10    # Need 10+ trades to calculate avg

  # Tier 2: Orderbook filters (live only)
  use_orderbook_filtering: true
  max_entry_spread: 2          # Only enter when spread <= 2c
  orderbook_timeout: 2.0
  orderbook_stale_threshold: 5.0
  tight_spread: 2
  normal_spread: 4
```

---

## 6. Validation Plan

### 6.1 Realistic Success Metrics (Quant Review)

| Metric | J-001 Baseline | Minimum | Target | Stretch |
|--------|---------------|---------|--------|---------|
| Target Hit Rate | 41.3% | **45%** | **50%** | 55% |
| Avg P&L/Journey | +1.34c | **+1.5c** | **+2.0c** | +2.5c |
| Win Rate | 60.7% | **62%** | **65%** | 68% |
| Profit Factor | 1.39 | **1.45** | **1.55** | 1.7 |
| Sample Size | - | **500** | 1,000 | 2,000 |

### 6.2 Bucket-Matched Validation (Quant Review)

**Critical**: Must validate edge controlling for entry price distribution.

**Validation Criteria**:
- Edge positive in **at least 3 out of 5** major price buckets
- No single bucket contributes >30% of total signals
- Weighted average edge positive after bucket matching

### 6.3 Backtestable Base Validation

**Objective**: Validate whale YES filter improves J-001

**Script**: `research/backtest/run_j002_odmr_base.py`

**Success Criteria** (pass/fail):
- [ ] Target hit rate >= 45% (vs 41.3% baseline)
- [ ] Avg P&L >= +1.5c (vs +1.34c baseline)
- [ ] Profit factor >= 1.45 (vs 1.39 baseline)
- [ ] Bucket-matched edge positive
- [ ] Sample size >= 500 journeys

### 6.4 Live Paper Trading Validation

**Objective**: Validate orderbook filter improves base strategy

**Duration**: 2 weeks minimum

**A/B Test**: Toggle `use_orderbook_filtering` parameter

**Success Criteria**:
- [ ] Orderbook-filtered: Target hit rate >= 50%
- [ ] Orderbook-filtered: Avg P&L >= +2.0c
- [ ] Rejection rate < 40% (not too aggressive)
- [ ] Base-only: Still >= 45% (sanity check)

---

## 7. Implementation Phases

### Phase 1: Backtestable Base (Week 1)

**Deliverables**:
1. [ ] Implement `research/backtest/strategies/j002_odmr_base.py`
2. [ ] Create `research/backtest/run_j002_odmr_base.py` runner
3. [ ] Run backtest with whale YES filter only
4. [ ] Validate against success criteria
5. [ ] Document results

**Go/No-Go**: If base strategy doesn't improve J-001, re-evaluate filter design.

### Phase 2: V3 Plugin Implementation (Week 2)

**Deliverables**:
1. [ ] Create `strategies/plugins/odmr.py` (following J001 pattern)
2. [ ] Create `strategies/config/odmr.yaml`
3. [ ] Integration tests with mock orderbook
4. [ ] Local testing with paper account

**Files Created**:
- `strategies/plugins/odmr.py`
- `strategies/config/odmr.yaml`
- Tests

### Phase 3: Exit Optimization (Week 3) - SEPARATE TRACK

**Per Quant Review**: Exit optimization is as important as entry optimization.

**Enhancements to Explore**:
1. **Increase Timeout**: 50 trades may be too short for recovery
   - Test 75 and 100 trade timeouts
2. **Dynamic Exit**: Adjust target based on dip size
   - Larger dips -> larger targets
3. **Trailing Stop**: Protect profits on strong recovery
   - After +3c gain, move stop to breakeven

**This is a separate validation track** - don't bundle with entry filter validation.

### Phase 4: Live Paper Trading (Week 4-5)

**Deliverables**:
1. [ ] Deploy to paper trader
2. [ ] Monitor for 2 weeks
3. [ ] A/B test base vs orderbook filtering
4. [ ] Collect bucket-matched metrics
5. [ ] Final validation report

### Phase 5: Production Decision (Week 6)

**Deliverables**:
1. [ ] Review all validation results
2. [ ] Document final parameters
3. [ ] Production deployment decision
4. [ ] Update VALIDATED_STRATEGIES.md (if validated)

---

## 8. Risk Mitigation

### 8.1 Filter Over-Aggressiveness

**Risk**: Whale filter rejects too many valid dips

**Mitigation**:
- Start with 2.0x multiplier (lower threshold)
- Monitor rejection rate (target: 30-40%)
- Adjust multiplier based on results

### 8.2 Exit Timing

**Risk**: Timeout exits dominate (J-001 had 36.7%)

**Mitigation**:
- Phase 3 exit optimization as separate track
- Increase timeout from 50 to 75 trades
- Add trailing stop after +3c gain

### 8.3 Orderbook Data Availability

**Risk**: Orderbook stale or unavailable

**Mitigation**:
- Graceful degradation to base-only filters if orderbook unavailable
- Track rejection stats to monitor impact
- Use REST fallback when WS degraded

### 8.4 Strategy Conflict

**Risk**: ODMR conflicts with RLM_NO

**Resolution**: No conflict - different trade sides and exit models.
- ODMR: Buy YES, exit on price
- RLM_NO: Buy NO, hold to settlement

---

## 9. Success Metrics Summary

### Minimum Viable Performance

| Metric | Minimum | Target |
|--------|---------|--------|
| Target Hit Rate | 45% | 50% |
| Avg P&L/Journey | +1.5c | +2.0c |
| Win Rate | 62% | 65% |
| Profit Factor | 1.45 | 1.55 |
| Bucket-Matched Edge | Positive | Positive in 3+ buckets |
| Sample Size | 500 | 1,000 |

### Improvement Over J-001

Must demonstrate improvement:
- Target hit rate: +4% absolute (41.3% -> 45%+)
- Avg P&L: +12% relative (+1.34c -> +1.5c+)
- Profit factor: +4% relative (1.39 -> 1.45+)

---

## 10. References

### Research Documents
- `research/RESEARCH_JOURNAL.md` - Session 2026-01-07c (Mean Reversion Analysis)
- `research/reports/journey_mean_reversion.json` - Mean reversion validation
- `research/reports/dip_buyer_journey_20260107_111738.json` - J-001 baseline

### Implementation References
- `backend/src/kalshiflow_rl/traderv3/strategies/plugins/j001_dip_buyer.py` - **Primary pattern reference**
- `backend/src/kalshiflow_rl/traderv3/strategies/plugins/rlm_no.py` - Plugin pattern
- `backend/src/kalshiflow_rl/traderv3/state/order_context.py` - OrderbookContext
- `backend/src/kalshiflow_rl/traderv3/services/trading_decision_service.py` - Trading service

---

## 11. Appendix: Key Patterns from J001

### Per-Market State Pattern
```python
# J001's DipMarketState - ODMR uses same pattern
@dataclass
class MarketJourneyState:
    market_ticker: str
    rolling_high_price: Optional[int] = None
    position_open: bool = False
    entry_price: Optional[int] = None
    entry_time: Optional[float] = None
    target_price: Optional[int] = None
    stop_loss_price: Optional[int] = None
```

### Entry Signal Pattern
```python
# Standard signal_id format
signal_id = f"{ticker}:{int(time.time() * 1000)}"

decision = TradingDecision(
    action="buy",
    market=ticker,
    side="yes",
    quantity=self._contracts_per_trade,
    price=entry_price,
    reason=f"ODMR dip: {dip_depth}c drop, whale YES detected",
    strategy_id="odmr",
    signal_params={...}
)
await self._context.trading_service.execute_decision(decision)
```

### Exit Order Pattern
```python
# Exit using same TradingDecisionService - no special journey_exit needed
signal_id = f"{ticker}:exit:{int(time.time() * 1000)}"

decision = TradingDecision(
    action="sell",
    market=ticker,
    side="yes",
    quantity=state.entry_contracts,
    price=exit_price,
    reason=f"ODMR exit ({exit_type})",
    strategy_id="odmr",
    signal_params={
        "exit_type": exit_type,  # "target", "stop_loss", "timeout"
        "entry_price": state.entry_price,
        "pnl_cents": exit_price - state.entry_price,
    }
)
await self._context.trading_service.execute_decision(decision)
```

### Processing Lock Pattern
```python
async def _handle_public_trade(self, trade_event: PublicTradeEvent) -> None:
    if not self._running or not self._context:
        return

    async with self._processing_lock:
        await self._process_trade(trade_event)
```

---

**End of Refined Plan**
