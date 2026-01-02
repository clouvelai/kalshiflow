# S-RLM-001 (Reverse Line Movement) Implementation Plan for V3 Trader

## Executive Summary

Implement S-RLM-001 strategy in the V3 Trader lifecycle mode. This is the **highest edge validated strategy** (+17.38% base, +24.88% optimal) that detects when retail bets YES but price moves toward NO.

**Signal**: When >65% of trades are YES, but YES price drops >=5c, bet NO

## 1. Architecture Analysis

### 1.1 Current V3 Architecture

The V3 Trader follows an **event-driven architecture** with these key components:

| Component | File | Purpose |
|-----------|------|---------|
| `EventBus` | `core/event_bus.py` | Central pub/sub for all events |
| `TradingDecisionService` | `services/trading_decision_service.py` | Strategy execution |
| `TrackedMarketsState` | `state/tracked_markets.py` | Market tracking with metadata |
| `V3TradesIntegration` | `clients/trades_integration.py` | Public trades WebSocket |
| `EventLifecycleService` | `services/event_lifecycle_service.py` | Market discovery |
| `Yes8090Service` | `services/yes_80_90_service.py` | **Reference pattern** for event-driven strategies |

### 1.2 Existing Event Types Available

From `event_bus.py`:
- `PUBLIC_TRADE_RECEIVED` - Individual public trades with `market_ticker`, `taker_side`, `yes_price`, `count`, `timestamp_ms`
- `ORDERBOOK_SNAPSHOT` / `ORDERBOOK_DELTA` - Current orderbook state
- `MARKET_TRACKED` - When a market is added to tracking
- `MARKET_DETERMINED` - When a market settles

### 1.3 Data Flow for RLM

```
Kalshi Public Trades WS
    |
    v
TradesClient (trades_client.py)
    | on_trade_callback
    v
V3TradesIntegration (trades_integration.py)
    | emit_public_trade()
    v
EventBus - PUBLIC_TRADE_RECEIVED event
    |
    +---> WhaleTracker (existing)
    +---> RLMService (NEW) <-- accumulates trades per market
                 |
                 | detect_rlm_signal()
                 v
         TradingDecisionService.execute_decision()
```

## 2. Data Requirements Analysis

### 2.1 Public Trade Data Fields

From `trades_client.py` line 255-296, each trade provides:
```python
trade_data = {
    "market_ticker": str,      # e.g., "KXNFL-25JAN05-DET"
    "yes_price": int,          # 0-100 cents
    "no_price": int,           # 0-100 cents (100 - yes_price)
    "count": int,              # Number of contracts
    "taker_side": str,         # "yes" or "no"
    "timestamp_ms": int,       # Milliseconds
}
```

### 2.2 RLM Signal Requirements

To detect RLM signal, we need **per-market aggregates**:

| Data | Source | Storage |
|------|--------|---------|
| Total YES trades (count) | PUBLIC_TRADE_RECEIVED | Per-market accumulator |
| Total NO trades (count) | PUBLIC_TRADE_RECEIVED | Per-market accumulator |
| First YES price | First trade in market | Per-market state |
| Last YES price | Most recent trade | Per-market state |
| Trade count | Count of trades | Per-market state |
| Market category | TrackedMarketsState | Pre-filtered |

### 2.3 Category Filtering

From VALIDATED_STRATEGIES.md lines 32-69, RLM works in:
- **Sports** (KXNFL*, KXNBA*, KXNCAAF*, etc.)
- **Crypto** (KXBTC*, KXETH*, etc.)
- **Entertainment** (KXNETFLIX*, KXSPOTIFY*, etc.)
- **Media_Mentions** (KXMRBEAST*, KXCOLBERT*, etc.)

**Exclude**: Weather, Economics, Politics

The `EventLifecycleService` already filters by category (line 50-54):
```python
DEFAULT_LIFECYCLE_CATEGORIES = ["sports", "media_mentions", "entertainment", "crypto"]
```

## 3. Implementation Design

### 3.1 New Service: `RLMService`

**Location**: `backend/src/kalshiflow_rl/traderv3/services/rlm_service.py`

**Pattern**: Follow `Yes8090Service` structure (event-driven, not cycle-based)

```python
@dataclass
class MarketTradeState:
    """Accumulates trade data for RLM signal detection."""
    market_ticker: str
    yes_trades: int = 0
    no_trades: int = 0
    first_yes_price: Optional[int] = None
    last_yes_price: Optional[int] = None
    first_trade_time: Optional[float] = None
    last_trade_time: Optional[float] = None

    @property
    def total_trades(self) -> int:
        return self.yes_trades + self.no_trades

    @property
    def yes_ratio(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.yes_trades / self.total_trades

    @property
    def price_drop(self) -> int:
        if self.first_yes_price is None or self.last_yes_price is None:
            return 0
        return self.first_yes_price - self.last_yes_price


class RLMService:
    """
    S-RLM-001: Reverse Line Movement NO strategy.

    Signal: >65% YES trades + YES price drops >=5c = bet NO
    Edge: +17.38% base, +24.88% optimal

    Subscribes to:
    - PUBLIC_TRADE_RECEIVED: Accumulate trades per market
    - MARKET_TRACKED: Initialize new market tracking
    - MARKET_DETERMINED: Cleanup completed markets
    """
```

### 3.2 Signal Detection Logic

From VALIDATED_STRATEGIES.md lines 1921-1932:

```python
def detect_rlm_signal(self, market_ticker: str) -> Optional[RLMSignal]:
    """
    Detect RLM signal for a market.

    Returns signal if:
    1. yes_trade_ratio > 0.65 (majority YES bets)
    2. price_drop >= 5 (YES price dropped 5+ cents)
    3. total_trades >= 15 (sufficient activity)
    """
    state = self._market_states.get(market_ticker)
    if not state:
        return None

    # Check trade count
    if state.total_trades < self._min_trades:
        return None

    # Check YES ratio
    if state.yes_ratio <= self._yes_threshold:
        return None

    # Check price drop
    if state.price_drop < self._min_price_drop:
        return None

    return RLMSignal(
        market_ticker=market_ticker,
        yes_ratio=state.yes_ratio,
        price_drop=state.price_drop,
        trade_count=state.total_trades,
    )
```

### 3.3 Event Handling

```python
async def _handle_public_trade(self, event: PublicTradeEvent) -> None:
    """Process incoming public trade."""
    if not self._running:
        return

    market_ticker = event.market_ticker

    # Only process markets we're tracking
    if market_ticker not in self._tracked_markets:
        return

    # Get or create market state
    if market_ticker not in self._market_states:
        self._market_states[market_ticker] = MarketTradeState(
            market_ticker=market_ticker
        )

    state = self._market_states[market_ticker]

    # Update trade counts
    if event.side == "yes":
        state.yes_trades += 1
    else:
        state.no_trades += 1

    # Track price movement (YES price)
    yes_price = event.price_cents if event.side == "yes" else (100 - event.price_cents)

    if state.first_yes_price is None:
        state.first_yes_price = yes_price
        state.first_trade_time = time.time()

    state.last_yes_price = yes_price
    state.last_trade_time = time.time()

    # Check for signal
    signal = self.detect_rlm_signal(market_ticker)
    if signal:
        await self._execute_signal(signal)
```

### 3.4 Integration with TrackedMarketsState

Subscribe to MARKET_TRACKED to know which markets to monitor:

```python
async def _handle_market_tracked(self, event: MarketTrackedEvent) -> None:
    """Handle new market being tracked."""
    market_ticker = event.market_ticker
    category = event.category.lower()

    # Category already filtered by EventLifecycleService
    # Just add to our tracked set
    self._tracked_markets.add(market_ticker)

    logger.debug(f"RLM now monitoring: {market_ticker} ({category})")
```

### 3.5 Coordinator Integration

Add to `coordinator.py` similar to `Yes8090Service`:

```python
# Initialize RLM service (if trading client available and strategy is RLM_NO)
self._rlm_service: Optional[RLMService] = None
if trading_client_integration and self._trading_service:
    if strategy == TradingStrategy.RLM_NO:
        self._rlm_service = RLMService(
            event_bus=event_bus,
            trading_service=self._trading_service,
            state_container=self._state_container,
            tracked_markets_state=self._tracked_markets_state,
            yes_threshold=0.65,
            min_trades=15,
            min_price_drop=5,
        )
        self._health_monitor.set_rlm_service(self._rlm_service)
        logger.info("RLMService initialized for Reverse Line Movement strategy")
```

## 4. File Changes Summary

### 4.1 New Files

| File | Purpose |
|------|---------|
| `traderv3/services/rlm_service.py` | RLM signal detection and execution |

### 4.2 Modified Files

| File | Changes |
|------|---------|
| `traderv3/services/trading_decision_service.py` | Add `TradingStrategy.RLM_NO` enum |
| `traderv3/core/coordinator.py` | Initialize RLMService when strategy=RLM_NO |
| `traderv3/core/health_monitor.py` | Add RLM health tracking |
| `traderv3/config/environment.py` | Add RLM config parameters |

### 4.3 Configuration

Add to `.env.paper`:
```bash
V3_TRADING_STRATEGY=rlm_no
RLM_YES_THRESHOLD=0.65
RLM_MIN_TRADES=15
RLM_MIN_PRICE_DROP=5
RLM_CONTRACTS_PER_TRADE=5
RLM_MAX_CONCURRENT=10
```

## 5. Data Flow Diagram

```
[Kalshi WS: market_lifecycle_v2] --> [LifecycleClient]
                                            |
                                            v
                                   [V3LifecycleIntegration]
                                            |
                                            | MARKET_LIFECYCLE_EVENT
                                            v
                                   [EventLifecycleService]
                                            |
                                            | Category filter (sports, crypto, etc.)
                                            v
                                   [TrackedMarketsState]
                                            |
                                            | MARKET_TRACKED event
                                            v
                                   +---> [RLMService] <---+
                                   |         |            |
                                   |         | track      |
                                   |         v            |
[Kalshi WS: trade] -------------> [TradesClient]         |
                                            |             |
                                            | PUBLIC_TRADE_RECEIVED
                                            v             |
                                   [RLMService]           |
                                            |             |
                                            | accumulate per market
                                            | detect_rlm_signal()
                                            v             |
                                   [TradingDecisionService]
                                            |
                                            | execute_decision()
                                            v
                                   [V3TradingClientIntegration]
                                            |
                                            | place_order()
                                            v
                                   [Kalshi REST API]
```

## 6. Key Design Decisions

### 6.1 Event-Driven vs Cycle-Based

**Decision**: Event-driven (like `Yes8090Service`)

**Rationale**:
- RLM signals emerge from trade flow, not orderbook state
- Need to react immediately when signal triggers
- Trade accumulation naturally event-driven

### 6.2 Per-Market State Accumulation

**Decision**: In-memory `Dict[str, MarketTradeState]`

**Rationale**:
- Markets are short-lived (hours to days)
- Need fast access for every trade
- Cleanup on MARKET_DETERMINED clears memory

### 6.3 Category Filtering

**Decision**: Use existing `EventLifecycleService` filtering

**Rationale**:
- Already filters to Sports, Crypto, Entertainment, Media_Mentions
- RLM validated on these same categories
- Avoid duplicate filtering logic

### 6.4 One Entry Per Market

**Decision**: Track `_executed_markets: Set[str]`

**Rationale**:
- Hold to settlement (no re-entry)
- Like `Yes8090Service._processed_markets`
- Clear on MARKET_DETERMINED for next cycle

## 7. Implementation Phases

### Phase 1: Core Service (Day 1)
1. Create `rlm_service.py` with `MarketTradeState` dataclass
2. Implement `_handle_public_trade()` trade accumulation
3. Implement `detect_rlm_signal()` detection logic
4. Add `TradingStrategy.RLM_NO` to enum

### Phase 2: Integration (Day 1)
1. Subscribe to `MARKET_TRACKED` for market discovery
2. Subscribe to `PUBLIC_TRADE_RECEIVED` for trade flow
3. Integrate with `TradingDecisionService` for execution
4. Add to `coordinator.py` initialization

### Phase 3: Configuration (Day 2)
1. Add config parameters to `V3Config`
2. Add environment variables
3. Add health monitoring
4. Add statistics tracking

### Phase 4: Testing (Day 2)
1. Unit tests for signal detection
2. Integration test with mock trades
3. Paper trading validation
4. Monitor signal quality

## 8. Risk Considerations

### 8.1 Data Quality
- **Risk**: Public trades may have gaps/delays
- **Mitigation**: Require minimum 15 trades before signal

### 8.2 Memory Growth
- **Risk**: Accumulating state for many markets
- **Mitigation**: Cleanup on MARKET_DETERMINED, bound `_market_states` size

### 8.3 Rate Limiting
- **Risk**: Too many signals at once
- **Mitigation**: Token bucket rate limiting (like Yes8090Service)

### 8.4 Category Drift
- **Risk**: Categories change or new ones appear
- **Mitigation**: Log rejected categories, periodic review

## 9. Success Metrics

| Metric | Target | Rationale |
|--------|--------|-----------|
| Signal detection rate | 100-500/day | Based on historical ~330/day |
| Win rate | >85% | Expected 90.2% |
| Edge per trade | >15% | Expected +17.38% |
| False positive rate | <5% | Tight parameters |

## 10. Open Questions for User

1. **Position sizing**: Start with 5 contracts per trade, or scale based on confidence?
2. **Max concurrent positions**: 10 seems conservative - increase?
3. **Cooldown**: After signal, wait before re-evaluating same market?
4. **Combination signals**: Should we also require "large move (5c+)" for optimal edge?

---

## Appendix A: Reference Implementation from VALIDATED_STRATEGIES.md

```python
def detect_rlm_signal(market: str, trades: list,
                       yes_threshold: float = 0.65,
                       min_trades: int = 15,
                       min_price_drop: int = 5) -> dict:
    """
    Detect Reverse Line Movement (RLM) signal.

    Returns dict with:
    - triggered: bool
    - yes_ratio: float
    - price_drop: int
    - n_trades: int
    - reason: str
    """
    market_trades = [t for t in trades if t['market_ticker'] == market]

    if len(market_trades) < min_trades:
        return {
            'triggered': False,
            'reason': f'insufficient_trades_{len(market_trades)}'
        }

    # Sort by timestamp
    market_trades = sorted(market_trades, key=lambda x: x.get('timestamp', 0))

    # Calculate YES ratio
    yes_trades = sum(1 for t in market_trades if t.get('taker_side') == 'yes')
    yes_ratio = yes_trades / len(market_trades)

    if yes_ratio <= yes_threshold:
        return {
            'triggered': False,
            'yes_ratio': yes_ratio,
            'reason': f'yes_ratio_too_low_{yes_ratio:.2f}'
        }

    # Calculate price movement
    first_yes_price = market_trades[0].get('yes_price', 50)
    last_yes_price = market_trades[-1].get('yes_price', 50)
    price_drop = first_yes_price - last_yes_price

    if price_drop < min_price_drop:
        return {
            'triggered': False,
            'yes_ratio': yes_ratio,
            'price_drop': price_drop,
            'reason': f'price_drop_too_small_{price_drop}'
        }

    return {
        'triggered': True,
        'yes_ratio': yes_ratio,
        'price_drop': price_drop,
        'n_trades': len(market_trades),
        'reason': f'rlm_signal_yes_{yes_ratio:.0%}_drop_{price_drop}c'
    }
```

## Appendix B: TradesClient Trade Data Format

From `trades_client.py`:
```python
# Trade message format from Kalshi:
{
    "type": "trade",
    "msg": {
        "market_ticker": "TICKER",
        "yes_price": 65,
        "no_price": 35,
        "count": 100,
        "taker_side": "yes" or "no",
        "ts": 1703700000 (seconds or milliseconds)
    }
}
```

Normalized by TradesClient to:
```python
trade_data = {
    "market_ticker": market_ticker,
    "yes_price": msg_data.get("yes_price", 0),
    "no_price": msg_data.get("no_price", 0),
    "count": msg_data.get("count", 0),
    "taker_side": msg_data.get("taker_side", "unknown"),
    "timestamp_ms": timestamp_ms,
}
```

## Appendix C: PublicTradeEvent Data Structure

From `event_bus.py`:
```python
@dataclass
class PublicTradeEvent:
    """
    Event data for public trades from Kalshi.
    """
    event_type: EventType = EventType.PUBLIC_TRADE_RECEIVED
    market_ticker: str = ""
    timestamp_ms: int = 0
    side: str = ""              # "yes" or "no"
    price_cents: int = 0        # Price for the taker_side
    count: int = 0              # Number of contracts
    received_at: float = 0.0
```

**Note**: The `price_cents` is the price for the `side` that was taken. So:
- If `side == "yes"`, `price_cents` is the YES price
- If `side == "no"`, `price_cents` is the NO price

To get YES price consistently:
```python
yes_price = event.price_cents if event.side == "yes" else (100 - event.price_cents)
```

---

## Implementation Summary (Completed 2025-12-30)

### User Configuration Decisions

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Position size | 100 contracts | Paper trading, can be aggressive |
| Re-entry | Allow on stronger signal | Can add to position if signal strengthens |
| Min price drop | 0c (any drop) | Configurable, start with base params |
| Max concurrent | 1000 positions | High limit for paper, configurable |

### Orderbook Integration Strategy

**Trades-Primary Signal + Orderbook for Execution**

1. **Signal Detection**: TRADES ONLY (Validated +17.38% edge)
   - Keep validated signal detection from public trades
   - No orderbook influence on signal trigger

2. **Execution**: USE ORDERBOOK (With Fallback)
   - Check NO spread before placing order
   - If NO spread <= 2c: Market order OK
   - If NO spread > 3c: Limit order at mid-price
   - **FALLBACK**: If orderbook unhealthy (503, timeout), use market order

3. **Logging**: CAPTURE FOR ANALYSIS (Phase 2)
   - Log orderbook state at signal time
   - Fields: `no_spread`, `yes_spread`, `top_no_bid_size`, `no_bid_volume`

### Configuration Parameters

From `environment.py`:
```python
# RLM Strategy Configuration
rlm_yes_threshold: float = 0.65      # >65% YES trades required
rlm_min_trades: int = 15             # Minimum trades before evaluating
rlm_min_price_drop: int = 0          # Any YES price drop (configurable)
rlm_contracts: int = 100             # Contracts per trade
rlm_max_concurrent: int = 1000       # Max concurrent positions
rlm_allow_reentry: bool = True       # Allow adding on stronger signal
rlm_orderbook_timeout: float = 2.0   # Orderbook fetch timeout (seconds)
rlm_tight_spread: int = 2            # Spread for market order (cents)
rlm_wide_spread: int = 3             # Spread threshold for limit order (cents)
```

### Files Created/Modified

| File | Status | Changes |
|------|--------|---------|
| `services/rlm_service.py` | ✅ Created | Full RLMService with trade accumulation, signal detection, orderbook-aware execution |
| `services/trading_decision_service.py` | ✅ Modified | Added `TradingStrategy.RLM_NO` enum |
| `config/environment.py` | ✅ Modified | Added all RLM config parameters |
| `core/coordinator.py` | ✅ Modified | RLMService initialization in lifecycle mode |
| `core/health_monitor.py` | ✅ Modified | Added RLM health tracking |
| `services/__init__.py` | ✅ Modified | Exported RLMService |

### Key Implementation Notes

1. **Lifecycle Mode Required**: RLMService is only initialized when `market_mode == "lifecycle"`
   - Requires `TrackedMarketsState` for market filtering
   - Uses `EventLifecycleService` category filtering

2. **Re-entry Logic**: Allows adding to existing position when:
   - Signal strengthens (higher YES ratio OR larger price drop)
   - Position not at max concurrent limit

3. **Rate Limiting**: Token bucket (10 trades/minute)
   - Prevents burst execution
   - Tracks rate-limited signals in stats

4. **Decision History**: Maintains last 100 decisions
   - Exposed via WebSocket for frontend console
   - Includes signal data, action taken, order ID

### Running the RLM Strategy

```bash
# Start V3 trader with RLM strategy in lifecycle mode
V3_TRADING_STRATEGY=rlm_no V3_MARKET_MODE=lifecycle ./scripts/run-v3.sh paper

# Or update .env.paper:
V3_TRADING_STRATEGY=rlm_no
V3_MARKET_MODE=lifecycle
```

### Next Steps (Phase 2)

1. **Candlesticks Integration**: Add to `TrackedMarketsSyncer` for historical price context
2. **Orderbook Signal Research**: Analyze logged orderbook data for secondary signals
3. **Trade Persistence Review**: Leverage existing trade storage in Kalshi Flow backend
