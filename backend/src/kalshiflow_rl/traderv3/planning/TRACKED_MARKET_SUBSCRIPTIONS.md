# Tracked Market Subscription Architecture

> Machine-readable architecture reference for coding agents.
> Last updated: 2024-12-30

## Core Principle

**TrackedMarketsState is THE source of truth for ALL subscriptions.**

We will NOT have any subscriptions to non-tracked markets in:
- Orderbook (OrderbookClient)
- Market Ticker (MarketTickerListener)
- Public Trades (when we change from firehose approach)

## Subscription Model

```
STARTUP:
├── Load all active tracked markets from DB
└── For EACH active market:
    ├── Subscribe orderbook ✅ (implemented)
    └── Subscribe ticker (TO ADD)

DURING SESSION:
├── Market discovered (lifecycle event) →
│   ├── Add to TrackedMarketsState
│   ├── Subscribe orderbook ✅ (implemented)
│   └── Subscribe ticker (TO ADD)
│
└── Market determined (lifecycle event) →
    ├── Update status in TrackedMarketsState
    ├── Unsubscribe orderbook ✅ (implemented)
    └── Unsubscribe ticker (TO ADD)
```

## Data Sources Per Tracked Market

| Data Type | Source | Subscription Trigger | Status |
|-----------|--------|---------------------|--------|
| Orderbook levels | OrderbookClient | TrackedMarket enter/exit | ✅ Implemented |
| Real-time prices | MarketTickerListener | TrackedMarket enter/exit | TO ADD |
| Trade activity | TradesClient (filtered) | RLMService filters for tracked | Future (RLM) |
| Position updates | PositionListener | TrackedMarket enter/exit | Future |
| Order fills | FillListener | TrackedMarket enter/exit | Future |

## Why This Design?

### Problem: Fragmented Subscription Management
Previously, subscriptions could come from multiple independent sources:
- Positions wanting ticker updates
- Lifecycle discovery wanting orderbook
- Watchlists wanting prices
- etc.

This leads to:
- Race conditions in subscribe/unsubscribe
- Memory leaks from orphaned subscriptions
- Confusion about who owns what

### Solution: Single Source of Truth
TrackedMarketsState owns all subscription decisions:
- When market enters tracked state → subscribe to ALL relevant channels
- When market exits tracked state → unsubscribe from ALL channels
- No other component manages subscriptions independently

### Implication for Positions
Positions only exist in tracked markets. You cannot have a position in a market you're not tracking. Therefore:
- Position tickers ⊆ Tracked market tickers
- No separate "position subscription" logic needed
- Positions just read from already-subscribed data

## Implementation Details

### Current State (Orderbook - Working)

```python
# In EventLifecycleService
self._on_subscribe: Callable[[str], Awaitable[bool]]  # Called on market tracked
self._on_unsubscribe: Callable[[str], Awaitable[bool]]  # Called on market determined

# Wired in Coordinator
self._event_lifecycle_service.set_subscribe_callback(
    self._orderbook_integration.subscribe_market
)
self._event_lifecycle_service.set_unsubscribe_callback(
    self._orderbook_integration.unsubscribe_market
)
```

### To Add (Ticker)

Option A: Chain callbacks
```python
async def subscribe_all(ticker: str) -> bool:
    await self._orderbook_integration.subscribe_market(ticker)
    await self._ticker_listener.subscribe(ticker)
    return True

self._event_lifecycle_service.set_subscribe_callback(subscribe_all)
```

Option B: Multiple callbacks in EventLifecycleService
```python
# Add list of callbacks instead of single callback
self._on_subscribe_callbacks: List[Callable] = []

async def _notify_subscribe(self, ticker: str):
    for callback in self._on_subscribe_callbacks:
        await callback(ticker)
```

### Startup Recovery

```python
# In Coordinator._connect_lifecycle()
# After loading from DB:
for ticker in recovered_tickers:
    await self._orderbook_integration.subscribe_market(ticker)  # ✅ exists
    await self._ticker_listener.subscribe(ticker)  # TO ADD
```

## RLM Integration (Future)

RLMService will own per-market trade state:

```python
@dataclass
class MarketTradeState:
    """Per-market trade accumulation for RLM signal detection."""
    market_ticker: str
    yes_trades: int = 0
    no_trades: int = 0
    first_yes_price: Optional[int] = None
    last_yes_price: Optional[int] = None
    # ... etc
```

RLMService will:
1. Subscribe to PUBLIC_TRADE_RECEIVED from EventBus
2. Filter for tracked markets using `TrackedMarketsState.is_tracked(ticker)`
3. Accumulate per-market trade state
4. Emit signals when RLM conditions met

This keeps trade processing in RLMService (strategy logic) rather than in subscription management.

## Files to Modify

| File | Change |
|------|--------|
| `core/coordinator.py` | Add ticker subscription in startup recovery loop |
| `services/event_lifecycle_service.py` | Add ticker subscribe/unsubscribe callbacks (or use chained callback) |

## Not Creating

- ~~TrackedMarketDataAggregator~~ - Subscriptions driven by TrackedMarkets directly
- ~~TickerSubscriptionRegistry~~ - Single source of truth eliminates need for ref counting
- ~~Separate position ticker logic~~ - Positions are subset of tracked markets
