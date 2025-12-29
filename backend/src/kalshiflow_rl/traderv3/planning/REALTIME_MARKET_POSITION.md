# Real-Time Market Position Price Tracking

**Status**: Planned
**Created**: 2024-12-28
**Author**: Claude + Sam

## Overview

Track actual market prices (bid/ask/last) for positions in real-time by listening to the Kalshi market ticker WebSocket channel. This provides visibility into current market prices independent of position cost basis.

## Problem Statement

Currently, V3 trader tracks:
- `total_traded` (cost basis from REST API)
- `market_exposure` (position value from WebSocket `market_positions`)

Missing:
- Current market bid/ask prices for positions
- Real-time price updates as market moves
- Ability to see spread and liquidity

## Solution Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           V3 Coordinator                                │
│                                                                         │
│  ┌───────────────┐   ┌─────────────────┐   ┌──────────────────────┐   │
│  │ Position      │   │ MarketTicker    │   │ State Container      │   │
│  │ Listener      │   │ Listener (NEW)  │   │                      │   │
│  │               │   │                 │   │ positions: {}        │   │
│  │ market_       │   │ ticker          │   │ market_prices: {} ◄──┼───┤
│  │ positions     │   │ channel         │   │                      │   │
│  └───────┬───────┘   └────────┬────────┘   └──────────────────────┘   │
│          │                    │                        │               │
│          │                    │                        │               │
│          └────────────────────┼────────────────────────┘               │
│                               │                                         │
│                          EventBus                                       │
│                               │                                         │
│                    ┌──────────▼──────────┐                             │
│                    │  WebSocketManager   │                             │
│                    │  → Frontend         │                             │
│                    └─────────────────────┘                             │
└─────────────────────────────────────────────────────────────────────────┘
```

## Kalshi Market Ticker WebSocket

**Documentation**: https://docs.kalshi.com/websockets/market-ticker

### Subscription Format
```json
{
  "id": 1,
  "cmd": "subscribe",
  "params": {
    "channels": ["ticker"],
    "market_tickers": ["INXD-25JAN03", "AAPL-25JAN03"]
  }
}
```

### Message Format
```json
{
  "type": "ticker",
  "sid": 1,
  "msg": {
    "market_ticker": "INXD-25JAN03",
    "price": 52,                    // last traded price (cents)
    "yes_bid": 50,
    "yes_ask": 54,
    "no_bid": 46,
    "no_ask": 50,
    "volume": 1500,
    "open_interest": 12000,
    "ts": 1703808000
  }
}
```

### Key Points
- Can filter to specific tickers (not forced to receive all)
- Updates sent when any ticker field changes
- Authentication required

## Implementation Details

### 1. MarketTickerListener Class

```python
# backend/src/kalshiflow_rl/traderv3/clients/market_ticker_listener.py

class MarketTickerListener:
    """
    WebSocket listener for Kalshi market ticker updates.

    Subscribes to ticker channel for position tickers and emits
    price updates via event bus for state container integration.

    Key Features:
    - Dynamic subscription management (add/remove tickers)
    - Throttled updates (configurable, default 500ms)
    - Automatic reconnection
    - Metrics tracking
    """

    def __init__(
        self,
        event_bus: "EventBus",
        ws_url: Optional[str] = None,
        throttle_ms: int = 500,
    ):
        self._event_bus = event_bus
        self._subscribed_tickers: Set[str] = set()
        self._last_update_time: Dict[str, float] = {}
        self._throttle_seconds = throttle_ms / 1000

    async def update_subscriptions(self, tickers: List[str]) -> None:
        """
        Update subscribed tickers.

        Computes diff and sends subscribe/unsubscribe commands as needed.
        Called when positions change.
        """
        new_tickers = set(tickers)
        to_add = new_tickers - self._subscribed_tickers
        to_remove = self._subscribed_tickers - new_tickers

        if to_add:
            await self._subscribe_tickers(list(to_add))
        if to_remove:
            await self._unsubscribe_tickers(list(to_remove))

        self._subscribed_tickers = new_tickers
```

### 2. State Container Updates

```python
# backend/src/kalshiflow_rl/traderv3/core/state_container.py

@dataclass
class MarketPriceData:
    """Market price snapshot for a single ticker."""
    ticker: str
    last_price: int      # cents (1-99)
    yes_bid: int         # cents
    yes_ask: int         # cents
    volume: int
    open_interest: int
    timestamp: float     # unix timestamp

class V3StateContainer:
    def __init__(self):
        # ... existing ...

        # Market price data (separate from positions)
        self._market_prices: Dict[str, MarketPriceData] = {}
        self._market_prices_version: int = 0

    def update_market_price(self, ticker: str, price_data: MarketPriceData) -> bool:
        """
        Update market price for a ticker.

        Called by coordinator when receiving ticker events.
        Does NOT modify position data - stored separately.
        """
        self._market_prices[ticker] = price_data
        self._market_prices_version += 1
        self._last_update = time.time()
        return True

    def get_trading_summary(self) -> Dict[str, Any]:
        # ... existing code ...

        # Add market prices to response
        summary["market_prices"] = {
            ticker: {
                "last_price": data.last_price,
                "yes_bid": data.yes_bid,
                "yes_ask": data.yes_ask,
                "spread": data.yes_ask - data.yes_bid,
                "timestamp": data.timestamp,
            }
            for ticker, data in self._market_prices.items()
        }
        summary["market_prices_version"] = self._market_prices_version

        return summary
```

### 3. Coordinator Integration (Minimal)

```python
# backend/src/kalshiflow_rl/traderv3/core/coordinator.py

class V3Coordinator:
    def __init__(self, ...):
        # ... existing ...
        self._market_ticker_listener: Optional[MarketTickerListener] = None

    async def _initialize_components(self) -> None:
        # ... existing ...

        # Market ticker listener for real-time prices
        self._market_ticker_listener = MarketTickerListener(
            event_bus=self._event_bus,
            throttle_ms=500,
        )

    async def _establish_connections(self) -> None:
        # ... existing orderbook, trades, position connections ...

        # Connect market ticker after position listener
        if self._market_ticker_listener:
            await self._connect_market_ticker()

    async def _connect_market_ticker(self) -> None:
        """Connect market ticker listener for position tickers."""
        try:
            # Get current position tickers
            tickers = []
            if self._state_container.trading_state:
                tickers = list(self._state_container.trading_state.positions.keys())

            # Start listener and subscribe
            await self._market_ticker_listener.start()
            if tickers:
                await self._market_ticker_listener.update_subscriptions(tickers)

            # Subscribe to position changes to update ticker subscriptions
            self._event_bus.subscribe(
                EventType.MARKET_POSITION_UPDATE,
                self._handle_position_change_for_ticker
            )

        except Exception as e:
            logger.warning(f"Market ticker listener failed: {e}")
            self._state_container.set_component_degraded("market_ticker", True, str(e))

    async def _handle_position_change_for_ticker(self, event) -> None:
        """Update ticker subscriptions when positions change."""
        if self._market_ticker_listener and self._state_container.trading_state:
            tickers = list(self._state_container.trading_state.positions.keys())
            await self._market_ticker_listener.update_subscriptions(tickers)
```

### 4. EventBus Updates

```python
# backend/src/kalshiflow_rl/traderv3/core/event_bus.py

class EventType(Enum):
    # ... existing ...
    MARKET_TICKER_UPDATE = "market_ticker_update"

class EventBus:
    async def emit_market_ticker_update(
        self,
        ticker: str,
        price_data: Dict[str, Any],
    ) -> bool:
        """Emit market ticker price update."""
        event = MarketEvent(
            event_type=EventType.MARKET_TICKER_UPDATE,
            market_ticker=ticker,
            sequence_number=0,
            timestamp_ms=int(time.time() * 1000),
            metadata={"price_data": price_data}
        )
        return await self._queue_event(event)
```

### 5. Frontend Display

```javascript
// frontend/src/hooks/useThrottledValue.js

import { useState, useEffect, useRef } from 'react';

/**
 * Throttle value updates to prevent UI chaos.
 * Max 1 update per throttleMs.
 */
export function useThrottledValue(value, throttleMs = 500) {
  const [throttledValue, setThrottledValue] = useState(value);
  const lastUpdateRef = useRef(0);
  const pendingValueRef = useRef(value);
  const timeoutRef = useRef(null);

  useEffect(() => {
    pendingValueRef.current = value;
    const now = Date.now();
    const timeSinceLastUpdate = now - lastUpdateRef.current;

    if (timeSinceLastUpdate >= throttleMs) {
      // Enough time passed, update immediately
      setThrottledValue(value);
      lastUpdateRef.current = now;
    } else {
      // Schedule update for remaining time
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
      timeoutRef.current = setTimeout(() => {
        setThrottledValue(pendingValueRef.current);
        lastUpdateRef.current = Date.now();
      }, throttleMs - timeSinceLastUpdate);
    }

    return () => {
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
    };
  }, [value, throttleMs]);

  return throttledValue;
}
```

```javascript
// In positions display component

const MarketPriceDisplay = ({ ticker, marketPrices }) => {
  const priceData = marketPrices?.[ticker];
  const lastPrice = useThrottledValue(priceData?.last_price ?? 0, 500);
  const priceFlash = useValueChangeFlash(lastPrice, { version: priceData?.timestamp });

  if (!priceData) return null;

  return (
    <div className={`text-xs font-mono ${getFlashClass(priceFlash)}`}>
      <span className="text-gray-400">MKT:</span>
      <span className="text-cyan-400 ml-1">{lastPrice}c</span>
      <span className="text-gray-600 ml-2">
        {priceData.yes_bid}c / {priceData.yes_ask}c
      </span>
    </div>
  );
};
```

## UX Design

### Position Card Layout
```
┌─────────────────────────────────────────────────────────┐
│ INXD-25JAN03                              YES 100 @ 45c │
│ ──────────────────────────────────────────────────────  │
│ Entry: $45.00           MKT: 52c  (50c / 54c)          │
│ Value: $52.00           Unrealized: +$7.00 (+15.6%)    │
└─────────────────────────────────────────────────────────┘
```

### Animation Rules
1. **Price changes**: Green flash for up, red flash for down (throttled to 500ms)
2. **No animation**: For bid/ask spread changes (too noisy)
3. **Shimmer**: On animated number counting (existing pattern)

### What We Display
- Last traded price (primary)
- Bid/ask spread (secondary, subtle)

### What We Don't Display Inline
- Volume (available but clutters UI)
- Open interest (available but clutters UI)

## Fallback Strategy (Future)

For initial state and data consistency, can optionally fetch via REST API:

**Endpoint**: `GET /markets?tickers=TICKER1,TICKER2`

**When to call**:
1. During Kalshi data sync (initial and periodic)
2. When WebSocket reconnects
3. When position is first opened (before ticker subscription propagates)

**Implementation Note**: Not in MVP. WebSocket provides real-time updates; REST is only needed if we want historical consistency or faster initial state.

## Testing Plan

1. **Unit tests**: MarketTickerListener subscription management
2. **Integration test**: Ticker updates flow to state container
3. **E2E test**: Price updates appear in frontend within throttle window
4. **Edge cases**:
   - Position closed → ticker unsubscribed
   - Position opened → ticker subscribed
   - WebSocket reconnect → subscriptions restored
   - Rapid price changes → throttled correctly

## Rollout

1. Backend: Add MarketTickerListener with feature flag (disabled by default)
2. Backend: Wire into coordinator, test locally
3. Frontend: Add throttle hook and display components
4. Enable and test E2E
5. Remove feature flag

## Open Questions

1. **Q**: Should we calculate unrealized P&L from market prices instead of `market_exposure`?
   **A**: Keep using `market_exposure` for P&L since it's Kalshi's authoritative value. Market prices are informational only.

2. **Q**: How many tickers can we subscribe to?
   **A**: Kalshi docs don't specify limit. Start with position count (typically <20). Monitor for issues.

3. **Q**: Should we persist market prices across sessions?
   **A**: No. Market prices are ephemeral and stale quickly. Fresh subscription on each session.
