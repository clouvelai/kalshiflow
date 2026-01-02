# Hybrid Mode Architecture Proposal

**Status**: Draft
**Author**: Claude (Agent)
**Date**: 2024-12-30
**Context**: Production and demo APIs share market tickers but have completely different prices/activity

## Executive Summary

This proposal designs a hybrid architecture for V3 trader that uses:
- **Production API** (`api.elections.kalshi.com`): Market discovery, orderbook WebSocket, trade feed, market data REST
- **Demo API** (`demo-api.kalshi.co`): Order placement, position management, portfolio/balance queries

This enables paper trading with realistic market discovery and live data while keeping execution safely on the demo account.

---

## 1. Current Architecture Overview

### 1.1 Key Components and Their API Dependencies

| Component | File Location | Current API Source | Operations |
|-----------|--------------|-------------------|------------|
| `OrderbookClient` | `data/orderbook_client.py` | `KALSHI_WS_URL` | WebSocket: orderbook snapshots/deltas |
| `LifecycleClient` | `clients/lifecycle_client.py` | `KALSHI_WS_URL` | WebSocket: market_lifecycle_v2 events |
| `TradesClient` | `clients/trades_client.py` | `KALSHI_WS_URL` | WebSocket: public trades stream |
| `KalshiDemoTradingClient` | `clients/demo_client.py` | `KALSHI_API_URL` + `KALSHI_WS_URL` | REST: orders, positions, balance, markets |
| `V3TradingClientIntegration` | `clients/trading_client_integration.py` | Uses demo_client | Wraps demo client for V3 |
| `ApiDiscoverySyncer` | `services/api_discovery_syncer.py` | trading_client.get_open_markets() | REST: market discovery |
| `MarketPriceSyncer` | `services/market_price_syncer.py` | trading_client.get_markets() | REST: market prices |
| `EventLifecycleService` | `services/event_lifecycle_service.py` | trading_client.get_market() | REST: category enrichment |

### 1.2 Current Configuration Flow

```
.env.paper:
  KALSHI_API_URL=https://demo-api.kalshi.co/trade-api/v2
  KALSHI_WS_URL=wss://demo-api.kalshi.co/trade-api/ws/v2
  KALSHI_API_KEY_ID=<demo_key>
  KALSHI_PRIVATE_KEY_CONTENT=<demo_private_key>

V3Config.from_env():
  - Reads single set of credentials
  - All clients use same API endpoints
```

### 1.3 Problem Statement

In paper trading mode:
1. **Demo API lacks real activity**: Synthetic/minimal trading activity, no real whale trades
2. **Demo prices diverge**: Orderbook, prices, and trade flow are completely different from production
3. **Demo discovery is limited**: Fewer events, different market availability, stale lifecycle events

---

## 2. Proposed Architecture

### 2.1 Dual Credential System

Add new environment variables for production data feed:

```bash
# .env.paper (hybrid mode enabled)
ENVIRONMENT=paper

# Primary: Demo API for trading (existing)
KALSHI_API_URL=https://demo-api.kalshi.co/trade-api/v2
KALSHI_WS_URL=wss://demo-api.kalshi.co/trade-api/ws/v2
KALSHI_API_KEY_ID=<demo_key>
KALSHI_PRIVATE_KEY_CONTENT=<demo_private_key>

# Secondary: Production API for data (NEW)
KALSHI_DATA_API_URL=https://api.elections.kalshi.com/trade-api/v2
KALSHI_DATA_WS_URL=wss://api.elections.kalshi.com/trade-api/ws/v2
KALSHI_DATA_API_KEY_ID=<production_key>
KALSHI_DATA_PRIVATE_KEY_CONTENT=<production_private_key>

# Hybrid mode flag (NEW)
V3_HYBRID_MODE=true
```

### 2.2 Client Separation

```
Production API (read-only data):
  - OrderbookClient -> Real orderbook snapshots/deltas
  - LifecycleClient -> Real market lifecycle events
  - TradesClient -> Real public trades (whale detection)
  - KalshiDataClient (NEW) -> Market discovery, prices, event lookup

Demo API (trading operations):
  - KalshiDemoTradingClient -> Order placement, position queries
  - Fill notifications -> Demo fills
  - Balance/portfolio -> Demo account
```

---

## 3. File Changes

### 3.1 New Files

| File | Purpose |
|------|---------|
| `traderv3/clients/data_auth.py` | Auth provider for production data API |
| `traderv3/clients/data_client.py` | Read-only REST client for market data |

### 3.2 Modified Files

| File | Changes |
|------|---------|
| `traderv3/config/environment.py` | Add hybrid mode config, data API credentials |
| `traderv3/core/coordinator.py` | Create data client, pass to WebSocket clients and services |
| `traderv3/clients/orderbook_integration.py` | Accept custom auth/URL for hybrid mode |
| `traderv3/clients/lifecycle_integration.py` | Accept custom auth/URL for hybrid mode |
| `traderv3/clients/trades_integration.py` | Accept custom auth/URL for hybrid mode |
| `traderv3/services/api_discovery_syncer.py` | Use data_client if available |
| `traderv3/services/market_price_syncer.py` | Use data_client if available |
| `traderv3/services/event_lifecycle_service.py` | Use data_client for category enrichment |

---

## 4. Detailed Design

### 4.1 V3Config Updates

```python
# In config/environment.py

@dataclass
class V3Config:
    # Existing fields...
    api_url: str
    ws_url: str
    api_key_id: str
    private_key_content: str

    # NEW: Data feed configuration (production API)
    data_api_url: Optional[str] = None
    data_ws_url: Optional[str] = None
    data_api_key_id: Optional[str] = None
    data_private_key_content: Optional[str] = None

    # NEW: Enable hybrid mode
    hybrid_mode: bool = False

    @classmethod
    def from_env(cls) -> "V3Config":
        # ... existing credential loading ...

        # Hybrid mode configuration
        hybrid_mode = os.environ.get("V3_HYBRID_MODE", "false").lower() == "true"

        data_api_url = None
        data_ws_url = None
        data_api_key_id = None
        data_private_key_content = None

        if hybrid_mode:
            data_api_url = os.environ.get("KALSHI_DATA_API_URL")
            data_ws_url = os.environ.get("KALSHI_DATA_WS_URL")
            data_api_key_id = os.environ.get("KALSHI_DATA_API_KEY_ID")
            data_private_key_content = os.environ.get("KALSHI_DATA_PRIVATE_KEY_CONTENT")

            if not all([data_api_url, data_ws_url, data_api_key_id, data_private_key_content]):
                raise ValueError(
                    "V3_HYBRID_MODE=true requires all KALSHI_DATA_* environment variables: "
                    "KALSHI_DATA_API_URL, KALSHI_DATA_WS_URL, KALSHI_DATA_API_KEY_ID, "
                    "KALSHI_DATA_PRIVATE_KEY_CONTENT"
                )

            logger.info("Hybrid mode enabled:")
            logger.info(f"  - Trading API: {api_url}")
            logger.info(f"  - Data API: {data_api_url}")
```

### 4.2 KalshiDataClient (New File)

```python
# New file: traderv3/clients/data_client.py

"""
Kalshi Data Client for Production Market Data.

Read-only API client for fetching market data from production Kalshi API.
Used in hybrid mode where data comes from production but trading goes to demo.

Key Responsibilities:
    1. **Market Discovery** - Fetch open markets and events
    2. **Price Queries** - Get current market prices for P&L
    3. **Category Enrichment** - Look up event categories

Architecture Position:
    Used by services that need market data but not trading:
    - ApiDiscoverySyncer: Market discovery
    - MarketPriceSyncer: Price updates
    - EventLifecycleService: Category lookup

Design Principles:
    - **Read-only**: No order placement or position modification
    - **Async-first**: All methods are async
    - **Error isolation**: Individual request errors don't break the client
"""

import asyncio
import logging
import json
import tempfile
import os
from typing import Dict, List, Optional, Any

import aiohttp
from kalshiflow.auth import KalshiAuth

logger = logging.getLogger("kalshiflow_rl.traderv3.clients.data_client")


class KalshiDataClient:
    """
    Read-only Kalshi API client for market data operations.

    Provides market discovery, price queries, and event lookup
    without trading capabilities.
    """

    def __init__(
        self,
        api_url: str,
        api_key_id: str,
        private_key_content: str,
    ):
        """
        Initialize data client.

        Args:
            api_url: Production API URL
            api_key_id: Production API key ID
            private_key_content: Production private key content
        """
        self.api_url = api_url
        self.api_key_id = api_key_id
        self._temp_key_file: Optional[str] = None
        self._auth: Optional[KalshiAuth] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._is_connected = False

        # Setup auth
        self._setup_auth(private_key_content)

        logger.info(f"KalshiDataClient initialized for {api_url}")

    def _setup_auth(self, private_key_content: str) -> None:
        """Setup authentication using same pattern as demo_client."""
        temp_fd, temp_path = tempfile.mkstemp(suffix='.pem', prefix='kalshi_data_key_')
        self._temp_key_file = temp_path

        with os.fdopen(temp_fd, 'w') as f:
            # Handle PEM format normalization
            if not private_key_content.startswith('-----BEGIN'):
                formatted = f"-----BEGIN PRIVATE KEY-----\n{private_key_content}\n-----END PRIVATE KEY-----"
            else:
                formatted = private_key_content.replace('\\n', '\n')
            f.write(formatted)

        self._auth = KalshiAuth(
            api_key_id=self.api_key_id,
            private_key_path=self._temp_key_file
        )

    async def connect(self) -> None:
        """Connect to the API."""
        self._session = aiohttp.ClientSession()

        # Test connection
        try:
            await self.get_markets(limit=1)
            self._is_connected = True
            logger.info("KalshiDataClient connected successfully")
        except Exception as e:
            await self._session.close()
            self._session = None
            raise RuntimeError(f"Failed to connect data client: {e}")

    async def disconnect(self) -> None:
        """Disconnect and cleanup."""
        if self._session:
            await self._session.close()
            self._session = None

        if self._temp_key_file:
            try:
                os.unlink(self._temp_key_file)
            except:
                pass

        self._is_connected = False
        logger.info("KalshiDataClient disconnected")

    def _create_auth_headers(self, method: str, path: str) -> Dict[str, str]:
        """Create authentication headers."""
        full_path = f"/trade-api/v2{path}"
        headers = self._auth.create_auth_headers(method, full_path)
        headers['Content-Type'] = 'application/json'
        return headers

    async def _make_request(self, method: str, path: str) -> Dict[str, Any]:
        """Make authenticated API request."""
        if not self._session:
            raise RuntimeError("Data client not connected")

        url = f"{self.api_url}{path}"
        headers = self._create_auth_headers(method, path)

        async with self._session.request(method, url, headers=headers) as response:
            text = await response.text()

            if response.status >= 400:
                raise RuntimeError(f"Data API error {response.status}: {text}")

            return json.loads(text) if text else {}

    # Market Data Methods (same interface as demo_client for compatibility)

    async def get_markets(
        self,
        limit: int = 100,
        tickers: Optional[List[str]] = None,
        status: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get markets with optional filtering."""
        params = [f"limit={limit}"]
        if tickers:
            params.append(f"tickers={','.join(tickers)}")
        if status:
            params.append(f"status={status}")

        query = "&".join(params)
        return await self._make_request("GET", f"/markets?{query}")

    async def get_market(self, ticker: str) -> Dict[str, Any]:
        """Get single market by ticker."""
        return await self._make_request("GET", f"/markets/{ticker}")

    async def get_event(self, event_ticker: str) -> Dict[str, Any]:
        """Get event for category enrichment."""
        response = await self._make_request("GET", f"/events/{event_ticker}")
        return response.get("event", {})

    async def get_events(
        self,
        status: Optional[str] = None,
        with_nested_markets: bool = False,
        limit: int = 200,
        cursor: Optional[str] = None,
        min_close_ts: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Get events with pagination for market discovery."""
        params = [f"limit={limit}"]
        if status:
            params.append(f"status={status}")
        if with_nested_markets:
            params.append("with_nested_markets=true")
        if cursor:
            params.append(f"cursor={cursor}")
        if min_close_ts:
            params.append(f"min_close_ts={min_close_ts}")

        query = "&".join(params)
        return await self._make_request("GET", f"/events?{query}")

    @property
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._is_connected
```

### 4.3 Coordinator Updates

```python
# In core/coordinator.py

class V3Coordinator:
    def __init__(self, ...):
        # Existing fields...

        # Hybrid mode: separate data client
        self._data_client: Optional['KalshiDataClient'] = None

    async def _establish_connections(self) -> None:
        """Establish all external connections."""
        # Create data client for hybrid mode FIRST
        if self._config.hybrid_mode:
            await self._create_data_client()

        # Orderbook connection (uses data URLs in hybrid mode)
        await self._connect_orderbook()

        # ... rest of connections ...

    async def _create_data_client(self) -> None:
        """Create production data client for hybrid mode."""
        from ..clients.data_client import KalshiDataClient

        logger.info("Creating production data client for hybrid mode...")

        self._data_client = KalshiDataClient(
            api_url=self._config.data_api_url,
            api_key_id=self._config.data_api_key_id,
            private_key_content=self._config.data_private_key_content,
        )
        await self._data_client.connect()

        logger.info("Production data client connected")

        await self._event_bus.emit_system_activity(
            activity_type="connection",
            message="Hybrid mode: Production data client connected",
            metadata={
                "api_url": self._config.data_api_url,
                "mode": "hybrid",
                "severity": "info"
            }
        )

    async def _start_api_discovery(self) -> None:
        """Start API discovery syncer."""
        # ... existing checks ...

        # Use data_client in hybrid mode, else trading_client
        api_client = self._data_client if self._config.hybrid_mode else None

        self._api_discovery_syncer = ApiDiscoverySyncer(
            data_client=api_client,  # NEW: production data
            trading_client=self._trading_client_integration,  # Fallback
            # ... rest of args ...
        )
```

### 4.4 OrderbookClient Updates

The OrderbookClient needs to accept custom WebSocket URL and auth:

```python
# In data/orderbook_client.py __init__:

def __init__(
    self,
    market_tickers: Optional[List[str]] = None,
    stats_collector=None,
    event_bus: Optional[Any] = None,
    ws_url: Optional[str] = None,  # NEW: Custom WebSocket URL
    auth: Optional[KalshiAuth] = None,  # NEW: Custom auth
):
    # Use custom or default
    self.ws_url = ws_url or config.KALSHI_WS_URL
    self._custom_auth = auth  # Store for connection
```

Then in the coordinator, when creating OrderbookClient for hybrid mode:

```python
# If hybrid mode, use production WebSocket
if self._config.hybrid_mode:
    from kalshiflow.auth import KalshiAuth
    from ..clients.data_auth import create_data_auth

    orderbook_client = OrderbookClient(
        market_tickers=market_tickers,
        event_bus=self._event_bus,
        ws_url=self._config.data_ws_url,
        auth=create_data_auth(
            self._config.data_api_key_id,
            self._config.data_private_key_content
        ),
    )
```

---

## 5. Risk Considerations

### 5.1 Price Mismatch (HIGH RISK)

**Risk**: Production orderbook prices differ from demo execution prices

**Impact**:
- Orders placed at production best bid/ask may not fill on demo
- P&L calculations based on production prices won't match demo fills
- Strategy signals based on production may not be valid on demo

**Mitigation**:
- Log clear warnings when hybrid mode is active
- Track "expected" (production) vs "actual" (demo) fill prices
- Use limit orders with wider spreads to increase fill probability
- Accept that paper trading P&L is approximate in hybrid mode

### 5.2 Market Availability Mismatch (MEDIUM RISK)

**Risk**: Markets discovered in production may not exist in demo

**Impact**:
- Order placement fails for production-discovered markets
- Lifecycle events for markets that don't exist in demo

**Mitigation**:
- Validate market exists in demo before order placement
- Gracefully handle "market not found" errors
- Log market availability mismatches for analysis

### 5.3 Authentication Complexity (LOW RISK)

**Risk**: Two sets of credentials to manage

**Impact**:
- Configuration errors more likely
- Credential rotation more complex
- Environment variable proliferation

**Mitigation**:
- Clear naming convention (KALSHI_* vs KALSHI_DATA_*)
- Validation on startup with helpful error messages
- Document both credential sets in .env.paper.example

### 5.4 Position State Consistency (NO RISK)

**Risk**: None - positions are correctly tracked in demo only

The trading client (demo) is the source of truth for:
- Account balance
- Open positions
- Open orders
- Portfolio value

Production data is only used for:
- Market discovery
- Orderbook data
- Price signals

---

## 6. Configuration Examples

### 6.1 Non-Hybrid Mode (Current Behavior)

```bash
ENVIRONMENT=paper
KALSHI_API_URL=https://demo-api.kalshi.co/trade-api/v2
KALSHI_WS_URL=wss://demo-api.kalshi.co/trade-api/ws/v2
KALSHI_API_KEY_ID=<demo_key>
KALSHI_PRIVATE_KEY_CONTENT=<demo_private_key>
# V3_HYBRID_MODE not set or false
# Uses demo for everything
```

### 6.2 Hybrid Mode (Production Data + Demo Trading)

```bash
ENVIRONMENT=paper
V3_HYBRID_MODE=true

# Trading credentials (demo) - for order execution
KALSHI_API_URL=https://demo-api.kalshi.co/trade-api/v2
KALSHI_WS_URL=wss://demo-api.kalshi.co/trade-api/ws/v2
KALSHI_API_KEY_ID=<demo_api_key>
KALSHI_PRIVATE_KEY_CONTENT=<demo_private_key>

# Data credentials (production) - for market data
KALSHI_DATA_API_URL=https://api.elections.kalshi.com/trade-api/v2
KALSHI_DATA_WS_URL=wss://api.elections.kalshi.com/trade-api/ws/v2
KALSHI_DATA_API_KEY_ID=<production_api_key>
KALSHI_DATA_PRIVATE_KEY_CONTENT=<production_private_key>
```

---

## 7. Implementation Plan

### Phase 1: Configuration (1 day)
- [ ] Add hybrid mode config to `environment.py`
- [ ] Add validation for data credentials
- [ ] Update `.env.paper.example` with new variables

### Phase 2: DataClient (1 day)
- [ ] Create `clients/data_client.py`
- [ ] Create `clients/data_auth.py` (optional, could inline)
- [ ] Add connect/disconnect lifecycle

### Phase 3: WebSocket Clients (1 day)
- [ ] Update `OrderbookClient` to accept custom auth/URL
- [ ] Update `LifecycleClient` to accept custom auth/URL
- [ ] Update `TradesClient` to accept custom auth/URL

### Phase 4: Services (1 day)
- [ ] Update `ApiDiscoverySyncer` to use data_client
- [ ] Update `MarketPriceSyncer` to use data_client
- [ ] Update `EventLifecycleService` to use data_client

### Phase 5: Coordinator (1 day)
- [ ] Add data_client creation in hybrid mode
- [ ] Pass data_client to services
- [ ] Pass custom auth/URLs to WebSocket clients
- [ ] Add hybrid mode logging and system activity

### Phase 6: Testing (1 day)
- [ ] Test non-hybrid mode still works
- [ ] Test hybrid mode with production data
- [ ] Validate orderbook connects to production
- [ ] Validate trading stays on demo

---

## 8. Open Questions

1. **Production trading support?** Should we also support hybrid mode for production trading (production data + production trading)? Currently the demo client blocks production URLs.

2. **Price display source?** Should MarketPriceSyncer use production prices for P&L calculations, or demo prices? Production is more realistic, but demo positions may fill at different prices.

3. **Market validation?** Should we proactively validate that production-discovered markets exist in demo before attempting trades?

---

## Appendix: Current File Locations

```
backend/src/kalshiflow_rl/
  traderv3/
    config/
      environment.py          # V3Config - add hybrid mode fields
    clients/
      demo_client.py          # KalshiDemoTradingClient - trading only
      data_client.py          # NEW: KalshiDataClient - data only
      data_auth.py            # NEW: Production auth helper
      orderbook_integration.py
      lifecycle_client.py
      trades_client.py
    services/
      api_discovery_syncer.py
      market_price_syncer.py
      event_lifecycle_service.py
    core/
      coordinator.py          # Main orchestrator - wire data client
  data/
    orderbook_client.py       # Accept custom auth/URL
```
