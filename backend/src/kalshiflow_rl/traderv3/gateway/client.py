"""KalshiGateway - Unified REST + WebSocket client for Kalshi API.

Single class consolidating all Kalshi communication. Uses httpx for REST,
websockets for WS, Pydantic v2 for typed responses, and aiolimiter for
rate limiting.

Usage:
    gw = KalshiGateway(api_url="https://demo-api.kalshi.co/trade-api/v2",
                       ws_url="wss://demo-api.kalshi.co/trade-api/ws/v2")
    await gw.connect()
    balance = await gw.get_balance()
    await gw.disconnect()
"""

import asyncio
import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional

import httpx

from .auth import GatewayAuth
from .errors import (
    KalshiAuthError,
    KalshiConnectionError,
    KalshiError,
    KalshiNotFoundError,
    KalshiOrderError,
    KalshiRateLimitError,
)
from .models import (
    Balance,
    Event,
    ExchangeStatus,
    Fill,
    Market,
    Order,
    OrderGroup,
    OrderResponse,
    Orderbook,
    Position,
    QueuePosition,
    Settlement,
)
from .rate_limiter import GatewayRateLimiter
from .ws_multiplexer import WSMultiplexer

logger = logging.getLogger("kalshiflow_rl.traderv3.gateway.client")


class KalshiGateway:
    """Unified Kalshi API client with REST + WebSocket support.

    Provides typed REST methods returning Pydantic models and a single
    multiplexed WebSocket connection for real-time data.
    """

    def __init__(
        self,
        api_url: str,
        ws_url: str,
        rate: float = 10.0,
        burst: int = 20,
    ):
        """
        Args:
            api_url: Kalshi REST API base URL (e.g. "https://demo-api.kalshi.co/trade-api/v2")
            ws_url: Kalshi WebSocket URL (e.g. "wss://demo-api.kalshi.co/trade-api/ws/v2")
            rate: Sustained requests per second.
            burst: Burst capacity for rapid order placement.
        """
        self._api_url = api_url.rstrip("/")
        self._ws_url = ws_url

        self._auth = GatewayAuth()
        self._limiter = GatewayRateLimiter(rate=rate, burst=burst)
        self._client: Optional[httpx.AsyncClient] = None
        self._ws: Optional[WSMultiplexer] = None

        self._connected = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Initialize auth, create HTTP client, verify connectivity."""
        self._auth.setup()
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(15.0))

        # Verify connectivity with exchange status
        try:
            status = await self.get_exchange_status()
            if not status.exchange_active:
                logger.warning("Exchange not active at connect time")
        except Exception as e:
            await self.disconnect()
            raise KalshiConnectionError(f"Failed to connect: {e}")

        self._connected = True
        logger.info("KalshiGateway connected")

    async def disconnect(self) -> None:
        """Close HTTP client, stop WS, cleanup auth."""
        if self._ws:
            await self._ws.stop()
            self._ws = None

        if self._client:
            await self._client.aclose()
            self._client = None

        self._auth.cleanup()
        self._connected = False
        logger.info("KalshiGateway disconnected")

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # WebSocket management
    # ------------------------------------------------------------------

    def get_ws(self) -> WSMultiplexer:
        """Get or create the WebSocket multiplexer.

        The multiplexer is created lazily on first access. Register
        callbacks before calling ws.start().
        """
        if self._ws is None:
            def auth_headers():
                headers = self._auth.rest_headers("GET", "/ws/v2")
                return headers

            self._ws = WSMultiplexer(
                ws_url=self._ws_url,
                auth_headers_fn=auth_headers,
            )
        return self._ws

    # ------------------------------------------------------------------
    # REST helpers
    # ------------------------------------------------------------------

    async def _request(
        self,
        method: str,
        path: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Make an authenticated, rate-limited REST request.

        Args:
            method: HTTP method
            path: API path without base URL (e.g. "/portfolio/balance")
            data: JSON body for POST/PUT/DELETE
            params: Query parameters

        Returns:
            Parsed JSON response dict.

        Raises:
            KalshiError subclass based on status code.
        """
        if not self._client:
            raise KalshiConnectionError("Gateway not connected")

        await self._limiter.acquire()

        # Auth headers use path without query params
        base_path = path.split("?")[0]
        headers = self._auth.rest_headers(method, base_path)

        url = f"{self._api_url}{path}"

        max_attempts = 2  # 1 initial + 1 retry for 502
        for attempt in range(1, max_attempts + 1):
            try:
                response = await self._client.request(
                    method,
                    url,
                    headers=headers,
                    json=data,
                    params=params,
                )

                # Retry once on 502 (common transient demo API error)
                if response.status_code == 502 and attempt < max_attempts:
                    logger.warning(f"502 on {method} {path} (attempt {attempt}), retrying")
                    await asyncio.sleep(1)
                    # Re-generate auth headers (timestamp changes)
                    headers = self._auth.rest_headers(method, base_path)
                    continue

                if response.status_code >= 400:
                    self._raise_for_status(response, method, path)

                return response.json() if response.text else {}

            except httpx.HTTPError as e:
                raise KalshiConnectionError(f"HTTP error on {method} {path}: {e}")

        raise KalshiConnectionError(f"Request failed after {max_attempts} attempts")

    def _raise_for_status(
        self, response: httpx.Response, method: str, path: str
    ) -> None:
        """Map HTTP status codes to KalshiError subtypes."""
        code = response.status_code
        body = response.text[:500]
        msg = f"{method} {path} → {code}: {body}"

        if code == 401 or code == 403:
            raise KalshiAuthError(msg, status_code=code, response_body=body)
        elif code == 404:
            raise KalshiNotFoundError(msg, status_code=code, response_body=body)
        elif code == 429:
            raise KalshiRateLimitError(msg, status_code=code, response_body=body)
        elif 400 <= code < 500:
            raise KalshiOrderError(msg, status_code=code, response_body=body)
        else:
            raise KalshiError(msg, status_code=code, response_body=body)

    # ------------------------------------------------------------------
    # Exchange
    # ------------------------------------------------------------------

    async def get_exchange_status(self) -> ExchangeStatus:
        """GET /exchange/status."""
        data = await self._request("GET", "/exchange/status")
        return ExchangeStatus.model_validate(data)

    # ------------------------------------------------------------------
    # Portfolio: Balance
    # ------------------------------------------------------------------

    async def get_balance(self) -> Balance:
        """GET /portfolio/balance."""
        data = await self._request("GET", "/portfolio/balance")
        return Balance.model_validate(data)

    # ------------------------------------------------------------------
    # Portfolio: Positions
    # ------------------------------------------------------------------

    async def get_positions(self) -> List[Position]:
        """GET /portfolio/positions. Returns list of positions."""
        data = await self._request("GET", "/portfolio/positions")
        raw = data.get("market_positions", data.get("positions", []))
        return [Position.model_validate(p) for p in raw]

    # ------------------------------------------------------------------
    # Portfolio: Orders
    # ------------------------------------------------------------------

    async def create_order(
        self,
        ticker: str,
        action: str,
        side: str,
        count: int,
        price: int,
        type: str = "limit",
        order_group_id: Optional[str] = None,
        expiration_ts: Optional[int] = None,
    ) -> OrderResponse:
        """POST /portfolio/orders. Place a single order.

        Args:
            ticker: Market ticker
            action: "buy" or "sell"
            side: "yes" or "no"
            count: Number of contracts
            price: Limit price in cents (1-99)
            type: "limit" or "market"
            order_group_id: Optional order group for portfolio limits
            expiration_ts: Optional unix timestamp for auto-cancel
        """
        body: Dict[str, Any] = {
            "ticker": ticker,
            "action": action,
            "side": side,
            "count": count,
            "type": type,
        }

        # Kalshi uses side-specific price fields
        if side == "yes":
            body["yes_price"] = price
        else:
            body["no_price"] = price

        if order_group_id:
            body["order_group_id"] = order_group_id
        if expiration_ts is not None:
            body["expiration_ts"] = expiration_ts

        data = await self._request("POST", "/portfolio/orders", data=body)
        return OrderResponse.model_validate(data)

    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """DELETE /portfolio/orders/{order_id}."""
        return await self._request("DELETE", f"/portfolio/orders/{order_id}")

    async def get_orders(
        self,
        ticker: Optional[str] = None,
        status: str = "resting",
    ) -> List[Order]:
        """GET /portfolio/orders with optional filters."""
        path = "/portfolio/orders"
        if ticker:
            path += f"?ticker={ticker}"

        data = await self._request("GET", path)
        all_orders = data.get("orders", [])

        if status:
            all_orders = [o for o in all_orders if o.get("status") == status]

        return [Order.model_validate(o) for o in all_orders]

    async def get_queue_positions(
        self,
        market_tickers: Optional[str] = None,
    ) -> List[QueuePosition]:
        """GET /portfolio/orders/queue_positions."""
        path = "/portfolio/orders/queue_positions"
        if market_tickers:
            path += f"?market_tickers={market_tickers}"

        data = await self._request("GET", path)
        raw = data.get("queue_positions", [])
        return [QueuePosition.model_validate(qp) for qp in raw]

    # ------------------------------------------------------------------
    # Portfolio: Order Groups
    # ------------------------------------------------------------------

    async def create_order_group(self, contracts_limit: int = 10000) -> OrderGroup:
        """POST /portfolio/order_groups."""
        body = {"contracts_limit": contracts_limit}
        data = await self._request("POST", "/portfolio/order_groups", data=body)
        return OrderGroup.model_validate(data)

    async def reset_order_group(self, order_group_id: str) -> Dict[str, Any]:
        """POST /portfolio/order_groups/{id}/reset - cancels all resting orders in group."""
        return await self._request(
            "POST", f"/portfolio/order_groups/{order_group_id}/reset"
        )

    # ------------------------------------------------------------------
    # Portfolio: Fills & Settlements
    # ------------------------------------------------------------------

    async def get_fills(
        self,
        ticker: Optional[str] = None,
        order_group_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Fill]:
        """GET /portfolio/fills."""
        params = []
        if ticker:
            params.append(f"ticker={ticker}")
        if order_group_id:
            params.append(f"order_group_id={order_group_id}")
        if limit != 100:
            params.append(f"limit={limit}")

        path = "/portfolio/fills"
        if params:
            path += "?" + "&".join(params)

        data = await self._request("GET", path)
        raw = data.get("fills", [])
        return [Fill.model_validate(f) for f in raw]

    async def get_settlements(self, limit: int = 200) -> List[Settlement]:
        """GET /portfolio/settlements with pagination."""
        all_settlements = []
        cursor = None

        while len(all_settlements) < limit:
            params = [f"limit={min(200, limit - len(all_settlements))}"]
            if cursor:
                params.append(f"cursor={cursor}")

            path = "/portfolio/settlements?" + "&".join(params)
            data = await self._request("GET", path)

            batch = data.get("settlements", [])
            all_settlements.extend(batch)

            cursor = data.get("cursor")
            if not cursor or not batch:
                break

        return [Settlement.model_validate(s) for s in all_settlements]

    # ------------------------------------------------------------------
    # Markets & Events
    # ------------------------------------------------------------------

    async def get_event(self, event_ticker: str) -> Event:
        """GET /events/{event_ticker} with nested markets."""
        data = await self._request("GET", f"/events/{event_ticker}")
        event_data = data.get("event", {})
        event_data["markets"] = data.get("markets", [])
        return Event.model_validate(event_data)

    async def get_events(
        self,
        status: Optional[str] = None,
        with_nested_markets: bool = False,
        limit: int = 200,
        cursor: Optional[str] = None,
        series_ticker: Optional[str] = None,
    ) -> Dict[str, Any]:
        """GET /events with filters. Returns raw dict with events + cursor."""
        params = [f"limit={limit}"]
        if status:
            params.append(f"status={status}")
        if with_nested_markets:
            params.append("with_nested_markets=true")
        if cursor:
            params.append(f"cursor={cursor}")
        if series_ticker:
            params.append(f"series_ticker={series_ticker}")

        path = "/events?" + "&".join(params)
        return await self._request("GET", path)

    async def get_market(self, ticker: str) -> Market:
        """GET /markets/{ticker}."""
        data = await self._request("GET", f"/markets/{ticker}")
        market_data = data.get("market", data)
        return Market.model_validate(market_data)

    async def get_markets(
        self,
        event_ticker: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[Market]:
        """GET /markets with filters."""
        params = [f"limit={limit}"]
        if event_ticker:
            params.append(f"event_ticker={event_ticker}")
        if status:
            params.append(f"status={status}")

        path = "/markets?" + "&".join(params)
        data = await self._request("GET", path)
        raw = data.get("markets", [])
        return [Market.model_validate(m) for m in raw]

    async def get_orderbook(self, ticker: str, depth: int = 5) -> Orderbook:
        """GET /markets/{ticker}/orderbook."""
        path = f"/markets/{ticker}/orderbook"
        if depth > 0:
            path += f"?depth={depth}"
        data = await self._request("GET", path)
        ob_data = data.get("orderbook", data)
        return Orderbook.model_validate(ob_data)

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def get_health(self) -> Dict[str, Any]:
        """Consolidated health check."""
        health: Dict[str, Any] = {
            "connected": self._connected,
        }
        if self._ws:
            health["ws"] = self._ws.get_health()
        return health
