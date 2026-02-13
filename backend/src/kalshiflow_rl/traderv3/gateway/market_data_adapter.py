"""MarketDataAdapter - Dict-based interface over KalshiGateway for read-only market data.

Adapts the typed KalshiGateway (which returns Pydantic models) to the dict-based
interface expected by monitor, discovery, and index. This allows hybrid mode to
swap in a production gateway for market data without changing any downstream
consumers.

When hybrid_data_mode is disabled, the coordinator uses the demo client directly
(which already returns dicts) — this adapter is only instantiated in hybrid mode.
"""

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .client import KalshiGateway

logger = logging.getLogger("kalshiflow_rl.traderv3.gateway.market_data_adapter")


class MarketDataAdapter:
    """Adapts KalshiGateway to the dict-based interface used by monitor/discovery.

    The gateway returns Pydantic models for typed endpoints (get_event,
    get_orderbook, etc.) but the monitor and discovery expect raw dicts
    matching the Kalshi REST API response shapes. This adapter converts
    Pydantic models back to dicts so existing consumers work unchanged.
    """

    def __init__(self, gateway: "KalshiGateway"):
        self._gw = gateway

    async def get_events(self, **kwargs) -> Dict[str, Any]:
        """GET /events — returns raw dict (gateway already returns dict)."""
        try:
            return await self._gw.get_events(**kwargs)
        except Exception as e:
            logger.warning(f"[ADAPTER] get_events failed: {e}")
            return {"events": []}

    async def get_event(self, event_ticker: str) -> Dict[str, Any]:
        """GET /events/{event_ticker} — convert Event model to dict.

        The demo client returns {"event_ticker": ..., "markets": [...], ...}
        (event fields + markets merged at top level). We match that shape.
        """
        try:
            event = await self._gw.get_event(event_ticker)
            d = event.model_dump()
            markets = d.pop("markets", [])
            d["markets"] = markets
            return d
        except Exception as e:
            logger.warning(f"[ADAPTER] get_event({event_ticker}) failed: {e}")
            return {}

    async def get_orderbook(self, ticker: str, depth: int = 5) -> Dict[str, Any]:
        """GET /markets/{ticker}/orderbook — convert Orderbook model to dict.

        Returns {"orderbook": {"yes": [...], "no": [...]}} to match demo client.
        """
        try:
            ob = await self._gw.get_orderbook(ticker, depth=depth)
            return {"orderbook": ob.model_dump()}
        except Exception as e:
            logger.warning(f"[ADAPTER] get_orderbook({ticker}) failed: {e}")
            return {"orderbook": {"yes": [], "no": []}}

    async def get_exchange_status(self) -> Dict[str, Any]:
        """GET /exchange/status — convert ExchangeStatus model to dict."""
        try:
            status = await self._gw.get_exchange_status()
            return status.model_dump()
        except Exception as e:
            logger.warning(f"[ADAPTER] get_exchange_status failed: {e}")
            return {"exchange_active": False, "trading_active": False}

    async def get_event_candlesticks(self, **kwargs) -> Dict[str, Any]:
        """GET /series/{series}/events/{event}/candlesticks — already returns dict."""
        return await self._gw.get_event_candlesticks(**kwargs)

    async def get_markets(self, **kwargs) -> Dict[str, Any]:
        """GET /markets — convert List[Market] to dict matching demo client shape."""
        try:
            markets = await self._gw.get_markets(**kwargs)
            return {"markets": [m.model_dump() for m in markets]}
        except Exception as e:
            logger.warning(f"[ADAPTER] get_markets failed: {e}")
            return {"markets": []}
