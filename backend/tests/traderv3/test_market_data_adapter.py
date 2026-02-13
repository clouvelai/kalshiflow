"""Unit tests for MarketDataAdapter error handling.

Tests verify that each method returns safe defaults when the underlying
KalshiGateway raises exceptions, instead of propagating unhandled errors.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from kalshiflow_rl.traderv3.gateway.market_data_adapter import MarketDataAdapter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_adapter(gateway=None):
    """Create a MarketDataAdapter with a mock gateway."""
    gw = gateway or MagicMock()
    return MarketDataAdapter(gw)


# ---------------------------------------------------------------------------
# get_orderbook error handling
# ---------------------------------------------------------------------------

class TestGetOrderbookErrorHandling:
    @pytest.mark.asyncio
    async def test_returns_safe_default_on_exception(self):
        gw = MagicMock()
        gw.get_orderbook = AsyncMock(side_effect=Exception("connection refused"))
        adapter = make_adapter(gw)

        result = await adapter.get_orderbook("TICKER-A")
        assert result == {"orderbook": {"yes": [], "no": []}}

    @pytest.mark.asyncio
    async def test_returns_safe_default_on_timeout(self):
        gw = MagicMock()
        gw.get_orderbook = AsyncMock(side_effect=TimeoutError("timed out"))
        adapter = make_adapter(gw)

        result = await adapter.get_orderbook("TICKER-A", depth=3)
        assert result == {"orderbook": {"yes": [], "no": []}}

    @pytest.mark.asyncio
    async def test_success_returns_normal_data(self):
        mock_ob = MagicMock()
        mock_ob.model_dump.return_value = {"yes": [[50, 10]], "no": [[60, 5]]}

        gw = MagicMock()
        gw.get_orderbook = AsyncMock(return_value=mock_ob)
        adapter = make_adapter(gw)

        result = await adapter.get_orderbook("TICKER-A")
        assert result == {"orderbook": {"yes": [[50, 10]], "no": [[60, 5]]}}


# ---------------------------------------------------------------------------
# get_event error handling
# ---------------------------------------------------------------------------

class TestGetEventErrorHandling:
    @pytest.mark.asyncio
    async def test_returns_empty_dict_on_exception(self):
        gw = MagicMock()
        gw.get_event = AsyncMock(side_effect=Exception("not found"))
        adapter = make_adapter(gw)

        result = await adapter.get_event("EVT-MISSING")
        assert result == {}

    @pytest.mark.asyncio
    async def test_success_returns_event_with_markets(self):
        mock_event = MagicMock()
        mock_event.model_dump.return_value = {
            "event_ticker": "EVT-1",
            "title": "Test Event",
            "markets": [{"ticker": "MKT-A"}],
        }

        gw = MagicMock()
        gw.get_event = AsyncMock(return_value=mock_event)
        adapter = make_adapter(gw)

        result = await adapter.get_event("EVT-1")
        assert result["event_ticker"] == "EVT-1"
        assert "markets" in result


# ---------------------------------------------------------------------------
# get_exchange_status error handling
# ---------------------------------------------------------------------------

class TestGetExchangeStatusErrorHandling:
    @pytest.mark.asyncio
    async def test_returns_inactive_defaults_on_exception(self):
        gw = MagicMock()
        gw.get_exchange_status = AsyncMock(side_effect=Exception("503 Service Unavailable"))
        adapter = make_adapter(gw)

        result = await adapter.get_exchange_status()
        assert result == {"exchange_active": False, "trading_active": False}

    @pytest.mark.asyncio
    async def test_success_returns_normal_status(self):
        mock_status = MagicMock()
        mock_status.model_dump.return_value = {
            "exchange_active": True,
            "trading_active": True,
        }

        gw = MagicMock()
        gw.get_exchange_status = AsyncMock(return_value=mock_status)
        adapter = make_adapter(gw)

        result = await adapter.get_exchange_status()
        assert result["exchange_active"] is True
        assert result["trading_active"] is True


# ---------------------------------------------------------------------------
# get_events error handling
# ---------------------------------------------------------------------------

class TestGetEventsErrorHandling:
    @pytest.mark.asyncio
    async def test_returns_empty_events_on_exception(self):
        gw = MagicMock()
        gw.get_events = AsyncMock(side_effect=Exception("rate limited"))
        adapter = make_adapter(gw)

        result = await adapter.get_events(limit=10)
        assert result == {"events": []}

    @pytest.mark.asyncio
    async def test_success_returns_normal_events(self):
        gw = MagicMock()
        gw.get_events = AsyncMock(return_value={"events": [{"event_ticker": "EVT-1"}]})
        adapter = make_adapter(gw)

        result = await adapter.get_events()
        assert len(result["events"]) == 1


# ---------------------------------------------------------------------------
# get_markets error handling
# ---------------------------------------------------------------------------

class TestGetMarketsErrorHandling:
    @pytest.mark.asyncio
    async def test_returns_empty_markets_on_exception(self):
        gw = MagicMock()
        gw.get_markets = AsyncMock(side_effect=Exception("API error"))
        adapter = make_adapter(gw)

        result = await adapter.get_markets(event_ticker="EVT-1")
        assert result == {"markets": []}

    @pytest.mark.asyncio
    async def test_success_returns_normal_markets(self):
        mock_market = MagicMock()
        mock_market.model_dump.return_value = {"ticker": "MKT-A", "title": "Market A"}

        gw = MagicMock()
        gw.get_markets = AsyncMock(return_value=[mock_market])
        adapter = make_adapter(gw)

        result = await adapter.get_markets()
        assert len(result["markets"]) == 1
        assert result["markets"][0]["ticker"] == "MKT-A"
