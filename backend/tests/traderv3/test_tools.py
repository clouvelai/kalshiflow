"""Unit tests for Captain V2 tools.

T2 tests — async with mocked gateway/memory. No network calls.
Tests each of the 10 tools returns correct Pydantic-shaped responses.
"""

import asyncio
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, Set
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.traderv3.conftest import make_event_meta, make_index, make_market_meta

from kalshiflow_rl.traderv3.single_arb.context_builder import ContextBuilder
from kalshiflow_rl.traderv3.single_arb.tools import (
    ALL_TOOLS,
    TOOL_CATEGORIES,
    ToolContext,
    get_context,
    set_context,
    get_market_state,
    get_market_movers,
    get_portfolio,
    place_order,
    execute_arb,
    cancel_order,
    get_resting_orders,
    search_news,
    recall_memory,
    store_insight,
    configure_sniper,
)


# ===========================================================================
# Fixtures
# ===========================================================================


def _make_mock_memory():
    """Create a mock SessionMemoryStore."""
    mem = MagicMock()
    mem.store = AsyncMock()
    mem.recall = AsyncMock(return_value=MagicMock(
        model_dump=lambda: {"query": "test", "results": [], "count": 0}
    ))
    return mem


def _make_mock_gateway():
    """Create a mock KalshiGateway."""
    gw = MagicMock()
    gw.get_balance = AsyncMock(return_value=MagicMock(balance=50000))
    gw.get_positions = AsyncMock(return_value=[])
    gw.get_orders = AsyncMock(return_value=[])

    # create_order returns OrderResponse (Pydantic) with .order -> Order
    mock_order = MagicMock(order_id="test-order-001", status="resting")
    mock_order_resp = MagicMock(order=mock_order)
    gw.create_order = AsyncMock(return_value=mock_order_resp)

    # cancel_order
    gw.cancel_order = AsyncMock(return_value={"order": {"order_id": "test-order-001", "status": "canceled"}})
    return gw


def _make_mock_session():
    """Create a mock TradingSession."""
    session = MagicMock()
    session.order_group_id = "test-group-001"
    session.order_ttl = 30
    session.captain_order_ids = set()
    return session


def _make_mock_search():
    """Create a mock TavilySearchService."""
    search = MagicMock()
    search.search_news = AsyncMock(return_value=[
        {"title": "Test Article", "url": "https://example.com", "content": "content", "score": 0.9},
    ])
    return search


@contextmanager
def inject_v2_context(
    index=None,
    gateway=None,
    memory=None,
    search=None,
    sniper=None,
    sniper_config=None,
    session=None,
    broadcast=None,
):
    """Context manager to inject V2 ToolContext and reset after."""
    from kalshiflow_rl.traderv3.single_arb import tools

    saved = tools._ctx

    idx = index or make_index(events=[])
    ctx = ToolContext(
        gateway=gateway or _make_mock_gateway(),
        index=idx,
        memory=memory or _make_mock_memory(),
        search=search,
        sniper=sniper,
        sniper_config=sniper_config,
        session=session or _make_mock_session(),
        context_builder=ContextBuilder(idx),
        broadcast=broadcast or AsyncMock(),
    )
    set_context(ctx)
    try:
        yield ctx
    finally:
        tools._ctx = saved


# ===========================================================================
# TestToolList
# ===========================================================================


class TestToolList:
    def test_13_tools(self):
        assert len(ALL_TOOLS) == 13

    def test_configure_sniper_not_in_all_tools(self):
        """configure_sniper removed from Captain's tool list to prevent sniper obsession."""
        tool_names = [t.name for t in ALL_TOOLS]
        assert "configure_sniper" not in tool_names

    def test_all_categorized(self):
        for tool in ALL_TOOLS:
            assert tool.name in TOOL_CATEGORIES, f"Tool {tool.name} not categorized"

    def test_categories_valid(self):
        valid = {"arb", "memory", "sniper", "system", "todo"}
        for name, cat in TOOL_CATEGORIES.items():
            assert cat in valid, f"Tool {name} has invalid category {cat}"


# ===========================================================================
# TestGetMarketState
# ===========================================================================


class TestGetMarketState:
    @pytest.mark.asyncio
    async def test_returns_dict(self):
        event = make_event_meta(event_ticker="EVT-1", n_markets=2)
        index = make_index(events=[event])
        with inject_v2_context(index=index):
            result = await get_market_state.ainvoke({})
        assert isinstance(result, dict)
        assert "events" in result
        assert result["total_events"] == 1

    @pytest.mark.asyncio
    async def test_empty_index(self):
        with inject_v2_context():
            result = await get_market_state.ainvoke({})
        assert result["total_events"] == 0


# ===========================================================================
# TestGetPortfolio
# ===========================================================================


class TestGetPortfolio:
    @pytest.mark.asyncio
    async def test_returns_dict(self):
        gw = _make_mock_gateway()
        with inject_v2_context(gateway=gw):
            result = await get_portfolio.ainvoke({})
        assert isinstance(result, dict)
        assert "balance_cents" in result
        assert result["balance_cents"] == 50000


# ===========================================================================
# TestPlaceOrder
# ===========================================================================


class TestPlaceOrder:
    @pytest.mark.asyncio
    async def test_success(self):
        gw = _make_mock_gateway()
        with inject_v2_context(gateway=gw) as ctx:
            result = await place_order.ainvoke({
                "ticker": "MKT-A",
                "side": "yes",
                "action": "buy",
                "contracts": 5,
                "price_cents": 40,
                "reasoning": "Test trade",
            })
        assert isinstance(result, dict)
        assert result.get("order_id") == "test-order-001"
        assert result.get("new_balance_cents") == 50000

    @pytest.mark.asyncio
    async def test_invalid_side(self):
        with inject_v2_context():
            result = await place_order.ainvoke({
                "ticker": "MKT-A",
                "side": "invalid",
                "action": "buy",
                "contracts": 5,
                "price_cents": 40,
                "reasoning": "Test",
            })
        assert "error" in result

    @pytest.mark.asyncio
    async def test_invalid_action(self):
        with inject_v2_context():
            result = await place_order.ainvoke({
                "ticker": "MKT-A",
                "side": "yes",
                "action": "invalid",
                "contracts": 5,
                "price_cents": 40,
                "reasoning": "Test",
            })
        assert "error" in result

    @pytest.mark.asyncio
    async def test_tracks_order_id(self):
        gw = _make_mock_gateway()
        with inject_v2_context(gateway=gw) as ctx:
            await place_order.ainvoke({
                "ticker": "MKT-A", "side": "yes", "action": "buy",
                "contracts": 5, "price_cents": 40, "reasoning": "Test",
            })
            assert "test-order-001" in ctx.captain_order_ids


# ===========================================================================
# TestCancelOrder
# ===========================================================================


class TestCancelOrder:
    @pytest.mark.asyncio
    async def test_success(self):
        gw = _make_mock_gateway()
        with inject_v2_context(gateway=gw):
            result = await cancel_order.ainvoke({
                "order_id": "test-order-001",
                "reason": "No longer needed",
            })
        assert result["status"] == "cancelled"


# ===========================================================================
# TestGetRestingOrders
# ===========================================================================


class TestGetRestingOrders:
    @pytest.mark.asyncio
    async def test_empty(self):
        gw = _make_mock_gateway()
        with inject_v2_context(gateway=gw):
            result = await get_resting_orders.ainvoke({})
        assert isinstance(result, dict)
        assert result["count"] == 0
        assert result["orders"] == []


# ===========================================================================
# TestSearchNews
# ===========================================================================


class TestSearchNews:
    @pytest.mark.asyncio
    async def test_success(self):
        search = _make_mock_search()
        with inject_v2_context(search=search):
            result = await search_news.ainvoke({"query": "test news"})
        assert isinstance(result, dict)
        assert result["count"] >= 1
        assert result["stored_in_memory"] is True
        assert result["cached"] is False

    @pytest.mark.asyncio
    async def test_no_search_service(self):
        """Without a search service, returns results from memory only (no error)."""
        with inject_v2_context(search=None):
            result = await search_news.ainvoke({"query": "test"})
        assert isinstance(result, dict)
        assert result["count"] == 0  # No memory entries, no Tavily = 0 articles

    @pytest.mark.asyncio
    async def test_cache_hit(self):
        """Second identical call returns cached result without calling search again."""
        search = _make_mock_search()
        with inject_v2_context(search=search):
            r1 = await search_news.ainvoke({"query": "test news"})
            assert r1["cached"] is False
            assert search.search_news.call_count == 1

            r2 = await search_news.ainvoke({"query": "test news"})
            assert r2["cached"] is True
            assert search.search_news.call_count == 1  # NOT called again

    @pytest.mark.asyncio
    async def test_cache_force_refresh(self):
        """force_refresh=True bypasses cache."""
        search = _make_mock_search()
        with inject_v2_context(search=search):
            r1 = await search_news.ainvoke({"query": "test news"})
            assert r1["cached"] is False

            r2 = await search_news.ainvoke({"query": "test news", "force_refresh": True})
            assert r2["cached"] is False
            assert search.search_news.call_count == 2  # Called again

    @pytest.mark.asyncio
    async def test_cache_expires(self):
        """Cache entry expires after TTL."""
        search = _make_mock_search()
        with inject_v2_context(search=search) as ctx:
            r1 = await search_news.ainvoke({"query": "test news"})
            assert r1["cached"] is False

            # Artificially age the cache entry (key includes depth)
            cache_key = "test news||week|fast"
            ts, data = ctx.news_cache[cache_key]
            ctx.news_cache[cache_key] = (ts - 301, data)

            r2 = await search_news.ainvoke({"query": "test news"})
            assert r2["cached"] is False
            assert search.search_news.call_count == 2

    @pytest.mark.asyncio
    async def test_cache_key_varies_by_params(self):
        """Different query/event_ticker/time_range produce different cache keys."""
        search = _make_mock_search()
        with inject_v2_context(search=search):
            await search_news.ainvoke({"query": "test news"})
            await search_news.ainvoke({"query": "test news", "time_range": "day"})
            assert search.search_news.call_count == 2  # Different key, separate fetch


# ===========================================================================
# TestRecallMemory
# ===========================================================================


class TestRecallMemory:
    @pytest.mark.asyncio
    async def test_returns_dict(self):
        with inject_v2_context():
            result = await recall_memory.ainvoke({"query": "test"})
        assert isinstance(result, dict)
        assert "query" in result


# ===========================================================================
# TestStoreInsight
# ===========================================================================


class TestStoreInsight:
    @pytest.mark.asyncio
    async def test_stores_and_returns(self):
        mem = _make_mock_memory()
        with inject_v2_context(memory=mem):
            result = await store_insight.ainvoke({"content": "learned something"})
        assert result["status"] == "stored"
        mem.store.assert_called_once()

    @pytest.mark.asyncio
    async def test_rejects_toxic_content(self):
        """store_insight should reject self-imposed restrictions."""
        mem = _make_mock_memory()
        toxic_contents = [
            "NEVER TRADE on events without 80% accuracy",
            "Pause trading until we recover from losses",
            "Crisis detected: implement recovery plan",
            "No directional trades until confidence improves",
            "Stop trading when VPIN is high",
        ]
        for content in toxic_contents:
            with inject_v2_context(memory=mem):
                result = await store_insight.ainvoke({"content": content})
            assert result["status"] == "rejected", f"Should reject: {content}"
            assert "Self-imposed restrictions" in result["reason"]

    @pytest.mark.asyncio
    async def test_allows_factual_content(self):
        """store_insight should allow factual observations."""
        mem = _make_mock_memory()
        factual_contents = [
            "Fed rate decision moved KXFED markets 8c in 15min",
            "VPIN spiked to 0.95 during sweep on MKT-A",
            "Complement fair value was 42c, market opened at 38c",
        ]
        for content in factual_contents:
            with inject_v2_context(memory=mem):
                result = await store_insight.ainvoke({"content": content})
            assert result["status"] == "stored", f"Should allow: {content}"


# ===========================================================================
# TestConfigureSniper
# ===========================================================================


class TestConfigureSniper:
    @pytest.mark.asyncio
    async def test_no_sniper(self):
        with inject_v2_context(sniper_config=None):
            result = await configure_sniper.ainvoke({"settings": {"enabled": True}})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_with_sniper(self):
        mock_config = MagicMock()
        mock_config.update.return_value = (["enabled"], [])
        mock_config.enabled = True
        mock_config.arb_min_edge = 3.0
        mock_config.max_capital = 100000
        mock_config.max_position = 25
        mock_config.cooldown = 10.0
        mock_config.vpin_reject_threshold = 0.7

        with inject_v2_context(sniper_config=mock_config):
            result = await configure_sniper.ainvoke({"settings": {"enabled": True}})
        assert result["status"] == "updated"


# ===========================================================================
# TestContextIsolation
# ===========================================================================


class TestContextIsolation:
    def test_set_and_get(self):
        from kalshiflow_rl.traderv3.single_arb import tools
        saved = tools._ctx
        try:
            idx = make_index(events=[])
            ctx = ToolContext(
                gateway=_make_mock_gateway(),
                index=idx,
                memory=_make_mock_memory(),
                search=None,
                sniper=None,
                sniper_config=None,
                session=_make_mock_session(),
                context_builder=ContextBuilder(idx),
                broadcast=None,
            )
            set_context(ctx)
            assert get_context() is ctx
        finally:
            tools._ctx = saved

    @pytest.mark.asyncio
    async def test_no_context_returns_error(self):
        from kalshiflow_rl.traderv3.single_arb import tools
        saved = tools._ctx
        try:
            tools._ctx = None
            result = await get_market_state.ainvoke({})
            assert "error" in result
        finally:
            tools._ctx = saved


# ===========================================================================
# TestCycleCapitalBudget (Gap 1)
# ===========================================================================


class TestCycleCapitalBudget:
    @pytest.mark.asyncio
    async def test_cycle_capital_tracks_place_order(self):
        """place_order increments cycle_capital_spent_cents."""
        event = make_event_meta(event_ticker="EVT-1", n_markets=2)
        index = make_index(events=[event])
        gw = _make_mock_gateway()
        with inject_v2_context(index=index, gateway=gw) as ctx:
            ctx.cycle_capital_spent_cents = 0
            result = await place_order.ainvoke({
                "ticker": "EVT-1-MKT-A",
                "side": "yes",
                "action": "buy",
                "contracts": 5,
                "price_cents": 40,
                "reasoning": "test",
            })
            assert result.get("error") is None
            assert ctx.cycle_capital_spent_cents == 5 * 40

    @pytest.mark.asyncio
    async def test_cycle_capital_tracks_execute_arb(self):
        """execute_arb accumulates cost from all executed legs."""
        # Build a 2-market event with edge
        event = make_event_meta(
            event_ticker="EVT-1", n_markets=2,
            market_prices=[{"yes_bid": 40, "yes_ask": 42}, {"yes_bid": 45, "yes_ask": 48}],
            mutually_exclusive=True,
        )
        index = make_index(events=[event])
        gw = _make_mock_gateway()
        gw.get_balance = AsyncMock(return_value=MagicMock(balance=100000))
        with inject_v2_context(index=index, gateway=gw) as ctx:
            ctx.cycle_capital_spent_cents = 0
            result = await execute_arb.ainvoke({
                "event_ticker": "EVT-1",
                "direction": "long",
                "max_contracts": 5,
                "reasoning": "test arb",
            })
            # Should have accumulated cost from executed legs
            assert ctx.cycle_capital_spent_cents > 0

    def test_cycle_capital_default_zero(self):
        """ToolContext.cycle_capital_spent_cents defaults to 0."""
        event = make_event_meta(event_ticker="EVT-1", n_markets=1)
        index = make_index(events=[event])
        with inject_v2_context(index=index) as ctx:
            assert ctx.cycle_capital_spent_cents == 0


# ===========================================================================
# TestGetMarketMovers
# ===========================================================================


class TestGetMarketMovers:
    @pytest.mark.asyncio
    async def test_no_context_returns_error(self):
        """get_market_movers without ToolContext should return error."""
        from kalshiflow_rl.traderv3.single_arb import tools
        saved = tools._ctx
        try:
            tools._ctx = None
            result = await get_market_movers.ainvoke({"event_ticker": "EVT-1"})
            assert "error" in result
        finally:
            tools._ctx = saved

    @pytest.mark.asyncio
    async def test_db_not_available_returns_graceful_error(self):
        """get_market_movers should return graceful error when DB unavailable."""
        with inject_v2_context():
            # rl_db is not configured in test environment, so import/pool will fail
            result = await get_market_movers.ainvoke({"event_ticker": "EVT-1"})
        assert "error" in result
        assert result.get("movers") == [] or result.get("movers") is None
