"""Unit tests for MM tools - enriched news intelligence.

Tests for _resolve_depth, _build_price_snapshot, get_market_movers,
get_resting_orders, and enriched search_news.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kalshiflow_rl.traderv3.market_maker.tools import (
    MMToolContext,
    _resolve_depth,
    _build_price_snapshot,
    get_mm_tools,
    set_context,
    get_context,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class FakeMicro:
    vpin: float = 0.5
    book_imbalance: float = 0.1
    volume_5m: int = 100
    whale_trade_count: int = 0
    avg_inter_trade_ms: float = 0.0
    rapid_sequence_count: int = 0
    buy_sell_ratio: float = 0.5
    consistent_size_ratio: float = 0.0


@dataclass
class FakeMarket:
    title: str = "Test Market"
    yes_bid: Optional[int] = 40
    yes_ask: Optional[int] = 45
    yes_mid: Optional[float] = 42.5
    spread: Optional[int] = 5
    micro: FakeMicro = field(default_factory=FakeMicro)
    open_interest: Optional[int] = 1000


@dataclass
class FakeEvent:
    event_ticker: str = "EVT-1"
    title: str = "Test Event"
    markets: Dict[str, FakeMarket] = field(default_factory=dict)
    mutually_exclusive: bool = True
    understanding: Optional[Dict] = None

    def market_sum(self):
        return 100.0


class FakeMMIndex:
    def __init__(self, events=None):
        self.events = events or {}
        self._arb_index = MagicMock()


def _make_ctx(cycle_mode=None, events=None, **overrides) -> MMToolContext:
    """Create a minimal MMToolContext for testing."""
    idx = FakeMMIndex(events or {})
    return MMToolContext(
        gateway=MagicMock(),
        index=idx,
        quote_engine=MagicMock(),
        memory=MagicMock(),
        search=MagicMock(),
        session=MagicMock(),
        cycle_mode=cycle_mode,
        **overrides,
    )


# ---------------------------------------------------------------------------
# _resolve_depth
# ---------------------------------------------------------------------------

class TestResolveDepth:
    def test_explicit_depth_returned(self):
        set_context(_make_ctx(cycle_mode="deep_scan"))
        assert _resolve_depth("fast") == "fast"
        assert _resolve_depth("ultra_fast") == "ultra_fast"
        assert _resolve_depth("advanced") == "advanced"

    def test_auto_reactive(self):
        set_context(_make_ctx(cycle_mode="reactive"))
        assert _resolve_depth("auto") == "ultra_fast"

    def test_auto_strategic(self):
        set_context(_make_ctx(cycle_mode="strategic"))
        assert _resolve_depth("auto") == "fast"

    def test_auto_deep_scan(self):
        set_context(_make_ctx(cycle_mode="deep_scan"))
        assert _resolve_depth("auto") == "advanced"

    def test_auto_no_mode(self):
        set_context(_make_ctx(cycle_mode=None))
        assert _resolve_depth("auto") == "fast"

    def test_auto_no_context(self):
        set_context(None)
        assert _resolve_depth("auto") == "fast"

    def test_auto_unknown_mode(self):
        set_context(_make_ctx(cycle_mode="unknown"))
        assert _resolve_depth("auto") == "fast"


# ---------------------------------------------------------------------------
# _build_price_snapshot
# ---------------------------------------------------------------------------

class TestBuildPriceSnapshot:
    @pytest.fixture(autouse=True)
    def setup(self):
        market = FakeMarket(yes_bid=40, yes_ask=45, yes_mid=42.5, spread=5)
        event = FakeEvent(markets={"MKT-A": market})
        ctx = _make_ctx(events={"EVT-1": event})
        set_context(ctx)

    @pytest.mark.asyncio
    async def test_returns_snapshot(self):
        snap = await _build_price_snapshot("EVT-1")
        assert snap is not None
        assert "MKT-A" in snap
        assert snap["MKT-A"]["yes_mid"] == 42.5
        assert snap["MKT-A"]["yes_bid"] == 40
        assert snap["MKT-A"]["yes_ask"] == 45
        assert "_ts" in snap

    @pytest.mark.asyncio
    async def test_none_for_unknown_event(self):
        snap = await _build_price_snapshot("EVT-NONEXISTENT")
        assert snap is None

    @pytest.mark.asyncio
    async def test_none_for_no_event_ticker(self):
        snap = await _build_price_snapshot(None)
        assert snap is None

    @pytest.mark.asyncio
    async def test_none_for_no_context(self):
        set_context(None)
        snap = await _build_price_snapshot("EVT-1")
        assert snap is None

    @pytest.mark.asyncio
    async def test_skips_markets_without_mid(self):
        market_no_mid = FakeMarket(yes_mid=None)
        event = FakeEvent(markets={"MKT-B": market_no_mid})
        set_context(_make_ctx(events={"EVT-2": event}))
        snap = await _build_price_snapshot("EVT-2")
        assert snap is None


# ---------------------------------------------------------------------------
# get_mm_tools
# ---------------------------------------------------------------------------

class TestGetMMTools:
    def test_tool_count(self):
        tools = get_mm_tools()
        assert len(tools) == 12

    def test_tool_names(self):
        tools = get_mm_tools()
        names = [t.name for t in tools]
        assert "search_news" in names
        assert "get_market_movers" in names
        assert "get_resting_orders" in names
        assert "recall_memory" in names
        assert "store_insight" in names
        assert "get_mm_state" in names
        assert "get_inventory" in names
        assert "get_quote_performance" in names
        assert "configure_quotes" in names
        assert "set_market_override" in names
        assert "pull_quotes" in names
        assert "resume_quotes" in names


# ---------------------------------------------------------------------------
# search_news (enriched)
# ---------------------------------------------------------------------------

class TestSearchNews:
    @pytest.fixture(autouse=True)
    def setup(self):
        from kalshiflow_rl.traderv3.market_maker.tools import search_news
        self.search_news = search_news

    @pytest.mark.asyncio
    async def test_no_context_returns_error(self):
        set_context(None)
        result = await self.search_news.ainvoke({"query": "test"})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_ultra_fast_memory_only(self):
        """ultra_fast depth should only use memory recall, not Tavily."""
        from kalshiflow_rl.traderv3.single_arb.models import RecallResult, MemoryEntry

        memory = AsyncMock()
        memory.recall = AsyncMock(return_value=RecallResult(
            query="test",
            results=[
                MemoryEntry(content="NEWS: Test article\nContent here", memory_type="news", similarity=0.9),
            ],
            count=1,
        ))

        ctx = _make_ctx(cycle_mode="reactive")
        ctx.memory = memory
        ctx.search = None  # No search service
        set_context(ctx)

        result = await self.search_news.ainvoke({"query": "test", "depth": "ultra_fast"})
        assert result["depth"] == "ultra_fast"
        assert result["count"] >= 1
        memory.recall.assert_called_once()


# ---------------------------------------------------------------------------
# get_market_movers
# ---------------------------------------------------------------------------

class TestGetMarketMovers:
    @pytest.fixture(autouse=True)
    def setup(self):
        from kalshiflow_rl.traderv3.market_maker.tools import get_market_movers
        self.get_market_movers = get_market_movers

    @pytest.mark.asyncio
    async def test_no_context(self):
        set_context(None)
        result = await self.get_market_movers.ainvoke({"event_ticker": "EVT-1"})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_db_unavailable(self):
        set_context(_make_ctx())
        with patch("kalshiflow_rl.traderv3.market_maker.tools.asyncio.wait_for", side_effect=asyncio.TimeoutError):
            result = await self.get_market_movers.ainvoke({"event_ticker": "EVT-1"})
        assert result["count"] == 0
        assert "error" in result


# ---------------------------------------------------------------------------
# get_resting_orders
# ---------------------------------------------------------------------------

class TestGetRestingOrders:
    @pytest.fixture(autouse=True)
    def setup(self):
        from kalshiflow_rl.traderv3.market_maker.tools import get_resting_orders
        self.get_resting_orders = get_resting_orders

    @pytest.mark.asyncio
    async def test_no_context(self):
        set_context(None)
        result = await self.get_resting_orders.ainvoke({})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_empty_orders(self):
        gw = AsyncMock()
        gw.get_orders = AsyncMock(return_value=[])
        ctx = _make_ctx()
        ctx.gateway = gw
        set_context(ctx)
        result = await self.get_resting_orders.ainvoke({})
        assert result["count"] == 0
        assert result["orders"] == []

    @pytest.mark.asyncio
    async def test_gateway_error(self):
        gw = AsyncMock()
        gw.get_orders = AsyncMock(side_effect=Exception("API error"))
        ctx = _make_ctx()
        ctx.gateway = gw
        set_context(ctx)
        result = await self.get_resting_orders.ainvoke({})
        assert "error" in result


# ---------------------------------------------------------------------------
# MMToolContext
# ---------------------------------------------------------------------------

class TestMMToolContext:
    def test_understanding_builder_field(self):
        ctx = _make_ctx()
        assert ctx.understanding_builder is None

    def test_understanding_builder_set(self):
        builder = MagicMock()
        ctx = _make_ctx(understanding_builder=builder)
        assert ctx.understanding_builder is builder

    def test_cycle_mode_field(self):
        ctx = _make_ctx(cycle_mode="strategic")
        assert ctx.cycle_mode == "strategic"
