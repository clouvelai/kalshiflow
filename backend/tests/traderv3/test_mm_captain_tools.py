"""Tests for MM tools folded into Captain (configure_quotes, pull_quotes, resume_quotes, get_quote_performance)."""

import asyncio
from dataclasses import dataclass, field
from typing import Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kalshiflow_rl.traderv3.single_arb.tools import (
    configure_quotes,
    get_quote_performance,
    pull_quotes,
    resume_quotes,
    set_context,
    ToolContext,
)
from kalshiflow_rl.traderv3.market_maker.models import QuoteConfig, QuoteState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_quote_engine(state: Optional[QuoteState] = None):
    """Create a mock QuoteEngine with configurable state."""
    engine = MagicMock()
    engine.state = state or QuoteState()
    engine.pull_all_quotes = AsyncMock(return_value=5)
    engine.resume_quotes = MagicMock()
    engine.is_running = True
    return engine


def _make_mock_mm_index():
    """Create a mock MMIndex."""
    index = MagicMock()
    index.total_realized_pnl.return_value = 42.5
    index.market_tickers = ["MKT-A", "MKT-B"]
    inv_a = MagicMock()
    inv_a.position = 10
    inv_a.realized_pnl_cents = 20
    inv_b = MagicMock()
    inv_b.position = 0
    inv_b.realized_pnl_cents = 0
    index.get_inventory.side_effect = lambda t: inv_a if t == "MKT-A" else inv_b
    return index


def _make_tool_context(quote_engine=None, mm_index=None, quote_config=None):
    """Create a ToolContext with MM fields populated."""
    ctx = ToolContext(
        gateway=MagicMock(),
        index=MagicMock(),
        memory=MagicMock(),
        search=None,
        sniper=None,
        sniper_config=None,
        session=MagicMock(),
        context_builder=MagicMock(),
        quote_engine=quote_engine,
        mm_index=mm_index,
        quote_config=quote_config,
    )
    set_context(ctx)
    return ctx


# ---------------------------------------------------------------------------
# configure_quotes
# ---------------------------------------------------------------------------

class TestConfigureQuotes:
    @pytest.mark.asyncio
    async def test_unavailable_without_quote_engine(self):
        _make_tool_context()
        result = await configure_quotes.ainvoke({"settings": {"enabled": False}})
        assert result["status"] == "unavailable"

    @pytest.mark.asyncio
    async def test_updates_config(self):
        qc = QuoteConfig(base_spread_cents=4, quote_size=10)
        _make_tool_context(
            quote_engine=_make_mock_quote_engine(),
            quote_config=qc,
        )
        result = await configure_quotes.ainvoke({
            "settings": {"base_spread_cents": 6, "quote_size": 20}
        })
        assert result["status"] == "updated"
        assert qc.base_spread_cents == 6
        assert qc.quote_size == 20
        assert len(result["changes"]) == 2

    @pytest.mark.asyncio
    async def test_no_changes(self):
        qc = QuoteConfig(base_spread_cents=4)
        _make_tool_context(
            quote_engine=_make_mock_quote_engine(),
            quote_config=qc,
        )
        result = await configure_quotes.ainvoke({
            "settings": {"base_spread_cents": 4}
        })
        assert result["status"] == "no_changes"
        assert len(result["changes"]) == 0

    @pytest.mark.asyncio
    async def test_ignores_unknown_keys(self):
        qc = QuoteConfig()
        _make_tool_context(
            quote_engine=_make_mock_quote_engine(),
            quote_config=qc,
        )
        result = await configure_quotes.ainvoke({
            "settings": {"unknown_key": 999}
        })
        assert result["status"] == "no_changes"


# ---------------------------------------------------------------------------
# pull_quotes
# ---------------------------------------------------------------------------

class TestPullQuotes:
    @pytest.mark.asyncio
    async def test_unavailable_without_engine(self):
        _make_tool_context()
        result = await pull_quotes.ainvoke({"reason": "test"})
        assert result["status"] == "unavailable"

    @pytest.mark.asyncio
    async def test_pulls_quotes(self):
        engine = _make_mock_quote_engine()
        _make_tool_context(quote_engine=engine)
        result = await pull_quotes.ainvoke({"reason": "vpin_spike"})
        assert result["status"] == "pulled"
        assert result["cancelled_orders"] == 5
        engine.pull_all_quotes.assert_awaited_once_with("vpin_spike")


# ---------------------------------------------------------------------------
# resume_quotes
# ---------------------------------------------------------------------------

class TestResumeQuotes:
    @pytest.mark.asyncio
    async def test_unavailable_without_engine(self):
        _make_tool_context()
        result = await resume_quotes.ainvoke({"reason": "test"})
        assert result["status"] == "unavailable"

    @pytest.mark.asyncio
    async def test_resumes(self):
        engine = _make_mock_quote_engine()
        _make_tool_context(quote_engine=engine)
        result = await resume_quotes.ainvoke({"reason": "vpin_subsided"})
        assert result["status"] == "resumed"
        engine.resume_quotes.assert_called_once()


# ---------------------------------------------------------------------------
# get_quote_performance
# ---------------------------------------------------------------------------

class TestGetQuotePerformance:
    @pytest.mark.asyncio
    async def test_returns_defaults_without_engine(self):
        _make_tool_context()
        result = await get_quote_performance.ainvoke({})
        assert result["total_fills_bid"] == 0
        assert result["total_fills_ask"] == 0

    @pytest.mark.asyncio
    async def test_returns_telemetry(self):
        state = QuoteState()
        state.total_fills_bid = 12
        state.total_fills_ask = 8
        state.total_requote_cycles = 100
        state.spread_captured_cents = 45.0
        state.adverse_selection_cents = 10.0
        state.fees_paid_cents = 5.0

        engine = _make_mock_quote_engine(state)
        mm_index = _make_mock_mm_index()
        _make_tool_context(quote_engine=engine, mm_index=mm_index)

        result = await get_quote_performance.ainvoke({})
        assert result["total_fills_bid"] == 12
        assert result["total_fills_ask"] == 8
        assert result["total_requote_cycles"] == 100
        assert result["spread_captured_cents"] == 45.0
        assert result["net_pnl_cents"] == 42.5
