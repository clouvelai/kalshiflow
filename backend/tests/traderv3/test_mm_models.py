"""Unit tests for MM models.py.

Tests for QuoteConfig, MarketInventory, QuoteState, MMAttentionItem,
and Pydantic model serialization.
"""

import time

import pytest

from kalshiflow_rl.traderv3.market_maker.models import (
    ActiveQuote,
    ConfigureQuotesResult,
    MarketInventory,
    MMAttentionItem,
    MMEventSnapshot,
    MMMarketSnapshot,
    PullQuotesResult,
    QuoteConfig,
    QuotePerformanceResult,
    QuoteState,
)


class TestQuoteConfig:
    """Tests for QuoteConfig dataclass."""

    def test_defaults(self):
        cfg = QuoteConfig()
        assert cfg.enabled is True
        assert cfg.base_spread_cents == 4
        assert cfg.quote_size == 10
        assert cfg.skew_factor == 0.5
        assert cfg.max_position == 100
        assert cfg.max_event_exposure == 500
        assert cfg.refresh_interval == 5.0
        assert cfg.market_overrides == {}

    def test_market_override(self):
        cfg = QuoteConfig(
            base_spread_cents=4,
            market_overrides={"MKT-A": {"base_spread_cents": 8}},
        )
        assert cfg.get_market_config("MKT-A", "base_spread_cents") == 8
        assert cfg.get_market_config("MKT-B", "base_spread_cents") == 4

    def test_market_override_fallback(self):
        cfg = QuoteConfig(quote_size=10)
        assert cfg.get_market_config("MKT-X", "quote_size") == 10

    def test_market_override_with_default(self):
        cfg = QuoteConfig()
        assert cfg.get_market_config("MKT-X", "nonexistent", "fallback") == "fallback"


class TestMarketInventory:
    """Tests for MarketInventory.record_fill()."""

    def test_initial_flat(self):
        inv = MarketInventory()
        assert inv.is_flat
        assert inv.position == 0

    def test_buy_yes_goes_long(self):
        inv = MarketInventory()
        inv.record_fill("yes", "buy", 40, 10)
        assert inv.position == 10
        assert inv.total_buys == 10

    def test_sell_yes_goes_short(self):
        inv = MarketInventory()
        inv.record_fill("yes", "sell", 60, 5)
        assert inv.position == -5
        assert inv.total_sells == 5

    def test_buy_no_goes_short(self):
        inv = MarketInventory()
        inv.record_fill("no", "buy", 40, 10)
        assert inv.position == -10

    def test_sell_no_goes_long(self):
        inv = MarketInventory()
        inv.record_fill("no", "sell", 40, 10)
        assert inv.position == 10

    def test_round_trip_realizes_pnl(self):
        inv = MarketInventory()
        inv.record_fill("yes", "buy", 40, 10)   # Buy 10 @ 40
        inv.record_fill("yes", "sell", 50, 10)  # Sell 10 @ 50
        assert inv.position == 0
        assert inv.realized_pnl_cents > 0  # Should be profit

    def test_multiple_buys_avg_entry(self):
        inv = MarketInventory()
        inv.record_fill("yes", "buy", 40, 5)
        inv.record_fill("yes", "buy", 50, 5)
        assert inv.position == 10
        assert inv.avg_entry_cents > 0

    def test_is_flat_after_full_close(self):
        inv = MarketInventory()
        inv.record_fill("yes", "buy", 40, 10)
        inv.record_fill("yes", "sell", 45, 10)
        assert inv.is_flat


class TestQuoteState:
    """Tests for QuoteState dataclass."""

    def test_defaults(self):
        qs = QuoteState()
        assert qs.active_quotes == 0
        assert qs.spread_multiplier == 1.0
        assert qs.fill_storm_active is False

    def test_to_dict(self):
        qs = QuoteState(active_quotes=4, spread_multiplier=2.0)
        d = qs.to_dict()
        assert d["active_quotes"] == 4
        assert d["spread_multiplier"] == 2.0
        assert "total_fills_bid" in d


class TestActiveQuote:
    """Tests for ActiveQuote dataclass."""

    def test_defaults(self):
        q = ActiveQuote()
        assert q.order_id == ""
        assert q.price_cents == 0
        assert q.queue_position is None

    def test_with_values(self):
        q = ActiveQuote(order_id="ord-1", side="yes", action="buy", price_cents=45, size=10)
        assert q.order_id == "ord-1"
        assert q.price_cents == 45


class TestMMAttentionItem:
    """Tests for MMAttentionItem."""

    def test_key(self):
        item = MMAttentionItem(event_ticker="EVT-1", market_ticker="MKT-A", category="fill")
        assert item.key == "EVT-1:MKT-A:fill"

    def test_key_no_market(self):
        item = MMAttentionItem(event_ticker="EVT-1", category="fill_storm")
        assert item.key == "EVT-1::fill_storm"

    def test_is_expired(self):
        item = MMAttentionItem(
            event_ticker="EVT-1",
            category="fill",
            created_at=time.time() - 200,
            ttl_seconds=120.0,
        )
        assert item.is_expired is True

    def test_not_expired(self):
        item = MMAttentionItem(
            event_ticker="EVT-1",
            category="fill",
            ttl_seconds=120.0,
        )
        assert item.is_expired is False

    def test_to_prompt(self):
        item = MMAttentionItem(
            event_ticker="EVT-1",
            market_ticker="MKT-A",
            urgency="high",
            category="fill",
            summary="Fill: bid 5@42c",
        )
        prompt = item.to_prompt()
        assert "[HIGH]" in prompt
        assert "EVT-1" in prompt
        assert "MKT-A" in prompt

    def test_to_dict(self):
        item = MMAttentionItem(event_ticker="EVT-1", category="fill", score=42.5)
        d = item.to_dict()
        assert d["event_ticker"] == "EVT-1"
        assert d["score"] == 42.5


class TestPydanticModels:
    """Tests for Pydantic model serialization."""

    def test_mm_market_snapshot_defaults(self):
        snap = MMMarketSnapshot(ticker="MKT-A", title="Market A")
        assert snap.ticker == "MKT-A"
        assert snap.yes_levels == []
        assert snap.maker_fee_cents == 0.0

    def test_mm_event_snapshot(self):
        snap = MMEventSnapshot(event_ticker="EVT-1", title="Event 1", market_count=3)
        assert snap.event_ticker == "EVT-1"
        assert snap.market_count == 3
        assert snap.markets == {}

    def test_configure_quotes_result(self):
        result = ConfigureQuotesResult(status="updated", changes=["spread: 4→6"])
        assert result.status == "updated"
        assert len(result.changes) == 1

    def test_pull_quotes_result(self):
        result = PullQuotesResult(status="pulled", cancelled_orders=8, reason="VPIN spike")
        assert result.cancelled_orders == 8


class TestNewsModelReExports:
    """Verify news models are re-exported from single_arb without duplication."""

    def test_news_article_import(self):
        from kalshiflow_rl.traderv3.market_maker.models import NewsArticle
        from kalshiflow_rl.traderv3.single_arb.models import NewsArticle as OriginalNewsArticle
        assert NewsArticle is OriginalNewsArticle

    def test_impact_pattern_import(self):
        from kalshiflow_rl.traderv3.market_maker.models import ImpactPattern
        from kalshiflow_rl.traderv3.single_arb.models import ImpactPattern as OriginalImpactPattern
        assert ImpactPattern is OriginalImpactPattern

    def test_swing_news_result_import(self):
        from kalshiflow_rl.traderv3.market_maker.models import SwingNewsResult
        from kalshiflow_rl.traderv3.single_arb.models import SwingNewsResult as OriginalSwingNewsResult
        assert SwingNewsResult is OriginalSwingNewsResult

    def test_news_article_creation(self):
        from kalshiflow_rl.traderv3.market_maker.models import NewsArticle
        article = NewsArticle(title="Test", content="Body", source="tavily")
        assert article.title == "Test"
        assert article.similar_patterns == []

    def test_swing_news_result_creation(self):
        from kalshiflow_rl.traderv3.market_maker.models import SwingNewsResult
        result = SwingNewsResult(query="test", depth="fast", count=0)
        assert result.depth == "fast"
        assert result.patterns_found == 0
