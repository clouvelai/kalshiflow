"""Unit tests for MMAttentionRouter.

Tests signal scoring, dedup, cooldowns, and drain behavior.
Uses asyncio.Event mocking.
"""

import asyncio
import time

import pytest

from kalshiflow_rl.traderv3.market_maker.attention import MMAttentionRouter
from kalshiflow_rl.traderv3.market_maker.models import MMAttentionItem


class TestMMAttentionRouterFill:
    """Tests for on_fill() signal."""

    def test_fill_creates_item(self):
        router = MMAttentionRouter()
        router.on_fill("EVT-1", "MKT-A", "bid", 42, 5, 5)
        items = router.pending_items
        assert len(items) == 1
        assert items[0].category == "fill"
        assert items[0].urgency == "high"

    def test_fill_score_scales_with_quantity(self):
        router = MMAttentionRouter()
        router.on_fill("EVT-1", "MKT-A", "bid", 42, 1, 1)
        router.on_fill("EVT-1", "MKT-B", "bid", 42, 15, 15)
        items = router.pending_items
        assert len(items) == 2
        # Larger fill should have higher score
        scores = {i.market_ticker: i.score for i in items}
        assert scores["MKT-B"] > scores["MKT-A"]

    def test_fill_sets_notify_event(self):
        router = MMAttentionRouter()
        assert not router.notify_event.is_set()
        router.on_fill("EVT-1", "MKT-A", "bid", 42, 5, 5)
        assert router.notify_event.is_set()


class TestMMAttentionRouterInventory:
    """Tests for on_inventory_warning() signal."""

    def test_no_warning_below_70pct(self):
        router = MMAttentionRouter()
        router.on_inventory_warning("EVT-1", "MKT-A", 50, 100)  # 50%
        assert len(router.pending_items) == 0

    def test_warning_above_70pct(self):
        router = MMAttentionRouter()
        router.on_inventory_warning("EVT-1", "MKT-A", 75, 100)  # 75%
        items = router.pending_items
        assert len(items) == 1
        assert items[0].category == "inventory_warning"
        assert items[0].urgency == "high"

    def test_immediate_above_90pct(self):
        router = MMAttentionRouter()
        router.on_inventory_warning("EVT-1", "MKT-A", 95, 100)  # 95%
        items = router.pending_items
        assert len(items) == 1
        assert items[0].urgency == "immediate"


class TestMMAttentionRouterVpin:
    """Tests for on_vpin_spike() signal."""

    def test_no_signal_below_80pct_threshold(self):
        router = MMAttentionRouter()
        router.on_vpin_spike("EVT-1", "MKT-A", 0.5, 0.95)  # Well below
        assert len(router.pending_items) == 0

    def test_signal_near_threshold(self):
        router = MMAttentionRouter()
        router.on_vpin_spike("EVT-1", "MKT-A", 0.80, 0.95)  # Above 80% of 0.95
        items = router.pending_items
        assert len(items) == 1
        assert items[0].category == "vpin_spike"

    def test_immediate_at_threshold(self):
        router = MMAttentionRouter()
        router.on_vpin_spike("EVT-1", "MKT-A", 0.96, 0.95)
        items = router.pending_items
        assert len(items) == 1
        assert items[0].urgency == "immediate"


class TestMMAttentionRouterSpread:
    """Tests for on_spread_change() signal."""

    def test_ignore_small_change(self):
        router = MMAttentionRouter()
        router.on_spread_change("EVT-1", "MKT-A", 5, 6)  # Change=1, < 2
        assert len(router.pending_items) == 0

    def test_signal_on_big_change(self):
        router = MMAttentionRouter()
        router.on_spread_change("EVT-1", "MKT-A", 3, 8)  # Change=5
        items = router.pending_items
        assert len(items) == 1
        assert items[0].category == "spread_change"

    def test_none_spreads_ignored(self):
        router = MMAttentionRouter()
        router.on_spread_change("EVT-1", "MKT-A", None, 5)
        assert len(router.pending_items) == 0


class TestMMAttentionRouterFillStorm:
    """Tests for on_fill_storm() signal."""

    def test_fill_storm_emits(self):
        router = MMAttentionRouter()
        router.on_fill_storm("EVT-1", 12, 30.0)
        items = router.pending_items
        assert len(items) == 1
        assert items[0].category == "fill_storm"
        assert items[0].urgency == "immediate"
        assert items[0].score == 80.0


class TestMMAttentionRouterDedup:
    """Tests for dedup and cooldown behavior."""

    def test_same_key_deduped(self):
        router = MMAttentionRouter()
        router.on_fill("EVT-1", "MKT-A", "bid", 42, 5, 5)
        router.on_fill("EVT-1", "MKT-A", "bid", 43, 3, 8)
        items = router.pending_items
        # Second one should be deduped (5s cooldown for fills)
        assert len(items) == 1

    def test_different_keys_not_deduped(self):
        router = MMAttentionRouter()
        router.on_fill("EVT-1", "MKT-A", "bid", 42, 5, 5)
        router.on_fill("EVT-1", "MKT-B", "bid", 43, 3, 3)
        items = router.pending_items
        assert len(items) == 2

    def test_below_min_score_filtered(self):
        router = MMAttentionRouter(min_score=50.0)
        router.on_spread_change("EVT-1", "MKT-A", 3, 5)  # score = 30 + 2*5 = 40 < 50
        assert len(router.pending_items) == 0


class TestMMAttentionRouterDrain:
    """Tests for drain() behavior."""

    def test_drain_returns_and_clears(self):
        router = MMAttentionRouter()
        router.on_fill("EVT-1", "MKT-A", "bid", 42, 5, 5)
        router.on_fill_storm("EVT-1", 12, 30.0)
        items = router.drain()
        assert len(items) == 2
        # After drain, pending should be empty
        assert len(router.pending_items) == 0

    def test_drain_clears_notify(self):
        router = MMAttentionRouter()
        router.on_fill("EVT-1", "MKT-A", "bid", 42, 5, 5)
        assert router.notify_event.is_set()
        router.drain()
        assert not router.notify_event.is_set()

    def test_drain_sorted_by_score(self):
        router = MMAttentionRouter()
        router.on_fill("EVT-1", "MKT-A", "bid", 42, 1, 1)       # score ~42
        router.on_fill_storm("EVT-1", 12, 30.0)                    # score 80
        items = router.drain()
        assert items[0].score >= items[1].score

    def test_expired_items_removed(self):
        router = MMAttentionRouter()
        # Manually add expired item
        item = MMAttentionItem(
            event_ticker="EVT-1",
            category="fill",
            score=50.0,
            created_at=time.time() - 200,
            ttl_seconds=60.0,
        )
        router._items.append(item)
        assert len(router.pending_items) == 0  # Expired items filtered

    def test_max_items_trimmed(self):
        router = MMAttentionRouter(max_items=3)
        # Add 5 fills with different market tickers to avoid dedup
        for i in range(5):
            router.on_fill("EVT-1", f"MKT-{i}", "bid", 42, 5, 5)
        assert len(router._items) <= 3
