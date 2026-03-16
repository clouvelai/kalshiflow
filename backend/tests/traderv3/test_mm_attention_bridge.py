"""Tests for MMAttentionRouter → Captain AttentionRouter bridge."""

import asyncio
import time
from unittest.mock import MagicMock

import pytest

from kalshiflow_rl.traderv3.market_maker.attention import MMAttentionRouter
from kalshiflow_rl.traderv3.market_maker.models import MMAttentionItem
from kalshiflow_rl.traderv3.single_arb.models import AttentionItem


class TestMMAttentionBridge:
    def test_bridge_called_on_emit(self):
        """When _captain_inject is set, _emit() should create and inject a Captain AttentionItem."""
        injected = []

        def capture_inject(item):
            injected.append(item)

        router = MMAttentionRouter(min_score=0, captain_inject=capture_inject)
        router.on_fill("EVT-1", "MKT-A", "yes", 50, 5, 10)

        assert len(injected) == 1
        item = injected[0]
        assert isinstance(item, AttentionItem)
        assert item.event_ticker == "EVT-1"
        assert item.category == "mm_fill"
        assert "[MM]" in item.summary

    def test_no_bridge_when_inject_is_none(self):
        """Without captain_inject, signals should still work locally."""
        router = MMAttentionRouter(min_score=0)
        router.on_fill("EVT-1", "MKT-A", "yes", 50, 5, 10)

        items = router.pending_items
        assert len(items) >= 1
        assert items[0].category == "fill"

    def test_bridge_preserves_urgency(self):
        """Urgency from MMAttentionItem should carry over to Captain AttentionItem."""
        injected = []

        router = MMAttentionRouter(min_score=0, captain_inject=lambda i: injected.append(i))

        # VPIN spike with high urgency
        router.on_vpin_spike("EVT-1", "MKT-A", 0.96, 0.95)

        assert len(injected) == 1
        assert injected[0].urgency == "immediate"
        assert injected[0].category == "mm_vpin_spike"

    def test_bridge_categories_prefixed_with_mm(self):
        """All bridged categories should be prefixed with mm_."""
        injected = []

        router = MMAttentionRouter(min_score=0, captain_inject=lambda i: injected.append(i))

        router.on_fill("EVT-1", "MKT-A", "yes", 50, 5, 10)
        router.on_spread_change("EVT-1", "MKT-A", 3, 8)
        router.on_inventory_warning("EVT-1", "MKT-A", 90, 100)

        categories = {i.category for i in injected}
        for cat in categories:
            assert cat.startswith("mm_"), f"Category {cat} not prefixed with mm_"

    def test_bridge_error_does_not_break_emit(self):
        """If captain_inject raises, _emit should still work for local items."""
        def bad_inject(item):
            raise RuntimeError("inject failed")

        router = MMAttentionRouter(min_score=0, captain_inject=bad_inject)
        router.on_fill("EVT-1", "MKT-A", "yes", 50, 5, 10)

        # Local items should still be tracked
        items = router.pending_items
        assert len(items) >= 1

    def test_dedup_still_works_with_bridge(self):
        """Dedup should prevent duplicate bridge calls."""
        injected = []

        router = MMAttentionRouter(min_score=0, captain_inject=lambda i: injected.append(i))

        # Same fill signal twice — dedup should block the second
        router.on_fill("EVT-1", "MKT-A", "yes", 50, 5, 10)
        router.on_fill("EVT-1", "MKT-A", "yes", 50, 5, 10)

        # Fill cooldown is 5s, so second should be deduped
        assert len(injected) == 1


class TestMMAttentionRouterInit:
    def test_captain_inject_stored(self):
        cb = MagicMock()
        router = MMAttentionRouter(captain_inject=cb)
        assert router._captain_inject is cb

    def test_default_no_inject(self):
        router = MMAttentionRouter()
        assert router._captain_inject is None
