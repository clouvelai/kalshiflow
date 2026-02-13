"""Unit tests for AttentionRouter.

T1/T2 tests — pure Python + async. No network calls.
Tests scoring, emission, deduplication, draining, and expiry.

Note: Edge thresholds were lowered to match Kalshi reality:
  EDGE_MIN_CENTS=0.5, EDGE_HIGH_CENTS=3.0, SCORE_EMIT_THRESHOLD=35
"""

import asyncio
import time

import pytest
from unittest.mock import AsyncMock, MagicMock

from tests.traderv3.conftest import make_event_meta, make_index, make_market_meta, make_config

from kalshiflow_rl.traderv3.single_arb.attention import (
    AttentionRouter,
    SCORE_EMIT_THRESHOLD,
    URGENCY_HIGH_THRESHOLD,
    URGENCY_IMMEDIATE_THRESHOLD,
    WEIGHT_EDGE_MAGNITUDE,
    EDGE_MIN_CENTS,
    EDGE_HIGH_CENTS,
    ACKNOWLEDGE_COOLDOWN,
    ACKNOWLEDGE_SCORE_INCREASE_PCT,
)
from kalshiflow_rl.traderv3.single_arb.models import AttentionItem


# ===========================================================================
# Helpers
# ===========================================================================

def make_router(events=None, config=None, auto_action_callback=None):
    """Create an AttentionRouter with a pre-populated index."""
    if events is None:
        events = [make_event_meta("EVT-1", n_markets=3)]
    index = make_index(events=events)
    config = config or make_config()
    return AttentionRouter(
        index=index,
        config=config,
        auto_action_callback=auto_action_callback,
    )


# ===========================================================================
# TestAttentionItem
# ===========================================================================

class TestAttentionItem:
    def test_dedup_key(self):
        item = AttentionItem(event_ticker="EVT-1", category="arb_opportunity")
        assert item.key == "EVT-1:arb_opportunity"

    def test_is_expired_fresh(self):
        item = AttentionItem(event_ticker="EVT-1", ttl_seconds=120.0)
        assert not item.is_expired

    def test_is_expired_stale(self):
        item = AttentionItem(
            event_ticker="EVT-1",
            ttl_seconds=1.0,
            created_at=time.time() - 5.0,
        )
        assert item.is_expired

    def test_to_prompt_immediate(self):
        item = AttentionItem(
            event_ticker="EVT-1",
            urgency="immediate",
            summary="long edge=8.2c",
        )
        text = item.to_prompt()
        assert "[IMMEDIATE]" in text
        assert "EVT-1" in text
        assert "long edge=8.2c" in text

    def test_to_prompt_with_market(self):
        item = AttentionItem(
            event_ticker="EVT-1",
            market_ticker="MKT-A",
            urgency="high",
            summary="vpin spike",
        )
        text = item.to_prompt()
        assert "MKT-A" in text

    def test_to_prompt_auto_handled(self):
        item = AttentionItem(
            event_ticker="EVT-1",
            urgency="normal",
            summary="stop loss",
            data={"auto_handled": "sold 5ct @ 40c"},
        )
        text = item.to_prompt()
        assert "(auto: sold 5ct @ 40c)" in text

    def test_to_dict(self):
        item = AttentionItem(event_ticker="EVT-1", score=65.3)
        d = item.to_dict()
        assert d["event_ticker"] == "EVT-1"
        assert d["score"] == 65.3


# ===========================================================================
# TestAttentionRouterScoring
# ===========================================================================

class TestAttentionRouterScoring:
    def test_no_items_for_no_edge(self):
        """Events with no edge should produce no items."""
        # Default factory: 3 markets with ~33c each, sum ~100, minimal edge
        router = make_router()
        router._evaluate_event("EVT-1")
        # Should have no items (edge too small)
        assert len(router._items) == 0

    def test_items_emitted_for_high_edge(self):
        """Events with significant edge should produce items."""
        # Create event where markets sum to much less than 100 (= positive long edge)
        prices = [
            {"yes_bid": 20, "yes_ask": 22},
            {"yes_bid": 20, "yes_ask": 22},
            {"yes_bid": 20, "yes_ask": 22},
        ]
        event = make_event_meta("EVT-EDGE", n_markets=3, market_prices=prices)
        router = make_router(events=[event])
        router._evaluate_event("EVT-EDGE")
        # Edge should be large enough to emit
        assert len(router._items) > 0

    def test_score_increases_with_edge(self):
        """Higher edge should produce higher score."""
        # Low edge event
        low_prices = [
            {"yes_bid": 30, "yes_ask": 33},
            {"yes_bid": 30, "yes_ask": 33},
            {"yes_bid": 30, "yes_ask": 33},
        ]
        # High edge event
        high_prices = [
            {"yes_bid": 15, "yes_ask": 17},
            {"yes_bid": 15, "yes_ask": 17},
            {"yes_bid": 15, "yes_ask": 17},
        ]
        low_event = make_event_meta("EVT-LOW", n_markets=3, market_prices=low_prices)
        high_event = make_event_meta("EVT-HIGH", n_markets=3, market_prices=high_prices)

        low_router = make_router(events=[low_event])
        high_router = make_router(events=[high_event])

        low_router._evaluate_event("EVT-LOW")
        high_router._evaluate_event("EVT-HIGH")

        low_scores = [i.score for i in low_router._items.values()]
        high_scores = [i.score for i in high_router._items.values()]

        if low_scores and high_scores:
            assert max(high_scores) >= max(low_scores)

    def test_position_pnl_scoring(self):
        """Positions with bad P&L + edge should emit position_risk items.

        P&L score (20) alone is below emit threshold (40), so we need edge
        from market pricing to contribute additional score.
        """
        # Create event with clear edge to get edge_score + pnl_score > 40
        prices = [
            {"yes_bid": 20, "yes_ask": 22},
            {"yes_bid": 20, "yes_ask": 22},
            {"yes_bid": 20, "yes_ask": 22},
        ]
        event = make_event_meta("EVT-POS", n_markets=3, market_prices=prices)
        router = make_router(events=[event])
        # Add a position with bad P&L
        router.update_positions([{
            "ticker": "EVT-POS-MKT-A",
            "event_ticker": "EVT-POS",
            "side": "yes",
            "quantity": 10,
            "pnl_per_ct": -15,  # Below -12 threshold
        }])
        router._evaluate_event("EVT-POS")
        risk_items = [i for i in router._items.values() if i.category == "position_risk"]
        assert len(risk_items) > 0

    def test_position_profit_scoring(self):
        """Positions with good P&L + edge should emit position_risk items.

        P&L score (20) alone is below emit threshold (40), so edge contributes.
        """
        prices = [
            {"yes_bid": 20, "yes_ask": 22},
            {"yes_bid": 20, "yes_ask": 22},
            {"yes_bid": 20, "yes_ask": 22},
        ]
        event = make_event_meta("EVT-PROF", n_markets=3, market_prices=prices)
        router = make_router(events=[event])
        router.update_positions([{
            "ticker": "EVT-PROF-MKT-A",
            "event_ticker": "EVT-PROF",
            "side": "yes",
            "quantity": 10,
            "pnl_per_ct": 15,  # Above +10 threshold
        }])
        router._evaluate_event("EVT-PROF")
        risk_items = [i for i in router._items.values() if i.category == "position_risk"]
        assert len(risk_items) > 0


# ===========================================================================
# TestAttentionRouterDraining
# ===========================================================================

class TestAttentionRouterDraining:
    def test_drain_returns_and_removes(self):
        """drain_items should return qualifying items and remove them."""
        router = make_router()
        # Manually insert a high-urgency item
        item = AttentionItem(
            event_ticker="EVT-1",
            urgency="immediate",
            category="arb_opportunity",
            score=75.0,
            summary="test",
        )
        router._items[item.key] = item

        drained = router.drain_items(min_urgency="high")
        assert len(drained) == 1
        assert drained[0].event_ticker == "EVT-1"
        # Should be removed
        assert len(router._items) == 0

    def test_drain_filters_by_urgency(self):
        """drain_items should filter by minimum urgency."""
        router = make_router()
        # Insert items of different urgencies
        for urgency, score in [("immediate", 80), ("high", 60), ("normal", 45)]:
            item = AttentionItem(
                event_ticker="EVT-1",
                urgency=urgency,
                category=f"test_{urgency}",
                score=score,
                summary="test",
            )
            router._items[item.key] = item

        # Drain only high+ urgency
        drained = router.drain_items(min_urgency="high")
        assert len(drained) == 2  # immediate + high
        assert len(router._items) == 1  # normal stays

    def test_drain_skips_expired(self):
        """drain_items should not return expired items."""
        router = make_router()
        item = AttentionItem(
            event_ticker="EVT-1",
            urgency="immediate",
            category="arb_opportunity",
            score=75.0,
            summary="test",
            created_at=time.time() - 300,  # Expired
            ttl_seconds=120.0,
        )
        router._items[item.key] = item

        drained = router.drain_items(min_urgency="high")
        assert len(drained) == 0

    def test_pending_items_peeks_without_draining(self):
        """pending_items should return items without removing them."""
        router = make_router()
        item = AttentionItem(
            event_ticker="EVT-1",
            urgency="normal",
            category="edge_emergence",
            score=45.0,
            summary="test",
        )
        router._items[item.key] = item

        pending = router.pending_items()
        assert len(pending) == 1
        # Should still be in the queue
        assert len(router._items) == 1


# ===========================================================================
# TestAttentionRouterDeduplication
# ===========================================================================

class TestAttentionRouterDeduplication:
    def test_higher_score_replaces(self):
        """Item with higher score should replace existing."""
        router = make_router()
        old = AttentionItem(
            event_ticker="EVT-1",
            category="arb_opportunity",
            score=50.0,
            summary="old",
        )
        router._emit_item(old)
        assert router._items[old.key].score == 50.0

        new = AttentionItem(
            event_ticker="EVT-1",
            category="arb_opportunity",
            score=70.0,
            summary="new",
        )
        router._emit_item(new)
        assert router._items[new.key].score == 70.0
        assert router._items[new.key].summary == "new"

    def test_lower_score_merges_data(self):
        """Item with lower score should merge data but keep higher score."""
        router = make_router()
        old = AttentionItem(
            event_ticker="EVT-1",
            category="arb_opportunity",
            score=70.0,
            summary="old",
            data={"edge": 5.0},
        )
        router._emit_item(old)

        new = AttentionItem(
            event_ticker="EVT-1",
            category="arb_opportunity",
            score=50.0,
            summary="new",
            data={"volume": 100},
        )
        router._emit_item(new)

        # Score should stay at 70 but data should be merged
        stored = router._items["EVT-1:arb_opportunity"]
        assert stored.score == 70.0
        assert stored.data.get("volume") == 100

    def test_expired_item_gets_replaced(self):
        """Expired items should always be replaced."""
        router = make_router()
        old = AttentionItem(
            event_ticker="EVT-1",
            category="arb_opportunity",
            score=90.0,
            summary="old",
            created_at=time.time() - 300,
            ttl_seconds=120.0,
        )
        router._items[old.key] = old

        new = AttentionItem(
            event_ticker="EVT-1",
            category="arb_opportunity",
            score=45.0,
            summary="new",
        )
        router._emit_item(new)
        assert router._items[new.key].score == 45.0


# ===========================================================================
# TestAttentionRouterNotification
# ===========================================================================

class TestAttentionRouterNotification:
    def test_high_urgency_sets_notify(self):
        """High-urgency items should set the notification event."""
        router = make_router()
        router._notify.clear()

        item = AttentionItem(
            event_ticker="EVT-1",
            urgency="immediate",
            category="arb_opportunity",
            score=80.0,
            summary="test",
        )
        router._emit_item(item)
        assert router._notify.is_set()

    def test_normal_urgency_does_not_notify(self):
        """Normal-urgency items should not set the notification event."""
        router = make_router()
        router._notify.clear()

        item = AttentionItem(
            event_ticker="EVT-1",
            urgency="normal",
            category="edge_emergence",
            score=45.0,
            summary="test",
        )
        router._emit_item(item)
        assert not router._notify.is_set()


# ===========================================================================
# TestAttentionRouterAutoActionCallback
# ===========================================================================

class TestAttentionRouterAutoActionCallback:
    @pytest.mark.asyncio
    async def test_callback_called_on_emit(self):
        """Auto-action callback should be called when items are emitted."""
        callback = AsyncMock()
        router = make_router(auto_action_callback=callback)

        item = AttentionItem(
            event_ticker="EVT-1",
            urgency="high",
            category="position_risk",
            score=60.0,
            summary="test",
        )
        router._emit_item(item)
        # Callback returns a coroutine, which is scheduled as a task via create_task
        callback.assert_called_once_with(item)
        # Allow the created task to complete
        await asyncio.sleep(0)

    @pytest.mark.asyncio
    async def test_notify_deferred_until_callback_completes(self):
        """Notification should NOT be set until auto_action callback finishes."""
        completed = asyncio.Event()

        async def slow_callback(item):
            # Simulate auto_action processing
            await asyncio.sleep(0.05)
            item.data["auto_handled"] = "stop_loss: sold 5ct"
            completed.set()

        router = make_router(auto_action_callback=slow_callback)
        router._notify.clear()

        item = AttentionItem(
            event_ticker="EVT-1",
            urgency="immediate",
            category="position_risk",
            score=80.0,
            summary="test",
        )
        router._emit_item(item)
        # Immediately after emit, notify should NOT be set (deferred)
        assert not router._notify.is_set()

        # Wait for callback to complete
        await completed.wait()
        await asyncio.sleep(0)  # Let done_callback run
        # Now notify should be set
        assert router._notify.is_set()
        assert item.data.get("auto_handled") == "stop_loss: sold 5ct"

    @pytest.mark.asyncio
    async def test_notify_immediate_without_callback(self):
        """Without auto_action callback, high-urgency items notify immediately."""
        router = make_router(auto_action_callback=None)
        router._notify.clear()

        item = AttentionItem(
            event_ticker="EVT-1",
            urgency="immediate",
            category="arb_opportunity",
            score=80.0,
            summary="test",
        )
        router._emit_item(item)
        assert router._notify.is_set()


# ===========================================================================
# TestAttentionRouterUpdatePositions
# ===========================================================================

class TestAttentionRouterUpdatePositions:
    def test_updates_position_state(self):
        """update_positions should store position data for P&L scoring."""
        router = make_router()
        router.update_positions([
            {"ticker": "MKT-A", "event_ticker": "EVT-1", "side": "yes", "quantity": 10, "pnl_per_ct": -5},
            {"ticker": "MKT-B", "event_ticker": "EVT-1", "side": "no", "quantity": 5, "pnl_per_ct": 3},
        ])
        assert len(router._positions) == 2
        assert router._positions["MKT-A"]["pnl_per_ct"] == -5

    def test_replaces_previous_positions(self):
        """Calling update_positions again should replace, not append."""
        router = make_router()
        router.update_positions([
            {"ticker": "MKT-A", "event_ticker": "EVT-1", "side": "yes", "quantity": 10, "pnl_per_ct": -5},
        ])
        assert len(router._positions) == 1

        router.update_positions([
            {"ticker": "MKT-B", "event_ticker": "EVT-1", "side": "no", "quantity": 5, "pnl_per_ct": 3},
        ])
        assert len(router._positions) == 1
        assert "MKT-B" in router._positions


# ===========================================================================
# TestAttentionRouterLifecycle
# ===========================================================================

class TestAttentionRouterLifecycle:
    @pytest.mark.asyncio
    async def test_start_stop(self):
        """Start and stop should not raise."""
        router = make_router()
        await router.start()
        assert router._running
        await router.stop()
        assert not router._running

    @pytest.mark.asyncio
    async def test_double_start(self):
        """Double start should be idempotent."""
        router = make_router()
        await router.start()
        await router.start()
        assert router._running
        await router.stop()

    def test_subscribe(self):
        """Subscribe should register callbacks on event bus."""
        router = make_router()
        event_bus = MagicMock()
        router.subscribe(event_bus)
        assert event_bus.subscribe.call_count == 4  # ORDERBOOK_SNAPSHOT, DELTA, TRADE, TICKER


# ===========================================================================
# TestInjectItem
# ===========================================================================

class TestInjectItem:
    def test_inject_item_external(self):
        """inject_item should make items visible via drain_items."""
        router = make_router()
        item = AttentionItem(
            event_ticker="EVT-SNIPER",
            category="sniper_execution",
            urgency="high",
            score=55.0,
            summary="sniper long arb 3/3 legs",
            data={"direction": "long", "edge_cents": 5.0},
        )
        router.inject_item(item)

        drained = router.drain_items(min_urgency="high")
        assert len(drained) == 1
        assert drained[0].event_ticker == "EVT-SNIPER"
        assert drained[0].category == "sniper_execution"

    def test_inject_item_triggers_notify(self):
        """inject_item with high urgency should set the notification event."""
        router = make_router()
        router._notify.clear()

        item = AttentionItem(
            event_ticker="EVT-X",
            category="sniper_execution",
            urgency="high",
            score=55.0,
            summary="test",
        )
        router.inject_item(item)
        assert router._notify.is_set()


# ===========================================================================
# TestAcknowledgmentCooldown
# ===========================================================================

class TestAcknowledgmentCooldown:
    def test_drain_records_acknowledgments(self):
        """drain_items should record acknowledged items with timestamps."""
        router = make_router()
        item = AttentionItem(
            event_ticker="EVT-1",
            urgency="immediate",
            category="arb_opportunity",
            score=75.0,
            summary="test",
        )
        router._items[item.key] = item

        drained = router.drain_items(min_urgency="high")
        assert len(drained) == 1
        assert item.key in router._acknowledged
        ack_time, ack_score = router._acknowledged[item.key]
        assert ack_score == 75.0
        assert time.time() - ack_time < 2.0

    def test_suppressed_during_cooldown(self):
        """Re-emitting the same signal within cooldown should be suppressed."""
        router = make_router()
        item1 = AttentionItem(
            event_ticker="EVT-1",
            urgency="immediate",
            category="arb_opportunity",
            score=75.0,
            summary="first",
        )
        router._emit_item(item1)
        assert item1.key in router._items

        # Drain (acknowledges the item)
        router.drain_items(min_urgency="high")
        assert item1.key in router._acknowledged
        assert item1.key not in router._items

        # Re-emit with same score — should be suppressed
        item2 = AttentionItem(
            event_ticker="EVT-1",
            urgency="immediate",
            category="arb_opportunity",
            score=75.0,
            summary="second",
        )
        router._emit_item(item2)
        assert item2.key not in router._items  # Suppressed

    def test_re_emit_after_cooldown_expires(self):
        """Re-emitting after cooldown expires should succeed."""
        router = make_router()
        item = AttentionItem(
            event_ticker="EVT-1",
            urgency="immediate",
            category="arb_opportunity",
            score=75.0,
            summary="test",
        )
        router._emit_item(item)
        router.drain_items(min_urgency="high")

        # Fake expired cooldown
        router._acknowledged[item.key] = (time.time() - ACKNOWLEDGE_COOLDOWN - 1, 75.0)

        # Re-emit — should succeed
        item2 = AttentionItem(
            event_ticker="EVT-1",
            urgency="immediate",
            category="arb_opportunity",
            score=75.0,
            summary="re-emitted",
        )
        router._emit_item(item2)
        assert item2.key in router._items
        assert router._items[item2.key].summary == "re-emitted"

    def test_re_emit_with_material_score_increase(self):
        """Re-emitting with >= 25% score increase should bypass cooldown."""
        router = make_router()
        item = AttentionItem(
            event_ticker="EVT-1",
            urgency="immediate",
            category="arb_opportunity",
            score=60.0,
            summary="original",
        )
        router._emit_item(item)
        router.drain_items(min_urgency="high")

        # Re-emit with 30% higher score (60 * 1.25 = 75, so 76 passes)
        item2 = AttentionItem(
            event_ticker="EVT-1",
            urgency="immediate",
            category="arb_opportunity",
            score=76.0,
            summary="higher score",
        )
        router._emit_item(item2)
        assert item2.key in router._items
        assert router._items[item2.key].score == 76.0
        # Acknowledgment should be cleared
        assert item2.key not in router._acknowledged

    def test_re_emit_with_small_score_increase_suppressed(self):
        """Re-emitting with < 25% score increase should still be suppressed."""
        router = make_router()
        item = AttentionItem(
            event_ticker="EVT-1",
            urgency="immediate",
            category="arb_opportunity",
            score=60.0,
            summary="original",
        )
        router._emit_item(item)
        router.drain_items(min_urgency="high")

        # Re-emit with only 10% higher score (60 * 1.25 = 75, 66 < 75)
        item2 = AttentionItem(
            event_ticker="EVT-1",
            urgency="immediate",
            category="arb_opportunity",
            score=66.0,
            summary="small bump",
        )
        router._emit_item(item2)
        assert item2.key not in router._items  # Still suppressed

    def test_expired_items_not_acknowledged(self):
        """Expired items removed during drain should not be acknowledged."""
        router = make_router()
        expired = AttentionItem(
            event_ticker="EVT-1",
            urgency="immediate",
            category="arb_opportunity",
            score=75.0,
            summary="expired",
            created_at=time.time() - 300,
            ttl_seconds=120.0,
        )
        router._items[expired.key] = expired

        drained = router.drain_items(min_urgency="high")
        assert len(drained) == 0
        # Expired items should NOT be acknowledged
        assert expired.key not in router._acknowledged

    def test_different_categories_independent(self):
        """Cooldown is per-key — different categories for the same event are independent."""
        router = make_router()
        arb_item = AttentionItem(
            event_ticker="EVT-1",
            urgency="immediate",
            category="arb_opportunity",
            score=75.0,
            summary="arb",
        )
        router._emit_item(arb_item)
        router.drain_items(min_urgency="high")

        # Emit a different category — should NOT be suppressed
        risk_item = AttentionItem(
            event_ticker="EVT-1",
            urgency="immediate",
            category="position_risk",
            score=65.0,
            summary="risk",
        )
        router._emit_item(risk_item)
        assert risk_item.key in router._items  # Not suppressed

    def test_stats_include_acknowledged_count(self):
        """get_stats should report acknowledged item count."""
        router = make_router()
        item = AttentionItem(
            event_ticker="EVT-1",
            urgency="immediate",
            category="arb_opportunity",
            score=75.0,
            summary="test",
        )
        router._emit_item(item)
        router.drain_items(min_urgency="high")

        stats = router.get_stats()
        assert stats["acknowledged_items"] == 1
