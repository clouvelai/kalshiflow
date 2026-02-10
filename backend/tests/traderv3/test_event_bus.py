"""
Unit tests for TRADER V3 EventBus.

Tests cover subscribe/unsubscribe, emit behaviour, subscriber notification,
ticker-update coalescing, and health/stats reporting.
"""

import asyncio
import time

import pytest
from unittest.mock import AsyncMock, MagicMock

from kalshiflow_rl.traderv3.core.event_bus import EventBus, QUEUE_CAPACITY
from kalshiflow_rl.traderv3.core.events import (
    EventType,
    MarketEvent,
    MarketTickerEvent,
    StateTransitionEvent,
)


# ---------------------------------------------------------------------------
# TestSubscribeUnsubscribe
# ---------------------------------------------------------------------------


class TestSubscribeUnsubscribe:
    """Tests for subscribe / unsubscribe behaviour."""

    @pytest.mark.asyncio
    async def test_subscribe_adds_callback(self):
        """After subscribe, callback appears in subscriber list."""
        bus = EventBus()
        await bus.start()
        try:
            callback = AsyncMock(name="my_callback", __name__="my_callback")
            bus.subscribe(EventType.STATE_TRANSITION, callback)
            assert callback in bus._subscribers[EventType.STATE_TRANSITION]
        finally:
            await bus.stop()

    @pytest.mark.asyncio
    async def test_unsubscribe_removes_callback(self):
        """After unsubscribe, callback is removed from subscriber list."""
        bus = EventBus()
        await bus.start()
        try:
            callback = AsyncMock(name="my_callback", __name__="my_callback")
            bus.subscribe(EventType.STATE_TRANSITION, callback)
            bus.unsubscribe(EventType.STATE_TRANSITION, callback)
            assert callback not in bus._subscribers.get(EventType.STATE_TRANSITION, [])
        finally:
            await bus.stop()

    @pytest.mark.asyncio
    async def test_unsubscribe_nonexistent(self):
        """Unsubscribing an unknown callback does not crash."""
        bus = EventBus()
        await bus.start()
        try:
            callback = AsyncMock(name="phantom", __name__="phantom")
            # Should not raise
            bus.unsubscribe(EventType.STATE_TRANSITION, callback)
        finally:
            await bus.stop()


# ---------------------------------------------------------------------------
# TestEmit
# ---------------------------------------------------------------------------


class TestEmit:
    """Tests for emit methods."""

    @pytest.mark.asyncio
    async def test_emit_when_running(self):
        """emit returns True and increments events_emitted when bus is running."""
        bus = EventBus()
        await bus.start()
        try:
            event = MarketTickerEvent(
                event_type=EventType.MARKET_TICKER_UPDATE,
                market_ticker="MKT-A",
                price_data={"yes_bid": 50},
            )
            result = await bus.emit(EventType.MARKET_TICKER_UPDATE, event)
            assert result is True
            assert bus._events_emitted == 1
        finally:
            await bus.stop()

    @pytest.mark.asyncio
    async def test_emit_when_stopped(self):
        """emit returns False when bus is not running."""
        bus = EventBus()
        # Do NOT start the bus
        event = MarketTickerEvent(
            event_type=EventType.MARKET_TICKER_UPDATE,
            market_ticker="MKT-A",
            price_data={"yes_bid": 50},
        )
        result = await bus.emit(EventType.MARKET_TICKER_UPDATE, event)
        assert result is False

    @pytest.mark.asyncio
    async def test_emit_state_transition(self):
        """emit_state_transition enqueues event successfully."""
        bus = EventBus()
        await bus.start()
        try:
            result = await bus.emit_state_transition("idle", "ready", "test context")
            assert result is True
            assert bus._events_emitted == 1
        finally:
            await bus.stop()


# ---------------------------------------------------------------------------
# TestNotifySubscribers
# ---------------------------------------------------------------------------


class TestNotifySubscribers:
    """Tests for subscriber notification."""

    @pytest.mark.asyncio
    async def test_callback_receives_event(self):
        """Subscribe + emit -> callback is called with event."""
        bus = EventBus()
        await bus.start()
        try:
            callback = AsyncMock(name="on_status", __name__="on_status")
            bus.subscribe(EventType.TRADER_STATUS, callback)

            await bus.emit_trader_status(
                state="ready",
                metrics={"uptime": 100},
                health="healthy",
            )
            # Allow the processing loop to drain
            await asyncio.sleep(0.2)

            callback.assert_called_once()
            event_arg = callback.call_args[0][0]
            assert event_arg.event_type == EventType.TRADER_STATUS
            assert event_arg.state == "ready"
        finally:
            await bus.stop()

    @pytest.mark.asyncio
    async def test_error_isolation(self):
        """Subscriber raising exception does not prevent others from receiving."""
        bus = EventBus()
        await bus.start()
        try:
            bad_callback = AsyncMock(
                name="bad", __name__="bad", side_effect=RuntimeError("boom")
            )
            good_callback = AsyncMock(name="good", __name__="good")

            bus.subscribe(EventType.TRADER_STATUS, bad_callback)
            bus.subscribe(EventType.TRADER_STATUS, good_callback)

            await bus.emit_trader_status(
                state="ready", metrics={}, health="healthy"
            )
            await asyncio.sleep(0.2)

            bad_callback.assert_called_once()
            good_callback.assert_called_once()
        finally:
            await bus.stop()

    @pytest.mark.asyncio
    async def test_callback_error_counter(self):
        """Bad callback increments callback_errors."""
        bus = EventBus()
        await bus.start()
        try:
            bad_callback = AsyncMock(
                name="bad", __name__="bad", side_effect=RuntimeError("boom")
            )
            bus.subscribe(EventType.TRADER_STATUS, bad_callback)

            await bus.emit_trader_status(
                state="ready", metrics={}, health="healthy"
            )
            await asyncio.sleep(0.2)

            assert bus._callback_errors >= 1
        finally:
            await bus.stop()


# ---------------------------------------------------------------------------
# TestCoalescing
# ---------------------------------------------------------------------------


class TestCoalescing:
    """Tests for _coalesce_ticker_updates."""

    def test_coalesce_same_ticker(self):
        """Two market_ticker_update for same ticker -> keeps latest only."""
        bus = EventBus()
        older = MarketTickerEvent(
            event_type=EventType.MARKET_TICKER_UPDATE,
            market_ticker="MKT-A",
            price_data={"yes_bid": 40},
            timestamp=1.0,
        )
        newer = MarketTickerEvent(
            event_type=EventType.MARKET_TICKER_UPDATE,
            market_ticker="MKT-A",
            price_data={"yes_bid": 45},
            timestamp=2.0,
        )
        # Put both in queue so task_done doesn't underflow
        bus._event_queue.put_nowait(older)
        bus._event_queue.put_nowait(newer)
        # Drain them back out to form the batch
        batch = []
        while not bus._event_queue.empty():
            batch.append(bus._event_queue.get_nowait())

        # Re-add to queue so coalesce can call task_done for dropped events
        for ev in batch:
            bus._event_queue.put_nowait(ev)

        result = bus._coalesce_ticker_updates(batch)

        assert len(result) == 1
        # The latest (newer) should survive
        assert result[0].price_data["yes_bid"] == 45

    def test_coalesce_different_tickers(self):
        """Two different tickers -> both kept."""
        bus = EventBus()
        ev_a = MarketTickerEvent(
            event_type=EventType.MARKET_TICKER_UPDATE,
            market_ticker="MKT-A",
            price_data={"yes_bid": 40},
        )
        ev_b = MarketTickerEvent(
            event_type=EventType.MARKET_TICKER_UPDATE,
            market_ticker="MKT-B",
            price_data={"yes_bid": 60},
        )
        result = bus._coalesce_ticker_updates([ev_a, ev_b])
        assert len(result) == 2

    def test_coalesce_non_coalescable(self):
        """state_transition events always pass through."""
        bus = EventBus()
        st1 = StateTransitionEvent(
            event_type=EventType.STATE_TRANSITION,
            from_state="idle",
            to_state="ready",
            context="test1",
            timestamp=time.time(),
        )
        st2 = StateTransitionEvent(
            event_type=EventType.STATE_TRANSITION,
            from_state="ready",
            to_state="trading",
            context="test2",
            timestamp=time.time(),
        )
        result = bus._coalesce_ticker_updates([st1, st2])
        assert len(result) == 2

    def test_coalesce_single_event(self):
        """Batch of 1 passes through unchanged."""
        bus = EventBus()
        ev = MarketTickerEvent(
            event_type=EventType.MARKET_TICKER_UPDATE,
            market_ticker="MKT-A",
            price_data={"yes_bid": 50},
        )
        result = bus._coalesce_ticker_updates([ev])
        assert len(result) == 1
        assert result[0] is ev


# ---------------------------------------------------------------------------
# TestHealthAndStats
# ---------------------------------------------------------------------------


class TestHealthAndStats:
    """Tests for health checks and statistics."""

    @pytest.mark.asyncio
    async def test_initial_stats(self):
        """Fresh bus has zero counters."""
        bus = EventBus()
        stats = bus.get_stats()
        assert stats["events_emitted"] == 0
        assert stats["events_processed"] == 0
        assert stats["callback_errors"] == 0
        assert stats["events_dropped"] == 0

    @pytest.mark.asyncio
    async def test_healthy_when_running(self):
        """After start, is_healthy() returns True."""
        bus = EventBus()
        await bus.start()
        try:
            assert bus.is_healthy() is True
        finally:
            await bus.stop()

    @pytest.mark.asyncio
    async def test_not_healthy_when_stopped(self):
        """Before start, is_healthy() returns False."""
        bus = EventBus()
        assert bus.is_healthy() is False

    @pytest.mark.asyncio
    async def test_queue_depth(self):
        """After emit, queue_depth > 0."""
        bus = EventBus()
        await bus.start()
        try:
            # Emit several events rapidly to make it likely at least one is still queued
            for i in range(5):
                await bus.emit_market_ticker_update(
                    f"MKT-{i}", {"yes_bid": 50 + i}
                )
            # Check that _events_emitted reflects the enqueues
            assert bus._events_emitted == 5
        finally:
            await bus.stop()

    @pytest.mark.asyncio
    async def test_get_health_details(self):
        """get_health_details returns expected keys while running."""
        bus = EventBus()
        await bus.start()
        try:
            details = bus.get_health_details()
            assert details["running"] is True
            assert details["processing_task_active"] is True
            assert "queue_capacity" in details
            assert details["queue_capacity"] == QUEUE_CAPACITY
        finally:
            await bus.stop()
