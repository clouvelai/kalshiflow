"""Unit tests for extended lifecycle handlers in EventLifecycleService.

Tests cover:
  1. _handle_activated transitions PENDING -> ACTIVE
  2. _handle_activated emits MARKET_ACTIVATED event
  3. _handle_activated updates parent TrackedEvent status
  4. _handle_activated ignores untracked markets
  5. _handle_activated ignores already-ACTIVE markets
  6. _handle_deactivated transitions ACTIVE -> DEACTIVATED
  7. _handle_deactivated ignores non-ACTIVE markets
  8. _handle_close_date_updated updates close_ts
  9. _handle_close_date_updated updates parent event latest_close_ts
  10. _handle_close_date_updated ignores zero close_ts
  11. _handle_created sets PENDING when open_ts in future
  12. _handle_created sets ACTIVE when open_ts in past
  13. _handle_created creates parent TrackedEvent
  14. All 6 lifecycle types are routed correctly
  15. _handle_settled updates status
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kalshiflow_rl.traderv3.state.tracked_markets import TrackedMarket, TrackedMarketsState, MarketStatus
from kalshiflow_rl.traderv3.state.tracked_events import TrackedEventsState, EventStatus
from kalshiflow_rl.traderv3.services.event_lifecycle_service import EventLifecycleService


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

def _make_event_bus() -> MagicMock:
    """Create a mock EventBus with all required async methods."""
    bus = MagicMock()
    bus.subscribe_to_market_lifecycle = AsyncMock()
    bus.emit_market_tracked = AsyncMock(return_value=True)
    bus.emit_market_activated = AsyncMock(return_value=True)
    bus.emit_market_determined = AsyncMock(return_value=True)
    bus.unsubscribe = MagicMock()
    return bus


def _make_tracked_markets(markets: Dict[str, TrackedMarket] | None = None) -> TrackedMarketsState:
    """Create a mock TrackedMarketsState."""
    state = MagicMock(spec=TrackedMarketsState)
    _markets = markets or {}

    state.is_tracked = MagicMock(side_effect=lambda t: t in _markets)
    state.get_market = MagicMock(side_effect=lambda t: _markets.get(t))
    state.add_market = AsyncMock(return_value=True)
    state.update_status = AsyncMock(return_value=True)
    state.update_market = AsyncMock(return_value=True)
    state.at_capacity = MagicMock(return_value=False)
    state.record_category_rejection = MagicMock()
    return state


def _make_tracked_events() -> TrackedEventsState:
    """Create a mock TrackedEventsState."""
    state = MagicMock(spec=TrackedEventsState)
    state.upsert_event = AsyncMock(return_value=True)
    state.add_market_to_event = AsyncMock(return_value=True)
    state.update_event_status = AsyncMock(return_value=True)
    state.update_close_ts = AsyncMock(return_value=True)
    return state


def _make_trading_client() -> MagicMock:
    """Create a mock trading client."""
    client = MagicMock()
    client.get_market = AsyncMock(return_value=None)
    return client


def _make_db() -> MagicMock:
    """Create a mock RLDatabase."""
    db = MagicMock()
    db.insert_lifecycle_event = AsyncMock()
    return db


def _make_market(
    ticker: str = "KXTEST-MKT-A",
    event_ticker: str = "KXTEST-25FEB12",
    status: MarketStatus = MarketStatus.PENDING,
    category: str = "Sports",
    **kwargs,
) -> TrackedMarket:
    return TrackedMarket(
        ticker=ticker,
        event_ticker=event_ticker,
        status=status,
        category=category,
        **kwargs,
    )


@pytest.fixture
def event_bus():
    return _make_event_bus()


@pytest.fixture
def tracked_events():
    return _make_tracked_events()


@pytest.fixture
def trading_client():
    return _make_trading_client()


@pytest.fixture
def db():
    return _make_db()


def _make_service(
    event_bus=None,
    tracked_markets=None,
    tracked_events=None,
    trading_client=None,
    db=None,
    categories=None,
):
    return EventLifecycleService(
        event_bus=event_bus or _make_event_bus(),
        tracked_markets=tracked_markets or _make_tracked_markets(),
        trading_client=trading_client or _make_trading_client(),
        db=db or _make_db(),
        categories=categories or ["sports", "crypto", "politics"],
        tracked_events=tracked_events or _make_tracked_events(),
    )


# ===========================================================================
# _handle_activated
# ===========================================================================


class TestHandleActivated:
    @pytest.mark.asyncio
    async def test_transitions_pending_to_active(self, event_bus, tracked_events, db):
        mkt = _make_market(status=MarketStatus.PENDING)
        tracked_markets = _make_tracked_markets({"KXTEST-MKT-A": mkt})
        svc = _make_service(
            event_bus=event_bus,
            tracked_markets=tracked_markets,
            tracked_events=tracked_events,
            db=db,
        )
        svc._running = True
        await svc._handle_activated("KXTEST-MKT-A", {})
        tracked_markets.update_status.assert_awaited_once_with("KXTEST-MKT-A", MarketStatus.ACTIVE)

    @pytest.mark.asyncio
    async def test_emits_market_activated_event(self, event_bus, tracked_events, db):
        mkt = _make_market(status=MarketStatus.PENDING)
        tracked_markets = _make_tracked_markets({"KXTEST-MKT-A": mkt})
        svc = _make_service(
            event_bus=event_bus,
            tracked_markets=tracked_markets,
            tracked_events=tracked_events,
            db=db,
        )
        svc._running = True
        await svc._handle_activated("KXTEST-MKT-A", {})
        event_bus.emit_market_activated.assert_awaited_once_with(
            market_ticker="KXTEST-MKT-A",
            event_ticker="KXTEST-25FEB12",
            category="Sports",
        )

    @pytest.mark.asyncio
    async def test_updates_parent_event_status(self, event_bus, tracked_events, db):
        mkt = _make_market(status=MarketStatus.PENDING)
        tracked_markets = _make_tracked_markets({"KXTEST-MKT-A": mkt})
        svc = _make_service(
            event_bus=event_bus,
            tracked_markets=tracked_markets,
            tracked_events=tracked_events,
            db=db,
        )
        svc._running = True
        await svc._handle_activated("KXTEST-MKT-A", {})
        tracked_events.update_event_status.assert_awaited_once_with(
            "KXTEST-25FEB12", EventStatus.ACTIVE
        )

    @pytest.mark.asyncio
    async def test_ignores_untracked_markets(self, event_bus, tracked_events, db):
        tracked_markets = _make_tracked_markets({})  # Empty
        svc = _make_service(
            event_bus=event_bus,
            tracked_markets=tracked_markets,
            tracked_events=tracked_events,
            db=db,
        )
        svc._running = True
        await svc._handle_activated("UNKNOWN-MKT", {})
        tracked_markets.update_status.assert_not_awaited()
        event_bus.emit_market_activated.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_ignores_already_active_markets(self, event_bus, tracked_events, db):
        mkt = _make_market(status=MarketStatus.ACTIVE)
        tracked_markets = _make_tracked_markets({"KXTEST-MKT-A": mkt})
        svc = _make_service(
            event_bus=event_bus,
            tracked_markets=tracked_markets,
            tracked_events=tracked_events,
            db=db,
        )
        svc._running = True
        await svc._handle_activated("KXTEST-MKT-A", {})
        tracked_markets.update_status.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_transitions_deactivated_to_active(self, event_bus, tracked_events, db):
        """DEACTIVATED markets can be re-activated."""
        mkt = _make_market(status=MarketStatus.DEACTIVATED)
        tracked_markets = _make_tracked_markets({"KXTEST-MKT-A": mkt})
        svc = _make_service(
            event_bus=event_bus,
            tracked_markets=tracked_markets,
            tracked_events=tracked_events,
            db=db,
        )
        svc._running = True
        await svc._handle_activated("KXTEST-MKT-A", {})
        tracked_markets.update_status.assert_awaited_once_with("KXTEST-MKT-A", MarketStatus.ACTIVE)


# ===========================================================================
# _handle_deactivated
# ===========================================================================


class TestHandleDeactivated:
    @pytest.mark.asyncio
    async def test_transitions_active_to_deactivated(self, event_bus, tracked_events, db):
        mkt = _make_market(status=MarketStatus.ACTIVE)
        tracked_markets = _make_tracked_markets({"KXTEST-MKT-A": mkt})
        svc = _make_service(
            event_bus=event_bus,
            tracked_markets=tracked_markets,
            tracked_events=tracked_events,
            db=db,
        )
        svc._running = True
        await svc._handle_deactivated("KXTEST-MKT-A", {})
        tracked_markets.update_status.assert_awaited_once_with("KXTEST-MKT-A", MarketStatus.DEACTIVATED)

    @pytest.mark.asyncio
    async def test_ignores_non_active_markets(self, event_bus, tracked_events, db):
        mkt = _make_market(status=MarketStatus.PENDING)
        tracked_markets = _make_tracked_markets({"KXTEST-MKT-A": mkt})
        svc = _make_service(
            event_bus=event_bus,
            tracked_markets=tracked_markets,
            tracked_events=tracked_events,
            db=db,
        )
        svc._running = True
        await svc._handle_deactivated("KXTEST-MKT-A", {})
        tracked_markets.update_status.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_ignores_untracked_markets(self, event_bus, tracked_events, db):
        tracked_markets = _make_tracked_markets({})
        svc = _make_service(
            event_bus=event_bus,
            tracked_markets=tracked_markets,
            tracked_events=tracked_events,
            db=db,
        )
        svc._running = True
        await svc._handle_deactivated("UNKNOWN-MKT", {})
        tracked_markets.update_status.assert_not_awaited()


# ===========================================================================
# _handle_close_date_updated
# ===========================================================================


class TestHandleCloseDateUpdated:
    @pytest.mark.asyncio
    async def test_updates_market_close_ts(self, event_bus, tracked_events, db):
        mkt = _make_market(status=MarketStatus.ACTIVE, close_ts=1000)
        tracked_markets = _make_tracked_markets({"KXTEST-MKT-A": mkt})
        svc = _make_service(
            event_bus=event_bus,
            tracked_markets=tracked_markets,
            tracked_events=tracked_events,
            db=db,
        )
        svc._running = True
        await svc._handle_close_date_updated("KXTEST-MKT-A", {"close_ts": 5000})
        tracked_markets.update_market.assert_awaited_once_with("KXTEST-MKT-A", close_ts=5000)

    @pytest.mark.asyncio
    async def test_updates_parent_event_close_ts(self, event_bus, tracked_events, db):
        mkt = _make_market(status=MarketStatus.ACTIVE, event_ticker="EVT-1")
        tracked_markets = _make_tracked_markets({"KXTEST-MKT-A": mkt})
        svc = _make_service(
            event_bus=event_bus,
            tracked_markets=tracked_markets,
            tracked_events=tracked_events,
            db=db,
        )
        svc._running = True
        await svc._handle_close_date_updated("KXTEST-MKT-A", {"close_ts": 5000})
        tracked_events.update_close_ts.assert_awaited_once_with("EVT-1", 5000)

    @pytest.mark.asyncio
    async def test_ignores_zero_close_ts(self, event_bus, tracked_events, db):
        mkt = _make_market(status=MarketStatus.ACTIVE)
        tracked_markets = _make_tracked_markets({"KXTEST-MKT-A": mkt})
        svc = _make_service(
            event_bus=event_bus,
            tracked_markets=tracked_markets,
            tracked_events=tracked_events,
            db=db,
        )
        svc._running = True
        await svc._handle_close_date_updated("KXTEST-MKT-A", {"close_ts": 0})
        tracked_markets.update_market.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_ignores_untracked_markets(self, event_bus, tracked_events, db):
        tracked_markets = _make_tracked_markets({})
        svc = _make_service(
            event_bus=event_bus,
            tracked_markets=tracked_markets,
            tracked_events=tracked_events,
            db=db,
        )
        svc._running = True
        await svc._handle_close_date_updated("UNKNOWN-MKT", {"close_ts": 5000})
        tracked_markets.update_market.assert_not_awaited()


# ===========================================================================
# _handle_created
# ===========================================================================


class TestHandleCreated:
    @pytest.mark.asyncio
    async def test_sets_pending_when_open_ts_in_future(self, event_bus, tracked_events, db):
        future_ts = int(time.time()) + 7200
        trading_client = _make_trading_client()
        trading_client.get_market = AsyncMock(return_value={
            "ticker": "MKT-NEW",
            "event_ticker": "EVT-1",
            "title": "Test",
            "category": "Crypto",
            "mutually_exclusive": True,
        })
        tracked_markets = _make_tracked_markets({})
        svc = _make_service(
            event_bus=event_bus,
            tracked_markets=tracked_markets,
            tracked_events=tracked_events,
            trading_client=trading_client,
            db=db,
        )
        svc._running = True
        await svc._handle_created("MKT-NEW", {"open_ts": future_ts, "close_ts": future_ts + 3600})
        # Market should be added
        tracked_markets.add_market.assert_awaited_once()
        added_market = tracked_markets.add_market.call_args[0][0]
        assert added_market.status == MarketStatus.PENDING

    @pytest.mark.asyncio
    async def test_sets_active_when_open_ts_in_past(self, event_bus, tracked_events, db):
        past_ts = int(time.time()) - 3600
        trading_client = _make_trading_client()
        trading_client.get_market = AsyncMock(return_value={
            "ticker": "MKT-NEW",
            "event_ticker": "EVT-1",
            "title": "Test",
            "category": "Crypto",
            "mutually_exclusive": True,
        })
        tracked_markets = _make_tracked_markets({})
        svc = _make_service(
            event_bus=event_bus,
            tracked_markets=tracked_markets,
            tracked_events=tracked_events,
            trading_client=trading_client,
            db=db,
        )
        svc._running = True
        await svc._handle_created("MKT-NEW", {"open_ts": past_ts, "close_ts": past_ts + 7200})
        tracked_markets.add_market.assert_awaited_once()
        added_market = tracked_markets.add_market.call_args[0][0]
        assert added_market.status == MarketStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_creates_parent_tracked_event(self, event_bus, tracked_events, db):
        past_ts = int(time.time()) - 3600
        trading_client = _make_trading_client()
        trading_client.get_market = AsyncMock(return_value={
            "ticker": "MKT-NEW",
            "event_ticker": "EVT-1",
            "title": "Test",
            "category": "Crypto",
            "series_ticker": "KXTEST",
            "mutually_exclusive": True,
        })
        tracked_markets = _make_tracked_markets({})
        svc = _make_service(
            event_bus=event_bus,
            tracked_markets=tracked_markets,
            tracked_events=tracked_events,
            trading_client=trading_client,
            db=db,
        )
        svc._running = True
        await svc._handle_created("MKT-NEW", {"open_ts": past_ts, "close_ts": past_ts + 7200})
        tracked_events.upsert_event.assert_awaited_once()
        tracked_events.add_market_to_event.assert_awaited_once_with("EVT-1", "MKT-NEW")


# ===========================================================================
# _handle_settled
# ===========================================================================


class TestHandleSettled:
    @pytest.mark.asyncio
    async def test_updates_status_to_settled(self, event_bus, tracked_events, db):
        mkt = _make_market(status=MarketStatus.DETERMINED)
        tracked_markets = _make_tracked_markets({"KXTEST-MKT-A": mkt})
        svc = _make_service(
            event_bus=event_bus,
            tracked_markets=tracked_markets,
            tracked_events=tracked_events,
            db=db,
        )
        svc._running = True
        await svc._handle_settled("KXTEST-MKT-A", {})
        tracked_markets.update_status.assert_awaited_once()
        call_args = tracked_markets.update_status.call_args
        assert call_args[0][1] == MarketStatus.SETTLED

    @pytest.mark.asyncio
    async def test_ignores_untracked_markets(self, event_bus, tracked_events, db):
        tracked_markets = _make_tracked_markets({})
        svc = _make_service(
            event_bus=event_bus,
            tracked_markets=tracked_markets,
            tracked_events=tracked_events,
            db=db,
        )
        svc._running = True
        await svc._handle_settled("UNKNOWN-MKT", {})
        tracked_markets.update_status.assert_not_awaited()


# ===========================================================================
# Lifecycle event routing
# ===========================================================================


class TestEventRouting:
    @pytest.mark.asyncio
    async def test_all_6_types_routed(self, event_bus, tracked_events, db):
        """Verify that all 6 lifecycle types are handled by _handle_lifecycle_event."""
        mkt = _make_market(status=MarketStatus.ACTIVE)
        tracked_markets = _make_tracked_markets({"KXTEST-MKT-A": mkt})
        svc = _make_service(
            event_bus=event_bus,
            tracked_markets=tracked_markets,
            tracked_events=tracked_events,
            db=db,
        )
        svc._running = True

        # Patch individual handlers to verify routing
        svc._handle_created = AsyncMock()
        svc._handle_activated = AsyncMock()
        svc._handle_deactivated = AsyncMock()
        svc._handle_close_date_updated = AsyncMock()
        svc._handle_determined = AsyncMock()
        svc._handle_settled = AsyncMock()

        # Create mock lifecycle events for each type
        from kalshiflow_rl.traderv3.core.events import MarketLifecycleEvent

        lifecycle_types = [
            "created", "activated", "deactivated",
            "close_date_updated", "determined", "settled",
        ]
        handlers = [
            svc._handle_created, svc._handle_activated, svc._handle_deactivated,
            svc._handle_close_date_updated, svc._handle_determined, svc._handle_settled,
        ]

        for ltype, handler in zip(lifecycle_types, handlers):
            event = MarketLifecycleEvent(
                market_ticker="KXTEST-MKT-A",
                lifecycle_event_type=ltype,
                payload={},
            )
            await svc._handle_lifecycle_event(event)
            handler.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_unknown_type_does_not_crash(self, event_bus, tracked_events, db):
        """Unknown lifecycle type is stored but doesn't crash."""
        tracked_markets = _make_tracked_markets({})
        svc = _make_service(
            event_bus=event_bus,
            tracked_markets=tracked_markets,
            tracked_events=tracked_events,
            db=db,
        )
        svc._running = True

        from kalshiflow_rl.traderv3.core.events import MarketLifecycleEvent
        event = MarketLifecycleEvent(
            market_ticker="KXTEST-MKT-A",
            lifecycle_event_type="unknown_type",
            payload={},
        )
        # Should not raise
        await svc._handle_lifecycle_event(event)
        # Audit trail should still be stored
        db.insert_lifecycle_event.assert_awaited_once()
