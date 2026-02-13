"""Unit tests for LifecyclePersistenceService.

Tests cover:
  1. load_from_db returns events/markets from mocked DB
  2. load_from_db handles DB errors gracefully (returns empty lists)
  3. persist_event creates fire-and-forget task
  4. persist_market creates fire-and-forget task
  5. deactivate_event/deactivate_market create tasks
  6. start() enables writes, stop() flushes pending
  7. persist calls are no-ops when not running
  8. flush() awaits all pending tasks
  9. get_stats() returns correct counts
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kalshiflow_rl.traderv3.services.lifecycle_persistence_service import (
    LifecyclePersistenceService,
)


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

def _make_mock_db(**overrides):
    """Create a mock RLDatabase with preset async returns."""
    db = MagicMock()
    db.get_active_tracked_events = AsyncMock(
        return_value=overrides.get("events", [{"event_ticker": "E1", "status": "active"}])
    )
    db.get_active_tracked_markets_v2 = AsyncMock(
        return_value=overrides.get("markets", [{"ticker": "MKT-A", "status": "active"}])
    )
    db.upsert_tracked_event = AsyncMock()
    db.upsert_tracked_market_v2 = AsyncMock()
    db.deactivate_tracked_event = AsyncMock()
    db.deactivate_tracked_market_v2 = AsyncMock()
    return db


@dataclass
class MockTrackedEvent:
    """Minimal mock for TrackedEvent used by persist_event."""
    event_ticker: str = "KXTEST-25FEB12"

    def to_dict(self) -> Dict[str, Any]:
        return {"event_ticker": self.event_ticker, "status": "active"}


@dataclass
class MockTrackedMarket:
    """Minimal mock for TrackedMarket used by persist_market."""
    ticker: str = "KXTEST-MKT-A"
    event_ticker: str = "KXTEST-25FEB12"
    title: str = "Test Market"
    category: str = "Sports"
    status: MagicMock = field(default_factory=lambda: MagicMock(value="active"))
    open_ts: int = 1000
    close_ts: int = 2000
    determined_ts: Optional[int] = None
    settled_ts: Optional[int] = None
    discovery_source: str = "lifecycle_ws"
    market_info: Dict[str, Any] = field(default_factory=dict)


@pytest.fixture
def db():
    return _make_mock_db()


@pytest.fixture
def service(db):
    return LifecyclePersistenceService(db=db)


# ===========================================================================
# load_from_db
# ===========================================================================


class TestLoadFromDb:
    @pytest.mark.asyncio
    async def test_returns_events_and_markets(self, service, db):
        events, markets = await service.load_from_db()
        assert len(events) == 1
        assert events[0]["event_ticker"] == "E1"
        assert len(markets) == 1
        assert markets[0]["ticker"] == "MKT-A"

    @pytest.mark.asyncio
    async def test_calls_db_methods(self, service, db):
        await service.load_from_db()
        db.get_active_tracked_events.assert_awaited_once()
        db.get_active_tracked_markets_v2.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_handles_db_error_gracefully(self):
        db = _make_mock_db()
        db.get_active_tracked_events = AsyncMock(side_effect=Exception("DB down"))
        svc = LifecyclePersistenceService(db=db)
        events, markets = await svc.load_from_db()
        assert events == []
        assert markets == []

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_data(self):
        db = _make_mock_db(events=[], markets=[])
        svc = LifecyclePersistenceService(db=db)
        events, markets = await svc.load_from_db()
        assert events == []
        assert markets == []


# ===========================================================================
# persist_event / persist_market
# ===========================================================================


class TestPersistOperations:
    @pytest.mark.asyncio
    async def test_persist_event_creates_task_when_running(self, service, db):
        await service.start()
        event = MockTrackedEvent()
        service.persist_event(event)
        # Let the fire-and-forget task run
        await asyncio.sleep(0.05)
        db.upsert_tracked_event.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_persist_market_creates_task_when_running(self, service, db):
        await service.start()
        market = MockTrackedMarket()
        service.persist_market(market)
        await asyncio.sleep(0.05)
        db.upsert_tracked_market_v2.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_persist_event_noop_when_not_running(self, service, db):
        event = MockTrackedEvent()
        service.persist_event(event)
        await asyncio.sleep(0.05)
        db.upsert_tracked_event.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_persist_market_noop_when_not_running(self, service, db):
        market = MockTrackedMarket()
        service.persist_market(market)
        await asyncio.sleep(0.05)
        db.upsert_tracked_market_v2.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_persist_event_handles_db_error(self, db):
        db.upsert_tracked_event = AsyncMock(side_effect=Exception("write fail"))
        svc = LifecyclePersistenceService(db=db)
        await svc.start()
        svc.persist_event(MockTrackedEvent())
        await asyncio.sleep(0.05)
        # Should not raise, error is logged
        stats = svc.get_stats()
        assert stats["error_count"] == 1


# ===========================================================================
# deactivate_event / deactivate_market
# ===========================================================================


class TestDeactivateOperations:
    @pytest.mark.asyncio
    async def test_deactivate_event_creates_task(self, service, db):
        await service.start()
        service.deactivate_event("KXTEST-25FEB12")
        await asyncio.sleep(0.05)
        db.deactivate_tracked_event.assert_awaited_once_with("KXTEST-25FEB12")

    @pytest.mark.asyncio
    async def test_deactivate_market_creates_task(self, service, db):
        await service.start()
        service.deactivate_market("MKT-A")
        await asyncio.sleep(0.05)
        db.deactivate_tracked_market_v2.assert_awaited_once_with("MKT-A")

    @pytest.mark.asyncio
    async def test_deactivate_noop_when_not_running(self, service, db):
        service.deactivate_event("KXTEST-25FEB12")
        service.deactivate_market("MKT-A")
        await asyncio.sleep(0.05)
        db.deactivate_tracked_event.assert_not_awaited()
        db.deactivate_tracked_market_v2.assert_not_awaited()


# ===========================================================================
# start / stop / flush
# ===========================================================================


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_start_enables_running(self, service):
        await service.start()
        assert service.get_stats()["running"] is True

    @pytest.mark.asyncio
    async def test_stop_disables_running_and_flushes(self, service, db):
        await service.start()
        service.persist_event(MockTrackedEvent())
        await service.stop()
        assert service.get_stats()["running"] is False
        # Flush should have awaited the pending write
        db.upsert_tracked_event.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_flush_awaits_all_pending(self, service, db):
        await service.start()
        # Create multiple writes
        service.persist_event(MockTrackedEvent(event_ticker="E1"))
        service.persist_event(MockTrackedEvent(event_ticker="E2"))
        await service.flush()
        assert db.upsert_tracked_event.await_count == 2


# ===========================================================================
# get_stats
# ===========================================================================


class TestGetStats:
    @pytest.mark.asyncio
    async def test_initial_stats(self, service):
        stats = service.get_stats()
        assert stats["running"] is False
        assert stats["write_count"] == 0
        assert stats["error_count"] == 0
        assert stats["pending_writes"] == 0

    @pytest.mark.asyncio
    async def test_stats_after_writes(self, service, db):
        await service.start()
        service.persist_event(MockTrackedEvent())
        await asyncio.sleep(0.05)
        stats = service.get_stats()
        assert stats["write_count"] == 1
        assert stats["error_count"] == 0
