"""Unit tests for TrackedEventsState and TrackedEvent.

Tests cover:
  1. TrackedEvent to_dict/from_dict round-trip serialization
  2. TrackedEventsState: upsert_event (new + update), add_market_to_event, update_event_status
  3. get_event, get_all, get_active, get_by_status
  4. update_close_ts
  5. get_timeline (chronological milestones with correct ordering)
  6. get_snapshot (complete state with all events)
  7. get_stats (counts by status)
  8. EventStatus enum values
  9. Version tracking (version increments on mutations)
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from kalshiflow_rl.traderv3.state.tracked_events import (
    EventStatus,
    TrackedEvent,
    TrackedEventsState,
)


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

def _make_event(
    event_ticker: str = "KXTEST-25FEB12",
    title: str = "Test Event",
    category: str = "Sports",
    status: EventStatus = EventStatus.PENDING,
    earliest_open_ts: int = 0,
    latest_close_ts: int = 0,
    market_tickers: list | None = None,
    **kwargs,
) -> TrackedEvent:
    return TrackedEvent(
        event_ticker=event_ticker,
        title=title,
        category=category,
        status=status,
        earliest_open_ts=earliest_open_ts,
        latest_close_ts=latest_close_ts,
        market_tickers=market_tickers or [],
        **kwargs,
    )


@pytest.fixture
def state():
    return TrackedEventsState()


# ===========================================================================
# EventStatus enum
# ===========================================================================


class TestEventStatus:
    def test_all_values(self):
        assert EventStatus.PENDING.value == "pending"
        assert EventStatus.ACTIVE.value == "active"
        assert EventStatus.PARTIALLY_DETERMINED.value == "partially_determined"
        assert EventStatus.DETERMINED.value == "determined"
        assert EventStatus.SETTLED.value == "settled"

    def test_enum_count(self):
        assert len(EventStatus) == 5


# ===========================================================================
# TrackedEvent serialization
# ===========================================================================


class TestTrackedEventSerialization:
    def test_to_dict_has_expected_keys(self):
        event = _make_event(market_tickers=["MKT-A", "MKT-B"])
        d = event.to_dict()
        assert d["event_ticker"] == "KXTEST-25FEB12"
        assert d["title"] == "Test Event"
        assert d["category"] == "Sports"
        assert d["status"] == "pending"
        assert d["market_tickers"] == ["MKT-A", "MKT-B"]
        assert d["market_count"] == 2

    def test_to_dict_time_to_close_none_when_zero(self):
        event = _make_event(latest_close_ts=0)
        d = event.to_dict()
        assert d["time_to_close_seconds"] is None

    def test_to_dict_time_to_close_positive_for_future(self):
        future_ts = int(time.time()) + 3600
        event = _make_event(latest_close_ts=future_ts)
        d = event.to_dict()
        assert d["time_to_close_seconds"] is not None
        assert d["time_to_close_seconds"] > 0

    def test_round_trip(self):
        original = _make_event(
            event_ticker="KXNFL-25JAN05",
            title="NFL Game",
            category="Sports",
            series_ticker="KXNFL",
            mutually_exclusive=False,
            status=EventStatus.ACTIVE,
            earliest_open_ts=1000,
            latest_close_ts=2000,
            discovery_source="api",
            market_tickers=["MKT-A", "MKT-B", "MKT-C"],
        )
        d = original.to_dict()
        restored = TrackedEvent.from_dict(d)

        assert restored.event_ticker == original.event_ticker
        assert restored.title == original.title
        assert restored.category == original.category
        assert restored.series_ticker == original.series_ticker
        assert restored.mutually_exclusive == original.mutually_exclusive
        assert restored.status == original.status
        assert restored.earliest_open_ts == original.earliest_open_ts
        assert restored.latest_close_ts == original.latest_close_ts
        assert restored.discovery_source == original.discovery_source
        assert restored.market_tickers == original.market_tickers

    def test_from_dict_invalid_status_falls_back_to_pending(self):
        d = {"event_ticker": "TEST", "status": "bogus_status"}
        event = TrackedEvent.from_dict(d)
        assert event.status == EventStatus.PENDING

    def test_from_dict_market_tickers_as_json_string(self):
        import json
        d = {
            "event_ticker": "TEST",
            "market_tickers": json.dumps(["MKT-A", "MKT-B"]),
        }
        event = TrackedEvent.from_dict(d)
        assert event.market_tickers == ["MKT-A", "MKT-B"]

    def test_from_dict_market_tickers_invalid_string(self):
        d = {"event_ticker": "TEST", "market_tickers": "not-json"}
        event = TrackedEvent.from_dict(d)
        assert event.market_tickers == []

    def test_from_dict_missing_fields_use_defaults(self):
        event = TrackedEvent.from_dict({})
        assert event.event_ticker == ""
        assert event.title == ""
        assert event.status == EventStatus.PENDING
        assert event.market_tickers == []


# ===========================================================================
# TrackedEventsState — upsert_event
# ===========================================================================


class TestUpsertEvent:
    @pytest.mark.asyncio
    async def test_upsert_new_event_returns_true(self, state):
        event = _make_event()
        result = await state.upsert_event(event)
        assert result is True
        assert state.total_count == 1

    @pytest.mark.asyncio
    async def test_upsert_existing_event_returns_false(self, state):
        event = _make_event()
        await state.upsert_event(event)
        result = await state.upsert_event(event)
        assert result is False

    @pytest.mark.asyncio
    async def test_upsert_update_merges_fields(self, state):
        original = _make_event(title="Original", category="", earliest_open_ts=1000, latest_close_ts=2000)
        await state.upsert_event(original)

        updated = _make_event(title="Updated", category="Sports", earliest_open_ts=900, latest_close_ts=3000)
        await state.upsert_event(updated)

        stored = state.get_event("KXTEST-25FEB12")
        assert stored.title == "Updated"
        assert stored.category == "Sports"
        # earliest_open_ts takes the min
        assert stored.earliest_open_ts == 900
        # latest_close_ts takes the max
        assert stored.latest_close_ts == 3000

    @pytest.mark.asyncio
    async def test_upsert_increments_version(self, state):
        v0 = state.version
        await state.upsert_event(_make_event())
        assert state.version == v0 + 1
        await state.upsert_event(_make_event())
        assert state.version == v0 + 2

    @pytest.mark.asyncio
    async def test_upsert_fires_change_callback(self, state):
        callback = MagicMock()
        state.set_change_callback(callback)
        await state.upsert_event(_make_event())
        assert callback.called


# ===========================================================================
# TrackedEventsState — add_market_to_event
# ===========================================================================


class TestAddMarketToEvent:
    @pytest.mark.asyncio
    async def test_add_market_success(self, state):
        await state.upsert_event(_make_event())
        result = await state.add_market_to_event("KXTEST-25FEB12", "MKT-A")
        assert result is True
        event = state.get_event("KXTEST-25FEB12")
        assert "MKT-A" in event.market_tickers

    @pytest.mark.asyncio
    async def test_add_market_nonexistent_event(self, state):
        result = await state.add_market_to_event("NOPE", "MKT-A")
        assert result is False

    @pytest.mark.asyncio
    async def test_add_market_duplicate_returns_false(self, state):
        await state.upsert_event(_make_event(market_tickers=["MKT-A"]))
        result = await state.add_market_to_event("KXTEST-25FEB12", "MKT-A")
        assert result is False

    @pytest.mark.asyncio
    async def test_add_market_increments_version(self, state):
        await state.upsert_event(_make_event())
        v = state.version
        await state.add_market_to_event("KXTEST-25FEB12", "MKT-A")
        assert state.version == v + 1


# ===========================================================================
# TrackedEventsState — update_event_status
# ===========================================================================


class TestUpdateEventStatus:
    @pytest.mark.asyncio
    async def test_update_status_success(self, state):
        await state.upsert_event(_make_event())
        result = await state.update_event_status("KXTEST-25FEB12", EventStatus.ACTIVE)
        assert result is True
        assert state.get_event("KXTEST-25FEB12").status == EventStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_update_status_same_returns_false(self, state):
        await state.upsert_event(_make_event(status=EventStatus.ACTIVE))
        result = await state.update_event_status("KXTEST-25FEB12", EventStatus.ACTIVE)
        assert result is False

    @pytest.mark.asyncio
    async def test_update_status_nonexistent_event(self, state):
        result = await state.update_event_status("NOPE", EventStatus.ACTIVE)
        assert result is False

    @pytest.mark.asyncio
    async def test_update_status_increments_version(self, state):
        await state.upsert_event(_make_event())
        v = state.version
        await state.update_event_status("KXTEST-25FEB12", EventStatus.ACTIVE)
        assert state.version == v + 1


# ===========================================================================
# TrackedEventsState — update_close_ts
# ===========================================================================


class TestUpdateCloseTs:
    @pytest.mark.asyncio
    async def test_update_close_ts_success(self, state):
        await state.upsert_event(_make_event(latest_close_ts=1000))
        result = await state.update_close_ts("KXTEST-25FEB12", 2000)
        assert result is True
        assert state.get_event("KXTEST-25FEB12").latest_close_ts == 2000

    @pytest.mark.asyncio
    async def test_update_close_ts_takes_max(self, state):
        await state.upsert_event(_make_event(latest_close_ts=3000))
        await state.update_close_ts("KXTEST-25FEB12", 2000)
        assert state.get_event("KXTEST-25FEB12").latest_close_ts == 3000

    @pytest.mark.asyncio
    async def test_update_close_ts_nonexistent_event(self, state):
        result = await state.update_close_ts("NOPE", 2000)
        assert result is False


# ===========================================================================
# TrackedEventsState — getters
# ===========================================================================


class TestGetters:
    @pytest.mark.asyncio
    async def test_get_event(self, state):
        await state.upsert_event(_make_event(event_ticker="E1"))
        assert state.get_event("E1") is not None
        assert state.get_event("NOPE") is None

    @pytest.mark.asyncio
    async def test_get_all(self, state):
        await state.upsert_event(_make_event(event_ticker="E1"))
        await state.upsert_event(_make_event(event_ticker="E2"))
        assert len(state.get_all()) == 2

    @pytest.mark.asyncio
    async def test_get_active_includes_pending_and_active(self, state):
        await state.upsert_event(_make_event(event_ticker="E1", status=EventStatus.PENDING))
        await state.upsert_event(_make_event(event_ticker="E2", status=EventStatus.ACTIVE))
        await state.upsert_event(_make_event(event_ticker="E3", status=EventStatus.DETERMINED))
        active = state.get_active()
        tickers = [e.event_ticker for e in active]
        assert "E1" in tickers
        assert "E2" in tickers
        assert "E3" not in tickers

    @pytest.mark.asyncio
    async def test_get_by_status(self, state):
        await state.upsert_event(_make_event(event_ticker="E1", status=EventStatus.SETTLED))
        await state.upsert_event(_make_event(event_ticker="E2", status=EventStatus.SETTLED))
        await state.upsert_event(_make_event(event_ticker="E3", status=EventStatus.ACTIVE))
        settled = state.get_by_status(EventStatus.SETTLED)
        assert len(settled) == 2


# ===========================================================================
# TrackedEventsState — get_timeline
# ===========================================================================


class TestTimeline:
    @pytest.mark.asyncio
    async def test_upcoming_open(self, state):
        future_ts = int(time.time()) + 7200
        await state.upsert_event(_make_event(
            event_ticker="E1",
            status=EventStatus.PENDING,
            earliest_open_ts=future_ts,
        ))
        timeline = state.get_timeline()
        assert len(timeline) == 1
        assert timeline[0]["type"] == "upcoming_open"
        assert timeline[0]["countdown_seconds"] > 0

    @pytest.mark.asyncio
    async def test_active_event(self, state):
        past_ts = int(time.time()) - 3600
        await state.upsert_event(_make_event(
            event_ticker="E1",
            status=EventStatus.ACTIVE,
            earliest_open_ts=past_ts,
            latest_close_ts=int(time.time()) + 7200,  # closes in 2h
        ))
        timeline = state.get_timeline()
        assert len(timeline) == 1
        assert timeline[0]["type"] == "active"

    @pytest.mark.asyncio
    async def test_closing_soon_within_one_hour(self, state):
        past_ts = int(time.time()) - 3600
        close_ts = int(time.time()) + 1800  # 30 min from now
        await state.upsert_event(_make_event(
            event_ticker="E1",
            status=EventStatus.ACTIVE,
            earliest_open_ts=past_ts,
            latest_close_ts=close_ts,
        ))
        timeline = state.get_timeline()
        assert len(timeline) == 1
        assert timeline[0]["type"] == "closing_soon"
        assert "countdown_seconds" in timeline[0]

    @pytest.mark.asyncio
    async def test_determined_event(self, state):
        await state.upsert_event(_make_event(
            event_ticker="E1",
            status=EventStatus.DETERMINED,
        ))
        timeline = state.get_timeline()
        assert len(timeline) == 1
        assert timeline[0]["type"] == "determined"

    @pytest.mark.asyncio
    async def test_timeline_sorted_by_timestamp(self, state):
        now = int(time.time())
        await state.upsert_event(_make_event(
            event_ticker="E-LATER",
            status=EventStatus.PENDING,
            earliest_open_ts=now + 7200,
        ))
        await state.upsert_event(_make_event(
            event_ticker="E-SOONER",
            status=EventStatus.PENDING,
            earliest_open_ts=now + 3600,
        ))
        timeline = state.get_timeline()
        assert len(timeline) == 2
        assert timeline[0]["event_ticker"] == "E-SOONER"
        assert timeline[1]["event_ticker"] == "E-LATER"

    @pytest.mark.asyncio
    async def test_settled_events_excluded_from_timeline(self, state):
        await state.upsert_event(_make_event(
            event_ticker="E1",
            status=EventStatus.SETTLED,
        ))
        timeline = state.get_timeline()
        assert len(timeline) == 0


# ===========================================================================
# TrackedEventsState — get_snapshot
# ===========================================================================


class TestSnapshot:
    @pytest.mark.asyncio
    async def test_snapshot_structure(self, state):
        await state.upsert_event(_make_event(event_ticker="E1"))
        snap = state.get_snapshot()
        assert "events" in snap
        assert "timeline" in snap
        assert "stats" in snap
        assert "version" in snap
        assert "timestamp" in snap

    @pytest.mark.asyncio
    async def test_snapshot_events_sorted_by_open_ts(self, state):
        await state.upsert_event(_make_event(event_ticker="E-LATE", earliest_open_ts=2000))
        await state.upsert_event(_make_event(event_ticker="E-EARLY", earliest_open_ts=1000))
        snap = state.get_snapshot()
        events = snap["events"]
        assert events[0]["event_ticker"] == "E-EARLY"
        assert events[1]["event_ticker"] == "E-LATE"


# ===========================================================================
# TrackedEventsState — get_stats
# ===========================================================================


class TestStats:
    @pytest.mark.asyncio
    async def test_stats_counts(self, state):
        await state.upsert_event(_make_event(event_ticker="E1", status=EventStatus.ACTIVE, category="Sports", market_tickers=["M1", "M2"]))
        await state.upsert_event(_make_event(event_ticker="E2", status=EventStatus.PENDING, category="Crypto", market_tickers=["M3"]))
        await state.upsert_event(_make_event(event_ticker="E3", status=EventStatus.DETERMINED, category="Sports"))

        stats = state.get_stats()
        assert stats["total"] == 3
        assert stats["by_status"]["active"] == 1
        assert stats["by_status"]["pending"] == 1
        assert stats["by_status"]["determined"] == 1
        assert stats["total_markets"] == 3  # M1 + M2 + M3
        assert stats["version"] == state.version

    @pytest.mark.asyncio
    async def test_stats_by_category_only_active_pending(self, state):
        await state.upsert_event(_make_event(event_ticker="E1", status=EventStatus.ACTIVE, category="Sports"))
        await state.upsert_event(_make_event(event_ticker="E2", status=EventStatus.DETERMINED, category="Sports"))
        stats = state.get_stats()
        # by_category only counts PENDING + ACTIVE
        assert stats["by_category"].get("Sports", 0) == 1


# ===========================================================================
# TrackedEventsState — version tracking
# ===========================================================================


class TestVersionTracking:
    @pytest.mark.asyncio
    async def test_initial_version_is_zero(self, state):
        assert state.version == 0

    @pytest.mark.asyncio
    async def test_has_changed_since(self, state):
        v = state.version
        assert state.has_changed_since(v) is False
        await state.upsert_event(_make_event())
        assert state.has_changed_since(v) is True

    @pytest.mark.asyncio
    async def test_total_count(self, state):
        assert state.total_count == 0
        await state.upsert_event(_make_event(event_ticker="E1"))
        assert state.total_count == 1
