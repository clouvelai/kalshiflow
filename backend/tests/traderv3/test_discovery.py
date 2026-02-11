"""Tests for TopVolumeDiscovery - volume-ranked event discovery."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from kalshiflow_rl.traderv3.single_arb.discovery import (
    DEFAULT_EVENT_COUNT,
    DEFAULT_MAX_MARKETS_PER_EVENT,
    TopVolumeDiscovery,
)
from kalshiflow_rl.traderv3.single_arb.index import EventArbIndex, EventMeta, MarketMeta


# ──────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────

def _make_market(ticker: str, volume_24h: int = 1000, status: str = "open"):
    return {
        "ticker": ticker,
        "event_ticker": ticker.rsplit("-", 1)[0] if "-" in ticker else "EVT",
        "status": status,
        "volume_24h": volume_24h,
        "open_interest": 100,
        "yes_bid": 40,
        "yes_ask": 60,
        "no_bid": 40,
        "no_ask": 60,
        "last_price": 50,
    }


def _make_event(event_ticker: str, num_markets: int = 3, volume_per_market: int = 1000, category: str = ""):
    markets = [
        _make_market(f"{event_ticker}-MKT{i}", volume_24h=volume_per_market)
        for i in range(num_markets)
    ]
    return {
        "event_ticker": event_ticker,
        "title": f"Event {event_ticker}",
        "series_ticker": f"S_{event_ticker}",
        "category": category,
        "mutually_exclusive": True,
        "sub_title": "",
        "markets": markets,
    }


def _make_client(events=None, pages=1):
    client = AsyncMock()
    events = events or []

    if pages <= 1:
        client.get_events.return_value = {"events": events, "cursor": ""}
    else:
        per_page = len(events) // pages
        call_count = 0

        async def paginated_get_events(**kwargs):
            nonlocal call_count
            start = call_count * per_page
            end = start + per_page if call_count < pages - 1 else len(events)
            page = events[start:end]
            cursor = f"page{call_count + 1}" if call_count < pages - 1 else ""
            call_count += 1
            return {"events": page, "cursor": cursor}

        client.get_events.side_effect = paginated_get_events

    return client


def _make_seed_index(event_ticker, num_markets=1):
    """Create an index with load_event mocked for seed loading."""
    index = EventArbIndex()
    meta = EventMeta(
        raw={}, event_ticker=event_ticker, series_ticker="S1",
        title=f"Seed {event_ticker}", category="", mutually_exclusive=False,
    )
    for i in range(num_markets):
        meta.markets[f"{event_ticker}-M{i}"] = MarketMeta.from_api(
            _make_market(f"{event_ticker}-M{i}"), event_ticker
        )
    index.load_event = AsyncMock(return_value=meta)
    return index, meta


# ──────────────────────────────────────────────────────────
#  Basic Discovery
# ──────────────────────────────────────────────────────────

class TestBasicDiscovery:
    @pytest.mark.asyncio
    async def test_discover_loads_top_n(self):
        events = [
            _make_event("HIGH", volume_per_market=10000),
            _make_event("MED", volume_per_market=5000),
            _make_event("LOW", volume_per_market=100),
        ]
        d = TopVolumeDiscovery(index=EventArbIndex(), trading_client=_make_client(events), event_count=2)
        loaded = await d.discover()
        assert loaded == 2
        assert "HIGH" in d.known_events
        assert "MED" in d.known_events
        assert "LOW" not in d.known_events

    @pytest.mark.asyncio
    async def test_ranks_by_total_volume(self):
        events = [
            _make_event("MED", num_markets=5, volume_per_market=2000),   # 10000
            _make_event("HIGH", num_markets=2, volume_per_market=10000), # 20000
        ]
        d = TopVolumeDiscovery(index=EventArbIndex(), trading_client=_make_client(events), event_count=1)
        await d.discover()
        assert "HIGH" in d.known_events
        assert "MED" not in d.known_events

    @pytest.mark.asyncio
    async def test_default_event_count(self):
        d = TopVolumeDiscovery(index=EventArbIndex(), trading_client=AsyncMock())
        assert d.event_count == DEFAULT_EVENT_COUNT

    @pytest.mark.asyncio
    async def test_empty_response(self):
        d = TopVolumeDiscovery(index=EventArbIndex(), trading_client=_make_client([]), event_count=5)
        loaded = await d.discover()
        assert loaded == 0

    @pytest.mark.asyncio
    async def test_idempotent(self):
        events = [_make_event("EVT1")]
        d = TopVolumeDiscovery(index=EventArbIndex(), trading_client=_make_client(events), event_count=5)
        first = await d.discover()
        second = await d.discover()
        assert first == 1
        assert second == 0

    @pytest.mark.asyncio
    async def test_pagination(self):
        events = [_make_event(f"EVT{i}") for i in range(6)]
        d = TopVolumeDiscovery(index=EventArbIndex(), trading_client=_make_client(events, pages=3), event_count=10)
        loaded = await d.discover()
        assert loaded == 6


# ──────────────────────────────────────────────────────────
#  Max Markets Filtering
# ──────────────────────────────────────────────────────────

class TestMaxMarketsFiltering:
    @pytest.mark.asyncio
    async def test_oversized_event_skipped(self):
        events = [
            _make_event("BIG", num_markets=100),
            _make_event("SMALL", num_markets=3),
        ]
        d = TopVolumeDiscovery(
            index=EventArbIndex(), trading_client=_make_client(events),
            event_count=10, max_markets_per_event=50,
        )
        await d.discover()
        assert "SMALL" in d.known_events
        assert "BIG" not in d.known_events
        assert d._stats.total_events_skipped_size == 1

    @pytest.mark.asyncio
    async def test_exactly_at_cap_allowed(self):
        events = [_make_event("EXACT", num_markets=50)]
        d = TopVolumeDiscovery(
            index=EventArbIndex(), trading_client=_make_client(events),
            event_count=10, max_markets_per_event=50,
        )
        await d.discover()
        assert "EXACT" in d.known_events


# ──────────────────────────────────────────────────────────
#  Seed Events
# ──────────────────────────────────────────────────────────

class TestSeedEvents:
    @pytest.mark.asyncio
    async def test_seed_events_loaded(self):
        index, _ = _make_seed_index("SEED1")
        d = TopVolumeDiscovery(
            index=index, trading_client=_make_client([]),
            event_count=5, seed_event_tickers=["SEED1"],
        )
        loaded = await d.discover()
        assert loaded == 1
        assert "SEED1" in d.known_events

    @pytest.mark.asyncio
    async def test_seed_failure_continues(self):
        index = EventArbIndex()
        index.load_event = AsyncMock(return_value=None)
        d = TopVolumeDiscovery(
            index=index, trading_client=_make_client([]),
            event_count=5, seed_event_tickers=["BAD"],
        )
        loaded = await d.discover()
        assert loaded == 0

    @pytest.mark.asyncio
    async def test_seed_size_cap(self):
        index, _ = _make_seed_index("BIG", num_markets=100)
        d = TopVolumeDiscovery(
            index=index, trading_client=_make_client([]),
            event_count=5, seed_event_tickers=["BIG"],
            max_markets_per_event=50,
        )
        loaded = await d.discover()
        assert loaded == 0
        assert d._stats.total_events_skipped_size == 1


# ──────────────────────────────────────────────────────────
#  Market Filtering
# ──────────────────────────────────────────────────────────

class TestMarketFiltering:
    @pytest.mark.asyncio
    async def test_closed_markets_excluded(self):
        event = _make_event("EVT1", num_markets=3)
        event["markets"][0]["status"] = "closed"
        event["markets"][1]["status"] = "settled"
        d = TopVolumeDiscovery(index=EventArbIndex(), trading_client=_make_client([event]), event_count=10)
        await d.discover()
        assert len(d._index.events["EVT1"].markets) == 1

    @pytest.mark.asyncio
    async def test_no_open_markets_skipped(self):
        event = _make_event("EVT1", num_markets=2)
        event["markets"][0]["status"] = "closed"
        event["markets"][1]["status"] = "closed"
        d = TopVolumeDiscovery(index=EventArbIndex(), trading_client=_make_client([event]), event_count=10)
        await d.discover()
        assert "EVT1" not in d.known_events


# ──────────────────────────────────────────────────────────
#  Stats
# ──────────────────────────────────────────────────────────

class TestStats:
    @pytest.mark.asyncio
    async def test_stats_updated(self):
        events = [_make_event("EVT1", num_markets=3)]
        d = TopVolumeDiscovery(index=EventArbIndex(), trading_client=_make_client(events), event_count=10)
        await d.discover()
        s = d.get_stats()
        assert s["total_fetches"] == 1
        assert s["total_events_discovered"] == 1
        assert s["total_markets_discovered"] == 3
        assert s["total_events_scanned"] == 1
        assert s["event_count"] == 10

    @pytest.mark.asyncio
    async def test_error_stats(self):
        client = AsyncMock()
        client.get_events.side_effect = Exception("API down")
        d = TopVolumeDiscovery(index=EventArbIndex(), trading_client=client, event_count=5)
        await d.discover()
        assert d.get_stats()["errors"] == 1


# ──────────────────────────────────────────────────────────
#  Callbacks
# ──────────────────────────────────────────────────────────

class TestCallbacks:
    @pytest.mark.asyncio
    async def test_subscribe_called(self):
        subscribe = AsyncMock()
        d = TopVolumeDiscovery(
            index=EventArbIndex(), trading_client=_make_client([_make_event("E1", num_markets=2)]),
            event_count=10, subscribe_callback=subscribe,
        )
        await d.discover()
        await asyncio.sleep(0.05)
        assert subscribe.called
        assert len(subscribe.call_args[0][0]) == 2

    @pytest.mark.asyncio
    async def test_broadcast_types(self):
        broadcast = AsyncMock()
        d = TopVolumeDiscovery(
            index=EventArbIndex(), trading_client=_make_client([_make_event("E1")]),
            event_count=10, broadcast_callback=broadcast,
        )
        await d.discover()
        await asyncio.sleep(0.05)
        types = [c.args[0]["type"] for c in broadcast.call_args_list]
        assert "discovery_update" in types
        assert "discovery_state" in types

    @pytest.mark.asyncio
    async def test_discovery_state_shape(self):
        broadcast = AsyncMock()
        d = TopVolumeDiscovery(
            index=EventArbIndex(), trading_client=_make_client([_make_event("E1", volume_per_market=5000)]),
            event_count=10, broadcast_callback=broadcast,
        )
        await d.discover()
        await asyncio.sleep(0.05)
        state_call = next(c for c in broadcast.call_args_list if c.args[0]["type"] == "discovery_state")
        data = state_call.args[0]["data"]
        assert "events" in data
        assert "stats" in data
        assert "timestamp" in data
        assert data["events"][0]["volume_24h"] == 15000

    @pytest.mark.asyncio
    async def test_subscribe_failure_doesnt_crash(self):
        d = TopVolumeDiscovery(
            index=EventArbIndex(), trading_client=_make_client([_make_event("E1")]),
            event_count=10, subscribe_callback=AsyncMock(side_effect=Exception("fail")),
        )
        await d.discover()
        await asyncio.sleep(0.05)
        assert "E1" in d.known_events


# ──────────────────────────────────────────────────────────
#  Index Integration
# ──────────────────────────────────────────────────────────

class TestIndexIntegration:
    @pytest.mark.asyncio
    async def test_market_tickers_registered(self):
        index = EventArbIndex()
        d = TopVolumeDiscovery(index=index, trading_client=_make_client([_make_event("E1", num_markets=3)]), event_count=10)
        await d.discover()
        assert "E1-MKT0" in index.market_tickers
        assert "E1-MKT2" in index.market_tickers

    @pytest.mark.asyncio
    async def test_event_meta_fields(self):
        index = EventArbIndex()
        d = TopVolumeDiscovery(index=index, trading_client=_make_client([_make_event("E1", category="Politics")]), event_count=10)
        await d.discover()
        meta = index.events["E1"]
        assert meta.title == "Event E1"
        assert meta.category == "Politics"
        assert meta.mutually_exclusive is True


# ──────────────────────────────────────────────────────────
#  Background Loop
# ──────────────────────────────────────────────────────────

class TestBackgroundLoop:
    @pytest.mark.asyncio
    async def test_start_stop(self):
        d = TopVolumeDiscovery(index=EventArbIndex(), trading_client=_make_client([]), event_count=5, refresh_interval=0.1)
        await d.start()
        assert d._running
        await d.stop()
        assert not d._running

    @pytest.mark.asyncio
    async def test_double_start_noop(self):
        d = TopVolumeDiscovery(index=EventArbIndex(), trading_client=_make_client([]), event_count=5)
        await d.start()
        task1 = d._task
        await d.start()
        assert d._task is task1
        await d.stop()


# ──────────────────────────────────────────────────────────
#  Snapshot
# ──────────────────────────────────────────────────────────

class TestSnapshot:
    @pytest.mark.asyncio
    async def test_empty_snapshot(self):
        d = TopVolumeDiscovery(index=EventArbIndex(), trading_client=_make_client([]), event_count=5)
        snap = d.get_discovery_snapshot()
        assert snap["events"] == []
        assert snap["event_count"] == 0

    @pytest.mark.asyncio
    async def test_sorted_by_volume(self):
        events = [
            _make_event("LOW", volume_per_market=100),
            _make_event("HIGH", volume_per_market=10000),
            _make_event("MED", volume_per_market=1000),
        ]
        d = TopVolumeDiscovery(index=EventArbIndex(), trading_client=_make_client(events), event_count=10)
        await d.discover()
        tickers = [e["event_ticker"] for e in d.get_discovery_snapshot()["events"]]
        assert tickers == ["HIGH", "MED", "LOW"]

    @pytest.mark.asyncio
    async def test_source_field(self):
        index, meta = _make_seed_index("SEED1")
        # Seed loading via index.load_event returns meta, but we also need to
        # register it in the index so get_discovery_snapshot can find it
        async def load_and_register(ticker, client):
            index._events[ticker] = meta
            for t in meta.markets:
                index._ticker_to_event[t] = ticker
            return meta
        index.load_event = AsyncMock(side_effect=load_and_register)

        events = [_make_event("VOL1"), _make_event("SEED1")]
        client = _make_client(events)
        d = TopVolumeDiscovery(
            index=index, trading_client=client,
            event_count=10, seed_event_tickers=["SEED1"],
        )
        await d.discover()
        sources = {e["event_ticker"]: e["source"] for e in d.get_discovery_snapshot()["events"]}
        assert sources["VOL1"] == "volume"
        assert sources["SEED1"] == "seed"


# ──────────────────────────────────────────────────────────
#  Volume Ranking
# ──────────────────────────────────────────────────────────

class TestVolumeRanking:
    def test_rank_order(self):
        d = TopVolumeDiscovery(index=EventArbIndex(), trading_client=AsyncMock(), event_count=10)
        events = [
            {"event_ticker": "A", "markets": [{"volume_24h": 100}, {"volume_24h": 200}]},
            {"event_ticker": "B", "markets": [{"volume_24h": 500}]},
            {"event_ticker": "C", "markets": [{"volume_24h": 50}]},
        ]
        ranked = d._rank_by_volume(events)
        assert [r[0]["event_ticker"] for r in ranked] == ["B", "A", "C"]
        assert [r[1] for r in ranked] == [500, 300, 50]

    def test_none_volume(self):
        d = TopVolumeDiscovery(index=EventArbIndex(), trading_client=AsyncMock(), event_count=10)
        ranked = d._rank_by_volume([{"event_ticker": "A", "markets": [{"volume_24h": None}]}])
        assert ranked[0][1] == 0

    def test_no_markets(self):
        d = TopVolumeDiscovery(index=EventArbIndex(), trading_client=AsyncMock(), event_count=10)
        ranked = d._rank_by_volume([{"event_ticker": "A", "markets": []}])
        assert ranked[0][1] == 0


# ──────────────────────────────────────────────────────────
#  Event Count Config
# ──────────────────────────────────────────────────────────

class TestEventCountConfig:
    @pytest.mark.asyncio
    async def test_custom_event_count(self):
        events = [_make_event(f"E{i}", volume_per_market=1000 * (10 - i)) for i in range(10)]
        d = TopVolumeDiscovery(index=EventArbIndex(), trading_client=_make_client(events), event_count=3)
        await d.discover()
        assert len(d.known_events) == 3

    @pytest.mark.asyncio
    async def test_fewer_events_than_count(self):
        events = [_make_event("E1"), _make_event("E2")]
        d = TopVolumeDiscovery(index=EventArbIndex(), trading_client=_make_client(events), event_count=10)
        loaded = await d.discover()
        assert loaded == 2


# ──────────────────────────────────────────────────────────
#  Eviction
# ──────────────────────────────────────────────────────────

class TestEviction:
    @pytest.mark.asyncio
    async def test_settled_events_evicted_on_refresh(self):
        """Event present in first pass but absent from API in second pass -> evicted."""
        events_pass1 = [_make_event("LIVE"), _make_event("SETTLED")]
        events_pass2 = [_make_event("LIVE")]

        client = AsyncMock()
        call_count = 0

        async def get_events_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"events": events_pass1, "cursor": ""}
            return {"events": events_pass2, "cursor": ""}

        client.get_events.side_effect = get_events_side_effect

        d = TopVolumeDiscovery(index=EventArbIndex(), trading_client=client, event_count=10)
        await d.discover()
        assert "SETTLED" in d.known_events
        assert "LIVE" in d.known_events

        await d.discover()
        assert "LIVE" in d.known_events
        assert "SETTLED" not in d.known_events

    @pytest.mark.asyncio
    async def test_eviction_cleans_index(self):
        """Verify _index._events and _ticker_to_event are cleaned on eviction."""
        events_pass1 = [_make_event("EVT1", num_markets=2)]
        events_pass2 = []  # empty won't trigger eviction due to guard

        # Use a two-pass where second pass has a different event
        events_pass2 = [_make_event("EVT2")]

        client = AsyncMock()
        call_count = 0

        async def get_events_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"events": events_pass1, "cursor": ""}
            return {"events": events_pass2, "cursor": ""}

        client.get_events.side_effect = get_events_side_effect

        index = EventArbIndex()
        d = TopVolumeDiscovery(index=index, trading_client=client, event_count=10)
        await d.discover()
        assert "EVT1" in index._events
        assert "EVT1-MKT0" in index._ticker_to_event

        await d.discover()
        assert "EVT1" not in index._events
        assert "EVT1-MKT0" not in index._ticker_to_event
        assert "EVT1-MKT1" not in index._ticker_to_event

    @pytest.mark.asyncio
    async def test_eviction_updates_known_tickers(self):
        """Verify _known_event_tickers shrinks after eviction."""
        events_pass1 = [_make_event("A"), _make_event("B"), _make_event("C")]
        events_pass2 = [_make_event("A")]

        client = AsyncMock()
        call_count = 0

        async def get_events_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"events": events_pass1, "cursor": ""}
            return {"events": events_pass2, "cursor": ""}

        client.get_events.side_effect = get_events_side_effect

        d = TopVolumeDiscovery(index=EventArbIndex(), trading_client=client, event_count=10)
        await d.discover()
        assert len(d.known_events) == 3

        await d.discover()
        assert d.known_events == {"A"}

    @pytest.mark.asyncio
    async def test_unsubscribe_callback_called(self):
        """Verify unsubscribe callback receives evicted market tickers."""
        events_pass1 = [_make_event("GONE", num_markets=2)]
        events_pass2 = [_make_event("NEW")]

        client = AsyncMock()
        call_count = 0

        async def get_events_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"events": events_pass1, "cursor": ""}
            return {"events": events_pass2, "cursor": ""}

        client.get_events.side_effect = get_events_side_effect
        unsubscribe = AsyncMock()

        d = TopVolumeDiscovery(
            index=EventArbIndex(), trading_client=client,
            event_count=10, unsubscribe_callback=unsubscribe,
        )
        await d.discover()
        await d.discover()

        unsubscribe.assert_called_once()
        removed_tickers = unsubscribe.call_args[0][0]
        assert "GONE-MKT0" in removed_tickers
        assert "GONE-MKT1" in removed_tickers

    @pytest.mark.asyncio
    async def test_active_events_not_evicted(self):
        """Events still in the API response stay tracked."""
        events = [_make_event("KEEP1"), _make_event("KEEP2")]
        client = _make_client(events)

        d = TopVolumeDiscovery(index=EventArbIndex(), trading_client=client, event_count=10)
        await d.discover()
        # Second pass with same events
        await d.discover()

        assert "KEEP1" in d.known_events
        assert "KEEP2" in d.known_events

    @pytest.mark.asyncio
    async def test_eviction_broadcasts_sent(self):
        """Verify eviction broadcasts discovery_eviction + event_arb_snapshot."""
        events_pass1 = [_make_event("OLD", num_markets=2)]
        events_pass2 = [_make_event("NEW")]

        client = AsyncMock()
        call_count = 0

        async def get_events_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"events": events_pass1, "cursor": ""}
            return {"events": events_pass2, "cursor": ""}

        client.get_events.side_effect = get_events_side_effect
        broadcast = AsyncMock()

        d = TopVolumeDiscovery(
            index=EventArbIndex(), trading_client=client,
            event_count=10, broadcast_callback=broadcast,
        )
        await d.discover()
        broadcast.reset_mock()

        await d.discover()
        await asyncio.sleep(0.05)

        types = [c.args[0]["type"] for c in broadcast.call_args_list]
        assert "discovery_eviction" in types
        assert "event_arb_snapshot" in types

        eviction_msg = next(c.args[0] for c in broadcast.call_args_list if c.args[0]["type"] == "discovery_eviction")
        assert eviction_msg["data"]["count"] == 1
        assert eviction_msg["data"]["evicted"][0]["event_ticker"] == "OLD"
        assert eviction_msg["data"]["evicted"][0]["title"] == "Event OLD"

    @pytest.mark.asyncio
    async def test_eviction_skipped_on_empty_fetch(self):
        """API failure returning empty list doesn't evict everything."""
        events_pass1 = [_make_event("SAFE1"), _make_event("SAFE2")]

        client = AsyncMock()
        call_count = 0

        async def get_events_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"events": events_pass1, "cursor": ""}
            # Simulate API returning empty (failure/outage)
            return {"events": [], "cursor": ""}

        client.get_events.side_effect = get_events_side_effect

        d = TopVolumeDiscovery(index=EventArbIndex(), trading_client=client, event_count=10)
        await d.discover()
        assert len(d.known_events) == 2

        await d.discover()
        # Events should NOT be evicted because the fetch returned empty
        assert len(d.known_events) == 2
        assert "SAFE1" in d.known_events
        assert "SAFE2" in d.known_events
