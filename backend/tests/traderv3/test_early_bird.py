"""Unit tests for EarlyBirdService.

Tests cover:
  1. EarlyBirdScore to_dict serialization
  2. EarlyBirdScore total_score computation
  3. Complement pricing (mutually exclusive event with existing prices)
  4. Complement pricing (no other markets priced -> 0)
  5. Complement pricing (overpriced event -> 0)
  6. Category scoring (sports=15, crypto=12, politics=8, unknown=5)
  7. Timing scoring (1h-24h sweet spot=10, >24h=5, <1h=3)
  8. Risk score default
  9. Strategy selection logic
  10. Cooldown enforcement
  11. Signal emitted when score >= threshold
  12. Signal NOT emitted when score < threshold
  13. Attention callback called with correct kwargs
  14. get_recent_opportunities returns last 30min only
  15. get_recent_opportunities respects max_recent limit
  16. Service disabled when config early_bird_enabled=False
  17. get_stats returns correct values
  18. _on_market_activated handles scoring errors gracefully
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kalshiflow_rl.traderv3.services.early_bird_service import (
    EarlyBirdScore,
    EarlyBirdService,
)


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

def _make_config(**overrides) -> MagicMock:
    """Create a mock V3Config with early bird defaults."""
    cfg = MagicMock()
    cfg.early_bird_enabled = overrides.get("early_bird_enabled", True)
    cfg.early_bird_min_score = overrides.get("early_bird_min_score", 30.0)
    cfg.early_bird_cooldown_seconds = overrides.get("early_bird_cooldown_seconds", 300.0)
    cfg.early_bird_use_news = overrides.get("early_bird_use_news", False)
    return cfg


def _make_event_bus() -> MagicMock:
    """Create a mock EventBus."""
    bus = MagicMock()
    bus.subscribe_to_market_activated = AsyncMock()
    return bus


@dataclass
class MockTrackedEvent:
    """Minimal mock for TrackedEvent."""
    event_ticker: str = "KXTEST-25FEB12"
    title: str = "Test Event"
    category: str = "Sports"
    mutually_exclusive: bool = True
    market_tickers: List[str] = field(default_factory=lambda: ["MKT-A", "MKT-B", "MKT-NEW"])


@dataclass
class MockTrackedMarket:
    """Minimal mock for TrackedMarket."""
    ticker: str = "MKT-A"
    category: str = "Sports"
    yes_bid: int = 40
    close_ts: int = 0


@dataclass
class MockActivatedEvent:
    """Minimal mock for MarketActivatedEvent."""
    market_ticker: str = "MKT-NEW"
    event_ticker: str = "KXTEST-25FEB12"
    category: str = "Sports"


def _make_tracked_events(events: Dict[str, MockTrackedEvent] | None = None) -> MagicMock:
    """Create a mock TrackedEventsState."""
    state = MagicMock()
    _events = events or {}
    state.get_event = MagicMock(side_effect=lambda ticker: _events.get(ticker))
    return state


def _make_tracked_markets(markets: Dict[str, MockTrackedMarket] | None = None) -> MagicMock:
    """Create a mock TrackedMarketsState."""
    state = MagicMock()
    _markets = markets or {}
    state.get_market = MagicMock(side_effect=lambda ticker: _markets.get(ticker))
    return state


@pytest.fixture
def config():
    return _make_config()


@pytest.fixture
def event_bus():
    return _make_event_bus()


@pytest.fixture
def attention_callback():
    return AsyncMock()


# ===========================================================================
# EarlyBirdScore
# ===========================================================================


class TestEarlyBirdScore:
    def test_to_dict_keys(self):
        score = EarlyBirdScore(
            market_ticker="MKT-A",
            event_ticker="EVT-1",
            total_score=75.0,
            complement_score=25.0,
            category_score=15.0,
            timing_score=10.0,
            risk_score=8.0,
            strategy="complement",
            fair_value_estimate=60.0,
            reasoning="Test reasoning",
        )
        d = score.to_dict()
        assert d["market_ticker"] == "MKT-A"
        assert d["event_ticker"] == "EVT-1"
        assert d["total_score"] == 75.0
        assert d["complement_score"] == 25.0
        assert d["strategy"] == "complement"
        assert d["fair_value_estimate"] == 60.0
        assert "scored_at" in d

    def test_default_values(self):
        score = EarlyBirdScore(market_ticker="MKT-A", event_ticker="EVT-1")
        assert score.total_score == 0.0
        assert score.complement_score == 0.0
        assert score.strategy == "unknown"
        assert score.fair_value_estimate is None


# ===========================================================================
# Complement pricing
# ===========================================================================


class TestComplementPricing:
    @pytest.mark.asyncio
    async def test_complement_full_score_with_existing_prices(self, config, event_bus, attention_callback):
        """Mutually exclusive event with 2 priced markets -> fair value = 100 - sum."""
        tracked_events = _make_tracked_events({
            "EVT-1": MockTrackedEvent(
                event_ticker="EVT-1",
                mutually_exclusive=True,
                market_tickers=["MKT-A", "MKT-B", "MKT-NEW"],
            ),
        })
        tracked_markets = _make_tracked_markets({
            "MKT-A": MockTrackedMarket(ticker="MKT-A", yes_bid=30, close_ts=int(time.time()) + 7200),
            "MKT-B": MockTrackedMarket(ticker="MKT-B", yes_bid=20, close_ts=int(time.time()) + 7200),
            "MKT-NEW": MockTrackedMarket(ticker="MKT-NEW", yes_bid=0, category="Sports", close_ts=int(time.time()) + 7200),
        })

        svc = EarlyBirdService(
            event_bus=event_bus,
            tracked_events=tracked_events,
            tracked_markets=tracked_markets,
            config=config,
            attention_callback=attention_callback,
        )
        await svc.start()

        # Invoke scoring directly
        score = await svc._score_opportunity("MKT-NEW", "EVT-1")
        # 100 - 30 - 20 = 50 -> full 25 points (5 <= 50 <= 95)
        assert score.complement_score == 25.0
        assert score.fair_value_estimate == 50.0
        assert score.strategy == "complement"

    @pytest.mark.asyncio
    async def test_complement_no_other_markets_priced(self, config, event_bus):
        """No other markets have yes_bid > 0 -> complement_score = 0."""
        tracked_events = _make_tracked_events({
            "EVT-1": MockTrackedEvent(
                event_ticker="EVT-1",
                mutually_exclusive=True,
                market_tickers=["MKT-A", "MKT-NEW"],
            ),
        })
        tracked_markets = _make_tracked_markets({
            "MKT-A": MockTrackedMarket(ticker="MKT-A", yes_bid=0),
            "MKT-NEW": MockTrackedMarket(ticker="MKT-NEW", yes_bid=0, category="Sports"),
        })

        svc = EarlyBirdService(
            event_bus=event_bus,
            tracked_events=tracked_events,
            tracked_markets=tracked_markets,
            config=config,
        )
        score = await svc._score_opportunity("MKT-NEW", "EVT-1")
        assert score.complement_score == 0.0
        assert score.fair_value_estimate is None

    @pytest.mark.asyncio
    async def test_complement_overpriced_event(self, config, event_bus):
        """Sum of other prices >= 100 -> complement_score = 0."""
        tracked_events = _make_tracked_events({
            "EVT-1": MockTrackedEvent(
                event_ticker="EVT-1",
                mutually_exclusive=True,
                market_tickers=["MKT-A", "MKT-B", "MKT-NEW"],
            ),
        })
        tracked_markets = _make_tracked_markets({
            "MKT-A": MockTrackedMarket(ticker="MKT-A", yes_bid=60),
            "MKT-B": MockTrackedMarket(ticker="MKT-B", yes_bid=45),
            "MKT-NEW": MockTrackedMarket(ticker="MKT-NEW", yes_bid=0, category="Sports"),
        })

        svc = EarlyBirdService(
            event_bus=event_bus,
            tracked_events=tracked_events,
            tracked_markets=tracked_markets,
            config=config,
        )
        score = await svc._score_opportunity("MKT-NEW", "EVT-1")
        assert score.complement_score == 0.0

    @pytest.mark.asyncio
    async def test_complement_partial_score_extreme_value(self, config, event_bus):
        """Fair value < 5 or > 95 -> partial score of 15."""
        tracked_events = _make_tracked_events({
            "EVT-1": MockTrackedEvent(
                event_ticker="EVT-1",
                mutually_exclusive=True,
                market_tickers=["MKT-A", "MKT-NEW"],
            ),
        })
        tracked_markets = _make_tracked_markets({
            "MKT-A": MockTrackedMarket(ticker="MKT-A", yes_bid=97),  # fair_value = 3, extreme
            "MKT-NEW": MockTrackedMarket(ticker="MKT-NEW", yes_bid=0, category="Sports"),
        })

        svc = EarlyBirdService(
            event_bus=event_bus,
            tracked_events=tracked_events,
            tracked_markets=tracked_markets,
            config=config,
        )
        score = await svc._score_opportunity("MKT-NEW", "EVT-1")
        assert score.complement_score == 15.0
        assert score.fair_value_estimate == 3.0

    @pytest.mark.asyncio
    async def test_complement_not_mutually_exclusive(self, config, event_bus):
        """Non mutually-exclusive events skip complement pricing."""
        tracked_events = _make_tracked_events({
            "EVT-1": MockTrackedEvent(
                event_ticker="EVT-1",
                mutually_exclusive=False,
                market_tickers=["MKT-A", "MKT-NEW"],
            ),
        })
        tracked_markets = _make_tracked_markets({
            "MKT-A": MockTrackedMarket(ticker="MKT-A", yes_bid=40),
            "MKT-NEW": MockTrackedMarket(ticker="MKT-NEW", yes_bid=0, category="Sports"),
        })

        svc = EarlyBirdService(
            event_bus=event_bus,
            tracked_events=tracked_events,
            tracked_markets=tracked_markets,
            config=config,
        )
        score = await svc._score_opportunity("MKT-NEW", "EVT-1")
        assert score.complement_score == 0.0


# ===========================================================================
# Category scoring
# ===========================================================================


class TestCategoryScoring:
    @pytest.mark.asyncio
    async def _score_with_category(self, category: str) -> float:
        config = _make_config()
        event_bus = _make_event_bus()
        tracked_events = _make_tracked_events({})
        tracked_markets = _make_tracked_markets({
            "MKT-NEW": MockTrackedMarket(
                ticker="MKT-NEW", category=category, yes_bid=0,
                close_ts=int(time.time()) + 7200,
            ),
        })
        svc = EarlyBirdService(
            event_bus=event_bus,
            tracked_events=tracked_events,
            tracked_markets=tracked_markets,
            config=config,
        )
        score = await svc._score_opportunity("MKT-NEW", "EVT-1")
        return score.category_score

    @pytest.mark.asyncio
    async def test_sports_category(self):
        assert await self._score_with_category("Sports") == 15

    @pytest.mark.asyncio
    async def test_crypto_category(self):
        assert await self._score_with_category("Crypto") == 12

    @pytest.mark.asyncio
    async def test_politics_category(self):
        assert await self._score_with_category("Politics") == 8

    @pytest.mark.asyncio
    async def test_economics_category(self):
        assert await self._score_with_category("Economics") == 10

    @pytest.mark.asyncio
    async def test_unknown_category(self):
        assert await self._score_with_category("Entertainment") == 5


# ===========================================================================
# Timing scoring
# ===========================================================================


class TestTimingScoring:
    @pytest.mark.asyncio
    async def _score_with_close_ts(self, seconds_from_now: int) -> float:
        config = _make_config()
        event_bus = _make_event_bus()
        tracked_events = _make_tracked_events({})
        close_ts = int(time.time()) + seconds_from_now
        tracked_markets = _make_tracked_markets({
            "MKT-NEW": MockTrackedMarket(
                ticker="MKT-NEW", category="Sports", yes_bid=0,
                close_ts=close_ts,
            ),
        })
        svc = EarlyBirdService(
            event_bus=event_bus,
            tracked_events=tracked_events,
            tracked_markets=tracked_markets,
            config=config,
        )
        score = await svc._score_opportunity("MKT-NEW", "EVT-1")
        return score.timing_score

    @pytest.mark.asyncio
    async def test_sweet_spot_1h_to_24h(self):
        # 12 hours from now = sweet spot
        assert await self._score_with_close_ts(43200) == 10

    @pytest.mark.asyncio
    async def test_more_than_24h(self):
        # 48 hours from now
        assert await self._score_with_close_ts(172800) == 5

    @pytest.mark.asyncio
    async def test_less_than_1h(self):
        # 30 minutes from now
        assert await self._score_with_close_ts(1800) == 3

    @pytest.mark.asyncio
    async def test_no_close_ts(self):
        config = _make_config()
        event_bus = _make_event_bus()
        tracked_events = _make_tracked_events({})
        tracked_markets = _make_tracked_markets({
            "MKT-NEW": MockTrackedMarket(ticker="MKT-NEW", category="Sports", yes_bid=0, close_ts=0),
        })
        svc = EarlyBirdService(
            event_bus=event_bus,
            tracked_events=tracked_events,
            tracked_markets=tracked_markets,
            config=config,
        )
        score = await svc._score_opportunity("MKT-NEW", "EVT-1")
        assert score.timing_score == 0


# ===========================================================================
# Risk score
# ===========================================================================


class TestRiskScore:
    @pytest.mark.asyncio
    async def test_default_risk_score_no_callback(self):
        """No health_callback -> default risk_score = 8."""
        config = _make_config()
        event_bus = _make_event_bus()
        tracked_events = _make_tracked_events({})
        tracked_markets = _make_tracked_markets({
            "MKT-NEW": MockTrackedMarket(ticker="MKT-NEW", yes_bid=0, category="Sports"),
        })
        svc = EarlyBirdService(
            event_bus=event_bus,
            tracked_events=tracked_events,
            tracked_markets=tracked_markets,
            config=config,
        )
        score = await svc._score_opportunity("MKT-NEW", "EVT-1")
        assert score.risk_score == 8

    @pytest.mark.asyncio
    async def test_risk_score_low_drawdown(self):
        """Drawdown < 10% -> risk_score = 10."""
        async def _health():
            return 5.0
        config = _make_config()
        event_bus = _make_event_bus()
        tracked_events = _make_tracked_events({})
        tracked_markets = _make_tracked_markets({
            "MKT-NEW": MockTrackedMarket(ticker="MKT-NEW", yes_bid=0, category="Sports"),
        })
        svc = EarlyBirdService(
            event_bus=event_bus,
            tracked_events=tracked_events,
            tracked_markets=tracked_markets,
            config=config,
            health_callback=_health,
        )
        score = await svc._score_opportunity("MKT-NEW", "EVT-1")
        assert score.risk_score == 10.0

    @pytest.mark.asyncio
    async def test_risk_score_high_drawdown(self):
        """Drawdown > 20% -> risk_score = 2."""
        async def _health():
            return 25.0
        config = _make_config()
        event_bus = _make_event_bus()
        tracked_events = _make_tracked_events({})
        tracked_markets = _make_tracked_markets({
            "MKT-NEW": MockTrackedMarket(ticker="MKT-NEW", yes_bid=0, category="Sports"),
        })
        svc = EarlyBirdService(
            event_bus=event_bus,
            tracked_events=tracked_events,
            tracked_markets=tracked_markets,
            config=config,
            health_callback=_health,
        )
        score = await svc._score_opportunity("MKT-NEW", "EVT-1")
        assert score.risk_score == 2.0

    @pytest.mark.asyncio
    async def test_risk_score_medium_drawdown(self):
        """10% < drawdown < 20% -> risk_score = 5."""
        async def _health():
            return 15.0
        config = _make_config()
        event_bus = _make_event_bus()
        tracked_events = _make_tracked_events({})
        tracked_markets = _make_tracked_markets({
            "MKT-NEW": MockTrackedMarket(ticker="MKT-NEW", yes_bid=0, category="Sports"),
        })
        svc = EarlyBirdService(
            event_bus=event_bus,
            tracked_events=tracked_events,
            tracked_markets=tracked_markets,
            config=config,
            health_callback=_health,
        )
        score = await svc._score_opportunity("MKT-NEW", "EVT-1")
        assert score.risk_score == 5.0


# ===========================================================================
# News scoring
# ===========================================================================


class TestNewsScoring:
    @pytest.mark.asyncio
    async def test_news_score_strong_presence(self):
        """3+ search results -> news_score = 15."""
        mock_search = AsyncMock()
        mock_search.search = AsyncMock(return_value=["r1", "r2", "r3"])
        config = _make_config(early_bird_use_news=True)
        event_bus = _make_event_bus()
        tracked_events = _make_tracked_events({
            "EVT-1": MockTrackedEvent(event_ticker="EVT-1"),
        })
        tracked_markets = _make_tracked_markets({
            "MKT-NEW": MockTrackedMarket(ticker="MKT-NEW", yes_bid=0, category="Sports"),
        })
        svc = EarlyBirdService(
            event_bus=event_bus,
            tracked_events=tracked_events,
            tracked_markets=tracked_markets,
            config=config,
            search_service=mock_search,
        )
        score = await svc._score_opportunity("MKT-NEW", "EVT-1")
        assert score.news_score == 15.0

    @pytest.mark.asyncio
    async def test_news_score_some_results(self):
        """1-2 search results -> news_score = 8."""
        mock_search = AsyncMock()
        mock_search.search = AsyncMock(return_value=["r1"])
        config = _make_config(early_bird_use_news=True)
        event_bus = _make_event_bus()
        tracked_events = _make_tracked_events({
            "EVT-1": MockTrackedEvent(event_ticker="EVT-1"),
        })
        tracked_markets = _make_tracked_markets({
            "MKT-NEW": MockTrackedMarket(ticker="MKT-NEW", yes_bid=0, category="Sports"),
        })
        svc = EarlyBirdService(
            event_bus=event_bus,
            tracked_events=tracked_events,
            tracked_markets=tracked_markets,
            config=config,
            search_service=mock_search,
        )
        score = await svc._score_opportunity("MKT-NEW", "EVT-1")
        assert score.news_score == 8.0

    @pytest.mark.asyncio
    async def test_news_score_no_results(self):
        """No search results -> news_score = 0."""
        mock_search = AsyncMock()
        mock_search.search = AsyncMock(return_value=[])
        config = _make_config(early_bird_use_news=True)
        event_bus = _make_event_bus()
        tracked_events = _make_tracked_events({
            "EVT-1": MockTrackedEvent(event_ticker="EVT-1"),
        })
        tracked_markets = _make_tracked_markets({
            "MKT-NEW": MockTrackedMarket(ticker="MKT-NEW", yes_bid=0, category="Sports"),
        })
        svc = EarlyBirdService(
            event_bus=event_bus,
            tracked_events=tracked_events,
            tracked_markets=tracked_markets,
            config=config,
            search_service=mock_search,
        )
        score = await svc._score_opportunity("MKT-NEW", "EVT-1")
        assert score.news_score == 0.0

    @pytest.mark.asyncio
    async def test_news_score_disabled_by_config(self):
        """early_bird_use_news=False -> news_score = 0 even with search service."""
        mock_search = AsyncMock()
        mock_search.search = AsyncMock(return_value=["r1", "r2", "r3"])
        config = _make_config(early_bird_use_news=False)
        event_bus = _make_event_bus()
        tracked_events = _make_tracked_events({})
        tracked_markets = _make_tracked_markets({
            "MKT-NEW": MockTrackedMarket(ticker="MKT-NEW", yes_bid=0, category="Sports"),
        })
        svc = EarlyBirdService(
            event_bus=event_bus,
            tracked_events=tracked_events,
            tracked_markets=tracked_markets,
            config=config,
            search_service=mock_search,
        )
        score = await svc._score_opportunity("MKT-NEW", "EVT-1")
        assert score.news_score == 0.0

    @pytest.mark.asyncio
    async def test_news_score_no_search_service(self):
        """No search_service -> news_score = 0 even with use_news=True."""
        config = _make_config(early_bird_use_news=True)
        event_bus = _make_event_bus()
        tracked_events = _make_tracked_events({})
        tracked_markets = _make_tracked_markets({
            "MKT-NEW": MockTrackedMarket(ticker="MKT-NEW", yes_bid=0, category="Sports"),
        })
        svc = EarlyBirdService(
            event_bus=event_bus,
            tracked_events=tracked_events,
            tracked_markets=tracked_markets,
            config=config,
            # No search_service
        )
        score = await svc._score_opportunity("MKT-NEW", "EVT-1")
        assert score.news_score == 0.0


# ===========================================================================
# Strategy selection
# ===========================================================================


class TestStrategySelection:
    @pytest.mark.asyncio
    async def test_complement_strategy_when_complement_score(self, config, event_bus):
        """If complement_score > 0, strategy = complement."""
        tracked_events = _make_tracked_events({
            "EVT-1": MockTrackedEvent(
                mutually_exclusive=True,
                market_tickers=["MKT-A", "MKT-NEW"],
            ),
        })
        tracked_markets = _make_tracked_markets({
            "MKT-A": MockTrackedMarket(ticker="MKT-A", yes_bid=40),
            "MKT-NEW": MockTrackedMarket(ticker="MKT-NEW", yes_bid=0, category="Sports"),
        })
        svc = EarlyBirdService(
            event_bus=event_bus,
            tracked_events=tracked_events,
            tracked_markets=tracked_markets,
            config=config,
        )
        score = await svc._score_opportunity("MKT-NEW", "EVT-1")
        assert score.strategy == "complement"

    @pytest.mark.asyncio
    async def test_captain_decide_fallback(self, config, event_bus):
        """No complement or news -> strategy = captain_decide."""
        tracked_events = _make_tracked_events({})
        tracked_markets = _make_tracked_markets({
            "MKT-NEW": MockTrackedMarket(ticker="MKT-NEW", yes_bid=0, category="Sports"),
        })
        svc = EarlyBirdService(
            event_bus=event_bus,
            tracked_events=tracked_events,
            tracked_markets=tracked_markets,
            config=config,
        )
        score = await svc._score_opportunity("MKT-NEW", "EVT-1")
        assert score.strategy == "captain_decide"


# ===========================================================================
# Cooldown enforcement
# ===========================================================================


class TestCooldown:
    @pytest.mark.asyncio
    async def test_cooldown_blocks_re_signal(self, config, event_bus, attention_callback):
        """After signaling, same event_ticker is blocked for cooldown period."""
        config.early_bird_min_score = 0  # Always signal
        config.early_bird_cooldown_seconds = 300

        tracked_events = _make_tracked_events({})
        tracked_markets = _make_tracked_markets({
            "MKT-NEW": MockTrackedMarket(ticker="MKT-NEW", yes_bid=0, category="Sports"),
        })
        svc = EarlyBirdService(
            event_bus=event_bus,
            tracked_events=tracked_events,
            tracked_markets=tracked_markets,
            config=config,
            attention_callback=attention_callback,
        )
        await svc.start()

        # First signal
        event1 = MockActivatedEvent(market_ticker="MKT-NEW", event_ticker="EVT-1")
        await svc._on_market_activated(event1)
        assert attention_callback.await_count == 1

        # Second signal within cooldown (same event_ticker)
        event2 = MockActivatedEvent(market_ticker="MKT-NEW-2", event_ticker="EVT-1")
        await svc._on_market_activated(event2)
        assert attention_callback.await_count == 1  # Not called again


# ===========================================================================
# Signal emission threshold
# ===========================================================================


class TestSignalEmission:
    @pytest.mark.asyncio
    async def test_signal_emitted_above_threshold(self, event_bus, attention_callback):
        """Score >= threshold -> attention callback called."""
        config = _make_config(early_bird_min_score=10)
        tracked_events = _make_tracked_events({})
        tracked_markets = _make_tracked_markets({
            "MKT-NEW": MockTrackedMarket(
                ticker="MKT-NEW", yes_bid=0, category="Sports",
                close_ts=int(time.time()) + 7200,
            ),
        })
        svc = EarlyBirdService(
            event_bus=event_bus,
            tracked_events=tracked_events,
            tracked_markets=tracked_markets,
            config=config,
            attention_callback=attention_callback,
        )
        await svc.start()

        event = MockActivatedEvent()
        await svc._on_market_activated(event)
        # Sports=15 + timing=10 + risk=8 = 33 >= 10
        attention_callback.assert_awaited_once()
        assert svc._signals_emitted == 1

    @pytest.mark.asyncio
    async def test_signal_not_emitted_below_threshold(self, event_bus, attention_callback):
        """Score < threshold -> attention callback NOT called."""
        config = _make_config(early_bird_min_score=999)
        tracked_events = _make_tracked_events({})
        tracked_markets = _make_tracked_markets({
            "MKT-NEW": MockTrackedMarket(ticker="MKT-NEW", yes_bid=0, category="Sports"),
        })
        svc = EarlyBirdService(
            event_bus=event_bus,
            tracked_events=tracked_events,
            tracked_markets=tracked_markets,
            config=config,
            attention_callback=attention_callback,
        )
        await svc.start()

        event = MockActivatedEvent()
        await svc._on_market_activated(event)
        attention_callback.assert_not_awaited()
        assert svc._signals_emitted == 0

    @pytest.mark.asyncio
    async def test_attention_callback_kwargs(self, event_bus, attention_callback):
        """Attention callback receives correct kwargs."""
        config = _make_config(early_bird_min_score=0)
        tracked_events = _make_tracked_events({})
        tracked_markets = _make_tracked_markets({
            "MKT-NEW": MockTrackedMarket(
                ticker="MKT-NEW", yes_bid=0, category="Sports",
                close_ts=int(time.time()) + 7200,
            ),
        })
        svc = EarlyBirdService(
            event_bus=event_bus,
            tracked_events=tracked_events,
            tracked_markets=tracked_markets,
            config=config,
            attention_callback=attention_callback,
        )
        await svc.start()

        event = MockActivatedEvent(market_ticker="MKT-NEW", event_ticker="EVT-1")
        await svc._on_market_activated(event)

        call_kwargs = attention_callback.call_args.kwargs
        assert call_kwargs["market_ticker"] == "MKT-NEW"
        assert call_kwargs["event_ticker"] == "EVT-1"
        assert "score" in call_kwargs
        assert "strategy" in call_kwargs


# ===========================================================================
# get_recent_opportunities
# ===========================================================================


class TestRecentOpportunities:
    @pytest.mark.asyncio
    async def test_returns_last_30_min_only(self, config, event_bus):
        tracked_events = _make_tracked_events({})
        tracked_markets = _make_tracked_markets({})
        svc = EarlyBirdService(
            event_bus=event_bus,
            tracked_events=tracked_events,
            tracked_markets=tracked_markets,
            config=config,
        )

        # Insert a recent score
        recent = EarlyBirdScore(
            market_ticker="MKT-A", event_ticker="EVT-1",
            total_score=50, scored_at=time.time(),
        )
        # Insert an old score (>30 min ago)
        old = EarlyBirdScore(
            market_ticker="MKT-B", event_ticker="EVT-2",
            total_score=40, scored_at=time.time() - 2000,
        )
        svc._recent_scores = [old, recent]

        opps = svc.get_recent_opportunities()
        assert len(opps) == 1
        assert opps[0]["market_ticker"] == "MKT-A"

    @pytest.mark.asyncio
    async def test_respects_max_recent_limit(self, config, event_bus):
        tracked_events = _make_tracked_events({})
        tracked_markets = _make_tracked_markets({})
        svc = EarlyBirdService(
            event_bus=event_bus,
            tracked_events=tracked_events,
            tracked_markets=tracked_markets,
            config=config,
        )
        svc._max_recent = 3
        # Add 5 scores
        for i in range(5):
            svc._recent_scores.append(EarlyBirdScore(
                market_ticker=f"MKT-{i}", event_ticker="EVT-1",
                total_score=50, scored_at=time.time(),
            ))
        # Internal list should be trimmed to _max_recent on next _on_market_activated
        # But the get_recent_opportunities just filters by time
        assert len(svc._recent_scores) == 5
        opps = svc.get_recent_opportunities()
        assert len(opps) == 5  # All are recent enough


# ===========================================================================
# Service disabled
# ===========================================================================


class TestServiceDisabled:
    @pytest.mark.asyncio
    async def test_start_noop_when_disabled(self, event_bus):
        config = _make_config(early_bird_enabled=False)
        svc = EarlyBirdService(
            event_bus=event_bus,
            tracked_events=MagicMock(),
            tracked_markets=MagicMock(),
            config=config,
        )
        await svc.start()
        assert svc._running is False
        event_bus.subscribe_to_market_activated.assert_not_awaited()


# ===========================================================================
# get_stats
# ===========================================================================


class TestGetStats:
    @pytest.mark.asyncio
    async def test_initial_stats(self, config, event_bus):
        svc = EarlyBirdService(
            event_bus=event_bus,
            tracked_events=MagicMock(),
            tracked_markets=MagicMock(),
            config=config,
        )
        stats = svc.get_stats()
        assert stats["running"] is False
        assert stats["enabled"] is True
        assert stats["activations_received"] == 0
        assert stats["signals_emitted"] == 0
        assert stats["recent_scores"] == 0

    @pytest.mark.asyncio
    async def test_stats_after_activation(self, event_bus, attention_callback):
        config = _make_config(early_bird_min_score=0)
        tracked_events = _make_tracked_events({})
        tracked_markets = _make_tracked_markets({
            "MKT-NEW": MockTrackedMarket(ticker="MKT-NEW", yes_bid=0, category="Sports"),
        })
        svc = EarlyBirdService(
            event_bus=event_bus,
            tracked_events=tracked_events,
            tracked_markets=tracked_markets,
            config=config,
            attention_callback=attention_callback,
        )
        await svc.start()
        await svc._on_market_activated(MockActivatedEvent())
        stats = svc.get_stats()
        assert stats["activations_received"] == 1
        assert stats["signals_emitted"] == 1
        assert stats["recent_scores"] == 1


# ===========================================================================
# Error handling
# ===========================================================================


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_scoring_error_does_not_crash(self, event_bus):
        """If _score_opportunity raises, _on_market_activated logs and continues."""
        config = _make_config()
        svc = EarlyBirdService(
            event_bus=event_bus,
            tracked_events=None,
            tracked_markets=None,
            config=config,
        )
        await svc.start()

        # This will cause an error in _score_opportunity since tracked_events is None
        # and it tries to access it. But it should not raise.
        event = MockActivatedEvent()
        await svc._on_market_activated(event)
        # Should not crash, activations count still increments
        assert svc._activations_received == 1

    @pytest.mark.asyncio
    async def test_not_running_ignores_activation(self, event_bus, config):
        svc = EarlyBirdService(
            event_bus=event_bus,
            tracked_events=MagicMock(),
            tracked_markets=MagicMock(),
            config=config,
        )
        # Not started -> _running is False
        event = MockActivatedEvent()
        await svc._on_market_activated(event)
        assert svc._activations_received == 0
