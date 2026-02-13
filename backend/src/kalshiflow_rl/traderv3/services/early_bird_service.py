"""
Early Bird Service - Detect and score newly activated market opportunities.

When MARKET_ACTIVATED fires from the lifecycle WebSocket, this service:
1. Contextualizes the opportunity (complement pricing, series patterns, news)
2. Scores it using EarlyBirdScore heuristics
3. Signals the Captain via AttentionRouter if score exceeds threshold

V1 design: Signal-to-Captain only (no auto-execute). Captain evaluates
each early bird opportunity with full LLM reasoning.

Key Responsibilities:
    - Subscribe to MARKET_ACTIVATED events via EventBus
    - Score opportunities using deterministic heuristics (no LLM)
    - Enforce per-event cooldown to prevent signal spam
    - Expose recent scores for Captain tool query
    - Signal AttentionRouter on above-threshold opportunities

Architecture Position:
    EventBus (MARKET_ACTIVATED) -> EarlyBirdService -> AttentionRouter -> Captain
"""

import asyncio
import logging
import time
from typing import Callable, Coroutine, Dict, Any, Optional, List, TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from ..core.event_bus import EventBus
    from ..state.tracked_events import TrackedEventsState
    from ..state.tracked_markets import TrackedMarketsState
    from ..config.environment import V3Config

logger = logging.getLogger("kalshiflow_rl.traderv3.services.early_bird_service")


@dataclass
class EarlyBirdScore:
    """Score breakdown for an early bird opportunity."""
    market_ticker: str
    event_ticker: str
    total_score: float = 0.0
    complement_score: float = 0.0    # 25 max - mutually exclusive residual pricing
    series_score: float = 0.0        # 20 max - series history anchor
    category_score: float = 0.0      # 15 max - category familiarity
    news_score: float = 0.0          # 20 max - news catalyst strength
    timing_score: float = 0.0        # 10 max - time horizon preference
    risk_score: float = 0.0          # 10 max - risk/capital headroom
    strategy: str = "unknown"        # complement, series, news, captain_decide
    fair_value_estimate: Optional[float] = None  # Estimated YES price in cents
    reasoning: str = ""
    scored_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "market_ticker": self.market_ticker,
            "event_ticker": self.event_ticker,
            "total_score": self.total_score,
            "complement_score": self.complement_score,
            "series_score": self.series_score,
            "category_score": self.category_score,
            "news_score": self.news_score,
            "timing_score": self.timing_score,
            "risk_score": self.risk_score,
            "strategy": self.strategy,
            "fair_value_estimate": self.fair_value_estimate,
            "reasoning": self.reasoning,
            "scored_at": self.scored_at,
        }


class EarlyBirdService:
    """Detect and score newly activated market opportunities.

    Subscribes to MARKET_ACTIVATED events, scores them using deterministic
    heuristics, and signals the Captain via AttentionRouter when score
    exceeds the configured threshold.
    """

    def __init__(
        self,
        event_bus: "EventBus",
        tracked_events: "TrackedEventsState",
        tracked_markets: "TrackedMarketsState",
        config: "V3Config",
        attention_callback=None,  # Callable to inject attention signal
        search_service=None,  # Optional Tavily search service for news scoring
        health_callback: Optional[Callable[..., Coroutine]] = None,  # async () -> float (drawdown_pct)
    ):
        self._event_bus = event_bus
        self._tracked_events = tracked_events
        self._tracked_markets = tracked_markets
        self._config = config
        self._attention_callback = attention_callback
        self._search_service = search_service
        self._health_callback = health_callback

        # Recent scores for Captain tool query
        self._recent_scores: List[EarlyBirdScore] = []
        self._max_recent = 20

        # Cooldown tracking: event_ticker -> last_signal_time
        self._cooldowns: Dict[str, float] = {}

        # Stats
        self._activations_received = 0
        self._signals_emitted = 0
        self._running = False

    async def start(self) -> None:
        """Start listening for MARKET_ACTIVATED events."""
        if not self._config.early_bird_enabled:
            logger.info("EarlyBirdService disabled by config")
            return
        self._running = True
        await self._event_bus.subscribe_to_market_activated(self._on_market_activated)
        logger.info("EarlyBirdService started")

    async def stop(self) -> None:
        """Stop the service."""
        self._running = False
        logger.info(
            f"EarlyBirdService stopped "
            f"(activations={self._activations_received}, signals={self._signals_emitted})"
        )

    async def _on_market_activated(self, event) -> None:
        """Handle MARKET_ACTIVATED event."""
        if not self._running:
            return
        self._activations_received += 1

        market_ticker = event.market_ticker
        event_ticker = event.event_ticker

        # Check cooldown
        now = time.time()
        last_signal = self._cooldowns.get(event_ticker, 0)
        if now - last_signal < self._config.early_bird_cooldown_seconds:
            logger.debug(f"Early bird cooldown active for {event_ticker}")
            return

        # Score the opportunity
        try:
            score = await self._score_opportunity(market_ticker, event_ticker)

            # Store for Captain tool query
            self._recent_scores.append(score)
            if len(self._recent_scores) > self._max_recent:
                self._recent_scores = self._recent_scores[-self._max_recent:]

            # Signal Captain if above threshold
            if score.total_score >= self._config.early_bird_min_score:
                self._cooldowns[event_ticker] = now
                self._signals_emitted += 1

                if self._attention_callback:
                    await self._attention_callback(
                        market_ticker=market_ticker,
                        event_ticker=event_ticker,
                        score=score.total_score,
                        strategy=score.strategy,
                        fair_value=score.fair_value_estimate,
                        score_breakdown={
                            "complement_score": score.complement_score,
                            "news_score": score.news_score,
                            "category_score": score.category_score,
                            "timing_score": score.timing_score,
                            "risk_score": score.risk_score,
                        },
                    )

                logger.info(
                    f"Early bird signal: {market_ticker} score={score.total_score:.0f} "
                    f"strategy={score.strategy} fair_value={score.fair_value_estimate}"
                )
            else:
                logger.debug(
                    f"Early bird below threshold: {market_ticker} score={score.total_score:.0f}"
                )

        except Exception as e:
            logger.error(f"Early bird scoring error for {market_ticker}: {e}", exc_info=True)

    async def _score_opportunity(
        self, market_ticker: str, event_ticker: str
    ) -> EarlyBirdScore:
        """Score a newly activated market opportunity."""
        score = EarlyBirdScore(
            market_ticker=market_ticker,
            event_ticker=event_ticker,
        )

        event = (
            self._tracked_events.get_event(event_ticker)
            if self._tracked_events
            else None
        )
        market = (
            self._tracked_markets.get_market(market_ticker)
            if self._tracked_markets
            else None
        )

        # Strategy 1: Complement Pricing (strongest - deterministic)
        if event and event.mutually_exclusive and len(event.market_tickers) > 1:
            complement = self._score_complement(event, market_ticker)
            score.complement_score = complement["score"]
            if complement.get("fair_value"):
                score.fair_value_estimate = complement["fair_value"]
                score.strategy = "complement"
                score.reasoning = complement.get("reasoning", "")

        # Strategy 2: Category familiarity
        if market and market.category:
            category_lower = market.category.lower()
            familiar_categories = {
                "sports": 15,
                "crypto": 12,
                "economics": 10,
                "politics": 8,
            }
            for cat, cat_score in familiar_categories.items():
                if cat in category_lower:
                    score.category_score = cat_score
                    break
            if score.category_score == 0:
                score.category_score = 5  # Base score for any known category

        # Strategy 3: Timing score
        if market:
            now = int(time.time())
            time_to_close = (market.close_ts - now) if market.close_ts else 0
            if 3600 <= time_to_close <= 86400:  # 1h-24h: sweet spot
                score.timing_score = 10
            elif time_to_close > 86400:  # >24h
                score.timing_score = 5
            elif 0 < time_to_close < 3600:  # <1h: too short
                score.timing_score = 3

        # Strategy 4: News scoring via Tavily (if wired)
        if self._config.early_bird_use_news and self._search_service:
            score.news_score = await self._score_news(market_ticker, event)

        # Strategy 5: Risk score from account health (or default)
        score.risk_score = await self._score_risk()

        # Compute total
        score.total_score = (
            score.complement_score
            + score.series_score
            + score.category_score
            + score.news_score
            + score.timing_score
            + score.risk_score
        )

        # Select best strategy
        if not score.strategy or score.strategy == "unknown":
            if score.complement_score > 0:
                score.strategy = "complement"
            elif score.news_score > 0:
                score.strategy = "news"
            else:
                score.strategy = "captain_decide"

        return score

    async def _score_news(self, market_ticker: str, event) -> float:
        """Score based on news catalyst. Returns 0-20."""
        if not self._search_service:
            return 0.0
        try:
            title = event.title if event else market_ticker
            results = await asyncio.wait_for(
                self._search_service.search(f"Kalshi {title} prediction market"),
                timeout=5.0,
            )
            if results and len(results) > 2:
                return 15.0  # Strong news presence
            elif results and len(results) > 0:
                return 8.0   # Some news coverage
            return 0.0
        except Exception:
            return 0.0

    async def _score_risk(self) -> float:
        """Risk score from actual drawdown. Returns 0-10."""
        if not self._health_callback:
            return 8.0
        try:
            drawdown = await self._health_callback()
            if drawdown > 20:
                return 2.0
            elif drawdown > 10:
                return 5.0
            else:
                return 10.0
        except Exception:
            return 8.0

    def _score_complement(
        self, event, new_market_ticker: str
    ) -> Dict[str, Any]:
        """Score based on complement pricing in mutually exclusive events."""
        result: Dict[str, Any] = {"score": 0.0, "fair_value": None, "reasoning": ""}

        # Sum YES prices of other markets in the event
        other_prices = []
        for ticker in event.market_tickers:
            if ticker == new_market_ticker:
                continue
            other_market = self._tracked_markets.get_market(ticker)
            if other_market and other_market.yes_bid > 0:
                other_prices.append(other_market.yes_bid)

        if not other_prices:
            return result

        price_sum = sum(other_prices)
        if price_sum >= 100:
            # Overpriced event - no clear complement
            return result

        # Fair value = 100 - sum(other prices)
        fair_value = 100 - price_sum

        if 5 <= fair_value <= 95:  # Reasonable range
            result["fair_value"] = fair_value
            result["score"] = 25.0  # Full complement score
            result["reasoning"] = (
                f"Mutually exclusive event: {len(other_prices)} markets priced, "
                f"sum={price_sum}c, residual={fair_value}c"
            )
        elif fair_value > 0:
            result["fair_value"] = fair_value
            result["score"] = 15.0  # Partial score for extreme values
            result["reasoning"] = f"Complement price {fair_value}c (extreme range)"

        return result

    def get_recent_opportunities(self) -> List[Dict[str, Any]]:
        """Get recently scored opportunities for Captain tool."""
        cutoff = time.time() - 1800  # Last 30 min
        return [
            s.to_dict()
            for s in self._recent_scores
            if s.scored_at > cutoff
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics for health endpoint."""
        return {
            "running": self._running,
            "enabled": self._config.early_bird_enabled if self._config else False,
            "activations_received": self._activations_received,
            "signals_emitted": self._signals_emitted,
            "recent_scores": len(self._recent_scores),
            "active_cooldowns": sum(
                1
                for t in self._cooldowns.values()
                if time.time() - t
                < (
                    self._config.early_bird_cooldown_seconds
                    if self._config
                    else 300
                )
            ),
        }
