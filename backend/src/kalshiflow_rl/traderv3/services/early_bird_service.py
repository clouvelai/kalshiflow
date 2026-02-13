"""
Early Bird Service - Detect and score newly activated market opportunities.

When MARKET_ACTIVATED fires from the lifecycle WebSocket, this service:
1. Contextualizes the opportunity (complement pricing, series patterns, news)
2. Scores it using EarlyBirdScore heuristics
3. Signals the Captain via AttentionRouter if score exceeds threshold

Key Responsibilities:
    - Subscribe to MARKET_ACTIVATED events via EventBus
    - Score opportunities using deterministic heuristics (no LLM)
    - Enforce per-market cooldown with event-level rate limiting
    - Expose recent scores for Captain tool query
    - Signal AttentionRouter on above-threshold opportunities
    - Optionally auto-execute complement trades (disabled by default)

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
        trading_gateway=None,  # Optional gateway for auto-execution
    ):
        self._event_bus = event_bus
        self._tracked_events = tracked_events
        self._tracked_markets = tracked_markets
        self._config = config
        self._attention_callback = attention_callback
        self._search_service = search_service
        self._health_callback = health_callback
        self._trading_gateway = trading_gateway

        # Recent scores for Captain tool query
        self._recent_scores: List[EarlyBirdScore] = []
        self._max_recent = 20

        # Per-market cooldown: market_ticker -> last_signal_time
        self._cooldowns: Dict[str, float] = {}

        # Event-level rate limiting: event_ticker -> [signal_timestamps]
        self._event_signal_times: Dict[str, List[float]] = {}
        self._max_signals_per_event = 5  # Max signals per event per cooldown window

        # Stats
        self._activations_received = 0
        self._signals_emitted = 0
        self._auto_executions = 0
        self._running = False

        # Background news tasks (fire-and-forget)
        self._pending_news_tasks: List[asyncio.Task] = []

    async def start(self) -> None:
        """Start listening for MARKET_ACTIVATED events."""
        if not self._config.early_bird_enabled:
            logger.info("EarlyBirdService disabled by config")
            return
        self._running = True
        await self._event_bus.subscribe_to_market_activated(self._on_market_activated)
        logger.info("EarlyBirdService started")

    async def run_startup_scan(self) -> int:
        """Evaluate existing ACTIVE markets for early bird opportunities.

        Called after startup to catch-up on markets that were already active
        before the system started. Only evaluates mutually exclusive events
        with 2+ markets (complement pricing requires other markets).

        Returns:
            Number of signals emitted.
        """
        if not self._running or not self._tracked_events:
            return 0

        signals_before = self._signals_emitted
        scanned = 0

        for event in self._tracked_events.get_all():
            if not event.mutually_exclusive or len(event.market_tickers) < 2:
                continue

            for market_ticker in event.market_tickers:
                market = self._tracked_markets.get_market(market_ticker) if self._tracked_markets else None
                if not market:
                    continue
                # Simulate a MARKET_ACTIVATED event
                from ..core.events.lifecycle_events import MarketActivatedEvent
                from ..core.events.types import EventType
                fake_event = MarketActivatedEvent(
                    event_type=EventType.MARKET_ACTIVATED,
                    market_ticker=market_ticker,
                    event_ticker=event.event_ticker,
                    category=market.category,
                    timestamp=time.time(),
                )
                await self._on_market_activated(fake_event)
                scanned += 1

        signals_new = self._signals_emitted - signals_before
        logger.info(
            f"Early bird startup scan: {scanned} markets scanned, "
            f"{signals_new} signals emitted"
        )
        return signals_new

    async def stop(self) -> None:
        """Stop the service."""
        self._running = False
        # Cancel pending news tasks
        for task in self._pending_news_tasks:
            if not task.done():
                task.cancel()
        self._pending_news_tasks.clear()
        logger.info(
            f"EarlyBirdService stopped "
            f"(activations={self._activations_received}, signals={self._signals_emitted}, "
            f"auto_executions={self._auto_executions})"
        )

    async def _on_market_activated(self, event) -> None:
        """Handle MARKET_ACTIVATED event."""
        if not self._running:
            return
        self._activations_received += 1

        market_ticker = event.market_ticker
        event_ticker = event.event_ticker

        # Per-market cooldown check
        now = time.time()
        last_signal = self._cooldowns.get(market_ticker, 0)
        if now - last_signal < self._config.early_bird_cooldown_seconds:
            logger.debug(f"Early bird cooldown active for market {market_ticker}")
            return

        # Event-level rate limit check
        if not self._check_event_rate_limit(event_ticker, now):
            logger.debug(f"Early bird event rate limit hit for {event_ticker}")
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
                self._cooldowns[market_ticker] = now
                self._record_event_signal(event_ticker, now)
                self._signals_emitted += 1

                # Auto-execute complement if enabled
                auto_handled = False
                if self._should_auto_execute(score):
                    auto_handled = await self._auto_execute_complement(score)

                if self._attention_callback:
                    data = {
                        "complement_score": score.complement_score,
                        "news_score": score.news_score,
                        "category_score": score.category_score,
                        "timing_score": score.timing_score,
                        "risk_score": score.risk_score,
                    }
                    if auto_handled:
                        data["auto_handled"] = "complement_maker"

                    await self._attention_callback(
                        market_ticker=market_ticker,
                        event_ticker=event_ticker,
                        score=score.total_score,
                        strategy=score.strategy,
                        fair_value=score.fair_value_estimate,
                        score_breakdown=data,
                    )

                logger.info(
                    f"Early bird signal: {market_ticker} score={score.total_score:.0f} "
                    f"strategy={score.strategy} fair_value={score.fair_value_estimate}"
                    f"{' [AUTO-EXECUTED]' if auto_handled else ''}"
                )
            else:
                logger.debug(
                    f"Early bird below threshold: {market_ticker} score={score.total_score:.0f}"
                )

        except Exception as e:
            logger.error(f"Early bird scoring error for {market_ticker}: {e}", exc_info=True)

    def _check_event_rate_limit(self, event_ticker: str, now: float) -> bool:
        """Check if event-level rate limit allows another signal."""
        cooldown = self._config.early_bird_cooldown_seconds
        times = self._event_signal_times.get(event_ticker, [])
        # Remove expired timestamps
        times = [t for t in times if now - t < cooldown]
        self._event_signal_times[event_ticker] = times
        return len(times) < self._max_signals_per_event

    def _record_event_signal(self, event_ticker: str, now: float) -> None:
        """Record a signal timestamp for event-level rate limiting."""
        if event_ticker not in self._event_signal_times:
            self._event_signal_times[event_ticker] = []
        self._event_signal_times[event_ticker].append(now)

    async def _score_opportunity(
        self, market_ticker: str, event_ticker: str
    ) -> EarlyBirdScore:
        """Score a newly activated market opportunity.

        Fast path: complement + category + timing + risk (no I/O).
        Slow path: news scoring fires as background task if enabled.
        """
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

        # Strategy 4: Risk score from account health (or default)
        score.risk_score = await self._score_risk()

        # Compute fast-path total (no news)
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
            else:
                score.strategy = "captain_decide"

        # Fire news scoring as background task (non-blocking)
        if self._config.early_bird_use_news and self._search_service:
            task = asyncio.create_task(
                self._background_news_score(score, event)
            )
            self._pending_news_tasks.append(task)
            task.add_done_callback(lambda t: self._pending_news_tasks.remove(t) if t in self._pending_news_tasks else None)

        return score

    async def _background_news_score(self, score: EarlyBirdScore, event) -> None:
        """Score news in background and update the stored score."""
        try:
            news_score = await self._score_news(score.market_ticker, event)
            if news_score > 0:
                score.news_score = news_score
                score.total_score += news_score
                if news_score > 0 and score.strategy == "captain_decide":
                    score.strategy = "news"
                logger.debug(
                    f"Background news score updated: {score.market_ticker} "
                    f"news={news_score} new_total={score.total_score}"
                )
        except Exception as e:
            logger.debug(f"Background news scoring failed for {score.market_ticker}: {e}")

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
        """Score based on complement pricing in mutually exclusive events.

        Uses midprice (yes_bid + yes_ask) / 2 when available for more
        accurate fair value. Falls back to yes_bid when no ask exists.
        Reduces score when average spread > 10c (pricing less reliable).
        """
        result: Dict[str, Any] = {"score": 0.0, "fair_value": None, "reasoning": ""}

        # Collect midprices of other markets in the event
        other_prices = []
        spreads = []
        for ticker in event.market_tickers:
            if ticker == new_market_ticker:
                continue
            other_market = self._tracked_markets.get_market(ticker)
            if other_market and other_market.yes_bid > 0:
                yes_ask = getattr(other_market, 'yes_ask', 0) or 0
                if yes_ask > 0:
                    midprice = (other_market.yes_bid + yes_ask) / 2
                    spreads.append(yes_ask - other_market.yes_bid)
                else:
                    midprice = other_market.yes_bid
                other_prices.append(midprice)

        if not other_prices:
            return result

        price_sum = sum(other_prices)
        if price_sum >= 100:
            # Overpriced event - no clear complement
            return result

        # Fair value = 100 - sum(midprices)
        fair_value = 100 - price_sum

        # Determine base score
        if 5 <= fair_value <= 95:  # Reasonable range
            base_score = 25.0
        elif fair_value > 0:
            base_score = 15.0  # Partial score for extreme values
        else:
            return result

        # Reduce score when average spread is wide (pricing less reliable)
        avg_spread = sum(spreads) / len(spreads) if spreads else 0
        if avg_spread > 10:
            base_score = min(base_score, 15.0)

        result["fair_value"] = fair_value
        result["score"] = base_score

        spread_note = f", avg_spread={avg_spread:.0f}c" if spreads else ""
        if 5 <= fair_value <= 95:
            result["reasoning"] = (
                f"Mutually exclusive event: {len(other_prices)} markets priced, "
                f"sum={price_sum:.0f}c, residual={fair_value:.0f}c{spread_note}"
            )
        else:
            result["reasoning"] = f"Complement price {fair_value:.0f}c (extreme range){spread_note}"

        return result

    # ------------------------------------------------------------------
    # Auto-execution (disabled by default)
    # ------------------------------------------------------------------

    def _should_auto_execute(self, score: EarlyBirdScore) -> bool:
        """Check if this score qualifies for auto-execution."""
        if not getattr(self._config, 'early_bird_auto_execute', False):
            return False
        if not self._trading_gateway:
            return False
        if score.strategy != "complement":
            return False
        if score.fair_value_estimate is None:
            return False
        if not (20 <= score.fair_value_estimate <= 80):
            return False
        if score.total_score < 60:
            return False
        return True

    async def _auto_execute_complement(self, score: EarlyBirdScore) -> bool:
        """Auto-execute complement maker orders (YES + NO limits at fair_value +/- 2c).

        Returns True if orders were placed successfully.
        """
        try:
            fair_value = score.fair_value_estimate
            yes_price = int(fair_value - 2)  # Buy YES below fair value
            no_price = int(100 - fair_value - 2)  # Buy NO below complement

            if yes_price < 5 or no_price < 5:
                return False

            # Place both legs via trading gateway
            gateway = self._trading_gateway
            results = []

            # YES leg
            yes_result = await gateway.place_order(
                ticker=score.market_ticker,
                side="yes",
                action="buy",
                count=10,  # Conservative default
                price=yes_price,
                order_type="limit",
            )
            results.append(yes_result)

            # NO leg
            no_result = await gateway.place_order(
                ticker=score.market_ticker,
                side="no",
                action="buy",
                count=10,
                price=no_price,
                order_type="limit",
            )
            results.append(no_result)

            self._auto_executions += 1
            logger.info(
                f"Auto-executed complement: {score.market_ticker} "
                f"YES@{yes_price}c NO@{no_price}c fair_value={fair_value:.0f}c"
            )
            return True

        except Exception as e:
            logger.warning(f"Auto-execution failed for {score.market_ticker}: {e}")
            return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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
            "auto_executions": self._auto_executions,
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
