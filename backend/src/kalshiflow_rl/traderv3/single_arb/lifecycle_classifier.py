"""
LifecycleClassifier - Haiku-powered event lifecycle stage classification.

Uses a lightweight Claude Haiku call to classify where an event is in its
lifecycle based on:
- Candlestick trend (7-day price trajectory)
- Volume trend (increasing/decreasing/flat)
- News count + recency
- Spread trajectory (widening/narrowing)
- Time to close
- Market count + price distribution

Lifecycle stages:
  dormant     - No meaningful activity, market is idle
  discovery   - Early interest, sparse news, wide spreads
  building    - Growing engagement, thesis-forming news, tightening spreads
  peak        - Maximum activity, breaking developments, highest edge potential
  convergence - Prices approaching resolution, declining uncertainty
  resolution  - Near close, binary outcome becoming clear

Results are cached for 30 minutes per event.
"""

import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger("kalshiflow_rl.traderv3.single_arb.lifecycle_classifier")

LIFECYCLE_STAGES = ["dormant", "discovery", "building", "peak", "convergence", "resolution"]

CLASSIFICATION_PROMPT = """You are classifying the lifecycle stage of a prediction market event.

Based on the data below, classify this event into exactly ONE stage:

STAGES:
- dormant: No meaningful activity, market is idle. Low volume, no news, wide spreads.
- discovery: Early interest, sparse news, wide spreads. Market is still forming opinions.
- building: Growing engagement, thesis-forming news, tightening spreads. Narrative developing.
- peak: Maximum activity, breaking developments, highest edge potential. Key news driving prices.
- convergence: Prices approaching resolution, declining uncertainty. Spreads narrowing to terminal.
- resolution: Near close (<2h), binary outcome becoming clear. Most markets near 0 or 100.

EVENT DATA:
{context}

Respond with ONLY a JSON object (no markdown, no explanation):
{{"stage": "<one of the stages above>", "confidence": <0.0-1.0>, "reasoning": "<1-2 sentences>", "recommended_action": "<observe|research|trade|exit|hold>"}}
"""


class LifecycleClassifier:
    """Haiku-powered event lifecycle stage classification."""

    def __init__(self, model: str = "haiku", cache_ttl: int = 1800):
        """
        Args:
            model: Model identifier for get_extraction_llm() (default: haiku)
            cache_ttl: Cache TTL in seconds (default 30 min)
        """
        self._model = model
        self._cache_ttl = cache_ttl
        self._cache: Dict[str, Dict] = {}  # event_ticker -> {result, cached_at}
        self._llm = None  # Lazy init

    def _get_llm(self):
        """Lazy-init LangChain LLM."""
        if self._llm is None:
            from .mentions_models import get_extraction_llm
            self._llm = get_extraction_llm(model=self._model, temperature=0.0, max_tokens=200)
            logger.info(f"[LIFECYCLE] LangChain LLM initialized (model={self._model})")
        return self._llm

    def _build_context(self, event, now: float) -> str:
        """Build classification context from EventMeta."""
        lines = []
        lines.append(f"Event: {event.title}")
        lines.append(f"Ticker: {event.event_ticker}")
        lines.append(f"Markets: {len(event.markets)}")
        lines.append(f"Mutually exclusive: {event.mutually_exclusive}")

        # Time to close
        time_to_close_hours = None
        for m in event.markets.values():
            if m.close_time:
                try:
                    from datetime import datetime
                    ct = m.close_time.replace("Z", "+00:00")
                    close_dt = datetime.fromisoformat(ct)
                    hours = (close_dt.timestamp() - now) / 3600
                    if time_to_close_hours is None or hours < time_to_close_hours:
                        time_to_close_hours = hours
                except (ValueError, TypeError):
                    pass
        if time_to_close_hours is not None:
            lines.append(f"Time to close: {time_to_close_hours:.1f} hours")

        # Price distribution
        prices = []
        spreads = []
        for m in event.markets.values():
            if m.yes_mid is not None:
                prices.append(m.yes_mid)
            if m.spread is not None:
                spreads.append(m.spread)
        if prices:
            lines.append(f"Price range: {min(prices):.0f}c - {max(prices):.0f}c (avg {sum(prices)/len(prices):.0f}c)")
        if spreads:
            lines.append(f"Spread range: {min(spreads)}-{max(spreads)}c (avg {sum(spreads)/len(spreads):.0f}c)")

        # Volume and microstructure
        total_vol = sum(m.micro.volume_5m for m in event.markets.values())
        total_trades = sum(m.trade_count for m in event.markets.values())
        total_whales = sum(m.micro.whale_trade_count for m in event.markets.values())
        lines.append(f"Volume (5min): {total_vol} contracts")
        lines.append(f"Total trades: {total_trades}")
        lines.append(f"Whale trades: {total_whales}")

        # Candlestick trend
        if event.candlesticks:
            cs = event.candlestick_summary()
            if cs:
                for ticker, info in list(cs.items())[:3]:  # Top 3 markets
                    trend = info.get("price_trend", "flat")
                    current = info.get("price_current")
                    avg_7d = info.get("price_7d_avg")
                    lines.append(f"  {ticker}: trend={trend}, current={current}c, 7d_avg={avg_7d}c")

        # News context
        if event.understanding:
            news = event.understanding.get("news_articles", [])
            lines.append(f"News articles: {len(news)}")
            if news:
                news_fetched = event.understanding.get("news_fetched_at", 0)
                if news_fetched:
                    lines.append(f"News freshness: {(now - news_fetched) / 3600:.1f} hours")
                # Headlines
                for a in news[:3]:
                    lines.append(f"  - {a.get('title', '')[:80]}")

        return "\n".join(lines)

    async def classify(self, event) -> Dict[str, Any]:
        """Classify event lifecycle stage using lightweight LLM.

        Args:
            event: EventMeta instance

        Returns:
            {stage, confidence, reasoning, recommended_action}
        """
        now = time.time()

        # Check cache
        cached = self._cache.get(event.event_ticker)
        if cached and (now - cached["cached_at"]) < self._cache_ttl:
            return cached["result"]

        # Build context
        context = self._build_context(event, now)
        prompt = CLASSIFICATION_PROMPT.format(context=context)

        try:
            from .llm_schemas import LifecycleResult as LifecycleResultSchema

            llm = self._get_llm()
            structured_llm = llm.with_structured_output(LifecycleResultSchema)
            parsed = await structured_llm.ainvoke(prompt)

            result = parsed.model_dump()

            # Validate stage
            stage = result.get("stage", "unknown")
            if stage not in LIFECYCLE_STAGES:
                logger.warning(f"[LIFECYCLE] Invalid stage '{stage}' for {event.event_ticker}, defaulting to 'discovery'")
                result["stage"] = "discovery"

            # Cache result
            self._cache[event.event_ticker] = {
                "result": result,
                "cached_at": now,
            }

            logger.info(
                f"[LIFECYCLE] {event.event_ticker}: stage={result['stage']} "
                f"confidence={result.get('confidence', 'N/A')} "
                f"action={result.get('recommended_action', 'N/A')}"
            )
            return result

        except Exception as e:
            logger.warning(f"[LIFECYCLE] Classification failed for {event.event_ticker}: {e}")
            # Return a heuristic fallback and cache it to avoid repeated failing API calls
            result = self._heuristic_classify(event, now)
            self._cache[event.event_ticker] = {
                "result": result,
                "cached_at": now,
            }
            return result

    def _heuristic_classify(self, event, now: float) -> Dict[str, Any]:
        """Fallback heuristic classification when LLM fails."""
        # Time to close
        time_to_close = None
        for m in event.markets.values():
            if m.close_time:
                try:
                    from datetime import datetime
                    ct = m.close_time.replace("Z", "+00:00")
                    close_dt = datetime.fromisoformat(ct)
                    hours = (close_dt.timestamp() - now) / 3600
                    if time_to_close is None or hours < time_to_close:
                        time_to_close = hours
                except (ValueError, TypeError):
                    pass

        total_vol = sum(m.micro.volume_5m for m in event.markets.values())
        total_trades = sum(m.trade_count for m in event.markets.values())

        if time_to_close is not None and time_to_close < 2:
            stage = "resolution"
        elif time_to_close is not None and time_to_close < 6:
            stage = "convergence"
        elif total_vol > 100 and total_trades > 50:
            stage = "peak"
        elif total_trades > 10:
            stage = "building"
        elif total_trades > 0:
            stage = "discovery"
        else:
            stage = "dormant"

        return {
            "stage": stage,
            "confidence": 0.4,
            "reasoning": "Heuristic fallback (LLM unavailable)",
            "recommended_action": {
                "dormant": "observe",
                "discovery": "observe",
                "building": "research",
                "peak": "trade",
                "convergence": "exit",
                "resolution": "hold",
            }.get(stage, "observe"),
        }

    def get_cached(self, event_ticker: str) -> Optional[Dict[str, Any]]:
        """Get cached classification if available."""
        cached = self._cache.get(event_ticker)
        if cached:
            return cached["result"]
        return None
