"""
Price Impact Agent - Transforms entity sentiment into market price impact.

Subscribes to Supabase Realtime INSERT events on both the reddit_entities
and news_entities tables, transforming raw entity sentiment into
market-specific price impact signals.

Data flow:
  Reddit:  Reddit stream → LLMEntityExtractor → reddit_entities → Realtime → _handle_entity_insert → _process_entity_record()
  News:    DDGS search   → LLMEntityExtractor → news_entities   → Realtime → _handle_news_insert  → _process_entity_record()

Key transformation rules:
- OUT markets: Invert sentiment (scandal = more likely OUT)
- WIN/CONFIRM/NOMINEE: Preserve sentiment direction

The agent:
1. Listens for new reddit_entities AND news_entities via Supabase Realtime
2. Normalizes news_entities fields to common format via adapter
3. Looks up entity → market mappings from EntityMarketIndex
4. Applies transformation rules based on market type
5. Inserts PriceImpactSignal into market_price_impacts table
6. Broadcasts signals to frontend via WebSocket
7. Strong Reddit signals trigger DDGS news search → news_entities pipeline
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from .base_agent import BaseAgent
from ..schemas.entity_schemas import (
    ExtractedEntity,
    PriceImpactSignal,
    RedditEntitySignal,
    compute_price_impact,
    IMPACT_RULES,
)
from ..services.price_impact_store import get_price_impact_store

if TYPE_CHECKING:
    from ..core.websocket_manager import V3WebSocketManager
    from ..core.event_bus import EventBus
    from ..services.entity_market_index import EntityMarketIndex

logger = logging.getLogger("kalshiflow_rl.traderv3.agents.price_impact_agent")


@dataclass
class PriceImpactAgentConfig:
    """Configuration for the Price Impact Agent."""

    # Supabase connection
    supabase_url: Optional[str] = None
    supabase_key: Optional[str] = None

    # Processing
    min_confidence: float = 0.5  # Minimum entity confidence to process
    min_sentiment_magnitude: int = 20  # Minimum |sentiment| to create impact

    # Reconnection (exponential backoff for full-recreation)
    reconnect_backoff_base: float = 5.0
    reconnect_backoff_max: float = 300.0  # 5 minutes max backoff

    # Timeouts for Supabase Realtime operations
    subscribe_timeout: float = 30.0
    unsubscribe_timeout: float = 10.0

    # SDK configuration (override defaults)
    realtime_timeout: int = 30  # SDK join push timeout (default 10s is too short)
    realtime_max_retries: int = 20  # SDK initial WS connect retries (default 5)

    # Nuclear reconnect: full client recreation after this many seconds unhealthy
    nuclear_reconnect_threshold: float = 600.0  # 10 minutes

    # LLM impact assessment
    llm_assessment_enabled: bool = True
    llm_assessment_model: str = "gpt-4o-mini"
    llm_assessment_timeout: float = 8.0  # seconds

    # News corroboration search
    news_search_enabled: bool = True
    news_search_min_impact: int = 40  # Minimum |price_impact_score| to trigger search
    news_search_min_confidence: float = 0.7  # Minimum confidence to trigger search
    news_search_cooldown_minutes: float = 15.0  # Cooldown per entity_id
    news_search_max_per_hour: int = 20  # Global rate limit
    news_search_max_age_minutes: int = 60  # Only consider articles within this window
    news_search_max_results: int = 8  # Max DDGS results per search
    news_search_model: str = "gpt-4o-mini"  # Model for corroboration scoring
    news_search_timeout: float = 10.0  # Timeout for LLM corroboration call

    enabled: bool = True


class NewsSearchCache:
    """Deduplication, rate limiting, and DDGS result caching for news searches."""

    def __init__(
        self,
        cooldown_minutes: float = 15.0,
        max_per_hour: int = 20,
        query_cache_ttl_seconds: float = 300.0,
    ):
        self._cooldown_seconds = cooldown_minutes * 60
        self._max_per_hour = max_per_hour
        self._query_cache_ttl = query_cache_ttl_seconds
        # entity_id -> last search timestamp
        self._entity_last_search: Dict[str, float] = {}
        # Timestamps of all searches in the last hour (for rate limiting)
        self._search_timestamps: List[float] = []
        # DDGS query -> (results, timestamp) for deduplication
        self._query_cache: Dict[str, tuple] = {}

    def can_search(self, entity_id: str) -> bool:
        """Check if a search is allowed for this entity (cooldown + rate limit)."""
        now = time.time()

        # Entity cooldown check
        last_search = self._entity_last_search.get(entity_id)
        if last_search is not None and (now - last_search) < self._cooldown_seconds:
            return False

        # Global rate limit check (prune old timestamps first)
        hour_ago = now - 3600
        self._search_timestamps = [t for t in self._search_timestamps if t > hour_ago]
        if len(self._search_timestamps) >= self._max_per_hour:
            return False

        return True

    def record_search(self, entity_id: str) -> None:
        """Record that a search was performed."""
        now = time.time()
        self._entity_last_search[entity_id] = now
        self._search_timestamps.append(now)

    def get_cached_results(self, query: str) -> Optional[List[dict]]:
        """Get cached DDGS results if still fresh."""
        entry = self._query_cache.get(query)
        if entry is None:
            return None
        results, cached_at = entry
        if (time.time() - cached_at) > self._query_cache_ttl:
            del self._query_cache[query]
            return None
        return results

    def cache_results(self, query: str, results: List[dict]) -> None:
        """Cache DDGS results for a query."""
        self._query_cache[query] = (results, time.time())

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        now = time.time()
        hour_ago = now - 3600
        self._search_timestamps = [t for t in self._search_timestamps if t > hour_ago]
        return {
            "entities_tracked": len(self._entity_last_search),
            "searches_this_hour": len(self._search_timestamps),
            "max_per_hour": self._max_per_hour,
            "query_cache_size": len(self._query_cache),
        }


class PriceImpactAgent(BaseAgent):
    """
    Agent that transforms entity sentiment into market price impact signals.

    Subscribes to Supabase Realtime for reddit_entities INSERT events
    and creates price impact signals for the Deep Agent to consume.
    """

    def __init__(
        self,
        config: Optional[PriceImpactAgentConfig] = None,
        websocket_manager: Optional["V3WebSocketManager"] = None,
        event_bus: Optional["EventBus"] = None,
        entity_index: Optional["EntityMarketIndex"] = None,
    ):
        """
        Initialize the Price Impact Agent.

        Args:
            config: Agent configuration
            websocket_manager: For broadcasting to frontend
            event_bus: For emitting events to other agents
            entity_index: Entity-to-market mapping index
        """
        super().__init__(
            name="price_impact",
            display_name="Price Impact Agent",
            event_bus=event_bus,
            websocket_manager=websocket_manager,
        )

        self._config = config or PriceImpactAgentConfig()
        self._entity_index = entity_index

        # Supabase Realtime channel
        self._supabase = None
        self._channel = None
        self._subscription_task: Optional[asyncio.Task] = None

        # Stats
        self._entities_received = 0
        self._impacts_created = 0
        self._entities_skipped = 0
        self._reconnect_attempts = 0

        # Health monitoring for silent disconnect detection
        self._last_signal_time: Optional[float] = None
        self._signal_timeout_seconds: float = 300.0  # 5 minutes without signal = reconnect
        self._health_check_interval: float = 60.0  # Check health every 60 seconds
        self._subscription_healthy: bool = False
        self._subscription_state: str = "disconnected"
        self._last_healthy_time: Optional[float] = None

        # LLM client for impact assessment
        self._llm_client = None
        self._llm_assessments = 0
        self._llm_failures = 0

        # News search cache and stats
        self._news_search_cache = NewsSearchCache(
            cooldown_minutes=self._config.news_search_cooldown_minutes,
            max_per_hour=self._config.news_search_max_per_hour,
        )
        self._news_searches_total = 0
        self._news_searches_enriched = 0
        self._news_searches_failed = 0

        # Callback for external signal handling
        self._signal_callback: Optional[Callable[[PriceImpactSignal], None]] = None

    def set_signal_callback(self, callback: Callable[[PriceImpactSignal], None]) -> None:
        """Set callback for when price impact signals are created."""
        self._signal_callback = callback

    async def _on_start(self) -> None:
        """Initialize resources on agent start."""
        if not self._config.enabled:
            logger.info("[price_impact] Agent disabled")
            return

        # Initialize Supabase client
        if not await self._init_supabase():
            logger.warning("[price_impact] Supabase not available, using event bus only")

        # Load recent signals from Supabase to populate in-memory store
        # This prevents the store from being empty after a restart
        if self._supabase:
            store = get_price_impact_store()
            loaded = await store.load_from_supabase(self._supabase, max_age_hours=2.0)
            logger.info(f"[price_impact] Loaded {loaded} signals from Supabase on startup")

        # Initialize signal time so health check works from startup
        self._last_signal_time = time.time()

        # Start Realtime subscription
        self._subscription_task = asyncio.create_task(self._subscription_loop())

        # Note: Event bus uses typed events (EventType enum), so we rely on
        # direct PriceImpactStore ingestion from RedditEntityAgent instead
        # of event bus subscription for cross-agent communication

        logger.info("[price_impact] Started listening for entity signals")

    async def _on_stop(self) -> None:
        """Cleanup resources on agent stop."""
        # Unsubscribe from Realtime with timeout
        if self._channel:
            try:
                await asyncio.wait_for(
                    self._channel.unsubscribe(),
                    timeout=self._config.unsubscribe_timeout,
                )
            except asyncio.TimeoutError:
                logger.warning("[price_impact] Unsubscribe timed out during shutdown")
            except Exception as e:
                logger.error(f"[price_impact] Unsubscribe error: {e}")

        if self._subscription_task and not self._subscription_task.done():
            self._subscription_task.cancel()
            try:
                await self._subscription_task
            except asyncio.CancelledError:
                pass

        logger.info(
            f"[price_impact] Stopped. Received: {self._entities_received}, "
            f"Created: {self._impacts_created}, Skipped: {self._entities_skipped}"
        )

    def _get_llm_client(self):
        """Get or create async OpenAI client for impact assessment."""
        if self._llm_client is None:
            import os
            from openai import AsyncOpenAI
            self._llm_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return self._llm_client

    def _get_or_create_extractor(self):
        """Lazy-init LLMEntityExtractor for news article entity extraction."""
        if not hasattr(self, "_entity_extractor") or self._entity_extractor is None:
            from ..nlp.llm_entity_extractor import LLMEntityExtractor
            self._entity_extractor = LLMEntityExtractor(
                model=self._config.news_search_model,
                timeout=self._config.news_search_timeout,
            )
        return self._entity_extractor

    async def _assess_price_impact(
        self,
        entity_name: str,
        sentiment_score: int,
        context_snippet: str,
        source_title: str,
        market_ticker: str,
        market_type: str,
    ) -> Optional[tuple]:
        """
        Use LLM to assess how entity news impacts a specific market.

        Returns: (price_impact_score, transformation_logic, suggested_side)
                 or None if LLM fails (signal should be skipped).
        """
        if not self._config.llm_assessment_enabled:
            impact = compute_price_impact(sentiment_score, market_type)
            logic = IMPACT_RULES.get(market_type, {}).get("description", "")
            side = "YES" if impact > 0 else "NO"
            return impact, logic, side

        try:
            import json as json_mod
            client = self._get_llm_client()

            news_context = context_snippet or source_title
            prompt = f"""Assess how this news impacts this specific prediction market.

Entity: {entity_name}
News context: {news_context}
Source headline: {source_title}

Market: {market_ticker}
Market type: {market_type}

Return JSON:
{{
  "price_impact": <integer -100 to +100>,
  "reasoning": "<1 sentence explaining the causal link between this news and this market>",
  "side": "<YES or NO>",
  "confidence": "<low|medium|high>"
}}

Guidelines:
- price_impact: How much does this news shift the market probability?
  * ±5-15: Tangentially related, weak signal
  * ±15-40: Relevant but not decisive
  * ±40-70: Directly relevant, clear directional impact
  * ±70-100: Major breaking news directly about this market
- For "OUT" markets (e.g. KXBONDIOUT): negative news about the person = positive impact (more likely ousted)
- For other markets: positive news = positive impact, negative news = negative impact
- If the news is NOT meaningfully connected to this market, return price_impact: 0
- confidence: how certain are you about the direction and magnitude?
- Return valid JSON only."""

            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=self._config.llm_assessment_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=150,
                    temperature=0,
                ),
                timeout=self._config.llm_assessment_timeout,
            )

            content = response.choices[0].message.content.strip()
            # Strip markdown code blocks
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(l for l in lines if not l.strip().startswith("```"))

            result = json_mod.loads(content)

            impact = max(-100, min(100, int(result.get("price_impact", 0))))
            reasoning = result.get("reasoning", "")
            side = result.get("side", "YES" if impact > 0 else "NO").upper()

            self._llm_assessments += 1
            return impact, reasoning, side

        except Exception as e:
            logger.error(f"[price_impact] LLM assessment failed for {entity_name}/{market_ticker}: {e}")
            self._llm_failures += 1
            return None  # Signal to caller: skip this market

    def _get_agent_stats(self) -> Dict[str, Any]:
        """Get agent-specific statistics."""
        # Calculate seconds since last signal for health monitoring
        seconds_since_signal = None
        if self._last_signal_time is not None:
            seconds_since_signal = time.time() - self._last_signal_time

        return {
            "entities_received": self._entities_received,
            "impacts_created": self._impacts_created,
            "entities_skipped": self._entities_skipped,
            "reconnect_attempts": self._reconnect_attempts,
            "supabase_connected": self._channel is not None,
            "subscription_healthy": self._subscription_healthy,
            "subscription_state": self._subscription_state,
            "entity_index_available": self._entity_index is not None,
            "last_signal_time": self._last_signal_time,
            "last_healthy_time": self._last_healthy_time,
            "seconds_since_signal": seconds_since_signal,
            "signal_timeout_seconds": self._signal_timeout_seconds,
            "llm_assessments": self._llm_assessments,
            "llm_failures": self._llm_failures,
            "news_search_enabled": self._config.news_search_enabled,
            "news_searches_total": self._news_searches_total,
            "news_searches_enriched": self._news_searches_enriched,
            "news_searches_failed": self._news_searches_failed,
            "news_search_cache": self._news_search_cache.get_stats(),
        }

    async def _init_supabase(self) -> bool:
        """Initialize Supabase async client for Realtime."""
        try:
            import os
            from supabase import acreate_client, AsyncClient

            url = self._config.supabase_url or os.getenv("SUPABASE_URL")
            key = self._config.supabase_key or os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_ANON_KEY")

            if not url or not key:
                logger.warning(
                    "[price_impact] SUPABASE_URL and SUPABASE_KEY required for Realtime"
                )
                return False

            # Use async client for Realtime support
            self._supabase: AsyncClient = await acreate_client(url, key)

            # Configure Realtime for stability (override SDK defaults)
            self._supabase.realtime.timeout = self._config.realtime_timeout
            self._supabase.realtime.max_retries = self._config.realtime_max_retries
            logger.info(
                f"[price_impact] Supabase async client initialized "
                f"(timeout={self._config.realtime_timeout}s, "
                f"max_retries={self._config.realtime_max_retries})"
            )
            return True

        except ImportError:
            logger.error("[price_impact] supabase package not installed")
            return False
        except Exception as e:
            logger.error(f"[price_impact] Supabase init error: {e}")
            return False

    async def _subscription_loop(self) -> None:
        """Main loop: subscribe once, let SDK handle transient failures.

        The SDK's built-in rejoin_timer uses exponential backoff (2^tries seconds)
        to retry channel joins on timeout. We only intervene with a full client
        recreation (nuclear reconnect) if the subscription has been unhealthy
        for longer than nuclear_reconnect_threshold (default 10 minutes).
        """
        while self._running:
            try:
                await self._subscribe_to_entities()

                while self._running and self._channel:
                    await asyncio.sleep(self._health_check_interval)

                    # If SDK's built-in reconnection has recovered, update timer
                    if self._subscription_healthy:
                        self._last_healthy_time = time.time()
                        continue

                    # Not healthy - how long has it been?
                    reference_time = (
                        self._last_healthy_time
                        or self._last_signal_time
                        or time.time()
                    )
                    unhealthy_duration = time.time() - reference_time

                    if unhealthy_duration > self._config.nuclear_reconnect_threshold:
                        logger.warning(
                            f"[price_impact] Unhealthy for {unhealthy_duration:.0f}s, "
                            f"recreating Supabase client (nuclear reconnect)"
                        )
                        await self._nuclear_reconnect()
                        break  # Break inner loop to re-subscribe with fresh client
                    else:
                        logger.info(
                            f"[price_impact] Subscription unhealthy for {unhealthy_duration:.0f}s, "
                            f"SDK rejoin_timer active (nuclear at {self._config.nuclear_reconnect_threshold:.0f}s)"
                        )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[price_impact] Subscription error: {e}")
                self._record_error(str(e))

                self._reconnect_attempts += 1
                backoff = min(
                    self._config.reconnect_backoff_base * (2 ** self._reconnect_attempts),
                    self._config.reconnect_backoff_max,
                )
                logger.info(f"[price_impact] Reconnect backoff: {backoff:.0f}s")
                await asyncio.sleep(backoff)

    async def _nuclear_reconnect(self) -> None:
        """Destroy and recreate Supabase client + channel. Last resort."""
        self._reconnect_attempts += 1

        # Clean up old channel
        if self._channel:
            try:
                await asyncio.wait_for(
                    self._channel.unsubscribe(),
                    timeout=self._config.unsubscribe_timeout,
                )
            except asyncio.TimeoutError:
                logger.warning("[price_impact] Unsubscribe timed out during nuclear reconnect")
            except Exception as e:
                logger.warning(f"[price_impact] Unsubscribe error during nuclear reconnect: {e}")
            self._channel = None

        # Close old Realtime connection
        if self._supabase and hasattr(self._supabase, "realtime"):
            try:
                await self._supabase.realtime.close()
            except Exception:
                pass

        self._supabase = None
        self._subscription_healthy = False
        self._subscription_state = "disconnected"

        # Recreate client
        if not await self._init_supabase():
            logger.error("[price_impact] Failed to recreate Supabase client in nuclear reconnect")
            return

        self._last_healthy_time = time.time()  # Reset timer to avoid immediate re-trigger

    async def _subscribe_to_entities(self) -> None:
        """Subscribe to reddit_entities and news_entities INSERT events.

        Uses a single Realtime channel with two table listeners. The SDK's
        subscribe(callback=...) form tracks subscription state transitions
        (SUBSCRIBED, TIMED_OUT, CHANNEL_ERROR). The SDK's built-in
        rejoin_timer handles transient failures automatically.
        """
        if not self._supabase:
            logger.warning("[price_impact] No Supabase client for subscription")
            return

        try:
            from realtime.types import RealtimeSubscribeStates

            self._channel = self._supabase.channel("entities-price-impact")

            # Reddit entities (existing)
            self._channel.on_postgres_changes(
                event="INSERT",
                schema="public",
                table="reddit_entities",
                callback=self._handle_entity_insert,
            )

            # News entities (new - same channel, different table)
            self._channel.on_postgres_changes(
                event="INSERT",
                schema="public",
                table="news_entities",
                callback=self._handle_news_insert,
            )

            def on_subscribe_state(state: RealtimeSubscribeStates, error=None):
                """Track subscription state - SDK calls this on state changes."""
                self._subscription_state = state.value if hasattr(state, "value") else str(state)
                if state == RealtimeSubscribeStates.SUBSCRIBED:
                    self._subscription_healthy = True
                    self._last_healthy_time = time.time()
                    self._reconnect_attempts = 0
                    logger.info("[price_impact] Subscribed to reddit_entities + news_entities Realtime")
                elif state == RealtimeSubscribeStates.TIMED_OUT:
                    self._subscription_healthy = False
                    logger.warning(
                        "[price_impact] Subscribe timed out - SDK rejoin_timer will retry"
                    )
                elif state == RealtimeSubscribeStates.CHANNEL_ERROR:
                    self._subscription_healthy = False
                    logger.error(f"[price_impact] Channel error: {error}")
                elif state == RealtimeSubscribeStates.CLOSED:
                    self._subscription_healthy = False
                    logger.info("[price_impact] Channel closed")

            await self._channel.subscribe(callback=on_subscribe_state)

        except Exception as e:
            logger.error(f"[price_impact] Subscription setup error: {e}")
            self._channel = None
            raise

    def _handle_entity_insert(self, payload: Dict[str, Any]) -> None:
        """
        Handle INSERT event from Supabase Realtime.

        This is called synchronously by the Supabase client,
        so we schedule the async processing.
        """
        try:
            # Supabase Realtime sends data in payload['data']['record']
            new_record = payload.get("data", {}).get("record", {})
            if not new_record:
                logger.debug(f"[price_impact] No record in payload: {list(payload.keys())}")
                return

            # Schedule async processing
            asyncio.create_task(self._process_entity_record(new_record))

        except Exception as e:
            logger.error(f"[price_impact] Handle insert error: {e}")

    def _handle_news_insert(self, payload: Dict[str, Any]) -> None:
        """
        Handle INSERT from news_entities — normalize to common format.

        Adapts news_entities fields to the field names _process_entity_record() expects,
        then schedules the same async processing path as Reddit entities.
        """
        try:
            record = payload.get("data", {}).get("record", {})
            if not record:
                return
            # Normalize to the format _process_entity_record() expects
            record["post_id"] = record.get("article_url", "")
            record["title"] = record.get("headline", "")
            record["subreddit"] = ""
            record["post_created_utc"] = record.get("published_at")
            record["source_type"] = "news_article"
            record["content_type"] = record.get("content_type", "news")
            record["source_domain"] = record.get("source_domain", "news")
            asyncio.create_task(self._process_entity_record(record))
        except Exception as e:
            logger.error(f"[price_impact] Handle news insert error: {e}")

    async def _handle_event_bus_signal(self, data: Dict[str, Any]) -> None:
        """Handle entity signal from event bus (from RedditEntityAgent)."""
        try:
            await self._process_entity_record(data)
        except Exception as e:
            logger.error(f"[price_impact] Event bus signal error: {e}")

    async def _process_entity_record(self, record: Dict[str, Any]) -> None:
        """
        Process a new reddit_entities record and create price impacts.

        Args:
            record: New record from Supabase or event bus
        """
        try:
            self._entities_received += 1
            self._record_event_processed()

            # Update health monitoring - track when we last received a signal
            self._last_signal_time = time.time()

            db_id = record.get("id")
            post_id = record.get("post_id", "")
            subreddit = record.get("subreddit", "")
            title = record.get("title", "")
            # Original Reddit timestamp - may be Unix float (event bus) or ISO string (Supabase Realtime)
            raw_created_utc = record.get("post_created_utc")
            post_created_utc = None
            if raw_created_utc is not None:
                if isinstance(raw_created_utc, (int, float)):
                    post_created_utc = float(raw_created_utc)
                elif isinstance(raw_created_utc, str):
                    try:
                        from datetime import datetime, timezone
                        dt = datetime.fromisoformat(raw_created_utc.replace("Z", "+00:00"))
                        post_created_utc = dt.timestamp()
                    except (ValueError, TypeError):
                        post_created_utc = None
            entities_data = record.get("entities", [])

            if not entities_data:
                self._entities_skipped += 1
                return

            logger.info(
                f"[price_impact] Processing {len(entities_data)} entities from {post_id}"
            )

            # Process each entity
            impacts_created = 0

            for entity_data in entities_data:
                # Handle both dict and object formats
                if isinstance(entity_data, dict):
                    entity_id = entity_data.get("entity_id", "")
                    canonical_name = entity_data.get("canonical_name", "")
                    entity_type = entity_data.get("entity_type", "")
                    sentiment_score = entity_data.get("sentiment_score", 0)
                    confidence = entity_data.get("confidence", 0.5)
                else:
                    entity_id = getattr(entity_data, "entity_id", "")
                    canonical_name = getattr(entity_data, "canonical_name", "")
                    entity_type = getattr(entity_data, "entity_type", "")
                    sentiment_score = getattr(entity_data, "sentiment_score", 0)
                    confidence = getattr(entity_data, "confidence", 0.5)

                # Skip low confidence entities
                if confidence < self._config.min_confidence:
                    continue

                # Skip low magnitude sentiment
                if abs(sentiment_score) < self._config.min_sentiment_magnitude:
                    continue

                # Look up markets for this entity
                if not self._entity_index:
                    logger.warning("[price_impact] No entity index available")
                    continue

                markets = self._entity_index.get_markets_for_entity(canonical_name)

                if not markets:
                    logger.debug(
                        f"[price_impact] No markets found for: {canonical_name}"
                    )
                    continue

                # Get context_snippet from entity data if available
                if isinstance(entity_data, dict):
                    context_snippet = entity_data.get("context_snippet", "")
                else:
                    context_snippet = getattr(entity_data, "context_snippet", "")

                # Get source metadata from record (set by reddit_entity_agent)
                source_type = record.get("source_type", "reddit_text")
                content_type = record.get("content_type", "text")
                source_domain = record.get("source_domain", "reddit.com")

                # Create price impact for each market
                for mapping in markets:
                    # Assess price impact using LLM (skips market on failure)
                    result = await self._assess_price_impact(
                        entity_name=canonical_name,
                        sentiment_score=sentiment_score,
                        context_snippet=context_snippet,
                        source_title=title,
                        market_ticker=mapping.market_ticker,
                        market_type=mapping.market_type,
                    )

                    if result is None:
                        # LLM failed — skip this market rather than producing low-quality signal
                        continue

                    price_impact, transformation_logic, suggested_side = result

                    # Create signal data
                    signal_data = {
                        "signal_id": f"{post_id}_{entity_id}_{mapping.market_ticker}",
                        "market_ticker": mapping.market_ticker,
                        "event_ticker": mapping.event_ticker,
                        "entity_id": entity_id,
                        "entity_name": canonical_name,
                        "sentiment_score": sentiment_score,
                        "price_impact_score": price_impact,
                        "market_type": mapping.market_type,
                        "transformation_logic": transformation_logic,
                        "confidence": confidence,
                        "suggested_side": suggested_side,
                        "source_post_id": post_id,
                        "source_subreddit": subreddit,
                        "source_title": title,  # Reddit post title for context
                        "context_snippet": context_snippet,  # Text around entity mention
                        "created_at": time.time(),
                        "source_created_at": post_created_utc,  # Original Reddit post timestamp
                        "source_type": source_type,  # reddit_text, video_transcript, article_extract
                        "content_type": content_type,  # text, video, link, image, social
                        "source_domain": source_domain,  # youtube.com, foxnews.com, reddit.com
                        "agent_status": "pending",  # Default status, updated by deep agent
                    }

                    # Ingest to in-memory store for fast DeepAgent queries
                    try:
                        store = get_price_impact_store()
                        store.ingest({
                            "signal_id": signal_data["signal_id"],
                            "market_ticker": signal_data["market_ticker"],
                            "entity_id": signal_data["entity_id"],
                            "entity_name": signal_data["entity_name"],
                            "sentiment_score": signal_data["sentiment_score"],
                            "price_impact_score": signal_data["price_impact_score"],
                            "confidence": signal_data["confidence"],
                            "market_type": signal_data["market_type"],
                            "event_ticker": signal_data["event_ticker"],
                            "transformation_logic": signal_data["transformation_logic"],
                            "source_subreddit": signal_data["source_subreddit"],
                            "source_title": signal_data["source_title"],
                            "context_snippet": signal_data["context_snippet"],
                            "created_at": signal_data["created_at"],
                            "source_created_at": signal_data["source_created_at"],
                            "source_type": signal_data["source_type"],
                            "content_type": signal_data["content_type"],
                            "source_domain": signal_data["source_domain"],
                            "agent_status": signal_data["agent_status"],
                        })
                    except Exception as e:
                        logger.warning(f"[price_impact] Store ingest failed: {e}")

                    # Broadcast to frontend
                    if self._ws_manager:
                        await self._ws_manager.broadcast_message("price_impact", signal_data)

                    # Signals are stored in PriceImpactStore for DeepAgent to query
                    # (EventBus uses typed events, not suitable for custom signal types)

                    # Persist to Supabase for DeepAgent queries
                    await self._insert_price_impact(signal_data, db_id)

                    # Update entity reddit stats in the index
                    if self._entity_index:
                        self._entity_index.update_entity_reddit_stats(
                            entity_id=entity_id,
                            mentions=1,
                            sentiment=float(sentiment_score),
                        )

                        # Broadcast entity signal update to frontend
                        entity = self._entity_index.get_canonical_entity(entity_id)
                        if entity and self._ws_manager:
                            await self._ws_manager.broadcast_entity_signal_update(
                                entity_id=entity_id,
                                canonical_name=canonical_name,
                                reddit_stats={
                                    # Use consistent field names (mention_count for frontend)
                                    "mention_count": entity.reddit_mentions,
                                    "aggregate_sentiment": entity.aggregate_sentiment,
                                    "last_signal_at": entity.last_reddit_signal,
                                },
                            )

                    # Trigger background news corroboration search if signal is strong enough
                    if self._should_search_news(signal_data):
                        asyncio.create_task(self._search_and_enrich(signal_data))

                    impacts_created += 1
                    self._impacts_created += 1

            if impacts_created > 0:
                logger.info(
                    f"[price_impact] Created {impacts_created} impacts from {post_id}"
                )

        except Exception as e:
            logger.error(f"[price_impact] Process error: {e}")
            self._record_error(str(e))

    def _should_search_news(self, signal_data: Dict[str, Any]) -> bool:
        """Check if a signal meets the threshold for news corroboration search."""
        if not self._config.news_search_enabled:
            return False

        # Loop prevention: news-sourced signals must not trigger further searches
        if signal_data.get("content_type") == "news":
            return False

        impact = abs(signal_data.get("price_impact_score", 0))
        confidence = signal_data.get("confidence", 0.0)
        entity_id = signal_data.get("entity_id", "")

        # Strong signal threshold
        if impact < self._config.news_search_min_impact:
            return False
        if confidence < self._config.news_search_min_confidence:
            return False

        # Dedup + rate limit
        if not self._news_search_cache.can_search(entity_id):
            return False

        return True

    @staticmethod
    def _build_news_query(entity_name: str, market_type: str) -> str:
        """Build a targeted DDGS query from entity name and market type context."""
        context_terms = {
            "OUT": "resign fired removed",
            "CONFIRM": "confirmation senate vote",
            "WIN": "election polls",
            "NOMINEE": "nomination candidate",
            "PRESIDENT": "president executive",
        }
        suffix_words = context_terms.get(market_type, "")
        suffix = suffix_words.split()[0] if suffix_words else ""
        return f"{entity_name} {suffix}".strip()

    async def _search_and_enrich(self, signal_data: Dict[str, Any]) -> None:
        """
        Background task: search DDGS for news, extract entities with
        LLMEntityExtractor, and INSERT into news_entities table.

        The Realtime subscription on news_entities handles the rest:
        _handle_news_insert() → _process_entity_record() → price impact signals.

        Flow: DDGS fetch → age filter → LLMEntityExtractor.extract() → INSERT news_entities
        """
        entity_id = signal_data.get("entity_id", "")
        entity_name = signal_data.get("entity_name", "")
        market_type = signal_data.get("market_type", "")
        signal_id = signal_data.get("signal_id", "")

        try:
            self._news_searches_total += 1
            self._news_search_cache.record_search(entity_id)

            # Step 1: Build query and search DDGS (with cache)
            query = self._build_news_query(entity_name, market_type)
            logger.info(f"[price_impact] News search for '{query}' (signal={signal_id})")

            cached = self._news_search_cache.get_cached_results(query)
            if cached is not None:
                articles = cached
                logger.info(f"[price_impact] Using cached DDGS results for '{query}' ({len(articles)} articles)")
            else:
                from duckduckgo_search import DDGS

                with DDGS() as ddgs:
                    raw_results = list(ddgs.news(
                        query,
                        max_results=self._config.news_search_max_results,
                        timelimit="d",  # 24h minimum granularity
                    ))

                # Step 2: Post-filter by age (enforce stricter window)
                from datetime import datetime, timezone, timedelta
                now = datetime.now(timezone.utc)
                max_age = timedelta(minutes=self._config.news_search_max_age_minutes)

                articles = []
                for r in raw_results:
                    published_str = r.get("date", "")
                    if published_str:
                        try:
                            published = datetime.fromisoformat(published_str.replace("Z", "+00:00"))
                            if (now - published) > max_age:
                                continue
                        except (ValueError, TypeError):
                            pass  # Include articles with unparseable dates

                    articles.append({
                        "title": r.get("title", ""),
                        "source": r.get("source", ""),
                        "snippet": r.get("body", "")[:300],
                        "url": r.get("url", ""),
                        "published_at": published_str,
                    })

                self._news_search_cache.cache_results(query, articles)

            logger.info(
                f"[price_impact] News search found {len(articles)} articles "
                f"within {self._config.news_search_max_age_minutes}min for '{query}'"
            )

            if not articles:
                logger.info(f"[price_impact] No articles found for '{query}', skipping")
                return

            # Step 3: Extract entities from articles using LLMEntityExtractor
            extractor = self._get_or_create_extractor()

            # Combine article titles + snippets for extraction
            combined_title = " | ".join(a["title"] for a in articles[:5] if a.get("title"))
            combined_body = "\n".join(
                f"[{a.get('source', '')}] {a.get('title', '')} — {a.get('snippet', '')}"
                for a in articles[:5]
            )

            entities = await extractor.extract(
                title=combined_title,
                subreddit="",  # Not Reddit
                body=combined_body,
            )

            if not entities:
                logger.info(f"[price_impact] No entities extracted from news for '{query}'")
                return

            # Step 4: Convert LLMExtractedEntity objects to JSONB format
            from ..schemas.entity_schemas import normalize_entity_id
            entities_jsonb = []
            aggregate_sentiment = 0
            for ent in entities:
                entities_jsonb.append({
                    "entity_id": normalize_entity_id(ent.name),
                    "canonical_name": ent.name,
                    "entity_type": ent.entity_type.lower(),
                    "sentiment_score": ent.sentiment,
                    "confidence": ent.confidence_float,
                    "context_snippet": ent.context,
                })
                aggregate_sentiment += ent.sentiment
            if entities_jsonb:
                aggregate_sentiment = aggregate_sentiment // len(entities_jsonb)

            # Step 5: INSERT into news_entities (upsert on article_url for dedup)
            if not self._supabase:
                logger.warning("[price_impact] No Supabase client for news entity insert")
                return

            # Use first article's URL as the dedup key
            primary_url = articles[0].get("url", f"news_{entity_id}_{int(time.time())}")
            primary_headline = articles[0].get("title", combined_title[:200])
            primary_source = articles[0].get("source", "")
            primary_published = articles[0].get("published_at", "")

            # Parse source_domain from URL
            source_domain = "news"
            try:
                from urllib.parse import urlparse
                parsed = urlparse(primary_url)
                if parsed.netloc:
                    source_domain = parsed.netloc.replace("www.", "")
            except Exception:
                pass

            insert_data = {
                "article_url": primary_url,
                "headline": primary_headline,
                "publisher": primary_source,
                "source_domain": source_domain,
                "entities": entities_jsonb,
                "aggregate_sentiment": aggregate_sentiment,
                "search_query": query,
                "triggered_by_entity": entity_id,
                "triggered_by_signal": signal_id,
                "content_type": "news",
                "extraction_source": "llm_extraction",
                "extraction_success": True,
            }

            # Parse published_at if available
            if primary_published:
                try:
                    insert_data["published_at"] = primary_published
                except Exception:
                    pass

            try:
                await self._supabase.table("news_entities") \
                    .upsert(insert_data, on_conflict="article_url") \
                    .execute()
                logger.info(
                    f"[price_impact] Inserted news entity: {primary_headline[:60]}... "
                    f"({len(entities_jsonb)} entities, triggered by {entity_name})"
                )
            except Exception as e:
                logger.error(f"[price_impact] news_entities upsert failed: {e}")
                self._news_searches_failed += 1
                return

            self._news_searches_enriched += 1

        except ImportError:
            logger.warning("[price_impact] duckduckgo_search not installed, news search disabled")
            self._news_searches_failed += 1
        except Exception as e:
            logger.error(f"[price_impact] News search/enrich error for {signal_id}: {e}")
            self._news_searches_failed += 1

    async def _insert_price_impact(
        self,
        signal_data: Dict[str, Any],
        reddit_entity_id: Optional[Any] = None,
    ) -> bool:
        """
        Insert a price impact signal into the market_price_impacts table.

        Args:
            signal_data: The signal data dict with all fields
            reddit_entity_id: Optional UUID reference to reddit_entities row

        Returns:
            True if insert succeeded, False otherwise
        """
        if not self._supabase:
            logger.debug("[price_impact] No Supabase client, skipping insert")
            return False

        try:
            from datetime import datetime, timezone

            # Build insert payload matching table schema
            insert_data = {
                "source_post_id": signal_data.get("source_post_id", ""),
                "source_subreddit": signal_data.get("source_subreddit", ""),
                "entity_id": signal_data.get("entity_id", ""),
                "entity_name": signal_data.get("entity_name", ""),
                "market_ticker": signal_data.get("market_ticker", ""),
                "event_ticker": signal_data.get("event_ticker", ""),
                "market_type": signal_data.get("market_type", ""),
                "sentiment_score": int(signal_data.get("sentiment_score", 0)),
                "price_impact_score": int(signal_data.get("price_impact_score", 0)),
                "confidence": float(signal_data.get("confidence", 0.5)),
                "transformation_logic": signal_data.get("transformation_logic", ""),
                "source_title": signal_data.get("source_title", ""),
                "context_snippet": signal_data.get("context_snippet", ""),
                "content_type": signal_data.get("content_type", ""),
                "source_domain": signal_data.get("source_domain", ""),
            }

            # Add source_created_at if available (convert Unix timestamp to ISO)
            source_created_at = signal_data.get("source_created_at")
            if source_created_at is not None:
                try:
                    insert_data["source_created_at"] = datetime.fromtimestamp(
                        float(source_created_at), tz=timezone.utc
                    ).isoformat()
                except (ValueError, TypeError, OSError):
                    pass  # Skip if timestamp is invalid

            # Add reddit_entity_id reference if available
            if reddit_entity_id:
                insert_data["reddit_entity_id"] = str(reddit_entity_id)

            # Insert into table (using async client)
            try:
                result = await self._supabase.table("market_price_impacts").insert(insert_data).execute()
            except Exception as schema_error:
                # Handle missing columns gracefully (migration not yet applied)
                error_str = str(schema_error)
                optional_columns = ["context_snippet", "source_title", "source_created_at", "content_type", "source_domain"]
                if any(col in error_str for col in optional_columns):
                    logger.warning(
                        "[price_impact] New columns not in schema, retrying without optional columns"
                    )
                    # Remove optional columns and retry
                    for col in optional_columns:
                        insert_data.pop(col, None)
                    result = await self._supabase.table("market_price_impacts").insert(insert_data).execute()
                else:
                    raise

            if result.data:
                logger.info(
                    f"[price_impact] Inserted signal {signal_data.get('signal_id', 'unknown')}"
                )
                return True
            else:
                logger.warning(
                    f"[price_impact] Insert returned no data for {signal_data.get('signal_id')}"
                )
                return False

        except Exception as e:
            logger.error(f"[price_impact] Insert error: {e}")
            self._record_error(f"insert_error: {e}")
            return False
