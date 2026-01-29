"""
Price Impact Agent - Transforms entity sentiment into market price impact.

Subscribes to Supabase Realtime INSERT events on the reddit_entities table,
transforming raw entity sentiment into market-specific price impact signals.

Key transformation rules:
- OUT markets: Invert sentiment (scandal = more likely OUT)
- WIN/CONFIRM/NOMINEE: Preserve sentiment direction

The agent:
1. Listens for new reddit_entities via Supabase Realtime
2. Looks up entity → market mappings from EntityMarketIndex
3. Applies transformation rules based on market type
4. Inserts PriceImpactSignal into market_price_impacts table
5. Broadcasts signals to frontend via WebSocket
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

    enabled: bool = True


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
        """Subscribe to reddit_entities table INSERT events.

        Uses the SDK's subscribe(callback=...) form to track subscription
        state transitions (SUBSCRIBED, TIMED_OUT, CHANNEL_ERROR). The SDK's
        built-in rejoin_timer handles transient failures automatically.
        """
        if not self._supabase:
            logger.warning("[price_impact] No Supabase client for subscription")
            return

        try:
            from realtime.types import RealtimeSubscribeStates

            self._channel = self._supabase.channel("reddit-entities-price-impact")

            channel_with_listener = self._channel.on_postgres_changes(
                event="INSERT",
                schema="public",
                table="reddit_entities",
                callback=self._handle_entity_insert,
            )

            def on_subscribe_state(state: RealtimeSubscribeStates, error=None):
                """Track subscription state - SDK calls this on state changes."""
                self._subscription_state = state.value if hasattr(state, "value") else str(state)
                if state == RealtimeSubscribeStates.SUBSCRIBED:
                    self._subscription_healthy = True
                    self._last_healthy_time = time.time()
                    self._reconnect_attempts = 0
                    logger.info("[price_impact] Subscribed to reddit_entities Realtime")
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

            await channel_with_listener.subscribe(callback=on_subscribe_state)

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
            post_created_utc = record.get("post_created_utc")  # Original Reddit timestamp
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
                    # Compute price impact based on market type
                    price_impact = compute_price_impact(sentiment_score, mapping.market_type)

                    # Determine suggested side
                    suggested_side = "YES" if price_impact > 0 else "NO"

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
                        "transformation_logic": IMPACT_RULES.get(mapping.market_type, {}).get("description", ""),
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

                    impacts_created += 1
                    self._impacts_created += 1

            if impacts_created > 0:
                logger.info(
                    f"[price_impact] Created {impacts_created} impacts from {post_id}"
                )

        except Exception as e:
            logger.error(f"[price_impact] Process error: {e}")
            self._record_error(str(e))

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
