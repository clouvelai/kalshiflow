"""
Extraction Signal Relay - Subscribes to extractions table and broadcasts signals.

Thin relay agent that:
1. Subscribes to `extractions` table via Supabase Realtime (INSERT events)
2. Broadcasts all extractions to frontend via WebSocket
3. No LLM calls, no entity lookup, no news search

The extraction pipeline (KalshiExtractor) already handles:
- Market linking (via merged prompt with ACTIVE KALSHI MARKETS)
- Direction/magnitude assessment (extraction attributes)
- Multi-class extraction (market_signal, entity_mention, context_factor, custom)

Data flow:
  Reddit/News → KalshiExtractor → extractions table → Realtime → THIS AGENT → WebSocket broadcast
  Deep agent queries extractions directly via get_extraction_signals() tool.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .base_agent import BaseAgent

if TYPE_CHECKING:
    from ..core.websocket_manager import V3WebSocketManager
    from ..core.event_bus import EventBus

logger = logging.getLogger("kalshiflow_rl.traderv3.agents.price_impact_agent")


@dataclass
class ExtractionRelayConfig:
    """Configuration for the Extraction Signal Relay."""

    # Supabase connection
    supabase_url: Optional[str] = None
    supabase_key: Optional[str] = None

    # Reconnection (exponential backoff for full-recreation)
    reconnect_backoff_base: float = 5.0
    reconnect_backoff_max: float = 300.0  # 5 minutes max backoff

    # Timeouts for Supabase Realtime operations
    subscribe_timeout: float = 30.0
    unsubscribe_timeout: float = 10.0

    # SDK configuration
    realtime_timeout: int = 30
    realtime_max_retries: int = 20

    # Nuclear reconnect threshold
    nuclear_reconnect_threshold: float = 600.0  # 10 minutes

    # Engagement refresh
    engagement_refresh_enabled: bool = True
    engagement_refresh_interval_seconds: float = 300.0  # 5 minutes
    engagement_refresh_window_hours: float = 4.0

    enabled: bool = True


# Keep old name as alias for backward compatibility with coordinator imports
PriceImpactAgentConfig = ExtractionRelayConfig


class PriceImpactAgent(BaseAgent):
    """
    Thin relay that subscribes to extractions Realtime and broadcasts to frontend.

    The extraction pipeline handles all intelligence (market linking, direction,
    magnitude). This agent just moves data from Supabase → WebSocket.
    """

    def __init__(
        self,
        config: Optional[ExtractionRelayConfig] = None,
        websocket_manager: Optional["V3WebSocketManager"] = None,
        event_bus: Optional["EventBus"] = None,
        entity_index: Optional[Any] = None,  # Ignored, kept for backward compat
    ):
        super().__init__(
            name="extraction_relay",
            display_name="Extraction Signal Relay",
            event_bus=event_bus,
            websocket_manager=websocket_manager,
        )

        self._config = config or ExtractionRelayConfig()

        # Supabase Realtime
        self._supabase = None
        self._channel = None
        self._subscription_task: Optional[asyncio.Task] = None
        self._engagement_refresh_task: Optional[asyncio.Task] = None

        # Stats
        self._extractions_received = 0
        self._market_signals_received = 0
        self._entity_mentions_received = 0
        self._context_factors_received = 0
        self._custom_extractions_received = 0
        self._broadcasts_sent = 0
        self._reconnect_attempts = 0

        # Health monitoring
        self._last_signal_time: Optional[float] = None
        self._subscription_healthy: bool = False
        self._subscription_state: str = "disconnected"
        self._last_healthy_time: Optional[float] = None

    async def _on_start(self) -> None:
        """Initialize resources on agent start."""
        if not self._config.enabled:
            logger.info("[extraction_relay] Agent disabled")
            return

        # Initialize Supabase client
        if not await self._init_supabase():
            logger.warning("[extraction_relay] Supabase not available")
            return

        # Initialize signal time so health check works from startup
        self._last_signal_time = time.time()

        # Start Realtime subscription
        self._subscription_task = asyncio.create_task(self._subscription_loop())

        # Start engagement refresh background task
        if self._config.engagement_refresh_enabled:
            self._engagement_refresh_task = asyncio.create_task(
                self._engagement_refresh_loop()
            )

        logger.info("[extraction_relay] Started listening for extractions via Supabase Realtime")

    async def _on_stop(self) -> None:
        """Cleanup resources on agent stop."""
        if self._channel:
            try:
                await asyncio.wait_for(
                    self._channel.unsubscribe(),
                    timeout=self._config.unsubscribe_timeout,
                )
            except asyncio.TimeoutError:
                logger.warning("[extraction_relay] Unsubscribe timed out during shutdown")
            except Exception as e:
                logger.error(f"[extraction_relay] Unsubscribe error: {e}")

        for task in (self._subscription_task, self._engagement_refresh_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        logger.info(
            f"[extraction_relay] Stopped. "
            f"Received: {self._extractions_received} extractions "
            f"({self._market_signals_received} signals, "
            f"{self._entity_mentions_received} entities, "
            f"{self._context_factors_received} context)"
        )

    def _get_agent_stats(self) -> Dict[str, Any]:
        """Get agent-specific statistics."""
        seconds_since_signal = None
        if self._last_signal_time is not None:
            seconds_since_signal = time.time() - self._last_signal_time

        return {
            "extractions_received": self._extractions_received,
            "market_signals_received": self._market_signals_received,
            "entity_mentions_received": self._entity_mentions_received,
            "context_factors_received": self._context_factors_received,
            "custom_extractions_received": self._custom_extractions_received,
            "broadcasts_sent": self._broadcasts_sent,
            "reconnect_attempts": self._reconnect_attempts,
            "supabase_connected": self._channel is not None,
            "subscription_healthy": self._subscription_healthy,
            "subscription_state": self._subscription_state,
            "seconds_since_signal": seconds_since_signal,
        }

    # =========================================================================
    # Supabase Realtime Subscription
    # =========================================================================

    async def _init_supabase(self) -> bool:
        """Initialize Supabase async client for Realtime."""
        try:
            import os
            from supabase import acreate_client, AsyncClient

            url = self._config.supabase_url or os.getenv("SUPABASE_URL")
            key = self._config.supabase_key or os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_ANON_KEY")

            if not url or not key:
                logger.warning("[extraction_relay] SUPABASE_URL and SUPABASE_KEY required")
                return False

            self._supabase: AsyncClient = await acreate_client(url, key)
            self._supabase.realtime.timeout = self._config.realtime_timeout
            self._supabase.realtime.max_retries = self._config.realtime_max_retries
            logger.info(
                f"[extraction_relay] Supabase client initialized "
                f"(timeout={self._config.realtime_timeout}s)"
            )
            return True

        except ImportError:
            logger.error("[extraction_relay] supabase package not installed")
            return False
        except Exception as e:
            logger.error(f"[extraction_relay] Supabase init error: {e}")
            return False

    async def _subscription_loop(self) -> None:
        """Main loop: subscribe to extractions table Realtime."""
        while self._running:
            try:
                await self._subscribe_to_extractions()

                while self._running and self._channel:
                    await asyncio.sleep(60.0)

                    if self._subscription_healthy:
                        self._last_healthy_time = time.time()
                        continue

                    reference_time = (
                        self._last_healthy_time
                        or self._last_signal_time
                        or time.time()
                    )
                    unhealthy_duration = time.time() - reference_time

                    if unhealthy_duration > self._config.nuclear_reconnect_threshold:
                        logger.warning(
                            f"[extraction_relay] Unhealthy for {unhealthy_duration:.0f}s, "
                            f"recreating client"
                        )
                        await self._nuclear_reconnect()
                        break

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[extraction_relay] Subscription error: {e}")
                self._record_error(str(e))
                self._reconnect_attempts += 1
                backoff = min(
                    self._config.reconnect_backoff_base * (2 ** self._reconnect_attempts),
                    self._config.reconnect_backoff_max,
                )
                await asyncio.sleep(backoff)

    async def _nuclear_reconnect(self) -> None:
        """Destroy and recreate Supabase client."""
        self._reconnect_attempts += 1

        if self._channel:
            try:
                await asyncio.wait_for(
                    self._channel.unsubscribe(),
                    timeout=self._config.unsubscribe_timeout,
                )
            except (asyncio.TimeoutError, Exception):
                pass
            self._channel = None

        if self._supabase and hasattr(self._supabase, "realtime"):
            try:
                await self._supabase.realtime.close()
            except Exception:
                pass

        self._supabase = None
        self._subscription_healthy = False
        self._subscription_state = "disconnected"

        if not await self._init_supabase():
            logger.error("[extraction_relay] Failed to recreate client")
            return

        self._last_healthy_time = time.time()

    async def _subscribe_to_extractions(self) -> None:
        """Subscribe to extractions table INSERT events."""
        if not self._supabase:
            return

        try:
            from realtime.types import RealtimeSubscribeStates

            self._channel = self._supabase.channel("extractions-relay")

            self._channel.on_postgres_changes(
                event="INSERT",
                schema="public",
                table="extractions",
                callback=self._handle_extraction_insert,
            )

            def on_subscribe_state(state: RealtimeSubscribeStates, error=None):
                self._subscription_state = state.value if hasattr(state, "value") else str(state)
                if state == RealtimeSubscribeStates.SUBSCRIBED:
                    self._subscription_healthy = True
                    self._last_healthy_time = time.time()
                    self._reconnect_attempts = 0
                    logger.info("[extraction_relay] Subscribed to extractions Realtime")
                elif state == RealtimeSubscribeStates.TIMED_OUT:
                    self._subscription_healthy = False
                    logger.warning("[extraction_relay] Subscribe timed out")
                elif state == RealtimeSubscribeStates.CHANNEL_ERROR:
                    self._subscription_healthy = False
                    logger.error(f"[extraction_relay] Channel error: {error}")
                elif state == RealtimeSubscribeStates.CLOSED:
                    self._subscription_healthy = False

            await self._channel.subscribe(callback=on_subscribe_state)

        except Exception as e:
            logger.error(f"[extraction_relay] Subscription setup error: {e}")
            self._channel = None
            raise

    # =========================================================================
    # Extraction Processing
    # =========================================================================

    def _handle_extraction_insert(self, payload: Dict[str, Any]) -> None:
        """Handle INSERT event from extractions table."""
        try:
            record = payload.get("data", {}).get("record", {})
            if not record:
                return
            asyncio.create_task(self._process_extraction(record))
        except Exception as e:
            logger.error(f"[extraction_relay] Handle insert error: {e}")

    async def _process_extraction(self, record: Dict[str, Any]) -> None:
        """Process a new extraction record and broadcast to frontend."""
        try:
            self._extractions_received += 1
            self._record_event_processed()
            self._last_signal_time = time.time()

            extraction_class = record.get("extraction_class", "")
            extraction_text = record.get("extraction_text", "")
            attributes = record.get("attributes", {})
            market_tickers = record.get("market_tickers", [])
            event_tickers = record.get("event_tickers", [])
            source_type = record.get("source_type", "")
            source_id = record.get("source_id", "")

            # Track by class
            if extraction_class == "market_signal":
                self._market_signals_received += 1
            elif extraction_class == "entity_mention":
                self._entity_mentions_received += 1
            elif extraction_class == "context_factor":
                self._context_factors_received += 1
            else:
                self._custom_extractions_received += 1

            # Build broadcast message
            broadcast_data = {
                "id": record.get("id", ""),
                "extraction_class": extraction_class,
                "extraction_text": extraction_text,
                "attributes": attributes,
                "market_tickers": market_tickers,
                "event_tickers": event_tickers,
                "source_type": source_type,
                "source_id": source_id,
                "source_subreddit": record.get("source_subreddit", ""),
                "engagement_score": record.get("engagement_score", 0),
                "engagement_comments": record.get("engagement_comments", 0),
                "created_at": record.get("created_at", ""),
            }

            # Broadcast to frontend
            if self._ws_manager:
                await self._ws_manager.broadcast_message("extraction", broadcast_data)
                self._broadcasts_sent += 1

                # Also broadcast a typed message for market signals (deep agent frontend)
                if extraction_class == "market_signal" and market_tickers:
                    await self._ws_manager.broadcast_message("market_signal", {
                        "market_ticker": market_tickers[0] if market_tickers else "",
                        "event_tickers": event_tickers,
                        "direction": attributes.get("direction", ""),
                        "magnitude": attributes.get("magnitude", 0),
                        "confidence": attributes.get("confidence", ""),
                        "reasoning": attributes.get("reasoning", ""),
                        "extraction_text": extraction_text,
                        "source_type": source_type,
                        "engagement_score": record.get("engagement_score", 0),
                        "created_at": record.get("created_at", ""),
                    })

            logger.info(
                f"[extraction_relay] {extraction_class}: "
                f"{extraction_text[:80]}... "
                f"markets={market_tickers} engagement={record.get('engagement_score', 0)}"
            )

        except Exception as e:
            logger.error(f"[extraction_relay] Process extraction error: {e}")
            self._record_error(str(e))

    # =========================================================================
    # Engagement Refresh
    # =========================================================================

    async def _engagement_refresh_loop(self) -> None:
        """Background task: re-fetch Reddit engagement scores for recent extractions."""
        while self._running:
            try:
                await asyncio.sleep(self._config.engagement_refresh_interval_seconds)
                await self._refresh_engagement()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[extraction_relay] Engagement refresh error: {e}")
                await asyncio.sleep(60.0)

    async def _refresh_engagement(self) -> None:
        """Re-fetch Reddit scores for recent extractions and update."""
        if not self._supabase:
            return

        try:
            from datetime import datetime, timezone, timedelta

            cutoff = datetime.now(timezone.utc) - timedelta(
                hours=self._config.engagement_refresh_window_hours
            )

            # Get recent Reddit extractions that need refresh
            result = await self._supabase.table("extractions") \
                .select("id, source_id, source_type, engagement_score") \
                .eq("source_type", "reddit_post") \
                .gte("created_at", cutoff.isoformat()) \
                .limit(100) \
                .execute()

            if not result.data:
                return

            # Batch lookup Reddit scores via PRAW
            try:
                import os
                import praw

                client_id = os.getenv("REDDIT_CLIENT_ID")
                client_secret = os.getenv("REDDIT_CLIENT_SECRET")
                if not client_id or not client_secret:
                    return

                reddit = praw.Reddit(
                    client_id=client_id,
                    client_secret=client_secret,
                    user_agent="kalshiflow:v2.0 (engagement refresh)",
                )

                updated = 0
                for row in result.data:
                    source_id = row.get("source_id", "")
                    if not source_id:
                        continue

                    try:
                        submission = await asyncio.to_thread(
                            reddit.submission, id=source_id
                        )
                        new_score = submission.score
                        new_comments = submission.num_comments
                        old_score = row.get("engagement_score", 0)

                        # Only update if score changed significantly (>20% or >50 delta)
                        if abs(new_score - old_score) > max(50, old_score * 0.2):
                            await self._supabase.table("extractions") \
                                .update({
                                    "engagement_score": new_score,
                                    "engagement_comments": new_comments,
                                    "engagement_updated_at": datetime.now(timezone.utc).isoformat(),
                                }) \
                                .eq("id", row["id"]) \
                                .execute()
                            updated += 1

                    except Exception:
                        continue

                if updated > 0:
                    logger.info(f"[extraction_relay] Refreshed engagement for {updated} extractions")

            except ImportError:
                pass  # PRAW not installed

        except Exception as e:
            logger.error(f"[extraction_relay] Engagement refresh error: {e}")
