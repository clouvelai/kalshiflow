"""
Reddit Entity Agent - Streams Reddit posts and extracts structured signals.

Uses PRAW for Reddit streaming and KalshiExtractor (langextract) for
structured extraction. One langextract call per post produces multiple
extraction classes (market_signal, entity_mention, context_factor, plus
per-event custom classes from understand_event).

Inserts extractions into Supabase `extractions` table. The Extraction
Signal Relay picks these up via Realtime and broadcasts to frontend.
"""

from __future__ import annotations

import asyncio
import logging
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .base_agent import BaseAgent

# Suppress PRAW async environment warning.
# PRAW detects an active event loop and emits a warning, but our blocking calls
# are already wrapped in asyncio.to_thread(). The remaining attribute accesses
# (title, url, etc.) are lightweight and safe on the async thread.
warnings.filterwarnings(
    "ignore",
    message=".*It appears that you are using PRAW in an asynchronous environment.*",
)

if TYPE_CHECKING:
    from ..core.websocket_manager import V3WebSocketManager
    from ..core.event_bus import EventBus
    from ..nlp.kalshi_extractor import KalshiExtractor, EventConfig, ExtractionRow

logger = logging.getLogger("kalshiflow_rl.traderv3.agents.reddit_entity_agent")


@dataclass
class RedditEntityAgentConfig:
    """Configuration for the Reddit Entity Agent."""

    # Subreddits to monitor
    subreddits: List[str] = field(default_factory=lambda: ["politics", "news"])

    # PRAW configuration
    skip_existing: bool = False  # False = get historical 100 items first
    post_limit: int = 100  # Max posts to keep in memory

    # Processing
    min_title_length: int = 20

    # Content extraction (video transcription, article extraction)
    content_extraction_enabled: bool = True
    video_transcription_enabled: bool = True
    article_extraction_enabled: bool = True
    video_max_duration_seconds: int = 300  # 5 minutes max
    video_daily_budget_minutes: float = 60.0
    article_max_chars: int = 10_000

    # Reddit metadata gating (skip low-engagement posts before LLM)
    reddit_min_score: int = 5
    reddit_min_comments: int = 5

    # Event config refresh interval (seconds)
    event_config_refresh_interval: float = 300.0  # 5 minutes

    enabled: bool = True


class RedditEntityAgent(BaseAgent):
    """
    Agent that streams Reddit posts and extracts structured signals.

    Pipeline:
    1. PRAW streams posts from configured subreddits
    2. Content extraction (video transcription, article text)
    3. Load active event configs from Supabase
    4. Single KalshiExtractor call per post (base + event specs merged)
    5. Parse langextract output → ExtractionRow objects
    6. Insert to Supabase `extractions` table (Realtime triggers downstream)
    7. Broadcast post metadata to frontend via WebSocket
    """

    def __init__(
        self,
        config: Optional[RedditEntityAgentConfig] = None,
        websocket_manager: Optional["V3WebSocketManager"] = None,
        event_bus: Optional["EventBus"] = None,
    ):
        super().__init__(
            name="reddit_entity",
            display_name="Reddit Entity Agent",
            event_bus=event_bus,
            websocket_manager=websocket_manager,
        )

        self._config = config or RedditEntityAgentConfig()

        # PRAW components
        self._reddit = None
        self._stream_task: Optional[asyncio.Task] = None

        # KalshiExtractor (replaces spaCy + LLM pipeline)
        self._extractor: Optional["KalshiExtractor"] = None

        # Supabase client for persistence
        self._supabase = None

        # Active event configs (refreshed periodically)
        self._event_configs: List["EventConfig"] = []
        self._event_configs_last_refresh: float = 0.0

        # Content extraction tools
        self._content_extractor = None
        self._video_transcriber = None

        # Duplicate prevention
        self._seen_post_ids: set = set()

        # History buffers for frontend snapshots
        from collections import deque
        self._recent_posts: deque = deque(maxlen=30)
        self._recent_extractions: deque = deque(maxlen=50)

        # Stats
        self._posts_processed = 0
        self._posts_skipped = 0
        self._posts_skipped_duplicate = 0
        self._posts_skipped_low_engagement = 0
        self._posts_inserted = 0
        self._extractions_total = 0
        self._extractions_market_signals = 0
        self._content_extractions = 0

        # Startup health tracking
        self._init_results: Dict[str, bool] = {
            "praw": False,
            "extractor": False,
            "supabase": False,
        }
        self._startup_health: str = "initializing"

        # Health broadcast task
        self._health_broadcast_task: Optional[asyncio.Task] = None

    async def _on_start(self) -> None:
        """Initialize resources on agent start."""
        if not self._config.enabled:
            logger.info("[reddit_entity] Agent disabled")
            return

        # Initialize PRAW
        self._init_results["praw"] = await self._init_praw()
        if not self._init_results["praw"]:
            logger.warning("[reddit_entity] PRAW not available, running in mock mode")

        # Initialize KalshiExtractor
        self._init_results["extractor"] = await self._init_extractor()
        if not self._init_results["extractor"]:
            logger.warning("[reddit_entity] KalshiExtractor not available, extraction disabled")

        # Initialize Supabase
        self._init_results["supabase"] = await self._init_supabase()
        if not self._init_results["supabase"]:
            logger.warning("[reddit_entity] Supabase not available, extractions won't persist")
        else:
            await self._load_seen_posts_from_db()
            await self._refresh_event_configs()

        # Compute startup health
        self._startup_health = self._compute_startup_health()
        logger.info(
            f"[reddit_entity] Startup health: {self._startup_health} "
            f"(praw={self._init_results['praw']}, extractor={self._init_results['extractor']}, "
            f"supabase={self._init_results['supabase']})"
        )

        # Initialize content extraction tools
        if self._config.content_extraction_enabled:
            await self._init_content_extractor()

        # Start streaming task
        self._stream_task = asyncio.create_task(self._stream_loop())

        # Start health broadcast task
        self._health_broadcast_task = asyncio.create_task(self._health_broadcast_loop())

        content_status = "enabled" if self._content_extractor else "disabled"
        logger.info(
            f"[reddit_entity] Started (langextract pipeline, content extraction {content_status}): "
            f"r/{' + r/'.join(self._config.subreddits)}"
        )

    def _compute_startup_health(self) -> str:
        """Compute startup health status."""
        praw_ok = self._init_results.get("praw", False)
        extractor_ok = self._init_results.get("extractor", False)
        supabase_ok = self._init_results.get("supabase", False)

        if praw_ok and extractor_ok and supabase_ok:
            return "healthy"
        if not praw_ok:
            return "unhealthy"
        return "degraded"

    async def _on_stop(self) -> None:
        """Cleanup resources on agent stop."""
        if self._stream_task and not self._stream_task.done():
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass

        if self._health_broadcast_task and not self._health_broadcast_task.done():
            self._health_broadcast_task.cancel()
            try:
                await self._health_broadcast_task
            except asyncio.CancelledError:
                pass

        if self._content_extractor:
            await self._content_extractor.close()

        logger.info(
            f"[reddit_entity] Stopped. Processed: {self._posts_processed}, "
            f"Extractions: {self._extractions_total} ({self._extractions_market_signals} signals), "
            f"Content: {self._content_extractions}, Skipped: {self._posts_skipped}"
        )

    def _get_agent_stats(self) -> Dict[str, Any]:
        """Get agent-specific statistics."""
        return {
            "posts_processed": self._posts_processed,
            "extractions_total": self._extractions_total,
            "extractions_market_signals": self._extractions_market_signals,
            "content_extractions": self._content_extractions,
            "posts_skipped": self._posts_skipped,
            "posts_skipped_duplicate": self._posts_skipped_duplicate,
            "posts_skipped_low_engagement": self._posts_skipped_low_engagement,
            "seen_posts_cached": len(self._seen_post_ids),
            "posts_inserted": self._posts_inserted,
            "subreddits": self._config.subreddits,
            "praw_available": self._reddit is not None,
            "extractor_available": self._extractor is not None,
            "supabase_available": self._supabase is not None,
            "content_extraction_enabled": self._content_extractor is not None,
            "event_configs_loaded": len(self._event_configs),
            "init_results": self._init_results.copy(),
            "startup_health": self._startup_health,
        }

    # === Initialization ===

    async def _init_praw(self) -> bool:
        """Initialize PRAW Reddit client."""
        try:
            import os
            import praw

            client_id = os.getenv("REDDIT_CLIENT_ID")
            client_secret = os.getenv("REDDIT_CLIENT_SECRET")
            user_agent = os.getenv("REDDIT_USER_AGENT", "kalshiflow:v1.0 (by /u/kalshiflow)")

            if not client_id or not client_secret:
                logger.warning("[reddit_entity] REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET required")
                return False

            self._reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent,
            )
            logger.info("[reddit_entity] PRAW initialized")
            return True

        except ImportError:
            logger.error("[reddit_entity] praw package not installed")
            return False
        except Exception as e:
            logger.error(f"[reddit_entity] PRAW init error: {e}")
            return False

    async def _init_extractor(self) -> bool:
        """Initialize KalshiExtractor."""
        try:
            from ..nlp.kalshi_extractor import KalshiExtractor

            self._extractor = KalshiExtractor()
            logger.info("[reddit_entity] KalshiExtractor initialized")
            return True

        except ImportError as e:
            logger.error(f"[reddit_entity] KalshiExtractor import failed: {e}")
            return False
        except Exception as e:
            logger.error(f"[reddit_entity] KalshiExtractor init error: {e}")
            return False

    async def _init_supabase(self) -> bool:
        """Initialize Supabase async client."""
        try:
            import os
            from supabase import acreate_client, AsyncClient

            url = os.getenv("SUPABASE_URL")
            key = os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_ANON_KEY")

            if not url or not key:
                logger.warning("[reddit_entity] SUPABASE_URL and SUPABASE_KEY required")
                return False

            self._supabase: AsyncClient = await acreate_client(url, key)
            logger.info("[reddit_entity] Supabase async client initialized")
            return True

        except ImportError:
            logger.error("[reddit_entity] supabase package not installed")
            return False
        except Exception as e:
            logger.error(f"[reddit_entity] Supabase init error: {e}")
            return False

    async def _init_content_extractor(self) -> bool:
        """Initialize content extraction tools."""
        try:
            from ..tools import (
                ContentExtractor,
                ContentExtractorConfig,
                VideoTranscriber,
                VideoTranscriberConfig,
            )

            if self._config.video_transcription_enabled:
                video_config = VideoTranscriberConfig(
                    max_duration_seconds=self._config.video_max_duration_seconds,
                    daily_budget_minutes=self._config.video_daily_budget_minutes,
                )
                self._video_transcriber = VideoTranscriber(config=video_config)
                logger.info("[reddit_entity] Video transcriber initialized")

            content_config = ContentExtractorConfig(
                article_extraction_enabled=self._config.article_extraction_enabled,
                video_transcription_enabled=self._config.video_transcription_enabled,
                max_output_chars=self._config.article_max_chars,
            )
            self._content_extractor = ContentExtractor(
                config=content_config,
                video_transcriber=self._video_transcriber,
            )
            logger.info("[reddit_entity] Content extractor initialized")
            return True

        except ImportError as e:
            logger.warning(f"[reddit_entity] Content extraction dependencies not available: {e}")
            return False
        except Exception as e:
            logger.error(f"[reddit_entity] Content extractor init failed: {e}")
            return False

    # === Event Config Management ===

    async def _refresh_event_configs(self) -> None:
        """Load active event configs from Supabase."""
        if not self._supabase:
            return

        try:
            from ..nlp.kalshi_extractor import EventConfig

            result = await self._supabase.table("event_configs") \
                .select("*") \
                .eq("is_active", True) \
                .execute()

            configs = []
            for row in result.data or []:
                configs.append(EventConfig(
                    event_ticker=row.get("event_ticker", ""),
                    event_title=row.get("event_title", ""),
                    primary_entity=row.get("primary_entity"),
                    primary_entity_type=row.get("primary_entity_type"),
                    description=row.get("description"),
                    key_drivers=row.get("key_drivers", []),
                    outcome_descriptions=row.get("outcome_descriptions", {}),
                    prompt_description=row.get("prompt_description"),
                    extraction_classes=row.get("extraction_classes", []),
                    examples=row.get("examples", []),
                    watchlist=row.get("watchlist", {}),
                    markets=row.get("markets", []),
                    is_active=row.get("is_active", True),
                    research_version=row.get("research_version", 1),
                ))

            self._event_configs = configs
            self._event_configs_last_refresh = time.time()
            logger.info(f"[reddit_entity] Loaded {len(configs)} active event configs")

        except Exception as e:
            logger.error(f"[reddit_entity] Event config refresh failed: {e}")

    async def _get_event_configs(self) -> List["EventConfig"]:
        """Get event configs, refreshing if stale."""
        elapsed = time.time() - self._event_configs_last_refresh
        if elapsed > self._config.event_config_refresh_interval:
            await self._refresh_event_configs()
        return self._event_configs

    # === Data Loading ===

    async def _load_seen_posts_from_db(self) -> None:
        """Pre-populate seen post IDs from database."""
        if not self._supabase:
            return

        try:
            # Load from extractions table (new system)
            result = await self._supabase.table("extractions").select(
                "source_id"
            ).eq(
                "source_type", "reddit_post"
            ).order(
                "created_at", desc=True
            ).limit(2000).execute()

            if result.data:
                self._seen_post_ids = {row["source_id"] for row in result.data if row.get("source_id")}
                logger.info(
                    f"[reddit_entity] Pre-loaded {len(self._seen_post_ids)} seen post IDs from DB"
                )
        except Exception as e:
            logger.warning(f"[reddit_entity] Could not pre-load seen posts: {e}")

    # === Streaming ===

    async def _stream_loop(self) -> None:
        """Main streaming loop for Reddit posts."""
        while self._running:
            try:
                if self._reddit:
                    await self._stream_reddit()
                else:
                    await asyncio.sleep(60.0)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[reddit_entity] Stream error: {e}")
                self._record_error(str(e))
                await asyncio.sleep(30.0)

    async def _stream_reddit(self) -> None:
        """Stream posts from Reddit using PRAW."""
        subreddit_str = "+".join(self._config.subreddits)
        subreddit = self._reddit.subreddit(subreddit_str)

        def get_next_submission(stream_iter):
            try:
                return next(stream_iter)
            except StopIteration:
                return None

        stream_iter = subreddit.stream.submissions(
            skip_existing=self._config.skip_existing
        )

        while self._running:
            submission = await asyncio.to_thread(get_next_submission, stream_iter)
            if submission is None or not self._running:
                break
            await self._process_submission(submission)

    # === Post Processing ===

    async def _process_submission(self, submission) -> None:
        """Process a single Reddit submission through KalshiExtractor."""
        try:
            title = submission.title
            post_id = submission.id
            selftext = getattr(submission, "selftext", "") or ""
            url = submission.url

            # Skip duplicates
            if post_id in self._seen_post_ids:
                self._posts_skipped_duplicate += 1
                return
            self._seen_post_ids.add(post_id)

            # Skip short titles
            if len(title) < self._config.min_title_length:
                self._posts_skipped += 1
                return

            # Engagement gating
            post_score = getattr(submission, "score", 0) or 0
            post_comments = getattr(submission, "num_comments", 0) or 0
            if (
                post_score < self._config.reddit_min_score
                and post_comments < self._config.reddit_min_comments
            ):
                self._posts_skipped_low_engagement += 1
                return

            self._posts_processed += 1
            self._record_event_processed()

            # Build post metadata
            post_data = {
                "post_id": post_id,
                "subreddit": submission.subreddit.display_name,
                "title": title,
                "url": url,
                "author": str(submission.author) if submission.author else "unknown",
                "score": post_score,
                "num_comments": post_comments,
                "created_utc": submission.created_utc,
            }

            # Broadcast post to frontend
            await self._broadcast_to_frontend("reddit_post", post_data)
            self._recent_posts.appendleft(post_data)

            # Build combined text
            combined_text = title
            if selftext:
                combined_text = f"{title}\n\n{selftext}"

            # Content extraction (video, article)
            if self._content_extractor and url:
                extracted = await self._content_extractor.extract(url, selftext)
                if extracted.success and extracted.text:
                    self._content_extractions += 1
                    combined_text = f"{combined_text}\n\n{extracted.text}"
                    logger.info(
                        f"[reddit_entity] Extracted {extracted.content_type} content "
                        f"({len(extracted.text)} chars) from {extracted.source_domain}"
                    )

            # Run KalshiExtractor
            if not self._extractor:
                return

            event_configs = await self._get_event_configs()

            source_created_at = None
            if submission.created_utc:
                source_created_at = datetime.fromtimestamp(
                    submission.created_utc, tz=timezone.utc
                ).isoformat()

            extraction_rows = await self._extractor.extract(
                title=title,
                body=combined_text,
                subreddit=post_data["subreddit"],
                engagement_score=post_score,
                engagement_comments=post_comments,
                event_configs=event_configs,
                source_type="reddit_post",
                source_id=post_id,
                source_url=url,
                source_created_at=source_created_at,
            )

            if not extraction_rows:
                return

            # Insert extractions to Supabase
            inserted = await self._insert_extractions(extraction_rows)

            # Update stats
            self._extractions_total += len(extraction_rows)
            self._extractions_market_signals += sum(
                1 for r in extraction_rows if r.extraction_class == "market_signal"
            )
            if inserted:
                self._posts_inserted += 1

            # Broadcast extraction summary to frontend
            for row in extraction_rows:
                extraction_data = {
                    "source_id": row.source_id,
                    "extraction_class": row.extraction_class,
                    "extraction_text": row.extraction_text[:200],
                    "attributes": row.attributes,
                    "market_tickers": row.market_tickers,
                    "event_tickers": row.event_tickers,
                    "source_subreddit": row.source_subreddit,
                    "engagement_score": row.engagement_score,
                }
                await self._broadcast_to_frontend("extraction", extraction_data)
                self._recent_extractions.appendleft(extraction_data)

            logger.info(
                f"[reddit_entity] {len(extraction_rows)} extractions from post {post_id} "
                f"({sum(1 for r in extraction_rows if r.extraction_class == 'market_signal')} signals)"
            )

        except Exception as e:
            logger.error(f"[reddit_entity] Process error: {e}")
            self._record_error(str(e))

    # === Persistence ===

    async def _insert_extractions(self, rows: List["ExtractionRow"]) -> bool:
        """Insert extraction rows into Supabase extractions table.

        Uses individual inserts with duplicate detection. The unique index
        idx_extractions_dedup ON (source_id, extraction_class, md5(extraction_text))
        prevents duplicate extractions when the same source is reprocessed.
        """
        if not self._supabase or not rows:
            return False

        inserted_count = 0
        duplicate_count = 0

        for row in rows:
            try:
                data = {
                    "source_type": row.source_type,
                    "source_id": row.source_id,
                    "source_url": row.source_url,
                    "source_subreddit": row.source_subreddit,
                    "extraction_class": row.extraction_class,
                    "extraction_text": row.extraction_text,
                    "attributes": row.attributes,
                    "market_tickers": row.market_tickers,
                    "event_tickers": row.event_tickers,
                    "engagement_score": row.engagement_score,
                    "engagement_comments": row.engagement_comments,
                    "source_created_at": row.source_created_at,
                }

                result = await self._supabase.table("extractions").insert(data).execute()

                if result.data:
                    inserted_count += 1

            except Exception as e:
                error_str = str(e)
                if "duplicate" in error_str.lower() or "unique" in error_str.lower() or "23505" in error_str:
                    duplicate_count += 1
                    logger.debug(
                        f"[reddit_entity] Duplicate extraction skipped: "
                        f"{row.extraction_class} from {row.source_id}"
                    )
                else:
                    logger.error(f"[reddit_entity] Extraction insert error: {e}")

        if duplicate_count > 0:
            logger.info(
                f"[reddit_entity] Inserted {inserted_count}, skipped {duplicate_count} duplicates "
                f"for source {rows[0].source_id}"
            )
        elif inserted_count > 0:
            logger.debug(
                f"[reddit_entity] Inserted {inserted_count} extractions "
                f"for source {rows[0].source_id}"
            )

        return inserted_count > 0

    # === Health ===

    async def _health_broadcast_loop(self) -> None:
        """Periodically broadcast health status to frontend."""
        while self._running:
            try:
                health_status = {
                    "agent_name": "reddit_entity",
                    "is_running": self._running,
                    "praw_available": self._reddit is not None,
                    "extractor_available": self._extractor is not None,
                    "supabase_available": self._supabase is not None,
                    "subreddits": self._config.subreddits,
                    "posts_processed": self._posts_processed,
                    "extractions_total": self._extractions_total,
                    "extractions_market_signals": self._extractions_market_signals,
                    "posts_skipped": self._posts_skipped,
                    "last_error": self._stats.last_error,
                    "errors_count": self._stats.errors_count,
                    "event_configs_loaded": len(self._event_configs),
                    "content_extractions": self._content_extractions,
                    "content_extractor_stats": (
                        self._content_extractor.get_stats() if self._content_extractor else {}
                    ),
                    "init_results": self._init_results.copy(),
                    "startup_health": self._startup_health,
                    "health": self._startup_health,
                }

                await self._broadcast_to_frontend("reddit_agent_health", health_status)
                await asyncio.sleep(5.0)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"[reddit_entity] Health broadcast error: {e}")
                await asyncio.sleep(5.0)

    # === Snapshot ===

    def get_entity_snapshot(self) -> Dict[str, Any]:
        """Get pipeline snapshot for new client initialization."""
        return {
            "reddit_posts": list(self._recent_posts),
            "extractions": list(self._recent_extractions),
            "stats": {
                "postsProcessed": self._posts_processed,
                "extractionsTotal": self._extractions_total,
                "extractionsMarketSignals": self._extractions_market_signals,
            },
            "is_active": self._running,
        }

    def get_event_configs(self) -> List[Dict[str, Any]]:
        """Get cached event configs for frontend display.

        Returns serializable list of active event configurations including
        event_ticker, event_title, primary_entity, markets list, etc.
        """
        configs = []
        for cfg in self._event_configs:
            configs.append({
                "event_ticker": cfg.event_ticker,
                "event_title": cfg.event_title,
                "primary_entity": getattr(cfg, "primary_entity", None),
                "primary_entity_type": getattr(cfg, "primary_entity_type", None),
                "markets": getattr(cfg, "markets", []),
                "is_active": getattr(cfg, "is_active", True),
                "research_version": getattr(cfg, "research_version", 1),
            })
        return configs
