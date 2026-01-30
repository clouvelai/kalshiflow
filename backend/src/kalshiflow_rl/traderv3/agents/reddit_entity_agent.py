"""
Reddit Entity Agent - Streams Reddit posts and extracts entities with sentiment.

Uses PRAW for Reddit streaming and a spaCy KB-backed pipeline for entity extraction.
Market entities are detected via EntityRuler patterns from the KnowledgeBase.
Sentiment is scored using batched LLM calls.
Inserts processed entities into Supabase reddit_entities table.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .base_agent import BaseAgent

if TYPE_CHECKING:
    from spacy.language import Language
    from ..core.websocket_manager import V3WebSocketManager
    from ..core.event_bus import EventBus
    from ..services.entity_market_index import EntityMarketIndex
    from ..nlp.knowledge_base import KalshiKnowledgeBase

logger = logging.getLogger("kalshiflow_rl.traderv3.agents.reddit_entity_agent")

# Stopwords to filter out common English words that NER may incorrectly extract as entities
ENTITY_STOPWORDS = {
    # Verbs commonly extracted incorrectly
    "gets", "get", "demoted", "says", "said", "told", "announces", "announced",
    "reported", "reports", "claims", "claimed", "wins", "won", "lost", "loses",
    "fired", "hired", "named", "calls", "called", "faces", "facing",
    # Common articles/prepositions
    "the", "a", "an", "in", "on", "at", "to", "for", "of", "with",
    # News buzzwords
    "breaking", "update", "just", "new", "live", "exclusive", "alert",
}


@dataclass
class RedditEntityAgentConfig:
    """Configuration for the Reddit Entity Agent."""

    # Subreddits to monitor
    subreddits: List[str] = field(default_factory=lambda: ["politics", "news"])

    # PRAW configuration
    skip_existing: bool = False  # False = get historical 100 items first
    post_limit: int = 100  # Max posts to keep in memory

    # spaCy configuration (KB-backed pipeline)
    spacy_model: str = "en_core_web_md"  # ~40MB, CPU optimized

    # LLM sentiment configuration
    sentiment_model: str = "gpt-4o-mini"
    sentiment_timeout: float = 10.0
    use_batched_sentiment: bool = True  # Batch sentiment for all entities in one call

    # Related entity extraction
    extract_related_entities: bool = True  # Extract PERSON/ORG/GPE/EVENT

    # Processing
    min_title_length: int = 20
    max_concurrent_llm: int = 3

    # Content extraction (video transcription, article extraction)
    content_extraction_enabled: bool = True
    video_transcription_enabled: bool = True
    article_extraction_enabled: bool = True
    video_max_duration_seconds: int = 300  # 5 minutes max
    video_daily_budget_minutes: float = 60.0  # ~$0.36/day at $0.006/min
    article_max_chars: int = 10_000

    # Market Impact Reasoning (for indirect market effects)
    market_impact_reasoning_enabled: bool = True
    market_impact_model: str = "gpt-4o-mini"
    market_impact_max_markets: int = 50  # Max markets to include in reasoning prompt

    # LLM entity extraction (replaces unreliable spaCy NER for Phase 2)
    llm_entity_extraction_enabled: bool = True
    llm_entity_model: str = "gpt-4o-mini"
    llm_entity_timeout: float = 10.0
    llm_entity_fallback_to_spacy: bool = True  # Use spaCy NER if LLM fails

    # Reddit metadata gating (Phase 4: skip low-engagement posts before LLM)
    reddit_min_score: int = 5  # Minimum post score to process through LLM
    reddit_min_comments: int = 5  # Minimum comment count to process through LLM

    enabled: bool = True


class RedditEntityAgent(BaseAgent):
    """
    Agent that streams Reddit posts and extracts entities with sentiment.

    Pipeline:
    1. PRAW streams posts from configured subreddits
    2. KB-backed spaCy pipeline extracts market entities via EntityRuler
    3. spaCy NER discovers related entities (PERSON, ORG, GPE, EVENT)
    4. Batched LLM scores sentiment for all entities (-100 to +100)
    5. Market entities stored in Supabase reddit_entities table
    6. Related entities stored for second-hand signal analysis
    7. Broadcasts to frontend via WebSocket
    """

    def __init__(
        self,
        config: Optional[RedditEntityAgentConfig] = None,
        websocket_manager: Optional["V3WebSocketManager"] = None,
        event_bus: Optional["EventBus"] = None,
        entity_index: Optional["EntityMarketIndex"] = None,
    ):
        """
        Initialize the Reddit Entity Agent.

        Args:
            config: Agent configuration
            websocket_manager: For broadcasting to frontend
            event_bus: For emitting events to other agents
            entity_index: EntityMarketIndex for market-led entity normalization
        """
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

        # spaCy components - KB-backed pipeline
        self._nlp: Optional["Language"] = None
        self._kb: Optional["KalshiKnowledgeBase"] = None

        # Related entity store
        self._related_entity_store = None

        # Supabase client for persistence
        self._supabase = None

        # Entity market index for normalization
        self._entity_index = entity_index

        # Sentiment task for batched scoring
        self._sentiment_task = None

        # Content extraction tools
        self._content_extractor = None
        self._video_transcriber = None

        # Market Impact Reasoner (for indirect market effects)
        self._market_impact_reasoner = None

        # LLM Entity Extractor (Phase 2 of two-phase pipeline)
        self._llm_entity_extractor = None

        # Relation Extractor (REL - extracts relations between entities)
        self._relation_extractor = None

        # Duplicate prevention - track seen post IDs in memory
        self._seen_post_ids: set = set()

        # History buffers for session persistence (snapshot on client connect)
        from collections import deque
        self._recent_posts: deque = deque(maxlen=30)
        self._recent_entities: deque = deque(maxlen=50)

        # Stats
        self._posts_processed = 0
        self._entities_extracted = 0
        self._posts_skipped = 0
        self._posts_skipped_duplicate = 0  # Track duplicates separately
        self._posts_inserted = 0
        self._entities_normalized = 0  # Track how many were normalized to market entities
        self._related_entities_extracted = 0
        self._content_extractions = 0  # Track content extraction attempts
        self._market_impact_reasoning_calls = 0  # Track reasoner invocations
        self._market_impact_signals_created = 0  # Track signals from reasoner
        self._llm_entity_extractions = 0  # Track LLM entity extraction calls
        self._llm_entity_kb_promotions = 0  # Track LLM entities promoted to market entities
        self._llm_market_matches = 0  # Track LLM direct market ticker matches
        self._llm_skipped_entities = 0  # Track entities with no market match
        self._posts_skipped_low_engagement = 0  # Track posts gated by metadata
        self._relation_extractions = 0  # Track REL extraction calls
        self._relations_extracted = 0  # Track individual relations found

        # Startup health tracking - tracks initialization status of critical components
        self._init_results: Dict[str, bool] = {
            "praw": False,
            "nlp_pipeline": False,
            "supabase": False,
        }
        self._startup_health: str = "initializing"  # "healthy", "degraded", "unhealthy"

        # Health broadcast task
        self._health_broadcast_task: Optional[asyncio.Task] = None

    async def _on_start(self) -> None:
        """Initialize resources on agent start."""
        if not self._config.enabled:
            logger.info("[reddit_entity] Agent disabled")
            return

        # Initialize PRAW (Reddit API) - track result
        self._init_results["praw"] = await self._init_praw()
        if not self._init_results["praw"]:
            logger.warning("[reddit_entity] PRAW not available, running in mock mode")

        # Initialize KB-backed NLP pipeline - track result
        self._init_results["nlp_pipeline"] = await self._init_kb_pipeline()
        if not self._init_results["nlp_pipeline"]:
            logger.warning("[reddit_entity] KB pipeline not available, entity extraction disabled")

        # Initialize Supabase for persistence (enables Realtime flow) - track result
        self._init_results["supabase"] = await self._init_supabase()
        if not self._init_results["supabase"]:
            logger.warning("[reddit_entity] Supabase not available, entities won't persist")
        else:
            # Pre-populate seen posts from DB to avoid re-processing on restart
            await self._load_seen_posts_from_db()

        # Compute startup health based on init results
        self._startup_health = self._compute_startup_health()
        logger.info(
            f"[reddit_entity] Startup health: {self._startup_health} "
            f"(praw={self._init_results['praw']}, nlp={self._init_results['nlp_pipeline']}, "
            f"supabase={self._init_results['supabase']})"
        )

        # Initialize related entity store
        if self._config.extract_related_entities:
            await self._init_related_entity_store()

        # Initialize batched sentiment task
        if self._config.use_batched_sentiment:
            from ..nlp.sentiment_task import BatchedSentimentTask
            self._sentiment_task = BatchedSentimentTask(
                model=self._config.sentiment_model,
                timeout=self._config.sentiment_timeout,
            )

        # Initialize content extraction tools
        if self._config.content_extraction_enabled:
            await self._init_content_extractor()

        # Initialize LLM entity extractor
        if self._config.llm_entity_extraction_enabled:
            await self._init_llm_entity_extractor()

        # Initialize market impact reasoner
        if self._config.market_impact_reasoning_enabled:
            await self._init_market_impact_reasoner()

        # Initialize relation extractor (REL)
        try:
            from ..nlp.relation_extractor import RelationExtractor
            self._relation_extractor = RelationExtractor(
                model=self._config.llm_entity_model,
                timeout=self._config.llm_entity_timeout,
            )
            logger.info("[reddit_entity] RelationExtractor initialized")
        except Exception as e:
            logger.warning(f"[reddit_entity] RelationExtractor init failed: {e}")

        # Start streaming task
        self._stream_task = asyncio.create_task(self._stream_loop())

        # Start health broadcast task (sends status every 5 seconds)
        self._health_broadcast_task = asyncio.create_task(self._health_broadcast_loop())

        content_status = "enabled" if self._content_extractor else "disabled"
        logger.info(
            f"[reddit_entity] Started (KB pipeline, content extraction {content_status}): "
            f"r/{' + r/'.join(self._config.subreddits)}"
        )

    def _compute_startup_health(self) -> str:
        """
        Compute startup health status based on initialization results.

        Returns:
            "healthy" - All critical components initialized
            "degraded" - Some components failed but agent can run
            "unhealthy" - Critical components failed, agent cannot function
        """
        praw_ok = self._init_results.get("praw", False)
        nlp_ok = self._init_results.get("nlp_pipeline", False)
        supabase_ok = self._init_results.get("supabase", False)

        # Healthy: All critical components working
        if praw_ok and nlp_ok and supabase_ok:
            return "healthy"

        # Unhealthy: No data source at all
        if not praw_ok:
            return "unhealthy"

        # Degraded: PRAW works but some processing missing
        return "degraded"

    async def _on_stop(self) -> None:
        """Cleanup resources on agent stop."""
        if self._stream_task and not self._stream_task.done():
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass

        # Cleanup health broadcast task
        if self._health_broadcast_task and not self._health_broadcast_task.done():
            self._health_broadcast_task.cancel()
            try:
                await self._health_broadcast_task
            except asyncio.CancelledError:
                pass

        # Cleanup content extractor
        if self._content_extractor:
            await self._content_extractor.close()

        logger.info(
            f"[reddit_entity] Stopped. Processed: {self._posts_processed}, "
            f"Extracted: {self._entities_extracted}, Content: {self._content_extractions}, "
            f"Skipped: {self._posts_skipped}"
        )

    def _get_agent_stats(self) -> Dict[str, Any]:
        """Get agent-specific statistics."""
        stats = {
            "posts_processed": self._posts_processed,
            "entities_extracted": self._entities_extracted,
            "entities_normalized": self._entities_normalized,
            "related_entities_extracted": self._related_entities_extracted,
            "content_extractions": self._content_extractions,
            "posts_skipped": self._posts_skipped,
            "posts_skipped_duplicate": self._posts_skipped_duplicate,
            "seen_posts_cached": len(self._seen_post_ids),
            "posts_inserted": self._posts_inserted,
            "subreddits": self._config.subreddits,
            "praw_available": self._reddit is not None,
            "nlp_available": self._nlp is not None,
            "kb_available": self._kb is not None,
            "supabase_available": self._supabase is not None,
            "entity_index_available": self._entity_index is not None,
            "batched_sentiment_active": self._sentiment_task is not None,
            "content_extraction_enabled": self._content_extractor is not None,
            # Market Impact Reasoning stats
            "market_impact_reasoning_enabled": self._market_impact_reasoner is not None,
            "market_impact_reasoning_calls": self._market_impact_reasoning_calls,
            "market_impact_signals_created": self._market_impact_signals_created,
            # LLM entity extraction stats
            "llm_entity_extraction_enabled": self._llm_entity_extractor is not None,
            "llm_entity_extractions": self._llm_entity_extractions,
            "llm_entity_kb_promotions": self._llm_entity_kb_promotions,
            "llm_market_matches": self._llm_market_matches,
            "llm_skipped_entities": self._llm_skipped_entities,
            # Metadata gating stats
            "posts_skipped_low_engagement": self._posts_skipped_low_engagement,
            "reddit_min_score": self._config.reddit_min_score,
            "reddit_min_comments": self._config.reddit_min_comments,
            # Startup health tracking
            "init_results": self._init_results.copy(),
            "startup_health": self._startup_health,
        }

        # Add content extraction sub-stats if available
        if self._content_extractor:
            stats["content_extractor_stats"] = self._content_extractor.get_stats()
        if self._video_transcriber:
            stats["video_transcriber_stats"] = self._video_transcriber.get_stats()

        return stats

    def set_entity_index(self, entity_index: "EntityMarketIndex") -> None:
        """Set the entity market index for normalization."""
        self._entity_index = entity_index

        # Update KB from entity index
        if self._entity_index:
            kb = self._entity_index.get_knowledge_base()
            if kb:
                self._kb = kb
                logger.info("[reddit_entity] KB updated from entity index")

        logger.info("[reddit_entity] Entity index attached for market-led normalization")

    async def _init_praw(self) -> bool:
        """Initialize PRAW Reddit client."""
        try:
            import os
            import praw

            client_id = os.getenv("REDDIT_CLIENT_ID")
            client_secret = os.getenv("REDDIT_CLIENT_SECRET")
            user_agent = os.getenv("REDDIT_USER_AGENT", "kalshiflow:v1.0 (by /u/kalshiflow)")

            if not client_id or not client_secret:
                logger.warning(
                    "[reddit_entity] REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET required"
                )
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

    async def _init_supabase(self) -> bool:
        """Initialize Supabase async client for persistence."""
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

    async def _init_kb_pipeline(self) -> bool:
        """
        Initialize the KB-backed NLP pipeline.

        Uses a spaCy pipeline with EntityRuler patterns from the
        KnowledgeBase for market entity detection and linking.
        """
        try:
            # Get KB from entity index
            if not self._entity_index:
                logger.warning("[reddit_entity] No entity index for KB pipeline")
                return False

            kb = self._entity_index.get_knowledge_base()
            if not kb:
                logger.warning("[reddit_entity] KB not available from entity index")
                return False

            self._kb = kb

            # Create pipeline with KB
            from ..nlp.pipeline import create_simple_entity_pipeline
            from ..nlp.extensions import register_custom_extensions

            # Register custom extensions
            register_custom_extensions()

            # Load pipeline in thread to avoid blocking
            def load_pipeline():
                return create_simple_entity_pipeline(
                    kb=kb,
                    model_name=self._config.spacy_model,
                )

            self._nlp = await asyncio.to_thread(load_pipeline)

            logger.info(
                f"[reddit_entity] KB pipeline initialized: "
                f"{self._kb.get_entity_count()} entities, "
                f"{self._kb.get_alias_count()} aliases"
            )
            return True

        except ImportError as e:
            logger.error(f"[reddit_entity] Missing dependency for KB pipeline: {e}")
            return False
        except Exception as e:
            logger.error(f"[reddit_entity] KB pipeline init error: {e}")
            return False

    async def _init_related_entity_store(self) -> bool:
        """Initialize the related entity store for second-hand signals."""
        try:
            from ..nlp.entity_store import RelatedEntityStore

            # Use existing Supabase client if available
            self._related_entity_store = RelatedEntityStore(
                supabase_client=self._supabase,
            )
            logger.info("[reddit_entity] Related entity store initialized")
            return True

        except Exception as e:
            logger.warning(f"[reddit_entity] Related entity store init failed: {e}")
            return False

    async def _init_content_extractor(self) -> bool:
        """Initialize content extraction tools (video transcription, article extraction)."""
        try:
            from ..tools import (
                ContentExtractor,
                ContentExtractorConfig,
                VideoTranscriber,
                VideoTranscriberConfig,
            )

            # Initialize video transcriber first
            if self._config.video_transcription_enabled:
                video_config = VideoTranscriberConfig(
                    max_duration_seconds=self._config.video_max_duration_seconds,
                    daily_budget_minutes=self._config.video_daily_budget_minutes,
                )
                self._video_transcriber = VideoTranscriber(config=video_config)
                logger.info("[reddit_entity] Video transcriber initialized")

            # Initialize content extractor (orchestrates video + article extraction)
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

    async def _init_market_impact_reasoner(self) -> bool:
        """Initialize the Market Impact Reasoner for indirect market effects."""
        try:
            from ..nlp.market_impact_reasoner import MarketImpactReasoner

            self._market_impact_reasoner = MarketImpactReasoner(
                model=self._config.market_impact_model,
                max_markets_in_prompt=self._config.market_impact_max_markets,
            )
            logger.info("[reddit_entity] Market Impact Reasoner initialized")
            return True

        except ImportError as e:
            logger.warning(f"[reddit_entity] Market Impact Reasoner dependencies not available: {e}")
            return False
        except Exception as e:
            logger.error(f"[reddit_entity] Market Impact Reasoner init failed: {e}")
            return False

    async def _init_llm_entity_extractor(self) -> bool:
        """Initialize the LLM entity extractor for Phase 2 extraction."""
        try:
            import os
            if not os.getenv("OPENAI_API_KEY"):
                logger.warning("[reddit_entity] OPENAI_API_KEY not set, LLM entity extraction disabled")
                return False

            from ..nlp.llm_entity_extractor import LLMEntityExtractor

            self._llm_entity_extractor = LLMEntityExtractor(
                model=self._config.llm_entity_model,
                timeout=self._config.llm_entity_timeout,
            )
            logger.info("[reddit_entity] LLM entity extractor initialized")
            return True

        except ImportError as e:
            logger.warning(f"[reddit_entity] LLM entity extractor dependencies not available: {e}")
            return False
        except Exception as e:
            logger.error(f"[reddit_entity] LLM entity extractor init failed: {e}")
            return False

    async def _insert_to_supabase(
        self, post_data: Dict[str, Any], entities: List[Dict[str, Any]]
    ) -> bool:
        """Insert post with entities into Supabase reddit_entities table."""
        if not self._supabase:
            return False

        try:
            # Calculate aggregate sentiment
            if entities:
                sentiments = [e.get("sentiment_score", 0) for e in entities]
                aggregate_sentiment = int(round(sum(sentiments) / len(sentiments)))
            else:
                aggregate_sentiment = 0

            # Build insert data matching table schema
            insert_data = {
                "post_id": post_data.get("post_id"),
                "subreddit": post_data.get("subreddit"),
                "title": post_data.get("title"),
                "url": post_data.get("url"),
                "author": post_data.get("author"),
                "score": post_data.get("score", 0),
                "num_comments": post_data.get("num_comments", 0),
                "post_created_utc": (
                    datetime.fromtimestamp(post_data["created_utc"], tz=timezone.utc).isoformat()
                    if post_data.get("created_utc") else None
                ),
                "entities": entities,
                "aggregate_sentiment": aggregate_sentiment,
                # Content extraction metadata
                "content_type": post_data.get("content_type"),
                "source_domain": post_data.get("source_domain"),
                "extraction_source": post_data.get("extraction_source"),
                "extraction_success": post_data.get("extraction_success", False),
                "extraction_error": post_data.get("extraction_error"),
            }

            # Insert (upsert on post_id to handle duplicates)
            # Handle missing columns gracefully (migration may not be applied yet)
            try:
                result = await self._supabase.table("reddit_entities").upsert(
                    insert_data, on_conflict="post_id"
                ).execute()
            except Exception as schema_error:
                error_str = str(schema_error)
                # If columns don't exist yet, retry without them
                if "content_type" in error_str or "source_domain" in error_str:
                    logger.warning(
                        "[reddit_entity] Content metadata columns not in schema, retrying without"
                    )
                    insert_data.pop("content_type", None)
                    insert_data.pop("source_domain", None)
                    insert_data.pop("extraction_source", None)
                    insert_data.pop("extraction_success", None)
                    insert_data.pop("extraction_error", None)
                    result = await self._supabase.table("reddit_entities").upsert(
                        insert_data, on_conflict="post_id"
                    ).execute()
                else:
                    raise

            if result.data:
                self._posts_inserted += 1
                logger.debug(f"[reddit_entity] Inserted post {post_data.get('post_id')}")
                return True

            return False

        except Exception as e:
            logger.error(f"[reddit_entity] Supabase insert error: {e}")
            return False

    async def _load_seen_posts_from_db(self) -> None:
        """Pre-populate seen post IDs from database to prevent re-processing on restart."""
        if not self._supabase:
            return

        try:
            # Load recent post IDs (last 2000 posts should cover most restarts)
            result = await self._supabase.table("reddit_entities").select(
                "post_id"
            ).order(
                "created_at", desc=True
            ).limit(2000).execute()

            if result.data:
                self._seen_post_ids = {row["post_id"] for row in result.data}
                logger.info(
                    f"[reddit_entity] Pre-loaded {len(self._seen_post_ids)} seen post IDs from DB"
                )
        except Exception as e:
            logger.warning(f"[reddit_entity] Could not pre-load seen posts: {e}")

    async def _stream_loop(self) -> None:
        """Main streaming loop for Reddit posts."""
        while self._running:
            try:
                if self._reddit:
                    await self._stream_reddit()
                else:
                    # Mock mode - wait and retry
                    await asyncio.sleep(60.0)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[reddit_entity] Stream error: {e}")
                self._record_error(str(e))
                await asyncio.sleep(30.0)

    async def _health_broadcast_loop(self) -> None:
        """Periodically broadcast health status to frontend."""
        while self._running:
            try:
                # Build health status
                health_status = {
                    "agent_name": "reddit_entity",
                    "is_running": self._running,
                    "praw_available": self._reddit is not None,
                    "nlp_available": self._nlp is not None,
                    "kb_available": self._kb is not None,
                    "supabase_available": self._supabase is not None,
                    "entity_index_available": self._entity_index is not None,
                    "subreddits": self._config.subreddits,
                    "posts_processed": self._posts_processed,
                    "entities_extracted": self._entities_extracted,
                    "posts_skipped": self._posts_skipped,
                    "last_error": self._stats.last_error,
                    "errors_count": self._stats.errors_count,
                    # Extraction stats for frontend visibility
                    "extraction_stats": self._content_extractor.get_stats() if self._content_extractor else {},
                    "video_stats": self._video_transcriber.get_stats() if self._video_transcriber else {},
                    "content_extractions": self._content_extractions,
                    # LLM entity extraction stats
                    "llm_entity_extraction_enabled": self._llm_entity_extractor is not None,
                    "llm_entity_extractions": self._llm_entity_extractions,
                    "llm_entity_kb_promotions": self._llm_entity_kb_promotions,
                    "llm_market_matches": self._llm_market_matches,
                    "llm_skipped_entities": self._llm_skipped_entities,
                    # Startup health tracking (new)
                    "init_results": self._init_results.copy(),
                    "startup_health": self._startup_health,
                }

                # Use startup_health for overall health status
                health_status["health"] = self._startup_health

                await self._broadcast_to_frontend("reddit_agent_health", health_status)
                await asyncio.sleep(5.0)  # Broadcast every 5 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"[reddit_entity] Health broadcast error: {e}")
                await asyncio.sleep(5.0)

    async def _stream_reddit(self) -> None:
        """Stream posts from Reddit using PRAW.

        PRAW is synchronous, so we run it in a thread to avoid blocking
        the async event loop (which would block HTTP endpoints).
        """
        subreddit_str = "+".join(self._config.subreddits)
        subreddit = self._reddit.subreddit(subreddit_str)

        # Run synchronous PRAW iteration in a thread pool
        def get_next_submission(stream_iter):
            """Get next submission from stream (blocking call)."""
            try:
                return next(stream_iter)
            except StopIteration:
                return None

        stream_iter = subreddit.stream.submissions(
            skip_existing=self._config.skip_existing
        )

        while self._running:
            # Run the blocking next() call in a thread
            submission = await asyncio.to_thread(get_next_submission, stream_iter)
            if submission is None or not self._running:
                break

            await self._process_submission(submission)

    async def _process_submission(self, submission) -> None:
        """Process a single Reddit submission."""
        try:
            title = submission.title
            post_id = submission.id
            selftext = getattr(submission, "selftext", "") or ""
            url = submission.url

            # Skip already-processed posts (prevents duplicate LLM calls on restart)
            if post_id in self._seen_post_ids:
                self._posts_skipped_duplicate += 1
                return
            self._seen_post_ids.add(post_id)

            # Skip short titles
            if len(title) < self._config.min_title_length:
                self._posts_skipped += 1
                return

            # Reddit metadata gating: skip low-engagement posts before LLM processing
            # Posts are still counted/broadcast but not processed through the pipeline
            post_score = getattr(submission, "score", 0) or 0
            post_comments = getattr(submission, "num_comments", 0) or 0
            if (
                post_score < self._config.reddit_min_score
                and post_comments < self._config.reddit_min_comments
            ):
                self._posts_skipped_low_engagement += 1
                logger.debug(
                    f"[reddit_entity] Skipping low-engagement post {post_id}: "
                    f"score={post_score}, comments={post_comments}"
                )
                return

            self._posts_processed += 1
            self._record_event_processed()

            # Broadcast post to frontend
            post_data = {
                "post_id": post_id,
                "subreddit": submission.subreddit.display_name,
                "title": title,
                "url": url,
                "author": str(submission.author) if submission.author else "unknown",
                "score": submission.score,
                "num_comments": submission.num_comments,
                "created_utc": submission.created_utc,
            }

            await self._broadcast_to_frontend("reddit_post", post_data)
            # Store in history buffer for session persistence
            self._recent_posts.appendleft(post_data)

            # Build combined text for entity extraction
            # Start with title + body text (for text posts)
            combined_text = title
            if selftext:
                combined_text = f"{title}\n\n{selftext}"

            # Track extraction metadata for persistence
            extraction_meta = {
                "content_type": "text",  # Default to text
                "source_domain": "reddit.com",  # Default to reddit
                "extraction_source": "selftext" if selftext else "none",
                "extraction_success": bool(selftext),
                "extraction_error": None,
            }

            # Extract additional content from URL (video transcription, article text)
            if self._content_extractor and url:
                extracted = await self._content_extractor.extract(url, selftext)
                # Update extraction metadata from result
                extraction_meta["content_type"] = extracted.content_type
                extraction_meta["source_domain"] = extracted.source_domain
                extraction_meta["extraction_source"] = extracted.source or "none"
                extraction_meta["extraction_success"] = extracted.success
                extraction_meta["extraction_error"] = extracted.error

                if extracted.success and extracted.text:
                    self._content_extractions += 1
                    combined_text = f"{combined_text}\n\n{extracted.text}"
                    logger.info(
                        f"[reddit_entity] Extracted {extracted.content_type} content "
                        f"({len(extracted.text)} chars) from {extracted.source_domain}"
                    )
                elif extracted.error:
                    logger.debug(
                        f"[reddit_entity] Extraction failed for {extracted.source_domain}: "
                        f"{extracted.error}"
                    )

            # Add extraction metadata to post_data for persistence
            post_data.update(extraction_meta)

            # Extract entities using KB-backed pipeline
            entities = []
            related_entities = []

            if self._nlp and self._kb:
                entities, related_entities = await self._extract_entities_kb(combined_text, post_data)

            # Process market entities
            for entity in entities:
                self._entities_extracted += 1
                await self._broadcast_to_frontend("entity_extracted", entity)
                # Store in history buffer for session persistence
                self._recent_entities.appendleft(entity)

            # Process related entities (second-hand signals)
            for related in related_entities:
                self._related_entities_extracted += 1
                await self._broadcast_to_frontend("related_entity", related)

            if entities:
                logger.debug(
                    f"[reddit_entity] {len(entities)} market entities, "
                    f"{len(related_entities)} related entities from post {post_id}"
                )
                # Insert into Supabase (triggers Realtime for Price Impact Agent)
                await self._insert_to_supabase(post_data, entities)

            # Match against objective entities (keyword-based, no LLM call)
            # Catches indirect effects: ICE → Government Shutdown via keyword matching
            if self._entity_index and entities:
                try:
                    entity_names = [
                        e.get("canonical_name", "") or e.get("entity_id", "")
                        for e in entities
                        if isinstance(e, dict)
                    ]
                    # Get categories from entities if available
                    entity_categories = []
                    for e in entities:
                        if isinstance(e, dict):
                            entity_categories.extend(e.get("categories", []))

                    obj_matches = self._entity_index.match_text_to_objective_entities(
                        text=combined_text,
                        extracted_entity_names=entity_names,
                        categories=entity_categories if entity_categories else None,
                    )
                    if obj_matches:
                        logger.info(
                            f"[reddit_entity] Objective entity matches for post {post_id}: "
                            f"{[(m.objective_entity.canonical_name, m.hit_score) for m in obj_matches[:3]]}"
                        )
                        # Feed objective entity matches to accumulator via price impact agent
                        # The accumulator will track these mentions and emit signals
                        from ..services.entity_accumulator import get_entity_accumulator
                        from ..schemas.kb_schemas import EntityMention
                        accumulator = get_entity_accumulator()
                        if accumulator:
                            # Use strongest entity's sentiment for objective entity mention
                            best_sentiment = 0
                            best_confidence = 0.5
                            for e in entities:
                                if isinstance(e, dict):
                                    s = abs(e.get("sentiment_score", 0))
                                    if s > abs(best_sentiment):
                                        best_sentiment = e.get("sentiment_score", 0)
                                        best_confidence = e.get("confidence", 0.5)

                            for match in obj_matches[:3]:  # Top 3 matches
                                obj = match.objective_entity
                                mention = EntityMention(
                                    entity_id=obj.entity_id,
                                    source_type="reddit_post",
                                    source_post_id=post_id,
                                    sentiment_score=best_sentiment,
                                    confidence=best_confidence,
                                    categories=obj.categories,
                                    reddit_score=post_data.get("score", 0),
                                    reddit_comments=post_data.get("num_comments", 0),
                                    context_snippet=title[:200],
                                    source_domain="reddit.com",
                                )
                                await accumulator.add_mention(
                                    mention=mention,
                                    canonical_name=obj.canonical_name,
                                    entity_category="objective",
                                    linked_market_tickers=obj.market_tickers,
                                )
                except Exception as obj_err:
                    logger.debug(f"[reddit_entity] Objective matching error: {obj_err}")

            # Extract relations between entities (REL)
            # Requires 2+ entities and an initialized relation extractor
            if self._relation_extractor and len(entities) >= 2:
                try:
                    from ..services.entity_accumulator import get_entity_accumulator
                    relations = await self._relation_extractor.extract_relations(
                        text=combined_text,
                        entities=entities,
                        source_post_id=post_id,
                    )
                    if relations:
                        self._relation_extractions += 1
                        self._relations_extracted += len(relations)
                        accumulator = get_entity_accumulator()
                        if accumulator:
                            for rel in relations:
                                await accumulator.add_relation(rel)
                        logger.info(
                            f"[reddit_entity] Extracted {len(relations)} relations "
                            f"from post {post_id}"
                        )
                except Exception as rel_err:
                    logger.debug(f"[reddit_entity] Relation extraction error: {rel_err}")

            # Run Market Impact Reasoning for indirect market effects
            # This catches impacts that direct entity→market mapping misses
            # (e.g., ICE shooting → government shutdown market)
            if self._market_impact_reasoner and self._entity_index:
                await self._run_market_impact_reasoning(
                    combined_text,
                    entities,
                    related_entities,
                    post_data,
                )

        except Exception as e:
            logger.error(f"[reddit_entity] Process error: {e}")
            self._record_error(str(e))

    async def _extract_entities_kb(
        self, text: str, post_data: Dict[str, Any]
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Two-phase entity extraction pipeline.

        Phase 1 (EntityRuler, unchanged): KB pattern match -> MARKET_ENTITY
            Fast, free, accurate. Uses spaCy EntityRuler with KB-generated patterns.

        Phase 2 (LLM Entity Extractor): Structured extraction + sentiment in ONE call
            Replaces unreliable spaCy NER AND separate batched sentiment call.
            Produces higher-quality entities from Reddit headlines.
            LLM-extracted entities are cross-referenced against KB for market linking.

        Falls back to old spaCy NER if LLM extraction is disabled or fails.

        Returns:
            Tuple of (market_entities, related_entities)
        """
        market_entities = []
        related_entities = []
        seen_entity_ids = set()

        if not self._nlp or not self._kb:
            return market_entities, related_entities

        try:
            # =================================================================
            # Phase 1: KB EntityRuler (fast, free, accurate) - UNCHANGED
            # =================================================================
            doc = self._nlp(text)

            # Collect Phase 1 market entities (EntityRuler matches)
            phase1_market_entities_for_sentiment = []

            for ent in doc.ents:
                if ent.text.lower() in ENTITY_STOPWORDS:
                    continue

                if ent.label_ == "MARKET_ENTITY":
                    entity_id = ent.ent_id_
                    if not entity_id or entity_id in seen_entity_ids:
                        continue
                    seen_entity_ids.add(entity_id)

                    metadata = self._kb.get_entity_metadata(entity_id)
                    if not metadata:
                        continue

                    phase1_market_entities_for_sentiment.append({
                        "text": metadata.canonical_name,
                        "label": "MARKET_ENTITY",
                        "entity_id": entity_id,
                        "metadata": metadata,
                        "matched_text": ent.text,
                    })

            # =================================================================
            # Phase 2: LLM Market-Aware Entity Extraction
            # =================================================================
            llm_entities = []
            use_llm = (
                self._llm_entity_extractor is not None
                and self._config.llm_entity_extraction_enabled
            )

            # Get cached enriched market context (pre-formatted, not per-post)
            market_prompt = ""
            if self._entity_index:
                market_prompt = self._entity_index.get_enriched_market_prompt()

            if use_llm:
                try:
                    if market_prompt:
                        llm_entities = await self._llm_entity_extractor.extract_with_markets(
                            title=post_data.get("title", ""),
                            subreddit=post_data.get("subreddit", ""),
                            body=text[:1000],
                            market_prompt=market_prompt,
                        )
                    else:
                        llm_entities = await self._llm_entity_extractor.extract(
                            title=post_data.get("title", ""),
                            subreddit=post_data.get("subreddit", ""),
                            body=text[:1000],
                        )
                    self._llm_entity_extractions += 1
                except Exception as llm_err:
                    logger.warning(f"[reddit_entity] LLM extraction failed: {llm_err}")
                    if self._config.llm_entity_fallback_to_spacy:
                        llm_entities = []  # Will fall through to spaCy NER below

            # Build sentiment map from LLM entities (name -> sentiment)
            # This is used for Phase 1 entities too, avoiding a separate sentiment call
            llm_sentiment_map = {}
            for llm_ent in llm_entities:
                llm_sentiment_map[llm_ent.name.lower()] = llm_ent.sentiment

            # Build context map from LLM entities (name -> per-entity context summary)
            llm_context_map = {}
            for llm_ent in llm_entities:
                if llm_ent.context:
                    llm_context_map[llm_ent.name.lower()] = llm_ent.context

            # Score Phase 1 market entities using LLM sentiment or fallback
            sentiments = {}
            if phase1_market_entities_for_sentiment:
                if llm_entities:
                    # Try to match Phase 1 entities to LLM sentiment results
                    unmatched = []
                    for e in phase1_market_entities_for_sentiment:
                        name_lower = e["text"].lower()
                        matched_lower = e["matched_text"].lower()
                        # Try exact name match, then matched text
                        if name_lower in llm_sentiment_map:
                            sentiments[e["text"]] = llm_sentiment_map[name_lower]
                        elif matched_lower in llm_sentiment_map:
                            sentiments[e["text"]] = llm_sentiment_map[matched_lower]
                        else:
                            # Try partial match (LLM may use different name form)
                            # Guard: require min 4 chars to avoid greedy matches like "trump" → "trump_organization"
                            found = False
                            if len(name_lower) >= 4:
                                for llm_name, llm_sent in llm_sentiment_map.items():
                                    if len(llm_name) >= 4 and (llm_name == name_lower or llm_name == matched_lower):
                                        # Exact match on either form — highest priority
                                        sentiments[e["text"]] = llm_sent
                                        found = True
                                        break
                                if not found:
                                    for llm_name, llm_sent in llm_sentiment_map.items():
                                        if len(llm_name) >= 4 and (llm_name in name_lower or name_lower in llm_name):
                                            sentiments[e["text"]] = llm_sent
                                            found = True
                                            break
                            if not found:
                                unmatched.append(e)

                    # For unmatched Phase 1 entities, use batched sentiment fallback
                    if unmatched:
                        if self._sentiment_task and self._config.use_batched_sentiment:
                            results = await self._sentiment_task.analyze_async(
                                text,
                                [(e["text"], e["label"]) for e in unmatched]
                            )
                            for r in results:
                                sentiments[r.entity_text] = r.sentiment_score
                        else:
                            for e in unmatched:
                                s = await self._get_sentiment(text, e["text"])
                                sentiments[e["text"]] = s
                else:
                    # No LLM entities — use old batched sentiment for Phase 1
                    if self._sentiment_task and self._config.use_batched_sentiment:
                        results = await self._sentiment_task.analyze_async(
                            text,
                            [(e["text"], e["label"]) for e in phase1_market_entities_for_sentiment]
                        )
                        sentiments = {r.entity_text: r.sentiment_score for r in results}
                    else:
                        for e in phase1_market_entities_for_sentiment:
                            s = await self._get_sentiment(text, e["text"])
                            sentiments[e["text"]] = s

            # Build Phase 1 market entity output
            market_entity_ids = []

            for e in phase1_market_entities_for_sentiment:
                sentiment = sentiments.get(e["text"], 0)
                metadata = e["metadata"]
                self._entities_normalized += 1
                market_entity_ids.append(e["entity_id"])

                market_entities.append({
                    "entity_id": e["entity_id"],
                    "canonical_name": metadata.canonical_name,
                    "entity_type": metadata.entity_type,
                    "sentiment_score": sentiment,
                    "confidence": 1.0,
                    "context_snippet": llm_context_map.get(e["text"].lower()) or llm_context_map.get(e["matched_text"].lower()) or text[:200],
                    "post_id": post_data.get("post_id"),
                    "subreddit": post_data.get("subreddit"),
                    "was_normalized": True,
                    "matched_text": e["matched_text"],
                    "market_ticker": metadata.market_ticker,
                    "market_type": metadata.market_type,
                })

                logger.info(
                    f"[reddit_entity] Phase1 KB matched '{e['matched_text']}' -> "
                    f"'{metadata.canonical_name}' (sentiment={sentiment})"
                )

            # =================================================================
            # Phase 2 output: LLM-driven market ticker matching
            # =================================================================
            if llm_entities and self._entity_index:
                from ..schemas.entity_schemas import normalize_entity_id

                for llm_ent in llm_entities:
                    normalized_id = normalize_entity_id(llm_ent.name)

                    # Skip if already seen in Phase 1
                    if normalized_id in seen_entity_ids:
                        continue
                    seen_entity_ids.add(normalized_id)

                    if llm_ent.market_tickers:
                        # LLM directly mapped this entity to market tickers
                        for ticker in llm_ent.market_tickers:
                            result = self._entity_index.get_market_mapping_by_ticker(ticker)
                            if not result:
                                logger.debug(
                                    f"[reddit_entity] LLM ticker '{ticker}' not found in index, skipping"
                                )
                                continue

                            entity_id, mapping = result
                            self._llm_market_matches += 1
                            market_entity_ids.append(entity_id)

                            # Get canonical entity for metadata
                            canonical = self._entity_index.get_canonical_entity(entity_id)
                            canonical_name = canonical.canonical_name if canonical else llm_ent.name
                            entity_type = canonical.entity_type if canonical else llm_ent.entity_type.lower()

                            market_entities.append({
                                "entity_id": entity_id,
                                "canonical_name": canonical_name,
                                "entity_type": entity_type,
                                "sentiment_score": llm_ent.sentiment,
                                "confidence": llm_ent.confidence_float,
                                "context_snippet": llm_ent.context or text[:200],
                                "post_id": post_data.get("post_id"),
                                "subreddit": post_data.get("subreddit"),
                                "was_normalized": True,
                                "matched_text": llm_ent.name,
                                "market_ticker": mapping.market_ticker,
                                "market_type": mapping.market_type,
                                "source": "llm_market_mapped",
                            })

                            logger.info(
                                f"[reddit_entity] Phase2 LLM market-mapped '{llm_ent.name}' -> "
                                f"'{canonical_name}' ({mapping.market_ticker}, "
                                f"sentiment={llm_ent.sentiment})"
                            )
                    else:
                        # No market tickers - treat as related entity
                        self._llm_skipped_entities += 1

                        related_entity = {
                            "entity_text": llm_ent.name,
                            "entity_type": llm_ent.entity_type,
                            "normalized_id": normalized_id,
                            "sentiment_score": llm_ent.sentiment,
                            "confidence": llm_ent.confidence_float,
                            "source_post_id": post_data.get("post_id"),
                            "source_subreddit": post_data.get("subreddit"),
                            "context_snippet": llm_ent.context or text[:200],
                            "co_occurring_market_entities": market_entity_ids.copy(),
                            "source": "llm_extracted",
                        }
                        related_entities.append(related_entity)

                        if self._related_entity_store:
                            from ..nlp.entity_store import RelatedEntity
                            store_data = {
                                k: v for k, v in related_entity.items()
                                if k != "source"
                            }
                            await self._related_entity_store.insert(
                                RelatedEntity(**store_data)
                            )

                        logger.debug(
                            f"[reddit_entity] Skipped '{llm_ent.name}' - no market match"
                        )

            elif not llm_entities and self._config.llm_entity_fallback_to_spacy:
                # =============================================================
                # Fallback: Use spaCy NER for related entities (old path)
                # =============================================================
                if self._config.extract_related_entities:
                    from ..schemas.entity_schemas import normalize_entity_id

                    spacy_related_for_sentiment = []

                    for ent in doc.ents:
                        if ent.text.lower() in ENTITY_STOPWORDS:
                            continue
                        if ent.label_ in {"PERSON", "ORG", "GPE", "EVENT", "NORP"}:
                            normalized_id = normalize_entity_id(ent.text)
                            if normalized_id in seen_entity_ids:
                                continue
                            seen_entity_ids.add(normalized_id)

                            spacy_related_for_sentiment.append({
                                "text": ent.text,
                                "label": ent.label_,
                                "entity_id": normalized_id,
                            })

                    # Score sentiment for spaCy-extracted related entities
                    if spacy_related_for_sentiment:
                        spacy_sentiments = {}
                        if self._sentiment_task and self._config.use_batched_sentiment:
                            results = await self._sentiment_task.analyze_async(
                                text,
                                [(e["text"], e["label"]) for e in spacy_related_for_sentiment]
                            )
                            spacy_sentiments = {r.entity_text: r.sentiment_score for r in results}
                        else:
                            for e in spacy_related_for_sentiment:
                                s = await self._get_sentiment(text, e["text"])
                                spacy_sentiments[e["text"]] = s

                        for e in spacy_related_for_sentiment:
                            sentiment = spacy_sentiments.get(e["text"], 0)
                            related_entity = {
                                "entity_text": e["text"],
                                "entity_type": e["label"],
                                "normalized_id": e["entity_id"],
                                "sentiment_score": sentiment,
                                "confidence": 0.8,
                                "source_post_id": post_data.get("post_id"),
                                "source_subreddit": post_data.get("subreddit"),
                                "context_snippet": text[:200],
                                "co_occurring_market_entities": market_entity_ids.copy(),
                            }
                            related_entities.append(related_entity)

                            if self._related_entity_store:
                                from ..nlp.entity_store import RelatedEntity
                                await self._related_entity_store.insert(
                                    RelatedEntity(**related_entity)
                                )

                            logger.debug(
                                f"[reddit_entity] Fallback spaCy entity '{e['text']}' "
                                f"({e['label']}, sentiment={sentiment})"
                            )

        except Exception as e:
            logger.error(f"[reddit_entity] KB entity extraction error: {e}")

        return market_entities, related_entities

    async def _run_market_impact_reasoning(
        self,
        content: str,
        market_entities: List[Dict[str, Any]],
        related_entities: List[Dict[str, Any]],
        post_data: Dict[str, Any],
    ) -> None:
        """
        Run Market Impact Reasoning for indirect market effects.

        This catches market impacts that direct entity→market mapping misses.
        Example: "ICE shooting in Minnesota" → government shutdown market.

        Args:
            content: Combined text content
            market_entities: Already-extracted market entities
            related_entities: Already-extracted related entities
            post_data: Post metadata
        """
        if not self._market_impact_reasoner or not self._entity_index:
            return

        try:
            from ..nlp.market_impact_reasoner import (
                MarketInfo,
                should_analyze_for_market_impact,
            )

            # Build entity list for filter check and reasoner
            all_entities = []
            for e in market_entities:
                all_entities.append((e.get("canonical_name", ""), "MARKET_ENTITY"))
            for e in related_entities:
                all_entities.append((e.get("entity_text", ""), e.get("entity_type", "")))

            # Calculate average sentiment magnitude
            sentiments = [abs(e.get("sentiment_score", 0)) for e in market_entities]
            sentiments.extend([abs(e.get("sentiment_score", 0)) for e in related_entities])
            avg_sentiment_magnitude = sum(sentiments) / len(sentiments) if sentiments else 0

            # Check if content qualifies for market impact reasoning
            if not should_analyze_for_market_impact(
                content=content,
                entities=all_entities,
                avg_sentiment_magnitude=avg_sentiment_magnitude,
                entity_mapped_count=len(market_entities),
            ):
                logger.debug("[reddit_entity] Content does not qualify for market impact reasoning")
                return

            # Get all markets from EntityMarketIndex
            all_markets_data = self._entity_index.get_all_markets_for_reasoner()
            if not all_markets_data:
                logger.debug("[reddit_entity] No markets available for impact reasoning")
                return

            # Convert to MarketInfo objects
            active_markets = [
                MarketInfo(
                    ticker=m["ticker"],
                    title=m["title"],
                    event_ticker=m.get("event_ticker", ""),
                )
                for m in all_markets_data
            ]

            # Get tickers already handled by entity mapping (to exclude from reasoning)
            exclude_tickers = [e.get("market_ticker", "") for e in market_entities if e.get("market_ticker")]

            # Run the reasoner
            self._market_impact_reasoning_calls += 1
            results = await self._market_impact_reasoner.analyze(
                content=content,
                entities=all_entities,
                active_markets=active_markets,
                exclude_tickers=exclude_tickers,
            )

            if not results:
                logger.debug(f"[reddit_entity] No market impacts found for post {post_data.get('post_id')}")
                return

            logger.info(
                f"[reddit_entity] 🎯 Market Impact Reasoning found {len(results)} indirect impacts "
                f"for post {post_data.get('post_id')}"
            )

            # Process and persist results
            for impact in results:
                self._market_impact_signals_created += 1

                # Broadcast to frontend
                impact_data = {
                    **impact,
                    "source_post_id": post_data.get("post_id"),
                    "source_subreddit": post_data.get("subreddit"),
                    "source_title": post_data.get("title"),
                }
                await self._broadcast_to_frontend("market_impact_reasoning", impact_data)

                # Persist to Supabase market_price_impacts table
                await self._insert_market_impact_signal(impact_data)

        except Exception as e:
            logger.error(f"[reddit_entity] Market impact reasoning error: {e}")

    async def _insert_market_impact_signal(self, impact_data: Dict[str, Any]) -> bool:
        """Insert a market impact reasoning signal into Supabase."""
        if not self._supabase:
            return False

        try:
            import time

            insert_data = {
                "source_post_id": impact_data.get("source_post_id", ""),
                "source_subreddit": impact_data.get("source_subreddit", ""),
                "entity_id": "market_impact_reasoning",  # Special marker
                "entity_name": "Indirect Impact",
                "market_ticker": impact_data.get("market_ticker", ""),
                "event_ticker": impact_data.get("event_ticker", ""),
                "market_type": "REASONED",  # Mark as reasoned (not entity-based)
                "sentiment_score": 0,  # Not applicable for reasoned signals
                "price_impact_score": impact_data.get("price_impact_score", 0),
                "confidence": impact_data.get("confidence_float", 0.5),
                "transformation_logic": impact_data.get("reasoning", ""),
                "source_title": impact_data.get("source_title", ""),
                "content_type": "market_impact_reasoning",
                "source_domain": "reddit.com",
            }

            result = await self._supabase.table("market_price_impacts").insert(insert_data).execute()

            if result.data:
                logger.debug(
                    f"[reddit_entity] Inserted market impact signal for {impact_data.get('market_ticker')}"
                )
                return True
            return False

        except Exception as e:
            logger.error(f"[reddit_entity] Market impact signal insert error: {e}")
            return False

    async def _get_sentiment(self, context: str, entity: str) -> int:
        """Get sentiment score for an entity in context using LLM."""
        try:
            import os
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

            prompt = f"""Analyze the sentiment about "{entity}" in this headline:

"{context}"

Rate the sentiment on a scale from -100 (extremely negative) to +100 (extremely positive).
Consider: Is the news good or bad for this entity? Does it help or hurt their reputation/position?

Respond with ONLY a number between -100 and 100."""

            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=self._config.sentiment_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=10,
                    temperature=0,
                ),
                timeout=self._config.sentiment_timeout,
            )

            result = response.choices[0].message.content.strip()
            # Parse the number
            sentiment = int(float(result))
            return max(-100, min(100, sentiment))

        except Exception as e:
            logger.debug(f"[reddit_entity] Sentiment error for {entity}: {e}")
            return 0  # Neutral on error

    def get_entity_snapshot(self) -> Dict[str, Any]:
        """
        Get entity pipeline snapshot for new client initialization.

        Returns a snapshot containing recent posts and entities for
        session persistence across page refreshes.

        Returns:
            Dict containing recent_posts, recent_entities, stats, and is_active
        """
        return {
            "reddit_posts": list(self._recent_posts),
            "entities": list(self._recent_entities),
            "stats": {
                # Use camelCase to match frontend expectations
                "postsProcessed": self._posts_processed,
                "entitiesExtracted": self._entities_extracted,
            },
            "is_active": self._running,
        }
