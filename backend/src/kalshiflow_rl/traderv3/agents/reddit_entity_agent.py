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

        # Stats
        self._posts_processed = 0
        self._entities_extracted = 0
        self._posts_skipped = 0
        self._posts_inserted = 0
        self._entities_normalized = 0  # Track how many were normalized to market entities
        self._related_entities_extracted = 0
        self._content_extractions = 0  # Track content extraction attempts

    async def _on_start(self) -> None:
        """Initialize resources on agent start."""
        if not self._config.enabled:
            logger.info("[reddit_entity] Agent disabled")
            return

        # Initialize PRAW (Reddit API)
        if not await self._init_praw():
            logger.warning("[reddit_entity] PRAW not available, running in mock mode")

        # Initialize KB-backed NLP pipeline
        if not await self._init_kb_pipeline():
            logger.warning("[reddit_entity] KB pipeline not available, entity extraction disabled")

        # Initialize Supabase for persistence (enables Realtime flow)
        if not await self._init_supabase():
            logger.warning("[reddit_entity] Supabase not available, entities won't persist")

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

        # Start streaming task
        self._stream_task = asyncio.create_task(self._stream_loop())

        content_status = "enabled" if self._content_extractor else "disabled"
        logger.info(
            f"[reddit_entity] Started (KB pipeline, content extraction {content_status}): "
            f"r/{' + r/'.join(self._config.subreddits)}"
        )

    async def _on_stop(self) -> None:
        """Cleanup resources on agent stop."""
        if self._stream_task and not self._stream_task.done():
            self._stream_task.cancel()
            try:
                await self._stream_task
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
            "posts_inserted": self._posts_inserted,
            "subreddits": self._config.subreddits,
            "praw_available": self._reddit is not None,
            "nlp_available": self._nlp is not None,
            "kb_available": self._kb is not None,
            "supabase_available": self._supabase is not None,
            "entity_index_available": self._entity_index is not None,
            "batched_sentiment_active": self._sentiment_task is not None,
            "content_extraction_enabled": self._content_extractor is not None,
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
                aggregate_sentiment = sum(sentiments) // len(sentiments)
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
                "post_created_utc": None,  # Would need conversion from Unix timestamp
                "entities": entities,
                "aggregate_sentiment": aggregate_sentiment,
            }

            # Insert (upsert on post_id to handle duplicates)
            result = await self._supabase.table("reddit_entities").upsert(
                insert_data, on_conflict="post_id"
            ).execute()

            if result.data:
                self._posts_inserted += 1
                logger.debug(f"[reddit_entity] Inserted post {post_data.get('post_id')}")
                return True

            return False

        except Exception as e:
            logger.error(f"[reddit_entity] Supabase insert error: {e}")
            return False

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

            # Skip short titles
            if len(title) < self._config.min_title_length:
                self._posts_skipped += 1
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

            # Build combined text for entity extraction
            # Start with title + body text (for text posts)
            combined_text = title
            if selftext:
                combined_text = f"{title}\n\n{selftext}"

            # Extract additional content from URL (video transcription, article text)
            if self._content_extractor and url:
                extracted = await self._content_extractor.extract(url, selftext)
                if extracted.success and extracted.text:
                    self._content_extractions += 1
                    combined_text = f"{combined_text}\n\n{extracted.text}"
                    logger.info(
                        f"[reddit_entity] Extracted {extracted.content_type} content "
                        f"({len(extracted.text)} chars) from {url[:50]}..."
                    )

            # Extract entities using KB-backed pipeline
            entities = []
            related_entities = []

            if self._nlp and self._kb:
                entities, related_entities = await self._extract_entities_kb(combined_text, post_data)

            # Process market entities
            for entity in entities:
                self._entities_extracted += 1
                await self._broadcast_to_frontend("entity_extracted", entity)

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

        except Exception as e:
            logger.error(f"[reddit_entity] Process error: {e}")
            self._record_error(str(e))

    async def _extract_entities_kb(
        self, text: str, post_data: Dict[str, Any]
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Extract entities using the KB-backed NLP pipeline.

        This is the new pipeline that:
        1. Uses EntityRuler patterns from KB for market entity detection
        2. Uses spaCy NER for general entity discovery (PERSON, ORG, GPE, EVENT)
        3. Links market entities to KB for metadata
        4. Scores sentiment for all entities (batched)

        Returns:
            Tuple of (market_entities, related_entities)
        """
        market_entities = []
        related_entities = []
        seen_entity_ids = set()

        if not self._nlp or not self._kb:
            return market_entities, related_entities

        try:
            # Process text with KB-backed pipeline
            doc = self._nlp(text)

            # Collect entities for batched sentiment
            entities_for_sentiment = []

            for ent in doc.ents:
                # Skip stopwords
                if ent.text.lower() in ENTITY_STOPWORDS:
                    continue

                if ent.label_ == "MARKET_ENTITY":
                    # Market-linked entity
                    entity_id = ent.ent_id_
                    if not entity_id or entity_id in seen_entity_ids:
                        continue
                    seen_entity_ids.add(entity_id)

                    # Get metadata from KB
                    metadata = self._kb.get_entity_metadata(entity_id)
                    if not metadata:
                        continue

                    entities_for_sentiment.append({
                        "text": metadata.canonical_name,
                        "label": "MARKET_ENTITY",
                        "entity_id": entity_id,
                        "metadata": metadata,
                        "matched_text": ent.text,
                    })

                elif ent.label_ in {"PERSON", "ORG", "GPE", "EVENT"}:
                    # Related entity (for second-hand signals)
                    if self._config.extract_related_entities:
                        from ..schemas.entity_schemas import normalize_entity_id
                        normalized_id = normalize_entity_id(ent.text)

                        if normalized_id in seen_entity_ids:
                            continue
                        seen_entity_ids.add(normalized_id)

                        entities_for_sentiment.append({
                            "text": ent.text,
                            "label": ent.label_,
                            "entity_id": normalized_id,
                            "metadata": None,
                            "matched_text": ent.text,
                        })

            # Batch sentiment analysis
            sentiments = {}
            if entities_for_sentiment:
                if self._sentiment_task and self._config.use_batched_sentiment:
                    # Use batched sentiment task
                    results = await self._sentiment_task.analyze_async(
                        text,
                        [(e["text"], e["label"]) for e in entities_for_sentiment]
                    )
                    sentiments = {r.entity_text: r.sentiment_score for r in results}
                else:
                    # Fall back to per-entity sentiment
                    for e in entities_for_sentiment:
                        sentiment = await self._get_sentiment(text, e["text"])
                        sentiments[e["text"]] = sentiment

            # Build output entities
            market_entity_ids = []  # Track for co-occurrence

            for e in entities_for_sentiment:
                sentiment = sentiments.get(e["text"], 0)

                if e["label"] == "MARKET_ENTITY":
                    metadata = e["metadata"]
                    self._entities_normalized += 1
                    market_entity_ids.append(e["entity_id"])

                    market_entities.append({
                        "entity_id": e["entity_id"],
                        "canonical_name": metadata.canonical_name,
                        "entity_type": metadata.entity_type,
                        "sentiment_score": sentiment,
                        "confidence": 1.0,
                        "context_snippet": text[:200],
                        "post_id": post_data.get("post_id"),
                        "subreddit": post_data.get("subreddit"),
                        "was_normalized": True,
                        "matched_text": e["matched_text"],
                        "market_ticker": metadata.market_ticker,
                        "market_type": metadata.market_type,
                    })

                    logger.info(
                        f"[reddit_entity] ✅ KB matched '{e['matched_text']}' -> "
                        f"'{metadata.canonical_name}' (sentiment={sentiment})"
                    )

                else:
                    # Related entity
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

                    # Store in related entity store
                    if self._related_entity_store:
                        from ..nlp.entity_store import RelatedEntity
                        await self._related_entity_store.insert(
                            RelatedEntity(**related_entity)
                        )

                    logger.debug(
                        f"[reddit_entity] Related entity '{e['text']}' ({e['label']}, "
                        f"sentiment={sentiment})"
                    )

        except Exception as e:
            logger.error(f"[reddit_entity] KB entity extraction error: {e}")

        return market_entities, related_entities

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
