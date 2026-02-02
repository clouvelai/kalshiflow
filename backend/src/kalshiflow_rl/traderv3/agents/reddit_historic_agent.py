"""
Reddit Historic Agent - Fetches top Reddit posts (past 24h) with comments
and runs extraction via the KalshiExtractor pipeline.

Produces a cached daily digest that the deep agent can query via
get_reddit_daily_digest(). Source type: `reddit_historic` (distinct
from live `reddit_post`).

All PRAW calls are synchronous, so they are wrapped in asyncio.to_thread().
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

logger = logging.getLogger("kalshiflow_rl.traderv3.agents.reddit_historic_agent")


@dataclass
class RedditHistoricAgentConfig:
    """Configuration for the Reddit Historic Agent."""

    subreddits: List[str] = field(default_factory=lambda: ["politics", "news"])
    posts_limit: int = 25
    comments_per_post: int = 20
    comment_max_chars: int = 200
    post_body_max_chars: int = 2000
    time_filter: str = "day"  # PRAW time_filter for .top()
    min_post_score: int = 10
    digest_cooldown_seconds: float = 21600.0  # 6 hours
    max_concurrent_posts: int = 5  # Semaphore bound for parallel processing

    enabled: bool = True


class RedditHistoricAgent(BaseAgent):
    """
    Agent that fetches top Reddit posts (past 24h) with comments and
    runs extraction via the KalshiExtractor pipeline.

    Pipeline:
    1. PRAW: fetch top N posts from configured subreddits (past 24h)
    2. For each post above min_score: fetch top M comments
    3. Build combined text: title + body + formatted comments
    4. Single KalshiExtractor call per post (source_type="reddit_historic")
    5. Insert to Supabase `extractions` table (dedup via unique index)
    6. Cache digest summary for deep agent tool access
    """

    def __init__(
        self,
        config: Optional[RedditHistoricAgentConfig] = None,
        websocket_manager: Optional["V3WebSocketManager"] = None,
        event_bus: Optional["EventBus"] = None,
    ):
        super().__init__(
            name="reddit_historic",
            display_name="Reddit Historic Agent",
            event_bus=event_bus,
            websocket_manager=websocket_manager,
        )

        self._config = config or RedditHistoricAgentConfig()

        # PRAW client
        self._reddit = None

        # KalshiExtractor
        self._extractor: Optional["KalshiExtractor"] = None

        # Supabase client
        self._supabase = None

        # Active event configs
        self._event_configs: List["EventConfig"] = []
        self._event_configs_last_refresh: float = 0.0

        # Cached digest
        self._cached_digest: Optional[Dict[str, Any]] = None
        self._last_digest_time: float = 0.0

        # Digest task
        self._digest_task: Optional[asyncio.Task] = None

        # Stats
        self._digests_completed = 0
        self._posts_processed = 0
        self._extractions_total = 0
        self._posts_skipped = 0

        # Startup health
        self._init_results: Dict[str, bool] = {
            "praw": False,
            "extractor": False,
            "supabase": False,
        }

    async def _on_start(self) -> None:
        """Initialize resources on agent start."""
        if not self._config.enabled:
            logger.info("[reddit_historic] Agent disabled")
            return

        self._init_results["praw"] = await self._init_praw()
        if not self._init_results["praw"]:
            logger.warning("[reddit_historic] PRAW not available")

        self._init_results["extractor"] = await self._init_extractor()
        if not self._init_results["extractor"]:
            logger.warning("[reddit_historic] KalshiExtractor not available")

        self._init_results["supabase"] = await self._init_supabase()
        if not self._init_results["supabase"]:
            logger.warning("[reddit_historic] Supabase not available")
        else:
            await self._refresh_event_configs()

        logger.info(
            f"[reddit_historic] Startup: praw={self._init_results['praw']}, "
            f"extractor={self._init_results['extractor']}, "
            f"supabase={self._init_results['supabase']}"
        )

        # Fire initial digest in background
        self._digest_task = asyncio.create_task(self._initial_digest())

        logger.info(
            f"[reddit_historic] Started: r/{' + r/'.join(self._config.subreddits)}"
        )

    async def _initial_digest(self) -> None:
        """Run initial digest, catching errors."""
        try:
            await asyncio.sleep(5.0)  # Let other agents settle
            await self._run_digest()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"[reddit_historic] Initial digest failed: {e}")
            self._record_error(str(e))

    async def _on_stop(self) -> None:
        """Cleanup resources on agent stop."""
        if self._digest_task and not self._digest_task.done():
            self._digest_task.cancel()
            try:
                await self._digest_task
            except asyncio.CancelledError:
                pass

        logger.info(
            f"[reddit_historic] Stopped. Digests: {self._digests_completed}, "
            f"Posts: {self._posts_processed}, Extractions: {self._extractions_total}"
        )

    def _get_agent_stats(self) -> Dict[str, Any]:
        """Get agent-specific statistics."""
        return {
            "digests_completed": self._digests_completed,
            "posts_processed": self._posts_processed,
            "posts_skipped": self._posts_skipped,
            "extractions_total": self._extractions_total,
            "subreddits": self._config.subreddits,
            "praw_available": self._reddit is not None,
            "extractor_available": self._extractor is not None,
            "supabase_available": self._supabase is not None,
            "event_configs_loaded": len(self._event_configs),
            "last_digest_time": self._last_digest_time,
            "has_cached_digest": self._cached_digest is not None,
            "init_results": self._init_results.copy(),
        }

    # === Initialization (same pattern as RedditEntityAgent) ===

    async def _init_praw(self) -> bool:
        """Initialize PRAW Reddit client."""
        try:
            import os
            import praw

            client_id = os.getenv("REDDIT_CLIENT_ID")
            client_secret = os.getenv("REDDIT_CLIENT_SECRET")
            user_agent = os.getenv("REDDIT_USER_AGENT", "kalshiflow:v1.0 (by /u/kalshiflow)")

            if not client_id or not client_secret:
                logger.warning("[reddit_historic] REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET required")
                return False

            self._reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent,
            )
            logger.info("[reddit_historic] PRAW initialized")
            return True

        except ImportError:
            logger.error("[reddit_historic] praw package not installed")
            return False
        except Exception as e:
            logger.error(f"[reddit_historic] PRAW init error: {e}")
            return False

    async def _init_extractor(self) -> bool:
        """Initialize KalshiExtractor."""
        try:
            from ..nlp.kalshi_extractor import KalshiExtractor

            self._extractor = KalshiExtractor()
            logger.info("[reddit_historic] KalshiExtractor initialized")
            return True

        except ImportError as e:
            logger.error(f"[reddit_historic] KalshiExtractor import failed: {e}")
            return False
        except Exception as e:
            logger.error(f"[reddit_historic] KalshiExtractor init error: {e}")
            return False

    async def _init_supabase(self) -> bool:
        """Initialize Supabase async client."""
        try:
            import os
            from supabase import acreate_client, AsyncClient

            url = os.getenv("SUPABASE_URL")
            key = os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_ANON_KEY")

            if not url or not key:
                logger.warning("[reddit_historic] SUPABASE_URL and SUPABASE_KEY required")
                return False

            self._supabase: AsyncClient = await acreate_client(url, key)
            logger.info("[reddit_historic] Supabase async client initialized")
            return True

        except ImportError:
            logger.error("[reddit_historic] supabase package not installed")
            return False
        except Exception as e:
            logger.error(f"[reddit_historic] Supabase init error: {e}")
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
            logger.info(f"[reddit_historic] Loaded {len(configs)} active event configs")

        except Exception as e:
            logger.error(f"[reddit_historic] Event config refresh failed: {e}")

    # === Public API ===

    async def run_digest(self) -> Dict[str, Any]:
        """
        Run the daily digest. Respects cooldown, returns cached or fresh.

        Returns:
            Dict with digest summary or status message.
        """
        now = time.time()
        elapsed = now - self._last_digest_time

        if elapsed < self._config.digest_cooldown_seconds and self._cached_digest:
            remaining = self._config.digest_cooldown_seconds - elapsed
            return {
                "status": "skipped_cooldown",
                "message": f"Digest ran {elapsed/3600:.1f}h ago. Next in {remaining/3600:.1f}h.",
                "cached_digest": self._cached_digest,
            }

        return await self._run_digest()

    def get_cached_digest(self) -> Optional[Dict[str, Any]]:
        """Return the last cached digest result, or None if not available."""
        return self._cached_digest

    # === Core Pipeline ===

    async def _run_digest(self) -> Dict[str, Any]:
        """Core digest pipeline."""
        if not self._reddit:
            return {"status": "error", "message": "PRAW not available"}

        if not self._extractor:
            return {"status": "error", "message": "KalshiExtractor not available"}

        logger.info("[reddit_historic] Starting digest run...")
        start_time = time.time()

        # Refresh event configs
        await self._refresh_event_configs()

        # Fetch top posts via PRAW (synchronous, run in thread)
        try:
            posts = await self._fetch_top_posts()
        except Exception as e:
            logger.error(f"[reddit_historic] Failed to fetch posts: {e}")
            self._record_error(str(e))
            return {"status": "error", "message": f"Failed to fetch posts: {e}"}

        if not posts:
            result = {
                "status": "completed",
                "message": "No posts above min_score threshold",
                "posts_processed": 0,
                "extractions_created": 0,
            }
            self._cached_digest = result
            self._last_digest_time = time.time()
            return result

        # Process posts concurrently with semaphore bound
        sem = asyncio.Semaphore(self._config.max_concurrent_posts)

        async def _process_with_semaphore(submission):
            async with sem:
                try:
                    return await self._process_historic_post(submission)
                except Exception as e:
                    logger.warning(f"[reddit_historic] Error processing post {getattr(submission, 'id', '?')}: {e}")
                    self._posts_skipped += 1
                    return None

        results = await asyncio.gather(
            *[_process_with_semaphore(sub) for sub in posts]
        )

        processed_posts = [r for r in results if r is not None]
        total_extractions = sum(r.get("extraction_count", 0) for r in processed_posts)

        # Build digest summary
        digest = self._build_digest_summary(processed_posts, total_extractions, time.time() - start_time)

        # Cache result
        self._cached_digest = digest
        self._last_digest_time = time.time()
        self._digests_completed += 1

        # Broadcast completion
        await self._broadcast_to_frontend("reddit_historic_digest", {
            "status": "completed",
            "posts_processed": len(processed_posts),
            "extractions_created": total_extractions,
            "duration_seconds": round(time.time() - start_time, 1),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        logger.info(
            f"[reddit_historic] Digest completed: {len(processed_posts)} posts, "
            f"{total_extractions} extractions in {time.time() - start_time:.1f}s"
        )

        return digest

    async def _fetch_top_posts(self) -> list:
        """Fetch top posts from configured subreddits via PRAW (in thread)."""
        subreddit_str = "+".join(self._config.subreddits)
        limit = self._config.posts_limit
        time_filter = self._config.time_filter
        min_score = self._config.min_post_score

        def _fetch():
            subreddit = self._reddit.subreddit(subreddit_str)
            posts = []
            for submission in subreddit.top(time_filter=time_filter, limit=limit):
                score = getattr(submission, "score", 0) or 0
                if score >= min_score:
                    posts.append(submission)
            return posts

        return await asyncio.to_thread(_fetch)

    async def _process_historic_post(self, submission) -> Optional[Dict[str, Any]]:
        """Process a single historic post through KalshiExtractor."""
        post_id = submission.id
        title = submission.title
        selftext = getattr(submission, "selftext", "") or ""
        url = submission.url
        post_score = getattr(submission, "score", 0) or 0
        num_comments = getattr(submission, "num_comments", 0) or 0
        subreddit_name = submission.subreddit.display_name

        self._posts_processed += 1
        self._record_event_processed()

        # Fetch top comments
        comments_text = await self._fetch_top_comments(submission)

        # Build combined text
        body_truncated = selftext[:self._config.post_body_max_chars] if selftext else ""
        combined_text = title
        if body_truncated:
            combined_text = f"{title}\n\n{body_truncated}"
        if comments_text:
            combined_text = f"{combined_text}\n\n--- Top Comments ---\n{comments_text}"

        # Source created_at from Reddit post timestamp
        source_created_at = None
        if submission.created_utc:
            source_created_at = datetime.fromtimestamp(
                submission.created_utc, tz=timezone.utc
            ).isoformat()

        # Run KalshiExtractor
        extraction_rows = await self._extractor.extract(
            title=title,
            body=combined_text,
            subreddit=subreddit_name,
            engagement_score=post_score,
            engagement_comments=num_comments,
            event_configs=self._event_configs,
            source_type="reddit_historic",
            source_id=f"historic_{post_id}",
            source_url=url,
            source_created_at=source_created_at,
        )

        extraction_count = 0
        market_signals = 0

        if extraction_rows:
            # Insert to Supabase
            await self._insert_extractions(extraction_rows)
            extraction_count = len(extraction_rows)
            market_signals = sum(
                1 for r in extraction_rows if r.extraction_class == "market_signal"
            )
            self._extractions_total += extraction_count

        return {
            "post_id": post_id,
            "title": title,
            "subreddit": subreddit_name,
            "score": post_score,
            "num_comments": num_comments,
            "url": url,
            "extraction_count": extraction_count,
            "market_signals": market_signals,
            "source_created_at": source_created_at,
        }

    async def _fetch_top_comments(self, submission) -> str:
        """Fetch top N comments from a submission (in thread)."""
        limit = self._config.comments_per_post
        max_chars = self._config.comment_max_chars

        def _fetch():
            try:
                submission.comments.replace_more(limit=0)
                comments = []
                for comment in submission.comments[:limit]:
                    body = getattr(comment, "body", "") or ""
                    score = getattr(comment, "score", 0) or 0
                    if score < 2:
                        continue
                    truncated = body[:max_chars]
                    if len(body) > max_chars:
                        truncated += "..."
                    comments.append(f"[score:{score}] {truncated}")
                return "\n".join(comments)
            except Exception as e:
                logger.debug(f"[reddit_historic] Comment fetch error: {e}")
                return ""

        return await asyncio.to_thread(_fetch)

    # === Persistence ===

    async def _insert_extractions(self, rows: List["ExtractionRow"]) -> bool:
        """Insert extraction rows into Supabase extractions table (batch)."""
        if not self._supabase or not rows:
            return False

        data = [
            {
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
            for row in rows
        ]

        # Try batch insert first
        try:
            result = await self._supabase.table("extractions").insert(data).execute()
            inserted = len(result.data or [])
            if inserted > 0:
                logger.debug(f"[reddit_historic] Batch inserted {inserted} extractions for {rows[0].source_id}")
            return inserted > 0
        except Exception as e:
            error_str = str(e)
            if "duplicate" in error_str.lower() or "unique" in error_str.lower() or "23505" in error_str:
                # Batch failed due to duplicates — fall back to individual inserts
                return await self._insert_extractions_individually(data, rows[0].source_id)
            logger.error(f"[reddit_historic] Batch insert error: {e}")
            return False

    async def _insert_extractions_individually(self, data: list, source_id: str) -> bool:
        """Fallback: insert rows one-by-one, skipping duplicates."""
        inserted_count = 0
        duplicate_count = 0

        for item in data:
            try:
                result = await self._supabase.table("extractions").insert(item).execute()
                if result.data:
                    inserted_count += 1
            except Exception as e:
                error_str = str(e)
                if "duplicate" in error_str.lower() or "unique" in error_str.lower() or "23505" in error_str:
                    duplicate_count += 1
                else:
                    logger.error(f"[reddit_historic] Individual insert error: {e}")

        if duplicate_count > 0:
            logger.info(
                f"[reddit_historic] Inserted {inserted_count}, skipped {duplicate_count} duplicates "
                f"for source {source_id}"
            )

        return inserted_count > 0

    # === Digest Summary ===

    def _build_digest_summary(
        self,
        processed_posts: List[Dict[str, Any]],
        total_extractions: int,
        duration_seconds: float,
    ) -> Dict[str, Any]:
        """Build a structured digest summary for the deep agent."""
        # Aggregate market signals across posts
        market_signal_counts: Dict[str, int] = {}
        for post in processed_posts:
            # Each post contributes its market_signals count
            # but we don't have per-ticker breakdown here,
            # so we provide post-level summaries
            pass

        # Sort posts by score descending
        sorted_posts = sorted(processed_posts, key=lambda p: p.get("score", 0), reverse=True)

        # Top posts summary (compact for LLM consumption)
        top_posts = []
        for post in sorted_posts[:15]:
            top_posts.append({
                "title": post["title"][:150],
                "subreddit": post["subreddit"],
                "score": post["score"],
                "comments": post["num_comments"],
                "extraction_count": post["extraction_count"],
                "market_signals": post["market_signals"],
                "source_created_at": post.get("source_created_at"),
            })

        return {
            "status": "completed",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": round(duration_seconds, 1),
            "subreddits": self._config.subreddits,
            "time_filter": self._config.time_filter,
            "summary": {
                "posts_processed": len(processed_posts),
                "posts_with_signals": sum(1 for p in processed_posts if p.get("market_signals", 0) > 0),
                "total_extractions": total_extractions,
                "total_market_signals": sum(p.get("market_signals", 0) for p in processed_posts),
            },
            "top_posts": top_posts,
            "message": (
                f"Processed {len(processed_posts)} top Reddit posts from the past 24h "
                f"across r/{', r/'.join(self._config.subreddits)}. "
                f"Found {total_extractions} extractions "
                f"({sum(p.get('market_signals', 0) for p in processed_posts)} market signals). "
                f"Use get_extraction_signals() to see aggregated signal data including these historic posts."
            ),
        }
