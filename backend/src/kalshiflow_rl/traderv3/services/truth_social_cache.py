"""
Truth Social Cache Service - Global cache for Truth Social posts and trending data.

This service maintains a long-lived in-memory cache of:
- Posts from users the authenticated account follows (auto "Trump circle")
- Trending hashtags
- Trending posts

The cache is refreshed periodically and queried by the evidence gathering pipeline.

Note: This now uses our custom TruthSocialApi instead of truthbrush, which uses
curl_cffi with chrome136 browser impersonation to bypass Cloudflare blocking.
"""

import asyncio
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any
from datetime import datetime, timedelta

logger = logging.getLogger("kalshiflow_rl.traderv3.services.truth_social_cache")


@dataclass
class TruthPost:
    """Represents a Truth Social post with metadata."""
    post_id: str
    author_handle: str
    content: str
    created_at: float  # Unix timestamp
    url: Optional[str] = None
    likes: int = 0
    reblogs: int = 0
    replies: int = 0
    is_verified: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def engagement_score(self) -> float:
        """Calculate a simple engagement score (likes + 2*reblogs + replies)."""
        return float(self.likes + 2 * self.reblogs + self.replies)

    def matches_query(self, keywords: List[str]) -> bool:
        """Check if post content matches any keyword (case-insensitive)."""
        content_lower = self.content.lower()
        return any(kw.lower() in content_lower for kw in keywords if kw.strip())


@dataclass
class TrendingData:
    """Trending hashtags and posts."""
    trending_tags: List[Dict[str, Any]] = field(default_factory=list)
    trending_posts: List[TruthPost] = field(default_factory=list)
    refreshed_at: float = field(default_factory=time.time)


class TruthSocialCacheService:
    """
    Global cache service for Truth Social content.

    Maintains in-memory cache with TTL-based expiration.
    Refreshes following list, posts, and trending data periodically.
    """

    def __init__(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        cache_refresh_seconds: float = 300.0,  # 5 minutes default
        trending_refresh_seconds: float = 600.0,  # 10 minutes default
        hours_back: float = 24.0,
        max_posts_per_user: int = 50,
    ):
        """
        Initialize Truth Social cache service.

        Args:
            username: Truth Social username (if None, loads from TRUTHSOCIAL_USERNAME env)
            password: Truth Social password (if None, loads from TRUTHSOCIAL_PASSWORD env)
            cache_refresh_seconds: How often to refresh posts from followed users
            trending_refresh_seconds: How often to refresh trending data
            hours_back: How far back to fetch posts (default 24h)
            max_posts_per_user: Maximum posts to fetch per followed user
        """
        self._username = username or os.getenv("TRUTHSOCIAL_USERNAME")
        self._password = password or os.getenv("TRUTHSOCIAL_PASSWORD")
        self._cache_refresh_seconds = cache_refresh_seconds
        self._trending_refresh_seconds = trending_refresh_seconds
        self._hours_back = hours_back
        self._max_posts_per_user = max_posts_per_user

        # Cache state
        self._posts: Dict[str, TruthPost] = {}  # post_id -> TruthPost
        self._posts_by_user: Dict[str, List[str]] = {}  # handle -> [post_ids]
        self._followed_handles: Set[str] = set()
        self._trending: TrendingData = TrendingData()

        # Background tasks
        self._running = False
        self._refresh_task: Optional[asyncio.Task] = None
        self._trending_task: Optional[asyncio.Task] = None
        self._last_refresh: Optional[float] = None
        self._last_trending_refresh: Optional[float] = None

        # Stats
        self._refresh_count = 0
        self._refresh_errors = 0
        self._following_discovery_failed = False

        # Custom Truth Social API client (lazy-loaded)
        # Uses curl_cffi with chrome136 to bypass Cloudflare (truthbrush uses chrome123 which is blocked)
        self._api = None

    @property
    def hours_back(self) -> float:
        """Get the configured hours_back window for post fetching."""
        return self._hours_back

    def _get_api(self):
        """Lazy-load custom Truth Social API client."""
        if self._api is None:
            try:
                # Use our custom API that bypasses Cloudflare
                from kalshiflow_rl.traderv3.services.truth_social_api import TruthSocialApi
                self._api = TruthSocialApi(username=self._username, password=self._password)
                logger.info(f"TruthSocialCacheService: API initialized for user {self._username}")
            except ImportError as e:
                logger.error(f"Truth Social API not available: {e}")
                raise RuntimeError("Truth Social API not available")
            except Exception as e:
                logger.error(f"Failed to initialize Truth Social API: {e}")
                raise
        return self._api

    async def start(self) -> bool:
        """
        Start the cache service (discover following + start refresh loops).

        Returns:
            True if started successfully, False if following discovery failed (hard-disable)
        """
        if self._running:
            logger.warning("TruthSocialCacheService already running")
            return True

        if not self._username or not self._password:
            logger.warning(
                "TruthSocialCacheService: Missing credentials - cache disabled. "
                "Set TRUTHSOCIAL_USERNAME and TRUTHSOCIAL_PASSWORD env vars."
            )
            return False

        logger.info("Starting TruthSocialCacheService...")

        # Try to discover following list (hard requirement)
        try:
            followed = await self._discover_following()
            if not followed:
                logger.error(
                    "TruthSocialCacheService: Following discovery failed - "
                    "cannot determine 'Trump circle'. Truth Social evidence DISABLED."
                )
                self._following_discovery_failed = True
                return False

            self._followed_handles = set(followed)
            logger.info(f"TruthSocialCacheService: Discovered {len(self._followed_handles)} followed users")
        except Exception as e:
            logger.error(
                f"TruthSocialCacheService: Following discovery error: {e} - "
                "Truth Social evidence DISABLED."
            )
            self._following_discovery_failed = True
            return False

        # Initial refresh
        await self._refresh_posts()
        await self._refresh_trending()

        # Start background refresh loops
        self._running = True
        self._refresh_task = asyncio.create_task(self._refresh_loop())
        self._trending_task = asyncio.create_task(self._trending_refresh_loop())

        logger.info("TruthSocialCacheService started successfully")
        return True

    async def stop(self) -> None:
        """Stop the cache service and cancel background tasks."""
        if not self._running:
            return

        logger.info("Stopping TruthSocialCacheService...")
        self._running = False

        if self._refresh_task:
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass

        if self._trending_task:
            self._trending_task.cancel()
            try:
                await self._trending_task
            except asyncio.CancelledError:
                pass

        logger.info("TruthSocialCacheService stopped")

    async def _discover_following(self) -> List[str]:
        """
        Discover list of users the authenticated account follows.

        Returns:
            List of user handles (without @ prefix)
        """
        try:
            api = self._get_api()
            # truthbrush API: user_following(username) returns a generator
            # We need to get the authenticated user's following list
            result = api.user_following(self._username)
            
            # Convert generator to list
            users = list(result) if result else []
            
            if not users:
                logger.warning("TruthSocialCacheService: user_following returned empty result")
                return []

            # Extract handles (may be dicts with 'username' or 'handle' key, or just strings)
            handles = []
            for user in users:
                if isinstance(user, str):
                    handles.append(user.lstrip("@"))
                elif isinstance(user, dict):
                    handle = user.get("username") or user.get("handle") or user.get("screen_name")
                    if handle:
                        handles.append(str(handle).lstrip("@"))

            logger.info(f"TruthSocialCacheService: Discovered {len(handles)} followed users: {handles[:5]}...")
            return handles

        except Exception as e:
            logger.error(f"TruthSocialCacheService: Following discovery failed: {e}", exc_info=True)
            raise

    async def _refresh_posts(self) -> None:
        """Refresh posts from all followed users."""
        if not self._followed_handles:
            return

        logger.debug(f"TruthSocialCacheService: Refreshing posts from {len(self._followed_handles)} users...")

        cutoff_time = time.time() - (self._hours_back * 3600)
        cutoff_datetime = datetime.fromtimestamp(cutoff_time)
        new_posts = {}
        posts_by_user = {}

        try:
            api = self._get_api()

            for handle in self._followed_handles:
                try:
                    # Our custom API: pull_statuses yields status dicts
                    # Run in executor since API is synchronous
                    loop = asyncio.get_event_loop()

                    def fetch_statuses():
                        return list(api.pull_statuses(
                            handle,
                            limit=self._max_posts_per_user,
                            created_after=cutoff_datetime
                        ))

                    statuses = await loop.run_in_executor(None, fetch_statuses)

                    if not statuses:
                        continue

                    user_posts = []
                    for status in statuses:
                        try:
                            post = self._parse_status(status)
                            if post and post.created_at >= cutoff_time:
                                new_posts[post.post_id] = post
                                user_posts.append(post.post_id)
                        except Exception as e:
                            logger.debug(f"Error parsing status from {handle}: {e}")
                            continue

                    if user_posts:
                        posts_by_user[handle] = user_posts

                except Exception as e:
                    logger.warning(f"Error fetching posts from {handle}: {e}")
                    continue

            # Update cache references (note: individual assignments are not atomic,
            # but this is acceptable since reads tolerate brief inconsistency)
            self._posts = new_posts
            self._posts_by_user = posts_by_user
            self._last_refresh = time.time()
            self._refresh_count += 1

            logger.info(
                f"TruthSocialCacheService: Refreshed {len(new_posts)} posts "
                f"from {len(posts_by_user)} users"
            )

        except Exception as e:
            logger.error(f"TruthSocialCacheService: Post refresh failed: {e}", exc_info=True)
            self._refresh_errors += 1

    def _parse_status(self, status: Dict[str, Any]) -> Optional[TruthPost]:
        """Parse a status dict from Truth Social API into TruthPost."""
        try:
            # Extract fields
            post_id = str(status.get("id") or status.get("status_id") or "")
            if not post_id:
                return None

            content = str(status.get("content") or status.get("text") or status.get("full_text") or "")
            # Strip HTML tags from content
            content = re.sub(r'<[^>]+>', '', content)
            
            # Parse created_at (may be ISO string or timestamp)
            created_at = status.get("created_at")
            if isinstance(created_at, str):
                try:
                    dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    created_at_ts = dt.timestamp()
                except Exception:
                    created_at_ts = time.time()
            elif isinstance(created_at, (int, float)):
                created_at_ts = float(created_at)
            else:
                created_at_ts = time.time()

            author = str(status.get("user", {}).get("username") or status.get("username") or "")

            # Engagement metrics
            likes = int(status.get("favourites_count") or status.get("likes_count") or status.get("likes") or 0)
            reblogs = int(status.get("reblogs_count") or status.get("reposts_count") or status.get("reblogs") or 0)
            replies = int(status.get("replies_count") or status.get("reply_count") or status.get("replies") or 0)

            # URL construction (if we have post ID and author)
            url = None
            if post_id and author:
                url = f"https://truthsocial.com/@{(author.lstrip('@'))}/posts/{post_id}"

            # Verified status
            user_obj = status.get("user", {})
            is_verified = bool(user_obj.get("verified") or user_obj.get("is_verified") or False)

            return TruthPost(
                post_id=post_id,
                author_handle=author,
                content=content,
                created_at=created_at_ts,
                url=url,
                likes=likes,
                reblogs=reblogs,
                replies=replies,
                is_verified=is_verified,
                metadata={
                    "raw_status": status,  # Keep raw for debugging
                }
            )

        except Exception as e:
            logger.debug(f"Error parsing status: {e}")
            return None

    async def _refresh_trending(self) -> None:
        """Refresh trending hashtags and posts."""
        try:
            api = self._get_api()
            loop = asyncio.get_event_loop()

            # Fetch trending tags
            trending_tags = []
            try:
                tags_result = await loop.run_in_executor(None, api.tags)
                if isinstance(tags_result, list):
                    trending_tags = tags_result[:20]  # Top 20
                elif isinstance(tags_result, dict):
                    trending_tags = tags_result.get("tags", [])[:20]
            except Exception as e:
                logger.debug(f"Error fetching trending tags: {e}")

            # Fetch trending posts
            trending_posts = []
            try:
                trends_result = await loop.run_in_executor(None, api.trending)
                if isinstance(trends_result, list):
                    for trend_item in trends_result[:30]:  # Top 30
                        post = self._parse_status(trend_item)
                        if post:
                            trending_posts.append(post)
                elif isinstance(trends_result, dict):
                    status_list = trends_result.get("statuses", trends_result.get("posts", []))
                    for status in status_list[:30]:
                        post = self._parse_status(status)
                        if post:
                            trending_posts.append(post)
            except Exception as e:
                logger.debug(f"Error fetching trending posts: {e}")

            self._trending = TrendingData(
                trending_tags=trending_tags,
                trending_posts=trending_posts,
                refreshed_at=time.time()
            )
            self._last_trending_refresh = time.time()

            logger.debug(
                f"TruthSocialCacheService: Refreshed {len(trending_tags)} trending tags "
                f"and {len(trending_posts)} trending posts"
            )

        except Exception as e:
            logger.warning(f"TruthSocialCacheService: Trending refresh failed: {e}")

    async def _refresh_loop(self) -> None:
        """Background loop to refresh posts periodically."""
        while self._running:
            try:
                await asyncio.sleep(self._cache_refresh_seconds)
                if self._running:
                    await self._refresh_posts()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"TruthSocialCacheService: Refresh loop error: {e}")

    async def _trending_refresh_loop(self) -> None:
        """Background loop to refresh trending data periodically."""
        while self._running:
            try:
                await asyncio.sleep(self._trending_refresh_seconds)
                if self._running:
                    await self._refresh_trending()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"TruthSocialCacheService: Trending refresh loop error: {e}")

    def query_posts(
        self,
        keywords: List[str],
        hours_back: Optional[float] = None,
        max_items: int = 25,
        include_trending: bool = True,
    ) -> List[TruthPost]:
        """
        Query cached posts matching keywords.

        Args:
            keywords: List of keywords to match (case-insensitive)
            hours_back: Optional time window (defaults to service's hours_back)
            max_items: Maximum number of posts to return
            include_trending: Whether to include trending posts in results

        Returns:
            List of TruthPost objects matching keywords, sorted by engagement score (desc)
        """
        if not keywords:
            return []

        cutoff_time = time.time() - ((hours_back or self._hours_back) * 3600)
        matches = []

        # Search cached posts
        for post in self._posts.values():
            if post.created_at < cutoff_time:
                continue
            if post.matches_query(keywords):
                matches.append(post)

        # Include trending posts if requested
        if include_trending:
            for post in self._trending.trending_posts:
                if post.created_at >= cutoff_time and post.matches_query(keywords):
                    # Avoid duplicates
                    if post.post_id not in {p.post_id for p in matches}:
                        matches.append(post)

        # Sort by engagement score (descending) and limit
        matches.sort(key=lambda p: p.engagement_score, reverse=True)
        return matches[:max_items]

    def get_trending_tags(self) -> List[Dict[str, Any]]:
        """Get cached trending hashtags."""
        return self._trending.trending_tags

    def get_trending_posts(self, max_items: int = 10) -> List[TruthPost]:
        """Get cached trending posts."""
        return self._trending.trending_posts[:max_items]

    def get_stats(self) -> Dict[str, Any]:
        """Get cache service statistics."""
        return {
            "running": self._running,
            "followed_handles_count": len(self._followed_handles),
            "cached_posts_count": len(self._posts),
            "trending_tags_count": len(self._trending.trending_tags),
            "trending_posts_count": len(self._trending.trending_posts),
            "refresh_count": self._refresh_count,
            "refresh_errors": self._refresh_errors,
            "last_refresh": self._last_refresh,
            "last_trending_refresh": self._last_trending_refresh,
            "following_discovery_failed": self._following_discovery_failed,
        }

    def is_available(self) -> bool:
        """Check if cache is available and operational."""
        return self._running and not self._following_discovery_failed

    def get_health_details(self) -> Dict[str, Any]:
        """
        Get health details for status reporting (matches syncer pattern).

        Returns comprehensive health info including:
        - Running/available status
        - Cache statistics (posts, authors)
        - Refresh timing and errors
        - Overall healthy flag
        """
        now = time.time()
        return {
            "running": self._running,
            "available": self.is_available(),
            "following_discovery_failed": self._following_discovery_failed,
            "followed_handles_count": len(self._followed_handles),
            "followed_handles_sample": list(self._followed_handles)[:5],  # First 5 for debugging
            "cached_posts_count": len(self._posts),
            "trending_tags_count": len(self._trending.trending_tags),
            "trending_posts_count": len(self._trending.trending_posts),
            "last_refresh": self._last_refresh,
            "last_refresh_age_seconds": round(now - self._last_refresh, 1) if self._last_refresh else None,
            "last_trending_refresh": self._last_trending_refresh,
            "last_trending_refresh_age_seconds": round(now - self._last_trending_refresh, 1) if self._last_trending_refresh else None,
            "refresh_count": self._refresh_count,
            "refresh_errors": self._refresh_errors,
            "cache_refresh_seconds": self._cache_refresh_seconds,
            "trending_refresh_seconds": self._trending_refresh_seconds,
            "hours_back": self._hours_back,
            "healthy": self.is_available() and self._refresh_count > 0 and self._refresh_errors == 0,
        }


# Global singleton instance
_global_cache: Optional[TruthSocialCacheService] = None


def get_truth_social_cache() -> Optional[TruthSocialCacheService]:
    """Get the global Truth Social cache instance."""
    return _global_cache


async def initialize_truth_social_cache() -> Optional[TruthSocialCacheService]:
    """
    Initialize and start the global Truth Social cache service.

    Returns:
        TruthSocialCacheService instance if started successfully, None if disabled/failed
    """
    global _global_cache

    if _global_cache is not None:
        return _global_cache

    # Check if enabled
    truth_flag = (os.getenv("TRUTHSOCIAL_EVIDENCE_ENABLED", "auto") or "auto").strip().lower()
    has_creds = bool(os.getenv("TRUTHSOCIAL_USERNAME")) and bool(os.getenv("TRUTHSOCIAL_PASSWORD"))

    if truth_flag in ("0", "false", "no", "n", "off"):
        logger.info("TruthSocialCacheService: Disabled via TRUTHSOCIAL_EVIDENCE_ENABLED")
        return None

    if truth_flag == "auto" and not has_creds:
        logger.info("TruthSocialCacheService: Auto-disabled (no credentials)")
        return None

    # Initialize service
    cache_refresh = float(os.getenv("TRUTHSOCIAL_CACHE_REFRESH_SECONDS", "300"))
    trending_refresh = float(os.getenv("TRUTHSOCIAL_TRENDING_REFRESH_SECONDS", "600"))
    hours_back = float(os.getenv("TRUTHSOCIAL_HOURS_BACK", "24"))

    _global_cache = TruthSocialCacheService(
        cache_refresh_seconds=cache_refresh,
        trending_refresh_seconds=trending_refresh,
        hours_back=hours_back,
    )

    # Start service (this will try following discovery)
    started = await _global_cache.start()
    if not started:
        _global_cache = None
        return None

    return _global_cache
