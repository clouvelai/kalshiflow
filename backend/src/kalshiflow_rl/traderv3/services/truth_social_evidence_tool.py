"""
Truth Social Evidence Tool - Converts cached Truth Social posts into Evidence objects.

This module provides the adapter between TruthSocialCacheService and the event-first
research pipeline's Evidence format.
"""

import logging
import time
from typing import List, Optional, Dict, Any

from ..state.event_research_context import Evidence, EvidenceReliability
from .truth_social_cache import TruthSocialCacheService, TruthPost

logger = logging.getLogger("kalshiflow_rl.traderv3.services.truth_social_evidence")


class TruthSocialEvidenceTool:
    """
    Tool that queries TruthSocialCacheService and converts results to Evidence.

    This adapter maintains compatibility with the existing Evidence dataclass
    while enriching it with Truth Social-specific metadata (engagement metrics, etc.).
    """

    def __init__(self, cache_service: Optional[TruthSocialCacheService] = None):
        """
        Initialize Truth Social evidence tool.

        Args:
            cache_service: TruthSocialCacheService instance (if None, uses global singleton)
        """
        self._cache = cache_service

    def gather(
        self,
        *,
        event_title: str,
        primary_driver: str,
        queries: List[str],
        hours_back: Optional[float] = None,
        max_items: int = 25,
        context: Optional[Dict[str, Any]] = None,
    ) -> Evidence:
        """
        Gather Truth Social evidence matching the query criteria.

        Args:
            event_title: Event title for context
            primary_driver: Primary driver keyword
            queries: List of search queries/keywords
            hours_back: Optional time window (uses cache default if None)
            max_items: Maximum posts to return
            context: Optional additional context (unused for now)

        Returns:
            Evidence object with Truth Social posts converted to key_evidence and sources
        """
        if not self._cache or not self._cache.is_available():
            return Evidence(
                evidence_summary="Truth Social evidence unavailable (cache not running or following discovery failed)",
                reliability=EvidenceReliability.LOW,
                metadata={"truth_social": {"status": "unavailable"}},
            )

        # Combine queries with event context
        search_keywords = list(queries)
        if primary_driver:
            search_keywords.append(primary_driver)
        # Add event title words (split and filter short ones)
        if event_title:
            title_words = [w.strip() for w in event_title.split() if len(w.strip()) > 3]
            search_keywords.extend(title_words[:3])  # Top 3 words from title

        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for kw in search_keywords:
            kw_lower = kw.lower()
            if kw_lower not in seen:
                seen.add(kw_lower)
                unique_keywords.append(kw)

        # Query cache
        matching_posts = self._cache.query_posts(
            keywords=unique_keywords,
            hours_back=hours_back,
            max_items=max_items,
            include_trending=True,
        )

        if not matching_posts:
            return Evidence(
                evidence_summary=f"No Truth Social posts found matching: {', '.join(unique_keywords[:5])}",
                reliability=EvidenceReliability.LOW,
                metadata={
                    "truth_social": {
                        "status": "no_matches",
                        "keywords_searched": unique_keywords,
                        "queries": queries,
                    }
                },
            )

        # Convert posts to evidence
        key_evidence = []
        sources = []
        top_posts_metadata = []

        # Sort by engagement score (already sorted by cache, but ensure)
        matching_posts.sort(key=lambda p: p.engagement_score, reverse=True)

        # Take top posts and extract evidence
        for post in matching_posts[:max_items]:
            # Extract key excerpts (first 200 chars or first sentence)
            excerpt = post.content[:200].strip()
            if len(post.content) > 200:
                excerpt += "..."
            
            # Format evidence line with author and engagement
            evidence_line = f"@{post.author_handle}: {excerpt}"
            if post.engagement_score > 0:
                evidence_line += f" [likes: {post.likes}, reblogs: {post.reblogs}, replies: {post.replies}]"
            
            key_evidence.append(evidence_line)

            # Add source URL if available
            if post.url:
                sources.append(post.url)
            else:
                # Fallback: construct from handle and post_id
                sources.append(f"truthsocial.com/@{post.author_handle}/posts/{post.post_id}")

            # Collect metadata for top posts
            top_posts_metadata.append({
                "post_id": post.post_id,
                "author": post.author_handle,
                "created_at": post.created_at,
                "url": post.url,
                "likes": post.likes,
                "reblogs": post.reblogs,
                "replies": post.replies,
                "engagement_score": post.engagement_score,
                "is_verified": post.is_verified,
            })

        # Assess reliability based on source quality and engagement
        # High reliability: verified accounts, high engagement
        # Medium: verified or high engagement
        # Low: unverified, low engagement
        verified_count = sum(1 for p in matching_posts if p.is_verified)
        high_engagement_count = sum(1 for p in matching_posts if p.engagement_score >= 100)

        if verified_count >= 2 and high_engagement_count >= 1:
            reliability = EvidenceReliability.HIGH
            reliability_reasoning = (
                f"Found {len(matching_posts)} Truth Social posts from verified accounts "
                f"with high engagement (verified: {verified_count}, high engagement: {high_engagement_count})"
            )
        elif verified_count >= 1 or high_engagement_count >= 2:
            reliability = EvidenceReliability.MEDIUM
            reliability_reasoning = (
                f"Found {len(matching_posts)} Truth Social posts "
                f"(verified: {verified_count}, high engagement: {high_engagement_count})"
            )
        else:
            reliability = EvidenceReliability.LOW
            reliability_reasoning = (
                f"Found {len(matching_posts)} Truth Social posts with low engagement "
                f"or unverified sources"
            )

        # Build evidence summary
        evidence_summary = (
            f"Found {len(matching_posts)} Truth Social posts matching keywords: "
            f"{', '.join(unique_keywords[:5])}. "
            f"From {len(set(p.author_handle for p in matching_posts))} unique accounts. "
            f"Top post engagement: {matching_posts[0].engagement_score:.0f} "
            f"(@{matching_posts[0].author_handle})."
        )

        # Get trending data for metadata
        trending_tags = self._cache.get_trending_tags()[:10]
        trending_posts = self._cache.get_trending_posts(max_items=5)

        cache_stats = self._cache.get_stats()

        return Evidence(
            key_evidence=key_evidence,
            evidence_summary=evidence_summary,
            sources=sources,
            sources_checked=len(matching_posts),
            reliability=reliability,
            reliability_reasoning=reliability_reasoning,
            metadata={
                "truth_social": {
                    "status": "success",
                    "posts_found": len(matching_posts),
                    "unique_authors": len(set(p.author_handle for p in matching_posts)),
                    "top_posts": top_posts_metadata,
                    "keywords_searched": unique_keywords,
                    "queries": queries,
                    "verified_count": verified_count,
                    "high_engagement_count": high_engagement_count,
                    "trending_tags": [
                        {
                            "tag": tag.get("name") or tag.get("tag") or str(tag),
                            "count": tag.get("count") or 0,
                        }
                        for tag in trending_tags
                    ],
                    "trending_posts_sample": [
                        {
                            "author": p.author_handle,
                            "engagement": p.engagement_score,
                            "url": p.url,
                        }
                        for p in trending_posts
                    ],
                    "cache_stats": {
                        "cached_posts_count": cache_stats.get("cached_posts_count", 0),
                        "followed_handles_count": cache_stats.get("followed_handles_count", 0),
                        "last_refresh": cache_stats.get("last_refresh"),
                    },
                    "gathered_at": time.time(),
                }
            },
        )
