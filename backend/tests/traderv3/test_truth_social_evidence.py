"""
Tests for Truth Social evidence tool and cache service.

These tests ensure graceful degradation when cache is unavailable,
following discovery fails, or credentials are missing.
"""

import os
import pytest
import time
from unittest.mock import Mock, AsyncMock, patch


@pytest.mark.asyncio
async def test_truth_social_cache_disabled_without_creds(monkeypatch):
    """Cache should not initialize if credentials are missing."""
    monkeypatch.delenv("TRUTHSOCIAL_USERNAME", raising=False)
    monkeypatch.delenv("TRUTHSOCIAL_PASSWORD", raising=False)
    monkeypatch.setenv("TRUTHSOCIAL_EVIDENCE_ENABLED", "auto")

    from kalshiflow_rl.traderv3.services.truth_social_cache import initialize_truth_social_cache

    cache = await initialize_truth_social_cache()
    assert cache is None


@pytest.mark.asyncio
async def test_truth_social_cache_hard_disable_on_following_failure(monkeypatch):
    """If following discovery fails, cache should not start (hard-disable per user preference)."""
    monkeypatch.setenv("TRUTHSOCIAL_USERNAME", "test_user")
    monkeypatch.setenv("TRUTHSOCIAL_PASSWORD", "test_pass")
    monkeypatch.setenv("TRUTHSOCIAL_EVIDENCE_ENABLED", "true")

    from kalshiflow_rl.traderv3.services.truth_social_cache import TruthSocialCacheService

    cache = TruthSocialCacheService(
        username="test_user",
        password="test_pass",
    )

    # Mock following discovery to fail
    async def mock_discover():
        raise Exception("Following discovery failed")

    cache._discover_following = mock_discover

    started = await cache.start()
    assert started is False
    assert cache._following_discovery_failed is True
    assert cache.is_available() is False


@pytest.mark.asyncio
async def test_truth_social_evidence_tool_returns_low_when_cache_unavailable():
    """Evidence tool should return LOW reliability evidence when cache is unavailable."""
    from kalshiflow_rl.traderv3.services.truth_social_evidence_tool import TruthSocialEvidenceTool
    from kalshiflow_rl.traderv3.state.event_research_context import EvidenceReliability

    # Tool with no cache
    tool = TruthSocialEvidenceTool(cache_service=None)

    evidence = tool.gather(
        event_title="Test Event",
        primary_driver="test driver",
        queries=["test"],
    )

    assert evidence.reliability == EvidenceReliability.LOW
    assert "unavailable" in evidence.evidence_summary.lower()
    assert evidence.metadata.get("truth_social", {}).get("status") == "unavailable"


@pytest.mark.asyncio
async def test_truth_social_evidence_tool_query_empty_results():
    """Evidence tool should handle empty query results gracefully."""
    from kalshiflow_rl.traderv3.services.truth_social_evidence_tool import TruthSocialEvidenceTool
    from kalshiflow_rl.traderv3.services.truth_social_cache import TruthSocialCacheService
    from kalshiflow_rl.traderv3.state.event_research_context import EvidenceReliability

    # Mock cache that returns empty results
    mock_cache = Mock(spec=TruthSocialCacheService)
    mock_cache.is_available.return_value = True
    mock_cache.query_posts.return_value = []  # No matching posts
    mock_cache.get_trending_tags.return_value = []
    mock_cache.get_trending_posts.return_value = []
    mock_cache.get_stats.return_value = {
        "cached_posts_count": 0,
        "followed_handles_count": 0,
    }
    mock_cache._hours_back = 24.0

    tool = TruthSocialEvidenceTool(cache_service=mock_cache)

    evidence = tool.gather(
        event_title="Test Event",
        primary_driver="nonexistent keyword",
        queries=["xyzabc123"],
    )

    assert evidence.reliability == EvidenceReliability.LOW
    assert "no matches" in evidence.evidence_summary.lower() or "no truth social" in evidence.evidence_summary.lower()
    assert evidence.metadata.get("truth_social", {}).get("status") in ("no_matches", "unavailable")


@pytest.mark.asyncio
async def test_truth_social_evidence_tool_metadata_includes_engagement():
    """Evidence tool should include engagement metrics (likes/reblogs/replies) in metadata."""
    from kalshiflow_rl.traderv3.services.truth_social_evidence_tool import TruthSocialEvidenceTool
    from kalshiflow_rl.traderv3.services.truth_social_cache import TruthSocialCacheService, TruthPost

    # Create mock posts with engagement data
    mock_posts = [
        TruthPost(
            post_id="post1",
            author_handle="testuser",
            content="Test post about Trump and politics",
            created_at=time.time() - 3600,  # 1 hour ago
            url="https://truthsocial.com/@testuser/posts/post1",
            likes=100,
            reblogs=20,
            replies=10,
            is_verified=True,
        )
    ]

    mock_cache = Mock(spec=TruthSocialCacheService)
    mock_cache.is_available.return_value = True
    mock_cache.query_posts.return_value = mock_posts
    mock_cache.get_trending_tags.return_value = []
    mock_cache.get_trending_posts.return_value = []
    mock_cache.get_stats.return_value = {
        "cached_posts_count": 1,
        "followed_handles_count": 5,
    }
    mock_cache._hours_back = 24.0

    tool = TruthSocialEvidenceTool(cache_service=mock_cache)

    evidence = tool.gather(
        event_title="Test Event",
        primary_driver="Trump",
        queries=["Trump", "politics"],
    )

    # Check metadata includes engagement data
    truth_meta = evidence.metadata.get("truth_social", {})
    assert truth_meta.get("status") == "success"
    assert truth_meta.get("posts_found", 0) > 0

    top_posts = truth_meta.get("top_posts", [])
    assert len(top_posts) > 0
    assert "likes" in top_posts[0]
    assert "reblogs" in top_posts[0]
    assert "replies" in top_posts[0]
    assert "engagement_score" in top_posts[0]


@pytest.mark.asyncio
async def test_event_research_service_truth_evidence_integration(monkeypatch):
    """EventResearchService should handle Truth Social evidence tool gracefully."""
    monkeypatch.setenv("OPENAI_API_KEY", "test_key")
    monkeypatch.setenv("TRUTHSOCIAL_EVIDENCE_ENABLED", "auto")
    monkeypatch.delenv("TRUTHSOCIAL_USERNAME", raising=False)
    monkeypatch.delenv("TRUTHSOCIAL_PASSWORD", raising=False)

    from kalshiflow_rl.traderv3.services.event_research_service import EventResearchService
    from kalshiflow_rl.traderv3.state.event_research_context import KeyDriverAnalysis

    svc = EventResearchService(
        trading_client=None,
        openai_api_key="test_key",
        web_search_enabled=False,
    )

    # Should not raise even if cache is unavailable
    evidence = await svc._gather_truth_social_evidence(
        driver_analysis=KeyDriverAnalysis(
            primary_driver="test driver",
            primary_driver_reasoning="test",
            causal_chain="test",
        ),
        event_title="Test Event",
        semantic_frame=None,
    )

    assert evidence is not None
    # Should be LOW reliability when cache unavailable
    assert evidence.reliability.value == "low"


@pytest.mark.asyncio
async def test_truth_social_cache_stop_cleanup():
    """Cache service should stop background tasks cleanly."""
    from kalshiflow_rl.traderv3.services.truth_social_cache import TruthSocialCacheService

    cache = TruthSocialCacheService(
        username="test_user",
        password="test_pass",
    )

    # Mock following discovery to succeed (return empty list for test)
    async def mock_discover():
        return []

    cache._discover_following = mock_discover
    cache._refresh_posts = AsyncMock()
    cache._refresh_trending = AsyncMock()

    # Start cache (will fail because following discovery returns empty, but that's fine for this test)
    started = await cache.start()
    # Just test that stop() doesn't raise
    await cache.stop()

    assert cache._running is False
