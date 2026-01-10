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

    # Mock cache (availability gate only; tool uses router for signals)
    mock_cache = Mock(spec=TruthSocialCacheService)
    mock_cache.is_available.return_value = True
    mock_cache.hours_back = 24.0

    # Mock router that returns no signals
    mock_router = Mock()
    mock_router.route.return_value = Mock(top_signals=[], stats={"posts_seen": 0, "unique_authors": 0, "verified_count": 0, "window_hours": 24.0})

    tool = TruthSocialEvidenceTool(cache_service=mock_cache, router=mock_router)

    evidence = tool.gather(
        event_title="Test Event",
        primary_driver="nonexistent keyword",
        queries=["xyzabc123"],
    )

    assert evidence.reliability == EvidenceReliability.LOW
    assert "no matches" in evidence.evidence_summary.lower() or "no truth social" in evidence.evidence_summary.lower()
    assert evidence.metadata.get("truth_social", {}).get("status") in ("no_matches", "unavailable")
    # New contract: top_signals only, no raw post cards
    truth_meta = evidence.metadata.get("truth_social", {})
    assert "top_signals" not in truth_meta or truth_meta.get("top_signals") in (None, [])
    assert "top_posts" not in truth_meta


@pytest.mark.asyncio
async def test_truth_social_evidence_tool_metadata_includes_engagement():
    """Evidence tool should include engagement_score in top_signals and not expose raw post content."""
    from kalshiflow_rl.traderv3.services.truth_social_evidence_tool import TruthSocialEvidenceTool
    from kalshiflow_rl.traderv3.services.truth_social_cache import TruthSocialCacheService
    from kalshiflow_rl.traderv3.services.truth_social_signal_store import DistilledTruthSignal

    # Cache gate
    mock_cache = Mock(spec=TruthSocialCacheService)
    mock_cache.is_available.return_value = True
    mock_cache.hours_back = 24.0

    # Router returns one distilled signal
    signal = DistilledTruthSignal(
        signal_id="ts:post1:0",
        created_at=time.time() - 3600,
        author_handle="testuser",
        is_verified=True,
        engagement_score=150.0,
        claim="Trump hinted at a new policy announcement soon.",
        claim_type="intent",
        entities=["Trump"],
        linked_roles=None,
        confidence=0.7,
        reasoning_short="Direct statement; high engagement; treat as narrative signal.",
        source_url="https://truthsocial.com/@testuser/posts/post1",
    )
    mock_router = Mock()
    mock_router.route.return_value = Mock(
        top_signals=[signal],
        stats={"posts_seen": 5, "unique_authors": 2, "verified_count": 1, "window_hours": 24.0},
    )

    tool = TruthSocialEvidenceTool(cache_service=mock_cache, router=mock_router)

    evidence = tool.gather(
        event_title="Test Event",
        primary_driver="Trump",
        queries=["Trump", "politics"],
    )

    # Check metadata includes distilled signals with engagement_score
    truth_meta = evidence.metadata.get("truth_social", {})
    assert truth_meta.get("status") == "success"
    assert truth_meta.get("signals_emitted", 0) > 0

    top_signals = truth_meta.get("top_signals", [])
    assert len(top_signals) > 0
    assert "engagement_score" in top_signals[0]
    assert "claim" in top_signals[0]
    assert "content" not in top_signals[0]
    assert "top_posts" not in truth_meta


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
