"""
Test harness for AgenticResearchService (standalone testing).

Can run without full trading system.
"""

import asyncio
import os
import pytest
from ...services.agentic_research_service import (
    AgenticResearchService,
    ResearchAssessment,
)


@pytest.mark.asyncio
async def test_research_service_standalone():
    """Test research service in isolation."""
    # Skip if no OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set, skipping test")
    
    # Create service
    service = AgenticResearchService(
        openai_api_key=api_key,
        openai_model="gpt-4o",
        openai_temperature=0.3,
        web_search_enabled=True,
        cache_ttl_seconds=60.0,
    )
    
    await service.start()
    
    try:
        # Test research
        assessment = await service.research_market(
            market_ticker="TEST-MARKET",
            market_title="Will it rain tomorrow in San Francisco?",
            market_category="weather",
            current_price_cents=45,  # 45% probability
            hours_to_close=24.0,
        )
        
        # Research is async, wait a bit
        await asyncio.sleep(5.0)
        
        # Get result
        result = await service.get_assessment("TEST-MARKET", wait_seconds=30.0)
        
        assert result is not None, "Assessment should be available"
        assert result.market_ticker == "TEST-MARKET"
        assert 0.0 <= result.agent_probability <= 1.0
        assert 0.0 <= result.confidence <= 1.0
        assert result.recommendation in ["BUY_YES", "BUY_NO", "HOLD"]
        
        print(f"\nAssessment Results:")
        print(f"  Agent Probability: {result.agent_probability:.2f}")
        print(f"  Market Probability: {result.market_price_probability:.2f}")
        print(f"  Mispricing: {result.mispricing_magnitude:+.2f}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Recommendation: {result.recommendation}")
        print(f"  Key Facts: {result.key_facts}")
        print(f"  Sources: {result.sources}")
        print(f"  Duration: {result.research_duration_seconds:.2f}s")
        
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_research_service_caching():
    """Test that research results are cached."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set, skipping test")
    
    service = AgenticResearchService(
        openai_api_key=api_key,
        cache_ttl_seconds=60.0,
        web_search_enabled=False,  # Disable web search for faster test
    )
    
    await service.start()
    
    try:
        # First research
        await service.research_market(
            market_ticker="TEST-CACHE",
            market_title="Test question",
            market_category="test",
            current_price_cents=50,
            hours_to_close=24.0,
        )
        
        # Wait for completion
        await asyncio.sleep(5.0)
        result1 = await service.get_assessment("TEST-CACHE", wait_seconds=30.0)
        
        assert result1 is not None
        
        # Second call should return cached result (much faster)
        start_time = asyncio.get_event_loop().time()
        result2 = await service.get_assessment("TEST-CACHE", wait_seconds=0.0)
        elapsed = asyncio.get_event_loop().time() - start_time
        
        assert result2 is not None
        assert result2.market_ticker == result1.market_ticker
        assert result2.agent_probability == result1.agent_probability
        assert elapsed < 0.1  # Should be instant (cached)
        
        print(f"\nCache test: Second call took {elapsed:.3f}s (cached)")
        
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_research_service_question_extraction():
    """Test question extraction logic."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set, skipping test")
    
    service = AgenticResearchService(
        openai_api_key=api_key,
        web_search_enabled=False,
    )
    
    # Test extractable questions
    assert service._extract_research_question("Will it rain tomorrow?") is not None
    assert service._extract_research_question("Will Trump win the election?") is not None
    
    # Test non-extractable (spread markets)
    assert service._extract_research_question("Team A vs Team B - spread") is None
    
    # Test empty
    assert service._extract_research_question("") is None


if __name__ == "__main__":
    # Run standalone test
    asyncio.run(test_research_service_standalone())