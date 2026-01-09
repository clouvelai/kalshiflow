#!/usr/bin/env python3
"""
Test script for AgenticResearchService in isolation.

Usage:
    cd backend

    # Test with a real market ticker from demo API:
    uv run python src/kalshiflow_rl/traderv3/scripts/run_agentic_research_test.py INXD-25JAN08

    # Run with mock test cases (no ticker):
    uv run python src/kalshiflow_rl/traderv3/scripts/run_agentic_research_test.py
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path

# Load environment from .env.paper (in backend folder)
from dotenv import load_dotenv
env_file = Path(__file__).parents[4] / ".env.paper"  # backend/.env.paper
if env_file.exists():
    load_dotenv(env_file, override=True)
else:
    print(f"Warning: {env_file} not found")

# Add parent package to path for script execution
sys.path.insert(0, str(Path(__file__).parents[1]))
from services.agentic_research_service import AgenticResearchService


def print_assessment(assessment, verbose: bool = True):
    """Pretty print a research assessment."""
    print(f"\n{'─'*60}")
    print(f"✓ RESEARCH COMPLETE")
    print(f"{'─'*60}")
    print(f"  Agent Fair Value:   {assessment.agent_probability:.0%}")
    print(f"  Market Price:       {assessment.market_price_probability:.0%}")
    print(f"  Mispricing:         {assessment.mispricing_magnitude:+.0%}")
    print(f"  Confidence:         {assessment.confidence:.0%}")
    print(f"  Recommendation:     {assessment.recommendation}")
    print(f"  Duration:           {assessment.research_duration_seconds:.1f}s")

    if verbose:
        print(f"\n  REASONING:")
        # Word wrap the reasoning
        words = assessment.reasoning.split()
        line = "    "
        for word in words:
            if len(line) + len(word) > 70:
                print(line)
                line = "    "
            line += word + " "
        if line.strip():
            print(line)

        if assessment.key_facts:
            print(f"\n  KEY FACTS:")
            for fact in assessment.key_facts[:3]:
                print(f"    • {fact[:100]}...")


async def research_ticker(ticker: str):
    """Research a real market ticker from the demo API."""

    print(f"\n{'='*60}")
    print(f"AGENTIC RESEARCH: {ticker}")
    print(f"{'='*60}")

    # Import demo client (path already set up at module level)
    from clients.demo_client import KalshiDemoTradingClient

    # Fetch market data from demo API
    print(f"\nFetching market data from demo API...")

    async with KalshiDemoTradingClient(mode="paper") as client:
        # Get market details
        response = await client.get_market(ticker)
        market = response.get("market", {})

        if not market:
            print(f"✗ Market '{ticker}' not found")
            return

        # Extract fields
        title = market.get("title", "Unknown")
        yes_bid = market.get("yes_bid") or market.get("last_price") or 50
        close_time = market.get("close_time")
        event_ticker = market.get("event_ticker")
        status = market.get("status", "unknown")

        # Calculate hours to close
        if close_time:
            try:
                close_dt = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
                hours_to_close = (close_dt - datetime.now(close_dt.tzinfo)).total_seconds() / 3600
            except:
                hours_to_close = 24.0
        else:
            hours_to_close = 24.0

        # Get category from event (markets often have empty category)
        category = market.get("category") or ""
        if not category and event_ticker:
            try:
                event = await client.get_event(event_ticker)
                category = event.get("category", "Unknown")
            except:
                category = "Unknown"

        print(f"\n  MARKET DATA:")
        print(f"    Ticker:    {ticker}")
        print(f"    Title:     {title}")
        print(f"    Category:  {category}")
        print(f"    Status:    {status}")
        print(f"    YES Bid:   {yes_bid}c ({yes_bid}% implied)")
        print(f"    Closes:    {hours_to_close:.1f} hours")

        if status != "open":
            print(f"\n⚠ Market is not open (status: {status})")

    # Create research service
    service = AgenticResearchService(
        openai_model="gpt-4o-mini",
        openai_temperature=0.3,
        web_search_enabled=True,
    )
    await service.start()

    print(f"\nResearching with web search + LLM...")

    # Trigger research
    await service.research_market(
        market_ticker=ticker,
        market_title=title,
        market_category=category,
        current_price_cents=yes_bid,
        hours_to_close=hours_to_close,
    )

    # Wait for result
    assessment = await service.get_assessment(ticker, wait_seconds=90.0)

    if assessment:
        print_assessment(assessment, verbose=True)
    else:
        print("\n✗ Research failed or timed out")

    await service.stop()


async def test_mock_markets():
    """Test with mock market data (no API calls)."""

    service = AgenticResearchService(
        openai_model="gpt-4o-mini",
        openai_temperature=0.3,
        web_search_enabled=True,
    )
    await service.start()

    test_cases = [
        {
            "ticker": "FED-25JAN-T4.50",
            "title": "Will the Federal Reserve set interest rates at or above 4.50% at their January 2025 meeting?",
            "category": "Economics",
            "price_cents": 85,
            "hours_to_close": 48.0,
        },
        {
            "ticker": "RAIN-NYC-25JAN08",
            "title": "Will it rain in New York City on January 8, 2025?",
            "category": "Weather",
            "price_cents": 35,
            "hours_to_close": 24.0,
        },
    ]

    print("\n" + "="*60)
    print("AGENTIC RESEARCH SERVICE TEST (MOCK DATA)")
    print("="*60)

    for test in test_cases:
        print(f"\n{'─'*60}")
        print(f"MARKET: {test['ticker']}")
        print(f"TITLE: {test['title']}")
        print(f"CATEGORY: {test['category']}")
        print(f"PRICE: {test['price_cents']}c")
        print("─"*60)

        await service.research_market(
            market_ticker=test["ticker"],
            market_title=test["title"],
            market_category=test["category"],
            current_price_cents=test["price_cents"],
            hours_to_close=test["hours_to_close"],
        )

        print("\nResearching...")
        assessment = await service.get_assessment(test["ticker"], wait_seconds=60.0)

        if assessment:
            print_assessment(assessment, verbose=False)
        else:
            print("\n✗ Research failed or timed out")

    await service.stop()


if __name__ == "__main__":
    # Check for ticker argument
    if len(sys.argv) > 1:
        ticker = sys.argv[1]
        asyncio.run(research_ticker(ticker))
    else:
        print("Usage: python test_agentic_research.py <TICKER>")
        print("       python test_agentic_research.py  (for mock test)")
        print("\nRunning mock test...")
        asyncio.run(test_mock_markets())
