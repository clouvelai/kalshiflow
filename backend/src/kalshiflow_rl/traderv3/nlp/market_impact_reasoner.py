"""
Market Impact Reasoner - Generic LLM reasoning for indirect market effects.

This component analyzes news content to identify which Kalshi markets might be
affected, even when there's no direct entity → market title match.

Example: "ICE shooting in Minnesota" → government shutdown market
- No entity in the content matches the shutdown market title
- But LLM reasoning can identify the causal chain

This is the key addition to catch signals the entity-sentiment pipeline misses.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("kalshiflow_rl.traderv3.nlp.market_impact_reasoner")


@dataclass
class MarketInfo:
    """Simplified market info for reasoner input."""

    ticker: str
    title: str
    event_ticker: str = ""


# Magnitude to price impact mapping (same scale as sentiment_task.py)
MAGNITUDE_TO_IMPACT: Dict[int, int] = {
    -2: -75,
    -1: -40,
    0: 0,
    1: 40,
    2: 75,
}

# Confidence label to float mapping
CONFIDENCE_TO_FLOAT: Dict[str, float] = {
    "low": 0.5,
    "medium": 0.7,
    "high": 0.9,
}


class MarketImpactReasoner:
    """
    Reasons about which markets are affected by news content.

    Uses LLM to identify causal chains between news events and market outcomes,
    catching indirect effects that entity-sentiment mapping misses.

    Usage:
        reasoner = MarketImpactReasoner()
        results = await reasoner.analyze(
            content="ICE agent killed in Minnesota during enforcement action...",
            entities=[("ICE", "ORG"), ("Minnesota", "GPE")],
            active_markets=[MarketInfo(ticker="KXGOVSHUT-26JAN31", title="Government shutdown on Saturday?")]
        )
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_markets_in_prompt: int = 50,
        timeout: float = 15.0,
    ):
        """
        Initialize the reasoner.

        Args:
            model: OpenAI model name
            max_markets_in_prompt: Maximum markets to include in prompt (to control cost)
            timeout: API call timeout in seconds
        """
        self._model = model
        self._max_markets = max_markets_in_prompt
        self._timeout = timeout
        self._async_client = None

    def _get_async_client(self):
        """Get or create async OpenAI client."""
        if self._async_client is None:
            from openai import AsyncOpenAI

            self._async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return self._async_client

    async def analyze(
        self,
        content: str,
        entities: List[Tuple[str, str]],  # List of (entity_text, entity_type)
        active_markets: List[MarketInfo],
        exclude_tickers: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Analyze content for market impacts.

        Args:
            content: News content (Reddit post title + body)
            entities: Extracted entities from content
            active_markets: List of active Kalshi markets to consider
            exclude_tickers: Tickers to exclude (already handled by entity mapping)

        Returns:
            List of market impact results with reasoning
        """
        if not content or not active_markets:
            return []

        # Filter out already-mapped markets
        exclude_set = set(exclude_tickers or [])
        markets_to_consider = [m for m in active_markets if m.ticker not in exclude_set]

        if not markets_to_consider:
            return []

        # Limit markets in prompt
        if len(markets_to_consider) > self._max_markets:
            # Prioritize by potential relevance - simple heuristic:
            # keep markets with shorter titles (usually more general/important)
            markets_to_consider = sorted(
                markets_to_consider, key=lambda m: len(m.title)
            )[: self._max_markets]

        try:
            client = self._get_async_client()
            prompt = self._build_prompt(content, entities, markets_to_consider)

            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1000,
                    temperature=0.2,  # Low temp for more consistent reasoning
                ),
                timeout=self._timeout,
            )

            return self._parse_response(response, markets_to_consider, content)

        except asyncio.TimeoutError:
            logger.warning("[market_impact_reasoner] Analysis timed out")
            return []
        except Exception as e:
            logger.warning(f"[market_impact_reasoner] Analysis failed: {e}")
            return []

    def _build_prompt(
        self,
        content: str,
        entities: List[Tuple[str, str]],
        markets: List[MarketInfo],
    ) -> str:
        """Build the market impact reasoning prompt."""
        # Format entities
        if entities:
            entity_section = ", ".join(
                f"{text} ({etype})" for text, etype in entities[:10]
            )
        else:
            entity_section = "(none extracted)"

        # Format markets - include ticker and title
        market_lines = []
        for m in markets:
            market_lines.append(f"- {m.ticker}: {m.title}")
        market_section = "\n".join(market_lines)

        # Truncate content if too long
        max_content_len = 2000
        if len(content) > max_content_len:
            content = content[:max_content_len] + "..."

        prompt = f"""Analyze this news content to identify which prediction markets might be affected.

NEWS CONTENT:
"{content}"

ENTITIES MENTIONED: {entity_section}

ACTIVE KALSHI MARKETS:
{market_section}

TASK: Identify markets that could be affected by this news, even if the connection is INDIRECT.

For each affected market:
1. Explain the CAUSAL CHAIN (how does this news affect the market outcome?)
2. Rate the IMPACT DIRECTION on YES price: "bullish" (YES more likely) or "bearish" (NO more likely)
3. Rate MAGNITUDE: -2 (strong bearish), -1 (mild bearish), 0 (neutral), +1 (mild bullish), +2 (strong bullish)
4. Rate CONFIDENCE: "low" (speculative), "medium" (reasonable inference), "high" (clear connection)

IMPORTANT:
- Only include markets with a PLAUSIBLE causal connection
- The causal chain should be explainable in 2-3 sentences
- Do NOT speculate on weak or imaginary links
- If NO markets are clearly affected, respond with "NO_IMPACTS"

FORMAT (one per market, separated by ---):
MARKET: [ticker]
IMPACT: [bullish/bearish]
MAGNITUDE: [number from -2 to +2]
CONFIDENCE: [low/medium/high]
REASONING: [2-3 sentences explaining the causal chain]
---"""

        return prompt

    def _parse_response(
        self,
        response,
        markets: List[MarketInfo],
        content: str,
    ) -> List[Dict[str, Any]]:
        """Parse LLM response into structured results."""
        results = []

        try:
            response_text = response.choices[0].message.content.strip()

            # Check for no impacts
            if "NO_IMPACTS" in response_text or not response_text:
                return []

            # Build ticker -> market lookup
            ticker_to_market = {m.ticker: m for m in markets}

            # Split by --- separator
            sections = response_text.split("---")

            for section in sections:
                section = section.strip()
                if not section:
                    continue

                # Parse each field
                market_match = re.search(r"MARKET:\s*([^\n]+)", section)
                impact_match = re.search(r"IMPACT:\s*(\w+)", section)
                magnitude_match = re.search(r"MAGNITUDE:\s*([+-]?\d+)", section)
                confidence_match = re.search(r"CONFIDENCE:\s*(\w+)", section)
                reasoning_match = re.search(
                    r"REASONING:\s*(.+?)(?=(?:MARKET:|$))", section, re.DOTALL
                )

                if not market_match:
                    continue

                ticker = market_match.group(1).strip()

                # Validate ticker exists
                if ticker not in ticker_to_market:
                    # Try fuzzy match
                    for t in ticker_to_market:
                        if ticker.lower() in t.lower() or t.lower() in ticker.lower():
                            ticker = t
                            break
                    else:
                        logger.debug(
                            f"[market_impact_reasoner] Unknown ticker: {ticker}"
                        )
                        continue

                market = ticker_to_market[ticker]

                # Extract values with defaults
                impact_direction = (
                    impact_match.group(1).lower() if impact_match else "neutral"
                )
                if impact_direction not in ("bullish", "bearish"):
                    impact_direction = "neutral"

                try:
                    magnitude = int(magnitude_match.group(1)) if magnitude_match else 0
                    magnitude = max(-2, min(2, magnitude))
                except ValueError:
                    magnitude = 0

                confidence = (
                    confidence_match.group(1).lower() if confidence_match else "low"
                )
                if confidence not in ("low", "medium", "high"):
                    confidence = "low"

                reasoning = (
                    reasoning_match.group(1).strip()
                    if reasoning_match
                    else "No reasoning provided"
                )

                # Build result
                result = {
                    "market_ticker": ticker,
                    "market_title": market.title,
                    "event_ticker": market.event_ticker,
                    "impact_direction": impact_direction,
                    "impact_magnitude": magnitude,
                    "price_impact_score": self._compute_impact_score(
                        impact_direction, magnitude
                    ),
                    "confidence": confidence,
                    "confidence_float": CONFIDENCE_TO_FLOAT.get(confidence, 0.5),
                    "reasoning": reasoning,
                    "source_type": "market_impact_reasoning",
                    "created_at": time.time(),
                }

                results.append(result)

        except Exception as e:
            logger.warning(f"[market_impact_reasoner] Parse error: {e}")

        return results

    def _compute_impact_score(self, direction: str, magnitude: int) -> int:
        """Convert direction + magnitude to price impact score."""
        base_score = MAGNITUDE_TO_IMPACT.get(abs(magnitude), 0)
        if direction == "bearish":
            return -base_score
        return base_score


# =============================================================================
# Filtering Functions - Determine which posts warrant market impact reasoning
# =============================================================================


def should_analyze_for_market_impact(
    content: str,
    entities: List[Tuple[str, str]],
    avg_sentiment_magnitude: float = 0.0,
    entity_mapped_count: int = 0,
) -> bool:
    """
    Determine if content should be analyzed for indirect market impacts.

    Filtering is important to avoid unnecessary LLM calls. We only analyze posts
    that are likely to have broader market implications beyond direct entity mapping.

    Args:
        content: Full content text
        entities: Extracted entities
        avg_sentiment_magnitude: Average |sentiment| across entities
        entity_mapped_count: Number of entities already mapped to markets

    Returns:
        True if content should be analyzed for indirect impacts
    """
    # Skip very short content
    if len(content) < 100:
        return False

    # Trigger keywords that suggest broader market implications
    trigger_keywords = [
        # Breaking/urgent news
        "breaking",
        "urgent",
        "just in",
        "developing",
        # Crisis/conflict
        "shooting",
        "attack",
        "killed",
        "dead",
        "crisis",
        "emergency",
        # Political events
        "shutdown",
        "veto",
        "impeach",
        "resign",
        "executive order",
        "tariff",
        # Major policy
        "bill passed",
        "legislation",
        "supreme court",
        "ruling",
    ]

    content_lower = content.lower()
    has_trigger = any(kw in content_lower for kw in trigger_keywords)

    # High sentiment posts (|avg| > 50) suggest significant news
    has_high_sentiment = avg_sentiment_magnitude > 50

    # Multiple entities suggest complex news with potential broader impact
    has_many_entities = len(entities) >= 3

    # If most entities already mapped, less need for indirect reasoning
    mostly_mapped = entity_mapped_count >= len(entities) * 0.8 if entities else True

    # Analyze if: trigger keyword OR (high sentiment AND not mostly mapped)
    # OR (many entities AND not mostly mapped)
    return has_trigger or (
        (has_high_sentiment or has_many_entities) and not mostly_mapped
    )


# =============================================================================
# Convenience Function
# =============================================================================


async def analyze_market_impact(
    content: str,
    entities: List[Tuple[str, str]],
    active_markets: List[MarketInfo],
    exclude_tickers: Optional[List[str]] = None,
    model: str = "gpt-4o-mini",
) -> List[Dict[str, Any]]:
    """
    Convenience function for market impact analysis.

    Args:
        content: News content
        entities: Extracted entities
        active_markets: Markets to consider
        exclude_tickers: Tickers already handled by entity mapping
        model: OpenAI model

    Returns:
        List of market impact results
    """
    reasoner = MarketImpactReasoner(model=model)
    return await reasoner.analyze(content, entities, active_markets, exclude_tickers)
