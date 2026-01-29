"""
Batched Sentiment Analysis Task.

Provides efficient sentiment scoring for multiple entities in a single
LLM call. This reduces API costs and latency compared to per-entity calls.

Features:
- Single LLM call scores all entities in a document
- Context-aware scoring considers entity position and surrounding text
- Async support for non-blocking processing
- Fallback to neutral sentiment on errors
- 5-point scale (-2 to +2) for reliable LLM classification
- LLM-provided confidence instead of hardcoded values
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("kalshiflow_rl.traderv3.nlp.sentiment_task")


# =============================================================================
# 5-Point Sentiment Scale Mappings
# =============================================================================

# Map categorical sentiment (-2 to +2) to price impact values
# These values are chosen to align with trading decision thresholds:
# - |impact| > 50: IMMEDIATE TRADE (strong signals)
# - |impact| > 30: Regular trade (moderate signals)
SENTIMENT_TO_IMPACT: Dict[int, int] = {
    -2: -75,  # Strongly negative (scandal, catastrophic)
    -1: -40,  # Mildly negative (criticism, setback)
    0: 0,     # Neutral
    1: 40,    # Mildly positive (endorsement, good news)
    2: 75,    # Strongly positive (major win, vindication)
}

# Map LLM confidence labels to float values
CONFIDENCE_TO_FLOAT: Dict[str, float] = {
    "low": 0.5,
    "medium": 0.7,
    "high": 0.9,
}


@dataclass
class EntitySentimentResult:
    """Result of sentiment analysis for a single entity."""

    entity_text: str
    entity_type: str  # MARKET_ENTITY, PERSON, ORG, etc.
    sentiment_category: int  # -2 to +2 (5-point scale)
    sentiment_score: int  # Mapped value: -75, -40, 0, 40, 75
    confidence: float  # From LLM: 0.5, 0.7, 0.9
    context_snippet: str  # Relevant surrounding text


class BatchedSentimentTask:
    """
    Batched sentiment analysis using LLM.

    Analyzes sentiment toward multiple entities in a single API call,
    which is more efficient than per-entity calls.

    Usage:
        task = BatchedSentimentTask()
        results = await task.analyze_async(
            text="Pam Bondi faces new scandal amid confirmation hearing.",
            entities=[("Pam Bondi", "MARKET_ENTITY")]
        )
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        context_window: int = 200,
        timeout: float = 10.0,
    ):
        """
        Initialize the sentiment task.

        Args:
            model: OpenAI model name for sentiment analysis
            context_window: Characters of context around entity mentions
            timeout: API call timeout in seconds
        """
        self._model = model
        self._context_window = context_window
        self._timeout = timeout
        self._sync_client = None
        self._async_client = None

    def _get_sync_client(self):
        """Get or create synchronous OpenAI client."""
        if self._sync_client is None:
            from openai import OpenAI
            self._sync_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return self._sync_client

    def _get_async_client(self):
        """Get or create async OpenAI client."""
        if self._async_client is None:
            from openai import AsyncOpenAI
            self._async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return self._async_client

    def analyze(
        self,
        text: str,
        entities: List[Tuple[str, str]],  # List of (entity_text, entity_type)
    ) -> List[EntitySentimentResult]:
        """
        Synchronously analyze sentiment for entities.

        Args:
            text: The full text containing entity mentions
            entities: List of (entity_text, entity_type) tuples

        Returns:
            List of EntitySentimentResult objects
        """
        if not entities:
            return []

        try:
            client = self._get_sync_client()
            prompt = self._build_prompt(text, entities)

            response = client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0,
            )

            return self._parse_response(response, entities, text)

        except Exception as e:
            logger.warning(f"[sentiment] Sync analysis failed: {e}")
            return self._fallback_results(entities, text)

    async def analyze_async(
        self,
        text: str,
        entities: List[Tuple[str, str]],  # List of (entity_text, entity_type)
    ) -> List[EntitySentimentResult]:
        """
        Asynchronously analyze sentiment for entities.

        Args:
            text: The full text containing entity mentions
            entities: List of (entity_text, entity_type) tuples

        Returns:
            List of EntitySentimentResult objects
        """
        if not entities:
            return []

        try:
            client = self._get_async_client()
            prompt = self._build_prompt(text, entities)

            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=100,
                    temperature=0,
                ),
                timeout=self._timeout,
            )

            return self._parse_response(response, entities, text)

        except asyncio.TimeoutError:
            logger.warning(f"[sentiment] Async analysis timed out")
            return self._fallback_results(entities, text)
        except Exception as e:
            logger.warning(f"[sentiment] Async analysis failed: {e}")
            return self._fallback_results(entities, text)

    def _build_prompt(
        self,
        text: str,
        entities: List[Tuple[str, str]],
    ) -> str:
        """
        Build the sentiment analysis prompt using a 5-point scale.

        The 5-point scale (-2 to +2) is more reliable for LLMs than
        fine-grained numeric scales. Each category maps directly to
        a trading decision tier.

        Args:
            text: The full text
            entities: List of (entity_text, entity_type) tuples

        Returns:
            Formatted prompt string
        """
        # Build entity list with types
        entity_list = []
        for i, (entity_text, entity_type) in enumerate(entities, 1):
            type_hint = self._get_type_hint(entity_type)
            entity_list.append(f"{i}. {entity_text} ({type_hint})")

        entity_section = "\n".join(entity_list)

        prompt = f"""Analyze sentiment toward each entity in this text.

For each entity, provide:
1. SENTIMENT (-2 to +2):
   -2: Strongly negative (scandal, criminal accusation, catastrophic failure)
   -1: Mildly negative (criticism, setback, unfavorable coverage)
    0: Neutral (factual mention, no clear valence)
   +1: Mildly positive (praise, endorsement, favorable coverage)
   +2: Strongly positive (major victory, vindication, overwhelming support)

2. CONFIDENCE (low/medium/high):
   low: Ambiguous or requires significant interpretation
   medium: Clear direction but some nuance
   high: Unambiguous, obvious sentiment

Examples:
"Bondi faces investigation over campaign finance" -> Pam Bondi: -2, high
"Analysts question Newsom's strategy" -> Gavin Newsom: -1, medium
"Hegseth receives veterans group endorsement" -> Pete Hegseth: +2, high
"Biden mentioned the infrastructure bill" -> Joe Biden: 0, high
"Reports suggest Kennedy may face scrutiny" -> RFK Jr: -1, low

Text: "{text}"

Entities to analyze:
{entity_section}

Respond ONLY with: entity_name: sentiment, confidence
One entity per line."""

        return prompt

    def _get_type_hint(self, entity_type: str) -> str:
        """Get a human-readable type hint for prompting."""
        hints = {
            "MARKET_ENTITY": "political figure",
            "PERSON": "person",
            "ORG": "organization",
            "GPE": "place",
            "EVENT": "event",
        }
        return hints.get(entity_type, "entity")

    def _parse_response(
        self,
        response,
        entities: List[Tuple[str, str]],
        text: str,
    ) -> List[EntitySentimentResult]:
        """
        Parse LLM response into EntitySentimentResult objects.

        Expected format from LLM:
            entity_name: sentiment, confidence
            entity_name: sentiment, confidence

        Examples:
            Pam Bondi: -2, high
            Gavin Newsom: -1, medium

        Args:
            response: OpenAI API response
            entities: List of (entity_text, entity_type) tuples
            text: Original text for context extraction

        Returns:
            List of EntitySentimentResult objects
        """
        results = []

        try:
            content = response.choices[0].message.content.strip()
            lines = content.split("\n")

            # Parse each line: "entity_name: sentiment, confidence"
            parsed_results: Dict[str, Tuple[int, float]] = {}

            for line in lines:
                line = line.strip()
                if not line or ":" not in line:
                    continue

                # Parse "entity_name: sentiment, confidence"
                match = re.match(r"(.+?):\s*([+-]?\d+)\s*,\s*(\w+)", line)
                if match:
                    entity_name = match.group(1).strip()
                    try:
                        sentiment_cat = int(match.group(2))
                        sentiment_cat = max(-2, min(2, sentiment_cat))
                    except ValueError:
                        sentiment_cat = 0

                    confidence_label = match.group(3).strip().lower()
                    confidence = CONFIDENCE_TO_FLOAT.get(confidence_label, 0.7)

                    parsed_results[entity_name.lower()] = (sentiment_cat, confidence)

            # Build results for each entity
            for entity_text, entity_type in entities:
                context = self._extract_context(text, entity_text)

                # Try to find matching parsed result
                entity_key = entity_text.lower()
                if entity_key in parsed_results:
                    sentiment_cat, confidence = parsed_results[entity_key]
                else:
                    # Try partial match
                    matched = False
                    for key, (cat, conf) in parsed_results.items():
                        if key in entity_key or entity_key in key:
                            sentiment_cat, confidence = cat, conf
                            matched = True
                            break
                    if not matched:
                        sentiment_cat, confidence = 0, 0.5

                # Map category to impact score
                sentiment_score = SENTIMENT_TO_IMPACT.get(sentiment_cat, 0)

                results.append(EntitySentimentResult(
                    entity_text=entity_text,
                    entity_type=entity_type,
                    sentiment_category=sentiment_cat,
                    sentiment_score=sentiment_score,
                    confidence=confidence,
                    context_snippet=context,
                ))

        except Exception as e:
            logger.warning(f"[sentiment] Parse error: {e}")
            results = self._fallback_results(entities, text)

        return results

    def _extract_context(self, text: str, entity: str) -> str:
        """
        Extract context snippet around entity mention.

        Args:
            text: Full text
            entity: Entity to find

        Returns:
            Context snippet (up to context_window chars)
        """
        text_lower = text.lower()
        entity_lower = entity.lower()

        idx = text_lower.find(entity_lower)
        if idx == -1:
            return text[:self._context_window]

        start = max(0, idx - self._context_window // 2)
        end = min(len(text), idx + len(entity) + self._context_window // 2)

        return text[start:end]

    def _fallback_results(
        self,
        entities: List[Tuple[str, str]],
        text: str,
    ) -> List[EntitySentimentResult]:
        """
        Generate fallback results with neutral sentiment.

        Used when LLM call fails.
        """
        results = []
        for entity_text, entity_type in entities:
            context = self._extract_context(text, entity_text)
            results.append(EntitySentimentResult(
                entity_text=entity_text,
                entity_type=entity_type,
                sentiment_category=0,
                sentiment_score=0,
                confidence=0.0,
                context_snippet=context,
            ))
        return results


# =============================================================================
# Convenience Functions
# =============================================================================


async def analyze_sentiment_batch(
    text: str,
    entities: List[Dict[str, Any]],
    model: str = "gpt-4o-mini",
) -> Dict[str, int]:
    """
    Convenience function for batch sentiment analysis.

    Args:
        text: Text containing entity mentions
        entities: List of entity dicts with 'text' and 'label' keys
        model: OpenAI model name

    Returns:
        Dict mapping entity text to sentiment score
    """
    task = BatchedSentimentTask(model=model)

    entity_tuples = [
        (e.get("text", ""), e.get("label", "ENTITY"))
        for e in entities
    ]

    results = await task.analyze_async(text, entity_tuples)

    return {r.entity_text: r.sentiment_score for r in results}


def analyze_sentiment_batch_sync(
    text: str,
    entities: List[Dict[str, Any]],
    model: str = "gpt-4o-mini",
) -> Dict[str, int]:
    """
    Synchronous convenience function for batch sentiment analysis.

    Args:
        text: Text containing entity mentions
        entities: List of entity dicts with 'text' and 'label' keys
        model: OpenAI model name

    Returns:
        Dict mapping entity text to sentiment score
    """
    task = BatchedSentimentTask(model=model)

    entity_tuples = [
        (e.get("text", ""), e.get("label", "ENTITY"))
        for e in entities
    ]

    results = task.analyze(text, entity_tuples)

    return {r.entity_text: r.sentiment_score for r in results}
