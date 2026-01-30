"""
LLM-Based Entity Extractor.

Extracts named entities AND scores sentiment in a single GPT-4o-mini call,
replacing the unreliable spaCy NER + separate batched sentiment approach.

This produces higher-quality entities from Reddit headlines by using LLM
understanding instead of off-the-shelf NER which misidentifies common words
like "TDS" and "I Like It" as entities while missing real ones like
"Alex Pretti", "Amazon", "Newsom".

Cost: ~$0.001/post with GPT-4o-mini (~$1.50/day at 50 posts/min).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from openai import AsyncOpenAI

from ..schemas.entity_schemas import LLMExtractedEntity

logger = logging.getLogger("kalshiflow_rl.traderv3.nlp.llm_entity_extractor")

# Prompt template for combined entity extraction + sentiment scoring
ENTITY_EXTRACTION_PROMPT = """Given this Reddit post, extract all named entities and score sentiment toward each.

Title: "{title}"
Subreddit: r/{subreddit}
Body: "{body}"

Return a JSON array. Each element:
{{
  "name": "canonical name",
  "type": "PERSON|ORG|GPE|EVENT|POLICY|NORP",
  "sentiment": -100 to +100,
  "confidence": "low|medium|high",
  "context": "1-2 sentence summary of what the post says about this entity"
}}

Rules:
- Extract ONLY proper nouns and named entities (people, places, organizations, specific events/policies)
- Do NOT extract common nouns, adjectives, verbs, or generic descriptions
- Do NOT extract subreddit names, Reddit jargon, or casual phrases
- Use the canonical/full form of names (e.g., "Gavin Newsom" not just "Newsom")
- If an acronym is used, expand it if possible (e.g., "AWS" stays "AWS" since that's the canonical name)
- Sentiment: Is the news good or bad for this entity? (-100 = catastrophic, +100 = triumph)
- Confidence: how clearly is sentiment expressed? (low = ambiguous, medium = clear direction, high = unambiguous)
- Context: What does the post say about this entity specifically? Be factual and concise.

If no named entities are found, return an empty array: []"""


MARKET_AWARE_EXTRACTION_PROMPT = """Given this Reddit post, extract all named entities, score sentiment,
AND identify which active Kalshi prediction markets each entity relates to.

Title: "{title}"
Subreddit: r/{subreddit}
Body: "{body}"

ACTIVE KALSHI MARKETS (ticker | title | keywords):
{market_list}

Return a JSON array. Each element:
{{
  "name": "canonical name",
  "type": "PERSON|ORG|GPE|EVENT|POLICY|NORP",
  "sentiment": -100 to +100,
  "confidence": "low|medium|high",
  "market_tickers": ["TICKER1", "TICKER2"],
  "context": "1-2 sentence summary of what the post says about this entity"
}}

Rules:
- Extract ONLY proper nouns and named entities (people, places, organizations, specific events/policies)
- Do NOT extract common nouns, adjectives, verbs, or generic descriptions
- Do NOT extract subreddit names, Reddit jargon, or casual phrases
- Use the canonical/full form of names (e.g., "Gavin Newsom" not just "Newsom")
- Sentiment: Is the news good or bad for this entity? (-100 = catastrophic, +100 = triumph)
- market_tickers: list tickers from ACTIVE MARKETS above that this entity directly relates to
- Use the keywords to help match entities to markets
- An entity can match zero, one, or multiple markets
- Only include tickers you are confident about - precision over recall
- Context: What does the post say about this entity specifically? Be factual and concise.
- If no named entities found, return: []"""


class LLMEntityExtractor:
    """
    Async component that uses GPT-4o-mini to extract entities and score
    sentiment in a single structured call per post.

    Falls back to empty list if LLM call fails or times out.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        timeout: float = 10.0,
    ):
        self._model = model
        self._timeout = timeout
        self._client: Optional["AsyncOpenAI"] = None

        # Stats
        self._calls_made = 0
        self._calls_succeeded = 0
        self._calls_failed = 0
        self._entities_extracted = 0

    def _get_client(self) -> "AsyncOpenAI":
        """Get or create async OpenAI client."""
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return self._client

    async def extract(
        self,
        title: str,
        subreddit: str = "",
        body: str = "",
    ) -> List[LLMExtractedEntity]:
        """
        Extract entities and sentiment from a Reddit post.

        Args:
            title: Post title (required)
            subreddit: Subreddit name
            body: Post body text (truncated to 500 chars)

        Returns:
            List of LLMExtractedEntity objects
        """
        if not title:
            return []

        self._calls_made += 1

        try:
            client = self._get_client()

            # Build prompt with truncated body
            prompt = ENTITY_EXTRACTION_PROMPT.format(
                title=title,
                subreddit=subreddit or "unknown",
                body=body[:500] if body else "(no body)",
            )

            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=700,
                    temperature=0,
                ),
                timeout=self._timeout,
            )

            content = response.choices[0].message.content.strip()

            # Parse JSON response (handle markdown code blocks)
            entities = self._parse_response(content)

            self._calls_succeeded += 1
            self._entities_extracted += len(entities)

            if entities:
                logger.info(
                    f"[llm_extractor] Extracted {len(entities)} entities: "
                    f"{[e.name for e in entities]}"
                )

            return entities

        except asyncio.TimeoutError:
            self._calls_failed += 1
            logger.warning("[llm_extractor] LLM call timed out")
            return []
        except Exception as e:
            self._calls_failed += 1
            logger.warning(f"[llm_extractor] LLM extraction failed: {e}")
            return []

    async def extract_with_markets(
        self,
        title: str,
        subreddit: str = "",
        body: str = "",
        market_prompt: str = "",
    ) -> List[LLMExtractedEntity]:
        """
        Extract entities with market-aware matching.

        Uses an enriched prompt that includes active Kalshi markets with keywords,
        allowing the LLM to directly map entities to market tickers.

        Args:
            title: Post title (required)
            subreddit: Subreddit name
            body: Post body text (truncated to 500 chars)
            market_prompt: Pre-formatted market list string (ticker | title | keywords)

        Returns:
            List of LLMExtractedEntity objects with market_tickers populated
        """
        if not title:
            return []

        # Fall back to standard extraction if no market context
        if not market_prompt:
            return await self.extract(title=title, subreddit=subreddit, body=body)

        self._calls_made += 1

        try:
            client = self._get_client()

            prompt = MARKET_AWARE_EXTRACTION_PROMPT.format(
                title=title,
                subreddit=subreddit or "unknown",
                body=body[:500] if body else "(no body)",
                market_list=market_prompt,
            )

            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1200,
                    temperature=0,
                ),
                timeout=self._timeout,
            )

            content = response.choices[0].message.content.strip()
            entities = self._parse_response(content)

            self._calls_succeeded += 1
            self._entities_extracted += len(entities)

            if entities:
                mapped = [e for e in entities if e.market_tickers]
                logger.info(
                    f"[llm_extractor] Market-aware: {len(entities)} entities, "
                    f"{len(mapped)} with market tickers: "
                    f"{[(e.name, list(e.market_tickers)) for e in mapped]}"
                )

            return entities

        except asyncio.TimeoutError:
            self._calls_failed += 1
            logger.warning("[llm_extractor] Market-aware LLM call timed out")
            return []
        except Exception as e:
            self._calls_failed += 1
            logger.warning(f"[llm_extractor] Market-aware extraction failed: {e}")
            return []

    def _parse_response(self, content: str) -> List[LLMExtractedEntity]:
        """
        Parse LLM response into LLMExtractedEntity objects.

        Handles:
        - Raw JSON arrays
        - Markdown code blocks (```json ... ```)
        - Malformed JSON with trailing commas
        """
        # Strip markdown code blocks
        if content.startswith("```"):
            lines = content.split("\n")
            # Remove first and last lines (``` markers)
            inner_lines = []
            in_block = False
            for line in lines:
                if line.strip().startswith("```") and not in_block:
                    in_block = True
                    continue
                elif line.strip() == "```" and in_block:
                    break
                elif in_block:
                    inner_lines.append(line)
            content = "\n".join(inner_lines).strip()

        if not content:
            return []

        try:
            raw = json.loads(content)
        except json.JSONDecodeError:
            # Try removing trailing commas
            import re
            cleaned = re.sub(r",\s*([}\]])", r"\1", content)
            try:
                raw = json.loads(cleaned)
            except json.JSONDecodeError:
                logger.warning(
                    f"[llm_extractor] Failed to parse JSON: {content[:200]}"
                )
                return []

        if not isinstance(raw, list):
            return []

        entities = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            try:
                entity = LLMExtractedEntity.from_dict(item)
                # Validate: skip empty names or very short names
                if len(entity.name.strip()) < 3:
                    continue
                # Validate sentiment range
                sentiment = max(-100, min(100, entity.sentiment))
                if sentiment != entity.sentiment:
                    entity = LLMExtractedEntity(
                        name=entity.name,
                        entity_type=entity.entity_type,
                        sentiment=sentiment,
                        confidence=entity.confidence,
                        market_tickers=entity.market_tickers,
                        context=entity.context,
                    )
                entities.append(entity)
            except Exception as e:
                logger.debug(f"[llm_extractor] Skipping malformed entity: {e}")

        return entities

    def get_stats(self) -> dict:
        """Get extractor statistics."""
        return {
            "calls_made": self._calls_made,
            "calls_succeeded": self._calls_succeeded,
            "calls_failed": self._calls_failed,
            "entities_extracted": self._entities_extracted,
            "model": self._model,
        }
