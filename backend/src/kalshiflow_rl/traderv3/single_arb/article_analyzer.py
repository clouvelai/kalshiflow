"""
ArticleAnalyzer - LLM + heuristic analysis of news articles for trading signals.

Key Responsibilities:
    - Analyze news article content for sentiment, relevance, and probability direction
    - Extract named entities and key factual claims
    - Provide fast, cheap pre-embedding analysis using utility-tier LLM (Gemini Flash)
    - Fall back to keyword-based heuristic analysis when LLM is unavailable

Architecture Position:
    Sits in the single_arb module alongside tools.py and context_builder.py.
    Called by tools or Captain subsystems that need to understand news content
    before it enters the memory/embedding pipeline.

Design Principles:
    - Self-contained: no dependencies beyond standard library, dataclasses, typing,
      logging, and the local mentions_models tier system
    - Graceful degradation: LLM failure falls back to heuristic (always returns a result)
    - Lazy initialization: LLM client created on first use, not at import time
    - Cheap by default: utility-tier model (Gemini Flash) keeps costs minimal
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .mentions_models import get_utility_model

logger = logging.getLogger("kalshiflow_rl.traderv3.single_arb.article_analyzer")

# ---------------------------------------------------------------------------
# Keyword lists for heuristic fallback
# ---------------------------------------------------------------------------

_BEARISH_KEYWORDS: List[str] = [
    "decline", "fall", "crash", "drop", "negative",
    "loss", "fail", "unlikely", "against",
]

_BULLISH_KEYWORDS: List[str] = [
    "rise", "surge", "gain", "positive",
    "win", "likely", "support", "approve", "pass",
]

# Regex for extracting capitalized multi-word sequences (named entities heuristic)
_ENTITY_RE = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ArticleAnalysis:
    """Result of analyzing a single news article for trading signals."""

    sentiment: float  # -1.0 (bearish) to +1.0 (bullish)
    market_relevance: float  # 0.0 to 1.0
    entities: List[str] = field(default_factory=list)
    key_claims: List[str] = field(default_factory=list)
    probability_direction: str = "neutral"  # "up" / "down" / "neutral" / "mixed"
    confidence: str = "low"  # "low" / "medium" / "high"


# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------

_ANALYSIS_PROMPT_TEMPLATE = """\
You are a prediction-market news analyst. Analyze the following article in the \
context of the prediction market event described below.

EVENT TITLE: {event_title}
EVENT DESCRIPTION: {event_description}

ARTICLE TITLE: {article_title}
ARTICLE CONTENT:
{article_content}

Return ONLY a JSON object (no markdown fences, no extra text) with these fields:
{{
  "sentiment": <float from -1.0 (bearish) to 1.0 (bullish)>,
  "market_relevance": <float from 0.0 to 1.0, how relevant this article is to the event>,
  "entities": [<up to 5 key named entities mentioned>],
  "key_claims": [<top 3 factual claims from the article>],
  "probability_direction": "<up|down|neutral|mixed> — implied direction for the YES probability",
  "confidence": "<low|medium|high> — your confidence in this analysis"
}}
"""


# ---------------------------------------------------------------------------
# Analyzer class
# ---------------------------------------------------------------------------

class ArticleAnalyzer:
    """Analyzes news articles for trading-relevant signals.

    Uses the utility-tier LLM (Gemini Flash by default) for fast, cheap analysis.
    Falls back to keyword-based heuristics when the LLM is unavailable or fails.
    """

    def __init__(self) -> None:
        self._llm_client = None  # Lazy-initialized on first LLM call

    async def _call_llm(self, prompt: str) -> str:
        """Call utility-tier LLM for article analysis."""
        if self._llm_client is None:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI

                self._llm_client = ChatGoogleGenerativeAI(
                    model=get_utility_model(),
                    temperature=0.0,
                )
            except ImportError:
                # Fall back to any available model
                try:
                    from langchain_anthropic import ChatAnthropic

                    self._llm_client = ChatAnthropic(
                        model="claude-haiku-4-5-20251001",
                        temperature=0.0,
                    )
                except ImportError:
                    return ""

        try:
            response = await self._llm_client.ainvoke(prompt)
            return response.content
        except Exception as e:
            logger.warning(f"[ARTICLE_ANALYZER] LLM call failed: {e}")
            return ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def analyze(
        self,
        article_content: str,
        article_title: str,
        event_title: str,
        event_description: str = "",
    ) -> ArticleAnalysis:
        """Analyze a single article using the utility-tier LLM.

        Falls back to ``analyze_heuristic()`` if the LLM call or JSON
        parsing fails.

        Args:
            article_content: Full text of the news article.
            article_title: Headline / title of the article.
            event_title: Title of the prediction-market event.
            event_description: Optional longer description of the event.

        Returns:
            ArticleAnalysis with sentiment, relevance, entities, claims,
            probability direction, and confidence.
        """
        prompt = _ANALYSIS_PROMPT_TEMPLATE.format(
            event_title=event_title,
            event_description=event_description or "(none)",
            article_title=article_title,
            article_content=article_content[:4000],  # cap to avoid token blowup
        )

        raw = await self._call_llm(prompt)
        if not raw:
            return self.analyze_heuristic(article_content, article_title)

        # Attempt to parse JSON from the LLM response
        try:
            parsed = self._parse_json(raw)
            return ArticleAnalysis(
                sentiment=float(self._clamp(parsed.get("sentiment", 0.0), -1.0, 1.0)),
                market_relevance=float(
                    self._clamp(parsed.get("market_relevance", 0.5), 0.0, 1.0)
                ),
                entities=self._to_str_list(parsed.get("entities", []))[:5],
                key_claims=self._to_str_list(parsed.get("key_claims", []))[:3],
                probability_direction=self._validate_direction(
                    parsed.get("probability_direction", "neutral")
                ),
                confidence=self._validate_confidence(
                    parsed.get("confidence", "low")
                ),
            )
        except Exception as e:
            logger.warning(
                f"[ARTICLE_ANALYZER] JSON parse failed, falling back to heuristic: {e}"
            )
            return self.analyze_heuristic(article_content, article_title)

    async def analyze_batch(
        self,
        articles: List[Dict],
        event_title: str,
        event_description: str = "",
    ) -> List[ArticleAnalysis]:
        """Analyze multiple articles sequentially.

        The utility-tier model is cheap and fast, so sequential processing
        is acceptable and avoids rate-limit issues.

        Args:
            articles: List of dicts, each with ``content`` and ``title`` keys.
            event_title: Title of the prediction-market event.
            event_description: Optional longer description of the event.

        Returns:
            List of ArticleAnalysis results, one per input article.
        """
        results: List[ArticleAnalysis] = []
        for article in articles:
            content = article.get("content", "")
            title = article.get("title", "")
            analysis = await self.analyze(
                article_content=content,
                article_title=title,
                event_title=event_title,
                event_description=event_description,
            )
            results.append(analysis)
        return results

    def analyze_heuristic(
        self,
        article_content: str,
        article_title: str = "",
    ) -> ArticleAnalysis:
        """No-LLM fallback using keyword matching.

        Counts bearish/bullish keyword occurrences to derive sentiment and
        probability direction. Extracts named entities via capitalized
        multi-word sequences.

        Args:
            article_content: Full text of the news article.
            article_title: Optional headline (included in keyword search).

        Returns:
            ArticleAnalysis with confidence always set to ``"low"``.
        """
        text = f"{article_title} {article_content}".lower()

        bullish_count = sum(text.count(kw) for kw in _BULLISH_KEYWORDS)
        bearish_count = sum(text.count(kw) for kw in _BEARISH_KEYWORDS)
        total = bullish_count + bearish_count
        sentiment = (bullish_count - bearish_count) / max(total, 1)

        # Derive probability direction from sentiment
        if sentiment > 0.2:
            direction = "up"
        elif sentiment < -0.2:
            direction = "down"
        else:
            direction = "neutral"

        # Extract entities from original (non-lowered) text
        raw_text = f"{article_title} {article_content}"
        entities = list(dict.fromkeys(_ENTITY_RE.findall(raw_text)))[:5]

        return ArticleAnalysis(
            sentiment=round(self._clamp(sentiment, -1.0, 1.0), 3),
            market_relevance=0.5,
            entities=entities,
            key_claims=[],
            probability_direction=direction,
            confidence="low",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json(raw: str) -> Dict:
        """Extract and parse the first JSON object from a string.

        Handles cases where the LLM wraps JSON in markdown code fences.
        """
        # Strip markdown fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            # Remove opening fence (with optional language tag) and closing fence
            cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
            cleaned = re.sub(r"\n?```\s*$", "", cleaned)

        return json.loads(cleaned)

    @staticmethod
    def _clamp(value, lo: float, hi: float) -> float:
        """Clamp a numeric value to [lo, hi]."""
        try:
            return max(lo, min(hi, float(value)))
        except (TypeError, ValueError):
            return (lo + hi) / 2.0

    @staticmethod
    def _to_str_list(value) -> List[str]:
        """Coerce a value to a list of strings."""
        if isinstance(value, list):
            return [str(v) for v in value if v]
        return []

    @staticmethod
    def _validate_direction(value: str) -> str:
        """Ensure probability_direction is one of the allowed values."""
        allowed = {"up", "down", "neutral", "mixed"}
        v = str(value).lower().strip()
        return v if v in allowed else "neutral"

    @staticmethod
    def _validate_confidence(value: str) -> str:
        """Ensure confidence is one of the allowed values."""
        allowed = {"low", "medium", "high"}
        v = str(value).lower().strip()
        return v if v in allowed else "low"
