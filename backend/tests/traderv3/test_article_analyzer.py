"""Tests for ArticleAnalyzer.

T1/T2 tests — pure Python + async + lightweight mocks. No external API calls.
"""

import json

import pytest
from unittest.mock import AsyncMock, patch

from kalshiflow_rl.traderv3.single_arb.article_analyzer import (
    ArticleAnalyzer,
    ArticleAnalysis,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def analyzer():
    return ArticleAnalyzer()


# ===========================================================================
# Heuristic Analysis — Sentiment & Direction
# ===========================================================================


class TestHeuristicBullish:
    """Positive sentiment content should produce direction='up'."""

    def test_bullish_keywords_produce_positive_sentiment(self, analyzer):
        content = "Markets surge as support grows. Likely to gain approval and pass."
        result = analyzer.analyze_heuristic(content)
        assert result.sentiment > 0.0
        assert result.probability_direction == "up"

    def test_strong_bullish_content(self, analyzer):
        content = "A massive surge in support. Investors gain confidence. Win likely."
        result = analyzer.analyze_heuristic(content)
        assert result.sentiment > 0.2
        assert result.probability_direction == "up"


class TestHeuristicBearish:
    """Negative sentiment content should produce direction='down'."""

    def test_bearish_keywords_produce_negative_sentiment(self, analyzer):
        content = "Markets crash as negative outlook causes sharp decline and loss."
        result = analyzer.analyze_heuristic(content)
        assert result.sentiment < 0.0
        assert result.probability_direction == "down"

    def test_strong_bearish_content(self, analyzer):
        # Avoid "pass" (bullish keyword) — use only bearish keywords
        content = "Unlikely outcome. Analysts against it. A decline and major loss expected. Fall and crash ahead."
        result = analyzer.analyze_heuristic(content)
        assert result.sentiment < -0.2
        assert result.probability_direction == "down"


class TestHeuristicNeutral:
    """Content with near-zero sentiment should produce direction='neutral'."""

    def test_no_keywords_neutral(self, analyzer):
        content = "The committee met today to discuss the agenda items."
        result = analyzer.analyze_heuristic(content)
        assert result.sentiment == 0.0
        assert result.probability_direction == "neutral"

    def test_balanced_keywords_neutral(self, analyzer):
        # One bullish ("gain"), one bearish ("decline") — balanced
        content = "Some gain here, but also a decline there."
        result = analyzer.analyze_heuristic(content)
        assert -0.2 <= result.sentiment <= 0.2
        assert result.probability_direction == "neutral"


class TestHeuristicMixed:
    """Content with both bullish and bearish keywords."""

    def test_mixed_content_has_bounded_sentiment(self, analyzer):
        content = "Markets surge but face a crash risk. Gain and loss coexist."
        result = analyzer.analyze_heuristic(content)
        assert -1.0 <= result.sentiment <= 1.0

    def test_mixed_content_returns_valid_direction(self, analyzer):
        content = "Likely to rise but could drop. Positive and negative signals."
        result = analyzer.analyze_heuristic(content)
        assert result.probability_direction in {"up", "down", "neutral"}


# ===========================================================================
# Entity Extraction
# ===========================================================================


class TestEntityExtraction:
    """Capitalized multi-word sequences should be found as entities."""

    def test_extracts_named_entities(self, analyzer):
        # Regex matches capitalized multi-word sequences including leading "The"
        content = "Federal Reserve announced new policy. White House responded."
        result = analyzer.analyze_heuristic(content)
        assert "Federal Reserve" in result.entities
        assert "White House" in result.entities

    def test_entity_dedup(self, analyzer):
        content = "Federal Reserve met today. Federal Reserve issued guidance."
        result = analyzer.analyze_heuristic(content)
        assert result.entities.count("Federal Reserve") == 1

    def test_entity_limit_max_5(self, analyzer):
        content = (
            "Apple Inc discussed results. "
            "Google Cloud expanded. "
            "Amazon Web announced. "
            "Goldman Sachs reported. "
            "Morgan Stanley upgraded. "
            "Bank Of something. "
            "Credit Suisse failed."
        )
        result = analyzer.analyze_heuristic(content)
        assert len(result.entities) <= 5


# ===========================================================================
# Heuristic Fixed Defaults
# ===========================================================================


class TestHeuristicDefaults:
    """Heuristic analysis always uses fixed confidence and market_relevance."""

    def test_confidence_always_low(self, analyzer):
        result = analyzer.analyze_heuristic("anything goes here")
        assert result.confidence == "low"

    def test_market_relevance_always_half(self, analyzer):
        result = analyzer.analyze_heuristic("any content")
        assert result.market_relevance == 0.5


# ===========================================================================
# _parse_json
# ===========================================================================


class TestParseJson:
    """JSON parsing helper handles raw and fenced JSON."""

    def test_valid_json(self):
        raw = '{"sentiment": 0.5, "market_relevance": 0.8}'
        result = ArticleAnalyzer._parse_json(raw)
        assert result["sentiment"] == 0.5
        assert result["market_relevance"] == 0.8

    def test_json_in_markdown_fences(self):
        raw = '```json\n{"sentiment": -0.3, "confidence": "high"}\n```'
        result = ArticleAnalyzer._parse_json(raw)
        assert result["sentiment"] == -0.3
        assert result["confidence"] == "high"

    def test_json_in_bare_fences(self):
        raw = '```\n{"direction": "up"}\n```'
        result = ArticleAnalyzer._parse_json(raw)
        assert result["direction"] == "up"

    def test_invalid_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            ArticleAnalyzer._parse_json("not json at all")


# ===========================================================================
# _clamp
# ===========================================================================


class TestClamp:
    """Clamp helper constrains values to [lo, hi]."""

    def test_within_range(self):
        assert ArticleAnalyzer._clamp(0.5, 0.0, 1.0) == 0.5

    def test_below_range(self):
        assert ArticleAnalyzer._clamp(-2.0, -1.0, 1.0) == -1.0

    def test_above_range(self):
        assert ArticleAnalyzer._clamp(5.0, 0.0, 1.0) == 1.0

    def test_non_numeric_returns_midpoint(self):
        assert ArticleAnalyzer._clamp("bad", 0.0, 1.0) == 0.5

    def test_none_returns_midpoint(self):
        assert ArticleAnalyzer._clamp(None, -1.0, 1.0) == 0.0


# ===========================================================================
# _validate_direction
# ===========================================================================


class TestValidateDirection:
    """Direction validation rejects unknown values."""

    @pytest.mark.parametrize("value", ["up", "down", "neutral", "mixed"])
    def test_valid_values(self, value):
        assert ArticleAnalyzer._validate_direction(value) == value

    def test_uppercase_normalized(self):
        assert ArticleAnalyzer._validate_direction("UP") == "up"

    def test_invalid_value_returns_neutral(self):
        assert ArticleAnalyzer._validate_direction("sideways") == "neutral"

    def test_empty_string_returns_neutral(self):
        assert ArticleAnalyzer._validate_direction("") == "neutral"


# ===========================================================================
# _validate_confidence
# ===========================================================================


class TestValidateConfidence:
    """Confidence validation rejects unknown values."""

    @pytest.mark.parametrize("value", ["low", "medium", "high"])
    def test_valid_values(self, value):
        assert ArticleAnalyzer._validate_confidence(value) == value

    def test_uppercase_normalized(self):
        assert ArticleAnalyzer._validate_confidence("HIGH") == "high"

    def test_invalid_value_returns_low(self):
        assert ArticleAnalyzer._validate_confidence("very_high") == "low"

    def test_empty_string_returns_low(self):
        assert ArticleAnalyzer._validate_confidence("") == "low"


# ===========================================================================
# _to_str_list
# ===========================================================================


class TestToStrList:
    """Coerces values to list of strings."""

    def test_list_input(self):
        assert ArticleAnalyzer._to_str_list(["a", "b", "c"]) == ["a", "b", "c"]

    def test_list_with_ints(self):
        assert ArticleAnalyzer._to_str_list([1, 2, 3]) == ["1", "2", "3"]

    def test_filters_falsy_items(self):
        assert ArticleAnalyzer._to_str_list(["a", "", None, "b"]) == ["a", "b"]

    def test_non_list_returns_empty(self):
        assert ArticleAnalyzer._to_str_list("not a list") == []

    def test_none_returns_empty(self):
        assert ArticleAnalyzer._to_str_list(None) == []


# ===========================================================================
# LLM analyze (mocked)
# ===========================================================================


class TestAnalyzeWithMockedLLM:
    """Full LLM path with _call_llm mocked to return valid JSON."""

    @pytest.mark.asyncio
    async def test_llm_returns_valid_json(self, analyzer):
        llm_response = json.dumps({
            "sentiment": 0.7,
            "market_relevance": 0.9,
            "entities": ["Federal Reserve", "Jerome Powell"],
            "key_claims": ["Rate hike expected", "Inflation slowing"],
            "probability_direction": "up",
            "confidence": "high",
        })

        with patch.object(analyzer, "_call_llm", new_callable=AsyncMock, return_value=llm_response):
            result = await analyzer.analyze(
                article_content="Fed likely to raise rates.",
                article_title="Fed Rate Decision",
                event_title="Will the Fed raise rates?",
                event_description="FOMC meeting decision on interest rates.",
            )

        assert isinstance(result, ArticleAnalysis)
        assert result.sentiment == 0.7
        assert result.market_relevance == 0.9
        assert result.entities == ["Federal Reserve", "Jerome Powell"]
        assert result.key_claims == ["Rate hike expected", "Inflation slowing"]
        assert result.probability_direction == "up"
        assert result.confidence == "high"

    @pytest.mark.asyncio
    async def test_llm_empty_response_falls_back_to_heuristic(self, analyzer):
        with patch.object(analyzer, "_call_llm", new_callable=AsyncMock, return_value=""):
            result = await analyzer.analyze(
                article_content="surge in support likely to gain approval",
                article_title="Title",
                event_title="Event",
            )

        # Should use heuristic — confidence is always "low"
        assert result.confidence == "low"
        assert result.market_relevance == 0.5

    @pytest.mark.asyncio
    async def test_llm_invalid_json_falls_back_to_heuristic(self, analyzer):
        with patch.object(analyzer, "_call_llm", new_callable=AsyncMock, return_value="not json"):
            result = await analyzer.analyze(
                article_content="content",
                article_title="title",
                event_title="event",
            )

        assert result.confidence == "low"
        assert result.market_relevance == 0.5

    @pytest.mark.asyncio
    async def test_llm_clamps_out_of_range_values(self, analyzer):
        llm_response = json.dumps({
            "sentiment": 5.0,
            "market_relevance": -2.0,
            "probability_direction": "up",
            "confidence": "medium",
        })

        with patch.object(analyzer, "_call_llm", new_callable=AsyncMock, return_value=llm_response):
            result = await analyzer.analyze(
                article_content="content",
                article_title="title",
                event_title="event",
            )

        assert result.sentiment == 1.0  # clamped from 5.0
        assert result.market_relevance == 0.0  # clamped from -2.0
