"""
GDELT News Intelligence Sub-Agent.

Replaces blunt truncation of GDELT results with a Haiku sub-agent that
analyzes the FULL GDELT response and returns structured trading intelligence.

Data flow:
  Deep Agent (Sonnet) calls get_news_intelligence(search_terms, context_hint)
    -> GDELTNewsAnalyzer.analyze()
        -> Check 15-min cache -> HIT? return cached
        -> MISS:
            -> GDELTDocClient.search_articles(max_records=250)
            -> (optional) GDELTClient.query_news() if BigQuery available
            -> Build prompt with full raw data
            -> Haiku call (~8K input, ~500 output -> ~$0.003)
            -> Parse structured JSON
            -> Cache 15 min
            -> Return compact intelligence to deep agent

Cost: ~$0.003 per call. With 15-min cache: realistic <$1/day.
"""

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Callable, Dict, List, Optional

from anthropic import AsyncAnthropic

from .gdelt_client import _gdelt_time_bucket

logger = logging.getLogger("kalshiflow_rl.traderv3.services.gdelt_news_analyzer")

# Default Haiku model for sub-agent analysis
DEFAULT_ANALYZER_MODEL = "claude-3-5-haiku-20241022"
FALLBACK_MODEL = "claude-sonnet-4-20250514"


@dataclass
class GDELTNewsAnalyzerConfig:
    """Configuration for the GDELT news intelligence sub-agent."""
    model: str = DEFAULT_ANALYZER_MODEL
    fallback_model: str = FALLBACK_MODEL
    cache_ttl_seconds: float = 900.0  # 15 minutes (aligned with GDELT update frequency)
    max_articles: int = 250  # Max articles to fetch from DOC API
    temperature: float = 0.0  # Deterministic for consistent analysis
    max_tokens: int = 1024  # Output cap for structured JSON


SUB_AGENT_SYSTEM_PROMPT = """You are a news intelligence analyst for a prediction market trader. Your job is to analyze raw GDELT news data and produce a structured trading intelligence report.

Rules:
- Output ONLY valid JSON matching the schema below. No markdown, no explanation.
- "no_signal" is a valid and common result. Do NOT fabricate signals.
- Freshness matters: articles from the last 2 hours are more actionable than older ones.
- Source diversity matters: 5 different outlets > 1 outlet repeated 5 times.
- Be specific: cite article counts, name sources, quote tone values.
- For sentiment, use the actual avg_tone values from the data.
- For market_signals, only include signals supported by concrete evidence from the articles.

Output JSON schema:
{
  "narrative_summary": "2-3 sentence summary of the news landscape",
  "key_developments": [
    {"headline": "Brief description", "source_count": 8, "recency": "breaking|recent|developing|stale"}
  ],
  "sentiment": {
    "overall": "strongly_negative|negative|mixed|neutral|positive|strongly_positive",
    "trend": "deteriorating|stable|improving",
    "avg_tone": -4.2,
    "confidence": "high|medium|low"
  },
  "source_analysis": {
    "total_articles": 23,
    "unique_sources": 12,
    "notable_sources": ["reuters.com", "apnews.com"],
    "geographic_spread": "domestic|international|global"
  },
  "market_signals": [
    {
      "signal": "Brief description",
      "direction": "bullish|bearish|neutral",
      "strength": "strong|moderate|weak|none",
      "evidence": "Specific evidence from articles",
      "time_sensitivity": "urgent|normal|low"
    }
  ],
  "freshness": {
    "newest_article_age_minutes": 12,
    "coverage_window_hours": 4,
    "is_breaking": true,
    "volume_trend": "surging|steady|declining"
  },
  "trading_recommendation": "act_now|monitor|wait|no_signal"
}"""


class GDELTNewsAnalyzer:
    """
    Sub-agent that analyzes full GDELT data and returns structured intelligence.

    Uses Haiku for fast, cheap analysis of raw article data. Results are cached
    for 15 minutes (aligned with GDELT's update frequency).
    """

    def __init__(
        self,
        config: Optional[GDELTNewsAnalyzerConfig] = None,
        token_usage_callback: Optional[Callable] = None,
    ):
        self._config = config or GDELTNewsAnalyzerConfig()

        # GDELT client references (wired by coordinator)
        self._gdelt_doc_client = None   # GDELTDocClient (free DOC API)
        self._gdelt_client = None       # GDELTClient (BigQuery GKG, optional)

        # Anthropic client
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            self._anthropic = AsyncAnthropic()
        else:
            self._anthropic = None
            logger.warning("[news_analyzer] No ANTHROPIC_API_KEY — analyzer disabled")

        # Token usage callback (for cost tracking in parent agent)
        self._token_usage_callback = token_usage_callback

        # 15-min TTL cache keyed by sorted lowercased search terms
        self._cache: Dict[str, tuple] = {}  # {key: (timestamp, result)}

        # Cost tracking
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_calls = 0
        self._total_cache_hits = 0
        self._total_errors = 0

        logger.info(
            f"[news_analyzer] Initialized (model={self._config.model}, "
            f"cache_ttl={self._config.cache_ttl_seconds}s, "
            f"max_articles={self._config.max_articles})"
        )

    # =========================================================================
    # Cache
    # =========================================================================

    def _cache_key(self, search_terms: List[str]) -> str:
        """Generate cache key from sorted lowercased search terms + GDELT time bucket."""
        normalized = sorted(t.strip().lower() for t in search_terms if t.strip())
        key_data = json.dumps({"terms": normalized, "bucket": _gdelt_time_bucket()})
        return hashlib.md5(key_data.encode()).hexdigest()

    def _get_cached(self, key: str) -> Optional[Dict[str, Any]]:
        """Return cached result if within TTL."""
        if key in self._cache:
            ts, result = self._cache[key]
            if time.time() - ts < self._config.cache_ttl_seconds:
                return result
            del self._cache[key]
        return None

    def _set_cache(self, key: str, result: Dict[str, Any]) -> None:
        """Store result in cache."""
        self._cache[key] = (time.time(), result)
        # Prune expired entries
        if len(self._cache) > 50:
            now = time.time()
            self._cache = {
                k: (ts, v) for k, (ts, v) in self._cache.items()
                if now - ts < self._config.cache_ttl_seconds
            }

    # =========================================================================
    # Main entry point
    # =========================================================================

    async def analyze(
        self,
        search_terms: List[str],
        context_hint: str = "",
    ) -> Dict[str, Any]:
        """
        Analyze GDELT news for the given search terms and return structured intelligence.

        Args:
            search_terms: List of terms to search (e.g., ["government shutdown", "funding"])
            context_hint: Optional hint about what the agent is investigating
                          (e.g., "Evaluating KXGOVTFUND-25FEB14 NO position")

        Returns:
            Structured intelligence dict with status, intelligence, and metadata
        """
        if not search_terms:
            return self._make_result("error", search_terms, error="No search terms provided")

        # Check cache
        cache_key = self._cache_key(search_terms)
        cached = self._get_cached(cache_key)
        if cached is not None:
            self._total_cache_hits += 1
            logger.info(f"[news_analyzer] Cache hit for terms={search_terms}")
            # Return a copy with cached status
            result = dict(cached)
            result["status"] = "cached"
            result["metadata"]["cached"] = True
            return result

        # Fetch raw GDELT data
        raw_data = await self._fetch_gdelt_data(search_terms)
        if raw_data.get("error"):
            return self._make_result("no_data", search_terms, error=raw_data["error"])

        article_count = raw_data.get("article_count", 0)
        if article_count == 0:
            return self._make_result("no_data", search_terms,
                                     intelligence=self._empty_intelligence(search_terms))

        # Build prompt and call sub-agent
        prompt = self._build_prompt(search_terms, raw_data, context_hint)
        start_ms = time.time() * 1000

        intelligence = await self._call_sub_agent(prompt, self._config.model)

        # If primary model failed, try fallback
        if intelligence is None and self._config.model != self._config.fallback_model:
            logger.warning(f"[news_analyzer] Primary model failed, trying fallback: {self._config.fallback_model}")
            intelligence = await self._call_sub_agent(prompt, self._config.fallback_model)

        elapsed_ms = int(time.time() * 1000 - start_ms)

        # If both models failed, return degraded result with raw stats
        if intelligence is None:
            self._total_errors += 1
            logger.warning("[news_analyzer] All models failed — returning degraded result")
            intelligence = self._degraded_intelligence(raw_data, search_terms)
            status = "degraded"
        else:
            status = "success"

        # Build full result — expire at next GDELT bucket boundary
        bucket_start = _gdelt_time_bucket()
        cache_expires_at = datetime.fromtimestamp(bucket_start + 900, tz=timezone.utc)
        result = {
            "status": status,
            "search_terms": search_terms,
            "intelligence": intelligence,
            "metadata": {
                "analyzer_model": self._config.model,
                "analysis_time_ms": elapsed_ms,
                "raw_article_count": article_count,
                "cached": False,
                "cache_expires_at": cache_expires_at.isoformat(),
                "cost_estimate_usd": round(self._estimate_cost(prompt, intelligence), 5),
            },
        }

        # Cache the result
        self._set_cache(cache_key, result)
        self._total_calls += 1

        logger.info(
            f"[news_analyzer] Analysis complete: terms={search_terms}, "
            f"articles={article_count}, status={status}, "
            f"elapsed={elapsed_ms}ms"
        )
        return result

    # =========================================================================
    # GDELT data fetching
    # =========================================================================

    async def _fetch_gdelt_data(self, search_terms: List[str]) -> Dict[str, Any]:
        """Fetch raw GDELT data from DOC API (primary) and optionally GKG."""
        result: Dict[str, Any] = {}

        # Primary: DOC API (free, always available)
        if self._gdelt_doc_client:
            try:
                doc_result = await self._gdelt_doc_client.search_articles(
                    search_terms=search_terms,
                    max_records=self._config.max_articles,
                    sort="datedesc",
                )
                if "error" not in doc_result:
                    result["doc_articles"] = doc_result.get("articles", [])
                    result["article_count"] = doc_result.get("article_count", 0)
                    result["source_diversity"] = doc_result.get("source_diversity", 0)
                    result["tone_summary"] = doc_result.get("tone_summary", {})
                else:
                    result["doc_error"] = doc_result.get("error", "Unknown error")
            except Exception as e:
                logger.warning(f"[news_analyzer] DOC API fetch failed: {e}")
                result["doc_error"] = str(e)

        # Optional: GKG (BigQuery, structured entities/themes)
        if self._gdelt_client:
            try:
                gkg_result = await self._gdelt_client.query_news(
                    search_terms=search_terms,
                    limit=50,  # Keep BigQuery usage low
                )
                if "error" not in gkg_result:
                    result["gkg_themes"] = gkg_result.get("key_themes", [])
                    result["gkg_persons"] = gkg_result.get("key_persons", [])
                    result["gkg_organizations"] = gkg_result.get("key_organizations", [])
                    result["gkg_article_count"] = gkg_result.get("article_count", 0)
            except Exception as e:
                logger.debug(f"[news_analyzer] GKG fetch failed (non-critical): {e}")

        # If DOC API returned no data and there's no GKG data either
        if not result.get("doc_articles") and not result.get("gkg_themes"):
            if result.get("doc_error"):
                result["error"] = result["doc_error"]
            else:
                result["article_count"] = 0

        # Combine DOC + GKG article counts so GKG data isn't discarded
        # when DOC API fails or returns 0
        doc_count = result.get("article_count", 0)
        gkg_count = result.get("gkg_article_count", 0)
        result["article_count"] = max(doc_count, gkg_count)

        return result

    # =========================================================================
    # Prompt construction
    # =========================================================================

    def _build_prompt(
        self,
        search_terms: List[str],
        raw_data: Dict[str, Any],
        context_hint: str,
    ) -> str:
        """Build the user message for the Haiku sub-agent."""
        parts = []

        parts.append(f"Search terms: {json.dumps(search_terms)}")
        if context_hint:
            parts.append(f"Context: {context_hint}")
        parts.append("")

        # DOC API articles (full data)
        articles = raw_data.get("doc_articles", [])
        if articles:
            parts.append(f"=== GDELT DOC API Articles ({len(articles)} total) ===")
            for i, article in enumerate(articles):
                title = article.get("title", "No title")
                source = article.get("source", "unknown")
                tone = article.get("tone", 0)
                seendate = article.get("seendate", "")
                url = article.get("url", "")
                parts.append(
                    f"[{i+1}] {title}\n"
                    f"    Source: {source} | Tone: {tone} | Date: {seendate}"
                )
                if url:
                    parts.append(f"    URL: {url}")
            parts.append("")

        # Tone summary
        tone_summary = raw_data.get("tone_summary", {})
        if tone_summary:
            parts.append(f"=== Tone Summary ===")
            parts.append(
                f"Avg tone: {tone_summary.get('avg_tone', 0)}, "
                f"Positive: {tone_summary.get('positive_count', 0)}, "
                f"Negative: {tone_summary.get('negative_count', 0)}, "
                f"Neutral: {tone_summary.get('neutral_count', 0)}"
            )
            parts.append("")

        # GKG entity/theme data (if available)
        gkg_themes = raw_data.get("gkg_themes", [])
        if gkg_themes:
            parts.append(f"=== GKG Themes (BigQuery) ===")
            for theme in gkg_themes[:15]:
                parts.append(f"  {theme.get('theme', '')}: {theme.get('count', 0)} mentions")
            parts.append("")

        gkg_persons = raw_data.get("gkg_persons", [])
        if gkg_persons:
            parts.append(f"=== GKG Persons ===")
            for person in gkg_persons[:10]:
                parts.append(f"  {person.get('person', '')}: {person.get('count', 0)} mentions")
            parts.append("")

        gkg_orgs = raw_data.get("gkg_organizations", [])
        if gkg_orgs:
            parts.append(f"=== GKG Organizations ===")
            for org in gkg_orgs[:10]:
                parts.append(f"  {org.get('org', '')}: {org.get('count', 0)} mentions")
            parts.append("")

        parts.append(
            f"Source diversity: {raw_data.get('source_diversity', 0)} unique outlets, "
            f"Total articles: {raw_data.get('article_count', 0)}"
        )

        return "\n".join(parts)

    # =========================================================================
    # Sub-agent call
    # =========================================================================

    async def _call_sub_agent(
        self,
        prompt: str,
        model: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Call the sub-agent model and parse structured JSON response.

        Returns parsed intelligence dict, or None on failure.
        """
        if not self._anthropic:
            logger.warning("[news_analyzer] Anthropic client not available")
            return None

        try:
            response = await self._anthropic.messages.create(
                model=model,
                max_tokens=self._config.max_tokens,
                temperature=self._config.temperature,
                system=SUB_AGENT_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )

            # Track token usage
            usage = response.usage
            input_tokens = getattr(usage, "input_tokens", 0)
            output_tokens = getattr(usage, "output_tokens", 0)
            self._total_input_tokens += input_tokens
            self._total_output_tokens += output_tokens

            # Report to parent agent's token tracker
            if self._token_usage_callback:
                self._token_usage_callback(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cache_read=getattr(usage, "cache_read_input_tokens", 0),
                    cache_created=getattr(usage, "cache_creation_input_tokens", 0),
                )

            # Extract text content
            raw_text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    raw_text += block.text

            if not raw_text.strip():
                logger.warning("[news_analyzer] Empty response from sub-agent")
                return None

            return self._parse_json_response(raw_text)

        except Exception as e:
            logger.error(f"[news_analyzer] Sub-agent call failed ({model}): {e}")
            return None

    def _parse_json_response(self, raw_text: str) -> Optional[Dict[str, Any]]:
        """Parse JSON from sub-agent response, with fallback strategies."""
        text = raw_text.strip()

        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Strip markdown code blocks
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # Find largest {...} block
        brace_start = text.find("{")
        if brace_start >= 0:
            # Find matching closing brace
            depth = 0
            last_close = -1
            for i in range(brace_start, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        last_close = i
                        break
            if last_close > brace_start:
                try:
                    return json.loads(text[brace_start:last_close + 1])
                except json.JSONDecodeError:
                    pass

        logger.warning(f"[news_analyzer] Failed to parse JSON from response: {text[:200]}")
        return None

    # =========================================================================
    # Fallback / degraded results
    # =========================================================================

    def _empty_intelligence(self, search_terms: List[str]) -> Dict[str, Any]:
        """Return empty intelligence when no articles found."""
        return {
            "narrative_summary": f"No news coverage found for: {', '.join(search_terms)}",
            "key_developments": [],
            "sentiment": {
                "overall": "neutral",
                "trend": "stable",
                "avg_tone": 0.0,
                "confidence": "low",
            },
            "source_analysis": {
                "total_articles": 0,
                "unique_sources": 0,
                "notable_sources": [],
                "geographic_spread": "domestic",
            },
            "market_signals": [],
            "freshness": {
                "newest_article_age_minutes": None,
                "coverage_window_hours": 4,
                "is_breaking": False,
                "volume_trend": "declining",
            },
            "trading_recommendation": "no_signal",
        }

    def _degraded_intelligence(
        self,
        raw_data: Dict[str, Any],
        search_terms: List[str],
    ) -> Dict[str, Any]:
        """Build degraded intelligence from raw stats (no AI analysis)."""
        tone = raw_data.get("tone_summary", {})
        avg_tone = tone.get("avg_tone", 0.0)
        article_count = raw_data.get("article_count", 0)
        source_diversity = raw_data.get("source_diversity", 0)

        # Determine sentiment from raw tone
        if avg_tone > 3.0:
            overall = "strongly_positive"
        elif avg_tone > 1.5:
            overall = "positive"
        elif avg_tone < -3.0:
            overall = "strongly_negative"
        elif avg_tone < -1.5:
            overall = "negative"
        else:
            overall = "mixed" if abs(avg_tone) > 0.5 else "neutral"

        # Extract notable sources from articles
        articles = raw_data.get("doc_articles", [])
        sources = list({a.get("source", "") for a in articles if a.get("source")})[:5]

        return {
            "narrative_summary": (
                f"[DEGRADED - raw stats only] {article_count} articles from "
                f"{source_diversity} sources. Avg tone: {avg_tone:.1f}. "
                f"AI analysis unavailable."
            ),
            "key_developments": [],
            "sentiment": {
                "overall": overall,
                "trend": "stable",
                "avg_tone": round(avg_tone, 2),
                "confidence": "low",
            },
            "source_analysis": {
                "total_articles": article_count,
                "unique_sources": source_diversity,
                "notable_sources": sources,
                "geographic_spread": "domestic",
            },
            "market_signals": [],
            "freshness": {
                "newest_article_age_minutes": None,
                "coverage_window_hours": 4,
                "is_breaking": False,
                "volume_trend": "steady",
            },
            "trading_recommendation": "no_signal",
        }

    # =========================================================================
    # Helpers
    # =========================================================================

    def _make_result(
        self,
        status: str,
        search_terms: List[str],
        intelligence: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build a standardized result dict."""
        result = {
            "status": status,
            "search_terms": search_terms,
            "intelligence": intelligence or self._empty_intelligence(search_terms),
            "metadata": {
                "analyzer_model": self._config.model,
                "analysis_time_ms": 0,
                "raw_article_count": 0,
                "cached": status == "cached",
                "cache_expires_at": None,
                "cost_estimate_usd": 0.0,
            },
        }
        if error:
            result["error"] = error
        return result

    def _estimate_cost(self, prompt: str, intelligence: Optional[Dict[str, Any]]) -> float:
        """Estimate USD cost of the sub-agent call."""
        # Rough token estimates: ~4 chars per token
        input_tokens = len(SUB_AGENT_SYSTEM_PROMPT + prompt) / 4
        output_tokens = len(json.dumps(intelligence or {})) / 4

        # Haiku pricing: $0.80 / 1M input, $4.00 / 1M output
        cost = (input_tokens * 0.80 + output_tokens * 4.0) / 1_000_000
        return cost

    def get_cost_stats(self) -> Dict[str, Any]:
        """Get token/cost monitoring stats."""
        # Haiku pricing
        input_cost = self._total_input_tokens * 0.80 / 1_000_000
        output_cost = self._total_output_tokens * 4.0 / 1_000_000

        return {
            "total_calls": self._total_calls,
            "cache_hits": self._total_cache_hits,
            "errors": self._total_errors,
            "cache_hit_rate": (
                round(self._total_cache_hits / (self._total_calls + self._total_cache_hits), 2)
                if (self._total_calls + self._total_cache_hits) > 0
                else 0.0
            ),
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "estimated_cost_usd": round(input_cost + output_cost, 4),
            "cache_entries": len(self._cache),
            "cache_ttl_seconds": self._config.cache_ttl_seconds,
            "model": self._config.model,
        }
