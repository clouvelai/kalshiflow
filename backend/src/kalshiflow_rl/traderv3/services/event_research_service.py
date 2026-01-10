"""
Event Research Service - Event-First Agentic Research Pipeline.

This service implements the event-first research architecture, which researches events
holistically before evaluating individual markets. This approach provides deeper understanding
through first-principles reasoning about what drives outcomes.

Architecture:
    Phase 1: EVENT DISCOVERY - Group tracked markets by event_ticker
    Phase 2: EVENT RESEARCH - "What is this event about?"
    Phase 3: KEY DRIVER IDENTIFICATION - "What single factor determines YES vs NO?"
    Phase 4: EVIDENCE GATHERING - Targeted search for key driver data
    Phase 5: MARKET EVALUATION - Batch assess all markets with shared event context
    Phase 6: TRADE DECISIONS - Execute on mispriced markets

Design Principles:
    - Event context is shared across all markets in an event
    - Key driver analysis uses first-principles reasoning
    - Evidence is gathered specifically for the identified key driver
    - Market evaluation includes microstructure signals
    - LLM prompts guide reasoning structure without being overly prescriptive

Usage:
    service = EventResearchService(trading_client)
    result = await service.research_event(event_ticker, markets, microstructure)
    for assessment in result.get_tradeable_assessments():
        # Execute trades
"""

import asyncio
import logging
import os
import re
import time
from typing import Dict, Any, Optional, List, Tuple, TYPE_CHECKING
from urllib.parse import urlparse

from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from pydantic import BaseModel, Field

from ..state.event_research_context import (
    EventContext,
    KeyDriverAnalysis,
    Evidence,
    EventResearchContext,
    MarketAssessment,
    EventResearchResult,
    EvidenceReliability,
    Confidence,
    FrameType,
    SemanticRole,
    SemanticFrame,
)

# Import versioned prompts and schemas
from ..prompts import (
    get_event_context_prompt,
    get_market_eval_prompt,
    ACTIVE_EVENT_CONTEXT_VERSION,
    ACTIVE_MARKET_EVAL_VERSION,
)
from ..prompts.schemas import (
    FullContextOutput,
    SemanticRoleOutput,
    SingleMarketAssessment,
    BatchMarketAssessmentOutput,
)
from ..prompts.schemas.market_assessment_v4_schema import (
    SingleMarketAssessmentV4,
    BatchMarketAssessmentV4Output,
)

if TYPE_CHECKING:
    from ..state.tracked_markets import TrackedMarket
    from ..state.microstructure_context import MicrostructureContext

logger = logging.getLogger("kalshiflow_rl.traderv3.services.event_research")


# === Source Quality Heuristics ===
# Trusted domains for evidence quality assessment

TRUSTED_DOMAINS = {
    # Major news agencies
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk",
    # Business/financial news
    "bloomberg.com", "wsj.com", "ft.com", "cnbc.com", "marketwatch.com",
    "finance.yahoo.com", "seekingalpha.com",
    # Sports
    "espn.com", "sports.yahoo.com", "nfl.com", "nba.com", "mlb.com",
    "theathleticdiscover.com", "theathletic.com",
    # Politics/policy
    "politico.com", "thehill.com", "rollcall.com", "c-span.org",
    # Government/official
    "whitehouse.gov", "federalreserve.gov", "sec.gov", "bls.gov",
    "congress.gov", "state.gov",
    # Data sources
    "tradingview.com", "coinmarketcap.com", "coingecko.com",
    # Wire services
    "prnewswire.com", "businesswire.com",
}

MEDIUM_TRUST_DOMAINS = {
    # General news
    "cnn.com", "nytimes.com", "washingtonpost.com", "theguardian.com",
    "usatoday.com", "latimes.com", "foxnews.com", "nbcnews.com",
    # Tech/business
    "techcrunch.com", "wired.com", "arstechnica.com", "venturebeat.com",
    # Crypto-specific
    "coindesk.com", "cointelegraph.com", "theblock.co",
    # Wikipedia (good for background, not breaking news)
    "wikipedia.org",
}


def assess_source_quality(url: str, text: str = "") -> Tuple[str, float]:
    """
    Assess the quality of a source based on domain and content heuristics.

    Args:
        url: Source URL
        text: Optional text content for additional heuristics

    Returns:
        Tuple of (quality_level, confidence_score)
        quality_level: "high", "medium", or "low"
        confidence_score: 0.0 to 1.0
    """
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()

        # Remove www. prefix
        if domain.startswith("www."):
            domain = domain[4:]

        # Check trusted domains
        if domain in TRUSTED_DOMAINS:
            return ("high", 0.9)

        # Check medium trust domains
        if domain in MEDIUM_TRUST_DOMAINS:
            return ("medium", 0.7)

        # Check for .gov or .edu domains
        if domain.endswith(".gov") or domain.endswith(".edu"):
            return ("high", 0.85)

        # Check for known low-quality indicators
        low_quality_indicators = [
            "blog", "forum", "reddit", "twitter", "x.com",
            "facebook", "tiktok", "youtube",
        ]
        for indicator in low_quality_indicators:
            if indicator in domain:
                return ("low", 0.3)

        # Default to medium for unknown sources
        return ("medium", 0.5)

    except Exception:
        return ("low", 0.3)


def generate_semantic_frame_queries(
    frame_type: FrameType,
    event_title: str,
    primary_driver: str,
    candidates: List[SemanticRole] = None,
    actors: List[SemanticRole] = None,
) -> List[str]:
    """
    Generate targeted search queries based on semantic frame type.

    Different frame types benefit from different query structures:
    - NOMINATION: Focus on actor intentions, candidate news, shortlist reports
    - COMPETITION: Focus on matchup, performance, odds, predictions
    - MEASUREMENT: Focus on data sources, forecasts, current values
    - OCCURRENCE: Focus on news about the event happening or not

    Args:
        frame_type: Type of semantic frame
        event_title: Event title for context
        primary_driver: Key driver from analysis
        candidates: List of candidate roles (for NOMINATION/COMPETITION)
        actors: List of actor roles (for NOMINATION)

    Returns:
        List of targeted search queries
    """
    queries = []

    if frame_type == FrameType.NOMINATION:
        # NOMINATION: Actor decides who gets position
        # Focus on: actor statements, shortlist reports, candidate news
        if actors:
            for actor in actors[:2]:  # Top 2 actors
                queries.append(f"{actor.canonical_name} {event_title} announcement")
                queries.append(f"{actor.canonical_name} nominee decision latest")
        if candidates:
            for candidate in candidates[:3]:  # Top 3 candidates
                queries.append(f"{candidate.canonical_name} frontrunner {event_title}")
        queries.append(f"{event_title} shortlist candidates")
        queries.append(f"{event_title} nomination timeline")

    elif frame_type == FrameType.COMPETITION:
        # COMPETITION: Entities compete, rules determine winner
        # Focus on: matchup analysis, recent performance, expert predictions
        if candidates:
            # Get matchup queries
            if len(candidates) >= 2:
                queries.append(
                    f"{candidates[0].canonical_name} vs {candidates[1].canonical_name} prediction"
                )
            for candidate in candidates[:3]:
                queries.append(f"{candidate.canonical_name} recent performance stats")
        queries.append(f"{event_title} odds prediction")
        queries.append(f"{event_title} expert analysis")

    elif frame_type == FrameType.MEASUREMENT:
        # MEASUREMENT: Numeric value vs threshold
        # Focus on: data sources, forecasts, current readings
        queries.append(f"{primary_driver} forecast prediction")
        queries.append(f"{primary_driver} latest data reading")
        queries.append(f"{event_title} consensus estimate")
        queries.append(f"{primary_driver} analyst expectations")

    elif frame_type == FrameType.OCCURRENCE:
        # OCCURRENCE: Event happens or doesn't
        # Focus on: likelihood, conditions, news
        queries.append(f"{event_title} likelihood")
        queries.append(f"{primary_driver} latest news")
        queries.append(f"{event_title} will it happen")

    elif frame_type == FrameType.ACHIEVEMENT:
        # ACHIEVEMENT: Subject reaches milestone
        queries.append(f"{event_title} progress toward")
        queries.append(f"{primary_driver} current status")

    elif frame_type == FrameType.MENTION:
        # MENTION: Someone says/references something
        queries.append(f"{event_title} speech transcript")
        queries.append(f"{event_title} expected remarks")

    # Always add a general query
    queries.append(f"{event_title} news today")

    # Deduplicate and limit
    seen = set()
    unique_queries = []
    for q in queries:
        q_lower = q.lower()
        if q_lower not in seen:
            seen.add(q_lower)
            unique_queries.append(q)

    return unique_queries[:5]  # Return top 5 queries


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    """Deduplicate a list while preserving order."""
    seen = set()
    out: List[str] = []
    for item in items:
        if not item:
            continue
        key = item.strip()
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


class EventResearchService:
    """
    Orchestrates event-first research pipeline.

    This service implements a 6-phase research approach that first understands
    the event holistically, identifies key drivers, gathers targeted evidence,
    then evaluates all markets in batch with shared context.
    """

    def __init__(
        self,
        trading_client: Any = None,
        openai_api_key: Optional[str] = None,
        openai_model: str = "gpt-4o",
        openai_temperature: float = 0.3,
        web_search_enabled: bool = True,
        min_mispricing_threshold: float = 0.10,
        cache_ttl_seconds: float = 3600.0,
    ):
        """
        Initialize event research service.

        Args:
            trading_client: Kalshi trading client for REST API calls
            openai_api_key: OpenAI API key (if None, loads from OPENAI_API_KEY env var)
            openai_model: OpenAI model to use (default: "gpt-4o")
            openai_temperature: Temperature for LLM (default: 0.3 for consistent reasoning)
            web_search_enabled: Enable web search for evidence gathering
            min_mispricing_threshold: Minimum mispricing to flag as tradeable (default: 10%)
            cache_ttl_seconds: TTL for cached semantic frames (default: 1 hour)
        """
        self._trading_client = trading_client
        self._api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var.")

        self._model = openai_model
        self._temperature = openai_temperature
        self._web_search_enabled = web_search_enabled
        self._min_mispricing = min_mispricing_threshold

        # Truth Social evidence (cache-backed; optional)
        # Cache service is initialized globally and accessed lazily via get_truth_social_cache()
        # We check availability lazily in _gather_truth_social_evidence() since cache may start after EventResearchService is created
        self._truth_social_cache = None  # Will be fetched lazily

        # Initialize LLM
        self._llm = ChatOpenAI(
            model=self._model,
            temperature=self._temperature,
            api_key=self._api_key,
        )

        # Initialize web search
        self._search_tool = None
        if self._web_search_enabled:
            try:
                self._search_tool = DuckDuckGoSearchRun()
            except Exception as e:
                logger.warning(f"Failed to initialize web search: {e}")
                self._web_search_enabled = False

        # Stats
        self._events_researched = 0
        self._total_llm_calls = 0
        self._total_research_seconds = 0.0

        # Parse failure tracking (for prompt quality monitoring)
        self._parse_failures = 0
        self._parse_repairs = 0
        self._prompt_version_event_context = ACTIVE_EVENT_CONTEXT_VERSION
        self._prompt_version_market_eval = ACTIVE_MARKET_EVAL_VERSION

        # Semantic frame cache
        self._frame_cache: Dict[str, EventResearchContext] = {}
        self._cache_ttl_seconds = cache_ttl_seconds
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_lock = asyncio.Lock()  # Thread safety for cache operations

        logger.info(
            f"EventResearchService initialized "
            f"(model={openai_model}, web_search={web_search_enabled}, cache_ttl={cache_ttl_seconds}s)"
        )

    async def _invoke_with_retry(self, chain, params: dict, max_retries: int = 3):
        """Invoke LLM chain with exponential backoff retry."""
        import random
        for attempt in range(max_retries):
            try:
                return await chain.ainvoke(params)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                wait = min(2 ** attempt + random.uniform(0, 1), 30)
                logger.warning(f"LLM retry {attempt+1}/{max_retries} after {wait:.1f}s: {e}")
                await asyncio.sleep(wait)

    # === Semantic Frame Caching ===

    def _get_cached_context(self, event_ticker: str) -> Optional[EventResearchContext]:
        """
        Check if we have a valid cached context for this event.

        Checks memory cache first, then falls back to database.

        Args:
            event_ticker: Event ticker to look up

        Returns:
            Cached EventResearchContext if valid, None otherwise
        """
        # Check memory cache first
        if event_ticker in self._frame_cache:
            cached = self._frame_cache[event_ticker]

            # Check TTL if semantic frame has extracted_at
            if cached.semantic_frame and cached.semantic_frame.extracted_at:
                age = time.time() - cached.semantic_frame.extracted_at
                # Handle future timestamp (clock skew) - treat as invalid
                if age < 0:
                    logger.warning(f"[CACHE INVALID] Future timestamp for {event_ticker} (age={age:.0f}s)")
                    del self._frame_cache[event_ticker]
                    return None
                if age > self._cache_ttl_seconds:
                    logger.info(f"[CACHE EXPIRED] {event_ticker} (age={age:.0f}s > ttl={self._cache_ttl_seconds}s)")
                    del self._frame_cache[event_ticker]
                    # Fall through to check database
                else:
                    return cached
            else:
                return cached

        # Memory cache miss - no need to check DB here, will be checked asynchronously
        return None

    async def _get_cached_context_async(self, event_ticker: str) -> Optional[EventResearchContext]:
        """
        Check if we have a valid cached context for this event.

        Checks memory cache first (with lock), then falls back to database.
        Thread-safe via asyncio.Lock for memory cache operations.

        Args:
            event_ticker: Event ticker to look up

        Returns:
            Cached EventResearchContext if valid, None otherwise
        """
        # Check memory cache first (with lock for thread safety)
        async with self._cache_lock:
            cached = self._get_cached_context(event_ticker)
            if cached:
                return cached

        # Memory cache miss - check database (outside lock - non-blocking I/O)
        db_context = await self._load_semantic_frame_from_db(event_ticker)
        if db_context:
            # Check TTL
            if db_context.semantic_frame and db_context.semantic_frame.extracted_at:
                age = time.time() - db_context.semantic_frame.extracted_at
                if age > self._cache_ttl_seconds:
                    logger.info(f"[DB CACHE EXPIRED] {event_ticker} (age={age:.0f}s > ttl={self._cache_ttl_seconds}s)")
                    return None

            # Populate memory cache (with lock for thread safety)
            async with self._cache_lock:
                self._frame_cache[event_ticker] = db_context
            logger.info(f"[DB CACHE HIT] Loaded semantic frame from database for {event_ticker}")
            return db_context

        return None

    async def _cache_context(self, event_ticker: str, context: EventResearchContext) -> None:
        """
        Cache an event research context.

        Thread-safe via asyncio.Lock.

        Args:
            event_ticker: Event ticker
            context: Complete research context to cache
        """
        # Set extraction timestamp if semantic frame exists
        if context.semantic_frame and not context.semantic_frame.extracted_at:
            context.semantic_frame.extracted_at = time.time()

        context.cached = True
        async with self._cache_lock:
            self._frame_cache[event_ticker] = context
        logger.info(f"[CACHE STORE] Cached semantic frame for {event_ticker}")

    async def _persist_semantic_frame(self, context: EventResearchContext) -> bool:
        """
        Persist semantic frame to database for recovery across restarts.

        Args:
            context: EventResearchContext with semantic frame to persist

        Returns:
            True if persisted successfully, False otherwise
        """
        if not context.semantic_frame:
            return False

        try:
            # Import Supabase client (lazy import to avoid circular deps)
            from supabase import create_client, Client
            from datetime import datetime, timezone

            supabase_url = os.getenv("SUPABASE_URL")
            supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY")

            if not supabase_url or not supabase_key:
                logger.debug("Supabase not configured, skipping persistence")
                return False

            supabase: Client = create_client(supabase_url, supabase_key)

            frame = context.semantic_frame
            # Note: JSONB columns receive native Python dicts/lists, not JSON strings
            data = {
                "event_ticker": frame.event_ticker,
                "frame_type": frame.frame_type.value,
                "question_template": frame.question_template,
                "primary_relation": frame.primary_relation,
                "actors": [a.to_dict() for a in frame.actors],  # Native list for JSONB
                "objects": [o.to_dict() for o in frame.objects],  # Native list for JSONB
                "candidates": [c.to_dict() for c in frame.candidates],  # Native list for JSONB
                "mutual_exclusivity": frame.mutual_exclusivity,
                "actor_controls_outcome": frame.actor_controls_outcome,
                "resolution_trigger": frame.resolution_trigger,
                "primary_search_queries": frame.primary_search_queries,
                "signal_keywords": frame.signal_keywords,
                "event_context": context.context.to_dict() if context.context else None,
                "key_driver_analysis": context.driver_analysis.to_dict() if context.driver_analysis else None,
                "extracted_at": datetime.fromtimestamp(frame.extracted_at, tz=timezone.utc).isoformat() if frame.extracted_at else None,
                "event_title": context.event_title,
                "event_category": context.event_category,
            }

            # Upsert (insert or update on conflict)
            result = supabase.table("semantic_frames").upsert(data).execute()

            if result.data:
                logger.info(f"[DB PERSIST] Saved semantic frame for {frame.event_ticker}")
                return True
            else:
                logger.warning(f"[DB PERSIST] No data returned for {frame.event_ticker}")
                return False

        except Exception as e:
            logger.warning(f"[DB PERSIST] Failed to save semantic frame: {e}")
            return False

    async def _load_semantic_frame_from_db(self, event_ticker: str) -> Optional[EventResearchContext]:
        """
        Load semantic frame from database.

        Args:
            event_ticker: Event ticker to load

        Returns:
            EventResearchContext if found, None otherwise
        """
        try:
            from supabase import create_client, Client
            from datetime import datetime

            supabase_url = os.getenv("SUPABASE_URL")
            supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY")

            if not supabase_url or not supabase_key:
                return None

            supabase: Client = create_client(supabase_url, supabase_key)

            result = supabase.table("semantic_frames").select("*").eq("event_ticker", event_ticker).execute()

            if not result.data:
                return None

            row = result.data[0]

            # Reconstruct SemanticFrame
            frame_type = FrameType(row["frame_type"]) if row.get("frame_type") in [e.value for e in FrameType] else FrameType.UNKNOWN

            # JSONB columns return native Python dicts/lists, not JSON strings
            actors_raw = row.get("actors") or []
            objects_raw = row.get("objects") or []
            candidates_raw = row.get("candidates") or []

            actors = [SemanticRole.from_dict(a) for a in actors_raw]
            objects = [SemanticRole.from_dict(o) for o in objects_raw]
            candidates = [SemanticRole.from_dict(c) for c in candidates_raw]

            # Parse extracted_at
            extracted_at = None
            if row.get("extracted_at"):
                try:
                    dt = datetime.fromisoformat(row["extracted_at"].replace("Z", "+00:00"))
                    extracted_at = dt.timestamp()
                except Exception:
                    pass

            semantic_frame = SemanticFrame(
                event_ticker=event_ticker,
                frame_type=frame_type,
                question_template=row.get("question_template", ""),
                primary_relation=row.get("primary_relation", ""),
                actors=actors,
                objects=objects,
                candidates=candidates,
                mutual_exclusivity=row.get("mutual_exclusivity", True),
                actor_controls_outcome=row.get("actor_controls_outcome", False),
                resolution_trigger=row.get("resolution_trigger", ""),
                primary_search_queries=row.get("primary_search_queries", []),
                signal_keywords=row.get("signal_keywords", []),
                extracted_at=extracted_at,
            )

            # Reconstruct EventContext if available
            event_context = None
            if row.get("event_context"):
                ctx = row["event_context"]
                event_context = EventContext(
                    event_description=ctx.get("event_description", ""),
                    core_question=ctx.get("core_question", ""),
                    resolution_criteria=ctx.get("resolution_criteria", ""),
                    resolution_objectivity=ctx.get("resolution_objectivity", "unknown"),
                    time_horizon=ctx.get("time_horizon", ""),
                )

            # Reconstruct KeyDriverAnalysis if available
            driver_analysis = None
            if row.get("key_driver_analysis"):
                kda = row["key_driver_analysis"]
                driver_analysis = KeyDriverAnalysis(
                    primary_driver=kda.get("primary_driver", ""),
                    primary_driver_reasoning=kda.get("primary_driver_reasoning", ""),
                    causal_chain=kda.get("causal_chain", ""),
                    secondary_factors=kda.get("secondary_factors", []),
                    tail_risks=kda.get("tail_risks", []),
                    base_rate=kda.get("base_rate", 0.5),
                    base_rate_reasoning=kda.get("base_rate_reasoning", ""),
                )

            return EventResearchContext(
                event_ticker=event_ticker,
                event_title=row.get("event_title", event_ticker),
                event_category=row.get("event_category", "Unknown"),
                context=event_context,
                driver_analysis=driver_analysis,
                semantic_frame=semantic_frame,
                cached=True,
            )

        except Exception as e:
            logger.warning(f"[DB LOAD] Failed to load semantic frame for {event_ticker}: {e}")
            return None

    async def _extract_full_context(
        self,
        event_ticker: str,
        event_title: str,
        event_category: str,
        markets: List["TrackedMarket"],
    ) -> EventResearchContext:
        """
        Combined extraction: EventContext + KeyDriverAnalysis + SemanticFrame
        in a single LLM call for efficiency.

        This replaces the separate Phase 2 and Phase 3 calls when cache miss occurs.

        Args:
            event_ticker: Event ticker
            event_title: Event title
            event_category: Event category
            markets: List of markets in this event (for candidate linking)

        Returns:
            Complete EventResearchContext with semantic frame
        """
        from datetime import datetime
        current_date = datetime.now().strftime("%B %d, %Y")

        # Build market list for candidate extraction
        market_list = "\n".join([
            f"- {m.ticker}: {m.title}"
            for m in markets
        ])

        num_markets = len(markets)

        # Use versioned prompt (v2 includes GROUNDING REQUIREMENTS, BASE RATE ANCHORING, CALIBRATION)
        prompt = get_event_context_prompt(current_date)
        logger.info(f"[PROMPT] Using event_context prompt version: {ACTIVE_EVENT_CONTEXT_VERSION}")

        try:
            chain = prompt | self._llm.with_structured_output(FullContextOutput)
            result = await self._invoke_with_retry(chain, {
                "event_title": event_title,
                "category": event_category,
                "market_list": market_list,
                "num_markets": num_markets,
            })

            # Build EventContext from result
            event_context = EventContext(
                event_description=result.event_description,
                core_question=result.core_question,
                resolution_criteria=result.resolution_criteria,
                resolution_objectivity=result.resolution_objectivity,
                time_horizon=result.time_horizon,
            )

            # Build KeyDriverAnalysis from result
            driver_analysis = KeyDriverAnalysis(
                primary_driver=result.primary_driver,
                primary_driver_reasoning=result.primary_driver_reasoning,
                causal_chain=result.causal_chain,
                secondary_factors=result.secondary_factors,
                tail_risks=result.tail_risks,
                base_rate=result.base_rate,
                base_rate_reasoning=result.base_rate_reasoning,
                edge_hypothesis=getattr(result, 'edge_hypothesis', "") or "",
            )

            # Build SemanticFrame from result
            frame_type = FrameType(result.frame_type.lower()) if result.frame_type.lower() in [e.value for e in FrameType] else FrameType.UNKNOWN

            # Convert actors (defensive null check in case LLM returns None)
            actors = [
                SemanticRole(
                    entity_id=a.entity_id,
                    canonical_name=a.canonical_name,
                    entity_type=a.entity_type,
                    role=a.role,
                    role_description=a.role_description or "",
                    market_ticker=a.market_ticker,
                    aliases=a.aliases or [],
                    search_queries=a.search_queries or [],
                )
                for a in (result.actors or [])
            ]

            # Convert objects
            objects = [
                SemanticRole(
                    entity_id=o.entity_id,
                    canonical_name=o.canonical_name,
                    entity_type=o.entity_type,
                    role=o.role,
                    role_description=o.role_description or "",
                    market_ticker=o.market_ticker,
                    aliases=o.aliases or [],
                    search_queries=o.search_queries or [],
                )
                for o in (result.objects or [])
            ]

            # Convert candidates
            candidates = [
                SemanticRole(
                    entity_id=c.entity_id,
                    canonical_name=c.canonical_name,
                    entity_type=c.entity_type,
                    role=c.role,
                    role_description=c.role_description or "",
                    market_ticker=c.market_ticker,
                    aliases=c.aliases or [],
                    search_queries=c.search_queries or [],
                )
                for c in (result.candidates or [])
            ]

            # Fallback: Ensure every market has a candidate entry
            extracted_tickers = {c.market_ticker for c in candidates if c.market_ticker}
            missing_count = 0
            for market in markets:
                if market.ticker not in extracted_tickers:
                    missing_count += 1
                    candidates.append(SemanticRole(
                        entity_id=f"MARKET:{market.ticker}",
                        canonical_name=market.title,
                        entity_type="OUTCOME",
                        role="candidate",
                        role_description=f"Outcome for market {market.ticker}",
                        market_ticker=market.ticker,
                        aliases=[],
                        search_queries=[],
                    ))

            if missing_count > 0:
                logger.info(
                    f"[CANDIDATE FALLBACK] Added {missing_count} missing candidates for {event_ticker} "
                    f"(LLM extracted {len(candidates) - missing_count}/{num_markets})"
                )

            semantic_frame = SemanticFrame(
                event_ticker=event_ticker,
                frame_type=frame_type,
                question_template=result.question_template or "",
                primary_relation=result.primary_relation or "",
                actors=actors,
                objects=objects,
                candidates=candidates,
                mutual_exclusivity=result.mutual_exclusivity,
                actor_controls_outcome=result.actor_controls_outcome,
                resolution_trigger=result.resolution_trigger or "",
                primary_search_queries=result.primary_search_queries or [],
                signal_keywords=result.signal_keywords or [],
                extracted_at=time.time(),
            )

            logger.info(
                f"[PHASE 2+3] Combined extraction complete for {event_ticker}: "
                f"frame_type={frame_type.value}, "
                f"actors={len(actors)}, candidates={len(candidates)}, "
                f"search_queries={len(result.primary_search_queries)}"
            )

            return EventResearchContext(
                event_ticker=event_ticker,
                event_title=event_title,
                event_category=event_category,
                context=event_context,
                driver_analysis=driver_analysis,
                semantic_frame=semantic_frame,
                market_tickers=[m.ticker for m in markets],
                llm_calls_made=1,  # Combined call
                cached=False,
            )

        except Exception as e:
            logger.error(f"Combined context extraction failed for {event_ticker}: {e}", exc_info=True)
            # Fallback to basic context
            return EventResearchContext(
                event_ticker=event_ticker,
                event_title=event_title,
                event_category=event_category,
                context=EventContext(
                    event_description=f"Event: {event_title}",
                    core_question=event_title,
                    resolution_criteria="Unknown",
                    resolution_objectivity="unknown",
                    time_horizon="Unknown",
                ),
                driver_analysis=KeyDriverAnalysis(
                    primary_driver="Unknown",
                    primary_driver_reasoning="Extraction failed",
                    causal_chain="Unknown",
                ),
                market_tickers=[m.ticker for m in markets],
                cached=False,
            )

    async def research_event(
        self,
        event_ticker: str,
        markets: List["TrackedMarket"],
        microstructure: Optional[Dict[str, "MicrostructureContext"]] = None,
    ) -> EventResearchResult:
        """
        Execute full event research pipeline.

        This is the main entry point. It runs all 6 phases:
        1. Fetch event details from REST API
        2. LLM: Understand event context
        3. LLM: Identify key driver
        4. Web search: Gather evidence for key driver
        5. LLM: Evaluate all markets in batch
        6. Return assessments

        Args:
            event_ticker: The event ticker (e.g., "KXNFL-25JAN05")
            markets: List of TrackedMarket objects in this event
            microstructure: Optional dict of market_ticker -> MicrostructureContext

        Returns:
            EventResearchResult with event context and per-market assessments
        """
        start_time = time.time()
        llm_calls = 0

        try:
            logger.info(f"Starting event research for {event_ticker} ({len(markets)} markets)")

            # Phase 1: Fetch event details
            event_details = await self._fetch_event_details(event_ticker)
            if not event_details:
                logger.warning(f"Could not fetch event details for {event_ticker}")
                # Use first market's info as fallback
                event_details = {
                    "event_ticker": event_ticker,
                    "title": markets[0].title if markets else event_ticker,
                    "category": markets[0].category if markets else "Unknown",
                }

            event_title = event_details.get("title") or event_ticker  # Handle empty string
            event_category = event_details.get("category") or "Unknown"

            logger.info(f"[PHASE 1] Fetched event details: {event_title}")

            # Enrich TrackedMarket titles from event's nested market data
            # Note: market "title" field is DEPRECATED in Kalshi API - use yes_sub_title instead
            event_markets = event_details.get("markets", [])
            if event_markets:
                market_lookup = {m.get("ticker"): m for m in event_markets}

                enriched_count = 0
                for tracked_market in markets:
                    if tracked_market.ticker in market_lookup:
                        api_market = market_lookup[tracked_market.ticker]

                        # Use yes_sub_title as primary name (title is DEPRECATED in Kalshi API)
                        if not tracked_market.title:
                            tracked_market.title = (
                                api_market.get("yes_sub_title") or
                                api_market.get("title") or  # Fallback to deprecated field
                                api_market.get("subtitle") or
                                ""
                            )

                        # Populate additional rich market fields
                        tracked_market.yes_sub_title = api_market.get("yes_sub_title", "")
                        tracked_market.no_sub_title = api_market.get("no_sub_title", "")
                        tracked_market.subtitle = api_market.get("subtitle", "")
                        tracked_market.rules_primary = api_market.get("rules_primary", "")
                        tracked_market.primary_participant_key = api_market.get("primary_participant_key", "")

                        if tracked_market.title:
                            enriched_count += 1

                if enriched_count > 0:
                    logger.info(f"[TITLE ENRICHMENT] Enriched {enriched_count}/{len(markets)} markets from event data")

            # Check cache before Phase 2+3 (memory + database)
            cached_context = await self._get_cached_context_async(event_ticker)

            if cached_context:
                # CACHE HIT: Skip Phase 2+3, use cached context
                self._cache_hits += 1
                logger.info(
                    f"[CACHE HIT] Using cached context for {event_ticker} "
                    f"(frame_type={cached_context.semantic_frame.frame_type.value if cached_context.semantic_frame else 'none'})"
                )

                # Update market tickers in case they changed
                cached_context.market_tickers = [m.ticker for m in markets]
                research_context = cached_context

            else:
                # CACHE MISS: Combined extraction (Phase 2+3 in one LLM call)
                self._cache_misses += 1
                logger.info(f"[CACHE MISS] Extracting semantic frame for {event_ticker}")

                research_context = await self._extract_full_context(
                    event_ticker=event_ticker,
                    event_title=event_title,
                    event_category=event_category,
                    markets=markets,
                )
                llm_calls += 1  # Combined call counts as 1

                # Cache the result for future calls (memory)
                await self._cache_context(event_ticker, research_context)

                # Persist to database for recovery across restarts (async, non-blocking)
                # Add error callback to log failures without blocking
                def _on_persist_done(task: asyncio.Task):
                    if task.exception():
                        logger.warning(f"[DB PERSIST] Background persistence failed: {task.exception()}")

                persist_task = asyncio.create_task(self._persist_semantic_frame(research_context))
                persist_task.add_done_callback(_on_persist_done)

            # Phase 4: Gather evidence (uses semantic frame queries if available)
            logger.info(f"[PHASE 4] Gathering evidence for {event_ticker}")
            evidence = await self._gather_evidence(
                driver_analysis=research_context.driver_analysis,
                event_title=event_title,
                semantic_frame=research_context.semantic_frame,
            )
            # No LLM call for web search
            logger.info(
                f"[PHASE 4] Evidence gathered: {len(evidence.sources)} sources, "
                f"reliability={evidence.reliability.value}"
            )

            # Update research context with evidence
            research_context.evidence = evidence

            # Phase 5: Evaluate markets in batch
            logger.info(f"[PHASE 5] Evaluating {len(markets)} markets for {event_ticker}")
            assessments = await self._evaluate_markets_batch(
                research_context=research_context,
                markets=markets,
                microstructure=microstructure or {},
            )
            llm_calls += 1

            # Count markets with edge for logging
            num_with_edge = sum(
                1 for a in assessments
                if abs(a.mispricing_magnitude) >= self._min_mispricing
            )
            logger.info(
                f"[PHASE 5] Batch assessment complete: {len(assessments)} markets, "
                f"{num_with_edge} with edge"
            )

            # Calculate stats
            duration = time.time() - start_time
            research_context.research_duration_seconds = duration
            research_context.llm_calls_made = llm_calls

            # Count markets with edge
            markets_with_edge = sum(
                1 for a in assessments
                if abs(a.mispricing_magnitude) >= self._min_mispricing
            )

            result = EventResearchResult(
                event_context=research_context,
                assessments=assessments,
                markets_evaluated=len(assessments),
                markets_with_edge=markets_with_edge,
                total_research_seconds=duration,
                success=True,
            )

            # Update stats
            self._events_researched += 1
            self._total_llm_calls += llm_calls
            self._total_research_seconds += duration

            logger.info(
                f"Event research complete for {event_ticker}: "
                f"{len(assessments)} markets evaluated, "
                f"{markets_with_edge} with edge, "
                f"{duration:.1f}s duration, "
                f"{llm_calls} LLM calls"
            )

            return result

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Event research failed for {event_ticker}: {e}", exc_info=True)

            # Return failure result
            return EventResearchResult(
                event_context=EventResearchContext(
                    event_ticker=event_ticker,
                    event_title=event_ticker,
                    event_category="Unknown",
                ),
                success=False,
                error_message=str(e),
                total_research_seconds=duration,
            )

    async def _fetch_event_details(self, event_ticker: str) -> Dict[str, Any]:
        """
        Phase 1: Fetch event details from REST API.

        Args:
            event_ticker: Event ticker to fetch

        Returns:
            Event details dict or empty dict if unavailable
        """
        if not self._trading_client:
            logger.debug("No trading client available for event fetch")
            return {}

        try:
            event = await self._trading_client.get_event(event_ticker)
            return event or {}
        except Exception as e:
            logger.warning(f"Failed to fetch event {event_ticker}: {e}")
            return {}

    # Note: _research_event_context() and _identify_key_driver() methods were removed.
    # They are superseded by _extract_full_context() which combines Phase 2 (context),
    # Phase 3 (key driver), and semantic frame extraction into a single efficient LLM call.

    async def _gather_evidence(
        self,
        driver_analysis: KeyDriverAnalysis,
        event_title: str,
        semantic_frame: Optional[SemanticFrame] = None,
    ) -> Evidence:
        """
        Phase 4: Use web search to gather evidence about the key driver.

        Uses semantic frame-aware query generation when frame is available:
        - NOMINATION: Actor intentions, candidate news, shortlist reports
        - COMPETITION: Matchup analysis, performance stats, predictions
        - MEASUREMENT: Data sources, forecasts, current readings
        - OCCURRENCE: Event likelihood, conditions, news

        Also applies source quality heuristics to assess evidence reliability.

        Args:
            driver_analysis: Phase 3 output with key driver
            event_title: Event title for context
            semantic_frame: Optional semantic frame with targeted queries

        Returns:
            Evidence with key facts and reliability assessment
        """
        # Run evidence tools in parallel (web + truth social). Each tool is best-effort.
        web_task = asyncio.create_task(
            self._gather_web_evidence(driver_analysis, event_title, semantic_frame)
        )
        truth_task = asyncio.create_task(
            self._gather_truth_social_evidence(driver_analysis, event_title, semantic_frame)
        )

        web_ev, truth_ev = await asyncio.gather(web_task, truth_task)

        # Merge into a single Evidence object (keep the existing downstream contract)
        merged = Evidence()
        merged.key_evidence = _dedupe_preserve_order((web_ev.key_evidence or []) + (truth_ev.key_evidence or []))[:10]
        merged.sources = _dedupe_preserve_order((web_ev.sources or []) + (truth_ev.sources or []))[:10]
        merged.sources_checked = int(web_ev.sources_checked or 0) + int(truth_ev.sources_checked or 0)

        # Tag evidence by source so LLM can distinguish provenance
        summary_parts = []
        if web_ev.evidence_summary:
            summary_parts.append(f"[WEB] {web_ev.evidence_summary.strip()}")
        if truth_ev.evidence_summary:
            summary_parts.append(f"[TRUTH SOCIAL] {truth_ev.evidence_summary.strip()}")
        merged.evidence_summary = "\n\n".join(summary_parts)[:900]

        # Reliability: take max of component reliabilities
        reliability_order = {
            EvidenceReliability.LOW: 1,
            EvidenceReliability.MEDIUM: 2,
            EvidenceReliability.HIGH: 3,
        }
        best = EvidenceReliability.LOW
        for r in (web_ev.reliability, truth_ev.reliability):
            if reliability_order.get(r, 0) > reliability_order.get(best, 0):
                best = r
        merged.reliability = best

        reasons = []
        if web_ev.reliability_reasoning:
            reasons.append(f"web: {web_ev.reliability_reasoning}")
        if truth_ev.reliability_reasoning:
            reasons.append(f"truth: {truth_ev.reliability_reasoning}")
        merged.reliability_reasoning = " | ".join([r.strip() for r in reasons if r.strip()])[:800]

        # Leave evidence_probability as default for now (not used in current Phase 5 prompt)
        
        # Merge metadata: combine truth_social metadata from both sources
        merged_metadata = {}
        if web_ev.metadata:
            merged_metadata.update(web_ev.metadata)
        if truth_ev.metadata:
            # Merge truth_social metadata specially
            if "truth_social" in merged_metadata and "truth_social" in truth_ev.metadata:
                # Merge nested truth_social dicts
                merged_truth = {**merged_metadata.get("truth_social", {}), **truth_ev.metadata["truth_social"]}
                merged_metadata["truth_social"] = merged_truth
            else:
                merged_metadata.update(truth_ev.metadata)
        
        merged.metadata = merged_metadata
        return merged

    async def _gather_web_evidence(
        self,
        driver_analysis: KeyDriverAnalysis,
        event_title: str,
        semantic_frame: Optional[SemanticFrame] = None,
    ) -> Evidence:
        """Web evidence gatherer (existing implementation extracted for composition)."""
        if not self._web_search_enabled or not self._search_tool:
            return Evidence(
                evidence_summary="Web search not available",
                reliability=EvidenceReliability.LOW,
                reliability_reasoning="web search disabled/unavailable",
            )

        try:
            # Use semantic frame-aware query generation (improved over v1)
            if semantic_frame and semantic_frame.primary_search_queries:
                # Primary: Use cached semantic frame queries
                search_queries = semantic_frame.primary_search_queries[:3]
                logger.info(f"[PHASE 4] Using semantic frame queries: {search_queries}")

                # Secondary: Generate frame-type-specific queries if needed
                if len(search_queries) < 3:
                    frame_queries = generate_semantic_frame_queries(
                        frame_type=semantic_frame.frame_type,
                        event_title=event_title,
                        primary_driver=driver_analysis.primary_driver,
                        candidates=semantic_frame.candidates,
                        actors=semantic_frame.actors,
                    )
                    # Add unique queries
                    existing = set(q.lower() for q in search_queries)
                    for q in frame_queries:
                        if q.lower() not in existing:
                            search_queries.append(q)
                            existing.add(q.lower())
                        if len(search_queries) >= 5:
                            break
                    logger.info(f"[PHASE 4] Enhanced with frame-type queries: {search_queries}")
            else:
                # Fallback to generic query
                search_queries = [f"{event_title} {driver_analysis.primary_driver}"]
                search_queries.append(f"{event_title} news today")

            # Run searches for all queries in parallel
            loop = asyncio.get_running_loop()

            async def _run_search(query: str) -> Optional[str]:
                """Execute a single search with timeout and error handling."""
                try:
                    result = await asyncio.wait_for(
                        loop.run_in_executor(None, self._search_tool.run, query),
                        timeout=15.0  # Shorter timeout per query
                    )
                    return result
                except asyncio.TimeoutError:
                    logger.warning(f"Search timed out for query: {query}")
                    return None
                except Exception as e:
                    logger.warning(f"Search failed for query '{query}': {e}")
                    return None

            # Execute all searches in parallel using asyncio.gather
            search_tasks = [_run_search(query) for query in search_queries]
            results = await asyncio.gather(*search_tasks)

            # Filter out None results from failed/timed-out searches
            all_results = [r for r in results if r is not None]

            # Combine results
            search_result = "\n\n".join(all_results) if all_results else ""

            # Parse results
            key_evidence = []
            sources = []

            # Extract key points (split by sentences, take meaningful ones)
            sentences = re.split(r'[.!?]+', search_result)
            for sentence in sentences[:10]:  # Top 10 sentences
                sentence = sentence.strip()
                if len(sentence) > 30:  # Skip very short fragments
                    key_evidence.append(sentence)

            # Extract URLs
            urls = re.findall(
                r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                search_result
            )
            sources.extend(urls[:5])

            # === Source Quality Assessment (v2 improvement) ===
            high_quality_sources = 0
            medium_quality_sources = 0
            low_quality_sources = 0
            source_quality_details = []

            for url in sources:
                quality, score = assess_source_quality(url)
                source_quality_details.append((url, quality, score))
                if quality == "high":
                    high_quality_sources += 1
                elif quality == "medium":
                    medium_quality_sources += 1
                else:
                    low_quality_sources += 1

            # Assess reliability based on evidence quantity AND source quality
            reliability = EvidenceReliability.MEDIUM
            if len(key_evidence) >= 5 and high_quality_sources >= 1:
                reliability = EvidenceReliability.HIGH
            elif len(key_evidence) >= 3 and (high_quality_sources + medium_quality_sources) >= 2:
                reliability = EvidenceReliability.HIGH
            elif len(key_evidence) < 2 or (high_quality_sources == 0 and medium_quality_sources == 0):
                reliability = EvidenceReliability.LOW

            # Build detailed reliability reasoning
            quality_breakdown = f"Sources: {high_quality_sources} high, {medium_quality_sources} medium, {low_quality_sources} low quality"
            reliability_reasoning = (
                f"Found {len(key_evidence)} evidence points from {len(sources)} sources "
                f"({len(all_results)} queries). {quality_breakdown}"
            )

            return Evidence(
                key_evidence=key_evidence[:5],
                evidence_summary=search_result[:500] if search_result else "No evidence found",
                sources=sources,
                sources_checked=len(all_results),
                reliability=reliability,
                reliability_reasoning=reliability_reasoning,
            )

        except asyncio.TimeoutError:
            logger.warning(f"Web search timed out for: {event_title}")
            return Evidence(
                evidence_summary="Web search timed out after 30 seconds",
                reliability=EvidenceReliability.LOW,
            )
        except Exception as e:
            logger.warning(f"Evidence gathering failed: {e}")
            return Evidence(
                evidence_summary=f"Search failed: {e}",
                reliability=EvidenceReliability.LOW,
                reliability_reasoning="web search failed",
            )

    async def _gather_truth_social_evidence(
        self,
        driver_analysis: KeyDriverAnalysis,
        event_title: str,
        semantic_frame: Optional[SemanticFrame] = None,
    ) -> Evidence:
        """
        Evidence gatherer from Truth Social cache (following-based "Trump circle").

        Uses the global TruthSocialCacheService which maintains a cache of posts from
        the authenticated user's following list (auto-discovered) plus trending data.

        This is intentionally conservative:
        - Treat as an "intent / narrative" signal, not independently verified fact.
        - Always returns an Evidence object; never raises.
        """
        # Lazy-load cache (may be None if not initialized or unavailable)
        if self._truth_social_cache is None:
            from .truth_social_cache import get_truth_social_cache
            self._truth_social_cache = get_truth_social_cache()

        if not self._truth_social_cache or not self._truth_social_cache.is_available():
            logger.debug(f"[TRUTH SOCIAL] {event_title[:40]}: cache unavailable or disabled")
            return Evidence(
                evidence_summary="Truth Social evidence disabled or unavailable",
                reliability=EvidenceReliability.LOW,
                reliability_reasoning="truth social cache not available",
                metadata={"truth_social": {"status": "unavailable"}},
            )

        try:
            from .truth_social_evidence_tool import TruthSocialEvidenceTool

            # Build search queries from semantic frame and event context
            queries = []
            if semantic_frame and semantic_frame.primary_search_queries:
                queries.extend(semantic_frame.primary_search_queries[:3])
            
            # Add driver and event title as additional keywords
            if driver_analysis.primary_driver:
                queries.append(driver_analysis.primary_driver)
            if event_title:
                queries.append(event_title)

            # Use the evidence tool to query cache
            evidence_tool = TruthSocialEvidenceTool(cache_service=self._truth_social_cache)
            evidence = evidence_tool.gather(
                event_title=event_title,
                primary_driver=driver_analysis.primary_driver or "",
                queries=queries,
                hours_back=self._truth_social_cache.hours_back if self._truth_social_cache else None,
                max_items=25,
            )

            # Log Truth Social evidence results for diagnostics
            ts_meta = evidence.metadata.get("truth_social", {})
            ts_status = ts_meta.get("status", "unknown")
            ts_posts = ts_meta.get("posts_found", 0)
            ts_authors = ts_meta.get("unique_authors", 0)
            logger.info(
                f"[TRUTH SOCIAL] {event_title[:40]}: "
                f"status={ts_status}, posts={ts_posts}, authors={ts_authors}, "
                f"queries={queries[:3]}"
            )

            return evidence

        except Exception as e:
            logger.warning(f"Truth Social evidence gathering failed: {e}", exc_info=True)
            return Evidence(
                evidence_summary=f"Truth Social evidence error: {str(e)[:100]}",
                reliability=EvidenceReliability.LOW,
                reliability_reasoning=f"truth social evidence tool error: {type(e).__name__}",
                metadata={"truth_social": {"status": "error", "error": str(e)[:200]}},
            )

    async def _evaluate_markets_batch(
        self,
        research_context: EventResearchContext,
        markets: List["TrackedMarket"],
        microstructure: Dict[str, "MicrostructureContext"],
    ) -> List[MarketAssessment]:
        """
        Phase 5: Evaluate all markets in batch with shared event context.

        This is efficient because the event research (Phases 2-4) is shared
        across all markets, and we evaluate them together in one LLM call.

        Args:
            research_context: Complete event research context
            markets: List of markets to evaluate
            microstructure: Dict of market_ticker -> MicrostructureContext

        Returns:
            List of MarketAssessment for each market
        """
        # Import classifier for clean signal formatting (v3 prompt)
        from .microstructure_classifier import MicrostructureClassifier
        classifier = MicrostructureClassifier()

        # Build market descriptions - NO PRICES (blind estimation)
        # NOTE: We deliberately DO NOT include market prices here
        # The LLM must estimate probabilities blind, then guess market price
        market_descriptions = []
        microstructure_summaries = []  # Collect for batch-level summary

        for market in markets:
            desc = f"MARKET: {market.ticker}\n"
            desc += f"TITLE: {market.title}\n"

            # Include microstructure context if available
            # Use classifier for clean, factual signal formatting (v3)
            if microstructure and market.ticker in microstructure:
                micro_ctx = microstructure[market.ticker]
                # Use classifier for cleaner, categorized output
                classified_signals = classifier.classify_to_prompt_string(micro_ctx)
                desc += f"\nMARKET SIGNALS:\n{classified_signals}\n"
                microstructure_summaries.append(f"{market.ticker}:\n{classified_signals}")

            market_descriptions.append(desc)

        markets_text = "\n---\n".join(market_descriptions)

        # Build batch-level microstructure signals summary for v3 prompt
        if microstructure_summaries:
            microstructure_signals = "\n\n".join(microstructure_summaries)
        else:
            microstructure_signals = "No microstructure signals available"

        from datetime import datetime
        current_date = datetime.now().strftime("%B %d, %Y")

        # Use versioned prompt (v2 includes BASE RATE ANCHORING, CALIBRATION SELF-CHECK, evidence citation)
        prompt = get_market_eval_prompt(current_date)

        # Get base rate from driver analysis (default to 0.5 if not available)
        base_rate = 0.5
        if research_context.driver_analysis and research_context.driver_analysis.base_rate:
            base_rate = research_context.driver_analysis.base_rate

        logger.info(
            f"[PROMPT] Using market_eval prompt version: {ACTIVE_MARKET_EVAL_VERSION}, "
            f"base_rate={base_rate:.0%}"
        )

        try:
            # Select schema based on active prompt version
            if ACTIVE_MARKET_EVAL_VERSION == "v4":
                output_schema = BatchMarketAssessmentV4Output
            else:
                output_schema = BatchMarketAssessmentOutput

            chain = prompt | self._llm.with_structured_output(output_schema)
            # Defensive access for driver_analysis in case of error fallback
            primary_driver = (
                research_context.driver_analysis.primary_driver
                if research_context.driver_analysis else "Unknown"
            )
            result = await self._invoke_with_retry(chain, {
                "event_context": research_context.to_prompt_string(),
                "markets_text": markets_text,
                "primary_driver": primary_driver,
                "base_rate": base_rate,  # CRITICAL: Pass base rate for v2/v3/v4 prompt anchoring
                "microstructure_signals": microstructure_signals,  # v3/v4.1: Batch-level classified signals
            })

            # Log mutual exclusivity warning if present (V4.1)
            mutual_warning = getattr(result, 'mutual_exclusivity_warning', None)
            if mutual_warning:
                logger.warning(f"[BATCH CONSISTENCY] {research_context.event_ticker}: {mutual_warning}")

            # Convert to MarketAssessment objects
            assessments = []
            for llm_assessment in result.assessments:
                # Find matching market
                market = next(
                    (m for m in markets if m.ticker == llm_assessment.market_ticker),
                    None
                )
                if not market:
                    continue

                # Get ACTUAL market price (mid-price)
                # Note: yes_bid=0 and yes_ask=0 means no orderbook data (not "price is 0")
                # In this case, fall back to market.price (last traded price) or 50c default
                yes_bid = market.yes_bid
                yes_ask = market.yes_ask

                # Calculate mid-price with proper fallback logic
                if yes_bid > 0 and yes_ask > 0:
                    # Both bid and ask available - use mid
                    actual_mid_price = (yes_bid + yes_ask) / 2
                elif yes_bid > 0:
                    # Only bid available - use bid
                    actual_mid_price = yes_bid
                elif yes_ask > 0:
                    # Only ask available - use ask
                    actual_mid_price = yes_ask
                elif market.price > 0:
                    # No bid/ask but have last traded price
                    actual_mid_price = market.price
                    logger.warning(
                        f"[PRICE] {market.ticker}: No bid/ask data, using last price {market.price}c"
                    )
                else:
                    # No price data at all - skip this market for now
                    logger.warning(
                        f"[PRICE] {market.ticker}: No price data available (bid={yes_bid}, ask={yes_ask}, price={market.price}), skipping"
                    )
                    continue

                market_prob = actual_mid_price / 100.0

                # LLM's estimates (blind)
                # V4 uses different field names - handle both
                if ACTIVE_MARKET_EVAL_VERSION == "v4":
                    evidence_prob = llm_assessment.probability
                    estimated_price = llm_assessment.market_price_guess
                else:
                    evidence_prob = llm_assessment.evidence_probability
                    estimated_price = llm_assessment.estimated_market_price

                # CALIBRATION: How well did LLM guess the market price?
                price_guess_error = estimated_price - actual_mid_price

                # EDGE: Difference between LLM's view and actual market
                mispricing = evidence_prob - market_prob

                # Log calibration for tracking
                logger.info(
                    f"[CALIBRATION] {market.ticker}: "
                    f"LLM_prob={evidence_prob:.0%} | "
                    f"LLM_guessed_price={estimated_price}c | "
                    f"actual_price={actual_mid_price:.0f}c | "
                    f"guess_error={price_guess_error:+.0f}c | "
                    f"edge={mispricing:+.1%}"
                )

                # Determine recommendation based on mispricing magnitude
                # (Still generates HOLD for logging, but plugin won't skip on it)
                if mispricing > 0.05:  # LLM thinks YES is underpriced
                    recommendation = "BUY_YES"
                elif mispricing < -0.05:  # LLM thinks NO is underpriced
                    recommendation = "BUY_NO"
                else:
                    recommendation = "HOLD"  # Small edge - logged but still traded

                # Map confidence
                confidence_map = {
                    "high": Confidence.HIGH,
                    "medium": Confidence.MEDIUM,
                    "low": Confidence.LOW,
                }
                confidence = confidence_map.get(
                    llm_assessment.confidence.lower(),
                    Confidence.MEDIUM
                )

                # Extract calibration fields with safe defaults (V4 uses different names)
                if ACTIVE_MARKET_EVAL_VERSION == "v4":
                    evidence_cited = getattr(llm_assessment, 'key_evidence', []) or []
                    what_would_change_mind = getattr(llm_assessment, 'what_would_change_mind', "") or ""
                    assumption_flags = []  # V4 doesn't have this
                    calibration_notes = ""  # V4 doesn't have this
                    # V4 derives evidence_quality from confidence
                    confidence_str = getattr(llm_assessment, 'confidence', "medium") or "medium"
                    evidence_quality = confidence_str
                else:
                    evidence_cited = getattr(llm_assessment, 'evidence_cited', []) or []
                    what_would_change_mind = getattr(llm_assessment, 'what_would_change_mind', "") or ""
                    assumption_flags = getattr(llm_assessment, 'assumption_flags', []) or []
                    calibration_notes = getattr(llm_assessment, 'calibration_notes', "") or ""
                    evidence_quality = getattr(llm_assessment, 'evidence_quality', "medium") or "medium"

                # Extract v3 base rate anchoring fields with safe defaults
                base_rate_used = getattr(llm_assessment, 'base_rate_used', base_rate) or base_rate
                adjustment_up = getattr(llm_assessment, 'adjustment_up', 0.0) or 0.0
                adjustment_up_reasoning = getattr(llm_assessment, 'adjustment_up_reasoning', "") or ""
                adjustment_down = getattr(llm_assessment, 'adjustment_down', 0.0) or 0.0
                adjustment_down_reasoning = getattr(llm_assessment, 'adjustment_down_reasoning', "") or ""

                # Log base rate anchoring for tracking
                if adjustment_up > 0 or adjustment_down > 0:
                    expected = base_rate_used + adjustment_up - adjustment_down
                    logger.info(
                        f"[BASE_RATE] {market.ticker}: "
                        f"base={base_rate_used:.0%} +{adjustment_up:.0%} -{adjustment_down:.0%} "
                        f"= expected {expected:.0%} vs actual {evidence_prob:.0%}"
                    )

                # V4.1 restored specific_question, driver_application still not present
                if ACTIVE_MARKET_EVAL_VERSION == "v4":
                    specific_question = getattr(llm_assessment, 'specific_question', "") or ""
                    driver_application = ""  # V4 doesn't have driver_application, derive from reasoning if needed
                else:
                    specific_question = getattr(llm_assessment, 'specific_question', "") or ""
                    driver_application = getattr(llm_assessment, 'driver_application', "") or ""

                # Log extreme probability flags for monitoring (V4.1)
                extreme_flag = getattr(llm_assessment, 'extreme_probability_flag', None)
                if extreme_flag:
                    logger.warning(
                        f"[EXTREME PROBABILITY] {market.ticker}: {extreme_flag}"
                    )

                assessment = MarketAssessment(
                    market_ticker=market.ticker,
                    market_title=market.title,
                    specific_question=specific_question,
                    driver_application=driver_application,
                    evidence_probability=evidence_prob,
                    market_probability=market_prob,
                    mispricing_magnitude=mispricing,
                    price_guess_cents=int(estimated_price),  # LLM's blind guess
                    price_guess_error_cents=int(price_guess_error),  # guess - actual
                    recommendation=recommendation,
                    confidence=confidence,
                    edge_explanation=llm_assessment.reasoning,
                    microstructure_summary="",  # Simplified for MVP
                    # v2 calibration fields
                    evidence_cited=evidence_cited,
                    what_would_change_mind=what_would_change_mind,
                    assumption_flags=assumption_flags,
                    calibration_notes=calibration_notes,
                    evidence_quality=evidence_quality,
                    # v3 base rate anchoring fields
                    base_rate_used=base_rate_used,
                    adjustment_up=adjustment_up,
                    adjustment_up_reasoning=adjustment_up_reasoning,
                    adjustment_down=adjustment_down,
                    adjustment_down_reasoning=adjustment_down_reasoning,
                    # v4.1 extreme probability monitoring
                    extreme_probability_flag=extreme_flag,
                )
                assessments.append(assessment)

            return assessments

        except Exception as e:
            logger.error(f"Batch market evaluation failed: {e}", exc_info=True)
            return []

    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        total_cache_requests = self._cache_hits + self._cache_misses
        total_parse_attempts = self._parse_failures + self._parse_repairs + self._total_llm_calls
        return {
            "events_researched": self._events_researched,
            "total_llm_calls": self._total_llm_calls,
            "total_research_seconds": round(self._total_research_seconds, 1),
            "avg_seconds_per_event": (
                round(self._total_research_seconds / self._events_researched, 1)
                if self._events_researched > 0 else 0
            ),
            "web_search_enabled": self._web_search_enabled,
            "model": self._model,
            "min_mispricing_threshold": self._min_mispricing,
            # Cache statistics
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate": (
                round(self._cache_hits / total_cache_requests, 2)
                if total_cache_requests > 0 else 0.0
            ),
            "frames_cached": len(self._frame_cache),
            "cache_ttl_seconds": self._cache_ttl_seconds,
            # Parse failure tracking (for prompt quality monitoring)
            "parse_failures": self._parse_failures,
            "parse_repairs": self._parse_repairs,
            "parse_failure_rate": (
                round(self._parse_failures / total_parse_attempts, 3)
                if total_parse_attempts > 0 else 0.0
            ),
            # Prompt version tracking
            "prompt_version_event_context": self._prompt_version_event_context,
            "prompt_version_market_eval": self._prompt_version_market_eval,
        }
