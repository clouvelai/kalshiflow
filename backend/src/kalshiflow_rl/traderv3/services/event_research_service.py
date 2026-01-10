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

        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are a prediction market analyst. Today is {current_date}.

Analyze this event deeply. Extract THREE things:
1. EVENT CONTEXT - What is this event about?
2. KEY DRIVER - What single factor determines the outcome?
3. SEMANTIC FRAME - The structural understanding (WHO has agency, WHAT they're deciding, WHICH choices exist)

For SEMANTIC FRAME, identify the type:
- NOMINATION: Actor DECIDES who gets position (e.g., "Trump nominates Fed Chair")
- COMPETITION: Entities compete, rules determine winner (e.g., "Chiefs vs Bills", elections)
- ACHIEVEMENT: Binary yes/no on reaching milestone (e.g., "Team wins championship")
- OCCURRENCE: Event happens or doesn't, no threshold (e.g., "Rain in NYC", "Government shutdown")
- MEASUREMENT: Numeric value compared to threshold (e.g., "CPI over 3%", "BTC above $100k")
- MENTION: Speech act - someone says/references something (e.g., "Will X mention Y?")

DISAMBIGUATION - when multiple seem to fit, choose MOST SPECIFIC:
- "BTC above $100k" = MEASUREMENT (numeric threshold), NOT ACHIEVEMENT
- "Team wins championship" = COMPETITION (vs other teams), NOT ACHIEVEMENT
- "Fed cuts rates" = OCCURRENCE (yes/no event), NOT MEASUREMENT
- "Trump picks Warsh" = NOMINATION (decision by actor), NOT OCCURRENCE

For CANDIDATES, link each possible outcome to its market ticker if applicable."""),

            ("user", """EVENT: {event_title}
CATEGORY: {category}

MARKETS ({num_markets} total - each market represents one possible outcome):
{market_list}

Analyze:

1. EVENT CONTEXT:
   - Brief description (1-2 sentences)
   - Core question being predicted
   - Resolution criteria (how YES/NO determined)
   - Resolution type: objective/subjective/mixed
   - Time horizon

2. KEY DRIVER:
   - Primary driver (the ONE factor that matters most)
   - Why this is the key driver (causal mechanism)
   - 2-3 secondary factors
   - Tail risks
   - Base rate for similar events (with reasoning)

3. SEMANTIC FRAME:
   - Frame type: NOMINATION / COMPETITION / ACHIEVEMENT / OCCURRENCE / MEASUREMENT / MENTION
   - Question template (e.g., "{{actor}} nominates {{candidate}} for {{position}}")
   - Primary relation (the verb: nominates, defeats, exceeds, says, etc.)
   - ACTORS: Who has agency? (name, entity_type, role, aliases)
   - OBJECTS: What's being acted upon? (name, entity_type) - leave empty if candidates ARE the objects

   - CANDIDATES: **You MUST create exactly {num_markets} candidate entries - one for each market above.**
     For each market ticker, provide:
     - canonical_name: The specific outcome this market represents (extract from market title)
     - market_ticker: MUST match the ticker exactly (copy from MARKETS list above)
     - aliases: Alternative names, abbreviations, variations for news matching
     - search_queries: 2-3 targeted news search queries for this specific outcome

   CANDIDATE EXAMPLES:
   For NOMINATION event "Trump's Fed Chair Nominee" with markets:
   - KXFEDCHAIRNOM-WARSH -> canonical_name: "Kevin Warsh", market_ticker: "KXFEDCHAIRNOM-WARSH", aliases: ["Warsh", "K. Warsh"]
   - KXFEDCHAIRNOM-HASSETT -> canonical_name: "Kevin Hassett", market_ticker: "KXFEDCHAIRNOM-HASSETT", aliases: ["Hassett"]

   For COMPETITION event "NFL Week 15: Chiefs vs Bills" with markets:
   - KXNFL-CHIEFS-WIN -> canonical_name: "Kansas City Chiefs win", market_ticker: "KXNFL-CHIEFS-WIN", aliases: ["KC Chiefs", "Chiefs"]
   - KXNFL-BILLS-WIN -> canonical_name: "Buffalo Bills win", market_ticker: "KXNFL-BILLS-WIN", aliases: ["Buffalo", "Bills"]

   - Does one actor control the outcome? (true/false)
   - Are outcomes mutually exclusive? (true if only one can win, false if multiple can resolve YES)
   - Resolution trigger (what event determines outcome)
   - 3-5 targeted search queries for finding news
   - Signal keywords (words that indicate important news)""")
        ])

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
        if not self._web_search_enabled or not self._search_tool:
            return Evidence(
                evidence_summary="Web search not available",
                reliability=EvidenceReliability.LOW,
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
        # Build market descriptions - NO PRICES (blind estimation)
        market_descriptions = []
        for market in markets:
            desc = f"MARKET: {market.ticker}\n"
            desc += f"TITLE: {market.title}\n"
            # NOTE: We deliberately DO NOT include market prices here
            # The LLM must estimate probabilities blind, then guess market price
            market_descriptions.append(desc)

        markets_text = "\n---\n".join(market_descriptions)

        from datetime import datetime
        current_date = datetime.now().strftime("%B %d, %Y")

        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are a prediction market TRADER looking for PROFITABLE opportunities. Today is {current_date}.

YOUR GOAL: Make money by finding markets where you have an edge over the market price.

HOW PREDICTION MARKETS WORK:
- Contracts pay $1 (100c) if YES resolves, $0 if NO resolves
- Buy YES at Xc → profit (100-X)c if YES, lose Xc if NO
- Buy NO at Yc → profit (100-Y)c if NO, lose Yc if YES
- EXPECTED VALUE: EV = P(correct) × profit - P(wrong) × loss
- Only bet when EV > 0 AND you have sufficient confidence

For each market, you will complete TWO SEPARATE tasks:

TASK 1 - YOUR PROBABILITY ESTIMATE:
Based ONLY on the evidence and reasoning, estimate the TRUE probability.
This is YOUR view - what you would bet your own money on.
Do NOT anchor on what the market might believe.

TASK 2 - MARKET PRICE GUESS:
AFTER forming your probability, separately guess what price the market is trading at.
This tests calibration - we compare your guess to actual price.
You will NOT be told actual prices.

WHAT MAKES MONEY:
- If you think YES is more likely than the price implies → buy YES
- If you think NO is more likely than the price implies → buy NO
- You don't need a huge edge - being right slightly more often = profit over time

KEY QUESTION: "Do I think this resolves YES or NO, and am I more confident than the market?" """),
            ("user", """EVENT RESEARCH:
{event_context}

KEY DRIVER: {primary_driver}

MARKETS TO EVALUATE:
{markets_text}

For EACH market, complete these steps IN ORDER:

1. UNDERSTAND: What specific question does this market ask?

2. APPLY: How does the key driver apply to THIS specific market?

3. YOUR ESTIMATE (Task 1): What is YOUR probability estimate for YES?
   - Range: 0.0 to 1.0
   - Base this ONLY on evidence and reasoning
   - Include 2-3 sentences of reasoning

4. MARKET GUESS (Task 2): What price (0-100 cents) do you think this market is trading at?
   - This is your guess of what OTHER traders believe
   - May differ from your estimate if you think market is mispriced

5. CONFIDENCE: How confident are you in your estimate?
   - high: Strong evidence, clear reasoning, 80%+ certainty
   - medium: Reasonable evidence, some uncertainty, 60-80% certainty
   - low: Weak or conflicting evidence, <60% certainty""")
        ])

        try:
            chain = prompt | self._llm.with_structured_output(BatchMarketAssessmentOutput)
            result = await self._invoke_with_retry(chain, {
                "event_context": research_context.to_prompt_string(),
                "markets_text": markets_text,
                "primary_driver": research_context.driver_analysis.primary_driver,
            })

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
                yes_bid = market.yes_bid or 0
                yes_ask = market.yes_ask or 100
                actual_mid_price = (yes_bid + yes_ask) / 2
                market_prob = actual_mid_price / 100.0

                # LLM's estimates (blind)
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

                # Extract v2 calibration fields with safe defaults
                evidence_cited = getattr(llm_assessment, 'evidence_cited', []) or []
                what_would_change_mind = getattr(llm_assessment, 'what_would_change_mind', "") or ""
                assumption_flags = getattr(llm_assessment, 'assumption_flags', []) or []
                calibration_notes = getattr(llm_assessment, 'calibration_notes', "") or ""
                evidence_quality = getattr(llm_assessment, 'evidence_quality', "medium") or "medium"

                assessment = MarketAssessment(
                    market_ticker=market.ticker,
                    market_title=market.title,
                    specific_question=llm_assessment.specific_question,
                    driver_application=llm_assessment.driver_application,
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
