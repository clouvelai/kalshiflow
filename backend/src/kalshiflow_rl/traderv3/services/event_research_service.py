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
from typing import Dict, Any, Optional, List, TYPE_CHECKING

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
)

if TYPE_CHECKING:
    from ..state.tracked_markets import TrackedMarket
    from ..state.microstructure_context import MicrostructureContext

logger = logging.getLogger("kalshiflow_rl.traderv3.services.event_research")


# === Pydantic models for structured LLM output ===

class EventContextOutput(BaseModel):
    """Structured output for Phase 2: Event understanding."""
    event_description: str = Field(
        description="2-3 sentence description of what this event is about"
    )
    core_question: str = Field(
        description="The fundamental question being asked (what outcome is being predicted?)"
    )
    resolution_criteria: str = Field(
        description="How will YES vs NO be determined? What specific criteria?"
    )
    resolution_objectivity: str = Field(
        description="Is resolution objective (clear data), subjective (judgment), or mixed? One of: objective, subjective, mixed"
    )
    time_horizon: str = Field(
        description="When will we know the outcome? Include key dates if relevant"
    )


class KeyDriverOutput(BaseModel):
    """Structured output for Phase 3: Key driver identification."""
    primary_driver: str = Field(
        description="The single most important factor determining YES vs NO. Be specific and measurable."
    )
    primary_driver_reasoning: str = Field(
        description="Why is this the key driver? What's the causal chain from driver to outcome?"
    )
    causal_chain: str = Field(
        description="The step-by-step mechanism: how does the driver lead to the outcome?"
    )
    secondary_factors: List[str] = Field(
        description="2-3 other factors that matter, in order of importance"
    )
    tail_risks: List[str] = Field(
        description="Low-probability events that could dramatically change the outcome"
    )
    base_rate: float = Field(
        description="Historical frequency of YES for similar events (0.0 to 1.0)",
        ge=0.0, le=1.0
    )
    base_rate_reasoning: str = Field(
        description="How did you determine the base rate? What counts as 'similar events'?"
    )


class SingleMarketAssessment(BaseModel):
    """Assessment for a single market within an event."""
    market_ticker: str = Field(description="The market ticker being assessed")
    specific_question: str = Field(
        description="What specific question does this market ask?"
    )
    driver_application: str = Field(
        description="How does the primary driver apply to THIS specific market?"
    )
    evidence_probability: float = Field(
        description="Your probability estimate for YES based on evidence (0.0 to 1.0)",
        ge=0.0, le=1.0
    )
    estimated_market_price: int = Field(
        description="What price (in cents, 0-100) do you think this market is CURRENTLY trading at? This is your guess of what the market believes.",
        ge=0, le=100
    )
    confidence: str = Field(
        description="Confidence in this assessment: high, medium, or low"
    )
    reasoning: str = Field(
        description="Brief reasoning for your probability estimate (2-3 sentences max)"
    )


class BatchMarketAssessmentOutput(BaseModel):
    """Structured output for Phase 5: Batch market evaluation."""
    assessments: List[SingleMarketAssessment] = Field(
        description="Assessment for each market in the event"
    )


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

        logger.info(
            f"EventResearchService initialized "
            f"(model={openai_model}, web_search={web_search_enabled})"
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

            event_title = event_details.get("title", event_ticker)
            event_category = event_details.get("category", "Unknown")

            logger.info(f"[PHASE 1] Fetched event details: {event_title}")

            # Phase 2: Understand event context
            logger.info(f"[PHASE 2] Understanding event context for {event_ticker}")
            event_context = await self._research_event_context(
                event_ticker=event_ticker,
                event_title=event_title,
                event_category=event_category,
                market_count=len(markets),
            )
            llm_calls += 1
            logger.info(f"[PHASE 2] Event context generated: {event_context.core_question[:50]}...")

            # Phase 3: Identify key driver
            logger.info(f"[PHASE 3] Identifying key driver for {event_ticker}")
            driver_analysis = await self._identify_key_driver(
                event_context=event_context,
                event_title=event_title,
                event_category=event_category,
            )
            llm_calls += 1
            logger.info(f"[PHASE 3] Key driver identified: {driver_analysis.primary_driver}")

            # Phase 4: Gather evidence
            logger.info(f"[PHASE 4] Gathering evidence for {event_ticker}")
            evidence = await self._gather_evidence(
                driver_analysis=driver_analysis,
                event_title=event_title,
            )
            # No LLM call for web search
            logger.info(
                f"[PHASE 4] Evidence gathered: {len(evidence.sources)} sources, "
                f"reliability={evidence.reliability.value}"
            )

            # Build complete event research context
            research_context = EventResearchContext(
                event_ticker=event_ticker,
                event_title=event_title,
                event_category=event_category,
                context=event_context,
                driver_analysis=driver_analysis,
                evidence=evidence,
                market_tickers=[m.ticker for m in markets],
            )

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

    async def _research_event_context(
        self,
        event_ticker: str,
        event_title: str,
        event_category: str,
        market_count: int,
    ) -> EventContext:
        """
        Phase 2: Use LLM to understand what the event is about.

        Args:
            event_ticker: Event ticker
            event_title: Event title
            event_category: Event category
            market_count: Number of markets in this event

        Returns:
            EventContext with event understanding
        """
        from datetime import datetime
        current_date = datetime.now().strftime("%B %d, %Y")

        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are a prediction market analyst. Today is {current_date}.
Quickly extract key facts about this event. Be concise."""),
            ("user", """EVENT: {event_title}
CATEGORY: {category}

Provide:
1. Brief description (1-2 sentences)
2. Core question being predicted
3. Resolution criteria (how YES/NO determined)
4. Resolution type: objective/subjective/mixed
5. Time horizon (when will we know?)""")
        ])

        try:
            chain = prompt | self._llm.with_structured_output(EventContextOutput)
            result = await self._invoke_with_retry(chain, {
                "event_title": event_title,
                "category": event_category,
                "market_count": market_count,
            })

            # Validate output
            if not result.event_description or len(result.event_description) < 20:
                raise ValueError("LLM returned invalid event description")

            return EventContext(
                event_description=result.event_description,
                core_question=result.core_question,
                resolution_criteria=result.resolution_criteria,
                resolution_objectivity=result.resolution_objectivity,
                time_horizon=result.time_horizon,
            )

        except Exception as e:
            logger.error(f"Event context extraction failed: {e}")
            return EventContext(
                event_description=f"Event: {event_title}",
                core_question=event_title,
                resolution_criteria="Unknown",
                resolution_objectivity="unknown",
                time_horizon="Unknown",
            )

    async def _identify_key_driver(
        self,
        event_context: EventContext,
        event_title: str,
        event_category: str,
    ) -> KeyDriverAnalysis:
        """
        Phase 3: Use LLM to identify what determines the outcome.

        This is the core of first-principles reasoning: What single factor
        most determines whether this event resolves YES or NO?

        Args:
            event_context: Phase 2 output
            event_title: Event title
            event_category: Event category

        Returns:
            KeyDriverAnalysis with primary driver and reasoning
        """
        from datetime import datetime
        current_date = datetime.now().strftime("%B %d, %Y")

        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are a prediction market analyst. Today is {current_date}.

Identify the SINGLE CRITICAL FACTOR that determines YES vs NO.
Be specific and measurable. Think first-principles."""),
            ("user", """EVENT: {event_title}

CORE QUESTION: {core_question}
RESOLUTION: {resolution_criteria}

Identify:
1. PRIMARY DRIVER - the ONE factor that matters most (be specific)
2. WHY this is the key driver (causal mechanism)
3. 2-3 secondary factors
4. Tail risks that could change the outcome
5. Base rate - how often do similar events resolve YES? (estimate with reasoning)""")
        ])

        try:
            chain = prompt | self._llm.with_structured_output(KeyDriverOutput)
            result = await self._invoke_with_retry(chain, {
                "event_title": event_title,
                "category": event_category,
                "event_description": event_context.event_description,
                "core_question": event_context.core_question,
                "resolution_criteria": event_context.resolution_criteria,
                "time_horizon": event_context.time_horizon,
            })

            # Validate output
            if not result.primary_driver or len(result.primary_driver) < 5:
                raise ValueError("LLM returned invalid key driver")

            return KeyDriverAnalysis(
                primary_driver=result.primary_driver,
                primary_driver_reasoning=result.primary_driver_reasoning,
                causal_chain=result.causal_chain,
                secondary_factors=result.secondary_factors,
                tail_risks=result.tail_risks,
                base_rate=result.base_rate,
                base_rate_reasoning=result.base_rate_reasoning,
            )

        except Exception as e:
            logger.error(f"Key driver identification failed: {e}")
            return KeyDriverAnalysis(
                primary_driver="Unknown",
                primary_driver_reasoning="Analysis failed",
                causal_chain="Unknown",
            )

    async def _gather_evidence(
        self,
        driver_analysis: KeyDriverAnalysis,
        event_title: str,
    ) -> Evidence:
        """
        Phase 4: Use web search to gather evidence about the key driver.

        This is targeted search focused on the primary driver, not generic
        event search.

        Args:
            driver_analysis: Phase 3 output with key driver
            event_title: Event title for context

        Returns:
            Evidence with key facts and reliability assessment
        """
        if not self._web_search_enabled or not self._search_tool:
            return Evidence(
                evidence_summary="Web search not available",
                reliability=EvidenceReliability.LOW,
            )

        try:
            # Construct targeted search query
            search_query = f"{event_title} {driver_analysis.primary_driver}"

            # Run search with timeout
            loop = asyncio.get_running_loop()
            search_result = await asyncio.wait_for(
                loop.run_in_executor(None, self._search_tool.run, search_query),
                timeout=30.0
            )

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

            # Assess reliability (simple heuristic)
            reliability = EvidenceReliability.MEDIUM
            if len(key_evidence) >= 5 and len(sources) >= 2:
                reliability = EvidenceReliability.HIGH
            elif len(key_evidence) < 2:
                reliability = EvidenceReliability.LOW

            return Evidence(
                key_evidence=key_evidence[:5],
                evidence_summary=search_result[:500] if search_result else "No evidence found",
                sources=sources,
                sources_checked=1,
                reliability=reliability,
                reliability_reasoning=f"Found {len(key_evidence)} evidence points from {len(sources)} sources",
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
            ("system", f"""You are a prediction market analyst. Today is {current_date}.

Your task: For each market, provide TWO estimates:
1. YOUR PROBABILITY - What do YOU think the true probability is based on the evidence?
2. MARKET'S PRICE - What do you think the market is CURRENTLY trading at? (blind guess)

IMPORTANT: You will NOT be told the actual market prices. You must:
- Form your own view based on evidence and reasoning
- Then guess what the market believes (this tests your calibration)

Think independently. Use the key driver analysis but also your own judgment."""),
            ("user", """EVENT RESEARCH:
{event_context}

KEY DRIVER: {primary_driver}

MARKETS TO EVALUATE:
{markets_text}

For EACH market, provide:
1. The specific question being asked
2. How the key driver applies
3. YOUR probability estimate (0.0-1.0) with brief reasoning
4. Your GUESS of the current market price (0-100 cents)
5. Your confidence (high/medium/low)""")
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
                )
                assessments.append(assessment)

            return assessments

        except Exception as e:
            logger.error(f"Batch market evaluation failed: {e}", exc_info=True)
            return []

    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
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
        }
