"""
Agentic Research Service - Standalone, Testable Component

This service can run independently for testing without requiring
the full trading system. It provides market research capabilities
through AI agents and web search using LangChain.

Design:
- Uses existing tracked markets (no separate discovery)
- Can be tested with mock market data
- Research results cached for efficiency
- LangChain-based web search + LLM reasoning
- Enhanced with microstructure context for better signal detection
"""

import asyncio
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from enum import Enum

from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ..state.microstructure_context import MicrostructureContext

logger = logging.getLogger("kalshiflow_rl.traderv3.services.agentic_research")


class ResearchStatus(Enum):
    """Status of a research task."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass
class ResearchTask:
    """A research task for a specific market."""
    market_ticker: str
    market_title: str
    market_category: str
    current_price_cents: int
    hours_to_close: float
    created_at: float = field(default_factory=time.time)
    status: ResearchStatus = ResearchStatus.PENDING

    # Extracted research question
    research_question: Optional[str] = None

    # Priority score (0-1) for queue ordering
    priority: float = 0.5

    # Microstructure context (trade flow, orderbook signals)
    microstructure_context: Optional[Any] = None  # MicrostructureContext


class MarketAssessment(BaseModel):
    """Structured output model for market assessment (MVP simplified)."""
    fair_value: float = Field(
        description="Your estimate of true YES probability (0.0 to 1.0)",
        ge=0.0, le=1.0
    )
    direction: str = Field(
        description="Is the market price too high, too low, or fair? One of: market_too_high, market_too_low, fairly_priced"
    )
    edge_reasoning: str = Field(
        description="2-3 sentences explaining what specific information the market might be missing"
    )
    uncertainty: str = Field(
        description="Your uncertainty level: high, medium, or low"
    )

    # For backward compatibility, map to old field names
    @property
    def probability(self) -> float:
        return self.fair_value

    @property
    def confidence(self) -> float:
        """Map uncertainty to confidence score."""
        return {"low": 0.85, "medium": 0.70, "high": 0.50}.get(self.uncertainty.lower(), 0.60)

    @property
    def reasoning(self) -> str:
        return self.edge_reasoning

    @property
    def key_factors(self) -> List[str]:
        return [self.edge_reasoning]


@dataclass
class ResearchAssessment:
    """Agent's assessment after research."""
    market_ticker: str
    agent_probability: float  # Agent's assessed probability (0-1)
    confidence: float  # Confidence in assessment (0-1)
    reasoning: str  # Chain of reasoning
    key_facts: List[str]  # Key facts discovered
    sources: List[str]  # Information sources used
    market_price_probability: float  # Market's implied probability
    mispricing_magnitude: float  # agent_prob - market_prob
    recommendation: str  # "BUY_YES", "BUY_NO", "HOLD"
    assessment_timestamp: float = field(default_factory=time.time)
    
    # Research metadata
    research_duration_seconds: float = 0.0
    sources_checked: int = 0


class AgenticResearchService:
    """
    Standalone service for agentic market research using LangChain.
    
    Can operate independently for testing or integrate with trading system.
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        openai_model: str = "gpt-4o",
        openai_temperature: float = 0.3,
        web_search_enabled: bool = True,
        cache_ttl_seconds: float = 300.0,  # 5 minute cache
        max_concurrent_research: int = 3,
    ):
        """
        Initialize research service.
        
        Args:
            openai_api_key: OpenAI API key (if None, loads from OPENAI_API_KEY env var)
            openai_model: OpenAI model to use (default: "gpt-4o")
            openai_temperature: Temperature for LLM (default: 0.3 for consistent reasoning)
            web_search_enabled: Enable web search (requires DuckDuckGo)
            cache_ttl_seconds: Cache assessment results for this duration
            max_concurrent_research: Max parallel research tasks
        """
        # Get OpenAI API key from parameter or environment
        self._api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var or pass openai_api_key parameter.")
        
        self._model = openai_model
        self._temperature = openai_temperature
        self._web_search_enabled = web_search_enabled
        self._cache_ttl = cache_ttl_seconds
        self._max_concurrent = max_concurrent_research
        
        # Initialize LangChain components
        self._llm = ChatOpenAI(
            model=self._model,
            temperature=self._temperature,
            api_key=self._api_key,
        )
        
        # Initialize web search tool if enabled
        self._search_tool = None
        if self._web_search_enabled:
            try:
                self._search_tool = DuckDuckGoSearchRun()
            except Exception as e:
                logger.warning(f"Failed to initialize web search tool: {e}. Web search will be disabled.")
                self._web_search_enabled = False
        
        # State
        self._research_queue: asyncio.Queue[ResearchTask] = asyncio.Queue()
        self._assessments: Dict[str, ResearchAssessment] = {}
        self._active_research: Dict[str, asyncio.Task] = {}
        self._running = False
        self._research_lock = asyncio.Lock()  # Protects _active_research

        # Backoff state for API failures
        self._consecutive_failures = 0
        self._backoff_until = 0.0

        # Cache cleanup
        self._last_cache_cleanup = time.time()
        self._cache_cleanup_interval = 60.0  # Cleanup every minute

        # Stats
        self._total_researched = 0
        self._successful_assessments = 0
        self._failed_researches = 0
        
        logger.info(
            f"AgenticResearchService initialized "
            f"(model={openai_model}, "
            f"web_search={web_search_enabled}, "
            f"cache_ttl={cache_ttl_seconds}s, "
            f"max_concurrent={max_concurrent_research})"
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

    async def start(self) -> None:
        """Start the research service background worker."""
        if self._running:
            return
        
        self._running = True
        logger.info("AgenticResearchService started")
    
    async def stop(self) -> None:
        """Stop the research service."""
        self._running = False
        
        # Cancel all active research
        for task in self._active_research.values():
            task.cancel()
        
        await asyncio.gather(*self._active_research.values(), return_exceptions=True)
        self._active_research.clear()
        
        logger.info("AgenticResearchService stopped")
    
    async def research_market(
        self,
        market_ticker: str,
        market_title: str,
        market_category: str,
        current_price_cents: int,
        hours_to_close: float,
        force_refresh: bool = False,
        microstructure_context: Optional["MicrostructureContext"] = None,
    ) -> Optional[ResearchAssessment]:
        """
        Research a market and return assessment.

        This is the main entry point. It will:
        1. Check cache (unless force_refresh)
        2. Extract researchable question
        3. Queue research if needed
        4. Return cached result or None (async completion)

        Args:
            market_ticker: Market ticker symbol
            market_title: Human-readable market title
            market_category: Market category
            current_price_cents: Current YES price in cents
            hours_to_close: Hours until market settlement
            force_refresh: Force new research even if cached
            microstructure_context: Optional trade flow and orderbook context

        Returns:
            ResearchAssessment if available, None if queued/researching
        """
        # Periodic cache cleanup
        if time.time() - self._last_cache_cleanup > self._cache_cleanup_interval:
            self._cleanup_expired_cache()
            self._last_cache_cleanup = time.time()

        # Check cache first (outside lock for performance)
        if not force_refresh:
            cached = self._get_cached_assessment(market_ticker)
            if cached:
                return cached

        # Check backoff
        if time.time() < self._backoff_until:
            logger.debug(f"Backing off research until {self._backoff_until:.0f}")
            return None

        # Use lock to prevent race condition when checking/adding to _active_research
        async with self._research_lock:
            # Double-check after acquiring lock
            if market_ticker in self._active_research:
                return None  # Research in progress

            # Extract researchable question
            research_question = self._extract_research_question(market_title)
            if not research_question:
                logger.debug(f"Cannot extract research question for: {market_title}")
                return None

            # Calculate priority
            priority = self._calculate_priority(
                current_price_cents, hours_to_close, market_category
            )

            # Create task
            task = ResearchTask(
                market_ticker=market_ticker,
                market_title=market_title,
                market_category=market_category,
                current_price_cents=current_price_cents,
                hours_to_close=hours_to_close,
                research_question=research_question,
                priority=priority,
                microstructure_context=microstructure_context,
            )

            # Start research (async, non-blocking)
            research_task = asyncio.create_task(
                self._run_research_pipeline(task)
            )
            self._active_research[market_ticker] = research_task

        return None  # Will be available when research completes

    def _cleanup_expired_cache(self) -> int:
        """Remove expired assessments from cache. Returns count removed."""
        now = time.time()
        expired = [
            ticker for ticker, assessment in self._assessments.items()
            if now - assessment.assessment_timestamp > self._cache_ttl
        ]
        for ticker in expired:
            self._assessments.pop(ticker, None)
        if expired:
            logger.debug(f"Cleaned up {len(expired)} expired cache entries")
        return len(expired)
    
    async def get_assessment(
        self,
        market_ticker: str,
        wait_seconds: float = 0.0,
    ) -> Optional[ResearchAssessment]:
        """
        Get assessment for a market (wait if researching).
        
        Args:
            market_ticker: Market to check
            wait_seconds: Max seconds to wait if research in progress
        
        Returns:
            ResearchAssessment or None
        """
        # Check cache first
        assessment = self._get_cached_assessment(market_ticker)
        if assessment:
            return assessment
        
        # Wait for active research if requested
        if wait_seconds > 0 and market_ticker in self._active_research:
            try:
                await asyncio.wait_for(
                    self._wait_for_research(market_ticker),
                    timeout=wait_seconds
                )
                return self._assessments.get(market_ticker)
            except asyncio.TimeoutError:
                return None
        
        return self._assessments.get(market_ticker)
    
    def _get_cached_assessment(
        self,
        market_ticker: str,
    ) -> Optional[ResearchAssessment]:
        """Get cached assessment if still valid."""
        assessment = self._assessments.get(market_ticker)
        if not assessment:
            return None
        
        age = time.time() - assessment.assessment_timestamp
        if age > self._cache_ttl:
            # Expired, remove from cache
            self._assessments.pop(market_ticker, None)
            return None
        
        return assessment
    
    async def _wait_for_research(self, market_ticker: str) -> None:
        """Wait for research to complete."""
        task = self._active_research.get(market_ticker)
        if task:
            await task
    
    def _extract_research_question(self, market_title: str) -> Optional[str]:
        """
        Extract a researchable question from market title.
        
        Simple version: Returns the title if it looks researchable.
        Can be enhanced with LLM extraction later.
        """
        if not market_title:
            return None
        
        # Simple heuristics for now
        # Skip spread markets (e.g., "Team A vs Team B - spread")
        if "spread" in market_title.lower():
            return None
        
        # Skip overly specific technical markets
        if market_title.startswith("KX"):
            return None
        
        # Return title as research question (can enhance with LLM later)
        return market_title
    
    def _calculate_priority(
        self,
        price_cents: int,
        hours_to_close: float,
        category: str,
    ) -> float:
        """
        Calculate research priority (0-1).
        
        Higher priority for:
        - Markets closing soon (but not too soon)
        - Markets away from 50c (more interesting)
        - Researchable categories (politics, sports, economics)
        """
        priority = 0.5  # Base priority
        
        # Time-based: prefer 1-24 hours to close
        if 1 <= hours_to_close <= 24:
            priority += 0.3
        elif hours_to_close < 1:
            priority -= 0.2  # Too soon, low priority
        elif hours_to_close > 7 * 24:
            priority -= 0.1  # Too far, lower priority
        
        # Price-based: prefer markets away from 50c
        price_distance_from_50 = abs(price_cents - 50) / 50.0
        priority += price_distance_from_50 * 0.2
        
        # Category-based
        researchable_categories = {
            "politics", "sports", "economics", "current events",
        }
        if category.lower() in researchable_categories:
            priority += 0.2
        
        return max(0.0, min(1.0, priority))
    
    async def _run_research_pipeline(
        self,
        task: ResearchTask,
    ) -> Optional[ResearchAssessment]:
        """
        Execute full research pipeline for a market.
        
        Steps:
        1. Web search (if enabled)
        2. Information synthesis
        3. LLM reasoning with structured output
        4. Probability assessment
        5. Value analysis
        """
        start_time = time.time()
        task.status = ResearchStatus.IN_PROGRESS
        
        try:
            logger.info(
                f"Starting research for {task.market_ticker}: "
                f"{task.research_question}"
            )
            
            # Step 1: Gather information via web search
            info_sources = []
            key_facts = []
            
            if self._web_search_enabled and self._search_tool:
                try:
                    search_results = await self._search_web(task.research_question)
                    info_sources.extend(search_results.get("sources", []))
                    key_facts.extend(search_results.get("facts", []))
                except Exception as e:
                    logger.warning(f"Web search failed for {task.market_ticker}: {e}")
                    # Continue without web search results
            else:
                logger.debug(f"Web search disabled or unavailable for {task.market_ticker}")
            
            # Step 2: LLM reasoning and assessment
            assessment = await self._reason_and_assess(
                question=task.research_question,
                market_title=task.market_title,
                category=task.market_category,
                key_facts=key_facts,
                sources=info_sources,
                market_price_cents=task.current_price_cents,
                hours_to_close=task.hours_to_close,
                microstructure_context=task.microstructure_context,
            )
            
            if not assessment:
                task.status = ResearchStatus.FAILED
                self._failed_researches += 1
                return None
            
            # Step 3: Value analysis
            market_prob = task.current_price_cents / 100.0
            mispricing = assessment.probability - market_prob
            
            # Step 4: Generate recommendation
            recommendation = self._generate_recommendation(
                agent_prob=assessment.probability,
                market_prob=market_prob,
                confidence=assessment.confidence,
                min_mispricing=0.10,  # 10% threshold
            )
            
            # Create final assessment
            duration = time.time() - start_time
            final_assessment = ResearchAssessment(
                market_ticker=task.market_ticker,
                agent_probability=assessment.probability,
                confidence=assessment.confidence,
                reasoning=assessment.reasoning,
                key_facts=assessment.key_factors[:5],  # Top 5 factors
                sources=info_sources[:3],  # Top 3 sources
                market_price_probability=market_prob,
                mispricing_magnitude=mispricing,
                recommendation=recommendation,
                assessment_timestamp=time.time(),
                research_duration_seconds=duration,
                sources_checked=len(info_sources),
            )
            
            # Store in cache
            self._assessments[task.market_ticker] = final_assessment
            task.status = ResearchStatus.COMPLETED
            self._successful_assessments += 1
            self._total_researched += 1
            
            logger.info(
                f"Research complete for {task.market_ticker}: "
                f"agent_prob={assessment.probability:.2f}, "
                f"market_prob={market_prob:.2f}, "
                f"mispricing={mispricing:+.2f}, "
                f"recommendation={recommendation}"
            )

            # Reset backoff on success
            self._consecutive_failures = 0

            return final_assessment

        except asyncio.CancelledError:
            # Clean shutdown - don't log as error, re-raise
            task.status = ResearchStatus.EXPIRED
            logger.info(f"Research cancelled for {task.market_ticker}")
            raise

        except Exception as e:
            logger.error(f"Research pipeline failed for {task.market_ticker}: {e}", exc_info=True)
            task.status = ResearchStatus.FAILED
            self._failed_researches += 1

            # Implement exponential backoff (max 5 minutes)
            self._consecutive_failures += 1
            backoff_seconds = min(30 * (2 ** self._consecutive_failures), 300)
            self._backoff_until = time.time() + backoff_seconds
            logger.warning(f"Research failed, backing off for {backoff_seconds}s")

            return None
        finally:
            self._active_research.pop(task.market_ticker, None)
    
    async def _search_web(self, question: str) -> Dict[str, Any]:
        """
        Search the web for information about the question.
        
        Uses LangChain's DuckDuckGo search tool.
        """
        if not self._search_tool:
            return {"facts": [], "sources": []}
        
        try:
            # Run search (synchronous tool, but we can run in executor for async)
            loop = asyncio.get_running_loop()
            search_result = await loop.run_in_executor(
                None,
                self._search_tool.run,
                question
            )
            
            # Parse search results
            # DuckDuckGo returns a string with results
            # For now, treat the result as a fact and extract URLs if present
            facts = [search_result[:500]]  # First 500 chars as fact
            sources = []

            # Try to extract URLs from search result
            urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', search_result)
            sources.extend(urls[:5])  # First 5 URLs as sources
            
            return {
                "facts": facts,
                "sources": sources,
            }
        except Exception as e:
            logger.warning(f"Web search error: {e}")
            return {"facts": [], "sources": []}
    
    async def _reason_and_assess(
        self,
        question: str,
        market_title: str,
        category: str,
        key_facts: List[str],
        sources: List[str],
        market_price_cents: int,
        hours_to_close: float,
        microstructure_context: Optional["MicrostructureContext"] = None,
    ) -> Optional[MarketAssessment]:
        """
        Use LangChain LLM to reason about the question and assess probability.

        Args:
            question: Research question
            market_title: Human-readable market title
            category: Market category
            key_facts: Key facts discovered from web search
            sources: Information sources used
            market_price_cents: Current YES price in cents
            hours_to_close: Hours until market settlement
            microstructure_context: Optional trade flow and orderbook context

        Returns:
            MarketAssessment with fair_value, direction, reasoning
        """
        try:
            # Build prompt context
            facts_text = "\n".join(f"- {fact}" for fact in key_facts) if key_facts else "No recent news found."
            market_probability = market_price_cents / 100.0

            # Build microstructure context string if available
            microstructure_text = ""
            if microstructure_context and not microstructure_context.is_empty:
                microstructure_text = f"\n\nMARKET MICROSTRUCTURE:\n{microstructure_context.to_prompt_string()}"
                logger.debug(f"Including microstructure context in prompt: {microstructure_context.total_trades} trades")

            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a prediction market analyst searching for MISPRICED markets.

The market price represents the collective estimate of all traders. Your job is to:
1. Identify if you have information the market might be missing
2. Assess whether the market price is too high, too low, or fair
3. Be HUMBLE - the market is usually right

KEY PRINCIPLES:
- The market price already incorporates most public information
- You need SPECIFIC evidence that contradicts market consensus
- If unsure, assume the market is fairly priced
- Base rates matter: consider how often similar events resolve YES historically

MICROSTRUCTURE SIGNALS (if available):
- Trade Flow: If most traders are buying YES but price is dropping, this indicates informed selling (potential NO opportunity)
- Volume Imbalance: Strong buy/sell pressure in orderbook can indicate directional sentiment
- Spread: Wide spreads indicate uncertainty; tight spreads indicate consensus
- Large Orders: Presence of large orders (10k+ contracts) may indicate institutional activity"""),
                ("user", """MARKET: {market_title}
CATEGORY: {category}
CURRENT PRICE: {market_price_cents}c (implies {market_probability:.0%} chance of YES)
TIME TO SETTLEMENT: {hours_to_close:.1f} hours

RESEARCH FINDINGS:
{facts_text}{microstructure_text}

Based on this information:
1. What is your estimate of the TRUE probability this resolves YES? (fair_value)
2. Is the market price too high, too low, or fairly priced? (direction)
3. What specific information might the market be missing? (edge_reasoning)
4. How uncertain are you? (uncertainty: high/medium/low)""")
            ])

            # Create chain with structured output
            chain = prompt | self._llm.with_structured_output(MarketAssessment)

            # Run chain with retry
            assessment = await self._invoke_with_retry(chain, {
                "market_title": market_title,
                "category": category,
                "market_price_cents": market_price_cents,
                "market_probability": market_probability,
                "hours_to_close": hours_to_close,
                "facts_text": facts_text,
                "microstructure_text": microstructure_text,
            })

            return assessment

        except Exception as e:
            logger.error(f"LLM reasoning failed: {e}", exc_info=True)
            return None
    
    def _generate_recommendation(
        self,
        agent_prob: float,
        market_prob: float,
        confidence: float,
        min_mispricing: float = 0.10,
    ) -> str:
        """Generate trading recommendation based on mispricing."""
        mispricing = agent_prob - market_prob
        abs_mispricing = abs(mispricing)
        
        # Need sufficient mispricing and confidence
        if abs_mispricing < min_mispricing:
            return "HOLD"
        
        if confidence < 0.60:  # Low confidence threshold
            return "HOLD"
        
        # Agent thinks higher probability than market
        if mispricing > 0:
            return "BUY_YES"
        # Agent thinks lower probability than market
        else:
            return "BUY_NO"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        return {
            "total_researched": self._total_researched,
            "successful_assessments": self._successful_assessments,
            "failed_researches": self._failed_researches,
            "active_research_count": len(self._active_research),
            "cached_assessments": len(self._assessments),
            "cache_ttl_seconds": self._cache_ttl,
            "web_search_enabled": self._web_search_enabled,
            "model": self._model,
        }