"""
Event Research Context - Data structures for event-first research pipeline.

This module provides data structures for the event-first agentic research architecture,
which researches events holistically before evaluating individual markets.

Architecture:
    1. EVENT DISCOVERY: Group tracked markets by event_ticker
    2. EVENT RESEARCH: "What is this event about?"
    3. KEY DRIVER IDENTIFICATION: "What single factor determines YES vs NO?"
    4. EVIDENCE GATHERING: Targeted search for key driver data
    5. MARKET EVALUATION: Batch assess all markets with shared event context
    6. TRADE DECISIONS: Execute on mispriced markets

Design Principles:
    - Event context is shared across all markets in an event
    - Key driver analysis uses first-principles reasoning
    - Evidence is gathered specifically for the identified key driver
    - Market evaluation includes microstructure signals
"""

import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum


class EvidenceReliability(Enum):
    """Reliability classification for gathered evidence."""
    HIGH = "high"        # Official sources, recent data, multiple confirmations
    MEDIUM = "medium"    # Credible sources but older or single source
    LOW = "low"          # Unverified, speculative, or conflicting


class Confidence(Enum):
    """Confidence level in assessment."""
    HIGH = "high"        # Strong evidence, clear causal chain
    MEDIUM = "medium"    # Reasonable evidence, some uncertainty
    LOW = "low"          # Weak evidence, high uncertainty


class FrameType(str, Enum):
    """
    Semantic frame types for prediction markets.

    Each frame type represents a different structural pattern for how
    the prediction market question is formulated and resolved.
    """
    NOMINATION = "nomination"      # Actor nominates Candidate for Position (Fed Chair, SCOTUS)
    COMPETITION = "competition"    # Competitor A vs Competitor B (NFL, NBA, elections)
    ACHIEVEMENT = "achievement"    # Subject achieves/exceeds Threshold ("BTC hits $100k")
    OCCURRENCE = "occurrence"      # Event happens or doesn't ("Rain in NYC tomorrow")
    MEASUREMENT = "measurement"    # Metric above/below Value ("CPI over 3%")
    MENTION = "mention"            # Subject mentions/references Target ("Will X mention Y?")
    UNKNOWN = "unknown"            # Could not determine frame type


@dataclass
class SemanticRole:
    """
    An entity playing a semantic role in the event frame.

    Semantic roles represent the participants in the event structure:
    - Actors: Entities with agency/decision-making power (e.g., Trump as nominator)
    - Objects: Things being acted upon (e.g., Fed Chair position)
    - Candidates: Possible outcomes, each linked to a market (e.g., Warsh, Hassett)

    For news matching:
    - aliases: Alternative names/spellings for entity recognition
    - search_queries: Targeted search queries specific to this entity
    """
    entity_id: str                       # Normalized ID: "PERSON:KEVIN_WARSH"
    canonical_name: str                  # Display name: "Kevin Warsh"
    entity_type: str                     # "PERSON", "ORGANIZATION", "POSITION", "EVENT"
    role: str                            # "nominator", "candidate", "position", "competitor"
    role_description: str = ""           # "Potential Fed Chair nominee"
    market_ticker: Optional[str] = None  # For candidates: linked market ticker
    aliases: List[str] = field(default_factory=list)  # ["Warsh", "Kevin Warsh", "K. Warsh"]
    search_queries: List[str] = field(default_factory=list)  # ["kevin warsh fed chair"]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "entity_id": self.entity_id,
            "canonical_name": self.canonical_name,
            "entity_type": self.entity_type,
            "role": self.role,
            "role_description": self.role_description,
            "market_ticker": self.market_ticker,
            "aliases": self.aliases,
            "search_queries": self.search_queries,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SemanticRole":
        """Create from dictionary."""
        return cls(
            entity_id=data.get("entity_id", ""),
            canonical_name=data.get("canonical_name", ""),
            entity_type=data.get("entity_type", ""),
            role=data.get("role", ""),
            role_description=data.get("role_description", ""),
            market_ticker=data.get("market_ticker"),
            aliases=data.get("aliases", []),
            search_queries=data.get("search_queries", []),
        )


@dataclass
class SemanticFrame:
    """
    Rich semantic understanding of an event's structure.

    The semantic frame captures the deep structure of a prediction market event:
    - Frame type: What kind of question this is (NOMINATION, COMPETITION, etc.)
    - Semantic roles: WHO (actors), WHAT (objects), WHICH (candidates)
    - Constraints: Mutual exclusivity, who controls outcome, resolution trigger
    - Search guidance: Targeted queries and signal keywords for evidence gathering

    This enables:
    1. Better context understanding for LLM reasoning
    2. Targeted search queries for evidence gathering
    3. Foundation for future news matching
    """
    event_ticker: str
    frame_type: FrameType

    # The semantic question structure
    question_template: str = ""          # "{actor} nominates {candidate} for {position}"
    primary_relation: str = ""           # "nominates", "defeats", "exceeds"

    # Semantic roles
    actors: List[SemanticRole] = field(default_factory=list)      # Who has agency
    objects: List[SemanticRole] = field(default_factory=list)     # What's being acted upon
    candidates: List[SemanticRole] = field(default_factory=list)  # Possible outcomes

    # Constraints (for agent reasoning)
    mutual_exclusivity: bool = True      # Only one candidate/outcome can be YES
    actor_controls_outcome: bool = False # Does a specific actor (e.g., Trump) decide?
    resolution_trigger: str = ""         # "First formal nomination", "Final score", etc.

    # For evidence gathering (replaces generic search)
    primary_search_queries: List[str] = field(default_factory=list)  # Event-level queries
    signal_keywords: List[str] = field(default_factory=list)         # "frontrunner", "shortlist"

    # Metadata
    extracted_at: Optional[float] = None  # Unix timestamp

    def get_search_queries_for_market(self, market_ticker: str) -> List[str]:
        """
        Get targeted search queries for a specific market.

        Combines event-level queries with candidate-specific queries.
        """
        queries = list(self.primary_search_queries)
        for candidate in self.candidates:
            if candidate.market_ticker == market_ticker:
                queries.extend(candidate.search_queries)
        return queries

    def get_candidate_for_market(self, market_ticker: str) -> Optional[SemanticRole]:
        """Get the candidate role associated with a specific market."""
        for candidate in self.candidates:
            if candidate.market_ticker == market_ticker:
                return candidate
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_ticker": self.event_ticker,
            "frame_type": self.frame_type.value,
            "question_template": self.question_template,
            "primary_relation": self.primary_relation,
            "actors": [a.to_dict() for a in self.actors],
            "objects": [o.to_dict() for o in self.objects],
            "candidates": [c.to_dict() for c in self.candidates],
            "mutual_exclusivity": self.mutual_exclusivity,
            "actor_controls_outcome": self.actor_controls_outcome,
            "resolution_trigger": self.resolution_trigger,
            "primary_search_queries": self.primary_search_queries,
            "signal_keywords": self.signal_keywords,
            "extracted_at": self.extracted_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SemanticFrame":
        """Create from dictionary."""
        frame_type_str = data.get("frame_type", "unknown")
        try:
            frame_type = FrameType(frame_type_str)
        except ValueError:
            frame_type = FrameType.UNKNOWN

        return cls(
            event_ticker=data.get("event_ticker", ""),
            frame_type=frame_type,
            question_template=data.get("question_template", ""),
            primary_relation=data.get("primary_relation", ""),
            actors=[SemanticRole.from_dict(a) for a in data.get("actors", [])],
            objects=[SemanticRole.from_dict(o) for o in data.get("objects", [])],
            candidates=[SemanticRole.from_dict(c) for c in data.get("candidates", [])],
            mutual_exclusivity=data.get("mutual_exclusivity", True),
            actor_controls_outcome=data.get("actor_controls_outcome", False),
            resolution_trigger=data.get("resolution_trigger", ""),
            primary_search_queries=data.get("primary_search_queries", []),
            signal_keywords=data.get("signal_keywords", []),
            extracted_at=data.get("extracted_at"),
        )

    def to_prompt_string(self) -> str:
        """Format semantic frame for LLM prompt consumption."""
        lines = []

        lines.append(f"FRAME TYPE: {self.frame_type.value.upper()}")
        if self.question_template:
            lines.append(f"QUESTION STRUCTURE: {self.question_template}")

        if self.actors:
            lines.append("")
            lines.append("ACTORS (decision-makers):")
            for actor in self.actors:
                aliases_str = f" (also: {', '.join(actor.aliases)})" if actor.aliases else ""
                lines.append(f"  - {actor.canonical_name}{aliases_str} [{actor.role}]")

        if self.objects:
            lines.append("")
            lines.append("OBJECTS:")
            for obj in self.objects:
                lines.append(f"  - {obj.canonical_name} [{obj.role}]")

        if self.candidates:
            lines.append("")
            lines.append(f"CANDIDATES ({len(self.candidates)} possible outcomes):")
            for candidate in self.candidates[:10]:  # Show first 10
                ticker_str = f" → {candidate.market_ticker}" if candidate.market_ticker else ""
                lines.append(f"  - {candidate.canonical_name}{ticker_str}")
            if len(self.candidates) > 10:
                lines.append(f"  ... and {len(self.candidates) - 10} more")

        lines.append("")
        lines.append("CONSTRAINTS:")
        lines.append(f"  - Mutual exclusivity: {self.mutual_exclusivity}")
        lines.append(f"  - Actor controls outcome: {self.actor_controls_outcome}")
        if self.resolution_trigger:
            lines.append(f"  - Resolution trigger: {self.resolution_trigger}")

        return "\n".join(lines)


@dataclass
class EventContext:
    """
    Phase 2 output: Understanding of what the event is about.

    Generated by LLM analysis of event metadata and initial research.
    """
    # What is this event about?
    event_description: str              # 2-3 sentence description
    core_question: str                  # The fundamental question being asked

    # How will outcome be determined?
    resolution_criteria: str            # What determines YES vs NO
    resolution_objectivity: str         # "objective", "subjective", "mixed"

    # Time horizon
    time_horizon: str                   # When will we know the outcome
    key_dates: List[str] = field(default_factory=list)  # Important dates before resolution

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_description": self.event_description,
            "core_question": self.core_question,
            "resolution_criteria": self.resolution_criteria,
            "resolution_objectivity": self.resolution_objectivity,
            "time_horizon": self.time_horizon,
            "key_dates": self.key_dates,
        }


@dataclass
class KeyDriverAnalysis:
    """
    Phase 3 output: Identification of what determines the outcome.

    Generated by LLM causal reasoning about the event.
    """
    # Primary driver
    primary_driver: str                 # The single most important factor
    primary_driver_reasoning: str       # Why this factor matters most
    causal_chain: str                   # How driver → outcome

    # Secondary factors
    secondary_factors: List[str] = field(default_factory=list)
    secondary_importance: str = ""      # How much they matter relative to primary

    # Tail risks
    tail_risks: List[str] = field(default_factory=list)
    what_could_go_wrong: str = ""       # What would invalidate the analysis

    # Base rate
    base_rate: float = 0.5              # Historical frequency of YES (0-1)
    base_rate_reasoning: str = ""       # How base rate was determined
    comparable_events: str = ""         # What "similar events" means

    # Edge hypothesis (v2 profit-focused)
    edge_hypothesis: str = ""           # Where might the market be wrong?

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "primary_driver": self.primary_driver,
            "primary_driver_reasoning": self.primary_driver_reasoning,
            "causal_chain": self.causal_chain,
            "secondary_factors": self.secondary_factors,
            "secondary_importance": self.secondary_importance,
            "tail_risks": self.tail_risks,
            "what_could_go_wrong": self.what_could_go_wrong,
            "base_rate": self.base_rate,
            "base_rate_reasoning": self.base_rate_reasoning,
            "comparable_events": self.comparable_events,
            "edge_hypothesis": self.edge_hypothesis,
        }


@dataclass
class Evidence:
    """
    Phase 4 output: Evidence gathered for the key driver.

    Generated by web search targeted at the identified primary driver.
    """
    # Key evidence about the primary driver
    key_evidence: List[str] = field(default_factory=list)
    evidence_summary: str = ""          # One paragraph summary

    # Sources
    sources: List[str] = field(default_factory=list)
    sources_checked: int = 0

    # Reliability assessment
    reliability: EvidenceReliability = EvidenceReliability.MEDIUM
    reliability_reasoning: str = ""

    # What evidence suggests
    evidence_probability: float = 0.5   # What probability evidence implies


@dataclass
class EventResearchContext:
    """
    Complete event-level research context shared across all markets in an event.

    This is the primary output of Phases 1-4, used as input to Phase 5 (market evaluation).

    Enhanced with SemanticFrame for deep structural understanding of the event.
    """
    # Event identification
    event_ticker: str
    event_title: str
    event_category: str

    # Phase 2: Event understanding
    context: EventContext = field(default_factory=lambda: EventContext(
        event_description="",
        core_question="",
        resolution_criteria="",
        resolution_objectivity="unknown",
        time_horizon="",
    ))

    # Phase 3: Key driver analysis
    driver_analysis: KeyDriverAnalysis = field(default_factory=KeyDriverAnalysis)

    # Phase 4: Evidence
    evidence: Evidence = field(default_factory=Evidence)

    # Semantic frame (NEW): Deep structural understanding
    semantic_frame: Optional[SemanticFrame] = None

    # Markets in this event
    market_tickers: List[str] = field(default_factory=list)

    # Metadata
    researched_at: float = field(default_factory=time.time)
    research_duration_seconds: float = 0.0
    llm_calls_made: int = 0
    cached: bool = False  # Was this context loaded from cache?

    def to_prompt_string(self) -> str:
        """Format event research context for LLM prompt consumption."""
        lines = []

        # Event description
        lines.append(f"EVENT: {self.event_title}")
        lines.append(f"CATEGORY: {self.event_category}")
        lines.append("")
        lines.append(f"DESCRIPTION: {self.context.event_description}")
        lines.append(f"CORE QUESTION: {self.context.core_question}")
        lines.append(f"RESOLUTION: {self.context.resolution_criteria}")
        lines.append(f"TIME HORIZON: {self.context.time_horizon}")

        # Key driver
        lines.append("")
        lines.append(f"PRIMARY DRIVER: {self.driver_analysis.primary_driver}")
        lines.append(f"WHY: {self.driver_analysis.primary_driver_reasoning}")
        lines.append(f"CAUSAL CHAIN: {self.driver_analysis.causal_chain}")

        # Base rate
        lines.append("")
        lines.append(f"BASE RATE: {self.driver_analysis.base_rate:.0%} for similar events")
        lines.append(f"REASONING: {self.driver_analysis.base_rate_reasoning}")

        # Secondary factors
        if self.driver_analysis.secondary_factors:
            lines.append("")
            lines.append("SECONDARY FACTORS:")
            for factor in self.driver_analysis.secondary_factors:
                lines.append(f"  - {factor}")

        # Tail risks
        if self.driver_analysis.tail_risks:
            lines.append("")
            lines.append("TAIL RISKS:")
            for risk in self.driver_analysis.tail_risks:
                lines.append(f"  - {risk}")

        # Evidence
        lines.append("")
        lines.append(f"EVIDENCE ({self.evidence.reliability.value} reliability):")
        lines.append(self.evidence.evidence_summary)
        if self.evidence.key_evidence:
            for ev in self.evidence.key_evidence[:5]:  # Top 5 evidence points
                lines.append(f"  - {ev}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/storage."""
        return {
            "event_ticker": self.event_ticker,
            "event_title": self.event_title,
            "event_category": self.event_category,
            "context": {
                "event_description": self.context.event_description,
                "core_question": self.context.core_question,
                "resolution_criteria": self.context.resolution_criteria,
                "resolution_objectivity": self.context.resolution_objectivity,
                "time_horizon": self.context.time_horizon,
                "key_dates": self.context.key_dates,
            },
            "driver_analysis": {
                "primary_driver": self.driver_analysis.primary_driver,
                "primary_driver_reasoning": self.driver_analysis.primary_driver_reasoning,
                "causal_chain": self.driver_analysis.causal_chain,
                "secondary_factors": self.driver_analysis.secondary_factors,
                "tail_risks": self.driver_analysis.tail_risks,
                "base_rate": self.driver_analysis.base_rate,
                "base_rate_reasoning": self.driver_analysis.base_rate_reasoning,
            },
            "evidence": {
                "key_evidence": self.evidence.key_evidence,
                "evidence_summary": self.evidence.evidence_summary,
                "sources": self.evidence.sources,
                "reliability": self.evidence.reliability.value,
                "evidence_probability": self.evidence.evidence_probability,
            },
            "semantic_frame": self.semantic_frame.to_dict() if self.semantic_frame else None,
            "market_tickers": self.market_tickers,
            "researched_at": self.researched_at,
            "research_duration_seconds": self.research_duration_seconds,
            "llm_calls_made": self.llm_calls_made,
            "cached": self.cached,
        }


@dataclass
class MarketAssessment:
    """
    Phase 5 output: Per-market assessment within an event.

    Generated by applying event context + key driver to specific market question.

    Calibration Fields (v2 additions - all with defaults for backward compatibility):
    - evidence_cited: Which evidence points support this estimate
    - what_would_change_mind: What would most change this estimate
    - assumption_flags: Assumptions made due to missing info
    - calibration_notes: Notes on confidence calibration
    - evidence_quality: Quality rating of supporting evidence
    """
    market_ticker: str
    market_title: str

    # The specific question this market asks
    specific_question: str = ""

    # How key driver applies to this market
    driver_application: str = ""        # How does primary driver affect THIS market?

    # Probability assessment
    evidence_probability: float = 0.5   # What evidence suggests
    market_probability: float = 0.5     # Current market price / 100
    mispricing_magnitude: float = 0.0   # evidence - market (positive = underpriced YES)

    # Price calibration (blind estimation)
    price_guess_cents: Optional[int] = None       # LLM's blind guess of market price
    price_guess_error_cents: Optional[int] = None # guess - actual (positive = overestimated)

    # Recommendation
    recommendation: str = "HOLD"        # BUY_YES, BUY_NO, HOLD
    confidence: Confidence = Confidence.MEDIUM
    edge_explanation: str = ""          # Why market is mispriced

    # Microstructure (if available)
    microstructure_summary: str = ""    # Summary of trade flow / orderbook signals

    # === Calibration Fields (v2 additions - all with defaults for backward compatibility) ===
    evidence_cited: List[str] = field(default_factory=list)  # Which evidence points support this
    what_would_change_mind: str = ""    # What would most change this estimate
    assumption_flags: List[str] = field(default_factory=list)  # Assumptions due to missing info
    calibration_notes: str = ""         # Notes on confidence calibration
    evidence_quality: str = "medium"    # Quality: high, medium, low

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/storage."""
        return {
            "market_ticker": self.market_ticker,
            "market_title": self.market_title,
            "specific_question": self.specific_question,
            "driver_application": self.driver_application,
            "evidence_probability": self.evidence_probability,
            "market_probability": self.market_probability,
            "mispricing_magnitude": self.mispricing_magnitude,
            "price_guess_cents": self.price_guess_cents,
            "price_guess_error_cents": self.price_guess_error_cents,
            "recommendation": self.recommendation,
            "confidence": self.confidence.value,
            "edge_explanation": self.edge_explanation,
            "microstructure_summary": self.microstructure_summary,
            # v2 calibration fields
            "evidence_cited": self.evidence_cited,
            "what_would_change_mind": self.what_would_change_mind,
            "assumption_flags": self.assumption_flags,
            "calibration_notes": self.calibration_notes,
            "evidence_quality": self.evidence_quality,
        }


@dataclass
class EventResearchResult:
    """
    Complete result of event-first research pipeline.

    Contains event context and all market assessments.
    """
    # Event-level context (shared)
    event_context: EventResearchContext

    # Per-market assessments
    assessments: List[MarketAssessment] = field(default_factory=list)

    # Summary statistics
    markets_evaluated: int = 0
    markets_with_edge: int = 0          # Markets with |mispricing| > threshold
    total_research_seconds: float = 0.0

    # Status
    success: bool = True
    error_message: Optional[str] = None

    def get_tradeable_assessments(
        self,
        min_mispricing: float = 0.10,
        min_confidence: Confidence = Confidence.MEDIUM,
    ) -> List[MarketAssessment]:
        """Get assessments that meet trading thresholds."""
        tradeable = []
        for assessment in self.assessments:
            if assessment.recommendation == "HOLD":
                continue
            if abs(assessment.mispricing_magnitude) < min_mispricing:
                continue
            # Confidence ordering: HIGH > MEDIUM > LOW
            confidence_order = {Confidence.HIGH: 3, Confidence.MEDIUM: 2, Confidence.LOW: 1}
            if confidence_order.get(assessment.confidence, 0) < confidence_order.get(min_confidence, 0):
                continue
            tradeable.append(assessment)
        return tradeable

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/storage."""
        return {
            "event_context": self.event_context.to_dict(),
            "assessments": [a.to_dict() for a in self.assessments],
            "markets_evaluated": self.markets_evaluated,
            "markets_with_edge": self.markets_with_edge,
            "total_research_seconds": self.total_research_seconds,
            "success": self.success,
            "error_message": self.error_message,
        }
