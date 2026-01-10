"""
Schema for Phase 2+3: Combined Event Context + Key Driver + Semantic Frame extraction.

This schema defines the structured output for the combined extraction prompt
that extracts event understanding, key driver analysis, and semantic frame
in a single LLM call for efficiency.

Version History:
    v1 (2025-01): Initial schema with basic fields
    v2 (2025-01): Added grounding and calibration fields
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class SemanticRoleOutput(BaseModel):
    """
    Output for a semantic role (actor, object, or candidate).

    Semantic roles represent entities in the event structure:
    - Actors: Decision-makers (e.g., "Trump" in nomination events)
    - Objects: Things being acted upon (e.g., "Fed Chair position")
    - Candidates: Possible outcomes linked to markets
    """
    entity_id: str = Field(
        description="Unique identifier for this entity (e.g., 'PERSON:TRUMP', 'TEAM:CELTICS')"
    )
    canonical_name: str = Field(
        description="The standard/official name for this entity"
    )
    entity_type: str = Field(
        description="Type of entity: PERSON, ORGANIZATION, TEAM, POSITION, METRIC, etc."
    )
    role: str = Field(
        description="Role in the semantic frame: 'actor', 'nominator', 'candidate', 'position', etc."
    )
    role_description: str = Field(
        default="",
        description="Brief description of this entity's role in the event"
    )
    market_ticker: Optional[str] = Field(
        default=None,
        description="If this entity corresponds to a specific market, its ticker (e.g., 'KXFEDCHAIRNOM-WARSH')"
    )
    aliases: List[str] = Field(
        default_factory=list,
        description="Alternative names/references for this entity (for news matching)"
    )
    search_queries: List[str] = Field(
        default_factory=list,
        description="Targeted search queries to find news about this entity"
    )


class FullContextOutput(BaseModel):
    """
    Combined output for context + driver + semantic frame.

    This model is used for efficient single-LLM-call extraction that combines:
    - Phase 2: Event Context understanding
    - Phase 3: Key Driver identification
    - Semantic Frame extraction

    On cache hit, we skip the LLM call entirely and use cached values.

    GROUNDING REQUIREMENTS (v2):
    - All factual claims must be supported by provided context or marked as assumptions
    - Base rates must cite comparable events or be marked "estimated"
    - Probability estimates must acknowledge uncertainty

    CALIBRATION (v2):
    - Base rate should anchor initial probability estimate
    - Adjust from base rate based on specific evidence
    - Express confidence bounds when uncertain
    """

    # === Event Context (Phase 2) ===
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

    # === Key Driver (Phase 3) ===
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

    # === Semantic Frame ===
    frame_type: str = Field(
        description="Type of semantic frame: NOMINATION, COMPETITION, ACHIEVEMENT, OCCURRENCE, MEASUREMENT, or MENTION"
    )
    question_template: str = Field(
        description="Template showing semantic structure, e.g., '{actor} nominates {candidate} for {position}'"
    )
    primary_relation: str = Field(
        description="The main verb/relation in the frame: 'nominates', 'defeats', 'exceeds', 'occurs', etc."
    )
    actors: List[SemanticRoleOutput] = Field(
        default_factory=list,
        description="Entities with agency/decision power (e.g., Trump nominates, Team competes)"
    )
    objects: List[SemanticRoleOutput] = Field(
        default_factory=list,
        description="Things being acted upon (e.g., Fed Chair position, championship title)"
    )
    candidates: List[SemanticRoleOutput] = Field(
        default_factory=list,
        description="Possible outcomes/choices linked to markets (e.g., Warsh, Hassett for Fed Chair)"
    )
    mutual_exclusivity: bool = Field(
        default=True,
        description="Can only ONE outcome happen? (True for most prediction markets)"
    )
    actor_controls_outcome: bool = Field(
        default=False,
        description="Does a specific actor (like Trump) directly decide the outcome?"
    )
    resolution_trigger: str = Field(
        default="",
        description="What specific event triggers resolution? e.g., 'First formal nomination announcement'"
    )
    primary_search_queries: List[str] = Field(
        default_factory=list,
        description="3-5 targeted search queries to find news about this event"
    )
    signal_keywords: List[str] = Field(
        default_factory=list,
        description="Keywords that signal important news: 'frontrunner', 'shortlist', 'reportedly', etc."
    )

    # === Grounding Fields (v2) ===
    grounding_notes: str = Field(
        default="",
        description="Notes on what information is from context vs assumed. If making assumptions, list them here."
    )
    base_rate_source: str = Field(
        default="estimated",
        description="Source of base rate: 'historical_data', 'comparable_events', 'domain_knowledge', or 'estimated'"
    )

    # === Calibration Fields (v2) ===
    uncertainty_factors: List[str] = Field(
        default_factory=list,
        description="Key factors that create uncertainty in this analysis"
    )
    information_gaps: List[str] = Field(
        default_factory=list,
        description="What information would most improve this analysis if available?"
    )
