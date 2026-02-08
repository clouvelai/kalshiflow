"""
Pydantic v2 schemas for LLM structured output extraction.

Used with LangChain's `llm.with_structured_output(Schema)` to replace
manual `json.loads()` + markdown stripping across all LLM call sites.

Schemas:
  - CausalModelExtraction: causal_model.py CausalModelBuilder.build()
  - LifecycleResult: lifecycle_classifier.py LifecycleClassifier.classify()
  - EventUnderstandingExtraction: event_understanding.py LLM synthesis
  - SpeakerExtraction: mentions_context.py _llm_extract_speakers_from_text()
  - WikipediaSpeakerExtraction: mentions_context.py _llm_extract_speaker_from_wikipedia()
  - LexemePackExtraction: mentions_tools.py _llm_parse_rules()
"""

from typing import List, Optional

from pydantic import BaseModel, Field


# =============================================================================
# causal_model.py - CausalModelBuilder.build()
# =============================================================================


class CausalDriverMarketLink(BaseModel):
    market_ticker: str = ""
    direction: str = "neutral"
    magnitude: float = Field(default=0.5, ge=0.0, le=1.0)
    mechanism: str = ""


class CausalDriverExtraction(BaseModel):
    name: str
    direction: str = "neutral"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    market_links: List[CausalDriverMarketLink] = Field(default_factory=list)
    evidence: str = ""


class CausalCatalystExtraction(BaseModel):
    name: str
    type: str = "expected"
    expected_date: str = ""
    affected_markets: List[str] = Field(default_factory=list)
    magnitude: float = Field(default=0.5, ge=0.0, le=1.0)


class CausalModelExtraction(BaseModel):
    """Schema for CausalModelBuilder.build() LLM output."""
    drivers: List[CausalDriverExtraction] = Field(default_factory=list)
    catalysts: List[CausalCatalystExtraction] = Field(default_factory=list)
    dominant_narrative: str = ""
    consensus_direction: str = "unclear"
    uncertainty_level: float = Field(default=0.5, ge=0.0, le=1.0)


# =============================================================================
# lifecycle_classifier.py - LifecycleClassifier.classify()
# =============================================================================


class LifecycleResult(BaseModel):
    """Schema for LifecycleClassifier.classify() LLM output."""
    stage: str = "discovery"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    reasoning: str = ""
    recommended_action: str = "observe"


# =============================================================================
# event_understanding.py - LLM synthesis
# =============================================================================


class TimelineSegment(BaseModel):
    name: str = ""
    start_offset_min: int = 0
    duration_min: int = 0
    description: str = ""


class EventUnderstandingExtraction(BaseModel):
    """Schema for event_understanding.py LLM synthesis output."""
    trading_summary: str = ""
    key_factors: List[str] = Field(default_factory=list)
    trading_considerations: List[str] = Field(default_factory=list)
    timeline: List[TimelineSegment] = Field(default_factory=list)


# =============================================================================
# mentions_context.py - _llm_extract_speakers_from_text()
# =============================================================================


class ExtractedSpeakerSchema(BaseModel):
    name: str = ""
    title: Optional[str] = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class SpeakerExtraction(BaseModel):
    """Schema for _llm_extract_speakers_from_text() LLM output."""
    speakers: List[ExtractedSpeakerSchema] = Field(default_factory=list)


# =============================================================================
# mentions_context.py - _llm_extract_speaker_from_wikipedia()
# =============================================================================


class WikipediaSpeakerExtraction(BaseModel):
    """Schema for _llm_extract_speaker_from_wikipedia() LLM output."""
    full_name: str = ""
    title: str = ""
    style_description: str = ""
    known_phrases: List[str] = Field(default_factory=list)
    background_relevant_to_speech: str = ""


# =============================================================================
# mentions_tools.py - _llm_parse_rules()
# =============================================================================


class LexemePackExtraction(BaseModel):
    """Schema for _llm_parse_rules() LLM output."""
    entity: str = ""
    accepted_forms: List[str] = Field(default_factory=list)
    prohibited_forms: List[str] = Field(default_factory=list)
    source_type: str = "any"
    speaker: Optional[str] = None
    time_window_start: Optional[str] = None
    time_window_end: Optional[str] = None
