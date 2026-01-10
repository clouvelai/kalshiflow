"""
Schema for Phase 5: Market Evaluation structured output.

This schema defines the structured output for batch market assessment
where all markets in an event are evaluated with shared context.

Version History:
    v1 (2025-01): Initial schema with basic fields
    v2 (2025-01): Added evidence_cited, what_would_change_mind, assumption_flags, calibration_notes
"""

from typing import List, Optional
from pydantic import BaseModel, Field, model_validator


class SingleMarketAssessment(BaseModel):
    """
    Assessment for a single market within an event.

    This schema captures the LLM's analysis of a specific market question,
    including probability estimate, confidence, and reasoning.

    CALIBRATION REQUIREMENTS (v2):
    - Probability must be justified by evidence or base rate
    - Confidence must reflect evidence quality and agreement
    - what_would_change_mind must be specific and measurable
    - If assumptions are made, they must be flagged
    """
    market_ticker: str = Field(
        description="The market ticker being assessed"
    )
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

    # === Calibration Fields (v2 additions - all with defaults for backward compatibility) ===
    evidence_cited: List[str] = Field(
        default_factory=list,
        description="Which specific evidence points support this estimate? List 1-3 key pieces of evidence."
    )
    what_would_change_mind: str = Field(
        default="",
        description="What single piece of information would most change this probability estimate?"
    )
    assumption_flags: List[str] = Field(
        default_factory=list,
        description="List any assumptions made due to missing information (e.g., 'Assumed no major news since last search')"
    )
    calibration_notes: str = Field(
        default="",
        description="Notes on confidence calibration: Why this confidence level? What could make you more/less confident?"
    )

    # === Source Quality (v2 addition) ===
    evidence_quality: str = Field(
        default="medium",
        description="Quality of evidence supporting this assessment: high (official/primary sources), medium (credible secondary), low (unverified/speculative)"
    )


class BatchMarketAssessmentOutput(BaseModel):
    """
    Structured output for Phase 5: Batch market evaluation.

    Contains assessments for all markets in an event.
    """
    assessments: List[SingleMarketAssessment] = Field(
        description="Assessment for each market in the event"
    )

    # === Batch-level calibration (v2 addition) ===
    cross_market_consistency_check: str = Field(
        default="",
        description="For mutually exclusive markets: Do probabilities sum to ~100%? Any inconsistencies noted?"
    )

    @model_validator(mode='after')
    def validate_mutual_exclusivity(self) -> 'BatchMarketAssessmentOutput':
        """
        Check if probabilities sum to ~100% for what appears to be mutually exclusive markets.

        This is a post-validation hook that flags when probabilities are inconsistent.
        It does NOT prevent invalid outputs - it flags them for downstream handling.

        Tolerance: 15% deviation from 100% is acceptable (85-115% sum).
        """
        if not self.assessments or len(self.assessments) < 2:
            return self

        # Sum all probabilities
        total_prob = sum(a.evidence_probability for a in self.assessments)

        # Check if outside tolerance (15%)
        if abs(total_prob - 1.0) > 0.15:
            # Determine if over or under
            if total_prob > 1.15:
                direction = "over-allocated"
                instruction = "reduce some probabilities"
            else:
                direction = "under-allocated"
                instruction = "some outcomes may be missing or under-weighted"

            warning = f"MUTUAL EXCLUSIVITY WARNING: Probabilities sum to {total_prob:.0%} ({direction}). If these are mutually exclusive outcomes, {instruction}."

            # Add warning to each assessment's assumption_flags
            for assessment in self.assessments:
                if warning not in assessment.assumption_flags:
                    assessment.assumption_flags.append(warning)

            # Also update the batch-level check if it wasn't set
            if not self.cross_market_consistency_check:
                self.cross_market_consistency_check = warning

        return self
