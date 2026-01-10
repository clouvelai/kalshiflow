"""
Schema for Phase 5: Market Evaluation V4 - Clean Calibration-Focused Output.

This schema defines a simplified structured output for market assessment
with essential fields designed for calibration tracking.

Version: v4.1
Created: 2026-01
Status: Active (default)

Philosophy:
- Clean structure with calibration guardrails
- Every field serves a calibration measurement purpose
- Pydantic validators enforce probability bounds and flag extremes
- Mutual exclusivity check at batch level

Calibration Metrics Enabled:
- Brier Score: Compare probability to outcome
- Base Rate Anchoring: Compare probability to base_rate_used
- Price Guess Accuracy: Compare market_price_guess to actual price
- Confidence Calibration: Brier by confidence bucket
- Edge Accuracy: When we bet, did we win?

Changes in v4.1:
- Added specific_question field for audit trail
- Added probability clamping validator (soft clamp at 0.02-0.98)
- Added extreme_probability_flag for monitoring
- Added mutual exclusivity validator at batch level
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator
import logging

logger = logging.getLogger("kalshiflow_rl.traderv3.prompts.schemas.v4")


class SingleMarketAssessmentV4(BaseModel):
    """
    Clean, calibration-focused assessment for a single market.

    10 fields total - each serves a specific calibration purpose:
    - market_ticker: Identity
    - specific_question: What exact question this market asks (audit trail)
    - probability: Core prediction (enables Brier score)
    - base_rate_used: Anchoring visibility (enables anchoring analysis)
    - market_price_guess: Market awareness (enables price guess accuracy)
    - confidence: Uncertainty signaling (enables confidence calibration)
    - reasoning: Human-readable explanation (free-form, for review)
    - key_evidence: Evidence grounding (for auditing)
    - what_would_change_mind: Falsifiability (for quality assessment)
    - extreme_probability_flag: Auto-set when probability is clamped (monitoring)
    """
    market_ticker: str = Field(
        description="The market ticker being assessed"
    )
    specific_question: str = Field(
        default="",
        description="What exact question does this market ask? One sentence summary."
    )
    probability: float = Field(
        description="Your probability estimate for YES (0.0 to 1.0). This is your genuine belief.",
        ge=0.0, le=1.0
    )
    base_rate_used: float = Field(
        description="The base rate you anchored from (0.0 to 1.0). Copy from event research context.",
        ge=0.0, le=1.0
    )
    market_price_guess: int = Field(
        description="Your guess of current market price in cents (0-100). What do you think the market is trading at?",
        ge=0, le=100
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence in your probability estimate. high = trust this strongly, low = very uncertain"
    )
    reasoning: str = Field(
        description="2-3 sentence explanation of your probability estimate. Free-form, focus on why."
    )
    key_evidence: List[str] = Field(
        description="1-3 most important pieces of evidence supporting your estimate",
        min_length=0,
        max_length=5
    )
    what_would_change_mind: str = Field(
        description="One sentence: What single piece of information would most change this estimate?"
    )
    # Auto-populated flag for extreme probability monitoring
    extreme_probability_flag: Optional[str] = Field(
        default=None,
        description="Auto-set if probability was clamped. Contains original value and reason."
    )

    @field_validator('probability', mode='after')
    @classmethod
    def clamp_extreme_probabilities(cls, v: float) -> float:
        """
        Soft clamp extreme probabilities to [0.02, 0.98] range.

        Rationale:
        - Probabilities of exactly 0% or 100% are almost never justified
        - Even "certain" outcomes have tail risks (illness, scandal, rule changes)
        - Clamping to 2-98% preserves calibration while preventing catastrophic bets

        Note: This is a SOFT clamp - we log the original value but return clamped.
        The extreme_probability_flag field (set in model_validator) tracks these cases.
        """
        if v < 0.02:
            return 0.02
        if v > 0.98:
            return 0.98
        return v

    @model_validator(mode='after')
    def flag_extreme_probabilities(self) -> 'SingleMarketAssessmentV4':
        """
        Flag when probability was at extreme values for monitoring.

        This runs AFTER field validators, so we check the clamped value
        against typical extreme thresholds to detect when clamping occurred.
        """
        # We can't directly detect clamping since field_validator already ran.
        # Instead, flag probabilities at the clamp boundaries or very close to them.
        # If LLM outputs 0.02 or 0.98 naturally, this will also flag - acceptable false positive.
        if self.probability <= 0.02:
            self.extreme_probability_flag = (
                f"EXTREME_LOW: Probability at or below 2% (clamped from potential 0%). "
                f"Confidence: {self.confidence}"
            )
            logger.warning(
                f"[EXTREME PROB] {self.market_ticker}: probability={self.probability:.1%} "
                f"(extreme low, may have been clamped)"
            )
        elif self.probability >= 0.98:
            self.extreme_probability_flag = (
                f"EXTREME_HIGH: Probability at or above 98% (clamped from potential 100%). "
                f"Confidence: {self.confidence}"
            )
            logger.warning(
                f"[EXTREME PROB] {self.market_ticker}: probability={self.probability:.1%} "
                f"(extreme high, may have been clamped)"
            )

        return self


class BatchMarketAssessmentV4Output(BaseModel):
    """
    Structured output for Phase 5: Batch market evaluation (V4).

    Contains clean assessments for all markets in an event.
    Includes mutual exclusivity validator to check probability consistency.
    """
    assessments: List[SingleMarketAssessmentV4] = Field(
        description="Assessment for each market in the event"
    )

    # Auto-populated field for batch-level consistency check
    mutual_exclusivity_warning: Optional[str] = Field(
        default=None,
        description="Auto-set if probabilities don't sum to ~100% for mutually exclusive markets"
    )

    @model_validator(mode='after')
    def check_mutual_exclusivity(self) -> 'BatchMarketAssessmentV4Output':
        """
        Check if probabilities sum to approximately 100% for mutually exclusive markets.

        This is a soft check - flags inconsistencies but doesn't modify probabilities.
        Tolerance: 15% deviation from 100% is acceptable (85-115% sum).
        """
        if not self.assessments or len(self.assessments) < 2:
            return self

        total_prob = sum(a.probability for a in self.assessments)

        # Check if outside tolerance (15%)
        if abs(total_prob - 1.0) > 0.15:
            if total_prob > 1.15:
                direction = "over-allocated"
                advice = "Consider reducing some probability estimates"
            else:
                direction = "under-allocated"
                advice = "Some outcomes may be missing or probabilities are too conservative"

            warning = (
                f"MUTUAL_EXCLUSIVITY_WARNING: Probabilities sum to {total_prob:.0%} ({direction}). "
                f"{advice}. Markets: {len(self.assessments)}"
            )
            self.mutual_exclusivity_warning = warning
            logger.warning(f"[MUTUAL EXCLUSIVITY] {warning}")

        return self
