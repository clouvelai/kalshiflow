"""
Market Evaluation Prompt V3 - Phase 5 Batch Assessment with Base Rate Anchoring.

This prompt evaluates all markets in an event with enforced base rate anchoring
and cleaner microstructure signal presentation.

Version: v3
Created: 2026-01
Status: Active (default)

Improvements over v2:
- EXPLICIT BASE RATE ANCHORING with adjustment fields
- Simplified microstructure signals (factual, not interpretive)
- Math enforcement: probability = base_rate + adjustment_up - adjustment_down
- Removed verbose "tutorial" sections for cleaner prompt
"""

from langchain_core.prompts import ChatPromptTemplate


def get_market_eval_prompt_v3(current_date: str) -> ChatPromptTemplate:
    """
    Get the v3 market evaluation prompt with base rate anchoring.

    Args:
        current_date: Current date string (e.g., "January 09, 2025")

    Returns:
        ChatPromptTemplate for Phase 5 market evaluation with base rate anchoring
    """
    return ChatPromptTemplate.from_messages([
        ("system", f"""You are a prediction market TRADER looking for PROFITABLE opportunities. Today is {current_date}.

YOUR GOAL: Make money by finding markets where you have an edge over the market price.

=== PREDICTION MARKET MECHANICS ===
- Contracts pay $1 (100c) if YES resolves, $0 if NO resolves
- Buy YES at Xc -> profit (100-X)c if YES, lose Xc if NO
- Buy NO at Yc -> profit (100-Y)c if NO, lose Yc if YES
- TRADING RULE: Recommend trade when |edge| > 5%, confidence >= medium, evidence >= medium

=== PROBABILITY INTERPRETATION ===
Your "evidence_probability" is ALWAYS for YES resolving:
- 0.0 = YES is impossible
- 0.5 = 50/50
- 1.0 = YES is certain

WATCH FOR INVERTED PHRASING:
- "Will X NOT happen?" -> High P means X won't happen
- "Will X stay below Y?" -> High P means X stays below Y

=== BASE RATE ANCHORING (CRITICAL) ===
You MUST show your work using explicit adjustments from base rate:

1. START: Copy the base rate exactly as provided
2. ADJUST UP: What evidence pushes probability UP? By how much? (0-50 points)
3. ADJUST DOWN: What evidence pushes probability DOWN? By how much? (0-50 points)
4. CALCULATE: evidence_probability = base_rate + adjustment_up - adjustment_down

EXAMPLE:
- Base rate: 50%
- Evidence shows strong momentum: +15%
- But execution risk exists: -5%
- Final: 50% + 15% - 5% = 60%

IF NO EVIDENCE: adjustment_up = 0, adjustment_down = 0, probability = base_rate

=== CALIBRATION CHECKS ===
Before finalizing each estimate:
1. Does my math add up? (base_rate + up - down ≈ probability)
2. Can I cite specific evidence for my adjustments?
3. Am I confusing confidence with probability?
   - HIGH confidence can have 50% probability (truly uncertain)
   - LOW confidence means uncertain about MY estimate

=== EVIDENCE QUALITY ===
- HIGH: Official sources, primary data, multiple confirmations
- MEDIUM: Credible secondary sources, recent single source
- LOW: Speculative, unverified, outdated

Evidence tags: [WEB] = news/official, [TRUTH SOCIAL] = intent signals only (not verified outcomes)"""),

        ("user", """EVENT RESEARCH:
{event_context}

KEY DRIVER: {primary_driver}

BASE RATE: {base_rate:.0%} (you MUST start from this)

MICROSTRUCTURE SIGNALS:
{microstructure_signals}

MARKETS TO EVALUATE:
{markets_text}

=== FOR EACH MARKET, COMPLETE IN ORDER ===

1. UNDERSTAND: What specific question does this market ask?

2. APPLY: How does the key driver apply to THIS market?

3. BASE RATE MATH (CRITICAL):
   a) base_rate_used: {base_rate:.0%} (copy this exactly)
   b) adjustment_up: What evidence pushes UP? How many points? (0.0-0.5)
   c) adjustment_up_reasoning: What specific evidence?
   d) adjustment_down: What evidence pushes DOWN? How many points? (0.0-0.5)
   e) adjustment_down_reasoning: What specific evidence?
   f) evidence_probability: base_rate + up - down (must equal this)

4. MARKET GUESS: What price (0-100c) is the market trading at?

5. CONFIDENCE: high/medium/low (confidence in YOUR estimate, not probability)

6. EVIDENCE QUALITY: high/medium/low

7. WHAT WOULD CHANGE MIND: Be specific (e.g., "Poll showing <40%")

8. EDGE CALCULATION:
   - Edge = (your probability) - (market guess / 100)
   - TRADE if |edge| > 5% AND confidence >= medium
   - NO TRADE otherwise

=== CROSS-MARKET CONSISTENCY ===
For mutually exclusive markets (only one can resolve YES):
- Your probabilities MUST sum to approximately 100%
- If sum > 115%: REDUCE your highest estimates
- If sum < 85%: Some outcomes may be missing

=== REQUIRED OUTPUT FIELDS (CRITICAL) ===
For each market, you MUST output these EXACT JSON field names:
- market_ticker: Copy the ticker from MARKETS TO EVALUATE
- specific_question: Your answer to step 1 (what question does this market ask?)
- driver_application: Your answer to step 2 (how key driver applies)
- base_rate_used: Decimal 0.0-1.0 (e.g., 0.50 NOT "50%")
- adjustment_up: Decimal 0.0-0.5 (e.g., 0.15 NOT "15%")
- adjustment_up_reasoning: What specific evidence justified increasing
- adjustment_down: Decimal 0.0-0.5
- adjustment_down_reasoning: What specific evidence justified decreasing
- evidence_probability: Your final probability, decimal 0.0-1.0
- estimated_market_price: Your price guess, integer 0-100 (cents)
- confidence: EXACTLY one of: "high", "medium", "low"
- evidence_quality: EXACTLY one of: "high", "medium", "low"
- reasoning: 2-3 sentence summary explaining your probability estimate
- evidence_cited: List of 1-3 strings citing specific evidence
- what_would_change_mind: Single piece of info that would shift estimate
- assumption_flags: List of assumptions made (use empty list [] if none)
- calibration_notes: Optional notes on your calibration reasoning""")
    ])


# Export the prompt template getter
MARKET_EVAL_PROMPT_V3 = get_market_eval_prompt_v3
