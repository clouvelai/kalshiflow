"""
Market Evaluation Prompt V1 - Phase 5 Batch Assessment.

This prompt evaluates all markets in an event with shared context,
producing probability estimates and trade recommendations.

Version: v1 (baseline)
Created: 2025-01
Status: Production baseline

Known Issues:
- May produce overconfident estimates
- Evidence citation is implicit, not explicit
- No explicit calibration self-check
"""

from langchain_core.prompts import ChatPromptTemplate


def get_market_eval_prompt_v1(current_date: str) -> ChatPromptTemplate:
    """
    Get the v1 market evaluation prompt.

    Args:
        current_date: Current date string (e.g., "January 09, 2025")

    Returns:
        ChatPromptTemplate for Phase 5 market evaluation
    """
    return ChatPromptTemplate.from_messages([
        ("system", f"""You are a prediction market TRADER looking for PROFITABLE opportunities. Today is {current_date}.

YOUR GOAL: Make money by finding markets where you have an edge over the market price.

HOW PREDICTION MARKETS WORK:
- Contracts pay $1 (100c) if YES resolves, $0 if NO resolves
- Buy YES at Xc -> profit (100-X)c if YES, lose Xc if NO
- Buy NO at Yc -> profit (100-Y)c if NO, lose Yc if YES
- EXPECTED VALUE: EV = P(correct) * profit - P(wrong) * loss
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
- If you think YES is more likely than the price implies -> buy YES
- If you think NO is more likely than the price implies -> buy NO
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


# Export the prompt template getter
MARKET_EVAL_PROMPT_V1 = get_market_eval_prompt_v1
