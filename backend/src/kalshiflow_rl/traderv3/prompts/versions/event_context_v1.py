"""
Event Context Prompt V1 - Phase 2+3 Combined Extraction.

This prompt extracts event context, key driver, and semantic frame
in a single LLM call for efficiency.

Version: v1 (baseline)
Created: 2025-01
Status: Production baseline

Known Issues:
- May generate ungrounded probabilities
- Base rates sometimes lack clear reasoning
- Semantic frame disambiguation could be clearer
"""

from langchain_core.prompts import ChatPromptTemplate


def get_event_context_prompt_v1(current_date: str) -> ChatPromptTemplate:
    """
    Get the v1 event context extraction prompt.

    Args:
        current_date: Current date string (e.g., "January 09, 2025")

    Returns:
        ChatPromptTemplate for Phase 2+3 extraction
    """
    return ChatPromptTemplate.from_messages([
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


# Export the prompt template getter
EVENT_CONTEXT_PROMPT_V1 = get_event_context_prompt_v1
