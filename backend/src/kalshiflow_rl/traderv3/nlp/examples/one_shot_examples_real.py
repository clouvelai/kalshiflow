"""
Market-agnostic synthetic examples for the KalshiExtractor.

Each example teaches a specific extraction BEHAVIOR pattern using placeholder
tickers. The model learns HOW to extract, not WHAT specific markets exist.

Market-specific examples come from event_configs.examples via understand_event().

Example behaviors taught:
1. High-magnitude market_signal — breaking news directly about an outcome
2. Low-magnitude entity_mention + context_factor — tangential news (teaches restraint)
3. Medium-magnitude market_signal + entity_mention — relevant but not decisive
4. Irrelevant content — teaches to extract only context_factor with low relevance
5. Multi-extraction — one post with signals for two different markets
"""

REAL_EXAMPLES: list = [
    # Example 1: HIGH-MAGNITUDE MARKET SIGNAL
    # Teaches: Breaking news directly about an outcome → high magnitude, high confidence
    {
        "input": (
            "SOURCE: reddit_post\n"
            "SUBREDDIT: r/politics\n"
            "ENGAGEMENT: 6200 upvotes, 1100 comments\n\n"
            "TITLE: Cabinet Secretary announces resignation effective immediately\n\n"
            "The Secretary confirmed in a press conference that they are stepping down "
            "from their position, citing personal reasons. The White House has accepted "
            "the resignation."
        ),
        "output": [
            {
                "extraction_class": "market_signal",
                "extraction_text": "Cabinet Secretary announces resignation effective immediately",
                "attributes": {
                    "market_ticker": "TICKER-EXAMPLE-01",
                    "direction": "bullish",
                    "magnitude": 90,
                    "confidence": "high",
                    "reasoning": "Direct announcement of resignation is near-definitive evidence of departure from office",
                },
            },
            {
                "extraction_class": "entity_mention",
                "extraction_text": "The Secretary confirmed in a press conference that they are stepping down",
                "attributes": {
                    "entity_name": "Cabinet Secretary",
                    "entity_type": "PERSON",
                    "sentiment": -30,
                    "confidence": "high",
                },
            },
        ],
    },
    # Example 2: LOW-MAGNITUDE / TANGENTIAL NEWS (teaches restraint)
    # Teaches: Tangential mention → entity_mention + context_factor only, NO market_signal
    {
        "input": (
            "SOURCE: reddit_post\n"
            "SUBREDDIT: r/news\n"
            "ENGAGEMENT: 450 upvotes, 90 comments\n\n"
            "TITLE: Former aide publishes memoir about time in administration\n\n"
            "A former White House aide has published a tell-all book about their experience. "
            "The book covers several policy debates but does not contain any major revelations "
            "beyond what was already publicly known."
        ),
        "output": [
            {
                "extraction_class": "entity_mention",
                "extraction_text": "Former aide publishes memoir about time in administration",
                "attributes": {
                    "entity_name": "White House",
                    "entity_type": "ORG",
                    "sentiment": -10,
                    "confidence": "medium",
                },
            },
            {
                "extraction_class": "context_factor",
                "extraction_text": "does not contain any major revelations beyond what was already publicly known",
                "attributes": {
                    "category": "political",
                    "relevance": "low",
                    "direction": "neutral",
                    "description": "Tell-all memoir with no new revelations has minimal market impact",
                },
            },
        ],
    },
    # Example 3: MEDIUM-MAGNITUDE SIGNAL + ENTITY
    # Teaches: Relevant but not decisive → medium magnitude, includes reasoning
    {
        "input": (
            "SOURCE: reddit_post\n"
            "SUBREDDIT: r/politics\n"
            "ENGAGEMENT: 3100 upvotes, 520 comments\n\n"
            "TITLE: Congressional committee subpoenas documents related to trade policy\n\n"
            "The Senate Finance Committee has issued subpoenas for documents related to "
            "upcoming trade negotiations. Sources say the investigation could delay the "
            "planned implementation timeline by several weeks."
        ),
        "output": [
            {
                "extraction_class": "market_signal",
                "extraction_text": "investigation could delay the planned implementation timeline by several weeks",
                "attributes": {
                    "market_ticker": "TICKER-EXAMPLE-02",
                    "direction": "bearish",
                    "magnitude": 45,
                    "confidence": "medium",
                    "reasoning": "Congressional subpoena could delay implementation, reducing near-term probability but not definitively blocking it",
                },
            },
            {
                "extraction_class": "entity_mention",
                "extraction_text": "Senate Finance Committee has issued subpoenas",
                "attributes": {
                    "entity_name": "Senate Finance Committee",
                    "entity_type": "ORG",
                    "sentiment": 0,
                    "confidence": "high",
                },
            },
        ],
    },
    # Example 4: IRRELEVANT CONTENT (the "null" example)
    # Teaches: Content with no market relevance → context_factor only, low relevance
    {
        "input": (
            "SOURCE: reddit_post\n"
            "SUBREDDIT: r/politics\n"
            "ENGAGEMENT: 280 upvotes, 65 comments\n\n"
            "TITLE: Local mayor announces new recycling program for downtown district\n\n"
            "The mayor of a mid-sized city unveiled a new curbside recycling initiative "
            "that will expand service to the downtown area starting next month."
        ),
        "output": [
            {
                "extraction_class": "context_factor",
                "extraction_text": "Local mayor announces new recycling program for downtown district",
                "attributes": {
                    "category": "social",
                    "relevance": "low",
                    "direction": "neutral",
                    "description": "Local municipal policy with no connection to federal or prediction market outcomes",
                },
            },
        ],
    },
    # Example 5: MULTI-EXTRACTION (one post, two market signals)
    # Teaches: A single source can generate signals for multiple markets
    {
        "input": (
            "SOURCE: reddit_post\n"
            "SUBREDDIT: r/news\n"
            "ENGAGEMENT: 7500 upvotes, 1800 comments\n\n"
            "TITLE: President announces sweeping executive action on both trade and immigration\n\n"
            "In a surprise move, the President signed two executive orders: one imposing new "
            "tariffs on imported goods, and another expanding border enforcement measures. "
            "Both orders take effect within 30 days."
        ),
        "output": [
            {
                "extraction_class": "market_signal",
                "extraction_text": "signed executive order imposing new tariffs on imported goods",
                "attributes": {
                    "market_ticker": "TICKER-EXAMPLE-03",
                    "direction": "bullish",
                    "magnitude": 75,
                    "confidence": "high",
                    "reasoning": "Direct presidential action signing tariff order makes implementation highly likely within 30 days",
                },
            },
            {
                "extraction_class": "market_signal",
                "extraction_text": "executive order expanding border enforcement measures",
                "attributes": {
                    "market_ticker": "TICKER-EXAMPLE-04",
                    "direction": "bullish",
                    "magnitude": 70,
                    "confidence": "high",
                    "reasoning": "Signed executive order on border enforcement directly increases probability of expanded enforcement actions",
                },
            },
            {
                "extraction_class": "entity_mention",
                "extraction_text": "President signed two executive orders",
                "attributes": {
                    "entity_name": "President",
                    "entity_type": "PERSON",
                    "sentiment": 0,
                    "confidence": "high",
                },
            },
            {
                "extraction_class": "context_factor",
                "extraction_text": "Both orders take effect within 30 days",
                "attributes": {
                    "category": "political",
                    "relevance": "high",
                    "direction": "positive",
                    "description": "Near-term implementation timeline increases urgency of signals for related markets",
                },
            },
        ],
    },
]
