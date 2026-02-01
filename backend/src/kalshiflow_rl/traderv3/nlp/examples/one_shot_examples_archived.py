"""
ARCHIVED real examples with market-specific tickers.

These were the original examples that referenced specific Kalshi tickers
(KXBONDIOUT-26APR30, KXTRUMPTARIFF-26FEB07, etc.) which become stale
when those markets expire.

Kept for reference. Market-specific examples now come exclusively from
event_configs.examples via the understand_event() pipeline.

Replaced by synthetic, market-agnostic examples in one_shot_examples_real.py
that teach extraction *behavior* rather than specific market knowledge.
"""

ARCHIVED_REAL_EXAMPLES: list = [
    {
        "input": (
            "SOURCE: reddit_post\n"
            "SUBREDDIT: r/politics\n"
            "ENGAGEMENT: 2847 upvotes, 531 comments\n\n"
            "TITLE: Pam Bondi under fire as new ethics probe launched into AG's handling of cases\n\n"
            "The Justice Department inspector general has opened a review into Attorney General "
            "Pam Bondi's decisions on several high-profile cases, raising questions about potential "
            "conflicts of interest."
        ),
        "output": [
            {
                "extraction_class": "market_signal",
                "extraction_text": "ethics probe launched into AG's handling of cases",
                "attributes": {
                    "market_ticker": "KXBONDIOUT-26APR30",
                    "direction": "bullish",
                    "magnitude": 55,
                    "confidence": "medium",
                    "reasoning": "Ethics probe into AG increases probability of departure from office",
                },
            },
            {
                "extraction_class": "entity_mention",
                "extraction_text": "Pam Bondi under fire as new ethics probe launched",
                "attributes": {
                    "entity_name": "Pam Bondi",
                    "entity_type": "PERSON",
                    "sentiment": -45,
                    "confidence": "high",
                },
            },
        ],
    },
    {
        "input": (
            "SOURCE: reddit_post\n"
            "SUBREDDIT: r/news\n"
            "ENGAGEMENT: 5120 upvotes, 892 comments\n\n"
            "TITLE: Trump says he will sign executive order on tariffs next week\n\n"
            "President Trump announced plans to sign a sweeping executive order imposing new "
            "tariffs on Chinese imports, escalating trade tensions."
        ),
        "output": [
            {
                "extraction_class": "market_signal",
                "extraction_text": "Trump says he will sign executive order on tariffs next week",
                "attributes": {
                    "market_ticker": "KXTRUMPTARIFF-26FEB07",
                    "direction": "bullish",
                    "magnitude": 75,
                    "confidence": "high",
                    "reasoning": "Direct statement from Trump about signing tariff EO makes it highly likely",
                },
            },
            {
                "extraction_class": "entity_mention",
                "extraction_text": "Trump announced plans to sign a sweeping executive order",
                "attributes": {
                    "entity_name": "Donald Trump",
                    "entity_type": "PERSON",
                    "sentiment": 0,
                    "confidence": "high",
                },
            },
            {
                "extraction_class": "context_factor",
                "extraction_text": "escalating trade tensions",
                "attributes": {
                    "category": "economic",
                    "relevance": "high",
                    "direction": "negative",
                    "description": "New tariffs on Chinese imports escalate trade war",
                },
            },
        ],
    },
    {
        "input": (
            "SOURCE: reddit_post\n"
            "SUBREDDIT: r/politics\n"
            "ENGAGEMENT: 312 upvotes, 87 comments\n\n"
            "TITLE: Local school board votes to ban cellphones in classrooms\n\n"
            "A school board in suburban Ohio voted to ban student cellphone use during class."
        ),
        "output": [
            {
                "extraction_class": "context_factor",
                "extraction_text": "school board votes to ban cellphones in classrooms",
                "attributes": {
                    "category": "social",
                    "relevance": "low",
                    "direction": "neutral",
                    "description": "Local education policy change with no federal market impact",
                },
            },
        ],
    },
    {
        "input": (
            "SOURCE: reddit_post\n"
            "SUBREDDIT: r/politics\n"
            "ENGAGEMENT: 8930 upvotes, 2100 comments\n\n"
            "TITLE: Government shutdown averted: Congress passes stopgap spending bill hours before deadline\n\n"
            "Both chambers of Congress passed a continuing resolution to fund the government "
            "through March 15, narrowly avoiding a shutdown."
        ),
        "output": [
            {
                "extraction_class": "market_signal",
                "extraction_text": "Government shutdown averted: Congress passes stopgap spending bill",
                "attributes": {
                    "market_ticker": "KXGOVSHUT-26FEB28",
                    "direction": "bearish",
                    "magnitude": 80,
                    "confidence": "high",
                    "reasoning": "Stopgap bill passed means no shutdown this cycle, reduces near-term shutdown probability",
                },
            },
            {
                "extraction_class": "entity_mention",
                "extraction_text": "Both chambers of Congress passed a continuing resolution",
                "attributes": {
                    "entity_name": "U.S. Congress",
                    "entity_type": "ORG",
                    "sentiment": 30,
                    "confidence": "high",
                },
            },
        ],
    },
    {
        "input": (
            "SOURCE: reddit_post\n"
            "SUBREDDIT: r/news\n"
            "ENGAGEMENT: 1540 upvotes, 340 comments\n\n"
            "TITLE: Pete Hegseth accused of sexual misconduct by former colleague\n\n"
            "A former Fox News colleague has publicly accused Defense Secretary Pete Hegseth "
            "of sexual misconduct, adding to a growing list of controversies around his tenure."
        ),
        "output": [
            {
                "extraction_class": "market_signal",
                "extraction_text": "Pete Hegseth accused of sexual misconduct by former colleague",
                "attributes": {
                    "market_ticker": "KXHEGSETHOUT-26MAR31",
                    "direction": "bullish",
                    "magnitude": 60,
                    "confidence": "medium",
                    "reasoning": "Sexual misconduct accusation increases probability of departure from office",
                },
            },
            {
                "extraction_class": "entity_mention",
                "extraction_text": "Pete Hegseth accused of sexual misconduct",
                "attributes": {
                    "entity_name": "Pete Hegseth",
                    "entity_type": "PERSON",
                    "sentiment": -65,
                    "confidence": "high",
                },
            },
        ],
    },
]
