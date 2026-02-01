"""
Base extraction prompt for the KalshiExtractor.

Defines the core extraction instructions, classes, and output format
that langextract uses to parse Reddit/news text into structured signals.

The base prompt is MARKET-AGNOSTIC: it defines extraction taxonomy and quality
guidelines without referencing any specific markets, tickers, or market types.

Market-specific context (active markets, market type semantics, event-specific
instructions) is injected dynamically by _build_merged_prompt() from event_configs.
"""

BASE_EXTRACTION_PROMPT = """You are a structured information extraction system for prediction markets.

Extract structured information from social media posts, news articles, and discussions
that may impact prediction market outcomes.

## Extraction Classes

### market_signal
A direct signal about a specific prediction market's outcome.
Attributes:
- market_ticker: str — The market ticker this signal applies to
- direction: str — "bullish" (YES price should go up) or "bearish" (YES price should go down)
- magnitude: int — 0-100 estimate of directional strength (0=noise, 100=definitive)
- confidence: str — "low", "medium", or "high"
- reasoning: str — 1-2 sentence explanation of WHY this impacts the market

### entity_mention
A named entity mentioned with sentiment context.
Attributes:
- entity_name: str — Canonical name (e.g., "Donald Trump", not "Trump")
- entity_type: str — PERSON | ORG | GPE | POLICY | EVENT | OUTCOME
- sentiment: int — -100 to +100 sentiment about this entity in context
- confidence: str — "low", "medium", or "high"

### context_factor
A background factor that may influence market outcomes.
Attributes:
- category: str — economic | political | social | legal | environmental
- relevance: str — "low", "medium", or "high"
- direction: str — "positive", "negative", or "neutral"
- description: str — Brief description of the factor

## Quality Guidelines

1. For entity_mention, use canonical names (full names, official organization names).
2. magnitude should reflect the STRENGTH of the signal, not just whether it exists:
   - 0-20: Tangential, weak connection
   - 20-50: Relevant but not decisive
   - 50-80: Directly relevant, clear impact
   - 80-100: Major breaking news directly about this outcome
3. If the text has NO relevance to any active market, extract only entity_mentions and context_factors.
4. Extract from the FULL text — title, body, and any additional content provided.
5. Source metadata (subreddit, engagement) is provided for context but should not bias extraction.
"""

# Market context template injected by _build_merged_prompt()
MARKET_CONTEXT_TEMPLATE = """
## Market-Specific Rules

1. ONLY extract market_signal when you can confidently link to a specific market ticker from the ACTIVE MARKETS list below.
2. For OUT markets (person leaving office): negative news about the person = BULLISH (more likely they leave).
3. For standard markets: positive news = BULLISH, negative news = BEARISH.
"""

# Template for formatting input text with metadata
INPUT_TEXT_TEMPLATE = """SOURCE: {source_type}
SUBREDDIT: r/{subreddit}
ENGAGEMENT: {score} upvotes, {comments} comments

TITLE: {title}

{body}"""
