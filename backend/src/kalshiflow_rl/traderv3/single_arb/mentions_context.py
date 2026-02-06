"""
Context gathering for mentions simulation.

Three layers of context:
1. EventContext - Structured facts about the event (Wikipedia-sourced)
2. EntityContext - What/who is each mention term (NameNet-style disambiguation)
3. TermRelevance - How each term connects to THIS specific event

Supports two simulation modes:
- BLIND: Generic event context only (baseline probability)
- INFORMED: Full context including term relevance and current news
"""

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import quote_plus

logger = logging.getLogger("kalshiflow_rl.traderv3.single_arb.mentions_context")


# =============================================================================
# EVENT CONTEXT CACHE (4-hour TTL, same pattern as transcript caching)
# =============================================================================

_CONTEXT_CACHE_TTL = 4 * 60 * 60  # 4 hours
_context_cache_dir: Optional[str] = None  # Set via set_context_cache_dir()


def set_context_cache_dir(cache_dir: str) -> None:
    """Set directory for EventContext cache (called by coordinator)."""
    global _context_cache_dir
    _context_cache_dir = cache_dir
    os.makedirs(cache_dir, exist_ok=True)


def _get_cached_context(event_ticker: str) -> Optional["EventContext"]:
    """Load cached context if fresh (within TTL)."""
    if not _context_cache_dir:
        return None
    path = os.path.join(_context_cache_dir, f"context_{event_ticker}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if time.time() - data.get("cached_at", 0) > _CONTEXT_CACHE_TTL:
            return None  # Expired
        return EventContext.from_dict(data["context"])
    except (json.JSONDecodeError, OSError, KeyError):
        return None


def _cache_context(event_ticker: str, ctx: "EventContext") -> None:
    """Cache context to disk."""
    if not _context_cache_dir:
        return
    path = os.path.join(_context_cache_dir, f"context_{event_ticker}.json")
    try:
        with open(path, "w") as f:
            json.dump({"context": ctx.to_dict(), "cached_at": time.time()}, f)
    except OSError:
        pass  # Silently fail, caching is optional


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class EventContext:
    """Structured facts about the event from Wikipedia.

    This is the foundation - who's playing, where, who's calling it.
    Used in BOTH blind and informed simulations.
    """

    event_ticker: str
    event_title: str
    domain: str  # "sports", "corporate", "politics", "entertainment"

    # Event identity
    event_type: str = ""  # "Super Bowl LX", "Q4 2025 Earnings Call"
    date: str = ""

    # Participants (teams, companies, speakers)
    participants: List[str] = field(default_factory=list)
    participant_details: Dict[str, str] = field(default_factory=dict)  # name -> Wikipedia summary

    # Key people
    key_figures: List[str] = field(default_factory=list)  # Players, executives, etc.
    figure_details: Dict[str, str] = field(default_factory=dict)  # name -> Wikipedia summary

    # Broadcast/presentation
    announcers: List[str] = field(default_factory=list)
    network: str = ""

    # Venue
    venue: str = ""
    location: str = ""

    # Historical context (past events of same type)
    historical_notes: List[str] = field(default_factory=list)

    # Source tracking
    wikipedia_urls: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EventContext":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class EntityContext:
    """NameNet-style entity resolution for a mention term.

    Answers: What/who IS this term? Disambiguated.
    """

    term: str

    # Disambiguation
    entity_type: str = ""  # "person", "phrase", "organization", "concept"
    canonical_name: str = ""  # "Taylor Swift" not "taylor swift"
    description: str = ""  # One-line Wikipedia description

    # Wikipedia data
    wikipedia_url: str = ""
    wikipedia_extract: str = ""  # First paragraph

    # Related entities (for co-occurrence hints)
    related_entities: List[str] = field(default_factory=list)

    # Category hints
    categories: List[str] = field(default_factory=list)  # "American singer", "NFL player"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TermRelevance:
    """How a specific term connects to a specific event.

    This is the KEY for informed simulation - why would this term be mentioned
    at THIS event specifically?
    """

    term: str
    event_ticker: str

    # Relevance assessment
    relevance_type: str = ""  # "direct_participant", "related_person", "current_news", "common_phrase", "unrelated"
    relevance_explanation: str = ""  # "Taylor Swift is dating Travis Kelce (Chiefs TE)"

    # Connection chain (how term links to event)
    connection_path: List[str] = field(default_factory=list)  # ["Taylor Swift", "dating", "Travis Kelce", "Chiefs", "Super Bowl"]

    # Relevance score (for ranking)
    relevance_score: float = 0.0  # 0.0 = unrelated, 1.0 = direct participant

    # Current news that strengthens relevance
    news_mentions: List[str] = field(default_factory=list)  # Recent headlines
    news_recency_days: int = 0  # How recent is the news

    # Historical frequency hints
    historical_mentions: List[str] = field(default_factory=list)  # "Said 5 times in Super Bowl LVIII"

    # Storyline potential
    storyline_hooks: List[str] = field(default_factory=list)  # Things announcers might say

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class InformedContext:
    """Full context for INFORMED simulation mode.

    Combines event facts + term-specific relevance + current news.
    """

    event: EventContext
    term_relevance: Dict[str, TermRelevance] = field(default_factory=dict)  # term -> relevance

    # Current storylines (news that doesn't need term filtering)
    storylines: List[str] = field(default_factory=list)

    # Term-specific storylines (news about specific terms)
    term_storylines: Dict[str, List[str]] = field(default_factory=dict)  # term -> [storylines]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event": self.event.to_dict(),
            "term_relevance": {t: r.to_dict() for t, r in self.term_relevance.items()},
            "storylines": self.storylines,
            "term_storylines": self.term_storylines,
        }


# =============================================================================
# WEB UTILITIES
# =============================================================================


async def _duckduckgo_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Search DuckDuckGo for results."""
    try:
        from duckduckgo_search import DDGS

        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                })
        return results

    except ImportError:
        logger.warning("duckduckgo_search not installed")
        return []
    except Exception as e:
        logger.warning(f"DuckDuckGo search failed: {e}")
        return []


async def _fetch_wikipedia(title: str) -> Optional[Dict[str, Any]]:
    """Fetch Wikipedia page summary and extract."""
    try:
        import httpx

        encoded_title = quote_plus(title)
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded_title}"

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, follow_redirects=True)

            if resp.status_code == 200:
                data = resp.json()
                return {
                    "title": data.get("title", ""),
                    "description": data.get("description", ""),
                    "extract": data.get("extract", ""),
                    "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                }
            elif resp.status_code == 404:
                # Try search
                search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={encoded_title}&format=json"
                search_resp = await client.get(search_url)
                if search_resp.status_code == 200:
                    results = search_resp.json().get("query", {}).get("search", [])
                    if results:
                        return await _fetch_wikipedia(results[0].get("title", ""))
        return None

    except Exception as e:
        logger.warning(f"Wikipedia fetch failed for '{title}': {e}")
        return None


async def _fetch_wikipedia_categories(title: str) -> List[str]:
    """Fetch Wikipedia categories for a page."""
    try:
        import httpx

        encoded_title = quote_plus(title)
        url = f"https://en.wikipedia.org/w/api.php?action=query&titles={encoded_title}&prop=categories&cllimit=20&format=json"

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                data = resp.json()
                pages = data.get("query", {}).get("pages", {})
                for page in pages.values():
                    cats = page.get("categories", [])
                    return [c.get("title", "").replace("Category:", "") for c in cats]
        return []

    except Exception as e:
        logger.debug(f"Wikipedia categories failed: {e}")
        return []


# =============================================================================
# CONTEXT GATHERING FUNCTIONS
# =============================================================================


async def gather_event_context(
    event_ticker: str,
    event_title: str,
    domain: Optional[str] = None,
) -> EventContext:
    """Gather structured event context from Wikipedia.

    This is domain-agnostic fact gathering about the event itself.
    No mention terms involved - just event facts.

    Caches results for 4 hours to avoid redundant HTTP calls.
    """
    # Check cache first
    cached = _get_cached_context(event_ticker)
    if cached:
        logger.debug(f"Using cached EventContext for {event_ticker}")
        return cached

    if not domain:
        domain = _detect_domain(event_ticker, event_title)

    ctx = EventContext(
        event_ticker=event_ticker,
        event_title=event_title,
        domain=domain,
    )

    # Domain-specific gathering
    if domain == "sports":
        await _gather_sports_event_context(ctx, event_title)
    elif domain == "corporate":
        await _gather_corporate_event_context(ctx, event_title)
    elif domain == "politics":
        await _gather_politics_event_context(ctx, event_title)
    else:
        await _gather_generic_event_context(ctx, event_title)

    # Cache before returning
    _cache_context(event_ticker, ctx)
    return ctx


async def _gather_sports_event_context(ctx: EventContext, event_title: str) -> None:
    """Gather sports event context (teams, players, venue, announcers)."""

    # Extract teams from title
    teams = _extract_teams_from_title(event_title)
    ctx.participants = teams

    # Detect specific event type
    if "super bowl" in event_title.lower():
        ctx.event_type = "Super Bowl"

        # Fetch Super Bowl Wikipedia
        wiki = await _fetch_wikipedia("Super Bowl")
        if wiki:
            ctx.historical_notes.append(wiki.get("extract", "")[:300])
            ctx.wikipedia_urls.append(wiki.get("url", ""))

    # Fetch each team's Wikipedia
    for team in teams:
        wiki = await _fetch_wikipedia(f"{team} NFL")
        if wiki:
            ctx.participant_details[team] = wiki.get("extract", "")[:300]
            ctx.wikipedia_urls.append(wiki.get("url", ""))

    # Try to find key players
    for team in teams[:2]:
        # Search for team roster / key players
        results = await _duckduckgo_search(f"{team} quarterback roster 2025", max_results=2)
        for r in results:
            snippet = r.get("snippet", "")
            # Extract player names (simplified - could use NER)
            # Just grab the team name for now
            ctx.key_figures.append(f"{team} players")

    # Try to find announcers
    results = await _duckduckgo_search(f"{event_title} broadcast announcers commentators", max_results=3)
    for r in results:
        # Could extract names with NER, for now just note the search
        ctx.wikipedia_urls.append(r.get("url", ""))

    # Venue search
    results = await _duckduckgo_search(f"{event_title} stadium venue location", max_results=2)
    for r in results:
        snippet = r.get("snippet", "")
        if "stadium" in snippet.lower() or "arena" in snippet.lower():
            ctx.venue = snippet[:150]


async def _gather_corporate_event_context(ctx: EventContext, event_title: str) -> None:
    """Gather corporate event context (company, executives)."""

    company = _extract_company_from_title(event_title)
    if company:
        ctx.participants = [company]
        ctx.event_type = "Earnings Call"

        # Fetch company Wikipedia
        wiki = await _fetch_wikipedia(company)
        if wiki:
            ctx.participant_details[company] = wiki.get("extract", "")[:300]
            ctx.wikipedia_urls.append(wiki.get("url", ""))

        # Search for executives
        results = await _duckduckgo_search(f"{company} CEO CFO executives", max_results=3)
        for r in results:
            ctx.wikipedia_urls.append(r.get("url", ""))


async def _gather_politics_event_context(ctx: EventContext, event_title: str) -> None:
    """Gather political event context (speaker, venue)."""

    speaker = _extract_speaker_from_title(event_title)
    if speaker:
        ctx.participants = [speaker]
        ctx.event_type = "Political Speech"

        # Fetch speaker Wikipedia
        wiki = await _fetch_wikipedia(speaker)
        if wiki:
            ctx.participant_details[speaker] = wiki.get("extract", "")[:300]
            ctx.wikipedia_urls.append(wiki.get("url", ""))


async def _gather_generic_event_context(ctx: EventContext, event_title: str) -> None:
    """Gather generic event context."""

    wiki = await _fetch_wikipedia(event_title)
    if wiki:
        ctx.historical_notes.append(wiki.get("extract", "")[:300])
        ctx.wikipedia_urls.append(wiki.get("url", ""))


async def resolve_entity(term: str) -> EntityContext:
    """NameNet-style entity resolution for a term.

    Answers: What/who IS this term?
    """
    entity = EntityContext(term=term)

    # Fetch Wikipedia
    wiki = await _fetch_wikipedia(term)
    if wiki:
        entity.canonical_name = wiki.get("title", term)
        entity.description = wiki.get("description", "")
        entity.wikipedia_extract = wiki.get("extract", "")[:500]
        entity.wikipedia_url = wiki.get("url", "")

        # Infer entity type from description
        desc_lower = entity.description.lower()
        if any(w in desc_lower for w in ["singer", "actor", "player", "politician", "person"]):
            entity.entity_type = "person"
        elif any(w in desc_lower for w in ["company", "corporation", "organization"]):
            entity.entity_type = "organization"
        elif any(w in desc_lower for w in ["phrase", "expression", "idiom", "saying"]):
            entity.entity_type = "phrase"
        else:
            entity.entity_type = "concept"

        # Fetch categories
        categories = await _fetch_wikipedia_categories(entity.canonical_name)
        entity.categories = categories[:10]
    else:
        # No Wikipedia page - might be a phrase or uncommon term
        entity.entity_type = "phrase" if " " in term else "concept"
        entity.canonical_name = term

    return entity


async def compute_term_relevance(
    term: str,
    event: EventContext,
    entity: EntityContext,
) -> TermRelevance:
    """Compute how a term relates to a specific event.

    This is the KEY function - determines WHY this term might be mentioned.
    """
    relevance = TermRelevance(
        term=term,
        event_ticker=event.event_ticker,
    )

    # Check for direct participation
    term_lower = term.lower()
    for participant in event.participants:
        if term_lower in participant.lower() or participant.lower() in term_lower:
            relevance.relevance_type = "direct_participant"
            relevance.relevance_score = 1.0
            relevance.relevance_explanation = f"{term} is a direct participant in {event.event_title}"
            return relevance

    # Check for key figure match
    for figure in event.key_figures:
        if term_lower in figure.lower():
            relevance.relevance_type = "key_figure"
            relevance.relevance_score = 0.9
            relevance.relevance_explanation = f"{term} is a key figure in {event.event_title}"
            return relevance

    # Search for connection between term and event
    search_query = f'"{term}" "{event.event_title}"'
    results = await _duckduckgo_search(search_query, max_results=5)

    if results:
        # Found connection in search results
        snippets = [r.get("snippet", "") for r in results]
        combined = " ".join(snippets)

        # Look for relationship indicators
        relationship_found = False
        for snippet in snippets:
            snippet_lower = snippet.lower()

            # Dating/relationship
            if any(w in snippet_lower for w in ["dating", "girlfriend", "boyfriend", "married", "relationship"]):
                relevance.relevance_type = "related_person"
                relevance.relevance_score = 0.7
                relevance.relevance_explanation = snippet[:200]
                relationship_found = True
                break

            # Attending/present
            if any(w in snippet_lower for w in ["attending", "attended", "present at", "watching"]):
                relevance.relevance_type = "current_news"
                relevance.relevance_score = 0.6
                relevance.relevance_explanation = snippet[:200]
                relationship_found = True
                break

            # News/controversy
            if any(w in snippet_lower for w in ["controversy", "news", "headlines", "trending"]):
                relevance.relevance_type = "current_news"
                relevance.relevance_score = 0.5
                relevance.relevance_explanation = snippet[:200]
                relationship_found = True
                break

        if not relationship_found and combined:
            # Generic connection found
            relevance.relevance_type = "mentioned_together"
            relevance.relevance_score = 0.3
            relevance.relevance_explanation = snippets[0][:200] if snippets else ""

        # Store news mentions
        relevance.news_mentions = [r.get("snippet", "")[:150] for r in results[:3]]

    # If no connection found
    if not relevance.relevance_type:
        # Check if it's a common phrase
        if entity.entity_type == "phrase":
            relevance.relevance_type = "common_phrase"
            relevance.relevance_score = 0.2
            relevance.relevance_explanation = f"'{term}' is a common phrase that may be used naturally"
        else:
            relevance.relevance_type = "unrelated"
            relevance.relevance_score = 0.1
            relevance.relevance_explanation = f"No clear connection between '{term}' and {event.event_title}"

    # Build connection path
    if relevance.relevance_type == "related_person":
        # Try to extract the connection chain
        relevance.connection_path = [term, "→", "related to", "→", event.participants[0] if event.participants else event.event_title]

    return relevance


async def gather_informed_context(
    event_ticker: str,
    event_title: str,
    mention_terms: List[str],
    include_news: bool = True,
    event_context: Optional[EventContext] = None,
) -> InformedContext:
    """Gather full context for INFORMED simulation mode.

    Includes:
    - Event facts (from Wikipedia)
    - Entity resolution for each term
    - Term-event relevance analysis
    - Current news storylines

    Args:
        event_ticker: Kalshi event ticker
        event_title: Event title for context
        mention_terms: List of terms to analyze
        include_news: Whether to fetch current news
        event_context: Optional pre-fetched EventContext (avoids duplicate HTTP calls)
    """
    # Get event context (reuse if provided, otherwise fetch)
    event = event_context if event_context else await gather_event_context(event_ticker, event_title)

    # Resolve each term and compute relevance
    term_relevance = {}
    term_storylines = {}

    for term in mention_terms:
        # Resolve entity
        entity = await resolve_entity(term)

        # Compute relevance to this event
        relevance = await compute_term_relevance(term, event, entity)
        term_relevance[term] = relevance

        # Gather term-specific storylines
        if include_news and relevance.relevance_score > 0.3:
            results = await _duckduckgo_search(f"{term} {event_title} news 2026", max_results=3)
            term_storylines[term] = [r.get("snippet", "")[:200] for r in results]

    # Gather general event storylines
    storylines = []
    if include_news:
        results = await _duckduckgo_search(f"{event_title} storylines preview 2026", max_results=5)
        storylines = [r.get("snippet", "")[:200] for r in results]

    return InformedContext(
        event=event,
        term_relevance=term_relevance,
        storylines=storylines,
        term_storylines=term_storylines,
    )


async def gather_blind_context(
    event_ticker: str,
    event_title: str,
) -> EventContext:
    """Gather BLIND context for baseline simulation.

    Just event facts, no term-specific information.
    Used to establish baseline probability.
    """
    return await gather_event_context(event_ticker, event_title)


async def gather_mention_contexts(
    event_ticker: str,
    event_title: str,
    mention_terms: List[str],
) -> Dict[str, TermRelevance]:
    """Gather term-specific context for each mention term.

    Returns Dict mapping term -> TermRelevance explaining
    WHY and HOW each term relates to this specific event.

    This builds on gather_event_context() but adds term-specific
    entity resolution and relevance computation.

    Args:
        event_ticker: Kalshi event ticker
        event_title: Event title for context
        mention_terms: List of terms to analyze

    Returns:
        Dict[str, TermRelevance] mapping each term to its relevance analysis
    """
    # Get event context once (reused for all terms)
    event = await gather_event_context(event_ticker, event_title)

    results: Dict[str, TermRelevance] = {}
    for term in mention_terms:
        # Resolve entity (Wikipedia lookup)
        entity = await resolve_entity(term)
        # Compute relevance to this event
        relevance = await compute_term_relevance(term, event, entity)
        results[term] = relevance

    return results


# =============================================================================
# PROMPT GENERATION
# =============================================================================


def generate_blind_prompt(
    event: EventContext,
    template,  # DomainTemplate
) -> str:
    """Generate blind simulation prompt (baseline).

    No mention terms, no term-specific news.
    """
    return template.get_simulation_prompt(
        event_title=event.event_title,
        participants=event.participants,
        announcers=event.announcers,
        venue=event.venue,
        storylines=event.historical_notes,  # Only historical, no current news
    )


def generate_informed_prompt(
    informed: InformedContext,
    template,  # DomainTemplate
    terms_to_include: Optional[List[str]] = None,
) -> str:
    """Generate informed simulation prompt (with context).

    Includes term-relevant storylines that might prompt natural mention.
    DOES NOT explicitly ask for terms - just provides context that makes
    them more likely to arise naturally.
    """
    # Combine general and term-specific storylines
    all_storylines = list(informed.storylines)

    # Add high-relevance term storylines
    for term, storylines in informed.term_storylines.items():
        relevance = informed.term_relevance.get(term)
        if relevance and relevance.relevance_score >= 0.5:
            # Add term storylines for relevant terms
            for storyline in storylines[:2]:
                if storyline not in all_storylines:
                    all_storylines.append(storyline)

    # Add relevance explanations as storylines for highly relevant terms
    for term, relevance in informed.term_relevance.items():
        if relevance.relevance_score >= 0.6 and relevance.relevance_explanation:
            storyline = relevance.relevance_explanation
            if storyline not in all_storylines:
                all_storylines.append(storyline)

    return template.get_simulation_prompt(
        event_title=informed.event.event_title,
        participants=informed.event.participants,
        announcers=informed.event.announcers,
        venue=informed.event.venue,
        storylines=all_storylines[:10],  # Limit to avoid overwhelming
    )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _detect_domain(event_ticker: str, event_title: str) -> str:
    """Detect domain from event ticker and title."""
    ticker_upper = event_ticker.upper()
    title_upper = event_title.upper() if event_title else ""

    if any(s in ticker_upper or s in title_upper for s in [
        "NFL", "NBA", "MLB", "NHL", "SUPER BOWL", "WORLD SERIES",
        "PLAYOFFS", "CHAMPIONSHIP", "GAME", "MATCH"
    ]):
        return "sports"

    if any(c in title_upper for c in [
        "EARNINGS", "CALL", "INVESTOR", "QUARTERLY"
    ]):
        return "corporate"

    if any(p in title_upper for p in [
        "TRUMP", "BIDEN", "PRESIDENT", "CONGRESS", "SPEECH", "ADDRESS", "BRIEFING"
    ]):
        return "politics"

    if any(e in title_upper for e in [
        "OSCAR", "GRAMMY", "EMMY", "AWARD", "SHOW"
    ]):
        return "entertainment"

    return "generic"


def _extract_teams_from_title(title: str) -> List[str]:
    """Extract team names from event title."""
    patterns = [
        r"(.+?)\s+(?:vs\.?|versus|at|@)\s+(.+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, title, re.IGNORECASE)
        if match:
            teams = [match.group(1).strip(), match.group(2).strip()]
            teams = [t for t in teams if t.lower() not in ["super bowl", "championship", "game"]]
            return teams[:2]
    return []


def _extract_company_from_title(title: str) -> Optional[str]:
    """Extract company name from earnings call title."""
    patterns = [
        r"^(\w+(?:\s+\w+)?)\s+(?:Q[1-4]|quarterly|earnings)",
        r"^(\w+(?:\s+\w+)?)\s+call",
    ]
    for pattern in patterns:
        match = re.search(pattern, title, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    words = title.split()
    return words[0] if words else None


def _extract_speaker_from_title(title: str) -> Optional[str]:
    """Extract speaker name from political speech title."""
    common_politicians = ["trump", "biden", "harris", "pence", "obama", "clinton"]
    title_lower = title.lower()
    for pol in common_politicians:
        if pol in title_lower:
            return pol.title()
    match = re.search(r"President\s+(\w+)", title, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None
