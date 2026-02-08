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
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple
from urllib.parse import quote_plus

if TYPE_CHECKING:
    from .mentions_templates import SpeakerPersona

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

    # Dynamic speaker personas (built by build_persona_from_event)
    speakers: List["SpeakerPersona"] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        # Convert SpeakerPersona objects to dicts
        if self.speakers:
            result["speakers"] = [
                s.to_dict() if hasattr(s, "to_dict") else s for s in self.speakers
            ]
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EventContext":
        from .mentions_templates import SpeakerPersona

        # Handle speakers field specially - convert dicts to SpeakerPersona objects
        speakers_data = data.pop("speakers", [])
        instance = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

        # Restore speakers as SpeakerPersona objects
        if speakers_data:
            instance.speakers = [
                SpeakerPersona.from_dict(s) if isinstance(s, dict) else s
                for s in speakers_data
            ]
        return instance


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
# SPEAKER PERSONA BUILDING (LLM-based extraction)
# =============================================================================


@dataclass
class ExtractedSpeaker:
    """Structured speaker info extracted by LLM."""
    name: str
    title: Optional[str] = None
    role: Optional[str] = None
    style_notes: Optional[str] = None
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


async def _llm_extract_speakers_from_text(
    text: str,
    event_title: str,
    role_hint: str,
) -> List[ExtractedSpeaker]:
    """Use LLM to extract speaker names and info from text.

    Much more robust than regex - handles:
    - All name formats (hyphenated, apostrophes, all-caps)
    - Context-aware extraction
    - Multiple speakers in one text
    """
    try:
        from .mentions_models import get_extraction_llm

        llm = get_extraction_llm()  # Uses DEFAULT_EXTRACTION_MODEL from config

        prompt = f"""Extract person names from this text who might be the {role_hint} for "{event_title}".

TEXT:
{text[:1500]}

Return JSON array of speakers found:
[{{"name": "Full Name", "title": "their title if mentioned", "confidence": 0.0-1.0}}]

Rules:
1. Only extract REAL person names, not organization names or places
2. Confidence: 1.0 = explicitly identified as {role_hint}, 0.7 = likely based on context, 0.3 = mentioned but unclear role
3. Return empty array [] if no relevant person names found
4. Names can have hyphens, apostrophes, or be all caps - extract them correctly

Return ONLY the JSON array, no explanation."""

        from .llm_schemas import SpeakerExtraction

        structured_llm = llm.with_structured_output(SpeakerExtraction)
        result = await structured_llm.ainvoke(prompt)

        return [
            ExtractedSpeaker(
                name=s.name,
                title=s.title,
                confidence=s.confidence,
            )
            for s in result.speakers
            if s.name
        ]

    except Exception as e:
        logger.debug(f"LLM speaker extraction failed: {e}")
        return []


async def _llm_extract_speaker_from_wikipedia(
    speaker_name: str,
) -> Optional[Dict[str, Any]]:
    """Use LLM to extract structured info from Wikipedia about a speaker.

    Returns structured data about speaking style, background, and example phrases.
    """
    wiki = await _fetch_wikipedia(speaker_name)
    if not wiki:
        return None

    extract = wiki.get("extract", "")
    description = wiki.get("description", "")

    if not extract:
        return None

    try:
        from .mentions_models import get_extraction_llm

        llm = get_extraction_llm()  # Uses DEFAULT_EXTRACTION_MODEL from config

        prompt = f"""Analyze this Wikipedia info about {speaker_name} and extract speaking style characteristics.

DESCRIPTION: {description}

EXTRACT:
{extract[:1000]}

Return JSON:
{{
    "full_name": "canonical full name",
    "title": "current job title",
    "style_description": "2-3 sentence description of how they speak/communicate",
    "known_phrases": ["any signature phrases or catchphrases they're known for"],
    "background_relevant_to_speech": "brief background that affects how they speak"
}}

Return ONLY valid JSON."""

        from .llm_schemas import WikipediaSpeakerExtraction

        structured_llm = llm.with_structured_output(WikipediaSpeakerExtraction)
        result = await structured_llm.ainvoke(prompt)

        return result.model_dump()

    except Exception as e:
        logger.debug(f"LLM Wikipedia extraction failed: {e}")
        return {
            "full_name": speaker_name,
            "title": description,
            "style_description": extract[:200] if extract else "",
        }


async def _search_for_speaker_name(
    event_title: str,
    role: str,
    search_terms: List[str],
) -> Optional[ExtractedSpeaker]:
    """Search web for speaker name using LLM-based extraction.

    Args:
        event_title: Event title for context
        role: Role to search for (e.g., "play_by_play", "press_secretary")
        search_terms: Additional search terms (e.g., ["broadcaster", "announcer"])

    Returns:
        ExtractedSpeaker if found, None otherwise
    """
    combined_text = ""

    for term in search_terms[:2]:  # Limit searches
        query = f"{event_title} {term} 2026"
        results = await _duckduckgo_search(query, max_results=3)

        for r in results:
            snippet = r.get("snippet", "")
            title = r.get("title", "")
            combined_text += f"\n{title}\n{snippet}\n"

    if not combined_text.strip():
        return None

    # Use LLM to extract speaker
    speakers = await _llm_extract_speakers_from_text(combined_text, event_title, role)

    if speakers:
        # Return highest confidence speaker
        speakers.sort(key=lambda s: s.confidence, reverse=True)
        best = speakers[0]
        logger.info(f"[PERSONA] LLM extracted speaker for {role}: {best.name} (conf={best.confidence})")
        return best

    return None


async def _fetch_speaker_style(speaker_name: str) -> Tuple[str, List[str]]:
    """Fetch style description and example phrases for a speaker from Wikipedia.

    Uses LLM to extract structured info for better simulation prompts.

    Args:
        speaker_name: Name of the speaker to look up

    Returns:
        Tuple of (style_description, example_phrases)
    """
    # Use LLM-enhanced extraction
    wiki_data = await _llm_extract_speaker_from_wikipedia(speaker_name)

    if not wiki_data:
        return ("", [])

    style_parts = []

    # Build style description
    if wiki_data.get("title"):
        style_parts.append(wiki_data["title"])
    if wiki_data.get("style_description"):
        style_parts.append(wiki_data["style_description"])
    if wiki_data.get("background_relevant_to_speech"):
        style_parts.append(wiki_data["background_relevant_to_speech"])

    style_description = ". ".join(style_parts)[:400] if style_parts else ""

    # Get known phrases
    known_phrases = wiki_data.get("known_phrases", [])
    if isinstance(known_phrases, list):
        known_phrases = [p for p in known_phrases if p and len(p) > 5][:5]
    else:
        known_phrases = []

    return (style_description, known_phrases)


async def _fetch_style_examples(
    speaker_name: str,
    role: str,
    n_examples: int = 3,
) -> List[str]:
    """Fetch example lines for a speaker from web/YouTube.

    Sources (in order of preference):
    1. YouTube transcript snippets (best for actual speech patterns)
    2. News quotes from web search
    3. Wikipedia excerpts

    Args:
        speaker_name: Name of the speaker
        role: Speaker role (for search context)
        n_examples: Number of examples to fetch

    Returns:
        List of example lines/quotes
    """
    examples: List[str] = []

    # Build role-specific search queries
    role_search_terms = {
        "play_by_play": ["announcing", "call", "broadcast"],
        "color_analyst": ["analysis", "commentary"],
        "press_secretary": ["press briefing", "White House"],
        "ceo": ["earnings call", "investor presentation"],
        "host": ["hosting", "introducing"],
    }

    search_suffix = role_search_terms.get(role, [role.replace("_", " ")])

    # Search for quotes/transcripts
    for suffix in search_suffix[:2]:
        query = f'"{speaker_name}" {suffix} transcript quote'
        results = await _duckduckgo_search(query, max_results=3)

        for r in results:
            snippet = r.get("snippet", "")
            # Look for quoted text in snippet
            quotes = _extract_quotes(snippet)
            for quote in quotes:
                if len(quote) > 20 and len(quote) < 200 and quote not in examples:
                    examples.append(quote)
                    if len(examples) >= n_examples:
                        return examples

    return examples[:n_examples]


def _extract_quotes(text: str) -> List[str]:
    """Extract quoted text from a string."""
    # Match text in quotes
    pattern = r'"([^"]{20,200})"'
    matches = re.findall(pattern, text)
    return matches


async def build_persona_from_event(
    event_ticker: str,
    event_title: str,
    template,  # DomainTemplate
    mentions_data: Optional[Dict[str, Any]] = None,
) -> List["SpeakerPersona"]:
    """Build speaker personas dynamically from event info.

    Called ONCE during gather_event_context(), cached with EventContext.

    Progressive enhancement levels (logged):
    - Level 0: Generic persona from template.speaker_roles
    - Level 1: Named persona from lexeme_pack.speaker OR web search
    - Level 2: + style description from Wikipedia (LLM-extracted)
    - Level 3: + example lines from Wikipedia/web

    PRIMARY SOURCE: lexeme_pack.speaker from event rules (most reliable)
    FALLBACK: LLM-based web search extraction

    Args:
        event_ticker: Kalshi event ticker
        event_title: Event title for search context
        template: DomainTemplate with speaker_roles
        mentions_data: Optional dict with lexeme_pack containing speaker info

    Returns:
        List of SpeakerPersona objects for each role
    """
    from .mentions_templates import SpeakerPersona, GENERIC_PERSONAS

    personas: List[SpeakerPersona] = []

    # Get roles from template
    roles = list(template.speaker_roles.keys())[:4]  # Limit to top 4 roles

    # Check if we have speaker from lexeme_pack (parsed from event rules)
    lexeme_speaker = None
    if mentions_data:
        lexeme_pack = mentions_data.get("lexeme_pack", {})
        lexeme_speaker = lexeme_pack.get("speaker")
        if lexeme_speaker:
            logger.info(f"[PERSONA] Using speaker from lexeme_pack: {lexeme_speaker}")

    for role in roles:
        enhancement_level = 0

        # Start with generic persona (Level 0)
        generic = GENERIC_PERSONAS.get(role, (role.replace("_", " "), ""))
        title = generic[0]
        default_style = generic[1]

        persona = SpeakerPersona(
            role=role,
            title=title,
            style_description=default_style,
        )

        speaker_name: Optional[str] = None
        extracted_speaker: Optional[ExtractedSpeaker] = None

        # PRIMARY: Use lexeme_pack.speaker if available and role matches
        # For sports events, the lexeme_speaker is the MONITORED person (e.g., "Trump"),
        # NOT a broadcaster. Only use as speaker for politics/corporate domains.
        is_sports = template.domain == "sports"
        if lexeme_speaker and not is_sports and role in ("speaker", "press_secretary", "president", "ceo"):
            speaker_name = lexeme_speaker
            enhancement_level = 1
            logger.info(f"[PERSONA] {role}: Using lexeme_pack speaker '{speaker_name}' (Level 1)")

        # FALLBACK: LLM-based web search
        if not speaker_name:
            search_terms = _get_search_terms_for_role(role)
            extracted_speaker = await _search_for_speaker_name(event_title, role, search_terms)
            if extracted_speaker:
                speaker_name = extracted_speaker.name
                if extracted_speaker.title:
                    persona.title = extracted_speaker.title
                enhancement_level = 1
                logger.info(f"[PERSONA] {role}: Web search found '{speaker_name}' (Level 1)")

        if speaker_name:
            persona.name = speaker_name

            # Fetch style from Wikipedia using LLM extraction (Level 2)
            style_desc, wiki_examples = await _fetch_speaker_style(speaker_name)
            if style_desc:
                persona.style_description = style_desc
                enhancement_level = 2
                logger.debug(f"[PERSONA] {role}: Added Wikipedia style (Level 2)")

            # Add Wikipedia examples first
            if wiki_examples:
                persona.example_lines.extend(wiki_examples)
                enhancement_level = 3
                logger.debug(f"[PERSONA] {role}: Added {len(wiki_examples)} Wikipedia examples (Level 3)")

            # Fetch additional example lines from web
            if len(persona.example_lines) < 3:
                web_examples = await _fetch_style_examples(
                    speaker_name, role, n_examples=3 - len(persona.example_lines)
                )
                if web_examples:
                    persona.example_lines.extend(web_examples)
                    enhancement_level = max(enhancement_level, 3)

            persona.source_urls.append(f"https://en.wikipedia.org/wiki/{quote_plus(speaker_name)}")

        personas.append(persona)
        logger.info(
            f"[PERSONA] Built {role}: "
            f"name='{persona.name or 'GENERIC'}' "
            f"level={enhancement_level} "
            f"examples={len(persona.example_lines)}"
        )

    return personas


def _get_search_terms_for_role(role: str) -> List[str]:
    """Get search terms for finding speaker name by role."""
    search_terms = {
        "play_by_play": ["announcer", "play-by-play", "commentator", "broadcaster"],
        "color_analyst": ["color commentator", "analyst", "color analyst"],
        "sideline_reporter": ["sideline reporter", "field reporter"],
        "press_secretary": ["press secretary", "spokesperson"],
        "ceo": ["CEO", "chief executive"],
        "cfo": ["CFO", "chief financial officer"],
        "host": ["host", "anchor"],
        "main_host": ["host", "anchor", "primetime host"],
        "celebrity_cohost": ["co-host", "celebrity host"],
        "halftime_host": ["halftime host", "halftime show host"],
    }
    return search_terms.get(role, [role.replace("_", " ")])


# =============================================================================
# CONTEXT GATHERING FUNCTIONS
# =============================================================================


async def gather_event_context(
    event_ticker: str,
    event_title: str,
    domain: Optional[str] = None,
    build_personas: bool = True,
    mentions_data: Optional[Dict[str, Any]] = None,
) -> EventContext:
    """Gather structured event context from Wikipedia.

    This is domain-agnostic fact gathering about the event itself.
    No mention terms involved - just event facts.

    Caches results for 4 hours to avoid redundant HTTP calls.

    Args:
        event_ticker: Kalshi event ticker
        event_title: Event title for context
        domain: Optional domain override
        build_personas: Whether to build speaker personas (default True)
        mentions_data: Optional dict with lexeme_pack (for speaker extraction from rules)
    """
    # Check cache first
    cached = _get_cached_context(event_ticker)
    if cached:
        logger.debug(f"Using cached EventContext for {event_ticker}")
        # Restore SpeakerPersona objects if stored as dicts
        if cached.speakers and isinstance(cached.speakers[0], dict):
            from .mentions_templates import SpeakerPersona
            cached.speakers = [SpeakerPersona.from_dict(s) for s in cached.speakers]
        return cached

    if not domain:
        domain = _detect_domain(event_ticker, event_title)

    ctx = EventContext(
        event_ticker=event_ticker,
        event_title=event_title,
        domain=domain,
    )

    # For mentions events, extract the real event name from the question-format title
    # e.g. "Will Trump say 'tariff' during the Super Bowl LX?" → "Super Bowl LX"
    search_title = event_title
    if mentions_data:
        real_event_name, monitored_subject = _extract_real_event_from_mentions_title(event_title)
        if real_event_name != event_title:
            search_title = real_event_name
            logger.info(f"[CONTEXT] Extracted real event '{real_event_name}' from mentions title")
        if monitored_subject and monitored_subject not in ctx.key_figures:
            ctx.key_figures.append(monitored_subject)
            logger.info(f"[CONTEXT] Added monitored subject '{monitored_subject}' to key_figures")
        # Re-detect domain using the cleaned event name
        domain = _detect_domain(event_ticker, search_title)
        ctx.domain = domain

    # Domain-specific gathering (use cleaned search_title for better lookups)
    if domain == "sports":
        await _gather_sports_event_context(ctx, search_title)
    elif domain == "corporate":
        await _gather_corporate_event_context(ctx, search_title)
    elif domain == "politics":
        await _gather_politics_event_context(ctx, search_title)
    else:
        await _gather_generic_event_context(ctx, search_title)

    # Build speaker personas (called once, cached with EventContext)
    # Uses mentions_data.lexeme_pack.speaker if available (most reliable source)
    if build_personas:
        try:
            from .mentions_templates import detect_template
            template = detect_template(event_ticker, event_title)
            ctx.speakers = await build_persona_from_event(
                event_ticker, event_title, template, mentions_data=mentions_data
            )
            logger.info(f"[PERSONA] Built {len(ctx.speakers)} speaker personas for {event_ticker}")
        except Exception as e:
            logger.warning(f"Failed to build speaker personas: {e}")
            ctx.speakers = []

    # Cache before returning
    _cache_context(event_ticker, ctx)
    return ctx


async def _gather_sports_event_context(ctx: EventContext, event_title: str) -> None:
    """Gather sports event context (teams, players, venue, announcers)."""

    # Extract teams from title
    teams = _extract_teams_from_title(event_title)
    ctx.participants = teams

    # Detect specific event type
    title_lower = event_title.lower()
    if "super bowl" in title_lower:
        ctx.event_type = "Super Bowl"

        # Try specific Super Bowl edition first (e.g., "Super Bowl LX")
        # Extract Roman numeral or number from title
        sb_specific = None
        sb_match = re.search(r"super bowl\s+([IVXLCDM]+|\d+)", event_title, re.IGNORECASE)
        if sb_match:
            sb_specific = f"Super Bowl {sb_match.group(1)}"

        if sb_specific:
            wiki = await _fetch_wikipedia(sb_specific)
            if wiki and wiki.get("extract"):
                ctx.historical_notes.append(wiki.get("extract", "")[:500])
                ctx.wikipedia_urls.append(wiki.get("url", ""))
                # Extract venue, network, date from the specific SB article
                extract = wiki.get("extract", "")
                _parse_super_bowl_details(ctx, extract)
                logger.info(f"[CONTEXT] Found specific Super Bowl article: {sb_specific}")
            else:
                # Fallback to generic Super Bowl
                wiki = await _fetch_wikipedia("Super Bowl")
                if wiki:
                    ctx.historical_notes.append(wiki.get("extract", "")[:300])
                    ctx.wikipedia_urls.append(wiki.get("url", ""))
        else:
            wiki = await _fetch_wikipedia("Super Bowl")
            if wiki:
                ctx.historical_notes.append(wiki.get("extract", "")[:300])
                ctx.wikipedia_urls.append(wiki.get("url", ""))

        # Search for broadcast team specifically
        broadcast_query = f"{sb_specific or 'Super Bowl'} 2026 broadcast team announcers"
        broadcast_results = await _duckduckgo_search(broadcast_query, max_results=3)
        for r in broadcast_results:
            snippet = r.get("snippet", "")
            title_r = r.get("title", "")
            combined = f"{title_r} {snippet}"
            # Use LLM extraction for announcer names
            speakers = await _llm_extract_speakers_from_text(
                combined, event_title, "broadcast announcer"
            )
            for speaker in speakers:
                if speaker.name and speaker.name not in ctx.announcers:
                    ctx.announcers.append(speaker.name)

    # Fetch each team's Wikipedia
    for team in teams:
        wiki = await _fetch_wikipedia(f"{team} NFL")
        if wiki:
            ctx.participant_details[team] = wiki.get("extract", "")[:300]
            ctx.wikipedia_urls.append(wiki.get("url", ""))

    # Try to find key players
    for team in teams[:2]:
        results = await _duckduckgo_search(f"{team} quarterback roster 2025", max_results=2)
        for r in results:
            snippet = r.get("snippet", "")
            ctx.key_figures.append(f"{team} players")

    # Try to find announcers (only if not already found via Super Bowl broadcast search)
    if not ctx.announcers:
        results = await _duckduckgo_search(f"{event_title} broadcast announcers commentators 2026", max_results=3)
        for r in results:
            ctx.wikipedia_urls.append(r.get("url", ""))

    # Venue search (only if not already found)
    if not ctx.venue:
        results = await _duckduckgo_search(f"{event_title} stadium venue location 2026", max_results=2)
        for r in results:
            snippet = r.get("snippet", "")
            if "stadium" in snippet.lower() or "arena" in snippet.lower():
                ctx.venue = snippet[:150]


def _parse_super_bowl_details(ctx: EventContext, extract: str) -> None:
    """Parse venue, network, date, location from a Super Bowl Wikipedia extract."""
    # Venue
    for pat in [
        r"(?:held|played|take place) at (?:the )?([A-Z][A-Za-z\s]+(?:Stadium|Arena|Dome|Field|Center|Centre))",
        r"([A-Z][A-Za-z\s]+(?:Stadium|Arena|Dome|Field|Center|Centre))",
    ]:
        vm = re.search(pat, extract)
        if vm:
            ctx.venue = vm.group(1).strip()
            break

    # Location
    for pat in [
        r"in ([A-Z][A-Za-z\s,]+(?:Louisiana|Florida|Arizona|California|Nevada|Texas|Georgia))",
        r"in ([A-Z][A-Za-z]+(?:,\s*[A-Z][A-Za-z]+)?)",
    ]:
        lm = re.search(pat, extract)
        if lm:
            ctx.location = lm.group(1).strip().rstrip(".")
            break

    # Network
    for pat in [
        r"(?:broadcast|televised|aired|airing) (?:by|on) ([A-Z]{2,4})",
        r"(Fox|CBS|NBC|ABC|ESPN|FOX)\s+(?:will |to )?(?:broadcast|televise|air)",
    ]:
        nm = re.search(pat, extract, re.IGNORECASE)
        if nm:
            ctx.network = nm.group(1).strip().upper()
            break

    # Date
    for pat in [
        r"(?:on |scheduled for )([A-Z][a-z]+ \d{1,2}, \d{4})",
        r"(February \d{1,2}, \d{4})",
    ]:
        dm = re.search(pat, extract)
        if dm:
            ctx.date = dm.group(1)
            break


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
    event_understanding: Optional[Dict[str, Any]] = None,
) -> InformedContext:
    """Gather full context for INFORMED simulation mode.

    Includes:
    - Event facts (from Wikipedia)
    - Entity resolution for each term
    - Term-event relevance analysis
    - Current news storylines (from event understanding + web search)

    Args:
        event_ticker: Kalshi event ticker
        event_title: Event title for context
        mention_terms: List of terms to analyze
        include_news: Whether to fetch current news
        event_context: Optional pre-fetched EventContext (avoids duplicate HTTP calls)
        event_understanding: Optional event understanding dict with news_articles
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

    # Pull news from event understanding first (avoids extra API calls)
    if event_understanding:
        news_articles = event_understanding.get("news_articles", [])
        for article in news_articles[:3]:
            headline = article.get("title", "")
            snippet = (article.get("content", "") or article.get("snippet", ""))[:150]
            if headline:
                storyline = f"{headline}: {snippet}" if snippet else headline
                storylines.append(storyline)

    if include_news:
        results = await _duckduckgo_search(f"{event_title} storylines preview 2026", max_results=5)
        for r in results:
            s = r.get("snippet", "")[:200]
            if s and s not in storylines:
                storylines.append(s)

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
    compressed: bool = True,  # Default to compressed for better signal/token
) -> str:
    """Generate blind simulation prompt (baseline).

    Templates ALL discovered event context elegantly:
    - Event identity (title, venue, date, network)
    - Participants with Wikipedia context
    - Key figures with details
    - Speaker personas with style + examples
    - Historical notes as storylines

    Args:
        event: EventContext with event facts
        template: DomainTemplate for the event type
        compressed: If True, use compressed format (5-10x more coverage per token)
    """
    # Build speaker personas section if available
    special_context = None
    if event.speakers:
        persona_intros = []
        example_sections = []
        for speaker in event.speakers[:3]:  # Top 3 speakers
            if hasattr(speaker, "to_prompt_intro"):
                intro = speaker.to_prompt_intro()
                if intro:
                    persona_intros.append(f"- {intro}")
                examples = speaker.to_examples_section()
                if examples:
                    example_sections.append(examples)

        if persona_intros:
            special_context = "SPEAKER PERSONAS:\n" + "\n".join(persona_intros)
            if example_sections:
                special_context += "\n" + "\n".join(example_sections[:2])  # Limit examples

    # Choose prompt format - pass ALL rich context
    if compressed:
        return template.get_compressed_simulation_prompt(
            event_title=event.event_title,
            participants=event.participants,
            announcers=event.announcers,
            venue=event.venue,
            storylines=event.historical_notes,
            special_context=special_context,
            # Rich context from EventContext
            participant_details=event.participant_details,
            key_figures=event.key_figures,
            figure_details=event.figure_details,
            network=event.network,
            date=event.date,
        )
    else:
        return template.get_simulation_prompt(
            event_title=event.event_title,
            participants=event.participants,
            announcers=event.announcers,
            venue=event.venue,
            storylines=event.historical_notes,
            special_context=special_context,
        )


def generate_informed_prompt(
    informed: InformedContext,
    template,  # DomainTemplate
    terms_to_include: Optional[List[str]] = None,
    compressed: bool = True,  # Default to compressed for better signal/token
) -> str:
    """Generate informed simulation prompt (with context).

    Templates ALL discovered context elegantly:
    - Event identity + participants + key figures (from EventContext)
    - Speaker personas with style + examples
    - Term-relevant storylines that might prompt natural mention
    - DOES NOT explicitly ask for terms - just provides context

    Args:
        informed: InformedContext with event facts and term relevance
        template: DomainTemplate for the event type
        terms_to_include: Optional list of terms to focus on
        compressed: If True, use compressed format (5-10x more coverage per token)
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

    # Build speaker personas section if available
    special_context = None
    event = informed.event
    if event.speakers:
        persona_intros = []
        example_sections = []
        for speaker in event.speakers[:3]:  # Top 3 speakers
            if hasattr(speaker, "to_prompt_intro"):
                intro = speaker.to_prompt_intro()
                if intro:
                    persona_intros.append(f"- {intro}")
                examples = speaker.to_examples_section()
                if examples:
                    example_sections.append(examples)

        if persona_intros:
            special_context = "SPEAKER PERSONAS:\n" + "\n".join(persona_intros)
            if example_sections:
                special_context += "\n" + "\n".join(example_sections[:2])  # Limit examples

    # Choose prompt format - pass ALL rich context
    if compressed:
        return template.get_compressed_simulation_prompt(
            event_title=event.event_title,
            participants=event.participants,
            announcers=event.announcers,
            venue=event.venue,
            storylines=all_storylines[:10],
            special_context=special_context,
            # Rich context from EventContext
            participant_details=event.participant_details,
            key_figures=event.key_figures,
            figure_details=event.figure_details,
            network=event.network,
            date=event.date,
        )
    else:
        return template.get_simulation_prompt(
            event_title=event.event_title,
            participants=event.participants,
            announcers=event.announcers,
            venue=event.venue,
            storylines=all_storylines[:10],
            special_context=special_context,
        )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _detect_domain(event_ticker: str, event_title: str, category: str = "") -> str:
    """Detect domain from event ticker, title, and Kalshi category."""
    ticker_upper = event_ticker.upper()
    title_upper = event_title.upper() if event_title else ""
    category_lower = category.lower() if category else ""

    # Use Kalshi category as primary signal when available
    if category_lower in ("sports",):
        return "sports"
    if category_lower in ("elections", "politics"):
        return "politics"
    if category_lower in ("companies",):
        return "corporate"
    if category_lower in ("entertainment",):
        return "entertainment"
    if category_lower in ("crypto",):
        return "crypto"

    # Fallback: keyword detection from ticker and title
    if any(s in ticker_upper or s in title_upper for s in [
        "NFL", "NBA", "MLB", "NHL", "SUPER BOWL", "WORLD SERIES",
        "PLAYOFFS", "CHAMPIONSHIP", "CHAMPION", "FOOTBALL", "BASEBALL",
        "BASKETBALL", "HOCKEY", "SOCCER", "GAME", "MATCH",
    ]):
        return "sports"

    if any(c in title_upper for c in [
        "EARNINGS", "CALL", "INVESTOR", "QUARTERLY",
    ]):
        return "corporate"

    if any(p in title_upper for p in [
        "TRUMP", "BIDEN", "PRESIDENT", "CONGRESS", "SPEECH", "ADDRESS",
        "BRIEFING", "NOMINEE", "NOMINATION", "DEMOCRATIC", "REPUBLICAN",
        "ELECTION", "VOTE", "GOVERNOR", "SENATE", "HOUSE",
    ]):
        return "politics"

    if any(e in title_upper for e in [
        "OSCAR", "GRAMMY", "EMMY", "AWARD", "SHOW",
    ]):
        return "entertainment"

    return "generic"


def _extract_real_event_from_mentions_title(title: str) -> Tuple[str, Optional[str]]:
    """Extract the underlying event name from a question-format mentions title.

    Mentions titles are questions like:
    - "Will Trump say 'tariff' during the Super Bowl LX?"
    - "How many times will the announcer say 'dynasty' at the Super Bowl LX?"

    Returns:
        Tuple of (real_event_name, monitored_subject)
        e.g. ("Super Bowl LX", "Trump")
    """
    monitored_subject = None

    # Extract monitored subject (the person being watched)
    subject_match = re.match(
        r"(?:Will|How many times will|How often will)\s+(.+?)\s+(?:say|mention|use)\b",
        title, re.IGNORECASE,
    )
    if subject_match:
        raw_subject = subject_match.group(1).strip()
        # Clean "the announcer" → None (not a specific person)
        if raw_subject.lower() not in ("the announcer", "the commentator", "the host", "an announcer"):
            monitored_subject = raw_subject

    # Extract real event name from "during/at/in the {EVENT}" patterns
    event_patterns = [
        r"(?:during|at|in)\s+(?:the\s+)?(.+?)\s*\??$",
        r"(?:during|at|in)\s+(.+?)\s*\??$",
    ]
    for pattern in event_patterns:
        event_match = re.search(pattern, title, re.IGNORECASE)
        if event_match:
            event_name = event_match.group(1).strip().rstrip("?")
            # Validate: the extracted name should look like an event, not residual text
            if len(event_name) > 3:
                return (event_name, monitored_subject)

    # Fallback: return original title
    return (title, monitored_subject)


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
