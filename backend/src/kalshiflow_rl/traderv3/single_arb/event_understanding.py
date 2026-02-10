"""
EventUnderstanding - Universal structured context for any event.

Three layers + extensible plugins:
1. Identity (from Kalshi API, always available)
2. Timeline + Key Actors (structured, machine-readable)
3. LLM Synthesis (market-focused trading insights)

All domain-detection, Wikipedia, and entity-extraction helpers are self-contained.
Cached to disk with 4-hour TTL.
"""

import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import quote_plus

from pydantic import BaseModel, Field

logger = logging.getLogger("kalshiflow_rl.traderv3.single_arb.event_understanding")

_UNDERSTANDING_CACHE_TTL_DEFAULT = 4 * 60 * 60  # 4 hours minimum
_UNDERSTANDING_CACHE_TTL_FALLBACK = 24 * 60 * 60  # 24 hours when no time info


# =============================================================================
# PYDANTIC MODELS (for LLM structured output)
# =============================================================================


class TimelineSegment(BaseModel):
    name: str = ""
    start_offset_min: int = 0
    duration_min: int = 0
    description: str = ""


class EventUnderstandingExtraction(BaseModel):
    trading_summary: str = ""
    key_factors: List[str] = Field(default_factory=list)
    trading_considerations: List[str] = Field(default_factory=list)
    timeline: List[TimelineSegment] = Field(default_factory=list)


# =============================================================================
# DOMAIN / ENTITY HELPERS
# =============================================================================


def _detect_domain(event_ticker: str, event_title: str, category: str = "") -> str:
    """Detect event domain from ticker, title, and category."""
    ticker_upper = event_ticker.upper()
    title_upper = event_title.upper() if event_title else ""
    category_lower = category.lower() if category else ""
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
    if any(
        s in ticker_upper or s in title_upper
        for s in [
            "NFL", "NBA", "MLB", "NHL", "SUPER BOWL", "WORLD SERIES",
            "PLAYOFFS", "CHAMPIONSHIP", "CHAMPION", "FOOTBALL", "BASEBALL",
            "BASKETBALL", "HOCKEY", "SOCCER", "GAME", "MATCH",
        ]
    ):
        return "sports"
    if any(c in title_upper for c in ["EARNINGS", "CALL", "INVESTOR", "QUARTERLY"]):
        return "corporate"
    if any(
        p in title_upper
        for p in [
            "TRUMP", "BIDEN", "PRESIDENT", "CONGRESS", "SPEECH", "ADDRESS",
            "BRIEFING", "NOMINEE", "NOMINATION", "DEMOCRATIC", "REPUBLICAN",
            "ELECTION", "VOTE", "GOVERNOR", "SENATE", "HOUSE",
        ]
    ):
        return "politics"
    if any(e in title_upper for e in ["OSCAR", "GRAMMY", "EMMY", "AWARD", "SHOW"]):
        return "entertainment"
    return "generic"


async def _fetch_wikipedia(title: str) -> Optional[Dict[str, Any]]:
    """Fetch Wikipedia summary via REST API, with search fallback on 404."""
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
                search_url = (
                    f"https://en.wikipedia.org/w/api.php"
                    f"?action=query&list=search&srsearch={encoded_title}&format=json"
                )
                search_resp = await client.get(search_url)
                if search_resp.status_code == 200:
                    results = search_resp.json().get("query", {}).get("search", [])
                    if results:
                        return await _fetch_wikipedia(results[0].get("title", ""))
        return None
    except Exception as e:
        logger.warning(f"Wikipedia fetch failed for '{title}': {e}")
        return None


def _extract_teams_from_title(title: str) -> List[str]:
    """Extract team names from 'Team A vs Team B' style titles."""
    patterns = [r"(.+?)\s+(?:vs\.?|versus|at|@)\s+(.+)"]
    for pattern in patterns:
        match = re.search(pattern, title, re.IGNORECASE)
        if match:
            teams = [match.group(1).strip(), match.group(2).strip()]
            teams = [t for t in teams if t.lower() not in ["super bowl", "championship", "game"]]
            return teams[:2]
    return []


def _extract_company_from_title(title: str) -> Optional[str]:
    """Extract company name from earnings/call style titles."""
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
    """Extract speaker name from political event titles."""
    common_politicians = ["trump", "biden", "harris", "pence", "obama", "clinton"]
    title_lower = title.lower()
    for pol in common_politicians:
        if pol in title_lower:
            return pol.title()
    match = re.search(r"President\s+(\w+)", title, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


# =============================================================================
# DATA MODEL
# =============================================================================


@dataclass
class EventUnderstanding:
    """Universal structured understanding of an event.

    Designed so any agent can reason over it: what's happening, when,
    who's involved, how it settles, and what to watch for.
    """

    # === IDENTITY (from Kalshi API, always available) ===
    event_ticker: str = ""
    title: str = ""
    category: str = ""
    domain: str = ""  # "sports", "politics", "corporate", "entertainment", "crypto", "weather"
    series_ticker: str = ""
    mutually_exclusive: bool = False
    market_count: int = 0

    # Settlement (parsed from rules_primary)
    settlement_summary: str = ""
    resolution_source: str = ""

    # === TIMELINE (structured, machine-readable) ===
    start_time: Optional[str] = None  # ISO 8601
    end_time: Optional[str] = None  # ISO 8601
    close_time: Optional[str] = None  # ISO 8601 market close
    time_to_close_hours: Optional[float] = None
    status: str = "upcoming"  # "upcoming", "live", "settled"

    timeline: List[Dict] = field(default_factory=list)
    # [{name, start_offset_min, duration_min, description}, ...]

    # === KEY ACTORS (structured) ===
    participants: List[Dict] = field(default_factory=list)
    # [{name, role, summary, wikipedia_url}, ...]

    key_figures: List[Dict] = field(default_factory=list)
    # [{name, role, relevance, summary}, ...]

    # === CONTEXT (from Wikipedia) ===
    event_summary: str = ""
    venue: str = ""
    location: str = ""
    network: str = ""
    date: str = ""
    historical_notes: List[str] = field(default_factory=list)
    wikipedia_urls: List[str] = field(default_factory=list)

    # === LLM SYNTHESIS (market-focused) ===
    trading_summary: str = ""
    key_factors: List[str] = field(default_factory=list)
    trading_considerations: List[str] = field(default_factory=list)

    # === NEWS CONTEXT (from web search) ===
    news_articles: List[Dict] = field(default_factory=list)
    # [{title, url, content, published_date, score, source}]
    news_summary: str = ""
    news_fetched_at: float = 0.0

    # === EXTENSIONS (domain-specific plugins) ===
    extensions: Dict[str, Any] = field(default_factory=dict)

    # === META ===
    gathered_at: float = 0.0
    cache_ttl_seconds: float = 0.0  # Per-instance TTL, set at build time
    version: int = 1

    @property
    def stale(self) -> bool:
        if self.gathered_at == 0:
            return True
        ttl = self.cache_ttl_seconds if self.cache_ttl_seconds > 0 else _UNDERSTANDING_CACHE_TTL_DEFAULT
        return (time.time() - self.gathered_at) > ttl

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["stale"] = self.stale
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EventUnderstanding":
        data.pop("stale", None)
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})


# =============================================================================
# EXTENSION INTERFACE
# =============================================================================


class UnderstandingExtension:
    """Base class for domain-specific understanding plugins."""

    key: str = "base"  # Key in extensions dict

    async def enrich(
        self,
        understanding: EventUnderstanding,
        event_meta: Any,
    ) -> None:
        """Enrich understanding with domain-specific data.

        Args:
            understanding: The EventUnderstanding being built
            event_meta: The EventMeta from the index
        """
        pass


# =============================================================================
# BUILDER
# =============================================================================


class UnderstandingBuilder:
    """Pipeline: Kalshi API identity -> Wikipedia context -> extensions -> LLM synthesis -> cache."""

    def __init__(self, cache_dir: Optional[str] = None, search_service=None):
        self._cache_dir = cache_dir
        self._search_service = search_service  # TavilySearchService (optional)
        self._extensions: List[UnderstandingExtension] = []
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def register_extension(self, ext: UnderstandingExtension) -> None:
        self._extensions.append(ext)

    async def build(
        self,
        event_meta: Any,
        force_refresh: bool = False,
    ) -> EventUnderstanding:
        """Build understanding for an event.

        Checks cache first unless force_refresh=True.
        """
        event_ticker = event_meta.event_ticker

        # Check cache
        if not force_refresh:
            cached = self._load_cache(event_ticker)
            if cached:
                logger.debug(f"[UNDERSTANDING] Cache hit for {event_ticker}")
                # Still run extensions to keep mentions data fresh
                for ext in self._extensions:
                    try:
                        await ext.enrich(cached, event_meta)
                    except Exception as e:
                        logger.debug(f"[UNDERSTANDING] Extension {ext.key} failed: {e}")
                return cached

        logger.info(f"[UNDERSTANDING] Building understanding for {event_ticker}")

        # Step 1: Identity from Kalshi API
        u = self._build_identity(event_meta)

        # Step 2: Wikipedia context
        try:
            await self._enrich_wikipedia(u, event_meta)
        except Exception as e:
            logger.warning(f"[UNDERSTANDING] Wikipedia enrichment failed: {e}")

        # Step 2.5: Web news enrichment (Tavily → DDG fallback)
        if self._search_service:
            try:
                await self._enrich_news(u, event_meta)
            except Exception as e:
                logger.warning(f"[UNDERSTANDING] News enrichment failed: {e}")

        # Step 3: Extensions
        for ext in self._extensions:
            try:
                await ext.enrich(u, event_meta)
            except Exception as e:
                logger.warning(f"[UNDERSTANDING] Extension {ext.key} failed: {e}")

        # Step 4: LLM synthesis
        await self._synthesize_trading_context(u)

        # Step 5: Compute event-aware cache TTL
        u.cache_ttl_seconds = self._compute_cache_ttl(u)

        # Step 6: Cache
        u.gathered_at = time.time()
        self._save_cache(event_ticker, u)

        logger.info(
            f"[UNDERSTANDING] Built for {event_ticker}: "
            f"participants={len(u.participants)} "
            f"factors={len(u.key_factors)} "
            f"extensions={list(u.extensions.keys())}"
        )
        return u

    @staticmethod
    def _compute_cache_ttl(u: EventUnderstanding) -> float:
        """Compute event-duration-aware cache TTL.

        If time_to_close is known: TTL = max(80% of remaining time, 4 hours).
        This means short events (~4h) cache for ~4h, multi-day events cache for days.
        Fallback: 24 hours if no time info available.
        """
        if u.time_to_close_hours is not None and u.time_to_close_hours > 0:
            remaining_seconds = u.time_to_close_hours * 3600
            duration_ttl = remaining_seconds * 0.8
            return max(duration_ttl, _UNDERSTANDING_CACHE_TTL_DEFAULT)
        return _UNDERSTANDING_CACHE_TTL_FALLBACK

    def _build_identity(self, event_meta: Any) -> EventUnderstanding:
        """Extract identity fields from Kalshi API data."""
        raw = getattr(event_meta, "raw", {})
        markets = getattr(event_meta, "markets", {})

        # Get close_time from first market
        close_time = None
        for m in markets.values():
            ct = getattr(m, "close_time", None)
            if ct:
                close_time = ct
                break

        # Compute time to close
        time_to_close = None
        if close_time:
            try:
                from datetime import datetime, timezone
                ct_dt = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
                delta = (ct_dt - datetime.now(timezone.utc)).total_seconds()
                time_to_close = max(0, delta / 3600)
            except Exception:
                pass

        # Settlement info from first market's rules
        settlement_summary = ""
        resolution_source = ""
        for m in markets.values():
            m_raw = getattr(m, "raw", {})
            rules = m_raw.get("rules_primary", "")
            if rules:
                # First sentence as summary
                settlement_summary = rules.split(".")[0].strip()[:200]
                resolution_source = m_raw.get("settlement_source_url", "")
                break

        # Detect status
        status = "upcoming"
        if time_to_close is not None:
            if time_to_close <= 0:
                status = "settled"
            elif time_to_close < 1:
                status = "live"

        category = getattr(event_meta, "category", "")
        domain = _detect_domain(
            event_meta.event_ticker,
            event_meta.title,
            category=category,
        )

        return EventUnderstanding(
            event_ticker=event_meta.event_ticker,
            title=event_meta.title,
            category=getattr(event_meta, "category", ""),
            domain=domain,
            series_ticker=getattr(event_meta, "series_ticker", ""),
            mutually_exclusive=getattr(event_meta, "mutually_exclusive", False),
            market_count=len(markets),
            settlement_summary=settlement_summary,
            resolution_source=resolution_source,
            close_time=close_time,
            time_to_close_hours=round(time_to_close, 2) if time_to_close is not None else None,
            status=status,
        )

    async def _enrich_wikipedia(self, u: EventUnderstanding, event_meta: Any) -> None:
        """Enrich with Wikipedia data using domain/entity helpers."""
        domain = u.domain
        title = u.title

        if domain == "sports":
            await self._enrich_sports(u, title)
        elif domain == "corporate":
            await self._enrich_corporate(u, title)
        elif domain == "politics":
            await self._enrich_politics(u, title)
        else:
            await self._enrich_generic(u, title)

        # Generic fallback: if domain-specific enrichment yielded nothing useful,
        # try Wikipedia with cleaned keywords from the title
        if not u.participants and not u.event_summary:
            keywords = self._extract_keywords_from_title(title)
            if len(keywords) > 3:
                wiki = await _fetch_wikipedia(keywords)
                if wiki:
                    u.event_summary = wiki.get("extract", "")[:300]
                    u.wikipedia_urls.append(wiki.get("url", ""))

    @staticmethod
    def _extract_keywords_from_title(title: str) -> str:
        """Extract meaningful keywords from question-format event titles.

        Strips common question prefixes and returns cleaned keywords for Wikipedia search.
        """
        cleaned = title
        # Strip question prefixes (most specific first)
        question_patterns = [
            r"^(?:What will|Will|How many times will|How often will)\s+.+?\s+(?:say|mention|use|announce|state)\s+(?:during|at|in)\s+(?:the\s+)?",
            r"^(?:What will|Will|How many|How often)\s+.+?\s+(?:say|mention|use)\s+",
            r"^(?:Who will (?:be|win)|What will be|Which)\s+(?:the\s+)?",
            r"^(?:What will|Will|Who will)\s+",
        ]
        for pat in question_patterns:
            m = re.match(pat, cleaned, re.IGNORECASE)
            if m:
                cleaned = cleaned[m.end():]
                break
        # Strip trailing question mark, year suffixes, and common prefixes
        cleaned = re.sub(r"\?$", "", cleaned).strip()
        cleaned = re.sub(r"\s+in\s+\d{4}$", "", cleaned).strip()
        cleaned = re.sub(r"^\s*the\s+", "", cleaned, flags=re.IGNORECASE).strip()
        return cleaned if len(cleaned) > 3 else title

    async def _enrich_sports(self, u: EventUnderstanding, title: str) -> None:
        title_lower = title.lower()

        teams = _extract_teams_from_title(title)
        for team in teams:
            wiki = await _fetch_wikipedia(f"{team} NFL")
            participant = {
                "name": team,
                "role": "team",
                "summary": "",
                "wikipedia_url": "",
            }
            if wiki:
                participant["summary"] = wiki.get("extract", "")[:200]
                participant["wikipedia_url"] = wiki.get("url", "")
                u.wikipedia_urls.append(wiki.get("url", ""))
            u.participants.append(participant)

        # Event-level wiki: recognize Super Bowl synonyms
        super_bowl_synonyms = [
            "super bowl",
            "pro football champion",
            "football champion",
        ]
        is_super_bowl = any(syn in title_lower for syn in super_bowl_synonyms)

        if is_super_bowl:
            # Try to find the specific Super Bowl edition from the event ticker or title
            # e.g., "SB26" in ticker, or detect year
            specific_sb = await self._find_specific_super_bowl(u, title)

            if not specific_sb:
                # Fallback to generic Super Bowl
                wiki = await _fetch_wikipedia("Super Bowl")
                if wiki:
                    u.event_summary = wiki.get("extract", "")[:300]
                    u.historical_notes.append(wiki.get("extract", "")[:300])
                    u.wikipedia_urls.append(wiki.get("url", ""))

            # Enrich with broadcast info for mentions markets
            await self._enrich_broadcast_info(u, title)

        # Fallback: if no teams and no event summary, try keyword extraction
        if not u.participants and not u.event_summary:
            keywords = self._extract_keywords_from_title(title)
            if len(keywords) > 3:
                wiki = await _fetch_wikipedia(keywords)
                if wiki:
                    u.event_summary = wiki.get("extract", "")[:300]
                    u.wikipedia_urls.append(wiki.get("url", ""))

    async def _find_specific_super_bowl(self, u: EventUnderstanding, title: str) -> bool:
        """Try to find the specific Super Bowl edition and populate detailed context."""
        # Extract Super Bowl number from ticker or title
        # Look for patterns like "SB26", "Super Bowl LIX", "Super Bowl 59"
        ticker = u.event_ticker
        sb_number = None

        # From ticker: "KXNFLMENTION-SB26" -> try "Super Bowl LX" (26th century = LX)
        sb_match = re.search(r"SB(\d+)", ticker)
        if sb_match:
            # Map year suffix to actual Super Bowl
            year_suffix = int(sb_match.group(1))
            # SB26 = 2026 Super Bowl = Super Bowl LX (60th)
            if year_suffix >= 20:
                sb_number_roman = self._to_roman(year_suffix + 34)  # 2026 -> SB 60 (LX)
                sb_search = f"Super Bowl {sb_number_roman}"
            else:
                sb_search = f"Super Bowl {year_suffix}"
        else:
            # Try to extract from title
            roman_match = re.search(r"Super Bowl ([IVXLCDM]+)", title, re.IGNORECASE)
            if roman_match:
                sb_search = f"Super Bowl {roman_match.group(1)}"
            else:
                return False

        wiki = await _fetch_wikipedia(sb_search)
        if wiki and wiki.get("extract"):
            extract = wiki["extract"]
            u.event_summary = extract[:500]
            u.wikipedia_urls.append(wiki.get("url", ""))

            # Try to extract structured info from the extract
            # Look for venue, date, network patterns
            extract_lower = extract.lower()

            # Venue extraction
            venue_patterns = [
                r"(?:held|played|take place) at (?:the )?([A-Z][A-Za-z\s]+(?:Stadium|Arena|Dome|Field|Center|Centre))",
                r"([A-Z][A-Za-z\s]+(?:Stadium|Arena|Dome|Field|Center|Centre))",
            ]
            for pat in venue_patterns:
                vm = re.search(pat, extract)
                if vm:
                    u.venue = vm.group(1).strip()
                    break

            # Location extraction
            loc_patterns = [
                r"in ([A-Z][A-Za-z\s,]+(?:Louisiana|Florida|Arizona|California|Nevada|Texas|Georgia))",
                r"in ([A-Z][A-Za-z]+(?:,\s*[A-Z][A-Za-z]+)?)",
            ]
            for pat in loc_patterns:
                lm = re.search(pat, extract)
                if lm:
                    u.location = lm.group(1).strip().rstrip(".")
                    break

            # Network extraction
            network_patterns = [
                r"(?:broadcast|televised|aired|airing) (?:by|on) ([A-Z]{2,4})",
                r"(Fox|CBS|NBC|ABC|ESPN|FOX)\s+(?:will |to )?(?:broadcast|televise|air)",
            ]
            for pat in network_patterns:
                nm = re.search(pat, extract, re.IGNORECASE)
                if nm:
                    u.network = nm.group(1).strip().upper()
                    break

            # Date extraction
            date_patterns = [
                r"(?:on |scheduled for )([A-Z][a-z]+ \d{1,2}, \d{4})",
                r"(February \d{1,2}, \d{4})",
                r"(\d{4} Super Bowl)",
            ]
            for pat in date_patterns:
                dm = re.search(pat, extract)
                if dm:
                    u.date = dm.group(1)
                    break

            # Add historical context
            u.historical_notes.append(extract[:300])

            return True

        return False

    @staticmethod
    def _to_roman(num: int) -> str:
        """Convert integer to Roman numeral."""
        val = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
        syms = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]
        result = ""
        for i, v in enumerate(val):
            while num >= v:
                result += syms[i]
                num -= v
        return result

    async def _enrich_broadcast_info(self, u: EventUnderstanding, title: str) -> None:
        """Enrich with broadcast-specific info (announcers, commentators).

        For mentions markets, knowing WHO is speaking is critical for simulation quality.
        """
        # Try to find broadcast team info
        # Common Super Bowl broadcast searches
        broadcast_searches = []
        if u.network:
            broadcast_searches.append(f"{u.network} Super Bowl broadcast")
        if u.date:
            year = re.search(r"\d{4}", u.date)
            if year:
                broadcast_searches.append(f"Super Bowl {year.group()} broadcast team")

        for search_term in broadcast_searches[:2]:
            wiki = await _fetch_wikipedia(search_term)
            if wiki and wiki.get("extract"):
                # Look for announcer names in the extract
                extract = wiki["extract"]

                # Common NFL announcer patterns
                announcer_patterns = [
                    r"(?:announced by|commentary by|called by|commentators?)\s+([A-Z][a-z]+ [A-Z][a-z]+(?:\s+and\s+[A-Z][a-z]+ [A-Z][a-z]+)?)",
                ]
                for pat in announcer_patterns:
                    am = re.search(pat, extract, re.IGNORECASE)
                    if am:
                        names = am.group(1).split(" and ")
                        for name in names:
                            name = name.strip()
                            if name and not any(p["name"] == name for p in u.participants):
                                u.participants.append({
                                    "name": name,
                                    "role": "announcer",
                                    "summary": f"{u.network} broadcast commentator",
                                    "wikipedia_url": "",
                                })
                break  # Stop after first successful search

    async def _enrich_corporate(self, u: EventUnderstanding, title: str) -> None:
        company = _extract_company_from_title(title)
        if company:
            wiki = await _fetch_wikipedia(company)
            participant = {
                "name": company,
                "role": "company",
                "summary": "",
                "wikipedia_url": "",
            }
            if wiki:
                participant["summary"] = wiki.get("extract", "")[:200]
                participant["wikipedia_url"] = wiki.get("url", "")
                u.wikipedia_urls.append(wiki.get("url", ""))
            u.participants.append(participant)

    async def _enrich_politics(self, u: EventUnderstanding, title: str) -> None:
        speaker = _extract_speaker_from_title(title)
        if speaker:
            wiki = await _fetch_wikipedia(speaker)
            participant = {
                "name": speaker,
                "role": "speaker",
                "summary": "",
                "wikipedia_url": "",
            }
            if wiki:
                participant["summary"] = wiki.get("extract", "")[:200]
                participant["wikipedia_url"] = wiki.get("url", "")
                u.wikipedia_urls.append(wiki.get("url", ""))
            u.participants.append(participant)

        # Event-level enrichment for election/nomination events
        if not u.event_summary:
            keywords = self._extract_keywords_from_title(title)
            if len(keywords) > 3:
                wiki = await _fetch_wikipedia(keywords)
                if wiki:
                    u.event_summary = wiki.get("extract", "")[:300]
                    u.wikipedia_urls.append(wiki.get("url", ""))

    async def _enrich_generic(self, u: EventUnderstanding, title: str) -> None:
        wiki = await _fetch_wikipedia(title)
        if wiki:
            u.event_summary = wiki.get("extract", "")[:300]
            u.historical_notes.append(wiki.get("extract", "")[:300])
            u.wikipedia_urls.append(wiki.get("url", ""))

    async def _enrich_news(self, u: EventUnderstanding, event_meta) -> None:
        """Fetch recent news articles about this event via Tavily (or DDG fallback)."""
        query = self._build_news_query(u)
        results = await self._search_service.search_news(
            query=query,
            event_ticker=u.event_ticker,
        )
        u.news_articles = results
        u.news_fetched_at = time.time()

        if results:
            logger.info(
                f"[UNDERSTANDING] News for {u.event_ticker}: "
                f"{len(results)} articles (source={results[0].get('source', '?')})"
            )

    def _build_news_query(self, u: EventUnderstanding) -> str:
        """Generate a search query from event context for news retrieval."""
        keywords = self._extract_keywords_from_title(u.title)

        # Add domain-specific suffixes for better results
        if u.domain == "sports":
            return f"{keywords} latest news predictions odds"
        elif u.domain == "politics":
            return f"{keywords} latest developments news"
        elif u.domain == "corporate":
            return f"{keywords} earnings news analysis"
        elif u.domain == "entertainment":
            return f"{keywords} latest news"
        return f"{keywords} latest news"

    def _generate_fallback_summary(self, u: EventUnderstanding) -> None:
        """Generate a useful trading_summary without LLM from available data."""
        parts = []
        if u.title:
            parts.append(u.title.rstrip("?").strip())
        if u.domain and u.domain != "generic":
            parts.append(f"({u.domain})")
        if u.mutually_exclusive:
            parts.append("- mutually exclusive outcomes, sum should be ~100%.")
        else:
            parts.append("- independent outcomes.")
        if u.market_count:
            parts.append(f"{u.market_count} markets.")
        if u.time_to_close_hours is not None:
            if u.time_to_close_hours > 48:
                parts.append(f"Closes in {u.time_to_close_hours / 24:.0f} days.")
            elif u.time_to_close_hours > 1:
                parts.append(f"Closes in {u.time_to_close_hours:.0f} hours.")
            else:
                parts.append(f"Closes in {u.time_to_close_hours * 60:.0f} minutes.")

        u.trading_summary = " ".join(parts)

        # Generate basic key_factors from available data
        factors = []
        if u.event_summary:
            factors.append(u.event_summary[:100])
        if u.mutually_exclusive:
            factors.append("Probabilities must sum to ~100% (arb opportunity if they don't)")
        if u.settlement_summary:
            factors.append(f"Settlement: {u.settlement_summary[:80]}")
        if u.participants:
            names = [p["name"] for p in u.participants[:3]]
            factors.append(f"Key participants: {', '.join(names)}")
        u.key_factors = factors[:5]

    async def _synthesize_trading_context(self, u: EventUnderstanding) -> None:
        """Use LLM to produce trading_summary, key_factors, trading_considerations.

        Invests more in initial quality (1200 tokens) since results are cached for
        the event's lifetime. Retries once on failure before falling back to heuristic.
        """
        for attempt in range(2):
            try:
                return await self._synthesize_trading_context_inner(u)
            except Exception as e:
                if attempt == 0:
                    logger.warning(f"[UNDERSTANDING] LLM synthesis attempt 1 failed, retrying: {e}")
                    continue
                logger.warning(f"[UNDERSTANDING] LLM synthesis failed after 2 attempts: {e}")
                self._generate_fallback_summary(u)

    async def _synthesize_trading_context_inner(self, u: EventUnderstanding) -> None:
        """Inner LLM synthesis call."""
        from .mentions_models import get_extraction_llm, get_subagent_model

        llm = get_extraction_llm(model=get_subagent_model(), temperature=0.2, max_tokens=1200)

        # Build compact context for synthesis
        context_parts = [f"Event: {u.title}"]
        if u.category:
            context_parts.append(f"Category: {u.category}")
        if u.domain:
            context_parts.append(f"Domain: {u.domain}")
        if u.participants:
            names = [p["name"] for p in u.participants[:5]]
            context_parts.append(f"Participants: {', '.join(names)}")
        if u.event_summary:
            context_parts.append(f"Summary: {u.event_summary[:300]}")
        if u.settlement_summary:
            context_parts.append(f"Settlement: {u.settlement_summary}")
        if u.time_to_close_hours is not None:
            context_parts.append(f"Time to close: {u.time_to_close_hours:.1f} hours")
        if u.mutually_exclusive:
            context_parts.append("Outcomes are mutually exclusive (prob sums to ~100%)")
        else:
            context_parts.append("Outcomes are independent")

        # Include news context if available
        if u.news_articles:
            news_snippets = []
            for article in u.news_articles[:5]:
                title = article.get("title", "")
                content = article.get("content", "")[:200]
                date = article.get("published_date", "")
                if title:
                    snippet = f"- {title}"
                    if date:
                        snippet += f" ({date})"
                    if content:
                        snippet += f": {content}"
                    news_snippets.append(snippet)
            if news_snippets:
                context_parts.append(f"Recent news:\n" + "\n".join(news_snippets))

        context_text = "\n".join(context_parts)

        prompt = f"""Analyze this prediction market event for a trader. Focus on what affects pricing.

{context_text}

Return JSON:
{{
    "trading_summary": "2-3 sentences: what a trader needs to know about this event",
    "key_factors": ["3-5 bullet points that affect market pricing"],
    "trading_considerations": ["2-3 points about liquidity, timing, edge sources"],
    "timeline": [{{"name": "segment name", "start_offset_min": 0, "duration_min": 60, "description": "brief"}}]
}}

Rules:
- Focus on PRICING, not biography
- Timeline should reflect the event's actual structure (pre-event, main event, post-event)
- trading_considerations: mention spreads, fee impact, time decay
- Return ONLY valid JSON"""

        structured_llm = llm.with_structured_output(EventUnderstandingExtraction)
        parsed = await structured_llm.ainvoke(prompt)

        u.trading_summary = parsed.trading_summary
        u.key_factors = parsed.key_factors[:5]
        u.trading_considerations = parsed.trading_considerations[:3]

        # Only use LLM timeline if we don't already have one
        if not u.timeline:
            u.timeline = [seg.model_dump() for seg in parsed.timeline[:5]]

    # === CACHE ===

    def _load_cache(self, event_ticker: str) -> Optional[EventUnderstanding]:
        if not self._cache_dir:
            return None
        path = os.path.join(self._cache_dir, f"understanding_{event_ticker}.json")
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r") as f:
                data = json.load(f)
            u = EventUnderstanding.from_dict(data)
            if u.stale:
                return None
            return u
        except (json.JSONDecodeError, OSError, KeyError) as e:
            logger.debug(f"[UNDERSTANDING] Cache load failed: {e}")
            return None

    def _save_cache(self, event_ticker: str, u: EventUnderstanding) -> None:
        if not self._cache_dir:
            return
        path = os.path.join(self._cache_dir, f"understanding_{event_ticker}.json")
        try:
            with open(path, "w") as f:
                json.dump(u.to_dict(), f, indent=2)
        except OSError as e:
            logger.debug(f"[UNDERSTANDING] Cache save failed: {e}")
