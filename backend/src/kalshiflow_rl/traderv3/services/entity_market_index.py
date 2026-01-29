"""
Entity-to-Market Index Service.

Builds and maintains an index mapping entities (people, organizations)
to Kalshi markets. Uses the yes_sub_title field from Kalshi market data
to identify which markets are about which entities.

Key functionality:
1. Fetches markets from Kalshi API
2. Extracts entity names from yes_sub_title
3. Normalizes names for fuzzy matching
4. Detects market type (OUT, WIN, CONFIRM, etc.)
5. Maintains entity → markets mapping
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from ..config.environment import DEFAULT_LIFECYCLE_CATEGORIES
from ..schemas.entity_schemas import (
    MarketMapping,
    EntityMarketEntry,
    CanonicalEntity,
    normalize_entity_id,
)
from ..nlp.knowledge_base import (
    KalshiKnowledgeBase,
    MarketData,
    set_kalshi_knowledge_base,
)

if TYPE_CHECKING:
    from ..clients.trading_client_integration import V3TradingClientIntegration
    from openai import AsyncOpenAI

logger = logging.getLogger("kalshiflow_rl.traderv3.services.entity_market_index")


# Market type detection patterns
MARKET_TYPE_PATTERNS = {
    "OUT": [
        r"\bout\b",
        r"leave",
        r"resign",
        r"withdraw",
        r"step.?down",
        r"fired",
        r"removed",
        r"shutdown",
        r"shut.*down",
    ],
    "CONFIRM": [
        r"confirm",
        r"approval",
        r"senate.*confirm",
        r"confirmation",
    ],
    "WIN": [
        r"\bwin\b",
        r"winner",
        r"victory",
        r"elect",
    ],
    "NOMINEE": [
        r"nomin",
        r"candidate",
        r"democratic.*nominee",
        r"republican.*nominee",
    ],
    "PRESIDENT": [
        r"president",
        r"potus",
        r"oval.?office",
        r"trump",
        r"biden",
    ],
    "MENTION": [
        r"mention",
        r"tweet",
        r"truth.*social",
        r"statement",
    ],
}

# ============================================================================
# LLM-Generated Alias System
# ============================================================================
# Instead of hardcoded nicknames, aliases are now generated dynamically
# by an LLM and cached in Supabase for persistence.
# See EntityMarketIndex._generate_llm_aliases() for the implementation.

# Common title prefixes to strip from names
TITLE_PREFIXES = [
    "mr", "mrs", "ms", "dr", "sen", "rep", "gov", "sec",
    "attorney general", "president", "vice president",
    "senator", "representative", "governor", "secretary",
    "speaker", "justice", "judge", "general", "admiral",
    "director", "ambassador",
]


def detect_market_type(event_title: str, market_ticker: str) -> str:
    """
    Detect market type from event title and ticker patterns.

    Args:
        event_title: Event title from Kalshi API
        market_ticker: Market ticker (e.g., "KXBONDIOUT-25FEB01")

    Returns:
        Market type: OUT, CONFIRM, WIN, NOMINEE, PRESIDENT, or UNKNOWN
    """
    combined = f"{event_title} {market_ticker}".lower()

    for market_type, patterns in MARKET_TYPE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, combined, re.IGNORECASE):
                return market_type

    return "UNKNOWN"


def build_aliases(
    canonical_name: str,
    entity_type: str = "person",
    llm_aliases: Optional[Set[str]] = None,
) -> Set[str]:
    """
    Build comprehensive alias set from canonical name.

    This generates all possible aliases for an entity including:
    - Normalized forms (lowercase, underscored)
    - Individual name parts
    - Initials for people
    - Title-stripped versions
    - LLM-generated aliases (if provided)

    Args:
        canonical_name: The canonical entity name (e.g., "Donald Trump")
        entity_type: Entity type ("person", "organization", "position")
        llm_aliases: Optional set of LLM-generated aliases to merge

    Returns:
        Set of aliases for matching
    """
    aliases = set()
    name_lower = canonical_name.lower().strip()

    # Standard variations
    aliases.add(name_lower)  # "donald trump"
    aliases.add(normalize_entity_id(name_lower))  # "donald_trump"

    # Individual name parts (>2 chars to avoid common words)
    parts = name_lower.split()
    for part in parts:
        if len(part) > 2:
            aliases.add(part)  # "donald", "trump"

    # Initials and last name (for people)
    if entity_type == "person" and len(parts) >= 2:
        # Last name alone
        aliases.add(parts[-1])  # "trump"
        # Initial + last name
        aliases.add(f"{parts[0][0]}. {parts[-1]}")  # "d. trump"
        # All initials
        aliases.add("".join(p[0] for p in parts if p))  # "dt"

    # Title removal - strip common titles
    for title in TITLE_PREFIXES:
        if name_lower.startswith(title + " "):
            stripped = name_lower[len(title) + 1:].strip()
            if stripped:
                aliases.add(stripped)
                # Also add normalized version of stripped name
                aliases.add(normalize_entity_id(stripped))

    # Merge LLM-generated aliases if provided
    if llm_aliases:
        aliases.update(llm_aliases)

    return aliases


def extract_name_variations(name: str) -> Set[str]:
    """
    Generate name variations for fuzzy matching.

    This is a wrapper around build_aliases for backward compatibility.

    Args:
        name: Entity name (e.g., "Pam Bondi")

    Returns:
        Set of name variations for matching
    """
    return build_aliases(name, entity_type="person")


def detect_entity_type(canonical_name: str, event_title: str = "", market_ticker: str = "") -> str:
    """
    Detect entity type from name and market context.

    Args:
        canonical_name: The canonical entity name
        event_title: Optional event title for context
        market_ticker: Optional market ticker for context

    Returns:
        Entity type: "person", "organization", or "position"
    """
    name_lower = canonical_name.lower()
    combined_context = f"{event_title} {market_ticker}".lower()

    # Position keywords
    position_keywords = [
        "secretary", "speaker", "pope", "president", "minister",
        "director", "chair", "ceo", "attorney general", "justice",
    ]
    for keyword in position_keywords:
        if keyword in name_lower or keyword in combined_context:
            # Check if it looks like a person (has multiple name parts)
            parts = canonical_name.split()
            if len(parts) >= 2 and all(p[0].isupper() for p in parts if p):
                return "person"
            return "position"

    # Organization keywords
    org_keywords = [
        "committee", "department", "party", "foundation",
        "inc", "corp", "llc", "council", "administration",
    ]
    for keyword in org_keywords:
        if keyword in name_lower:
            return "organization"

    # Default to person for names that look like people
    parts = canonical_name.split()
    if len(parts) >= 2:
        # Multiple capitalized words likely means a person's name
        return "person"

    return "person"  # Default


@dataclass
class EntityMarketIndexConfig:
    """Configuration for the Entity-Market Index."""

    refresh_interval_seconds: float = 300.0  # 5 minutes
    min_confidence: float = 0.6  # Minimum mapping confidence

    # Dynamic category-based discovery (no hardcoded series!)
    categories: List[str] = field(default_factory=lambda: list(DEFAULT_LIFECYCLE_CATEGORIES))
    sports_prefixes: List[str] = field(default_factory=list)  # e.g., ["KXNFL"]
    min_hours_to_settlement: float = 4.0
    max_days_to_settlement: int = 30

    exclude_patterns: List[str] = field(default_factory=lambda: [
        r"^test",
        r"^demo",
    ])
    enabled: bool = True


class EntityMarketIndex:
    """
    Service that maintains a mapping from entities to Kalshi markets.

    The index is built by:
    1. Fetching events/markets from Kalshi API
    2. Extracting entity names from yes_sub_title
    3. Normalizing names for consistent matching
    4. Detecting market types from event titles/tickers

    Usage:
        index = EntityMarketIndex(trading_client)
        await index.start()
        markets = index.get_markets_for_entity("pam_bondi")
    """

    def __init__(
        self,
        trading_client: Optional["V3TradingClientIntegration"] = None,
        config: Optional[EntityMarketIndexConfig] = None,
        websocket_manager=None,
    ):
        """
        Initialize the Entity-Market Index.

        Args:
            trading_client: Kalshi trading client for API access
            config: Index configuration
            websocket_manager: WebSocket manager for broadcasting entity updates
        """
        self._trading_client = trading_client
        self._config = config or EntityMarketIndexConfig()
        self._websocket_manager = websocket_manager

        # The main index: entity_id -> EntityMarketEntry
        self._index: Dict[str, EntityMarketEntry] = {}

        # Canonical entities with full alias support
        self._canonical_entities: Dict[str, CanonicalEntity] = {}

        # Reverse index: market_ticker -> list of entity_ids
        self._reverse_index: Dict[str, List[str]] = {}

        # Name variations for fuzzy matching: variation -> entity_id
        self._name_variations: Dict[str, str] = {}

        # All markets cache (for MarketImpactReasoner - includes markets without entities)
        self._all_markets_cache: List[Dict[str, Any]] = []

        # Alias lookup: alias -> entity_id (comprehensive alias matching)
        self._alias_lookup: Dict[str, str] = {}

        # State
        self._running = False
        self._last_refresh_at: Optional[float] = None
        self._refresh_task: Optional[asyncio.Task] = None

        # Stats
        self._total_entities = 0
        self._total_markets = 0
        self._refresh_count = 0
        self._total_aliases = 0
        self._llm_aliases_generated = 0

        # spaCy Knowledge Base (for NLP pipeline integration)
        self._kb: Optional[KalshiKnowledgeBase] = None
        self._kb_lock = asyncio.Lock()  # Protects KB access during refresh

        # LLM client for alias generation
        self._llm_client: Optional["AsyncOpenAI"] = None
        self._llm_alias_model: str = "gpt-4o-mini"

        # In-memory alias cache: entity_id -> set of LLM-generated aliases
        # This is populated from Supabase on startup and updated on generation
        self._llm_alias_cache: Dict[str, Set[str]] = {}

        # Supabase client for alias persistence (lazy initialized)
        self._supabase = None

        # Keyword cache: event_ticker -> list of keywords
        self._keyword_cache: Dict[str, List[str]] = {}

        # Reverse ticker index: market_ticker -> (entity_id, MarketMapping)
        self._ticker_to_entity: Dict[str, Tuple[str, MarketMapping]] = {}

        # Enriched market list (built once per refresh, not per-post)
        self._enriched_market_list: List[Dict[str, str]] = []
        self._enriched_market_prompt: str = ""

    async def build_index(
        self,
        trading_client: Optional["V3TradingClientIntegration"] = None,
        categories: Optional[List[str]] = None,
    ) -> None:
        """
        Build the entity-market index from Kalshi API.

        This is a one-shot build without starting the periodic refresh.
        Use start() for continuous operation.

        Args:
            trading_client: Kalshi trading client for API access
            categories: Categories to filter (e.g., ["Politics"]) - uses config default if not specified
        """
        if trading_client:
            self._trading_client = trading_client

        if categories:
            self._config.categories = categories

        # Load cached data from Supabase before building
        await self._load_cached_aliases()
        await self._load_cached_keywords()

        await self._refresh_index()
        logger.info(
            f"[entity_index] Built index: {self._total_entities} entities, "
            f"{self._total_markets} markets"
        )

    async def start(self) -> None:
        """Start the index service with periodic refresh."""
        if self._running:
            logger.warning("[entity_index] Already running")
            return

        if not self._config.enabled:
            logger.info("[entity_index] Entity market index disabled")
            return

        logger.info("[entity_index] Starting entity-market index service")
        self._running = True

        # Load cached LLM aliases from Supabase
        await self._load_cached_aliases()

        # Load cached event keywords from Supabase
        await self._load_cached_keywords()

        # Initial build
        await self._refresh_index()

        # Start periodic refresh
        self._refresh_task = asyncio.create_task(self._refresh_loop())

        logger.info(
            f"[entity_index] Started with {self._total_entities} entities, "
            f"{self._total_markets} markets"
        )

    async def stop(self) -> None:
        """Stop the index service."""
        if not self._running:
            return

        logger.info("[entity_index] Stopping entity-market index service")
        self._running = False

        if self._refresh_task and not self._refresh_task.done():
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass

        logger.info("[entity_index] Stopped")

    # =========================================================================
    # LLM Alias Generation
    # =========================================================================

    def _get_llm_client(self) -> "AsyncOpenAI":
        """Get or create async OpenAI client for alias generation."""
        if self._llm_client is None:
            from openai import AsyncOpenAI
            self._llm_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return self._llm_client

    def _get_supabase_client(self):
        """Get or create Supabase client for alias persistence."""
        if self._supabase is None:
            from supabase import create_client
            url = os.getenv("SUPABASE_URL") or os.getenv("DATABASE_URL", "").replace(
                "postgresql://", "https://"
            ).split("?")[0].replace(":5432", "")
            key = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_ANON_KEY", "")
            if url and key:
                self._supabase = create_client(url, key)
        return self._supabase

    async def _load_cached_aliases(self) -> None:
        """Load LLM-generated aliases from Supabase into memory cache."""
        try:
            supabase = self._get_supabase_client()
            if not supabase:
                logger.debug("[entity_index] No Supabase client for alias cache")
                return

            # Fetch all cached aliases
            response = supabase.table("entity_aliases").select("*").execute()
            if response.data:
                for row in response.data:
                    entity_id = row.get("entity_id")
                    aliases = row.get("aliases", [])
                    if entity_id and aliases:
                        self._llm_alias_cache[entity_id] = set(aliases)

                logger.info(
                    f"[entity_index] Loaded {len(self._llm_alias_cache)} entities "
                    f"with cached LLM aliases"
                )
        except Exception as e:
            # Table might not exist yet - that's ok
            if "entity_aliases" not in str(e):
                logger.debug(f"[entity_index] Alias cache load: {e}")

    async def _save_aliases_to_cache(
        self, entity_id: str, canonical_name: str, aliases: Set[str]
    ) -> None:
        """Save LLM-generated aliases to Supabase for persistence."""
        try:
            supabase = self._get_supabase_client()
            if not supabase:
                return

            # Upsert the aliases
            supabase.table("entity_aliases").upsert({
                "entity_id": entity_id,
                "canonical_name": canonical_name,
                "aliases": list(aliases),
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }, on_conflict="entity_id").execute()

        except Exception as e:
            logger.debug(f"[entity_index] Alias cache save error: {e}")

    # =========================================================================
    # Event Keyword Generation (for enriched market context)
    # =========================================================================

    async def _load_cached_keywords(self) -> None:
        """Load event keywords from Supabase into memory cache."""
        try:
            supabase = self._get_supabase_client()
            if not supabase:
                logger.debug("[entity_index] No Supabase client for keyword cache")
                return

            response = supabase.table("entity_event_keywords").select("*").execute()
            if response.data:
                for row in response.data:
                    event_ticker = row.get("event_ticker")
                    keywords = row.get("keywords", [])
                    if event_ticker and keywords:
                        self._keyword_cache[event_ticker] = keywords

                logger.info(
                    f"[entity_index] Loaded {len(self._keyword_cache)} events "
                    f"with cached keywords"
                )
        except Exception as e:
            if "entity_event_keywords" not in str(e):
                logger.debug(f"[entity_index] Keyword cache load: {e}")

    async def _save_keywords_to_cache(
        self, event_ticker: str, event_title: str, keywords: List[str]
    ) -> None:
        """Save event keywords to Supabase for persistence."""
        try:
            supabase = self._get_supabase_client()
            if not supabase:
                return

            supabase.table("entity_event_keywords").upsert({
                "event_ticker": event_ticker,
                "event_title": event_title,
                "keywords": keywords,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }, on_conflict="event_ticker").execute()

        except Exception as e:
            logger.debug(f"[entity_index] Keyword cache save error: {e}")

    async def _generate_event_keywords(
        self,
        event_ticker: str,
        event_title: str,
        event_subtitle: str,
        yes_sub_title: str,
        category: str = "",
    ) -> List[str]:
        """
        Generate matching keywords for an event using LLM. Cached in Supabase.

        Given event metadata, generates 5-10 keywords/phrases that Reddit posts
        might use when discussing this event.

        Args:
            event_ticker: Event ticker (e.g., "KXBONDIOUT")
            event_title: Event title
            event_subtitle: Event subtitle
            yes_sub_title: YES outcome subtitle
            category: Event category (e.g., "Politics")

        Returns:
            List of keyword strings
        """
        # Check in-memory cache first
        if event_ticker in self._keyword_cache:
            return self._keyword_cache[event_ticker]

        try:
            client = self._get_llm_client()

            prompt = f"""Generate 5-10 keywords/phrases that people on Reddit might use when discussing this prediction market event.

Event: "{event_title}"
Subtitle: "{event_subtitle}"
Outcome: "{yes_sub_title}"
Category: {category or "General"}

Include:
- People's names (full and short forms)
- Key topics and issues
- Common abbreviations or acronyms
- Related concepts that would appear in news/Reddit posts about this
- Informal terms people might use

Return ONLY a JSON array of lowercase strings. Example: ["keyword1", "key phrase 2", "acronym"]
Generate 5-10 keywords maximum."""

            response = await client.chat.completions.create(
                model=self._llm_alias_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.3,
            )

            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            content = content.strip()

            keywords = json.loads(content)
            keywords = [k.lower().strip() for k in keywords if k]

            # Cache in memory and Supabase
            self._keyword_cache[event_ticker] = keywords
            await self._save_keywords_to_cache(event_ticker, event_title, keywords)

            logger.info(
                f"[entity_index] Generated {len(keywords)} keywords for event "
                f"'{event_ticker}': {keywords[:5]}{'...' if len(keywords) > 5 else ''}"
            )

            return keywords

        except json.JSONDecodeError as e:
            logger.warning(f"[entity_index] Keyword parse error for '{event_ticker}': {e}")
            return []
        except Exception as e:
            logger.warning(f"[entity_index] Keyword generation error for '{event_ticker}': {e}")
            return []

    # =========================================================================
    # LLM Alias Generation
    # =========================================================================

    async def _generate_llm_aliases(
        self, canonical_name: str, entity_type: str, context: str = ""
    ) -> Set[str]:
        """
        Generate aliases for an entity using LLM.

        Uses GPT-4o-mini to generate common nicknames, abbreviations,
        and alternative names that might appear in news headlines.

        Args:
            canonical_name: The canonical entity name (e.g., "Donald Trump")
            entity_type: Entity type ("person", "organization", "position")
            context: Optional context like event title for better alias generation

        Returns:
            Set of lowercase aliases
        """
        entity_id = normalize_entity_id(canonical_name)

        # Check in-memory cache first
        if entity_id in self._llm_alias_cache:
            return self._llm_alias_cache[entity_id]

        # Generate via LLM
        try:
            client = self._get_llm_client()

            prompt = f"""Generate common alternative names, nicknames, and shortened forms for the {entity_type}: "{canonical_name}"

These aliases will be used to match mentions in news headlines. Include:
- Common nicknames (e.g., "Bibi" for Benjamin Netanyahu)
- Shortened versions (e.g., "RFK Jr" for Robert Kennedy Jr)
- Title variations (e.g., "Senator Cruz" for Ted Cruz)
- Initials if commonly used (e.g., "DJT" for Donald Trump)
- Informal names (e.g., "Bernie" for Bernie Sanders)

{f'Context: {context}' if context else ''}

Return ONLY a JSON array of strings in lowercase. Example: ["nickname1", "alias2", "short form"]
Do NOT include the canonical name itself: "{canonical_name.lower()}"
Generate 3-8 aliases maximum."""

            response = await client.chat.completions.create(
                model=self._llm_alias_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.3,
            )

            # Parse JSON response
            content = response.choices[0].message.content.strip()
            # Handle potential markdown code blocks
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            content = content.strip()

            aliases_list = json.loads(content)
            aliases = set(alias.lower().strip() for alias in aliases_list if alias)

            # Cache in memory and Supabase
            self._llm_alias_cache[entity_id] = aliases
            await self._save_aliases_to_cache(entity_id, canonical_name, aliases)
            self._llm_aliases_generated += 1

            logger.info(
                f"[entity_index] Generated {len(aliases)} LLM aliases for '{canonical_name}': "
                f"{list(aliases)[:5]}{'...' if len(aliases) > 5 else ''}"
            )

            return aliases

        except json.JSONDecodeError as e:
            logger.warning(
                f"[entity_index] LLM alias parse error for '{canonical_name}': {e}"
            )
            return set()
        except Exception as e:
            logger.warning(
                f"[entity_index] LLM alias generation error for '{canonical_name}': {e}"
            )
            return set()

    @staticmethod
    def _extract_aliases_from_text(
        event_title: str,
        event_subtitle: str,
        yes_sub_title: str,
    ) -> Set[str]:
        """
        Extract additional entity aliases from event title and subtitle.

        Parses proper nouns (capitalized multi-word sequences) from the event
        title/subtitle that differ from the yes_sub_title entity. This captures
        entity references in market questions, e.g., "Will Amazon stock drop?"
        has "Amazon" in the title.

        Args:
            event_title: Event title text
            event_subtitle: Event subtitle text
            yes_sub_title: The primary entity name (already handled separately)

        Returns:
            Set of additional lowercase aliases extracted from title/subtitle
        """
        extra_aliases: Set[str] = set()
        yes_lower = yes_sub_title.lower().strip()

        for text in (event_title, event_subtitle):
            if not text:
                continue
            text_lower = text.lower()

            # Skip if the text IS the yes_sub_title (no new info)
            if text_lower.strip() == yes_lower:
                continue

            # Extract capitalized word sequences (proper nouns)
            # Matches sequences like "Amazon", "Gavin Newsom", "Project Dawn"
            proper_nouns = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", text)
            for noun in proper_nouns:
                noun_lower = noun.lower()
                # Skip if it's already the primary entity
                if noun_lower == yes_lower:
                    continue
                # Skip very short matches (likely not entities)
                if len(noun_lower) < 3:
                    continue
                # Skip common non-entity capitalized words
                skip_words = {
                    "will", "what", "who", "when", "where", "how", "does",
                    "the", "yes", "monday", "tuesday", "wednesday",
                    "thursday", "friday", "saturday", "sunday",
                    "january", "february", "march", "april", "may",
                    "june", "july", "august", "september", "october",
                    "november", "december",
                }
                if noun_lower in skip_words:
                    continue
                extra_aliases.add(noun_lower)

            # Also extract ALL-CAPS acronyms (e.g., "AWS", "ICE", "NASA")
            acronyms = re.findall(r"\b([A-Z]{2,6})\b", text)
            for acr in acronyms:
                acr_lower = acr.lower()
                if len(acr_lower) >= 2 and acr_lower != yes_lower:
                    extra_aliases.add(acr_lower)

        return extra_aliases

    async def _refresh_loop(self) -> None:
        """Periodically refresh the index."""
        while self._running:
            try:
                await asyncio.sleep(self._config.refresh_interval_seconds)
                await self._refresh_index()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[entity_index] Refresh error: {e}")
                await asyncio.sleep(30.0)  # Brief pause on error

    async def _refresh_index(self) -> None:
        """Refresh the entity-market index from Kalshi API using category-based discovery."""
        if not self._trading_client:
            logger.warning("[entity_index] No trading client available")
            return

        logger.info(f"[entity_index] Refreshing index with categories: {self._config.categories}")

        new_index: Dict[str, EntityMarketEntry] = {}
        new_canonical_entities: Dict[str, CanonicalEntity] = {}
        new_reverse_index: Dict[str, List[str]] = {}
        new_name_variations: Dict[str, str] = {}
        new_alias_lookup: Dict[str, str] = {}
        new_ticker_to_entity: Dict[str, Tuple[str, MarketMapping]] = {}

        try:
            # Single call to get all markets for configured categories
            markets = await self._trading_client.get_open_markets(
                categories=self._config.categories,
                sports_prefixes=self._config.sports_prefixes if self._config.sports_prefixes else None,
                min_hours_to_settlement=self._config.min_hours_to_settlement,
                max_days_to_settlement=self._config.max_days_to_settlement,
            )

            logger.info(
                f"[entity_index] Fetched {len(markets)} markets for categories: {self._config.categories}"
            )

            # Cache all markets for MarketImpactReasoner (includes markets without entities)
            self._all_markets_cache = markets

            current_time = time.time()
            markets_processed = 0

            for market in markets:
                market_ticker = market.get("ticker", "")
                yes_sub_title = market.get("yes_sub_title", "")
                event_ticker = market.get("event_ticker", "")
                # Use event-level title/subtitle (propagated from get_open_markets)
                # Fall back to market-level title/subtitle
                event_title = (
                    market.get("event_title", "")
                    or market.get("title", "")
                    or market.get("subtitle", "")
                )
                event_subtitle = market.get("event_subtitle", "")

                if not yes_sub_title or not market_ticker:
                    continue

                # Skip excluded patterns
                if any(
                    re.search(p, market_ticker.lower())
                    for p in self._config.exclude_patterns
                ):
                    continue

                markets_processed += 1

                # Detect market type
                market_type = detect_market_type(event_title, market_ticker)

                # Create entity from yes_sub_title
                entity_id = normalize_entity_id(yes_sub_title)

                # Detect entity type from context
                entity_type = detect_entity_type(yes_sub_title, event_title, market_ticker)

                # Create mapping
                mapping = MarketMapping(
                    market_ticker=market_ticker,
                    event_ticker=event_ticker,
                    market_type=market_type,
                    yes_sub_title=yes_sub_title,
                    confidence=0.9 if market_type != "UNKNOWN" else 0.7,
                )

                # Get LLM-generated aliases (from cache or generate new)
                # Note: This uses cached aliases if available, generates async if not
                llm_aliases = self._llm_alias_cache.get(entity_id, set())

                # Build comprehensive aliases with LLM aliases included
                aliases = build_aliases(yes_sub_title, entity_type, llm_aliases)

                # Extract additional aliases from event title and subtitle
                # This captures entity references in market questions
                # e.g., "Will Amazon stock drop?" has "Amazon" in the title
                extra_aliases = self._extract_aliases_from_text(
                    event_title, event_subtitle, yes_sub_title
                )
                aliases.update(extra_aliases)

                # Add to legacy index (backward compatibility)
                if entity_id not in new_index:
                    new_index[entity_id] = EntityMarketEntry(
                        entity_id=entity_id,
                        canonical_name=yes_sub_title,
                        markets=[mapping],
                        last_updated=current_time,
                    )
                else:
                    # Check if market already exists
                    existing_tickers = {
                        m.market_ticker
                        for m in new_index[entity_id].markets
                    }
                    if market_ticker not in existing_tickers:
                        new_index[entity_id].markets.append(mapping)

                # Add to CanonicalEntity index
                if entity_id not in new_canonical_entities:
                    new_canonical_entities[entity_id] = CanonicalEntity(
                        entity_id=entity_id,
                        canonical_name=yes_sub_title,
                        entity_type=entity_type,
                        aliases=aliases,
                        markets=[mapping],
                        created_at=current_time,
                        last_seen_at=current_time,
                    )
                else:
                    # Update existing canonical entity
                    existing = new_canonical_entities[entity_id]
                    existing.aliases.update(aliases)
                    existing.last_seen_at = current_time
                    existing_tickers = {m.market_ticker for m in existing.markets}
                    if market_ticker not in existing_tickers:
                        existing.markets.append(mapping)

                # Build reverse index
                if market_ticker not in new_reverse_index:
                    new_reverse_index[market_ticker] = []
                if entity_id not in new_reverse_index[market_ticker]:
                    new_reverse_index[market_ticker].append(entity_id)

                # Build name variations (legacy)
                for variation in aliases:
                    new_name_variations[variation] = entity_id

                # Build comprehensive alias lookup
                for alias in aliases:
                    new_alias_lookup[alias.lower()] = entity_id

                # Build reverse ticker -> entity index
                new_ticker_to_entity[market_ticker] = (entity_id, mapping)

            # Update indices atomically
            self._index = new_index
            self._canonical_entities = new_canonical_entities
            self._reverse_index = new_reverse_index
            self._name_variations = new_name_variations
            self._alias_lookup = new_alias_lookup
            self._ticker_to_entity = new_ticker_to_entity
            self._last_refresh_at = time.time()
            self._refresh_count += 1
            self._total_aliases = len(new_alias_lookup)

            self._total_entities = len(new_index)
            self._total_markets = sum(len(e.markets) for e in new_index.values())

            logger.info(
                f"[entity_index] Refresh complete: {self._total_entities} entities, "
                f"{self._total_markets} market mappings, {self._total_aliases} aliases "
                f"(from {markets_processed} politics markets)"
            )

            # CRITICAL: Generate LLM aliases BEFORE KB sync (not after!)
            # Otherwise new entities miss NER matching for 1 refresh cycle
            entities_needing_aliases = [
                (entity_id, ce.canonical_name, ce.entity_type)
                for entity_id, ce in new_canonical_entities.items()
                if entity_id not in self._llm_alias_cache
            ]
            if entities_needing_aliases:
                # Limit to first 10 new entities per refresh to avoid rate limits
                entities_to_generate = entities_needing_aliases[:10]
                logger.info(
                    f"[entity_index] Generating LLM aliases for {len(entities_to_generate)} "
                    f"new entities BEFORE KB sync"
                )
                # Generate aliases SYNCHRONOUSLY (await) before KB sync
                tasks = [
                    self._generate_llm_aliases(canonical_name, entity_type)
                    for _, canonical_name, entity_type in entities_to_generate
                ]
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                    # Update alias lookups with newly generated aliases
                    for entity_id, ce in new_canonical_entities.items():
                        if entity_id in self._llm_alias_cache:
                            ce.aliases.update(self._llm_alias_cache[entity_id])
                            for alias in self._llm_alias_cache[entity_id]:
                                self._alias_lookup[alias.lower()] = entity_id
                    # Update total aliases count after LLM generation
                    self._total_aliases = len(self._alias_lookup)

            # Generate event keywords for enriched market context
            # Collect unique event tickers that need keywords
            events_needing_keywords = set()
            for market in markets:
                event_ticker = market.get("event_ticker", "")
                if event_ticker and event_ticker not in self._keyword_cache:
                    events_needing_keywords.add(event_ticker)

            if events_needing_keywords:
                # Limit to first 10 new events per refresh to avoid rate limits
                events_to_generate = list(events_needing_keywords)[:10]
                logger.info(
                    f"[entity_index] Generating keywords for {len(events_to_generate)} "
                    f"new events"
                )
                keyword_tasks = []
                for evt_ticker in events_to_generate:
                    # Find a market with this event ticker for metadata
                    evt_market = next(
                        (m for m in markets if m.get("event_ticker") == evt_ticker), None
                    )
                    if evt_market:
                        keyword_tasks.append(
                            self._generate_event_keywords(
                                event_ticker=evt_ticker,
                                event_title=evt_market.get("event_title", "") or evt_market.get("title", ""),
                                event_subtitle=evt_market.get("event_subtitle", ""),
                                yes_sub_title=evt_market.get("yes_sub_title", ""),
                                category=evt_market.get("category", ""),
                            )
                        )
                if keyword_tasks:
                    await asyncio.gather(*keyword_tasks, return_exceptions=True)

            # Build enriched market list (for LLM market-aware extraction)
            new_enriched_list = []
            for entity_id, ce in new_canonical_entities.items():
                for m in ce.markets:
                    keywords = self._keyword_cache.get(m.event_ticker, [])
                    new_enriched_list.append({
                        "ticker": m.market_ticker,
                        "title": m.yes_sub_title,
                        "keywords": ", ".join(keywords[:8]),
                    })
            self._enriched_market_list = new_enriched_list[:200]

            # Pre-format the prompt string so extract_with_markets() doesn't rebuild per-post
            self._enriched_market_prompt = "\n".join(
                f"{m['ticker']} | {m['title']} | {m['keywords']}" for m in self._enriched_market_list
            )

            logger.info(
                f"[entity_index] Enriched market list: {len(self._enriched_market_list)} markets, "
                f"{len(self._keyword_cache)} events with keywords"
            )

            # Synchronize Knowledge Base for NLP pipeline
            # NOTE: This MUST happen AFTER LLM alias generation so KB has complete aliases
            await self._sync_knowledge_base()

            # Broadcast entity index snapshot to connected clients
            if self._websocket_manager:
                try:
                    await self._websocket_manager.broadcast_entity_index_snapshot()
                except Exception as broadcast_error:
                    logger.warning(f"[entity_index] Broadcast error: {broadcast_error}")

        except Exception as e:
            logger.error(f"[entity_index] Refresh failed: {e}")

    def get_markets_for_entity(
        self,
        entity_id_or_name: str,
        market_types: Optional[List[str]] = None,
    ) -> List[MarketMapping]:
        """
        Get all markets associated with an entity.

        Args:
            entity_id_or_name: Entity ID or name (supports fuzzy matching)
            market_types: Optional filter for specific market types

        Returns:
            List of MarketMapping objects
        """
        # First try direct lookup
        entity_id = normalize_entity_id(entity_id_or_name)
        entry = self._index.get(entity_id)

        # If not found, try name variations
        if not entry:
            for variation in extract_name_variations(entity_id_or_name):
                if variation in self._name_variations:
                    mapped_id = self._name_variations[variation]
                    entry = self._index.get(mapped_id)
                    if entry:
                        break

        if not entry:
            return []

        markets = entry.markets

        # Filter by market type if specified
        if market_types:
            market_types_upper = [mt.upper() for mt in market_types]
            markets = [m for m in markets if m.market_type in market_types_upper]

        return markets

    def get_entity_for_market(self, market_ticker: str) -> Optional[EntityMarketEntry]:
        """
        Get the entity associated with a market.

        Args:
            market_ticker: Kalshi market ticker

        Returns:
            EntityMarketEntry if found, None otherwise
        """
        entity_ids = self._reverse_index.get(market_ticker, [])
        if not entity_ids:
            return None

        # Return the first entity (typically there's only one)
        return self._index.get(entity_ids[0])

    def find_entity_by_name(self, name: str) -> Optional[EntityMarketEntry]:
        """
        Find an entity by name using fuzzy matching.

        Args:
            name: Entity name to search for

        Returns:
            EntityMarketEntry if found, None otherwise
        """
        for variation in extract_name_variations(name):
            if variation in self._name_variations:
                entity_id = self._name_variations[variation]
                return self._index.get(entity_id)
        return None

    def get_all_entities(self) -> List[EntityMarketEntry]:
        """Get all entities in the index."""
        return list(self._index.values())

    def get_all_canonical_entities(self) -> List[CanonicalEntity]:
        """Get all canonical entities with aliases and aggregated signals."""
        return list(self._canonical_entities.values())

    def get_all_markets_for_reasoner(self) -> List[Dict[str, str]]:
        """
        Get all cached markets in format suitable for MarketImpactReasoner.

        Returns markets including those without direct entity matches,
        enabling indirect impact reasoning.

        Returns:
            List of dicts with ticker, title, event_ticker fields
        """
        result = []
        for market in self._all_markets_cache:
            ticker = market.get("ticker", "")
            title = market.get("title", "") or market.get("subtitle", "")
            event_ticker = market.get("event_ticker", "")

            if ticker and title:
                result.append({
                    "ticker": ticker,
                    "title": title,
                    "event_ticker": event_ticker,
                })
        return result

    def get_canonical_entity(self, entity_id: str) -> Optional[CanonicalEntity]:
        """Get a specific canonical entity by ID."""
        return self._canonical_entities.get(entity_id)

    def find_entity_by_alias(self, alias: str) -> Optional[CanonicalEntity]:
        """
        Find an entity by any of its aliases.

        Uses the comprehensive alias lookup for fast O(1) matching.

        Args:
            alias: Any alias (nickname, partial name, etc.)

        Returns:
            CanonicalEntity if found, None otherwise
        """
        alias_lower = alias.lower().strip()
        entity_id = self._alias_lookup.get(alias_lower)
        if entity_id:
            return self._canonical_entities.get(entity_id)
        return None

    def get_market_mapping_by_ticker(self, ticker: str) -> Optional[Tuple[str, MarketMapping]]:
        """
        Reverse lookup: ticker -> (entity_id, mapping) for LLM direct matches.

        Args:
            ticker: Market ticker to look up

        Returns:
            Tuple of (entity_id, MarketMapping) if found, None otherwise
        """
        return self._ticker_to_entity.get(ticker)

    def get_enriched_market_list(self) -> List[Dict[str, str]]:
        """Return cached enriched market list (rebuilt on index refresh only)."""
        return self._enriched_market_list

    def get_enriched_market_prompt(self) -> str:
        """Return pre-formatted market list string for LLM prompt."""
        return self._enriched_market_prompt

    def update_entity_reddit_stats(
        self,
        entity_id: str,
        mentions: int = 0,
        sentiment: float = 0.0,
    ) -> None:
        """
        Update reddit signal stats for an entity.

        Called by the price impact agent when new signals arrive.

        Args:
            entity_id: Entity ID to update
            mentions: Number of new mentions to add
            sentiment: Sentiment value to aggregate
        """
        entity = self._canonical_entities.get(entity_id)
        if entity:
            entity.reddit_mentions += mentions
            # Running average for sentiment
            if entity.aggregate_sentiment == 0.0:
                entity.aggregate_sentiment = sentiment
            else:
                entity.aggregate_sentiment = (entity.aggregate_sentiment + sentiment) / 2
            entity.last_reddit_signal = time.time()

    def set_websocket_manager(self, websocket_manager) -> None:
        """Set the WebSocket manager for broadcasting entity updates."""
        self._websocket_manager = websocket_manager

    async def _sync_knowledge_base(self) -> None:
        """
        Synchronize the spaCy KnowledgeBase with current market data.

        Uses atomic swap pattern to prevent race conditions:
        1. Creates NEW KB instance
        2. Populates it completely
        3. Atomically swaps: self._kb = new_kb
        4. Sets global KB instance for NLP pipeline access

        This ensures the NLP pipeline never sees a partially-populated KB.
        """
        try:
            import spacy

            # Get or create shared vocab
            from ..nlp.pipeline import get_shared_vocab, set_shared_vocab

            vocab = get_shared_vocab()
            if vocab is None:
                # Load a blank model to get vocab
                nlp = spacy.blank("en")
                vocab = nlp.vocab
                set_shared_vocab(vocab)

            # Create NEW KB instance (atomic swap pattern - never clear existing)
            new_kb = KalshiKnowledgeBase(vocab, entity_vector_length=64)

            # Build market data for KB population
            markets_for_kb = self._get_markets_for_kb()

            # Create alias builder that includes LLM aliases from cache
            # This is critical - without LLM aliases, the KB won't match Reddit entities!
            def alias_builder_with_llm(name: str, entity_type: str) -> Set[str]:
                entity_id = normalize_entity_id(name)
                llm_aliases = self._llm_alias_cache.get(entity_id, set())
                return build_aliases(name, entity_type, llm_aliases)

            # Populate new KB with LLM-enhanced alias builder
            new_kb.populate_from_markets(markets_for_kb, alias_builder=alias_builder_with_llm)

            # Atomic swap under lock to prevent concurrent access issues
            async with self._kb_lock:
                self._kb = new_kb
                # Set global KB instance
                set_kalshi_knowledge_base(new_kb)

            logger.info(
                f"[entity_index] KB synced: {new_kb.get_entity_count()} entities, "
                f"{new_kb.get_alias_count()} aliases"
            )

        except Exception as e:
            logger.error(f"[entity_index] KB sync failed: {e}")

    def _get_markets_for_kb(self) -> List[MarketData]:
        """
        Build MarketData list for KB population from current index.

        Returns:
            List of MarketData objects for KB.populate_from_markets()
        """
        markets = []

        for entity_id, canonical_entity in self._canonical_entities.items():
            for mapping in canonical_entity.markets:
                markets.append(MarketData(
                    ticker=mapping.market_ticker,
                    event_ticker=mapping.event_ticker,
                    yes_sub_title=canonical_entity.canonical_name,
                    market_type=mapping.market_type,
                    volume_24h=None,  # Could enhance with real volume data
                ))

        return markets

    def get_knowledge_base(self) -> Optional[KalshiKnowledgeBase]:
        """
        Get the synchronized KnowledgeBase.

        Returns:
            KalshiKnowledgeBase if synced, None otherwise
        """
        return self._kb

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        kb_stats = None
        if self._kb:
            kb_stats = {
                "entity_count": self._kb.get_entity_count(),
                "alias_count": self._kb.get_alias_count(),
                "entity_vector_length": self._kb.entity_vector_length,
            }

        return {
            "running": self._running,
            "enabled": self._config.enabled,
            "total_entities": self._total_entities,
            "total_markets": self._total_markets,
            "total_aliases": self._total_aliases,
            "llm_aliases_generated": self._llm_aliases_generated,
            "llm_alias_cache_size": len(self._llm_alias_cache),
            "refresh_count": self._refresh_count,
            "last_refresh_at": self._last_refresh_at,
            "categories": self._config.categories,
            "knowledge_base": kb_stats,
        }


# Global singleton instance
_global_entity_index: Optional[EntityMarketIndex] = None


def get_entity_market_index() -> Optional[EntityMarketIndex]:
    """Get the global EntityMarketIndex instance."""
    return _global_entity_index


def set_entity_market_index(index: EntityMarketIndex) -> None:
    """Set the global EntityMarketIndex instance."""
    global _global_entity_index
    _global_entity_index = index
    logger.info("[entity_index] Set global EntityMarketIndex instance")
