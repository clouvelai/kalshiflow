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
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

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
    ],
}

# ============================================================================
# Political Nicknames Lookup Table
# ============================================================================
# Maps canonical names to common nicknames/aliases for entity matching.
# These are used in addition to the algorithmically-generated aliases.

POLITICAL_NICKNAMES: Dict[str, Set[str]] = {
    # US Political Figures
    "donald trump": {"the donald", "djt", "45", "potus", "trump", "donald j trump", "president trump"},
    "joe biden": {"biden", "joe", "46", "potus", "president biden", "sleepy joe"},
    "kamala harris": {"harris", "kamala", "vp", "vice president harris"},
    "elizabeth warren": {"liz warren", "pocahontas", "warren", "senator warren"},
    "pete hegseth": {"hegseth", "pete"},
    "marco rubio": {"rubio", "marco", "senator rubio", "little marco"},
    "ted cruz": {"cruz", "ted", "senator cruz", "rafael cruz"},
    "bernie sanders": {"bernie", "sanders", "senator sanders"},
    "nancy pelosi": {"pelosi", "speaker pelosi"},
    "mitch mcconnell": {"mcconnell", "mitch", "senator mcconnell"},
    "kevin mccarthy": {"mccarthy", "speaker mccarthy"},
    "mike johnson": {"johnson", "speaker johnson"},
    "kristi noem": {"noem", "kristi", "governor noem"},
    "pam bondi": {"bondi", "pam", "ag bondi", "attorney general bondi"},
    "tulsi gabbard": {"gabbard", "tulsi"},
    "robert kennedy": {"rfk", "rfk jr", "kennedy", "bobby kennedy"},
    "vivek ramaswamy": {"vivek", "ramaswamy"},
    "ron desantis": {"desantis", "ron", "governor desantis"},
    "gavin newsom": {"newsom", "gavin", "governor newsom"},
    "elon musk": {"musk", "elon"},
    "jd vance": {"vance", "jd", "senator vance"},

    # International Leaders
    "vladimir putin": {"putin", "vladimir"},
    "xi jinping": {"xi", "jinping", "president xi"},
    "benjamin netanyahu": {"netanyahu", "bibi"},
    "emmanuel macron": {"macron", "emmanuel"},
    "olaf scholz": {"scholz", "olaf"},
    "keir starmer": {"starmer", "keir"},
    "justin trudeau": {"trudeau", "justin"},
    "narendra modi": {"modi", "narendra"},
    "pope francis": {"francis", "pope", "the pope"},
    "ali khamenei": {"khamenei", "ayatollah", "supreme leader"},
}

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


def build_aliases(canonical_name: str, entity_type: str = "person") -> Set[str]:
    """
    Build comprehensive alias set from canonical name.

    This generates all possible aliases for an entity including:
    - Normalized forms (lowercase, underscored)
    - Individual name parts
    - Initials for people
    - Title-stripped versions
    - Known nicknames from lookup table

    Args:
        canonical_name: The canonical entity name (e.g., "Donald Trump")
        entity_type: Entity type ("person", "organization", "position")

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

    # Common nicknames from lookup table
    aliases.update(POLITICAL_NICKNAMES.get(name_lower, set()))

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
    series_filters: List[str] = field(default_factory=lambda: [
        # Production series (US politics)
        "KXCABINET",
        "KXPRES",
        "KXSENATE",
        "KXHOUSE",
        # Demo API series (international politics/leaders)
        "KXNEWPOPE",
        "KXXISUCCESSOR",
        "KXNEXTISRAELPM",
        "KXNEXTIRANLEADER",
        "KXLALEADEROUT",
        "KXG7LEADEROUT",
        "KXAFRICALEADEROUT",
        "KXNEXTSPEAKER",
    ])
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

        # spaCy Knowledge Base (for NLP pipeline integration)
        self._kb: Optional[KalshiKnowledgeBase] = None

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
            categories: Optional list of categories to filter (e.g., ["Politics"])
        """
        if trading_client:
            self._trading_client = trading_client

        if categories:
            # Map categories to series filters
            category_series_map = {
                "politics": [
                    # Production series (US politics)
                    "KXCABINET", "KXPRES", "KXSENATE", "KXHOUSE",
                    # Demo API series (international politics/leaders)
                    "KXNEWPOPE", "KXXISUCCESSOR", "KXNEXTISRAELPM",
                    "KXNEXTIRANLEADER", "KXLALEADEROUT", "KXG7LEADEROUT",
                    "KXAFRICALEADEROUT", "KXNEXTSPEAKER",
                ],
                "media_mentions": [],
                "entertainment": [],
                "crypto": [],
                "sports": ["KXNFL", "KXNBA", "KXMLB"],
            }
            series_filters = []
            for cat in categories:
                series_filters.extend(category_series_map.get(cat.lower(), []))
            if series_filters:
                self._config.series_filters = series_filters

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
        """Refresh the entity-market index from Kalshi API."""
        if not self._trading_client:
            logger.warning("[entity_index] No trading client available")
            return

        logger.info("[entity_index] Refreshing entity-market index...")

        new_index: Dict[str, EntityMarketEntry] = {}
        new_canonical_entities: Dict[str, CanonicalEntity] = {}
        new_reverse_index: Dict[str, List[str]] = {}
        new_name_variations: Dict[str, str] = {}
        new_alias_lookup: Dict[str, str] = {}

        try:
            # Fetch events for configured series
            events_processed = 0
            markets_processed = 0
            current_time = time.time()

            for series_filter in self._config.series_filters:
                try:
                    # Get events matching the series
                    events = await self._fetch_events_for_series(series_filter)

                    for event in events:
                        event_ticker = event.get("event_ticker", "")
                        event_title = event.get("title", "")
                        markets = event.get("markets", [])

                        events_processed += 1

                        for market in markets:
                            market_ticker = market.get("ticker", "")
                            yes_sub_title = market.get("yes_sub_title", "")

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

                            # Build comprehensive aliases
                            aliases = build_aliases(yes_sub_title, entity_type)

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

                except Exception as e:
                    logger.error(
                        f"[entity_index] Error fetching series {series_filter}: {e}"
                    )
                    continue

            # Update indices atomically
            self._index = new_index
            self._canonical_entities = new_canonical_entities
            self._reverse_index = new_reverse_index
            self._name_variations = new_name_variations
            self._alias_lookup = new_alias_lookup
            self._last_refresh_at = time.time()
            self._refresh_count += 1
            self._total_aliases = len(new_alias_lookup)

            self._total_entities = len(new_index)
            self._total_markets = sum(len(e.markets) for e in new_index.values())

            logger.info(
                f"[entity_index] Refresh complete: {self._total_entities} entities, "
                f"{self._total_markets} market mappings, {self._total_aliases} aliases "
                f"(from {events_processed} events, {markets_processed} markets)"
            )

            # Synchronize Knowledge Base for NLP pipeline
            await self._sync_knowledge_base()

            # Broadcast entity index snapshot to connected clients
            if self._websocket_manager:
                try:
                    await self._websocket_manager.broadcast_entity_index_snapshot()
                except Exception as broadcast_error:
                    logger.warning(f"[entity_index] Broadcast error: {broadcast_error}")

        except Exception as e:
            logger.error(f"[entity_index] Refresh failed: {e}")

    async def _fetch_events_for_series(self, series_filter: str) -> List[Dict[str, Any]]:
        """
        Fetch events matching a series filter.

        Args:
            series_filter: Series prefix to filter (e.g., "KXCABINET")

        Returns:
            List of event dicts with markets
        """
        events = []

        try:
            # Use trading client to get events
            logger.info(f"[entity_index] Fetching events for series: {series_filter}")
            raw_events = await self._trading_client.get_events(
                series_ticker=series_filter,
                status="open",
            )
            logger.info(f"[entity_index] Series {series_filter}: found {len(raw_events or [])} raw events")

            for event in raw_events or []:
                event_ticker = event.get("event_ticker", "")
                if not event_ticker:
                    continue

                # Fetch full event with markets
                full_event = await self._trading_client.get_event(event_ticker)
                if full_event:
                    markets_count = len(full_event.get("markets", []))
                    logger.debug(f"[entity_index] Event {event_ticker}: {markets_count} markets")
                    events.append(full_event)

        except Exception as e:
            logger.error(f"[entity_index] Error fetching events for {series_filter}: {e}", exc_info=True)

        return events

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

        This method:
        1. Creates KB if not exists
        2. Clears existing KB data
        3. Repopulates from current market index
        4. Sets global KB instance for NLP pipeline access
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

            # Create KB if needed
            if self._kb is None:
                self._kb = KalshiKnowledgeBase(vocab, entity_vector_length=64)
            else:
                # Clear existing KB for fresh population
                self._kb.clear()

            # Build market data for KB population
            markets_for_kb = self._get_markets_for_kb()

            # Populate KB
            self._kb.populate_from_markets(markets_for_kb, alias_builder=build_aliases)

            # Set global KB instance
            set_kalshi_knowledge_base(self._kb)

            logger.info(
                f"[entity_index] KB synced: {self._kb.get_entity_count()} entities, "
                f"{self._kb.get_alias_count()} aliases"
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
            "refresh_count": self._refresh_count,
            "last_refresh_at": self._last_refresh_at,
            "series_filters": self._config.series_filters,
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
