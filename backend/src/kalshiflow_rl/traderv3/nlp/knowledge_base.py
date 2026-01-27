"""
Knowledge Base for Kalshi Market Entities.

Provides a spaCy-compatible Knowledge Base (KB) wrapper around InMemoryLookupKB
that maps entities to Kalshi markets. This enables proper entity disambiguation
using prior probabilities and entity vectors.

Key Features:
- Entities stored with vectors for semantic similarity
- Aliases mapped with prior probabilities for disambiguation
- Metadata includes market_ticker, market_type, event_ticker
- Integrates with spaCy EntityLinker for entity resolution
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from spacy.vocab import Vocab
    from spacy.kb import InMemoryLookupKB, Candidate

logger = logging.getLogger("kalshiflow_rl.traderv3.nlp.knowledge_base")


@dataclass
class EntityMetadata:
    """
    Metadata for a KB entity.

    Stores market-related information for downstream processing
    after entity linking.
    """

    entity_id: str  # Normalized: "pam_bondi"
    canonical_name: str  # Display: "Pam Bondi"
    market_ticker: str  # "KXBONDIOUT-25FEB01"
    market_type: str  # "OUT", "WIN", "CONFIRM", etc.
    event_ticker: str  # "KXBONDIOUT"
    entity_type: str = "person"  # "person", "organization", "position"
    volume_24h: Optional[float] = None  # Trading volume for prior calculation


@dataclass
class MarketData:
    """
    Market data structure for KB population.

    Simplified view of Kalshi market data needed for KB building.
    """

    ticker: str
    event_ticker: str
    yes_sub_title: str
    market_type: str
    volume_24h: Optional[float] = None


def normalize_entity_id(name: str) -> str:
    """
    Normalize an entity name to a consistent ID format.

    Args:
        name: Entity name (e.g., "Pam Bondi", "Pete Hegseth")

    Returns:
        Normalized ID (e.g., "pam_bondi", "pete_hegseth")
    """
    import re

    normalized = name.lower().strip()
    normalized = re.sub(r"[^\w\s]", "", normalized)  # Remove punctuation
    normalized = re.sub(r"\s+", "_", normalized)  # Spaces to underscores
    return normalized


class KalshiKnowledgeBase:
    """
    Knowledge Base mapping entities to Kalshi markets.

    Wraps spaCy's InMemoryLookupKB with market-specific functionality:
    - Entities with trading volume-based frequency (for prior calculation)
    - Comprehensive alias sets (nicknames, initials, variations)
    - Metadata for market-to-entity resolution

    Each entity in the KB represents a market-able entity with:
    - entity_id: Normalized entity identifier (e.g., "pam_bondi")
    - vector: Entity embedding for semantic similarity
    - metadata: market_ticker, market_type, event_ticker, canonical_name
    """

    def __init__(
        self,
        vocab: "Vocab",
        entity_vector_length: int = 64,
    ):
        """
        Initialize the Knowledge Base.

        Args:
            vocab: spaCy Vocab object (shared with nlp pipeline)
            entity_vector_length: Dimension of entity vectors
        """
        from spacy.kb import InMemoryLookupKB

        self._kb = InMemoryLookupKB(vocab, entity_vector_length)
        self._entity_metadata: Dict[str, EntityMetadata] = {}
        self._alias_to_entities: Dict[str, List[str]] = {}  # alias -> list of entity_ids
        self._entity_vector_length = entity_vector_length
        self._vocab = vocab

    @property
    def entity_vector_length(self) -> int:
        """Get the entity vector length."""
        return self._entity_vector_length

    @property
    def entity_metadata(self) -> Dict[str, EntityMetadata]:
        """Get all entity metadata."""
        return self._entity_metadata

    def populate_from_markets(
        self,
        markets: List[MarketData],
        alias_builder: Optional[Callable[[str, str], set]] = None,
    ) -> None:
        """
        Populate KB from Kalshi market data.

        For each market:
        1. Add entity with frequency (based on trading volume)
        2. Add all aliases with prior probabilities
        3. Store metadata for downstream processing

        Args:
            markets: List of MarketData objects to populate from
            alias_builder: Optional function to build aliases (name, type) -> set
        """
        from ..services.entity_market_index import build_aliases, detect_entity_type

        # Use provided alias builder or default
        if alias_builder is None:
            alias_builder = build_aliases

        entities_added = 0
        aliases_added = 0

        for market in markets:
            entity_id = normalize_entity_id(market.yes_sub_title)

            # Generate entity vector (deterministic hash-based)
            vector = self._generate_entity_vector(market.yes_sub_title)

            # Calculate frequency from trading volume
            freq = max(int(market.volume_24h or 100), 1)

            # Add entity if not exists
            if entity_id not in self._entity_metadata:
                try:
                    self._kb.add_entity(
                        entity=entity_id,
                        freq=freq,
                        entity_vector=vector,
                    )
                    entities_added += 1
                except Exception as e:
                    logger.warning(f"[kb] Failed to add entity {entity_id}: {e}")
                    continue

            # Detect entity type
            entity_type = detect_entity_type(
                market.yes_sub_title,
                "",  # event_title
                market.ticker,
            )

            # Store/update metadata
            self._entity_metadata[entity_id] = EntityMetadata(
                entity_id=entity_id,
                canonical_name=market.yes_sub_title,
                market_ticker=market.ticker,
                market_type=market.market_type,
                event_ticker=market.event_ticker,
                entity_type=entity_type,
                volume_24h=market.volume_24h,
            )

            # Build and add aliases
            aliases = alias_builder(market.yes_sub_title, entity_type)
            for alias in aliases:
                alias_lower = alias.lower()
                prior = self._compute_alias_prior(alias_lower, market.yes_sub_title)

                # Track alias -> entities mapping
                if alias_lower not in self._alias_to_entities:
                    self._alias_to_entities[alias_lower] = []
                if entity_id not in self._alias_to_entities[alias_lower]:
                    self._alias_to_entities[alias_lower].append(entity_id)

                try:
                    # Check if alias already exists for this entity
                    existing_entities = self._alias_to_entities.get(alias_lower, [])
                    if len(existing_entities) == 1 and existing_entities[0] == entity_id:
                        # Already added, skip
                        continue

                    # Add alias to KB
                    self._kb.add_alias(
                        alias=alias_lower,
                        entities=[entity_id],
                        probabilities=[prior],
                    )
                    aliases_added += 1
                except Exception as e:
                    # Alias may already exist, try to update
                    logger.debug(f"[kb] Alias '{alias_lower}' already exists: {e}")

        logger.info(
            f"[kb] Populated KB: {entities_added} entities, {aliases_added} aliases "
            f"from {len(markets)} markets"
        )

    def _generate_entity_vector(self, name: str) -> np.ndarray:
        """
        Generate a deterministic entity vector from name.

        Uses hash-based approach for consistency without requiring
        external embeddings. This provides a simple but effective
        vector for the KB.

        Args:
            name: Entity name

        Returns:
            numpy array of shape (entity_vector_length,)
        """
        # Use SHA256 hash for deterministic vector
        hash_bytes = hashlib.sha256(name.lower().encode()).digest()

        # Convert to float array
        vector = np.frombuffer(hash_bytes[:self._entity_vector_length], dtype=np.uint8)
        vector = vector.astype(np.float32) / 255.0  # Normalize to [0, 1]

        # Pad if needed
        if len(vector) < self._entity_vector_length:
            padding = np.zeros(self._entity_vector_length - len(vector), dtype=np.float32)
            vector = np.concatenate([vector, padding])

        return vector[:self._entity_vector_length]

    def _compute_alias_prior(self, alias: str, canonical_name: str) -> float:
        """
        Compute prior probability for an alias.

        Higher prior for:
        - Exact matches (full name)
        - Longer aliases (more specific)

        Lower prior for:
        - Very short aliases (ambiguous)
        - Single-word partial matches

        Args:
            alias: The alias string
            canonical_name: The canonical entity name

        Returns:
            Prior probability (0.0 to 1.0)
        """
        canonical_lower = canonical_name.lower()

        # Exact match gets highest prior
        if alias == canonical_lower:
            return 0.95

        # Full last name match
        name_parts = canonical_lower.split()
        if len(name_parts) >= 2 and alias == name_parts[-1]:
            return 0.85

        # Length-based heuristic (longer = more specific)
        if len(alias) >= 10:
            return 0.80
        if len(alias) >= 6:
            return 0.70
        if len(alias) >= 4:
            return 0.60

        # Short aliases are more ambiguous
        return 0.40

    def get_candidates(
        self, mention: str
    ) -> Iterator["Candidate"]:
        """
        Get candidate entities for a mention.

        This method is used by spaCy's EntityLinker to get KB candidates.

        Args:
            mention: The text mention to look up

        Yields:
            Candidate objects from the KB
        """
        mention_lower = mention.lower().strip()
        return self._kb.get_alias_candidates(mention_lower)

    def get_entity_metadata(self, entity_id: str) -> Optional[EntityMetadata]:
        """
        Get metadata for an entity.

        Args:
            entity_id: The entity ID

        Returns:
            EntityMetadata if found, None otherwise
        """
        return self._entity_metadata.get(entity_id)

    def get_aliases(self, entity_id: str) -> List[str]:
        """
        Get all aliases for an entity.

        Args:
            entity_id: The entity ID

        Returns:
            List of aliases
        """
        aliases = []
        for alias, entities in self._alias_to_entities.items():
            if entity_id in entities:
                aliases.append(alias)
        return aliases

    def contains_entity(self, entity_id: str) -> bool:
        """Check if entity exists in KB."""
        return entity_id in self._entity_metadata

    def get_entity_count(self) -> int:
        """Get total number of entities in KB."""
        return len(self._entity_metadata)

    def get_alias_count(self) -> int:
        """Get total number of unique aliases."""
        return len(self._alias_to_entities)

    def get_kb(self) -> "InMemoryLookupKB":
        """Get the underlying spaCy KB object."""
        return self._kb

    def clear(self) -> None:
        """Clear the KB."""
        # Recreate KB (InMemoryLookupKB doesn't have a clear method)
        from spacy.kb import InMemoryLookupKB

        self._kb = InMemoryLookupKB(self._vocab, self._entity_vector_length)
        self._entity_metadata.clear()
        self._alias_to_entities.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize KB state for debugging/logging."""
        return {
            "entity_count": self.get_entity_count(),
            "alias_count": self.get_alias_count(),
            "entity_vector_length": self._entity_vector_length,
            "entities": list(self._entity_metadata.keys())[:10],  # Sample
        }


# ============================================================================
# Global Singleton
# ============================================================================

_global_kb: Optional[KalshiKnowledgeBase] = None


def get_kalshi_knowledge_base() -> Optional[KalshiKnowledgeBase]:
    """Get the global KalshiKnowledgeBase instance."""
    return _global_kb


def set_kalshi_knowledge_base(kb: KalshiKnowledgeBase) -> None:
    """Set the global KalshiKnowledgeBase instance."""
    global _global_kb
    _global_kb = kb
    logger.info(f"[kb] Set global KalshiKnowledgeBase: {kb.get_entity_count()} entities")
