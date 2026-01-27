"""
Custom spaCy Extensions for Entity Processing.

Registers custom attributes on Doc and Span objects for:
- Sentiment scores per entity
- Market type information
- Confidence scores
- Entity linking metadata

These extensions enable the unified pipeline to carry all entity
information through processing stages.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from spacy.tokens import Doc, Span, Token

logger = logging.getLogger("kalshiflow_rl.traderv3.nlp.extensions")

# Track if extensions have been registered
_extensions_registered = False


@dataclass
class EntityExtensions:
    """
    Container for entity extension attributes.

    These are the custom attributes added to spaCy Span objects
    for entities processed by our pipeline.
    """

    # Sentiment score (-100 to +100)
    sentiment: int = 0

    # Market type (OUT, WIN, CONFIRM, etc.)
    market_type: str = ""

    # Entity type (MARKET_ENTITY, PERSON, ORG, GPE, EVENT)
    entity_type: str = ""

    # Disambiguation confidence (0.0 to 1.0)
    kb_confidence: float = 0.0

    # Is this a market-linked entity?
    is_market_entity: bool = False

    # Source metadata
    source_alias: str = ""  # The alias that matched


def register_custom_extensions() -> bool:
    """
    Register custom extensions on spaCy Doc and Span objects.

    These extensions are used throughout the NLP pipeline to carry
    entity-specific information.

    Returns:
        True if extensions were registered, False if already registered
    """
    global _extensions_registered

    if _extensions_registered:
        return False

    # =========================================================================
    # Span Extensions (for entities)
    # =========================================================================

    # Sentiment score for this entity (-100 to +100)
    if not Span.has_extension("sentiment"):
        Span.set_extension("sentiment", default=0)

    # Market type for market-linked entities
    if not Span.has_extension("market_type"):
        Span.set_extension("market_type", default="")

    # Disambiguation confidence from EntityLinker
    if not Span.has_extension("kb_confidence"):
        Span.set_extension("kb_confidence", default=0.0)

    # Is this a market-linked entity?
    if not Span.has_extension("is_market_entity"):
        Span.set_extension("is_market_entity", default=False)

    # The alias text that matched (for debugging)
    if not Span.has_extension("source_alias"):
        Span.set_extension("source_alias", default="")

    # Market ticker if linked
    if not Span.has_extension("market_ticker"):
        Span.set_extension("market_ticker", default="")

    # Event ticker if linked
    if not Span.has_extension("event_ticker"):
        Span.set_extension("event_ticker", default="")

    # Canonical name from KB
    if not Span.has_extension("canonical_name"):
        Span.set_extension("canonical_name", default="")

    # =========================================================================
    # Doc Extensions (for document-level info)
    # =========================================================================

    # Overall document sentiment (average of entity sentiments)
    if not Doc.has_extension("aggregate_sentiment"):
        Doc.set_extension("aggregate_sentiment", default=0)

    # Number of market entities found
    if not Doc.has_extension("market_entity_count"):
        Doc.set_extension("market_entity_count", default=0)

    # Number of related (non-market) entities found
    if not Doc.has_extension("related_entity_count"):
        Doc.set_extension("related_entity_count", default=0)

    # Processing metadata
    if not Doc.has_extension("nlp_pipeline_version"):
        Doc.set_extension("nlp_pipeline_version", default="v2_kb")

    # Market entities list (for convenience)
    if not Doc.has_extension("market_entities"):
        Doc.set_extension("market_entities", getter=_get_market_entities)

    # Related entities list (for convenience)
    if not Doc.has_extension("related_entities"):
        Doc.set_extension("related_entities", getter=_get_related_entities)

    _extensions_registered = True
    logger.info("[extensions] Registered custom spaCy extensions")
    return True


def _get_market_entities(doc: Doc) -> list:
    """
    Get all MARKET_ENTITY spans from a Doc.

    This is a getter function for the Doc._.market_entities extension.
    """
    return [
        ent for ent in doc.ents
        if ent.label_ == "MARKET_ENTITY" or ent._.is_market_entity
    ]


def _get_related_entities(doc: Doc) -> list:
    """
    Get all non-market entity spans from a Doc.

    This is a getter function for the Doc._.related_entities extension.
    Returns entities with labels: PERSON, ORG, GPE, EVENT
    """
    related_labels = {"PERSON", "ORG", "GPE", "EVENT"}
    return [
        ent for ent in doc.ents
        if ent.label_ in related_labels and not ent._.is_market_entity
    ]


def ensure_extensions_registered() -> None:
    """
    Ensure extensions are registered.

    Call this at the start of any function that uses custom extensions.
    """
    if not _extensions_registered:
        register_custom_extensions()


def get_entity_extensions(span: Span) -> EntityExtensions:
    """
    Get all custom extensions for an entity span as a dataclass.

    Args:
        span: A spaCy Span object (entity)

    Returns:
        EntityExtensions with all custom attribute values
    """
    ensure_extensions_registered()

    return EntityExtensions(
        sentiment=span._.sentiment,
        market_type=span._.market_type,
        entity_type=span.label_,
        kb_confidence=span._.kb_confidence,
        is_market_entity=span._.is_market_entity,
        source_alias=span._.source_alias,
    )


def set_entity_extensions(
    span: Span,
    sentiment: Optional[int] = None,
    market_type: Optional[str] = None,
    kb_confidence: Optional[float] = None,
    is_market_entity: Optional[bool] = None,
    source_alias: Optional[str] = None,
    market_ticker: Optional[str] = None,
    event_ticker: Optional[str] = None,
    canonical_name: Optional[str] = None,
) -> None:
    """
    Set custom extensions on an entity span.

    Only sets attributes that are provided (not None).

    Args:
        span: A spaCy Span object (entity)
        sentiment: Sentiment score (-100 to +100)
        market_type: Market type (OUT, WIN, CONFIRM, etc.)
        kb_confidence: Disambiguation confidence
        is_market_entity: Whether this is a market-linked entity
        source_alias: The alias text that matched
        market_ticker: Market ticker if linked
        event_ticker: Event ticker if linked
        canonical_name: Canonical name from KB
    """
    ensure_extensions_registered()

    if sentiment is not None:
        span._.sentiment = sentiment
    if market_type is not None:
        span._.market_type = market_type
    if kb_confidence is not None:
        span._.kb_confidence = kb_confidence
    if is_market_entity is not None:
        span._.is_market_entity = is_market_entity
    if source_alias is not None:
        span._.source_alias = source_alias
    if market_ticker is not None:
        span._.market_ticker = market_ticker
    if event_ticker is not None:
        span._.event_ticker = event_ticker
    if canonical_name is not None:
        span._.canonical_name = canonical_name


def update_doc_stats(doc: Doc) -> None:
    """
    Update document-level statistics after entity processing.

    Call this after all entities have been processed to update
    aggregate stats.

    Args:
        doc: A spaCy Doc object
    """
    ensure_extensions_registered()

    market_ents = doc._.market_entities
    related_ents = doc._.related_entities

    doc._.market_entity_count = len(market_ents)
    doc._.related_entity_count = len(related_ents)

    # Calculate aggregate sentiment
    all_sentiments = [ent._.sentiment for ent in doc.ents if ent._.sentiment != 0]
    if all_sentiments:
        doc._.aggregate_sentiment = sum(all_sentiments) // len(all_sentiments)
