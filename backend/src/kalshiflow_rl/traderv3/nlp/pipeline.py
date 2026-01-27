"""
spaCy Pipeline Factory for Entity Extraction and Sentiment Analysis.

Creates a unified NLP pipeline that combines:
1. EntityRuler - Pattern-based matching for known market entities
2. LLM NER - General NER to discover all entities (for second-hand signals)
3. EntityLinker - Links market entities to Knowledge Base
4. LLM Sentiment - Batched sentiment scoring for all entities

Pipeline Flow:
    Reddit Post
        ↓
    [sentencizer] → [EntityRuler: MARKET_ENTITY] → [spaCy NER: general]
        ↓
    [EntityLinker: link MARKET_ENTITY → KB]
        ↓
    [LLM Sentiment: score ALL entities in one batched call]
        ↓
    Output: doc.ents with kb_id_, sentiment, entity_type
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import spacy
from spacy.language import Language
from spacy.vocab import Vocab

if TYPE_CHECKING:
    from .knowledge_base import KalshiKnowledgeBase

logger = logging.getLogger("kalshiflow_rl.traderv3.nlp.pipeline")

# Global shared vocab (for consistency across pipelines)
_shared_vocab: Optional["Vocab"] = None


def get_shared_vocab() -> Optional["Vocab"]:
    """Get the shared spaCy Vocab instance."""
    return _shared_vocab


def set_shared_vocab(vocab: "Vocab") -> None:
    """Set the shared spaCy Vocab instance."""
    global _shared_vocab
    _shared_vocab = vocab
    logger.info("[pipeline] Set shared spaCy Vocab")


def create_hybrid_entity_pipeline(
    kb: "KalshiKnowledgeBase",
    model_name: str = "en_core_web_md",
    include_llm_ner: bool = False,
    include_sentiment: bool = True,
    sentiment_model: str = "gpt-4o-mini",
) -> "Language":
    """
    Create hybrid spaCy pipeline with KB-backed entity extraction.

    This pipeline combines:
    1. EntityRuler - Matches known market entities from KB
    2. spaCy NER - Catches general entities (PERSON, ORG, GPE, etc.)
    3. Custom components for entity linking and sentiment

    Args:
        kb: KalshiKnowledgeBase with market entities
        model_name: spaCy model to use as base
        include_llm_ner: Whether to include LLM-based NER (expensive)
        include_sentiment: Whether to include sentiment component
        sentiment_model: OpenAI model for sentiment analysis

    Returns:
        Configured spaCy Language pipeline
    """
    import spacy
    from .extensions import register_custom_extensions

    # Register custom extensions first
    register_custom_extensions()

    # Load base model
    try:
        nlp = spacy.load(model_name)
        logger.info(f"[pipeline] Loaded spaCy model: {model_name}")
    except OSError:
        # Fall back to blank English model
        logger.warning(f"[pipeline] Model {model_name} not found, using blank 'en'")
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")

    # Store shared vocab
    set_shared_vocab(nlp.vocab)

    # 1. Add EntityRuler for KNOWN market entities (runs first, high priority)
    # This MUST run before the built-in NER to catch market entities
    # overwrite_ents=True ensures EntityRuler patterns take precedence
    if "entity_ruler" not in nlp.pipe_names:
        ruler = nlp.add_pipe(
            "entity_ruler",
            before="ner" if "ner" in nlp.pipe_names else None,
            config={"overwrite_ents": True},
        )
        patterns = build_patterns_from_kb(kb)
        ruler.add_patterns(patterns)
        logger.info(f"[pipeline] Added EntityRuler with {len(patterns)} patterns")
    else:
        # Update existing ruler
        ruler = nlp.get_pipe("entity_ruler")
        patterns = build_patterns_from_kb(kb)
        ruler.add_patterns(patterns)

    # 2. Add custom EntityLinker component (links MARKET_ENTITY to KB)
    # Set the global KB so the linker can access it
    from .knowledge_base import set_kalshi_knowledge_base
    set_kalshi_knowledge_base(kb)

    if "kb_entity_linker" not in nlp.pipe_names:
        nlp.add_pipe("kb_entity_linker", after="entity_ruler")

    # 3. Add sentiment component if requested
    if include_sentiment and "batched_sentiment" not in nlp.pipe_names:
        nlp.add_pipe(
            "batched_sentiment",
            last=True,
            config={"model": sentiment_model},
        )

    logger.info(f"[pipeline] Created hybrid pipeline: {nlp.pipe_names}")
    return nlp


def build_patterns_from_kb(kb: "KalshiKnowledgeBase") -> List[Dict[str, Any]]:
    """
    Convert KB aliases to EntityRuler patterns.

    Each pattern maps an alias to the MARKET_ENTITY label with
    the entity_id stored in the pattern ID field.

    Patterns use the token-based format with LOWER attribute for
    case-insensitive matching.

    Args:
        kb: KalshiKnowledgeBase with entities and aliases

    Returns:
        List of pattern dicts for EntityRuler
    """
    patterns = []
    seen_patterns = set()

    for entity_id, metadata in kb.entity_metadata.items():
        aliases = kb.get_aliases(entity_id)

        for alias in aliases:
            alias_lower = alias.lower()

            # Skip very short aliases (too many false positives)
            if len(alias_lower) < 3:
                continue

            # Skip duplicates
            if alias_lower in seen_patterns:
                continue
            seen_patterns.add(alias_lower)

            # Build token-based pattern for case-insensitive matching
            # Each token in the alias becomes a pattern element with LOWER attribute
            tokens = alias_lower.split()
            if len(tokens) == 1:
                # Single token pattern
                pattern = [{"LOWER": tokens[0]}]
            else:
                # Multi-token pattern
                pattern = [{"LOWER": token} for token in tokens]

            patterns.append({
                "label": "MARKET_ENTITY",
                "pattern": pattern,
                "id": entity_id,  # Links pattern to KB entity
            })

    logger.info(f"[pipeline] Built {len(patterns)} EntityRuler patterns from KB")
    return patterns


def create_simple_entity_pipeline(
    kb: "KalshiKnowledgeBase",
    model_name: str = "en_core_web_md",
) -> "Language":
    """
    Create a simplified entity pipeline without LLM components.

    This is faster and cheaper than the full hybrid pipeline,
    using only EntityRuler + built-in spaCy NER.

    Args:
        kb: KalshiKnowledgeBase with market entities
        model_name: spaCy model to use as base

    Returns:
        Configured spaCy Language pipeline
    """
    import spacy
    from .extensions import register_custom_extensions

    # Register custom extensions
    register_custom_extensions()

    # Load base model
    try:
        nlp = spacy.load(model_name)
    except OSError:
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")

    # Store shared vocab
    set_shared_vocab(nlp.vocab)

    # Add EntityRuler for market entities
    # overwrite_ents=True ensures EntityRuler patterns take precedence over NER
    if "entity_ruler" not in nlp.pipe_names:
        ruler = nlp.add_pipe(
            "entity_ruler",
            before="ner" if "ner" in nlp.pipe_names else None,
            config={"overwrite_ents": True},
        )
        patterns = build_patterns_from_kb(kb)
        ruler.add_patterns(patterns)

    # Add KB entity linker (uses global KB singleton)
    from .knowledge_base import set_kalshi_knowledge_base
    set_kalshi_knowledge_base(kb)

    if "kb_entity_linker" not in nlp.pipe_names:
        nlp.add_pipe("kb_entity_linker", after="entity_ruler")

    logger.info(f"[pipeline] Created simple pipeline: {nlp.pipe_names}")
    return nlp


# =============================================================================
# Custom Pipeline Components
# =============================================================================


@Language.factory(
    "kb_entity_linker",
    assigns=["span._.is_market_entity", "span._.market_type"],
    requires=["doc.ents"],
    default_config={},
)
def create_kb_entity_linker(nlp, name: str):
    """Factory for KB Entity Linker component.

    Uses the global KalshiKnowledgeBase singleton set via set_kalshi_knowledge_base().
    """
    from .knowledge_base import get_kalshi_knowledge_base
    kb = get_kalshi_knowledge_base()
    return KBEntityLinker(kb)


class KBEntityLinker:
    """
    Custom spaCy component that links MARKET_ENTITY spans to KB.

    For each entity with label MARKET_ENTITY:
    1. Looks up entity_id from the pattern ID
    2. Gets metadata from KB
    3. Sets custom extensions (market_type, market_ticker, etc.)
    """

    def __init__(self, kb: Optional["KalshiKnowledgeBase"]):
        """
        Initialize the linker.

        Args:
            kb: KalshiKnowledgeBase instance
        """
        self._kb = kb

    def __call__(self, doc: "spacy.tokens.Doc") -> "spacy.tokens.Doc":
        """
        Process a doc and link market entities to KB.

        Args:
            doc: spaCy Doc object

        Returns:
            Doc with linked entities
        """
        from .extensions import set_entity_extensions

        if self._kb is None:
            return doc

        for ent in doc.ents:
            if ent.label_ == "MARKET_ENTITY":
                # Get entity_id from pattern ID (stored by EntityRuler)
                entity_id = ent.ent_id_

                if entity_id:
                    metadata = self._kb.get_entity_metadata(entity_id)
                    if metadata:
                        set_entity_extensions(
                            ent,
                            is_market_entity=True,
                            market_type=metadata.market_type,
                            market_ticker=metadata.market_ticker,
                            event_ticker=metadata.event_ticker,
                            canonical_name=metadata.canonical_name,
                            source_alias=ent.text.lower(),
                            kb_confidence=1.0,  # Pattern match = high confidence
                        )

        return doc


@Language.factory(
    "batched_sentiment",
    assigns=["span._.sentiment"],
    requires=["doc.ents"],
    default_config={"model": "gpt-4o-mini", "batch_size": 10},
)
def create_batched_sentiment(nlp, name: str, model: str, batch_size: int = 10):
    """Factory for Batched Sentiment component."""
    return BatchedSentimentComponent(model, batch_size)


class BatchedSentimentComponent:
    """
    Custom spaCy component for batched sentiment analysis.

    Scores sentiment for all entities in a document using a single
    LLM call (or batched calls for large entity sets).

    Note: This is a synchronous wrapper. For async processing,
    use the BatchedSentimentTask directly.
    """

    def __init__(self, model: str, batch_size: int = 10):
        """
        Initialize the sentiment component.

        Args:
            model: OpenAI model name
            batch_size: Max entities per LLM call
        """
        self._model = model
        self._batch_size = batch_size
        self._client = None

    def _get_client(self):
        """Get or create OpenAI client."""
        if self._client is None:
            import os
            from openai import OpenAI
            self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return self._client

    def __call__(self, doc: "spacy.tokens.Doc") -> "spacy.tokens.Doc":
        """
        Process a doc and score sentiment for all entities.

        Args:
            doc: spaCy Doc object

        Returns:
            Doc with sentiment scores
        """
        from .extensions import set_entity_extensions

        # Get entities that need sentiment scoring
        entities = list(doc.ents)
        if not entities:
            return doc

        # Build batched prompt
        entity_texts = []
        for ent in entities:
            # Use canonical name if available, else original text
            name = ent._.canonical_name if ent._.canonical_name else ent.text
            entity_texts.append(name)

        if not entity_texts:
            return doc

        try:
            # Score all entities in one call
            sentiments = self._batch_sentiment(doc.text, entity_texts)

            # Apply sentiments to entities
            for ent, sentiment in zip(entities, sentiments):
                set_entity_extensions(ent, sentiment=sentiment)

        except Exception as e:
            logger.warning(f"[sentiment] Batch sentiment failed: {e}")
            # Set neutral sentiment on failure
            for ent in entities:
                set_entity_extensions(ent, sentiment=0)

        return doc

    def _batch_sentiment(self, context: str, entities: List[str]) -> List[int]:
        """
        Get sentiment for multiple entities in one LLM call.

        Args:
            context: The full text context
            entities: List of entity names

        Returns:
            List of sentiment scores (-100 to +100)
        """
        client = self._get_client()

        # Build prompt
        entity_list = "\n".join(f"{i+1}. {e}" for i, e in enumerate(entities))
        prompt = f"""Analyze the sentiment toward each entity in this text.

Text: "{context}"

Entities:
{entity_list}

For each entity, rate the sentiment from -100 (extremely negative) to +100 (extremely positive).
Consider: Is the news good or bad for this entity? Does it help or hurt their reputation/position?

Respond with ONLY a comma-separated list of numbers in order, like: 50,-30,0,75
"""

        try:
            response = client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0,
            )

            result = response.choices[0].message.content.strip()

            # Parse comma-separated numbers
            sentiments = []
            for val in result.split(","):
                try:
                    score = int(float(val.strip()))
                    score = max(-100, min(100, score))
                    sentiments.append(score)
                except ValueError:
                    sentiments.append(0)

            # Pad if needed
            while len(sentiments) < len(entities):
                sentiments.append(0)

            return sentiments[:len(entities)]

        except Exception as e:
            logger.warning(f"[sentiment] LLM call failed: {e}")
            return [0] * len(entities)


