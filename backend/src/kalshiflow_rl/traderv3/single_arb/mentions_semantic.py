"""
Semantic analysis for mentions markets using WordNet.

Provides:
- Synonym detection (what DOESN'T count per settlement rules)
- Semantic expansion for search queries
- Co-occurrence prediction based on semantic relationships
- Settlement rule interpretation (lemma forms, word relationships)

Uses the Open English WordNet (OEWN) 2025 edition.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

logger = logging.getLogger("kalshiflow_rl.traderv3.single_arb.mentions_semantic")

# Module-level WordNet instance (lazy loaded)
_wordnet = None
_wordnet_loaded = False


def _ensure_wordnet():
    """Ensure WordNet is downloaded and loaded."""
    global _wordnet, _wordnet_loaded

    if _wordnet_loaded:
        return _wordnet

    try:
        import wn

        # Check if OEWN is already downloaded
        try:
            _wordnet = wn.Wordnet("oewn:2024")
            _wordnet_loaded = True
            logger.info("WordNet (OEWN 2024) loaded")
            return _wordnet
        except wn.Error:
            pass

        # Try downloading
        logger.info("Downloading Open English WordNet 2024...")
        wn.download("oewn:2024")
        _wordnet = wn.Wordnet("oewn:2024")
        _wordnet_loaded = True
        logger.info("WordNet (OEWN 2024) downloaded and loaded")
        return _wordnet

    except ImportError:
        logger.warning("wn package not installed - semantic features disabled")
        _wordnet_loaded = True  # Mark as attempted
        return None
    except Exception as e:
        logger.error(f"Failed to load WordNet: {e}")
        _wordnet_loaded = True
        return None


@dataclass
class SemanticContext:
    """Semantic analysis result for a term."""

    term: str
    # Direct synonyms (words that mean the same - these DON'T count per rules)
    synonyms: List[str] = field(default_factory=list)
    # Hypernyms (more general terms)
    hypernyms: List[str] = field(default_factory=list)
    # Hyponyms (more specific terms)
    hyponyms: List[str] = field(default_factory=list)
    # Related terms (co-occurrence hints)
    related: List[str] = field(default_factory=list)
    # All morphological forms of the lemma
    lemma_forms: List[str] = field(default_factory=list)
    # Definitions for disambiguation
    definitions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "term": self.term,
            "synonyms": self.synonyms,
            "hypernyms": self.hypernyms,
            "hyponyms": self.hyponyms,
            "related": self.related,
            "lemma_forms": self.lemma_forms,
            "definitions": self.definitions,
        }


def get_semantic_context(term: str, max_items: int = 10) -> SemanticContext:
    """Get semantic context for a term using WordNet.

    Args:
        term: The word to analyze
        max_items: Maximum items per category

    Returns:
        SemanticContext with synonyms, hypernyms, hyponyms, related terms
    """
    wordnet = _ensure_wordnet()
    ctx = SemanticContext(term=term)

    if wordnet is None:
        return ctx

    try:
        # Get all synsets for the term
        synsets = wordnet.synsets(term)
        if not synsets:
            # Try lowercase
            synsets = wordnet.synsets(term.lower())

        if not synsets:
            return ctx

        synonyms_set: Set[str] = set()
        hypernyms_set: Set[str] = set()
        hyponyms_set: Set[str] = set()
        related_set: Set[str] = set()
        definitions: List[str] = []

        for synset in synsets[:5]:  # Limit synsets to check
            # Get definition
            defn = synset.definition()
            if defn:
                definitions.append(defn)

            # Get synonyms (other words in the same synset)
            for word in synset.words():
                lemma = word.lemma()
                if lemma.lower() != term.lower():
                    synonyms_set.add(lemma)

            # Get hypernyms (more general)
            hypers = synset.hypernyms()
            for hyper in (hypers[:3] if hypers else []):
                for word in hyper.words()[:2]:
                    hypernyms_set.add(word.lemma())

            # Get hyponyms (more specific)
            hypos = synset.hyponyms()
            for hypo in (hypos[:3] if hypos else []):
                for word in hypo.words()[:2]:
                    hyponyms_set.add(word.lemma())

            # Get related via 'also' relations
            try:
                also_rels = synset.also()
                for rel in (also_rels[:2] if also_rels else []):
                    for word in rel.words()[:2]:
                        related_set.add(word.lemma())
            except AttributeError:
                # Some versions may not have 'also' method
                pass

        # Get lemma forms (morphological variants)
        words = wordnet.words(term)
        if not words:
            words = wordnet.words(term.lower())
        if words:
            for form in words[0].forms()[:5]:
                if form.lower() != term.lower():
                    ctx.lemma_forms.append(form)

        ctx.synonyms = list(synonyms_set)[:max_items]
        ctx.hypernyms = list(hypernyms_set)[:max_items]
        ctx.hyponyms = list(hyponyms_set)[:max_items]
        ctx.related = list(related_set)[:max_items]
        ctx.definitions = definitions[:3]

    except Exception as e:
        logger.warning(f"Semantic analysis failed for '{term}': {e}")

    return ctx


def get_prohibited_forms(term: str) -> List[str]:
    """Get forms that should NOT count as mentions per strict rules.

    Settlement rules typically require the EXACT word. Synonyms don't count.
    This returns synonyms and related terms that might be confused for the target.

    Args:
        term: The target mention term

    Returns:
        List of words that should NOT count as mentions
    """
    ctx = get_semantic_context(term)
    prohibited = set()

    # Synonyms definitely don't count
    prohibited.update(ctx.synonyms)

    # Hypernyms (more general) don't count
    # e.g., "tariff" doesn't match "tax" (hypernym)
    prohibited.update(ctx.hypernyms)

    # Hyponyms (more specific) probably don't count either
    # e.g., "tariff" doesn't match "protective_tariff"
    prohibited.update(ctx.hyponyms)

    # Remove the original term
    prohibited.discard(term.lower())
    prohibited.discard(term)

    return list(prohibited)


def get_accepted_forms(term: str) -> List[str]:
    """Get morphological forms that likely DO count as the same mention.

    These are different forms of the same lemma (plurals, verb conjugations).
    Note: Settlement rules vary - some accept all forms, some only exact matches.

    Args:
        term: The target mention term

    Returns:
        List of acceptable forms (including the original)
    """
    wordnet = _ensure_wordnet()
    forms = {term, term.lower(), term.capitalize(), term.upper()}

    if wordnet is None:
        return list(forms)

    try:
        words = wordnet.words(term)
        if not words:
            words = wordnet.words(term.lower())
        if words:
            for form in words[0].forms():
                forms.add(form)
                forms.add(form.lower())
                forms.add(form.capitalize())
    except Exception as e:
        logger.debug(f"Could not get forms for '{term}': {e}")

    return list(forms)


def expand_search_queries(term: str) -> List[str]:
    """Expand a term into related search queries for corpus sourcing.

    This helps find relevant frequency data across related concepts.

    Args:
        term: The target mention term

    Returns:
        List of expanded search terms/phrases
    """
    ctx = get_semantic_context(term)
    queries = [term]

    # Add related terms for broader search
    queries.extend(ctx.related[:3])

    # Add hypernyms for category-level searches
    for hyper in ctx.hypernyms[:2]:
        queries.append(f"{term} {hyper}")

    return queries


def predict_cooccurrence(term_a: str, term_b: str) -> float:
    """Predict co-occurrence likelihood based on semantic similarity.

    Higher values indicate terms are more likely to co-occur.

    Args:
        term_a: First term
        term_b: Second term

    Returns:
        Co-occurrence score (0.0 to 1.0)
    """
    wordnet = _ensure_wordnet()

    if wordnet is None:
        return 0.5  # Neutral default

    try:
        synsets_a = wordnet.synsets(term_a.lower())
        synsets_b = wordnet.synsets(term_b.lower())

        if not synsets_a or not synsets_b:
            return 0.3  # Low confidence if not in WordNet

        # Check for direct relationships
        for sa in synsets_a[:3]:
            for sb in synsets_b[:3]:
                # Same synset = very high co-occurrence
                if sa.id() == sb.id():
                    return 0.95

                # Hypernym/hyponym relationship
                if sb in sa.hypernyms() or sa in sb.hypernyms():
                    return 0.8

                if sb in sa.hyponyms() or sa in sb.hyponyms():
                    return 0.75

                # Check for shared hypernym (sibling concepts)
                hyper_a = set(h.id() for h in sa.hypernyms())
                hyper_b = set(h.id() for h in sb.hypernyms())
                if hyper_a & hyper_b:
                    return 0.65

        return 0.4  # No direct relationship found

    except Exception as e:
        logger.debug(f"Co-occurrence prediction failed: {e}")
        return 0.5


def interpret_settlement_rule(rule_text: str, target_term: str) -> Dict:
    """Use semantic analysis to interpret settlement rule strictness.

    Analyzes the rule text to determine:
    - Whether synonyms count
    - Whether plurals count
    - Whether different cases count

    Args:
        rule_text: The settlement rules text
        target_term: The term being counted

    Returns:
        Dict with interpretation guidance
    """
    rule_lower = rule_text.lower()

    interpretation = {
        "target_term": target_term,
        "synonyms_count": False,  # Default: synonyms don't count
        "plurals_count": True,  # Default: plurals likely count
        "case_sensitive": False,  # Default: case insensitive
        "exact_phrase_required": False,
        "notes": [],
    }

    # Check for strict language
    strict_phrases = [
        "exact word",
        "exactly",
        "only the word",
        "specifically",
        "must say",
        "does not count",
        "doesn't count",
        "do not count",
        "won't count",
        "will not count",
    ]
    for phrase in strict_phrases:
        if phrase in rule_lower:
            interpretation["synonyms_count"] = False
            interpretation["notes"].append(f"Strict language detected: '{phrase}'")

    # Check for permissive language
    permissive_phrases = [
        "any form",
        "variations",
        "includes",
        "such as",
        "or similar",
    ]
    for phrase in permissive_phrases:
        if phrase in rule_lower:
            interpretation["plurals_count"] = True
            interpretation["notes"].append(f"Permissive language detected: '{phrase}'")

    # Check if rule explicitly mentions synonyms
    if "synonym" in rule_lower:
        if "not" in rule_lower or "don't" in rule_lower or "doesn't" in rule_lower:
            interpretation["synonyms_count"] = False
            interpretation["notes"].append("Rule explicitly excludes synonyms")
        else:
            interpretation["synonyms_count"] = True
            interpretation["notes"].append("Rule may include synonyms")

    # Get semantic context for additional guidance
    ctx = get_semantic_context(target_term)
    interpretation["prohibited_synonyms"] = ctx.synonyms[:5]
    interpretation["accepted_forms"] = get_accepted_forms(target_term)

    return interpretation


def is_wordnet_available() -> bool:
    """Check if WordNet is available for use."""
    wordnet = _ensure_wordnet()
    return wordnet is not None
