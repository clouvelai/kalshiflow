"""
Tier 1: Deterministic text + entity matching.

Three-step pipeline:
  1. Feature extraction (normalize text, extract entities/numbers)
  2. Event matching (jaccard + entity overlap + fuzzy ratio)
  3. Market matching (within matched event pairs)

Handles 60-80% of pairs without any LLM call.
"""

import logging
import re
import string
from collections import defaultdict
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from ..fetchers.models import MatchCandidate, NormalizedEvent, NormalizedMarket

logger = logging.getLogger("kalshiflow_rl.pair_index.matchers.text_matcher")

# ── Stop words ──────────────────────────────────────────────────────────

STOP_WORDS: FrozenSet[str] = frozenset({
    "will", "the", "be", "by", "in", "on", "to", "of", "a", "an", "is",
    "it", "or", "and", "for", "at", "this", "that", "with", "from", "not",
    "has", "have", "had", "do", "does", "did", "are", "was", "were", "been",
    "market", "price", "before", "after", "above", "below", "more", "than",
    "what", "who", "which", "when", "where", "how", "if", "than", "then",
    "yes", "no",
})

# ── Category normalization ──────────────────────────────────────────────

# Broad group mapping — unknown categories fall through to "other"
CATEGORY_GROUPS: Dict[str, str] = {
    "elections": "politics", "politics": "politics", "geopolitics": "politics",
    "trump": "politics", "trump presidency": "politics", "us election": "politics",
    "world elections": "politics", "global elections": "politics",
    "foreign policy": "politics", "courts": "politics", "economic policy": "politics",
    "economics": "finance", "economy": "finance", "finance": "finance",
    "financials": "finance", "crypto": "finance", "bitcoin": "finance",
    "ethereum": "finance", "solana": "finance", "xrp": "finance",
    "crypto prices": "finance", "stock prices": "finance", "equities": "finance",
    "commodities": "finance", "earnings": "finance", "big tech": "finance",
    "pre-market": "finance", "companies": "finance", "business": "finance",
    "inflation": "finance", "derivatives": "finance",
    "climate and weather": "science", "weather": "science",
    "science and technology": "science", "science & technology": "science",
    "tech": "science", "ai": "science", "science": "science",
    "climate & science": "science", "space": "science", "health": "science",
    "pandemics": "science",
    "entertainment": "culture", "culture": "culture", "awards": "culture",
    "movies": "culture", "music": "culture", "reality tv": "culture",
    "video games": "culture", "games": "culture",
    "sports": "sports", "esports": "sports", "nfl": "sports", "nba": "sports",
    "soccer": "sports", "cricket": "sports", "ufc": "sports", "tennis": "sports",
    "formula 1": "sports", "basketball": "sports", "nfl playoffs": "sports",
    "ncaab": "sports", "mlb": "sports",
    "world": "world", "social": "world",
}

# Keywords for fallback category detection
CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "politics": ["election", "president", "congress", "senate", "governor", "vote", "ballot",
                 "nominee", "impeach", "legislation", "democrat", "republican", "cabinet",
                 "supreme court", "scotus", "pope", "parliament"],
    "finance": ["stock", "bitcoin", "crypto", "price", "fed", "interest rate", "gdp",
                "inflation", "s&p", "nasdaq", "dow", "treasury", "forex", "gold", "oil",
                "etf", "ipo"],
    "science": ["weather", "hurricane", "temperature", "earthquake", "climate", "ai",
                "openai", "spacex", "nasa", "pandemic", "virus", "vaccine"],
    "sports": ["nfl", "nba", "mlb", "nhl", "soccer", "football", "basketball",
               "baseball", "hockey", "championship", "super bowl", "world cup",
               "playoffs", "mvp", "draft"],
    "culture": ["oscar", "emmy", "grammy", "movie", "film", "album", "song",
                "netflix", "box office", "reality tv", "bachelor"],
}


def normalize_category(cat: str) -> str:
    """Map a venue category string to a broad group."""
    cat_lower = cat.lower().strip()
    if cat_lower in CATEGORY_GROUPS:
        return CATEGORY_GROUPS[cat_lower]
    # Keyword fallback
    for group, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in cat_lower:
                return group
    return "other"


# ── Feature extraction ──────────────────────────────────────────────────

# Patterns
_NUMBER_PATTERN = re.compile(r'\b(\d{4}|\d+%|\$\d[\d,.]*[KMBkmb]?|\d+(?:\.\d+)?)\b')
_ENTITY_PATTERN = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b')
_PUNCTUATION_TABLE = str.maketrans("", "", string.punctuation)


def extract_features(text: str) -> Tuple[str, FrozenSet[str], List[str], List[str]]:
    """
    Extract text features for matching.

    Returns:
        (normalized_text, token_set, entities, numbers)
    """
    # 1. Extract numbers/dates before normalization
    numbers = _NUMBER_PATTERN.findall(text)

    # 2. Extract entities (capitalized proper nouns from original text)
    #    Filter out sentence-initial common words that happen to be capitalized
    raw_entities = _ENTITY_PATTERN.findall(text)
    seen: Set[str] = set()
    unique_entities = []
    first_word = text.split()[0].rstrip("?.,!:;") if text else ""
    for e in raw_entities:
        e_lower = e.lower()
        # Remove sentence-initial stop words from multi-word entities
        # e.g. "Will Trump" -> filter out "Will" if it's a stop word
        words = e.split()
        if words and words[0] == first_word and words[0].lower() in STOP_WORDS:
            words = words[1:]
        if not words:
            continue
        cleaned = " ".join(words).lower()
        if cleaned not in seen and cleaned not in STOP_WORDS:
            seen.add(cleaned)
            unique_entities.append(cleaned)
    entities = unique_entities

    # 3. Normalize: lowercase, strip punctuation, remove stop words
    lowered = text.lower().translate(_PUNCTUATION_TABLE)
    tokens = lowered.split()
    meaningful = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]

    normalized = " ".join(meaningful)
    token_set = frozenset(meaningful)

    return normalized, token_set, entities, numbers


def populate_features(event: NormalizedEvent) -> None:
    """Populate text features on an event and all its markets."""
    norm, tokens, ents, _ = extract_features(event.title)
    event.normalized_title = norm
    event.token_set = tokens
    event.entities = ents

    for market in event.markets:
        m_norm, m_tokens, m_ents, m_nums = extract_features(market.question)
        market.normalized_text = m_norm
        market.token_set = m_tokens
        market.entities = m_ents
        market.numbers = m_nums


# ── Scoring functions ───────────────────────────────────────────────────

def jaccard(a: FrozenSet[str], b: FrozenSet[str]) -> float:
    """Jaccard similarity between two token sets."""
    if not a and not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union > 0 else 0.0


def entity_overlap(a: List[str], b: List[str]) -> float:
    """Fraction of entity words shared between two lists.

    Decomposes multi-word entities into individual words for matching,
    so "presidential election" partially matches "presidential election winner".
    """
    if not a and not b:
        return 0.0
    if not a or not b:
        return 0.0
    # Decompose multi-word entities into individual words
    words_a: Set[str] = set()
    for e in a:
        words_a.update(e.lower().split())
    words_b: Set[str] = set()
    for e in b:
        words_b.update(e.lower().split())
    # Remove stop words from entity words
    words_a -= STOP_WORDS
    words_b -= STOP_WORDS
    if not words_a or not words_b:
        return 0.0
    intersection = len(words_a & words_b)
    min_len = min(len(words_a), len(words_b))
    return intersection / min_len if min_len > 0 else 0.0


def fuzzy_ratio(a: str, b: str) -> float:
    """Fuzzy string similarity ratio (0-1). Uses rapidfuzz if available, difflib fallback."""
    if not a or not b:
        return 0.0
    try:
        from rapidfuzz import fuzz
        return fuzz.ratio(a, b) / 100.0
    except ImportError:
        from difflib import SequenceMatcher
        return SequenceMatcher(None, a, b).ratio()


def combined_event_score(
    token_score: float,
    entity_score: float,
    fuzzy_score: float,
    w_token: float = 0.35,
    w_entity: float = 0.40,
    w_fuzzy: float = 0.25,
) -> float:
    """Weighted combination of event-level scores."""
    return w_token * token_score + w_entity * entity_score + w_fuzzy * fuzzy_score


# ── Main matcher ────────────────────────────────────────────────────────

class TextMatcher:
    """Tier 1 deterministic matcher."""

    def __init__(
        self,
        event_threshold: float = 0.50,
        market_threshold: float = 0.55,
    ):
        self.event_threshold = event_threshold
        self.market_threshold = market_threshold

    def match_events(
        self,
        kalshi_events: List[NormalizedEvent],
        poly_events: List[NormalizedEvent],
    ) -> Tuple[
        List[Tuple[NormalizedEvent, NormalizedEvent, float]],
        List[NormalizedEvent],
        List[NormalizedEvent],
    ]:
        """
        Match Kalshi events to Polymarket events using text similarity.

        Returns:
            (matched_pairs, unmatched_kalshi, unmatched_poly)
        """
        # Ensure features are populated
        for e in kalshi_events:
            if not e.token_set:
                populate_features(e)
        for e in poly_events:
            if not e.token_set:
                populate_features(e)

        # Group by broad category
        k_by_cat: Dict[str, List[NormalizedEvent]] = defaultdict(list)
        p_by_cat: Dict[str, List[NormalizedEvent]] = defaultdict(list)

        for e in kalshi_events:
            k_by_cat[normalize_category(e.category)].append(e)
        for e in poly_events:
            p_by_cat[normalize_category(e.category)].append(e)

        # Also add everything to "other" for cross-category matching
        shared_cats = set(k_by_cat.keys()) & set(p_by_cat.keys())
        logger.info(f"Tier 1: {len(shared_cats)} shared categories: {sorted(shared_cats)}")

        # Score all pairs within shared categories
        scored: List[Tuple[NormalizedEvent, NormalizedEvent, float]] = []

        for cat in shared_cats:
            for ke in k_by_cat[cat]:
                for pe in p_by_cat[cat]:
                    t_score = jaccard(ke.token_set, pe.token_set)
                    e_score = entity_overlap(ke.entities, pe.entities)
                    f_score = fuzzy_ratio(ke.normalized_title, pe.normalized_title)
                    score = combined_event_score(t_score, e_score, f_score)

                    if score >= self.event_threshold:
                        scored.append((ke, pe, score))

        # Also try cross-category for "other" items
        other_k = k_by_cat.get("other", [])
        other_p = p_by_cat.get("other", [])
        if other_k and other_p:
            for ke in other_k:
                for pe in other_p:
                    t_score = jaccard(ke.token_set, pe.token_set)
                    e_score = entity_overlap(ke.entities, pe.entities)
                    f_score = fuzzy_ratio(ke.normalized_title, pe.normalized_title)
                    score = combined_event_score(t_score, e_score, f_score)
                    if score >= self.event_threshold:
                        scored.append((ke, pe, score))

        # Greedy 1:1 assignment by score descending
        scored.sort(key=lambda x: x[2], reverse=True)
        used_k: Set[str] = set()
        used_p: Set[str] = set()
        matched: List[Tuple[NormalizedEvent, NormalizedEvent, float]] = []

        for ke, pe, score in scored:
            if ke.event_id in used_k or pe.event_id in used_p:
                continue
            used_k.add(ke.event_id)
            used_p.add(pe.event_id)
            matched.append((ke, pe, score))

        unmatched_k = [e for e in kalshi_events if e.event_id not in used_k]
        unmatched_p = [e for e in poly_events if e.event_id not in used_p]

        logger.info(
            f"Tier 1 event matching: {len(matched)} matched, "
            f"{len(unmatched_k)} unmatched Kalshi, {len(unmatched_p)} unmatched Poly"
        )
        return matched, unmatched_k, unmatched_p

    def match_markets(
        self,
        event_pairs: List[Tuple[NormalizedEvent, NormalizedEvent, float]],
    ) -> Tuple[List[MatchCandidate], List[Tuple[NormalizedMarket, NormalizedMarket, float]]]:
        """
        Match markets within matched event pairs.

        Returns:
            (confirmed_matches, near_misses) where near_misses have
            scores between 0.35 and market_threshold (candidates for LLM confirmation).
        """
        confirmed: List[MatchCandidate] = []
        near_misses: List[Tuple[NormalizedMarket, NormalizedMarket, float]] = []

        for ke, pe, event_score in event_pairs:
            k_markets = [m for m in ke.markets if m.is_active]
            p_markets = [m for m in pe.markets if m.is_active]

            if not k_markets or not p_markets:
                continue

            # Binary events (1 market each): direct match
            if len(k_markets) == 1 and len(p_markets) == 1:
                km = k_markets[0]
                pm = p_markets[0]
                # Still compute score for auditing
                m_score = self._market_score(km, pm)
                confirmed.append(MatchCandidate(
                    kalshi_market=km,
                    poly_market=pm,
                    kalshi_event=ke,
                    poly_event=pe,
                    event_text_score=event_score,
                    market_text_score=m_score,
                    combined_score=(event_score + m_score) / 2,
                    match_method="text",
                    match_tier=1,
                    match_signals=["binary_event", "tier1_text"],
                ))
                continue

            # Multi-market: pairwise scoring + greedy 1:1
            scored_pairs: List[Tuple[NormalizedMarket, NormalizedMarket, float]] = []
            for km in k_markets:
                for pm in p_markets:
                    m_score = self._market_score(km, pm)
                    scored_pairs.append((km, pm, m_score))

            scored_pairs.sort(key=lambda x: x[2], reverse=True)
            used_k: Set[str] = set()
            used_p: Set[str] = set()

            for km, pm, m_score in scored_pairs:
                if km.market_id in used_k or pm.market_id in used_p:
                    continue

                if m_score >= self.market_threshold:
                    used_k.add(km.market_id)
                    used_p.add(pm.market_id)
                    confirmed.append(MatchCandidate(
                        kalshi_market=km,
                        poly_market=pm,
                        kalshi_event=ke,
                        poly_event=pe,
                        event_text_score=event_score,
                        market_text_score=m_score,
                        combined_score=(event_score + m_score) / 2,
                        match_method="text",
                        match_tier=1,
                        match_signals=["multi_market", "tier1_text"],
                    ))
                elif m_score >= 0.35:
                    # Near miss — candidate for LLM confirmation
                    near_misses.append((km, pm, m_score))

        logger.info(
            f"Tier 1 market matching: {len(confirmed)} confirmed, "
            f"{len(near_misses)} near misses for LLM review"
        )
        return confirmed, near_misses

    def _market_score(self, km: NormalizedMarket, pm: NormalizedMarket) -> float:
        """Compute similarity score between two markets."""
        # Ensure features populated
        if not km.token_set:
            n, t, e, nums = extract_features(km.question)
            km.normalized_text, km.token_set, km.entities, km.numbers = n, t, e, nums
        if not pm.token_set:
            n, t, e, nums = extract_features(pm.question)
            pm.normalized_text, pm.token_set, pm.entities, pm.numbers = n, t, e, nums

        j = jaccard(km.token_set, pm.token_set)
        e = entity_overlap(km.entities, pm.entities)
        f = fuzzy_ratio(km.normalized_text, pm.normalized_text)

        # Numbers must match for market-level (different thresholds = different markets)
        num_penalty = 0.0
        if km.numbers and pm.numbers:
            k_nums = set(km.numbers)
            p_nums = set(pm.numbers)
            if k_nums != p_nums and not (k_nums & p_nums):
                num_penalty = 0.15

        # Weights: entity overlap matters most at market level
        score = 0.30 * j + 0.40 * e + 0.30 * f - num_penalty
        return max(0.0, score)
