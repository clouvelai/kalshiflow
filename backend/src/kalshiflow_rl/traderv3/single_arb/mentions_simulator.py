"""
Blind LLM roleplay simulation for mentions probability estimation.

Core idea:
1. Generate realistic transcripts WITHOUT knowing what terms we're looking for
2. Scan generated transcripts post-hoc for mention terms
3. Estimate P(term) = appearances / simulations

This avoids the bias of asking "would you say X?" and instead lets the LLM
generate natural content based on its training data from real broadcasts.
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("kalshiflow_rl.traderv3.single_arb.mentions_simulator")


@dataclass
class SimulationResult:
    """Result of a single blind roleplay simulation."""

    event_ticker: str
    simulation_id: str
    transcript: str  # Full generated transcript
    segment_transcripts: Dict[str, str] = field(default_factory=dict)  # Per-segment
    generated_at: float = 0.0  # Unix timestamp
    model: str = "haiku"
    context_hash: str = ""  # For cache invalidation
    word_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SimulationResult":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class TermMatch:
    """A single match of a term in a transcript."""

    span: str  # Exact matched text
    position: int  # Character position in transcript
    segment: str  # Which segment it was found in
    context: str  # Surrounding text (50 chars each side)
    match_type: str  # "exact" or "morphological"


@dataclass
class TermScanResult:
    """Results of scanning for a single term across simulations."""

    term: str
    exact_matches: List[TermMatch] = field(default_factory=list)
    morphological_matches: List[TermMatch] = field(default_factory=list)
    total_exact: int = 0
    total_morphological: int = 0
    # Binary counts: how many simulations had at least one match
    simulations_with_exact: int = 0
    simulations_with_any: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "term": self.term,
            "exact_matches": [
                {"span": m.span, "position": m.position, "segment": m.segment, "context": m.context}
                for m in self.exact_matches
            ],
            "morphological_matches": [
                {"span": m.span, "position": m.position, "segment": m.segment, "context": m.context}
                for m in self.morphological_matches
            ],
            "total_exact": self.total_exact,
            "total_morphological": self.total_morphological,
            "simulations_with_exact": self.simulations_with_exact,
            "simulations_with_any": self.simulations_with_any,
        }


@dataclass
class ProbabilityEstimate:
    """Probability estimate for a single term."""

    term: str
    probability: float  # exact_appearances / n_simulations
    probability_with_variants: float  # (exact + morphological) / n_simulations
    confidence_interval: Tuple[float, float] = (0.0, 1.0)  # 95% CI
    n_simulations: int = 0
    appearances_per_simulation: List[int] = field(default_factory=list)
    variance: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "term": self.term,
            "probability": self.probability,
            "probability_with_variants": self.probability_with_variants,
            "confidence_interval": list(self.confidence_interval),
            "n_simulations": self.n_simulations,
            "appearances_per_simulation": self.appearances_per_simulation,
            "variance": self.variance,
        }


class MentionsSimulator:
    """LLM roleplay simulator for mentions probability estimation."""

    def __init__(self, cache_dir: str):
        """Initialize simulator with cache directory.

        Args:
            cache_dir: Directory for persisting simulation results
        """
        self.cache_dir = cache_dir
        self._cache: Dict[str, List[SimulationResult]] = {}  # event_ticker -> simulations
        self._load_cache()

    async def simulate(
        self,
        event_ticker: str,
        prompt: str,
        n_simulations: int = 10,
        model: str = "haiku",
        context_hash: Optional[str] = None,
    ) -> List[SimulationResult]:
        """Run N blind roleplay simulations.

        Args:
            event_ticker: Event identifier for caching
            prompt: Full roleplay prompt (must be blind - no mention terms)
            n_simulations: Number of simulations to run
            model: LLM model to use ("haiku" recommended for speed/cost)
            context_hash: Hash of context for cache invalidation

        Returns:
            List of SimulationResult objects
        """
        # Check cache first
        if context_hash:
            cached = self._get_cached(event_ticker, context_hash, n_simulations)
            if cached:
                logger.info(f"Using {len(cached)} cached simulations for {event_ticker}")
                return cached

        # Run new simulations
        results = []
        logger.info(f"Running {n_simulations} blind simulations for {event_ticker}...")

        # Run simulations concurrently (batch of 5 at a time to avoid rate limits)
        batch_size = 5
        for i in range(0, n_simulations, batch_size):
            batch_n = min(batch_size, n_simulations - i)
            tasks = [
                self._run_single_simulation(event_ticker, prompt, model, context_hash)
                for _ in range(batch_n)
            ]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for r in batch_results:
                if isinstance(r, SimulationResult):
                    results.append(r)
                else:
                    logger.warning(f"Simulation failed: {r}")

            # Small delay between batches
            if i + batch_size < n_simulations:
                await asyncio.sleep(1.0)

        # Cache results
        if context_hash and results:
            self._cache_results(event_ticker, context_hash, results)

        logger.info(f"Completed {len(results)} simulations for {event_ticker}")
        return results

    async def _run_single_simulation(
        self,
        event_ticker: str,
        prompt: str,
        model: str,
        context_hash: Optional[str],
    ) -> SimulationResult:
        """Run a single blind roleplay simulation.

        Uses centralized model config (default: Gemini 2.0 Flash).
        Override via MENTIONS_SIMULATION_MODEL env var or model parameter.
        """
        try:
            from .mentions_models import get_simulation_llm, DEFAULT_SIMULATION_MODEL

            # Use specified model or default from config
            model_to_use = model if model else DEFAULT_SIMULATION_MODEL
            llm = get_simulation_llm(model=model_to_use)

            response = await llm.ainvoke(prompt)
            transcript = response.content

            # Parse segments from transcript
            segments = self._parse_segments(transcript)

            return SimulationResult(
                event_ticker=event_ticker,
                simulation_id=str(uuid.uuid4())[:8],
                transcript=transcript,
                segment_transcripts=segments,
                generated_at=time.time(),
                model=model_to_use,
                context_hash=context_hash or "",
                word_count=len(transcript.split()),
            )

        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            raise

    def _parse_segments(self, transcript: str) -> Dict[str, str]:
        """Parse segment headers from transcript."""
        segments = {}

        # Look for [SEGMENT_NAME] headers
        pattern = r"\[([A-Z0-9_]+)\]"
        matches = list(re.finditer(pattern, transcript))

        for i, match in enumerate(matches):
            segment_name = match.group(1).lower()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(transcript)
            segments[segment_name] = transcript[start:end].strip()

        return segments

    def scan_for_terms(
        self,
        simulations: List[SimulationResult],
        terms: List[str],
        use_wordnet: bool = True,
    ) -> Dict[str, TermScanResult]:
        """Post-hoc scan for mention terms in simulated transcripts.

        Args:
            simulations: List of simulation results
            terms: List of terms to look for
            use_wordnet: Whether to use WordNet for morphological variants

        Returns:
            Dict mapping term -> TermScanResult with simulations_with_match count
        """
        results = {}

        for term in terms:
            scan_result = TermScanResult(term=term)
            simulations_with_exact = 0
            simulations_with_any = 0

            # Get accepted forms (morphological variants)
            accepted_forms = self._get_accepted_forms(term, use_wordnet)

            for sim in simulations:
                transcript = sim.transcript
                sim_had_exact = False
                sim_had_any = False

                # Scan for exact matches
                exact_matches = self._scan_text(transcript, [term], "exact", sim.segment_transcripts)
                scan_result.exact_matches.extend(exact_matches)
                scan_result.total_exact += len(exact_matches)
                if exact_matches:
                    sim_had_exact = True
                    sim_had_any = True

                # Scan for morphological variants (excluding the exact term)
                if accepted_forms:
                    variant_forms = [f for f in accepted_forms if f.lower() != term.lower()]
                    morph_matches = self._scan_text(
                        transcript, variant_forms, "morphological", sim.segment_transcripts
                    )
                    scan_result.morphological_matches.extend(morph_matches)
                    scan_result.total_morphological += len(morph_matches)
                    if morph_matches:
                        sim_had_any = True

                if sim_had_exact:
                    simulations_with_exact += 1
                if sim_had_any:
                    simulations_with_any += 1

            # Store the binary simulation counts for probability calculation
            scan_result.simulations_with_exact = simulations_with_exact
            scan_result.simulations_with_any = simulations_with_any

            results[term] = scan_result

        return results

    def _scan_text(
        self,
        transcript: str,
        forms: List[str],
        match_type: str,
        segments: Dict[str, str],
    ) -> List[TermMatch]:
        """Scan transcript for term forms."""
        matches = []

        for form in forms:
            # Case-insensitive word boundary search
            pattern = re.compile(r"\b" + re.escape(form) + r"\b", re.IGNORECASE)

            for match in pattern.finditer(transcript):
                # Get context
                start = max(0, match.start() - 50)
                end = min(len(transcript), match.end() + 50)
                context = transcript[start:end]

                # Determine which segment
                segment = self._find_segment(match.start(), segments, transcript)

                matches.append(
                    TermMatch(
                        span=match.group(),
                        position=match.start(),
                        segment=segment,
                        context=context,
                        match_type=match_type,
                    )
                )

        return matches

    def _find_segment(
        self,
        position: int,
        segments: Dict[str, str],
        full_transcript: str,
    ) -> str:
        """Find which segment a position belongs to."""
        # Find segment headers and their positions
        pattern = r"\[([A-Z0-9_]+)\]"
        headers = [(m.group(1).lower(), m.start()) for m in re.finditer(pattern, full_transcript)]

        current_segment = "unknown"
        for name, start in headers:
            if start <= position:
                current_segment = name
            else:
                break

        return current_segment

    def _get_accepted_forms(self, term: str, use_wordnet: bool) -> List[str]:
        """Get accepted morphological forms of a term."""
        forms = {term, term.lower(), term.upper(), term.capitalize()}

        if use_wordnet:
            try:
                from .mentions_semantic import get_accepted_forms

                wordnet_forms = get_accepted_forms(term)
                forms.update(wordnet_forms)
            except (ImportError, Exception) as e:
                logger.debug(f"WordNet not available: {e}")

        return list(forms)

    def estimate_probability(
        self,
        scan_results: Dict[str, TermScanResult],
        n_simulations: int,
    ) -> Dict[str, ProbabilityEstimate]:
        """Compute probability estimates from scan results.

        Args:
            scan_results: Dict from scan_for_terms()
            n_simulations: Number of simulations that were run

        Returns:
            Dict mapping term -> ProbabilityEstimate
        """
        estimates = {}

        for term, scan in scan_results.items():
            # Use binary simulation counts (how many sims had at least one match)
            sims_with_exact = getattr(scan, 'simulations_with_exact', 0)
            sims_with_any = getattr(scan, 'simulations_with_any', 0)

            # Probability = proportion of simulations with at least one appearance
            p_exact = sims_with_exact / max(n_simulations, 1)
            p_with_variants = sims_with_any / max(n_simulations, 1)

            # Clamp to [0, 1]
            p_exact = min(1.0, max(0.0, p_exact))
            p_with_variants = min(1.0, max(0.0, p_with_variants))

            # Wilson score confidence interval for binomial proportion
            ci = self._wilson_ci(sims_with_exact, n_simulations)

            estimates[term] = ProbabilityEstimate(
                term=term,
                probability=p_exact,
                probability_with_variants=p_with_variants,
                confidence_interval=ci,
                n_simulations=n_simulations,
                appearances_per_simulation=[sims_with_exact],  # Binary count
                variance=p_exact * (1 - p_exact),  # Binomial variance
            )

        return estimates

    def _wilson_ci(
        self,
        successes: int,
        n: int,
        z: float = 1.96,  # 95% CI
    ) -> Tuple[float, float]:
        """Compute Wilson score confidence interval."""
        if n == 0:
            return (0.0, 1.0)

        p_hat = successes / n
        denominator = 1 + z * z / n
        center = (p_hat + z * z / (2 * n)) / denominator

        # Guard against floating point issues that can produce negative values under sqrt
        variance_term = p_hat * (1 - p_hat) / n + z * z / (4 * n * n)
        variance_term = max(0.0, variance_term)  # Ensure non-negative
        margin = z * (variance_term ** 0.5) / denominator

        lower = max(0.0, center - margin)
        upper = min(1.0, center + margin)

        return (lower, upper)

    # =============================================================================
    # CACHING
    # =============================================================================

    def _load_cache(self) -> None:
        """Load persisted simulations from disk."""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
            return

        cache_index_path = os.path.join(self.cache_dir, "simulation_index.json")
        if not os.path.exists(cache_index_path):
            return

        try:
            with open(cache_index_path, "r") as f:
                index = json.load(f)

            for event_ticker, sim_files in index.items():
                self._cache[event_ticker] = []
                for sim_file in sim_files[:20]:  # Limit loaded simulations
                    sim_path = os.path.join(self.cache_dir, sim_file)
                    if os.path.exists(sim_path):
                        with open(sim_path, "r") as f:
                            data = json.load(f)
                            self._cache[event_ticker].append(SimulationResult.from_dict(data))

            logger.info(f"Loaded simulation cache: {len(self._cache)} events")

        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load simulation cache: {e}")

    def _save_cache(self) -> None:
        """Persist simulation cache to disk."""
        os.makedirs(self.cache_dir, exist_ok=True)

        index = {}
        for event_ticker, simulations in self._cache.items():
            index[event_ticker] = []
            for sim in simulations[:20]:  # Limit persisted simulations
                sim_file = f"sim_{event_ticker}_{sim.simulation_id}.json"
                sim_path = os.path.join(self.cache_dir, sim_file)
                try:
                    with open(sim_path, "w") as f:
                        json.dump(sim.to_dict(), f, indent=2)
                    index[event_ticker].append(sim_file)
                except OSError as e:
                    logger.warning(f"Failed to save simulation: {e}")

        index_path = os.path.join(self.cache_dir, "simulation_index.json")
        try:
            with open(index_path, "w") as f:
                json.dump(index, f, indent=2)
        except OSError as e:
            logger.warning(f"Failed to save simulation index: {e}")

    def _get_cached(
        self,
        event_ticker: str,
        context_hash: str,
        n_simulations: int,
    ) -> Optional[List[SimulationResult]]:
        """Get cached simulations if available and fresh."""
        cached = self._cache.get(event_ticker, [])

        # Filter by context hash
        matching = [s for s in cached if s.context_hash == context_hash]

        if len(matching) >= n_simulations:
            return matching[:n_simulations]

        return None

    def _cache_results(
        self,
        event_ticker: str,
        context_hash: str,
        results: List[SimulationResult],
    ) -> None:
        """Cache simulation results."""
        if event_ticker not in self._cache:
            self._cache[event_ticker] = []

        # Remove old results with different context hash
        self._cache[event_ticker] = [
            s for s in self._cache[event_ticker] if s.context_hash == context_hash
        ]

        # Add new results
        self._cache[event_ticker].extend(results)

        # Persist
        self._save_cache()

    def clear_cache(self, event_ticker: Optional[str] = None) -> None:
        """Clear cache for an event or all events."""
        if event_ticker:
            self._cache.pop(event_ticker, None)
        else:
            self._cache.clear()
        self._save_cache()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


@dataclass
class ComparisonResult:
    """Result of comparing blind vs informed simulations."""

    term: str
    blind_probability: float
    informed_probability: float
    delta: float  # informed - blind
    relevance_score: float  # from TermRelevance
    relevance_type: str
    interpretation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _extract_news_snippets(informed_context) -> List[str]:
    """Extract news snippets from InformedContext for result payload."""
    news_snippets = []
    # Add general storylines
    for storyline in informed_context.storylines[:3]:
        if storyline and storyline not in news_snippets:
            news_snippets.append(storyline[:150])
    # Add term-specific news mentions
    for term, relevance in informed_context.term_relevance.items():
        for mention in relevance.news_mentions[:2]:
            if mention and mention[:150] not in news_snippets:
                news_snippets.append(mention[:150])
    # Limit to 5 snippets max
    return news_snippets[:5]


async def _run_simulation_core(
    event_ticker: str,
    event_title: str,
    mention_terms: List[str],
    n_simulations: int,
    mode: str,  # "blind" or "informed"
    cache_dir: Optional[str] = None,
    event_context: Optional[Any] = None,  # EventContext, reuse if provided
    mentions_data: Optional[Dict[str, Any]] = None,  # For speaker persona extraction
) -> Dict[str, Any]:
    """Core simulation logic for both blind and informed modes.

    This internal function handles the common simulation pipeline:
    1. Template detection
    2. Context gathering (mode-specific)
    3. Prompt generation (mode-specific)
    4. Simulation execution
    5. Term scanning
    6. Probability estimation
    7. Result formatting

    Args:
        event_ticker: Kalshi event ticker
        event_title: Event title for context
        mention_terms: List of terms to estimate probabilities for
        n_simulations: Number of simulations to run
        mode: "blind" for baseline or "informed" for context-aware
        cache_dir: Optional cache directory for simulation persistence
        event_context: Optional pre-fetched EventContext (avoids duplicate HTTP calls)

    Returns:
        Dict with probability estimates, scan results, and metadata
    """
    from .mentions_templates import detect_template
    from .mentions_context import (
        gather_event_context,
        gather_informed_context,
        generate_blind_prompt,
        generate_informed_prompt,
        EventContext,
    )

    if not cache_dir:
        cache_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "memory", "data", "simulations"
        )

    # 1. Detect template from event structure
    template = detect_template(event_ticker, event_title)

    # 2. Gather context (mode-specific, but can reuse EventContext)
    news_snippets: List[str] = []
    informed_context = None
    context_for_result: Any = None

    if mode == "blind":
        if event_context is None:
            event_context = await gather_event_context(
                event_ticker, event_title, mentions_data=mentions_data
            )
        context_for_result = event_context
    else:
        informed_context = await gather_informed_context(
            event_ticker, event_title, mention_terms,
            include_news=True,
            event_context=event_context,
        )
        news_snippets = _extract_news_snippets(informed_context)
        context_for_result = informed_context
        if event_context is None:
            event_context = informed_context.event

    simulator = MentionsSimulator(cache_dir=cache_dir)

    # 4. Generate prompt from template (compressed format for token efficiency)
    if mode == "blind":
        prompt = generate_blind_prompt(event_context, template, compressed=True)
    else:
        prompt = generate_informed_prompt(informed_context, template, compressed=True)

    # 5. Run simulations
    context_hash = hashlib.md5(
        (prompt + f"{mode.upper()}_COMP" + str(mention_terms)).encode()
    ).hexdigest()[:12]

    all_simulations = await simulator.simulate(
        event_ticker=event_ticker,
        prompt=prompt,
        n_simulations=n_simulations,
        model=None,
        context_hash=context_hash,
    )

    # 6. Scan for terms across ALL simulations
    scan_results = simulator.scan_for_terms(
        simulations=all_simulations,
        terms=mention_terms,
        use_wordnet=True,
    )

    # 7. Compute probability estimates
    estimates = simulator.estimate_probability(
        scan_results=scan_results,
        n_simulations=len(all_simulations),
    )

    # 8. Build result (mode-specific payload)
    result = {
        "mode": mode,
        "simulation_format": "compressed",
        "event_ticker": event_ticker,
        "event_title": event_title,
        "template": template.event_type,
        "domain": template.domain,
        "n_simulations": len(all_simulations),
        "estimates": {term: est.to_dict() for term, est in estimates.items()},
        "scan_results": {term: scan.to_dict() for term, scan in scan_results.items()},
        "news_snippets": news_snippets,
    }

    # Add mode-specific context fields
    if mode == "blind":
        result["context"] = context_for_result.to_dict()
    else:
        result["informed_context"] = context_for_result.to_dict()
        result["term_relevance"] = {
            t: r.to_dict() for t, r in informed_context.term_relevance.items()
        }

    return result


async def run_comparison_simulation(
    event_ticker: str,
    event_title: str,
    mention_terms: List[str],
    n_simulations: int = 10,
    cache_dir: Optional[str] = None,
    mentions_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run BOTH blind and informed simulations and compare.

    This is the key edge detection function:
    - BLIND gives baseline P (generic event, no news)
    - INFORMED gives context-adjusted P (with news and term relevance)
    - DELTA = INFORMED - BLIND tells you how much context matters

    If delta is large and positive, context makes the term more likely.
    If delta is near zero, the term's probability doesn't depend on context.

    Optimization: Fetches EventContext once and reuses for both simulations,
    saving ~5-10 HTTP calls compared to running separately.

    Args:
        event_ticker: Kalshi event ticker
        event_title: Event title for context
        mention_terms: List of terms to estimate probabilities for
        n_simulations: Number of simulations (10 recommended)
        cache_dir: Optional cache directory
        mentions_data: Optional dict with lexeme_pack (for speaker extraction from rules)
    """
    from .mentions_context import gather_event_context

    # Fetch EventContext once for both simulations
    event_context = await gather_event_context(
        event_ticker, event_title, mentions_data=mentions_data
    )

    # Run both simulations, passing shared EventContext
    blind_result = await _run_simulation_core(
        event_ticker=event_ticker,
        event_title=event_title,
        mention_terms=mention_terms,
        n_simulations=n_simulations,
        mode="blind",
        cache_dir=cache_dir,
        event_context=event_context,
        mentions_data=mentions_data,
    )

    informed_result = await _run_simulation_core(
        event_ticker=event_ticker,
        event_title=event_title,
        mention_terms=mention_terms,
        n_simulations=n_simulations,
        mode="informed",
        cache_dir=cache_dir,
        event_context=event_context,
        mentions_data=mentions_data,
    )

    # Get term relevance from informed result (already computed)
    term_relevance = informed_result.get("term_relevance", {})

    # Compare results
    comparisons = []
    for term in mention_terms:
        blind_est = blind_result["estimates"].get(term, {})
        informed_est = informed_result["estimates"].get(term, {})
        relevance = term_relevance.get(term, {})

        blind_p = blind_est.get("probability", 0.0)
        informed_p = informed_est.get("probability", 0.0)
        delta = informed_p - blind_p

        # Interpret the delta
        if delta > 0.2:
            interpretation = f"Strong context effect: {term} much more likely given current news/relevance"
        elif delta > 0.1:
            interpretation = f"Moderate context effect: {term} somewhat more likely with context"
        elif delta > 0.05:
            interpretation = f"Weak context effect: {term} slightly more likely with context"
        elif delta < -0.1:
            interpretation = f"Negative context effect: {term} less likely given current context"
        else:
            interpretation = f"No context effect: {term} probability stable regardless of context"

        comparison = ComparisonResult(
            term=term,
            blind_probability=blind_p,
            informed_probability=informed_p,
            delta=delta,
            relevance_score=relevance.get("relevance_score", 0.0),
            relevance_type=relevance.get("relevance_type", "unknown"),
            interpretation=interpretation,
        )
        comparisons.append(comparison)

    # Sort by absolute delta (biggest context effects first)
    comparisons.sort(key=lambda c: abs(c.delta), reverse=True)

    return {
        "event_ticker": event_ticker,
        "event_title": event_title,
        "n_simulations": n_simulations,
        "blind_result": blind_result,
        "informed_result": informed_result,
        "comparisons": [c.to_dict() for c in comparisons],
        "summary": {
            "terms_with_context_effect": len([c for c in comparisons if abs(c.delta) > 0.1]),
            "terms_stable": len([c for c in comparisons if abs(c.delta) <= 0.1]),
            "largest_positive_delta": max(c.delta for c in comparisons) if comparisons else 0,
            "largest_negative_delta": min(c.delta for c in comparisons) if comparisons else 0,
        },
    }


async def run_mentions_simulation(
    event_ticker: str,
    event_title: str,
    mention_terms: List[str],
    n_simulations: int = 10,
    cache_dir: Optional[str] = None,
    mode: str = "informed",  # "blind", "informed", or "compare"
    mentions_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run mentions simulation pipeline.

    Main entry point for the simulation system.

    Args:
        event_ticker: Kalshi event ticker
        event_title: Event title for context
        mention_terms: List of terms to estimate probabilities for
        n_simulations: Number of simulations (10 recommended for trading)
        cache_dir: Optional cache directory
        mode: "blind" (baseline), "informed" (with context), or "compare" (both)
        mentions_data: Optional dict with lexeme_pack (for speaker extraction from rules)

    Returns:
        Dict with probability estimates, scan results, and metadata
    """
    if mode == "compare":
        return await run_comparison_simulation(
            event_ticker, event_title, mention_terms, n_simulations, cache_dir, mentions_data
        )
    else:
        # Use core for both blind and informed modes
        return await _run_simulation_core(
            event_ticker=event_ticker,
            event_title=event_title,
            mention_terms=mention_terms,
            n_simulations=n_simulations,
            mode=mode if mode in ("blind", "informed") else "informed",
            cache_dir=cache_dir,
            mentions_data=mentions_data,
        )


# =============================================================================
# COUNTER-FACTUAL SIMULATION
# =============================================================================


@dataclass
class CounterfactualResult:
    """Result of counter-factual (avoidance) simulation."""

    term: str
    n_simulations: int
    leakages: int  # How many times the model said the term anyway
    leakage_rate: float  # leakages / n_simulations
    leakage_contexts: List[str]  # Context around each leakage
    interpretation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


async def run_counterfactual_simulation(
    event_ticker: str,
    event_title: str,
    term: str,
    n_simulations: int = 5,
    cache_dir: Optional[str] = None,
    mentions_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run counter-factual simulation measuring avoidance difficulty.

    Generates transcripts with explicit instruction to AVOID mentioning
    the term, then counts how many times the LLM "leaks" and says it anyway.

    High leakage = term is hard to avoid = high baseline probability
    Low leakage = term is easy to avoid = term not naturally part of discourse

    This is useful for:
    - Validating blind simulation results (should correlate)
    - Detecting terms that are so central they're nearly unavoidable
    - Understanding the "floor" probability for a term

    Args:
        event_ticker: Kalshi event ticker
        event_title: Event title for context
        term: Single term to test avoidance of
        n_simulations: Number of simulations (default 5)
        cache_dir: Optional cache directory
        mentions_data: Optional dict with lexeme_pack (for speaker extraction from rules)

    Returns:
        Dict with leakage_rate, leakage contexts, and interpretation
    """
    from .mentions_templates import detect_template
    from .mentions_context import gather_event_context, generate_blind_prompt

    logger.info(f"Running counterfactual simulation for '{term}' in {event_ticker}")

    if not cache_dir:
        cache_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "memory", "data", "simulations"
        )

    # Get event context and template
    event_context = await gather_event_context(
        event_ticker, event_title, mentions_data=mentions_data
    )
    template = detect_template(event_ticker, event_title)

    # Generate base prompt
    base_prompt = generate_blind_prompt(event_context, template)

    # Add avoidance constraint
    avoidance_instruction = f"""

CRITICAL CONSTRAINT:
You must AVOID using the word "{term}" or any form/variant of it.
Do NOT say "{term}", "{term.lower()}", "{term.upper()}", or related forms.
If you would naturally say this word, find an alternative way to express the same idea.
This is a test of whether you can generate natural content while avoiding specific words.

Generate the transcript now, WITHOUT using the word "{term}".
"""

    counterfactual_prompt = base_prompt + avoidance_instruction

    # Compute context hash for potential caching
    context_hash = hashlib.md5(
        (counterfactual_prompt + term + "COUNTERFACTUAL").encode()
    ).hexdigest()[:12]

    # Run simulations (uses DEFAULT_SIMULATION_MODEL from mentions_models.py)
    simulator = MentionsSimulator(cache_dir=cache_dir)
    simulations = await simulator.simulate(
        event_ticker=f"{event_ticker}_cf_{term[:10]}",
        prompt=counterfactual_prompt,
        n_simulations=n_simulations,
        model=None,  # Use default from config
        context_hash=context_hash,
    )

    # Scan for leakages (term appearing despite avoidance instruction)
    leakages = 0
    leakage_contexts: List[str] = []

    for sim in simulations:
        transcript = sim.transcript
        # Case-insensitive search for the term
        pattern = re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)
        matches = list(pattern.finditer(transcript))

        if matches:
            leakages += 1
            # Get context around first match
            match = matches[0]
            start = max(0, match.start() - 50)
            end = min(len(transcript), match.end() + 50)
            context = transcript[start:end]
            leakage_contexts.append(f"...{context}...")

    leakage_rate = leakages / max(n_simulations, 1)

    # Interpret the result
    if leakage_rate >= 0.8:
        interpretation = (
            f"VERY HIGH leakage ({leakage_rate:.0%}): '{term}' is nearly unavoidable in this context. "
            f"The term is deeply embedded in the natural discourse. Expect high baseline P."
        )
    elif leakage_rate >= 0.5:
        interpretation = (
            f"HIGH leakage ({leakage_rate:.0%}): '{term}' is difficult to avoid. "
            f"The term is strongly associated with this event type. Moderate-high baseline P."
        )
    elif leakage_rate >= 0.2:
        interpretation = (
            f"MODERATE leakage ({leakage_rate:.0%}): '{term}' leaks occasionally. "
            f"The term has some natural association but isn't central. Moderate baseline P."
        )
    else:
        interpretation = (
            f"LOW leakage ({leakage_rate:.0%}): '{term}' is easy to avoid. "
            f"The term is not naturally part of this event's discourse. Low baseline P expected."
        )

    result = CounterfactualResult(
        term=term,
        n_simulations=len(simulations),
        leakages=leakages,
        leakage_rate=leakage_rate,
        leakage_contexts=leakage_contexts[:5],  # Limit to 5 examples
        interpretation=interpretation,
    )

    return {
        "event_ticker": event_ticker,
        "event_title": event_title,
        "counterfactual_result": result.to_dict(),
        "template": template.event_type,
        "domain": template.domain,
        "usage_note": (
            "High leakage_rate suggests the term is central to the event discourse. "
            "Compare with blind simulation P - they should correlate. "
            "If blind P is much lower than leakage_rate, simulations may be under-estimating."
        ),
    }


