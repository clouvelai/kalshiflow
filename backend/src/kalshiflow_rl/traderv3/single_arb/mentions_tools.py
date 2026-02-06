"""
LangChain tools for the MentionsSpecialist subagent.

Simplified tool set focused on blind LLM roleplay simulation.

Tools:
1. simulate_probability - PRIMARY: Run blind simulations and estimate probabilities
2. get_event_context - Get blind context for an event (no mention terms)
3. get_mention_context - Get term-specific context (for prior adjustment)
4. query_wordnet - Get accepted/prohibited forms for a term
5. get_mentions_rules - Get parsed settlement rules (LexemePackLite)
6. get_mentions_summary - Current count and evidence for live events

Module-level globals are injected at startup by the coordinator.
"""

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

logger = logging.getLogger("kalshiflow_rl.traderv3.single_arb.mentions_tools")

# Track in-progress background simulations
_pending_simulations: Dict[str, Dict[str, Any]] = {}  # event_ticker -> {started_at, mode, terms, ...}

# --- Dependency injection via module globals ---
_index = None           # EventArbIndex
_memory_store = None    # DualMemoryStore (file + vector)
_config = None          # V3Config
_mentions_data_dir = None  # Directory for persisting mentions state

# Simulator instance (lazy loaded)
_simulator = None

# Path to persisted mentions data
MENTIONS_STATE_FILE = "mentions_state.json"


def set_mentions_dependencies(
    index=None,
    memory_store=None,
    config=None,
    mentions_data_dir=None,
) -> None:
    """Set shared dependencies for mentions tools."""
    global _index, _memory_store, _config, _mentions_data_dir
    if index is not None:
        _index = index
    if memory_store is not None:
        _memory_store = memory_store
    if config is not None:
        _config = config
    if mentions_data_dir is not None:
        _mentions_data_dir = mentions_data_dir
        os.makedirs(mentions_data_dir, exist_ok=True)


def _get_simulator():
    """Get or create simulator instance."""
    global _simulator
    if _simulator is None:
        from .mentions_simulator import MentionsSimulator
        cache_dir = os.path.join(
            _mentions_data_dir or os.path.dirname(os.path.abspath(__file__)),
            "simulations"
        )
        _simulator = MentionsSimulator(cache_dir=cache_dir)
    return _simulator


# --- Data Structures ---

@dataclass
class LexemePackLite:
    """Simplified per-market rule object for mentions counting."""
    entity: str                              # e.g., "tariff"
    accepted_forms: List[str] = field(default_factory=list)  # ["tariff", "tariffs", "Tariff"]
    prohibited_forms: List[str] = field(default_factory=list)  # ["trade barrier", "import tax"]
    source_type: str = "any"                 # "speech" | "tweet" | "any"
    speaker: Optional[str] = None            # "Trump" | None (any speaker)
    time_window_start: Optional[str] = None  # ISO timestamp
    time_window_end: Optional[str] = None    # ISO timestamp
    raw_rules: str = ""                      # Original rules_primary text
    market_ticker: Optional[str] = None      # Associated market ticker

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LexemePackLite":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# --- Persistence Helpers ---

def _get_mentions_state_path() -> str:
    """Get path to mentions state file."""
    if _mentions_data_dir:
        return os.path.join(_mentions_data_dir, MENTIONS_STATE_FILE)
    # Fallback to module directory
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "memory", "data", MENTIONS_STATE_FILE
    )


def _load_persisted_mentions_state() -> Dict[str, Dict[str, Any]]:
    """Load persisted mentions state from disk."""
    path = _get_mentions_state_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to load mentions state: {e}")
        return {}


def _save_persisted_mentions_state(state: Dict[str, Dict[str, Any]]) -> None:
    """Save mentions state to disk."""
    path = _get_mentions_state_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
    except OSError as e:
        logger.warning(f"Failed to save mentions state: {e}")


def _sync_event_mentions_to_disk(event_ticker: str) -> None:
    """Sync a single event's mentions_data to persistent storage."""
    if not _index:
        return
    event = _index.events.get(event_ticker)
    if not event:
        return
    state = _load_persisted_mentions_state()
    state[event_ticker] = event.mentions_data
    _save_persisted_mentions_state(state)


def restore_mentions_state_from_disk() -> None:
    """Restore mentions_data for all events from persisted storage."""
    if not _index:
        logger.warning("Cannot restore mentions state: index not available")
        return
    state = _load_persisted_mentions_state()
    restored = 0
    for event_ticker, mentions_data in state.items():
        event = _index.events.get(event_ticker)
        if event:
            event.mentions_data = mentions_data
            restored += 1
    if restored > 0:
        logger.info(f"Restored mentions state for {restored} events from disk")


# --- LLM Helpers ---

async def _llm_parse_rules(rules_text: str, market_title: str) -> LexemePackLite:
    """Use LLM to parse settlement rules into LexemePackLite."""
    try:
        from langchain_anthropic import ChatAnthropic

        llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)

        prompt = f"""Analyze this Kalshi market's settlement rules and extract the counting criteria.

Market Title: {market_title}

Settlement Rules:
{rules_text}

Extract the following as JSON:
{{
    "entity": "the word/phrase being counted (e.g., 'tariff')",
    "accepted_forms": ["list", "of", "exact", "forms", "that", "count"],
    "prohibited_forms": ["synonyms", "or", "phrases", "that", "do", "NOT", "count"],
    "source_type": "speech" | "tweet" | "any",
    "speaker": "name of required speaker or null if any",
    "time_window_start": "ISO timestamp or null",
    "time_window_end": "ISO timestamp or null"
}}

CRITICAL RULES:
1. accepted_forms should include the base word, plurals, capitalizations that EXPLICITLY count per rules
2. prohibited_forms should include synonyms mentioned as NOT counting
3. If rules don't specify forms, include reasonable variations of the entity
4. Pay attention to "does not count" or "only counts if" language

Return ONLY valid JSON, no explanation."""

        response = await llm.ainvoke(prompt)
        content = response.content

        # Parse JSON from response
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        parsed = json.loads(content.strip())

        return LexemePackLite(
            entity=parsed.get("entity", ""),
            accepted_forms=parsed.get("accepted_forms", []),
            prohibited_forms=parsed.get("prohibited_forms", []),
            source_type=parsed.get("source_type", "any"),
            speaker=parsed.get("speaker"),
            time_window_start=parsed.get("time_window_start"),
            time_window_end=parsed.get("time_window_end"),
            raw_rules=rules_text,
        )

    except Exception as e:
        logger.error(f"Failed to parse rules with LLM: {e}")
        return LexemePackLite(raw_rules=rules_text)


# =============================================================================
# PRIMARY TOOL: Blind LLM Simulation
# =============================================================================

# Maximum history entries to keep per event (rolling window)
MAX_HISTORY_ENTRIES = 10


def _compute_trend(current: float, previous: float) -> str:
    """Compute trend indicator based on delta."""
    delta = current - previous
    if delta > 0.05:
        return "↑"
    elif delta < -0.05:
        return "↓"
    return "→"


def _build_deltas(
    current_estimates: Dict[str, Dict[str, Any]],
    baseline_estimates: Optional[Dict[str, Dict[str, Any]]],
    previous_estimates: Optional[Dict[str, Dict[str, Any]]],
) -> Dict[str, Dict[str, Any]]:
    """Build delta tracking dict for each term."""
    deltas = {}
    for term, est in current_estimates.items():
        current_prob = est.get("probability", 0.0)

        # Get baseline probability
        baseline_prob = 0.0
        if baseline_estimates and term in baseline_estimates:
            baseline_prob = baseline_estimates[term].get("probability", 0.0)

        # Get previous probability
        previous_prob = current_prob  # Default to current if no previous
        if previous_estimates and term in previous_estimates:
            previous_prob = previous_estimates[term].get("probability", 0.0)

        deltas[term] = {
            "current": current_prob,
            "baseline": baseline_prob,
            "previous": previous_prob,
            "delta_from_baseline": round(current_prob - baseline_prob, 4),
            "delta_from_previous": round(current_prob - previous_prob, 4),
            "trend": _compute_trend(current_prob, previous_prob),
        }
    return deltas


@tool
async def simulate_probability(
    event_ticker: str,
    terms: Optional[List[str]] = None,
    n_simulations: int = 10,
    mode: str = "informed",
    force_resimulate: bool = False,
) -> Dict[str, Any]:
    """Estimate mention probability via LLM roleplay simulation.

    This is the PRIMARY tool for mentions probability estimation.

    MODES:
    - "blind": Baseline probability (generic event context, no term news)
             First blind run becomes the stable baseline.
    - "informed": Context-adjusted probability (includes term relevance and news)
                 Returns deltas showing change from baseline and previous.

    HISTORY TRACKING:
    - baseline_estimates: First blind run (stable reference)
    - current_estimates: Latest run (updated each call)
    - estimate_history: Rolling window of last 10 refreshes with news context

    DELTA TRACKING:
    Returns deltas showing probability changes:
    - delta_from_baseline: How much context shifted P from initial blind
    - delta_from_previous: How much P changed since last refresh
    - trend: ↑ (increase >5%), ↓ (decrease >5%), → (stable)

    Args:
        event_ticker: Kalshi event ticker
        terms: Specific terms to estimate (default: all from lexeme pack)
        n_simulations: Number of simulations (default: 10, min for trading)
        mode: "blind" or "informed" (default: "informed")
        force_resimulate: Ignore cache and re-run simulations

    Returns:
        Dict with probability estimates, deltas from baseline/previous,
        news_context that informed the run, and history count.

    Example:
        >>> result = await simulate_probability("KXSBLX", ["Taylor Swift"], 10, "informed")
        >>> result["deltas"]["Taylor Swift"]
        {"current": 0.45, "baseline": 0.10, "previous": 0.40,
         "delta_from_baseline": 0.35, "delta_from_previous": 0.05, "trend": "↑"}
    """
    if not _index:
        return {"error": "EventArbIndex not available"}

    event = _index.events.get(event_ticker)
    if not event:
        return {"error": f"Event {event_ticker} not found"}

    # Get terms from lexeme pack if not specified
    if not terms:
        mentions_data = event.mentions_data or {}
        lexeme_pack = mentions_data.get("lexeme_pack", {})
        if lexeme_pack:
            entity = lexeme_pack.get("entity", "")
            terms = lexeme_pack.get("accepted_forms", [entity]) if entity else []

        if not terms:
            return {
                "error": "No terms specified and no lexeme_pack found. Provide terms or parse rules first.",
                "hint": "Use get_mentions_rules() to see if rules have been parsed.",
            }

    # Validate mode (removed "compare" - use separate blind + informed calls instead)
    if mode not in ["blind", "informed"]:
        return {"error": f"Invalid mode: {mode}. Use 'blind' or 'informed'."}

    # Run simulation
    try:
        from .mentions_simulator import run_mentions_simulation

        simulator = _get_simulator()
        if force_resimulate:
            simulator.clear_cache(event_ticker)

        result = await run_mentions_simulation(
            event_ticker=event_ticker,
            event_title=event.title,
            mention_terms=terms,
            n_simulations=n_simulations,
            cache_dir=os.path.join(_mentions_data_dir or "", "simulations") if _mentions_data_dir else None,
            mode=mode,
        )

        # Initialize mentions_data if needed
        if event.mentions_data is None:
            event.mentions_data = {}

        current_estimates = result.get("estimates", {})
        news_snippets = result.get("news_snippets", [])[:3]  # Top 3 news headlines
        current_ts = time.time()

        # --- History Tracking Logic ---

        # Get existing history data
        baseline_estimates = event.mentions_data.get("baseline_estimates")
        previous_estimates = event.mentions_data.get("current_estimates")
        estimate_history = event.mentions_data.get("estimate_history", [])

        # For blind mode: set as baseline if no baseline exists
        if mode == "blind":
            if baseline_estimates is None:
                event.mentions_data["baseline_estimates"] = current_estimates
                baseline_estimates = current_estimates
                logger.info(f"[MENTIONS] Set baseline estimates for {event_ticker}")

        # Update current estimates
        event.mentions_data["current_estimates"] = current_estimates

        # Add to history (rolling window)
        history_entry = {
            "ts": current_ts,
            "mode": mode,
            "estimates": {term: est.get("probability", 0.0) for term, est in current_estimates.items()},
            "news_snippets": news_snippets,
            "n_simulations": n_simulations,
        }
        estimate_history.append(history_entry)

        # Keep only last MAX_HISTORY_ENTRIES
        if len(estimate_history) > MAX_HISTORY_ENTRIES:
            estimate_history = estimate_history[-MAX_HISTORY_ENTRIES:]
        event.mentions_data["estimate_history"] = estimate_history

        # Update timestamps
        event.mentions_data["last_simulation_ts"] = current_ts
        event.mentions_data["last_refresh_ts"] = current_ts
        event.mentions_data["simulation_mode"] = mode

        # Persist to disk
        _sync_event_mentions_to_disk(event_ticker)

        # --- Build Deltas ---
        deltas = _build_deltas(current_estimates, baseline_estimates, previous_estimates)

        # --- Build Response ---
        response = {
            "event_ticker": event_ticker,
            "terms_analyzed": terms,
            "mode": mode,
            "n_simulations": result.get("n_simulations", n_simulations),
            "estimates": current_estimates,
            "deltas": deltas,
            "news_context": news_snippets,
            "history_count": len(estimate_history),
            "context_hash": result.get("context_hash", ""),
            "template": result.get("template", ""),
            "domain": result.get("domain", ""),
        }

        # Add usage guidance based on mode
        if mode == "blind":
            response["usage_note"] = (
                "BLIND mode: Stable baseline established. "
                "Use mode='informed' for context-aware refreshes with delta tracking."
            )
        else:
            response["usage_note"] = (
                "INFORMED mode: Context-aware estimate with news. "
                "Check 'deltas' for change from baseline and previous. "
                "Positive delta_from_baseline = context increases P(mention)."
            )

        return response

    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        return {"error": f"Simulation failed: {e}"}


# =============================================================================
# ASYNC SIMULATION TOOLS
# =============================================================================


async def _run_simulation_background(
    event_ticker: str,
    event_title: str,
    terms: List[str],
    n_simulations: int,
    mode: str,
) -> None:
    """Background simulation runner - writes results to event.mentions_data."""
    global _pending_simulations
    try:
        from .mentions_simulator import run_mentions_simulation

        simulator = _get_simulator()

        result = await run_mentions_simulation(
            event_ticker=event_ticker,
            event_title=event_title,
            mention_terms=terms,
            n_simulations=n_simulations,
            cache_dir=os.path.join(_mentions_data_dir or "", "simulations") if _mentions_data_dir else None,
            mode=mode,
        )

        # Store results in event.mentions_data (same logic as simulate_probability)
        if _index:
            event = _index.events.get(event_ticker)
            if event:
                if event.mentions_data is None:
                    event.mentions_data = {}

                current_estimates = result.get("estimates", {})
                news_snippets = result.get("news_snippets", [])[:3]
                current_ts = time.time()

                baseline_estimates = event.mentions_data.get("baseline_estimates")
                previous_estimates = event.mentions_data.get("current_estimates")
                estimate_history = event.mentions_data.get("estimate_history", [])

                # For blind mode: set as baseline if no baseline exists
                if mode == "blind" and baseline_estimates is None:
                    event.mentions_data["baseline_estimates"] = current_estimates
                    logger.info(f"[MENTIONS] Background: Set baseline estimates for {event_ticker}")

                # Update current estimates
                event.mentions_data["current_estimates"] = current_estimates

                # Add to history
                history_entry = {
                    "ts": current_ts,
                    "mode": mode,
                    "estimates": {term: est.get("probability", 0.0) for term, est in current_estimates.items()},
                    "news_snippets": news_snippets,
                    "n_simulations": n_simulations,
                    "async": True,
                }
                estimate_history.append(history_entry)
                if len(estimate_history) > MAX_HISTORY_ENTRIES:
                    estimate_history = estimate_history[-MAX_HISTORY_ENTRIES:]
                event.mentions_data["estimate_history"] = estimate_history

                event.mentions_data["last_simulation_ts"] = current_ts
                event.mentions_data["last_refresh_ts"] = current_ts
                event.mentions_data["simulation_mode"] = mode

                _sync_event_mentions_to_disk(event_ticker)

        logger.info(f"[MENTIONS] Background simulation complete for {event_ticker} (mode={mode})")

    except Exception as e:
        logger.error(f"[MENTIONS] Background simulation failed for {event_ticker}: {e}")
    finally:
        # Remove from pending
        _pending_simulations.pop(event_ticker, None)


@tool
async def trigger_simulation(
    event_ticker: str,
    terms: Optional[List[str]] = None,
    n_simulations: int = 5,
    mode: str = "informed",
) -> Dict[str, Any]:
    """Start a simulation in the background (non-blocking).

    Returns IMMEDIATELY. The simulation runs asynchronously.
    Check get_mentions_summary() later for results.

    Use this when you want to start a simulation without waiting.
    Captain can continue with other work and check results next cycle.

    Args:
        event_ticker: Kalshi event ticker
        terms: Specific terms to estimate (default: from lexeme pack)
        n_simulations: Number of simulations (default: 5)
        mode: "blind" (baseline) or "informed" (context-aware)

    Returns:
        Dict with triggered status and estimated completion time.

    Example:
        >>> await trigger_simulation("KXSBLX", mode="blind", n_simulations=10)
        {"status": "triggered", "check_after_seconds": 20, ...}

        # Later, check results:
        >>> await get_mentions_summary("KXSBLX")
        {"current_estimates": {...}, ...}
    """
    global _pending_simulations

    if not _index:
        return {"error": "EventArbIndex not available"}

    event = _index.events.get(event_ticker)
    if not event:
        return {"error": f"Event {event_ticker} not found"}

    # Check if already running
    if event_ticker in _pending_simulations:
        pending = _pending_simulations[event_ticker]
        elapsed = time.time() - pending["started_at"]
        return {
            "status": "already_running",
            "event_ticker": event_ticker,
            "mode": pending["mode"],
            "elapsed_seconds": round(elapsed, 1),
            "usage_note": "A simulation is already in progress. Check get_mentions_summary() for results.",
        }

    # Get terms from lexeme pack if not specified
    if not terms:
        mentions_data = event.mentions_data or {}
        lexeme_pack = mentions_data.get("lexeme_pack", {})
        if lexeme_pack:
            entity = lexeme_pack.get("entity", "")
            terms = lexeme_pack.get("accepted_forms", [entity]) if entity else []

        if not terms:
            return {
                "error": "No terms specified and no lexeme_pack found. Provide terms or parse rules first.",
                "hint": "Use get_mentions_rules() to see if rules have been parsed.",
            }

    # Validate mode
    if mode not in ["blind", "informed"]:
        return {"error": f"Invalid mode: {mode}. Use 'blind' or 'informed'."}

    # Track as pending
    _pending_simulations[event_ticker] = {
        "started_at": time.time(),
        "mode": mode,
        "terms": terms,
        "n_simulations": n_simulations,
    }

    # Start background task
    loop = asyncio.get_running_loop()
    loop.create_task(_run_simulation_background(
        event_ticker=event_ticker,
        event_title=event.title,
        terms=terms,
        n_simulations=n_simulations,
        mode=mode,
    ))

    # Estimate completion time (roughly 2 seconds per simulation)
    estimated_seconds = n_simulations * 2

    return {
        "status": "triggered",
        "event_ticker": event_ticker,
        "mode": mode,
        "terms": terms,
        "n_simulations": n_simulations,
        "estimated_completion_seconds": estimated_seconds,
        "usage_note": f"Simulation started in background. Check get_mentions_summary() in ~{estimated_seconds}s for results.",
    }


# =============================================================================
# CONTEXT TOOLS
# =============================================================================

@tool
async def get_event_context(
    event_ticker: str,
) -> Dict[str, Any]:
    """Get BLIND event context for simulation.

    Returns context about the event (teams, venue, historical notes).
    This is domain-specific fact gathering used as the foundation
    for both blind and informed simulations.

    Use this to understand what context the simulation is using.

    Args:
        event_ticker: Kalshi event ticker

    Returns:
        Dict with EventContext
    """
    if not _index:
        return {"error": "EventArbIndex not available"}

    event = _index.events.get(event_ticker)
    if not event:
        return {"error": f"Event {event_ticker} not found"}

    try:
        from .mentions_context import gather_blind_context

        context = await gather_blind_context(
            event_ticker=event_ticker,
            event_title=event.title,
        )

        return {
            "event_ticker": event_ticker,
            "context": context.to_dict(),
            "note": "This is the base event context used for blind simulation (baseline probability).",
        }

    except Exception as e:
        logger.error(f"Context gathering failed: {e}")
        return {"error": f"Context gathering failed: {e}"}


@tool
async def get_mention_context(
    event_ticker: str,
    terms: List[str],
) -> Dict[str, Any]:
    """Get term-specific context for prior adjustment.

    Returns WHY each term might appear in the event, based on:
    - Wikipedia context (is term related to event?)
    - Historical mentions (has term been said before?)
    - Related entities (what might trigger the term?)

    This is SEPARATE from simulation - used for:
    1. Prior adjustment (relevant terms get boosted)
    2. Post-hoc interpretation
    3. Edge confidence

    Args:
        event_ticker: Kalshi event ticker
        terms: List of terms to research

    Returns:
        Dict mapping term -> TermRelevance
    """
    if not _index:
        return {"error": "EventArbIndex not available"}

    event = _index.events.get(event_ticker)
    if not event:
        return {"error": f"Event {event_ticker} not found"}

    try:
        from .mentions_context import gather_mention_contexts

        contexts = await gather_mention_contexts(
            event_ticker=event_ticker,
            event_title=event.title,
            mention_terms=terms,
        )

        return {
            "event_ticker": event_ticker,
            "term_contexts": {term: ctx.to_dict() for term, ctx in contexts.items()},
            "usage_note": (
                "relevance_score indicates connection strength. "
                "1.0 = direct participant, 0.7 = related person, "
                "0.5 = current news, 0.2 = common phrase, 0.1 = unrelated."
            ),
        }

    except Exception as e:
        logger.error(f"Mention context failed: {e}")
        return {"error": f"Mention context failed: {e}"}


# =============================================================================
# SEMANTIC TOOLS
# =============================================================================

@tool
async def query_wordnet(term: str) -> Dict[str, Any]:
    """Query WordNet for semantic context about a mention term.

    Use this to understand:
    - What synonyms DON'T count (settlement rules require exact words)
    - What related terms might co-occur
    - What morphological forms exist (plurals, verb forms)

    Args:
        term: The word to analyze

    Returns:
        Dict with synonyms, hypernyms, related terms, accepted forms
    """
    try:
        from .mentions_semantic import (
            get_semantic_context,
            get_prohibited_forms,
            get_accepted_forms,
            is_wordnet_available,
        )

        if not is_wordnet_available():
            return {
                "error": "WordNet not available",
                "hint": "WordNet may still be downloading. Try again in a moment.",
            }

        ctx = get_semantic_context(term)
        prohibited = get_prohibited_forms(term)
        accepted = get_accepted_forms(term)

        return {
            "term": term,
            "semantic_context": ctx.to_dict(),
            "prohibited_forms": prohibited,
            "accepted_forms": accepted,
            "usage_note": (
                "prohibited_forms are synonyms that do NOT count per strict settlement rules. "
                "accepted_forms are morphological variants that likely DO count."
            ),
        }

    except Exception as e:
        logger.error(f"WordNet query failed: {e}")
        return {"error": f"WordNet query failed: {e}"}


# =============================================================================
# RULES AND STATE TOOLS
# =============================================================================

@tool
async def get_mentions_rules(event_ticker: str) -> Dict[str, Any]:
    """Get the parsed settlement rules (LexemePackLite) for a mentions event.

    Returns the entity being counted, accepted forms, prohibited forms,
    and other constraints parsed from the market's rules_primary.

    Args:
        event_ticker: The Kalshi event ticker

    Returns:
        Dict with lexeme_pack (rules), or error if not a mentions event
    """
    if not _index:
        return {"error": "EventArbIndex not available"}

    event = _index.events.get(event_ticker)
    if not event:
        return {"error": f"Event {event_ticker} not found"}

    mentions_data = event.mentions_data
    if not mentions_data:
        return {
            "error": f"Event {event_ticker} has no mentions_data. May not be a mentions market.",
            "hint": "Ensure the event was identified as a mentions market during loading.",
        }

    lexeme_pack = mentions_data.get("lexeme_pack")
    if not lexeme_pack:
        return {
            "error": "No lexeme_pack found. Rules may not have been parsed yet.",
            "mentions_data_keys": list(mentions_data.keys()),
        }

    # Include simulation estimates if available
    sim_estimates = mentions_data.get("simulation_estimates", {})
    baseline_estimates = mentions_data.get("baseline_estimates", {})
    current_estimates = mentions_data.get("current_estimates", {})
    history_count = len(mentions_data.get("estimate_history", []))

    return {
        "event_ticker": event_ticker,
        "lexeme_pack": lexeme_pack,
        "simulation_estimates": sim_estimates,
        "baseline_estimates": baseline_estimates,
        "current_estimates": current_estimates,
        "history_count": history_count,
        "last_simulation_ts": mentions_data.get("last_simulation_ts"),
        "last_refresh_ts": mentions_data.get("last_refresh_ts"),
        "current_count": mentions_data.get("current_count", 0),
        "evidence_count": len(mentions_data.get("evidence", [])),
    }


@tool
async def get_mentions_summary(event_ticker: str) -> Dict[str, Any]:
    """Get a summary of current mentions state for an event.

    Returns current count, evidence list, simulation estimates, history, and lexeme pack info.
    Also shows if a background simulation is in progress.

    Args:
        event_ticker: The event to summarize

    Returns:
        Dict with full mentions state summary including:
        - baseline_estimates: First blind run (stable reference)
        - current_estimates: Latest simulation result
        - estimate_history: Rolling window of last 10 refreshes
        - last_refresh_ts: Timestamp of most recent refresh
        - simulation_in_progress: True if async simulation is running
    """
    global _pending_simulations

    if not _index:
        return {"error": "EventArbIndex not available"}

    event = _index.events.get(event_ticker)
    if not event:
        return {"error": f"Event {event_ticker} not found"}

    # Check if simulation is in progress
    pending = _pending_simulations.get(event_ticker)
    simulation_in_progress = pending is not None
    pending_info = None
    if pending:
        elapsed = time.time() - pending["started_at"]
        pending_info = {
            "mode": pending["mode"],
            "elapsed_seconds": round(elapsed, 1),
            "n_simulations": pending["n_simulations"],
        }

    mentions_data = event.mentions_data
    if not mentions_data:
        return {
            "event_ticker": event_ticker,
            "is_mentions_market": False,
            "current_count": 0,
            "evidence": [],
            "simulation_estimates": {},
            "baseline_estimates": {},
            "current_estimates": {},
            "estimate_history": [],
            "history_count": 0,
            "simulation_in_progress": simulation_in_progress,
            "pending_simulation": pending_info,
        }

    lexeme_pack = mentions_data.get("lexeme_pack", {})
    sim_estimates = mentions_data.get("simulation_estimates", {})
    baseline_estimates = mentions_data.get("baseline_estimates", {})
    current_estimates = mentions_data.get("current_estimates", {})
    estimate_history = mentions_data.get("estimate_history", [])

    return {
        "event_ticker": event_ticker,
        "is_mentions_market": bool(lexeme_pack),
        "entity": lexeme_pack.get("entity", ""),
        "accepted_forms": lexeme_pack.get("accepted_forms", []),
        "current_count": mentions_data.get("current_count", 0),
        "evidence_count": len(mentions_data.get("evidence", [])),
        "evidence": mentions_data.get("evidence", [])[-10:],  # Last 10 for brevity
        # Simulation data
        "simulation_estimates": sim_estimates,
        "baseline_estimates": baseline_estimates,
        "current_estimates": current_estimates,
        "estimate_history": estimate_history[-5:],  # Last 5 for brevity in response
        "history_count": len(estimate_history),
        # Timestamps
        "last_simulation_ts": mentions_data.get("last_simulation_ts"),
        "last_refresh_ts": mentions_data.get("last_refresh_ts"),
        "simulation_mode": mentions_data.get("simulation_mode", ""),
        # Async status
        "simulation_in_progress": simulation_in_progress,
        "pending_simulation": pending_info,
    }


# =============================================================================
# EDGE DETECTION TOOL
# =============================================================================

@tool
async def compute_edge(
    term: str,
    baseline_probability: float,
    informed_probability: float,
    market_yes_price: int,
    market_no_price: int,
    confidence: float = 0.5,
    baseline_weight: float = 0.6,
    spread_adjustment: int = 2,
) -> Dict[str, Any]:
    """Compute edge between blended probability and market price.

    RECENCY BIAS CORRECTION:
    News makes terms seem more likely than they are. The baseline (blind) estimate
    is grounded in historical broadcast patterns. We blend both:
    - baseline_probability: What history suggests (default 60% weight)
    - informed_probability: What current news suggests (default 40% weight)
    - blended_probability: Used for edge calculation

    Edge = blended_fair_value - market_price - spread - fees

    Only recommend trades where edge exceeds spread + fees + buffer.

    Args:
        term: The mention term
        baseline_probability: P(term) from blind simulation (stable baseline)
        informed_probability: P(term) from informed simulation (context-aware)
        market_yes_price: Current market YES price (cents)
        market_no_price: Current market NO price (cents)
        confidence: Confidence in simulation estimate (0.0 to 1.0)
        baseline_weight: How much to weight baseline (default 0.6 = 60%)
        spread_adjustment: Additional cents to require beyond spread

    Returns:
        Dict with edge analysis, recency bias adjustment, and trade recommendation

    Example:
        >>> await compute_edge("Taylor Swift", baseline_probability=0.10,
        ...                    informed_probability=0.45, market_yes_price=60, ...)
        {"blended_probability": 0.24, "recency_bias_adjustment": 0.21, ...}
    """
    # Blend baseline and informed to correct for recency bias
    blended_probability = (
        baseline_weight * baseline_probability +
        (1 - baseline_weight) * informed_probability
    )

    # Recency bias adjustment = how much informed overestimates vs blended
    recency_bias_adjustment = informed_probability - blended_probability

    # Convert blended probability to fair value in cents
    yes_fair = round(blended_probability * 100)
    no_fair = 100 - yes_fair

    # Calculate raw edge (using blended fair value)
    yes_edge = yes_fair - market_yes_price
    no_edge = no_fair - market_no_price

    # Calculate spread
    spread = 100 - market_yes_price - market_no_price

    # Fee per contract (typical Kalshi fee)
    fee_per_side = 7  # cents

    # Minimum edge required: spread + fees + buffer
    min_edge = spread + fee_per_side + spread_adjustment

    # Apply confidence discount
    confidence_discount = 1.0 - confidence
    adjusted_yes_edge = yes_edge * (1 - confidence_discount * 0.5)
    adjusted_no_edge = no_edge * (1 - confidence_discount * 0.5)

    result = {
        "term": term,
        # Probability estimates
        "baseline_probability": round(baseline_probability, 4),
        "informed_probability": round(informed_probability, 4),
        "blended_probability": round(blended_probability, 4),
        "baseline_weight": baseline_weight,
        "recency_bias_adjustment": round(recency_bias_adjustment, 4),
        # Fair values (from blended)
        "yes_fair_value_cents": yes_fair,
        "no_fair_value_cents": no_fair,
        # Market state
        "market_yes_price": market_yes_price,
        "market_no_price": market_no_price,
        "spread": spread,
        # Edge calculations
        "yes_edge": yes_edge,
        "no_edge": no_edge,
        "min_edge_required": min_edge,
        "confidence": confidence,
        # Recommendation
        "recommendation": "PASS",
        "reason": "",
    }

    # Make recommendation
    if adjusted_yes_edge >= min_edge:
        result["recommendation"] = "BUY_YES"
        result["reason"] = f"YES edge ({yes_edge}c) exceeds min required ({min_edge}c). Blended P={blended_probability:.0%} vs market {market_yes_price}c."
    elif adjusted_no_edge >= min_edge:
        result["recommendation"] = "BUY_NO"
        result["reason"] = f"NO edge ({no_edge}c) exceeds min required ({min_edge}c). Blended P={1-blended_probability:.0%} vs market {market_no_price}c."
    elif confidence < 0.6:
        result["reason"] = f"Confidence too low ({confidence:.2f}). Need more simulations."
    else:
        result["reason"] = f"Edge ({max(yes_edge, no_edge)}c) below min required ({min_edge}c). Blended P={blended_probability:.0%}."

    # Add recency bias warning if significant
    if recency_bias_adjustment > 0.15:
        result["recency_bias_warning"] = (
            f"High recency bias detected: informed ({informed_probability:.0%}) >> baseline ({baseline_probability:.0%}). "
            f"News may be inflating P(mention). Blended estimate ({blended_probability:.0%}) corrects for this."
        )

    return result


# =============================================================================
# DEPRECATED TOOLS (kept for backward compatibility during transition)
# =============================================================================

@tool
async def record_evidence(
    event_ticker: str,
    hit: Dict[str, Any],
    source_url: str,
) -> Dict[str, Any]:
    """Record a confirmed mention hit to memory store.

    DEPRECATED: This tool is from the old live-counting approach.
    The new simulation approach estimates probability pre-event.

    Args:
        event_ticker: The event this mention belongs to
        hit: Confirmed hit dict (must have span, start_char, end_char, reason)
        source_url: URL of the source document

    Returns:
        Dict with recorded status
    """
    if not _index:
        return {"error": "EventArbIndex not available"}

    event = _index.events.get(event_ticker)
    if not event:
        return {"error": f"Event {event_ticker} not found"}

    mentions_data = event.mentions_data or {}

    # Create evidence entry
    evidence_id = str(uuid.uuid4())[:8]
    evidence_entry = {
        "id": evidence_id,
        "span": hit.get("span", ""),
        "start_char": hit.get("start_char", 0),
        "end_char": hit.get("end_char", 0),
        "context": hit.get("context", ""),
        "source_url": source_url,
        "source_type": hit.get("source_type", "unknown"),
        "reasoning": hit.get("reason", ""),
        "confidence": hit.get("confidence", 0.0),
        "recorded_at": datetime.utcnow().isoformat(),
    }

    # Initialize fields if needed
    if "evidence" not in mentions_data:
        mentions_data["evidence"] = []
    if "current_count" not in mentions_data:
        mentions_data["current_count"] = 0

    mentions_data["evidence"].append(evidence_entry)
    mentions_data["current_count"] = mentions_data.get("current_count", 0) + 1
    mentions_data["last_scan_ts"] = time.time()

    event.mentions_data = mentions_data
    _sync_event_mentions_to_disk(event_ticker)

    return {
        "recorded": True,
        "evidence_id": evidence_id,
        "new_total_count": mentions_data["current_count"],
        "note": "DEPRECATED: Prefer simulate_probability() for probability estimation.",
    }
