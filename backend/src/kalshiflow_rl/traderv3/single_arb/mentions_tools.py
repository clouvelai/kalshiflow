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
_file_store = None      # FileMemoryStore (journal.jsonl)
_config = None          # V3Config
_mentions_data_dir = None  # Directory for persisting mentions state
_broadcast_callback = None  # Async callback to broadcast mentions updates to frontend
_budget_manager = None  # SimulationBudgetManager (optional)

# Simulator instance (lazy loaded)
_simulator = None

# Path to persisted mentions data
MENTIONS_STATE_FILE = "mentions_state.json"


def set_mentions_dependencies(
    index=None,
    file_store=None,
    config=None,
    mentions_data_dir=None,
    budget_manager=None,
) -> None:
    """Set shared dependencies for mentions tools."""
    global _index, _file_store, _config, _mentions_data_dir, _budget_manager
    if index is not None:
        _index = index
    if file_store is not None:
        _file_store = file_store
    if config is not None:
        _config = config
    if mentions_data_dir is not None:
        _mentions_data_dir = mentions_data_dir
        os.makedirs(mentions_data_dir, exist_ok=True)
    if budget_manager is not None:
        _budget_manager = budget_manager


def set_mentions_broadcast_callback(callback) -> None:
    """Set the broadcast callback for emitting mentions updates to frontend."""
    global _broadcast_callback
    _broadcast_callback = callback


async def _broadcast_mentions_update(
    event_ticker: str,
    current_estimates: Dict[str, Dict[str, Any]],
    mode: str,
    baseline_estimates: Optional[Dict[str, Dict[str, Any]]],
    deltas: Dict[str, Dict[str, Any]],
    news_snippets: List[str],
    history_count: int,
    simulation_in_progress: bool = False,
) -> None:
    """Broadcast mentions update to frontend via WebSocket."""
    if not _broadcast_callback:
        return

    try:
        # Build terms list with probability and confidence interval
        terms = []
        for term, est in current_estimates.items():
            terms.append({
                "term": term,
                "probability": est.get("probability", 0),
                "confidence_interval": est.get("confidence_interval", [0, 1]),
                "n_simulations": est.get("n_simulations", 0),
            })

        await _broadcast_callback({
            "type": "mentions_update",
            "data": {
                "event_ticker": event_ticker,
                "terms": terms,
                "mode": mode,
                "baseline_estimates": baseline_estimates or {},
                "deltas": deltas,
                "news_context": news_snippets[:3],
                "history_count": history_count,
                "simulation_in_progress": simulation_in_progress,
            }
        })
    except Exception as e:
        logger.debug(f"Failed to broadcast mentions update: {e}")


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
        from .mentions_models import get_extraction_llm

        llm = get_extraction_llm()  # Uses DEFAULT_EXTRACTION_MODEL from config

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

        from .llm_schemas import LexemePackExtraction

        structured_llm = llm.with_structured_output(LexemePackExtraction)
        result = await structured_llm.ainvoke(prompt)

        return LexemePackLite(
            entity=result.entity,
            accepted_forms=result.accepted_forms,
            prohibited_forms=result.prohibited_forms,
            source_type=result.source_type,
            speaker=result.speaker,
            time_window_start=result.time_window_start,
            time_window_end=result.time_window_end,
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
    logger.info(f"[MENTIONS:SIMULATE] event={event_ticker} terms={terms} n={n_simulations} mode={mode}")

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

    # Check budget before running
    if _budget_manager:
        budget_status = _budget_manager.get_budget_status(event_ticker)
        if budget_status and budget_status.get("simulations_remaining", 999) <= 0:
            return {
                "error": "Simulation budget exhausted for this event",
                "budget": budget_status,
                "hint": "Budget is $1/event. All simulations have been used.",
            }

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
            mentions_data=event.mentions_data,  # Pass for speaker persona extraction
        )

        # Initialize mentions_data if needed
        if event.mentions_data is None:
            event.mentions_data = {}

        current_estimates = result.get("estimates", {})
        news_snippets = result.get("news_snippets", [])[:3]  # Top 3 news headlines

        # Merge event-level news from understanding (richer signal, no extra API calls)
        if _index:
            understanding = getattr(event, "understanding", None) or {}
            for article in understanding.get("news_articles", [])[:3]:
                headline = article.get("title", "")
                if headline and headline not in news_snippets:
                    news_snippets.append(headline[:150])
            news_snippets = news_snippets[:5]  # Cap at 5 total

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

        # --- Broadcast to frontend ---
        await _broadcast_mentions_update(
            event_ticker=event_ticker,
            current_estimates=current_estimates,
            mode=mode,
            baseline_estimates=baseline_estimates,
            deltas=deltas,
            news_snippets=news_snippets,
            history_count=len(estimate_history),
            simulation_in_progress=False,
        )

        # --- Record against budget ---
        if _budget_manager:
            _budget_manager.record_simulation(event_ticker, n_simulations, mode)

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

        # Add budget info
        if _budget_manager:
            budget_status = _budget_manager.get_budget_status(event_ticker)
            if budget_status:
                response["budget"] = {
                    "phase": budget_status["phase"],
                    "simulations_used": budget_status["total_simulations"],
                    "simulations_remaining": budget_status["simulations_remaining"],
                    "cost_used": budget_status["total_estimated_cost"],
                    "budget_pct_used": budget_status["budget_pct_used"],
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
    mentions_data: Optional[Dict[str, Any]] = None,
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
            mentions_data=mentions_data,  # Pass for speaker persona extraction
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

                # Build deltas and broadcast to frontend
                deltas = _build_deltas(current_estimates, baseline_estimates, previous_estimates)
                await _broadcast_mentions_update(
                    event_ticker=event_ticker,
                    current_estimates=current_estimates,
                    mode=mode,
                    baseline_estimates=event.mentions_data.get("baseline_estimates"),
                    deltas=deltas,
                    news_snippets=news_snippets,
                    history_count=len(estimate_history),
                    simulation_in_progress=False,
                )

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
    logger.info(f"[MENTIONS:TRIGGER] event={event_ticker} terms={terms} n={n_simulations} mode={mode}")
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

    # Check budget before triggering
    if _budget_manager:
        budget_status = _budget_manager.get_budget_status(event_ticker)
        if budget_status and budget_status.get("simulations_remaining", 999) <= 0:
            return {
                "status": "budget_exhausted",
                "event_ticker": event_ticker,
                "budget": budget_status,
                "usage_note": "Budget exhausted. No more simulations will be scheduled.",
            }

    # Track as pending
    _pending_simulations[event_ticker] = {
        "started_at": time.time(),
        "mode": mode,
        "terms": terms,
        "n_simulations": n_simulations,
    }

    # Broadcast simulation_in_progress status to frontend
    mentions_data = event.mentions_data or {}
    current_estimates = mentions_data.get("current_estimates", {})
    baseline_estimates = mentions_data.get("baseline_estimates", {})
    await _broadcast_mentions_update(
        event_ticker=event_ticker,
        current_estimates=current_estimates,
        mode=mode,
        baseline_estimates=baseline_estimates,
        deltas={},
        news_snippets=[],
        history_count=len(mentions_data.get("estimate_history", [])),
        simulation_in_progress=True,
    )

    # Start background task
    loop = asyncio.get_running_loop()
    loop.create_task(_run_simulation_background(
        event_ticker=event_ticker,
        event_title=event.title,
        terms=terms,
        n_simulations=n_simulations,
        mode=mode,
        mentions_data=event.mentions_data,  # Pass for speaker persona extraction
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

    # Budget info
    budget_info = None
    if _budget_manager:
        budget_info = _budget_manager.get_budget_status(event_ticker)

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
        # Budget
        "budget": budget_info,
    }


# =============================================================================
# GET_MENTIONS_STATUS TOOL (Simplified Entry Point)
# =============================================================================

# Staleness threshold in seconds (5 minutes)
STALENESS_THRESHOLD_SECONDS = 300


@tool
async def get_mentions_status(event_ticker: str) -> Dict[str, Any]:
    """Get mentions probability status. Returns cached values IMMEDIATELY.

    THIS IS THE RECOMMENDED ENTRY POINT for the MentionsSpecialist.
    Handles the complexity of baseline establishment and refresh triggering.

    BEHAVIOR:
    - If NO BASELINE exists: Runs SYNCHRONOUS blind simulation (blocks ~20s)
      This establishes the stable baseline required for edge calculation.
    - If BASELINE exists but STALE (>5 min): Triggers BACKGROUND informed refresh
      You get current values NOW, refresh happens async.
    - If BASELINE exists and FRESH: Returns immediately with all data.

    RETURNS:
    - baseline_estimates: Stable blind probabilities (reference point)
    - current_estimates: Latest context-aware probabilities
    - has_baseline: True if baseline exists (REQUIRED for trading)
    - ready_for_trading: True if baseline exists AND confidence >= 0.6
    - refresh_triggered: True if background refresh was started
    - simulation_stale: True if >5 min since last simulation

    USE THIS FOR EDGE CALCULATION:
    Once ready_for_trading=True, call compute_edge() with:
    - baseline_probability from baseline_estimates
    - informed_probability from current_estimates (or baseline if no current)
    - market_yes_price, market_no_price from orderbook

    Args:
        event_ticker: Kalshi event ticker

    Returns:
        Dict with baseline/current estimates and status flags

    Example:
        >>> status = await get_mentions_status("KXSBLX")
        >>> if status["ready_for_trading"]:
        ...     baseline_p = status["baseline_estimates"]["Taylor Swift"]["probability"]
        ...     current_p = status["current_estimates"]["Taylor Swift"]["probability"]
        ...     edge = await compute_edge("Taylor Swift", baseline_p, current_p, 60, 42)
    """
    global _pending_simulations
    logger.info(f"[MENTIONS:STATUS] event={event_ticker}")

    if not _index:
        return {"error": "EventArbIndex not available"}

    event = _index.events.get(event_ticker)
    if not event:
        return {"error": f"Event {event_ticker} not found"}

    mentions_data = event.mentions_data
    if not mentions_data:
        return {
            "error": f"Event {event_ticker} has no mentions_data. May not be a mentions market.",
            "is_mentions_market": False,
        }

    lexeme_pack = mentions_data.get("lexeme_pack")
    if not lexeme_pack:
        return {
            "error": "No lexeme_pack found. Rules may not have been parsed yet.",
            "is_mentions_market": False,
        }

    entity = lexeme_pack.get("entity", "")
    terms = lexeme_pack.get("accepted_forms", [entity]) if entity else []

    if not terms:
        return {
            "error": "No terms found in lexeme_pack.",
            "is_mentions_market": True,
            "has_baseline": False,
        }

    baseline_estimates = mentions_data.get("baseline_estimates", {})
    current_estimates = mentions_data.get("current_estimates", {})
    last_sim_ts = mentions_data.get("last_simulation_ts", 0)
    is_stale = (time.time() - last_sim_ts) > STALENESS_THRESHOLD_SECONDS if last_sim_ts else True

    # Check if simulation is already in progress
    pending = _pending_simulations.get(event_ticker)
    simulation_in_progress = pending is not None

    # --- BASELINE ESTABLISHMENT (SYNCHRONOUS if missing) ---
    needs_baseline = not baseline_estimates
    baseline_just_established = False

    if needs_baseline and not simulation_in_progress:
        logger.info(f"[MENTIONS:STATUS] No baseline for {event_ticker}, running SYNC blind simulation...")
        try:
            from .mentions_simulator import run_mentions_simulation

            result = await run_mentions_simulation(
                event_ticker=event_ticker,
                event_title=event.title,
                mention_terms=terms[:5],
                n_simulations=10,  # Full baseline
                cache_dir=os.path.join(_mentions_data_dir or "", "simulations") if _mentions_data_dir else None,
                mode="blind",
                mentions_data=mentions_data,  # Pass for speaker persona extraction
            )

            # Store baseline
            new_estimates = result.get("estimates", {})
            if new_estimates:
                mentions_data["baseline_estimates"] = new_estimates
                mentions_data["current_estimates"] = new_estimates  # Initialize current too
                mentions_data["last_simulation_ts"] = time.time()
                mentions_data["simulation_mode"] = "blind"
                event.mentions_data = mentions_data
                _sync_event_mentions_to_disk(event_ticker)

                baseline_estimates = new_estimates
                current_estimates = new_estimates
                baseline_just_established = True
                is_stale = False

                logger.info(
                    f"[MENTIONS:STATUS] Baseline established for {event_ticker}: "
                    f"P({entity})={new_estimates.get(entity, {}).get('probability', 'N/A')}"
                )

        except Exception as e:
            logger.error(f"[MENTIONS:STATUS] Baseline establishment failed: {e}")
            return {
                "error": f"Failed to establish baseline: {e}",
                "is_mentions_market": True,
                "has_baseline": False,
            }

    # --- BACKGROUND REFRESH (ASYNC if stale) ---
    refresh_triggered = False
    if baseline_estimates and is_stale and not simulation_in_progress and not baseline_just_established:
        # Trigger background informed refresh
        logger.info(f"[MENTIONS:STATUS] Triggering background refresh for {event_ticker}")
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_run_simulation_background(
                event_ticker=event_ticker,
                event_title=event.title,
                terms=terms[:5],
                n_simulations=5,  # Lighter refresh
                mode="informed",
                mentions_data=mentions_data,  # Pass for speaker persona extraction
            ))
            refresh_triggered = True
        except Exception as e:
            logger.warning(f"[MENTIONS:STATUS] Failed to trigger background refresh: {e}")

    # --- BUILD RESPONSE ---

    # Get primary entity probability for quick reference
    baseline_p = baseline_estimates.get(entity, {}).get("probability") if entity else None
    current_p = current_estimates.get(entity, {}).get("probability") if entity else None

    # Confidence for trading decision
    confidence = baseline_estimates.get(entity, {}).get("confidence_interval", [0, 1])
    ci_width = confidence[1] - confidence[0] if len(confidence) == 2 else 1.0
    confidence_score = max(0.0, 1.0 - ci_width)  # Narrower CI = higher confidence

    # Ready for trading if we have baseline AND reasonable confidence
    n_sims = baseline_estimates.get(entity, {}).get("n_simulations", 0) if entity else 0
    ready_for_trading = bool(baseline_estimates) and n_sims >= 10 and confidence_score >= 0.6

    # Build budget info
    budget_info = None
    if _budget_manager:
        budget_info = _budget_manager.get_budget_status(event_ticker)

    return {
        "event_ticker": event_ticker,
        "is_mentions_market": True,
        "entity": entity,
        "terms": terms[:5],

        # Probability estimates
        "baseline_estimates": baseline_estimates,
        "current_estimates": current_estimates,
        "baseline_probability": baseline_p,
        "current_probability": current_p,

        # Status flags
        "has_baseline": bool(baseline_estimates),
        "simulation_stale": is_stale,
        "refresh_triggered": refresh_triggered,
        "simulation_in_progress": simulation_in_progress,
        "baseline_just_established": baseline_just_established,

        # Trading readiness
        "ready_for_trading": ready_for_trading,
        "confidence_score": round(confidence_score, 2),
        "n_simulations": n_sims,

        # Budget
        "budget": budget_info,

        # Timestamps
        "last_simulation_ts": last_sim_ts,
        "staleness_threshold_seconds": STALENESS_THRESHOLD_SECONDS,

        # Usage guidance
        "usage_note": (
            "If ready_for_trading=True, call compute_edge(term, baseline_probability, "
            "current_probability, market_yes, market_no) for edge calculation."
            if ready_for_trading else
            "Baseline established. If stale, background refresh triggered. "
            "Check again next cycle or wait for refresh."
            if baseline_estimates else
            "No baseline yet. This call should have established it (check for errors)."
        ),
    }


# =============================================================================
# COUNTER-FACTUAL SIMULATION TOOL
# =============================================================================


@tool
async def run_counterfactual(
    event_ticker: str,
    term: str,
    n_simulations: int = 5,
) -> Dict[str, Any]:
    """Run counter-factual simulation to measure avoidance difficulty.

    This tool tests how HARD it is to avoid mentioning a term.
    It generates transcripts with explicit "DO NOT say X" instructions,
    then counts how often the LLM "leaks" and says it anyway.

    INTERPRETATION:
    - High leakage (>80%): Term is nearly unavoidable - deeply embedded in discourse
    - Moderate leakage (20-50%): Term has natural association but isn't central
    - Low leakage (<20%): Term is easy to avoid - not naturally part of discourse

    USE CASES:
    1. Validate blind simulation results (leakage should correlate with P)
    2. Detect terms that are so central they're nearly unavoidable
    3. Understand the "floor" probability for a term
    4. Identify when blind P might be under-estimating

    Args:
        event_ticker: Kalshi event ticker
        term: Single term to test avoidance of
        n_simulations: Number of simulations (default 5 for speed)

    Returns:
        Dict with leakage_rate, leakage_contexts, interpretation

    Example:
        >>> result = await run_counterfactual("KXSBLX", "Super Bowl", 5)
        >>> result["counterfactual_result"]["leakage_rate"]
        0.8  # 80% leakage - "Super Bowl" is nearly unavoidable in Super Bowl broadcast
    """
    logger.info(f"[MENTIONS:COUNTERFACTUAL] event={event_ticker} term={term} n={n_simulations}")

    if not _index:
        return {"error": "EventArbIndex not available"}

    event = _index.events.get(event_ticker)
    if not event:
        return {"error": f"Event {event_ticker} not found"}

    try:
        from .mentions_simulator import run_counterfactual_simulation

        result = await run_counterfactual_simulation(
            event_ticker=event_ticker,
            event_title=event.title,
            term=term,
            n_simulations=n_simulations,
            cache_dir=os.path.join(_mentions_data_dir or "", "simulations") if _mentions_data_dir else None,
            mentions_data=event.mentions_data,  # Pass for speaker persona extraction
        )

        # Add comparison guidance
        cf_result = result.get("counterfactual_result", {})
        leakage_rate = cf_result.get("leakage_rate", 0)

        # Get baseline probability for comparison if available
        baseline_p = None
        mentions_data = event.mentions_data or {}
        baseline_estimates = mentions_data.get("baseline_estimates", {})
        if term in baseline_estimates:
            baseline_p = baseline_estimates[term].get("probability")

        comparison_note = ""
        if baseline_p is not None:
            if leakage_rate > baseline_p + 0.2:
                comparison_note = (
                    f"WARNING: Leakage ({leakage_rate:.0%}) >> Blind P ({baseline_p:.0%}). "
                    f"Blind simulation may be under-estimating. Consider weighting toward leakage."
                )
            elif leakage_rate < baseline_p - 0.2:
                comparison_note = (
                    f"NOTE: Leakage ({leakage_rate:.0%}) << Blind P ({baseline_p:.0%}). "
                    f"Blind P may reflect informed context effect. Results are consistent."
                )
            else:
                comparison_note = (
                    f"Leakage ({leakage_rate:.0%}) ≈ Blind P ({baseline_p:.0%}). "
                    f"Results are consistent - good confidence in probability estimate."
                )

        result["baseline_probability"] = baseline_p
        result["comparison_note"] = comparison_note

        return result

    except Exception as e:
        logger.error(f"Counterfactual simulation failed: {e}")
        return {"error": f"Counterfactual simulation failed: {e}"}


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
    ci_lower: Optional[float] = None,
    ci_upper: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute edge between blended probability and market price.

    CONFIDENCE INTERVAL GATE:
    If the market price falls within the simulation's confidence interval,
    there is NO detectable edge - the market and model agree within uncertainty.
    This prevents trading on point estimates when the true probability is uncertain.

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
        ci_lower: Lower bound of confidence interval (from baseline_estimates CI). If None, uses heuristic.
        ci_upper: Upper bound of confidence interval (from baseline_estimates CI). If None, uses heuristic.

    Returns:
        Dict with edge analysis, recency bias adjustment, and trade recommendation

    Example:
        >>> await compute_edge("Taylor Swift", baseline_probability=0.10,
        ...                    informed_probability=0.45, market_yes_price=60, ...,
        ...                    ci_lower=0.0, ci_upper=0.278)
        {"recommendation": "PASS", "reason": "Market price within confidence interval..."}
    """
    logger.info(f"[MENTIONS:EDGE] term={term} baseline={baseline_probability} informed={informed_probability} yes={market_yes_price} no={market_no_price}")

    # Blend baseline and informed to correct for recency bias
    blended_probability = (
        baseline_weight * baseline_probability +
        (1 - baseline_weight) * informed_probability
    )

    # Recency bias adjustment = how much informed overestimates vs blended
    recency_bias_adjustment = informed_probability - blended_probability

    # --- CONFIDENCE INTERVAL GATE ---
    # If CI not provided, compute a heuristic CI from baseline_probability
    # Wilson score interval approximation for small samples
    import math
    if ci_lower is None or ci_upper is None:
        # Default to wide interval reflecting uncertainty
        # Approximate using normal approximation with n=10 simulations
        n = 10  # default simulation count
        p = baseline_probability
        z = 1.96  # 95% CI
        denominator = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / denominator
        spread_ci = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator
        ci_lower = max(0.0, center - spread_ci)
        ci_upper = min(1.0, center + spread_ci)

    market_yes_frac = market_yes_price / 100.0
    market_within_ci = ci_lower <= market_yes_frac <= ci_upper

    # Convert blended probability to fair value in cents
    yes_fair = round(blended_probability * 100)
    no_fair = 100 - yes_fair

    # Calculate raw edge (using blended fair value)
    yes_edge = yes_fair - market_yes_price
    no_edge = no_fair - market_no_price

    # Calculate spread
    spread = 100 - market_yes_price - market_no_price

    # Kalshi taker fee: roundup(0.07 * contracts * price * (1-price))
    # For 1 contract: max fee is 2c (at 50c), min is 1c at extremes
    price_frac = market_yes_price / 100.0
    fee_per_side = max(1, math.ceil(0.07 * 1 * price_frac * (1 - price_frac) * 100))

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
        # Confidence interval
        "confidence_interval": [round(ci_lower, 4), round(ci_upper, 4)],
        "market_within_ci": market_within_ci,
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

    # --- CI GATE: If market price is within confidence interval, NO EDGE ---
    if market_within_ci:
        result["recommendation"] = "PASS"
        result["reason"] = (
            f"Market price ({market_yes_price}c = {market_yes_frac:.0%}) is WITHIN the simulation's "
            f"confidence interval [{ci_lower:.1%}, {ci_upper:.1%}]. No detectable edge. "
            f"The market and your model agree within uncertainty. "
            f"When your model says {baseline_probability:.0%} but CI extends to {ci_upper:.0%}, "
            f"a market price of {market_yes_price}c is not mispriced - your estimate is just uncertain."
        )
        return result

    # Make recommendation
    if adjusted_yes_edge >= min_edge:
        result["recommendation"] = "BUY_YES"
        result["reason"] = f"YES edge ({yes_edge}c) exceeds min required ({min_edge}c). Blended P={blended_probability:.0%} vs market {market_yes_price}c. Market OUTSIDE CI [{ci_lower:.1%}, {ci_upper:.1%}]."
    elif adjusted_no_edge >= min_edge:
        result["recommendation"] = "BUY_NO"
        result["reason"] = f"NO edge ({no_edge}c) exceeds min required ({min_edge}c). Blended P={1-blended_probability:.0%} vs market {market_no_price}c. Market OUTSIDE CI [{ci_lower:.1%}, {ci_upper:.1%}]."
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
