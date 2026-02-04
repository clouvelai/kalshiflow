"""
Read-only data snapshot tools for the arb deep agent.

Bridges shared systems (PairRegistry, SpreadMonitor, EventCodex, FileMemoryStore)
to the LangGraph agent via tool calls.

IMPORTANT: All tools MUST return bounded output to avoid LLM context overflow.
Target: each tool output < 4K tokens (~16K chars).
"""

import logging
import time
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

logger = logging.getLogger("kalshiflow_rl.traderv3.deep_agent.tools.data")

_pair_registry = None
_spread_monitor = None
_event_codex = None
_file_store = None

# Max items returned by list tools to stay within context budget
_MAX_PAIRS = 20
_MAX_SPREAD_ROWS = 20

# Validation TTL: after this many seconds, validations are considered expired
_VALIDATION_TTL_SECONDS = 1800  # 30 minutes


def set_shared_data(
    pair_registry=None,
    spread_monitor=None,
    event_codex=None,
    file_store=None,
) -> None:
    """Set shared data references for all data tools."""
    global _pair_registry, _spread_monitor, _event_codex, _file_store
    if pair_registry is not None:
        _pair_registry = pair_registry
    if spread_monitor is not None:
        _spread_monitor = spread_monitor
    if event_codex is not None:
        _event_codex = event_codex
    if file_store is not None:
        _file_store = file_store


@tool
async def get_pair_snapshot(
    pair_id: Optional[str] = None,
    event_ticker: Optional[str] = None,
) -> Dict[str, Any]:
    """Get paired markets with current spread state.

    Returns active pairs (max 20), optionally filtered by pair_id or event_ticker.
    Each pair includes: kalshi_ticker, question, match_confidence, and live
    spread state (kalshi_mid, poly_yes, spread_cents) from SpreadMonitor.

    Args:
        pair_id: Filter to a specific pair UUID
        event_ticker: Filter to pairs under a specific Kalshi event

    Returns:
        Dict with pairs list, total_count, and truncated flag
    """
    if not _pair_registry:
        return {"error": "PairRegistry not available"}

    pairs = _pair_registry.get_all_active()

    if pair_id:
        pairs = [p for p in pairs if p.id == pair_id]
    elif event_ticker:
        pairs = [p for p in pairs if p.kalshi_event_ticker == event_ticker]

    total = len(pairs)
    pairs = pairs[:_MAX_PAIRS]

    result = []
    for p in pairs:
        entry = {
            "pair_id": p.id,
            "kalshi_ticker": p.kalshi_ticker,
            "kalshi_event_ticker": p.kalshi_event_ticker,
            "poly_condition_id": p.poly_condition_id,
            "poly_token_id_yes": p.poly_token_id_yes,
            "question": p.question[:120] if p.question else None,
            "match_method": p.match_method,
            "match_confidence": p.match_confidence,
        }

        if _spread_monitor:
            spread_state = _spread_monitor.get_spread_state(p.id)
            if spread_state:
                entry["kalshi_yes_mid"] = spread_state.kalshi_yes_mid
                entry["poly_yes_cents"] = spread_state.poly_yes_cents
                entry["spread_cents"] = spread_state.spread_cents
                entry["tradeable"] = spread_state.tradeable
                entry["is_fresh"] = spread_state.is_fresh()
            else:
                entry["spread_state"] = "not_tracked"

        result.append(entry)

    return {"pairs": result, "total_count": total, "truncated": total > _MAX_PAIRS}


@tool
async def get_spread_snapshot(
    min_spread_cents: int = 0,
) -> Dict[str, Any]:
    """Get spread dashboard sorted by absolute spread (biggest opportunities first).

    Shows top 20 pairs with live Kalshi/Poly prices and computed spread.
    Use this to identify trading opportunities.

    Args:
        min_spread_cents: Only return pairs with abs(spread) >= this value (default 0 = all)

    Returns:
        Dict with spreads list (compact), total_count, and truncated flag
    """
    if not _spread_monitor:
        return {"info": "SpreadMonitor not running"}

    dashboard = _spread_monitor.get_spread_dashboard()

    if min_spread_cents > 0:
        dashboard = [
            d for d in dashboard
            if abs(d.get("spread_cents") or 0) >= min_spread_cents
        ]

    total = len(dashboard)
    dashboard = dashboard[:_MAX_SPREAD_ROWS]

    # Return only essential fields per row to stay within context budget
    # Build pair_id -> event_ticker lookup from registry
    event_lookup = {}
    if _pair_registry:
        for p in _pair_registry.get_all_active():
            event_lookup[p.id] = p.kalshi_event_ticker

    compact = []
    for d in dashboard:
        pid = d.get("pair_id")
        compact.append({
            "pair_id": pid,
            "kalshi_ticker": d.get("kalshi_ticker"),
            "kalshi_event_ticker": event_lookup.get(pid),
            "kalshi_yes_mid": d.get("kalshi_yes_mid"),
            "poly_yes_cents": d.get("poly_yes_cents"),
            "spread_cents": d.get("spread_cents"),
            "tradeable": d.get("tradeable"),
            "is_fresh": d.get("is_fresh"),
        })

    return {"spreads": compact, "total_count": total, "truncated": total > _MAX_SPREAD_ROWS}


@tool
async def get_event_codex(event_ticker: str) -> Dict[str, Any]:
    """Get EventCodex enrichment data for an event (event-level view).

    Returns Kalshi event details, Polymarket metadata, and per-market
    candlestick data for ALL markets in the event. Use this to assess
    whether the entire event is suitable for arb trading.

    Candle data adapts to whatever window the codex fetched (default 1h,
    1-min intervals). All available data points are returned.

    Args:
        event_ticker: Kalshi event ticker (e.g. 'KXFEDCHAIRNOM-26')

    Returns:
        Dict with event metadata, markets with full candle history, or error
    """
    if not _event_codex:
        return {"error": "EventCodex not available"}

    entry = _event_codex.get_entry(event_ticker)
    if not entry:
        available = list(_event_codex._cache.keys())[:20]
        return {"error": f"No codex entry for {event_ticker}", "available_events": available}

    result = {
        "event_ticker": entry.kalshi_event_ticker,
        "series_ticker": entry.series_ticker,
        "title": entry.title,
        "category": entry.category,
        "kalshi_subtitle": entry.kalshi_subtitle,
        "kalshi_mutually_exclusive": entry.kalshi_mutually_exclusive,
        "kalshi_strike_date": entry.kalshi_strike_date,
        "kalshi_markets_count": len(entry.kalshi_markets),
        "poly_event_id": entry.poly_event_id,
        "poly_title": entry.poly_title,
        "poly_description": (entry.poly_description or "")[:300],
        "poly_volume": entry.poly_volume,
        "poly_volume_24h": entry.poly_volume_24h,
        "poly_liquidity": entry.poly_liquidity,
    }

    # All markets with full candle history (adapts to codex window)
    markets = []
    for mc in entry.market_candles[:10]:  # max 10 markets per event
        m = {
            "kalshi_ticker": mc.kalshi_ticker,
            "question": mc.question[:120] if mc.question else None,
            "pair_id": mc.pair_id,
        }
        if mc.kalshi:
            m["kalshi_candles"] = [{"ts": c.ts, "close": c.close, "volume": c.volume} for c in mc.kalshi]
        if mc.poly:
            m["poly_candles"] = [{"ts": c.ts, "close": c.close} for c in mc.poly]
        markets.append(m)

    result["markets"] = markets
    return result


@tool
async def get_validation_status(event_ticker: str) -> Dict[str, Any]:
    """Check cached validation status for an event.

    Returns the EventAnalyst's previous validation (approved/rejected/unknown)
    for the entire event. Validation is event-level: all markets in an event
    are approved or rejected together.

    Args:
        event_ticker: Kalshi event ticker (e.g. 'KXFEDCHAIRNOM-29')

    Returns:
        Dict with status (approved/rejected/unknown), reasoning, validated_at
    """
    if not _file_store:
        return {"status": "unknown", "reason": "FileMemoryStore not available"}

    validation = _file_store.get_validation(event_ticker)
    if not validation:
        return {"status": "unknown", "reason": "Not yet validated by EventAnalyst"}

    age = time.time() - validation.get("validated_at", 0)
    if age > _VALIDATION_TTL_SECONDS:
        return {
            "status": "expired",
            "reason": f"Validation expired ({age / 60:.0f}m old, TTL={_VALIDATION_TTL_SECONDS // 60}m)",
            "previous": validation,
        }

    return validation
