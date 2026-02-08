"""Module-level @tool functions for the Captain agent.

Each function delegates to a single CaptainToolContext instance.
This reduces 10+ module globals to 1 while staying 100% compatible
with create_deep_agent() and LangChain's @tool decorator.
"""

from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

from .context import CaptainToolContext

# Single module-level context - set by coordinator at startup
_ctx: Optional[CaptainToolContext] = None


def set_context(ctx: CaptainToolContext) -> None:
    """Set the shared CaptainToolContext. Called by coordinator."""
    global _ctx
    _ctx = ctx


@tool
async def get_events_summary() -> List[Dict[str, Any]]:
    """Get a compact summary of all monitored events for scanning.

    Returns a lightweight list with edge calculations, data coverage,
    and MENTIONS DATA per event. Use this for initial scanning, then
    drill into specific events with get_event_snapshot().

    Returns:
        List of dicts with event_ticker, title, market_count, edge info, mentions data
    """
    if not _ctx:
        return {"error": "Context not initialized"}
    return await _ctx.get_events_summary()


@tool
async def get_event_snapshot(event_ticker: str, _ts: Optional[str] = None) -> Dict[str, Any]:
    """Get current arb state for a specific event. ALWAYS returns real-time data.

    Returns all markets with full orderbook depth, probability sums,
    edge calculations, signal operators, and recent trade activity.

    Args:
        event_ticker: The Kalshi event ticker (e.g., "KXFEDCHAIRNOM")
        _ts: Cache-buster. Pass current timestamp to ensure fresh data.

    Returns:
        Dict with markets (incl. depth, trades, volume), prob sums, edges, and signals
    """
    _ = _ts
    if not _ctx:
        return {"error": "Context not initialized"}
    return await _ctx.get_event_snapshot(event_ticker)


@tool
async def get_market_orderbook(market_ticker: str) -> Dict[str, Any]:
    """Get full orderbook depth for a single market.

    Returns up to 5 levels of YES bids (yes_levels) and NO bids (no_levels),
    plus BBO and spread.

    Args:
        market_ticker: The Kalshi market ticker

    Returns:
        Dict with yes_levels, no_levels, BBO, spread, and freshness
    """
    if not _ctx:
        return {"error": "Context not initialized"}
    return await _ctx.get_market_orderbook(market_ticker)


@tool
async def get_trade_history(ticker: Optional[str] = None, limit: int = 20) -> Dict[str, Any]:
    """Get fills, settlements, and P&L. This is your report card.

    Args:
        ticker: Filter to specific market ticker (optional)
        limit: Maximum fills to return (default 20)
    """
    if not _ctx:
        return {"error": "Context not initialized"}
    return await _ctx.get_trade_history(ticker=ticker, limit=limit)


@tool
async def get_positions() -> Dict[str, Any]:
    """Get positions for tracked events with realtime P&L.

    Cost from API (total_traded), current value from live orderbook.
    """
    if not _ctx:
        return {"error": "Context not initialized"}
    return await _ctx.get_positions()


@tool
async def get_balance() -> Dict[str, Any]:
    """Get current account balance.

    Returns balance in cents and dollars.
    """
    if not _ctx:
        return {"error": "Context not initialized"}
    return await _ctx.get_balance()


@tool
async def update_understanding(
    event_ticker: str,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """Rebuild the structured understanding for an event.

    Use this to refresh event context (Wikipedia, LLM synthesis, extensions).

    Args:
        event_ticker: The Kalshi event ticker
        force_refresh: If True, bypass cache and rebuild from scratch

    Returns:
        Dict with understanding summary or error
    """
    if not _ctx:
        return {"error": "Context not initialized"}
    return await _ctx.update_understanding(event_ticker, force_refresh=force_refresh)
