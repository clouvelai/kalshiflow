"""Polymarket read-only tools for the arb deep agent."""

import logging
from typing import Any, Dict, List

from langchain_core.tools import tool

logger = logging.getLogger("kalshiflow_rl.traderv3.deep_agent.tools.poly")

_poly_client = None


def set_poly_client(client: Any) -> None:
    """Set the Polymarket client instance for all Poly tools."""
    global _poly_client
    _poly_client = client


@tool
async def poly_get_events(limit: int = 50, active: bool = True) -> List[Dict[str, Any]]:
    """Get active events from Polymarket (Gamma API).

    Each event contains nested markets with condition_ids, token_ids, and prices.
    Use this to browse what's available on Polymarket before matching to Kalshi.

    Args:
        limit: Maximum events to return
        active: Only return active events

    Returns:
        List of event dicts with title, slug, markets[] (each with condition_id, clobTokenIds)
    """
    if not _poly_client:
        logger.warning("poly_get_events called but Polymarket client not initialized")
        return [{"error": "Polymarket client not available"}]
    try:
        events = await _poly_client.get_events(limit=limit, active=active)
        return events or []
    except Exception as e:
        return [{"error": str(e)}]


@tool
async def poly_get_markets(limit: int = 100, active: bool = True) -> List[Dict[str, Any]]:
    """Get active markets from Polymarket (Gamma API).

    Returns individual markets (outcomes). Each has a condition_id and clobTokenIds.
    For event-level browsing, use poly_get_events instead.

    Args:
        limit: Maximum markets to return
        active: Only return active markets

    Returns:
        List of market dicts with question, condition_id, clobTokenIds, outcomePrices
    """
    if not _poly_client:
        return [{"error": "Polymarket client not available"}]
    try:
        markets = await _poly_client.get_markets(limit=limit, active=active)
        return markets or []
    except Exception as e:
        return [{"error": str(e)}]


@tool
async def poly_search_events(query: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Search Polymarket events by text query.

    Use this to find Polymarket equivalents of Kalshi events. For example,
    search 'Fed Chair' to find all Fed Chair nomination events.

    Args:
        query: Search query string (e.g. 'Fed Chair', 'Bitcoin price')
        limit: Maximum results

    Returns:
        List of matching event dicts with nested markets
    """
    if not _poly_client:
        return [{"error": "Polymarket client not available"}]
    try:
        events = await _poly_client.search_events(query=query, limit=limit)
        return events or []
    except Exception as e:
        return [{"error": str(e)}]


