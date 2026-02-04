"""Kalshi API tools for the arb deep agent."""

import logging
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

logger = logging.getLogger("kalshiflow_rl.traderv3.deep_agent.tools.kalshi")

_trading_client = None


def set_trading_client(client: Any) -> None:
    """Set the trading client instance for all Kalshi tools."""
    global _trading_client
    _trading_client = client


@tool
async def kalshi_get_events(
    limit: int = 50,
    status: str = "open",
    series_ticker: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Get Kalshi events with their nested markets.

    Each event contains multiple markets (outcomes). Use this to browse
    what's available on Kalshi before matching to Polymarket.

    Args:
        limit: Maximum events to return
        status: Event status filter (open, closed, settled)
        series_ticker: Filter by series ticker

    Returns:
        List of event dicts with event_ticker, title, category, markets[]
    """
    if not _trading_client:
        logger.warning("kalshi_get_events called but trading client not initialized")
        return [{"error": "Trading client not available"}]
    try:
        result = await _trading_client.get_events(
            limit=limit,
            status=status,
            with_nested_markets=True,
            series_ticker=series_ticker,
        )
        # Demo client returns {"events": [...], "cursor": "..."}
        if isinstance(result, dict):
            return result.get("events", [])
        return result or []
    except Exception as e:
        return [{"error": str(e)}]


@tool
async def kalshi_get_markets(
    event_ticker: Optional[str] = None,
    series_ticker: Optional[str] = None,
    status: str = "open",
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """Get Kalshi markets with optional filters.

    Use event_ticker to get all outcome markets under a specific event.

    Args:
        event_ticker: Filter by event ticker (e.g. 'KXFEDCHAIRNOM')
        series_ticker: Filter by series ticker
        status: Market status (open, closed, settled)
        limit: Maximum markets to return

    Returns:
        List of market dicts with ticker, title, yes_bid, yes_ask, volume
    """
    if not _trading_client:
        return [{"error": "Trading client not available"}]
    try:
        result = await _trading_client.get_markets(
            limit=limit,
            status=status,
            event_ticker=event_ticker,
            series_ticker=series_ticker,
        )
        # Demo client returns {"markets": [...], "cursor": "..."}
        if isinstance(result, dict):
            return result.get("markets", [])
        return result or []
    except Exception as e:
        return [{"error": str(e)}]


@tool
async def kalshi_get_orderbook(ticker: str) -> Dict[str, Any]:
    """Get the current orderbook for a specific Kalshi market ticker.

    Args:
        ticker: Market ticker (e.g., 'INXD-25FEB03-B7700')

    Returns:
        Dict with yes_bids, yes_asks arrays and metadata
    """
    if not _trading_client:
        return {"error": "Trading client not available"}
    try:
        book = await _trading_client.get_orderbook(ticker=ticker)
        return book or {}
    except Exception as e:
        return {"error": str(e)}


@tool
async def kalshi_create_order(
    ticker: str,
    side: str,
    action: str = "buy",
    count: int = 10,
    price_cents: int = 50,
    order_type: str = "limit",
) -> Dict[str, Any]:
    """Place an order on Kalshi.

    Args:
        ticker: Market ticker
        side: 'yes' or 'no'
        action: 'buy' or 'sell'
        count: Number of contracts
        price_cents: Limit price in cents (1-99)
        order_type: 'limit' or 'market'

    Returns:
        Order result with order_id, status
    """
    if not _trading_client:
        return {"error": "Trading client not available"}
    try:
        result = await _trading_client.create_order(
            ticker=ticker,
            side=side,
            action=action,
            count=count,
            type=order_type,
            yes_price=price_cents if side == "yes" else None,
            no_price=price_cents if side == "no" else None,
        )
        return result or {"status": "submitted"}
    except Exception as e:
        return {"error": str(e)}


@tool
async def kalshi_cancel_order(order_id: str) -> Dict[str, Any]:
    """Cancel a resting order on Kalshi.

    Args:
        order_id: The order ID to cancel

    Returns:
        Cancellation result
    """
    if not _trading_client:
        return {"error": "Trading client not available"}
    try:
        result = await _trading_client.cancel_order(order_id=order_id)
        return result or {"status": "cancelled"}
    except Exception as e:
        return {"error": str(e)}


@tool
async def kalshi_get_balance() -> Dict[str, Any]:
    """Get account balance and portfolio value.

    Returns:
        Dict with balance, portfolio_value (in cents)
    """
    if not _trading_client:
        return {"error": "Trading client not available"}
    try:
        balance = await _trading_client.get_balance()
        return balance or {}
    except Exception as e:
        return {"error": str(e)}


@tool
async def kalshi_get_positions() -> List[Dict[str, Any]]:
    """Get all open positions.

    Returns:
        List of position dicts with ticker, side, quantity, avg_price
    """
    if not _trading_client:
        return [{"error": "Trading client not available"}]
    try:
        positions = await _trading_client.get_positions()
        return positions or []
    except Exception as e:
        return [{"error": str(e)}]


@tool
async def kalshi_get_settlements(limit: int = 50) -> List[Dict[str, Any]]:
    """Get settlement history for resolved markets.

    Args:
        limit: Maximum settlements to return

    Returns:
        List of settlement dicts with ticker, result, payout
    """
    if not _trading_client:
        return [{"error": "Trading client not available"}]
    try:
        settlements = await _trading_client.get_settlements(limit=limit)
        return settlements or []
    except Exception as e:
        return [{"error": str(e)}]


@tool
async def kalshi_get_fills(limit: int = 50) -> List[Dict[str, Any]]:
    """Get recent order fill history.

    Args:
        limit: Maximum fills to return

    Returns:
        List of fill dicts with ticker, side, price, count
    """
    if not _trading_client:
        return [{"error": "Trading client not available"}]
    try:
        fills = await _trading_client.get_fills(limit=limit)
        return fills or []
    except Exception as e:
        return [{"error": str(e)}]
