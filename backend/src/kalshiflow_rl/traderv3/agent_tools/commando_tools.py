"""Module-level @tool functions for the TradeCommando subagent.

Each function delegates to a single CommandoToolContext instance.
"""

from typing import Any, Dict, Optional

from langchain_core.tools import tool

from .context import CommandoToolContext

# Single module-level context - set by coordinator at startup
_ctx: Optional[CommandoToolContext] = None


def set_context(ctx: CommandoToolContext) -> None:
    """Set the shared CommandoToolContext. Called by coordinator."""
    global _ctx
    _ctx = ctx


@tool
async def place_order(
    ticker: str,
    side: str,
    contracts: int,
    price_cents: int,
    reasoning: str,
    action: str = "buy",
) -> Dict[str, Any]:
    """Place a single order. Auto-sets configurable TTL and uses session order group.
    Auto-records trade to memory.

    Args:
        ticker: Market ticker
        side: "yes" or "no"
        contracts: Number of contracts (1-100)
        price_cents: Limit price in cents (1-99)
        reasoning: Trade thesis (REQUIRED - stored in memory)
        action: "buy" or "sell" (default "buy")
    """
    if not _ctx:
        return {"error": "Context not initialized"}
    return await _ctx.place_order(
        ticker=ticker,
        side=side,
        contracts=contracts,
        price_cents=price_cents,
        reasoning=reasoning,
        action=action,
    )


@tool
async def execute_arb(
    event_ticker: str,
    direction: str,
    max_contracts: int,
    reasoning: str,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Execute a multi-leg arb trade on a single event.

    LONG ARB: Buy YES on every market in the event.
    SHORT ARB: Buy NO on every market in the event.

    Args:
        event_ticker: Event to trade
        direction: "long" (buy all YES) or "short" (buy all NO)
        max_contracts: Maximum contracts per leg
        reasoning: Why this trade should be made
        dry_run: If True, compute everything but don't place orders (preview mode)
    """
    if not _ctx:
        return {"error": "Context not initialized"}
    return await _ctx.execute_arb(
        event_ticker=event_ticker,
        direction=direction,
        max_contracts=max_contracts,
        reasoning=reasoning,
        dry_run=dry_run,
    )


@tool
async def cancel_order(order_id: str, reason: str = "") -> Dict[str, Any]:
    """Cancel a resting order.

    Args:
        order_id: The order ID to cancel
        reason: Optional reason for cancellation
    """
    if not _ctx:
        return {"error": "Context not initialized"}
    return await _ctx.cancel_order(order_id, reason=reason)


@tool
async def get_resting_orders(ticker: Optional[str] = None) -> Dict[str, Any]:
    """Get currently open/resting orders with queue position info.

    Args:
        ticker: Filter to specific market ticker (optional)
    """
    if not _ctx:
        return {"error": "Context not initialized"}
    return await _ctx.get_resting_orders(ticker=ticker)


@tool
async def get_market_orderbook(market_ticker: str) -> Dict[str, Any]:
    """Get full orderbook depth for a single market.

    Args:
        market_ticker: The Kalshi market ticker
    """
    if not _ctx:
        return {"error": "Context not initialized"}
    return await _ctx.get_market_orderbook(market_ticker)


@tool
async def get_recent_trades(event_ticker: str) -> Dict[str, Any]:
    """Get recent public trades across all markets in an event.

    Args:
        event_ticker: The Kalshi event ticker
    """
    if not _ctx:
        return {"error": "Context not initialized"}
    return await _ctx.get_recent_trades(event_ticker)


@tool
async def get_balance() -> Dict[str, Any]:
    """Get current account balance."""
    if not _ctx:
        return {"error": "Context not initialized"}
    return await _ctx.get_balance()


@tool
async def get_positions() -> Dict[str, Any]:
    """Get positions for tracked events with realtime P&L."""
    if not _ctx:
        return {"error": "Context not initialized"}
    return await _ctx.get_positions()


@tool
async def record_learning(
    content: str,
    category: str = "learning",
    target_file: str = "AGENTS.md",
) -> Dict[str, Any]:
    """Record a learning or insight to memory.

    Args:
        content: The learning/insight text to store
        category: Category (learning, mistake, strategy, observation, trade_result)
        target_file: Which memory file this relates to (AGENTS.md, SIGNALS.md, PLAYBOOK.md)
    """
    if not _ctx:
        return {"error": "Context not initialized"}
    return _ctx.record_learning(content, category=category, target_file=target_file)
