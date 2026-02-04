"""Database tools for the arb deep agent."""

import logging
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

logger = logging.getLogger("kalshiflow_rl.traderv3.deep_agent.tools.db")

_supabase_client = None
_pair_registry = None
_spread_monitor = None
_state_container = None


def set_dependencies(
    supabase: Any = None,
    pair_registry: Any = None,
    spread_monitor: Any = None,
    state_container: Any = None,
) -> None:
    """Set dependencies for DB tools."""
    global _supabase_client, _pair_registry, _spread_monitor, _state_container
    if supabase is not None:
        _supabase_client = supabase
    if pair_registry is not None:
        _pair_registry = pair_registry
    if spread_monitor is not None:
        _spread_monitor = spread_monitor
    if state_container is not None:
        _state_container = state_container


@tool
async def get_pair_history(pair_id: str, limit: int = 100) -> List[Dict[str, Any]]:
    """Get historical price ticks for a paired market.

    Args:
        pair_id: UUID of the paired market
        limit: Maximum ticks to return

    Returns:
        List of price tick dicts sorted by timestamp desc
    """
    if not _supabase_client:
        return [{"error": "Database not available"}]
    try:
        result = _supabase_client.table("price_ticks").select("*").eq(
            "pair_id", pair_id
        ).order("created_at", desc=True).limit(limit).execute()
        return result.data or []
    except Exception as e:
        return [{"error": str(e)}]


@tool
async def log_trade(
    pair_id: str,
    kalshi_ticker: str,
    side: str,
    action: str,
    contracts: int,
    price_cents: int,
    reasoning: str,
    spread_at_entry: Optional[int] = None,
) -> Dict[str, Any]:
    """Record an arb trade with reasoning.

    Args:
        pair_id: UUID of the paired market
        kalshi_ticker: Kalshi market ticker
        side: 'yes' or 'no'
        action: 'buy' or 'sell'
        contracts: Number of contracts
        price_cents: Trade price in cents
        reasoning: Why this trade was made
        spread_at_entry: Spread when trade was initiated

    Returns:
        Dict with trade_id and status
    """
    if not _supabase_client:
        return {"error": "Database not available"}
    try:
        result = _supabase_client.table("arb_trades").insert({
            "pair_id": pair_id,
            "kalshi_ticker": kalshi_ticker,
            "side": side,
            "action": action,
            "contracts": contracts,
            "price_cents": price_cents,
            "spread_at_entry": spread_at_entry,
            "reasoning": reasoning,
            "status": "logged",
        }).execute()
        trade_data = result.data[0] if result.data else {}
        return {"trade_id": trade_data.get("id"), "status": "logged"}
    except Exception as e:
        return {"error": str(e)}


@tool
async def get_system_state() -> Dict[str, Any]:
    """Get combined system state: balance, positions, pairs, recent trades.

    Returns:
        Dict with balance, positions, active_pairs, recent_trades
    """
    state = {}

    if _state_container and _state_container.trading_state:
        ts = _state_container.trading_state
        state["balance_cents"] = ts.balance
        state["portfolio_value_cents"] = ts.portfolio_value
        state["position_count"] = ts.position_count
        state["order_count"] = ts.order_count

    if _pair_registry:
        state["active_pairs"] = _pair_registry.count
        state["pairs"] = [
            {"kalshi_ticker": p.kalshi_ticker, "question": p.question, "confidence": p.match_confidence}
            for p in _pair_registry.get_all_active()[:20]
        ]

    if _spread_monitor:
        state["top_spreads"] = _spread_monitor.get_spread_dashboard()[:10]
        state["spread_monitor"] = _spread_monitor.get_status()

    if _supabase_client:
        try:
            result = _supabase_client.table("arb_trades").select("*").order(
                "created_at", desc=True
            ).limit(10).execute()
            state["recent_trades"] = result.data or []
        except Exception:
            state["recent_trades"] = []

    return state
