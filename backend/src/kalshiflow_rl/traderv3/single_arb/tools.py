"""
LangChain tools for the ArbCaptain.

Module-level globals are injected at startup by the coordinator.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Set

from langchain_core.tools import tool

logger = logging.getLogger("kalshiflow_rl.traderv3.single_arb.tools")

# --- Dependency injection via module globals ---
_index = None           # EventArbIndex
_trading_client = None  # KalshiDemoTradingClient
_memory_store = None    # DualMemoryStore (file + vector)
_config = None          # V3Config


def set_dependencies(
    index=None,
    trading_client=None,
    memory_store=None,
    config=None,
) -> None:
    """Set shared dependencies for all tools."""
    global _index, _trading_client, _memory_store, _config
    if index is not None:
        _index = index
    if trading_client is not None:
        _trading_client = trading_client
    if memory_store is not None:
        _memory_store = memory_store
    if config is not None:
        _config = config


# --- Tools ---

@tool
async def get_event_snapshot(event_ticker: str) -> Dict[str, Any]:
    """Get current arb state for a specific event.

    Returns all markets with full orderbook depth, probability sums,
    edge calculations, signal operators, and recent trade activity.

    Args:
        event_ticker: The Kalshi event ticker (e.g., "KXFEDCHAIRNOM")

    Returns:
        Dict with markets (incl. depth, trades, volume), prob sums, edges, and signals
    """
    if not _index:
        return {"error": "EventArbIndex not available"}

    snapshot = _index.get_event_snapshot(event_ticker)
    if not snapshot:
        return {"error": f"Event {event_ticker} not found in index"}

    return snapshot


@tool
async def get_all_events() -> Dict[str, Any]:
    """Get summary of all monitored events with arb state.

    Returns each event's probability sums, edges, and market coverage.
    """
    if not _index:
        return {"error": "EventArbIndex not available"}

    return _index.get_snapshot()


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
    if not _index:
        return {"error": "EventArbIndex not available"}

    event_ticker = _index.get_event_for_ticker(market_ticker)
    if not event_ticker:
        return {"error": f"Market {market_ticker} not tracked"}

    event = _index.events.get(event_ticker)
    if not event:
        return {"error": f"Event {event_ticker} not found"}

    market = event.markets.get(market_ticker)
    if not market:
        return {"error": f"Market {market_ticker} not found in event"}

    return {
        "ticker": market.ticker,
        "title": market.title,
        "yes_levels": market.yes_levels,
        "no_levels": market.no_levels,
        "yes_bid": market.yes_bid,
        "yes_ask": market.yes_ask,
        "yes_bid_size": market.yes_bid_size,
        "yes_ask_size": market.yes_ask_size,
        "spread": market.spread,
        "source": market.source,
        "freshness_seconds": round(market.freshness_seconds, 1),
    }


@tool
async def get_recent_trades(event_ticker: str) -> Dict[str, Any]:
    """Get recent public trades across all markets in an event.

    Returns the last trades from the trade WS channel for each market,
    merged and sorted by timestamp (newest first).

    Args:
        event_ticker: The Kalshi event ticker

    Returns:
        Dict with trades list, per-market trade counts, and most active market
    """
    if not _index:
        return {"error": "EventArbIndex not available"}

    event = _index.events.get(event_ticker)
    if not event:
        return {"error": f"Event {event_ticker} not found"}

    all_trades = []
    market_stats = {}
    for m in event.markets.values():
        market_stats[m.ticker] = {
            "title": m.title,
            "trade_count": m.trade_count,
            "last_trade_price": m.last_trade_price,
            "last_trade_side": m.last_trade_side,
        }
        for t in m.recent_trades[:10]:
            all_trades.append({**t, "market_ticker": m.ticker, "title": m.title})

    # Sort by timestamp descending
    all_trades.sort(key=lambda t: t.get("ts", 0), reverse=True)

    most_active = event.most_active_market()
    return {
        "event_ticker": event_ticker,
        "trades": all_trades[:30],
        "trade_count_total": sum(s["trade_count"] for s in market_stats.values()),
        "market_stats": market_stats,
        "most_active_ticker": most_active.ticker if most_active else None,
    }


@tool
async def execute_arb(
    event_ticker: str,
    direction: str,
    max_contracts: int,
    reasoning: str,
) -> Dict[str, Any]:
    """Execute a multi-leg arb trade on a single event.

    LONG ARB: Buy YES on every market in the event.
    SHORT ARB: Buy NO on every market in the event.

    Each leg is placed as a limit order at the current best price.
    All legs must succeed for the arb to be profitable.

    Args:
        event_ticker: Event to trade
        direction: "long" (buy all YES) or "short" (buy all NO)
        max_contracts: Maximum contracts per leg
        reasoning: Why this trade should be made

    Returns:
        Dict with status (completed/partial/aborted), legs executed, total cost
    """
    if not _trading_client:
        return {"status": "aborted", "reason": "Trading client not available"}
    if not _index:
        return {"status": "aborted", "reason": "EventArbIndex not available"}

    event_state = _index.events.get(event_ticker)
    if not event_state:
        return {"status": "aborted", "reason": f"Event {event_ticker} not found"}

    if not event_state.all_markets_have_data:
        return {
            "status": "aborted",
            "reason": f"Not all markets have data ({event_state.markets_with_data}/{event_state.markets_total})",
        }

    # Check balance
    try:
        balance_resp = await _trading_client.get_balance()
        balance = balance_resp.get("balance", 0)
    except Exception as e:
        return {"status": "aborted", "reason": f"Balance check failed: {e}"}

    # Build legs
    legs_executed = []
    total_cost = 0
    errors = []

    for book in event_state.markets.values():
        if direction == "long":
            # Buy YES at the ask
            if book.yes_ask is None:
                errors.append(f"{book.ticker}: no YES ask")
                continue
            side = "yes"
            price = book.yes_ask
        elif direction == "short":
            # Buy NO (sell YES) → NO price = 100 - YES bid
            if book.yes_bid is None:
                errors.append(f"{book.ticker}: no YES bid")
                continue
            side = "no"
            price = 100 - book.yes_bid
        else:
            return {"status": "aborted", "reason": f"Invalid direction: {direction}"}

        # Size: minimum of max_contracts and available liquidity
        contracts = min(max_contracts, book.yes_ask_size if direction == "long" else book.yes_bid_size)
        contracts = max(contracts, 1)  # At least 1

        # Check balance for this leg
        leg_cost = contracts * price
        if total_cost + leg_cost > balance:
            errors.append(f"{book.ticker}: insufficient balance for {contracts}@{price}c")
            continue

        try:
            logger.info(
                f"[SINGLE_ARB:TRADE] action=buy ticker={book.ticker} "
                f"side={side} contracts={contracts} price={price}c"
            )
            order_resp = await _trading_client.create_order(
                ticker=book.ticker,
                action="buy",
                side=side,
                count=contracts,
                price=price,
                type="limit",
            )
            order = order_resp.get("order", order_resp)
            order_id = order.get("order_id", "")

            legs_executed.append({
                "ticker": book.ticker,
                "title": book.title,
                "side": side,
                "contracts": contracts,
                "price_cents": price,
                "order_id": order_id,
                "status": order.get("status", "placed"),
            })
            total_cost += leg_cost

        except Exception as e:
            error_msg = f"{book.ticker}: order failed: {e}"
            logger.error(f"[SINGLE_ARB:TRADE_ERROR] {error_msg}")
            errors.append(error_msg)

    status = "completed" if len(legs_executed) == event_state.markets_total else "partial"
    if len(legs_executed) == 0:
        status = "aborted"

    result = {
        "status": status,
        "event_ticker": event_ticker,
        "direction": direction,
        "legs_executed": len(legs_executed),
        "legs_total": event_state.markets_total,
        "total_cost_cents": total_cost,
        "legs": legs_executed,
        "errors": errors,
        "reasoning": reasoning,
    }

    logger.info(
        f"[SINGLE_ARB:TRADE_RESULT] event={event_ticker} direction={direction} "
        f"status={status} legs={len(legs_executed)}/{event_state.markets_total} "
        f"cost={total_cost}c"
    )

    return result


@tool
async def memory_store(
    content: str,
    memory_type: str = "learning",
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Store a learning or insight in memory.

    Use this to record patterns, mistakes, strategy adjustments, or observations
    that should persist across Captain cycles.

    Args:
        content: The learning/insight text to store
        memory_type: Category (learning, mistake, strategy, observation, trade_result)
        metadata: Optional metadata dict (event_ticker, edge, etc.)

    Returns:
        Dict with status and storage details
    """
    if not _memory_store:
        return {"error": "Memory store not available"}

    try:
        _memory_store.append(
            content=content,
            memory_type=memory_type,
            metadata=metadata,
        )
        return {"status": "stored", "type": memory_type, "file_stored": True}
    except Exception as e:
        return {"error": str(e)}


@tool
async def memory_search(
    query: str,
    limit: int = 5,
) -> Dict[str, Any]:
    """Search past experiences and learnings in memory.

    Uses semantic search (vector similarity) when available, with keyword
    fallback. Results are merged and deduplicated.

    Args:
        query: Search query (keywords or natural language)
        limit: Maximum results to return

    Returns:
        Dict with matching memory entries
    """
    if not _memory_store:
        return {"error": "Memory store not available"}

    try:
        # Use async hybrid search (semantic + keyword) if DualMemoryStore
        if hasattr(_memory_store, 'search'):
            results = await _memory_store.search(query=query, limit=limit)
        else:
            results = _memory_store.search_journal(query=query, limit=limit)
        return {
            "results": results,
            "count": len(results),
            "query": query,
        }
    except Exception as e:
        return {"error": str(e)}


@tool
async def get_positions() -> Dict[str, Any]:
    """Get current open positions on Kalshi.

    Returns all positions with market tickers and quantities.
    """
    if not _trading_client:
        return {"error": "Trading client not available"}

    try:
        resp = await _trading_client.get_positions()
        positions = resp.get("market_positions", resp.get("positions", []))
        return {
            "positions": positions[:50],
            "count": len(positions),
        }
    except Exception as e:
        return {"error": str(e)}


@tool
async def get_balance() -> Dict[str, Any]:
    """Get current account balance.

    Returns balance in cents and dollars.
    """
    if not _trading_client:
        return {"error": "Trading client not available"}

    try:
        resp = await _trading_client.get_balance()
        balance_cents = resp.get("balance", 0)
        return {
            "balance_cents": balance_cents,
            "balance_dollars": round(balance_cents / 100, 2),
        }
    except Exception as e:
        return {"error": str(e)}


# --- MemoryCurator tools ---


def _get_file_store():
    """Get the FileMemoryStore."""
    return _memory_store


@tool
async def get_memory_stats() -> Dict[str, Any]:
    """Get memory store statistics.

    Returns entry counts by type, staleness info, and storage details.
    Used by MemoryCurator to assess memory health.

    Returns:
        Dict with journal_entries, type_counts, validations_cached, etc.
    """
    if not _memory_store:
        return {"error": "Memory store not available"}
    return _memory_store.get_stats()


@tool
async def dedup_memories(similarity_threshold: float = 0.88) -> Dict[str, Any]:
    """Find and mark duplicate memories based on content similarity.

    Scans recent journal entries for near-duplicates (keyword overlap).
    Returns pairs of duplicates found for review.

    Args:
        similarity_threshold: Minimum overlap ratio to flag as duplicate (0-1)

    Returns:
        Dict with duplicate pairs found and dedup count
    """
    file_store = _get_file_store()
    if not file_store:
        return {"error": "Memory store not available"}

    entries = file_store.get_journal(limit=200)

    duplicates = []
    seen_content: Dict[str, Dict] = {}

    for entry in entries:
        content = entry.get("content", "").lower().strip()
        if not content:
            continue

        words: Set[str] = set(content.split())
        found_dup = False
        for existing_content, existing_entry in seen_content.items():
            existing_words: Set[str] = set(existing_content.split())
            if not words or not existing_words:
                continue
            overlap = len(words & existing_words) / max(len(words | existing_words), 1)
            if overlap >= similarity_threshold:
                duplicates.append({
                    "original": existing_entry.get("content", "")[:100],
                    "duplicate": content[:100],
                    "overlap": round(overlap, 3),
                })
                found_dup = True
                break

        if not found_dup:
            seen_content[content] = entry

    return {
        "duplicates_found": len(duplicates),
        "entries_scanned": len(entries),
        "pairs": duplicates[:20],
    }


@tool
async def consolidate_memories(memory_type: str = "learning") -> Dict[str, Any]:
    """Merge related memories of a given type into a summary.

    Reads all entries of a type, groups by topic similarity, and identifies
    candidates for consolidation. Does NOT auto-merge (returns recommendations).

    Args:
        memory_type: Type of memories to consolidate (learning, mistake, etc.)

    Returns:
        Dict with consolidation recommendations
    """
    file_store = _get_file_store()
    if not file_store:
        return {"error": "Memory store not available"}

    entries = file_store.get_journal(limit=100, memory_type=memory_type)

    groups: Dict[str, List[Dict]] = {}
    ungrouped = []

    for entry in entries:
        meta = entry.get("metadata", {})
        event_ticker = meta.get("event_ticker")
        ticker = meta.get("ticker") or meta.get("market_ticker")

        key = event_ticker or ticker
        if key:
            groups.setdefault(key, []).append(entry)
        else:
            ungrouped.append(entry)

    recommendations = []
    for key, group in groups.items():
        if len(group) >= 3:
            recommendations.append({
                "group_key": key,
                "count": len(group),
                "newest": group[0].get("content", "")[:100],
                "oldest": group[-1].get("content", "")[:100],
                "recommendation": "consolidate",
            })

    return {
        "type": memory_type,
        "total_entries": len(entries),
        "groups": len(groups),
        "ungrouped": len(ungrouped),
        "consolidation_candidates": len(recommendations),
        "recommendations": recommendations[:10],
    }


@tool
async def prune_stale_memories(max_age_hours: float = 168.0) -> Dict[str, Any]:
    """Identify stale memories older than max_age_hours.

    Reports stale entries but does NOT delete them (read-only analysis).
    Returns count and preview for curator to decide.

    Args:
        max_age_hours: Maximum age in hours before considering stale (default 168 = 7 days)

    Returns:
        Dict with stale entry count and previews
    """
    file_store = _get_file_store()
    if not file_store:
        return {"error": "Memory store not available"}

    entries = file_store.get_journal(limit=500)

    cutoff = time.time() - (max_age_hours * 3600)
    stale = []

    for entry in entries:
        ts = entry.get("timestamp", 0)
        if ts and ts < cutoff:
            stale.append({
                "content": entry.get("content", "")[:100],
                "type": entry.get("type"),
                "age_hours": round((time.time() - ts) / 3600, 1),
            })

    return {
        "max_age_hours": max_age_hours,
        "total_scanned": len(entries),
        "stale_count": len(stale),
        "stale_preview": stale[:15],
    }
