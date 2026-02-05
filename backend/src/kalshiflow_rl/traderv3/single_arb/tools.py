"""
LangChain tools for the ArbCaptain.

Module-level globals are injected at startup by the coordinator.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

logger = logging.getLogger("kalshiflow_rl.traderv3.single_arb.tools")

# --- Dependency injection via module globals ---
_index = None           # EventArbIndex
_trading_client = None  # KalshiDemoTradingClient
_memory_store = None    # DualMemoryStore (file + vector)
_config = None          # V3Config
_order_group_id = None  # Session order group ID
_order_ttl = 60         # Default TTL in seconds (1 min for demo)


def set_dependencies(
    index=None,
    trading_client=None,
    memory_store=None,
    config=None,
    order_group_id=None,
    order_ttl=None,
) -> None:
    """Set shared dependencies for all tools."""
    global _index, _trading_client, _memory_store, _config, _order_group_id, _order_ttl
    if index is not None:
        _index = index
    if trading_client is not None:
        _trading_client = trading_client
    if memory_store is not None:
        _memory_store = memory_store
    if config is not None:
        _config = config
    if order_group_id is not None:
        _order_group_id = order_group_id
    if order_ttl is not None:
        _order_ttl = order_ttl


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

    # Trim noisy fields from each market to reduce token usage
    _market_keep_fields = {
        "ticker", "title", "yes_bid", "yes_ask", "yes_bid_size", "yes_ask_size",
        "yes_levels", "no_levels", "spread", "freshness_seconds",
        "last_trade_price", "last_trade_side", "trade_count",
    }
    if "markets" in snapshot:
        trimmed = {}
        for ticker, mkt in snapshot["markets"].items():
            trimmed[ticker] = {k: v for k, v in mkt.items() if k in _market_keep_fields}
        snapshot["markets"] = trimmed

    # Also remove event-level noise
    for key in ("subtitle", "loaded_at"):
        snapshot.pop(key, None)

    return snapshot


@tool
async def get_events_summary() -> List[Dict[str, Any]]:
    """Get a compact summary of all monitored events for scanning.

    Returns a lightweight list with edge calculations and data coverage
    per event. Use this for initial scanning, then drill into specific
    events with get_event_snapshot().

    Returns:
        List of dicts with event_ticker, title, market_count, edge info
    """
    if not _index:
        return {"error": "EventArbIndex not available"}

    fee = _index._fee_per_contract
    summary = []
    for et, event in _index.events.items():
        sum_bid = event.market_sum_bid()
        sum_ask = event.market_sum_ask()
        long_e = event.long_edge(fee)
        short_e = event.short_edge(fee)

        summary.append({
            "event_ticker": et,
            "title": event.title,
            "market_count": event.markets_total,
            "markets_with_data": event.markets_with_data,
            "all_markets_have_data": event.all_markets_have_data,
            "sum_yes_bid": sum_bid,
            "sum_yes_ask": sum_ask,
            "long_edge": round(long_e, 1) if long_e is not None else None,
            "short_edge": round(short_e, 1) if short_e is not None else None,
        })

    return summary


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
    if not _trading_client:
        return {"error": "Trading client not available"}

    if side not in ("yes", "no"):
        return {"error": f"Invalid side: {side}. Must be 'yes' or 'no'."}
    if action not in ("buy", "sell"):
        return {"error": f"Invalid action: {action}. Must be 'buy' or 'sell'."}
    if not (1 <= contracts <= 100):
        return {"error": f"Contracts must be 1-100, got {contracts}"}
    if not (1 <= price_cents <= 99):
        return {"error": f"Price must be 1-99 cents, got {price_cents}"}

    try:
        expiration_ts = int(time.time()) + _order_ttl

        order_kwargs = {
            "ticker": ticker,
            "action": action,
            "side": side,
            "count": contracts,
            "price": price_cents,
            "type": "limit",
            "expiration_ts": expiration_ts,
        }
        if _order_group_id:
            order_kwargs["order_group_id"] = _order_group_id

        logger.info(
            f"[SINGLE_ARB:TRADE] action={action} ticker={ticker} "
            f"side={side} contracts={contracts} price={price_cents}c ttl={_order_ttl}s"
        )

        order_resp = await _trading_client.create_order(**order_kwargs)
        order = order_resp.get("order", order_resp)
        order_id = order.get("order_id", "")
        status = order.get("status", "placed")

        # Auto-record to memory
        if _memory_store:
            try:
                _memory_store.append(
                    content=f"TRADE: {action} {contracts} {side} {ticker} @{price_cents}c | {reasoning}",
                    memory_type="trade",
                    metadata={
                        "order_id": order_id,
                        "ticker": ticker,
                        "side": side,
                        "action": action,
                        "contracts": contracts,
                        "price_cents": price_cents,
                        "status": status,
                    },
                )
            except Exception as e:
                logger.debug(f"Memory record failed: {e}")

        return {
            "order_id": order_id,
            "status": status,
            "ticker": ticker,
            "side": side,
            "action": action,
            "contracts": contracts,
            "price_cents": price_cents,
            "ttl_seconds": _order_ttl,
            "order_group": _order_group_id[:8] if _order_group_id else None,
        }

    except Exception as e:
        # Record failed order to memory
        if _memory_store:
            try:
                _memory_store.append(
                    content=f"FAILED ORDER: {action} {contracts} {side} {ticker} @{price_cents}c | {reasoning} | error: {e}",
                    memory_type="trade",
                    metadata={
                        "ticker": ticker,
                        "side": side,
                        "action": action,
                        "contracts": contracts,
                        "price_cents": price_cents,
                        "status": "failed",
                        "error": str(e),
                    },
                )
            except Exception:
                pass
        return {"error": f"Order failed: {e}"}


@tool
async def get_trade_history(ticker: Optional[str] = None, limit: int = 20) -> Dict[str, Any]:
    """Get fills, settlements, and P&L. This is your report card.

    Args:
        ticker: Filter to specific market ticker (optional)
        limit: Maximum fills to return (default 20)
    """
    if not _trading_client:
        return {"error": "Trading client not available"}

    try:
        # Get fills
        fills_resp = await _trading_client.get_fills(ticker=ticker)
        raw_fills = fills_resp.get("fills", [])[:limit]
        fills = [
            {
                "ticker": f.get("ticker"),
                "side": f.get("side"),
                "action": f.get("action"),
                "count": f.get("count"),
                "yes_price": f.get("yes_price"),
                "no_price": f.get("no_price"),
                "order_id": f.get("order_id"),
                "created_time": f.get("created_time"),
            }
            for f in raw_fills
        ]

        # Get settlements
        settlements_resp = await _trading_client.get_settlements(max_settlements=50)
        raw_settlements = settlements_resp if isinstance(settlements_resp, list) else settlements_resp.get("settlements", [])
        settlements = [
            {
                "ticker": s.get("ticker") or s.get("market_ticker"),
                "market_result": s.get("market_result") or s.get("result"),
                "revenue": s.get("revenue"),
                "payout": s.get("payout"),
                "settled_time": s.get("settled_time") or s.get("settlement_time"),
            }
            for s in raw_settlements[:20]
        ]

        # Compute P&L from settlements
        total_revenue = sum(s.get("revenue", 0) or 0 for s in settlements)
        total_payout = sum(s.get("payout", 0) or 0 for s in settlements)
        net_pnl = total_payout - total_revenue

        return {
            "fills": fills,
            "fill_count": len(fills),
            "settlements": settlements,
            "settlement_count": len(settlements),
            "pnl_summary": {
                "total_revenue_cents": total_revenue,
                "total_payout_cents": total_payout,
                "net_pnl_cents": net_pnl,
            },
        }

    except Exception as e:
        return {"error": f"Trade history failed: {e}"}


@tool
async def get_resting_orders(ticker: Optional[str] = None) -> Dict[str, Any]:
    """Get currently open/resting orders with queue position info.

    Returns order details plus queue position (contracts ahead of yours).

    Args:
        ticker: Filter to specific market ticker (optional)
    """
    if not _trading_client:
        return {"error": "Trading client not available"}

    try:
        # Get orders
        resp = await _trading_client.get_orders(ticker=ticker, status="resting")
        raw_orders = resp.get("orders", [])

        # Get queue positions (best-effort)
        queue_map = {}
        try:
            qp_resp = await _trading_client.get_queue_positions(
                market_tickers=ticker if ticker else None,
            )
            for qp in qp_resp.get("queue_positions", []):
                queue_map[qp.get("order_id")] = qp.get("queue_position", None)
        except Exception:
            pass

        orders = [
            {
                "order_id": o.get("order_id"),
                "ticker": o.get("ticker"),
                "side": o.get("side"),
                "action": o.get("action"),
                "price": o.get("price") or o.get("yes_price"),
                "remaining_count": o.get("remaining_count"),
                "created_time": o.get("created_time"),
                "expiration_time": o.get("expiration_time"),
                "queue_position": queue_map.get(o.get("order_id")),
            }
            for o in raw_orders
        ]

        return {
            "count": len(orders),
            "orders": orders,
        }

    except Exception as e:
        return {"error": f"Get orders failed: {e}"}


@tool
async def cancel_order(order_id: str, reason: str = "") -> Dict[str, Any]:
    """Cancel a resting order.

    Args:
        order_id: The order ID to cancel
        reason: Optional reason for cancellation
    """
    if not _trading_client:
        return {"error": "Trading client not available"}

    try:
        await _trading_client.cancel_order(order_id)

        # Record cancellation to memory if reason provided
        if reason and _memory_store:
            try:
                _memory_store.append(
                    content=f"CANCEL: order {order_id[:8]}... | {reason}",
                    memory_type="trade",
                    metadata={"order_id": order_id, "action": "cancel"},
                )
            except Exception:
                pass

        return {"status": "cancelled", "order_id": order_id}

    except Exception as e:
        err_str = str(e).lower()
        if "not found" in err_str or "404" in err_str:
            return {"status": "already_gone", "order_id": order_id}
        # Record failed cancel to memory
        if _memory_store:
            try:
                _memory_store.append(
                    content=f"FAILED CANCEL: order {order_id[:8]}... | error: {e}",
                    memory_type="trade",
                    metadata={"order_id": order_id, "action": "cancel", "status": "failed", "error": str(e)},
                )
            except Exception:
                pass
        return {"error": f"Cancel failed: {e}"}


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
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Execute a multi-leg arb trade on a single event.

    LONG ARB: Buy YES on every market in the event.
    SHORT ARB: Buy NO on every market in the event.

    Each leg is placed as a limit order at the current best price.
    All legs must succeed for the arb to be profitable.

    Use dry_run=True to preview the trade without placing orders.

    Args:
        event_ticker: Event to trade
        direction: "long" (buy all YES) or "short" (buy all NO)
        max_contracts: Maximum contracts per leg
        reasoning: Why this trade should be made
        dry_run: If True, compute everything but don't place orders (preview mode)

    Returns:
        Dict with status (preview/completed/partial/aborted), legs, total cost
    """
    if not _trading_client:
        return {"error": "Trading client not available"}
    if not _index:
        return {"error": "EventArbIndex not available"}

    event_state = _index.events.get(event_ticker)
    if not event_state:
        return {"error": f"Event {event_ticker} not found"}

    if not event_state.all_markets_have_data:
        return {
            "error": f"Not all markets have data ({event_state.markets_with_data}/{event_state.markets_total})",
        }

    if direction not in ("long", "short"):
        return {"error": f"Invalid direction: {direction}. Must be 'long' or 'short'."}

    # Check balance
    try:
        balance_resp = await _trading_client.get_account_info()
        balance = balance_resp.get("balance", 0)
    except Exception as e:
        return {"error": f"Balance check failed: {e}"}

    # Build legs
    legs = []
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
        else:
            # Buy NO (sell YES) → NO price = 100 - YES bid
            if book.yes_bid is None:
                errors.append(f"{book.ticker}: no YES bid")
                continue
            side = "no"
            price = 100 - book.yes_bid

        # Size: minimum of max_contracts and available liquidity
        contracts = min(max_contracts, book.yes_ask_size if direction == "long" else book.yes_bid_size)
        contracts = max(contracts, 1)  # At least 1

        # Check balance for this leg
        leg_cost = contracts * price
        if total_cost + leg_cost > balance:
            errors.append(f"{book.ticker}: insufficient balance for {contracts}@{price}c")
            continue

        legs.append({
            "ticker": book.ticker,
            "title": book.title,
            "side": side,
            "contracts": contracts,
            "price_cents": price,
        })
        total_cost += leg_cost

    # Dry run: return preview without placing orders
    if dry_run:
        return {
            "status": "preview",
            "event_ticker": event_ticker,
            "direction": direction,
            "legs_planned": len(legs),
            "legs_total": event_state.markets_total,
            "estimated_cost_cents": total_cost,
            "balance_cents": balance,
            "balance_after_cents": balance - total_cost,
            "legs": legs,
            "errors": errors,
            "reasoning": reasoning,
        }

    # Execute orders
    legs_executed = []
    exec_cost = 0

    for leg in legs:
        try:
            logger.info(
                f"[SINGLE_ARB:TRADE] action=buy ticker={leg['ticker']} "
                f"side={leg['side']} contracts={leg['contracts']} price={leg['price_cents']}c"
            )
            order_kwargs = {
                "ticker": leg["ticker"],
                "action": "buy",
                "side": leg["side"],
                "count": leg["contracts"],
                "price": leg["price_cents"],
                "type": "limit",
                "expiration_ts": int(time.time()) + _order_ttl,
            }
            if _order_group_id:
                order_kwargs["order_group_id"] = _order_group_id

            order_resp = await _trading_client.create_order(**order_kwargs)
            order = order_resp.get("order", order_resp)
            order_id = order.get("order_id", "")

            legs_executed.append({
                **leg,
                "order_id": order_id,
                "status": order.get("status", "placed"),
            })
            exec_cost += leg["contracts"] * leg["price_cents"]

        except Exception as e:
            error_msg = f"{leg['ticker']}: order failed: {e}"
            logger.error(f"[SINGLE_ARB:TRADE_ERROR] {error_msg}")
            errors.append(error_msg)

    status = "completed" if len(legs_executed) == event_state.markets_total else "partial"
    if len(legs_executed) == 0:
        status = "failed"

    result = {
        "status": status,
        "event_ticker": event_ticker,
        "direction": direction,
        "legs_executed": len(legs_executed),
        "legs_total": event_state.markets_total,
        "total_cost_cents": exec_cost,
        "legs": legs_executed,
        "errors": errors,
        "reasoning": reasoning,
    }

    # Auto-record to memory (including failures)
    if _memory_store:
        try:
            prefix = "ARB" if status != "failed" else "FAILED ARB"
            _memory_store.append(
                content=f"{prefix}: {direction} {event_ticker} | {len(legs_executed)}/{event_state.markets_total} legs | cost={exec_cost}c | {reasoning}",
                memory_type="trade",
                metadata={
                    "event_ticker": event_ticker,
                    "direction": direction,
                    "legs_executed": len(legs_executed),
                    "total_cost_cents": exec_cost,
                    "status": status,
                    "order_ids": [l.get("order_id") for l in legs_executed],
                    "errors": errors if errors else None,
                },
            )
        except Exception as e:
            logger.debug(f"Memory record failed: {e}")

    logger.info(
        f"[SINGLE_ARB:TRADE_RESULT] event={event_ticker} direction={direction} "
        f"status={status} legs={len(legs_executed)}/{event_state.markets_total} "
        f"cost={exec_cost}c"
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
async def get_positions() -> Dict[str, Any]:
    """Get current positions on Kalshi, filtered to arb-tracked markets.

    Returns only positions whose ticker is in the current EventArbIndex.
    Also reports total position count so the agent knows about legacy positions.
    """
    if not _trading_client:
        return {"error": "Trading client not available"}

    try:
        resp = await _trading_client.get_positions()
        positions = resp.get("market_positions", resp.get("positions", []))
        total = len(positions)

        # Filter to arb-tracked market tickers
        tracked = set(_index.market_tickers) if _index else set()
        arb_positions = [
            p for p in positions
            if p.get("ticker") in tracked
        ]

        return {
            "arb_positions": arb_positions,
            "arb_position_count": len(arb_positions),
            "total_positions": total,
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
        resp = await _trading_client.get_account_info()
        balance_cents = resp.get("balance", 0)
        return {
            "balance_cents": balance_cents,
            "balance_dollars": round(balance_cents / 100, 2),
        }
    except Exception as e:
        return {"error": str(e)}
