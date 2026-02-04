"""
Trade execution tools for the arb deep agent.

Two self-contained tools that handle the full order lifecycle:
- buy_arb_position: Validate -> orderbook -> place -> wait -> resolve -> log
- sell_arb_position: Validate -> orderbook -> place -> wait -> resolve -> log

Both ALWAYS return terminal status: completed, cancelled, or aborted.
"""

import asyncio
import logging
import time
from typing import Any, Callable, Coroutine, Dict, Optional

from langchain_core.tools import tool

logger = logging.getLogger("kalshiflow_rl.traderv3.deep_agent.tools.trade")

_trading_client = None
_pair_registry = None
_spread_monitor = None
_supabase = None
_file_store = None
_event_callback: Optional[Callable[..., Coroutine]] = None

# Trade execution config
FILL_POLL_INTERVAL = 3.0  # seconds between fill checks
FILL_TIMEOUT = 60.0  # max wait for fill


def set_trade_deps(
    trading_client=None,
    pair_registry=None,
    spread_monitor=None,
    supabase=None,
    file_store=None,
    event_callback=None,
) -> None:
    """Set dependencies for trade tools."""
    global _trading_client, _pair_registry, _spread_monitor, _supabase, _file_store, _event_callback
    if trading_client is not None:
        _trading_client = trading_client
    if pair_registry is not None:
        _pair_registry = pair_registry
    if spread_monitor is not None:
        _spread_monitor = spread_monitor
    if supabase is not None:
        _supabase = supabase
    if file_store is not None:
        _file_store = file_store
    if event_callback is not None:
        _event_callback = event_callback


async def _emit(subtype: str, data: Dict[str, Any]) -> None:
    """Emit a streaming event to the frontend."""
    if not _event_callback:
        return
    try:
        await _event_callback({
            "type": "agent_message",
            "subtype": subtype,
            **data,
        })
    except Exception as e:
        logger.warning(f"Trade event emission failed: {e}")


async def _log_to_db(
    pair_id: str,
    kalshi_ticker: str,
    side: str,
    action: str,
    contracts: int,
    price_cents: int,
    reasoning: str,
    status: str,
    spread_at_entry: Optional[int] = None,
    order_id: Optional[str] = None,
    pnl_cents: Optional[int] = None,
) -> None:
    """Log trade to arb_trades table."""
    if not _supabase:
        return
    try:
        row = {
            "pair_id": pair_id,
            "kalshi_ticker": kalshi_ticker,
            "side": side,
            "action": action,
            "contracts": contracts,
            "price_cents": price_cents,
            "reasoning": reasoning,
            "status": status,
        }
        if spread_at_entry is not None:
            row["spread_at_entry"] = spread_at_entry
        if order_id:
            row["order_id"] = order_id
        if pnl_cents is not None:
            row["pnl_cents"] = pnl_cents
        _supabase.table("arb_trades").insert(row).execute()
    except Exception as e:
        logger.warning(f"Failed to log trade to DB: {e}")


async def _get_orderbook(ticker: str) -> Dict[str, Any]:
    """Fetch orderbook for a ticker."""
    if not _trading_client:
        return {}
    try:
        return await _trading_client.get_orderbook(ticker=ticker) or {}
    except Exception as e:
        logger.warning(f"Orderbook fetch failed for {ticker}: {e}")
        return {}


def _best_ask(orderbook: Dict[str, Any], side: str) -> Optional[int]:
    """Get best ask price from orderbook for given side."""
    if side == "yes":
        asks = orderbook.get("yes", {}).get("asks", []) or orderbook.get("yes_asks", [])
    else:
        asks = orderbook.get("no", {}).get("asks", []) or orderbook.get("no_asks", [])
    if not asks:
        return None
    # asks sorted ascending by price
    return asks[0].get("price") if isinstance(asks[0], dict) else asks[0]


def _best_bid(orderbook: Dict[str, Any], side: str) -> Optional[int]:
    """Get best bid price from orderbook for given side."""
    if side == "yes":
        bids = orderbook.get("yes", {}).get("bids", []) or orderbook.get("yes_bids", [])
    else:
        bids = orderbook.get("no", {}).get("bids", []) or orderbook.get("no_bids", [])
    if not bids:
        return None
    # bids sorted descending by price
    return bids[0].get("price") if isinstance(bids[0], dict) else bids[0]


async def _wait_for_fill(order_id: str, ticker: str, side: str, contracts: int) -> Dict[str, Any]:
    """Poll for order fill. Returns fill status."""
    start = time.time()
    while (time.time() - start) < FILL_TIMEOUT:
        await asyncio.sleep(FILL_POLL_INTERVAL)

        await _emit("trade_waiting", {
            "order_id": order_id,
            "elapsed": round(time.time() - start, 1),
            "ticker": ticker,
        })

        try:
            fills_resp = await _trading_client.get_fills(ticker=ticker)
            fills = fills_resp.get("fills", []) if isinstance(fills_resp, dict) else []

            for fill in fills:
                if fill.get("order_id") == order_id:
                    return {
                        "filled": True,
                        "fill_price": fill.get("yes_price") or fill.get("no_price"),
                        "fill_count": fill.get("count", contracts),
                        "fill_id": fill.get("trade_id"),
                    }
        except Exception as e:
            logger.warning(f"Fill check failed: {e}")

    return {"filled": False, "reason": "ttl_expired"}


@tool
async def buy_arb_position(
    pair_id: str,
    side: str,
    contracts: int,
    max_price_cents: int,
    reasoning: str,
) -> Dict[str, Any]:
    """Buy an arb position on Kalshi. Handles the full order lifecycle.

    State machine: VALIDATE -> ORDERBOOK -> PLACE -> WAIT -> RESOLVE -> LOG
    Always returns terminal status: completed, cancelled, or aborted.

    Args:
        pair_id: UUID of the paired market to trade
        side: 'yes' or 'no' (which Kalshi contract to buy)
        contracts: Number of contracts to buy
        max_price_cents: Maximum limit price in cents (1-99)
        reasoning: Why this trade should be made

    Returns:
        Dict with status (completed/cancelled/aborted), price, contracts, order_id
    """
    if not _trading_client:
        return {"status": "aborted", "reason": "Trading client not available"}

    # VALIDATE
    pair = _pair_registry.get_by_id(pair_id) if _pair_registry else None
    if not pair:
        return {"status": "aborted", "reason": f"Pair {pair_id} not found in registry"}

    ticker = pair.kalshi_ticker

    # Check validation exists
    if _file_store:
        validation = _file_store.get_validation(pair_id)
        if not validation or validation.get("status") != "approved":
            return {"status": "aborted", "reason": f"Pair not validated (status: {validation.get('status') if validation else 'unknown'})"}

    # Check balance
    try:
        balance_resp = await _trading_client.get_balance()
        balance = balance_resp.get("balance", 0)
        cost = contracts * max_price_cents
        if balance < cost:
            return {"status": "aborted", "reason": f"Insufficient balance: {balance}c < {cost}c needed"}
    except Exception as e:
        return {"status": "aborted", "reason": f"Balance check failed: {e}"}

    # ORDERBOOK
    book = await _get_orderbook(ticker)
    best = _best_ask(book, side)
    if best is None:
        return {"status": "aborted", "reason": f"No {side} liquidity on {ticker}"}

    entry_price = min(max_price_cents, best)

    # Get current spread for logging
    spread_at_entry = None
    if _spread_monitor:
        ss = _spread_monitor.get_spread_state(pair_id)
        if ss:
            spread_at_entry = ss.spread_cents

    # PLACE
    logger.info(f"[V3:TRADE_PLACED] action=buy ticker={ticker} side={side} contracts={contracts} price={entry_price}c pair_id={pair_id}")
    await _emit("trade_placed", {
        "action": "buy",
        "ticker": ticker,
        "side": side,
        "contracts": contracts,
        "price": entry_price,
        "pair_id": pair_id,
    })

    try:
        order_resp = await _trading_client.create_order(
            ticker=ticker,
            action="buy",
            side=side,
            count=contracts,
            price=entry_price,
            type="limit",
        )
    except Exception as e:
        logger.error(f"[V3:TRADE_RESULT] action=buy ticker={ticker} status=aborted reason=order_placement_failed error={e}")
        await _log_to_db(pair_id, ticker, side, "buy", contracts, entry_price, reasoning, "aborted", spread_at_entry)
        return {"status": "aborted", "reason": f"Order placement failed: {e}"}

    order = order_resp.get("order", order_resp)
    order_id = order.get("order_id", "")
    order_status = order.get("status", "")

    # Immediate fill check
    if order_status in ("executed", "filled"):
        fill_price = order.get("yes_price") or order.get("no_price") or entry_price
        result = {
            "status": "completed",
            "action": "buy",
            "ticker": ticker,
            "side": side,
            "contracts": contracts,
            "price_cents": fill_price,
            "order_id": order_id,
            "spread_at_entry": spread_at_entry,
        }
        logger.info(f"[V3:TRADE_RESULT] action=buy ticker={ticker} status=completed price={fill_price}c contracts={contracts}")
        await _log_to_db(pair_id, ticker, side, "buy", contracts, fill_price, reasoning, "completed", spread_at_entry, order_id)
        await _emit("trade_terminal", {"result": result})
        return result

    # WAIT for fill
    fill_result = await _wait_for_fill(order_id, ticker, side, contracts)

    if fill_result.get("filled"):
        result = {
            "status": "completed",
            "action": "buy",
            "ticker": ticker,
            "side": side,
            "contracts": fill_result.get("fill_count", contracts),
            "price_cents": fill_result.get("fill_price", entry_price),
            "order_id": order_id,
            "spread_at_entry": spread_at_entry,
        }
        logger.info(f"[V3:TRADE_RESULT] action=buy ticker={ticker} status=completed price={result['price_cents']}c contracts={result['contracts']}")
        await _log_to_db(pair_id, ticker, side, "buy", contracts, result["price_cents"], reasoning, "completed", spread_at_entry, order_id)
        await _emit("trade_terminal", {"result": result})
        return result

    # TTL expired -> cancel
    try:
        await _trading_client.cancel_order(order_id=order_id)
    except Exception as e:
        logger.warning(f"Cancel after TTL failed: {e}")

    result = {
        "status": "cancelled",
        "reason": "ttl_expired",
        "action": "buy",
        "ticker": ticker,
        "side": side,
        "contracts": contracts,
        "price_cents": entry_price,
        "order_id": order_id,
    }
    logger.info(f"[V3:TRADE_RESULT] action=buy ticker={ticker} status=cancelled reason=ttl_expired")
    await _log_to_db(pair_id, ticker, side, "buy", contracts, entry_price, reasoning, "cancelled", spread_at_entry, order_id)
    await _emit("trade_terminal", {"result": result})
    return result


@tool
async def sell_arb_position(
    pair_id: str,
    side: str,
    contracts: int,
    min_price_cents: int,
    reasoning: str,
) -> Dict[str, Any]:
    """Sell an existing arb position on Kalshi. Handles the full order lifecycle.

    State machine: VALIDATE -> ORDERBOOK -> PLACE -> WAIT -> RESOLVE -> LOG
    Always returns terminal status: completed, cancelled, or aborted.

    Args:
        pair_id: UUID of the paired market
        side: 'yes' or 'no' (which Kalshi contract to sell)
        contracts: Number of contracts to sell
        min_price_cents: Minimum limit price in cents (1-99)
        reasoning: Why this position should be sold

    Returns:
        Dict with status (completed/cancelled/aborted), price, contracts, pnl_cents
    """
    if not _trading_client:
        return {"status": "aborted", "reason": "Trading client not available"}

    pair = _pair_registry.get_by_id(pair_id) if _pair_registry else None
    if not pair:
        return {"status": "aborted", "reason": f"Pair {pair_id} not found in registry"}

    ticker = pair.kalshi_ticker

    # Check position exists
    try:
        positions_resp = await _trading_client.get_positions()
        positions = positions_resp.get("market_positions", []) if isinstance(positions_resp, dict) else []
        matching = [p for p in positions if p.get("ticker") == ticker]
        if not matching:
            return {"status": "aborted", "reason": f"No position in {ticker}"}

        pos = matching[0]
        available = pos.get(f"{side}_quantity", 0) or pos.get("quantity", 0)
        if available < contracts:
            return {"status": "aborted", "reason": f"Insufficient position: have {available}, want to sell {contracts}"}
    except Exception as e:
        return {"status": "aborted", "reason": f"Position check failed: {e}"}

    # ORDERBOOK
    book = await _get_orderbook(ticker)
    best = _best_bid(book, side)
    if best is None:
        return {"status": "aborted", "reason": f"No {side} bids on {ticker}"}

    sell_price = max(min_price_cents, best)

    spread_at_entry = None
    if _spread_monitor:
        ss = _spread_monitor.get_spread_state(pair_id)
        if ss:
            spread_at_entry = ss.spread_cents

    # PLACE
    logger.info(f"[V3:TRADE_PLACED] action=sell ticker={ticker} side={side} contracts={contracts} price={sell_price}c pair_id={pair_id}")
    await _emit("trade_placed", {
        "action": "sell",
        "ticker": ticker,
        "side": side,
        "contracts": contracts,
        "price": sell_price,
        "pair_id": pair_id,
    })

    try:
        order_resp = await _trading_client.create_order(
            ticker=ticker,
            action="sell",
            side=side,
            count=contracts,
            price=sell_price,
            type="limit",
        )
    except Exception as e:
        logger.error(f"[V3:TRADE_RESULT] action=sell ticker={ticker} status=aborted reason=order_placement_failed error={e}")
        await _log_to_db(pair_id, ticker, side, "sell", contracts, sell_price, reasoning, "aborted", spread_at_entry)
        return {"status": "aborted", "reason": f"Order placement failed: {e}"}

    order = order_resp.get("order", order_resp)
    order_id = order.get("order_id", "")
    order_status = order.get("status", "")

    # Compute P&L vs entry (from recent fills)
    entry_price_estimate = None
    try:
        fills_resp = await _trading_client.get_fills(ticker=ticker)
        fills = fills_resp.get("fills", []) if isinstance(fills_resp, dict) else []
        buy_fills = [f for f in fills if f.get("action") == "buy" and f.get("side") == side]
        if buy_fills:
            entry_price_estimate = buy_fills[-1].get("yes_price") or buy_fills[-1].get("no_price")
    except Exception:
        pass

    if order_status in ("executed", "filled"):
        fill_price = order.get("yes_price") or order.get("no_price") or sell_price
        pnl = (fill_price - entry_price_estimate) * contracts if entry_price_estimate else None
        result = {
            "status": "completed",
            "action": "sell",
            "ticker": ticker,
            "side": side,
            "contracts": contracts,
            "price_cents": fill_price,
            "order_id": order_id,
            "pnl_cents": pnl,
            "spread_at_entry": spread_at_entry,
        }
        logger.info(f"[V3:TRADE_RESULT] action=sell ticker={ticker} status=completed price={fill_price}c pnl={pnl}c")
        await _log_to_db(pair_id, ticker, side, "sell", contracts, fill_price, reasoning, "completed", spread_at_entry, order_id, pnl)
        await _emit("trade_terminal", {"result": result})
        return result

    # WAIT
    fill_result = await _wait_for_fill(order_id, ticker, side, contracts)

    if fill_result.get("filled"):
        fill_price = fill_result.get("fill_price", sell_price)
        pnl = (fill_price - entry_price_estimate) * contracts if entry_price_estimate else None
        result = {
            "status": "completed",
            "action": "sell",
            "ticker": ticker,
            "side": side,
            "contracts": fill_result.get("fill_count", contracts),
            "price_cents": fill_price,
            "order_id": order_id,
            "pnl_cents": pnl,
            "spread_at_entry": spread_at_entry,
        }
        logger.info(f"[V3:TRADE_RESULT] action=sell ticker={ticker} status=completed price={fill_price}c pnl={pnl}c")
        await _log_to_db(pair_id, ticker, side, "sell", contracts, fill_price, reasoning, "completed", spread_at_entry, order_id, pnl)
        await _emit("trade_terminal", {"result": result})
        return result

    # TTL expired -> cancel
    try:
        await _trading_client.cancel_order(order_id=order_id)
    except Exception as e:
        logger.warning(f"Cancel after TTL failed: {e}")

    result = {
        "status": "cancelled",
        "reason": "ttl_expired",
        "action": "sell",
        "ticker": ticker,
        "side": side,
        "contracts": contracts,
        "price_cents": sell_price,
        "order_id": order_id,
    }
    logger.info(f"[V3:TRADE_RESULT] action=sell ticker={ticker} status=cancelled reason=ttl_expired")
    await _log_to_db(pair_id, ticker, side, "sell", contracts, sell_price, reasoning, "cancelled", spread_at_entry, order_id)
    await _emit("trade_terminal", {"result": result})
    return result
