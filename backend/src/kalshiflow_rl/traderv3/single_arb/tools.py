"""Captain V2 tools - 13 tools for single-agent Captain.

Single ToolContext dataclass replaces 14 module-level globals.
All tools return Pydantic models (serialized via .model_dump()).
Order tools return new_balance so agent never needs a separate balance query.
"""

import asyncio
import collections
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, TYPE_CHECKING

from langchain_core.tools import tool

from .models import (
    ArbLegResult,
    ArbResult,
    ImpactPattern,
    MarketState,
    NewsArticle,
    NewsSearchResult,
    OrderResult,
    PortfolioState,
    RecallResult,
    RestingOrder,
    SwingNewsResult,
)

if TYPE_CHECKING:
    from .account_health import AccountHealthService
    from .auto_actions import AutoActionManager
    from .context_builder import ContextBuilder
    from .decision_ledger import DecisionLedger
    from .index import EventArbIndex
    from .memory.session_store import SessionMemoryStore
    from .sniper import Sniper, SniperConfig
    from .swing_detector import SwingDetector
    from .tavily_service import TavilySearchService
    from ..agent_tools.session import TradingSession
    from ..gateway.client import KalshiGateway
    from ..services.early_bird_service import EarlyBirdService
    from ..market_maker.quote_engine import QuoteEngine
    from ..market_maker.index import MMIndex
    from ..market_maker.models import QuoteConfig

logger = logging.getLogger("kalshiflow_rl.traderv3.single_arb.tools_v2")


NEWS_CACHE_TTL = 300  # 5 minutes


async def retry_api(coro_factory, max_retries: int = 3, label: str = "api_call"):
    """Retry an async API call with exponential backoff + jitter.

    Args:
        coro_factory: Callable that returns a new coroutine on each call (e.g., lambda: gw.get_balance())
        max_retries: Maximum attempts (default 3)
        label: Label for logging

    Returns:
        The result of the coroutine on success.

    Raises:
        The last exception after all retries exhausted.
        Immediately raises on 4xx client errors (no retry).
    """
    last_err = None
    for attempt in range(max_retries):
        try:
            return await coro_factory()
        except Exception as e:
            last_err = e
            err_str = str(e).lower()
            # Don't retry 4xx client errors (bad request, not found, etc.)
            if any(code in err_str for code in ("400", "401", "403", "404", "422")):
                raise
            if attempt < max_retries - 1:
                delay = (2 ** attempt) * 0.5 + random.uniform(0, 0.5)
                logger.warning(f"[RETRY] {label} attempt {attempt + 1} failed: {e}, retrying in {delay:.1f}s")
                await asyncio.sleep(delay)
    raise last_err


@dataclass
class ToolContext:
    """Single dependency container for all V2 tools."""
    gateway: "KalshiGateway"
    index: "EventArbIndex"
    memory: "SessionMemoryStore"
    search: Optional["TavilySearchService"]
    sniper: Optional["Sniper"]
    sniper_config: Optional["SniperConfig"]
    session: "TradingSession"
    context_builder: "ContextBuilder"
    broadcast: Optional[Callable[..., Coroutine]] = None
    health_service: Optional["AccountHealthService"] = None
    auto_actions: Optional["AutoActionManager"] = None
    swing_detector: Optional["SwingDetector"] = None
    decision_ledger: Optional["DecisionLedger"] = None
    early_bird_service: Optional["EarlyBirdService"] = None
    quote_engine: Optional["QuoteEngine"] = None
    mm_index: Optional["MMIndex"] = None
    quote_config: Optional["QuoteConfig"] = None
    max_contracts_per_market: int = 200  # From V3Config.captain_max_contracts_per_market
    cycle_mode: Optional[str] = None  # "reactive", "strategic", "deep_scan"
    captain_order_ids: Set[str] = field(default_factory=set)
    order_initial_states: Dict[str, dict] = field(default_factory=dict)
    news_cache: collections.OrderedDict = field(default_factory=collections.OrderedDict)
    cycle_capital_spent_cents: int = 0
    _positions_cache: Optional[List] = field(default=None)
    _positions_cached_at: float = field(default=0.0)


# Module-level context (set by coordinator at startup)
_ctx: Optional[ToolContext] = None


def set_context(ctx: ToolContext) -> None:
    """Set the shared ToolContext. Called by coordinator at startup."""
    global _ctx
    _ctx = ctx


def get_context() -> Optional[ToolContext]:
    """Get the current ToolContext (for testing)."""
    return _ctx


def cleanup_terminal_orders(max_age_seconds: float = 86400) -> int:
    """Remove terminal orders older than max_age from tracking sets.

    Called periodically by coordinator to prevent unbounded growth.
    Returns number of cleaned entries.
    """
    if not _ctx:
        return 0
    now = time.time()
    stale_ids = []
    for order_id, state in list(_ctx.order_initial_states.items()):
        placed_at = state.get("placed_at", 0)
        status = state.get("status", "")
        if status in ("executed", "filled", "expired", "canceled", "cancelled"):
            if now - placed_at > max_age_seconds:
                stale_ids.append(order_id)
        elif now - placed_at > max_age_seconds * 2:
            # Non-terminal but very old — clean up anyway
            stale_ids.append(order_id)

    for oid in stale_ids:
        _ctx.captain_order_ids.discard(oid)
        _ctx.order_initial_states.pop(oid, None)

    if stale_ids:
        logger.info(f"[TOOLS:CLEANUP] Removed {len(stale_ids)} stale order entries")
    return len(stale_ids)


async def _get_cached_positions(event_ticker: Optional[str] = None, ttl: float = 10.0) -> List:
    """Fetch positions with a short TTL cache to avoid redundant API calls.

    Args:
        event_ticker: Optional event ticker to filter positions
        ttl: Cache TTL in seconds (default 10s)

    Returns:
        List of position objects from the API
    """
    now = time.time()
    if _ctx._positions_cache is not None and (now - _ctx._positions_cached_at) < ttl:
        return _ctx._positions_cache
    positions = await retry_api(
        lambda: _ctx.gateway.get_positions(event_ticker=event_ticker),
        max_retries=2, label="get_positions_cached",
    )
    _ctx._positions_cache = positions
    _ctx._positions_cached_at = now
    return positions


async def _fetch_balance_with_budget_check(label: str) -> tuple:
    """Fetch balance and warn if cycle capital spend exceeds 30%.

    Returns (balance_cents, balance_dollars) or (None, None) on failure.
    """
    try:
        bal = await _ctx.gateway.get_balance()
        balance_cents = bal.balance
        balance_dollars = round(balance_cents / 100, 2)
        if balance_cents and _ctx.cycle_capital_spent_cents > balance_cents * 0.3:
            logger.warning(
                f"[TOOLS:BUDGET] Cycle capital spend {_ctx.cycle_capital_spent_cents}c "
                f"exceeds 30% of balance {balance_cents}c"
            )
        return balance_cents, balance_dollars
    except Exception as e:
        logger.warning(f"[TOOLS:{label}] Balance fetch failed: {e}")
        return None, None


# --- Tool 1: get_market_state ---

@tool
async def get_market_state() -> Dict[str, Any]:
    """Get current market state for all monitored events.

    DO NOT call in REACTIVE or STRATEGIC cycles — data is in your briefing.
    Only use in DEEP_SCAN if you need per-market detail beyond the compact summary.

    Returns:
        MarketState with all events, markets, edges, volume, and regime
    """
    if not _ctx:
        return {"error": "ToolContext not available"}
    state = _ctx.context_builder.build_market_state()
    return state.model_dump()


# --- Tool 2: get_portfolio ---

@tool
async def get_portfolio() -> Dict[str, Any]:
    """Get balance + all positions + P&L in one call.

    Your cycle briefing already includes portfolio data. place_order and execute_arb
    return new_balance in their response. Only call if positions changed mid-cycle
    AND are not reflected in order responses.

    Returns:
        PortfolioState with balance, positions, and P&L
    """
    if not _ctx:
        return {"error": "ToolContext not available"}
    portfolio = await _ctx.context_builder.build_portfolio_state(_ctx.gateway)
    return portfolio.model_dump()


# --- Tool 3: place_order ---

@tool
async def place_order(
    ticker: str,
    side: str,
    action: str,
    contracts: int,
    price_cents: int,
    reasoning: str,
) -> Dict[str, Any]:
    """Place a single order. Returns new balance after placement.

    Auto-sets TTL, uses session order group, records to memory,
    and broadcasts to frontend.

    Args:
        ticker: Market ticker (e.g., KXMARKET-YES)
        side: "yes" or "no"
        action: "buy" or "sell"
        contracts: Number of contracts (1 to max_contracts_per_market)
        price_cents: Limit price in cents (1-99)
        reasoning: Trade thesis (stored in memory)

    Returns:
        OrderResult with order_id, status, and new_balance_cents
    """
    if not _ctx:
        return {"error": "ToolContext not available"}

    if side not in ("yes", "no"):
        return {"error": f"Invalid side: {side}. Must be 'yes' or 'no'."}
    if action not in ("buy", "sell"):
        return {"error": f"Invalid action: {action}. Must be 'buy' or 'sell'."}
    max_contracts = _ctx.max_contracts_per_market
    if not (1 <= contracts <= max_contracts):
        return {"error": f"Contracts must be 1-{max_contracts}, got {contracts}"}
    if not (1 <= price_cents <= 99):
        return {"error": f"Price must be 1-99 cents, got {price_cents}"}

    gw = _ctx.gateway
    session = _ctx.session

    # Position conflict guard (block buying opposite side on same market)
    if action == "buy":
        try:
            event_for_ticker = _ctx.index.get_event_for_ticker(ticker) if _ctx.index else None
            raw_positions = await _get_cached_positions(event_ticker=event_for_ticker)
            for pos in raw_positions:
                pos_ticker = pos.ticker if hasattr(pos, "ticker") else pos.get("ticker")
                pos_count = pos.position if hasattr(pos, "position") else pos.get("position", 0)
                if pos_ticker != ticker or pos_count == 0:
                    continue
                existing_side = "yes" if pos_count > 0 else "no"
                if existing_side != side:
                    return {
                        "error": f"POSITION CONFLICT: Hold {abs(pos_count)} {existing_side.upper()} on {ticker}. "
                        f"Exit first with place_order(action='sell', side='{existing_side}', ...). "
                        f"Buying {side.upper()} while holding {existing_side.upper()} locks in losses.",
                    }
        except Exception as e:
            logger.warning(f"[TOOLS:TRADE] Position conflict check failed (blocking order): {e}")
            return {"error": f"Position conflict check failed: {e}. Cannot verify positions — order blocked for safety."}

    # Edge gate: reject buy orders with no edge (sell/exit orders bypass)
    if action == "buy" and _ctx.index:
        event_ticker = _ctx.index.get_event_for_ticker(ticker)
        if event_ticker:
            event = _ctx.index.events.get(event_ticker)
            market = event.markets.get(ticker) if event else None
            if market:
                if side == "yes" and market.yes_ask is not None and price_cents >= market.yes_ask:
                    logger.info(
                        f"[TOOLS:EDGE_GATE] BLOCKED buy YES {ticker}@{price_cents}c "
                        f"(ask={market.yes_ask}c, no edge)"
                    )
                    return {
                        "error": f"NO EDGE: Buying YES@{price_cents}c but ask is {market.yes_ask}c. "
                        f"You'd be paying at or above the ask with 0 edge. Bid lower or skip.",
                        "market_yes_ask": market.yes_ask, "market_yes_bid": market.yes_bid,
                    }
                if side == "no" and market.yes_bid is not None:
                    no_ask = 100 - market.yes_bid
                    if price_cents >= no_ask:
                        logger.info(
                            f"[TOOLS:EDGE_GATE] BLOCKED buy NO {ticker}@{price_cents}c "
                            f"(no_ask={no_ask}c, no edge)"
                        )
                        return {
                            "error": f"NO EDGE: Buying NO@{price_cents}c but NO ask is {no_ask}c. "
                            f"You'd be paying at or above the ask with 0 edge. Bid lower or skip.",
                            "market_yes_bid": market.yes_bid, "no_ask": no_ask,
                        }
                # Log edge value for analysis
                if side == "yes" and market.yes_ask is not None:
                    edge = market.yes_ask - price_cents
                    logger.info(f"[TOOLS:EDGE] {ticker} YES buy@{price_cents}c ask={market.yes_ask}c edge={edge}c")
                elif side == "no" and market.yes_bid is not None:
                    no_ask = 100 - market.yes_bid
                    edge = no_ask - price_cents
                    logger.info(f"[TOOLS:EDGE] {ticker} NO buy@{price_cents}c no_ask={no_ask}c edge={edge}c")

    try:
        expiration_ts = int(time.time()) + session.order_ttl

        order_kwargs = {
            "ticker": ticker,
            "action": action,
            "side": side,
            "count": contracts,
            "price": price_cents,
            "type": "limit",
            "expiration_ts": expiration_ts,
        }
        if session.order_group_id:
            order_kwargs["order_group_id"] = session.order_group_id

        logger.info(
            f"[V2:TRADE] action={action} ticker={ticker} side={side} "
            f"contracts={contracts} price={price_cents}c ttl={session.order_ttl}s"
        )

        # Place order via gateway (with retry for transient failures)
        order_resp = await retry_api(
            lambda: gw.create_order(**order_kwargs),
            max_retries=2, label=f"create_order({ticker})",
        )
        order = order_resp.order
        order_id = order.order_id
        status = order.status or "placed"

        # Track as Captain order
        if order_id:
            _ctx.captain_order_ids.add(order_id)
            session.captain_order_ids.add(order_id)
            _ctx.order_initial_states[order_id] = {
                "ticker": ticker, "side": side, "action": action,
                "contracts": contracts, "price_cents": price_cents,
                "placed_at": time.time(), "ttl_seconds": session.order_ttl,
                "status": status,
            }

        # Bust position cache after successful placement
        _ctx._positions_cache = None

        # Track cycle capital spend
        _ctx.cycle_capital_spent_cents += contracts * price_cents

        # Record to decision ledger (fire-and-forget)
        if _ctx.decision_ledger:
            event_ticker_for_ledger = _ctx.index.get_event_for_ticker(ticker) if _ctx.index else None
            asyncio.create_task(_ctx.decision_ledger.record_decision(
                order_id=order_id, source="captain",
                event_ticker=event_ticker_for_ledger, market_ticker=ticker,
                side=side, action=action, contracts=contracts,
                limit_price_cents=price_cents, reasoning=reasoning,
                cycle_mode=_ctx.cycle_mode,
            ))

        new_balance_cents, new_balance_dollars = await _fetch_balance_with_budget_check("TRADE")

        # Auto-record to memory (fire-and-forget)
        asyncio.create_task(_ctx.memory.store(
            content=f"TRADE: {action} {contracts} {side} {ticker} @{price_cents}c | {reasoning}",
            memory_type="trade",
            metadata={"order_id": order_id, "ticker": ticker, "side": side,
                       "action": action, "contracts": contracts, "price_cents": price_cents},
        ))

        # Broadcast to frontend
        if _ctx.broadcast:
            event_ticker = _ctx.index.get_event_for_ticker(ticker) if _ctx.index else None
            asyncio.create_task(_ctx.broadcast({
                "type": "arb_trade_executed",
                "data": {
                    "order_id": order_id, "event_ticker": event_ticker,
                    "kalshi_ticker": ticker, "side": side, "action": action,
                    "contracts": contracts, "price_cents": price_cents,
                    "status": status, "reasoning": reasoning,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
            }))

        result = OrderResult(
            order_id=order_id, status=status, ticker=ticker,
            side=side, action=action, contracts=contracts,
            price_cents=price_cents, ttl_seconds=session.order_ttl,
            new_balance_cents=new_balance_cents,
            new_balance_dollars=new_balance_dollars,
        )
        return result.model_dump()

    except Exception as e:
        asyncio.create_task(_ctx.memory.store(
            content=f"FAILED ORDER: {action} {contracts} {side} {ticker} @{price_cents}c | {reasoning} | error: {e}",
            memory_type="trade",
            metadata={"ticker": ticker, "status": "failed", "error": str(e)},
        ))
        return OrderResult(error=f"Order failed: {e}").model_dump()


# --- Tool 4: execute_arb ---

@tool
async def execute_arb(
    event_ticker: str,
    direction: str,
    max_contracts: int,
    reasoning: str,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Execute a multi-leg arb trade. Returns new balance after execution.

    LONG ARB: Buy YES on every market. SHORT ARB: Buy NO on every market.
    Each leg placed as limit order at current best price.
    Use dry_run=True to preview without placing orders.

    Args:
        event_ticker: Event to trade
        direction: "long" (buy all YES) or "short" (buy all NO)
        max_contracts: Maximum contracts per leg
        reasoning: Why this trade should be made
        dry_run: Preview mode (no actual orders)

    Returns:
        ArbResult with status, legs, cost, and new_balance_cents
    """
    if not _ctx:
        return {"error": "ToolContext not available"}

    index = _ctx.index
    gw = _ctx.gateway
    session = _ctx.session
    event_state = index.events.get(event_ticker)

    if not event_state:
        return {"error": f"Event {event_ticker} not found"}
    if not event_state.all_markets_have_data:
        return {"error": f"Not all markets have data ({event_state.markets_with_data}/{event_state.markets_total})"}
    if direction not in ("long", "short"):
        return {"error": f"Invalid direction: {direction}. Must be 'long' or 'short'."}
    if not event_state.mutually_exclusive and direction == "short":
        return {"error": f"INDEPENDENT EVENT: {event_ticker} has mutually_exclusive=False. Short arb is not risk-free."}

    # Check balance (with retry)
    try:
        balance_obj = await retry_api(lambda: gw.get_balance(), max_retries=2, label="arb_balance")
        balance = balance_obj.balance
    except Exception as e:
        return {"error": f"Balance check failed: {e}"}

    # Build legs with uniform contract count across all markets
    legs = []
    errors = []
    min_available = max_contracts

    # First pass: validate all markets and find minimum available size
    for book in event_state.markets.values():
        if direction == "long":
            if book.yes_ask is None:
                errors.append(f"{book.ticker}: no YES ask")
                continue
            side, price = "yes", book.yes_ask
            available = book.yes_ask_size
        else:
            if book.yes_bid is None:
                errors.append(f"{book.ticker}: no YES bid")
                continue
            side, price = "no", 100 - book.yes_bid
            available = book.yes_bid_size

        min_available = min(min_available, available)
        legs.append(ArbLegResult(
            ticker=book.ticker, title=book.title,
            side=side, contracts=0, price_cents=price,
        ))

    # Compute uniform contract count (same for every leg)
    uniform_contracts = max(1, min_available)
    total_cost = sum(leg.price_cents * uniform_contracts for leg in legs)

    if total_cost > balance:
        # Scale down to fit balance
        total_price_per_contract = sum(leg.price_cents for leg in legs)
        if total_price_per_contract > 0:
            uniform_contracts = max(1, balance // total_price_per_contract)
            total_cost = total_price_per_contract * uniform_contracts

    # Apply uniform count to all legs
    for leg in legs:
        leg.contracts = uniform_contracts

    if dry_run:
        return ArbResult(
            status="preview", event_ticker=event_ticker, direction=direction,
            legs_executed=len(legs), legs_total=event_state.markets_total,
            legs=legs, total_cost_cents=total_cost, errors=errors,
            new_balance_cents=balance - total_cost,
            new_balance_dollars=round((balance - total_cost) / 100, 2),
        ).model_dump()

    # Execute all legs in parallel (following sniper._execute_arb pattern)
    expiration_ts = int(time.time()) + session.order_ttl

    async def _place_arb_leg(leg: ArbLegResult):
        """Place a single arb leg order. Returns (leg, order_resp) on success."""
        order_kwargs = {
            "ticker": leg.ticker, "action": "buy", "side": leg.side,
            "count": leg.contracts, "price": leg.price_cents,
            "type": "limit",
            "expiration_ts": expiration_ts,
        }
        if session.order_group_id:
            order_kwargs["order_group_id"] = session.order_group_id
        return leg, await gw.create_order(**order_kwargs)

    results = await asyncio.gather(
        *[_place_arb_leg(leg) for leg in legs],
        return_exceptions=True,
    )

    # Process results
    exec_legs = []
    exec_cost = 0
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            errors.append(f"{legs[i].ticker}: {result}")
            continue
        leg, order_resp = result
        order = order_resp.order
        order_id = order.order_id

        if order_id:
            _ctx.captain_order_ids.add(order_id)
            session.captain_order_ids.add(order_id)
            _ctx.order_initial_states[order_id] = {
                "ticker": leg.ticker, "side": leg.side, "action": "buy",
                "contracts": leg.contracts, "price_cents": leg.price_cents,
                "placed_at": time.time(), "ttl_seconds": session.order_ttl,
                "status": order.status or "placed",
            }

        exec_legs.append(ArbLegResult(
            ticker=leg.ticker, title=leg.title,
            side=leg.side, contracts=leg.contracts,
            price_cents=leg.price_cents,
            order_id=order_id, status=order.status or "placed",
        ))
        exec_cost += leg.contracts * leg.price_cents

        # Record each leg to decision ledger (fire-and-forget)
        if _ctx.decision_ledger:
            asyncio.create_task(_ctx.decision_ledger.record_decision(
                order_id=order_id, source="captain",
                event_ticker=event_ticker, market_ticker=leg.ticker,
                side=leg.side, action="buy", contracts=leg.contracts,
                limit_price_cents=leg.price_cents, reasoning=reasoning,
                cycle_mode=_ctx.cycle_mode,
            ))

    status = "completed" if len(exec_legs) == event_state.markets_total else "partial"
    if not exec_legs:
        status = "failed"

    # Log partial fill situation (don't try to unwind — Captain handles next cycle)
    if status == "partial":
        logger.warning(
            f"[TOOLS:ARB] PARTIAL {event_ticker} {direction} "
            f"legs={len(exec_legs)}/{event_state.markets_total} — "
            f"Captain should review unhedged exposure next cycle"
        )

    # Track cycle capital spend for arb legs
    _ctx.cycle_capital_spent_cents += exec_cost

    new_balance_cents, new_balance_dollars = await _fetch_balance_with_budget_check("ARB")

    # Record + broadcast
    asyncio.create_task(_ctx.memory.store(
        content=f"ARB: {direction} {event_ticker} | {len(exec_legs)}/{event_state.markets_total} legs | cost={exec_cost}c | {reasoning}",
        memory_type="trade",
        metadata={"event_ticker": event_ticker, "direction": direction, "status": status},
    ))
    if _ctx.broadcast and exec_legs:
        for leg in exec_legs:
            asyncio.create_task(_ctx.broadcast({
                "type": "arb_trade_executed",
                "data": {
                    "order_id": leg.order_id, "event_ticker": event_ticker,
                    "kalshi_ticker": leg.ticker, "direction": direction,
                    "side": leg.side, "contracts": leg.contracts,
                    "price_cents": leg.price_cents, "status": leg.status,
                    "reasoning": reasoning,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
            }))

    return ArbResult(
        status=status, event_ticker=event_ticker, direction=direction,
        legs_executed=len(exec_legs), legs_total=event_state.markets_total,
        legs=exec_legs, total_cost_cents=exec_cost,
        new_balance_cents=new_balance_cents,
        new_balance_dollars=new_balance_dollars,
        errors=errors,
    ).model_dump()


# --- Tool 5: cancel_order ---

@tool
async def cancel_order(order_id: str, reason: str = "") -> Dict[str, Any]:
    """Cancel a resting order.

    Args:
        order_id: The order ID to cancel
        reason: Optional reason for cancellation

    Returns:
        Status dict with order_id
    """
    if not _ctx:
        return {"error": "ToolContext not available"}
    try:
        await retry_api(
            lambda: _ctx.gateway.cancel_order(order_id),
            max_retries=2, label=f"cancel_order({order_id[:8]})",
        )
        if reason:
            asyncio.create_task(_ctx.memory.store(
                content=f"CANCEL: order {order_id[:8]}... | {reason}",
                memory_type="trade",
                metadata={"order_id": order_id, "action": "cancel"},
            ))
        new_balance_cents, new_balance_dollars = await _fetch_balance_with_budget_check("CANCEL")
        return {
            "status": "cancelled", "order_id": order_id,
            "new_balance_cents": new_balance_cents,
            "new_balance_dollars": new_balance_dollars,
        }
    except Exception as e:
        if "not found" in str(e).lower() or "404" in str(e):
            return {"status": "already_gone", "order_id": order_id}
        return {"error": f"Cancel failed: {e}"}


# --- Tool 6: get_resting_orders ---

@tool
async def get_resting_orders(ticker: Optional[str] = None) -> Dict[str, Any]:
    """Get currently open/resting orders with queue position info.

    Args:
        ticker: Filter to specific market ticker (optional)

    Returns:
        List of RestingOrder with queue positions
    """
    if not _ctx:
        return {"error": "ToolContext not available"}
    gw = _ctx.gateway
    try:
        raw_orders = await gw.get_orders(ticker=ticker, status="resting")

        queue_map = {}
        if raw_orders:
            try:
                market_ticker = ticker or raw_orders[0].ticker
                if market_ticker:
                    qp_list = await gw.get_queue_positions(market_tickers=market_ticker)
                    for qp in qp_list:
                        queue_map[qp.order_id] = qp.queue_position
            except Exception:
                pass

        orders = [
            RestingOrder(
                order_id=o.order_id,
                ticker=o.ticker,
                side=o.side,
                action=o.action,
                price_cents=o.price or o.yes_price or 0,
                remaining_count=o.remaining_count,
                queue_position=queue_map.get(o.order_id),
            ).model_dump()
            for o in raw_orders
        ]
        return {"count": len(orders), "orders": orders}
    except Exception as e:
        return {"error": f"Get orders failed: {e}"}


# --- Tool 7: search_news ---

async def _build_price_snapshot(event_ticker: Optional[str]) -> Optional[Dict]:
    """Build enriched price snapshot for news-price impact tracking."""
    if not event_ticker or not _ctx or not _ctx.index:
        return None
    event = _ctx.index.events.get(event_ticker)
    if not event:
        return None
    snap = {}
    for mt, mm in event.markets.items():
        if mm.yes_mid is not None:
            snap[mt] = {
                "yes_bid": mm.yes_bid,
                "yes_ask": mm.yes_ask,
                "yes_mid": mm.yes_mid,
                "spread": mm.spread,
                "volume_5m": mm.micro.volume_5m,
                "book_imbalance": round(mm.micro.book_imbalance, 3),
                "open_interest": mm.open_interest,
            }
    if snap:
        snap["_ts"] = time.time()
        return snap
    return None


async def _enrich_with_patterns(articles: List[NewsArticle]) -> int:
    """Enrich articles with similar_patterns from the swing-news index.

    For each article, embed the title+content and search for historically
    similar news that moved prices. Returns count of patterns found.

    Budget: 10s total, 3s per embedding, 3s per DB query. Stops enriching
    remaining articles on budget exhaustion.
    """
    try:
        from kalshiflow_rl.data.database import rl_db
        pool = await asyncio.wait_for(rl_db.get_pool(), timeout=3.0)
    except (Exception, asyncio.TimeoutError):
        return 0

    if not _ctx or not _ctx.memory:
        return 0

    total_patterns = 0
    embeddings = _ctx.memory._get_embeddings()
    if not embeddings:
        return 0

    loop = asyncio.get_running_loop()
    deadline = loop.time() + 10.0

    for article in articles:
        if not article.title:
            continue
        # Check total budget
        if loop.time() >= deadline:
            logger.info(f"[TOOLS:PATTERNS] Budget exhausted after enriching {total_patterns} patterns, skipping remaining articles")
            break
        try:
            embed_text = f"{article.title} {article.content[:300]}"
            embedding = await asyncio.wait_for(
                asyncio.to_thread(embeddings.embed_query, embed_text),
                timeout=3.0,
            )

            async with asyncio.timeout(3.0):
                async with pool.acquire() as conn:
                    rows = await conn.fetch(
                        "SELECT * FROM find_similar_impact_patterns($1::vector, $2, $3)",
                        str(embedding), 0.5, 3,
                    )

            for row in rows:
                article.similar_patterns.append(ImpactPattern(
                    news_title=row["news_title"] or "",
                    news_url=row["news_url"] or "",
                    direction=row["direction"] or "",
                    change_cents=row["change_cents"] or 0.0,
                    confidence=row["causal_confidence"] or 0.0,
                    similarity=row["similarity"] or 0.0,
                    event_ticker=row["event_ticker"] or "",
                    market_ticker=row["market_ticker"] or "",
                ))
                total_patterns += 1
        except asyncio.TimeoutError:
            logger.debug(f"[TOOLS:PATTERNS] Timeout enriching article: {article.title[:50]}")
        except Exception as e:
            logger.debug(f"[TOOLS:PATTERNS] Pattern enrichment error: {e}")

    return total_patterns


def _resolve_depth(depth: str) -> str:
    """Resolve 'auto' depth based on current cycle mode."""
    if depth != "auto":
        return depth
    if not _ctx or not _ctx.cycle_mode:
        return "fast"
    mode_map = {"reactive": "ultra_fast", "strategic": "fast", "deep_scan": "advanced"}
    return mode_map.get(_ctx.cycle_mode, "fast")


@tool
async def search_news(
    query: str,
    event_ticker: Optional[str] = None,
    depth: str = "auto",
    time_range: str = "week",
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """Search for news with pattern-aware impact predictions.

    Three depth tiers (auto-selected by cycle mode, or override manually):
    - ultra_fast: Memory recall only (0 credits, <10ms). Best for reactive cycles.
    - fast: Memory + Tavily basic search (1 credit, ~2s). Default for strategic.
    - advanced: Memory + Tavily advanced + full content + analysis (2 credits, ~5s). For deep_scan.

    Each article is enriched with similar_patterns: historical news that looked
    similar AND moved prices. Use these predictions to trade ahead of price moves.

    Args:
        query: Search query (be specific: "Fed rate decision March 2026")
        event_ticker: Optional event ticker for budget tracking
        depth: "ultra_fast", "fast", "advanced", or "auto" (default "auto" = cycle-mode based)
        time_range: "day", "week", "month" (default "week")
        force_refresh: Bypass cache and fetch fresh results (default False)

    Returns:
        SwingNewsResult with articles, pattern counts, and depth used
    """
    if not _ctx:
        return {"error": "ToolContext not available"}

    effective_depth = _resolve_depth(depth)

    # Check cache (skip for ultra_fast since it's already instant)
    cache_key = f"{query.lower().strip()}|{event_ticker or ''}|{time_range}|{effective_depth}"
    if not force_refresh and effective_depth != "ultra_fast" and cache_key in _ctx.news_cache:
        cached_ts, cached_result = _ctx.news_cache[cache_key]
        if time.time() - cached_ts < NEWS_CACHE_TTL:
            _ctx.news_cache.move_to_end(cache_key)
            cached_result["cached"] = True
            return cached_result

    articles: List[NewsArticle] = []
    tavily_answer = ""

    # --- TIER 1: ultra_fast (memory only) ---
    # Always run: recall from FAISS + pgvector, filtering for swing_news type
    try:
        recalled = await _ctx.memory.recall(
            query=query, limit=5, memory_types=["swing_news", "news"],
        )
        if recalled and recalled.results:
            for mem in recalled.results:
                articles.append(NewsArticle(
                    title=mem.content.split("\n")[0].replace("SWING_NEWS: ", "").replace("NEWS: ", "")[:100],
                    content=mem.content[:500],
                    source="memory",
                    score=mem.similarity,
                ))
    except Exception as e:
        logger.debug(f"[TOOLS:SEARCH] Memory recall error: {e}")

    # --- TIER 2: fast (+ Tavily basic search) ---
    if effective_depth in ("fast", "advanced") and _ctx.search:
        try:
            tavily_depth = "fast" if effective_depth == "fast" else "advanced"
            tavily_topic = "finance" if effective_depth == "advanced" else "news"
            include_raw = effective_depth == "advanced"
            include_answer_mode = "advanced" if effective_depth == "advanced" else True

            raw_results = await asyncio.wait_for(
                _ctx.search.search_news(
                    query=query,
                    time_range=time_range,
                    include_raw_content=include_raw,
                    event_ticker=event_ticker or "",
                    search_depth=tavily_depth,
                ),
                timeout=12.0,
            )

            for r in raw_results[:5]:
                # Check for answer summary
                if r.get("answer_summary") and not tavily_answer:
                    tavily_answer = r["answer_summary"]
                articles.append(NewsArticle(
                    title=r.get("title", ""),
                    url=r.get("url", ""),
                    content=r.get("content", "")[:500],
                    raw_content=r.get("raw_content", "")[:2000] if include_raw else "",
                    published_date=r.get("published_date", ""),
                    score=r.get("score", 0.0),
                    source=r.get("source", ""),
                ))
        except asyncio.TimeoutError:
            logger.warning("[TOOLS:SEARCH] Tavily search timed out after 12s, using memory-only results")
        except Exception as e:
            logger.warning(f"[TOOLS:SEARCH] Tavily search error: {e}")

    # Deduplicate articles by title similarity
    seen_titles: List[str] = []
    deduped: List[NewsArticle] = []
    for art in articles:
        title_lower = art.title.lower().strip()
        if any(title_lower == s for s in seen_titles):
            continue
        seen_titles.append(title_lower)
        deduped.append(art)
    articles = deduped[:8]

    # --- Pattern enrichment (all tiers) ---
    patterns_found = 0
    if articles:
        try:
            patterns_found = await _enrich_with_patterns(articles)
        except Exception as e:
            logger.debug(f"[TOOLS:SEARCH] Pattern enrichment failed: {e}")

    # --- Auto-store to memory (fast + advanced only) ---
    stored = False
    if effective_depth in ("fast", "advanced") and articles:
        price_snapshot = await _build_price_snapshot(event_ticker)
        for article in articles[:3]:
            if article.source == "memory":
                continue  # Don't re-store memory recalls
            memory_content = (
                f"NEWS: {article.title}\n"
                f"Source: {article.source} | {article.published_date}\n"
                f"URL: {article.url}\n\n"
                f"{article.content}"
            )
            metadata = {
                "query": query,
                "event_ticker": event_ticker,
                "news_url": article.url,
                "news_title": article.title,
                "news_published_at": article.published_date or None,
                "news_source": article.source,
                "price_snapshot": price_snapshot,
            }
            asyncio.create_task(_ctx.memory.store(
                content=memory_content,
                memory_type="news",
                metadata=metadata,
            ))
        stored = True

    result = SwingNewsResult(
        query=query,
        articles=articles,
        count=len(articles),
        stored_in_memory=stored,
        cached=False,
        depth=effective_depth,
        patterns_found=patterns_found,
        tavily_answer=tavily_answer,
    ).model_dump()

    # Store in cache (LRU eviction at 200 entries)
    if effective_depth != "ultra_fast":
        _ctx.news_cache[cache_key] = (time.time(), result)
        if len(_ctx.news_cache) > 200:
            _ctx.news_cache.popitem(last=False)

    return result


# --- Tool 8: recall_memory ---

@tool
async def recall_memory(query: str, limit: int = 5, memory_type: str = "") -> Dict[str, Any]:
    """Search session + persistent memory for relevant context.

    Hybrid search: fast FAISS (this session) + pgvector (all sessions).
    Results sorted by similarity.

    Args:
        query: What to search for (e.g., "Fed rate decision analysis")
        limit: Max results (default 5)
        memory_type: Filter by type (e.g., "trade_outcome", "trade", "news", "learning"). Empty = all types.

    Returns:
        RecallResult with matched memories
    """
    if not _ctx:
        return {"error": "ToolContext not available"}
    memory_types = [memory_type] if memory_type else None
    try:
        result = await asyncio.wait_for(
            _ctx.memory.recall(query=query, limit=limit, memory_types=memory_types),
            timeout=10.0,
        )
        return result.model_dump()
    except asyncio.TimeoutError:
        logger.warning("[TOOLS:RECALL] Memory recall timed out after 10s")
        return RecallResult(results=[], count=0).model_dump()


# --- Tool 9: store_insight ---

# Patterns that indicate self-imposed restrictions (toxic behavioral rules)
TOXIC_PATTERNS = [
    "never trade", "pause trading", "freeze", "crisis",
    "wait for accuracy", "recovery plan", "mandatory",
    "no directional", "avoid all", "stop trading",
    "trading freeze", "self-imposed", "accuracy gate",
    "performance threshold",
]


@tool
async def store_insight(
    content: str,
    memory_type: str = "learning",
    event_ticker: str = "",
    tags: str = "",
) -> Dict[str, Any]:
    """Store a learning or insight to session + persistent memory.

    Stored in both FAISS (session, instant recall) and pgvector (persistent,
    survives restarts). Use for lessons, observations, or rules.

    ALWAYS include event_ticker when storing insights about a specific event.
    This enables event-scoped recall later.

    Args:
        content: The insight to store (e.g., "IF VPIN > 0.85 THEN reduce position size")
        memory_type: Category: learning, observation, pattern, mistake (default "learning")
        event_ticker: Event this insight relates to (e.g., "KXFEDCHAIRNOM-29"). Always include for event-specific insights.
        tags: Comma-separated tags for categorization (e.g., "arb,fee,lesson")

    Returns:
        Confirmation dict
    """
    if not _ctx:
        return {"error": "ToolContext not available"}

    # Block self-imposed restrictions from being stored
    content_lower = content.lower()
    for pattern in TOXIC_PATTERNS:
        if pattern in content_lower:
            return {
                "status": "rejected",
                "reason": "Self-imposed restrictions are not allowed. Store factual observations only.",
            }

    metadata: Dict[str, Any] = {}
    if event_ticker:
        metadata["event_ticker"] = event_ticker
    if tags:
        metadata["tags"] = [t.strip() for t in tags.split(",") if t.strip()]
    await _ctx.memory.store(content=content, memory_type=memory_type, metadata=metadata)
    return {"status": "stored", "memory_type": memory_type, "length": len(content),
            "event_ticker": event_ticker or None}


# --- Tool 10: configure_sniper ---

@tool
async def configure_sniper(settings: Dict[str, Any]) -> Dict[str, Any]:
    """Configure the automated sniper execution layer.

    The sniper auto-executes S1_ARB opportunities on the hot path.
    You CONFIGURE it, it EXECUTES autonomously.

    Args:
        settings: Dict of config parameters. Common keys:
            enabled: bool - Turn sniper on/off
            arb_min_edge: float - Minimum edge in cents to trigger (default 3.0)
            max_capital: int - Max capital in cents (default 100000 = $1000, minimum 5000 = $50)
            max_position: int - Max contracts per leg (default 25)
            cooldown: float - Seconds between trades on same event (default 10)
            vpin_reject_threshold: float - VPIN above this blocks trades (default 0.85)

    Returns:
        Dict with changed fields and current config
    """
    if not _ctx or not _ctx.sniper_config:
        return {"error": "Sniper not available"}

    if not settings:
        return {"error": "No parameters provided. Pass enabled=True/False, arb_min_edge, etc."}

    changed, unknown = _ctx.sniper_config.update(**settings)
    result = {
        "status": "updated" if changed else "no_changes",
        "changed": changed,
        "current_config": {
            "enabled": _ctx.sniper_config.enabled,
            "arb_min_edge": _ctx.sniper_config.arb_min_edge,
            "max_capital": _ctx.sniper_config.max_capital,
            "max_position": _ctx.sniper_config.max_position,
            "cooldown": _ctx.sniper_config.cooldown,
            "vpin_reject_threshold": _ctx.sniper_config.vpin_reject_threshold,
        },
    }
    if unknown:
        result["unknown_fields"] = unknown
    return result


# --- Tool 11: get_account_health ---

@tool
async def get_account_health() -> Dict[str, Any]:
    """Get account health snapshot: balance, drawdown, settlements, stale positions, alerts.

    Health data is pre-injected in STRATEGIC and DEEP_SCAN briefings.
    Only call after a significant mid-cycle balance change where you need fresh drawdown data.

    Returns:
        AccountHealthStatus with balance, drawdown, settlements, alerts, and status
    """
    if not _ctx or not _ctx.health_service:
        return {"error": "AccountHealthService not available"}
    status = _ctx.health_service.get_health_status()
    return status.model_dump()


# --- Tool 12: configure_automation ---

@tool
async def configure_automation(settings: Dict[str, Any]) -> Dict[str, Any]:
    """Configure automated trading actions (stop_loss, time_exit, regime_gate).

    Captain retains override authority over all auto-actions. Auto-actions fire
    deterministically when conditions are met, but you can tune thresholds,
    enable/disable per action, and set per-ticker/event overrides.

    Args:
        settings: Must include "action" key. Valid actions: "stop_loss", "time_exit", "regime_gate".
            Common settings:
            - "action": str (required) — which auto-action to configure
            - "enabled": bool — enable/disable action globally
            - "ticker": str — set per-ticker override (for stop_loss)
            - "event": str — set per-event override (for time_exit, regime_gate)

            stop_loss settings:
            - "threshold": int — P&L threshold in cents/contract (default -12)

            time_exit settings:
            - "threshold_minutes": int — minutes before close to exit (default 30)
            - "hold_through": bool — per-event override to hold through settlement

            regime_gate settings:
            - "cooldown": float — seconds to pause sniper after toxic regime (default 300)
            - "ignore_regime": bool — per-event override to ignore regime changes

    Returns:
        Updated configuration for the specified action, or all actions if action="status".
    """
    if not _ctx or not _ctx.auto_actions:
        return {"error": "Auto-actions not available"}

    action_name = settings.get("action", "")
    if action_name == "status":
        return _ctx.auto_actions.get_stats()

    if not action_name:
        return {"error": "Missing 'action' key. Valid: stop_loss, time_exit, regime_gate, status"}

    # Remove the "action" key before passing to configure
    config_settings = {k: v for k, v in settings.items() if k != "action"}
    return _ctx.auto_actions.configure(action_name, config_settings)


# --- Tool 13: get_market_movers ---

@tool
async def get_market_movers(
    event_ticker: str,
    min_change_cents: int = 5,
) -> Dict[str, Any]:
    """Find news articles that moved prices for this event.

    Queries the news_price_impacts table (populated every 30min) for articles
    where price changed >= min_change_cents at 1h, 4h, or 24h after publication.
    Use in DEEP_SCAN to understand which news types actually move prices.

    Args:
        event_ticker: Event to query (e.g., "KXEVENT-ABC")
        min_change_cents: Minimum price change to qualify (default 5c)

    Returns:
        Dict with movers list and count
    """
    if not _ctx:
        return {"error": "ToolContext not available"}

    try:
        from kalshiflow_rl.data.database import rl_db
        pool = await asyncio.wait_for(rl_db.get_pool(), timeout=3.0)
    except asyncio.TimeoutError:
        return {"error": "Database pool busy", "movers": [], "count": 0}
    except Exception as e:
        return {"error": f"Database not available: {e}", "movers": [], "count": 0}

    try:
        async with asyncio.timeout(5.0):
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT * FROM find_market_movers($1, $2, $3)",
                    event_ticker, min_change_cents, 10,
                )

        movers = [
            {
                "news_title": row["news_title"],
                "news_url": row["news_url"],
                "change_cents": row["change_cents"],
                "direction": row["direction"],
                "delay_hours": row["delay_hours"],
                "market_ticker": row["market_ticker"],
            }
            for row in rows
        ]
        return {"event_ticker": event_ticker, "movers": movers, "count": len(movers)}

    except asyncio.TimeoutError:
        logger.warning(f"[TOOLS:MARKET_MOVERS] Query timed out for {event_ticker}")
        return {"error": "Database pool busy", "movers": [], "count": 0}
    except Exception as e:
        logger.warning(f"[TOOLS:MARKET_MOVERS] Query failed: {e}")
        return {"error": f"Query failed: {e}", "movers": [], "count": 0}


# --- Tool 14: get_early_bird_opportunities ---

@tool
async def get_early_bird_opportunities() -> Dict[str, Any]:
    """Get recently activated markets with early bird opportunity scores.

    Returns scored opportunities for markets that just opened for trading.
    Score breakdown includes complement pricing, category familiarity, timing, and risk.
    Use in STRATEGIC or DEEP_SCAN to discover fresh markets worth investigating.

    Returns:
        Dict with opportunities list and count
    """
    if not _ctx or not _ctx.early_bird_service:
        return {"opportunities": [], "note": "Early bird service not available", "count": 0}

    opportunities = _ctx.early_bird_service.get_recent_opportunities()
    return {"opportunities": opportunities, "count": len(opportunities)}


# --- Tool 15: configure_quotes ---

@tool
async def configure_quotes(settings: Dict[str, Any]) -> Dict[str, Any]:
    """Configure the QuoteEngine (market maker) parameters.

    Adjustable settings (pass only the ones you want to change):
      - enabled (bool): Enable/disable quoting
      - base_spread_cents (int): Minimum spread width (default 4)
      - quote_size (int): Contracts per side per market (default 10)
      - skew_factor (float): Inventory skew multiplier (default 0.5)
      - max_position (int): Max contracts per market one side (default 100)
      - refresh_interval (float): Seconds between requote cycles (default 5.0)

    Args:
        settings: Dict of QuoteConfig fields to update

    Returns:
        Dict with status, current config, and list of changes made
    """
    from ..market_maker.models import ConfigureQuotesResult

    if not _ctx or not _ctx.quote_engine:
        return ConfigureQuotesResult(
            status="unavailable",
            config={},
            changes=["QuoteEngine not active"],
        ).model_dump()

    qc = _ctx.quote_config
    if not qc:
        return ConfigureQuotesResult(
            status="unavailable",
            config={},
            changes=["QuoteConfig not available"],
        ).model_dump()

    ALLOWED = {
        "enabled", "base_spread_cents", "quote_size", "skew_factor",
        "skew_cap_cents", "max_position", "max_event_exposure",
        "refresh_interval", "pull_quotes_threshold",
    }
    changes = []
    for key, value in settings.items():
        if key not in ALLOWED:
            continue
        old = getattr(qc, key, None)
        if old != value:
            setattr(qc, key, value)
            changes.append(f"{key}: {old} → {value}")

    config_dict = {
        "enabled": qc.enabled,
        "base_spread_cents": qc.base_spread_cents,
        "quote_size": qc.quote_size,
        "skew_factor": qc.skew_factor,
        "max_position": qc.max_position,
        "refresh_interval": qc.refresh_interval,
    }

    return ConfigureQuotesResult(
        status="updated" if changes else "no_changes",
        config=config_dict,
        changes=changes,
    ).model_dump()


# --- Tool 16: pull_quotes ---

@tool
async def pull_quotes(reason: str) -> Dict[str, Any]:
    """Emergency pull all market maker quotes.

    Cancels all active MM quotes across all markets. Use when:
    - VPIN spike detected on MM markets
    - Adverse news on MM events
    - Inventory risk too high

    Args:
        reason: Why quotes are being pulled (logged for telemetry)

    Returns:
        Dict with status and count of cancelled orders
    """
    from ..market_maker.models import PullQuotesResult

    if not _ctx or not _ctx.quote_engine:
        return PullQuotesResult(
            status="unavailable", reason="QuoteEngine not active"
        ).model_dump()

    cancelled = await _ctx.quote_engine.pull_all_quotes(reason)
    return PullQuotesResult(
        status="pulled",
        cancelled_orders=cancelled,
        reason=reason,
    ).model_dump()


# --- Tool 17: resume_quotes ---

@tool
async def resume_quotes(reason: str) -> Dict[str, Any]:
    """Resume market maker quoting after a pull.

    Clears the pulled state so QuoteEngine resumes posting 2-sided quotes
    on the next refresh cycle.

    Args:
        reason: Why quotes are being resumed (logged for telemetry)

    Returns:
        Dict with status
    """
    from ..market_maker.models import PullQuotesResult

    if not _ctx or not _ctx.quote_engine:
        return PullQuotesResult(
            status="unavailable", reason="QuoteEngine not active"
        ).model_dump()

    _ctx.quote_engine.resume_quotes()
    return PullQuotesResult(
        status="resumed",
        cancelled_orders=0,
        reason=reason,
    ).model_dump()


# --- Tool 18: get_quote_performance ---

@tool
async def get_quote_performance() -> Dict[str, Any]:
    """Get market maker performance telemetry.

    Returns fills, spread capture, adverse selection, fees, net P&L,
    and operational stats. Use in STRATEGIC/DEEP_SCAN to evaluate MM health.

    Returns:
        Dict with MM performance metrics
    """
    from ..market_maker.models import QuotePerformanceResult

    if not _ctx or not _ctx.quote_engine:
        return QuotePerformanceResult().model_dump()

    state = _ctx.quote_engine.state
    mm_index = _ctx.mm_index
    realized = mm_index.total_realized_pnl() if mm_index else 0.0
    total_fills = state.total_fills_bid + state.total_fills_ask
    total_cycles = max(state.total_requote_cycles, 1)

    return QuotePerformanceResult(
        total_fills_bid=state.total_fills_bid,
        total_fills_ask=state.total_fills_ask,
        total_requote_cycles=state.total_requote_cycles,
        spread_captured_cents=round(state.spread_captured_cents, 2),
        adverse_selection_cents=round(state.adverse_selection_cents, 2),
        fees_paid_cents=round(state.fees_paid_cents, 2),
        net_pnl_cents=round(realized, 2),
        quote_uptime_pct=state.quote_uptime_pct,
        fill_rate_pct=round(total_fills / total_cycles * 100, 1) if total_fills else 0.0,
        spread_multiplier=state.spread_multiplier,
        fill_storm_active=state.fill_storm_active,
    ).model_dump()


# --- Tool list for Captain V2 ---

ALL_TOOLS = [
    get_market_state,
    get_portfolio,
    place_order,
    execute_arb,
    cancel_order,
    get_resting_orders,
    search_news,
    recall_memory,
    store_insight,
    get_account_health,
    configure_automation,
    configure_sniper,
    get_market_movers,
    get_early_bird_opportunities,
    configure_quotes,
    pull_quotes,
    resume_quotes,
    get_quote_performance,
]

# Tool categorization for frontend event routing
TOOL_CATEGORIES = {
    "get_market_state": "arb",
    "get_portfolio": "arb",
    "place_order": "arb",
    "execute_arb": "arb",
    "cancel_order": "arb",
    "get_resting_orders": "arb",
    "search_news": "arb",
    "recall_memory": "memory",
    "store_insight": "memory",
    "configure_automation": "sniper",
    "configure_sniper": "sniper",
    "get_account_health": "system",
    "get_market_movers": "arb",
    "get_early_bird_opportunities": "arb",
    "configure_quotes": "mm",
    "pull_quotes": "mm",
    "resume_quotes": "mm",
    "get_quote_performance": "mm",
    "write_todos": "todo",
}
