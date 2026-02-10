"""Captain V2 tools - 10 tools for single-agent Captain.

Single ToolContext dataclass replaces 14 module-level globals.
All tools return Pydantic models (serialized via .model_dump()).
Order tools return new_balance so agent never needs a separate balance query.
"""

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, TYPE_CHECKING

from langchain_core.tools import tool

from .models import (
    ArbLegResult,
    ArbResult,
    MarketState,
    NewsArticle,
    NewsSearchResult,
    OrderResult,
    PortfolioState,
    RecallResult,
    RestingOrder,
)

if TYPE_CHECKING:
    from .account_health import AccountHealthService
    from .context_builder import ContextBuilder
    from .index import EventArbIndex
    from .memory.session_store import SessionMemoryStore
    from .sniper import Sniper, SniperConfig
    from .tavily_service import TavilySearchService
    from ..agent_tools.session import TradingSession
    from ..gateway.client import KalshiGateway

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
    captain_order_ids: Set[str] = field(default_factory=set)
    order_initial_states: Dict[str, dict] = field(default_factory=dict)
    news_cache: Dict[str, tuple] = field(default_factory=dict)
    cycle_capital_spent_cents: int = 0


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


# --- Tool 1: get_market_state ---

@tool
async def get_market_state() -> Dict[str, Any]:
    """Get current market state for all monitored events.

    Returns structured JSON with events, markets, edges, regime, and semantics.
    Usually unnecessary since the cycle briefing already contains this data.
    Use for mid-cycle refresh if you suspect data has changed significantly.

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

    Returns balance, positions (sorted by unrealized P&L), and totals.
    Each position includes exit_price and unrealized_pnl_cents.

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
        contracts: Number of contracts (1-100)
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
    if not (1 <= contracts <= 100):
        return {"error": f"Contracts must be 1-100, got {contracts}"}
    if not (1 <= price_cents <= 99):
        return {"error": f"Price must be 1-99 cents, got {price_cents}"}

    gw = _ctx.gateway
    session = _ctx.session

    # Position conflict guard (block buying opposite side on same market)
    if action == "buy":
        try:
            event_for_ticker = _ctx.index.get_event_for_ticker(ticker) if _ctx.index else None
            raw_positions = await gw.get_positions(
                event_ticker=event_for_ticker,
            )
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

        # Track cycle capital spend
        _ctx.cycle_capital_spent_cents += contracts * price_cents

        # Fetch new balance (warn on failure instead of silent pass)
        new_balance_cents = None
        new_balance_dollars = None
        try:
            bal = await gw.get_balance()
            new_balance_cents = bal.balance
            new_balance_dollars = round(new_balance_cents / 100, 2)
            # Warn if cycle spend exceeds 30% of balance
            if new_balance_cents and _ctx.cycle_capital_spent_cents > new_balance_cents * 0.3:
                logger.warning(
                    f"[TOOLS:BUDGET] Cycle capital spend {_ctx.cycle_capital_spent_cents}c "
                    f"exceeds 30% of balance {new_balance_cents}c"
                )
        except Exception as bal_err:
            logger.warning(f"[TOOLS:TRADE] Balance fetch failed after order: {bal_err}")

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

    # Build legs
    legs = []
    total_cost = 0
    errors = []

    for book in event_state.markets.values():
        if direction == "long":
            if book.yes_ask is None:
                errors.append(f"{book.ticker}: no YES ask")
                continue
            side, price = "yes", book.yes_ask
        else:
            if book.yes_bid is None:
                errors.append(f"{book.ticker}: no YES bid")
                continue
            side, price = "no", 100 - book.yes_bid

        contracts = min(max_contracts, book.yes_ask_size if direction == "long" else book.yes_bid_size)
        contracts = max(contracts, 1)
        leg_cost = contracts * price
        if total_cost + leg_cost > balance:
            errors.append(f"{book.ticker}: insufficient balance")
            continue

        legs.append(ArbLegResult(
            ticker=book.ticker, title=book.title,
            side=side, contracts=contracts, price_cents=price,
        ))
        total_cost += leg_cost

    if dry_run:
        return ArbResult(
            status="preview", event_ticker=event_ticker, direction=direction,
            legs_executed=len(legs), legs_total=event_state.markets_total,
            legs=legs, total_cost_cents=total_cost, errors=errors,
            new_balance_cents=balance - total_cost,
            new_balance_dollars=round((balance - total_cost) / 100, 2),
        ).model_dump()

    # Execute orders
    exec_legs = []
    exec_cost = 0
    for leg in legs:
        try:
            order_kwargs = {
                "ticker": leg.ticker, "action": "buy", "side": leg.side,
                "count": leg.contracts, "price": leg.price_cents,
                "type": "limit",
                "expiration_ts": int(time.time()) + session.order_ttl,
            }
            if session.order_group_id:
                order_kwargs["order_group_id"] = session.order_group_id

            order_resp = await gw.create_order(**order_kwargs)
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
        except Exception as e:
            errors.append(f"{leg.ticker}: {e}")

    status = "completed" if len(exec_legs) == event_state.markets_total else "partial"
    if not exec_legs:
        status = "failed"

    # Unwind on partial fill: cancel successful legs to avoid unhedged exposure
    if status == "partial" and exec_legs:
        unwind_count = 0
        for el in exec_legs:
            if el.order_id:
                try:
                    await gw.cancel_order(el.order_id)
                    unwind_count += 1
                except Exception as cancel_err:
                    errors.append(f"unwind {el.ticker}: {cancel_err}")
        if unwind_count:
            errors.append(f"Unwound {unwind_count}/{len(exec_legs)} legs due to partial fill")
            status = "unwound"

    # Track cycle capital spend for arb legs
    _ctx.cycle_capital_spent_cents += exec_cost

    # Get new balance (warn on failure instead of silent pass)
    new_balance_cents = None
    new_balance_dollars = None
    try:
        bal = await gw.get_balance()
        new_balance_cents = bal.balance
        new_balance_dollars = round(new_balance_cents / 100, 2)
        # Warn if cycle spend exceeds 30% of balance
        if new_balance_cents and _ctx.cycle_capital_spent_cents > new_balance_cents * 0.3:
            logger.warning(
                f"[TOOLS:BUDGET] Cycle capital spend {_ctx.cycle_capital_spent_cents}c "
                f"exceeds 30% of balance {new_balance_cents}c"
            )
    except Exception as bal_err:
        logger.warning(f"[TOOLS:ARB] Balance fetch failed after arb: {bal_err}")

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
        return {"status": "cancelled", "order_id": order_id}
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

@tool
async def search_news(
    query: str,
    event_ticker: Optional[str] = None,
    time_range: str = "week",
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """Search web for news. Results cached 5 min; set force_refresh=True for fresh data.

    Uses Tavily (advanced) with DuckDuckGo fallback. Results auto-persisted
    to pgvector for cross-session recall.

    Args:
        query: Search query (be specific: "Fed rate decision March 2026")
        event_ticker: Optional event ticker for budget tracking
        time_range: "day", "week", "month" (default "week")
        force_refresh: Bypass cache and fetch fresh results (default False)

    Returns:
        NewsSearchResult with articles, count, and cached flag
    """
    if not _ctx or not _ctx.search:
        return {"error": "Search service not available"}

    # Check cache
    cache_key = f"{query.lower().strip()}|{event_ticker or ''}|{time_range}"
    if not force_refresh and cache_key in _ctx.news_cache:
        cached_ts, cached_result = _ctx.news_cache[cache_key]
        if time.time() - cached_ts < NEWS_CACHE_TTL:
            cached_result["cached"] = True
            return cached_result

    try:
        raw_results = await _ctx.search.search_news(
            query=query,
            time_range=time_range,
            event_ticker=event_ticker or "",
        )

        articles = [
            NewsArticle(
                title=r.get("title", ""),
                url=r.get("url", ""),
                content=r.get("content", "")[:500],
                published_date=r.get("published_date", ""),
                score=r.get("score", 0.0),
                source=r.get("source", ""),
            )
            for r in raw_results[:5]
        ]

        # Auto-store headlines in memory for future recall
        stored = False
        if articles:
            headlines = "; ".join(a.title for a in articles[:3] if a.title)

            # Build enriched price snapshot for news-price impact tracking
            price_snapshot = None
            if event_ticker and _ctx.index:
                event = _ctx.index.events.get(event_ticker)
                if event:
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
                        price_snapshot = snap

            asyncio.create_task(_ctx.memory.store(
                content=f"NEWS [{query}]: {headlines}",
                memory_type="news",
                metadata={
                    "query": query,
                    "event_ticker": event_ticker,
                    "article_count": len(articles),
                    "news_url": articles[0].url if articles else None,
                    "news_title": articles[0].title if articles else None,
                    "price_snapshot": price_snapshot,
                },
            ))
            stored = True

        result = NewsSearchResult(
            query=query,
            articles=articles,
            count=len(articles),
            stored_in_memory=stored,
            cached=False,
        ).model_dump()

        # Store in cache
        _ctx.news_cache[cache_key] = (time.time(), result)

        return result

    except Exception as e:
        return {"error": f"News search failed: {e}"}


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
    result = await _ctx.memory.recall(query=query, limit=limit, memory_types=memory_types)
    return result.model_dump()


# --- Tool 9: store_insight ---

@tool
async def store_insight(
    content: str,
    memory_type: str = "learning",
) -> Dict[str, Any]:
    """Store a learning or insight to session + persistent memory.

    Stored in both FAISS (session, instant recall) and pgvector (persistent,
    survives restarts). Use for lessons, observations, or rules.

    Args:
        content: The insight to store (e.g., "IF VPIN > 0.85 THEN reduce position size")
        memory_type: Category: learning, observation, rule, mistake (default "learning")

    Returns:
        Confirmation dict
    """
    if not _ctx:
        return {"error": "ToolContext not available"}
    await _ctx.memory.store(content=content, memory_type=memory_type)
    return {"status": "stored", "memory_type": memory_type, "length": len(content)}


# --- Tool 10: configure_sniper ---

@tool
async def configure_sniper(settings: Dict[str, Any]) -> Dict[str, Any]:
    """Configure the automated sniper execution layer.

    The sniper auto-executes S1_ARB opportunities on the hot path.
    You CONFIGURE it, it EXECUTES autonomously.

    Args:
        settings: Dict of config parameters. Common keys:
            enabled: bool - Turn sniper on/off
            arb_min_edge: float - Minimum edge in cents to trigger (default 2.0)
            max_capital: int - Max capital in cents (default 5000 = $50)
            max_position: int - Max contracts per leg (default 10)
            cooldown: float - Seconds between trades on same event (default 30)
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

    Zero-I/O call (reads cached state from background health service).
    Use when you need to check account hygiene, review settlements,
    or diagnose balance issues.

    Returns:
        AccountHealthStatus with balance, drawdown, settlements, alerts, and status
    """
    if not _ctx or not _ctx.health_service:
        return {"error": "AccountHealthService not available"}
    status = _ctx.health_service.get_health_status()
    return status.model_dump()


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
    configure_sniper,
    get_account_health,
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
    "configure_sniper": "sniper",
    "get_account_health": "system",
    "write_todos": "todo",
}
