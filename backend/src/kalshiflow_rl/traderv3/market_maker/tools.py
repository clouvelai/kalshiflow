"""Admiral tools - 12 tools for the market making LLM agent.

Single MMToolContext dataclass holds all dependencies.
All tools return Pydantic models (serialized via .model_dump()).

Tools:
  Observation: get_mm_state, get_inventory, get_quote_performance, get_resting_orders
  Execution: configure_quotes, set_market_override, pull_quotes, resume_quotes
  Intelligence: search_news, get_market_movers, recall_memory, store_insight
"""

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional, TYPE_CHECKING

from langchain_core.tools import tool

from .models import (
    ConfigureQuotesResult,
    ImpactPattern,
    InventoryResult,
    MMStateResult,
    NewsArticle,
    PullQuotesResult,
    QuotePerformanceResult,
    SwingNewsResult,
)
from ..single_arb.models import RestingOrder

if TYPE_CHECKING:
    from .index import MMIndex
    from .quote_engine import QuoteEngine
    from ..single_arb.event_understanding import UnderstandingBuilder
    from ..single_arb.memory.session_store import SessionMemoryStore
    from ..single_arb.tavily_service import TavilySearchService
    from ..agent_tools.session import TradingSession
    from ..gateway.client import KalshiGateway

logger = logging.getLogger("kalshiflow_rl.traderv3.market_maker.tools")

NEWS_CACHE_TTL = 300  # 5 minutes


async def retry_api(coro_factory, max_retries: int = 3, label: str = "api_call"):
    """Retry an async API call with exponential backoff + jitter."""
    last_err = None
    for attempt in range(max_retries):
        try:
            return await coro_factory()
        except Exception as e:
            last_err = e
            err_str = str(e).lower()
            if any(code in err_str for code in ("400", "401", "403", "404", "422")):
                raise
            if attempt < max_retries - 1:
                delay = (2 ** attempt) * 0.5 + random.uniform(0, 0.5)
                logger.warning(f"[RETRY] {label} attempt {attempt + 1} failed: {e}, retrying in {delay:.1f}s")
                await asyncio.sleep(delay)
    raise last_err


@dataclass
class MMToolContext:
    """Single dependency container for all Admiral tools."""
    gateway: "KalshiGateway"
    index: "MMIndex"
    quote_engine: "QuoteEngine"
    memory: "SessionMemoryStore"
    search: Optional["TavilySearchService"]
    session: "TradingSession"
    broadcast: Optional[Callable[..., Coroutine]] = None
    cycle_mode: Optional[str] = None
    news_cache: Dict[str, tuple] = field(default_factory=dict)
    understanding_builder: Optional["UnderstandingBuilder"] = None


# Module-level context
_ctx: Optional[MMToolContext] = None


def set_context(ctx: MMToolContext) -> None:
    global _ctx
    _ctx = ctx


def get_context() -> Optional[MMToolContext]:
    return _ctx


# ------------------------------------------------------------------
# Observation Tools
# ------------------------------------------------------------------

@tool
async def get_mm_state() -> Dict[str, Any]:
    """Get full market maker state for all active events.

    Returns market data, our quotes, inventory, and fair values for every market.
    Only call in DEEP_SCAN or when you need full detail beyond the briefing.
    """
    if not _ctx:
        return {"error": "Tools not initialized"}

    events = []
    for et in _ctx.index.events:
        snap = _ctx.index.get_event_snapshot(et)
        if snap:
            events.append(snap.model_dump())

    config = _ctx.quote_engine.config
    state = _ctx.quote_engine.state

    return MMStateResult(
        events=[_ctx.index.get_event_snapshot(et) for et in _ctx.index.events if _ctx.index.get_event_snapshot(et)],
        quote_config={
            "enabled": config.enabled,
            "base_spread_cents": config.base_spread_cents,
            "quote_size": config.quote_size,
            "skew_factor": config.skew_factor,
            "max_position": config.max_position,
            "max_event_exposure": config.max_event_exposure,
            "refresh_interval": config.refresh_interval,
            "pull_quotes_threshold": config.pull_quotes_threshold,
        },
        quote_state=state.to_dict(),
    ).model_dump()


@tool
async def get_inventory() -> Dict[str, Any]:
    """Get positions, P&L, and exposure across all markets.

    Returns per-market inventory plus aggregate stats.
    Your briefing includes inventory data — only call mid-cycle if needed.
    """
    if not _ctx:
        return {"error": "Tools not initialized"}

    markets = []
    for et, event in _ctx.index.events.items():
        for ticker in event.markets:
            inv = _ctx.index.get_inventory(ticker)
            market = event.markets[ticker]
            markets.append({
                "ticker": ticker,
                "event_ticker": et,
                "title": market.title,
                "position": inv.position,
                "avg_entry_cents": round(inv.avg_entry_cents, 2),
                "realized_pnl_cents": round(inv.realized_pnl_cents, 2),
                "unrealized_pnl_cents": round(inv.unrealized_pnl_cents, 2),
                "total_buys": inv.total_buys,
                "total_sells": inv.total_sells,
            })

    # Get balance
    try:
        balance = await retry_api(
            lambda: _ctx.gateway.get_balance(), label="get_balance"
        )
        balance_cents = balance.balance if hasattr(balance, 'balance') else 0
    except Exception:
        balance_cents = 0

    total_pos = _ctx.index.total_position_contracts()
    total_realized = _ctx.index.total_realized_pnl()
    total_unrealized = _ctx.index.total_unrealized_pnl()

    return InventoryResult(
        markets=markets,
        total_position_contracts=total_pos,
        total_realized_pnl_cents=round(total_realized, 2),
        total_unrealized_pnl_cents=round(total_unrealized, 2),
        total_fees_paid_cents=round(_ctx.quote_engine.state.fees_paid_cents, 2),
        balance_cents=balance_cents,
        balance_dollars=round(balance_cents / 100, 2),
        event_exposure=total_pos,
        max_event_exposure=_ctx.quote_engine.config.max_event_exposure,
    ).model_dump()


@tool
async def get_quote_performance() -> Dict[str, Any]:
    """Get quote engine performance metrics.

    Returns fill rates, spread capture, adverse selection, fees, and net P&L.
    """
    if not _ctx:
        return {"error": "Tools not initialized"}

    state = _ctx.quote_engine.state
    total_fills = state.total_fills_bid + state.total_fills_ask
    net_pnl = state.spread_captured_cents - state.adverse_selection_cents - state.fees_paid_cents

    return QuotePerformanceResult(
        total_fills_bid=state.total_fills_bid,
        total_fills_ask=state.total_fills_ask,
        total_requote_cycles=state.total_requote_cycles,
        spread_captured_cents=round(state.spread_captured_cents, 2),
        adverse_selection_cents=round(state.adverse_selection_cents, 2),
        fees_paid_cents=round(state.fees_paid_cents, 2),
        net_pnl_cents=round(net_pnl, 2),
        quote_uptime_pct=round(state.quote_uptime_pct, 1),
        fill_rate_pct=0.0,  # TODO: compute from quote exposure time
        avg_spread_captured=round(state.spread_captured_cents / total_fills, 2) if total_fills > 0 else 0.0,
        spread_multiplier=state.spread_multiplier,
        fill_storm_active=state.fill_storm_active,
    ).model_dump()


# ------------------------------------------------------------------
# Execution Tools
# ------------------------------------------------------------------

@tool
async def configure_quotes(settings: Dict[str, Any]) -> Dict[str, Any]:
    """Configure the quote engine parameters.

    Adjustable settings:
      base_spread_cents (int): Minimum spread width in cents
      quote_size (int): Contracts per side per market
      skew_factor (float): Inventory skew multiplier (0-2)
      skew_cap_cents (float): Max skew offset
      max_position (int): Max contracts per market per side
      max_event_exposure (int): Max total contracts across event
      refresh_interval (float): Seconds between requote cycles
      pull_quotes_threshold (float): VPIN threshold to pull quotes
      enabled (bool): Enable/disable quoting

    Args:
        settings: Dict of key-value pairs to update
    """
    if not _ctx:
        return {"error": "Tools not initialized"}

    config = _ctx.quote_engine.config
    changes = []

    allowed_keys = {
        "base_spread_cents", "quote_size", "skew_factor", "skew_cap_cents",
        "max_position", "max_event_exposure", "refresh_interval",
        "pull_quotes_threshold", "enabled", "cancel_on_fill",
        "fill_storm_threshold", "fill_storm_window",
    }

    for key, value in settings.items():
        if key not in allowed_keys:
            continue
        old = getattr(config, key, None)
        if old != value:
            setattr(config, key, value)
            changes.append(f"{key}: {old} → {value}")

    if changes:
        logger.info(f"[ADMIRAL:CONFIGURE] {', '.join(changes)}")

    return ConfigureQuotesResult(
        status="updated" if changes else "no_changes",
        config={k: getattr(config, k) for k in allowed_keys if hasattr(config, k)},
        changes=changes,
    ).model_dump()


@tool
async def set_market_override(ticker: str, settings: Dict[str, Any]) -> Dict[str, Any]:
    """Set per-market configuration overrides.

    Overrides apply to a specific market ticker, falling back to base config
    for unset keys.

    Args:
        ticker: Market ticker to override
        settings: Dict of key-value pairs (same keys as configure_quotes)
    """
    if not _ctx:
        return {"error": "Tools not initialized"}

    config = _ctx.quote_engine.config
    if ticker not in config.market_overrides:
        config.market_overrides[ticker] = {}

    changes = []
    for key, value in settings.items():
        old = config.market_overrides[ticker].get(key)
        config.market_overrides[ticker][key] = value
        changes.append(f"{ticker}.{key}: {old} → {value}")

    logger.info(f"[ADMIRAL:OVERRIDE] {', '.join(changes)}")

    return ConfigureQuotesResult(
        status="updated",
        config=config.market_overrides.get(ticker, {}),
        changes=changes,
    ).model_dump()


@tool
async def pull_quotes(reason: str = "manual") -> Dict[str, Any]:
    """Emergency: cancel ALL resting quotes immediately.

    Use when:
      - VPIN spike indicates toxic flow
      - Unexpected news/event
      - Need to reassess fair values
      - Any time you want to pause market making

    Args:
        reason: Why quotes are being pulled (logged)
    """
    if not _ctx:
        return {"error": "Tools not initialized"}

    cancelled = await _ctx.quote_engine.pull_all_quotes(reason)

    return PullQuotesResult(
        status="pulled",
        cancelled_orders=cancelled,
        reason=reason,
    ).model_dump()


@tool
async def resume_quotes(reason: str = "") -> Dict[str, Any]:
    """Resume quoting after a pull.

    Quotes will resume on the next refresh cycle.

    Args:
        reason: Why quotes are being resumed (optional, logged)
    """
    if not _ctx:
        return {"error": "Tools not initialized"}

    _ctx.quote_engine.resume_quotes()

    return PullQuotesResult(
        status="resumed",
        cancelled_orders=0,
        reason=reason or "manual_resume",
    ).model_dump()


# ------------------------------------------------------------------
# Intelligence Helpers (ported from single_arb/tools.py)
# ------------------------------------------------------------------

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
                "open_interest": getattr(mm, "open_interest", None),
            }
    if snap:
        snap["_ts"] = time.time()
        return snap
    return None


async def _enrich_with_patterns(articles: List[NewsArticle]) -> int:
    """Enrich articles with similar_patterns from the swing-news index.

    For each article, embed the title+content and search for historically
    similar news that moved prices. Returns count of patterns found.
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
        if loop.time() >= deadline:
            logger.info(f"[TOOLS:PATTERNS] Budget exhausted after enriching {total_patterns} patterns")
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


# ------------------------------------------------------------------
# Intelligence Tools
# ------------------------------------------------------------------

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
    similar AND moved prices. Use these predictions to inform fair value and spread.

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
        return {"error": "Tools not initialized"}

    effective_depth = _resolve_depth(depth)

    # Check cache (skip for ultra_fast since it's already instant)
    cache_key = f"{query.lower().strip()}|{event_ticker or ''}|{time_range}|{effective_depth}"
    if not force_refresh and effective_depth != "ultra_fast" and cache_key in _ctx.news_cache:
        cached_ts, cached_result = _ctx.news_cache[cache_key]
        if time.time() - cached_ts < NEWS_CACHE_TTL:
            cached_result["cached"] = True
            return cached_result

    articles: List[NewsArticle] = []
    tavily_answer = ""

    # --- TIER 1: ultra_fast (memory only) ---
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
                continue
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

    # Store in cache
    if effective_depth != "ultra_fast":
        _ctx.news_cache[cache_key] = (time.time(), result)

    return result


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
        return {"error": "Tools not initialized"}

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


@tool
async def get_resting_orders(ticker: Optional[str] = None) -> Dict[str, Any]:
    """Get currently open/resting orders with queue position info.

    Args:
        ticker: Filter to specific market ticker (optional)

    Returns:
        List of RestingOrder with queue positions
    """
    if not _ctx:
        return {"error": "Tools not initialized"}
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


@tool
async def recall_memory(query: str, limit: int = 5) -> Dict[str, Any]:
    """Search session + persistent memory for relevant context.

    Args:
        query: Natural language query
        limit: Max results to return
    """
    if not _ctx:
        return {"error": "Tools not initialized"}

    try:
        result = await _ctx.memory.recall(query, limit=limit)
        return {
            "query": query,
            "results": [
                {
                    "content": r.content,
                    "memory_type": r.memory_type,
                    "similarity": round(r.similarity, 3),
                }
                for r in result.results
            ],
            "count": result.count,
        }
    except Exception as e:
        return {"error": f"Memory recall failed: {e}", "query": query}


@tool
async def store_insight(
    content: str,
    memory_type: str = "learning",
    tags: str = "",
) -> Dict[str, Any]:
    """Persist a market making insight to memory.

    Use for:
      - Spread optimization learnings
      - Market microstructure patterns
      - Adverse selection observations
      - Event-specific fair value notes

    Args:
        content: The insight text
        memory_type: Type tag (learning, observation, strategy, error)
        tags: Comma-separated tags
    """
    if not _ctx:
        return {"error": "Tools not initialized"}

    try:
        tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
        tag_list.append("market_maker")

        await _ctx.memory.store(
            content=content,
            memory_type=memory_type,
            tags=tag_list,
        )
        return {
            "status": "stored",
            "content_preview": content[:100],
            "memory_type": memory_type,
            "tags": tag_list,
        }
    except Exception as e:
        return {"error": f"Store failed: {e}"}


# ------------------------------------------------------------------
# Tool List
# ------------------------------------------------------------------

def get_mm_tools() -> List:
    """Return all Admiral tools."""
    return [
        get_mm_state,
        get_inventory,
        get_quote_performance,
        get_resting_orders,
        configure_quotes,
        set_market_override,
        pull_quotes,
        resume_quotes,
        search_news,
        get_market_movers,
        recall_memory,
        store_insight,
    ]
