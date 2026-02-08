"""
LangChain tools for the ArbCaptain.

Module-level globals are injected at startup by the coordinator.
"""

import hashlib
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
_broadcast_callback = None  # Async callback to broadcast events to frontend
_captain_order_ids = set()  # Track order IDs placed by Captain this session
_order_initial_states: Dict[str, dict] = {}  # Track initial order state for lifecycle tracking
_understanding_builder = None  # UnderstandingBuilder for event context
_search_service = None  # TavilySearchService for web search
_news_store = None      # NewsStore for persisting news articles
_impact_tracker = None  # PriceImpactTracker for news-price correlation
_causal_model_builder = None  # CausalModelBuilder for structured drivers
_sniper = None          # Sniper execution layer
_sniper_config = None   # SniperConfig (mutable by Captain)


def set_dependencies(
    index=None,
    trading_client=None,
    file_store=None,
    memory_store=None,
    config=None,
    order_group_id=None,
    order_ttl=None,
    broadcast_callback=None,
    reset_session=False,
    understanding_builder=None,
    search_service=None,
    news_store=None,
    impact_tracker=None,
    causal_model_builder=None,
    sniper=None,
) -> None:
    """Set shared dependencies for all tools.

    Args:
        file_store: Deprecated alias for memory_store (backwards compat)
        memory_store: DualMemoryStore instance (preferred)
        reset_session: If True, clears captain_order_ids (call on new session start)
    """
    global _index, _trading_client, _memory_store, _config, _order_group_id, _order_ttl, _broadcast_callback, _captain_order_ids, _understanding_builder, _search_service, _news_store, _impact_tracker, _causal_model_builder, _sniper, _sniper_config
    if index is not None:
        _index = index
    if trading_client is not None:
        _trading_client = trading_client
    if memory_store is not None:
        _memory_store = memory_store
    elif file_store is not None:
        _memory_store = file_store
    if config is not None:
        _config = config
    if order_group_id is not None:
        _order_group_id = order_group_id
        # New order group = new session, clear tracked orders
        _captain_order_ids.clear()
        _order_initial_states.clear()
    if order_ttl is not None:
        _order_ttl = order_ttl
    if broadcast_callback is not None:
        _broadcast_callback = broadcast_callback
    if understanding_builder is not None:
        _understanding_builder = understanding_builder
    if search_service is not None:
        _search_service = search_service
    if news_store is not None:
        _news_store = news_store
    if impact_tracker is not None:
        _impact_tracker = impact_tracker
    if causal_model_builder is not None:
        _causal_model_builder = causal_model_builder
    if sniper is not None:
        _sniper = sniper
        _sniper_config = sniper.config
    if reset_session:
        _captain_order_ids.clear()
        _order_initial_states.clear()


# --- Helper functions ---

def _compute_entity_fingerprint(trades: List[Dict]) -> Dict[str, Any]:
    """Generate a fingerprint from a cluster of similar trades.

    Returns fingerprint ID and pattern details.
    """
    if not trades:
        return {"fingerprint": "UNKNOWN", "pattern": {}}

    # Size signature: modal size bucket
    sizes = [t.get("count", 1) for t in trades]
    avg_size = sum(sizes) / len(sizes) if sizes else 0
    size_sig = "S" if avg_size < 20 else "M" if avg_size < 100 else "L"

    # Timing signature: average inter-trade delta
    deltas = [t.get("delta_ms", 0) for t in trades if t.get("delta_ms", 0) > 0]
    avg_delta = sum(deltas) / len(deltas) if deltas else 0
    timing_sig = "F" if avg_delta < 500 else "N" if avg_delta < 5000 else "S"

    # Hash from pattern
    pattern = f"{size_sig}_{timing_sig}_{avg_size:.0f}_{avg_delta:.0f}"
    fingerprint = hashlib.md5(pattern.encode()).hexdigest()[:6]

    return {
        "fingerprint": f"ENT_{fingerprint.upper()}",
        "pattern": {
            "size_sig": size_sig,
            "timing_sig": timing_sig,
            "avg_size": round(avg_size, 1),
            "avg_delta_ms": round(avg_delta, 1),
        },
    }


def _cluster_trades_by_pattern(trades: List[Dict]) -> List[List[Dict]]:
    """Group trades by similar characteristics (size, timing, side).

    Returns list of trade clusters, each cluster being a potential entity.
    """
    if not trades:
        return []

    # Simple clustering: group by size bucket and side tendency
    clusters = {}
    for t in trades:
        count = t.get("count", 1)
        side = t.get("side", "unknown")

        # Size bucket
        if count <= 10:
            size_key = "tiny"
        elif count <= 50:
            size_key = "small"
        elif count <= 100:
            size_key = "medium"
        else:
            size_key = "large"

        key = f"{size_key}_{side}"
        if key not in clusters:
            clusters[key] = []
        clusters[key].append(t)

    # Filter to clusters with >= 3 trades (enough to identify a pattern)
    return [c for c in clusters.values() if len(c) >= 3]


def _guess_entity_type(trades: List[Dict]) -> str:
    """Guess entity type based on behavior patterns."""
    if not trades:
        return "UNKNOWN"

    # Check for market maker patterns (consistent sizes, both sides)
    sides = set(t.get("side") for t in trades)
    sizes = [t.get("count", 1) for t in trades]
    avg_size = sum(sizes) / len(sizes) if sizes else 0
    size_variance = sum((s - avg_size) ** 2 for s in sizes) / len(sizes) if sizes else 0

    # Low size variance = consistent sizing = likely automated
    is_consistent_size = size_variance < (avg_size * 0.3) ** 2

    # Both sides = market maker
    if len(sides) > 1 and is_consistent_size:
        return "MM_BOT"

    # Consistent large sizes = whale or arb bot
    if avg_size >= 50 and is_consistent_size:
        return "ARB_BOT"

    # Large inconsistent = whale
    if avg_size >= 100:
        return "WHALE"

    # Consistent small sizes = automated
    if is_consistent_size:
        return "UNKNOWN_AUTO"

    return "RETAIL"


# --- Tools ---

@tool
async def get_event_snapshot(event_ticker: str, _ts: Optional[str] = None) -> Dict[str, Any]:
    """Get current arb state for a specific event. ALWAYS returns real-time data.

    Returns all markets with full orderbook depth, probability sums,
    edge calculations, signal operators, and recent trade activity.

    Args:
        event_ticker: The Kalshi event ticker (e.g., "KXFEDCHAIRNOM")
        _ts: Cache-buster. Pass current timestamp to ensure fresh data.

    Returns:
        Dict with markets (incl. depth, trades, volume), prob sums, edges, and signals
    """
    # _ts parameter exists to bust cache - unique input = cache miss
    _ = _ts
    if not _index:
        return {"error": "EventArbIndex not available"}

    snapshot = _index.get_event_snapshot(event_ticker)
    if not snapshot:
        return {"error": f"Event {event_ticker} not found in index"}

    # Trim noisy fields from each market to reduce token usage
    _market_keep_fields = {
        "ticker", "title", "yes_bid", "yes_ask", "yes_bid_size", "yes_ask_size",
        "yes_levels", "no_levels", "spread", "freshness_seconds",
        "last_trade_price", "last_trade_side", "trade_count", "micro",
    }
    if "markets" in snapshot:
        trimmed = {}
        for ticker, mkt in snapshot["markets"].items():
            trimmed[ticker] = {k: v for k, v in mkt.items() if k in _market_keep_fields}
        snapshot["markets"] = trimmed

    # Also remove event-level noise
    for key in ("subtitle", "loaded_at"):
        snapshot.pop(key, None)

    # Add candlestick summary (compact token-efficient version)
    event = _index.events.get(event_ticker)
    if event and event.candlesticks:
        snapshot["candlestick_summary"] = event.candlestick_summary()

    # Add mentions budget info if available
    if event and event.mentions_data and event.mentions_data.get("lexeme_pack"):
        from .mentions_tools import _budget_manager
        if _budget_manager:
            budget_status = _budget_manager.get_budget_status(event_ticker)
            if budget_status:
                snapshot["mentions_budget"] = {
                    "phase": budget_status["phase"],
                    "simulations_used": budget_status["total_simulations"],
                    "simulations_remaining": budget_status["simulations_remaining"],
                    "cost_used": budget_status["total_estimated_cost"],
                    "budget_pct_used": budget_status["budget_pct_used"],
                    "next_scheduled_ts": budget_status["next_scheduled_ts"],
                }

    # Add lifecycle classification if available
    if event and event.lifecycle:
        snapshot["lifecycle"] = event.lifecycle

    # Add full causal model if available
    if event and event.causal_model:
        snapshot["causal_model"] = event.causal_model

    return snapshot


@tool
async def get_events_summary() -> List[Dict[str, Any]]:
    """Get a compact summary of all monitored events for scanning.

    Returns a lightweight list with edge calculations, data coverage,
    event structure (mutually_exclusive, market_regime), and MENTIONS DATA per event.
    Use this for initial scanning, then drill into specific events with get_event_snapshot().

    KEY FIELDS for strategy selection:
    - mutually_exclusive: True = S1 (sum arb) may apply. False = independent outcomes.
    - market_regime: "pre_event" (>24h), "live" (1-24h), "settling" (<1h) — guides strategy.
    - time_to_close_hours: Hours until earliest market closes.

    Returns:
        List of dicts with event_ticker, title, market_count, edge info, structure, mentions data
    """
    if not _index:
        return {"error": "EventArbIndex not available"}

    fee = _index._fee_per_contract
    now = time.time()
    summary = []
    for et, event in _index.events.items():
        sum_bid = event.market_sum_bid()
        sum_ask = event.market_sum_ask()
        long_e = event.long_edge(fee)
        short_e = event.short_edge(fee)

        # Microstructure summary
        total_vol5m = sum(m.micro.volume_5m for m in event.markets.values())
        total_whale = sum(m.micro.whale_trade_count for m in event.markets.values())

        # Compute time to close from earliest market close_time
        time_to_close_hours = None
        market_regime = "unknown"
        for m in event.markets.values():
            if m.close_time:
                try:
                    from datetime import datetime, timezone
                    # Parse ISO timestamp (handles both Z and +00:00 formats)
                    ct = m.close_time.replace("Z", "+00:00")
                    close_dt = datetime.fromisoformat(ct)
                    hours = (close_dt.timestamp() - now) / 3600
                    if time_to_close_hours is None or hours < time_to_close_hours:
                        time_to_close_hours = hours
                except (ValueError, TypeError):
                    pass

        if time_to_close_hours is not None:
            time_to_close_hours = round(time_to_close_hours, 1)
            if time_to_close_hours > 24:
                market_regime = "pre_event"
            elif time_to_close_hours > 1:
                market_regime = "live"
            else:
                market_regime = "settling"

        summary_item = {
            "event_ticker": et,
            "title": event.title,
            "market_count": event.markets_total,
            "markets_with_data": event.markets_with_data,
            "all_markets_have_data": event.all_markets_have_data,
            "mutually_exclusive": event.mutually_exclusive,
            "time_to_close_hours": time_to_close_hours,
            "market_regime": market_regime,
            "sum_yes_bid": sum_bid,
            "sum_yes_ask": sum_ask,
            "long_edge": round(long_e, 1) if long_e is not None else None,
            "short_edge": round(short_e, 1) if short_e is not None else None,
            "volume_5m": total_vol5m,
            "whale_trades": total_whale,
        }

        # Add mentions data if this is a mentions market
        mentions_data = event.mentions_data
        if mentions_data and mentions_data.get("lexeme_pack"):
            lexeme_pack = mentions_data["lexeme_pack"]
            entity = lexeme_pack.get("entity", "")

            summary_item["is_mentions_market"] = True
            summary_item["mentions_entity"] = entity

            # Get baseline and current probability for the primary entity
            baseline = mentions_data.get("baseline_estimates", {})
            current = mentions_data.get("current_estimates", {})

            if entity and entity in baseline:
                summary_item["baseline_probability"] = round(
                    baseline[entity].get("probability", 0.0), 3
                )
            else:
                summary_item["baseline_probability"] = None

            if entity and entity in current:
                summary_item["current_probability"] = round(
                    current[entity].get("probability", 0.0), 3
                )
            else:
                summary_item["current_probability"] = None

            # Freshness check (5 min = 300s)
            last_sim = mentions_data.get("last_simulation_ts", 0)
            summary_item["simulation_stale"] = (time.time() - last_sim) > 300
            summary_item["has_baseline"] = bool(baseline)

            # Budget info
            from .mentions_tools import _budget_manager
            if _budget_manager:
                budget_status = _budget_manager.get_budget_status(et)
                if budget_status:
                    summary_item["simulation_phase"] = budget_status["phase"]
                    summary_item["budget_remaining"] = budget_status["budget_remaining"]
                    # CI width for confidence indicator
                    if entity and entity in baseline:
                        ci = baseline[entity].get("confidence_interval", [0, 1])
                        if len(ci) == 2:
                            summary_item["ci_width"] = round(ci[1] - ci[0], 3)

        # Add news context from understanding
        if event.understanding:
            u = event.understanding
            news = u.get("news_articles", [])
            if news:
                summary_item["news_headlines"] = [a.get("title", "")[:80] for a in news[:3]]
                news_fetched_at = u.get("news_fetched_at", 0)
                if news_fetched_at:
                    summary_item["news_freshness_hours"] = round(
                        (time.time() - news_fetched_at) / 3600, 1
                    )
            if u.get("trading_summary"):
                summary_item["trading_summary"] = u["trading_summary"][:150]

        # Add candlestick trend info if available
        if event.candlesticks:
            cs = event.candlestick_summary()
            if cs:
                trends = {}
                for ticker, info in cs.items():
                    trends[ticker] = {
                        "trend": info.get("price_trend", "flat"),
                        "current": info.get("price_current"),
                        "7d_avg": info.get("price_7d_avg"),
                        "high": info.get("price_high"),
                        "low": info.get("price_low"),
                    }
                summary_item["candlestick_trends"] = trends

        # Add lifecycle stage if classified
        if event.lifecycle:
            summary_item["lifecycle"] = {
                "stage": event.lifecycle.get("stage", "unknown"),
                "confidence": event.lifecycle.get("confidence"),
                "recommended_action": event.lifecycle.get("recommended_action"),
            }

        # Add causal model compact summary (top drivers, next catalyst, consensus)
        if event.causal_model:
            cm = event.causal_model
            active_drivers = [d for d in cm.get("drivers", []) if d.get("status") == "active"]
            top_drivers = active_drivers[:3]
            summary_item["causal"] = {
                "consensus": cm.get("consensus_direction", "unclear"),
                "narrative": cm.get("dominant_narrative", "")[:120],
                "drivers": [
                    {"name": d["name"], "direction": d["direction"], "confidence": d.get("confidence", 0.5)}
                    for d in top_drivers
                ],
            }
            # Next upcoming catalyst
            upcoming = [c for c in cm.get("catalysts", []) if not c.get("occurred")]
            if upcoming:
                next_cat = upcoming[0]
                summary_item["causal"]["next_catalyst"] = {
                    "name": next_cat["name"],
                    "expected_date": next_cat.get("expected_date", ""),
                    "type": next_cat.get("type", "expected"),
                }

        summary.append(summary_item)

    return summary


@tool
async def update_causal_model(
    event_ticker: str,
    action: str,
    driver_name: Optional[str] = None,
    driver_direction: Optional[str] = None,
    driver_confidence: Optional[float] = None,
    driver_evidence: Optional[str] = None,
    market_links: Optional[str] = None,
    catalyst_name: Optional[str] = None,
    catalyst_type: Optional[str] = None,
    catalyst_date: Optional[str] = None,
    affected_markets: Optional[str] = None,
) -> Dict[str, Any]:
    """Maintain the structured causal model for an event.

    The causal model tracks what's DRIVING prices NOW — drivers, catalysts, and entity links.
    Use this tool to incrementally update causal knowledge across cycles.

    Actions:
      - refresh: Rebuild the causal model from scratch (LLM call, use sparingly)
      - add_driver: Add a new price driver (requires driver_name, driver_direction)
      - validate_driver: Confirm a driver is still active (requires driver_name)
      - invalidate_driver: Mark a driver as invalidated (requires driver_name)
      - add_catalyst: Add an upcoming catalyst/event (requires catalyst_name)
      - mark_catalyst: Mark a catalyst as occurred (requires catalyst_name)
      - prune: Remove stale drivers (>2h without validation)
      - view: Return current causal model (no mutations)

    Args:
        event_ticker: The Kalshi event ticker
        action: One of: refresh, add_driver, validate_driver, invalidate_driver, add_catalyst, mark_catalyst, prune, view
        driver_name: Name of driver (for add/validate/invalidate actions)
        driver_direction: Direction: bullish, bearish, neutral, ambiguous (for add_driver)
        driver_confidence: Confidence 0.0-1.0 (for add_driver)
        driver_evidence: Evidence source string (for add_driver)
        market_links: Comma-separated market tickers this driver affects (for add_driver)
        catalyst_name: Name of catalyst (for add_catalyst/mark_catalyst)
        catalyst_type: Type: scheduled, expected, conditional (for add_catalyst)
        catalyst_date: Expected date string (for add_catalyst)
        affected_markets: Comma-separated market tickers (for add_catalyst)

    Returns:
        Updated causal model or action result
    """
    if not _index:
        return {"error": "EventArbIndex not available"}

    event = _index.events.get(event_ticker)
    if not event:
        return {"error": f"Event {event_ticker} not found"}

    valid_actions = {"refresh", "add_driver", "validate_driver", "invalidate_driver",
                     "add_catalyst", "mark_catalyst", "prune", "view"}
    if action not in valid_actions:
        return {"error": f"Invalid action '{action}'. Valid: {valid_actions}"}

    from .causal_model import CausalModel, Driver, MarketLink, Catalyst

    # Load existing model or create empty
    if event.causal_model:
        model = CausalModel.from_dict(event.causal_model)
    else:
        model = CausalModel(event_ticker=event_ticker)

    if action == "view":
        return model.to_dict()

    if action == "refresh":
        if not _causal_model_builder:
            return {"error": "CausalModelBuilder not available"}
        model = await _causal_model_builder.build(event)
        event.causal_model = model.to_dict()
        # Broadcast update
        if _broadcast_callback:
            await _broadcast_callback({
                "type": "causal_model_update",
                "data": {"event_ticker": event_ticker, "causal_model": event.causal_model},
            })
        return {"status": "rebuilt", "drivers": len(model.drivers), "catalysts": len(model.catalysts)}

    if action == "add_driver":
        if not driver_name:
            return {"error": "driver_name required for add_driver"}
        links = []
        if market_links:
            for mt in market_links.split(","):
                mt = mt.strip()
                if mt:
                    links.append(MarketLink(market_ticker=mt, direction=driver_direction or "neutral"))
        driver = model.add_driver(
            name=driver_name,
            direction=driver_direction or "neutral",
            confidence=driver_confidence or 0.5,
            market_links=[{"market_ticker": ml.market_ticker, "direction": ml.direction} for ml in links],
            evidence=driver_evidence or "",
        )
        event.causal_model = model.to_dict()
        if _broadcast_callback:
            await _broadcast_callback({
                "type": "causal_model_update",
                "data": {"event_ticker": event_ticker, "causal_model": event.causal_model},
            })
        return {"status": "driver_added", "driver_id": driver.id, "total_drivers": len(model.drivers)}

    if action == "validate_driver":
        if not driver_name:
            return {"error": "driver_name required for validate_driver"}
        # Find driver by name (case-insensitive)
        found = False
        for d in model.drivers:
            if d.name.lower() == driver_name.lower():
                model.validate_driver(d.id)
                found = True
                break
        if not found:
            return {"error": f"Driver '{driver_name}' not found"}
        event.causal_model = model.to_dict()
        return {"status": "driver_validated", "driver_name": driver_name}

    if action == "invalidate_driver":
        if not driver_name:
            return {"error": "driver_name required for invalidate_driver"}
        found = False
        for d in model.drivers:
            if d.name.lower() == driver_name.lower():
                model.invalidate_driver(d.id)
                found = True
                break
        if not found:
            return {"error": f"Driver '{driver_name}' not found"}
        event.causal_model = model.to_dict()
        if _broadcast_callback:
            await _broadcast_callback({
                "type": "causal_model_update",
                "data": {"event_ticker": event_ticker, "causal_model": event.causal_model},
            })
        return {"status": "driver_invalidated", "driver_name": driver_name}

    if action == "add_catalyst":
        if not catalyst_name:
            return {"error": "catalyst_name required for add_catalyst"}
        affected = []
        if affected_markets:
            affected = [m.strip() for m in affected_markets.split(",") if m.strip()]
        catalyst = Catalyst(
            name=catalyst_name,
            type=catalyst_type or "expected",
            expected_date=catalyst_date or "",
            affected_markets=affected,
        )
        model.add_catalyst(catalyst)
        event.causal_model = model.to_dict()
        if _broadcast_callback:
            await _broadcast_callback({
                "type": "causal_model_update",
                "data": {"event_ticker": event_ticker, "causal_model": event.causal_model},
            })
        return {"status": "catalyst_added", "catalyst_id": catalyst.id, "total_catalysts": len(model.catalysts)}

    if action == "mark_catalyst":
        if not catalyst_name:
            return {"error": "catalyst_name required for mark_catalyst"}
        found = False
        for c in model.catalysts:
            if c.name.lower() == catalyst_name.lower():
                model.mark_catalyst_occurred(c.id)
                found = True
                break
        if not found:
            return {"error": f"Catalyst '{catalyst_name}' not found"}
        event.causal_model = model.to_dict()
        return {"status": "catalyst_marked_occurred", "catalyst_name": catalyst_name}

    if action == "prune":
        before = len([d for d in model.drivers if d.status == "active"])
        model.prune_stale()
        after = len([d for d in model.drivers if d.status == "active"])
        event.causal_model = model.to_dict()
        if _broadcast_callback:
            await _broadcast_callback({
                "type": "causal_model_update",
                "data": {"event_ticker": event_ticker, "causal_model": event.causal_model},
            })
        return {"status": "pruned", "active_before": before, "active_after": after}

    return {"error": f"Unhandled action: {action}"}


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
        # Microstructure signals
        "microprice": round(market.microprice, 2) if market.microprice is not None else None,
        "ofi": round(market.micro.ofi, 3),
        "ofi_ema": round(market.micro.ofi_ema, 3),
        "vpin": round(market.micro.vpin, 3),
        "book_imbalance": round(market.micro.book_imbalance, 3),
    }


@tool
async def get_event_candlesticks(event_ticker: str, period: str = "6h") -> Dict[str, Any]:
    """Get candlestick OHLC data for all markets in an event.

    Returns 7-day price history at the specified interval.
    The "6h" period is cached (fetched at startup, refreshed every 30 min).
    Other periods ("1h", "1d") trigger a fresh API call.

    Args:
        event_ticker: The Kalshi event ticker
        period: Candle interval - "6h" (cached), "1h", or "1d" (fresh fetch)

    Returns:
        Dict with per-market OHLC candles, volume, and open interest
    """
    if not _index:
        return {"error": "EventArbIndex not available"}

    event = _index.events.get(event_ticker)
    if not event:
        return {"error": f"Event {event_ticker} not found"}

    period_map = {"1h": 60, "6h": 360, "1d": 1440}
    interval = period_map.get(period, 360)

    # Use cached data for 6h
    if period == "6h" and event.candlesticks:
        raw = event.candlesticks
        tickers = raw.get("market_tickers", [])
        candle_lists = raw.get("market_candlesticks", [])
        result = {}
        for i, ticker in enumerate(tickers):
            if i < len(candle_lists) and candle_lists[i]:
                result[ticker] = candle_lists[i][-20:]  # Last 20 candles for token efficiency
        return {
            "event_ticker": event_ticker,
            "period": period,
            "markets": result,
            "fetched_at": event.candlesticks_fetched_at,
            "cached": True,
        }

    # Fresh fetch for other periods
    if not _trading_client:
        return {"error": "Trading client not available for fresh fetch"}

    if not event.series_ticker:
        return {"error": f"No series_ticker for {event_ticker}"}

    try:
        import time as _time
        now = int(_time.time())
        start_ts = now - (7 * 24 * 60 * 60)
        resp = await _trading_client.get_event_candlesticks(
            series_ticker=event.series_ticker,
            event_ticker=event_ticker,
            start_ts=start_ts,
            end_ts=now,
            period_interval=interval,
        )
        tickers = resp.get("market_tickers", [])
        candle_lists = resp.get("market_candlesticks", [])
        result = {}
        for i, ticker in enumerate(tickers):
            if i < len(candle_lists) and candle_lists[i]:
                result[ticker] = candle_lists[i][-20:]
        return {
            "event_ticker": event_ticker,
            "period": period,
            "markets": result,
            "cached": False,
        }
    except Exception as e:
        return {"error": f"Candlestick fetch failed: {e}"}


@tool
async def get_market_candlesticks(market_ticker: str, period: str = "6h") -> Dict[str, Any]:
    """Get candlestick OHLC data for a single market.

    Extracts data from the event-level candlestick cache.

    Args:
        market_ticker: The Kalshi market ticker
        period: Candle interval - "6h" (cached), "1h", or "1d" (fresh fetch)

    Returns:
        Dict with OHLC candles, volume, and open interest for one market
    """
    if not _index:
        return {"error": "EventArbIndex not available"}

    event_ticker = _index.get_event_for_ticker(market_ticker)
    if not event_ticker:
        return {"error": f"Market {market_ticker} not tracked"}

    event = _index.events.get(event_ticker)
    if not event:
        return {"error": f"Event {event_ticker} not found"}

    # Extract from cached event candlesticks
    if period == "6h" and event.candlesticks:
        tickers = event.candlesticks.get("market_tickers", [])
        candle_lists = event.candlesticks.get("market_candlesticks", [])
        for i, t in enumerate(tickers):
            if t == market_ticker and i < len(candle_lists) and candle_lists[i]:
                return {
                    "market_ticker": market_ticker,
                    "event_ticker": event_ticker,
                    "period": period,
                    "candles": candle_lists[i][-20:],
                    "candle_count": len(candle_lists[i]),
                    "cached": True,
                }
        return {"error": f"No candlestick data for {market_ticker}"}

    # Fresh fetch via event-level API
    result = await get_event_candlesticks.ainvoke({
        "event_ticker": event_ticker,
        "period": period,
    })
    if "error" in result:
        return result

    market_data = result.get("markets", {}).get(market_ticker)
    if not market_data:
        return {"error": f"No candlestick data for {market_ticker} in fresh fetch"}

    return {
        "market_ticker": market_ticker,
        "event_ticker": event_ticker,
        "period": period,
        "candles": market_data,
        "cached": False,
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

    # --- POSITION CONFLICT GUARD ---
    # Block buying the opposite side on the same market (e.g., YES while holding NO).
    # Holding both sides simultaneously guarantees losses on at least one side.
    # If you want to hedge, exit the existing position first, then enter the new one.
    if action == "buy":
        try:
            pos_resp = await _trading_client.get_positions()
            all_positions = pos_resp.get("market_positions", pos_resp.get("positions", []))
            for pos in all_positions:
                if pos.get("ticker") != ticker:
                    continue
                position_count = pos.get("position", 0)
                if position_count == 0:
                    continue
                existing_side = "yes" if position_count > 0 else "no"
                if existing_side != side:
                    return {
                        "error": f"POSITION CONFLICT: Already hold {abs(position_count)} {existing_side.upper()} "
                        f"on {ticker}. Exit existing position first with place_order(action='sell', side='{existing_side}', ...). "
                        f"Buying {side.upper()} while holding {existing_side.upper()} on the SAME market locks in losses.",
                    }
        except Exception as e:
            logger.warning(f"[SINGLE_ARB:CONFLICT_CHECK] Position check failed: {e}")

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

        # Track this order as a Captain order
        if order_id:
            _captain_order_ids.add(order_id)
            _order_initial_states[order_id] = {
                "ticker": ticker, "side": side, "action": action,
                "contracts": contracts, "price_cents": price_cents,
                "placed_at": time.time(), "ttl_seconds": _order_ttl,
                "status": status,
            }

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

        # Emit arb_trade_executed event to frontend
        if _broadcast_callback:
            try:
                event_ticker = _index.get_event_for_ticker(ticker) if _index else None
                await _broadcast_callback({
                    "type": "arb_trade_executed",
                    "data": {
                        "order_id": order_id,
                        "event_ticker": event_ticker,
                        "kalshi_ticker": ticker,
                        "direction": None,  # Single-leg trade, no direction
                        "side": side,
                        "action": action,
                        "contracts": contracts,
                        "price_cents": price_cents,
                        "status": status,
                        "reasoning": reasoning,
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    },
                })
            except Exception as e:
                logger.debug(f"Failed to emit arb_trade_executed: {e}")

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

        # Get queue positions (best-effort) - only if we have orders to check
        queue_map = {}
        if raw_orders:
            try:
                # Use first order's ticker if no specific ticker provided
                market_ticker = ticker or raw_orders[0].get("ticker")
                if market_ticker:
                    qp_resp = await _trading_client.get_queue_positions(
                        market_tickers=market_ticker,
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

    Returns the last trades merged and sorted by timestamp (newest first).
    Titles omitted for token efficiency - use get_event_snapshot for market details.

    Args:
        event_ticker: The Kalshi event ticker

    Returns:
        Dict with trades list and most active market ticker
    """
    if not _index:
        return {"error": "EventArbIndex not available"}

    event = _index.events.get(event_ticker)
    if not event:
        return {"error": f"Event {event_ticker} not found"}

    all_trades = []
    total_count = 0
    for m in event.markets.values():
        total_count += m.trade_count
        for t in m.recent_trades[:5]:  # 5 per market max
            all_trades.append({
                "ticker": m.ticker,
                "price": t.get("yes_price"),
                "count": t.get("count", 1),
                "side": t.get("taker_side"),
                "ts": t.get("ts"),
            })

    # Sort by timestamp descending
    all_trades.sort(key=lambda t: t.get("ts", 0), reverse=True)

    most_active = event.most_active_market()
    return {
        "event_ticker": event_ticker,
        "trades": all_trades[:15],  # Reduced from 30
        "trade_count_total": total_count,
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

    # --- INDEPENDENT EVENT WARNING ---
    # On independent events, sum > 100% is NORMAL, not arbitrage
    if not event_state.mutually_exclusive and direction == "short":
        return {
            "error": (
                f"INDEPENDENT EVENT: {event_ticker} has mutually_exclusive=False. "
                f"Markets are independent - each can resolve YES independently. "
                f"Sum of probabilities CAN exceed 100% on independent events. "
                f"Buying NO on all markets is NOT risk-free arbitrage here. "
                f"Use place_order() for directional bets on individual markets instead."
            ),
        }

    # --- POSITION CONFLICT WARNING ---
    # Warn (but don't block) if holding opposite positions on markets in this event.
    # Legitimate hedging exists, but accidental conflicts are the #1 source of losses.
    conflict_warnings = []
    try:
        pos_resp = await _trading_client.get_positions()
        all_positions = pos_resp.get("market_positions", pos_resp.get("positions", []))
        event_tickers = set(event_state.markets.keys())
        for pos in all_positions:
            if pos.get("ticker") not in event_tickers:
                continue
            position_count = pos.get("position", 0)
            if position_count == 0:
                continue
            existing_side = "yes" if position_count > 0 else "no"
            new_side = "yes" if direction == "long" else "no"
            if existing_side != new_side:
                conflict_warnings.append(
                    f"WARNING: Hold {abs(position_count)} {existing_side.upper()} on {pos.get('ticker')}. "
                    f"This {direction} arb will add {new_side.upper()} - creating opposing positions."
                )
    except Exception as e:
        logger.warning(f"[SINGLE_ARB:CONFLICT_CHECK] Position check failed: {e}")

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
        preview = {
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
        if conflict_warnings:
            preview["conflict_warnings"] = conflict_warnings
        return preview

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

            # Track this order as a Captain order
            if order_id:
                _captain_order_ids.add(order_id)
                _order_initial_states[order_id] = {
                    "ticker": leg["ticker"], "side": leg["side"], "action": "buy",
                    "contracts": leg["contracts"], "price_cents": leg["price_cents"],
                    "placed_at": time.time(), "ttl_seconds": _order_ttl,
                    "status": order.get("status", "placed"),
                    "arb_event": event_ticker, "arb_direction": direction,
                }

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
    if conflict_warnings:
        result["conflict_warnings"] = conflict_warnings

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

    # Emit arb_trade_executed event to frontend for each executed leg
    if _broadcast_callback and legs_executed:
        try:
            import asyncio
            for leg in legs_executed:
                await _broadcast_callback({
                    "type": "arb_trade_executed",
                    "data": {
                        "order_id": leg.get("order_id"),
                        "event_ticker": event_ticker,
                        "kalshi_ticker": leg.get("ticker"),
                        "direction": direction,
                        "side": leg.get("side"),
                        "contracts": leg.get("contracts"),
                        "price_cents": leg.get("price_cents"),
                        "status": leg.get("status"),
                        "reasoning": reasoning,
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    },
                })
        except Exception as e:
            logger.debug(f"Failed to emit arb_trade_executed: {e}")

    return result


@tool
async def record_learning(
    content: str,
    category: str = "learning",
    target_file: str = "AGENTS.md",
) -> Dict[str, Any]:
    """Record a learning or insight to memory.

    Appends to journal.jsonl (permanent audit trail) and optionally
    notes which memory file (AGENTS.md, SIGNALS.md, PLAYBOOK.md) should
    be updated. To actually update the file, use edit_file via the
    FilesystemBackend (e.g. edit /memories/AGENTS.md).

    Args:
        content: The learning/insight text to store
        category: Category (learning, mistake, strategy, observation, trade_result)
        target_file: Which memory file this relates to (AGENTS.md, SIGNALS.md, PLAYBOOK.md)

    Returns:
        Dict with status and storage details
    """
    if not _memory_store:
        return {"error": "File store not available"}

    try:
        _memory_store.append(
            content=content,
            memory_type=category,
            metadata={"target_file": target_file},
        )
        return {"status": "stored", "category": category, "target_file": target_file}
    except Exception as e:
        return {"error": str(e)}


@tool
async def get_positions() -> Dict[str, Any]:
    """Get positions for tracked events with realtime P&L.

    Cost from API (total_traded), current value from live orderbook.
    """
    if not _trading_client or not _index:
        return {"error": "Not initialized"}

    resp = await _trading_client.get_positions()
    all_positions = resp.get("market_positions", resp.get("positions", []))
    tracked = set(_index.market_tickers) if _index else set()

    positions = []
    for pos in all_positions:
        ticker = pos.get("ticker")
        if ticker not in tracked:
            continue

        position_count = pos.get("position", 0)
        qty = abs(position_count)
        side = "yes" if position_count > 0 else "no"

        # Cost from API (what we actually paid)
        cost = pos.get("total_traded", 0)
        realized_pnl = pos.get("realized_pnl", 0)
        fees = pos.get("fees_paid", 0)

        # Current value from LIVE orderbook
        current_value = 0
        exit_price = None
        event_ticker = _index.get_event_for_ticker(ticker)

        # Calculate exit price from live orderbook
        if event_ticker and qty > 0:
            event = _index.events.get(event_ticker)
            if event:
                market = event.markets.get(ticker)
                if market:
                    if side == "yes":
                        exit_price = market.yes_bid
                    else:
                        # NO position: exit by selling NO, which is buying YES at yes_ask
                        exit_price = 100 - market.yes_ask if market.yes_ask else None
                    if exit_price:
                        current_value = exit_price * qty

        unrealized_pnl = current_value - cost if cost > 0 else 0

        positions.append({
            "ticker": ticker,
            "event_ticker": event_ticker,
            "side": side,
            "quantity": qty,
            "cost": cost,
            "exit_price": exit_price,
            "current_value": current_value,
            "unrealized_pnl": unrealized_pnl,
            "realized_pnl": realized_pnl,
            "fees_paid": fees,
        })

    positions.sort(key=lambda p: p["unrealized_pnl"], reverse=True)

    return {
        "positions": positions,
        "count": len(positions),
        "total_cost": sum(p["cost"] for p in positions),
        "total_value": sum(p["current_value"] for p in positions),
        "total_unrealized_pnl": sum(p["unrealized_pnl"] for p in positions),
    }


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


@tool
async def report_issue(
    title: str,
    severity: str,
    category: str,
    description: str,
    proposed_fix: str = "",
) -> Dict[str, Any]:
    """Report an issue the Captain observed for future self-fixing.

    Issues are tracked in issues.jsonl and can be auto-fixed by scripts/self-fix.sh.
    Use this when you detect: tool failures, memory contradictions, trade outcomes
    that don't match reasoning, prompt gaps, or config problems.

    Args:
        title: Short description (e.g., "execute_arb returns wrong leg count")
        severity: "critical", "high", "medium", or "low"
        category: "memory_corruption", "bad_trade_logic", "tool_failure",
                  "prompt_gap", "pattern_detection_error", or "config_issue"
        description: Detailed description with evidence
        proposed_fix: What you think should change (optional but helpful)

    Returns:
        Dict with issue id and status
    """
    from .memory.issues import report_issue as _report_issue
    try:
        issue = _report_issue(
            title=title,
            description=description,
            severity=severity,
            category=category,
            proposed_fix=proposed_fix,
            source_agent="captain",
        )
        return {"status": "reported", "issue_id": issue["id"], "severity": severity}
    except Exception as e:
        return {"error": f"Failed to report issue: {e}"}


@tool
async def get_issues() -> Dict[str, Any]:
    """Get summary of open issues and recent resolutions.

    Use this to see what issues have been reported (including by you in prior cycles)
    to avoid reporting duplicates and to track self-improvement progress.

    Returns:
        Dict with summary counts and list of open issues
    """
    from .memory.issues import get_open_issues, get_issues_summary
    try:
        summary = get_issues_summary()
        open_issues = get_open_issues()
        return {
            "summary": summary,
            "open_issues": [
                {
                    "id": i["id"],
                    "title": i["title"],
                    "severity": i["severity"],
                    "category": i["category"],
                    "timestamp": i["timestamp"],
                }
                for i in open_issues[:10]  # Limit for token efficiency
            ],
        }
    except Exception as e:
        return {"error": f"Failed to get issues: {e}"}


@tool
async def analyze_microstructure(event_ticker: str) -> Dict[str, Any]:
    """Analyze market microstructure for automated trading patterns.

    Returns trade flow analysis, size clustering, timing patterns,
    and whale activity for the ChevalDeTroie to interpret.

    Args:
        event_ticker: The event to analyze

    Returns:
        Dict with trade_flow, size_distribution, timing_analysis, whale_trades, anomalies
    """
    if not _index:
        return {"error": "EventArbIndex not available"}

    event = _index.events.get(event_ticker)
    if not event:
        return {"error": f"Event {event_ticker} not found"}

    all_trades = []
    for m in event.markets.values():
        for t in m.recent_trades:
            all_trades.append({**t, "market_ticker": m.ticker})

    if not all_trades:
        return {
            "event_ticker": event_ticker,
            "total_trades": 0,
            "trade_flow": [],
            "size_distribution": {"1-10": 0, "11-50": 0, "51-100": 0, "101-500": 0, "500+": 0},
            "whale_trades": [],
            "whale_count": 0,
            "timing_histogram": {},
            "rapid_sequences": 0,
            "rapid_sequence_samples": [],
            "detected_entities": [],
        }

    # Sort by timestamp
    all_trades.sort(key=lambda t: t.get("ts", 0))

    # Compute inter-trade deltas (ms)
    trade_flow = []
    for i, t in enumerate(all_trades):
        prev_ts = all_trades[i - 1].get("ts", 0) if i > 0 else t.get("ts", 0)
        curr_ts = t.get("ts", 0)
        delta_ms = (curr_ts - prev_ts) * 1000 if i > 0 else 0
        trade_flow.append({
            "market": t["market_ticker"],
            "price": t.get("yes_price"),
            "count": t.get("count", 1),
            "side": t.get("taker_side"),
            "ts": curr_ts,
            "delta_ms": round(delta_ms),
        })

    # Size distribution
    size_buckets = {"1-10": 0, "11-50": 0, "51-100": 0, "101-500": 0, "500+": 0}
    for t in all_trades:
        count = t.get("count", 1)
        if count <= 10:
            size_buckets["1-10"] += 1
        elif count <= 50:
            size_buckets["11-50"] += 1
        elif count <= 100:
            size_buckets["51-100"] += 1
        elif count <= 500:
            size_buckets["101-500"] += 1
        else:
            size_buckets["500+"] += 1

    # Whale trades (>= 100 contracts)
    whale_trades = [t for t in trade_flow if t["count"] >= 100]

    # Timing pattern: trades per second-of-minute
    timing_histogram = {}
    for t in all_trades:
        ts = t.get("ts", 0)
        sec = int(ts) % 60
        timing_histogram[sec] = timing_histogram.get(sec, 0) + 1

    # Anomalies: look for rapid sequences (< 100ms between trades)
    rapid_sequences = []
    seq = []
    for t in trade_flow:
        if 0 < t["delta_ms"] < 100:
            seq.append(t)
        else:
            if len(seq) >= 3:
                rapid_sequences.append(seq)
            seq = [t] if 0 < t["delta_ms"] < 100 else []
    # Check final sequence
    if len(seq) >= 3:
        rapid_sequences.append(seq)

    # Entity detection: cluster trades by pattern and fingerprint each cluster
    detected_entities = []
    clusters = _cluster_trades_by_pattern(trade_flow)
    for cluster in clusters:
        fp_result = _compute_entity_fingerprint(cluster)
        entity_type = _guess_entity_type(cluster)

        # Confidence based on cluster size
        if len(cluster) >= 10:
            confidence = "high"
        elif len(cluster) >= 5:
            confidence = "medium"
        else:
            confidence = "low"

        detected_entities.append({
            "fingerprint": fp_result["fingerprint"],
            "type_guess": entity_type,
            "trade_count": len(cluster),
            "avg_size": fp_result["pattern"].get("avg_size", 0),
            "timing_pattern": fp_result["pattern"].get("timing_sig", "?"),
            "confidence": confidence,
        })

    # Sort by trade count descending
    detected_entities.sort(key=lambda e: e["trade_count"], reverse=True)

    return {
        "event_ticker": event_ticker,
        "total_trades": len(all_trades),
        "trade_flow": trade_flow[-30:],  # Last 30 for token efficiency
        "size_distribution": size_buckets,
        "whale_trades": whale_trades,
        "whale_count": len(whale_trades),
        "timing_histogram": timing_histogram,
        "rapid_sequences": len(rapid_sequences),
        "rapid_sequence_samples": rapid_sequences[:3],  # First 3 for examples
        "detected_entities": detected_entities[:10],  # Top 10 entities
    }


@tool
async def analyze_orderbook_patterns(market_ticker: str) -> Dict[str, Any]:
    """Analyze orderbook for MM/bot signatures.

    Looks at current orderbook state for:
    - Symmetric quoting (bids and asks at equal distances from mid)
    - Round number clustering (orders at 50, 60, 70, etc.)
    - Size patterns (consistent sizes suggest automation)
    - Layering (multiple levels with similar sizes)

    Args:
        market_ticker: The market to analyze

    Returns:
        Dict with mm_likelihood, patterns detected, anomalies
    """
    if not _index:
        return {"error": "Index not available"}

    event_ticker = _index.get_event_for_ticker(market_ticker)
    if not event_ticker:
        return {"error": f"Market {market_ticker} not tracked"}

    event = _index.events.get(event_ticker)
    market = event.markets.get(market_ticker) if event else None
    if not market:
        return {"error": f"Market {market_ticker} not found"}

    yes_levels = market.yes_levels or []
    no_levels = market.no_levels or []

    patterns = []

    # 1. Check for symmetric quoting (MM signature)
    if market.yes_bid and market.yes_ask:
        mid = (market.yes_bid + market.yes_ask) / 2
        bid_dist = mid - market.yes_bid
        ask_dist = market.yes_ask - mid
        if abs(bid_dist - ask_dist) <= 1:  # Within 1 cent
            patterns.append("symmetric_quotes")

    # 2. Check for round number clustering
    all_prices = [l[0] for l in yes_levels] + [l[0] for l in no_levels]
    round_count = sum(1 for p in all_prices if p % 5 == 0)
    if len(all_prices) > 0 and round_count / len(all_prices) > 0.6:
        patterns.append("round_number_clustering")

    # 3. Check for consistent sizes (automation signal)
    all_sizes = [l[1] for l in yes_levels] + [l[1] for l in no_levels]
    if len(all_sizes) >= 3:
        avg_size = sum(all_sizes) / len(all_sizes)
        variance = sum((s - avg_size) ** 2 for s in all_sizes) / len(all_sizes)
        if variance < (avg_size * 0.3) ** 2:  # Low variance
            patterns.append("consistent_sizes")

    # 4. Check for layering (multiple levels, similar sizes)
    if len(yes_levels) >= 3 or len(no_levels) >= 3:
        patterns.append("layered_book")

    # MM likelihood score
    mm_signals = {"symmetric_quotes", "consistent_sizes", "layered_book"}
    mm_score = len(set(patterns) & mm_signals) / 3.0

    return {
        "market_ticker": market_ticker,
        "spread": market.spread,
        "yes_levels_count": len(yes_levels),
        "no_levels_count": len(no_levels),
        "patterns_detected": patterns,
        "mm_likelihood": round(mm_score, 2),
        "mm_likelihood_label": "high" if mm_score > 0.6 else "medium" if mm_score > 0.3 else "low",
        "top_bid": {"price": market.yes_bid, "size": market.yes_bid_size},
        "top_ask": {"price": market.yes_ask, "size": market.yes_ask_size},
    }


@tool
async def update_understanding(
    event_ticker: str,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """Rebuild the structured understanding for an event.

    Use this to refresh event context (Wikipedia, LLM synthesis, extensions).
    The understanding includes: trading_summary, key_factors, participants,
    timeline, trading_considerations, and domain-specific extensions.

    Args:
        event_ticker: The Kalshi event ticker
        force_refresh: If True, bypass cache and rebuild from scratch

    Returns:
        Dict with understanding summary or error
    """
    if not _index:
        return {"error": "EventArbIndex not available"}
    if not _understanding_builder:
        return {"error": "UnderstandingBuilder not available"}

    event = _index.events.get(event_ticker)
    if not event:
        return {"error": f"Event {event_ticker} not found"}

    try:
        understanding = await _understanding_builder.build(
            event, force_refresh=force_refresh
        )

        # Store on EventMeta
        event.understanding = understanding.to_dict()

        # Broadcast update to frontend
        if _broadcast_callback:
            try:
                await _broadcast_callback({
                    "type": "event_understanding_update",
                    "data": {
                        "event_ticker": event_ticker,
                        "understanding": understanding.to_dict(),
                    },
                })
            except Exception as e:
                logger.debug(f"Failed to broadcast understanding update: {e}")

        return {
            "status": "updated",
            "event_ticker": event_ticker,
            "trading_summary": understanding.trading_summary,
            "key_factors": understanding.key_factors,
            "participants": len(understanding.participants),
            "extensions": list(understanding.extensions.keys()),
            "version": understanding.version,
            "stale": understanding.stale,
        }

    except Exception as e:
        return {"error": f"Understanding build failed: {e}"}


@tool
async def search_event_news(
    event_ticker: str,
    query: str = "",
    topic: str = "news",
    time_range: str = "week",
) -> Dict[str, Any]:
    """Search for recent news and context about an event. Up to 5 searches per cycle.

    Use topic="news" for breaking developments and recent coverage.
    Use topic="general" for deeper background, analysis, and historical context.

    Returns articles with titles, URLs, content snippets, relevance scores,
    and published dates. Advanced depth provides multiple snippets per source.

    Args:
        event_ticker: The Kalshi event ticker
        query: Specific search query (leave empty to auto-generate from event title)
        topic: "news" for recent coverage, "general" for background/analysis
        time_range: "day", "week", "month" - how far back to search
    """
    if not _search_service:
        return {"error": "Search service not available (no TAVILY_API_KEY configured)"}
    if not _index:
        return {"error": "EventArbIndex not available"}

    event = _index.events.get(event_ticker)
    if not event:
        return {"error": f"Event {event_ticker} not found"}

    # Auto-generate query from event title if not provided
    if not query:
        from .event_understanding import UnderstandingBuilder
        query = UnderstandingBuilder._extract_keywords_from_title(event.title)
        if not query or len(query) < 5:
            query = event.title
        query += " latest news"

    if topic == "news":
        results = await _search_service.search_news(
            query=query,
            time_range=time_range,
            event_ticker=event_ticker,
        )
    else:
        results = await _search_service.search(
            query=query,
            topic="general",
            time_range=time_range,
            event_ticker=event_ticker,
        )

    # Auto-persist articles to news store if available
    if results and _news_store:
        try:
            stored = await _news_store.persist_articles(results, event_ticker)
            if stored > 0:
                logger.debug(f"[NEWS_STORE] Persisted {stored} articles for {event_ticker}")
        except Exception as e:
            logger.debug(f"[NEWS_STORE] Auto-persist failed: {e}")

    return {
        "event_ticker": event_ticker,
        "query": query,
        "topic": topic,
        "time_range": time_range,
        "articles": results,
        "count": len(results),
    }


@tool
async def recall_context(
    query: str,
    memory_types: str = "",
    event_ticker: str = "",
    limit: int = 5,
) -> Dict[str, Any]:
    """Search past learnings, trades, news, and observations by semantic similarity.

    Use this to recall relevant context before making decisions:
    - Before trading: recall_context("similar trades on [market type]")
    - When news is stale: recall_context("[event topic] recent developments")
    - After a loss: recall_context("mistakes on [market type]")
    - For thesis building: recall_context("[topic] price impact patterns")

    Args:
        query: Natural language search query
        memory_types: Comma-separated filter (e.g. "trade,learning,news"). Empty = all types.
        event_ticker: Filter to specific event (optional)
        limit: Max results (default 5)

    Returns:
        Dict with results list, each containing content, type, similarity score, and age
    """
    if not _memory_store:
        return {"error": "Memory store not available"}

    types_list = [t.strip() for t in memory_types.split(",") if t.strip()] or None

    try:
        results = await _memory_store.search(
            query=query,
            limit=limit,
            memory_types=types_list,
            event_ticker=event_ticker or None,
        )

        # Format results with age info
        now = time.time()
        formatted = []
        for r in results:
            entry = {
                "content": r.get("content", ""),
                "type": r.get("memory_type", r.get("type", "unknown")),
                "similarity": round(r.get("similarity", 0.0), 3) if "similarity" in r else None,
            }
            # Compute age from timestamp
            ts = r.get("timestamp") or r.get("created_at")
            if ts:
                if isinstance(ts, (int, float)):
                    age_hours = (now - ts) / 3600
                else:
                    try:
                        from datetime import datetime, timezone
                        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                        age_hours = (now - dt.timestamp()) / 3600
                    except (ValueError, TypeError):
                        age_hours = None
                if age_hours is not None:
                    entry["age_hours"] = round(age_hours, 1)

            # Include source info
            entry["source"] = r.get("_source", "unknown")
            formatted.append(entry)

        return {
            "query": query,
            "results": formatted,
            "count": len(formatted),
        }
    except Exception as e:
        return {"error": f"Recall failed: {e}"}


@tool
async def search_news_history(
    query: str,
    event_ticker: str = "",
    days: int = 7,
    limit: int = 5,
) -> Dict[str, Any]:
    """Search persisted news articles by semantic similarity.

    Unlike search_event_news (which fetches fresh from Tavily), this searches
    previously stored articles with their price snapshots at time of storage.

    Args:
        query: Natural language search query
        event_ticker: Filter to specific event (optional)
        days: How far back to search (default 7)
        limit: Max results (default 5)

    Returns:
        Dict with articles including price_snapshot at time of storage
    """
    if not _news_store:
        # Fall back to general recall if no news store
        if _memory_store:
            results = await _memory_store.search(
                query=query,
                limit=limit,
                memory_types=["news"],
                event_ticker=event_ticker or None,
            )
            return {"query": query, "results": results, "count": len(results), "source": "vector_fallback"}
        return {"error": "News store not available"}

    try:
        results = await _news_store.search_news(
            query=query,
            event_ticker=event_ticker or None,
            hours=days * 24,
            limit=limit,
        )
        return {
            "query": query,
            "event_ticker": event_ticker,
            "results": results,
            "count": len(results),
        }
    except Exception as e:
        return {"error": f"News search failed: {e}"}


@tool
async def get_price_movers(
    event_ticker: str,
    min_change_cents: int = 5,
    limit: int = 5,
) -> Dict[str, Any]:
    """Find news articles that correlated with significant price movements.

    Returns articles where prices moved >= min_change_cents within 1-24h
    after the article was stored. Use this to learn what kind of news
    moves this market.

    Args:
        event_ticker: The event to check
        min_change_cents: Minimum price change to qualify as a "mover" (default 5)
        limit: Max results (default 5)

    Returns:
        Dict with market-moving articles and their price impact data
    """
    if not _impact_tracker:
        return {"error": "Price impact tracker not available"}

    try:
        movers = await _impact_tracker.find_market_movers(
            event_ticker=event_ticker,
            min_change_cents=min_change_cents,
            limit=limit,
        )
        return {
            "event_ticker": event_ticker,
            "min_change_cents": min_change_cents,
            "movers": movers,
            "count": len(movers),
        }
    except Exception as e:
        return {"error": f"Price movers query failed: {e}"}


@tool
async def manage_thesis(
    action: str,
    thesis_id: str = "",
    title: str = "",
    event_ticker: str = "",
    evidence: str = "",
    implication: str = "",
    confidence_delta: float = 0.0,
) -> Dict[str, Any]:
    """Manage multi-day strategic theses.

    Theses persist across cycles in THESES.md and help build conviction
    from accumulating evidence over time.

    Actions:
      - "create": Start a new thesis. Requires title, event_ticker, implication.
      - "add_evidence": Add supporting evidence to an existing thesis. Requires thesis_id, evidence.
      - "validate": Mark thesis as validated (confidence >= 0.6, 3+ evidence points).
      - "invalidate": Mark thesis as invalidated with reason. Requires thesis_id, evidence (reason).
      - "list": List all active theses.

    Guardrails:
      - Max 5 active theses. Create will fail if at limit.
      - Confidence increases max 0.1 per cycle (prevents conviction spikes).
      - Theses auto-expire after 7 days without new evidence.
      - First 10 cycles: thesis creation blocked (need observation data first).

    Args:
        action: "create", "add_evidence", "validate", "invalidate", or "list"
        thesis_id: ID of existing thesis (for add_evidence/validate/invalidate)
        title: Short thesis title (for create, e.g. "tariff-escalation")
        event_ticker: Related event (for create)
        evidence: Evidence text (for create/add_evidence) or reason (for invalidate)
        implication: Trading implication (for create, e.g. "YES probability should increase")
        confidence_delta: How much to adjust confidence (max 0.1, for add_evidence)
    """
    import json as _json
    import os as _os

    theses_path = _os.path.join(
        _os.path.dirname(_os.path.abspath(__file__)), "memory", "data", "theses.json"
    )

    # Load existing theses
    theses = []
    if _os.path.exists(theses_path):
        try:
            with open(theses_path, "r") as f:
                theses = _json.load(f)
        except (ValueError, OSError):
            theses = []

    def _save():
        _os.makedirs(_os.path.dirname(theses_path), exist_ok=True)
        with open(theses_path, "w") as f:
            _json.dump(theses, f, indent=2)

    now = time.time()

    # Auto-expire stale theses (>7 days without evidence)
    for t in theses:
        if t.get("status") not in ("active", "validating"):
            continue
        last_evidence_ts = t.get("last_evidence_ts", t.get("created_at", 0))
        if (now - last_evidence_ts) > 7 * 24 * 3600:
            t["status"] = "expired"
            t["expired_at"] = now
            # Archive to vector memory
            if _memory_store:
                try:
                    _memory_store.append(
                        content=f"EXPIRED THESIS: {t['title']} | {t.get('implication', '')} | evidence_count={len(t.get('evidence', []))}",
                        memory_type="thesis_archived",
                        metadata={"thesis_id": t["id"], "event_ticker": t.get("event_ticker", "")},
                    )
                except Exception:
                    pass

    active_theses = [t for t in theses if t.get("status") in ("active", "validating")]

    if action == "list":
        _save()
        return {
            "active_theses": [
                {
                    "id": t["id"],
                    "title": t["title"],
                    "event_ticker": t.get("event_ticker", ""),
                    "confidence": t.get("confidence", 0.3),
                    "evidence_count": len(t.get("evidence", [])),
                    "status": t["status"],
                    "age_hours": round((now - t.get("created_at", now)) / 3600, 1),
                    "implication": t.get("implication", ""),
                }
                for t in active_theses
            ],
            "total_active": len(active_theses),
            "total_archived": len([t for t in theses if t["status"] in ("expired", "invalidated", "validated")]),
        }

    if action == "create":
        if not title or not event_ticker:
            return {"error": "create requires title and event_ticker"}
        if len(active_theses) >= 5:
            return {"error": "Max 5 active theses. Invalidate or expire one first."}

        import hashlib
        thesis_id = f"T-{hashlib.md5(f'{title}{now}'.encode()).hexdigest()[:6].upper()}"

        # Get current price snapshot
        price_snapshot = {}
        if _index:
            event = _index.events.get(event_ticker)
            if event:
                for ticker, m in event.markets.items():
                    if m.yes_bid is not None:
                        price_snapshot[ticker] = {"yes_bid": m.yes_bid, "yes_ask": m.yes_ask}

        new_thesis = {
            "id": thesis_id,
            "title": title,
            "event_ticker": event_ticker,
            "implication": implication,
            "confidence": 0.3,
            "status": "active",
            "evidence": [{"text": evidence, "ts": now}] if evidence else [],
            "price_at_creation": price_snapshot,
            "created_at": now,
            "last_evidence_ts": now,
        }
        theses.append(new_thesis)
        _save()
        _update_theses_md(theses)
        return {"status": "created", "thesis_id": thesis_id, "confidence": 0.3}

    if action == "add_evidence":
        if not thesis_id or not evidence:
            return {"error": "add_evidence requires thesis_id and evidence"}

        target = next((t for t in theses if t["id"] == thesis_id), None)
        if not target:
            return {"error": f"Thesis {thesis_id} not found"}
        if target["status"] not in ("active", "validating"):
            return {"error": f"Thesis {thesis_id} is {target['status']}, cannot add evidence"}

        target["evidence"].append({"text": evidence, "ts": now})
        target["last_evidence_ts"] = now

        # Adjust confidence (capped at +0.1 per cycle)
        delta = min(confidence_delta, 0.1) if confidence_delta > 0 else max(confidence_delta, -0.2)
        target["confidence"] = max(0.0, min(1.0, target["confidence"] + delta))

        # Auto-promote to validating at 3+ evidence
        if len(target["evidence"]) >= 3 and target["status"] == "active":
            target["status"] = "validating"

        _save()
        _update_theses_md(theses)
        return {
            "status": "evidence_added",
            "thesis_id": thesis_id,
            "confidence": target["confidence"],
            "evidence_count": len(target["evidence"]),
            "thesis_status": target["status"],
        }

    if action == "validate":
        if not thesis_id:
            return {"error": "validate requires thesis_id"}
        target = next((t for t in theses if t["id"] == thesis_id), None)
        if not target:
            return {"error": f"Thesis {thesis_id} not found"}
        if target["confidence"] < 0.6:
            return {"error": f"Confidence {target['confidence']:.2f} < 0.6 required for validation"}
        if len(target["evidence"]) < 3:
            return {"error": f"Need 3+ evidence points, have {len(target['evidence'])}"}
        target["status"] = "validated"
        target["validated_at"] = now
        _save()
        _update_theses_md(theses)
        return {"status": "validated", "thesis_id": thesis_id}

    if action == "invalidate":
        if not thesis_id:
            return {"error": "invalidate requires thesis_id"}
        target = next((t for t in theses if t["id"] == thesis_id), None)
        if not target:
            return {"error": f"Thesis {thesis_id} not found"}
        target["status"] = "invalidated"
        target["invalidated_at"] = now
        target["invalidation_reason"] = evidence
        _save()
        _update_theses_md(theses)
        # Archive to vector memory
        if _memory_store:
            try:
                _memory_store.append(
                    content=f"INVALIDATED THESIS: {target['title']} | reason: {evidence}",
                    memory_type="thesis_archived",
                    metadata={"thesis_id": thesis_id, "event_ticker": target.get("event_ticker", "")},
                )
            except Exception:
                pass
        return {"status": "invalidated", "thesis_id": thesis_id}

    return {"error": f"Unknown action: {action}. Use create/add_evidence/validate/invalidate/list."}


def _update_theses_md(theses: list) -> None:
    """Write active theses to THESES.md for Captain system prompt."""
    import os as _os

    theses_md_path = _os.path.join(
        _os.path.dirname(_os.path.abspath(__file__)), "memory", "data", "THESES.md"
    )

    active = [t for t in theses if t.get("status") in ("active", "validating", "validated")]
    now = time.time()

    lines = ["# Active Theses\n", "\n"]
    if not active:
        lines.append("(No active theses. Use manage_thesis(action='create') to start one.)\n")
    else:
        for t in active[:5]:
            age_days = (now - t.get("created_at", now)) / 86400
            lines.append(
                f"## {t['id']} | {t['title']} | confidence: {t.get('confidence', 0.3):.1f} | "
                f"status: {t['status']} | age: {age_days:.0f}d\n"
            )
            if t.get("implication"):
                lines.append(f"Implication: {t['implication']}\n")
            evidence_list = t.get("evidence", [])
            lines.append(f"Evidence: {len(evidence_list)} points\n")
            # Show last 2 evidence items
            for e in evidence_list[-2:]:
                lines.append(f"  - {e.get('text', '')[:100]}\n")
            lines.append("\n")

    _os.makedirs(_os.path.dirname(theses_md_path), exist_ok=True)
    with open(theses_md_path, "w") as f:
        f.writelines(lines)


# --- Sniper Execution Layer Tools ---


@tool
async def configure_sniper(**kwargs) -> Dict[str, Any]:
    """Configure the Sniper execution layer at runtime.

    The Sniper handles sub-second S1_ARB market rebalancing execution.
    You own strategy; Sniper owns speed. Tune these parameters based on
    market conditions and your risk appetite.

    Settable fields:
      - enabled (bool): Master switch. Set True to activate Sniper execution.
      - max_position (int): Max contracts per market per arb (default 25).
      - max_capital (int): Max active capital at risk in cents (default 5000 = $50).
      - cooldown (float): Seconds between trades on same event (default 10.0).
      - max_trades_per_cycle (int): Max trades between your cycles (default 5).
      - arb_min_edge (float): Min edge in cents after fees for S1_ARB (default 3.0).
      - vpin_reject_threshold (float): VPIN above this = toxic flow, skip (default 0.7).
      - order_ttl (int): Order time-to-live in seconds (default 30).
      - leg_timeout (float): Per-leg placement timeout in seconds (default 5.0).
      - s4_obi_enabled (bool): Enable OBI momentum strategy (default False).
      - s6_spread_enabled (bool): Enable spread capture (default False).
      - s7_sweep_enabled (bool): Enable sweep following (default False).

    Note: Typo'd field names are detected and returned in 'unknown_fields'.

    Example: configure_sniper(enabled=True, arb_min_edge=2.5, max_position=15)

    Returns:
        Current config with list of changed fields (and unknown fields if any typos).
    """
    if not _sniper_config:
        return {"error": "Sniper not initialized. Is sniper_enabled=true in config?"}

    changed, unknown = _sniper_config.update(**kwargs)
    result = {
        "config": _sniper_config.to_dict(),
        "changed": changed,
    }
    if unknown:
        result["unknown_fields"] = unknown
        result["warning"] = f"Unknown fields ignored: {unknown}. Check spelling."
    if changed:
        logger.info(f"[SNIPER:CONFIG] Captain updated: {changed}")
    return result


@tool
async def get_sniper_status() -> Dict[str, Any]:
    """Get Sniper execution layer status and telemetry.

    Returns real-time Sniper state including:
      - Running status and current config
      - Trade counts (this cycle, total, executed, rejected)
      - Capital deployed, active orders
      - Last rejection reason (for diagnosing risk gate blocks)
      - Recent execution actions with latency

    Use this to monitor Sniper performance and diagnose issues.
    """
    if not _sniper:
        return {"error": "Sniper not initialized. Is sniper_enabled=true in config?"}

    return {
        "config": _sniper.config.to_dict(),
        "state": _sniper.state.to_dict(),
    }


@tool
async def kill_sniper_positions(reason: str = "captain_request") -> Dict[str, Any]:
    """Emergency: cancel ALL active Sniper orders immediately.

    Use when:
      - Market conditions have changed and you want to stop Sniper activity
      - You detect adverse selection or toxic flow across multiple markets
      - You want to reset Sniper state before reconfiguring

    Args:
        reason: Why you're killing positions (logged for audit)

    Returns:
        Dict with cancelled count and any errors.
    """
    if not _sniper:
        return {"error": "Sniper not initialized. Is sniper_enabled=true in config?"}

    result = await _sniper.kill_positions(reason=reason)
    logger.info(f"[SNIPER:KILL] Captain killed positions: {result}")
    return result


@tool
async def get_orderbook_intelligence(event_ticker: str) -> Dict[str, Any]:
    """Synthesize orderbook microstructure intelligence for an event.

    Aggregates microprice, OFI, VPIN, sweep status, and liquidity-adjusted
    edge across all markets in an event. Use before making trading decisions
    to understand the microstructure regime.

    Returns per-market:
      - microprice: Depth-weighted fair value (better than mid)
      - ofi / ofi_ema: Order flow imbalance (positive = buy pressure)
      - vpin: Volume-synchronized PIN (>0.7 = toxic informed flow)
      - sweep_active/direction: Aggressive sweep in progress
      - book_imbalance: Bid vs ask depth ratio

    Returns event-level:
      - liquidity_adjusted_edge: Edge after walking depth (not just BBO)
      - regime: "normal", "toxic" (high VPIN), "sweep" (active sweep), "thin" (low depth)

    Args:
        event_ticker: Event to analyze
    """
    if not _index:
        return {"error": "EventArbIndex not available"}

    event = _index.events.get(event_ticker)
    if not event:
        return {"error": f"Event {event_ticker} not found"}

    markets_intel = {}
    regime_flags = set()

    for ticker, m in event.markets.items():
        intel = {
            "microprice": round(m.microprice, 2) if m.microprice is not None else None,
            "yes_mid": m.yes_mid,
            "spread": m.spread,
            "ofi": round(m.micro.ofi, 3),
            "ofi_ema": round(m.micro.ofi_ema, 3),
            "vpin": round(m.micro.vpin, 3),
            "book_imbalance": round(m.micro.book_imbalance, 3),
            "total_bid_depth": m.micro.total_bid_depth,
            "total_ask_depth": m.micro.total_ask_depth,
            "sweep_active": m.micro.sweep_active,
            "sweep_direction": m.micro.sweep_direction,
            "sweep_size": m.micro.sweep_size,
            "volume_5m": m.micro.volume_5m,
            "buy_sell_ratio": round(m.micro.buy_sell_ratio, 2),
        }
        markets_intel[ticker] = intel

        # Flag regime (use config threshold if available, else default 0.7)
        vpin_threshold = _sniper_config.vpin_reject_threshold if _sniper_config else 0.7
        if m.micro.vpin > vpin_threshold:
            regime_flags.add("toxic")
        if m.micro.sweep_active:
            regime_flags.add("sweep")
        if (m.micro.total_bid_depth + m.micro.total_ask_depth) < 20:
            regime_flags.add("thin")

    # Liquidity-adjusted edge (walk depth levels)
    la_long = event.liquidity_adjusted_edge("long", target_contracts=10)
    la_short = event.liquidity_adjusted_edge("short", target_contracts=10)

    regime = "normal"
    if "toxic" in regime_flags:
        regime = "toxic"
    elif "sweep" in regime_flags:
        regime = "sweep"
    elif "thin" in regime_flags:
        regime = "thin"

    return {
        "event_ticker": event_ticker,
        "regime": regime,
        "regime_flags": list(regime_flags),
        "markets": markets_intel,
        "liquidity_adjusted_edge_long": la_long,
        "liquidity_adjusted_edge_short": la_short,
    }
