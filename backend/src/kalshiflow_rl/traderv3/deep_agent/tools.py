"""
Deep Agent Tools - Callable functions for the self-improving agent.

Provides tools for:
- Extraction signals (get_extraction_signals) - PRIMARY DATA SOURCE
- Event understanding (understand_event) - builds extraction specs
- Market data retrieval (get_markets)
- Trade execution (trade)
- Session state querying (get_session_state)
- Memory file access (read_memory, write_memory)

The agent trades on extraction signals from the langextract pipeline.
The get_extraction_signals() tool queries the extractions table
populated by KalshiExtractor via the entity extraction system.

These tools are designed for use with Claude's native tool calling
and provide structured outputs for the agent to reason about.
"""

import asyncio
import json
import logging
import os
import re
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Literal, Optional, TYPE_CHECKING

from ..nlp.kalshi_extractor import EventConfig
from ..state.trading_attachment import TrackedMarketOrder

if TYPE_CHECKING:
    from ..core.websocket_manager import V3WebSocketManager
    from ..clients.trading_client_integration import V3TradingClientIntegration
    from ..core.state_container import V3StateContainer
    from ..state.tracked_markets import TrackedMarketsState
    from ..services.event_position_tracker import EventPositionTracker

logger = logging.getLogger("kalshiflow_rl.traderv3.deep_agent.tools")

# Default memory directory
DEFAULT_MEMORY_DIR = Path(__file__).parent / "memory"


@dataclass
class MarketInfo:
    """Structured market information returned by get_markets."""
    ticker: str
    event_ticker: str
    title: str
    yes_bid: int
    yes_ask: int
    spread: int
    volume_24h: int
    status: str
    last_trade_price: Optional[int] = None
    data_quality: str = "live"  # "live" | "estimated" | "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TradeResult:
    """Result of a trade execution."""
    success: bool
    ticker: str
    side: str
    contracts: int
    price_cents: Optional[int] = None
    order_id: Optional[str] = None
    error: Optional[str] = None
    limit_price_cents: Optional[int] = None  # The computed limit price used for the order
    order_status: Optional[str] = None  # Kalshi order status (resting, executed, etc.)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)



@dataclass
class SessionState:
    """Current trading session state."""
    balance_cents: int
    portfolio_value_cents: int
    realized_pnl_cents: int
    unrealized_pnl_cents: int
    total_pnl_cents: int
    position_count: int
    open_order_count: int
    trade_count: int
    win_rate: float
    positions: List[Dict[str, Any]] = field(default_factory=list)
    open_orders: List[Dict[str, Any]] = field(default_factory=list)
    recent_fills: List[Dict[str, Any]] = field(default_factory=list)
    is_valid: bool = True  # False when state container is unavailable or errored
    error: Optional[str] = None  # Error description when is_valid is False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TruePerformance:
    """True performance data from Kalshi API via state container, filtered by strategy."""
    # Account-level (not strategy-filtered)
    balance_cents: int
    portfolio_value_cents: int

    # Strategy-filtered positions
    positions: List[Dict[str, Any]] = field(default_factory=list)
    position_count: int = 0
    unrealized_pnl_cents: int = 0

    # Strategy-filtered settlements
    settlements: List[Dict[str, Any]] = field(default_factory=list)
    realized_pnl_cents: int = 0
    total_pnl_cents: int = 0
    trade_count: int = 0
    win_count: int = 0
    loss_count: int = 0
    win_rate: float = 0.0

    # Per-event breakdown (from settlements + positions)
    by_event: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    is_valid: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FillRecord:
    """A recorded fill from trade execution."""
    order_id: str
    ticker: str
    side: str  # "yes" or "no"
    contracts: int
    price_cents: int
    timestamp: float
    reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EventMarketInfo:
    """Info about a single market within an event."""
    ticker: str
    title: str
    yes_price: Optional[int]  # Current YES price in cents (None if unknown)
    no_price: Optional[int]   # Current NO price (100 - yes_price, None if unknown)
    has_position: bool
    position_side: str  # "yes", "no", or "none"
    position_contracts: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EventContextInfo:
    """
    Context about an event's markets and positions.

    Explains mutual exclusivity relationships and risk levels
    for markets within the same event.
    """
    event_ticker: str
    market_count: int
    markets: List[EventMarketInfo]
    yes_sum: int              # Sum of all YES prices in cents
    no_sum: int               # N*100 - yes_sum
    risk_level: str           # ARBITRAGE, NORMAL, HIGH_RISK, GUARANTEED_LOSS
    has_positions: bool       # True if any position in this event
    position_count: int       # Number of markets with positions
    total_contracts: int      # Total contracts across all markets
    mutual_exclusivity_note: str  # Human-readable explanation

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)



class DeepAgentTools:
    """
    Container for all deep agent tools with shared state.

    This class holds references to the trading client, state container,
    and WebSocket manager, providing them to the individual tool functions.
    """

    def __init__(
        self,
        trading_client: Optional['V3TradingClientIntegration'] = None,
        state_container: Optional['V3StateContainer'] = None,
        websocket_manager: Optional['V3WebSocketManager'] = None,
        memory_dir: Optional[Path] = None,
        tracked_markets: Optional['TrackedMarketsState'] = None,
        event_position_tracker: Optional['EventPositionTracker'] = None,
    ):
        """
        Initialize tools with shared dependencies.

        Args:
            trading_client: Client for executing trades and fetching market data
            state_container: Container for trading state (positions, orders, etc.)
            websocket_manager: WebSocket manager for streaming updates
            memory_dir: Directory for memory files (defaults to ./memory)
            tracked_markets: State container for tracked markets (for event context)
            event_position_tracker: Tracker for event-level position risk
        """
        self._trading_client = trading_client
        self._state_container = state_container
        self._ws_manager = websocket_manager
        self._memory_dir = memory_dir or DEFAULT_MEMORY_DIR
        self._tracked_markets = tracked_markets
        self._event_position_tracker = event_position_tracker
        self._supabase_client = None

        # Vector memory service (set by agent after startup)
        self._vector_memory = None  # VectorMemoryService instance

        # Trade executor (set by plugin after startup)
        self._trade_executor = None  # TradeExecutor instance

        # Token usage callback (set by agent to accumulate tool API costs)
        self._token_usage_callback = None

        # Reddit Historic Agent reference (set by coordinator after startup)
        self._reddit_historic_agent = None

        # GDELT client references (set by coordinator after startup)
        self._gdelt_client = None       # BigQuery GKG (structured entities/themes)
        self._gdelt_doc_client = None   # Free DOC API (article search/timelines)

        # GDELT News Analyzer sub-agent (set by coordinator after startup)
        self._news_analyzer = None      # GDELTNewsAnalyzer (Haiku sub-agent)

        # Microstructure services (set by coordinator after startup)
        self._orderbook_integration = None   # V3OrderbookIntegration
        self._trade_flow_service = None      # TradeFlowService

        # RF-1: Per-event dollar exposure tracking (resets on restart)
        self._event_exposure_cents: Dict[str, int] = {}
        # $1,000 cap per event — limits single-event concentration risk.
        # Sized for paper trading; scale with bankroll for production.
        self._max_event_exposure_cents: int = 100_000

        # Fresh news enforcement (wired from agent config)
        self._require_fresh_news: bool = True
        self._max_news_age_hours: float = 2.0

        # RF-5: Startup timestamp for marking pre-existing signals as historical
        self._startup_time: float = time.time()

        # Ensure memory directory exists
        self._memory_dir.mkdir(parents=True, exist_ok=True)

        # Recent fills tracking for session state
        self._recent_fills: Deque[FillRecord] = deque(maxlen=50)

        # GDELT query tracking for reflection (stores last N queries for trade snapshots)
        self._recent_gdelt_queries: Deque[Dict[str, Any]] = deque(maxlen=20)
        # Full GDELT results for WebSocket snapshot (new clients get recent results)
        self._recent_gdelt_results: Deque[Dict[str, Any]] = deque(maxlen=10)

        # Track tickers the deep agent has traded (for position filtering)
        # Only positions in tickers we've actually traded are shown to the agent,
        # preventing 100+ old positions from previous sessions from appearing.
        self._traded_tickers: set = set()

        # Cache of event tickers that failed understand_event (e.g., KXQUICKSETTLE).
        # Prevents repeated LLM calls for events that always produce empty/bad JSON.
        self._failed_event_tickers: set = set()

        # Circuit breaker callback: set by agent.py to check if a ticker is blacklisted.
        # Signature: (ticker: str) -> tuple[bool, Optional[str]]
        # Returns (is_blacklisted, reason_message)
        self._circuit_breaker_checker: Optional[Callable] = None

        # Track tool usage for metrics
        self._tool_calls = {
            "get_extraction_signals": 0,
            "understand_event": 0,
            "get_markets": 0,
            "preflight_check": 0,
            "trade": 0,
            "get_session_state": 0,
            "get_true_performance": 0,
            "read_memory": 0,
            "write_memory": 0,
            "append_memory": 0,
            "get_event_context": 0,
            "evaluate_extractions": 0,
            "refine_event": 0,
            "get_extraction_quality": 0,
            "get_reddit_daily_digest": 0,
            "write_cycle_summary": 0,
            "query_gdelt_news": 0,
            "query_gdelt_events": 0,
            "search_gdelt_articles": 0,
            "get_gdelt_volume_timeline": 0,
            "get_news_intelligence": 0,
            "get_microstructure": 0,
            "get_candlesticks": 0,
            "read_todos": 0,
            "write_todos": 0,
            "search_memory": 0,
            "submit_trade_intent": 0,
        }


    def reconstruct_event_exposure(self) -> Dict[str, int]:
        """
        Reconstruct _event_exposure_cents from current positions on startup.

        Reads positions from state_container and maps each to its event_ticker
        via tracked_markets. Estimates exposure as abs(contracts) * avg_entry_price.

        Returns:
            Dict of {event_ticker: exposure_cents} that was reconstructed
        """
        if not self._state_container or not self._tracked_markets:
            logger.info("[deep_agent.tools] No state_container or tracked_markets — skipping exposure reconstruction")
            return {}

        try:
            summary = self._state_container.get_trading_summary()
            positions_details = summary.get("positions_details", [])

            if not positions_details:
                logger.info("[deep_agent.tools] No positions — exposure reconstruction skipped")
                return {}

            reconstructed: Dict[str, int] = {}

            for pos in positions_details:
                ticker = pos.get("ticker", "")
                total_cost = pos.get("total_cost", 0)
                position_count = abs(pos.get("position", 0))

                if not ticker or position_count == 0:
                    continue

                # Look up event_ticker from tracked_markets
                market = self._tracked_markets.get_market(ticker)
                if not market or not market.event_ticker:
                    logger.debug(f"[deep_agent.tools] No tracked market for {ticker}, skipping exposure")
                    continue

                event_ticker = market.event_ticker

                # Use total_cost as the exposure estimate (cost basis in cents)
                if total_cost <= 0:
                    logger.warning(
                        f"[deep_agent.tools] Skipping {ticker}: total_cost={total_cost}, "
                        f"cannot reconstruct exposure without cost basis"
                    )
                    continue
                reconstructed[event_ticker] = reconstructed.get(event_ticker, 0) + total_cost

            # Apply to state
            self._event_exposure_cents = reconstructed

            if reconstructed:
                for et, exp in reconstructed.items():
                    logger.info(
                        f"[deep_agent.tools] Reconstructed exposure: {et} = "
                        f"${exp/100:.2f} / ${self._max_event_exposure_cents/100:.0f}"
                    )

            logger.info(
                f"[deep_agent.tools] Exposure reconstruction complete: "
                f"{len(reconstructed)} events, total ${sum(reconstructed.values())/100:.2f}"
            )
            return reconstructed

        except Exception as e:
            logger.error(f"[deep_agent.tools] Exposure reconstruction failed: {e}")
            return {}

    def _get_supabase(self):
        """Get or create cached Supabase client."""
        if self._supabase_client is None:
            url = os.getenv("SUPABASE_URL")
            key = os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_ANON_KEY")
            if url and key:
                from supabase import create_client
                self._supabase_client = create_client(url, key)
        return self._supabase_client






    # === Microstructure Tools ===

    async def get_microstructure(
        self,
        market_ticker: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get real-time microstructure signals: orderbook + trade flow.

        Single-market mode (ticker provided): Returns detailed orderbook and
        trade flow data for one market.

        Scan mode (no ticker): Returns a summary of all tracked markets with
        trade activity, sorted by total trades descending.

        Args:
            market_ticker: Optional specific market ticker for detailed view.

        Returns:
            Dict with microstructure data or error message.
        """
        self._tool_calls["get_microstructure"] += 1

        has_trade_flow = self._trade_flow_service is not None
        has_orderbook = self._orderbook_integration is not None

        if not has_trade_flow and not has_orderbook:
            return {"error": "Microstructure services not available yet. They are wired after orderbook connects."}

        if market_ticker:
            # === Single-market detailed mode ===
            data: Dict[str, Any] = {"ticker": market_ticker}
            has_data = False

            # Trade flow state
            if has_trade_flow:
                tf_state = self._trade_flow_service.get_market_state(market_ticker)
                if tf_state:
                    data["yes_trades"] = tf_state.get("yes_trades", 0)
                    data["no_trades"] = tf_state.get("no_trades", 0)
                    data["total_trades"] = tf_state.get("total_trades", 0)
                    data["yes_ratio"] = tf_state.get("yes_ratio", 0.0)
                    data["price_drop"] = tf_state.get("price_drop", 0)
                    data["last_yes_price"] = tf_state.get("last_yes_price")
                    has_data = True

            # Orderbook signals
            if has_orderbook:
                ob_signals = self._orderbook_integration.get_orderbook_signals(market_ticker)
                if ob_signals:
                    data["orderbook"] = {
                        "imbalance_ratio": ob_signals.get("imbalance_ratio"),
                        "delta_count": ob_signals.get("delta_count", 0),
                        "spread_open": ob_signals.get("spread_open"),
                        "spread_close": ob_signals.get("spread_close"),
                        "spread_high": ob_signals.get("spread_high"),
                        "spread_low": ob_signals.get("spread_low"),
                        "large_order_count": ob_signals.get("large_order_count", 0),
                    }
                    has_data = True

            # L2 orderbook depth (top 5 price levels per side)
            if market_ticker:
                try:
                    from ...data.orderbook_state import get_shared_orderbook_state
                    ob_state = await get_shared_orderbook_state(market_ticker)
                    top_levels = await ob_state.get_top_levels(5)
                    # Convert {price: size} dicts to [[price, size], ...] for readability
                    data["orderbook_depth"] = {
                        "yes_bids": [[p, q] for p, q in top_levels.get("yes_bids", {}).items()],
                        "yes_asks": [[p, q] for p, q in top_levels.get("yes_asks", {}).items()],
                        "no_bids": [[p, q] for p, q in top_levels.get("no_bids", {}).items()],
                        "no_asks": [[p, q] for p, q in top_levels.get("no_asks", {}).items()],
                        "source": "websocket",
                    }
                    has_data = True
                except Exception as e:
                    logger.debug(f"Could not get orderbook depth for {market_ticker}: {e}")
                    # REST fallback when WS data unavailable
                    if self._trading_client:
                        try:
                            rest_book = await self._trading_client._client.get_orderbook(market_ticker, depth=5)
                            ob_data = rest_book.get("orderbook", {})
                            data["orderbook_depth"] = {
                                "yes_bids": ob_data.get("yes", []),
                                "no_bids": ob_data.get("no", []),
                                "source": "rest_fallback",
                            }
                            has_data = True
                        except Exception:
                            pass

            data["has_data"] = has_data

            # Build human-readable summary
            if has_data:
                parts = []
                if "total_trades" in data:
                    parts.append(f"{data['total_trades']} trades ({data.get('yes_ratio', 0):.0%} YES)")
                if "price_drop" in data and data["price_drop"] != 0:
                    parts.append(f"price move {data['price_drop']:+d}c")
                ob = data.get("orderbook", {})
                if ob.get("imbalance_ratio") is not None:
                    imb = ob["imbalance_ratio"]
                    parts.append(f"imbalance {imb:+.2f}")
                if ob.get("spread_close") is not None:
                    parts.append(f"spread {ob['spread_close']}c")
                if ob.get("large_order_count", 0) > 0:
                    parts.append(f"{ob['large_order_count']} large orders")
                data["summary"] = " | ".join(parts) if parts else "No activity"

            return {"data": data, "has_data": has_data}

        else:
            # === Scan mode: all markets with activity ===
            markets = []

            # Get trade flow states
            trade_flow_by_ticker: Dict[str, Dict] = {}
            if has_trade_flow:
                for s in self._trade_flow_service.get_market_states():
                    ticker = s.get("market_ticker", "")
                    if ticker:
                        trade_flow_by_ticker[ticker] = s

            # Merge with orderbook signals
            all_tickers = set(trade_flow_by_ticker.keys())

            for ticker in all_tickers:
                tf = trade_flow_by_ticker.get(ticker, {})
                total_trades = tf.get("total_trades", 0)
                if total_trades == 0:
                    continue

                entry: Dict[str, Any] = {
                    "ticker": ticker,
                    "total_trades": total_trades,
                    "yes_ratio": tf.get("yes_ratio", 0.0),
                    "price_drop": tf.get("price_drop", 0),
                }

                # Merge orderbook signals if available
                if has_orderbook:
                    ob_signals = self._orderbook_integration.get_orderbook_signals(ticker)
                    if ob_signals:
                        entry["spread"] = ob_signals.get("spread_close")
                        entry["imbalance"] = ob_signals.get("imbalance_ratio")
                        entry["large_order_count"] = ob_signals.get("large_order_count", 0)

                # Build one-line summary
                parts = [f"{total_trades}T"]
                parts.append(f"{entry.get('yes_ratio', 0):.0%}Y")
                pd = entry.get("price_drop", 0)
                if pd != 0:
                    parts.append(f"{pd:+d}c")
                if entry.get("spread") is not None:
                    parts.append(f"sp={entry['spread']}c")
                if entry.get("imbalance") is not None:
                    parts.append(f"imb={entry['imbalance']:+.2f}")
                if entry.get("large_order_count", 0) > 0:
                    parts.append(f"{entry['large_order_count']}whale")
                entry["summary"] = " ".join(parts)

                markets.append(entry)

            # Sort by total trades descending
            markets.sort(key=lambda m: m.get("total_trades", 0), reverse=True)

            return {
                "markets": markets,
                "total_tracked": len(markets),
                "has_data": len(markets) > 0,
            }

    # === Candlestick / OHLC Tool ===

    async def get_candlesticks(
        self,
        event_ticker: str,
        period: str = "hourly",
        hours_back: int = 24,
    ) -> Dict[str, Any]:
        """
        Fetch OHLC candlestick data for all markets in an event.

        Uses the efficient event-level endpoint (single API call for all markets).

        Args:
            event_ticker: Event ticker (e.g., "KXBONDIOUT-28")
            period: Candle period - "1min", "hourly", or "daily"
            hours_back: How many hours of history to fetch (default: 24)

        Returns:
            Dict with per-market OHLC summary and recent candles
        """
        self._tool_calls["get_candlesticks"] += 1

        if not self._trading_client:
            return {"error": "Trading client not available"}

        # Map period to interval minutes
        intervals = {"1min": 1, "hourly": 60, "daily": 1440}
        interval = intervals.get(period, 60)

        # Get event's markets to find series_ticker
        if not self._tracked_markets:
            return {"error": "No tracked markets available"}

        markets_by_event = self._tracked_markets.get_markets_by_event()
        event_markets = markets_by_event.get(event_ticker, [])
        if not event_markets:
            return {"error": f"No tracked markets for event {event_ticker}"}

        # Extract series_ticker from first market's ticker
        sample_ticker = event_markets[0].ticker
        series_ticker = sample_ticker.split("-")[0] if "-" in sample_ticker else sample_ticker

        # Time range
        end_ts = int(time.time())
        start_ts = end_ts - (hours_back * 3600)

        try:
            response = await self._trading_client.get_event_candlesticks(
                series_ticker=series_ticker,
                event_ticker=event_ticker,
                start_ts=start_ts,
                end_ts=end_ts,
                period_interval=interval,
            )
        except Exception as e:
            logger.error(f"[deep_agent.tools.get_candlesticks] API error: {e}")
            return {"error": f"Failed to fetch candlesticks: {e}"}

        # Format for agent consumption
        result_markets = []
        tickers = response.get("market_tickers", [])
        all_candles = response.get("market_candlesticks", [])

        for i, ticker in enumerate(tickers):
            candles = all_candles[i] if i < len(all_candles) else []
            if not candles:
                continue

            first = candles[0].get("price", {})
            last = candles[-1].get("price", {})
            all_highs = [c.get("price", {}).get("high", 0) for c in candles]
            all_lows = [
                c.get("price", {}).get("low", 0) for c in candles
                if c.get("price", {}).get("low", 0) > 0
            ]
            total_vol = sum(c.get("volume", 0) for c in candles)

            result_markets.append({
                "ticker": ticker,
                "candle_count": len(candles),
                "open_price": first.get("open"),
                "close_price": last.get("close"),
                "high": max(all_highs) if all_highs else None,
                "low": min(all_lows) if all_lows else None,
                "total_volume": total_vol,
                "candles": candles[-10:],  # Last 10 candles for detail
            })

        logger.info(
            f"[deep_agent.tools.get_candlesticks] {event_ticker}: "
            f"{len(result_markets)} markets, period={period}, hours_back={hours_back}"
        )

        return {
            "event_ticker": event_ticker,
            "period": period,
            "hours_back": hours_back,
            "markets": result_markets,
        }

    # === Market Tools ===

    async def get_markets(
        self,
        event_ticker: Optional[str] = None,
        limit: int = 20,
    ) -> List[MarketInfo]:
        """
        Get current market data.

        Uses tracked_markets state as primary source (already synced from API).
        Falls back to direct API calls if needed.

        Args:
            event_ticker: Optional event to filter markets
            limit: Maximum number of markets to return

        Returns:
            List of MarketInfo objects with current prices and spreads
        """
        self._tool_calls["get_markets"] += 1
        logger.info(f"[deep_agent.tools.get_markets] event_ticker={event_ticker}, limit={limit}")

        markets = []

        # Primary source: Use tracked_markets state (already has live data)
        if self._tracked_markets:
            try:
                if event_ticker:
                    # Get markets for specific event
                    markets_by_event = self._tracked_markets.get_markets_by_event()
                    event_markets = markets_by_event.get(event_ticker, [])
                else:
                    # Get all tracked markets using public method
                    event_markets = self._tracked_markets.get_all()

                for m in event_markets[:limit]:
                    # T3.3: Propagate None instead of hardcoded 50c fallback
                    raw_bid = m.yes_bid if hasattr(m, 'yes_bid') else None
                    raw_ask = m.yes_ask if hasattr(m, 'yes_ask') else None

                    if raw_bid is not None and raw_ask is not None:
                        yes_bid = raw_bid
                        yes_ask = raw_ask
                        dq = "live"
                    elif hasattr(m, 'price') and m.price is not None:
                        yes_bid = m.price
                        yes_ask = m.price
                        dq = "estimated"
                    else:
                        yes_bid = 0
                        yes_ask = 0
                        dq = "unknown"

                    # T3.4: Convert status to string, default to "unknown" not "active"
                    status_val = m.status
                    if hasattr(status_val, 'value'):
                        status_val = status_val.value
                    elif status_val is None:
                        status_val = "unknown"

                    markets.append(MarketInfo(
                        ticker=m.ticker,
                        event_ticker=m.event_ticker,
                        title=m.yes_sub_title or m.title or "",
                        yes_bid=yes_bid,
                        yes_ask=yes_ask,
                        spread=yes_ask - yes_bid,
                        volume_24h=getattr(m, 'volume_24h', 0) or 0,
                        status=str(status_val),
                        last_trade_price=getattr(m, 'last_price', None),
                        data_quality=dq,
                    ))

                if markets:
                    logger.info(f"[deep_agent.tools.get_markets] Returned {len(markets)} markets from tracked_markets")
                    return markets
                else:
                    logger.info(f"[deep_agent.tools.get_markets] No markets in tracked_markets for event={event_ticker}, trying API")
                    # Fall through to API fallback

            except Exception as e:
                logger.warning(f"[deep_agent.tools.get_markets] Error reading tracked_markets: {e}")
                # Fall through to API fallback

        # Fallback: Direct API call via trading client (for untracked events/markets)
        if self._trading_client:
            try:
                raw_markets = []

                if event_ticker:
                    # Try to get event data from API (for markets not in tracked_markets)
                    logger.info(f"[deep_agent.tools.get_markets] Fetching event {event_ticker} from API")
                    event_data = await self._trading_client.get_event(event_ticker)
                    if event_data:
                        raw_markets = event_data.get("markets", [])[:limit]
                        logger.info(f"[deep_agent.tools.get_markets] Got {len(raw_markets)} markets from event API")

                # If no event_ticker or event lookup failed, try to get specific tickers
                # from recent price impact signals
                if not raw_markets and not event_ticker:
                    # Get recent signal tickers from extractions table
                    try:
                        supabase = self._get_supabase()

                        if supabase:
                            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=2)

                            # Get market_tickers arrays from recent extraction signals
                            result = supabase.table("extractions") \
                                .select("market_tickers") \
                                .eq("extraction_class", "market_signal") \
                                .gte("created_at", cutoff_time.isoformat()) \
                                .limit(50) \
                                .execute()

                            # Unnest market_tickers arrays to get unique tickers
                            signal_tickers = list(set(
                                ticker
                                for row in (result.data or [])
                                for ticker in (row.get("market_tickers") or [])
                                if ticker
                            ))

                            if signal_tickers:
                                logger.info(f"[deep_agent.tools.get_markets] Fetching {len(signal_tickers)} signal market tickers from API")
                                raw_markets = await self._trading_client.get_markets(signal_tickers)
                                logger.info(f"[deep_agent.tools.get_markets] Got {len(raw_markets)} markets from API for signal tickers")
                    except Exception as e:
                        logger.warning(f"[deep_agent.tools.get_markets] Could not fetch signal tickers: {e}")

                for m in raw_markets:
                    yes_bid = m.get("yes_bid", 0) or 0
                    yes_ask = m.get("yes_ask", 0) or 0

                    markets.append(MarketInfo(
                        ticker=m.get("ticker", ""),
                        event_ticker=m.get("event_ticker", ""),
                        title=m.get("yes_sub_title") or m.get("title", ""),
                        yes_bid=yes_bid,
                        yes_ask=yes_ask,
                        spread=yes_ask - yes_bid,
                        volume_24h=m.get("volume_24h", 0) or 0,
                        status=m.get("status", "unknown"),
                        last_trade_price=m.get("last_price"),
                    ))

                if markets:
                    logger.info(f"[deep_agent.tools.get_markets] Returned {len(markets)} markets from API")
                    return markets

            except Exception as e:
                logger.error(f"[deep_agent.tools.get_markets] API Error: {e}")

        logger.warning("[deep_agent.tools.get_markets] No market data source available")
        return []

    # === Event Context ===

    async def get_event_context(
        self,
        event_ticker: str,
    ) -> Optional[EventContextInfo]:
        """
        Get context about an event and its related markets.

        This tool helps understand mutual exclusivity relationships between
        markets in the same event. For example, in a presidential primary
        event, only ONE candidate can win - all markets are mutually exclusive.

        Args:
            event_ticker: The event ticker (e.g., "KXPRESNOMD")

        Returns:
            EventContextInfo with markets, risk levels, and positions, or None if not found
        """
        self._tool_calls["get_event_context"] += 1
        logger.info(f"[deep_agent.tools.get_event_context] event_ticker={event_ticker}")

        if not self._tracked_markets:
            logger.warning("[deep_agent.tools.get_event_context] No tracked_markets available")
            return None

        # Get markets grouped by event
        markets_by_event = self._tracked_markets.get_markets_by_event()
        event_markets = markets_by_event.get(event_ticker)

        if not event_markets:
            logger.warning(f"[deep_agent.tools.get_event_context] Event {event_ticker} not found")
            return None

        # Get positions from state container
        positions = {}
        if self._state_container:
            trading_state = self._state_container.trading_state
            if trading_state:
                positions = trading_state.positions or {}

        # Build market info list
        market_infos = []
        yes_sum = 0
        total_contracts = 0
        position_count = 0

        missing_price_count = 0
        for market in event_markets:
            ticker = market.ticker
            # T3.5: Use None instead of fabricated 50c
            yes_price = market.yes_ask if market.yes_ask > 0 else market.price
            if yes_price is None or yes_price <= 0:
                yes_price = None
                missing_price_count += 1

            no_price = (100 - yes_price) if yes_price is not None else None
            if yes_price is not None:
                yes_sum += yes_price

            # Check position
            position = positions.get(ticker, {})
            pos_contracts = position.get("position", 0)
            has_position = pos_contracts != 0

            if pos_contracts > 0:
                pos_side = "yes"
            elif pos_contracts < 0:
                pos_side = "no"
            else:
                pos_side = "none"

            if has_position:
                position_count += 1
                total_contracts += abs(pos_contracts)

            # Use yes_sub_title for display (actual outcome name)
            title = market.yes_sub_title or market.title

            market_infos.append(EventMarketInfo(
                ticker=ticker,
                title=title,
                yes_price=yes_price,
                no_price=no_price,
                has_position=has_position,
                position_side=pos_side,
                position_contracts=abs(pos_contracts),
            ))

        # Calculate aggregates
        n_markets = len(market_infos)
        priced_markets = n_markets - missing_price_count
        no_sum = priced_markets * 100 - yes_sum if priced_markets > 0 else 0

        # Get risk level from event position tracker if available
        risk_level = "NORMAL"
        if self._event_position_tracker:
            groups = self._event_position_tracker.get_event_groups()
            if event_ticker in groups:
                risk_level = groups[event_ticker].risk_level

        # Generate human-readable explanation
        mutual_exclusivity_note = self._generate_mutual_exclusivity_note(
            event_ticker, n_markets, yes_sum, risk_level, position_count
        )
        # T3.5: Note missing price data in the explanation
        if missing_price_count > 0:
            mutual_exclusivity_note += (
                f" NOTE: {missing_price_count}/{n_markets} markets have no price data. "
                f"YES sum is based on {priced_markets} priced markets only."
            )

        logger.info(
            f"[deep_agent.tools.get_event_context] {event_ticker}: {n_markets} markets, "
            f"YES_sum={yes_sum}c, risk={risk_level}, positions={position_count}"
        )

        return EventContextInfo(
            event_ticker=event_ticker,
            market_count=n_markets,
            markets=market_infos,
            yes_sum=yes_sum,
            no_sum=no_sum,
            risk_level=risk_level,
            has_positions=position_count > 0,
            position_count=position_count,
            total_contracts=total_contracts,
            mutual_exclusivity_note=mutual_exclusivity_note,
        )

    def _generate_mutual_exclusivity_note(
        self,
        event_ticker: str,
        n_markets: int,
        yes_sum: int,
        risk_level: str,
        position_count: int,
    ) -> str:
        """Generate a human-readable explanation of mutual exclusivity."""
        notes = []

        # Basic explanation
        notes.append(
            f"Event {event_ticker} contains {n_markets} mutually exclusive markets. "
            f"Only ONE market can resolve YES - all others resolve NO."
        )

        # YES sum interpretation
        if yes_sum > 100:
            notes.append(
                f"YES prices sum to {yes_sum}c (>${100}). "
                f"NO contracts are cheap - buying NO on all markets would profit."
            )
        elif yes_sum < 100:
            notes.append(
                f"YES prices sum to {yes_sum}c (<$100). "
                f"NO contracts are expensive - holding NO on all markets guarantees loss."
            )
        else:
            notes.append(f"YES prices sum to exactly {yes_sum}c (fair pricing).")

        # Risk level explanation
        risk_explanations = {
            "ARBITRAGE": "This is an ARBITRAGE opportunity - NO is underpriced.",
            "NORMAL": "Pricing is within normal range.",
            "HIGH_RISK": "WARNING: Holding multiple NO positions is risky at these prices.",
            "GUARANTEED_LOSS": "DANGER: Multiple NO positions will result in guaranteed loss!",
        }
        notes.append(risk_explanations.get(risk_level, ""))

        # Position advice
        if position_count > 1:
            notes.append(
                f"You have positions in {position_count} markets within this event. "
                f"These positions are CORRELATED - they all depend on the same outcome."
            )

        return " ".join(notes)

    # === Preflight Check ===

    async def preflight_check(
        self,
        ticker: str,
        side: Literal["yes", "no"],
        contracts: int,
        execution_strategy: Literal["aggressive", "moderate", "passive"] = "aggressive",
    ) -> Dict[str, Any]:
        """
        Bundled pre-trade check: market data + event context + safety checks.

        Replaces the need to call get_markets() + get_event_context() separately,
        saving 2 LLM round-trips per evaluated signal.

        Args:
            ticker: Market ticker to check
            side: Intended trade side ("yes" or "no")
            contracts: Intended number of contracts
            execution_strategy: Pricing strategy for limit price computation

        Returns:
            Dict with market data, event context, computed limit price,
            safety checks (spread, circuit breaker, exposure, risk, data quality),
            and a tradeable boolean.
        """
        self._tool_calls["preflight_check"] += 1
        logger.info(f"[deep_agent.tools.preflight_check] ticker={ticker}, side={side}, contracts={contracts}")

        result: Dict[str, Any] = {
            "ticker": ticker,
            "event_ticker": "",
            "title": "",
            "yes_bid": 0,
            "yes_ask": 0,
            "spread": 0,
            "intended_side": side,
            "estimated_limit_price": 0,
            "estimated_cost_cents": 0,
            "event_context": None,
            "checks": {
                "spread_ok": False,
                "circuit_breaker_ok": True,
                "event_exposure_ok": True,
                "risk_level_ok": True,
                "data_quality_ok": False,
                "all_clear": False,
            },
            "tradeable": False,
            "blockers": [],
            "warnings": [],
        }

        # Step 1: Get market data from tracked_markets
        market_data = None
        if self._tracked_markets:
            try:
                market_data = self._tracked_markets.get_market(ticker)
            except Exception as e:
                logger.warning(f"[deep_agent.tools.preflight_check] Error reading tracked_markets: {e}")

        if not market_data:
            # API fallback: fetch market directly and add to tracked markets
            if self._trading_client and self._tracked_markets:
                try:
                    api_market = await self._trading_client.get_market(ticker)
                    if api_market:
                        from ..state.tracked_markets import TrackedMarket, MarketStatus

                        open_ts, close_ts = 0, 0
                        try:
                            ot = api_market.get("open_time", "")
                            ct = api_market.get("close_time", "")
                            if ot and isinstance(ot, str):
                                open_ts = int(datetime.fromisoformat(ot.replace("Z", "+00:00")).timestamp())
                            if ct and isinstance(ct, str):
                                close_ts = int(datetime.fromisoformat(ct.replace("Z", "+00:00")).timestamp())
                        except (ValueError, TypeError):
                            pass

                        new_market = TrackedMarket(
                            ticker=ticker,
                            event_ticker=api_market.get("event_ticker", ""),
                            title=api_market.get("title", ""),
                            category=api_market.get("category", ""),
                            yes_sub_title=api_market.get("yes_sub_title", ""),
                            no_sub_title=api_market.get("no_sub_title", ""),
                            subtitle=api_market.get("subtitle", ""),
                            rules_primary=api_market.get("rules_primary", ""),
                            status=MarketStatus.ACTIVE,
                            open_ts=open_ts,
                            close_ts=close_ts,
                            tracked_at=time.time(),
                            market_info=api_market,
                            discovery_source="preflight_fallback",
                            volume=api_market.get("volume", 0),
                            volume_24h=api_market.get("volume_24h", 0),
                            open_interest=api_market.get("open_interest", 0),
                            yes_bid=api_market.get("yes_bid", 0) or 0,
                            yes_ask=api_market.get("yes_ask", 0) or 0,
                            price=api_market.get("last_price", 0) or 0,
                        )
                        await self._tracked_markets.add_market(new_market)
                        market_data = new_market
                        logger.info(f"[deep_agent.tools.preflight_check] API fallback: added {ticker} to tracked markets")
                except Exception as e:
                    logger.warning(f"[deep_agent.tools.preflight_check] API fallback failed for {ticker}: {e}")

        if not market_data:
            result["blockers"].append(f"Market {ticker} not found in tracked markets")
            return result

        # Extract market info
        raw_bid = market_data.yes_bid if hasattr(market_data, 'yes_bid') else None
        raw_ask = market_data.yes_ask if hasattr(market_data, 'yes_ask') else None

        if raw_bid is not None and raw_ask is not None and (raw_bid > 0 or raw_ask > 0):
            result["yes_bid"] = raw_bid
            result["yes_ask"] = raw_ask
            result["spread"] = raw_ask - raw_bid
            result["checks"]["data_quality_ok"] = True
        else:
            result["blockers"].append(f"No live bid/ask data (bid={raw_bid}, ask={raw_ask})")

        result["event_ticker"] = market_data.event_ticker or ""
        result["title"] = market_data.yes_sub_title or market_data.title or ""

        # Step 2: Compute estimated limit price
        if result["checks"]["data_quality_ok"]:
            limit_price = self._compute_limit_price(
                side, result["yes_bid"], result["yes_ask"], execution_strategy
            )
            result["estimated_limit_price"] = limit_price
            result["estimated_cost_cents"] = contracts * limit_price

        # Step 3: Spread check
        spread = result["spread"]
        max_spread = 12
        if result["checks"]["data_quality_ok"]:
            result["checks"]["spread_ok"] = spread <= max_spread
            if spread > max_spread:
                result["blockers"].append(f"Spread too wide: {spread}c (max {max_spread}c)")

        # Step 4: Circuit breaker check
        if self._circuit_breaker_checker:
            is_blacklisted, reason = self._circuit_breaker_checker(ticker)
            if is_blacklisted:
                result["checks"]["circuit_breaker_ok"] = False
                result["blockers"].append(reason or f"Circuit breaker triggered for {ticker}")

        # Step 5: Event exposure check
        event_ticker = result["event_ticker"]
        if event_ticker and result["estimated_cost_cents"] > 0:
            existing_cents = self._event_exposure_cents.get(event_ticker, 0)
            new_cost_cents = result["estimated_cost_cents"]
            total_after = existing_cents + new_cost_cents
            if total_after > self._max_event_exposure_cents:
                result["checks"]["event_exposure_ok"] = False
                max_dollars = self._max_event_exposure_cents / 100
                result["blockers"].append(
                    f"Would exceed ${max_dollars:.0f}/event cap. "
                    f"Existing: ${existing_cents/100:.2f}, "
                    f"New: ${new_cost_cents/100:.2f}, "
                    f"Total: ${total_after/100:.2f}"
                )

        # Step 6: Event context + risk level check
        if event_ticker:
            event_ctx = await self.get_event_context(event_ticker)
            if event_ctx:
                result["event_context"] = {
                    "market_count": event_ctx.market_count,
                    "yes_sum": event_ctx.yes_sum,
                    "risk_level": event_ctx.risk_level,
                    "position_count": event_ctx.position_count,
                    "existing_positions": [
                        {
                            "ticker": em.ticker,
                            "side": em.position_side,
                            "contracts": em.position_contracts,
                        }
                        for em in event_ctx.markets
                        if em.has_position
                    ],
                }

                # Risk level check
                risk = event_ctx.risk_level
                if risk in ("HIGH_RISK", "GUARANTEED_LOSS"):
                    result["checks"]["risk_level_ok"] = False
                    result["blockers"].append(f"Event risk level: {risk}")

                # Warnings for existing positions in same event
                for em in event_ctx.markets:
                    if em.has_position and em.ticker != ticker:
                        result["warnings"].append(
                            f"Existing position in same event: {em.ticker} {em.position_side.upper()} x{em.position_contracts}"
                        )

        # Step 7: Compute overall tradeable flag
        checks = result["checks"]
        checks["all_clear"] = all([
            checks["spread_ok"],
            checks["circuit_breaker_ok"],
            checks["event_exposure_ok"],
            checks["risk_level_ok"],
            checks["data_quality_ok"],
        ])
        result["tradeable"] = checks["all_clear"]

        logger.info(
            f"[deep_agent.tools.preflight_check] {ticker}: tradeable={result['tradeable']}, "
            f"blockers={len(result['blockers'])}, warnings={len(result['warnings'])}"
        )

        return result

    # === Trade Execution ===

    @staticmethod
    def _compute_limit_price(side: str, yes_bid: int, yes_ask: int, execution_strategy: str, action: str = "buy") -> int:
        """Compute limit price in cents given side, bid/ask, strategy, and action.

        For BUY orders: aggressive crosses the spread (pay more for immediate fill).
        For SELL orders: aggressive crosses the spread (accept less for immediate fill).

        NO-side price derivation: NO_bid = 100 - YES_ask, NO_ask = 100 - YES_bid.
        When yes_bid == 0, we use yes_ask as the NO-side anchor (NO_bid = 100 - yes_ask).
        When yes_ask == 0, we use yes_bid as the NO-side anchor (NO_ask = 100 - yes_bid).

        Returns a clamped price in [1, 99].
        """
        midpoint = (yes_bid + yes_ask) // 2 if (yes_bid > 0 and yes_ask > 0) else 0

        # Normalize unrecognized strategies to aggressive
        if execution_strategy not in ("aggressive", "moderate", "passive"):
            execution_strategy = "aggressive"

        # Pre-compute NO-side anchor prices
        no_bid = (100 - yes_ask) if yes_ask > 0 else 0   # What you'd get selling NO
        no_ask = (100 - yes_bid) if yes_bid > 0 else 0   # What you'd pay buying NO
        no_mid = (no_bid + no_ask) // 2 if (no_bid > 0 and no_ask > 0) else 0

        if action == "sell":
            # SELL: price is the minimum you'll accept
            # aggressive = sell at bid (immediate fill, lowest price)
            # passive = sell near ask (best price, may not fill)
            if execution_strategy == "aggressive":
                if side == "yes":
                    price = yes_bid if yes_bid > 0 else yes_ask
                else:
                    # Sell NO aggressive = sell at NO bid (100 - yes_ask)
                    price = no_bid if no_bid > 0 else no_ask
            elif execution_strategy == "moderate":
                if side == "yes":
                    price = midpoint if midpoint > 0 else yes_bid
                else:
                    price = no_mid if no_mid > 0 else (no_bid if no_bid > 0 else no_ask)
            else:  # passive
                if side == "yes":
                    price = yes_ask - 1 if yes_ask > 1 else (yes_bid if yes_bid > 0 else 1)
                else:
                    # Sell NO passive = near NO ask (best price, may not fill)
                    price = no_ask - 1 if no_ask > 1 else (no_bid if no_bid > 0 else 1)
        else:
            # BUY: price is the maximum you'll pay
            # aggressive = buy at ask (immediate fill, highest cost)
            # passive = buy near bid (cheapest, may not fill)
            if execution_strategy == "aggressive":
                if side == "yes":
                    price = yes_ask
                else:
                    # Buy NO aggressive = buy at NO ask (100 - yes_bid)
                    # Fallback to NO bid if no NO ask available
                    price = no_ask if no_ask > 0 else no_bid
            elif execution_strategy == "moderate":
                if side == "yes":
                    price = midpoint + 1 if midpoint > 0 else yes_ask
                else:
                    price = no_mid + 1 if no_mid > 0 else (no_ask if no_ask > 0 else no_bid)
            else:  # passive
                if side == "yes":
                    price = yes_bid + 1 if yes_bid > 0 else yes_ask
                else:
                    # Buy NO passive = near NO bid + 1 (cheapest)
                    price = no_bid + 1 if no_bid > 0 else (no_ask if no_ask > 0 else 1)

        return max(1, min(99, price))

    async def trade(
        self,
        ticker: str,
        side: Literal["yes", "no"],
        contracts: int,
        reasoning: str,
        execution_strategy: Literal["aggressive", "moderate", "passive"] = "aggressive",
        action: Literal["buy", "sell"] = "buy",
    ) -> TradeResult:
        """
        Execute a trade on the market.

        action="buy" (default): Open a new position by buying contracts.
        action="sell": Close an existing position by selling contracts you own.

        Places a limit order with pricing determined by execution_strategy:
        - aggressive: Cross the spread (buy at ask / sell at bid). Fastest fill.
        - moderate: Place at midpoint + 1c. Saves ~half the spread, decent fill probability.
        - passive: Place near the bid + 1c (buy) or ask - 1c (sell). May not fill.

        For buy orders, enforces per-event dollar exposure cap (_max_event_exposure_cents)
        to prevent concentration risk. Sell orders skip exposure checks since they
        reduce exposure.

        Args:
            ticker: Market ticker to trade
            side: "yes" or "no"
            contracts: Number of contracts (1-500)
            reasoning: Why you're making this trade (stored for reflection)
            execution_strategy: How aggressively to price the order (default: aggressive)
            action: "buy" to open position, "sell" to close existing position (default: buy)

        Returns:
            TradeResult with success status and fill details
        """
        self._tool_calls["trade"] += 1
        logger.info(f"[deep_agent.tools.trade] ticker={ticker}, side={side}, contracts={contracts}, action={action}")
        logger.info(f"[deep_agent.tools.trade] reasoning: {reasoning[:100]}...")

        # Validate inputs
        if contracts < 1 or contracts > 500:
            return TradeResult(
                success=False,
                ticker=ticker,
                side=side,
                contracts=contracts,
                error="Contracts must be between 1 and 500",
            )

        if side not in ("yes", "no"):
            return TradeResult(
                success=False,
                ticker=ticker,
                side=side,
                contracts=contracts,
                error="Side must be 'yes' or 'no'",
            )

        if not self._trading_client:
            return TradeResult(
                success=False,
                ticker=ticker,
                side=side,
                contracts=contracts,
                error="No trading client available",
            )

        try:
            # Step 1: Fetch current market data to get bid/ask prices
            market_data = await self._trading_client.get_market(ticker)
            if not market_data:
                return TradeResult(
                    success=False,
                    ticker=ticker,
                    side=side,
                    contracts=contracts,
                    error=f"Could not fetch market data for {ticker}",
                )

            # Step 2: Calculate limit price based on execution strategy
            # Kalshi markets have yes_bid/yes_ask. NO price = 100 - YES price.
            yes_ask = market_data.get("yes_ask")
            yes_bid = market_data.get("yes_bid")

            # T1.1: Validate bid/ask before computing execution price
            if yes_ask is None or yes_bid is None or (yes_ask == 0 and yes_bid == 0):
                return TradeResult(
                    success=False,
                    ticker=ticker,
                    side=side,
                    contracts=contracts,
                    error=f"Market data incomplete: yes_bid={yes_bid}, yes_ask={yes_ask}",
                )

            # T1.1b: For NO-side trades, require at least one valid anchor price
            # to avoid computing meaningless 1c limit prices
            if side == "no" and yes_bid == 0 and yes_ask == 0:
                return TradeResult(
                    success=False,
                    ticker=ticker,
                    side=side,
                    contracts=contracts,
                    error=f"No valid anchor for NO-side price: yes_bid={yes_bid}, yes_ask={yes_ask}",
                )

            # T1.3: Re-check spread with fresh API data (mirrors preflight check)
            # This closes the race window between preflight (cached prices) and trade (fresh prices)
            spread = yes_ask - yes_bid if (yes_ask > 0 and yes_bid > 0) else 0
            max_spread = 12
            if spread > max_spread:
                return TradeResult(
                    success=False,
                    ticker=ticker,
                    side=side,
                    contracts=contracts,
                    error=f"Spread too wide at trade time: {spread}c (max {max_spread}c, yes_bid={yes_bid}, yes_ask={yes_ask})",
                )

            limit_price = self._compute_limit_price(side, yes_bid, yes_ask, execution_strategy, action)

            # RF-1: Enforce per-event dollar exposure cap (buy orders only)
            # Sell orders reduce exposure, so skip the cap check
            # Use limit_price as estimate (actual fill may be better but we don't know yet)
            event_ticker = market_data.get("event_ticker", "")
            if event_ticker and action == "buy":
                new_cost_cents = contracts * limit_price  # Conservative: limit_price >= actual fill
                existing_cents = self._event_exposure_cents.get(event_ticker, 0)
                if existing_cents + new_cost_cents > self._max_event_exposure_cents:
                    max_dollars = self._max_event_exposure_cents / 100
                    return TradeResult(
                        success=False,
                        ticker=ticker,
                        side=side,
                        contracts=contracts,
                        error=(
                            f"Blocked: trade would exceed ${max_dollars:.0f}/event cap. "
                            f"Existing: ${existing_cents/100:.2f}, "
                            f"New: ${new_cost_cents/100:.2f}, "
                            f"Total: ${(existing_cents + new_cost_cents)/100:.2f}"
                        ),
                    )

            # Validate we have a valid price
            if limit_price <= 0 or limit_price >= 100:
                return TradeResult(
                    success=False,
                    ticker=ticker,
                    side=side,
                    contracts=contracts,
                    error=f"Invalid price calculated: {limit_price}c (yes_bid={yes_bid}, yes_ask={yes_ask})",
                )

            logger.info(
                f"[deep_agent.tools.trade] Using limit price {limit_price}c for {side} "
                f"(strategy={execution_strategy}, yes_bid={yes_bid}, yes_ask={yes_ask}, mid={(yes_bid + yes_ask) / 2})"
            )

            # Step 3: Place limit order through trading client
            # action="buy" opens a position, action="sell" closes an existing position
            # side determines whether we're trading YES or NO contracts
            result = await self._trading_client.place_order(
                ticker=ticker,
                action=action,  # "buy" to open, "sell" to close
                side=side,      # "yes" or "no"
                count=contracts,
                price=limit_price,  # Limit price in cents
                order_type="limit",  # All Kalshi orders are limit orders
            )

            # Check for success in response - place_order returns {"order": {...}}
            order = result.get("order")
            success = order is not None

            if success:
                # Log raw API response for debugging impossible price issues
                logger.debug(f"[deep_agent.tools.trade] Raw API order response: {order}")

                # Extract order details
                order_id = order.get("order_id", "")
                # For market orders, avg_price comes from fills.
                # Demo API may not return avg_price or price, so fall back to limit_price.
                raw_avg_price = order.get("avg_price") or order.get("price") or limit_price
                avg_price = self._clamp_binary_price(raw_avg_price, limit_price, f"order {order_id}")

                # Determine order status for fill tracking and broadcast
                order_status = order.get("status", "unknown")

                # Broadcast to WebSocket
                if self._ws_manager:
                    await self._ws_manager.broadcast_message("deep_agent_trade", {
                        "ticker": ticker,
                        "side": side,
                        "action": action,
                        "contracts": contracts,
                        "price_cents": avg_price,
                        "limit_price_cents": limit_price,
                        "order_id": order_id,
                        "order_status": order_status,
                        "reasoning": reasoning[:500],
                        "timestamp": time.strftime("%H:%M:%S"),
                    })

                logger.info(f"[deep_agent.tools.trade] Order placed successfully: {order_id} @ {avg_price}c")

                # RF-1: Record event exposure for dollar cap enforcement
                # Use actual fill price when available, fall back to limit price for resting orders
                # Buy orders increase exposure; sell orders release it
                if event_ticker:
                    fill_price = avg_price if avg_price and avg_price != limit_price else limit_price
                    if action == "buy":
                        buy_cost = contracts * fill_price
                        self._event_exposure_cents[event_ticker] = (
                            self._event_exposure_cents.get(event_ticker, 0) + buy_cost
                        )
                    else:  # sell
                        # Release exposure based on what was recorded at buy time (limit_price),
                        # not the current sell price, to keep the tracker balanced
                        sell_release = contracts * fill_price
                        self._event_exposure_cents[event_ticker] = max(
                            0, self._event_exposure_cents.get(event_ticker, 0) - sell_release
                        )
                    logger.info(
                        f"[deep_agent.tools.trade] Event {event_ticker} exposure: "
                        f"${self._event_exposure_cents[event_ticker]/100:.2f} / "
                        f"${self._max_event_exposure_cents/100:.0f} "
                        f"(action={action})"
                    )

                # Only record fill metrics when the order actually executed immediately.
                # Resting orders haven't filled yet — their fills will be detected
                # by _sync_trading_attachments() when the next REST sync runs.
                # Recording phantom fills for resting orders causes the agent's
                # recent_fills to disagree with actual positions, triggering E5 halts.
                is_immediate_fill = order_status in ("executed", "filled")

                if is_immediate_fill:
                    # Record order fill in session tracker for P&L metrics
                    if self._state_container:
                        order_cost_cents = contracts * limit_price
                        self._state_container.record_order_fill(order_cost_cents, contracts)
                        logger.info(f"[deep_agent.tools.trade] Session recorded: {contracts} contracts @ {limit_price}c = {order_cost_cents}c")

                    # Record fill in recent fills for session state visibility
                    self.record_fill(
                        order_id=order_id,
                        ticker=ticker,
                        side=side,
                        contracts=contracts,
                        price_cents=avg_price or limit_price,
                        reasoning=reasoning,
                    )
                else:
                    logger.info(
                        f"[deep_agent.tools.trade] Order {order_id[:8]}... is {order_status}, "
                        f"deferring fill recording until confirmed by sync"
                    )

                # Wire strategy_id into state container for per-strategy P&L tracking.
                # Use the actual order status from the API response so that
                # immediately-executed orders are tracked as "filled" from the start.
                if self._state_container:
                    try:
                        signal_id = f"deep_agent:{ticker}:{int(time.time() * 1000)}"
                        attachment_status = "filled" if is_immediate_fill else "resting"
                        now = time.time()
                        await self._state_container.update_order_in_attachment(
                            ticker=ticker,
                            order_id=order_id,
                            order_data=TrackedMarketOrder(
                                order_id=order_id,
                                signal_id=signal_id,
                                action=action,
                                side=side,
                                count=contracts,
                                price=limit_price,
                                status=attachment_status,
                                placed_at=now,
                                fill_count=contracts if is_immediate_fill else 0,
                                fill_avg_price=avg_price if is_immediate_fill else 0,
                                filled_at=now if is_immediate_fill else None,
                                strategy_id="deep_agent",
                            ),
                        )
                        logger.info(
                            f"[deep_agent.tools.trade] Order tracked in attachment: "
                            f"{ticker} {order_id[:8]}... status={attachment_status}, strategy_id=deep_agent"
                        )
                    except Exception as e:
                        logger.warning(f"[deep_agent.tools.trade] Failed to track order in attachment: {e}")

                return TradeResult(
                    success=True,
                    ticker=ticker,
                    side=side,
                    contracts=contracts,
                    price_cents=avg_price,
                    order_id=order_id,
                    limit_price_cents=limit_price,
                    order_status=order_status,
                )
            else:
                error_msg = result.get("error") or result.get("message") or "Unknown error"
                logger.warning(f"[deep_agent.tools.trade] Order failed: {error_msg}")
                return TradeResult(
                    success=False,
                    ticker=ticker,
                    side=side,
                    contracts=contracts,
                    error=error_msg,
                )

        except Exception as e:
            logger.error(f"[deep_agent.tools.trade] Error: {e}")
            return TradeResult(
                success=False,
                ticker=ticker,
                side=side,
                contracts=contracts,
                error=str(e),
            )

    # === Event Exposure Management ===

    def release_event_exposure(self, event_ticker: str, contracts: int, entry_price_cents: int) -> None:
        """Release event exposure when a position settles or is exited.

        Called from reflection handler when a PendingTrade settles. Without this,
        _event_exposure_cents only increases, progressively locking the agent out
        of trading on events it has already exited.
        """
        if not event_ticker or event_ticker not in self._event_exposure_cents:
            return

        release_amount = contracts * entry_price_cents
        previous = self._event_exposure_cents[event_ticker]
        self._event_exposure_cents[event_ticker] = max(0, previous - release_amount)

        logger.info(
            f"[deep_agent.tools] Released exposure for {event_ticker}: "
            f"${release_amount/100:.2f} freed, "
            f"${self._event_exposure_cents[event_ticker]/100:.2f} remaining"
        )

        # Clean up zero-exposure entries
        if self._event_exposure_cents[event_ticker] == 0:
            del self._event_exposure_cents[event_ticker]

    # === Fill Recording ===

    def record_fill(
        self,
        order_id: str,
        ticker: str,
        side: str,
        contracts: int,
        price_cents: int,
        reasoning: str = "",
    ) -> None:
        """
        Record a trade fill for session state tracking.

        Called automatically after successful trade execution.

        Args:
            order_id: The order ID from Kalshi
            ticker: Market ticker
            side: "yes" or "no"
            contracts: Number of contracts filled
            price_cents: Fill price in cents
            reasoning: Agent's reasoning for the trade
        """
        fill = FillRecord(
            order_id=order_id,
            ticker=ticker,
            side=side,
            contracts=contracts,
            price_cents=price_cents,
            timestamp=time.time(),
            reasoning=reasoning[:200] if reasoning else "",
        )
        self._recent_fills.appendleft(fill)
        self._traded_tickers.add(ticker)
        logger.debug(f"[deep_agent.tools] Recorded fill: {ticker} {side} {contracts}@{price_cents}c")

    def get_recent_fills(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent fills for session state.

        Args:
            limit: Maximum number of fills to return

        Returns:
            List of fill records as dicts
        """
        return [fill.to_dict() for fill in list(self._recent_fills)[:limit]]

    # === Price Validation Helpers ===

    @staticmethod
    def _clamp_binary_price(raw_price: int, fallback: int, context: str = "") -> int:
        """Validate a binary contract price is in the 1-99c range.

        The Kalshi demo API sometimes returns impossible prices (>100c) --
        likely centi-cent values or cumulative costs.  This helper detects
        those and falls back to the provided fallback price.
        """
        try:
            price = int(raw_price)
        except (TypeError, ValueError):
            price = 0
        if 1 <= price <= 99:
            return price
        logger.warning(
            "[deep_agent.tools] Impossible price %sc clamped to %sc (%s)",
            raw_price, fallback, context,
        )
        return int(fallback) if 1 <= int(fallback) <= 99 else 50

    # === Strategy Filtering Helpers ===

    def _filter_positions_by_strategy(self, positions_details: List[Dict]) -> List[Dict]:
        """Filter position dicts to deep_agent strategy, _traded_tickers fallback.

        Returns the raw position dicts (no transformation applied).
        """
        result = []
        for pos in positions_details:
            ticker = pos.get("ticker", "")
            if pos.get("strategy_id") == "deep_agent":
                result.append(pos)
            elif ticker in self._traded_tickers:
                result.append(pos)
        return result

    def _filter_settlements_by_strategy(self, settlements: List[Dict]) -> List[Dict]:
        """Filter settlements to deep_agent strategy, _traded_tickers fallback."""
        agent_settlements = [
            s for s in settlements
            if s.get("strategy_id") == "deep_agent"
        ]
        if not agent_settlements and self._traded_tickers:
            agent_settlements = [
                s for s in settlements
                if s.get("ticker") in self._traded_tickers
            ]
        return agent_settlements

    # === Session State ===

    async def get_session_state(self) -> SessionState:
        """
        Get current trading session state.

        Returns:
            SessionState with balance, positions, P&L, and recent activity
        """
        self._tool_calls["get_session_state"] += 1
        logger.info("[deep_agent.tools.get_session_state] Fetching session state")

        if not self._state_container:
            logger.warning("[deep_agent.tools.get_session_state] No state container available")
            return SessionState(
                balance_cents=0,
                portfolio_value_cents=0,
                realized_pnl_cents=0,
                unrealized_pnl_cents=0,
                total_pnl_cents=0,
                position_count=0,
                open_order_count=0,
                trade_count=0,
                win_rate=0.0,
                is_valid=False,
                error="No state container available",
            )

        try:
            summary = self._state_container.get_trading_summary()
            pnl = summary.get("pnl", {})

            # Get positions with details, filtered to deep_agent strategy.
            # Primary: strategy_id on TradingAttachment (survives restarts).
            # Fallback: _traded_tickers set (current session only).
            #
            # Bug 5 fix: _format_position_details() returns dicts with
            #   "position" (signed int, +YES/-NO) and "side" ("yes"/"no"),
            #   NOT "yes_contracts"/"no_contracts".  Use the correct fields.
            positions = []
            for pos in self._filter_positions_by_strategy(summary.get("positions_details", [])):
                ticker = pos.get("ticker")

                # Read side and absolute contract count from formatted position data
                pos_side = pos.get("side", "none")
                pos_contracts = abs(pos.get("position", 0))

                if pos_contracts == 0:
                    continue  # Skip empty positions

                # Compute per-contract avg price from total_cost
                total_cost = pos.get("total_cost", 0) or 0
                raw_avg = total_cost // pos_contracts if pos_contracts > 0 else 0
                avg_price = raw_avg if 1 <= raw_avg <= 99 else 0

                positions.append({
                    "ticker": ticker,
                    "side": pos_side,
                    "contracts": pos_contracts,
                    "avg_price": avg_price,
                    "current_value": pos.get("current_value"),
                    "unrealized_pnl": pos.get("unrealized_pnl"),
                })

            # Collect open orders from TradingAttachments (deep_agent only)
            open_orders = []
            now = time.time()
            if self._state_container:
                for ticker in set(list(self._traded_tickers) + [p["ticker"] for p in positions]):
                    attachment = self._state_container.get_trading_attachment(ticker)
                    if not attachment:
                        continue
                    for order in attachment.orders.values():
                        if (order.strategy_id == "deep_agent"
                                and order.status in ("resting", "pending", "partial")):
                            open_orders.append({
                                "order_id": order.order_id,
                                "ticker": ticker,
                                "side": order.side,
                                "action": order.action,
                                "contracts": order.count,
                                "price_cents": order.price,
                                "status": order.status,
                                "fill_count": order.fill_count,
                                "placed_at": order.placed_at,
                                "age_seconds": round(now - order.placed_at, 1),
                            })

            # Calculate win rate from settlements — filtered to deep_agent strategy
            agent_settlements = self._filter_settlements_by_strategy(summary.get("settlements", []))
            wins = sum(1 for s in agent_settlements if s.get("net_pnl", 0) > 0)
            total_settled = len(agent_settlements)
            win_rate = wins / total_settled if total_settled > 0 else 0.0
            agent_realized_pnl = sum(s.get("net_pnl", 0) for s in agent_settlements)

            # Compute unrealized P&L from agent's open positions
            agent_unrealized_pnl = sum(
                p.get("unrealized_pnl", 0) for p in positions
                if p.get("unrealized_pnl") is not None
            )

            # Bug 5 fix: balance and portfolio_value from get_trading_summary()
            # are already in cents (from TraderState).  Do NOT multiply by 100 again.
            return SessionState(
                balance_cents=int(summary.get("balance", 0)),
                portfolio_value_cents=int(summary.get("portfolio_value", 0)),
                realized_pnl_cents=int(agent_realized_pnl),
                unrealized_pnl_cents=int(agent_unrealized_pnl),
                total_pnl_cents=int(agent_realized_pnl + agent_unrealized_pnl),
                position_count=len(positions),  # Filtered count (this session only)
                open_order_count=len(open_orders),
                trade_count=total_settled,
                win_rate=win_rate,
                positions=positions,
                open_orders=open_orders,
                recent_fills=self.get_recent_fills(limit=10),
            )

        except Exception as e:
            logger.error(f"[deep_agent.tools.get_session_state] Error: {e}")
            return SessionState(
                balance_cents=0,
                portfolio_value_cents=0,
                realized_pnl_cents=0,
                unrealized_pnl_cents=0,
                total_pnl_cents=0,
                position_count=0,
                open_order_count=0,
                trade_count=0,
                win_rate=0.0,
                is_valid=False,
                error=str(e),
            )

    # === True Performance (Strategy-Filtered from Kalshi API) ===

    async def get_true_performance(self) -> TruePerformance:
        """
        Get TRUE trading performance from Kalshi API, filtered by strategy_id="deep_agent".

        Returns strategy-filtered P&L (realized + unrealized), positions, settlements,
        win rate, and per-event breakdown. This is the ground truth for the agent's
        trading performance — use it instead of the scorecard for P&L assessment.

        Unrealized P&L from open positions is included for immediate feedback
        before markets settle.

        Returns:
            TruePerformance dataclass with all strategy-filtered metrics
        """
        self._tool_calls["get_true_performance"] += 1
        logger.info("[deep_agent.tools.get_true_performance] Fetching true performance")

        if not self._state_container:
            return TruePerformance(
                balance_cents=0,
                portfolio_value_cents=0,
                is_valid=False,
                error="No state container available",
            )

        try:
            summary = self._state_container.get_trading_summary()

            # Account-level (not strategy-filtered)
            # Bug 5 fix: balance/portfolio_value are already in cents from TraderState.
            balance_cents = int(summary.get("balance", 0))
            portfolio_value_cents = int(summary.get("portfolio_value", 0))

            # --- Filter positions by strategy_id or _traded_tickers ---
            agent_positions = self._filter_positions_by_strategy(summary.get("positions_details", []))

            # Build clean position list with unrealized P&L
            # Bug 5 fix: _format_position_details() uses "position" (signed int)
            # and "side" ("yes"/"no"), not "yes_contracts"/"no_contracts".
            positions = []
            unrealized_pnl_cents = 0
            for pos in agent_positions:
                ticker = pos.get("ticker", "")
                pos_side = pos.get("side", "none")
                pos_contracts = abs(pos.get("position", 0))

                if pos_contracts == 0:
                    continue  # No position

                pos_unrealized = pos.get("unrealized_pnl", 0) or 0
                unrealized_pnl_cents += pos_unrealized

                # Compute per-contract avg price from total_cost
                total_cost = pos.get("total_cost", 0) or 0
                raw_avg = total_cost // pos_contracts if pos_contracts > 0 else 0
                avg_price = raw_avg if 1 <= raw_avg <= 99 else 0

                positions.append({
                    "ticker": ticker,
                    "event_ticker": pos.get("event_ticker", ""),
                    "side": pos_side,
                    "contracts": pos_contracts,
                    "avg_price": avg_price,
                    "current_value": pos.get("current_value"),
                    "unrealized_pnl": pos_unrealized,
                })

            # --- Filter settlements by strategy_id ---
            agent_settlements = self._filter_settlements_by_strategy(summary.get("settlements", []))

            # Compute realized P&L from settlements
            realized_pnl_cents = sum(s.get("net_pnl", 0) for s in agent_settlements)
            total_pnl_cents = realized_pnl_cents + unrealized_pnl_cents
            win_count = sum(1 for s in agent_settlements if s.get("net_pnl", 0) > 0)
            loss_count = sum(1 for s in agent_settlements if s.get("net_pnl", 0) < 0)
            trade_count = len(agent_settlements)
            win_rate = win_count / trade_count if trade_count > 0 else 0.0

            # Clean settlement data for return
            settlements = []
            for s in agent_settlements:
                settlements.append({
                    "ticker": s.get("ticker", ""),
                    "event_ticker": s.get("event_ticker", ""),
                    "side": s.get("side", ""),
                    "contracts": s.get("contracts", 0),
                    "net_pnl": s.get("net_pnl", 0),
                    "is_win": s.get("net_pnl", 0) > 0,
                    "settled_at": s.get("settled_at", ""),
                })

            # --- Per-event breakdown from settlements AND positions ---
            by_event: Dict[str, Dict[str, Any]] = {}

            # Settlements by event
            for s in agent_settlements:
                et = s.get("event_ticker", "unknown")
                if et not in by_event:
                    by_event[et] = {
                        "settled_trades": 0, "wins": 0, "losses": 0,
                        "realized_pnl": 0, "unrealized_pnl": 0,
                        "open_positions": 0,
                    }
                by_event[et]["settled_trades"] += 1
                pnl = s.get("net_pnl", 0)
                by_event[et]["realized_pnl"] += pnl
                if pnl > 0:
                    by_event[et]["wins"] += 1
                elif pnl < 0:
                    by_event[et]["losses"] += 1

            # Open positions by event (for unrealized P&L per event)
            for pos in positions:
                et = pos.get("event_ticker", "unknown")
                if et not in by_event:
                    by_event[et] = {
                        "settled_trades": 0, "wins": 0, "losses": 0,
                        "realized_pnl": 0, "unrealized_pnl": 0,
                        "open_positions": 0,
                    }
                by_event[et]["unrealized_pnl"] += pos.get("unrealized_pnl", 0)
                by_event[et]["open_positions"] += 1

            logger.info(
                f"[deep_agent.tools.get_true_performance] "
                f"positions={len(positions)}, settlements={trade_count}, "
                f"realized=${realized_pnl_cents/100:.2f}, unrealized=${unrealized_pnl_cents/100:.2f}, "
                f"total=${total_pnl_cents/100:.2f}, win_rate={win_rate:.0%}"
            )

            return TruePerformance(
                balance_cents=balance_cents,
                portfolio_value_cents=portfolio_value_cents,
                positions=positions,
                position_count=len(positions),
                unrealized_pnl_cents=unrealized_pnl_cents,
                settlements=settlements,
                realized_pnl_cents=realized_pnl_cents,
                total_pnl_cents=total_pnl_cents,
                trade_count=trade_count,
                win_count=win_count,
                loss_count=loss_count,
                win_rate=win_rate,
                by_event=by_event,
            )

        except Exception as e:
            logger.error(f"[deep_agent.tools.get_true_performance] Error: {e}")
            return TruePerformance(
                balance_cents=0,
                portfolio_value_cents=0,
                is_valid=False,
                error=str(e),
            )

    # === Memory Tools ===

    async def read_memory(self, filename: str) -> str:
        """
        Read a memory file.

        Args:
            filename: Name of memory file (e.g., "learnings.md", "strategy.md")

        Returns:
            Contents of the file, or empty string if not found
        """
        self._tool_calls["read_memory"] += 1

        # Sanitize filename to prevent path traversal
        safe_name = Path(filename).name
        file_path = self._memory_dir / safe_name

        logger.info(f"[deep_agent.tools.read_memory] Reading {safe_name}")

        try:
            if file_path.exists():
                content = file_path.read_text(encoding="utf-8")
                logger.info(f"[deep_agent.tools.read_memory] Read {len(content)} chars from {safe_name}")
                return content
            else:
                logger.warning(f"[deep_agent.tools.read_memory] File not found: {safe_name}")
                return ""
        except Exception as e:
            logger.error(f"[deep_agent.tools.read_memory] Error reading {safe_name}: {e}")
            # T4.6: Return error message so agent knows memory is unavailable
            return f"[ERROR: Could not read {safe_name}: {e}]"

    # Files that must NOT be fully replaced (use append_memory instead)
    _APPEND_ONLY_FILES = {"learnings.md", "mistakes.md", "patterns.md", "golden_rules.md", "cycle_journal.md"}

    async def _broadcast_memory_update(self, filename: str, content: str) -> None:
        """Broadcast memory file update to WebSocket clients."""
        if self._ws_manager:
            await self._ws_manager.broadcast_message("deep_agent_memory_update", {
                "filename": filename,
                "content_preview": content[:200] + "..." if len(content) > 200 else content,
                "timestamp": time.strftime("%H:%M:%S"),
            })

    # Strategy cap: hard limit on strategy.md to prevent bloat
    _STRATEGY_MAX_CHARS = 3000

    async def write_memory(self, filename: str, content: str) -> bool:
        """
        Write to a memory file (full replacement).

        Only strategy.md should be fully replaced. For learnings.md, mistakes.md,
        and patterns.md, use append_memory() to avoid accidental data loss.

        For strategy.md: enforces 3000 char cap and maintains a version counter.

        Args:
            filename: Name of memory file (e.g., "strategy.md")
            content: Content to write

        Returns:
            True if successful, False otherwise
        """
        self._tool_calls["write_memory"] += 1

        # Sanitize filename
        safe_name = Path(filename).name
        file_path = self._memory_dir / safe_name

        # Guard: prevent full replacement of append-only memory files
        if safe_name in self._APPEND_ONLY_FILES:
            logger.warning(
                f"[deep_agent.tools.write_memory] BLOCKED full replace of {safe_name}. "
                f"Use append_memory() instead to avoid data loss."
            )
            return False

        # Strategy-specific: version counter + cap enforcement
        if safe_name == "strategy.md":
            content = self._apply_strategy_cap(file_path, content)

        logger.info(f"[deep_agent.tools.write_memory] Writing {len(content)} chars to {safe_name}")

        # Log strategy evolution for auditability
        if safe_name == "strategy.md" and file_path.exists():
            try:
                old = file_path.read_text(encoding="utf-8")
                old_lines = len(old.splitlines())
                new_lines = len(content.splitlines())
                logger.info(
                    f"[deep_agent.tools.write_memory] Strategy evolution: "
                    f"{old_lines} -> {new_lines} lines, {len(old)} -> {len(content)} chars"
                )
            except Exception:
                pass

        try:
            file_path.write_text(content, encoding="utf-8")
            await self._broadcast_memory_update(safe_name, content)

            logger.info(f"[deep_agent.tools.write_memory] Successfully wrote {safe_name}")
            return True

        except Exception as e:
            logger.error(f"[deep_agent.tools.write_memory] Error writing {safe_name}: {e}")
            return False

    def _apply_strategy_cap(self, file_path: Path, content: str) -> str:
        """Apply version counter and size cap to strategy.md content."""
        # Extract and increment version counter
        version = 1
        if file_path.exists():
            try:
                old_content = file_path.read_text(encoding="utf-8")
                match = re.search(r'<!-- version:(\d+)', old_content)
                if match:
                    version = int(match.group(1)) + 1
            except Exception:
                pass

        # Strip any existing version header from new content
        content = re.sub(r'<!-- version:\d+ updated:\S+ -->\n?', '', content).lstrip()

        # Add version header
        timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M")
        header = f"<!-- version:{version} updated:{timestamp} -->\n"

        content = header + content

        # Enforce cap: truncate at last complete ## section boundary
        if len(content) > self._STRATEGY_MAX_CHARS:
            truncated = content[:self._STRATEGY_MAX_CHARS]
            # Find the last ## section header and truncate there
            last_section = truncated.rfind("\n## ")
            if last_section > len(header) + 100:
                content = truncated[:last_section].rstrip() + "\n"
            else:
                # Fallback: truncate at last newline
                last_nl = truncated.rfind("\n")
                content = truncated[:last_nl].rstrip() + "\n" if last_nl > 0 else truncated

            logger.warning(
                f"[deep_agent.tools.write_memory] Strategy capped at {len(content)} chars "
                f"(limit: {self._STRATEGY_MAX_CHARS})"
            )

        return content

    async def append_memory(self, filename: str, content: str) -> dict:
        """
        Append to a memory file with automatic size management.

        If the file exceeds 10,000 characters after appending, the oldest
        half of the content is archived to memory_archive/{date}/ to prevent
        unbounded growth while preserving history.

        Args:
            filename: Name of memory file
            content: Content to append

        Returns:
            Dict with success status and metadata
        """
        self._tool_calls["append_memory"] += 1

        safe_name = Path(filename).name
        existing = await self.read_memory(safe_name)
        new_content = existing + "\n" + content if existing else content

        # Size guard: archive oldest half if over 10,000 chars
        archived = False
        archive_path = ""
        if len(new_content) > 10000:
            midpoint = len(new_content) // 2
            # Find the nearest newline after midpoint to avoid cutting mid-line
            split_point = new_content.find("\n", midpoint)
            if split_point == -1:
                split_point = midpoint

            archived_content = new_content[:split_point]
            new_content = new_content[split_point:].lstrip("\n")

            # Archive the older half
            archive_dir = self._memory_dir / "memory_archive" / datetime.now().strftime("%Y-%m-%d")
            archive_dir.mkdir(parents=True, exist_ok=True)
            archive_file = archive_dir / f"{safe_name}.{datetime.now().strftime('%H%M%S')}"
            archive_file.write_text(archived_content, encoding="utf-8")
            archive_path = str(archive_file)
            archived = True

            logger.info(
                f"[deep_agent.tools.append_memory] Archived {len(archived_content)} chars "
                f"from {safe_name} to {archive_path}"
            )

        # Write directly instead of calling write_memory(), which blocks
        # append-only files (learnings.md, mistakes.md, patterns.md).
        file_path = self._memory_dir / safe_name
        try:
            file_path.write_text(new_content, encoding="utf-8")
            await self._broadcast_memory_update(safe_name, new_content)

            logger.info(f"[deep_agent.tools.append_memory] Successfully wrote {len(new_content)} chars to {safe_name}")
            success = True

            # Dual-write to vector store (fire-and-forget, non-blocking)
            if self._vector_memory and safe_name in self._get_vector_memory_types():
                asyncio.create_task(self._safe_vector_store(content, safe_name))
        except Exception as e:
            logger.error(f"[deep_agent.tools.append_memory] Error writing {safe_name}: {e}")
            success = False

        return {
            "success": success,
            "new_length": len(new_content),
            "archived": archived,
            "archive_path": archive_path if archived else None,
        }

    # === Vector Memory Helpers ===

    @staticmethod
    def _get_vector_memory_types():
        """Return the set of filenames that should be dual-written to vector store."""
        from .vector_memory import VectorMemoryService
        return VectorMemoryService.FILENAME_TO_TYPE

    async def _safe_vector_store(self, content: str, filename: str) -> None:
        """Fire-and-forget vector store write. Never raises."""
        try:
            from .vector_memory import VectorMemoryService
            memory_type = VectorMemoryService.FILENAME_TO_TYPE.get(filename)
            if memory_type and self._vector_memory:
                await self._vector_memory.store(content, memory_type, {
                    "source_file": filename,
                })
        except Exception as e:
            logger.debug(f"[tools] Vector store write failed (non-fatal): {e}")

    async def search_memory(
        self, query: str, types: Optional[list] = None, limit: int = 8
    ) -> str:
        """Semantic search across all historical vector memories."""
        self._tool_calls["search_memory"] += 1
        if not self._vector_memory:
            return "Vector memory not available. Use read_memory for flat-file access."
        try:
            results = await self._vector_memory.recall(
                query=query, types=types, limit=limit
            )
            if not results:
                return "No matching memories found."
            lines = []
            for r in results:
                age_marker = ""
                if r.get("access_count", 0) >= 5:
                    age_marker = f" [recalled {r['access_count']}x]"
                lines.append(
                    f"- [{r['memory_type']}]{age_marker} {r['content'][:400]}"
                )
            return f"Found {len(results)} memories:\n" + "\n".join(lines)
        except Exception as e:
            return f"Search failed: {e}. Use read_memory for flat-file access."

    # === Trade Intent Submission ===

    async def submit_trade_intent(
        self,
        market_ticker: str,
        side: str,
        contracts: int,
        thesis: str,
        confidence: str = "medium",
        exit_criteria: str = "",
        max_price_cents: int = 99,
        execution_strategy: str = "aggressive",
        action: str = "buy",
    ) -> Dict[str, Any]:
        """Submit a trade intent to the executor for execution.

        The deep agent calls this instead of trade() directly. The executor
        handles preflight, pricing, and fill verification.
        """
        self._tool_calls["submit_trade_intent"] += 1

        if not self._trade_executor:
            return {
                "error": "Trade executor not available",
                "status": "failed",
            }

        # Resolve event_ticker from tracked markets
        event_ticker = ""
        if self._tracked_markets:
            markets_by_event = self._tracked_markets.get_markets_by_event()
            for evt, tickers in markets_by_event.items():
                if market_ticker in tickers:
                    event_ticker = evt
                    break

        from .trade_executor import TradeIntent
        import uuid as _uuid

        intent = TradeIntent(
            intent_id=str(_uuid.uuid4())[:8],
            event_ticker=event_ticker,
            market_ticker=market_ticker,
            side=side,
            action=action,
            contracts=contracts,
            max_price_cents=max_price_cents,
            thesis=thesis,
            exit_criteria=exit_criteria or f"Hold until settlement or thesis invalidated",
            confidence=confidence,
            execution_strategy=execution_strategy,
            created_at=time.time(),
            status="pending",
        )

        await self._trade_executor.submit_intent(intent)

        # Embed thesis in vector memory for future recall
        if self._vector_memory and thesis:
            try:
                summary = (
                    f"Trade thesis for {market_ticker} ({side} {action}): {thesis[:300]}"
                )
                asyncio.create_task(
                    self._vector_memory.store_signal_summary(
                        event_ticker, summary, signal_type="thesis"
                    )
                )
            except Exception as e:
                logger.debug(f"[tools] Thesis embed failed (non-fatal): {e}")

        return {
            "intent_id": intent.intent_id,
            "market_ticker": market_ticker,
            "side": side,
            "action": action,
            "contracts": contracts,
            "max_price_cents": max_price_cents,
            "confidence": confidence,
            "execution_strategy": execution_strategy,
            "status": "submitted",
            "note": "Intent queued for executor. It will run preflight, check price, and execute within 30s.",
        }

    # === TODO Task Planning Tools ===

    TODOS_FILENAME = "todos.json"

    async def read_todos(self) -> Dict[str, Any]:
        """
        Read the current TODO task list.

        Returns:
            Dict with 'items' (list of todo items) and metadata.
            Each item has: task, priority (high/medium/low), status (pending/done),
            created_cycle, completed_cycle (if done).
        """
        self._tool_calls["read_todos"] += 1
        todos_path = self._memory_dir / self.TODOS_FILENAME

        if not todos_path.exists():
            return {"items": [], "total": 0, "pending": 0, "done": 0}

        try:
            data = json.loads(todos_path.read_text(encoding="utf-8"))
            items = data.get("items", [])
            pending = [i for i in items if i.get("status") != "done"]
            done = [i for i in items if i.get("status") == "done"]
            return {
                "items": items,
                "total": len(items),
                "pending": len(pending),
                "done": len(done),
            }
        except Exception as e:
            logger.error(f"[deep_agent.tools.read_todos] Error reading todos: {e}")
            return {"items": [], "total": 0, "pending": 0, "done": 0, "error": str(e)}

    async def write_todos(
        self,
        items: List[Dict[str, Any]],
        current_cycle: int = 0,
    ) -> Dict[str, Any]:
        """
        Write the TODO task list (full replace).

        Each item should have:
        - task (str): Description of the task
        - priority (str): "high", "medium", or "low"
        - status (str): "pending" or "done"

        Items marked "done" are auto-expired after 10 cycles from completion.

        Args:
            items: List of todo items
            current_cycle: Current cycle number (for expiry tracking)

        Returns:
            Dict with success status and metadata
        """
        self._tool_calls["write_todos"] += 1
        todos_path = self._memory_dir / self.TODOS_FILENAME

        # Normalize items
        normalized = []
        for item in items:
            entry = {
                "task": item.get("task", ""),
                "priority": item.get("priority", "medium"),
                "status": item.get("status", "pending"),
                "created_cycle": item.get("created_cycle", current_cycle),
            }
            if entry["status"] == "done":
                entry["completed_cycle"] = item.get("completed_cycle", current_cycle)
            normalized.append(entry)

        # Auto-expire: remove done items older than 10 cycles
        if current_cycle > 0:
            normalized = [
                item for item in normalized
                if item.get("status") != "done"
                or (current_cycle - item.get("completed_cycle", current_cycle)) < 10
            ]

        data = {
            "items": normalized,
            "last_updated_cycle": current_cycle,
            "updated_at": datetime.now().isoformat(),
        }

        try:
            todos_path.write_text(
                json.dumps(data, indent=2, default=str),
                encoding="utf-8",
            )
            pending = len([i for i in normalized if i.get("status") != "done"])
            done = len([i for i in normalized if i.get("status") == "done"])
            logger.info(
                f"[deep_agent.tools.write_todos] Wrote {len(normalized)} todos "
                f"({pending} pending, {done} done)"
            )
            return {
                "success": True,
                "total": len(normalized),
                "pending": pending,
                "done": done,
            }
        except Exception as e:
            logger.error(f"[deep_agent.tools.write_todos] Error writing todos: {e}")
            return {"success": False, "error": str(e)}

    # === Understand Event Tool ===

    async def understand_event(
        self,
        event_ticker: str,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """
        Build or refresh understanding of a Kalshi event.

        Produces the langextract specification (prompt, examples, extraction classes)
        for this event. Results stored in event_configs table and automatically used
        by the extraction pipeline.

        Idempotent: if event_configs row exists and last_researched_at < 24h ago
        and force_refresh=False, returns cached config.

        Args:
            event_ticker: The event ticker to research (e.g., "KXBONDIOUT")
            force_refresh: Force re-research even if recently researched

        Returns:
            Dict with event understanding and langextract spec
        """
        self._tool_calls["understand_event"] += 1
        logger.info(f"[deep_agent.tools.understand_event] event={event_ticker}, force={force_refresh}")

        # Fast-path: return generic response for events that previously failed
        # (e.g., KXQUICKSETTLE synthetic markets that always produce empty JSON)
        if not force_refresh and event_ticker in self._failed_event_tickers:
            logger.debug(f"[deep_agent.tools.understand_event] Skipping {event_ticker} (cached failure)")
            return {
                "status": "cached_failure",
                "event_ticker": event_ticker,
                "event_title": f"Synthetic quick-settle market ({event_ticker})",
                "description": "This is a synthetic quick-settle market used for rapid settlement testing. No meaningful research is available.",
                "primary_entity": "",
                "key_drivers": [],
                "markets_count": 0,
                "extraction_classes_count": 0,
                "watchlist": {},
            }

        supabase = self._get_supabase()
        if not supabase:
            return {"error": "Supabase not available", "event_ticker": event_ticker}

        # Check cache first (unless force_refresh)
        if not force_refresh:
            try:
                cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
                result = supabase.table("event_configs") \
                    .select("*") \
                    .eq("event_ticker", event_ticker) \
                    .gte("last_researched_at", cutoff.isoformat()) \
                    .execute()

                if result.data:
                    cached = result.data[0]
                    logger.info(
                        f"[deep_agent.tools.understand_event] Using cached config for {event_ticker} "
                        f"(version {cached.get('research_version', 1)})"
                    )
                    return {
                        "status": "cached",
                        "event_ticker": event_ticker,
                        "event_title": cached.get("event_title", ""),
                        "description": cached.get("description", ""),
                        "primary_entity": cached.get("primary_entity", ""),
                        "key_drivers": cached.get("key_drivers", []),
                        "markets_count": len(cached.get("markets", [])),
                        "research_version": cached.get("research_version", 1),
                        "extraction_classes_count": len(cached.get("extraction_classes", [])),
                        "watchlist": cached.get("watchlist", {}),
                    }
            except Exception as e:
                logger.warning(f"[deep_agent.tools.understand_event] Cache check failed: {e}")

        # Fetch event markets from tracked_markets
        markets_data = []
        if self._tracked_markets:
            markets_by_event = self._tracked_markets.get_markets_by_event()
            event_markets = markets_by_event.get(event_ticker, [])
            for m in event_markets:
                markets_data.append({
                    "ticker": m.ticker,
                    "yes_sub_title": m.yes_sub_title or m.title or "",
                    "type": getattr(m, "market_type", "UNKNOWN"),
                    "yes_price": m.yes_ask if hasattr(m, "yes_ask") and m.yes_ask else None,
                })

        if not markets_data and self._trading_client:
            # API fallback: fetch event directly from Kalshi API
            try:
                event_data = await self._trading_client.get_event(event_ticker)
                if event_data:
                    api_markets = event_data.get("markets", [])
                    for m in api_markets:
                        m_ticker = m.get("ticker", "")
                        if not m_ticker:
                            continue
                        markets_data.append({
                            "ticker": m_ticker,
                            "yes_sub_title": m.get("yes_sub_title", "") or m.get("title", ""),
                            "type": m.get("market_type", "UNKNOWN"),
                            "yes_price": m.get("yes_ask") or m.get("last_price"),
                        })
                    if markets_data:
                        logger.info(
                            f"[deep_agent.tools.understand_event] API fallback: "
                            f"found {len(markets_data)} markets for {event_ticker}"
                        )
            except Exception as e:
                logger.warning(f"[deep_agent.tools.understand_event] API fallback failed for {event_ticker}: {e}")

        if not markets_data:
            return {
                "error": f"No markets found for event {event_ticker}",
                "event_ticker": event_ticker,
            }

        # Build event title from first market
        event_title = markets_data[0].get("yes_sub_title", event_ticker)

        # Single Claude call for structured understanding
        try:
            from anthropic import AsyncAnthropic

            client = AsyncAnthropic()
            market_list = "\n".join(
                f"  - {m['ticker']}: {m['yes_sub_title']} (type={m['type']}, price={m.get('yes_price', '?')}c)"
                for m in markets_data
            )

            prompt = f"""Analyze this Kalshi prediction market event and produce a structured understanding.

EVENT: {event_ticker}
MARKETS:
{market_list}

Return a JSON object with these fields:
{{
  "event_title": "Short title for this event",
  "primary_entity": "Main person/org this event is about",
  "primary_entity_type": "person | org | policy | event",
  "description": "2-3 sentence description of what this event is about",
  "key_drivers": ["List of 3-5 factors that determine outcomes"],
  "outcome_descriptions": {{"yes_sub_title": "What YES means for each market"}},
  "prompt_description": "Extraction instructions specific to this event (what to look for in news/social media that would impact these markets)",
  "extraction_classes": [
    {{
      "class_name": "event-specific extraction class name",
      "description": "When to extract this class",
      "attributes": {{"attr_name": "description of attribute"}}
    }}
  ],
  "watchlist": {{
    "entities": ["Names to watch for"],
    "keywords": ["Keywords that indicate relevant news"],
    "aliases": {{"canonical_name": ["alias1", "alias2"]}}
  }},
  "example_input": "A realistic example Reddit post title about this event",
  "example_output": [
    {{
      "extraction_class": "market_signal",
      "extraction_text": "relevant quote from the example",
      "attributes": {{
        "market_ticker": "{markets_data[0]['ticker']}",
        "direction": "bullish",
        "magnitude": 50,
        "confidence": "medium",
        "reasoning": "Why this impacts the market"
      }}
    }}
  ]
}}

Be specific to THIS event. The extraction instructions should help an LLM identify
relevant information in Reddit posts and news articles that impact these specific markets."""

            response = await client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=2000,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}],
            )

            # Track token usage via callback
            if self._token_usage_callback and hasattr(response, 'usage') and response.usage:
                u = response.usage
                self._token_usage_callback(
                    input_tokens=getattr(u, 'input_tokens', 0) or 0,
                    output_tokens=getattr(u, 'output_tokens', 0) or 0,
                    cache_read=getattr(u, 'cache_read_input_tokens', 0) or 0,
                    cache_created=getattr(u, 'cache_creation_input_tokens', 0) or 0,
                )

            # Parse response
            content = response.content[0].text.strip()
            # Strip markdown code blocks if present
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(l for l in lines if not l.strip().startswith("```"))

            # Guard against empty response body
            if not content:
                logger.warning(f"[deep_agent.tools.understand_event] Empty response from Claude for {event_ticker}")
                self._failed_event_tickers.add(event_ticker)
                return {"error": "Empty response from research model", "event_ticker": event_ticker}

            research = json.loads(content)

        except Exception as e:
            logger.error(f"[deep_agent.tools.understand_event] Research failed: {e}")
            self._failed_event_tickers.add(event_ticker)
            return {"error": f"Research failed: {str(e)}", "event_ticker": event_ticker}

        # Build examples JSON from research
        examples_json = []
        if research.get("example_input") and research.get("example_output"):
            examples_json.append({
                "input": research["example_input"],
                "output": research["example_output"],
            })

        # UPSERT into event_configs
        try:
            upsert_data = {
                "event_ticker": event_ticker,
                "event_title": research.get("event_title", event_title),
                "primary_entity": research.get("primary_entity", ""),
                "primary_entity_type": research.get("primary_entity_type", ""),
                "description": research.get("description", ""),
                "key_drivers": research.get("key_drivers", []),
                "outcome_descriptions": research.get("outcome_descriptions", {}),
                "prompt_description": research.get("prompt_description", ""),
                "extraction_classes": research.get("extraction_classes", []),
                "examples": examples_json,
                "watchlist": research.get("watchlist", {}),
                "markets": markets_data,
                "is_active": True,
                "last_researched_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
            }

            # Get current version for increment
            existing = supabase.table("event_configs") \
                .select("research_version") \
                .eq("event_ticker", event_ticker) \
                .execute()

            current_version = 0
            if existing.data:
                current_version = existing.data[0].get("research_version", 0)
            upsert_data["research_version"] = current_version + 1

            supabase.table("event_configs").upsert(
                upsert_data, on_conflict="event_ticker"
            ).execute()

            logger.info(
                f"[deep_agent.tools.understand_event] Stored event config for {event_ticker} "
                f"(version {upsert_data['research_version']}, "
                f"{len(research.get('extraction_classes', []))} custom classes)"
            )

        except Exception as e:
            logger.error(f"[deep_agent.tools.understand_event] Upsert failed: {e}")
            return {"error": f"Storage failed: {str(e)}", "event_ticker": event_ticker}

        # Broadcast to frontend
        if self._ws_manager:
            await self._ws_manager.broadcast_message("event_understood", {
                "event_ticker": event_ticker,
                "event_title": research.get("event_title", ""),
                "primary_entity": research.get("primary_entity", ""),
                "markets_count": len(markets_data),
                "extraction_classes": len(research.get("extraction_classes", [])),
                "research_version": upsert_data["research_version"],
                "timestamp": time.strftime("%H:%M:%S"),
            })

        return {
            "status": "researched",
            "event_ticker": event_ticker,
            "event_title": research.get("event_title", ""),
            "description": research.get("description", ""),
            "primary_entity": research.get("primary_entity", ""),
            "key_drivers": research.get("key_drivers", []),
            "extraction_classes_count": len(research.get("extraction_classes", [])),
            "watchlist_entities": len(research.get("watchlist", {}).get("entities", [])),
            "markets_count": len(markets_data),
            "research_version": upsert_data["research_version"],
        }

    # === Extraction Signal Tools ===

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        """Safely convert a value to float, returning default on failure.

        Handles None, non-numeric strings (e.g. JSONPath templates like
        '@/model_response.magnitude'), and other unparseable values that
        may arrive from JSONB attribute fields.
        """
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    def _process_extraction_row(self, row: Dict, market_signals: Dict[str, Dict]) -> None:
        """Process a single SQL RPC row into the per-market signals accumulator."""
        ticker = row.get("market_ticker", "")
        if not ticker:
            return

        extraction_class = row.get("extraction_class", "")
        direction = row.get("direction", "")
        occ_count = int(row.get("occurrence_count", 0) or 0)
        unique_src = int(row.get("unique_sources", 0) or 0)
        avg_eng = self._safe_float(row.get("avg_engagement"))
        max_eng = int(row.get("max_engagement", 0) or 0)
        total_comments = int(row.get("total_comments", 0) or 0)
        avg_mag = self._safe_float(row.get("avg_magnitude"))
        avg_sent = self._safe_float(row.get("avg_sentiment"))
        last_seen = row.get("last_seen_at", "")
        first_seen = row.get("first_seen_at", "")
        oldest_source = row.get("oldest_source_at", "")
        newest_source = row.get("newest_source_at", "")

        if ticker not in market_signals:
            market_signals[ticker] = {
                "market_ticker": ticker,
                "event_tickers": [],
                "occurrence_count": 0,
                "unique_source_count": 0,
                "total_engagement": 0,
                "max_engagement": 0,
                "total_comments": 0,
                "directions": {"bullish": 0, "bearish": 0},
                "avg_magnitude": 0,
                "_magnitude_sum": 0.0,
                "_magnitude_count": 0,
                "last_seen_at": None,
                "first_seen_at": None,
                "oldest_source_at": None,
                "newest_source_at": None,
                "recent_extractions": [],
                "entity_mentions": [],
                "context_factors": [],
                "_entity_agg": {},
                "_context_agg": {},
            }

        sig = market_signals[ticker]

        # Update last_seen / first_seen (extraction save time)
        if last_seen and (not sig["last_seen_at"] or last_seen > sig["last_seen_at"]):
            sig["last_seen_at"] = last_seen
        if first_seen and (not sig["first_seen_at"] or first_seen < sig["first_seen_at"]):
            sig["first_seen_at"] = first_seen

        # Update oldest/newest source timestamps (original post publication time)
        if oldest_source and (not sig["oldest_source_at"] or oldest_source < sig["oldest_source_at"]):
            sig["oldest_source_at"] = oldest_source
        if newest_source and (not sig["newest_source_at"] or newest_source > sig["newest_source_at"]):
            sig["newest_source_at"] = newest_source

        # Branch by extraction_class
        if extraction_class == "market_signal":
            sig["occurrence_count"] += occ_count
            sig["unique_source_count"] = max(sig["unique_source_count"], unique_src)
            sig["total_engagement"] += round(avg_eng * occ_count)
            sig["max_engagement"] = max(sig["max_engagement"], max_eng)
            sig["total_comments"] += total_comments

            if direction in sig["directions"]:
                sig["directions"][direction] += occ_count

            if avg_mag:
                sig["_magnitude_sum"] += avg_mag * occ_count
                sig["_magnitude_count"] += occ_count

        elif extraction_class == "entity_mention":
            entity_key = direction or "unknown_entity"
            agg = sig["_entity_agg"]
            if entity_key not in agg:
                agg[entity_key] = {
                    "entity_name": entity_key,
                    "mention_count": 0,
                    "avg_sentiment": 0.0,
                    "_sentiment_sum": 0.0,
                    "unique_sources": 0,
                }
            ea = agg[entity_key]
            ea["mention_count"] += occ_count
            ea["_sentiment_sum"] += avg_sent * occ_count
            ea["unique_sources"] = max(ea["unique_sources"], unique_src)

        elif extraction_class == "context_factor":
            ctx_key = direction or "unknown"
            cagg = sig["_context_agg"]
            if ctx_key not in cagg:
                cagg[ctx_key] = {
                    "category": ctx_key,
                    "direction": direction,
                    "mention_count": 0,
                }
            cagg[ctx_key]["mention_count"] += occ_count

        # Custom per-event classes also contribute to occurrence counts
        else:
            sig["occurrence_count"] += occ_count

    def _finalize_signal_aggregations(self, market_signals: Dict[str, Dict]) -> List[Dict]:
        """Finalize per-market signal aggregations and return signal list."""
        signals = []
        for ticker, sig in market_signals.items():
            # Compute avg_magnitude from accumulated values
            if sig["_magnitude_count"] > 0:
                sig["avg_magnitude"] = round(sig["_magnitude_sum"] / sig["_magnitude_count"], 1)
            del sig["_magnitude_sum"]
            del sig["_magnitude_count"]

            # Compute directional consensus
            total_dir = sig["directions"]["bullish"] + sig["directions"]["bearish"]
            if total_dir > 0:
                sig["consensus"] = "bullish" if sig["directions"]["bullish"] > sig["directions"]["bearish"] else "bearish"
                sig["consensus_strength"] = round(max(sig["directions"].values()) / total_dir, 2)
            else:
                sig["consensus"] = "neutral"
                sig["consensus_strength"] = 0.0

            # Finalize entity_mentions
            for ea in sig["_entity_agg"].values():
                mc = ea["mention_count"]
                sig["entity_mentions"].append({
                    "entity_name": ea["entity_name"],
                    "mention_count": mc,
                    "avg_sentiment": round(ea["_sentiment_sum"] / mc, 1) if mc > 0 else 0.0,
                    "unique_sources": ea["unique_sources"],
                })
            del sig["_entity_agg"]

            # Sort entity_mentions by mention_count desc
            sig["entity_mentions"].sort(key=lambda e: e["mention_count"], reverse=True)

            # Finalize context_factors
            sig["context_factors"] = list(sig["_context_agg"].values())
            sig["context_factors"].sort(key=lambda c: c["mention_count"], reverse=True)
            del sig["_context_agg"]

            signals.append(sig)
        return signals

    def _filter_signals_by_scope(self, signals: List[Dict], event_ticker: Optional[str]) -> List[Dict]:
        """Filter signals to tracked markets and optionally by event_ticker."""
        # Filter to tracked markets only (belt-and-suspenders)
        if self._tracked_markets:
            tracked = {m.ticker for m in self._tracked_markets.get_all()}
            before_count = len(signals)
            signals = [s for s in signals if s["market_ticker"] in tracked]
            filtered = before_count - len(signals)
            if filtered > 0:
                logger.info(
                    f"[deep_agent.tools.get_extraction_signals] Filtered {filtered} signals "
                    f"with non-tracked tickers"
                )

        # Event ticker filter (Python-side since SQL doesn't support it)
        if event_ticker:
            event_tickers_set: set = set()
            if self._tracked_markets:
                markets_by_event = self._tracked_markets.get_markets_by_event()
                event_markets = markets_by_event.get(event_ticker, [])
                event_tickers_set = {m.ticker for m in event_markets}

            if event_tickers_set:
                signals = [s for s in signals if s["market_ticker"] in event_tickers_set]
                for s in signals:
                    s["event_tickers"] = [event_ticker]

        return signals

    async def _attach_extraction_snippets(self, signals: List[Dict], window_hours: float, limit: int) -> None:
        """Attach recent extraction snippets (top 3 by engagement per market)."""
        top_tickers = [s["market_ticker"] for s in signals[:limit]]
        if not top_tickers:
            return

        supabase = self._get_supabase()
        if not supabase:
            return

        try:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)

            snippets_result = supabase.table("extractions") \
                .select("market_tickers, extraction_text, attributes, source_subreddit, engagement_score, extraction_class, source_created_at, created_at") \
                .eq("extraction_class", "market_signal") \
                .gte("created_at", cutoff.isoformat()) \
                .order("engagement_score", desc=True) \
                .limit(limit * 3) \
                .execute()

            # Build snippets lookup by ticker
            snippets_by_ticker: Dict[str, List[Dict]] = {}
            for row in (snippets_result.data or []):
                tickers = row.get("market_tickers", [])
                attrs = row.get("attributes", {})
                for t in tickers:
                    if t in top_tickers:
                        if t not in snippets_by_ticker:
                            snippets_by_ticker[t] = []
                        if len(snippets_by_ticker[t]) < 3:
                            snippets_by_ticker[t].append({
                                "text": (row.get("extraction_text", "") or "")[:200],
                                "direction": attrs.get("direction", ""),
                                "magnitude": attrs.get("magnitude"),
                                "confidence": attrs.get("confidence", ""),
                                "reasoning": (attrs.get("reasoning", "") or "")[:150],
                                "source_subreddit": row.get("source_subreddit", ""),
                                "engagement": row.get("engagement_score", 0),
                                "source_created_at": row.get("source_created_at"),
                                "extracted_at": row.get("created_at"),
                            })

            # Attach snippets to signals
            for sig in signals:
                sig["recent_extractions"] = snippets_by_ticker.get(sig["market_ticker"], [])

        except Exception as e:
            logger.warning(f"[deep_agent.tools.get_extraction_signals] Snippets query failed: {e}")

    async def get_extraction_signals(
        self,
        market_ticker: Optional[str] = None,
        event_ticker: Optional[str] = None,
        window_hours: float = 4.0,
        limit: int = 20,
    ) -> Dict[str, Any]:
        """
        Get aggregated extraction signals from the extraction pipeline.

        Uses the SQL RPC function get_extraction_signals() for efficient
        server-side aggregation across ALL extraction classes (market_signal,
        entity_mention, context_factor, and per-event custom classes).

        Returns enriched signal data per market including entity mentions
        and context factors alongside the core market_signal data.

        Args:
            market_ticker: Filter by specific market
            event_ticker: Filter by event
            window_hours: Time window for aggregation (default: 4 hours)
            limit: Maximum signals to return

        Returns:
            Dict with aggregated signal data per market, including
            entity_mentions and context_factors lists
        """
        self._tool_calls["get_extraction_signals"] += 1

        # Enforce max news age: cap window_hours to configured maximum
        if self._require_fresh_news and window_hours > self._max_news_age_hours:
            logger.info(
                f"[deep_agent.tools.get_extraction_signals] Capping window_hours "
                f"from {window_hours}h to {self._max_news_age_hours}h (require_fresh_news=True)"
            )
            window_hours = self._max_news_age_hours

        supabase = self._get_supabase()
        if not supabase:
            return {"signals": [], "message": "Supabase not available"}

        try:
            # Step 1: Call SQL RPC for server-side aggregation
            rpc_params: Dict[str, Any] = {"p_window_hours": window_hours}
            if market_ticker:
                rpc_params["p_market_ticker"] = market_ticker

            rpc_result = supabase.rpc("get_extraction_signals", rpc_params).execute()
            rpc_rows = rpc_result.data or []

            if not rpc_rows:
                return {
                    "signals": [],
                    "message": f"No extraction signals in the last {window_hours}h",
                    "window_hours": window_hours,
                }

            # Step 2: Process rows into per-market signals
            market_signals: Dict[str, Dict] = {}
            for row in rpc_rows:
                self._process_extraction_row(row, market_signals)

            # Step 3: Finalize aggregations
            signals = self._finalize_signal_aggregations(market_signals)

            # Step 4: Filter by scope
            signals = self._filter_signals_by_scope(signals, event_ticker)

            # Step 5: Attach snippets
            await self._attach_extraction_snippets(signals, window_hours, limit)

            # Step 6: Sort and return
            signals.sort(key=lambda s: s["occurrence_count"], reverse=True)
            top_signals = signals[:limit]

            # Step 7: Background embed top signals into vector memory
            if self._vector_memory and top_signals:
                try:
                    summaries = []
                    for s in top_signals[:5]:
                        ticker = s.get("market_ticker", "?")
                        consensus = s.get("consensus", "?")
                        strength = s.get("consensus_strength", 0)
                        sources = s.get("unique_source_count", 0)
                        summaries.append(
                            f"{ticker}: consensus={consensus} ({strength:.0%}), {sources} sources"
                        )
                    summary = f"Extraction signals ({window_hours}h): " + "; ".join(summaries)
                    asyncio.create_task(
                        self._vector_memory.store_signal_summary("", summary, "signal")
                    )
                except Exception as e:
                    logger.debug(f"[tools] Signal embed failed (non-fatal): {e}")

            return {
                "signals": top_signals,
                "total_markets": len(signals),
                "window_hours": window_hours,
                "message": f"{len(signals)} markets with extraction signals in the last {window_hours}h",
            }

        except Exception as e:
            logger.error(f"[deep_agent.tools.get_extraction_signals] Error: {e}")
            return {"signals": [], "error": str(e)}

    # === Reddit Daily Digest ===

    async def get_reddit_daily_digest(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Get Reddit daily digest - top posts from past 24h with analysis.

        Never blocks the agent cycle. Returns cached data immediately or
        kicks off a background refresh and returns a pending status.
        """
        self._tool_calls["get_reddit_daily_digest"] += 1

        if not self._reddit_historic_agent:
            return {"status": "unavailable", "message": "Reddit historic agent not initialized."}

        # Always try cache first
        cached = self._reddit_historic_agent.get_cached_digest()

        if force_refresh or not cached:
            # Trigger background refresh (non-blocking)
            import asyncio
            asyncio.create_task(self._reddit_historic_agent.run_digest())
            if cached:
                return {**cached, "note": "Refresh triggered in background. Current data shown."}
            return {
                "status": "pending",
                "message": "Reddit digest is being prepared in background. Check again next cycle.",
            }

        return cached

    # === GDELT News Intelligence ===

    async def query_gdelt_news(
        self,
        search_terms: List[str],
        window_hours: Optional[float] = None,
        tone_filter: Optional[str] = None,
        source_filter: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Query GDELT Global Knowledge Graph for mainstream news coverage.

        Searches across thousands of news sources worldwide. Use to confirm
        or discover signals beyond Reddit. GDELT is updated every 15 minutes.

        Args:
            search_terms: List of terms to search (persons, orgs, themes)
            window_hours: How far back to look (default: 4 hours)
            tone_filter: Optional: "positive", "negative", or None
            source_filter: Optional: filter by source name (partial match)
            limit: Max articles to return

        Returns:
            Dict with article_count, source_diversity, tone_summary,
            key_themes, key_persons, key_organizations, top_articles, timeline
        """
        self._tool_calls["query_gdelt_news"] += 1

        if not self._gdelt_client:
            return {"status": "unavailable", "message": "GDELT client not initialized. Set GDELT_ENABLED=true and GDELT_GCP_PROJECT_ID."}

        try:
            result = await self._gdelt_client.query_news(
                search_terms=search_terms,
                window_hours=window_hours,
                tone_filter=tone_filter,
                source_filter=source_filter,
                limit=limit,
            )

            # Track query metadata for trade snapshot / reflection
            query_ts = time.time()
            self._recent_gdelt_queries.appendleft({
                "search_terms": search_terms,
                "article_count": result.get("article_count", 0),
                "source_diversity": result.get("source_diversity", 0),
                "avg_tone": result.get("tone_summary", {}).get("avg_tone", 0),
                "timestamp": query_ts,
            })

            # Store full result for WebSocket snapshot (new client persistence)
            if result.get("article_count", 0) >= 0 and "error" not in result:
                self._recent_gdelt_results.appendleft({
                    "search_terms": search_terms,
                    "window_hours": window_hours,
                    "tone_filter": tone_filter,
                    "article_count": result.get("article_count", 0),
                    "source_diversity": result.get("source_diversity", 0),
                    "tone_summary": result.get("tone_summary", {}),
                    "key_themes": result.get("key_themes", [])[:10],
                    "key_persons": result.get("key_persons", [])[:8],
                    "key_organizations": result.get("key_organizations", [])[:8],
                    "top_articles": result.get("top_articles", [])[:5],
                    "timeline": result.get("timeline", []),
                    "cached": result.get("_cached", False),
                    "timestamp": time.strftime("%H:%M:%S"),
                    "timestamp_unix": query_ts,
                })

            return result
        except Exception as e:
            logger.error(f"[deep_agent.tools.query_gdelt_news] Error: {e}")
            return {"error": str(e), "article_count": 0}

    # === GDELT Events Database (BigQuery Actor-Event-Actor Triples) ===

    async def query_gdelt_events(
        self,
        actor_names: List[str],
        country_filter: Optional[str] = None,
        window_hours: Optional[float] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Query GDELT Events Database for Actor-Event-Actor triples.

        Searches structured event data with CAMEO coding and GoldsteinScale
        conflict/cooperation scoring. Use to understand HOW actors interact
        (investigations, threats, cooperation) beyond just news coverage.

        Args:
            actor_names: Entity names to search in both Actor1 and Actor2 positions
            country_filter: Optional 3-letter ISO country code
            window_hours: How far back to look (default: 4 hours)
            limit: Max events to return (default: 50)

        Returns:
            Dict with event_count, quad_class_summary, goldstein_summary,
            top_event_triples, top_actors, event_code_distribution, geo_hotspots
        """
        self._tool_calls["query_gdelt_events"] += 1

        if not self._gdelt_client:
            return {"status": "unavailable", "message": "GDELT client not initialized. Set GDELT_ENABLED=true and GDELT_GCP_PROJECT_ID."}

        try:
            result = await self._gdelt_client.query_events(
                actor_names=actor_names,
                country_filter=country_filter,
                window_hours=window_hours,
                limit=limit,
            )

            # Track query metadata for trade snapshot / reflection
            query_ts = time.time()
            self._recent_gdelt_queries.appendleft({
                "actor_names": actor_names,
                "event_count": result.get("event_count", 0),
                "avg_goldstein": result.get("goldstein_summary", {}).get("avg", 0),
                "timestamp": query_ts,
                "source": "events",
            })

            # Store full result for WebSocket snapshot
            if result.get("event_count", 0) >= 0 and "error" not in result:
                self._recent_gdelt_results.appendleft({
                    "actor_names": actor_names,
                    "window_hours": window_hours,
                    "event_count": result.get("event_count", 0),
                    "quad_class_summary": result.get("quad_class_summary", {}),
                    "goldstein_summary": result.get("goldstein_summary", {}),
                    "top_event_triples": result.get("top_event_triples", [])[:10],
                    "top_actors": result.get("top_actors", [])[:10],
                    "event_code_distribution": result.get("event_code_distribution", [])[:10],
                    "geo_hotspots": result.get("geo_hotspots", [])[:5],
                    "cached": result.get("_cached", False),
                    "source": "events",
                    "timestamp": time.strftime("%H:%M:%S"),
                    "timestamp_unix": query_ts,
                })

            return result
        except Exception as e:
            logger.error(f"[deep_agent.tools.query_gdelt_events] Error: {e}")
            return {"error": str(e), "event_count": 0}

    # === GDELT DOC API (Free Article Search) ===

    async def search_gdelt_articles(
        self,
        search_terms: List[str],
        timespan: Optional[str] = None,
        tone_filter: Optional[str] = None,
        max_records: Optional[int] = None,
        sort: str = "datedesc",
    ) -> Dict[str, Any]:
        """
        Search GDELT DOC API for articles (FREE, no BigQuery needed).

        Faster and simpler than query_gdelt_news. Best for:
        - Quick article lookup when you need titles and URLs
        - Checking recent coverage of a topic
        - Fallback when BigQuery GKG is not available

        Args:
            search_terms: List of terms to search
            timespan: Time window (e.g., "4h", "1d", "2w")
            tone_filter: "positive", "negative", or None
            max_records: Max articles (default: 75, max: 250)
            sort: "datedesc", "dateasc", "tonedesc", "toneasc"

        Returns:
            Dict with article_count, source_diversity, tone_summary, articles (with title, url, tone)
        """
        self._tool_calls["search_gdelt_articles"] += 1

        if not self._gdelt_doc_client:
            return {"status": "unavailable", "message": "GDELT DOC client not initialized."}

        try:
            result = await self._gdelt_doc_client.search_articles(
                search_terms=search_terms,
                timespan=timespan,
                tone_filter=tone_filter,
                max_records=max_records,
                sort=sort,
            )

            # Track in recent GDELT queries (same deque as GKG)
            self._recent_gdelt_queries.appendleft({
                "search_terms": search_terms,
                "article_count": result.get("article_count", 0),
                "source_diversity": result.get("source_diversity", 0),
                "avg_tone": result.get("tone_summary", {}).get("avg_tone", 0),
                "timestamp": time.time(),
                "source": "doc_api",
            })

            return result
        except Exception as e:
            logger.error(f"[deep_agent.tools.search_gdelt_articles] Error: {e}")
            return {"error": str(e), "article_count": 0}

    async def get_gdelt_volume_timeline(
        self,
        search_terms: List[str],
        timespan: Optional[str] = None,
        tone_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get GDELT coverage volume timeline (FREE, no BigQuery needed).

        Shows how media coverage of a topic changes over time. Useful for:
        - Detecting coverage surges (breaking news)
        - Identifying when a story peaked
        - Comparing coverage trends across topics

        Args:
            search_terms: List of terms to search
            timespan: Time window (e.g., "4h", "1d", "2w")
            tone_filter: "positive", "negative", or None

        Returns:
            Dict with timeline array of {date, value} points
        """
        self._tool_calls["get_gdelt_volume_timeline"] += 1

        if not self._gdelt_doc_client:
            return {"status": "unavailable", "message": "GDELT DOC client not initialized."}

        try:
            return await self._gdelt_doc_client.get_volume_timeline(
                search_terms=search_terms,
                timespan=timespan,
                tone_filter=tone_filter,
            )
        except Exception as e:
            logger.error(f"[deep_agent.tools.get_gdelt_volume_timeline] Error: {e}")
            return {"error": str(e), "timeline": []}

    # === GDELT News Intelligence (Haiku Sub-Agent) ===

    async def get_news_intelligence(
        self,
        search_terms: List[str],
        context_hint: str = "",
    ) -> Dict[str, Any]:
        """
        Get structured news intelligence via Haiku sub-agent analysis of GDELT data.

        Preferred over raw GDELT tools — analyzes full article data and returns
        compact, actionable trading intelligence with sentiment, market signals,
        and freshness assessment. Results cached for 15 minutes.

        Args:
            search_terms: Terms to search (e.g., ["government shutdown", "funding"])
            context_hint: Optional context about what you're investigating
                          (e.g., "Evaluating KXGOVTFUND-25FEB14 NO position")

        Returns:
            Structured intelligence with narrative_summary, sentiment, market_signals,
            freshness, and trading_recommendation (act_now/monitor/wait/no_signal)
        """
        self._tool_calls["get_news_intelligence"] += 1

        if not self._news_analyzer:
            return {
                "status": "unavailable",
                "message": "News analyzer not initialized. GDELT DOC client required.",
            }

        try:
            result = await self._news_analyzer.analyze(
                search_terms=search_terms,
                context_hint=context_hint,
            )

            # Track in recent GDELT queries for trade snapshots
            self._recent_gdelt_queries.appendleft({
                "search_terms": search_terms,
                "article_count": result.get("metadata", {}).get("raw_article_count", 0),
                "status": result.get("status", "unknown"),
                "trading_recommendation": result.get("intelligence", {}).get("trading_recommendation", "unknown"),
                "timestamp": time.time(),
                "source": "news_intelligence",
            })

            # Store for WebSocket snapshot (new clients get recent results on page refresh)
            if result.get("status") not in ("error", "unavailable"):
                self._recent_gdelt_results.appendleft({
                    "search_terms": search_terms,
                    "source": "news_intelligence",
                    "article_count": result.get("metadata", {}).get("raw_article_count", 0),
                    "status": result.get("status", "unknown"),
                    "intelligence": result.get("intelligence", {}),
                    "cached": result.get("metadata", {}).get("cached", False),
                    "timestamp": time.strftime("%H:%M:%S"),
                    "timestamp_unix": time.time(),
                })

                # Background embed research result into vector memory
                if self._vector_memory:
                    try:
                        intel = result.get("intelligence", {})
                        narrative = intel.get("narrative_summary", "")
                        rec = intel.get("trading_recommendation", "")
                        if narrative:
                            summary = (
                                f"GDELT research [{', '.join(search_terms[:3])}]: "
                                f"{narrative[:300]}. Recommendation: {rec}"
                            )
                            asyncio.create_task(
                                self._vector_memory.store_signal_summary(
                                    "", summary, "research"
                                )
                            )
                    except Exception as e2:
                        logger.debug(f"[tools] Research embed failed (non-fatal): {e2}")

            return result
        except Exception as e:
            logger.error(f"[deep_agent.tools.get_news_intelligence] Error: {e}")
            return {"status": "error", "error": str(e)}

    def get_tool_stats(self) -> Dict[str, int]:
        """Get tool usage statistics."""
        return self._tool_calls.copy()
