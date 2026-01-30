"""
Deep Agent Tools - Callable functions for the self-improving agent.

Provides tools for:
- Price impact signals (get_price_impacts) - PRIMARY DATA SOURCE
- Market data retrieval (get_markets)
- Trade execution (trade)
- Session state querying (get_session_state)
- Memory file access (read_memory, write_memory)

The agent trades EXCLUSIVELY on Reddit entity signals transformed into
price impact scores. The get_price_impacts() tool queries the
market_price_impacts table populated by the entity pipeline.

These tools are designed for use with Claude's native tool calling
and provide structured outputs for the agent to reason about.
"""

import asyncio
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Literal, Optional, TYPE_CHECKING

from ..services.entity_accumulator import get_entity_accumulator

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
    recent_fills: List[Dict[str, Any]] = field(default_factory=list)
    is_valid: bool = True  # False when state container is unavailable or errored
    error: Optional[str] = None  # Error description when is_valid is False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PriceImpactItem:
    """A price impact signal from the entity pipeline."""
    signal_id: str
    market_ticker: str
    entity_id: str
    entity_name: str
    sentiment_score: int      # Mapped from 5-point scale: -75, -40, 0, 40, 75
    price_impact_score: int   # Transformed for market type: -75, -40, 0, 40, 75
    confidence: float         # From LLM: 0.5, 0.7, 0.9
    market_type: str          # OUT, WIN, CONFIRM, NOMINEE
    event_ticker: str
    transformation_logic: str # Explains the sentiment→impact transformation
    source_subreddit: str
    created_at: str           # ISO timestamp (when signal was processed)
    suggested_side: str       # "yes" or "no" based on impact direction
    source_title: str = ""    # Reddit post title for context
    context_snippet: str = "" # Text around entity mention
    source_created_at: str = ""  # ISO timestamp of original Reddit post
    content_type: str = ""    # Content type: text, video, link, image, social
    source_domain: str = ""   # Source domain: youtube.com, foxnews.com, reddit.com
    # Liquidity info (populated when liquidity gating is enabled)
    market_spread: Optional[int] = None  # Spread in cents (None if unknown)
    is_tradeable: bool = True  # False if spread exceeds threshold

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


@dataclass
class TradeOpportunityAssessment:
    """Quantitative assessment of a trade opportunity.

    Combines signal data with current market prices to compute:
    - Expected edge (how much the market should move per signal)
    - Signal-implied fair price
    - Priced-in assessment (has the market already moved?)
    - Risk/reward analysis (max profit, max loss, expected profit)
    - Overall trade quality verdict
    """
    market_ticker: str

    # Current market state
    current_yes_bid: int
    current_yes_ask: int
    current_spread: int

    # Signal analysis
    signal_impact: int
    signal_confidence: float
    suggested_side: str               # "yes" or "no"
    expected_edge_cents: int          # Signal-implied price move
    signal_implied_fair_price: int    # Where price "should" be per signal

    # Entry price
    entry_price_cents: int            # What you'd actually pay

    # Priced-in assessment
    price_24h_ago: Optional[int]          # Historical price (None if unavailable)
    price_move_since_signal: Optional[int]  # How much market already moved
    priced_in_pct: Optional[float]        # % of expected move already realized
    priced_in_verdict: str                # NOT_PRICED_IN | PARTIALLY_PRICED_IN | FULLY_PRICED_IN

    # Risk/reward
    max_profit_cents: int             # Best case per contract
    max_loss_cents: int               # Worst case per contract
    expected_profit_cents: int        # Remaining edge per contract
    reward_risk_ratio: float          # expected_profit / max_loss
    risk_reward_verdict: str          # FAVORABLE | MARGINAL | UNFAVORABLE

    # Overall
    trade_quality: str                # STRONG | MODERATE | WEAK | AVOID
    reasoning: str                    # Human-readable summary

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
        price_impact_store: Optional[Any] = None,
        tracked_markets: Optional['TrackedMarketsState'] = None,
        event_position_tracker: Optional['EventPositionTracker'] = None,
        signal_tracker: Optional[Any] = None,
    ):
        """
        Initialize tools with shared dependencies.

        Args:
            trading_client: Client for executing trades and fetching market data
            state_container: Container for trading state (positions, orders, etc.)
            websocket_manager: WebSocket manager for streaming updates
            memory_dir: Directory for memory files (defaults to ./memory)
            price_impact_store: Store for querying price impact signals
            tracked_markets: State container for tracked markets (for event context)
            event_position_tracker: Tracker for event-level position risk
            signal_tracker: Lifecycle tracker for signal evaluation state
        """
        self._trading_client = trading_client
        self._state_container = state_container
        self._ws_manager = websocket_manager
        self._memory_dir = memory_dir or DEFAULT_MEMORY_DIR
        self._price_impact_store = price_impact_store
        self._tracked_markets = tracked_markets
        self._event_position_tracker = event_position_tracker
        self._signal_tracker = signal_tracker
        self._supabase_client = None

        # RF-1: Per-event dollar exposure tracking (resets on restart)
        self._event_exposure_cents: Dict[str, int] = {}
        # $100 cap per event — limits single-event concentration risk.
        # Sized for paper trading; scale with bankroll for production.
        self._max_event_exposure_cents: int = 10000

        # RF-5: Startup timestamp for marking pre-existing signals as historical
        self._startup_time: float = time.time()

        # Ensure memory directory exists
        self._memory_dir.mkdir(parents=True, exist_ok=True)

        # Recent fills tracking for session state
        self._recent_fills: Deque[FillRecord] = deque(maxlen=50)

        # Track tickers the deep agent has traded (for position filtering)
        # Only positions in tickers we've actually traded are shown to the agent,
        # preventing 100+ old positions from previous sessions from appearing.
        self._traded_tickers: set = set()

        # Track tool usage for metrics
        self._tool_calls = {
            "get_price_impacts": 0,
            "get_markets": 0,
            "assess_trade_opportunity": 0,
            "trade": 0,
            "get_session_state": 0,
            "read_memory": 0,
            "write_memory": 0,
            "append_memory": 0,
            "get_event_context": 0,
            "get_entity_signals": 0,
        }


    def _get_supabase(self):
        """Get or create cached Supabase client."""
        if self._supabase_client is None:
            url = os.getenv("SUPABASE_URL")
            key = os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_ANON_KEY")
            if url and key:
                from supabase import create_client
                self._supabase_client = create_client(url, key)
        return self._supabase_client

    # === Price Impact Tools (PRIMARY DATA SOURCE) ===

    async def _get_market_spreads_batch(
        self,
        tickers: List[str],
    ) -> Dict[str, int]:
        """
        Fetch spreads for multiple markets efficiently.

        Args:
            tickers: List of market tickers to fetch spreads for

        Returns:
            Dict mapping ticker -> spread in cents (None if unknown)
        """
        spreads: Dict[str, Optional[int]] = {}

        if not tickers:
            return spreads

        # Try tracked_markets first (already synced, fast)
        if self._tracked_markets:
            try:
                for ticker in tickers:
                    market = self._tracked_markets.get(ticker)
                    if market:
                        # T3.1: Use is not None checks instead of or-0/or-100
                        yes_bid = market.yes_bid if hasattr(market, 'yes_bid') else None
                        yes_ask = market.yes_ask if hasattr(market, 'yes_ask') else None
                        if yes_bid is not None and yes_ask is not None:
                            spreads[ticker] = yes_ask - yes_bid
                        else:
                            spreads[ticker] = None
            except Exception as e:
                logger.warning(f"[deep_agent.tools._get_market_spreads_batch] tracked_markets error: {e}")

        # For any missing tickers, try API
        missing_tickers = [t for t in tickers if t not in spreads]
        if missing_tickers and self._trading_client:
            try:
                markets = await self._trading_client.get_markets(missing_tickers)
                for m in markets:
                    ticker = m.get("ticker", "")
                    yes_bid = m.get("yes_bid")
                    yes_ask = m.get("yes_ask")
                    if yes_bid is not None and yes_ask is not None:
                        spreads[ticker] = yes_ask - yes_bid
                    else:
                        spreads[ticker] = None
            except Exception as e:
                logger.warning(f"[deep_agent.tools._get_market_spreads_batch] API error: {e}")

        # T3.2: Mark unknown tickers as None instead of assuming max spread
        unknown_tickers = [t for t in tickers if t not in spreads]
        if unknown_tickers:
            logger.warning(
                f"[deep_agent.tools._get_market_spreads_batch] "
                f"{len(unknown_tickers)} tickers unresolved: {unknown_tickers[:5]}"
            )
            for ticker in unknown_tickers:
                spreads[ticker] = None

        return spreads

    async def get_price_impacts(
        self,
        market_ticker: Optional[str] = None,
        entity_id: Optional[str] = None,
        min_confidence: float = 0.5,
        min_impact_magnitude: int = 30,
        limit: int = 20,
        max_age_hours: float = 2.0,
        max_spread_cents: int = 15,
        only_tradeable: bool = True,
    ) -> List[PriceImpactItem]:
        """
        Query recent price impact signals from the entity pipeline.

        This is the PRIMARY data source for the self-improving agent.
        Signals come from Reddit entity extraction → sentiment scoring →
        market-specific price impact transformation.

        Uses direct Supabase query for reliability (persists across restarts).

        LIQUIDITY GATING: By default, only returns signals for markets with
        spreads <= max_spread_cents. This prevents the agent from seeing
        signals it cannot act on (the root cause of repeated execution failures).

        Args:
            market_ticker: Filter by specific market
            entity_id: Filter by specific entity
            min_confidence: Minimum confidence threshold (0.0 to 1.0)
            min_impact_magnitude: Minimum |price_impact_score| to include
            limit: Maximum number of signals to return
            max_age_hours: Maximum signal age in hours
            max_spread_cents: Maximum spread for tradeable markets (default: 15c)
            only_tradeable: If True, filter out signals for illiquid markets

        Returns:
            List of PriceImpactItem objects sorted by created_at DESC
        """
        self._tool_calls["get_price_impacts"] += 1
        logger.info(
            f"[deep_agent.tools.get_price_impacts] market={market_ticker}, entity={entity_id}, "
            f"min_conf={min_confidence}, min_impact={min_impact_magnitude}, limit={limit}"
        )

        # Stage 1: Query Supabase (primary) then in-memory store (fallback)
        items, supabase_error, store_error = await self._query_impact_sources(
            market_ticker, entity_id, min_confidence, min_impact_magnitude, limit, max_age_hours,
        )

        # T2.1: Log pipeline health when both sources return nothing
        if not items:
            if supabase_error and store_error:
                logger.error(
                    f"[deep_agent.tools.get_price_impacts] SIGNAL PIPELINE DOWN: "
                    f"Supabase={supabase_error}, Store={store_error}"
                )
            elif supabase_error:
                logger.warning(
                    f"[deep_agent.tools.get_price_impacts] Supabase failed ({supabase_error}), store empty"
                )

        # Stage 2: Liquidity gating
        if items:
            items = await self._apply_liquidity_gating(items, max_spread_cents, only_tradeable, limit)

        # Stage 3: Signal lifecycle filtering
        if items and self._signal_tracker:
            items = self._apply_signal_lifecycle_filter(items)

        logger.info(f"[deep_agent.tools.get_price_impacts] Returning {len(items)} tradeable signals")
        return items

    async def _query_impact_sources(
        self,
        market_ticker: Optional[str],
        entity_id: Optional[str],
        min_confidence: float,
        min_impact_magnitude: int,
        limit: int,
        max_age_hours: float,
    ) -> tuple:
        """Query Supabase (primary) then in-memory store (fallback) for price impacts.

        Returns:
            (items, supabase_error, store_error) tuple
        """
        items: List[PriceImpactItem] = []
        supabase_error = None
        store_error = None

        # Primary: Direct Supabase query (reliable, persists across restarts)
        try:
            supabase = self._get_supabase()

            if supabase:
                from datetime import timezone, timedelta

                cutoff_time = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)

                query = supabase.table("market_price_impacts") \
                    .select("*") \
                    .gte("confidence", min_confidence) \
                    .gte("created_at", cutoff_time.isoformat()) \
                    .order("created_at", desc=True) \
                    .limit(limit * 2)

                if market_ticker:
                    query = query.eq("market_ticker", market_ticker)
                if entity_id:
                    query = query.eq("entity_id", entity_id)

                result = query.execute()

                for row in result.data or []:
                    if abs(row.get("price_impact_score", 0)) >= min_impact_magnitude:
                        items.append(PriceImpactItem(
                            signal_id=row.get("id", ""),
                            market_ticker=row.get("market_ticker", ""),
                            entity_id=row.get("entity_id", ""),
                            entity_name=row.get("entity_name", ""),
                            sentiment_score=row.get("sentiment_score", 0),
                            price_impact_score=row.get("price_impact_score", 0),
                            confidence=row.get("confidence", 0.5),
                            market_type=row.get("market_type", ""),
                            event_ticker=row.get("event_ticker", ""),
                            transformation_logic=row.get("transformation_logic", ""),
                            source_subreddit=row.get("source_subreddit", ""),
                            created_at=row.get("created_at", ""),
                            suggested_side="yes" if row.get("price_impact_score", 0) > 0 else "no",
                            source_title=row.get("source_title", ""),
                            context_snippet=row.get("context_snippet", ""),
                            source_created_at=row.get("source_created_at", ""),
                            content_type=row.get("content_type", ""),
                            source_domain=row.get("source_domain", ""),
                        ))

                logger.info(f"[deep_agent.tools.get_price_impacts] Found {len(items)} raw signals from Supabase")

        except Exception as e:
            supabase_error = str(e)
            logger.warning(f"[deep_agent.tools.get_price_impacts] Supabase query failed: {e}")

        # Fallback: Try in-memory store if Supabase failed (items is empty)
        if not items:
            store = self._price_impact_store
            if store is None:
                from ..services.price_impact_store import get_price_impact_store
                store = get_price_impact_store()

            if store:
                try:
                    store_stats = store.get_stats()
                    logger.info(
                        f"[deep_agent.tools.get_price_impacts] Store stats: {store_stats['signal_count']} signals"
                    )
                    signals = await store.get_impacts_for_trading(
                        min_confidence=min_confidence,
                        min_impact_magnitude=min_impact_magnitude,
                        limit=limit,
                        max_age_hours=max_age_hours,
                    )

                    for signal in signals:
                        if market_ticker and signal.market_ticker != market_ticker:
                            continue
                        if entity_id and signal.entity_id != entity_id:
                            continue

                        # Format source_created_at if available
                        source_created_at_str = ""
                        source_created_at = getattr(signal, "source_created_at", None)
                        if source_created_at:
                            source_created_at_str = datetime.fromtimestamp(source_created_at).isoformat()

                        items.append(PriceImpactItem(
                            signal_id=signal.signal_id,
                            market_ticker=signal.market_ticker,
                            entity_id=signal.entity_id,
                            entity_name=signal.entity_name,
                            sentiment_score=signal.sentiment_score,
                            price_impact_score=signal.price_impact_score,
                            confidence=signal.confidence,
                            market_type=signal.market_type,
                            event_ticker=signal.event_ticker,
                            transformation_logic=signal.transformation_logic,
                            source_subreddit=signal.source_subreddit,
                            created_at=datetime.fromtimestamp(signal.created_at).isoformat(),
                            suggested_side="yes" if signal.price_impact_score > 0 else "no",
                            source_title=getattr(signal, "source_title", ""),
                            context_snippet=getattr(signal, "context_snippet", ""),
                            source_created_at=source_created_at_str,
                            content_type=getattr(signal, "content_type", ""),
                            source_domain=getattr(signal, "source_domain", ""),
                        ))

                    logger.info(f"[deep_agent.tools.get_price_impacts] Found {len(items)} raw signals from store")

                except Exception as e:
                    store_error = str(e)
                    logger.error(f"[deep_agent.tools.get_price_impacts] Store query error: {e}")

        return items, supabase_error, store_error

    async def _apply_liquidity_gating(
        self,
        items: List[PriceImpactItem],
        max_spread_cents: int,
        only_tradeable: bool,
        limit: int,
    ) -> List[PriceImpactItem]:
        """Filter signals to only include those for tradeable (liquid) markets."""
        tickers = list(set(item.market_ticker for item in items))
        spreads = await self._get_market_spreads_batch(tickers)

        filtered_items = []
        illiquid_count = 0
        unknown_count = 0

        for item in items:
            spread = spreads.get(item.market_ticker)
            item.market_spread = spread

            if spread is None:
                # T3.2: Unknown spread = don't trade (conservative)
                unknown_count += 1
                item.is_tradeable = False
            elif spread > max_spread_cents:
                illiquid_count += 1
                item.is_tradeable = False
            else:
                item.is_tradeable = True

            if item.is_tradeable or not only_tradeable:
                filtered_items.append(item)

        if illiquid_count > 0 or unknown_count > 0:
            logger.info(
                f"[deep_agent.tools.get_price_impacts] Filtered: "
                f"{illiquid_count} illiquid (spread > {max_spread_cents}c), "
                f"{unknown_count} unknown spread"
            )

        return filtered_items[:limit]

    def _apply_signal_lifecycle_filter(self, items: List[PriceImpactItem]) -> List[PriceImpactItem]:
        """Register signals with tracker and filter out terminal/exhausted ones."""
        for item in items:
            # RF-5: Signals that predate this session are marked historical
            # so they aren't re-evaluated after a restart.
            is_historical = False
            if item.created_at:
                try:
                    signal_dt = datetime.fromisoformat(
                        item.created_at.replace("Z", "+00:00")
                    )
                    is_historical = signal_dt.timestamp() < self._startup_time
                except (ValueError, AttributeError) as e:
                    # T2.3: Conservative — treat unparseable timestamps as old
                    is_historical = True
                    logger.warning(
                        f"[deep_agent.tools] Signal timestamp parse failed for "
                        f"{item.signal_id}, treating as historical: {e}"
                    )
            self._signal_tracker.register_signal(
                signal_id=item.signal_id,
                market_ticker=item.market_ticker,
                entity_name=item.entity_name,
                is_historical=is_historical,
            )

        all_ids = [item.signal_id for item in items]
        actionable_ids = set(self._signal_tracker.get_actionable_signals(all_ids))

        filtered_items = []
        lifecycle_filtered = 0
        for item in items:
            if item.signal_id in actionable_ids:
                filtered_items.append(item)
            else:
                lifecycle_filtered += 1

        if lifecycle_filtered > 0:
            logger.info(
                f"[deep_agent.tools.get_price_impacts] Filtered {lifecycle_filtered} "
                f"terminal/historical signals via lifecycle tracker"
            )

        return filtered_items

    # === Entity Signal Tools ===

    async def get_entity_signals(
        self,
        min_signal_strength: float = 0.3,
        min_mentions: int = 1,
        limit: int = 15,
    ) -> Dict[str, Any]:
        """Get accumulated entity signals from the knowledge base.

        Shows entities with building narratives (multiple mentions,
        rising sentiment, high engagement). Use alongside get_price_impacts()
        for a complete picture.

        Returns entities sorted by signal_strength (strongest first).
        Each includes linked_market_tickers for actionable trading and
        relations (extracted entity-to-entity relationships like SUPPORTS,
        OPPOSES, CAUSES) for causal chain reasoning.
        """
        self._tool_calls["get_entity_signals"] += 1

        accumulator = get_entity_accumulator()
        if not accumulator:
            return {
                "entity_signals": [],
                "message": "Entity accumulator not available",
                "total_active": 0,
            }

        summaries = accumulator.get_entity_signal_summaries(
            min_strength=min_signal_strength,
            min_mentions=min_mentions,
            limit=limit,
        )

        return {
            "entity_signals": [s.to_dict() for s in summaries],
            "total_active": len(summaries),
            "window_hours": summaries[0].window_hours if summaries else 2.0,
            "message": (
                f"{len(summaries)} entities with active signals"
                if summaries
                else "No entities above threshold"
            ),
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
                    # Get recent signal tickers and fetch those markets
                    try:
                        supabase = self._get_supabase()

                        if supabase:
                            from datetime import timezone, timedelta
                            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=2)

                            # Get unique market tickers from recent signals
                            result = supabase.table("market_price_impacts") \
                                .select("market_ticker") \
                                .gte("created_at", cutoff_time.isoformat()) \
                                .limit(20) \
                                .execute()

                            signal_tickers = list(set(
                                row.get("market_ticker") for row in (result.data or [])
                                if row.get("market_ticker")
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

    # === Trade Opportunity Assessment ===

    async def _fetch_price_history(
        self,
        market_ticker: str,
        event_ticker: str,
    ) -> Optional[int]:
        """
        Fetch the YES price from ~24 hours ago using candlestick data.

        Uses hourly candles (period_interval=60) for the last 24 hours.
        Returns the close price of the earliest candle found.

        Args:
            market_ticker: Market ticker (e.g., "KXGOVSHUT-26JAN31")
            event_ticker: Event ticker for series extraction

        Returns:
            YES price in cents from ~24h ago, or None if unavailable
        """
        if not self._trading_client:
            return None

        try:
            # Extract series_ticker: use event_ticker prefix or market_ticker prefix
            # event_ticker like "KXGOVSHUT" stays as-is
            # event_ticker like "KXTRUMPSAY-26FEB02" -> "KXTRUMPSAY"
            series_ticker = event_ticker.split("-")[0] if event_ticker else market_ticker.split("-")[0]

            now = int(time.time())
            start_ts = now - (24 * 3600)  # 24 hours ago

            # Access the underlying demo client via the integration
            client = getattr(self._trading_client, '_client', None)
            if client is None:
                return None

            response = await client.get_market_candlesticks(
                series_ticker=series_ticker,
                ticker=market_ticker,
                start_ts=start_ts,
                end_ts=now,
                period_interval=60,  # Hourly candles
            )

            candlesticks = response.get("candlesticks", [])
            if not candlesticks:
                return None

            # Get the earliest candle's close price as "price 24h ago"
            for candle in candlesticks:
                price_data = candle.get("price", {})
                close_price = price_data.get("close")
                if close_price is not None and close_price > 0:
                    return int(close_price)

            return None

        except Exception as e:
            logger.debug(f"[deep_agent.tools._fetch_price_history] Failed for {market_ticker}: {e}")
            return None

    async def assess_trade_opportunity(
        self,
        market_ticker: str,
        signal_impact_score: int,
        signal_confidence: float,
        suggested_side: str,
    ) -> TradeOpportunityAssessment:
        """
        Quantitative assessment of a trade opportunity.

        Combines signal data with current market prices to compute expected
        edge, priced-in status, and risk/reward ratio. Call this after
        get_price_impacts() to evaluate whether a signal is worth trading.

        Multi-step analysis flow:
            1. Fetch current bid/ask from tracked markets or API fallback
            2. Compute expected edge from BASE_EDGE lookup * confidence
            3. Derive signal-implied fair price (midpoint +/- edge)
            4. Calculate entry price, max profit/loss, and risk/reward
            5. Detect if signal is already priced in (edge < spread)
            6. Return quality verdict (strong/marginal/weak/negative_edge)

        Args:
            market_ticker: Market to assess
            signal_impact_score: Price impact from signal (-75 to +75)
            signal_confidence: Signal confidence (0.5, 0.7, 0.9)
            suggested_side: "yes" or "no"

        Returns:
            TradeOpportunityAssessment with edge, risk/reward, and quality verdict
        """
        self._tool_calls["assess_trade_opportunity"] += 1
        logger.info(
            f"[deep_agent.tools.assess_trade_opportunity] ticker={market_ticker}, "
            f"impact={signal_impact_score}, conf={signal_confidence}, side={suggested_side}"
        )

        # --- Step 1: Get current market prices ---
        yes_bid = None
        yes_ask = None
        event_ticker = ""
        has_live_data = False

        if self._tracked_markets:
            market = self._tracked_markets.get(market_ticker)
            if market:
                yes_bid = market.yes_bid if hasattr(market, 'yes_bid') else None
                yes_ask = market.yes_ask if hasattr(market, 'yes_ask') else None
                event_ticker = market.event_ticker if hasattr(market, 'event_ticker') else ""
                if yes_bid is not None and yes_ask is not None:
                    has_live_data = True

        # Fallback to API if tracked_markets didn't have data
        if not has_live_data and self._trading_client:
            try:
                market_data = await self._trading_client.get_market(market_ticker)
                if market_data:
                    yes_bid = market_data.get("yes_bid")
                    yes_ask = market_data.get("yes_ask")
                    event_ticker = market_data.get("event_ticker", "") or event_ticker
                    if yes_bid is not None and yes_ask is not None:
                        has_live_data = True
            except Exception as e:
                logger.warning(f"[deep_agent.tools.assess_trade_opportunity] API fallback failed: {e}")

        # T1.3: If both sources failed, return NO_DATA assessment instead of
        # computing edge from fabricated 0/100 sentinel prices
        if not has_live_data:
            logger.warning(
                f"[deep_agent.tools.assess_trade_opportunity] No live market data for {market_ticker}"
            )
            return TradeOpportunityAssessment(
                market_ticker=market_ticker,
                current_yes_bid=0,
                current_yes_ask=0,
                current_spread=0,
                signal_impact=signal_impact_score,
                signal_confidence=signal_confidence,
                suggested_side=suggested_side,
                expected_edge_cents=0,
                signal_implied_fair_price=0,
                entry_price_cents=0,
                price_24h_ago=None,
                price_move_since_signal=None,
                priced_in_pct=None,
                priced_in_verdict="UNKNOWN",
                max_profit_cents=0,
                max_loss_cents=0,
                expected_profit_cents=0,
                reward_risk_ratio=0.0,
                risk_reward_verdict="UNFAVORABLE",
                trade_quality="NO_DATA",
                reasoning=f"Cannot assess: no live market data available for {market_ticker}. "
                          f"Both tracked_markets and API returned no bid/ask prices.",
            )

        # At this point yes_bid and yes_ask are guaranteed non-None
        yes_bid = yes_bid or 0  # handle 0 as valid
        yes_ask = yes_ask or 0

        spread = yes_ask - yes_bid
        midpoint = (yes_bid + yes_ask) // 2

        # --- Step 2: Expected edge model ---
        # Base edge in cents per impact level, calibrated from backtest observations:
        # impact 75 (max) -> 25c expected move, impact 40 (moderate) -> 12c.
        # Fallback for other values: abs_impact // 3 (conservative linear scale).
        BASE_EDGE = {75: 25, 40: 12}
        abs_impact = abs(signal_impact_score)
        base_edge = BASE_EDGE.get(abs_impact, max(1, abs_impact // 3))
        expected_edge = int(base_edge * signal_confidence)

        # --- Step 3: Signal-implied fair price ---
        if suggested_side == "yes":
            fair_price = midpoint + expected_edge
        else:
            fair_price = midpoint - expected_edge
        fair_price = max(3, min(97, fair_price))

        # --- Step 4: Entry price and risk/reward ---
        if suggested_side == "yes":
            entry = yes_ask                          # Buying YES at ask
            max_profit = 100 - entry                 # YES resolves at 100
            max_loss = entry                         # YES resolves at 0
            expected_profit = fair_price - entry     # Signal says it'll reach fair_price
        else:
            entry = 100 - yes_bid if yes_bid > 0 else 100  # Buying NO
            max_profit = 100 - entry                 # NO resolves at 100
            max_loss = entry                         # NO resolves at 0
            expected_profit = (100 - fair_price) - entry

        reward_risk = round(expected_profit / max_loss, 2) if max_loss > 0 else 0.0

        # Risk/reward verdict
        if reward_risk >= 1.5:
            rr_verdict = "FAVORABLE"
        elif reward_risk >= 0.5:
            rr_verdict = "MARGINAL"
        else:
            rr_verdict = "UNFAVORABLE"

        # --- Step 5: Priced-in assessment ---
        price_24h_ago: Optional[int] = None
        price_move: Optional[int] = None
        priced_in_pct: Optional[float] = None

        # Try candlestick-based assessment
        if event_ticker:
            price_24h_ago = await self._fetch_price_history(market_ticker, event_ticker)

        if price_24h_ago is not None and expected_edge > 0:
            # Compute how much the market already moved toward the signal direction
            if suggested_side == "yes":
                price_move = midpoint - price_24h_ago
            else:
                price_move = price_24h_ago - midpoint
            priced_in_pct = round(abs(price_move) / expected_edge * 100, 1) if expected_edge > 0 else 0.0

        # Determine priced-in verdict
        if priced_in_pct is not None:
            if priced_in_pct >= 80:
                priced_in_verdict = "FULLY_PRICED_IN"
            elif priced_in_pct >= 40:
                priced_in_verdict = "PARTIALLY_PRICED_IN"
            else:
                priced_in_verdict = "NOT_PRICED_IN"
        else:
            # Fallback: use expected_profit vs expected_edge
            if expected_profit < 3:
                priced_in_verdict = "FULLY_PRICED_IN"
            elif expected_edge > 0 and expected_profit < expected_edge * 0.5:
                priced_in_verdict = "PARTIALLY_PRICED_IN"
            else:
                priced_in_verdict = "NOT_PRICED_IN"

        # --- Step 6: Trade quality ---
        if (expected_profit >= 15 and reward_risk >= 1.5
                and priced_in_verdict != "FULLY_PRICED_IN"):
            quality = "STRONG"
        elif (expected_profit >= 8 and reward_risk >= 1.0
              and priced_in_verdict != "FULLY_PRICED_IN"):
            quality = "MODERATE"
        elif expected_profit >= 3 and reward_risk >= 0.5:
            quality = "WEAK"
        else:
            quality = "AVOID"

        # --- Step 7: Human-readable reasoning ---
        reasoning_parts = [
            f"Signal: impact={signal_impact_score}, conf={signal_confidence}, side={suggested_side}.",
            f"Market: bid={yes_bid}c, ask={yes_ask}c, spread={spread}c.",
            f"Edge model: {expected_edge}c expected move -> fair_price={fair_price}c.",
            f"Entry at {entry}c: expected_profit={expected_profit}c, max_loss={max_loss}c, R:R={reward_risk:.1f}x.",
        ]
        if price_24h_ago is not None:
            reasoning_parts.append(
                f"24h ago: {price_24h_ago}c, moved {price_move}c ({priced_in_pct:.0f}% priced in)."
            )
        reasoning_parts.append(f"Verdict: {quality} ({rr_verdict}, {priced_in_verdict}).")
        reasoning = " ".join(reasoning_parts)

        logger.info(f"[deep_agent.tools.assess_trade_opportunity] {market_ticker}: quality={quality}, edge={expected_profit}c, R:R={reward_risk}")

        return TradeOpportunityAssessment(
            market_ticker=market_ticker,
            current_yes_bid=yes_bid,
            current_yes_ask=yes_ask,
            current_spread=spread,
            signal_impact=signal_impact_score,
            signal_confidence=signal_confidence,
            suggested_side=suggested_side,
            expected_edge_cents=expected_edge,
            signal_implied_fair_price=fair_price,
            entry_price_cents=entry,
            price_24h_ago=price_24h_ago,
            price_move_since_signal=price_move,
            priced_in_pct=priced_in_pct,
            priced_in_verdict=priced_in_verdict,
            max_profit_cents=max_profit,
            max_loss_cents=max_loss,
            expected_profit_cents=expected_profit,
            reward_risk_ratio=reward_risk,
            risk_reward_verdict=rr_verdict,
            trade_quality=quality,
            reasoning=reasoning,
        )

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

    # === Trade Execution ===

    @staticmethod
    def _compute_limit_price(side: str, yes_bid: int, yes_ask: int, execution_strategy: str) -> int:
        """Compute limit price in cents given side, bid/ask, and strategy.

        Returns a clamped price in [1, 99].
        """
        midpoint = (yes_bid + yes_ask) // 2 if yes_bid > 0 else 0

        if execution_strategy == "aggressive":
            if side == "yes":
                price = yes_ask
            else:
                price = 100 - yes_bid if yes_bid > 0 else 0
        elif execution_strategy == "moderate":
            if side == "yes":
                price = midpoint + 1 if midpoint > 0 else yes_ask
            else:
                price = (100 - midpoint) + 1 if midpoint > 0 else (100 - yes_bid if yes_bid > 0 else 0)
        elif execution_strategy == "passive":
            if side == "yes":
                price = yes_bid + 1 if yes_bid > 0 else yes_ask
            else:
                price = (100 - yes_ask) + 1 if yes_ask > 0 else (100 - yes_bid if yes_bid > 0 else 0)
        else:
            # Fallback to aggressive
            if side == "yes":
                price = yes_ask
            else:
                price = 100 - yes_bid if yes_bid > 0 else 0

        return max(1, min(99, price))

    async def trade(
        self,
        ticker: str,
        side: Literal["yes", "no"],
        contracts: int,
        reasoning: str,
        execution_strategy: Literal["aggressive", "moderate", "passive"] = "aggressive",
    ) -> TradeResult:
        """
        Execute a trade on the market.

        Places a limit order with pricing determined by execution_strategy:
        - aggressive: Cross the spread (buy at ask). Fastest fill, highest cost.
        - moderate: Place at midpoint + 1c. Saves ~half the spread, decent fill probability.
        - passive: Place near the bid + 1c. Cheapest entry, may not fill.

        Enforces per-event dollar exposure cap (_max_event_exposure_cents) to prevent
        concentration risk. Exposure is released when trades settle via
        release_event_exposure().

        Args:
            ticker: Market ticker to trade
            side: "yes" or "no"
            contracts: Number of contracts (1-100)
            reasoning: Why you're making this trade (stored for reflection)
            execution_strategy: How aggressively to price the order (default: aggressive)

        Returns:
            TradeResult with success status and fill details
        """
        self._tool_calls["trade"] += 1
        logger.info(f"[deep_agent.tools.trade] ticker={ticker}, side={side}, contracts={contracts}")
        logger.info(f"[deep_agent.tools.trade] reasoning: {reasoning[:100]}...")

        # Validate inputs
        if contracts < 1 or contracts > 100:
            return TradeResult(
                success=False,
                ticker=ticker,
                side=side,
                contracts=contracts,
                error="Contracts must be between 1 and 100",
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

            limit_price = self._compute_limit_price(side, yes_bid, yes_ask, execution_strategy)

            # RF-1: Enforce per-event dollar exposure cap
            event_ticker = market_data.get("event_ticker", "")
            if event_ticker:
                new_cost_cents = contracts * limit_price
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
            # For buying YES/NO contracts: action is always "buy"
            # side determines whether we're buying YES or NO contracts
            result = await self._trading_client.place_order(
                ticker=ticker,
                action="buy",  # Always buying (to sell, we'd buy the opposite side)
                side=side,     # "yes" or "no"
                count=contracts,
                price=limit_price,  # Limit price in cents
                order_type="limit",  # All Kalshi orders are limit orders
            )

            # Check for success in response - place_order returns {"order": {...}}
            order = result.get("order")
            success = order is not None

            if success:
                # Extract order details
                order_id = order.get("order_id", "")
                # For market orders, avg_price comes from fills
                avg_price = order.get("avg_price") or order.get("price")

                # Broadcast to WebSocket
                if self._ws_manager:
                    await self._ws_manager.broadcast_message("deep_agent_trade", {
                        "ticker": ticker,
                        "side": side,
                        "contracts": contracts,
                        "price_cents": avg_price,
                        "order_id": order_id,
                        "reasoning": reasoning[:200],
                        "timestamp": time.strftime("%H:%M:%S"),
                    })

                logger.info(f"[deep_agent.tools.trade] Order placed successfully: {order_id} @ {avg_price}c")

                # RF-1: Record event exposure for dollar cap enforcement
                if event_ticker:
                    cost = contracts * limit_price
                    self._event_exposure_cents[event_ticker] = (
                        self._event_exposure_cents.get(event_ticker, 0) + cost
                    )
                    logger.info(
                        f"[deep_agent.tools.trade] Event {event_ticker} exposure: "
                        f"${self._event_exposure_cents[event_ticker]/100:.2f} / "
                        f"${self._max_event_exposure_cents/100:.0f}"
                    )

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

                return TradeResult(
                    success=True,
                    ticker=ticker,
                    side=side,
                    contracts=contracts,
                    price_cents=avg_price,
                    order_id=order_id,
                    limit_price_cents=limit_price,
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

            # Get positions with details, filtered to only tickers the deep
            # agent has traded. The Kalshi API returns ALL positions (including
            # old ones from previous sessions), which pollutes the agent's view.
            # If no trades have happened yet, show NO positions (empty list).
            positions = []
            for pos in summary.get("positions_details", []):
                ticker = pos.get("ticker")

                # Filter: only show positions in tickers we've actually traded
                if ticker not in self._traded_tickers:
                    continue

                # T1.5: Explicit position side determination
                yes_c = pos.get("yes_contracts", 0) or 0
                no_c = pos.get("no_contracts", 0) or 0
                if yes_c > 0:
                    pos_side = "yes"
                    pos_contracts = yes_c
                elif no_c > 0:
                    pos_side = "no"
                    pos_contracts = no_c
                else:
                    pos_side = "none"
                    pos_contracts = 0

                positions.append({
                    "ticker": ticker,
                    "side": pos_side,
                    "contracts": pos_contracts,
                    "avg_price": pos.get("avg_price"),
                    "current_value": pos.get("market_value"),
                    "unrealized_pnl": pos.get("unrealized_pnl"),
                })

            # Calculate win rate from settlements
            settlements = summary.get("settlements", [])
            wins = sum(1 for s in settlements if s.get("pnl", 0) > 0)
            total_settled = len(settlements)
            win_rate = wins / total_settled if total_settled > 0 else 0.0

            return SessionState(
                balance_cents=int(summary.get("balance", 0) * 100),
                portfolio_value_cents=int(summary.get("portfolio_value", 0) * 100),
                realized_pnl_cents=int(pnl.get("realized", 0) * 100),
                unrealized_pnl_cents=int(pnl.get("unrealized", 0) * 100),
                total_pnl_cents=int(pnl.get("total", 0) * 100),
                position_count=len(positions),  # Filtered count (this session only)
                open_order_count=summary.get("order_count", 0),
                trade_count=total_settled,
                win_rate=win_rate,
                positions=positions,
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
    _APPEND_ONLY_FILES = {"learnings.md", "mistakes.md", "patterns.md"}

    async def write_memory(self, filename: str, content: str) -> bool:
        """
        Write to a memory file (full replacement).

        Only strategy.md should be fully replaced. For learnings.md, mistakes.md,
        and patterns.md, use append_memory() to avoid accidental data loss.

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

        logger.info(f"[deep_agent.tools.write_memory] Writing {len(content)} chars to {safe_name}")

        try:
            file_path.write_text(content, encoding="utf-8")

            # Broadcast memory update to WebSocket
            if self._ws_manager:
                await self._ws_manager.broadcast_message("deep_agent_memory_update", {
                    "filename": safe_name,
                    "content_preview": content[:200] + "..." if len(content) > 200 else content,
                    "timestamp": time.strftime("%H:%M:%S"),
                })

            logger.info(f"[deep_agent.tools.write_memory] Successfully wrote {safe_name}")
            return True

        except Exception as e:
            logger.error(f"[deep_agent.tools.write_memory] Error writing {safe_name}: {e}")
            return False

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

            # Broadcast memory update to WebSocket
            if self._ws_manager:
                await self._ws_manager.broadcast_message("deep_agent_memory_update", {
                    "filename": safe_name,
                    "content_preview": new_content[:200] + "..." if len(new_content) > 200 else new_content,
                    "timestamp": time.strftime("%H:%M:%S"),
                })

            logger.info(f"[deep_agent.tools.append_memory] Successfully wrote {len(new_content)} chars to {safe_name}")
            success = True
        except Exception as e:
            logger.error(f"[deep_agent.tools.append_memory] Error writing {safe_name}: {e}")
            success = False

        return {
            "success": success,
            "new_length": len(new_content),
            "archived": archived,
            "archive_path": archive_path if archived else None,
        }

    def get_tool_stats(self) -> Dict[str, int]:
        """Get tool usage statistics."""
        return self._tool_calls.copy()
