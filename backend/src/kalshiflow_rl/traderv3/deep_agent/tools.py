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

These tools are designed to be used with LangChain's tool calling
and provide structured outputs for the agent to reason about.
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, TYPE_CHECKING

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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NewsItem:
    """A news item from search."""
    title: str
    url: str
    snippet: str
    source: str
    published_at: Optional[str] = None
    relevance_score: float = 0.5

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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PriceImpactItem:
    """A price impact signal from the entity pipeline."""
    signal_id: str
    market_ticker: str
    entity_id: str
    entity_name: str
    sentiment_score: int      # Original entity sentiment: -100 to +100
    price_impact_score: int   # Transformed for market type: -100 to +100
    confidence: float         # Signal confidence: 0.0 to 1.0
    market_type: str          # OUT, WIN, CONFIRM, NOMINEE
    event_ticker: str
    transformation_logic: str # Explains the sentiment→impact transformation
    source_subreddit: str
    created_at: str           # ISO timestamp
    suggested_side: str       # "yes" or "no" based on impact direction

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EventMarketInfo:
    """Info about a single market within an event."""
    ticker: str
    title: str
    yes_price: int  # Current YES price in cents
    no_price: int   # Current NO price (100 - yes_price)
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
        return {
            "event_ticker": self.event_ticker,
            "market_count": self.market_count,
            "markets": [m.to_dict() for m in self.markets],
            "yes_sum": self.yes_sum,
            "no_sum": self.no_sum,
            "risk_level": self.risk_level,
            "has_positions": self.has_positions,
            "position_count": self.position_count,
            "total_contracts": self.total_contracts,
            "mutual_exclusivity_note": self.mutual_exclusivity_note,
        }


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
        """
        self._trading_client = trading_client
        self._state_container = state_container
        self._ws_manager = websocket_manager
        self._memory_dir = memory_dir or DEFAULT_MEMORY_DIR
        self._price_impact_store = price_impact_store
        self._tracked_markets = tracked_markets
        self._event_position_tracker = event_position_tracker

        # Ensure memory directory exists
        self._memory_dir.mkdir(parents=True, exist_ok=True)

        # Track tool usage for metrics
        self._tool_calls = {
            "get_price_impacts": 0,
            "get_markets": 0,
            "trade": 0,
            "get_session_state": 0,
            "read_memory": 0,
            "write_memory": 0,
            "get_event_context": 0,
        }

    # === Price Impact Tools (PRIMARY DATA SOURCE) ===

    async def get_price_impacts(
        self,
        market_ticker: Optional[str] = None,
        entity_id: Optional[str] = None,
        min_confidence: float = 0.5,
        min_impact_magnitude: int = 30,
        limit: int = 20,
        max_age_hours: float = 2.0,
    ) -> List[PriceImpactItem]:
        """
        Query recent price impact signals from the entity pipeline.

        This is the PRIMARY data source for the self-improving agent.
        Signals come from Reddit entity extraction → sentiment scoring →
        market-specific price impact transformation.

        Uses direct Supabase query for reliability (persists across restarts).

        Args:
            market_ticker: Filter by specific market
            entity_id: Filter by specific entity
            min_confidence: Minimum confidence threshold (0.0 to 1.0)
            min_impact_magnitude: Minimum |price_impact_score| to include
            limit: Maximum number of signals to return
            max_age_hours: Maximum signal age in hours

        Returns:
            List of PriceImpactItem objects sorted by created_at DESC
        """
        self._tool_calls["get_price_impacts"] += 1
        logger.info(
            f"[get_price_impacts] market={market_ticker}, entity={entity_id}, "
            f"min_conf={min_confidence}, min_impact={min_impact_magnitude}, limit={limit}"
        )

        items = []

        # Primary: Direct Supabase query (reliable, persists across restarts)
        try:
            import os
            from supabase import create_client

            url = os.getenv("SUPABASE_URL")
            key = os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_ANON_KEY")

            if url and key:
                from datetime import timezone, timedelta

                supabase = create_client(url, key)
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
                        ))
                        if len(items) >= limit:
                            break

                logger.info(f"[get_price_impacts] Returned {len(items)} signals from Supabase")
                return items

        except Exception as e:
            logger.warning(f"[get_price_impacts] Supabase query failed: {e}")

        # Fallback: Try in-memory store if Supabase failed
        store = self._price_impact_store
        if store is None:
            from ..services.price_impact_store import get_price_impact_store
            store = get_price_impact_store()

        if store:
            try:
                store_stats = store.get_stats()
                logger.info(
                    f"[get_price_impacts] Store stats: {store_stats['signal_count']} signals"
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
                    ))

                logger.info(f"[get_price_impacts] Returned {len(items)} signals from store")

            except Exception as e:
                logger.error(f"[get_price_impacts] Store query error: {e}")

        return items

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
        logger.info(f"[get_markets] event_ticker={event_ticker}, limit={limit}")

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
                    yes_bid = m.yes_bid if hasattr(m, 'yes_bid') else (m.price or 50)
                    yes_ask = m.yes_ask if hasattr(m, 'yes_ask') else (m.price or 50)

                    markets.append(MarketInfo(
                        ticker=m.ticker,
                        event_ticker=m.event_ticker,
                        title=m.yes_sub_title or m.title or "",
                        yes_bid=yes_bid,
                        yes_ask=yes_ask,
                        spread=yes_ask - yes_bid,
                        volume_24h=getattr(m, 'volume_24h', 0) or 0,
                        status=m.status or "active",
                        last_trade_price=getattr(m, 'last_price', None),
                    ))

                logger.info(f"[get_markets] Returned {len(markets)} markets from tracked_markets")
                return markets

            except Exception as e:
                logger.warning(f"[get_markets] Error reading tracked_markets: {e}")
                # Fall through to API fallback

        # Fallback: Direct API call via trading client
        if self._trading_client:
            try:
                raw_markets = []

                if event_ticker:
                    # Try to get event data from API (for markets not in tracked_markets)
                    logger.info(f"[get_markets] Fetching event {event_ticker} from API")
                    event_data = await self._trading_client.get_event(event_ticker)
                    if event_data:
                        raw_markets = event_data.get("markets", [])[:limit]
                        logger.info(f"[get_markets] Got {len(raw_markets)} markets from event API")

                # If no event_ticker or event lookup failed, try to get specific tickers
                # from recent price impact signals
                if not raw_markets and not event_ticker:
                    # Get recent signal tickers and fetch those markets
                    try:
                        import os
                        from supabase import create_client

                        url = os.getenv("SUPABASE_URL")
                        key = os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_ANON_KEY")

                        if url and key:
                            supabase = create_client(url, key)
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
                                logger.info(f"[get_markets] Fetching {len(signal_tickers)} signal market tickers from API")
                                raw_markets = await self._trading_client.get_markets(signal_tickers)
                                logger.info(f"[get_markets] Got {len(raw_markets)} markets from API for signal tickers")
                    except Exception as e:
                        logger.warning(f"[get_markets] Could not fetch signal tickers: {e}")

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
                    logger.info(f"[get_markets] Returned {len(markets)} markets from API")
                    return markets

            except Exception as e:
                logger.error(f"[get_markets] API Error: {e}")

        logger.warning("[get_markets] No market data source available")
        return []

    # === News Search ===

    async def search_news(
        self,
        query: str,
        max_results: int = 10,
        max_age_hours: float = 24.0,
    ) -> List[NewsItem]:
        """
        Search for price-moving news.

        Uses DuckDuckGo search for finding relevant news articles.

        Args:
            query: Search query (e.g., "Newsom 2028 presidential polls")
            max_results: Maximum number of results to return
            max_age_hours: Maximum age of news in hours

        Returns:
            List of NewsItem objects with title, URL, and snippet
        """
        self._tool_calls["search_news"] += 1
        logger.info(f"[search_news] query='{query}', max_results={max_results}")

        news_items = []

        try:
            # Import DDGS for search
            from duckduckgo_search import DDGS

            with DDGS() as ddgs:
                # Use news search for fresher results
                results = list(ddgs.news(
                    query,
                    max_results=max_results,
                    timelimit="d" if max_age_hours <= 24 else "w",
                ))

            for r in results:
                news_items.append(NewsItem(
                    title=r.get("title", ""),
                    url=r.get("url", ""),
                    snippet=r.get("body", "")[:500],  # Truncate long snippets
                    source=r.get("source", ""),
                    published_at=r.get("date"),
                    relevance_score=0.5,  # Default, could be enhanced with scoring
                ))

            logger.info(f"[search_news] Found {len(news_items)} results")
            return news_items

        except ImportError:
            logger.error("[search_news] duckduckgo_search not installed")
            return []
        except Exception as e:
            logger.error(f"[search_news] Error: {e}")
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
        logger.info(f"[get_event_context] event_ticker={event_ticker}")

        if not self._tracked_markets:
            logger.warning("[get_event_context] No tracked_markets available")
            return None

        # Get markets grouped by event
        markets_by_event = self._tracked_markets.get_markets_by_event()
        event_markets = markets_by_event.get(event_ticker)

        if not event_markets:
            logger.warning(f"[get_event_context] Event {event_ticker} not found")
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

        for market in event_markets:
            ticker = market.ticker
            yes_price = market.yes_ask if market.yes_ask > 0 else market.price
            if yes_price <= 0:
                yes_price = 50  # Default

            no_price = 100 - yes_price
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
        no_sum = n_markets * 100 - yes_sum

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

        logger.info(
            f"[get_event_context] {event_ticker}: {n_markets} markets, "
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

    async def trade(
        self,
        ticker: str,
        side: Literal["yes", "no"],
        contracts: int,
        reasoning: str,
    ) -> TradeResult:
        """
        Execute a trade on the market.

        Args:
            ticker: Market ticker to trade
            side: "yes" or "no"
            contracts: Number of contracts (1-100)
            reasoning: Why you're making this trade (stored for reflection)

        Returns:
            TradeResult with success status and fill details
        """
        self._tool_calls["trade"] += 1
        logger.info(f"[trade] ticker={ticker}, side={side}, contracts={contracts}")
        logger.info(f"[trade] reasoning: {reasoning[:100]}...")

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
            # Execute market order through trading client
            # Use place_order() which is the actual method on V3TradingClientIntegration
            # For buying YES/NO contracts: action is always "buy"
            # side determines whether we're buying YES or NO contracts
            result = await self._trading_client.place_order(
                ticker=ticker,
                action="buy",  # Always buying (to sell, we'd buy the opposite side)
                side=side,     # "yes" or "no"
                count=contracts,
                order_type="market",  # Market order for immediate execution
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

                logger.info(f"[trade] Order placed successfully: {order_id} @ {avg_price}c")

                return TradeResult(
                    success=True,
                    ticker=ticker,
                    side=side,
                    contracts=contracts,
                    price_cents=avg_price,
                    order_id=order_id,
                )
            else:
                error_msg = result.get("error") or result.get("message") or "Unknown error"
                logger.warning(f"[trade] Order failed: {error_msg}")
                return TradeResult(
                    success=False,
                    ticker=ticker,
                    side=side,
                    contracts=contracts,
                    error=error_msg,
                )

        except Exception as e:
            logger.error(f"[trade] Error: {e}")
            return TradeResult(
                success=False,
                ticker=ticker,
                side=side,
                contracts=contracts,
                error=str(e),
            )

    # === Session State ===

    async def get_session_state(self) -> SessionState:
        """
        Get current trading session state.

        Returns:
            SessionState with balance, positions, P&L, and recent activity
        """
        self._tool_calls["get_session_state"] += 1
        logger.info("[get_session_state] Fetching session state")

        if not self._state_container:
            logger.warning("[get_session_state] No state container available")
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
            )

        try:
            summary = self._state_container.get_trading_summary()
            pnl = summary.get("pnl", {})

            # Get positions with details
            positions = []
            for pos in summary.get("positions_details", []):
                positions.append({
                    "ticker": pos.get("ticker"),
                    "side": "yes" if pos.get("yes_contracts", 0) > 0 else "no",
                    "contracts": pos.get("yes_contracts", 0) or pos.get("no_contracts", 0),
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
                position_count=summary.get("position_count", 0),
                open_order_count=summary.get("order_count", 0),
                trade_count=total_settled,
                win_rate=win_rate,
                positions=positions,
                recent_fills=[],  # TODO: Add recent fills
            )

        except Exception as e:
            logger.error(f"[get_session_state] Error: {e}")
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

        logger.info(f"[read_memory] Reading {safe_name}")

        try:
            if file_path.exists():
                content = file_path.read_text(encoding="utf-8")
                logger.info(f"[read_memory] Read {len(content)} chars from {safe_name}")
                return content
            else:
                logger.warning(f"[read_memory] File not found: {safe_name}")
                return ""
        except Exception as e:
            logger.error(f"[read_memory] Error reading {safe_name}: {e}")
            return ""

    async def write_memory(self, filename: str, content: str) -> bool:
        """
        Write to a memory file.

        Args:
            filename: Name of memory file (e.g., "learnings.md")
            content: Content to write

        Returns:
            True if successful, False otherwise
        """
        self._tool_calls["write_memory"] += 1

        # Sanitize filename
        safe_name = Path(filename).name
        file_path = self._memory_dir / safe_name

        logger.info(f"[write_memory] Writing {len(content)} chars to {safe_name}")

        try:
            file_path.write_text(content, encoding="utf-8")

            # Broadcast memory update to WebSocket
            if self._ws_manager:
                await self._ws_manager.broadcast_message("deep_agent_memory_update", {
                    "filename": safe_name,
                    "content_preview": content[:200] + "..." if len(content) > 200 else content,
                    "timestamp": time.strftime("%H:%M:%S"),
                })

            logger.info(f"[write_memory] Successfully wrote {safe_name}")
            return True

        except Exception as e:
            logger.error(f"[write_memory] Error writing {safe_name}: {e}")
            return False

    async def append_memory(self, filename: str, content: str) -> bool:
        """
        Append to a memory file.

        Args:
            filename: Name of memory file
            content: Content to append

        Returns:
            True if successful
        """
        existing = await self.read_memory(filename)
        new_content = existing + "\n" + content if existing else content
        return await self.write_memory(filename, new_content)

    def get_tool_stats(self) -> Dict[str, int]:
        """Get tool usage statistics."""
        return self._tool_calls.copy()


# === Module-level convenience functions ===
# These are used when tools need to be passed to LangChain without a class instance

_global_tools: Optional[DeepAgentTools] = None


def set_global_tools(tools: DeepAgentTools) -> None:
    """Set the global tools instance."""
    global _global_tools
    _global_tools = tools


async def get_price_impacts(
    market_ticker: Optional[str] = None,
    entity_id: Optional[str] = None,
    min_confidence: float = 0.5,
    limit: int = 20,
) -> List[Dict]:
    """Get price impact signals from the entity pipeline."""
    if _global_tools:
        items = await _global_tools.get_price_impacts(
            market_ticker=market_ticker,
            entity_id=entity_id,
            min_confidence=min_confidence,
            limit=limit,
        )
        return [i.to_dict() for i in items]
    return []


async def get_markets(event_ticker: Optional[str] = None, limit: int = 20) -> List[Dict]:
    """Get current market data."""
    if _global_tools:
        markets = await _global_tools.get_markets(event_ticker, limit)
        return [m.to_dict() for m in markets]
    return []


async def trade(ticker: str, side: str, contracts: int, reasoning: str) -> Dict:
    """Execute a trade."""
    if _global_tools:
        result = await _global_tools.trade(ticker, side, contracts, reasoning)
        return result.to_dict()
    return {"success": False, "error": "Tools not initialized"}


async def get_session_state() -> Dict:
    """Get session state."""
    if _global_tools:
        state = await _global_tools.get_session_state()
        return state.to_dict()
    return {}


async def read_memory(filename: str) -> str:
    """Read memory file."""
    if _global_tools:
        return await _global_tools.read_memory(filename)
    return ""


async def write_memory(filename: str, content: str) -> bool:
    """Write memory file."""
    if _global_tools:
        return await _global_tools.write_memory(filename, content)
    return False
