"""
RLM NO Strategy Plugin - Reverse Line Movement for TRADER V3.

This plugin implements the validated +17.38% edge RLM strategy:
Bet NO when majority trade YES but price drops.

Key Mechanics:
    - Signal: >65% YES trades + YES price dropped from open + >=15 trades
    - Entry: Market or limit order based on orderbook spread
    - Exit: Hold to settlement (no early exits in MVP)
    - Re-entry: Allow adding to position on stronger signal

Architecture Position:
    - Registered with StrategyRegistry as "rlm_no"
    - Subscribes to PUBLIC_TRADE_RECEIVED events
    - Subscribes to TMO_FETCHED events (for true market open)
    - Accumulates per-market trade state
    - Creates TradingDecision objects with strategy_id="rlm_no"
    - Delegates execution to TradingDecisionService

Reference: research/strategies/h014_rlm_validation.md
"""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Set, TYPE_CHECKING

from ..protocol import Strategy, StrategyContext
from ..registry import StrategyRegistry
from ..coordinator import StrategyCoordinator
from ...core.events import EventType
from ...core.event_bus import PublicTradeEvent, MarketDeterminedEvent, TMOFetchedEvent
from ...core.state_machine import TraderState
from ....data.orderbook_state import get_shared_orderbook_state
from ...state.order_context import OrderbookContext, BBODepthTier
from ...state.trading_attachment import TrackedMarketOrder

if TYPE_CHECKING:
    from ...services.trading_decision_service import TradingDecisionService, TradingDecision
    from ...core.state_container import V3StateContainer
    from ...state.tracked_markets import TrackedMarketsState
    from ...clients.orderbook_integration import V3OrderbookIntegration
    from ...core.event_bus import EventBus

logger = logging.getLogger("kalshiflow_rl.traderv3.strategies.plugins.rlm_no")


# Decision history size
DECISION_HISTORY_SIZE = 100

# Recent tracked trades buffer size
RECENT_TRACKED_TRADES_SIZE = 30


@dataclass
class TrackedTrade:
    """
    Trade record for tracked markets only.

    Used for the Trade Processing panel to display recent trades
    that passed the filter (tracked markets).
    """
    trade_id: str
    market_ticker: str
    side: str  # "yes" | "no"
    price_cents: int
    count: int
    timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for WebSocket transport."""
        return {
            "trade_id": self.trade_id,
            "market_ticker": self.market_ticker,
            "side": self.side,
            "price_cents": self.price_cents,
            "count": self.count,
            "timestamp": self.timestamp,
            "age_seconds": int(time.time() - self.timestamp),
        }


@dataclass
class MarketTradeState:
    """
    Per-market trade accumulation for RLM signal detection.

    Tracks:
        - Trade direction counts (YES vs NO) - reset after signal execution
        - Price movement from market open (first trade) to current
        - Position state for re-entry logic

    IMPORTANT: first_yes_price is set once on the first trade seen and NEVER reset.
    This matches the validated strategy which measures price drop from market open
    throughout the entire market lifecycle. This allows signals to fire multiple times
    as price continues to drop (with larger positions for stronger signals per S-001).
    """
    market_ticker: str
    yes_trades: int = 0
    no_trades: int = 0
    first_yes_price: Optional[int] = None  # First observed YES price (fallback if no TMO)
    last_yes_price: Optional[int] = None   # Current trade price (updated each trade, reset after signal)
    first_trade_time: Optional[float] = None  # When we started observing
    last_trade_time: Optional[float] = None   # Current trade time - reset after signal

    # True Market Open (TMO) - fetched from candlestick API
    # This is the actual market opening price, more accurate than first_yes_price
    true_market_open: Optional[int] = None  # YES price at true market open (cents)

    # Position tracking for re-entry
    position_contracts: int = 0
    entry_yes_ratio: Optional[float] = None  # Ratio when first entered
    entry_price_drop: Optional[int] = None   # Drop when first entered

    # Historical tracking (not reset by _reset_signal_state)
    signal_trigger_count: int = 0

    @property
    def total_trades(self) -> int:
        return self.yes_trades + self.no_trades

    @property
    def yes_ratio(self) -> float:
        return self.yes_trades / self.total_trades if self.total_trades > 0 else 0.0

    @property
    def open_price(self) -> Optional[int]:
        """Get the market open price - prefer TMO, fallback to first observed."""
        return self.true_market_open if self.true_market_open is not None else self.first_yes_price

    @property
    def price_drop(self) -> int:
        """Calculate price drop from market open (TMO if available) to current price."""
        open_price = self.open_price
        if open_price is None or self.last_yes_price is None:
            return 0
        return open_price - self.last_yes_price

    @property
    def using_tmo(self) -> bool:
        """Check if TMO is available for price drop calculation."""
        return self.true_market_open is not None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/transport."""
        return {
            "market_ticker": self.market_ticker,
            "yes_trades": self.yes_trades,
            "no_trades": self.no_trades,
            "total_trades": self.total_trades,
            "yes_ratio": round(self.yes_ratio, 3),
            "first_yes_price": self.first_yes_price,
            "last_yes_price": self.last_yes_price,
            "true_market_open": self.true_market_open,
            "open_price": self.open_price,  # TMO or first observed
            "using_tmo": self.using_tmo,
            "price_drop": self.price_drop,
            "position_contracts": self.position_contracts,
            "signal_trigger_count": self.signal_trigger_count,
        }


@dataclass
class RLMSignal:
    """
    Represents a detected RLM trading signal.

    Attributes:
        market_ticker: Market ticker for this signal
        yes_ratio: Ratio of YES trades (0.0 - 1.0)
        price_drop: YES price drop in cents from open
        trade_count: Total trades observed
        is_reentry: Whether this is a re-entry signal
        detected_at: Timestamp when signal was detected
    """
    market_ticker: str
    yes_ratio: float
    price_drop: int
    trade_count: int
    is_reentry: bool = False
    detected_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/transport."""
        return {
            "market_ticker": self.market_ticker,
            "yes_ratio": round(self.yes_ratio, 3),
            "price_drop": self.price_drop,
            "trade_count": self.trade_count,
            "is_reentry": self.is_reentry,
            "detected_at": self.detected_at,
            "age_seconds": int(time.time() - self.detected_at),
        }


@dataclass
class RLMDecision:
    """
    Records a decision about an RLM signal.

    Attributes:
        signal_id: Unique identifier (market_ticker:timestamp)
        timestamp: When decision was made
        action: Decision action ("executed", "skipped_*", "failed", "reentry")
        reason: Human-readable explanation
        order_id: Kalshi order ID if executed
        signal_data: Original signal data for reference
    """
    signal_id: str
    timestamp: float
    action: str
    reason: str
    order_id: Optional[str] = None
    signal_data: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for WebSocket transport."""
        result = {
            "signal_id": self.signal_id,
            "timestamp": self.timestamp,
            "action": self.action,
            "reason": self.reason,
            "order_id": self.order_id[:8] if self.order_id else None,
            "age_seconds": int(time.time() - self.timestamp),
        }
        if self.signal_data:
            result["market_ticker"] = self.signal_data.get("market_ticker", "")
            result["yes_ratio"] = self.signal_data.get("yes_ratio", 0)
            result["price_drop"] = self.signal_data.get("price_drop", 0)
        return result


@StrategyRegistry.register("rlm_no")
class RLMNoStrategy:
    """
    RLM NO Strategy Plugin - Reverse Line Movement Trading.

    Implements the validated +17.38% edge strategy: bet NO when majority
    trade YES but price drops. This strategy was validated on 1.7M+ trades
    across 72k+ unique markets.

    Attributes:
        name: "rlm_no"
        display_name: "Reverse Line Movement NO"
        subscribed_events: PUBLIC_TRADE_RECEIVED, TMO_FETCHED, MARKET_DETERMINED

    Key Features:
        - Trade direction tracking from public trade feed
        - Price movement detection from market open
        - Orderbook-aware execution (market vs limit based on spread)
        - Re-entry on stronger signals
        - Category filtering via lifecycle discovery
    """

    name: str = "rlm_no"
    display_name: str = "Reverse Line Movement NO"
    subscribed_events: Set[EventType] = {
        EventType.PUBLIC_TRADE_RECEIVED,
        EventType.TMO_FETCHED,
        EventType.MARKET_DETERMINED,
    }

    def __init__(self):
        """Initialize the RLM NO strategy."""
        self._context: Optional[StrategyContext] = None
        self._running: bool = False
        self._started_at: Optional[float] = None

        # Signal detection parameters (loaded from config)
        self._yes_threshold: float = 0.65
        self._min_trades: int = 15
        self._min_price_drop: int = 0
        self._contracts_per_trade: int = 100
        self._max_concurrent: int = 1000
        self._allow_reentry: bool = True

        # Orderbook execution parameters
        self._orderbook_timeout: float = 2.0
        self._tight_spread: int = 2
        self._normal_spread: int = 4
        self._max_spread: int = 10

        # Time-to-settlement filter
        self._min_hours_to_settlement: float = 4.0
        self._max_days_to_settlement: int = 30

        # Rate limiting (internal - coordinator also has global rate limit)
        self._max_tokens: int = 10
        self._tokens: float = 10.0
        self._token_refill_seconds: float = 6.0
        self._last_refill_time: float = time.time()
        self._rate_limited_count: int = 0

        # Per-market trade accumulation state
        self._market_states: Dict[str, MarketTradeState] = {}

        # Track markets with open positions
        self._markets_with_positions: Set[str] = set()

        # Track markets at per-market position max
        self._maxed_markets: Set[str] = set()
        self._per_market_max: int = 100

        # Decision history
        self._decision_history: deque[RLMDecision] = deque(maxlen=DECISION_HISTORY_SIZE)

        # Recent tracked trades buffer
        self._recent_tracked_trades: deque[TrackedTrade] = deque(maxlen=RECENT_TRACKED_TRADES_SIZE)

        # Statistics
        self._stats = {
            "trades_processed": 0,
            "trades_filtered": 0,
            "signals_detected": 0,
            "signals_executed": 0,
            "signals_skipped": 0,
            "signals_skipped_maxed": 0,
            "reentries": 0,
            "orderbook_fallbacks": 0,
        }
        self._last_execution_time: Optional[float] = None

        # Processing lock
        self._processing_lock = asyncio.Lock()

        # State broadcast throttling
        self._last_state_emit: Dict[str, float] = {}
        self._state_emit_interval = 0.5

        # Trade counter for unique IDs
        self._trade_counter: int = 0

        # Background task for periodic orderbook signal broadcast
        self._signal_broadcaster_task: Optional[asyncio.Task] = None
        self._signal_broadcast_interval = 10.0

        logger.debug("RLMNoStrategy initialized")

    async def start(self, context: StrategyContext) -> None:
        """
        Start the RLM NO strategy.

        Loads configuration, subscribes to events, and begins processing.

        Args:
            context: Shared strategy context
        """
        if self._running:
            logger.warning("RLMNoStrategy is already running")
            return

        self._context = context
        self._running = True
        self._started_at = time.time()

        # Load configuration from coordinator if available
        self._load_config_from_coordinator()

        # Subscribe to public trade events
        await context.event_bus.subscribe_to_public_trade(self._handle_public_trade)
        logger.info("Subscribed to PUBLIC_TRADE_RECEIVED events")

        # Subscribe to market determined events (for cleanup)
        await context.event_bus.subscribe_to_market_determined(self._handle_market_determined)
        logger.info("Subscribed to MARKET_DETERMINED events for cleanup")

        # Subscribe to TMO fetched events
        await context.event_bus.subscribe_to_tmo_fetched(self._handle_tmo_fetched)
        logger.info("Subscribed to TMO_FETCHED events for price improvement")

        # Start orderbook signal broadcaster task
        self._signal_broadcaster_task = asyncio.create_task(
            self._run_signal_broadcaster(),
            name="rlm_no_signal_broadcaster"
        )
        logger.info("Started orderbook signal broadcaster task")

        # Emit startup event
        await context.event_bus.emit_system_activity(
            activity_type="strategy_start",
            message=f"RLM NO strategy started",
            metadata={
                "strategy": "rlm_no",
                "yes_threshold": f"{self._yes_threshold:.0%}",
                "min_trades": self._min_trades,
                "max_concurrent": self._max_concurrent,
            }
        )

        logger.info(
            f"RLMNoStrategy started: yes_threshold={self._yes_threshold:.0%}, "
            f"min_trades={self._min_trades}, min_price_drop={self._min_price_drop}c, "
            f"contracts={self._contracts_per_trade}, max_concurrent={self._max_concurrent}, "
            f"reentry={'ON' if self._allow_reentry else 'OFF'}"
        )

    def _load_config_from_coordinator(self) -> None:
        """Load configuration from strategy coordinator if available."""
        if not self._context:
            return

        if not self._context.config:
            logger.warning(
                "No config in context - using HARDCODED DEFAULTS! "
                f"contracts={self._contracts_per_trade}, threshold={self._yes_threshold}"
            )
            return

        config = self._context.config
        params = config.params

        # Load all parameters from YAML config
        self._yes_threshold = params.get("yes_threshold", self._yes_threshold)
        self._min_trades = params.get("min_trades", self._min_trades)
        self._min_price_drop = params.get("min_price_drop", self._min_price_drop)
        self._contracts_per_trade = params.get("contracts_per_trade", self._contracts_per_trade)
        self._max_concurrent = params.get("max_concurrent", self._max_concurrent)
        self._allow_reentry = params.get("allow_reentry", self._allow_reentry)
        self._orderbook_timeout = params.get("orderbook_timeout", self._orderbook_timeout)
        self._tight_spread = params.get("tight_spread", self._tight_spread)
        self._normal_spread = params.get("normal_spread", self._normal_spread)
        self._max_spread = params.get("max_spread", self._max_spread)
        self._min_hours_to_settlement = params.get("min_hours_to_settlement", self._min_hours_to_settlement)
        self._max_days_to_settlement = params.get("max_days_to_settlement", self._max_days_to_settlement)

        logger.info(
            f"Loaded config from YAML: {config.name} "
            f"(contracts={self._contracts_per_trade}, threshold={self._yes_threshold}, "
            f"min_trades={self._min_trades}, min_drop={self._min_price_drop}c, "
            f"max_concurrent={self._max_concurrent})"
        )

    async def stop(self) -> None:
        """
        Stop the RLM NO strategy.

        Unsubscribes from events and cleans up resources.
        """
        if not self._running:
            return

        logger.info("Stopping RLMNoStrategy...")
        self._running = False

        # Cancel signal broadcaster task
        if self._signal_broadcaster_task and not self._signal_broadcaster_task.done():
            self._signal_broadcaster_task.cancel()
            try:
                await self._signal_broadcaster_task
            except asyncio.CancelledError:
                pass
            logger.debug("Signal broadcaster task cancelled")
        self._signal_broadcaster_task = None

        # Unsubscribe from EventBus
        if self._context and self._context.event_bus:
            self._context.event_bus.unsubscribe(
                EventType.PUBLIC_TRADE_RECEIVED, self._handle_public_trade
            )
            self._context.event_bus.unsubscribe(
                EventType.MARKET_DETERMINED, self._handle_market_determined
            )
            self._context.event_bus.unsubscribe(
                EventType.TMO_FETCHED, self._handle_tmo_fetched
            )
            logger.debug("Unsubscribed from all events")

            # Emit shutdown event
            await self._context.event_bus.emit_system_activity(
                activity_type="strategy_stop",
                message=f"RLM NO strategy stopped",
                metadata={
                    "strategy": "rlm_no",
                    "signals_detected": self._stats["signals_detected"],
                    "signals_executed": self._stats["signals_executed"],
                }
            )

        logger.info(
            f"RLMNoStrategy stopped. Stats: "
            f"detected={self._stats['signals_detected']}, "
            f"executed={self._stats['signals_executed']}, "
            f"skipped={self._stats['signals_skipped']}"
        )

    def is_healthy(self) -> bool:
        """
        Check if the strategy is healthy.

        Returns:
            True if the strategy is running
        """
        return self._running

    def get_stats(self) -> Dict[str, Any]:
        """
        Get strategy statistics.

        Returns stats in the format expected by WebSocketManager._broadcast_trade_processing():
        - trades_processed: Number of trades processed (passed filter)
        - trades_filtered: Number of trades filtered out (not tracked)
        - signals_detected: Number of RLM signals detected
        - signals_executed: Number of signals that resulted in orders
        - signals_skipped: Number of signals skipped (rate limited, maxed, etc)
        - rate_limited_count: Number of signals skipped due to rate limiting
        - reentries: Number of re-entry signals

        Returns:
            Dictionary with strategy statistics
        """
        uptime = time.time() - self._started_at if self._started_at else 0

        return {
            "name": self.name,
            "display_name": self.display_name,
            "running": self._running,
            "uptime_seconds": uptime,
            # Core trade processing stats (required by WebSocketManager)
            "trades_processed": self._stats["trades_processed"],
            "trades_filtered": self._stats["trades_filtered"],
            "signals_detected": self._stats["signals_detected"],
            "signals_executed": self._stats["signals_executed"],
            "signals_skipped": self._stats["signals_skipped"],
            "rate_limited_count": self._rate_limited_count,
            "reentries": self._stats["reentries"],
            # Additional stats
            "last_signal_at": self._last_execution_time,
            "markets_tracking": len(self._market_states),
            "config": {
                "yes_threshold": self._yes_threshold,
                "min_trades": self._min_trades,
                "min_price_drop": self._min_price_drop,
                "contracts_per_trade": self._contracts_per_trade,
                "max_concurrent": self._max_concurrent,
                "allow_reentry": self._allow_reentry,
            },
        }

    # ============================================================
    # Event Handlers
    # ============================================================

    async def _handle_public_trade(self, trade_event: PublicTradeEvent) -> None:
        """Handle public trade events."""
        if not self._running or not self._context:
            return

        # Wait for system to be fully ready
        machine_state = self._context.state_container.machine_state
        if machine_state != TraderState.READY:
            logger.debug(f"Skipping RLM processing - system not ready (state={machine_state})")
            return

        # Filter: only process trades for tracked markets
        if not self._context.tracked_markets:
            return
        if not self._context.tracked_markets.is_tracked(trade_event.market_ticker):
            self._stats["trades_filtered"] += 1
            return

        # Use lock to prevent concurrent processing
        async with self._processing_lock:
            await self._process_trade(trade_event)

    async def _handle_market_determined(self, event: MarketDeterminedEvent) -> None:
        """Handle market determined events for cleanup."""
        ticker = event.market_ticker
        cleaned = False

        if ticker in self._market_states:
            del self._market_states[ticker]
            cleaned = True

        if ticker in self._last_state_emit:
            del self._last_state_emit[ticker]
            cleaned = True

        if ticker in self._markets_with_positions:
            self._markets_with_positions.discard(ticker)
            cleaned = True

        if ticker in self._maxed_markets:
            self._maxed_markets.discard(ticker)
            cleaned = True

        if cleaned:
            logger.info(f"RLMNoStrategy cleanup for determined market: {ticker}")

    async def _handle_tmo_fetched(self, event: TMOFetchedEvent) -> None:
        """Handle True Market Open fetched events."""
        ticker = event.market_ticker
        tmo = event.true_market_open

        if ticker not in self._market_states:
            self._market_states[ticker] = MarketTradeState(market_ticker=ticker)

        state = self._market_states[ticker]
        old_tmo = state.true_market_open

        state.true_market_open = tmo

        if old_tmo is None:
            improvement = ""
            if state.first_yes_price is not None:
                diff = state.first_yes_price - tmo
                if diff != 0:
                    improvement = f" (diff from first observed: {diff:+d}c)"
            logger.info(f"TMO set for {ticker}: {tmo}c{improvement}")
        else:
            logger.debug(f"TMO updated for {ticker}: {old_tmo}c -> {tmo}c")

    # ============================================================
    # Trade Processing
    # ============================================================

    async def _process_trade(self, trade_event: PublicTradeEvent) -> None:
        """Process a public trade and update market state."""
        self._stats["trades_processed"] += 1
        market_ticker = trade_event.market_ticker

        # Check if this market is at per-market position max
        if await self._check_and_handle_maxed_position(market_ticker):
            pass  # Continue to update state but skip signal detection

        # Store in tracked trades buffer
        self._trade_counter += 1
        trade_id = f"{market_ticker}:{trade_event.timestamp_ms}:{self._trade_counter}"
        tracked_trade = TrackedTrade(
            trade_id=trade_id,
            market_ticker=market_ticker,
            side=trade_event.side,
            price_cents=trade_event.price_cents,
            count=trade_event.count,
            timestamp=time.time(),
        )
        self._recent_tracked_trades.append(tracked_trade)

        # Get or create market state
        if market_ticker not in self._market_states:
            self._market_states[market_ticker] = MarketTradeState(
                market_ticker=market_ticker
            )

        state = self._market_states[market_ticker]

        # Update trade counts
        if trade_event.side == "yes":
            state.yes_trades += 1
        else:
            state.no_trades += 1

        # Calculate YES price from trade
        yes_price = trade_event.price_cents if trade_event.side == "yes" else (100 - trade_event.price_cents)

        # Track price movement from market open
        if state.first_yes_price is None:
            state.first_yes_price = yes_price
            state.first_trade_time = time.time()
            logger.debug(f"Set market open anchor for {market_ticker}: first_yes_price={yes_price}c")

        state.last_yes_price = yes_price
        state.last_trade_time = time.time()

        # Emit trade arrived event
        if self._context and self._context.event_bus:
            await self._context.event_bus.emit_rlm_trade_arrived(
                market_ticker=market_ticker,
                side=trade_event.side,
                count=trade_event.count,
                price_cents=trade_event.price_cents,
            )

        # Emit state update (throttled)
        if self._should_emit_state_update(market_ticker):
            state_dict = state.to_dict()
            if self._context and self._context.orderbook_integration:
                orderbook_signals = self._context.orderbook_integration.get_orderbook_signals(market_ticker)
                if orderbook_signals:
                    state_dict["orderbook_signals"] = orderbook_signals

            if self._context and self._context.event_bus:
                await self._context.event_bus.emit_rlm_market_update(
                    market_ticker=market_ticker,
                    state=state_dict,
                )
            self._last_state_emit[market_ticker] = time.time()

        # Skip signal detection if market is at max position
        if market_ticker in self._maxed_markets:
            return

        # Check for signal
        signal = self._detect_signal(state)
        if signal:
            await self._execute_signal(signal)

    async def _check_and_handle_maxed_position(self, market_ticker: str) -> bool:
        """Check if a market is at per-market position max."""
        if not self._context:
            return False

        trading_state = self._context.state_container.trading_state
        positions = trading_state.positions if trading_state else {}
        market_position = positions.get(market_ticker, {})
        position_size = abs(market_position.get("position", 0))

        is_maxed = position_size >= self._per_market_max

        if is_maxed:
            if market_ticker not in self._maxed_markets:
                self._maxed_markets.add(market_ticker)
                self._stats["signals_skipped_maxed"] += 1

                attachment = self._context.state_container.get_trading_attachment(market_ticker)
                if attachment:
                    attachment.mark_position_maxed()

                await self._context.event_bus.emit_system_activity(
                    activity_type="position_maxed",
                    message=f"Position maxed for {market_ticker}: {position_size} contracts (limit: {self._per_market_max})",
                    metadata={
                        "market": market_ticker,
                        "position": position_size,
                        "limit": self._per_market_max,
                    }
                )
                logger.info(
                    f"Market {market_ticker} at max position ({position_size} contracts) - "
                    f"skipping signal detection"
                )
            return True
        else:
            hysteresis_threshold = int(self._per_market_max * 0.8)
            if market_ticker in self._maxed_markets and position_size < hysteresis_threshold:
                self._maxed_markets.discard(market_ticker)

                attachment = self._context.state_container.get_trading_attachment(market_ticker)
                if attachment:
                    attachment.clear_position_maxed()

                logger.info(
                    f"Market {market_ticker} position reduced below hysteresis threshold "
                    f"({position_size} < {hysteresis_threshold}) - signal detection resumed"
                )
            return False

    # ============================================================
    # Signal Detection
    # ============================================================

    def _detect_signal(self, state: MarketTradeState) -> Optional[RLMSignal]:
        """Detect if market state meets RLM signal criteria."""
        if not self._context:
            return None

        # Check minimum trades
        if state.total_trades < self._min_trades:
            return None

        # Check YES ratio
        if state.yes_ratio <= self._yes_threshold:
            return None

        # Check price drop
        if state.price_drop < self._min_price_drop:
            return None

        # Belt-and-suspenders: validate time-to-settlement
        if self._context.tracked_markets:
            tracked = self._context.tracked_markets.get_market(state.market_ticker)
            if tracked and tracked.close_ts > 0:
                now_ts = int(time.time())
                hours_to_settlement = (tracked.close_ts - now_ts) / 3600

                if hours_to_settlement < self._min_hours_to_settlement:
                    logger.debug(
                        f"Signal skipped for {state.market_ticker}: "
                        f"settling in {hours_to_settlement:.1f}h < {self._min_hours_to_settlement}h minimum"
                    )
                    return None

                max_hours = self._max_days_to_settlement * 24
                if hours_to_settlement > max_hours:
                    logger.debug(
                        f"Signal skipped for {state.market_ticker}: "
                        f"settling in {hours_to_settlement/24:.1f}d > {self._max_days_to_settlement}d maximum"
                    )
                    return None

        # Check concurrent positions
        trading_state = self._context.state_container.trading_state
        positions = trading_state.positions if trading_state else {}
        current_position_count = len(positions)

        has_position = state.market_ticker in positions
        is_reentry = has_position and self._allow_reentry

        if current_position_count >= self._max_concurrent and not is_reentry:
            return None

        # For re-entry, check if signal is stronger than entry
        if is_reentry:
            if state.entry_yes_ratio is not None and state.entry_price_drop is not None:
                if state.yes_ratio <= state.entry_yes_ratio and state.price_drop <= state.entry_price_drop:
                    return None

        self._stats["signals_detected"] += 1

        return RLMSignal(
            market_ticker=state.market_ticker,
            yes_ratio=state.yes_ratio,
            price_drop=state.price_drop,
            trade_count=state.total_trades,
            is_reentry=is_reentry,
        )

    # ============================================================
    # Signal Execution
    # ============================================================

    async def _execute_signal(self, signal: RLMSignal) -> None:
        """Execute an RLM signal by placing a NO order."""
        from ...services.trading_decision_service import TradingDecision, TradingStrategy

        if not self._context or not self._context.trading_service:
            logger.warning("Cannot execute signal - no trading service available")
            return

        signal_id = f"{signal.market_ticker}:{int(signal.detected_at * 1000)}"
        trade_succeeded = False

        # Increment trigger count
        state = self._market_states.get(signal.market_ticker)
        if state:
            state.signal_trigger_count += 1

        try:
            # Check rate limit
            self._refill_tokens()
            if not self._consume_token():
                self._rate_limited_count += 1
                logger.debug(f"Rate limited signal {signal_id}")
                return

            # Emit processing start
            await self._context.event_bus.emit_system_activity(
                activity_type="rlm_signal",
                message=f"RLM Signal: {signal.market_ticker} YES={signal.yes_ratio:.0%} drop={signal.price_drop}c",
                metadata={"signal_id": signal_id, "status": "processing", **signal.to_dict()}
            )

            # Get orderbook for execution price
            entry_price = None
            ob_context: Optional[OrderbookContext] = None

            try:
                orderbook_state = await asyncio.wait_for(
                    get_shared_orderbook_state(signal.market_ticker),
                    timeout=self._orderbook_timeout
                )
                snapshot = await orderbook_state.get_snapshot()

                ob_context = OrderbookContext.from_orderbook_snapshot(
                    snapshot,
                    tight_spread=self._tight_spread,
                    normal_spread=self._normal_spread,
                )

                # REST fallback if WS connection is degraded
                if self._context.orderbook_integration:
                    ws_healthy = self._context.orderbook_integration.is_connection_healthy()

                    if not ws_healthy:
                        refreshed = await self._context.orderbook_integration.refresh_orderbook_if_stale(
                            signal.market_ticker
                        )
                        if refreshed:
                            snapshot = await orderbook_state.get_snapshot()
                            ob_context = OrderbookContext.from_orderbook_snapshot(
                                snapshot,
                                tight_spread=self._tight_spread,
                                normal_spread=self._normal_spread,
                            )
                            self._stats["rest_refresh_successes"] = self._stats.get("rest_refresh_successes", 0) + 1

                        if ob_context.should_skip_due_to_staleness():
                            logger.warning(
                                f"Stale orderbook for {signal.market_ticker}: "
                                f"WS degraded, REST refresh failed, skipping signal"
                            )
                            self._stats["signals_skipped"] += 1
                            self._stats["orderbook_stale_skips"] = self._stats.get("orderbook_stale_skips", 0) + 1
                            self._record_decision(
                                signal_id=signal_id,
                                action="skipped",
                                reason="WS degraded, REST refresh failed",
                                signal_data=signal.to_dict()
                            )
                            return

                if ob_context.no_best_ask is None or ob_context.no_best_bid is None:
                    logger.warning(f"No orderbook data for {signal.market_ticker}, skipping signal")
                    self._stats["signals_skipped"] += 1
                    return

                # Tiered spread rejection
                if signal.price_drop >= 20:
                    effective_max_spread = 8
                elif signal.price_drop >= 10:
                    effective_max_spread = 5
                else:
                    effective_max_spread = 3

                if ob_context.no_spread is not None and ob_context.no_spread > effective_max_spread:
                    logger.warning(
                        f"Spread too wide for {signal.market_ticker}: {ob_context.no_spread}c > {effective_max_spread}c max, skipping"
                    )
                    self._stats["signals_skipped"] += 1
                    self._record_decision(
                        signal_id=signal_id,
                        action="skipped",
                        reason=f"Spread {ob_context.no_spread}c > tiered max {effective_max_spread}c (drop={signal.price_drop}c)",
                        signal_data=signal.to_dict()
                    )
                    return

                # Log orderbook state
                self._log_orderbook_at_signal(signal, snapshot)

                # Position-aware pricing
                is_high_edge = signal.price_drop >= 10

                entry_price = ob_context.get_recommended_entry_price(
                    aggressive=is_high_edge,
                    max_spread=self._max_spread
                )

                if entry_price is None:
                    entry_price = self._calculate_no_entry_price(
                        ob_context.no_best_ask, ob_context.no_best_bid,
                        ob_context.no_spread or 0, aggressive=is_high_edge
                    )

                aggr_label = " [aggressive]" if is_high_edge else ""
                depth_label = f" depth={ob_context.bbo_depth_tier.value}" if ob_context.bbo_depth_tier != BBODepthTier.UNKNOWN else ""
                imbalance_label = f" imbal={ob_context.bid_imbalance:+.2f}" if ob_context.bid_imbalance is not None else ""
                logger.info(
                    f"RLM pricing: {signal.market_ticker} spread={ob_context.no_spread}c "
                    f"bid={ob_context.no_best_bid} ask={ob_context.no_best_ask} -> entry={entry_price}c "
                    f"({ob_context.spread_tier.value}{aggr_label}{depth_label}{imbalance_label})"
                )

            except (asyncio.TimeoutError, Exception) as e:
                logger.warning(f"Orderbook unavailable for {signal.market_ticker}: {e}, skipping signal")
                self._stats["orderbook_fallbacks"] += 1
                self._stats["signals_skipped"] += 1
                return

            # Flat sizing
            scaled_quantity = self._contracts_per_trade
            scale_label = "1x"

            # Create trading decision with strategy_id
            action_type = "reentry" if signal.is_reentry else "executed"
            state = self._market_states.get(signal.market_ticker)
            signal_params = {
                "yes_trades": state.yes_trades if state else 0,
                "no_trades": state.no_trades if state else 0,
                "total_trades": state.total_trades if state else 0,
                "yes_ratio": round(signal.yes_ratio, 4),
                "price_drop_cents": signal.price_drop,
                "true_market_open": state.true_market_open if state else None,
                "last_yes_price": state.last_yes_price if state else None,
                "using_tmo": state.using_tmo if state else False,
                "is_reentry": signal.is_reentry,
                "entry_yes_ratio": state.entry_yes_ratio if state else None,
                "position_scale": scale_label,
                "signal_trigger_count": state.signal_trigger_count if state else 1,
                "aggressive_pricing": ob_context.no_spread is not None and ob_context.no_spread <= 2 and signal.price_drop >= 10,
                "signal_detected_at": signal.detected_at,
            }

            # Create decision with both old enum (for backward compat) and new strategy_id
            decision = TradingDecision(
                action="buy",
                market=signal.market_ticker,
                side="no",
                quantity=scaled_quantity,
                price=entry_price,
                reason=f"RLM signal: {signal.yes_ratio:.0%} YES, -{signal.price_drop}c drop ({scale_label})",
                confidence=min(0.9, 0.7 + signal.price_drop * 0.01),
                strategy=TradingStrategy.RLM_NO,  # Keep for backward compatibility
                signal_params=signal_params,
            )
            # Add strategy_id for new plugin system
            decision.strategy_id = "rlm_no"

            logger.info(
                f"Executing RLM signal: {signal.market_ticker} "
                f"NO @ {entry_price or 'market'}c, {scaled_quantity} contracts ({scale_label}) "
                f"({'reentry' if signal.is_reentry else 'new'})"
            )

            # Execute through trading service
            success = await self._context.trading_service.execute_decision(decision)

            if success:
                trade_succeeded = True
                self._stats["signals_executed"] += 1
                if signal.is_reentry:
                    self._stats["reentries"] += 1
                self._last_execution_time = time.time()

                state = self._market_states.get(signal.market_ticker)
                if state:
                    state.position_contracts += scaled_quantity
                    if not signal.is_reentry:
                        state.entry_yes_ratio = signal.yes_ratio
                        state.entry_price_drop = signal.price_drop

                self._record_decision(
                    signal_id=signal_id,
                    action=action_type,
                    reason=f"Bought {scaled_quantity} NO @ {entry_price or 'market'}c ({scale_label})",
                    signal_data=signal.to_dict()
                )

                await self._context.event_bus.emit_system_activity(
                    activity_type="rlm_execute",
                    message=f"RLM Executed: {signal.market_ticker} {scaled_quantity} NO ({scale_label})",
                    metadata={
                        "signal_id": signal_id,
                        "status": "success",
                        "market": signal.market_ticker,
                        "side": "no",
                        "quantity": scaled_quantity,
                        "price": entry_price,
                        "is_reentry": signal.is_reentry,
                        "scale": scale_label,
                    }
                )
            else:
                self._stats["signals_skipped"] += 1
                self._record_decision(
                    signal_id=signal_id,
                    action="failed",
                    reason=f"Order execution failed",
                    signal_data=signal.to_dict()
                )

        except Exception as e:
            self._stats["signals_skipped"] += 1
            logger.error(f"Error executing RLM signal: {e}")
            self._record_decision(
                signal_id=signal_id,
                action="error",
                reason=f"Exception: {str(e)}",
                signal_data=signal.to_dict()
            )

        finally:
            self._reset_signal_state(signal.market_ticker, preserve_entry=trade_succeeded)

    def _calculate_no_entry_price(
        self,
        best_no_ask: int,
        best_no_bid: int,
        spread: int,
        aggressive: bool = False
    ) -> int:
        """Calculate optimal entry price for buying NO contracts."""
        if spread <= self._tight_spread:
            if aggressive:
                return best_no_ask
            return best_no_ask - 1
        elif spread <= self._normal_spread:
            if aggressive:
                return best_no_ask - 1
            return (best_no_ask + best_no_bid) // 2
        else:
            if aggressive:
                return best_no_bid + (spread * 85 // 100)
            return best_no_bid + (spread * 3 // 4)

    def _log_orderbook_at_signal(self, signal: RLMSignal, orderbook: Dict[str, Any]) -> None:
        """Log orderbook state at signal time for analysis."""
        try:
            no_bids = orderbook.get("no_bids", {})
            yes_bids = orderbook.get("yes_bids", {})
            no_asks = orderbook.get("no_asks", {})

            log_data = {
                "signal_market": signal.market_ticker,
                "signal_yes_ratio": signal.yes_ratio,
                "signal_price_drop": signal.price_drop,
                "no_bid_volume": sum(no_bids.values()) if no_bids else 0,
                "yes_bid_volume": sum(yes_bids.values()) if yes_bids else 0,
                "no_spread": (min(no_asks.keys()) - max(no_bids.keys())) if no_asks and no_bids else None,
                "top_no_bid_size": list(no_bids.values())[0] if no_bids else 0,
                "timestamp": time.time(),
            }

            logger.debug(f"RLM orderbook at signal: {log_data}")

        except Exception as e:
            logger.debug(f"Error logging orderbook: {e}")

    def _reset_signal_state(self, market_ticker: str, preserve_entry: bool = True) -> None:
        """Reset signal detection state after trade attempt."""
        state = self._market_states.get(market_ticker)
        if state:
            state.yes_trades = 0
            state.no_trades = 0
            state.last_yes_price = None
            state.last_trade_time = None

            if not preserve_entry:
                state.entry_yes_ratio = None
                state.entry_price_drop = None

            logger.debug(
                f"Reset RLM signal state for {market_ticker} (preserve_entry={preserve_entry}, "
                f"first_yes_price={state.first_yes_price}c preserved from market open)"
            )

    def _record_decision(
        self,
        signal_id: str,
        action: str,
        reason: str,
        order_id: Optional[str] = None,
        signal_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a signal decision in history."""
        decision = RLMDecision(
            signal_id=signal_id,
            timestamp=time.time(),
            action=action,
            reason=reason,
            order_id=order_id,
            signal_data=signal_data,
        )
        self._decision_history.append(decision)

    # ============================================================
    # Rate Limiting
    # ============================================================

    def _refill_tokens(self) -> None:
        """Refill rate limit tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self._last_refill_time

        if elapsed >= self._token_refill_seconds:
            tokens_to_add = elapsed / self._token_refill_seconds
            self._tokens = min(self._max_tokens, self._tokens + tokens_to_add)
            self._last_refill_time = now

    def _consume_token(self) -> bool:
        """Consume a rate limit token. Returns True if available."""
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False

    def _should_emit_state_update(self, market_ticker: str) -> bool:
        """Check if we should emit a state update for this market."""
        last_emit = self._last_state_emit.get(market_ticker, 0.0)
        return (time.time() - last_emit) >= self._state_emit_interval

    # ============================================================
    # Signal Broadcasting
    # ============================================================

    async def _run_signal_broadcaster(self) -> None:
        """Periodically broadcast orderbook signals for all tracked markets."""
        logger.info(f"Starting orderbook signal broadcaster (interval={self._signal_broadcast_interval}s)")

        while self._running:
            try:
                await asyncio.sleep(self._signal_broadcast_interval)
                if not self._running:
                    break
                await self._broadcast_orderbook_signals()
            except asyncio.CancelledError:
                logger.info("Signal broadcaster task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in signal broadcaster: {e}")

        logger.info("Signal broadcaster stopped")

    async def _broadcast_orderbook_signals(self) -> None:
        """Emit orderbook signals for all tracked markets with data."""
        if not self._context or not self._context.orderbook_integration:
            return

        if not self._context.tracked_markets:
            return

        tracked_tickers = self._context.tracked_markets.get_active_tickers()

        broadcast_count = 0
        for market_ticker in tracked_tickers:
            signals = self._context.orderbook_integration.get_orderbook_signals(market_ticker)
            if signals and signals.get("snapshot_count", 0) > 0:
                state = self._market_states.get(market_ticker)
                if state:
                    state_dict = state.to_dict()
                else:
                    state_dict = {
                        "market_ticker": market_ticker,
                        "yes_trades": 0,
                        "no_trades": 0,
                        "total_trades": 0,
                        "yes_ratio": 0.0,
                    }
                state_dict["orderbook_signals"] = signals
                await self._context.event_bus.emit_rlm_market_update(
                    market_ticker=market_ticker,
                    state=state_dict,
                )
                broadcast_count += 1

        if broadcast_count > 0:
            logger.debug(f"Broadcast orderbook signals for {broadcast_count} markets")

    # ============================================================
    # Public Query Methods (for RLMService compatibility)
    # ============================================================

    def get_market_states(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get current market states for monitoring."""
        states = list(self._market_states.values())
        states.sort(key=lambda s: s.total_trades, reverse=True)

        result = []
        for state in states[:limit]:
            state_dict = state.to_dict()
            if self._context and self._context.orderbook_integration:
                orderbook_signals = self._context.orderbook_integration.get_orderbook_signals(
                    state.market_ticker
                )
                if orderbook_signals:
                    state_dict["orderbook_signals"] = orderbook_signals
            result.append(state_dict)

        return result

    def get_decision_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent decision history."""
        decisions = list(self._decision_history)
        decisions.reverse()
        return [d.to_dict() for d in decisions[:limit]]

    def get_recent_tracked_trades(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent tracked trades for Trade Processing panel."""
        trades = list(self._recent_tracked_trades)
        trades.reverse()
        return [t.to_dict() for t in trades[:limit]]
