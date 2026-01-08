"""
ODMR Strategy Plugin - Orderbook-Driven Mean Reversion for TRADER V3.

This plugin implements the validated +85% win rate dip buying strategy with
optional ODMR enhancements (whale filter + orderbook filtering).

Key Mechanics:
    - Signal: Price drops >=10c from rolling high (all-time high YES price)
    - Entry: Buy YES when dip detected, price bounds 15c-85c, >=5 trades
    - Exit: +5c target (limit), 15min timeout (market), -40c stop (market)
    - Wide stop loss validated - tight stops hurt performance

ODMR Enhancements (optional, for A/B testing):
    - Tier 1 Whale Filter: Require recent whale YES trade (2x avg size)
    - Tier 2 Orderbook Filter: Stricter spread (2c), bid imbalance required

Validated Backtest Results:
    - 85.1% win rate on 85,189 journeys
    - +$4,492.88 total P&L (profit factor 2.26)
    - 82% of dips hit +5c recovery target
    - Average +5.3c per journey

Architecture Position:
    - Registered with StrategyRegistry as "odmr"
    - Subscribes to PUBLIC_TRADE_RECEIVED events for dip detection
    - Subscribes to ORDERBOOK_SNAPSHOT events for spread checking
    - Subscribes to ORDER_FILL events for entry/exit tracking
    - Subscribes to MARKET_DETERMINED events for cleanup
    - Creates TradingDecision objects with strategy_id="odmr"
    - Delegates execution to TradingDecisionService
    - Background task monitors timeout and stop loss conditions

Reference: research/backtest/strategies/odmr_analysis.md
"""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Set, TYPE_CHECKING

from ..protocol import Strategy, StrategyContext
from ..registry import StrategyRegistry
from ...core.events import EventType
from ...core.event_bus import (
    PublicTradeEvent,
    MarketDeterminedEvent,
    OrderFillEvent,
)
from ...core.state_machine import TraderState
from ....data.orderbook_state import get_shared_orderbook_state
from ...state.order_context import OrderbookContext
from ...state.trading_attachment import TrackedMarketOrder

if TYPE_CHECKING:
    from ...services.trading_decision_service import TradingDecisionService, TradingDecision
    from ...core.state_container import V3StateContainer
    from ...state.tracked_markets import TrackedMarketsState
    from ...clients.orderbook_integration import V3OrderbookIntegration
    from ...core.event_bus import EventBus

logger = logging.getLogger("kalshiflow_rl.traderv3.strategies.plugins.odmr")

# Decision history size
DECISION_HISTORY_SIZE = 100

# Position monitor interval
POSITION_MONITOR_INTERVAL = 1.0  # Check every 1 second

# Tracked trades buffer size (for Trade Processing panel)
RECENT_TRACKED_TRADES_SIZE = 30


@dataclass
class DipMarketState:
    """
    Per-market state tracking for dip detection and position management.

    Tracks:
        - Rolling high YES price (all-time high observed)
        - Current price and trade count
        - Position state for active exits
        - ODMR trade history for whale detection
    """
    market_ticker: str
    rolling_high: int = 0           # All-time high YES price observed
    current_price: int = 0          # Latest trade YES price
    trade_count: int = 0            # Total trades observed
    first_trade_time: Optional[float] = None  # When we started observing

    # Position state (when we have an open position)
    position_open: bool = False
    entry_price: int = 0            # YES entry price in cents
    entry_time: Optional[datetime] = None
    contracts_held: int = 0
    exit_order_id: Optional[str] = None  # Target exit order (limit)
    target_exit_price: int = 0       # Entry + recovery target

    # Exit retry tracking
    exit_retry_count: int = 0                       # Number of exit retries attempted
    last_exit_attempt_time: Optional[datetime] = None  # When we last tried to exit

    # Signal tracking
    dip_detected_at: Optional[float] = None
    dip_depth: int = 0               # How much price dropped from high

    # ODMR Trade History Tracking (for whale detection)
    recent_trades: deque = field(default_factory=lambda: deque(maxlen=20))
    trade_size_history: deque = field(default_factory=lambda: deque(maxlen=50))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/transport."""
        return {
            "market_ticker": self.market_ticker,
            "rolling_high": self.rolling_high,
            "current_price": self.current_price,
            "trade_count": self.trade_count,
            "position_open": self.position_open,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "contracts_held": self.contracts_held,
            "target_exit_price": self.target_exit_price,
            "dip_depth": self.dip_depth,
            "exit_retry_count": self.exit_retry_count,
            "last_exit_attempt_time": self.last_exit_attempt_time.isoformat() if self.last_exit_attempt_time else None,
        }


@dataclass
class TradeRecord:
    """
    Lightweight trade record for ODMR whale detection.

    Tracks individual trades for calculating whale thresholds
    (trades significantly larger than average).
    """
    timestamp: float
    taker_side: str  # "yes" or "no"
    count: int
    yes_price: int


@dataclass
class TrackedTrade:
    """
    Trade record for tracked markets - matches RLM NO pattern for UI consistency.

    Used for the Trade Processing panel in the frontend to show recent
    trade activity in tracked markets.
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
class DipSignal:
    """
    Represents a detected dip buying signal.

    Attributes:
        market_ticker: Market ticker for this signal
        rolling_high: YES price high before dip
        current_price: Current YES price (dipped)
        dip_depth: How many cents price dropped
        trade_count: Total trades observed
        detected_at: Timestamp when signal was detected
    """
    market_ticker: str
    rolling_high: int
    current_price: int
    dip_depth: int
    trade_count: int
    detected_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/transport."""
        return {
            "market_ticker": self.market_ticker,
            "rolling_high": self.rolling_high,
            "current_price": self.current_price,
            "dip_depth": self.dip_depth,
            "trade_count": self.trade_count,
            "detected_at": self.detected_at,
            "age_seconds": int(time.time() - self.detected_at),
        }


@dataclass
class DipDecision:
    """
    Records a decision about a dip signal or exit.

    Attributes:
        signal_id: Unique identifier
        timestamp: When decision was made
        action: Decision action ("entry", "exit_target", "exit_timeout", "exit_stop", etc.)
        reason: Human-readable explanation
        order_id: Kalshi order ID if executed
        signal_data: Original signal data for reference
        strategy_id: Strategy identifier for frontend filtering
    """
    signal_id: str
    timestamp: float
    action: str
    reason: str
    order_id: Optional[str] = None
    signal_data: Optional[Dict[str, Any]] = None
    pnl_cents: Optional[int] = None
    strategy_id: str = "odmr"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for WebSocket transport."""
        result = {
            "signal_id": self.signal_id,
            "timestamp": self.timestamp,
            "action": self.action,
            "reason": self.reason,
            "order_id": self.order_id[:8] if self.order_id else None,
            "age_seconds": int(time.time() - self.timestamp),
            "strategy_id": self.strategy_id,
        }
        if self.signal_data:
            result["market_ticker"] = self.signal_data.get("market_ticker", "")
            result["dip_depth"] = self.signal_data.get("dip_depth", 0)
        if self.pnl_cents is not None:
            result["pnl_cents"] = self.pnl_cents
        return result


@StrategyRegistry.register("odmr")
class ODMRStrategy:
    """
    ODMR Strategy Plugin - Orderbook-Driven Mean Reversion Trading.

    Implements the validated +85.1% win rate strategy: buy YES when price
    dips from rolling high, exit on recovery target or timeout.

    Optional ODMR enhancements (configurable for A/B testing):
        - Tier 1 Whale Filter: Require recent whale YES trade
        - Tier 2 Orderbook Filter: Stricter spread + bid imbalance

    Attributes:
        name: "odmr"
        display_name: "ODMR (Orderbook-Driven Mean Reversion)"
        subscribed_events: PUBLIC_TRADE_RECEIVED, ORDER_FILL, MARKET_DETERMINED

    Key Features:
        - Rolling high tracking from public trade feed
        - Dip detection when price drops >=10c from high
        - Target exit at +5c from entry (limit order)
        - Timeout exit at 15 minutes (market order)
        - Wide stop loss at -40c (validated as optimal)
        - Background position monitor for timeout/stop
    """

    name: str = "odmr"
    display_name: str = "ODMR (Orderbook-Driven Mean Reversion)"
    subscribed_events: Set[EventType] = {
        EventType.PUBLIC_TRADE_RECEIVED,
        EventType.ORDER_FILL,
        EventType.MARKET_DETERMINED,
        EventType.ORDERBOOK_SNAPSHOT,
    }

    def __init__(self):
        """Initialize the ODMR strategy."""
        self._context: Optional[StrategyContext] = None
        self._running: bool = False
        self._started_at: Optional[float] = None

        # Signal detection parameters (loaded from config)
        self._dip_threshold_cents: int = 10        # Minimum dip from rolling high
        self._min_trades_before_dip: int = 5       # Minimum trades before signal valid
        self._price_floor_cents: int = 15          # Minimum YES price to enter
        self._price_ceiling_cents: int = 85        # Maximum YES price to enter

        # Exit parameters
        self._recovery_target_cents: int = 5       # +5c target from entry
        self._timeout_seconds: int = 900           # 15 minutes
        self._stop_loss_cents: int = 40            # Wide stop (validated optimal)

        # Position sizing
        self._entry_contracts: int = 10            # Contracts per entry
        self._entry_spread_max_cents: int = 8      # Max spread to enter

        # Risk limits
        self._max_positions: int = 5               # Max concurrent positions
        self._max_exposure_cents: int = 5000       # $50 max total exposure
        self._max_per_market: int = 20             # Max contracts per market

        # ODMR Tier 1: Whale Filter (off by default for A/B testing)
        self._enable_whale_filter: bool = False
        self._whale_multiplier: float = 2.0       # Trade is "whale" if >= 2x avg
        self._whale_lookback: int = 5             # Check last N trades
        self._min_history_for_whale: int = 10     # Min history to calculate avg

        # ODMR Tier 2: Orderbook Filter (off by default for A/B testing)
        self._use_orderbook_filtering: bool = False
        self._odmr_spread_max_cents: int = 2      # Stricter spread for ODMR
        self._require_bid_imbalance: bool = True  # Require bid_depth > ask_depth

        # Per-market state tracking
        self._market_states: Dict[str, DipMarketState] = {}

        # Track markets with open positions
        self._open_positions: Set[str] = set()

        # Decision history
        self._decision_history: deque[DipDecision] = deque(maxlen=DECISION_HISTORY_SIZE)

        # Statistics
        self._stats = {
            "dips_detected": 0,
            "entries_attempted": 0,
            "entries_filled": 0,
            "exits_target": 0,
            "exits_timeout": 0,
            "exits_stop_loss": 0,
            "exits_settlement": 0,
            "trades_processed": 0,
            "trades_filtered": 0,
            # ODMR Tier 1: Whale filter stats
            "whale_filter_passed": 0,
            "whale_filter_rejected": 0,
            "whale_filter_skipped_insufficient_history": 0,
            # ODMR Tier 2: Orderbook filter stats
            "orderbook_filter_passed": 0,
            "orderbook_filter_rejected_spread": 0,
            "orderbook_filter_rejected_imbalance": 0,
            "orderbook_filter_skipped_unavailable": 0,
            # Exit order management stats
            "exit_order_cancellations": 0,
            "exit_order_cancel_failures": 0,
            "exit_retries_triggered": 0,
            "exit_max_retries_reached": 0,
        }

        # P&L tracking
        self._session_pnl_cents: int = 0
        self._winning_trades: int = 0
        self._losing_trades: int = 0

        # Background task for position monitoring
        self._position_monitor_task: Optional[asyncio.Task] = None

        # Processing lock
        self._processing_lock = asyncio.Lock()

        # State broadcast throttling
        self._last_state_emit: Dict[str, float] = {}
        self._state_emit_interval = 0.5

        # Trade counter for unique IDs (TrackedTrade mechanism)
        self._trade_counter: int = 0

        # Recent tracked trades buffer (for Trade Processing panel)
        self._recent_tracked_trades: deque[TrackedTrade] = deque(maxlen=RECENT_TRACKED_TRADES_SIZE)

        logger.debug("ODMRStrategy initialized")

    async def start(self, context: StrategyContext) -> None:
        """
        Start the ODMR strategy.

        Loads configuration, subscribes to events, and begins processing.

        Args:
            context: Shared strategy context
        """
        if self._running:
            logger.warning("ODMRStrategy is already running")
            return

        self._context = context
        self._running = True
        self._started_at = time.time()

        # Load configuration from coordinator if available
        self._load_config_from_coordinator()

        # Subscribe to public trade events for dip detection
        await context.event_bus.subscribe_to_public_trade(self._handle_public_trade)
        logger.info("Subscribed to PUBLIC_TRADE_RECEIVED events")

        # Subscribe to order fill events for entry/exit tracking
        await context.event_bus.subscribe_to_order_fill(self._handle_order_fill)
        logger.info("Subscribed to ORDER_FILL events")

        # Subscribe to market determined events for cleanup
        await context.event_bus.subscribe_to_market_determined(self._handle_market_determined)
        logger.info("Subscribed to MARKET_DETERMINED events for cleanup")

        # Start position monitor background task
        self._position_monitor_task = asyncio.create_task(
            self._position_monitor_loop(),
            name="odmr_position_monitor"
        )
        logger.info("Started position monitor background task")

        # Emit startup event
        await context.event_bus.emit_system_activity(
            activity_type="strategy_start",
            message=f"ODMR strategy started",
            metadata={
                "strategy": "odmr",
                "dip_threshold": f"{self._dip_threshold_cents}c",
                "recovery_target": f"+{self._recovery_target_cents}c",
                "timeout": f"{self._timeout_seconds}s",
                "stop_loss": f"-{self._stop_loss_cents}c",
                "max_positions": self._max_positions,
            }
        )

        logger.info(
            f"ODMRStrategy started: dip_threshold={self._dip_threshold_cents}c, "
            f"min_trades={self._min_trades_before_dip}, price_bounds={self._price_floor_cents}-{self._price_ceiling_cents}c, "
            f"target=+{self._recovery_target_cents}c, timeout={self._timeout_seconds}s, stop=-{self._stop_loss_cents}c"
        )

    def _load_config_from_coordinator(self) -> None:
        """Load configuration from strategy coordinator if available."""
        if not self._context:
            return

        if not self._context.config:
            logger.warning(
                "No config in context - using HARDCODED DEFAULTS! "
                f"dip_threshold={self._dip_threshold_cents}c, contracts={self._entry_contracts}"
            )
            return

        config = self._context.config
        params = config.params

        # Load signal detection parameters
        self._dip_threshold_cents = params.get("dip_threshold_cents", self._dip_threshold_cents)
        self._min_trades_before_dip = params.get("min_trades_before_dip", self._min_trades_before_dip)
        self._price_floor_cents = params.get("price_floor_cents", self._price_floor_cents)
        self._price_ceiling_cents = params.get("price_ceiling_cents", self._price_ceiling_cents)

        # Load exit parameters
        self._recovery_target_cents = params.get("recovery_target_cents", self._recovery_target_cents)
        self._timeout_seconds = params.get("timeout_seconds", self._timeout_seconds)
        self._stop_loss_cents = params.get("stop_loss_cents", self._stop_loss_cents)

        # Load position sizing
        self._entry_contracts = params.get("entry_contracts", self._entry_contracts)
        self._entry_spread_max_cents = params.get("entry_spread_max_cents", self._entry_spread_max_cents)

        # Load risk limits from config top-level
        self._max_positions = config.max_positions
        self._max_exposure_cents = config.max_exposure_cents
        self._max_per_market = params.get("max_per_market", self._max_per_market)

        # Load ODMR Tier 1: Whale filter parameters
        self._enable_whale_filter = params.get("enable_whale_filter", self._enable_whale_filter)
        self._whale_multiplier = params.get("whale_multiplier", self._whale_multiplier)
        self._whale_lookback = params.get("whale_lookback", self._whale_lookback)
        self._min_history_for_whale = params.get("min_history_for_whale", self._min_history_for_whale)

        # Load ODMR Tier 2: Orderbook filter parameters
        self._use_orderbook_filtering = params.get("use_orderbook_filtering", self._use_orderbook_filtering)
        self._odmr_spread_max_cents = params.get("odmr_spread_max_cents", self._odmr_spread_max_cents)
        self._require_bid_imbalance = params.get("require_bid_imbalance", self._require_bid_imbalance)

        logger.info(
            f"Loaded config from YAML: {config.name} "
            f"(dip={self._dip_threshold_cents}c, target=+{self._recovery_target_cents}c, "
            f"timeout={self._timeout_seconds}s, stop=-{self._stop_loss_cents}c, "
            f"contracts={self._entry_contracts}, max_positions={self._max_positions}, "
            f"whale_filter={self._enable_whale_filter}, orderbook_filter={self._use_orderbook_filtering})"
        )

    async def stop(self) -> None:
        """
        Stop the ODMR strategy.

        Unsubscribes from events and cleans up resources.
        """
        if not self._running:
            return

        logger.info("Stopping ODMRStrategy...")
        self._running = False

        # Cancel position monitor task
        if self._position_monitor_task and not self._position_monitor_task.done():
            self._position_monitor_task.cancel()
            try:
                await self._position_monitor_task
            except asyncio.CancelledError:
                pass
            logger.debug("Position monitor task cancelled")
        self._position_monitor_task = None

        # Unsubscribe from EventBus
        if self._context and self._context.event_bus:
            self._context.event_bus.unsubscribe(
                EventType.PUBLIC_TRADE_RECEIVED, self._handle_public_trade
            )
            self._context.event_bus.unsubscribe(
                EventType.ORDER_FILL, self._handle_order_fill
            )
            self._context.event_bus.unsubscribe(
                EventType.MARKET_DETERMINED, self._handle_market_determined
            )
            logger.debug("Unsubscribed from all events")

            # Emit shutdown event
            await self._context.event_bus.emit_system_activity(
                activity_type="strategy_stop",
                message=f"ODMR strategy stopped",
                metadata={
                    "strategy": "odmr",
                    "dips_detected": self._stats["dips_detected"],
                    "entries_filled": self._stats["entries_filled"],
                    "session_pnl_cents": self._session_pnl_cents,
                }
            )

        logger.info(
            f"ODMRStrategy stopped. Stats: "
            f"dips={self._stats['dips_detected']}, "
            f"entries={self._stats['entries_filled']}, "
            f"pnl=${self._session_pnl_cents / 100:.2f}"
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

        Returns:
            Dictionary with strategy statistics
        """
        uptime = time.time() - self._started_at if self._started_at else 0
        total_exits = (
            self._stats["exits_target"] +
            self._stats["exits_timeout"] +
            self._stats["exits_stop_loss"] +
            self._stats["exits_settlement"]
        )
        win_rate = (
            self._winning_trades / (self._winning_trades + self._losing_trades)
            if (self._winning_trades + self._losing_trades) > 0
            else 0.0
        )

        return {
            "name": self.name,
            "display_name": self.display_name,
            "running": self._running,
            "uptime_seconds": uptime,
            # Core stats
            "dips_detected": self._stats["dips_detected"],
            "entries_attempted": self._stats["entries_attempted"],
            "entries_filled": self._stats["entries_filled"],
            # Aliases for aggregation compatibility (coordinator expects these)
            "signals_detected": self._stats["dips_detected"],
            "signals_executed": self._stats["entries_filled"],
            "exits_target": self._stats["exits_target"],
            "exits_timeout": self._stats["exits_timeout"],
            "exits_stop_loss": self._stats["exits_stop_loss"],
            "exits_settlement": self._stats["exits_settlement"],
            "total_exits": total_exits,
            # Trade processing stats
            "trades_processed": self._stats["trades_processed"],
            "trades_filtered": self._stats["trades_filtered"],
            # P&L stats
            "session_pnl_cents": self._session_pnl_cents,
            "winning_trades": self._winning_trades,
            "losing_trades": self._losing_trades,
            "win_rate": round(win_rate, 3),
            # Position stats
            "open_positions": len(self._open_positions),
            "markets_tracking": len(self._market_states),
            # Config
            "config": {
                "dip_threshold_cents": self._dip_threshold_cents,
                "min_trades_before_dip": self._min_trades_before_dip,
                "price_floor_cents": self._price_floor_cents,
                "price_ceiling_cents": self._price_ceiling_cents,
                "recovery_target_cents": self._recovery_target_cents,
                "timeout_seconds": self._timeout_seconds,
                "stop_loss_cents": self._stop_loss_cents,
                "entry_contracts": self._entry_contracts,
                "max_positions": self._max_positions,
            },
            # ODMR-specific stats
            "odmr": {
                "enabled": {
                    "whale_filter": self._enable_whale_filter,
                    "orderbook_filtering": self._use_orderbook_filtering,
                },
                "whale_filter": {
                    "passed": self._stats["whale_filter_passed"],
                    "rejected": self._stats["whale_filter_rejected"],
                    "skipped_insufficient_history": self._stats["whale_filter_skipped_insufficient_history"],
                },
                "orderbook_filter": {
                    "passed": self._stats["orderbook_filter_passed"],
                    "rejected_spread": self._stats["orderbook_filter_rejected_spread"],
                    "rejected_imbalance": self._stats["orderbook_filter_rejected_imbalance"],
                    "skipped_unavailable": self._stats["orderbook_filter_skipped_unavailable"],
                },
                "exit_management": {
                    "order_cancellations": self._stats["exit_order_cancellations"],
                    "cancel_failures": self._stats["exit_order_cancel_failures"],
                    "retries_triggered": self._stats["exit_retries_triggered"],
                    "max_retries_reached": self._stats["exit_max_retries_reached"],
                },
            },
        }

    # ============================================================
    # Event Handlers
    # ============================================================

    async def _handle_public_trade(self, trade_event: PublicTradeEvent) -> None:
        """Handle public trade events for dip detection."""
        if not self._running or not self._context:
            return

        # Wait for system to be fully ready
        machine_state = self._context.state_container.machine_state
        if machine_state != TraderState.READY:
            logger.debug(f"Skipping ODMR processing - system not ready (state={machine_state})")
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

    async def _handle_order_fill(self, event: OrderFillEvent) -> None:
        """Handle order fill events for entry/exit tracking."""
        if not self._running or not self._context:
            return

        ticker = event.market_ticker
        state = self._market_states.get(ticker)
        if not state:
            return

        # Check if this is our entry fill (buying YES)
        if event.action == "buy" and event.side == "yes" and not state.position_open:
            # Entry filled
            state.position_open = True
            state.entry_price = event.price_cents
            state.entry_time = datetime.now(timezone.utc)
            state.contracts_held = event.count
            state.target_exit_price = state.entry_price + self._recovery_target_cents

            self._open_positions.add(ticker)
            self._stats["entries_filled"] += 1

            logger.info(
                f"ODMR: Entry FILLED for {ticker} - "
                f"{event.count} YES @ {event.price_cents}c, target={state.target_exit_price}c, "
                f"stop={state.entry_price - self._stop_loss_cents}c"
            )

            # Place target exit order (limit sell YES at target price)
            await self._place_target_exit(state)

        # Check if this is our exit fill (selling YES)
        elif event.action == "sell" and event.side == "yes" and state.position_open:
            # Exit filled - calculate P&L for this fill
            exit_price = event.price_cents
            filled_count = event.count
            pnl_cents = (exit_price - state.entry_price) * filled_count
            self._session_pnl_cents += pnl_cents

            # Track win/loss
            if pnl_cents >= 0:
                self._winning_trades += 1
            else:
                self._losing_trades += 1

            # Determine exit type
            exit_type = "unknown"
            if exit_price >= state.target_exit_price:
                exit_type = "target"
                self._stats["exits_target"] += 1
            elif exit_price <= state.entry_price - self._stop_loss_cents:
                exit_type = "stop_loss"
                self._stats["exits_stop_loss"] += 1
            else:
                # Could be timeout or manual
                exit_type = "timeout"
                self._stats["exits_timeout"] += 1

            # Handle partial fills correctly
            remaining_contracts = state.contracts_held - filled_count

            if remaining_contracts <= 0:
                # Fully exited
                logger.info(
                    f"ODMR: Exit COMPLETE ({exit_type}) for {ticker} - "
                    f"{filled_count} contracts @ {exit_price}c, PnL={pnl_cents:+d}c, "
                    f"session_pnl=${self._session_pnl_cents / 100:.2f}"
                )

                # Record decision with signal_data for UI display (MARKET/DIP columns)
                self._record_decision(
                    signal_id=f"{ticker}:exit:{int(time.time() * 1000)}",
                    action=f"exit_{exit_type}",
                    reason=f"Exit at {exit_price}c (entry={state.entry_price}c)",
                    pnl_cents=pnl_cents,
                    signal_data={
                        "market_ticker": state.market_ticker,
                        "dip_depth": state.dip_depth,
                        "entry_price": state.entry_price,
                        "rolling_high": state.rolling_high,
                    },
                )

                # Emit activity event
                await self._context.event_bus.emit_system_activity(
                    activity_type=f"odmr_exit_{exit_type}",
                    message=f"ODMR Exit ({exit_type}): {ticker} @ {exit_price}c, PnL={pnl_cents:+d}c",
                    metadata={
                        "market": ticker,
                        "exit_price": exit_price,
                        "entry_price": state.entry_price,
                        "pnl_cents": pnl_cents,
                        "exit_type": exit_type,
                        "session_pnl_cents": self._session_pnl_cents,
                    }
                )

                # Reset position state
                self._reset_position_state(state)
            else:
                # Partial fill - keep tracking remaining position
                logger.warning(
                    f"ODMR: Exit PARTIAL ({exit_type}) for {ticker} - "
                    f"{filled_count}/{state.contracts_held} contracts filled @ {exit_price}c, "
                    f"{remaining_contracts} remaining, PnL so far={pnl_cents:+d}c"
                )
                state.contracts_held = remaining_contracts

                # Verify sync with trading state after partial fill
                await self._verify_position_for_exit(state)
                # Note: If mismatch, state.contracts_held is already corrected by the verify method

    async def _handle_market_determined(self, event: MarketDeterminedEvent) -> None:
        """Handle market determined events for cleanup."""
        ticker = event.market_ticker
        cleaned = False

        state = self._market_states.get(ticker)
        if state:
            # If we had an open position, record as settlement exit
            if state.position_open:
                self._stats["exits_settlement"] += 1
                # Note: Actual P&L from settlement comes through position updates
                logger.info(f"ODMR position settled: {ticker}")

            del self._market_states[ticker]
            cleaned = True

        if ticker in self._open_positions:
            self._open_positions.discard(ticker)
            cleaned = True

        if ticker in self._last_state_emit:
            del self._last_state_emit[ticker]
            cleaned = True

        if cleaned:
            logger.info(f"ODMRStrategy cleanup for determined market: {ticker}")

    # ============================================================
    # Trade Processing
    # ============================================================

    async def _process_trade(self, trade_event: PublicTradeEvent) -> None:
        """Process a public trade and update market state for dip detection."""
        self._stats["trades_processed"] += 1
        market_ticker = trade_event.market_ticker

        # Get or create market state
        if market_ticker not in self._market_states:
            self._market_states[market_ticker] = DipMarketState(
                market_ticker=market_ticker
            )

        state = self._market_states[market_ticker]

        # Calculate YES price from trade
        yes_price = trade_event.price_cents if trade_event.side == "yes" else (100 - trade_event.price_cents)

        # Update state
        state.current_price = yes_price
        state.trade_count += 1

        if state.first_trade_time is None:
            state.first_trade_time = time.time()

        # Update rolling high
        if yes_price > state.rolling_high:
            state.rolling_high = yes_price

        # Track trade history for ODMR whale detection
        trade_record = TradeRecord(
            timestamp=time.time(),
            taker_side=trade_event.side,
            count=trade_event.count,
            yes_price=yes_price,
        )
        state.recent_trades.append(trade_record)
        state.trade_size_history.append(trade_event.count)

        # Store in tracked trades buffer (for Trade Processing panel / UI display)
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

        # If we already have a position, check for stop loss trigger
        if state.position_open:
            stop_price = state.entry_price - self._stop_loss_cents
            if yes_price <= stop_price:
                logger.info(
                    f"ODMR Stop loss triggered: {market_ticker} "
                    f"current={yes_price}c <= stop={stop_price}c"
                )
                await self._execute_stop_loss(state)
            return

        # Check for dip signal (only if no position)
        signal = self._detect_dip_signal(state)
        if signal:
            await self._execute_entry(signal)

    def _detect_dip_signal(self, state: DipMarketState) -> Optional[DipSignal]:
        """Detect if market state meets dip signal criteria."""
        if not self._context:
            return None

        # Already have a position in this market
        if state.position_open or state.market_ticker in self._open_positions:
            return None

        # Check minimum trades
        if state.trade_count < self._min_trades_before_dip:
            return None

        # Check price bounds
        if state.current_price < self._price_floor_cents:
            return None
        if state.current_price > self._price_ceiling_cents:
            return None

        # Calculate dip depth
        dip_depth = state.rolling_high - state.current_price

        # Check if dip is significant enough
        if dip_depth < self._dip_threshold_cents:
            return None

        # Check max concurrent positions
        if len(self._open_positions) >= self._max_positions:
            logger.debug(
                f"ODMR max positions reached ({len(self._open_positions)}/{self._max_positions}), "
                f"skipping dip signal for {state.market_ticker}"
            )
            return None

        # ODMR Tier 1: Whale YES filter (optional, for A/B testing)
        if self._enable_whale_filter:
            whale_result = self._check_whale_yes_filter(state)
            if whale_result == "insufficient_history":
                self._stats["whale_filter_skipped_insufficient_history"] += 1
                logger.debug(
                    f"ODMR: Whale filter skipped (insufficient history) for {state.market_ticker}"
                )
                return None
            elif whale_result == "no_whale":
                self._stats["whale_filter_rejected"] += 1
                logger.info(
                    f"ODMR: Whale filter REJECTED for {state.market_ticker} - no whale YES trade in recent history"
                )
                return None
            else:
                self._stats["whale_filter_passed"] += 1
                logger.info(
                    f"ODMR: Whale filter PASSED for {state.market_ticker}"
                )

        # Signal detected
        self._stats["dips_detected"] += 1

        logger.info(
            f"ODMR: Dip signal DETECTED for {state.market_ticker} - "
            f"depth={dip_depth}c (high={state.rolling_high}c -> current={state.current_price}c)"
        )

        return DipSignal(
            market_ticker=state.market_ticker,
            rolling_high=state.rolling_high,
            current_price=state.current_price,
            dip_depth=dip_depth,
            trade_count=state.trade_count,
        )

    def _check_whale_yes_filter(self, state: DipMarketState) -> str:
        """
        ODMR Tier 1: Check for whale YES trade in recent history.

        A whale trade is defined as >= whale_multiplier * average trade size.
        Only passes if a recent whale trade was on the YES side (informed buying).

        Returns:
            'passed': Whale YES trade detected
            'no_whale': No whale YES trade in lookback window
            'insufficient_history': Not enough trade history to calculate
        """
        if len(state.trade_size_history) < self._min_history_for_whale:
            return "insufficient_history"

        # Calculate average trade size
        avg_size = sum(state.trade_size_history) / len(state.trade_size_history)
        whale_threshold = avg_size * self._whale_multiplier

        # Check recent trades for whale YES
        recent_trades = list(state.recent_trades)[-self._whale_lookback:]
        for trade in recent_trades:
            if trade.taker_side == "yes" and trade.count >= whale_threshold:
                logger.debug(
                    f"ODMR whale YES detected: {state.market_ticker} "
                    f"trade={trade.count} >= threshold={whale_threshold:.1f} (avg={avg_size:.1f})"
                )
                return "passed"

        return "no_whale"

    def _check_yes_bid_imbalance(self, snapshot: Dict[str, Any]) -> str:
        """
        ODMR Tier 2: Check for bullish orderbook imbalance.

        Requires YES bid depth > YES ask depth, indicating more buyers
        than sellers at the current price level.

        Returns:
            'passed': Bullish imbalance (bid_depth > ask_depth)
            'rejected': Bearish imbalance (ask_depth >= bid_depth)
            'unavailable': Orderbook data not available
        """
        if not snapshot:
            return "unavailable"

        yes_bids = snapshot.get("yes_bids", {})
        yes_asks = snapshot.get("yes_asks", {})

        if not yes_bids or not yes_asks:
            return "unavailable"

        yes_bid_total = sum(yes_bids.values())
        yes_ask_total = sum(yes_asks.values())

        if yes_bid_total > yes_ask_total:
            logger.debug(
                f"ODMR bid imbalance passed: bids={yes_bid_total} > asks={yes_ask_total}"
            )
            return "passed"

        logger.debug(
            f"ODMR bid imbalance rejected: bids={yes_bid_total} <= asks={yes_ask_total}"
        )
        return "rejected"

    # ============================================================
    # Order Execution
    # ============================================================

    async def _execute_entry(self, signal: DipSignal) -> None:
        """Execute a dip entry by placing a YES buy order."""
        from ...services.trading_decision_service import TradingDecision, TradingStrategy

        if not self._context or not self._context.trading_service:
            logger.warning("Cannot execute entry - no trading service available")
            return

        signal_id = f"{signal.market_ticker}:{int(signal.detected_at * 1000)}"
        state = self._market_states.get(signal.market_ticker)
        if not state:
            return

        self._stats["entries_attempted"] += 1

        try:
            # Get orderbook for execution price and spread check
            entry_price = None
            current_spread: Optional[int] = None

            try:
                orderbook_state = await asyncio.wait_for(
                    get_shared_orderbook_state(signal.market_ticker),
                    timeout=2.0
                )
                snapshot = await orderbook_state.get_snapshot()

                ob_context = OrderbookContext.from_orderbook_snapshot(
                    snapshot,
                    tight_spread=2,
                    normal_spread=5,
                )

                # ODMR Tier 2: Use stricter spread when orderbook filtering enabled
                spread_max = (
                    self._odmr_spread_max_cents
                    if self._use_orderbook_filtering
                    else self._entry_spread_max_cents
                )

                # Check YES side spread (we're buying YES)
                current_spread = ob_context.yes_spread

                if current_spread is not None and current_spread > spread_max:
                    if self._use_orderbook_filtering:
                        self._stats["orderbook_filter_rejected_spread"] += 1
                    logger.info(
                        f"ODMR Entry skipped: {signal.market_ticker} spread {current_spread}c > {spread_max}c max"
                    )
                    self._record_decision(
                        signal_id=signal_id,
                        action="skipped_spread",
                        reason=f"Spread {current_spread}c > {spread_max}c max",
                        signal_data=signal.to_dict(),
                    )
                    return

                # ODMR Tier 2: Check bid imbalance (optional)
                if self._use_orderbook_filtering and self._require_bid_imbalance:
                    imbalance_result = self._check_yes_bid_imbalance(snapshot)
                    if imbalance_result == "unavailable":
                        self._stats["orderbook_filter_skipped_unavailable"] += 1
                        logger.debug(
                            f"ODMR bid imbalance skipped (unavailable): {signal.market_ticker}"
                        )
                        # Continue anyway - don't block on unavailable data
                    elif imbalance_result == "rejected":
                        self._stats["orderbook_filter_rejected_imbalance"] += 1
                        logger.info(
                            f"ODMR Entry skipped: {signal.market_ticker} bid imbalance rejected"
                        )
                        self._record_decision(
                            signal_id=signal_id,
                            action="skipped_imbalance",
                            reason="Bearish orderbook imbalance",
                            signal_data=signal.to_dict(),
                        )
                        return
                    else:
                        self._stats["orderbook_filter_passed"] += 1

                # Get YES best ask for entry price
                if ob_context.yes_best_ask is not None:
                    entry_price = ob_context.yes_best_ask
                else:
                    # Fallback to current trade price
                    entry_price = signal.current_price

            except (asyncio.TimeoutError, Exception) as e:
                logger.warning(f"Orderbook unavailable for {signal.market_ticker}: {e}, using trade price")
                entry_price = signal.current_price

            # Create trading decision
            decision = TradingDecision(
                action="buy",
                market=signal.market_ticker,
                side="yes",
                quantity=self._entry_contracts,
                price=entry_price,
                reason=f"ODMR dip: high={signal.rolling_high}c -> {signal.current_price}c (-{signal.dip_depth}c)",
                confidence=min(0.9, 0.6 + signal.dip_depth * 0.02),
                strategy=TradingStrategy.HOLD,  # Backward compat placeholder
                signal_params={
                    "rolling_high": signal.rolling_high,
                    "current_price": signal.current_price,
                    "dip_depth": signal.dip_depth,
                    "trade_count": signal.trade_count,
                    "signal_detected_at": signal.detected_at,
                },
            )
            decision.strategy_id = "odmr"

            # Update state before execution
            state.dip_detected_at = signal.detected_at
            state.dip_depth = signal.dip_depth

            logger.info(
                f"ODMR Entry signal: {signal.market_ticker} "
                f"high={signal.rolling_high}c -> current={signal.current_price}c (dip={signal.dip_depth}c), "
                f"entry_price={entry_price}c, contracts={self._entry_contracts}"
            )

            # Execute through trading service
            success = await self._context.trading_service.execute_decision(decision)

            if success:
                self._record_decision(
                    signal_id=signal_id,
                    action="entry_placed",
                    reason=f"Buy {self._entry_contracts} YES @ {entry_price}c",
                    signal_data=signal.to_dict(),
                )

                await self._context.event_bus.emit_system_activity(
                    activity_type="odmr_entry",
                    message=f"ODMR Entry: {signal.market_ticker} dip {signal.dip_depth}c, buying YES @ {entry_price}c",
                    metadata={
                        "signal_id": signal_id,
                        "market": signal.market_ticker,
                        "rolling_high": signal.rolling_high,
                        "current_price": signal.current_price,
                        "dip_depth": signal.dip_depth,
                        "entry_price": entry_price,
                        "contracts": self._entry_contracts,
                    }
                )
            else:
                self._record_decision(
                    signal_id=signal_id,
                    action="entry_failed",
                    reason="Order execution failed",
                    signal_data=signal.to_dict(),
                )

        except Exception as e:
            logger.error(f"Error executing ODMR entry: {e}")
            self._record_decision(
                signal_id=signal_id,
                action="entry_error",
                reason=f"Exception: {str(e)}",
                signal_data=signal.to_dict(),
            )

    async def _place_target_exit(self, state: DipMarketState) -> None:
        """Place target exit order (limit sell YES at target price)."""
        from ...services.trading_decision_service import TradingDecision, TradingStrategy

        if not self._context or not self._context.trading_service:
            logger.warning("ODMR: Cannot place target exit - no trading service available")
            return

        # Verify actual position before placing sell order
        verified_quantity = await self._verify_position_for_exit(state)
        if verified_quantity <= 0:
            logger.warning(
                f"ODMR: Skipping target exit for {state.market_ticker} - no verified position"
            )
            self._reset_position_state(state)
            return

        try:
            decision = TradingDecision(
                action="sell",
                market=state.market_ticker,
                side="yes",
                quantity=verified_quantity,
                price=state.target_exit_price,
                reason=f"ODMR target exit: entry={state.entry_price}c + {self._recovery_target_cents}c target",
                confidence=0.8,
                strategy=TradingStrategy.HOLD,
                signal_params={
                    "entry_price": state.entry_price,
                    "target_price": state.target_exit_price,
                    "recovery_target": self._recovery_target_cents,
                },
            )
            decision.strategy_id = "odmr"

            success = await self._context.trading_service.execute_decision(decision)

            if success:
                # Capture order_id from trading attachment for cancellation support
                await self._capture_exit_order_id(state)
                logger.info(
                    f"ODMR: Target exit PLACED for {state.market_ticker} - "
                    f"SELL {state.contracts_held} YES @ {state.target_exit_price}c"
                    f"{f' (order_id: {state.exit_order_id[:8]}...)' if state.exit_order_id else ''}"
                )
            else:
                logger.warning(f"ODMR: Target exit order FAILED for {state.market_ticker}")

        except Exception as e:
            logger.error(f"ODMR: Error placing target exit for {state.market_ticker}: {e}")

    async def _capture_exit_order_id(self, state: DipMarketState) -> None:
        """
        Capture the exit order ID from the trading attachment after placing target exit.

        Looks for the most recent resting sell order for YES in this market's
        trading attachment. This enables proper cancellation when we need to
        switch to timeout or stop loss exits.

        Args:
            state: Market state to update with order ID
        """
        if not self._context or not self._context.state_container:
            return

        try:
            # Small delay to let the order propagate to the attachment
            await asyncio.sleep(0.1)

            attachment = self._context.state_container.get_trading_attachment(state.market_ticker)
            if not attachment or not attachment.orders:
                logger.debug(f"ODMR: No trading attachment found for {state.market_ticker}")
                return

            # Find the most recent sell order for YES (our target exit)
            # Orders are stored by order_id, so we iterate and find the best match
            for order_id, order in attachment.orders.items():
                # Check for resting sell order on YES side (lowercase per API convention)
                if (order.side == "yes" and
                    order.action == "sell" and
                    order.status == "resting"):
                    state.exit_order_id = order_id
                    logger.debug(
                        f"ODMR: Captured exit order ID {order_id[:8]}... for {state.market_ticker}"
                    )
                    return

            logger.debug(f"ODMR: No matching resting sell order found for {state.market_ticker}")

        except Exception as e:
            logger.warning(f"ODMR: Error capturing exit order ID for {state.market_ticker}: {e}")

    async def _cancel_target_exit(self, state: DipMarketState) -> bool:
        """
        Cancel existing target exit order before placing alternative exit.

        Called before timeout or stop loss exits to ensure we don't have
        conflicting orders. Uses the trading client integration directly
        for cancellation.

        Args:
            state: Market state containing the exit order ID to cancel

        Returns:
            True if cancellation succeeded or no order to cancel, False on failure
        """
        if not state.exit_order_id:
            return True  # No order to cancel

        if not self._context:
            logger.warning("ODMR: Cannot cancel target exit - no context available")
            return False

        try:
            # Get trading client integration from context
            integration = self._context.trading_client_integration
            if not integration:
                logger.warning("ODMR: Cannot cancel target exit - no trading client integration")
                return False

            await integration.cancel_order(state.exit_order_id)
            logger.info(
                f"ODMR: Cancelled target exit {state.exit_order_id[:8]}... for {state.market_ticker}"
            )
            self._stats["exit_order_cancellations"] += 1
            state.exit_order_id = None
            return True

        except Exception as e:
            logger.error(
                f"ODMR: Failed to cancel target exit {state.exit_order_id[:8]}... "
                f"for {state.market_ticker}: {e}"
            )
            self._stats["exit_order_cancel_failures"] += 1
            return False

    async def _execute_timeout_exit(self, state: DipMarketState) -> None:
        """Execute timeout exit (market sell YES)."""
        from ...services.trading_decision_service import TradingDecision, TradingStrategy

        if not self._context or not self._context.trading_service:
            return

        # Verify actual position before placing sell order
        verified_quantity = await self._verify_position_for_exit(state)
        if verified_quantity <= 0:
            logger.warning(
                f"ODMR: Skipping timeout exit for {state.market_ticker} - no verified position"
            )
            self._reset_position_state(state)
            return

        # Check if there's already a pending exit order
        if self._has_pending_exit_order(state):
            logger.info(
                f"ODMR: Skipping timeout exit for {state.market_ticker} - exit order already resting"
            )
            return

        try:
            # Cancel any existing target exit order first
            if state.exit_order_id:
                cancel_success = await self._cancel_target_exit(state)
                if not cancel_success:
                    logger.warning(
                        f"ODMR: Could not cancel target exit for {state.market_ticker}, "
                        "proceeding with timeout exit anyway"
                    )

            # Get current best bid for market sell
            exit_price = state.current_price  # Fallback
            try:
                orderbook_state = await asyncio.wait_for(
                    get_shared_orderbook_state(state.market_ticker),
                    timeout=1.0
                )
                snapshot = await orderbook_state.get_snapshot()
                ob_context = OrderbookContext.from_orderbook_snapshot(snapshot)
                if ob_context.yes_best_bid is not None:
                    exit_price = ob_context.yes_best_bid
            except Exception:
                pass

            decision = TradingDecision(
                action="sell",
                market=state.market_ticker,
                side="yes",
                quantity=verified_quantity,
                price=exit_price,  # Aggressive price for quick fill
                reason=f"ODMR timeout exit: {self._timeout_seconds}s elapsed",
                confidence=0.7,
                strategy=TradingStrategy.HOLD,
                signal_params={
                    "entry_price": state.entry_price,
                    "exit_price": exit_price,
                    "exit_type": "timeout",
                    "elapsed_seconds": self._timeout_seconds,
                },
            )
            decision.strategy_id = "odmr"

            success = await self._context.trading_service.execute_decision(decision)

            if success:
                logger.info(
                    f"ODMR Timeout exit placed: {state.market_ticker} "
                    f"SELL {verified_quantity} YES @ {exit_price}c"
                )
            else:
                logger.warning(f"ODMR Timeout exit order failed for {state.market_ticker}")
                # Trigger retry mechanism - set retry count to start retry loop
                state.exit_retry_count = 1
                state.last_exit_attempt_time = datetime.now(timezone.utc)
                self._stats["exit_retries_triggered"] += 1

        except Exception as e:
            logger.error(f"Error executing timeout exit: {e}")
            # Also trigger retry on exception
            state.exit_retry_count = 1
            state.last_exit_attempt_time = datetime.now(timezone.utc)

    async def _execute_stop_loss(self, state: DipMarketState) -> None:
        """Execute stop loss exit (market sell YES)."""
        from ...services.trading_decision_service import TradingDecision, TradingStrategy

        if not self._context or not self._context.trading_service:
            return

        # Verify actual position before placing sell order
        verified_quantity = await self._verify_position_for_exit(state)
        if verified_quantity <= 0:
            logger.warning(
                f"ODMR: Skipping stop loss exit for {state.market_ticker} - no verified position"
            )
            self._reset_position_state(state)
            return

        try:
            # Cancel any existing target exit order first
            if state.exit_order_id:
                cancel_success = await self._cancel_target_exit(state)
                if not cancel_success:
                    logger.warning(
                        f"ODMR: Could not cancel target exit for {state.market_ticker}, "
                        "proceeding with stop loss exit anyway"
                    )

            # Get current best bid for market sell
            exit_price = state.current_price  # Fallback
            try:
                orderbook_state = await asyncio.wait_for(
                    get_shared_orderbook_state(state.market_ticker),
                    timeout=1.0
                )
                snapshot = await orderbook_state.get_snapshot()
                ob_context = OrderbookContext.from_orderbook_snapshot(snapshot)
                if ob_context.yes_best_bid is not None:
                    exit_price = ob_context.yes_best_bid
            except Exception:
                pass

            decision = TradingDecision(
                action="sell",
                market=state.market_ticker,
                side="yes",
                quantity=verified_quantity,
                price=exit_price,  # Aggressive price for quick fill
                reason=f"ODMR stop loss: -{self._stop_loss_cents}c from entry",
                confidence=0.9,  # High confidence to exit
                strategy=TradingStrategy.HOLD,
                signal_params={
                    "entry_price": state.entry_price,
                    "exit_price": exit_price,
                    "exit_type": "stop_loss",
                    "stop_loss_cents": self._stop_loss_cents,
                },
            )
            decision.strategy_id = "odmr"

            success = await self._context.trading_service.execute_decision(decision)

            if success:
                logger.info(
                    f"ODMR Stop loss exit placed: {state.market_ticker} "
                    f"SELL {verified_quantity} YES @ {exit_price}c"
                )
            else:
                logger.warning(f"ODMR Stop loss exit order failed for {state.market_ticker}")
                # Trigger retry mechanism
                state.exit_retry_count = 1
                state.last_exit_attempt_time = datetime.now(timezone.utc)
                self._stats["exit_retries_triggered"] += 1

        except Exception as e:
            logger.error(f"Error executing stop loss exit: {e}")
            # Also trigger retry on exception
            state.exit_retry_count = 1
            state.last_exit_attempt_time = datetime.now(timezone.utc)

    def _reset_position_state(self, state: DipMarketState) -> None:
        """Reset position-related state after exit."""
        state.position_open = False
        state.entry_price = 0
        state.entry_time = None
        state.contracts_held = 0
        state.exit_order_id = None
        state.target_exit_price = 0
        state.dip_detected_at = None
        state.dip_depth = 0

        # Reset exit retry tracking
        state.exit_retry_count = 0
        state.last_exit_attempt_time = None

        # Keep rolling_high for future dip detection
        # Keep trade_count and current_price

        self._open_positions.discard(state.market_ticker)

    # ============================================================
    # Background Position Monitor
    # ============================================================

    # Exit retry configuration
    EXIT_RETRY_MAX_ATTEMPTS: int = 3
    EXIT_RETRY_INTERVAL_SECONDS: float = 5.0
    EXIT_RETRY_PRICE_AGGRESSION_CENTS: int = 2  # Each retry is 2c more aggressive

    async def _position_monitor_loop(self) -> None:
        """
        Background task to monitor positions for timeout, stop loss, and exit retries.

        Runs every 1 second to check:
        1. Timeout condition (15 minutes elapsed)
        2. Exit retry for failed exits (5 second interval between retries)
        3. Stop loss condition is checked in _process_trade
        """
        logger.info("ODMR Position monitor started")

        while self._running:
            try:
                await asyncio.sleep(POSITION_MONITOR_INTERVAL)

                if not self._running:
                    break

                now = datetime.now(timezone.utc)

                for ticker in list(self._open_positions):
                    state = self._market_states.get(ticker)
                    if not state or not state.position_open or not state.entry_time:
                        continue

                    # Check for exit retry (position still open after exit attempt)
                    if state.exit_retry_count > 0 and state.last_exit_attempt_time:
                        time_since_attempt = (now - state.last_exit_attempt_time).total_seconds()
                        if time_since_attempt >= self.EXIT_RETRY_INTERVAL_SECONDS:
                            await self._execute_exit_with_retry(state)
                        continue  # Skip timeout check if we're in retry mode

                    # Check timeout condition
                    elapsed = (now - state.entry_time).total_seconds()
                    if elapsed >= self._timeout_seconds:
                        logger.info(
                            f"ODMR Timeout triggered: {ticker} "
                            f"elapsed={elapsed:.0f}s >= {self._timeout_seconds}s"
                        )
                        await self._execute_timeout_exit(state)

            except asyncio.CancelledError:
                logger.info("ODMR Position monitor cancelled")
                break
            except Exception as e:
                logger.error(f"Error in ODMR position monitor: {e}")

        logger.info("ODMR Position monitor stopped")

    async def _execute_exit_with_retry(self, state: DipMarketState) -> None:
        """
        Execute exit with retry logic and increasingly aggressive pricing.

        Called when a previous exit attempt did not fill. Each retry uses a
        more aggressive price (2c lower than previous) to increase fill probability.

        Max 3 retries before giving up and logging max retries reached.

        Args:
            state: Market state with position needing exit
        """
        from ...services.trading_decision_service import TradingDecision, TradingStrategy

        if not self._context or not self._context.trading_service:
            return

        # Verify actual position before placing sell order (especially important for retries)
        verified_quantity = await self._verify_position_for_exit(state)
        if verified_quantity <= 0:
            logger.warning(
                f"ODMR: Skipping exit retry for {state.market_ticker} - no verified position"
            )
            self._reset_position_state(state)
            return

        # Check if max retries reached
        if state.exit_retry_count >= self.EXIT_RETRY_MAX_ATTEMPTS:
            logger.error(
                f"ODMR: Max exit retries ({self.EXIT_RETRY_MAX_ATTEMPTS}) reached for {state.market_ticker}. "
                f"Position likely stale or already closed. Cleaning up state."
            )
            self._stats["exit_max_retries_reached"] += 1
            # Force cleanup of stale position to prevent infinite retry loops
            # This happens when ODMR thinks there's a position but the trading client doesn't have it
            logger.warning(
                f"ODMR: Force-closing stale position for {state.market_ticker} "
                f"(entry={state.entry_price}c, contracts={verified_quantity})"
            )
            self._reset_position_state(state)
            return

        try:
            # Get current best bid and make it more aggressive based on retry count
            exit_price = state.current_price  # Fallback
            try:
                orderbook_state = await asyncio.wait_for(
                    get_shared_orderbook_state(state.market_ticker),
                    timeout=1.0
                )
                snapshot = await orderbook_state.get_snapshot()
                ob_context = OrderbookContext.from_orderbook_snapshot(snapshot)
                if ob_context.yes_best_bid is not None:
                    exit_price = ob_context.yes_best_bid
            except Exception:
                pass

            # Make price more aggressive with each retry
            # Retry 1: -2c, Retry 2: -4c, Retry 3: -6c
            price_reduction = state.exit_retry_count * self.EXIT_RETRY_PRICE_AGGRESSION_CENTS
            aggressive_price = max(1, exit_price - price_reduction)  # Never below 1c

            state.exit_retry_count += 1
            state.last_exit_attempt_time = datetime.now(timezone.utc)
            self._stats["exit_retries_triggered"] += 1

            logger.info(
                f"ODMR Exit retry {state.exit_retry_count}/{self.EXIT_RETRY_MAX_ATTEMPTS} for {state.market_ticker}: "
                f"SELL {verified_quantity} YES @ {aggressive_price}c (base={exit_price}c, reduction={price_reduction}c)"
            )

            decision = TradingDecision(
                action="sell",
                market=state.market_ticker,
                side="yes",
                quantity=verified_quantity,
                price=aggressive_price,
                reason=f"ODMR exit retry {state.exit_retry_count}: aggressive price {aggressive_price}c",
                confidence=0.85,  # Higher confidence for retries
                strategy=TradingStrategy.HOLD,
                signal_params={
                    "entry_price": state.entry_price,
                    "exit_price": aggressive_price,
                    "exit_type": "retry",
                    "retry_count": state.exit_retry_count,
                    "price_reduction": price_reduction,
                },
            )
            decision.strategy_id = "odmr"

            success = await self._context.trading_service.execute_decision(decision)

            if success:
                logger.info(
                    f"ODMR Exit retry order placed: {state.market_ticker} "
                    f"SELL {state.contracts_held} YES @ {aggressive_price}c"
                )
            else:
                logger.warning(
                    f"ODMR Exit retry order failed for {state.market_ticker} "
                    f"(attempt {state.exit_retry_count})"
                )

        except Exception as e:
            logger.error(f"Error in ODMR exit retry for {state.market_ticker}: {e}")

    # ============================================================
    # Position Verification Helpers
    # ============================================================

    async def _verify_position_for_exit(self, state: DipMarketState) -> int:
        """
        Verify actual position from trading state before placing sell orders.

        Prevents trading errors when ODMR's tracked state becomes desynchronized
        from the actual position (e.g., after manual interventions, order rejections,
        or sync delays).

        Args:
            state: DipMarketState containing tracked contracts_held

        Returns:
            Actual contract count from trading attachment (0 if no position)

        Side Effects:
            - Emits odmr_sync_warning activity event if mismatch detected
            - Syncs state.contracts_held with actual position
        """
        if not self._context or not self._context.state_container:
            logger.warning(
                f"ODMR: Cannot verify position for {state.market_ticker} - no state container"
            )
            return 0

        attachment = self._context.state_container.get_trading_attachment(state.market_ticker)

        # No trading attachment means no position
        if not attachment:
            if state.contracts_held > 0:
                logger.warning(
                    f"ODMR: Position mismatch {state.market_ticker}: "
                    f"tracked={state.contracts_held}, actual=0 (no attachment)"
                )
                await self._context.event_bus.emit_system_activity(
                    activity_type="odmr_sync_warning",
                    message=f"ODMR: Position mismatch {state.market_ticker}: tracked={state.contracts_held}, actual=0",
                    metadata={
                        "market": state.market_ticker,
                        "tracked_contracts": state.contracts_held,
                        "actual_contracts": 0,
                        "issue": "no_position"
                    }
                )
                state.contracts_held = 0
            return 0

        # No position in attachment
        if not attachment.position:
            if state.contracts_held > 0:
                logger.warning(
                    f"ODMR: Position mismatch {state.market_ticker}: "
                    f"tracked={state.contracts_held}, actual=0 (no position in attachment)"
                )
                await self._context.event_bus.emit_system_activity(
                    activity_type="odmr_sync_warning",
                    message=f"ODMR: Position mismatch {state.market_ticker}: tracked={state.contracts_held}, actual=0",
                    metadata={
                        "market": state.market_ticker,
                        "tracked_contracts": state.contracts_held,
                        "actual_contracts": 0,
                        "issue": "no_position"
                    }
                )
                state.contracts_held = 0
            return 0

        actual_position = attachment.position.count

        # Check for zero or negative (shouldn't have negative but defensive)
        if actual_position <= 0:
            if state.contracts_held > 0:
                logger.warning(
                    f"ODMR: Position mismatch {state.market_ticker}: "
                    f"tracked={state.contracts_held}, actual={actual_position}"
                )
                await self._context.event_bus.emit_system_activity(
                    activity_type="odmr_sync_warning",
                    message=f"ODMR: Position mismatch {state.market_ticker}: tracked={state.contracts_held}, actual={actual_position}",
                    metadata={
                        "market": state.market_ticker,
                        "tracked_contracts": state.contracts_held,
                        "actual_contracts": actual_position,
                        "issue": "zero_or_negative"
                    }
                )
                state.contracts_held = 0
            return 0

        # Check for mismatch (actual exists but differs from tracked)
        if actual_position != state.contracts_held:
            logger.warning(
                f"ODMR: Position mismatch {state.market_ticker}: "
                f"tracked={state.contracts_held}, actual={actual_position}"
            )
            await self._context.event_bus.emit_system_activity(
                activity_type="odmr_sync_warning",
                message=f"ODMR: Position mismatch {state.market_ticker}: tracked={state.contracts_held}, actual={actual_position}",
                metadata={
                    "market": state.market_ticker,
                    "tracked_contracts": state.contracts_held,
                    "actual_contracts": actual_position,
                    "issue": "mismatch"
                }
            )
            state.contracts_held = actual_position

        return actual_position

    def _has_pending_exit_order(self, state: DipMarketState) -> bool:
        """
        Check if there's already a resting sell order for this position.

        Used to avoid placing duplicate exit orders (e.g., timeout exit when
        target exit is already resting).

        Args:
            state: DipMarketState for the market

        Returns:
            True if a resting sell order exists, False otherwise
        """
        if not self._context or not self._context.state_container:
            return False

        attachment = self._context.state_container.get_trading_attachment(state.market_ticker)
        if not attachment or not attachment.orders:
            return False

        # Check for any resting sell order on YES side
        for order in attachment.orders.values():
            if (order.side == "yes" and
                order.action == "sell" and
                order.status == "resting"):
                logger.debug(
                    f"ODMR: Found pending exit order {order.order_id[:8]}... for {state.market_ticker}"
                )
                return True

        return False

    # ============================================================
    # Utility Methods
    # ============================================================

    def _record_decision(
        self,
        signal_id: str,
        action: str,
        reason: str,
        order_id: Optional[str] = None,
        signal_data: Optional[Dict[str, Any]] = None,
        pnl_cents: Optional[int] = None,
    ) -> None:
        """Record a signal decision in history."""
        decision = DipDecision(
            signal_id=signal_id,
            timestamp=time.time(),
            action=action,
            reason=reason,
            order_id=order_id,
            signal_data=signal_data,
            pnl_cents=pnl_cents,
        )
        self._decision_history.append(decision)

    # ============================================================
    # Public Query Methods
    # ============================================================

    def get_market_states(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get current market states for monitoring."""
        states = list(self._market_states.values())
        # Sort by open positions first, then by trade count
        states.sort(key=lambda s: (not s.position_open, -s.trade_count))
        return [s.to_dict() for s in states[:limit]]

    def get_decision_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent decision history."""
        decisions = list(self._decision_history)
        decisions.reverse()
        return [d.to_dict() for d in decisions[:limit]]

    def get_open_positions(self) -> List[Dict[str, Any]]:
        """Get all open positions with details."""
        positions = []
        for ticker in self._open_positions:
            state = self._market_states.get(ticker)
            if state and state.position_open:
                # Calculate unrealized P&L
                unrealized_pnl = (state.current_price - state.entry_price) * state.contracts_held

                # Calculate time remaining
                time_remaining = None
                if state.entry_time:
                    elapsed = (datetime.now(timezone.utc) - state.entry_time).total_seconds()
                    time_remaining = max(0, self._timeout_seconds - elapsed)

                positions.append({
                    "market_ticker": ticker,
                    "entry_price": state.entry_price,
                    "current_price": state.current_price,
                    "target_exit_price": state.target_exit_price,
                    "contracts_held": state.contracts_held,
                    "unrealized_pnl_cents": unrealized_pnl,
                    "time_remaining_seconds": time_remaining,
                    "dip_depth": state.dip_depth,
                })
        return positions

    def get_recent_tracked_trades(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get recent tracked trades for Trade Processing panel.

        Returns trades in reverse chronological order (most recent first),
        matching the RLM NO pattern for UI consistency.

        Args:
            limit: Maximum number of trades to return (default 20)

        Returns:
            List of trade dictionaries with trade_id, market_ticker, side,
            price_cents, count, timestamp, and age_seconds
        """
        trades = list(self._recent_tracked_trades)
        trades.reverse()
        return [t.to_dict() for t in trades[:limit]]
