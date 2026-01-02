"""
RLM (Reverse Line Movement) Trading Service for TRADER V3.

Implements the validated +17.38% edge strategy: bet NO when majority trade YES but price drops.
This strategy was validated on 1.7M+ trades across 72k+ unique markets.

Key Mechanics:
    - Signal: >65% YES trades + YES price dropped from open + >=15 trades
    - Entry: Market or limit order based on orderbook spread
    - Exit: Hold to settlement (no early exits in MVP)
    - Re-entry: Allow adding to position on stronger signal

Architecture Position:
    - Subscribes to PUBLIC_TRADE_RECEIVED events
    - Subscribes to MARKET_TRACKED events (for lifecycle-discovered markets)
    - Accumulates per-market trade state
    - Creates TradingDecision objects
    - Delegates execution to TradingDecisionService

Reference: traderv3/planning/rlm_v1.md
"""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Set, TYPE_CHECKING

from ..core.event_bus import EventBus, EventType, PublicTradeEvent, MarketDeterminedEvent
from ..core.state_machine import TraderState
from ...data.orderbook_state import get_shared_orderbook_state

if TYPE_CHECKING:
    from ..services.trading_decision_service import TradingDecisionService, TradingDecision
    from ..core.state_container import V3StateContainer
    from ..state.tracked_markets import TrackedMarketsState

logger = logging.getLogger("kalshiflow_rl.traderv3.services.rlm")


# Decision history size
DECISION_HISTORY_SIZE = 100

# Recent tracked trades buffer size
RECENT_TRACKED_TRADES_SIZE = 30

# Rate limiting: 10 trades per minute
DEFAULT_MAX_TRADES_PER_MINUTE = 10
DEFAULT_TOKEN_REFILL_SECONDS = 6.0


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
        - Trade direction counts (YES vs NO)
        - Price movement from first trade to current
        - Position state for re-entry logic
    """
    market_ticker: str
    yes_trades: int = 0
    no_trades: int = 0
    first_yes_price: Optional[int] = None  # Anchor (from first trade)
    last_yes_price: Optional[int] = None   # Current (updated each trade)
    first_trade_time: Optional[float] = None
    last_trade_time: Optional[float] = None

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
    def price_drop(self) -> int:
        if self.first_yes_price is None or self.last_yes_price is None:
            return 0
        return self.first_yes_price - self.last_yes_price

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


class RLMService:
    """
    Event-driven RLM (Reverse Line Movement) trading service.

    Subscribes to public trade events and accumulates per-market trade state.
    When RLM signal conditions are met (>65% YES trades + price drops),
    executes trades to bet NO.

    Key Features:
        - Trade direction tracking from public trade feed
        - Price movement detection from market open
        - Orderbook-aware execution (market vs limit based on spread)
        - Re-entry on stronger signals
        - Category filtering via lifecycle discovery
    """

    def __init__(
        self,
        event_bus: EventBus,
        trading_service: 'TradingDecisionService',
        state_container: 'V3StateContainer',
        tracked_markets_state: 'TrackedMarketsState',
        yes_threshold: float = 0.65,
        min_trades: int = 15,
        min_price_drop: int = 0,
        contracts_per_trade: int = 100,
        max_concurrent: int = 1000,
        allow_reentry: bool = True,
        orderbook_timeout: float = 2.0,
        tight_spread: int = 2,
        normal_spread: int = 4,
        max_spread: int = 10,
        max_trades_per_minute: int = DEFAULT_MAX_TRADES_PER_MINUTE,
        token_refill_seconds: float = DEFAULT_TOKEN_REFILL_SECONDS,
    ):
        """
        Initialize RLM service.

        Args:
            event_bus: V3 EventBus for event subscription
            trading_service: Service for executing trades
            state_container: Container for trading state (positions, orders)
            tracked_markets_state: State for lifecycle-discovered markets
            yes_threshold: Minimum YES trade ratio to trigger signal (default: 0.65)
            min_trades: Minimum trades before evaluating signal (default: 15)
            min_price_drop: Minimum YES price drop in cents (default: 0 = any drop)
            contracts_per_trade: Contracts per trade (default: 100)
            max_concurrent: Maximum concurrent positions (default: 1000)
            allow_reentry: Allow adding to position on stronger signal (default: True)
            orderbook_timeout: Timeout for orderbook fetch in seconds (default: 2.0)
            tight_spread: Spread <= this uses aggressive fill (ask - 1c)
            normal_spread: Spread <= this uses midpoint pricing
            max_spread: Spread > this skips signal (protects from bad fills)
            max_trades_per_minute: Rate limit - trades per minute (default: 10)
            token_refill_seconds: Seconds between token refills (default: 6.0)
        """
        self._event_bus = event_bus
        self._trading_service = trading_service
        self._state_container = state_container
        self._tracked_markets_state = tracked_markets_state

        # Signal detection parameters
        self._yes_threshold = yes_threshold
        self._min_trades = min_trades
        self._min_price_drop = min_price_drop
        self._contracts_per_trade = contracts_per_trade
        self._max_concurrent = max_concurrent
        self._allow_reentry = allow_reentry

        # Orderbook execution parameters
        self._orderbook_timeout = orderbook_timeout
        self._tight_spread = tight_spread
        self._normal_spread = normal_spread
        self._max_spread = max_spread

        # Token bucket rate limiting
        self._max_tokens = max_trades_per_minute
        self._tokens = float(max_trades_per_minute)
        self._token_refill_seconds = token_refill_seconds
        self._last_refill_time = time.time()
        self._rate_limited_count = 0

        # Per-market trade accumulation state (in-memory, rebuilds on restart)
        self._market_states: Dict[str, MarketTradeState] = {}

        # Track markets with open positions (for re-entry logic)
        self._markets_with_positions: Set[str] = set()

        # Decision history (circular buffer)
        self._decision_history: deque[RLMDecision] = deque(maxlen=DECISION_HISTORY_SIZE)

        # Recent tracked trades buffer (for Trade Processing panel)
        self._recent_tracked_trades: deque[TrackedTrade] = deque(maxlen=RECENT_TRACKED_TRADES_SIZE)

        # Statistics
        self._stats = {
            "trades_processed": 0,
            "trades_filtered": 0,
            "signals_detected": 0,
            "signals_executed": 0,
            "signals_skipped": 0,
            "reentries": 0,
            "orderbook_fallbacks": 0,
        }
        self._last_execution_time: Optional[float] = None

        # State
        self._running = False
        self._started_at: Optional[float] = None
        self._processing_lock = asyncio.Lock()

        # RLM state broadcast throttling (max 2 updates/sec per market = 500ms min interval)
        self._last_state_emit: Dict[str, float] = {}
        self._state_emit_interval = 0.5  # 500ms between state updates per market

        # Atomic counter for unique trade IDs (handles multiple trades in same millisecond)
        self._trade_counter: int = 0

        logger.info(
            f"RLMService initialized: yes_threshold={yes_threshold:.0%}, "
            f"min_trades={min_trades}, min_price_drop={min_price_drop}c, "
            f"contracts={contracts_per_trade}, max_concurrent={max_concurrent}, "
            f"reentry={'ON' if allow_reentry else 'OFF'}"
        )

    async def start(self) -> None:
        """Start the RLM service."""
        if self._running:
            logger.warning("RLMService is already running")
            return

        logger.info("Starting RLMService")
        self._running = True
        self._started_at = time.time()

        # Subscribe to public trade events
        await self._event_bus.subscribe_to_public_trade(self._handle_public_trade)
        logger.info("Subscribed to PUBLIC_TRADE_RECEIVED events")

        # Subscribe to market determined events (for cleanup)
        await self._event_bus.subscribe_to_market_determined(self._handle_market_determined)
        logger.info("Subscribed to MARKET_DETERMINED events for cleanup")

        # Emit startup event
        await self._event_bus.emit_system_activity(
            activity_type="strategy_start",
            message=f"RLM (Reverse Line Movement) strategy started",
            metadata={
                "strategy": "RLM_NO",
                "yes_threshold": f"{self._yes_threshold:.0%}",
                "min_trades": self._min_trades,
                "max_concurrent": self._max_concurrent,
            }
        )

        logger.info("RLMService started successfully")

    async def stop(self) -> None:
        """Stop the RLM service."""
        if not self._running:
            return

        logger.info("Stopping RLMService...")
        self._running = False

        # Unsubscribe from EventBus to prevent stale handlers
        self._event_bus.unsubscribe(EventType.PUBLIC_TRADE_RECEIVED, self._handle_public_trade)
        logger.debug("Unsubscribed from PUBLIC_TRADE_RECEIVED")

        self._event_bus.unsubscribe(EventType.MARKET_DETERMINED, self._handle_market_determined)
        logger.debug("Unsubscribed from MARKET_DETERMINED")

        # Emit shutdown event
        await self._event_bus.emit_system_activity(
            activity_type="strategy_stop",
            message=f"RLM strategy stopped",
            metadata={
                "strategy": "RLM_NO",
                "signals_detected": self._stats["signals_detected"],
                "signals_executed": self._stats["signals_executed"],
            }
        )

        logger.info(
            f"RLMService stopped. Stats: "
            f"detected={self._stats['signals_detected']}, "
            f"executed={self._stats['signals_executed']}, "
            f"skipped={self._stats['signals_skipped']}"
        )

    async def _handle_public_trade(self, trade_event: PublicTradeEvent) -> None:
        """
        Handle public trade events.

        Filters to tracked markets and accumulates trade state.

        Args:
            trade_event: Public trade event from EventBus
        """
        if not self._running:
            return

        # Wait for system to be fully ready before processing RLM signals
        # This prevents failures during startup when trading client isn't connected yet
        machine_state = self._state_container.machine_state
        if machine_state != TraderState.READY:
            logger.debug(f"Skipping RLM processing - system not ready (state={machine_state})")
            return

        # Filter: only process trades for tracked markets (lifecycle-discovered)
        if not self._tracked_markets_state.is_tracked(trade_event.market_ticker):
            self._stats["trades_filtered"] += 1
            return

        # Use lock to prevent concurrent processing
        async with self._processing_lock:
            await self._process_trade(trade_event)

    async def _process_trade(self, trade_event: PublicTradeEvent) -> None:
        """
        Process a public trade and update market state.

        Args:
            trade_event: Public trade event
        """
        self._stats["trades_processed"] += 1
        market_ticker = trade_event.market_ticker

        # Store in tracked trades buffer (for Trade Processing panel)
        # Generate trade_id from market_ticker, timestamp_ms, and counter for uniqueness
        # Counter ensures uniqueness when multiple trades occur in the same millisecond
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
        # If trade is YES, price_cents is YES price
        # If trade is NO, YES price is 100 - price_cents
        yes_price = trade_event.price_cents if trade_event.side == "yes" else (100 - trade_event.price_cents)

        # Track price movement
        if state.first_yes_price is None:
            state.first_yes_price = yes_price
            state.first_trade_time = time.time()

        state.last_yes_price = yes_price
        state.last_trade_time = time.time()

        # Emit trade arrived event (every trade - lightweight for UI pulse)
        await self._event_bus.emit_rlm_trade_arrived(
            market_ticker=market_ticker,
            side=trade_event.side,
            count=trade_event.count,
            price_cents=trade_event.price_cents,
        )

        # Emit state update (throttled to max 2/sec per market)
        if self._should_emit_state_update(market_ticker):
            await self._event_bus.emit_rlm_market_update(
                market_ticker=market_ticker,
                state=state.to_dict(),
            )
            self._last_state_emit[market_ticker] = time.time()

        # Check for signal
        signal = self._detect_signal(state)
        if signal:
            await self._execute_signal(signal)

    def _detect_signal(self, state: MarketTradeState) -> Optional[RLMSignal]:
        """
        Detect if market state meets RLM signal criteria.

        Signal Criteria (ALL must be true):
            1. total_trades >= min_trades (sufficient activity)
            2. yes_ratio > yes_threshold (majority YES trades)
            3. price_drop >= min_price_drop (price moved toward NO)
            4. Not at max concurrent positions (unless re-entry)

        Args:
            state: Market trade state

        Returns:
            RLMSignal if criteria met, None otherwise
        """
        # Check minimum trades
        if state.total_trades < self._min_trades:
            return None

        # Check YES ratio
        if state.yes_ratio <= self._yes_threshold:
            return None

        # Check price drop
        if state.price_drop < self._min_price_drop:
            return None

        # Check concurrent positions
        trading_state = self._state_container.trading_state
        positions = trading_state.positions if trading_state else {}
        current_position_count = len(positions)

        # Determine if this is a re-entry
        has_position = state.market_ticker in positions
        is_reentry = has_position and self._allow_reentry

        # Skip if at max positions and not re-entry
        if current_position_count >= self._max_concurrent and not is_reentry:
            return None

        # For re-entry, check if signal is stronger than entry
        if is_reentry:
            if state.entry_yes_ratio is not None and state.entry_price_drop is not None:
                # Signal must be stronger (higher ratio OR bigger drop)
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

    async def _execute_signal(self, signal: RLMSignal) -> None:
        """
        Execute an RLM signal by placing a NO order.

        Uses orderbook to determine order type (market vs limit).
        Signal state is always reset after execution (one-shot behavior).

        Args:
            signal: Detected RLM signal
        """
        from ..services.trading_decision_service import TradingDecision, TradingStrategy

        signal_id = f"{signal.market_ticker}:{int(signal.detected_at * 1000)}"
        trade_succeeded = False  # Track success for reset logic

        # Increment trigger count (persists across resets for historical tracking)
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
            await self._event_bus.emit_system_activity(
                activity_type="rlm_signal",
                message=f"RLM Signal: {signal.market_ticker} YES={signal.yes_ratio:.0%} drop={signal.price_drop}c",
                metadata={"signal_id": signal_id, "status": "processing", **signal.to_dict()}
            )
            # Get orderbook for execution price
            entry_price = None

            try:
                orderbook_state = await asyncio.wait_for(
                    get_shared_orderbook_state(signal.market_ticker),
                    timeout=self._orderbook_timeout
                )
                snapshot = await orderbook_state.get_snapshot()

                # Get NO spread
                no_asks = snapshot.get("no_asks", {})
                no_bids = snapshot.get("no_bids", {})

                if no_asks and no_bids:
                    best_no_ask = min(no_asks.keys()) if no_asks else 99
                    best_no_bid = max(no_bids.keys()) if no_bids else 1
                    spread = best_no_ask - best_no_bid

                    # P1: Max spread rejection - protect from bad fills in illiquid markets
                    if spread > self._max_spread:
                        logger.warning(
                            f"Spread too wide for {signal.market_ticker}: {spread}c > {self._max_spread}c max, skipping signal"
                        )
                        self._stats["signals_skipped"] += 1
                        self._record_decision(
                            signal_id=signal_id,
                            action="skipped",
                            reason=f"Spread {spread}c > max {self._max_spread}c",
                            signal_data=signal.to_dict()
                        )
                        return

                    # Log orderbook state for Phase 2 analysis
                    self._log_orderbook_at_signal(signal, snapshot)

                    # P1: Position-aware pricing - stronger signals can afford more slippage
                    # 2x positions (20c+ drop) have 30% edge, can pay for guaranteed fills
                    # 1.5x positions (10-20c drop) have 17-19% edge, be moderately aggressive
                    # 1x positions (5-10c drop) have 12% edge, prioritize price improvement
                    is_high_edge = signal.price_drop >= 10  # 1.5x or 2x positions

                    # Spread-aware pricing using config thresholds
                    entry_price = self._calculate_no_entry_price(
                        best_no_ask, best_no_bid, spread, aggressive=is_high_edge
                    )

                    # Log pricing decision for strategy validation (use config thresholds)
                    if spread <= self._tight_spread:
                        spread_type = "tight"
                    elif spread <= self._normal_spread:
                        spread_type = "normal"
                    else:
                        spread_type = "wide"
                    aggr_label = " [aggressive]" if is_high_edge else ""
                    logger.info(
                        f"RLM pricing: {signal.market_ticker} spread={spread}c "
                        f"bid={best_no_bid} ask={best_no_ask} → entry={entry_price}c ({spread_type}{aggr_label})"
                    )
                else:
                    # No orderbook data - skip this signal
                    logger.warning(f"No orderbook data for {signal.market_ticker}, skipping signal")
                    self._stats["signals_skipped"] += 1
                    return

            except (asyncio.TimeoutError, Exception) as e:
                logger.warning(f"Orderbook unavailable for {signal.market_ticker}: {e}, skipping signal")
                self._stats["orderbook_fallbacks"] += 1
                self._stats["signals_skipped"] += 1
                return

            # S-001: Scale position by signal strength (price drop magnitude)
            # Research shows: 5-10c = +11.9% edge, 10-20c = +17-19.5% edge, 20c+ = +30.7% edge
            if signal.price_drop >= 20:
                scaled_quantity = self._contracts_per_trade * 2  # 2x for strongest signals
                scale_label = "2x"
            elif signal.price_drop >= 10:
                scaled_quantity = int(self._contracts_per_trade * 1.5)  # 1.5x for strong signals
                scale_label = "1.5x"
            else:
                scaled_quantity = self._contracts_per_trade  # 1x for baseline signals (5-10c)
                scale_label = "1x"

            # Create trading decision
            action_type = "reentry" if signal.is_reentry else "executed"
            decision = TradingDecision(
                action="buy",
                market=signal.market_ticker,
                side="no",
                quantity=scaled_quantity,
                price=entry_price,
                reason=f"RLM signal: {signal.yes_ratio:.0%} YES, -{signal.price_drop}c drop ({scale_label})",
                confidence=min(0.9, 0.7 + signal.price_drop * 0.01),  # Higher confidence for bigger drops
                strategy=TradingStrategy.RLM_NO,
            )

            logger.info(
                f"Executing RLM signal: {signal.market_ticker} "
                f"NO @ {entry_price or 'market'}c, {scaled_quantity} contracts ({scale_label}) "
                f"({'reentry' if signal.is_reentry else 'new'})"
            )

            # Execute through trading service
            success = await self._trading_service.execute_decision(decision)

            if success:
                trade_succeeded = True  # Mark success for reset logic
                self._stats["signals_executed"] += 1
                if signal.is_reentry:
                    self._stats["reentries"] += 1
                self._last_execution_time = time.time()

                # Update market state with entry parameters
                state = self._market_states.get(signal.market_ticker)
                if state:
                    state.position_contracts += scaled_quantity
                    if not signal.is_reentry:
                        state.entry_yes_ratio = signal.yes_ratio
                        state.entry_price_drop = signal.price_drop

                # Record successful decision
                self._record_decision(
                    signal_id=signal_id,
                    action=action_type,
                    reason=f"Bought {scaled_quantity} NO @ {entry_price or 'market'}c ({scale_label})",
                    signal_data=signal.to_dict()
                )

                # Emit success event
                await self._event_bus.emit_system_activity(
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
            # One-shot behavior: always reset signal state after execution attempt
            # preserve_entry=True on success keeps entry params for re-entry comparison
            self._reset_signal_state(signal.market_ticker, preserve_entry=trade_succeeded)

    def _calculate_no_entry_price(
        self,
        best_no_ask: int,
        best_no_bid: int,
        spread: int,
        aggressive: bool = False
    ) -> int:
        """
        Calculate optimal entry price for buying NO contracts.

        Spread-aware pricing optimized for RLM time-sensitive signals:
        - Tight spread (<=tight_spread): hit near ask for guaranteed fill
        - Normal spread (<=normal_spread): use midpoint for price improvement
        - Wide spread (>normal_spread): 75% toward ask for higher fill probability

        Position-aware adjustment (aggressive=True for high-edge signals):
        - High-edge signals (10c+ drop) can afford slippage, prioritize fills
        - Low-edge signals (5-10c drop) prioritize price improvement

        Uses configurable thresholds: tight_spread, normal_spread (set in config).
        Max spread rejection happens before this function is called.

        Args:
            best_no_ask: Best NO ask price in cents (what we pay to buy immediately)
            best_no_bid: Best NO bid price in cents (what we'd get to sell)
            spread: Bid-ask spread in cents
            aggressive: If True, use more aggressive pricing for guaranteed fills

        Returns:
            Entry price in cents
        """
        if spread <= self._tight_spread:
            # Tight spread: aggressive hits ask, normal joins queue below ask
            if aggressive:
                return best_no_ask  # Hit ask for guaranteed fill
            return best_no_ask - 1  # Join queue just below ask
        elif spread <= self._normal_spread:
            # Normal spread: aggressive pays near ask, normal uses midpoint
            if aggressive:
                return best_no_ask - 1  # Pay 1c below ask
            return (best_no_ask + best_no_bid) // 2  # Use midpoint
        else:
            # Wide spread: aggressive goes 85% toward ask, normal goes 75%
            if aggressive:
                return best_no_bid + (spread * 85 // 100)  # 85% toward ask
            return best_no_bid + (spread * 3 // 4)  # 75% toward ask

    def _log_orderbook_at_signal(self, signal: RLMSignal, orderbook: Dict[str, Any]) -> None:
        """
        Log orderbook state at signal time for Phase 2 analysis.

        Args:
            signal: RLM signal
            orderbook: Orderbook snapshot
        """
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
        """
        Reset signal detection state after trade attempt.

        This makes the signal "one-shot" - market must re-accumulate
        fresh trades to trigger another signal.

        Args:
            market_ticker: Market to reset
            preserve_entry: If True, keep entry_* params for re-entry comparison
        """
        state = self._market_states.get(market_ticker)
        if state:
            # Reset signal detection counters
            state.yes_trades = 0
            state.no_trades = 0
            state.first_yes_price = None
            state.last_yes_price = None
            state.first_trade_time = None
            state.last_trade_time = None
            # Keep position_contracts (we still own the position)
            # Keep entry_yes_ratio and entry_price_drop if preserve_entry=True
            if not preserve_entry:
                state.entry_yes_ratio = None
                state.entry_price_drop = None

            logger.debug(f"Reset RLM signal state for {market_ticker} (preserve_entry={preserve_entry})")

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
        """
        Check if we should emit a state update for this market.

        Throttles state updates to max 2 per second per market (500ms interval).
        This prevents overwhelming the WebSocket with rapid updates during
        high-frequency trading periods.

        Args:
            market_ticker: Market ticker to check

        Returns:
            True if enough time has passed since last emit
        """
        last_emit = self._last_state_emit.get(market_ticker, 0.0)
        return (time.time() - last_emit) >= self._state_emit_interval

    async def _handle_market_determined(self, event: MarketDeterminedEvent) -> None:
        """
        Handle market determined events for cleanup.

        Removes all state for a market that has been determined/settled
        to prevent unbounded memory growth.

        Args:
            event: Market determined event from EventBus
        """
        ticker = event.market_ticker
        cleaned = False

        # Remove from market states (trade accumulation)
        if ticker in self._market_states:
            del self._market_states[ticker]
            cleaned = True

        # Remove from state emit throttling dict
        if ticker in self._last_state_emit:
            del self._last_state_emit[ticker]
            cleaned = True

        # Remove from position tracking set
        if ticker in self._markets_with_positions:
            self._markets_with_positions.discard(ticker)
            cleaned = True

        if cleaned:
            logger.info(f"RLMService cleanup for determined market: {ticker}")

    # Public methods for stats and monitoring

    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        return {
            "strategy": "RLM_NO",
            "running": self._running,
            "started_at": self._started_at,
            "uptime_seconds": int(time.time() - self._started_at) if self._started_at else 0,
            "markets_tracking": len(self._market_states),
            "last_execution": self._last_execution_time,
            "rate_limited_count": self._rate_limited_count,
            **self._stats,
        }

    def get_market_states(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get current market states for monitoring."""
        states = list(self._market_states.values())
        # Sort by total trades descending
        states.sort(key=lambda s: s.total_trades, reverse=True)
        return [s.to_dict() for s in states[:limit]]

    def get_decision_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent decision history."""
        decisions = list(self._decision_history)
        decisions.reverse()  # Most recent first
        return [d.to_dict() for d in decisions[:limit]]

    def get_recent_tracked_trades(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent tracked trades for Trade Processing panel (newest first)."""
        trades = list(self._recent_tracked_trades)
        trades.reverse()  # Most recent first
        return [t.to_dict() for t in trades[:limit]]

    def is_healthy(self) -> bool:
        """Check if service is healthy."""
        return self._running
