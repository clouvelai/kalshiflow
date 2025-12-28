"""
YES at 80-90c Trading Service for TRADER V3.

Implements the validated +5.1% edge strategy: buy YES when price is 80-90c.
This strategy was validated on 1.6M+ resolved trades across 72k+ unique markets.

Key Mechanics:
    - Signal: YES ask price between 80-90c with liquidity >= 10 contracts
    - Entry: Limit order at calculated price based on spread
    - Exit: Hold to settlement (no early exits in MVP)
    - Risk: Max 100 concurrent positions for stress testing

Architecture Position:
    - Subscribes to ORDERBOOK_SNAPSHOT/DELTA events
    - Detects 80-90c signals in YES ask prices
    - Creates TradingDecision objects
    - Delegates execution to TradingDecisionService

Reference: traderv3/planning/MVP_BEST_STRATEGY.md
"""

import asyncio
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Set, TYPE_CHECKING

from ..core.event_bus import EventBus, EventType
from ...data.orderbook_state import get_shared_orderbook_state

if TYPE_CHECKING:
    from ..services.trading_decision_service import TradingDecisionService, TradingDecision
    from ..core.state_container import V3StateContainer

logger = logging.getLogger("kalshiflow_rl.traderv3.services.yes_80_90")


# Configuration from environment (can be overridden)
DEFAULT_MIN_PRICE_CENTS = int(os.getenv("YES8090_MIN_PRICE", "80"))
DEFAULT_MAX_PRICE_CENTS = int(os.getenv("YES8090_MAX_PRICE", "90"))
DEFAULT_MIN_LIQUIDITY = int(os.getenv("YES8090_MIN_LIQUIDITY", "10"))
DEFAULT_MAX_SPREAD_CENTS = int(os.getenv("YES8090_MAX_SPREAD", "5"))
DEFAULT_CONTRACTS_PER_TRADE = int(os.getenv("YES8090_CONTRACTS", "100"))
DEFAULT_TIER_A_CONTRACTS = int(os.getenv("YES8090_TIER_A_CONTRACTS", "150"))
DEFAULT_MAX_CONCURRENT = int(os.getenv("YES8090_MAX_CONCURRENT", "100"))
DECISION_HISTORY_SIZE = 100

# Rate limiting: Very forgiving for demo environment with limited opportunities
# 10 trades per minute = 1 token per 6 seconds
DEFAULT_MAX_TRADES_PER_MINUTE = 10
DEFAULT_TOKEN_REFILL_SECONDS = 6.0


@dataclass
class Yes8090Signal:
    """
    Represents a detected YES at 80-90c trading signal.

    Attributes:
        market_ticker: Market ticker for this signal
        yes_ask_cents: Best YES ask price in cents
        yes_ask_size: Size available at best ask
        yes_bid_cents: Best YES bid price in cents
        spread_cents: Bid-ask spread in cents
        tier: Signal tier ('A' for 83-87c, 'B' for edges)
        detected_at: Timestamp when signal was detected
    """
    market_ticker: str
    yes_ask_cents: int
    yes_ask_size: int
    yes_bid_cents: int
    spread_cents: int
    tier: str
    detected_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/transport."""
        return {
            "market_ticker": self.market_ticker,
            "yes_ask_cents": self.yes_ask_cents,
            "yes_ask_size": self.yes_ask_size,
            "yes_bid_cents": self.yes_bid_cents,
            "spread_cents": self.spread_cents,
            "tier": self.tier,
            "detected_at": self.detected_at,
            "age_seconds": int(time.time() - self.detected_at)
        }


@dataclass
class Yes8090Decision:
    """
    Records a decision about a YES 80-90c signal.

    Attributes:
        signal_id: Unique identifier (market_ticker:timestamp)
        timestamp: When decision was made
        action: Decision action ("executed", "skipped_*", "failed")
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
            result["yes_ask_cents"] = self.signal_data.get("yes_ask_cents", 0)
            result["tier"] = self.signal_data.get("tier", "B")
        return result


class Yes8090Service:
    """
    Event-driven YES at 80-90c trading service.

    Subscribes to orderbook events and immediately processes signals,
    executing trades when conditions are met.

    Key Features:
        - Signal detection in 80-90c YES ask range
        - Liquidity and spread filtering
        - Deduplication (one entry per market per session)
        - Tier-based position sizing (A: 83-87c, B: edges)
        - Decision history tracking
    """

    def __init__(
        self,
        event_bus: EventBus,
        trading_service: 'TradingDecisionService',
        state_container: 'V3StateContainer',
        min_price_cents: int = DEFAULT_MIN_PRICE_CENTS,
        max_price_cents: int = DEFAULT_MAX_PRICE_CENTS,
        min_liquidity: int = DEFAULT_MIN_LIQUIDITY,
        max_spread_cents: int = DEFAULT_MAX_SPREAD_CENTS,
        contracts_per_trade: int = DEFAULT_CONTRACTS_PER_TRADE,
        tier_a_contracts: int = DEFAULT_TIER_A_CONTRACTS,
        max_concurrent: int = DEFAULT_MAX_CONCURRENT,
        max_trades_per_minute: int = DEFAULT_MAX_TRADES_PER_MINUTE,
        token_refill_seconds: float = DEFAULT_TOKEN_REFILL_SECONDS,
    ):
        """
        Initialize YES 80-90c service.

        Args:
            event_bus: V3 EventBus for event subscription
            trading_service: Service for executing trades
            state_container: Container for trading state (positions, orders)
            min_price_cents: Minimum YES ask price (default: 80)
            max_price_cents: Maximum YES ask price (default: 90)
            min_liquidity: Minimum contracts at best ask (default: 10)
            max_spread_cents: Maximum bid-ask spread (default: 5)
            contracts_per_trade: Default contracts per trade (default: 100)
            tier_a_contracts: Contracts for Tier A signals (default: 150)
            max_concurrent: Maximum concurrent positions (default: 100)
            max_trades_per_minute: Rate limit - trades per minute (default: 10)
            token_refill_seconds: Seconds between token refills (default: 6.0)
        """
        self._event_bus = event_bus
        self._trading_service = trading_service
        self._state_container = state_container

        # Signal detection parameters
        self._min_price = min_price_cents
        self._max_price = max_price_cents
        self._min_liquidity = min_liquidity
        self._max_spread = max_spread_cents
        self._contracts_per_trade = contracts_per_trade
        self._tier_a_contracts = tier_a_contracts
        self._max_concurrent = max_concurrent

        # Token bucket rate limiting (forgiving: 10/min for demo environment)
        self._max_tokens = max_trades_per_minute
        self._tokens = float(max_trades_per_minute)  # Start with full bucket
        self._token_refill_seconds = token_refill_seconds
        self._last_refill_time = time.time()
        self._rate_limited_count = 0

        # Deduplication: track markets we've already processed this session
        # Prevents re-entering a market after signal detected
        self._processed_markets: Set[str] = set()

        # Decision history (circular buffer)
        self._decision_history: deque[Yes8090Decision] = deque(maxlen=DECISION_HISTORY_SIZE)

        # Statistics
        self._signals_detected = 0
        self._signals_executed = 0
        self._signals_skipped = 0
        self._last_execution_time: Optional[float] = None

        # State
        self._running = False
        self._started_at: Optional[float] = None
        self._processing_lock = asyncio.Lock()

        logger.info(
            f"Yes8090Service initialized: price_range={min_price_cents}-{max_price_cents}c, "
            f"liquidity>={min_liquidity}, spread<={max_spread_cents}c, "
            f"contracts={contracts_per_trade} (Tier A: {tier_a_contracts}), "
            f"max_positions={max_concurrent}, rate_limit={max_trades_per_minute}/min"
        )

    async def start(self) -> None:
        """Start the YES 80-90c service."""
        if self._running:
            logger.warning("Yes8090Service is already running")
            return

        logger.info("Starting Yes8090Service")
        self._running = True
        self._started_at = time.time()

        # Subscribe to orderbook events
        await self._event_bus.subscribe_to_orderbook_snapshot(self._handle_orderbook)
        await self._event_bus.subscribe_to_orderbook_delta(self._handle_orderbook)
        logger.info("Subscribed to ORDERBOOK_SNAPSHOT and ORDERBOOK_DELTA events")

        # Emit startup event
        await self._event_bus.emit_system_activity(
            activity_type="strategy_start",
            message=f"YES 80-90c strategy started - monitoring for signals",
            metadata={
                "strategy": "YES_80_90",
                "price_range": f"{self._min_price}-{self._max_price}c",
                "max_positions": self._max_concurrent
            }
        )

        logger.info("Yes8090Service started successfully")

    async def stop(self) -> None:
        """Stop the YES 80-90c service."""
        if not self._running:
            return

        logger.info("Stopping Yes8090Service...")
        self._running = False

        # Emit shutdown event
        await self._event_bus.emit_system_activity(
            activity_type="strategy_stop",
            message=f"YES 80-90c strategy stopped",
            metadata={
                "strategy": "YES_80_90",
                "signals_detected": self._signals_detected,
                "signals_executed": self._signals_executed
            }
        )

        logger.info(
            f"Yes8090Service stopped. Stats: "
            f"detected={self._signals_detected}, executed={self._signals_executed}, "
            f"skipped={self._signals_skipped}"
        )

    async def _handle_orderbook(self, market_ticker: str, metadata: Dict[str, Any]) -> None:
        """
        Handle orderbook snapshot/delta events.

        Checks if the market has a valid YES 80-90c signal and executes
        if all conditions are met.

        Args:
            market_ticker: Market that had orderbook update
            metadata: Orderbook data including price levels
        """
        if not self._running:
            return

        # Use lock to prevent concurrent signal processing
        async with self._processing_lock:
            await self._process_orderbook(market_ticker, metadata)

    async def _process_orderbook(self, market_ticker: str, metadata: Dict[str, Any]) -> None:
        """
        Process orderbook update and detect signals.

        Args:
            market_ticker: Market ticker
            metadata: Orderbook data
        """
        # Skip if already processed this market
        if market_ticker in self._processed_markets:
            return

        # Check if we have too many positions
        trading_state = self._state_container.trading_state
        positions = trading_state.positions if trading_state else {}

        if len(positions) >= self._max_concurrent:
            # Don't log every time - only on new markets
            return

        # Skip if we already have a position in this market
        if market_ticker in positions:
            self._processed_markets.add(market_ticker)
            return

        # Skip if we have open orders in this market
        orders = trading_state.orders if trading_state else {}
        if any(o.get("ticker") == market_ticker for o in orders.values()):
            return

        # Get orderbook state for this market
        try:
            orderbook_state = await get_shared_orderbook_state(market_ticker)
            snapshot = await orderbook_state.get_snapshot()
        except Exception as e:
            logger.debug(f"Could not get orderbook state for {market_ticker}: {e}")
            return

        # Detect signal
        signal = self._detect_signal(market_ticker, snapshot)
        if signal:
            await self._execute_signal(signal)

    def _detect_signal(self, market_ticker: str, orderbook: Dict[str, Any]) -> Optional[Yes8090Signal]:
        """
        Detect if orderbook meets YES 80-90c signal criteria.

        Signal Criteria (ALL must be true):
            1. Best YES ask price >= min_price AND <= max_price
            2. Best YES ask size >= min_liquidity
            3. Bid-ask spread <= max_spread
            4. Market NOT already processed

        Args:
            market_ticker: Market ticker
            orderbook: Orderbook snapshot dict with yes_asks, yes_bids, etc.

        Returns:
            Yes8090Signal if criteria met, None otherwise
        """
        yes_asks = orderbook.get("yes_asks", {})
        yes_bids = orderbook.get("yes_bids", {})

        # No orderbook data
        if not yes_asks:
            return None

        # Get best YES ask (lowest price in asks dict)
        # yes_asks is already sorted ascending, so first key is lowest
        try:
            sorted_asks = sorted(yes_asks.keys())
            if not sorted_asks:
                return None
            best_ask_price = sorted_asks[0]
            best_ask_size = yes_asks[best_ask_price]
        except (IndexError, KeyError, TypeError):
            return None

        # Check price range
        if not (self._min_price <= best_ask_price <= self._max_price):
            return None

        # Check liquidity
        if best_ask_size < self._min_liquidity:
            return None

        # Get best YES bid for spread calculation
        try:
            sorted_bids = sorted(yes_bids.keys(), reverse=True)
            best_bid_price = sorted_bids[0] if sorted_bids else 0
        except (IndexError, TypeError):
            best_bid_price = 0

        # Calculate spread
        # Edge case: when no bid exists, spread defaults to 99 cents
        # This correctly filters out illiquid markets (spread > max_spread)
        spread = best_ask_price - best_bid_price if best_bid_price > 0 else 99

        # Check spread
        if spread > self._max_spread:
            return None

        # Determine signal tier
        # Tier A: 83-87c (sweet spot, higher edge)
        # Tier B: 80-83c or 87-90c (edges, still valid)
        tier = "A" if 83 <= best_ask_price <= 87 else "B"

        self._signals_detected += 1

        return Yes8090Signal(
            market_ticker=market_ticker,
            yes_ask_cents=best_ask_price,
            yes_ask_size=best_ask_size,
            yes_bid_cents=best_bid_price,
            spread_cents=spread,
            tier=tier
        )

    async def _execute_signal(self, signal: Yes8090Signal) -> None:
        """
        Execute a YES 80-90c signal by placing an order.

        Args:
            signal: Detected signal with all parameters
        """
        from ..services.trading_decision_service import TradingDecision, TradingStrategy

        signal_id = f"{signal.market_ticker}:{int(signal.detected_at * 1000)}"

        # Check rate limit before executing
        self._refill_tokens()
        if not self._consume_token():
            # Rate limited - don't mark as processed so we can retry
            self._rate_limited_count += 1
            logger.debug(f"Rate limited signal {signal_id}, will retry on next update")
            return

        # Mark as processed AFTER rate limit check passes
        self._processed_markets.add(signal.market_ticker)

        # Emit processing start
        await self._event_bus.emit_system_activity(
            activity_type="yes_80_90_signal",
            message=f"Signal detected: {signal.market_ticker} @ {signal.yes_ask_cents}c (Tier {signal.tier})",
            metadata={"signal_id": signal_id, "status": "processing", **signal.to_dict()}
        )

        try:
            # Calculate entry price
            entry_price = self._calculate_entry_price(
                signal.yes_ask_cents,
                signal.yes_bid_cents,
                signal.spread_cents
            )

            # Determine position size based on tier
            contracts = self._tier_a_contracts if signal.tier == "A" else self._contracts_per_trade

            # Create trading decision
            decision = TradingDecision(
                action="buy",
                market=signal.market_ticker,
                side="yes",
                quantity=contracts,
                price=entry_price,
                reason=f"YES 80-90c signal at {signal.yes_ask_cents}c (Tier {signal.tier})",
                confidence=0.85 if signal.tier == "A" else 0.75,
                strategy=TradingStrategy.YES_80_90,
            )

            logger.info(
                f"Executing YES 80-90c signal: {signal.market_ticker} "
                f"@ {entry_price}c, {contracts} contracts (Tier {signal.tier})"
            )

            # Execute through trading service
            success = await self._trading_service.execute_decision(decision)

            if success:
                self._signals_executed += 1
                self._last_execution_time = time.time()

                # Record successful decision
                self._record_decision(
                    signal_id=signal_id,
                    action="executed",
                    reason=f"Bought {contracts} YES @ {entry_price}c (Tier {signal.tier})",
                    signal_data=signal.to_dict()
                )

                # Emit success event
                await self._event_bus.emit_system_activity(
                    activity_type="yes_80_90_execute",
                    message=f"Executed: {signal.market_ticker} {contracts} YES @ {entry_price}c",
                    metadata={
                        "signal_id": signal_id,
                        "status": "success",
                        "market": signal.market_ticker,
                        "side": "yes",
                        "quantity": contracts,
                        "price": entry_price,
                        "tier": signal.tier
                    }
                )
            else:
                self._signals_skipped += 1

                self._record_decision(
                    signal_id=signal_id,
                    action="failed",
                    reason=f"Order execution failed for {signal.market_ticker}",
                    signal_data=signal.to_dict()
                )

                await self._event_bus.emit_system_activity(
                    activity_type="yes_80_90_execute",
                    message=f"Failed: {signal.market_ticker} order execution failed",
                    metadata={"signal_id": signal_id, "status": "failed"}
                )

        except Exception as e:
            self._signals_skipped += 1
            logger.error(f"Error executing YES 80-90c signal: {e}")

            self._record_decision(
                signal_id=signal_id,
                action="error",
                reason=f"Exception: {str(e)}",
                signal_data=signal.to_dict()
            )

    def _calculate_entry_price(
        self,
        best_ask: int,
        best_bid: int,
        spread: int
    ) -> int:
        """
        Calculate optimal entry price based on spread.

        Entry Price Strategy:
            - Tight spread (<=2c): best_ask - 1c (try to get filled)
            - Normal spread (<=4c): midpoint
            - Wide spread (>4c): best_bid + 1c (be aggressive)

        Args:
            best_ask: Best YES ask price in cents
            best_bid: Best YES bid price in cents
            spread: Bid-ask spread in cents

        Returns:
            Entry price in cents
        """
        if spread <= 2:
            # Tight spread - hit slightly below ask
            return best_ask - 1
        elif spread <= 4:
            # Normal spread - use midpoint
            return (best_ask + best_bid) // 2
        else:
            # Wide spread - be aggressive from bid side
            return best_bid + 1

    def _record_decision(
        self,
        signal_id: str,
        action: str,
        reason: str,
        order_id: Optional[str] = None,
        signal_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record a signal decision in history.

        Args:
            signal_id: Unique signal identifier
            action: Decision action (executed, skipped_*, error)
            reason: Human-readable explanation
            order_id: Order ID if executed
            signal_data: Original signal data
        """
        decision = Yes8090Decision(
            signal_id=signal_id,
            timestamp=time.time(),
            action=action,
            reason=reason,
            order_id=order_id,
            signal_data=signal_data,
        )
        self._decision_history.append(decision)

        logger.debug(f"YES 80-90c decision: {action} - {reason}")

    def _refill_tokens(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self._last_refill_time

        # Calculate tokens to add
        tokens_to_add = elapsed / self._token_refill_seconds

        # Add tokens (cap at max)
        self._tokens = min(self._max_tokens, self._tokens + tokens_to_add)
        self._last_refill_time = now

    def _consume_token(self) -> bool:
        """
        Try to consume a token for a trade.

        Returns:
            True if token was consumed, False if rate limited
        """
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False

    def get_decision_history(self) -> List[Dict[str, Any]]:
        """
        Get recent decision history.

        Returns:
            List of decision dicts, most recent first
        """
        return [d.to_dict() for d in reversed(self._decision_history)]

    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        uptime = time.time() - self._started_at if self._started_at else 0

        return {
            "running": self._running,
            "uptime_seconds": uptime,
            "signals_detected": self._signals_detected,
            "signals_executed": self._signals_executed,
            "signals_skipped": self._signals_skipped,
            "rate_limited_count": self._rate_limited_count,
            "markets_processed": len(self._processed_markets),
            "last_execution_time": self._last_execution_time,
            "decision_history_size": len(self._decision_history),
            "tokens_available": round(self._tokens, 1),
            "config": {
                "price_range": f"{self._min_price}-{self._max_price}c",
                "min_liquidity": self._min_liquidity,
                "max_spread": self._max_spread,
                "contracts_default": self._contracts_per_trade,
                "contracts_tier_a": self._tier_a_contracts,
                "max_concurrent": self._max_concurrent,
                "rate_limit_per_min": self._max_tokens,
            }
        }

    def get_health_details(self) -> Dict[str, Any]:
        """Get detailed health information."""
        stats = self.get_stats()
        return {
            "healthy": self._running,
            "running": self._running,
            "signals_detected": stats["signals_detected"],
            "signals_executed": stats["signals_executed"],
            "markets_processed": stats["markets_processed"],
            "uptime_seconds": stats["uptime_seconds"],
        }

    def is_healthy(self) -> bool:
        """Check if service is healthy."""
        return self._running

    def reset_processed_markets(self) -> int:
        """
        Reset the processed markets set.

        Used for testing or to allow re-entry into markets after
        positions have been closed.

        Returns:
            Number of markets that were cleared
        """
        count = len(self._processed_markets)
        self._processed_markets.clear()
        logger.info(f"Reset processed markets - cleared {count} entries")
        return count
