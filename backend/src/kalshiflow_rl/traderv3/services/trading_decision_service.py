"""
Trading Decision Service for V3 trader.

Handles trading logic and decision-making for the V3 trading system.
This service is designed to be the single point for all trading decisions,
supporting both hardcoded strategies and RL model integration.

Key Responsibilities:
    1. Evaluate market conditions and generate trading decisions
    2. Execute trades through the trading client
    3. Support multiple strategies (HOLD, WHALE_FOLLOWER, RL, custom)
    4. Track decision history and whale follow state

Design Principles:
    - Single point for all trading decisions
    - Strategy pattern for easy extension
    - Event-driven updates via EventBus
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from enum import Enum
from dataclasses import dataclass, field
from collections import deque

if TYPE_CHECKING:
    from ..clients.trading_client_integration import V3TradingClientIntegration
    from ..core.state_container import V3StateContainer
    from ..core.event_bus import EventBus
    from ..services.whale_tracker import WhaleTracker
    from kalshiflow_rl.data.models import OrderbookSnapshot

logger = logging.getLogger("kalshiflow_rl.traderv3.services.trading_decision")

# Constants for whale following
WHALE_FOLLOW_CONTRACTS = 5  # Fixed 5 contracts per follow
WHALE_MAX_AGE_SECONDS = 120  # Skip whales older than 2 minutes


class TradingStrategy(Enum):
    """Available trading strategies."""
    HOLD = "hold"  # Never trade (safe default)
    WHALE_FOLLOWER = "whale_follower"  # Follow the Whale strategy
    PAPER_TEST = "paper_test"  # Simple test trades for paper mode
    RL_MODEL = "rl_model"  # Use trained RL model
    CUSTOM = "custom"  # Custom strategy implementation


@dataclass
class FollowedWhale:
    """
    Tracks a whale trade that we have followed.

    Attributes:
        whale_id: Unique identifier (market_ticker:timestamp_ms)
        our_order_id: Order ID from Kalshi response
        placed_at: When we placed the follow order (time.time())
        market_ticker: Market ticker
        side: "yes" or "no"
        our_count: Number of contracts we bought
        whale_size_cents: Original whale size in cents
    """
    whale_id: str
    our_order_id: str
    placed_at: float
    market_ticker: str
    side: str
    our_count: int
    price_cents: int  # Price we bought at
    whale_size_cents: int

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for WebSocket transport to frontend."""
        # Calculate cost and payout like whale queue does
        cost_cents = self.our_count * self.price_cents
        payout_cents = self.our_count * 100  # $1.00 per contract if wins
        return {
            "whale_id": self.whale_id,
            "order_id": self.our_order_id[:8] if self.our_order_id else "unknown",
            "market_ticker": self.market_ticker,
            "side": self.side,
            "our_count": self.our_count,
            "price_cents": self.price_cents,
            "cost_dollars": cost_cents / 100,
            "payout_dollars": payout_cents / 100,
            "size_dollars": cost_cents / 100,  # Same as cost for our trades
            "whale_size_cents": self.whale_size_cents,
            "placed_at": self.placed_at,
            "age_seconds": int(time.time() - self.placed_at)
        }


@dataclass
class WhaleDecision:
    """
    Tracks a decision made about a whale trade.

    Used for auditing and visibility into why whales were followed/skipped.

    Attributes:
        whale_id: Unique identifier (market_ticker:timestamp_ms)
        timestamp: When the decision was made (time.time())
        action: Decision action ("followed", "skipped_age", "skipped_position", "skipped_orders", "already_followed", "rate_limited")
        reason: Human-readable explanation
        order_id: Order ID if whale was followed (None otherwise)
        market_ticker: Market ticker
        side: "yes" or "no"
        whale_size_dollars: Size of the whale trade in dollars
    """
    whale_id: str
    timestamp: float
    action: str
    reason: str
    order_id: Optional[str] = None
    market_ticker: str = ""
    side: str = ""
    whale_size_dollars: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for WebSocket transport to frontend."""
        return {
            "whale_id": self.whale_id,
            "timestamp": self.timestamp,
            "action": self.action,
            "reason": self.reason,
            "order_id": self.order_id[:8] if self.order_id else None,
            "market_ticker": self.market_ticker,
            "side": self.side,
            "whale_size_dollars": self.whale_size_dollars,
            "age_seconds": int(time.time() - self.timestamp)
        }


@dataclass
class TradingDecision:
    """Represents a trading decision."""
    action: str  # "buy", "sell", "hold"
    market: str
    side: str  # "yes" or "no"
    quantity: int
    price: Optional[int] = None  # If None, use market order
    reason: str = ""
    confidence: float = 0.0
    strategy: TradingStrategy = TradingStrategy.HOLD


class TradingDecisionService:
    """
    Service for making trading decisions in the V3 trader.

    Responsibilities:
    - Evaluate market conditions
    - Generate trading decisions
    - Execute trades through trading client
    - Track decision history
    - Support multiple strategies (HOLD, WHALE_FOLLOWER, RL, custom)
    """

    def __init__(
        self,
        trading_client: Optional['V3TradingClientIntegration'],
        state_container: 'V3StateContainer',
        event_bus: 'EventBus',
        strategy: TradingStrategy = TradingStrategy.HOLD,
        whale_tracker: Optional['WhaleTracker'] = None
    ):
        """
        Initialize trading decision service.

        Args:
            trading_client: Trading client integration for order execution
            state_container: State container for accessing system state
            event_bus: Event bus for emitting trading events
            strategy: Trading strategy to use
            whale_tracker: Optional whale tracker for WHALE_FOLLOWER strategy
        """
        self._trading_client = trading_client
        self._state_container = state_container
        self._event_bus = event_bus
        self._strategy = strategy
        self._whale_tracker = whale_tracker

        # Decision tracking
        self._decision_count = 0
        self._trade_count = 0
        self._last_decision: Optional[TradingDecision] = None

        # Whale following state (in-memory, cleared on restart)
        self._followed_whales: Dict[str, FollowedWhale] = {}

        # Decision history for audit trail (last 100 decisions)
        self._decision_history: deque[WhaleDecision] = deque(maxlen=100)
        self._decision_stats = {
            "whales_detected": 0,
            "whales_followed": 0,
            "whales_skipped": 0,
            "skipped_age": 0,
            "skipped_position": 0,
            "skipped_orders": 0,
            "already_followed": 0,
            "rate_limited": 0
        }

        # RL model (loaded lazily if needed)
        self._rl_model = None
        self._rl_model_loaded = False

        logger.info(f"Trading Decision Service initialized with strategy: {strategy.value}")
        if whale_tracker and strategy == TradingStrategy.WHALE_FOLLOWER:
            logger.info("Whale Follower strategy enabled - will follow big bets")
    
    async def evaluate_market(
        self, 
        market: str, 
        orderbook: Optional['OrderbookSnapshot'] = None
    ) -> Optional[TradingDecision]:
        """
        Evaluate a market and generate a trading decision.
        
        Args:
            market: Market ticker to evaluate
            orderbook: Current orderbook snapshot (optional)
        
        Returns:
            Trading decision or None if no action needed
        """
        self._decision_count += 1
        
        # Get current positions and orders
        trading_state = self._state_container.trading_state
        positions = trading_state.positions if trading_state else {}
        orders = trading_state.orders if trading_state else {}

        # Check if we already have exposure in this market
        has_position = market in positions
        # orders is Dict[order_id, order_data] where order_data has 'ticker' key
        has_orders = any(o.get("ticker") == market for o in orders.values())
        
        if has_orders:
            logger.debug(f"Skipping {market} - has open orders")
            return None
        
        # Apply strategy
        decision = None
        
        if self._strategy == TradingStrategy.HOLD:
            # Never trade
            decision = TradingDecision(
                action="hold",
                market=market,
                side="",
                quantity=0,
                reason="HOLD strategy - no trading",
                strategy=self._strategy
            )

        elif self._strategy == TradingStrategy.WHALE_FOLLOWER:
            # Whale follower is handled differently in TradingFlowOrchestrator
            # via evaluate_whale_queue() - this is a fallback for market-based calls
            decision = TradingDecision(
                action="hold",
                market=market,
                side="",
                quantity=0,
                reason="WHALE_FOLLOWER uses evaluate_whale_queue()",
                strategy=self._strategy
            )

        elif self._strategy == TradingStrategy.PAPER_TEST:
            # Simple test strategy for paper mode
            decision = await self._paper_test_strategy(
                market, orderbook, has_position
            )
        
        elif self._strategy == TradingStrategy.RL_MODEL:
            # Use RL model for decision
            decision = await self._rl_model_strategy(
                market, orderbook, has_position
            )
        
        elif self._strategy == TradingStrategy.CUSTOM:
            # Custom strategy implementation
            decision = await self._custom_strategy(
                market, orderbook, has_position
            )
        
        # Store decision
        if decision and decision.action != "hold":
            self._last_decision = decision
            logger.info(
                f"Trading decision for {market}: {decision.action} "
                f"{decision.quantity} {decision.side} @ {decision.price or 'market'} "
                f"(reason: {decision.reason})"
            )
        
        return decision
    
    async def _paper_test_strategy(
        self,
        market: str,
        orderbook: Optional['OrderbookSnapshot'],
        has_position: bool
    ) -> TradingDecision:
        """
        Simple test strategy for paper trading validation.
        
        Buys 5 contracts at mid-price if no position,
        sells if we have a position.
        """
        if not orderbook:
            return TradingDecision(
                action="hold",
                market=market,
                side="",
                quantity=0,
                reason="No orderbook data",
                strategy=self._strategy
            )
        
        # Calculate mid price
        best_bid = orderbook.yes[0][0] if orderbook.yes else 0
        best_ask = orderbook.no[0][0] if orderbook.no else 100
        mid_price = (best_bid + best_ask) // 2
        
        if has_position:
            # Sell position
            return TradingDecision(
                action="sell",
                market=market,
                side="yes",  # Assuming we bought yes
                quantity=5,
                price=mid_price,
                reason="Paper test: closing position",
                confidence=0.5,
                strategy=self._strategy
            )
        else:
            # Buy if price is reasonable
            if 20 <= mid_price <= 80:
                return TradingDecision(
                    action="buy",
                    market=market,
                    side="yes" if mid_price < 50 else "no",
                    quantity=5,
                    price=mid_price,
                    reason=f"Paper test: mid price {mid_price}",
                    confidence=0.5,
                    strategy=self._strategy
                )
        
        return TradingDecision(
            action="hold",
            market=market,
            side="",
            quantity=0,
            reason="Paper test: conditions not met",
            strategy=self._strategy
        )
    
    async def _rl_model_strategy(
        self,
        market: str,
        orderbook: Optional['OrderbookSnapshot'],
        has_position: bool
    ) -> TradingDecision:
        """
        Use RL model for trading decision.
        """
        # TODO: Implement RL model integration
        # For now, return hold
        return TradingDecision(
            action="hold",
            market=market,
            side="",
            quantity=0,
            reason="RL model not yet integrated",
            strategy=self._strategy
        )
    
    async def _custom_strategy(
        self,
        market: str,
        orderbook: Optional['OrderbookSnapshot'],
        has_position: bool
    ) -> TradingDecision:
        """
        Custom trading strategy implementation.
        Override this method for custom strategies.
        """
        return TradingDecision(
            action="hold",
            market=market,
            side="",
            quantity=0,
            reason="Custom strategy not implemented",
            strategy=self._strategy
        )

    def _record_whale_decision(
        self,
        whale_id: str,
        action: str,
        reason: str,
        order_id: Optional[str] = None,
        market_ticker: str = "",
        side: str = "",
        whale_size_dollars: float = 0.0
    ) -> None:
        """
        Record a whale decision in the audit history.

        Args:
            whale_id: Unique identifier for the whale
            action: Decision action (followed, skipped_*, etc.)
            reason: Human-readable explanation
            order_id: Order ID if whale was followed
            market_ticker: Market ticker
            side: Side of the trade
            whale_size_dollars: Size of the whale trade
        """
        decision = WhaleDecision(
            whale_id=whale_id,
            timestamp=time.time(),
            action=action,
            reason=reason,
            order_id=order_id,
            market_ticker=market_ticker,
            side=side,
            whale_size_dollars=whale_size_dollars
        )
        self._decision_history.append(decision)

        # Update stats
        self._decision_stats["whales_detected"] += 1
        if action == "followed":
            self._decision_stats["whales_followed"] += 1
        else:
            self._decision_stats["whales_skipped"] += 1
            # Track specific skip reasons
            if action in self._decision_stats:
                self._decision_stats[action] += 1

    async def evaluate_whale_queue(
        self,
        whale_queue: List[Dict[str, Any]]
    ) -> Optional[TradingDecision]:
        """
        Evaluate the whale queue and generate a trading decision.

        This is the main entry point for WHALE_FOLLOWER strategy.
        Iterates through whales in priority order and returns a decision
        for the first valid whale to follow.

        Records all decisions (follow and skip) in the audit history.

        Args:
            whale_queue: List of whale dicts from WhaleTracker.get_queue_state()

        Returns:
            TradingDecision to follow a whale, or None if no valid whale found
        """
        if not whale_queue:
            logger.debug("No whales in queue")
            return None

        # Get current positions to check for existing exposure
        trading_state = self._state_container.trading_state
        positions = trading_state.positions if trading_state else {}
        orders = trading_state.orders if trading_state else {}

        # Build set of markets with open orders for fast lookup
        # orders is Dict[order_id, order_data] where order_data has 'ticker' key
        markets_with_orders = {o.get("ticker") for o in orders.values() if o.get("ticker")}

        now_ms = int(time.time() * 1000)

        for whale in whale_queue:
            # Build whale ID
            market_ticker = whale.get("market_ticker", "")
            timestamp_ms = whale.get("timestamp_ms", 0)
            whale_id = f"{market_ticker}:{timestamp_ms}"
            side = whale.get("side", "yes")
            whale_size_dollars = whale.get("whale_size_dollars", 0)

            # Skip if already followed
            if whale_id in self._followed_whales:
                logger.debug(f"Skipping whale {whale_id[:20]}... - already followed")
                self._record_whale_decision(
                    whale_id=whale_id,
                    action="already_followed",
                    reason="Already followed this whale",
                    market_ticker=market_ticker,
                    side=side,
                    whale_size_dollars=whale_size_dollars
                )
                continue

            # Skip if too old
            age_seconds = (now_ms - timestamp_ms) / 1000.0
            if age_seconds > WHALE_MAX_AGE_SECONDS:
                logger.debug(f"Skipping whale {whale_id[:20]}... - too old ({age_seconds:.1f}s)")
                self._record_whale_decision(
                    whale_id=whale_id,
                    action="skipped_age",
                    reason=f"Too old ({age_seconds:.0f}s > {WHALE_MAX_AGE_SECONDS}s limit)",
                    market_ticker=market_ticker,
                    side=side,
                    whale_size_dollars=whale_size_dollars
                )
                continue

            # Skip if we have position in this market
            if market_ticker in positions:
                logger.debug(f"Skipping whale {whale_id[:20]}... - already have position")
                self._record_whale_decision(
                    whale_id=whale_id,
                    action="skipped_position",
                    reason=f"Already have position in {market_ticker}",
                    market_ticker=market_ticker,
                    side=side,
                    whale_size_dollars=whale_size_dollars
                )
                continue

            # Skip if we have open orders in this market
            if market_ticker in markets_with_orders:
                logger.debug(f"Skipping whale {whale_id[:20]}... - have open orders")
                self._record_whale_decision(
                    whale_id=whale_id,
                    action="skipped_orders",
                    reason=f"Have open orders in {market_ticker}",
                    market_ticker=market_ticker,
                    side=side,
                    whale_size_dollars=whale_size_dollars
                )
                continue

            # Found a valid whale to follow
            price_cents = whale.get("price_cents", 50)
            whale_size_cents = int(whale_size_dollars * 100)

            logger.info(
                f"Following whale: {market_ticker} {side.upper()} @ {price_cents}c "
                f"(whale size: ${whale_size_cents/100:.2f}, age: {age_seconds:.1f}s)"
            )

            # Note: Decision is recorded in _execute_buy when order_id is available
            # We return the TradingDecision here, and the order execution will record
            # the "followed" decision with the actual order_id

            return TradingDecision(
                action="buy",
                market=market_ticker,
                side=side,
                quantity=WHALE_FOLLOW_CONTRACTS,
                price=price_cents,  # Same price as whale
                reason=f"whale:{whale_id}",
                confidence=0.7,
                strategy=TradingStrategy.WHALE_FOLLOWER
            )

        logger.debug("No valid whales to follow")
        return None

    def _extract_whale_id_from_reason(self, reason: str) -> Optional[str]:
        """
        Extract whale ID from decision reason.

        Args:
            reason: Decision reason string (format: "whale:{market}:{timestamp}")

        Returns:
            Whale ID or None if not a whale follow
        """
        if reason.startswith("whale:"):
            return reason[6:]  # Strip "whale:" prefix
        return None

    async def execute_decision(self, decision: TradingDecision) -> bool:
        """
        Execute a trading decision.
        
        Args:
            decision: Trading decision to execute
        
        Returns:
            True if execution successful, False otherwise
        """
        if not self._trading_client:
            logger.error("No trading client configured - cannot execute trades")
            return False
        
        if decision.action == "hold":
            return True  # No action needed
        
        try:
            # Emit decision event
            await self._event_bus.emit_system_activity(
                activity_type="trading_decision",
                message=f"Executing {decision.action} {decision.quantity} {decision.side} on {decision.market}",
                metadata={
                    "market": decision.market,
                    "action": decision.action,
                    "side": decision.side,
                    "quantity": decision.quantity,
                    "price": decision.price,
                    "strategy": decision.strategy.value
                }
            )
            
            # Execute based on action type
            if decision.action == "buy":
                success = await self._execute_buy(decision)
            elif decision.action == "sell":
                success = await self._execute_sell(decision)
            else:
                logger.warning(f"Unknown action: {decision.action}")
                return False
            
            if success:
                self._trade_count += 1
                logger.info(f"Successfully executed trade #{self._trade_count}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error executing decision: {e}")
            await self._event_bus.emit_system_activity(
                activity_type="trading_error",
                message=f"Failed to execute trade: {str(e)}",
                metadata={"error": str(e), "decision": decision.__dict__}
            )
            return False
    
    async def _execute_buy(self, decision: TradingDecision) -> bool:
        """
        Execute a buy order through the trading client.

        For whale following, also tracks the follow in _followed_whales.

        Args:
            decision: Trading decision with buy details

        Returns:
            True if order placed successfully, False otherwise
        """
        try:
            # Get order group ID for portfolio limits
            order_group_id = self._trading_client.get_order_group_id()

            # Place the order
            response = await self._trading_client.place_order(
                ticker=decision.market,
                action="buy",
                side=decision.side,
                count=decision.quantity,
                price=decision.price,
                order_type="limit",
                order_group_id=order_group_id
            )

            order_id = response.get("order", {}).get("order_id", "unknown")

            logger.info(
                f"BUY order placed: {decision.quantity} {decision.side} {decision.market} "
                f"@ {decision.price}c (order_id: {order_id[:8]}...)"
            )

            # Track whale follow if this is a whale following decision
            whale_id = self._extract_whale_id_from_reason(decision.reason)
            if whale_id:
                whale_size_cents = 0
                # Extract whale size from reason if available (format: whale:market:timestamp)
                parts = whale_id.split(":")
                if len(parts) >= 2:
                    market_ticker = parts[0]
                    timestamp_ms = int(parts[1]) if parts[1].isdigit() else 0
                else:
                    market_ticker = decision.market
                    timestamp_ms = 0

                self._followed_whales[whale_id] = FollowedWhale(
                    whale_id=whale_id,
                    our_order_id=order_id,
                    placed_at=time.time(),
                    market_ticker=market_ticker,
                    side=decision.side,
                    our_count=decision.quantity,
                    price_cents=decision.price or 0,
                    whale_size_cents=whale_size_cents
                )

                # Record the successful follow decision in audit history
                self._record_whale_decision(
                    whale_id=whale_id,
                    action="followed",
                    reason=f"Followed ${whale_size_cents/100:.2f} whale at {decision.price}c",
                    order_id=order_id,
                    market_ticker=market_ticker,
                    side=decision.side,
                    whale_size_dollars=whale_size_cents / 100
                )

                logger.info(
                    f"Tracked whale follow: {whale_id[:30]}... "
                    f"(total followed: {len(self._followed_whales)})"
                )

                # Emit whale follow event
                await self._event_bus.emit_system_activity(
                    activity_type="whale_follow",
                    message=f"Following whale on {market_ticker} {decision.side.upper()}",
                    metadata={
                        "whale_id": whale_id,
                        "order_id": order_id,
                        "market": market_ticker,
                        "side": decision.side,
                        "quantity": decision.quantity,
                        "price": decision.price
                    }
                )

            return True

        except Exception as e:
            logger.error(f"BUY order failed: {e}")
            await self._event_bus.emit_system_activity(
                activity_type="trading_error",
                message=f"BUY order failed: {str(e)}",
                metadata={
                    "market": decision.market,
                    "side": decision.side,
                    "quantity": decision.quantity,
                    "price": decision.price,
                    "error": str(e)
                }
            )
            return False

    async def _execute_sell(self, decision: TradingDecision) -> bool:
        """
        Execute a sell order through the trading client.

        Args:
            decision: Trading decision with sell details

        Returns:
            True if order placed successfully, False otherwise
        """
        try:
            # Get order group ID for portfolio limits
            order_group_id = self._trading_client.get_order_group_id()

            # Place the order
            response = await self._trading_client.place_order(
                ticker=decision.market,
                action="sell",
                side=decision.side,
                count=decision.quantity,
                price=decision.price,
                order_type="limit",
                order_group_id=order_group_id
            )

            order_id = response.get("order", {}).get("order_id", "unknown")

            logger.info(
                f"SELL order placed: {decision.quantity} {decision.side} {decision.market} "
                f"@ {decision.price}c (order_id: {order_id[:8]}...)"
            )

            return True

        except Exception as e:
            logger.error(f"SELL order failed: {e}")
            await self._event_bus.emit_system_activity(
                activity_type="trading_error",
                message=f"SELL order failed: {str(e)}",
                metadata={
                    "market": decision.market,
                    "side": decision.side,
                    "quantity": decision.quantity,
                    "price": decision.price,
                    "error": str(e)
                }
            )
            return False
    
    def set_strategy(self, strategy: TradingStrategy) -> None:
        """Change the trading strategy."""
        logger.info(f"Changing strategy from {self._strategy.value} to {strategy.value}")
        self._strategy = strategy
    
    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        stats = {
            "strategy": self._strategy.value,
            "decision_count": self._decision_count,
            "trade_count": self._trade_count,
            "last_decision": self._last_decision.__dict__ if self._last_decision else None,
            "rl_model_loaded": self._rl_model_loaded,
            "whales_followed": len(self._followed_whales)
        }

        # Add whale tracker availability info
        if self._strategy == TradingStrategy.WHALE_FOLLOWER:
            stats["whale_tracker_available"] = self._whale_tracker is not None
            if self._whale_tracker:
                queue_state = self._whale_tracker.get_queue_state()
                stats["whale_queue_size"] = queue_state["stats"]["queue_size"]

        return stats
    
    def is_healthy(self) -> bool:
        """Check if service is healthy."""
        return True  # Simple health check for now

    def get_followed_whale_ids(self) -> set:
        """
        Get set of whale IDs that have been followed.

        Returns:
            Set of whale IDs (format: "{market_ticker}:{timestamp_ms}")
            that have been followed by this trading session.
        """
        return set(self._followed_whales.keys())

    def get_followed_whales(self) -> List[Dict[str, Any]]:
        """
        Get all followed whales with full details for frontend display.

        Returns:
            List of dicts with whale details including:
            - whale_id, order_id, market_ticker, side
            - our_count, whale_size_cents, placed_at, age_seconds
        """
        return [fw.to_dict() for fw in self._followed_whales.values()]

    def get_decision_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get recent whale decision history for audit display.

        Args:
            limit: Maximum number of decisions to return (default 20)

        Returns:
            List of decision dicts, most recent first, including:
            - whale_id, timestamp, action, reason
            - order_id (if followed), market_ticker, side, whale_size_dollars
        """
        # Return most recent decisions first, limited to requested count
        decisions = list(self._decision_history)
        decisions.reverse()  # Most recent first
        return [d.to_dict() for d in decisions[:limit]]

    def get_decision_stats(self) -> Dict[str, Any]:
        """
        Get whale decision statistics.

        Returns:
            Dict containing:
            - whales_detected: Total whales evaluated
            - whales_followed: Whales we followed
            - whales_skipped: Whales we skipped (all reasons)
            - skipped_age: Skipped due to age limit
            - skipped_position: Skipped due to existing position
            - skipped_orders: Skipped due to open orders
            - already_followed: Skipped because already followed
            - rate_limited: Skipped due to rate limiting
        """
        return self._decision_stats.copy()