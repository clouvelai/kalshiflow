"""
Whale Execution Service for TRADER V3 - Event-Driven Whale Following.

Purpose:
    This service subscribes to WHALE_QUEUE_UPDATED events and immediately
    executes trades to follow whales, removing the 30-second cycle delay
    from the trading flow orchestrator.

Key Responsibilities:
    1. **Event Subscription** - Subscribe to WHALE_QUEUE_UPDATED from EventBus
    2. **Token Bucket Rate Limiting** - Configurable trades/minute limit
    3. **Immediate Execution** - Process whales as they arrive (no cycle delays)
    4. **Decision History** - Track why each whale was followed/skipped
    5. **Delegation** - Use TradingDecisionService.execute_decision() for orders

Architecture Position:
    The WhaleExecutionService sits between the WhaleTracker and TradingDecisionService:
    - WhaleTracker: Detects whales, emits WHALE_QUEUE_UPDATED events
    - WhaleExecutionService: Receives events, validates whales, rate limits, executes
    - TradingDecisionService: Executes actual orders via trading client

Design Principles:
    - **Event-driven**: Reacts to whale queue updates immediately
    - **Rate Limited**: Token bucket prevents excessive trading
    - **Non-blocking**: Async processing, never blocks event loop
    - **Observable**: Complete decision history for debugging
"""

import asyncio
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, TYPE_CHECKING

from ..core.event_bus import EventBus, EventType, WhaleQueueEvent

if TYPE_CHECKING:
    from ..services.trading_decision_service import TradingDecisionService, TradingDecision
    from ..services.whale_tracker import WhaleTracker
    from ..core.state_container import V3StateContainer

logger = logging.getLogger("kalshiflow_rl.traderv3.services.whale_execution")


# Configuration constants (can be overridden by environment variables)
DEFAULT_MAX_TRADES_PER_MINUTE = int(os.getenv("WHALE_MAX_TRADES_PER_MINUTE", "3"))
DEFAULT_TOKEN_REFILL_SECONDS = float(os.getenv("WHALE_TOKEN_REFILL_SECONDS", "20"))
WHALE_MAX_AGE_SECONDS = 120  # Skip whales older than 2 minutes
DECISION_HISTORY_SIZE = 100


@dataclass
class WhaleDecision:
    """
    Records a decision about whether to follow a whale.

    This provides complete visibility into why whales were/weren't followed,
    enabling debugging and strategy refinement.

    Attributes:
        whale_id: Unique identifier (market_ticker:timestamp_ms)
        timestamp: When the decision was made (time.time())
        action: What happened ("followed", "skipped_age", "skipped_position", etc.)
        reason: Human-readable explanation
        order_id: Kalshi order ID if followed, None otherwise
        whale_data: Original whale data dict for reference
    """
    whale_id: str
    timestamp: float
    action: str
    reason: str
    order_id: Optional[str] = None
    whale_data: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for WebSocket transport."""
        result = {
            "whale_id": self.whale_id,
            "timestamp": self.timestamp,
            "action": self.action,
            "reason": self.reason,
            "order_id": self.order_id[:8] if self.order_id else None,
            "age_seconds": int(time.time() - self.timestamp),
        }
        # Include whale data fields needed by Decision Audit panel
        if self.whale_data:
            result["market_ticker"] = self.whale_data.get("market_ticker", "")
            result["side"] = self.whale_data.get("side", "")
            result["whale_size_dollars"] = self.whale_data.get("whale_size_dollars", 0)
        return result


class WhaleExecutionService:
    """
    Event-driven whale execution with token bucket rate limiting.

    Subscribes to WHALE_QUEUE_UPDATED events and immediately processes
    whales, bypassing the 30-second trading cycle delay.

    Key Features:
        - Token bucket rate limiting (configurable trades/minute)
        - Immediate whale processing on queue updates
        - Complete decision history tracking
        - Skip logic for age, positions, orders, rate limits
    """

    def __init__(
        self,
        event_bus: EventBus,
        trading_service: 'TradingDecisionService',
        state_container: 'V3StateContainer',
        whale_tracker: Optional['WhaleTracker'] = None,
        max_trades_per_minute: int = DEFAULT_MAX_TRADES_PER_MINUTE,
        token_refill_seconds: float = DEFAULT_TOKEN_REFILL_SECONDS,
    ):
        """
        Initialize whale execution service.

        Args:
            event_bus: V3 EventBus for event subscription
            trading_service: Service for executing trades
            state_container: Container for trading state (positions, orders)
            whale_tracker: Optional tracker for removing processed whales from queue
            max_trades_per_minute: Maximum trades per minute (token bucket capacity)
            token_refill_seconds: Seconds between token refills
        """
        self._event_bus = event_bus
        self._trading_service = trading_service
        self._state_container = state_container
        self._whale_tracker = whale_tracker

        # Token bucket configuration
        self._max_tokens = max_trades_per_minute
        self._tokens = float(max_trades_per_minute)  # Start with full bucket
        self._token_refill_seconds = token_refill_seconds
        self._last_refill_time = time.time()

        # Decision history (circular buffer)
        self._decision_history: deque[WhaleDecision] = deque(maxlen=DECISION_HISTORY_SIZE)

        # Statistics
        self._whales_processed = 0
        self._whales_followed = 0
        self._whales_skipped = 0
        self._rate_limited_count = 0
        self._last_execution_time: Optional[float] = None

        # Deduplication: track whale IDs that have been evaluated (followed OR skipped)
        # This prevents re-evaluating the same whale on every WHALE_QUEUE_UPDATED event
        self._evaluated_whale_ids: set[str] = set()

        # State
        self._running = False
        self._started_at: Optional[float] = None
        self._processing_lock = asyncio.Lock()

        logger.info(
            f"WhaleExecutionService initialized: max_trades={max_trades_per_minute}/min, "
            f"refill_rate=1/{token_refill_seconds}s"
        )

    async def start(self) -> None:
        """Start the whale execution service."""
        if self._running:
            logger.warning("WhaleExecutionService is already running")
            return

        logger.info("Starting WhaleExecutionService")
        self._running = True
        self._started_at = time.time()

        # Subscribe to whale queue events
        await self._event_bus.subscribe_to_whale_queue(self._handle_whale_queue)
        logger.info("Subscribed to WHALE_QUEUE_UPDATED events")

        logger.info("WhaleExecutionService started")

    async def stop(self) -> None:
        """Stop the whale execution service."""
        if not self._running:
            return

        logger.info("Stopping WhaleExecutionService...")
        self._running = False

        logger.info(
            f"WhaleExecutionService stopped. Stats: "
            f"processed={self._whales_processed}, followed={self._whales_followed}, "
            f"skipped={self._whales_skipped}, rate_limited={self._rate_limited_count}"
        )

    async def _handle_whale_queue(self, event: WhaleQueueEvent) -> None:
        """
        Handle WHALE_QUEUE_UPDATED event.

        Iterates through the whale queue and executes valid whales
        immediately (no 30-second cycle delay).

        Args:
            event: WhaleQueueEvent containing the current whale queue
        """
        if not self._running:
            return

        # Use lock to prevent concurrent processing
        async with self._processing_lock:
            await self._process_whale_queue(event.queue or [])

    async def _process_whale_queue(self, whale_queue: List[Dict[str, Any]]) -> None:
        """
        Process the whale queue and execute valid whales.

        DEDUPLICATION: Each whale is only evaluated ONCE. After evaluation
        (whether followed, skipped, or failed), the whale_id is added to
        _evaluated_whale_ids and will be ignored on subsequent queue updates.

        Args:
            whale_queue: List of whale dicts from WhaleTracker
        """
        if not whale_queue:
            return

        # Refill tokens before processing
        self._refill_tokens()

        # Get current positions and orders for filtering
        trading_state = self._state_container.trading_state
        positions = trading_state.positions if trading_state else {}
        orders = trading_state.orders if trading_state else {}

        # Build set of markets with open orders for fast lookup
        markets_with_orders = {
            o.get("ticker") for o in orders.values() if o.get("ticker")
        }

        now_ms = int(time.time() * 1000)
        new_whales_evaluated = 0

        for whale in whale_queue:
            # Build whale ID
            market_ticker = whale.get("market_ticker", "")
            timestamp_ms = whale.get("timestamp_ms", 0)
            whale_id = f"{market_ticker}:{timestamp_ms}"

            # CRITICAL: Skip whales we've already evaluated (followed OR skipped)
            # This prevents re-evaluating the same whale on every WHALE_QUEUE_UPDATED event
            if whale_id in self._evaluated_whale_ids:
                continue

            self._whales_processed += 1
            new_whales_evaluated += 1

            # Emit processing start event for frontend animation
            await self._event_bus.emit_system_activity(
                activity_type="whale_processing",
                message=f"Processing whale {market_ticker}",
                metadata={"whale_id": whale_id, "status": "processing"}
            )

            # Check if too old - mark as evaluated even if skipped
            age_seconds = (now_ms - timestamp_ms) / 1000.0
            if age_seconds > WHALE_MAX_AGE_SECONDS:
                self._evaluated_whale_ids.add(whale_id)
                self._record_decision(
                    whale_id=whale_id,
                    action="skipped_age",
                    reason=f"Whale is {age_seconds:.1f}s old (max: {WHALE_MAX_AGE_SECONDS}s)",
                    whale_data=whale,
                )
                self._whales_skipped += 1
                await self._emit_processing_complete(whale_id, "skipped_age")
                await self._remove_from_queue(whale_id)
                continue

            # Check if we have position in this market - mark as evaluated
            if market_ticker in positions:
                self._evaluated_whale_ids.add(whale_id)
                self._record_decision(
                    whale_id=whale_id,
                    action="skipped_position",
                    reason=f"Already have position in {market_ticker}",
                    whale_data=whale,
                )
                self._whales_skipped += 1
                await self._emit_processing_complete(whale_id, "skipped_position")
                await self._remove_from_queue(whale_id)
                continue

            # Check if we have open orders in this market - mark as evaluated
            if market_ticker in markets_with_orders:
                self._evaluated_whale_ids.add(whale_id)
                self._record_decision(
                    whale_id=whale_id,
                    action="skipped_orders",
                    reason=f"Have open orders in {market_ticker}",
                    whale_data=whale,
                )
                self._whales_skipped += 1
                await self._emit_processing_complete(whale_id, "skipped_orders")
                await self._remove_from_queue(whale_id)
                continue

            # Check rate limit - do NOT mark as evaluated (we want to retry later)
            # Also do NOT remove from queue - we want to retry when tokens refill
            if not self._consume_token():
                self._record_decision(
                    whale_id=whale_id,
                    action="rate_limited",
                    reason=f"Rate limited ({self._tokens:.1f} tokens remaining)",
                    whale_data=whale,
                )
                self._rate_limited_count += 1
                self._whales_skipped += 1
                # Clear processing animation but don't remove from queue
                await self._emit_processing_complete(whale_id, "rate_limited")
                # Stop processing more whales until tokens refill
                # Don't add to evaluated AND don't remove - we want to retry when tokens refill
                break

            # Execute the whale follow - mark as evaluated regardless of outcome
            self._evaluated_whale_ids.add(whale_id)
            success, order_id = await self._execute_whale_follow(whale, whale_id)

            if success:
                self._record_decision(
                    whale_id=whale_id,
                    action="followed",
                    reason=f"Followed whale on {market_ticker} {whale.get('side', 'yes').upper()}",
                    order_id=order_id,
                    whale_data=whale,
                )
                self._whales_followed += 1
                self._last_execution_time = time.time()
                await self._emit_processing_complete(whale_id, "followed")
                await self._remove_from_queue(whale_id)
            else:
                self._record_decision(
                    whale_id=whale_id,
                    action="failed",
                    reason=f"Failed to execute order on {market_ticker}",
                    whale_data=whale,
                )
                self._whales_skipped += 1
                await self._emit_processing_complete(whale_id, "failed")
                await self._remove_from_queue(whale_id)

        # Log if we processed new whales
        if new_whales_evaluated > 0:
            logger.debug(
                f"Evaluated {new_whales_evaluated} new whales "
                f"(total tracked: {len(self._evaluated_whale_ids)})"
            )

    async def _execute_whale_follow(
        self,
        whale: Dict[str, Any],
        whale_id: str
    ) -> tuple[bool, Optional[str]]:
        """
        Execute a whale follow trade.

        Args:
            whale: Whale data dict
            whale_id: Unique whale identifier

        Returns:
            Tuple of (success, order_id)
        """
        from ..services.trading_decision_service import TradingDecision, TradingStrategy

        # Import constant from trading decision service
        WHALE_FOLLOW_CONTRACTS = 5

        try:
            # Build trading decision
            market_ticker = whale.get("market_ticker", "")
            side = whale.get("side", "yes")
            price_cents = whale.get("price_cents", 50)

            decision = TradingDecision(
                action="buy",
                market=market_ticker,
                side=side,
                quantity=WHALE_FOLLOW_CONTRACTS,
                price=price_cents,
                reason=f"whale:{whale_id}",
                confidence=0.7,
                strategy=TradingStrategy.WHALE_FOLLOWER,
            )

            logger.info(
                f"Executing whale follow: {market_ticker} {side.upper()} @ {price_cents}c "
                f"(whale size: ${whale.get('whale_size_dollars', 0):.2f})"
            )

            # Execute through trading service
            success = await self._trading_service.execute_decision(decision)

            # Get order ID from followed whales (trading service tracks it)
            order_id = None
            if success:
                followed = self._trading_service._followed_whales.get(whale_id)
                if followed:
                    order_id = followed.our_order_id

            return success, order_id

        except Exception as e:
            logger.error(f"Error executing whale follow: {e}")
            return False, None

    async def _emit_processing_complete(self, whale_id: str, action: str) -> None:
        """
        Emit processing complete event for frontend animation.

        Args:
            whale_id: Unique whale identifier
            action: Decision action (followed, skipped_*, rate_limited, failed)
        """
        await self._event_bus.emit_system_activity(
            activity_type="whale_processing",
            message=f"Processed whale: {action}",
            metadata={"whale_id": whale_id, "status": "complete", "action": action}
        )

    async def _remove_from_queue(self, whale_id: str) -> None:
        """
        Remove whale from tracker queue after terminal decision.

        Called after: followed, failed, skipped_age, skipped_position, skipped_orders.
        NOT called after: rate_limited (we want to retry when tokens refill).

        Args:
            whale_id: Unique whale identifier (market_ticker:timestamp_ms)
        """
        if self._whale_tracker:
            removed = await self._whale_tracker.remove_whale(whale_id)
            if removed:
                logger.debug(f"Removed whale from queue: {whale_id}")

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

    def _record_decision(
        self,
        whale_id: str,
        action: str,
        reason: str,
        order_id: Optional[str] = None,
        whale_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record a whale decision in history.

        Args:
            whale_id: Unique whale identifier
            action: Decision action (followed, skipped_*, rate_limited, failed)
            reason: Human-readable explanation
            order_id: Order ID if followed
            whale_data: Original whale data
        """
        decision = WhaleDecision(
            whale_id=whale_id,
            timestamp=time.time(),
            action=action,
            reason=reason,
            order_id=order_id,
            whale_data=whale_data,
        )
        self._decision_history.append(decision)

        # Log non-trivial decisions
        if action not in ("skipped_age",):  # Skip logging old whale skips
            logger.debug(f"Whale decision: {action} - {reason}")

    def get_decision_history(self) -> List[Dict[str, Any]]:
        """
        Get recent decision history.

        Returns:
            List of decision dicts, most recent first
        """
        return [d.to_dict() for d in reversed(self._decision_history)]

    def get_stats(self) -> Dict[str, Any]:
        """Get execution service statistics."""
        uptime = time.time() - self._started_at if self._started_at else 0

        # Calculate categorized skip counts from decision history
        skipped_age = 0
        skipped_position = 0
        skipped_orders = 0
        already_followed = 0
        rate_limited = 0
        followed = 0
        failed = 0

        for decision in self._decision_history:
            action = decision.action
            if action == "skipped_age":
                skipped_age += 1
            elif action == "skipped_position":
                skipped_position += 1
            elif action == "skipped_orders":
                skipped_orders += 1
            elif action == "already_followed":
                already_followed += 1
            elif action == "rate_limited":
                rate_limited += 1
            elif action == "followed":
                followed += 1
            elif action == "failed":
                failed += 1

        return {
            "running": self._running,
            "uptime_seconds": uptime,
            "whales_processed": self._whales_processed,
            "whales_followed": self._whales_followed,
            "whales_skipped": self._whales_skipped,
            "rate_limited_count": self._rate_limited_count,
            "tokens_remaining": round(self._tokens, 2),
            "max_tokens": self._max_tokens,
            "token_refill_seconds": self._token_refill_seconds,
            "last_execution_time": self._last_execution_time,
            "decision_history_size": len(self._decision_history),
            "unique_whales_evaluated": len(self._evaluated_whale_ids),
            # Categorized skip reasons for Decision Audit panel
            "skipped_age": skipped_age,
            "skipped_position": skipped_position,
            "skipped_orders": skipped_orders,
            "already_followed": already_followed,
            "rate_limited": rate_limited,
            "followed": followed,
            "failed": failed,
        }

    def get_health_details(self) -> Dict[str, Any]:
        """Get detailed health information."""
        stats = self.get_stats()
        return {
            "healthy": self._running,
            "running": self._running,
            "whales_followed": stats["whales_followed"],
            "whales_skipped": stats["whales_skipped"],
            "rate_limited_count": stats["rate_limited_count"],
            "tokens_remaining": stats["tokens_remaining"],
            "uptime_seconds": stats["uptime_seconds"],
            "unique_whales_evaluated": stats["unique_whales_evaluated"],
        }

    def is_healthy(self) -> bool:
        """Check if service is healthy."""
        return self._running
