"""
Session P&L Tracker for V3 Trader.

This module provides session-level P&L tracking, capturing starting state
at session start and computing P&L metrics throughout the session. It tracks
both position P&L and session cash flow (orders placed, settlements received).

Purpose:
    The SessionPnLTracker consolidates all session P&L logic that was
    previously spread across the V3StateContainer, making it easier to
    understand and test session-level financial metrics.

Key Responsibilities:
    1. **Session Initialization** - Capture starting balance/portfolio at session start
    2. **Order Fill Recording** - Track cash spent on orders during session
    3. **Settlement Recording** - Track cash received from settlements during session
    4. **TTL Cancellation Tracking** - Count orders cancelled due to TTL expiry
    5. **P&L Computation** - Calculate session P&L with realized/unrealized breakdown

ALL VALUES IN CENTS - consistent with TraderState.
"""

import time
import logging
from typing import Dict, Any, List, Optional

from .trader_state import SessionPnLState

logger = logging.getLogger("kalshiflow_rl.traderv3.state.session_pnl_tracker")


class SessionPnLTracker:
    """
    Tracks session-level P&L and cash flow metrics.

    This tracker captures the starting state when initialized and tracks
    all financial activity during the session. It provides a clean API
    for recording fills, settlements, and computing P&L.

    Attributes:
        _session_pnl_state: SessionPnLState with starting equity
        _session_cash_invested: Cents spent on orders this session
        _session_cash_received: Cents received from settlements this session
        _session_orders_count: Orders placed this session
        _session_settlements_count: Positions settled this session
        _session_total_fees_paid: Total fees in cents this session
        _session_orders_cancelled_ttl: Orders cancelled due to TTL expiry

    Usage:
        ```python
        tracker = SessionPnLTracker()

        # Initialize on first sync
        tracker.initialize(balance=50000, portfolio_value=25000)

        # Record activity
        tracker.record_order_fill(cost_cents=500, contracts=5)
        tracker.record_settlement(cash_received=600, fees=5)
        tracker.record_ttl_cancellation(count=2)

        # Get P&L summary
        pnl = tracker.get_pnl_summary(
            current_balance=49500,
            current_portfolio_value=25500,
            positions_details=[...]
        )
        ```
    """

    def __init__(self):
        """Initialize session P&L tracker with zero state."""
        # Session P&L state - captures starting equity
        self._session_pnl_state: Optional[SessionPnLState] = None

        # Session cash flow tracking (resets each trader startup)
        # Tracks only activity during THIS session, not existing positions
        self._session_cash_invested: int = 0      # Cents spent on orders this session
        self._session_cash_received: int = 0      # Cents received from settlements this session
        self._session_orders_count: int = 0       # Orders placed this session
        self._session_settlements_count: int = 0  # Positions settled this session
        self._session_total_fees_paid: int = 0    # Total fees in cents this session
        self._session_orders_cancelled_ttl: int = 0  # Orders cancelled due to TTL expiry

        logger.debug("SessionPnLTracker initialized")

    @property
    def is_initialized(self) -> bool:
        """Check if session P&L tracking has been initialized."""
        return self._session_pnl_state is not None

    @property
    def session_pnl_state(self) -> Optional[SessionPnLState]:
        """Get the underlying SessionPnLState (for advanced access)."""
        return self._session_pnl_state

    @property
    def session_cash_invested(self) -> int:
        """Total cents spent on orders this session."""
        return self._session_cash_invested

    @property
    def session_cash_received(self) -> int:
        """Total cents received from settlements this session."""
        return self._session_cash_received

    @property
    def session_orders_count(self) -> int:
        """Number of orders placed this session."""
        return self._session_orders_count

    @property
    def session_settlements_count(self) -> int:
        """Number of positions settled this session."""
        return self._session_settlements_count

    @property
    def session_total_fees_paid(self) -> int:
        """Total fees paid in cents this session."""
        return self._session_total_fees_paid

    @property
    def session_orders_cancelled_ttl(self) -> int:
        """Number of orders cancelled due to TTL expiry this session."""
        return self._session_orders_cancelled_ttl

    def initialize(self, balance: int, portfolio_value: int) -> bool:
        """
        Initialize session P&L tracking on first sync.

        Called by the coordinator after the first successful Kalshi sync
        to capture the starting state for session P&L calculation.

        Args:
            balance: Starting balance in cents
            portfolio_value: Starting portfolio value in cents

        Returns:
            True if initialized, False if already initialized (no-op)

        Note:
            This is a one-time initialization per session. Subsequent calls
            are ignored to preserve the original session start state.
        """
        if self._session_pnl_state is not None:
            logger.debug("Session P&L already initialized, ignoring")
            return False

        self._session_pnl_state = SessionPnLState(
            session_start_time=time.time(),
            starting_balance=balance,
            starting_portfolio_value=portfolio_value
        )
        logger.info(
            f"Session P&L initialized: starting equity "
            f"{balance + portfolio_value} cents "
            f"(balance={balance}, portfolio={portfolio_value})"
        )
        return True

    def record_order_fill(self, cost_cents: int, contracts: int = 1) -> None:
        """
        Record cash spent on an order fill during this session.

        Called by TradingDecisionService after a successful order execution
        to track session cash flow metrics.

        Args:
            cost_cents: Total cost of the order in cents (price x contracts)
            contracts: Number of contracts filled (for logging)

        Side Effects:
            - Increments _session_cash_invested by cost_cents
            - Increments _session_orders_count by 1
        """
        self._session_cash_invested += cost_cents
        self._session_orders_count += 1

        logger.info(
            f"Order fill recorded: {contracts} contracts @ {cost_cents}c | "
            f"Session totals: invested={self._session_cash_invested}c, orders={self._session_orders_count}"
        )

    def record_settlement(self, cash_received: int, fees: int) -> None:
        """
        Record a position settlement during this session.

        Called when a position closes (via WebSocket or sync) to track
        cash received and fees paid.

        Args:
            cash_received: Cash received from settlement in cents (cost basis + P&L)
            fees: Fees paid in cents

        Side Effects:
            - Increments _session_cash_received by cash_received
            - Increments _session_settlements_count by 1
            - Increments _session_total_fees_paid by fees
        """
        self._session_cash_received += max(0, cash_received)  # Can't receive negative cash
        self._session_settlements_count += 1
        self._session_total_fees_paid += fees

        logger.debug(
            f"Settlement recorded: received={cash_received}c, fees={fees}c | "
            f"Session totals: received={self._session_cash_received}c, "
            f"fees={self._session_total_fees_paid}c, settlements={self._session_settlements_count}"
        )

    def record_ttl_cancellation(self, count: int) -> None:
        """
        Record orders cancelled due to TTL expiry during this session.

        Called by TradingFlowOrchestrator after successfully cancelling
        expired resting orders.

        Args:
            count: Number of orders cancelled

        Side Effects:
            - Increments _session_orders_cancelled_ttl by count
        """
        self._session_orders_cancelled_ttl += count

        logger.info(
            f"TTL cancellation recorded: {count} orders | "
            f"Session total: {self._session_orders_cancelled_ttl}"
        )

    def compute_pnl(
        self,
        current_balance: int,
        current_portfolio_value: int,
        positions_details: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Compute session P&L from current state.

        Delegates to SessionPnLState.compute_pnl() for the core calculation,
        then adds session cash flow metrics.

        Args:
            current_balance: Current balance in cents
            current_portfolio_value: Current portfolio value in cents
            positions_details: List of position dicts with realized_pnl and unrealized_pnl

        Returns:
            Dict with session P&L metrics (all values in cents except percent):
                - session_start_time: Timestamp when session began
                - starting_equity: Starting balance + portfolio_value
                - current_equity: Current balance + portfolio_value
                - session_pnl: Change in equity (current - starting)
                - session_pnl_percent: P&L as percentage
                - invested_amount: Current portfolio value
                - realized_pnl: Sum of realized P&L from positions
                - unrealized_pnl: Sum of unrealized P&L from positions

        Note:
            Session cash flow metrics (invested, received, counts) are NOT
            included here - they are added by the caller (get_pnl_summary).
        """
        if not self._session_pnl_state:
            return {
                "session_start_time": 0,
                "starting_equity": 0,
                "current_equity": current_balance + current_portfolio_value,
                "session_pnl": 0,
                "session_pnl_percent": 0.0,
                "invested_amount": current_portfolio_value,
                "realized_pnl": 0,
                "unrealized_pnl": 0,
            }

        return self._session_pnl_state.compute_pnl(
            current_balance,
            current_portfolio_value,
            positions_details
        )

    def get_pnl_summary(
        self,
        current_balance: int,
        current_portfolio_value: int,
        positions_details: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Get complete P&L summary including session cash flow metrics.

        This is the primary method for getting P&L data for broadcasting.
        It combines the core P&L calculation with session cash flow tracking.

        Args:
            current_balance: Current balance in cents
            current_portfolio_value: Current portfolio value in cents
            positions_details: List of position dicts with P&L data

        Returns:
            Dict with all P&L and cash flow metrics for the session
        """
        pnl = self.compute_pnl(current_balance, current_portfolio_value, positions_details)

        # Add session cash flow metrics
        pnl["session_cash_invested"] = self._session_cash_invested
        pnl["session_cash_received"] = self._session_cash_received
        pnl["session_orders_count"] = self._session_orders_count
        pnl["session_settlements_count"] = self._session_settlements_count
        pnl["session_total_fees_paid"] = self._session_total_fees_paid
        pnl["session_orders_cancelled_ttl"] = self._session_orders_cancelled_ttl

        return pnl

    def reset(self) -> None:
        """
        Reset all session P&L state.

        Called when the state container is reset (for testing or recovery).
        """
        self._session_pnl_state = None
        self._session_cash_invested = 0
        self._session_cash_received = 0
        self._session_orders_count = 0
        self._session_settlements_count = 0
        self._session_total_fees_paid = 0
        self._session_orders_cancelled_ttl = 0

        logger.info("SessionPnLTracker reset complete")
