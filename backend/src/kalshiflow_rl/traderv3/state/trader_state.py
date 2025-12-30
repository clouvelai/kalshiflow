"""
Trader state representation for V3.

ALL VALUES IN CENTS - Kalshi's native unit.
No dollar conversions, no derived calculations.
Just raw data from Kalshi API.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional


@dataclass
class OrderGroupState:
    """
    Order group state representation.
    Tracks an order group session for portfolio limits.

    Note: The Kalshi API only returns {is_auto_cancel_enabled, orders[]} for order groups.
    Other fields are tracked locally or derived.
    """
    order_group_id: str  # UUID of the order group
    created_at: float  # Timestamp when created (local tracking)
    status: str = "active"  # active, closed, failed (local tracking)
    order_ids: List[str] = field(default_factory=list)  # Order IDs from API
    is_auto_cancel_enabled: bool = False  # From API

    def is_active(self) -> bool:
        """Check if order group is active and usable."""
        return self.status == "active" and self.order_group_id

    def to_metadata(self) -> dict:
        """Format for state machine metadata."""
        return {
            "order_group_id": self.order_group_id[:8] if self.order_group_id else "none",
            "status": self.status,
            "order_count": len(self.order_ids),
            "is_auto_cancel_enabled": self.is_auto_cancel_enabled
        }


@dataclass
class TraderState:
    """
    Complete trader state representation.
    ALL VALUES IN CENTS - no dollar conversions.
    """
    # Account state (from /portfolio/balance)
    balance: int  # Available cash in cents
    portfolio_value: int  # Value of all positions in cents
    
    # Positions (from /portfolio/positions)
    positions: Dict[str, Any] = field(default_factory=dict)  # Raw position data by ticker
    position_count: int = 0  # Number of active positions
    
    # Orders (from /orders)
    orders: Dict[str, Any] = field(default_factory=dict)  # Raw order data by order_id
    order_count: int = 0  # Number of open orders
    
    # Settlements (from /portfolio/settlements)
    settlements: List[Dict[str, Any]] = field(default_factory=list)  # Recent settlements
    
    # Order group state (optional)
    order_group: Optional[OrderGroupState] = None  # Order group session if active
    
    # Metadata
    sync_timestamp: float = 0.0
    
    @classmethod
    def from_kalshi_data(
        cls, 
        balance_data: dict, 
        positions_data: dict, 
        orders_data: dict, 
        settlements_data: list
    ) -> 'TraderState':
        """
        Create TraderState from raw Kalshi API responses.
        
        Args:
            balance_data: Response from get_account_info()
            positions_data: Response from get_positions()
            orders_data: Response from get_orders()
            settlements_data: List of settlements
            
        Returns:
            TraderState instance with raw Kalshi data
        """
        # Extract positions by ticker - only include ACTIVE positions (position != 0)
        # Kalshi API returns all markets ever traded, including closed positions (position=0)
        positions_by_ticker = {}
        for pos in positions_data.get("market_positions", []):
            ticker = pos.get("ticker")
            # Only include positions with actual holdings (non-zero position count)
            if ticker and pos.get("position", 0) != 0:
                positions_by_ticker[ticker] = pos
        
        # Extract orders by order_id - only include RESTING (open) orders
        # Kalshi API returns orders with all statuses: resting, canceled, executed
        # We only want active orders that can still be filled or canceled
        orders_by_id = {}
        for order in orders_data.get("orders", []):
            order_id = order.get("order_id")
            status = order.get("status", "")
            # Only include resting (open) orders, not executed or canceled
            if order_id and status == "resting":
                orders_by_id[order_id] = order
        
        return cls(
            balance=balance_data.get("balance", 0),  # Already in cents
            portfolio_value=balance_data.get("portfolio_value", 0),  # Already in cents
            positions=positions_by_ticker,
            position_count=len(positions_by_ticker),
            orders=orders_by_id,
            order_count=len(orders_by_id),
            settlements=settlements_data,
            sync_timestamp=time.time()
        )


@dataclass
class StateChange:
    """
    Tracks simple changes between syncs.
    All values are deltas (can be positive or negative).
    """
    balance_change: int = 0  # Change in cents
    portfolio_value_change: int = 0  # Change in cents
    position_count_change: int = 0  # Change in number of positions
    order_count_change: int = 0  # Change in number of orders
    
    def format_metadata(self) -> dict:
        """
        Format changes for state machine metadata.

        Returns:
            Dict with formatted change strings
        """
        return {
            "balance_change": f"{self.balance_change:+d}" if self.balance_change != 0 else "0",
            "portfolio_change": f"{self.portfolio_value_change:+d}" if self.portfolio_value_change != 0 else "0",
            "positions_change": f"{self.position_count_change:+d}" if self.position_count_change != 0 else "0",
            "orders_change": f"{self.order_count_change:+d}" if self.order_count_change != 0 else "0"
        }


@dataclass
class SessionPnLState:
    """
    Session-level P&L tracking.

    Captures starting balance and portfolio value at session start to compute
    session P&L. Resets when the trader restarts.

    ALL VALUES IN CENTS - consistent with TraderState.
    """
    session_start_time: float  # Timestamp when session began
    starting_balance: int  # Balance at session start (cents)
    starting_portfolio_value: int  # Portfolio value at session start (cents)

    @property
    def starting_equity(self) -> int:
        """Total equity at session start (balance + portfolio_value)."""
        return self.starting_balance + self.starting_portfolio_value

    def compute_pnl(
        self,
        current_balance: int,
        current_portfolio_value: int,
        positions_details: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Compute session P&L from current state.

        Args:
            current_balance: Current balance in cents
            current_portfolio_value: Current portfolio value in cents
            positions_details: List of position dicts with realized_pnl and unrealized_pnl

        Returns:
            Dict with session P&L metrics (all values in cents except percent)
        """
        current_equity = current_balance + current_portfolio_value
        session_pnl = current_equity - self.starting_equity
        pnl_percent = (session_pnl / self.starting_equity * 100) if self.starting_equity > 0 else 0.0

        # Aggregate realized and unrealized P&L from positions
        total_realized = sum(p.get("realized_pnl", 0) for p in (positions_details or []))
        total_unrealized = sum(p.get("unrealized_pnl", 0) for p in (positions_details or []))

        return {
            "session_start_time": self.session_start_time,
            "starting_equity": self.starting_equity,
            "current_equity": current_equity,
            "session_pnl": session_pnl,
            "session_pnl_percent": round(pnl_percent, 2),
            "invested_amount": current_portfolio_value,  # Amount currently in positions
            "realized_pnl": total_realized,
            "unrealized_pnl": total_unrealized,
        }