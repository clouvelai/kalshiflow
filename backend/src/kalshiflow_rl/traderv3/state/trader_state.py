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
        # Extract positions by ticker
        positions_by_ticker = {}
        for pos in positions_data.get("market_positions", []):
            ticker = pos.get("ticker")
            if ticker:
                positions_by_ticker[ticker] = pos
        
        # Extract orders by order_id
        orders_by_id = {}
        for order in orders_data.get("orders", []):
            order_id = order.get("order_id")
            if order_id:
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