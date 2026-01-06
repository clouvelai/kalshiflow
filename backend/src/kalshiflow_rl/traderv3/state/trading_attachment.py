"""
Trading Attachment - Orders, Positions, and P&L Tracking for Tracked Markets.

This module provides data structures for attaching trading state to lifecycle-discovered
tracked markets, enabling real-time visibility of orders, positions, and P&L per market.

Architecture:
    - TradingAttachment attaches to TrackedMarket (not embedded in it)
    - Populated from existing TradingStateSyncer sync cycle (no new API calls)
    - Real-time updates from FillListener events (between syncs)
    - Session recovery automatic via first sync after market discovery

Key Data Flow:
    1. TradingStateSyncer.sync_with_kalshi() -> StateContainer.update_trading_state()
    2. update_trading_state() -> _sync_trading_attachments() (hooks into existing sync)
    3. For each position/order in synced data, if market is tracked, update attachment
    4. FillListener.ORDER_FILL -> mark_order_filled_in_attachment() (real-time)
"""

import time
from datetime import datetime
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum

logger = logging.getLogger("kalshiflow_rl.traderv3.state.trading_attachment")


class TradingState(Enum):
    """Trading lifecycle state for a tracked market."""
    MONITORING = "monitoring"           # Just watching, no action taken
    SIGNAL_READY = "signal_ready"       # Signal detected, ready to trade
    ORDER_PENDING = "order_pending"     # Order sent, waiting for confirmation
    ORDER_RESTING = "order_resting"     # Order confirmed, waiting for fill
    POSITION_OPEN = "position_open"     # Have active position in market
    AWAITING_SETTLEMENT = "awaiting_settlement"  # Market determined, waiting payout
    SETTLED = "settled"                 # Position closed, P&L realized


@dataclass
class TrackedMarketOrder:
    """
    An order we've placed in a tracked market.

    Kept in attachment for audit trail even after fill/cancel.

    Attributes:
        order_id: Kalshi order UUID
        signal_id: Links to triggering signal (whale_id, rlm_signal, etc.)
        action: "buy" or "sell"
        side: "yes" or "no"
        count: Contracts ordered
        price: Price in cents
        status: "pending", "resting", "partial", "filled", "cancelled"
        placed_at: Timestamp when order was placed
        fill_count: Contracts filled so far
        fill_avg_price: Average fill price in cents
        filled_at: Timestamp when fully filled
        cancelled_at: Timestamp when cancelled (if cancelled)
        strategy_id: String identifier for the strategy that placed this order
                    (e.g., "rlm_no", "s013"). Used for per-strategy P&L tracking.
    """
    order_id: str
    signal_id: Optional[str]  # Links to triggering signal (whale_id, rlm_signal, etc.)
    action: str               # "buy" or "sell"
    side: str                 # "yes" or "no"
    count: int                # Contracts ordered
    price: int                # Price in cents
    status: str               # "pending", "resting", "partial", "filled", "cancelled"
    placed_at: float          # Timestamp when order was placed
    fill_count: int = 0       # Contracts filled so far
    fill_avg_price: int = 0   # Average fill price in cents
    filled_at: Optional[float] = None      # Timestamp when fully filled
    cancelled_at: Optional[float] = None   # Timestamp when cancelled (if cancelled)
    strategy_id: Optional[str] = None      # Strategy that placed this order

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for WebSocket broadcast."""
        return {
            "order_id": self.order_id,
            "signal_id": self.signal_id,
            "action": self.action,
            "side": self.side,
            "count": self.count,
            "price": self.price,
            "status": self.status,
            "placed_at": self.placed_at,
            "fill_count": self.fill_count,
            "fill_avg_price": self.fill_avg_price,
            "filled_at": self.filled_at,
            "cancelled_at": self.cancelled_at,
            "strategy_id": self.strategy_id,
        }

    @classmethod
    def from_kalshi_order(cls, order_data: Dict[str, Any], signal_id: Optional[str] = None) -> 'TrackedMarketOrder':
        """Create from Kalshi API order data (from sync)."""
        # Determine status from Kalshi order data
        status = order_data.get("status", "resting")
        if status == "active":
            status = "resting"

        # Get price from yes_price or no_price
        price = order_data.get("yes_price") or order_data.get("no_price") or 0

        # Parse created_time properly - Kalshi returns ISO string, we need Unix timestamp
        created_time = order_data.get("created_time")
        if isinstance(created_time, str):
            try:
                # Handle ISO format: "2025-12-27T10:30:00Z" or "2025-12-27T10:30:00+00:00"
                dt = datetime.fromisoformat(created_time.replace("Z", "+00:00"))
                placed_at = dt.timestamp()
            except (ValueError, TypeError):
                placed_at = time.time()
        elif isinstance(created_time, (int, float)):
            placed_at = float(created_time)
        else:
            placed_at = time.time()

        return cls(
            order_id=order_data.get("order_id", ""),
            signal_id=signal_id or f"recovered:{order_data.get('ticker', 'unknown')}",
            action=order_data.get("action", "buy"),
            side=order_data.get("side", "yes"),
            count=order_data.get("remaining_count", 0),
            price=price,
            status=status,
            placed_at=placed_at,
            fill_count=order_data.get("filled_count", 0),
        )


@dataclass
class TrackedMarketPosition:
    """
    Our position in a tracked market.

    Updated from periodic sync and real-time fill events.
    """
    side: str                 # "yes" or "no"
    count: int                # Number of contracts (always positive)
    avg_entry_price: int      # Weighted average entry price (cents)
    total_cost: int           # Total cents invested (cost basis)
    current_value: int        # Current market value (from position data)
    unrealized_pnl: int       # current_value - total_cost
    realized_pnl: int         # From partial closes
    fees_paid: int = 0        # Fees accumulated
    last_updated: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for WebSocket broadcast."""
        return {
            "side": self.side,
            "count": self.count,
            "avg_entry_price": self.avg_entry_price,
            "total_cost": self.total_cost,
            "current_value": self.current_value,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "fees_paid": self.fees_paid,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_kalshi_position(cls, position_data: Dict[str, Any]) -> 'TrackedMarketPosition':
        """Create from Kalshi API position data (from sync)."""
        position_count = position_data.get("position", 0)
        side = "yes" if position_count > 0 else "no"
        count = abs(position_count)

        # Correct field names from TraderState.positions
        total_cost = position_data.get("total_traded", 0)  # Entry cost (cost basis)
        current_value = position_data.get("market_exposure", 0)  # Current market value

        # Calculate avg entry price (no direct field available)
        avg_entry = round(total_cost / count) if count > 0 else 0

        return cls(
            side=side,
            count=count,
            avg_entry_price=avg_entry,
            total_cost=total_cost,
            current_value=current_value,
            unrealized_pnl=current_value - total_cost,
            realized_pnl=position_data.get("realized_pnl", 0),
            fees_paid=position_data.get("fees_paid", 0),
            last_updated=time.time(),
        )


@dataclass
class TrackedMarketSettlement:
    """
    Final outcome when a market settles.

    Captured when position becomes 0 and market is determined.

    Attributes:
        result: "yes", "no", or "void"
        determined_at: When market was determined (Unix timestamp)
        settled_at: When position was settled (payout received)
        final_position: Contracts we held at settlement
        final_pnl: Net P&L in cents (revenue - cost_basis - fees)
        revenue: Payout received in cents
        cost_basis: Original cost in cents
        fees: Fees paid in cents
        strategy_id: Strategy that opened this position. May be "mixed" if
                    multiple strategies contributed to the position.
        per_order_pnl: Per-order P&L breakdown for multi-order positions.
                      Maps order_id -> pnl_cents. Enables accurate per-strategy
                      P&L calculation when multiple orders contributed to position.
    """
    result: str               # "yes", "no", "void"
    determined_at: float      # When market was determined
    settled_at: Optional[float] = None  # When position was settled (payout received)
    final_position: int = 0   # Contracts we held at settlement
    final_pnl: int = 0        # Net P&L in cents (revenue - cost_basis - fees)
    revenue: int = 0          # Payout received in cents
    cost_basis: int = 0       # Original cost in cents
    fees: int = 0             # Fees paid in cents
    strategy_id: Optional[str] = None   # Strategy that opened this position
    per_order_pnl: Dict[str, int] = field(default_factory=dict)  # order_id -> pnl_cents

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for WebSocket broadcast."""
        return {
            "result": self.result,
            "determined_at": self.determined_at,
            "settled_at": self.settled_at,
            "final_position": self.final_position,
            "final_pnl": self.final_pnl,
            "revenue": self.revenue,
            "cost_basis": self.cost_basis,
            "fees": self.fees,
            "strategy_id": self.strategy_id,
            "per_order_pnl": self.per_order_pnl,
        }


@dataclass
class TradingAttachment:
    """
    Trading state attached to a TrackedMarket.

    Provides unified view of our trading activity in a specific market:
    - Open orders (pending, resting)
    - Historical orders (filled, cancelled) - kept for audit trail
    - Current position (if any)
    - Settlement data (when market determines)

    This is populated from the existing TradingStateSyncer sync cycle,
    not from a separate API call.
    """
    ticker: str
    trading_state: TradingState = TradingState.MONITORING
    orders: Dict[str, TrackedMarketOrder] = field(default_factory=dict)
    position: Optional[TrackedMarketPosition] = None
    settlement: Optional[TrackedMarketSettlement] = None
    is_position_maxed: bool = False  # True when position is at per-market max
    version: int = 0  # For change detection
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for WebSocket broadcast."""
        return {
            "ticker": self.ticker,
            "trading_state": self.trading_state.value,
            "orders": [order.to_dict() for order in self.orders.values()],
            "position": self.position.to_dict() if self.position else None,
            "settlement": self.settlement.to_dict() if self.settlement else None,
            "is_position_maxed": self.is_position_maxed,
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def bump_version(self) -> None:
        """Increment version and update timestamp."""
        self.version += 1
        self.updated_at = time.time()

    @property
    def has_exposure(self) -> bool:
        """Do we have orders or position in this market?"""
        has_active_orders = any(
            o.status in ("pending", "resting", "partial")
            for o in self.orders.values()
        )
        has_position = self.position is not None and self.position.count > 0
        return has_active_orders or has_position

    @property
    def total_pnl(self) -> int:
        """Total P&L (realized + unrealized) in cents."""
        if self.settlement:
            return self.settlement.final_pnl
        if self.position:
            return self.position.realized_pnl + self.position.unrealized_pnl
        return 0

    @property
    def active_orders(self) -> List[TrackedMarketOrder]:
        """Get orders that are still active (not filled or cancelled)."""
        return [
            o for o in self.orders.values()
            if o.status in ("pending", "resting", "partial")
        ]

    def update_trading_state(self) -> None:
        """Update trading_state based on current orders/position/settlement."""
        if self.settlement:
            self.trading_state = TradingState.SETTLED
        elif self.position and self.position.count > 0:
            # Check if market is determined (caller should set this)
            self.trading_state = TradingState.POSITION_OPEN
        elif self.active_orders:
            # Check order statuses
            statuses = [o.status for o in self.active_orders]
            if "pending" in statuses:
                self.trading_state = TradingState.ORDER_PENDING
            else:
                self.trading_state = TradingState.ORDER_RESTING
        else:
            self.trading_state = TradingState.MONITORING

    def mark_signal_ready(self, signal_id: str) -> None:
        """
        Mark that a signal is ready to act on for this market.

        Called when the RLM service generates a signal for a tracked market.
        Only transitions from MONITORING state to avoid disrupting active orders/positions.

        Args:
            signal_id: Identifier of the signal (e.g., whale_id, rlm_signal_id)
        """
        if self.trading_state == TradingState.MONITORING:
            self.trading_state = TradingState.SIGNAL_READY
            self.bump_version()

    def mark_position_maxed(self) -> None:
        """
        Mark that this market's position is at the per-market maximum.

        Called when RLM detects a signal but position is already at max.
        Prevents signal spam and provides UI visibility.
        """
        if not self.is_position_maxed:
            self.is_position_maxed = True
            self.bump_version()

    def clear_position_maxed(self) -> None:
        """
        Clear the position maxed flag.

        Called when position reduces below max (e.g., after partial close or settlement).
        Allows signals to fire again for this market.
        """
        if self.is_position_maxed:
            self.is_position_maxed = False
            self.bump_version()


# Type alias for container
TradingAttachments = Dict[str, TradingAttachment]
