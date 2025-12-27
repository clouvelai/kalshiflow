"""
Kalshi-as-Truth State Manager for V3 Trader.

Lightweight state management that treats Kalshi API as the single source
of truth, with WebSocket events applied between syncs for responsiveness.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("kalshiflow_rl.traderv3.state.kalshi_state")


class StateSource(Enum):
    """Source of state update."""
    KALSHI_SYNC = "kalshi_sync"  # Direct API sync (truth)
    WEBSOCKET = "websocket"      # WebSocket event (interim update)
    LOCAL = "local"              # Local calculation/estimation


@dataclass
class Position:
    """Position in a market."""
    ticker: str
    side: str  # "yes" or "no"
    quantity: int
    avg_price: float
    realized_pnl: int  # In cents
    unrealized_pnl: int  # In cents
    last_update: float = field(default_factory=time.time)
    source: StateSource = StateSource.KALSHI_SYNC


@dataclass
class Order:
    """Open order."""
    order_id: str
    ticker: str
    side: str
    action: str  # "buy" or "sell"
    quantity: int
    price: int
    status: str
    created_at: float
    last_update: float = field(default_factory=time.time)
    source: StateSource = StateSource.KALSHI_SYNC


@dataclass
class KalshiState:
    """
    Complete state from Kalshi API.
    This is the authoritative state that we sync periodically.
    """
    # Account state
    balance: int  # In cents
    portfolio_value: int  # In cents
    
    # Positions
    positions: Dict[str, Position] = field(default_factory=dict)
    position_count: int = 0
    
    # Orders
    open_orders: List[Order] = field(default_factory=list)
    order_count: int = 0
    
    # Sync metadata
    last_sync: float = field(default_factory=time.time)
    sync_count: int = 0
    
    # WebSocket updates applied since last sync
    ws_updates_applied: int = 0
    dirty_markets: Set[str] = field(default_factory=set)  # Markets with WS updates
    
    def mark_dirty(self, market: str) -> None:
        """Mark a market as having WebSocket updates."""
        self.dirty_markets.add(market)
        self.ws_updates_applied += 1
    
    def clear_dirty(self) -> None:
        """Clear dirty flags after sync."""
        self.dirty_markets.clear()
        self.ws_updates_applied = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "balance": self.balance,
            "portfolio_value": self.portfolio_value,
            "position_count": self.position_count,
            "order_count": self.order_count,
            "last_sync": self.last_sync,
            "sync_count": self.sync_count,
            "ws_updates_applied": self.ws_updates_applied,
            "dirty_markets": list(self.dirty_markets),
            "positions": {
                ticker: {
                    "side": pos.side,
                    "quantity": pos.quantity,
                    "avg_price": pos.avg_price,
                    "realized_pnl": pos.realized_pnl,
                    "unrealized_pnl": pos.unrealized_pnl
                }
                for ticker, pos in self.positions.items()
            }
        }


class KalshiStateManager:
    """
    Manages state with Kalshi API as the source of truth.
    
    Design principles:
    1. Kalshi API is always the authoritative source
    2. WebSocket events update state between syncs
    3. Force sync before critical operations (trading)
    4. Track which data is from sync vs WebSocket
    5. Simple, clear, maintainable
    """
    
    def __init__(self):
        """Initialize the state manager."""
        self._state = KalshiState()
        self._lock = asyncio.Lock()
        
        # Track sync health
        self._last_successful_sync: Optional[float] = None
        self._consecutive_sync_failures = 0
        self._max_sync_failures = 3
        
        # WebSocket event buffer (for events received during sync)
        self._event_buffer: List[Dict[str, Any]] = []
        self._buffering = False
        
        logger.info("Kalshi State Manager initialized")
    
    async def sync_with_kalshi(
        self,
        balance: int,
        positions: List[Dict[str, Any]],
        orders: List[Dict[str, Any]]
    ) -> KalshiState:
        """
        Perform full state sync with Kalshi API.
        
        This is the authoritative update that overwrites any
        WebSocket-based interim state.
        
        Args:
            balance: Account balance in cents
            positions: List of position dicts from API
            orders: List of open order dicts from API
        
        Returns:
            Updated state after sync
        """
        async with self._lock:
            # Start buffering WebSocket events during sync
            self._buffering = True
            self._event_buffer.clear()
            
            try:
                # Create new state from API data
                new_state = KalshiState(
                    balance=balance,
                    portfolio_value=self._calculate_portfolio_value(balance, positions),
                    last_sync=time.time(),
                    sync_count=self._state.sync_count + 1
                )
                
                # Process positions
                for pos_data in positions:
                    ticker = pos_data.get("ticker", pos_data.get("market"))
                    if not ticker:
                        continue
                    
                    position = Position(
                        ticker=ticker,
                        side=pos_data.get("side", "yes"),
                        quantity=pos_data.get("quantity", 0),
                        avg_price=pos_data.get("avg_price", 0),
                        realized_pnl=pos_data.get("realized_pnl", 0),
                        unrealized_pnl=pos_data.get("unrealized_pnl", 0),
                        source=StateSource.KALSHI_SYNC
                    )
                    new_state.positions[ticker] = position
                
                new_state.position_count = len(new_state.positions)
                
                # Process orders
                for order_data in orders:
                    order = Order(
                        order_id=order_data.get("order_id", ""),
                        ticker=order_data.get("ticker", ""),
                        side=order_data.get("side", ""),
                        action=order_data.get("action", ""),
                        quantity=order_data.get("quantity", 0),
                        price=order_data.get("price", 0),
                        status=order_data.get("status", ""),
                        created_at=order_data.get("created_at", time.time()),
                        source=StateSource.KALSHI_SYNC
                    )
                    new_state.open_orders.append(order)
                
                new_state.order_count = len(new_state.open_orders)
                
                # Replace old state
                old_state = self._state
                self._state = new_state
                
                # Track successful sync
                self._last_successful_sync = time.time()
                self._consecutive_sync_failures = 0
                
                logger.info(
                    f"State synced: Balance ${balance/100:.2f}, "
                    f"{new_state.position_count} positions, "
                    f"{new_state.order_count} orders"
                )
                
                # Apply any buffered events that arrived during sync
                if self._event_buffer:
                    logger.debug(f"Applying {len(self._event_buffer)} buffered events")
                    for event in self._event_buffer:
                        await self._apply_websocket_event_internal(event)
                
                return self._state
                
            finally:
                # Stop buffering
                self._buffering = False
                self._event_buffer.clear()
    
    async def apply_websocket_event(self, event: Dict[str, Any]) -> None:
        """
        Apply a WebSocket event to update state between syncs.
        
        These updates are interim and will be overwritten by the
        next sync, but provide responsiveness.
        
        Args:
            event: WebSocket event data
        """
        async with self._lock:
            # If we're syncing, buffer the event
            if self._buffering:
                self._event_buffer.append(event)
                return
            
            await self._apply_websocket_event_internal(event)
    
    async def _apply_websocket_event_internal(self, event: Dict[str, Any]) -> None:
        """Internal method to apply WebSocket event."""
        event_type = event.get("type")
        
        if event_type == "fill":
            # Update position based on fill
            ticker = event.get("ticker")
            if ticker:
                self._state.mark_dirty(ticker)
                # Simple position update - will be corrected on next sync
                if ticker not in self._state.positions:
                    self._state.positions[ticker] = Position(
                        ticker=ticker,
                        side=event.get("side", "yes"),
                        quantity=event.get("quantity", 0),
                        avg_price=event.get("price", 0),
                        realized_pnl=0,
                        unrealized_pnl=0,
                        source=StateSource.WEBSOCKET
                    )
                else:
                    # Update existing position
                    pos = self._state.positions[ticker]
                    pos.quantity += event.get("quantity", 0)
                    pos.last_update = time.time()
                    pos.source = StateSource.WEBSOCKET
        
        elif event_type == "order_status":
            # Update order status
            order_id = event.get("order_id")
            if order_id:
                for order in self._state.open_orders:
                    if order.order_id == order_id:
                        order.status = event.get("status", order.status)
                        order.last_update = time.time()
                        order.source = StateSource.WEBSOCKET
                        
                        # Remove if filled or cancelled
                        if order.status in ["filled", "cancelled"]:
                            self._state.open_orders.remove(order)
                            self._state.order_count = len(self._state.open_orders)
                        break
    
    def _calculate_portfolio_value(
        self,
        balance: int,
        positions: List[Dict[str, Any]]
    ) -> int:
        """Calculate total portfolio value."""
        total = balance
        for pos in positions:
            # Add position value (simplified)
            quantity = pos.get("quantity", 0)
            price = pos.get("avg_price", 0)
            total += quantity * price
        return total
    
    async def get_state(self) -> KalshiState:
        """Get current state (mix of sync and WebSocket updates)."""
        async with self._lock:
            return self._state
    
    def get_sync_health(self) -> Dict[str, Any]:
        """Get sync health information."""
        return {
            "last_successful_sync": self._last_successful_sync,
            "time_since_sync": (
                time.time() - self._last_successful_sync 
                if self._last_successful_sync else None
            ),
            "consecutive_failures": self._consecutive_sync_failures,
            "sync_count": self._state.sync_count,
            "ws_updates_applied": self._state.ws_updates_applied,
            "dirty_markets": list(self._state.dirty_markets)
        }
    
    def needs_sync(self, max_age: float = 30.0) -> bool:
        """
        Check if state needs to be synced.
        
        Args:
            max_age: Maximum age in seconds before sync needed
        
        Returns:
            True if sync is needed
        """
        # Need sync if never synced
        if not self._last_successful_sync:
            return True
        
        # Need sync if too old
        age = time.time() - self._last_successful_sync
        if age > max_age:
            return True
        
        # Need sync if too many WS updates
        if self._state.ws_updates_applied > 10:
            return True
        
        # Need sync if critical markets are dirty
        # (In production, check if dirty markets have positions)
        if len(self._state.dirty_markets) > 5:
            return True
        
        return False
    
    def record_sync_failure(self) -> None:
        """Record a sync failure for health tracking."""
        self._consecutive_sync_failures += 1
        logger.warning(f"Sync failure #{self._consecutive_sync_failures}")
    
    def is_healthy(self) -> bool:
        """Check if state manager is healthy."""
        # Unhealthy if too many sync failures
        if self._consecutive_sync_failures >= self._max_sync_failures:
            return False
        
        # Unhealthy if sync is very stale (>5 minutes)
        if self._last_successful_sync:
            age = time.time() - self._last_successful_sync
            if age > 300:  # 5 minutes
                return False
        
        return True