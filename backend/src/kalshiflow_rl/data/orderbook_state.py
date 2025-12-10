"""
In-memory orderbook state management for RL Trading Subsystem.

Provides OrderbookState for efficient price level operations using SortedDict,
and SharedOrderbookState with thread-safe access and subscriber notifications.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Callable, Tuple
from decimal import Decimal
from copy import deepcopy
from sortedcontainers import SortedDict

from ..config import config

logger = logging.getLogger("kalshiflow_rl.orderbook_state")


class OrderbookState:
    """
    Efficient orderbook state representation using SortedDict.
    
    Maintains separate yes/no order books with bid/ask sides.
    Provides fast price level operations and spread calculations.
    """
    
    def __init__(self, market_ticker: str):
        """
        Initialize orderbook state for a market.
        
        Args:
            market_ticker: Market ticker (e.g., "INXD-25JAN03")
        """
        self.market_ticker = market_ticker
        self.last_sequence = 0
        self.last_update_time = 0
        
        # Use SortedDict for efficient price level operations
        # Key: price (int, cents), Value: size (int)
        self.yes_bids = SortedDict()  # Highest price first (reverse=True)
        self.yes_asks = SortedDict()  # Lowest price first (reverse=False)
        self.no_bids = SortedDict()   # Highest price first (reverse=True) 
        self.no_asks = SortedDict()   # Lowest price first (reverse=False)
        
        # Pre-reverse the bid dictionaries for proper ordering
        self.yes_bids = SortedDict(lambda x: -x)  # Descending order
        self.no_bids = SortedDict(lambda x: -x)   # Descending order
        
        # Cache frequently computed values
        self._yes_spread_cache: Optional[int] = None
        self._no_spread_cache: Optional[int] = None
        self._yes_mid_cache: Optional[Decimal] = None
        self._no_mid_cache: Optional[Decimal] = None
        self._cache_sequence = 0
    
    def apply_snapshot(self, snapshot_data: Dict[str, Any]) -> None:
        """
        Apply a full orderbook snapshot.
        
        Args:
            snapshot_data: Dict with yes_bids, yes_asks, no_bids, no_asks, sequence
        """
        self.last_sequence = snapshot_data.get('sequence_number', 0)
        self.last_update_time = snapshot_data.get('timestamp_ms', int(time.time() * 1000))
        
        # Clear existing state
        self.yes_bids.clear()
        self.yes_asks.clear()
        self.no_bids.clear()
        self.no_asks.clear()
        
        # Apply new snapshot
        self._apply_price_levels(self.yes_bids, snapshot_data.get('yes_bids', {}))
        self._apply_price_levels(self.yes_asks, snapshot_data.get('yes_asks', {}))
        self._apply_price_levels(self.no_bids, snapshot_data.get('no_bids', {}))
        self._apply_price_levels(self.no_asks, snapshot_data.get('no_asks', {}))
        
        # Invalidate cache
        self._invalidate_cache()
        
        logger.debug(
            f"Applied snapshot for {self.market_ticker}: seq={self.last_sequence}, "
            f"yes_levels={len(self.yes_bids) + len(self.yes_asks)}, "
            f"no_levels={len(self.no_bids) + len(self.no_asks)}"
        )
    
    def apply_delta(self, delta_data: Dict[str, Any]) -> bool:
        """
        Apply an orderbook delta update.
        
        Args:
            delta_data: Dict with side, action, price, old_size, new_size, sequence
            
        Returns:
            bool: True if delta was applied successfully
        """
        sequence_number = delta_data.get('sequence_number', 0)
        
        # Sequence validation (skip if sequence_number is 0 - temporary fix)
        if sequence_number != 0 and sequence_number <= self.last_sequence:
            logger.warning(
                f"Out-of-order delta for {self.market_ticker}: "
                f"seq={sequence_number}, last_seq={self.last_sequence}"
            )
            return False
        
        side = delta_data.get('side')
        action = delta_data.get('action')
        price = delta_data.get('price')
        new_size = delta_data.get('new_size', 0)
        
        if not side or not action or price is None:
            logger.error(f"Invalid delta data: {delta_data}")
            return False
        
        # Select the appropriate order book based on side and price
        # For remove actions, we need to determine which book by checking existing price levels
        if side == 'yes':
            if action == 'remove':
                # For remove, check which book has this price level
                if price in self.yes_bids:
                    book = self.yes_bids
                elif price in self.yes_asks:
                    book = self.yes_asks
                else:
                    # Price not found in either book, that's OK for remove
                    return True
            elif price <= 50:  # Yes bids
                book = self.yes_bids
            elif price > 50:  # Yes asks
                book = self.yes_asks
            else:
                logger.error(f"Invalid yes side price/action: {price}/{action}")
                return False
        elif side == 'no':
            if action == 'remove':
                # For remove, check which book has this price level
                if price in self.no_bids:
                    book = self.no_bids
                elif price in self.no_asks:
                    book = self.no_asks
                else:
                    # Price not found in either book, that's OK for remove
                    return True
            elif price <= 50:  # No bids
                book = self.no_bids
            elif price > 50:  # No asks
                book = self.no_asks
            else:
                logger.error(f"Invalid no side price/action: {price}/{action}")
                return False
        else:
            logger.error(f"Invalid side: {side}")
            return False
        
        # Apply the delta
        if action in ['add', 'update']:
            if new_size > 0:
                book[price] = new_size
            else:
                book.pop(price, None)  # Remove if size is 0
        elif action == 'remove':
            book.pop(price, None)
        else:
            logger.error(f"Unknown action: {action}")
            return False
        
        self.last_sequence = sequence_number
        self.last_update_time = delta_data.get('timestamp_ms', int(time.time() * 1000))
        self._invalidate_cache()
        
        return True
    
    def _apply_price_levels(self, book: SortedDict, price_levels: Dict) -> None:
        """Apply price levels from snapshot data."""
        for price_key, size in price_levels.items():
            try:
                # Handle both int and string keys
                if isinstance(price_key, int):
                    price = price_key
                else:
                    price = int(float(price_key))  # Convert string to integer cents
                    
                if size > 0:
                    book[price] = int(size)
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid price level: {price_key}={size}, error: {e}")
    
    def get_yes_spread(self) -> Optional[int]:
        """Get yes side spread in cents."""
        if self._yes_spread_cache is not None and self._cache_sequence == self.last_sequence:
            return self._yes_spread_cache
        
        best_bid = self._get_best_price(self.yes_bids, True)
        best_ask = self._get_best_price(self.yes_asks, False)
        
        if best_bid is not None and best_ask is not None:
            self._yes_spread_cache = best_ask - best_bid
        else:
            self._yes_spread_cache = None
        
        self._cache_sequence = self.last_sequence
        return self._yes_spread_cache
    
    def get_no_spread(self) -> Optional[int]:
        """Get no side spread in cents."""
        if self._no_spread_cache is not None and self._cache_sequence == self.last_sequence:
            return self._no_spread_cache
        
        best_bid = self._get_best_price(self.no_bids, True)
        best_ask = self._get_best_price(self.no_asks, False)
        
        if best_bid is not None and best_ask is not None:
            self._no_spread_cache = best_ask - best_bid
        else:
            self._no_spread_cache = None
        
        self._cache_sequence = self.last_sequence
        return self._no_spread_cache
    
    def get_yes_mid_price(self) -> Optional[Decimal]:
        """Get yes side mid price."""
        if self._yes_mid_cache is not None and self._cache_sequence == self.last_sequence:
            return self._yes_mid_cache
        
        best_bid = self._get_best_price(self.yes_bids, True)
        best_ask = self._get_best_price(self.yes_asks, False)
        
        if best_bid is not None and best_ask is not None:
            self._yes_mid_cache = Decimal(best_bid + best_ask) / 2
        else:
            self._yes_mid_cache = None
        
        self._cache_sequence = self.last_sequence
        return self._yes_mid_cache
    
    def get_no_mid_price(self) -> Optional[Decimal]:
        """Get no side mid price."""
        if self._no_mid_cache is not None and self._cache_sequence == self.last_sequence:
            return self._no_mid_cache
        
        best_bid = self._get_best_price(self.no_bids, True)
        best_ask = self._get_best_price(self.no_asks, False)
        
        if best_bid is not None and best_ask is not None:
            self._no_mid_cache = Decimal(best_bid + best_ask) / 2
        else:
            self._no_mid_cache = None
        
        self._cache_sequence = self.last_sequence
        return self._no_mid_cache
    
    def _get_best_price(self, book: SortedDict, is_bid: bool) -> Optional[int]:
        """Get best price from order book."""
        if not book:
            return None
        
        if is_bid:
            # For bids, we want the highest price (first in descending order)
            return next(iter(book))
        else:
            # For asks, we want the lowest price (first in ascending order)
            return next(iter(book))
    
    def get_top_levels(self, num_levels: int = 5) -> Dict[str, Any]:
        """
        Get top N price levels for each side.
        
        Args:
            num_levels: Number of levels to return per side
            
        Returns:
            Dict with top levels for each side
        """
        return {
            'yes_bids': dict(list(self.yes_bids.items())[:num_levels]),
            'yes_asks': dict(list(self.yes_asks.items())[:num_levels]),
            'no_bids': dict(list(self.no_bids.items())[:num_levels]),
            'no_asks': dict(list(self.no_asks.items())[:num_levels])
        }
    
    def get_total_volume(self) -> int:
        """Get total volume across all price levels."""
        return (
            sum(self.yes_bids.values()) + sum(self.yes_asks.values()) +
            sum(self.no_bids.values()) + sum(self.no_asks.values())
        )
    
    def _invalidate_cache(self) -> None:
        """Invalidate computed value caches."""
        self._yes_spread_cache = None
        self._no_spread_cache = None
        self._yes_mid_cache = None
        self._no_mid_cache = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert orderbook state to dictionary representation."""
        return {
            'market_ticker': self.market_ticker,
            'last_sequence': self.last_sequence,
            'last_update_time': self.last_update_time,
            'yes_bids': dict(self.yes_bids),
            'yes_asks': dict(self.yes_asks),
            'no_bids': dict(self.no_bids),
            'no_asks': dict(self.no_asks),
            'yes_spread': self.get_yes_spread(),
            'no_spread': self.get_no_spread(),
            'yes_mid_price': float(self.get_yes_mid_price()) if self.get_yes_mid_price() else None,
            'no_mid_price': float(self.get_no_mid_price()) if self.get_no_mid_price() else None,
            'total_volume': self.get_total_volume()
        }


class SharedOrderbookState:
    """
    Thread-safe wrapper for OrderbookState with subscriber notifications.
    
    Provides atomic access to orderbook state and broadcasts updates
    to multiple subscribers without blocking the main update process.
    """
    
    def __init__(self, market_ticker: str):
        """
        Initialize shared orderbook state.
        
        Args:
            market_ticker: Market ticker for this orderbook
        """
        self.market_ticker = market_ticker
        self._state = OrderbookState(market_ticker)
        self._lock = asyncio.Lock()
        self._subscribers: List[Callable[[Dict[str, Any]], None]] = []
        self._last_notification_time = 0
        self._notification_throttle = 0.1  # Minimum 100ms between notifications
        
        logger.info(f"Initialized shared orderbook state for {market_ticker}")
    
    async def apply_snapshot(self, snapshot_data: Dict[str, Any]) -> None:
        """Thread-safe snapshot application with subscriber notification."""
        async with self._lock:
            self._state.apply_snapshot(snapshot_data)
            await self._notify_subscribers('snapshot')
    
    async def apply_delta(self, delta_data: Dict[str, Any]) -> bool:
        """Thread-safe delta application with subscriber notification."""
        async with self._lock:
            success = self._state.apply_delta(delta_data)
            if success:
                await self._notify_subscribers('delta')
            return success
    
    async def get_snapshot(self) -> Dict[str, Any]:
        """Get atomic snapshot of current orderbook state."""
        async with self._lock:
            return deepcopy(self._state.to_dict())
    
    async def get_top_levels(self, num_levels: int = 5) -> Dict[str, Any]:
        """Get top price levels atomically."""
        async with self._lock:
            return self._state.get_top_levels(num_levels)
    
    async def get_spreads_and_mids(self) -> Dict[str, Any]:
        """Get spreads and mid prices atomically."""
        async with self._lock:
            return {
                'yes_spread': self._state.get_yes_spread(),
                'no_spread': self._state.get_no_spread(),
                'yes_mid_price': float(self._state.get_yes_mid_price()) if self._state.get_yes_mid_price() else None,
                'no_mid_price': float(self._state.get_no_mid_price()) if self._state.get_no_mid_price() else None
            }
    
    def add_subscriber(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Add a subscriber for orderbook updates.
        
        Args:
            callback: Function to call on updates, receives update_type and state
        """
        self._subscribers.append(callback)
        logger.debug(f"Added subscriber for {self.market_ticker}, total: {len(self._subscribers)}")
    
    def remove_subscriber(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Remove a subscriber."""
        if callback in self._subscribers:
            self._subscribers.remove(callback)
            logger.debug(f"Removed subscriber for {self.market_ticker}, total: {len(self._subscribers)}")
    
    async def _notify_subscribers(self, update_type: str) -> None:
        """Notify all subscribers of an update (with throttling)."""
        current_time = time.time()
        
        # Throttle notifications to prevent spam
        if current_time - self._last_notification_time < self._notification_throttle:
            return
        
        if not self._subscribers:
            return
        
        # Create notification data
        notification_data = {
            'market_ticker': self.market_ticker,
            'update_type': update_type,
            'timestamp': current_time,
            'state_summary': {
                'last_sequence': self._state.last_sequence,
                'last_update_time': self._state.last_update_time,
                'yes_spread': self._state.get_yes_spread(),
                'no_spread': self._state.get_no_spread(),
                'total_volume': self._state.get_total_volume()
            }
        }
        
        # Notify subscribers (don't block on failures)
        for callback in self._subscribers[:]:  # Copy list to avoid modification during iteration
            try:
                callback(notification_data)
            except Exception as e:
                logger.error(f"Subscriber notification failed for {self.market_ticker}: {e}")
        
        self._last_notification_time = current_time
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about this shared state."""
        return {
            'market_ticker': self.market_ticker,
            'last_sequence': self._state.last_sequence,
            'last_update_time': self._state.last_update_time,
            'subscribers_count': len(self._subscribers),
            'total_volume': self._state.get_total_volume(),
            'price_level_count': (
                len(self._state.yes_bids) + len(self._state.yes_asks) +
                len(self._state.no_bids) + len(self._state.no_asks)
            )
        }


# Global registry of shared orderbook states
_orderbook_states: Dict[str, SharedOrderbookState] = {}
_states_lock = asyncio.Lock()


async def get_shared_orderbook_state(market_ticker: str) -> SharedOrderbookState:
    """
    Get or create a shared orderbook state for a market.
    
    Args:
        market_ticker: Market ticker
        
    Returns:
        SharedOrderbookState instance for the market
    """
    async with _states_lock:
        if market_ticker not in _orderbook_states:
            _orderbook_states[market_ticker] = SharedOrderbookState(market_ticker)
            logger.info(f"Created new shared orderbook state for {market_ticker}")
        
        return _orderbook_states[market_ticker]


async def get_all_orderbook_states() -> Dict[str, SharedOrderbookState]:
    """Get all registered orderbook states."""
    async with _states_lock:
        return dict(_orderbook_states)