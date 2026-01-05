"""
In-memory orderbook state management for RL Trading Subsystem.

Provides OrderbookState for efficient price level operations using SortedDict,
and SharedOrderbookState with thread-safe access and subscriber notifications.

Supports dependency injection via ServiceContainer to replace global singleton patterns.
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

# Maximum number of price levels to retain per side (bid/ask)
# Reduces memory footprint while keeping sufficient depth for trading decisions
MAX_ORDERBOOK_LEVELS = 5


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
        
        Kalshi only provides bids due to the reciprocal nature of binary markets.
        We derive asks from bids using the relationships:
        - YES_BID at price X → NO_ASK at price (100 - X)
        - NO_BID at price Y → YES_ASK at price (100 - Y)
        
        Args:
            snapshot_data: Dict with yes_bids, no_bids (asks are derived)
        """
        self.last_sequence = snapshot_data.get('sequence_number', 0)
        self.last_update_time = snapshot_data.get('timestamp_ms', int(time.time() * 1000))
        
        # Clear existing state
        self.yes_bids.clear()
        self.yes_asks.clear()
        self.no_bids.clear()
        self.no_asks.clear()
        
        # Apply bid data from snapshot
        yes_bids_data = snapshot_data.get('yes_bids', {})
        no_bids_data = snapshot_data.get('no_bids', {})
        
        self._apply_price_levels(self.yes_bids, yes_bids_data)
        self._apply_price_levels(self.no_bids, no_bids_data)
        
        # CRITICAL FIX: Derive asks from bids using correct Kalshi arbitrage relationship
        # NO_BID at Y → YES_ASK at (100 - Y) [NOT 99-Y, that was wrong!]
        for price_str, size in no_bids_data.items():
            price = int(price_str)
            ask_price = 100 - price
            if 1 <= ask_price <= 99:  # Ensure valid price range
                self.yes_asks[ask_price] = size
        
        # YES_BID at X → NO_ASK at (100 - X) [NOT 99-X, that was wrong!]
        for price_str, size in yes_bids_data.items():
            price = int(price_str)
            ask_price = 100 - price
            if 1 <= ask_price <= 99:  # Ensure valid price range
                self.no_asks[ask_price] = size
        
        # Prune to max levels and invalidate cache
        self._prune_to_max_levels()
        self._invalidate_cache()

        logger.debug(
            f"Applied snapshot for {self.market_ticker}: seq={self.last_sequence}, "
            f"yes_levels={len(self.yes_bids) + len(self.yes_asks)}, "
            f"no_levels={len(self.no_bids) + len(self.no_asks)} "
            f"(pruned to max {MAX_ORDERBOOK_LEVELS} per side)"
        )
    
    def apply_delta(self, delta_data: Dict[str, Any]) -> bool:
        """
        Apply an orderbook delta update.
        
        Kalshi only provides bid-side deltas. We apply the delta to the bid side
        and then update the corresponding ask side using the reciprocal relationship:
        - YES_BID at X → NO_ASK at (100 - X)
        - NO_BID at Y → YES_ASK at (100 - Y)
        
        Args:
            delta_data: Dict with side, action, price, old_size, new_size, sequence
            
        Returns:
            bool: True if delta was applied successfully
        """
        sequence_number = delta_data.get('sequence_number', 0)
        
        # Sequence validation (allow missing sequences in some cases)
        if sequence_number <= self.last_sequence:
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
        
        # Kalshi sends deltas for bids only. We update the bid side
        # and derive the corresponding ask update.
        if side == 'yes':
            # Apply to YES bid
            if action in ['add', 'update']:
                if new_size > 0:
                    self.yes_bids[price] = new_size
                    # CRITICAL FIX: Derive NO ask at (100 - price) [NOT 99 - price!]
                    derived_ask_price = 100 - price
                    if 1 <= derived_ask_price <= 99:
                        self.no_asks[derived_ask_price] = new_size
                else:
                    # Remove if size is 0
                    self.yes_bids.pop(price, None)
                    derived_ask_price = 100 - price
                    self.no_asks.pop(derived_ask_price, None)
            elif action == 'remove':
                # Remove from YES bids
                self.yes_bids.pop(price, None)
                # Remove corresponding NO ask
                derived_ask_price = 100 - price
                self.no_asks.pop(derived_ask_price, None)
            else:
                logger.error(f"Unknown action: {action}")
                return False
                
        elif side == 'no':
            # Apply to NO bid
            if action in ['add', 'update']:
                if new_size > 0:
                    self.no_bids[price] = new_size
                    # CRITICAL FIX: Derive YES ask at (100 - price) [NOT 99 - price!]
                    derived_ask_price = 100 - price
                    if 1 <= derived_ask_price <= 99:
                        self.yes_asks[derived_ask_price] = new_size
                else:
                    # Remove if size is 0
                    self.no_bids.pop(price, None)
                    derived_ask_price = 100 - price
                    self.yes_asks.pop(derived_ask_price, None)
            elif action == 'remove':
                # Remove from NO bids
                self.no_bids.pop(price, None)
                # Remove corresponding YES ask
                derived_ask_price = 100 - price
                self.yes_asks.pop(derived_ask_price, None)
            else:
                logger.error(f"Unknown action: {action}")
                return False
        else:
            logger.error(f"Invalid side: {side}")
            return False
        
        self.last_sequence = sequence_number
        self.last_update_time = delta_data.get('timestamp_ms', int(time.time() * 1000))

        # Prune to max levels (only if needed - deltas rarely cause overflow)
        self._prune_to_max_levels()
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

    def _prune_to_max_levels(self) -> None:
        """
        Prune each order book side to MAX_ORDERBOOK_LEVELS.

        SortedDict maintains order, so we keep the first N entries:
        - Bids: sorted descending (highest price first) - keep best N bids
        - Asks: sorted ascending (lowest price first) - keep best N asks
        """
        max_levels = MAX_ORDERBOOK_LEVELS

        # Prune YES bids (descending order, keep highest N)
        if len(self.yes_bids) > max_levels:
            keys_to_keep = list(self.yes_bids.keys())[:max_levels]
            for key in list(self.yes_bids.keys()):
                if key not in keys_to_keep:
                    del self.yes_bids[key]

        # Prune YES asks (ascending order, keep lowest N)
        if len(self.yes_asks) > max_levels:
            keys_to_keep = list(self.yes_asks.keys())[:max_levels]
            for key in list(self.yes_asks.keys()):
                if key not in keys_to_keep:
                    del self.yes_asks[key]

        # Prune NO bids (descending order, keep highest N)
        if len(self.no_bids) > max_levels:
            keys_to_keep = list(self.no_bids.keys())[:max_levels]
            for key in list(self.no_bids.keys()):
                if key not in keys_to_keep:
                    del self.no_bids[key]

        # Prune NO asks (ascending order, keep lowest N)
        if len(self.no_asks) > max_levels:
            keys_to_keep = list(self.no_asks.keys())[:max_levels]
            for key in list(self.no_asks.keys()):
                if key not in keys_to_keep:
                    del self.no_asks[key]
    
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

    async def apply_rest_snapshot(self, rest_data: Dict[str, Any]) -> None:
        """
        Apply orderbook data from REST API response (fallback for stale WS data).

        Converts REST format to internal format and updates state.

        REST format: { "orderbook": { "yes": [[price, count], ...], "no": [[price, count], ...] } }
        Internal format: { "yes_bids": {price: size}, "no_bids": {price: size} }
        Note: Keys are converted to int internally via _apply_price_levels.

        Args:
            rest_data: REST API response from /markets/{ticker}/orderbook
        """
        orderbook = rest_data.get("orderbook", {})

        # Convert REST [[price, count], ...] to internal {price_str: count} format
        yes_bids = {str(p): c for p, c in orderbook.get("yes", [])}
        no_bids = {str(p): c for p, c in orderbook.get("no", [])}

        # Build snapshot data in expected format
        snapshot_data = {
            "yes_bids": yes_bids,
            "no_bids": no_bids,
            "sequence_number": 0,  # REST doesn't provide sequence
            "timestamp_ms": int(time.time() * 1000),  # Use current time
        }

        async with self._lock:
            self._state.apply_snapshot(snapshot_data)
            await self._notify_subscribers('rest_snapshot')

        logger.info(
            f"Applied REST snapshot for {self.market_ticker}: "
            f"yes_levels={len(yes_bids)}, no_levels={len(no_bids)}"
        )

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


# ===============================================================================
# Dependency Injection Support
# ===============================================================================

# Global registry of shared orderbook states (maintained for backward compatibility)
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
    """
    Get all registered orderbook states.
    """
    async with _states_lock:
        return dict(_orderbook_states)


async def remove_shared_orderbook_state(market_ticker: str) -> bool:
    """
    Remove a shared orderbook state from the global registry.

    Call this when unsubscribing from a market to prevent memory leaks.

    Args:
        market_ticker: Market ticker to remove

    Returns:
        True if state was removed, False if it didn't exist
    """
    async with _states_lock:
        if market_ticker in _orderbook_states:
            del _orderbook_states[market_ticker]
            logger.info(f"Removed shared orderbook state for {market_ticker}")
            return True
        return False


async def remove_shared_orderbook_states(market_tickers: List[str]) -> int:
    """
    Remove multiple shared orderbook states from the global registry.

    Args:
        market_tickers: List of market tickers to remove

    Returns:
        Number of states that were removed
    """
    removed_count = 0
    async with _states_lock:
        for ticker in market_tickers:
            if ticker in _orderbook_states:
                del _orderbook_states[ticker]
                removed_count += 1

        if removed_count > 0:
            logger.info(f"Removed {removed_count} shared orderbook states from global registry")

    return removed_count


def get_global_registry_size() -> int:
    """
    Get the current size of the global orderbook state registry.

    Useful for monitoring memory usage.
    """
    return len(_orderbook_states)


async def cleanup_global_orderbook_states() -> None:
    """
    Clean up the global orderbook state registry.
    
    Used for testing and migration cleanup.
    """
    global _orderbook_states
    async with _states_lock:
        _orderbook_states.clear()
        logger.info("Global orderbook state registry cleared")