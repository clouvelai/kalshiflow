"""
Kalshi data synchronization service for V3.

Fetches current state from Kalshi API and tracks changes.
"""

import logging
import time
from typing import Optional, Tuple

from ..state.trader_state import TraderState, StateChange

logger = logging.getLogger("kalshiflow_rl.traderv3.sync.kalshi_data_sync")


class KalshiDataSync:
    """
    Synchronizes trader state with Kalshi API.
    
    This service:
    - Fetches balance, positions, orders, and settlements from Kalshi
    - Builds a TraderState from the raw data
    - Tracks changes between syncs
    - Works entirely in cents (Kalshi's native unit)
    """
    
    def __init__(self, trading_client):
        """
        Initialize sync service.
        
        Args:
            trading_client: KalshiDemoTradingClient instance
        """
        self._client = trading_client
        self._current_state: Optional[TraderState] = None
        self._sync_count = 0
    
    async def sync_with_kalshi(self) -> Tuple[TraderState, Optional[StateChange]]:
        """
        Perform full state synchronization with Kalshi.
        
        Returns:
            Tuple of (new_state, changes_from_previous)
            changes will be None on first sync
        """
        start_time = time.time()
        self._sync_count += 1
        
        # Store previous state for comparison
        previous_state = self._current_state
        
        try:
            logger.info(f"Starting Kalshi data sync #{self._sync_count}...")
            
            # Fetch all data from Kalshi API
            # Note: These are the actual API calls to Kalshi
            
            # 1. Get balance (returns balance and portfolio_value in cents)
            logger.debug("Fetching account balance...")
            balance_response = await self._client.get_account_info()
            
            # 2. Get positions (returns market_positions array)
            logger.debug("Fetching positions...")
            positions_response = await self._client.get_positions()
            
            # 3. Get orders (returns orders array)
            logger.debug("Fetching orders...")
            orders_response = await self._client.get_orders()
            
            # 4. Get settlements (optional, may fail)
            settlements = []
            try:
                logger.debug("Fetching settlements...")
                settlements_response = await self._client.get_settlements()
                settlements = settlements_response.get("settlements", [])
            except Exception as e:
                logger.warning(f"Could not fetch settlements: {e}")
            
            # Build new state from raw Kalshi data
            new_state = TraderState.from_kalshi_data(
                balance_data=balance_response,
                positions_data=positions_response,
                orders_data=orders_response,
                settlements_data=settlements
            )
            
            # Calculate changes if we have a previous state
            changes = None
            if previous_state:
                changes = StateChange(
                    balance_change=new_state.balance - previous_state.balance,
                    portfolio_value_change=new_state.portfolio_value - previous_state.portfolio_value,
                    position_count_change=new_state.position_count - previous_state.position_count,
                    order_count_change=new_state.order_count - previous_state.order_count
                )
                
                # Log changes
                if changes.balance_change != 0 or changes.portfolio_value_change != 0:
                    logger.info(
                        f"Account changes - Balance: {changes.balance_change:+d} cents, "
                        f"Portfolio: {changes.portfolio_value_change:+d} cents"
                    )
                
                if changes.position_count_change != 0 or changes.order_count_change != 0:
                    logger.info(
                        f"Position/Order changes - Positions: {changes.position_count_change:+d}, "
                        f"Orders: {changes.order_count_change:+d}"
                    )
            
            # Update current state
            self._current_state = new_state
            
            duration_ms = (time.time() - start_time) * 1000
            
            # Log summary
            logger.info(
                f"✅ Kalshi data sync #{self._sync_count} complete in {duration_ms:.1f}ms - "
                f"Balance: {new_state.balance} cents, Portfolio: {new_state.portfolio_value} cents, "
                f"Positions: {new_state.position_count}, Orders: {new_state.order_count}"
            )
            
            return new_state, changes
            
        except Exception as e:
            logger.error(f"❌ Kalshi data sync failed: {e}")
            raise
    
    @property
    def current_state(self) -> Optional[TraderState]:
        """Get current trader state."""
        return self._current_state
    
    @property
    def sync_count(self) -> int:
        """Get number of syncs performed."""
        return self._sync_count
    
    def has_state(self) -> bool:
        """Check if we have synced state."""
        return self._current_state is not None
    
    async def refresh_state(self) -> Optional[TraderState]:
        """
        Refresh the current state without tracking changes.
        Useful for periodic updates.
        
        Returns:
            Current trader state or None on error
        """
        try:
            state, _ = await self.sync_with_kalshi()
            return state
        except Exception as e:
            logger.error(f"Failed to refresh state: {e}")
            return None