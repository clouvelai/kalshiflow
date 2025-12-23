"""
PositionTracker - Extracted from OrderManager for TRADER 2.0

Handles position tracking and P&L calculations from order fills.
Focused extraction of working position management functionality from the monolithic OrderManager.
"""

import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import IntEnum

logger = logging.getLogger("kalshiflow_rl.trading.services.position_tracker")


class OrderSide(IntEnum):
    """Trading side enumeration."""
    BUY = 0
    SELL = 1


class ContractSide(IntEnum):
    """Contract side enumeration."""
    YES = 0
    NO = 1


@dataclass
class Position:
    """Represents a position in a specific market."""
    ticker: str
    contracts: int       # +contracts for YES, -contracts for NO (Kalshi convention)
    cost_basis: float    # Total cost in dollars
    realized_pnl: float  # Cumulative realized P&L in dollars
    
    @property
    def is_long_yes(self) -> bool:
        """True if long YES contracts."""
        return self.contracts > 0
    
    @property
    def is_long_no(self) -> bool:
        """True if long NO contracts (negative contracts)."""
        return self.contracts < 0
    
    @property
    def is_flat(self) -> bool:
        """True if no position."""
        return self.contracts == 0
    
    def get_unrealized_pnl(self, current_yes_price: float = None, yes_bid: float = None, yes_ask: float = None) -> float:
        """
        Calculate unrealized P&L based on current market price.
        
        Args:
            current_yes_price: Current YES price as a probability (0.0-1.0) - DEPRECATED, use bid/ask
            yes_bid: Current YES bid price (what we can sell at)
            yes_ask: Current YES ask price (what we must pay to buy)
            
        Returns:
            Unrealized P&L in dollars
        """
        if self.is_flat:
            return 0.0
        
        # Use bid/ask if provided, otherwise fall back to mid price for backward compatibility
        if yes_bid is not None and yes_ask is not None:
            if self.is_long_yes:
                # Long YES: use bid price (what we can sell at)
                exit_price = yes_bid
                current_value = self.contracts * exit_price
            else:
                # Long NO: use ask price to calculate NO position value
                # NO bid = 1 - YES ask (what we can sell NO at)
                no_bid = 1.0 - yes_ask
                current_value = abs(self.contracts) * no_bid
        else:
            # Fallback to mid price for backward compatibility
            if current_yes_price is None:
                return 0.0  # Can't calculate without price
            
            current_value = abs(self.contracts) * current_yes_price
            if self.is_long_no:
                current_value = abs(self.contracts) * (1.0 - current_yes_price)
        
        return current_value - self.cost_basis
    
    def get_total_pnl(self, current_yes_price: float = None, yes_bid: float = None, yes_ask: float = None) -> float:
        """Get total P&L (realized + unrealized)."""
        return self.realized_pnl + self.get_unrealized_pnl(current_yes_price, yes_bid, yes_ask)
    
    def average_cost_per_contract(self) -> float:
        """Get average cost per contract."""
        if self.contracts == 0:
            return 0.0
        return self.cost_basis / abs(self.contracts)


@dataclass
class FillInfo:
    """Information about an order fill for position tracking."""
    ticker: str
    side: OrderSide
    contract_side: ContractSide
    quantity: int
    fill_price: int  # Price in cents
    fill_timestamp: float
    order_id: str


class PositionTracker:
    """
    Handles position tracking and P&L calculations.
    
    Extracted from OrderManager to focus solely on position management,
    updating positions from fills, and calculating P&L.
    """
    
    def __init__(
        self, 
        initial_cash_balance: float = 0.0, 
        status_logger: Optional['StatusLogger'] = None,
        websocket_manager=None
    ):
        """
        Initialize PositionTracker.
        
        Args:
            initial_cash_balance: Starting cash balance
            status_logger: Optional StatusLogger for activity tracking
            websocket_manager: Global WebSocketManager for broadcasting (optional)
        """
        self.cash_balance = initial_cash_balance
        self.positions: Dict[str, Position] = {}
        self.status_logger = status_logger
        self.websocket_manager = websocket_manager
        
        # API-reported portfolio value (for positions we may not be tracking)
        self.api_portfolio_value = 0.0
        
        # Trade history for analysis
        self.trade_history: List[FillInfo] = []
        
        logger.info(f"PositionTracker initialized with ${initial_cash_balance:.2f}")
    
    def update_from_fill(self, fill_info: FillInfo) -> None:
        """
        Update position from a fill.
        
        Extracted from OrderManager._process_fill() logic.
        
        Args:
            fill_info: Fill information
        """
        try:
            # Add to trade history
            self.trade_history.append(fill_info)
            
            # Calculate fill cost in dollars
            fill_cost = (fill_info.fill_price / 100.0) * fill_info.quantity
            
            # Update cash balance
            if fill_info.side == OrderSide.BUY:
                self.cash_balance -= fill_cost  # Pay for bought contracts
            else:
                self.cash_balance += fill_cost  # Receive for sold contracts
            
            # Update position
            ticker = fill_info.ticker
            if ticker not in self.positions:
                self.positions[ticker] = Position(
                    ticker=ticker,
                    contracts=0,
                    cost_basis=0.0,
                    realized_pnl=0.0
                )
            
            position = self.positions[ticker]
            
            # Calculate position change based on Kalshi convention
            if fill_info.contract_side == ContractSide.YES:
                if fill_info.side == OrderSide.BUY:
                    contract_change = fill_info.quantity  # +YES
                else:
                    contract_change = -fill_info.quantity  # Sell YES
            else:  # NO contracts
                if fill_info.side == OrderSide.BUY:
                    contract_change = -fill_info.quantity  # Buy NO = -YES
                else:
                    contract_change = fill_info.quantity  # Sell NO = +YES
            
            # Check if this trade closes existing position (realizes P&L)
            if (position.contracts > 0 and contract_change < 0) or \
               (position.contracts < 0 and contract_change > 0):
                # Position reduction - calculate realized P&L
                reduction_amount = min(abs(contract_change), abs(position.contracts))
                
                # Calculate average cost per contract for the position being closed
                if position.contracts != 0:
                    avg_cost_per_contract = position.cost_basis / abs(position.contracts)
                    
                    # Calculate realized P&L
                    if position.contracts > 0:  # Closing long YES position
                        if fill_info.side == OrderSide.SELL:
                            realized_pnl = reduction_amount * (fill_info.fill_price / 100.0 - avg_cost_per_contract)
                        else:
                            # Buying NO to close YES (unusual but possible)
                            realized_pnl = reduction_amount * ((100 - fill_info.fill_price) / 100.0 - avg_cost_per_contract)
                    else:  # Closing long NO position
                        if fill_info.side == OrderSide.SELL:
                            # Selling NO to close (unusual)
                            realized_pnl = reduction_amount * (fill_info.fill_price / 100.0 - avg_cost_per_contract)
                        else:
                            # Buying YES to close NO
                            realized_pnl = reduction_amount * (avg_cost_per_contract - fill_info.fill_price / 100.0)
                    
                    position.realized_pnl += realized_pnl
                    
                    # Update cost basis proportionally
                    remaining_contracts = abs(position.contracts) - reduction_amount
                    if remaining_contracts > 0:
                        position.cost_basis *= remaining_contracts / abs(position.contracts)
                    else:
                        position.cost_basis = 0.0
                    
                    logger.info(f"Position reduction: {ticker} realized ${realized_pnl:.2f}")
            else:
                # Position increase - add to cost basis
                position.cost_basis += fill_cost
            
            # Update contract count
            position.contracts += contract_change
            
            # Broadcast position update
            if self.websocket_manager:
                import asyncio
                asyncio.create_task(self._broadcast_position_update(ticker, position, fill_info))
                asyncio.create_task(self._broadcast_portfolio_update())
            
            # Log activity
            if self.status_logger:
                import asyncio
                asyncio.create_task(self.status_logger.log_action_result(
                    "position_updated",
                    f"{ticker} {position.contracts} contracts (${position.get_total_pnl():.2f} P&L)",
                    0.0
                ))
            
            logger.info(f"Position updated: {ticker} {position.contracts} contracts @ ${position.average_cost_per_contract():.2f}")
            
        except Exception as e:
            logger.error(f"Error updating position from fill: {e}")
            if self.status_logger:
                import asyncio
                asyncio.create_task(self.status_logger.log_action_result(
                    "position_error",
                    f"{fill_info.ticker} - {str(e)[:50]}",
                    0.0
                ))
    
    def get_position(self, ticker: str) -> Optional[Position]:
        """Get position for a specific ticker."""
        return self.positions.get(ticker)
    
    def get_all_positions(self) -> Dict[str, Position]:
        """Get all positions."""
        return self.positions.copy()
    
    def get_active_positions(self) -> Dict[str, Position]:
        """Get positions with non-zero contracts."""
        return {ticker: pos for ticker, pos in self.positions.items() if not pos.is_flat}
    
    def calculate_total_pnl(self, market_prices: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, float]:
        """
        Calculate total P&L across all positions.
        
        Args:
            market_prices: Dict of {ticker: {"bid": float, "ask": float}} for unrealized P&L
            
        Returns:
            Dict with realized_pnl, unrealized_pnl, total_pnl, portfolio_value, total_value
        """
        total_realized = sum(pos.realized_pnl for pos in self.positions.values())
        total_unrealized = 0.0
        portfolio_value = 0.0
        
        # Calculate portfolio value (market value of all positions)
        active_positions = self.get_active_positions()
        for ticker, position in active_positions.items():
            if market_prices and ticker in market_prices:
                # Use market prices if available
                prices = market_prices[ticker]
                yes_bid = prices.get("bid")
                yes_ask = prices.get("ask")
                if yes_bid is not None and yes_ask is not None:
                    # Calculate unrealized P&L
                    total_unrealized += position.get_unrealized_pnl(
                        yes_bid=yes_bid / 100.0,  # Convert cents to dollars
                        yes_ask=yes_ask / 100.0
                    )
                    # Calculate market value of position
                    if position.is_long_yes:
                        # Long YES: value at bid price
                        portfolio_value += abs(position.contracts) * (yes_bid / 100.0)
                    else:
                        # Long NO: value at NO bid (1 - YES ask)
                        no_bid = 1.0 - (yes_ask / 100.0)
                        portfolio_value += abs(position.contracts) * no_bid
                else:
                    # Fallback to cost basis if no market price
                    portfolio_value += position.cost_basis
            else:
                # No market price available, use cost basis
                portfolio_value += position.cost_basis
        
        return {
            "realized_pnl": total_realized,
            "unrealized_pnl": total_unrealized,
            "total_pnl": total_realized + total_unrealized,
            "cash_balance": self.cash_balance,
            "portfolio_value": portfolio_value,  # Market value of positions
            "total_value": self.cash_balance + portfolio_value  # Cash + positions (not P&L)
        }
    
    def get_portfolio_summary(self, market_prices: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, Any]:
        """Get portfolio summary for status reporting."""
        pnl_info = self.calculate_total_pnl(market_prices)
        active_positions = self.get_active_positions()
        
        # Use API portfolio value if we have no tracked positions but API says there's value
        portfolio_value = pnl_info["portfolio_value"]
        if portfolio_value == 0.0 and self.api_portfolio_value > 0:
            # API knows about positions we're not tracking
            portfolio_value = self.api_portfolio_value
        
        # Calculate total value using the correct portfolio value
        total_value = self.cash_balance + portfolio_value
        
        return {
            "cash_balance": self.cash_balance,
            "portfolio_value": portfolio_value,              # Market value of positions (from API or calculated)
            "position_value": portfolio_value,               # Alias for compatibility
            "total_value": total_value,                      # Cash + positions
            "positions_count": len(active_positions),
            "positions_total": len(self.positions),
            "total_pnl": pnl_info["total_pnl"],
            "realized_pnl": pnl_info["realized_pnl"],
            "unrealized_pnl": pnl_info["unrealized_pnl"],
            "trades_today": len(self.trade_history),
            "active_tickers": list(active_positions.keys()),
            "largest_position": max([
                abs(pos.contracts) for pos in active_positions.values()
            ]) if active_positions else 0,
            "last_trade": self.trade_history[-1].fill_timestamp if self.trade_history else None
        }
    
    def sync_positions_from_api(self, api_positions: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
        """
        Sync positions from Kalshi API data.
        
        Args:
            api_positions: Raw position data from Kalshi API
            
        Returns:
            Dict of {ticker: sync_status} for reporting
        """
        sync_results = {}
        
        try:
            # Clear existing positions that aren't in API
            api_tickers = set(api_positions.keys())
            local_tickers = set(self.positions.keys())
            
            # Remove positions no longer in API
            for ticker in local_tickers - api_tickers:
                if self.positions[ticker].is_flat:
                    del self.positions[ticker]
                    sync_results[ticker] = "removed_flat"
                else:
                    sync_results[ticker] = "kept_nonzero"
            
            # Update positions from API
            for ticker, api_pos in api_positions.items():
                try:
                    # Parse API position format
                    # Expected format: {"ticker": str, "position": int, "total_cost": float, ...}
                    api_contracts = api_pos.get("position", 0)
                    api_cost = api_pos.get("total_cost", 0.0)
                    
                    if ticker not in self.positions:
                        self.positions[ticker] = Position(
                            ticker=ticker,
                            contracts=api_contracts,
                            cost_basis=api_cost,
                            realized_pnl=0.0
                        )
                        sync_results[ticker] = "created"
                    else:
                        # Update existing position
                        local_pos = self.positions[ticker]
                        
                        # Check for drift
                        contract_drift = abs(local_pos.contracts - api_contracts)
                        cost_drift = abs(local_pos.cost_basis - api_cost)
                        
                        if contract_drift > 0 or cost_drift > 0.01:  # 1 cent tolerance
                            logger.warning(f"Position drift detected for {ticker}: "
                                         f"contracts {local_pos.contracts} -> {api_contracts}, "
                                         f"cost ${local_pos.cost_basis:.2f} -> ${api_cost:.2f}")
                            
                            # Update to API values (API is source of truth)
                            local_pos.contracts = api_contracts
                            local_pos.cost_basis = api_cost
                            sync_results[ticker] = f"synced_drift_{contract_drift}_{cost_drift:.2f}"
                        else:
                            sync_results[ticker] = "in_sync"
                
                except Exception as e:
                    logger.error(f"Error syncing position {ticker}: {e}")
                    sync_results[ticker] = f"error_{str(e)[:20]}"
            
            logger.info(f"Position sync completed: {len(sync_results)} positions processed")
            
        except Exception as e:
            logger.error(f"Error during position sync: {e}")
            return {"sync_error": str(e)}
        
        return sync_results
    
    def get_position_statistics(self) -> Dict[str, Any]:
        """Get position statistics for monitoring."""
        active_positions = self.get_active_positions()
        
        if not active_positions:
            return {
                "positions_active": 0,
                "total_exposure": 0.0,
                "largest_position": 0,
                "avg_position_size": 0.0,
                "long_positions": 0,
                "short_positions": 0
            }
        
        exposures = [abs(pos.contracts * pos.average_cost_per_contract()) for pos in active_positions.values()]
        position_sizes = [abs(pos.contracts) for pos in active_positions.values()]
        long_count = sum(1 for pos in active_positions.values() if pos.is_long_yes)
        short_count = sum(1 for pos in active_positions.values() if pos.is_long_no)
        
        return {
            "positions_active": len(active_positions),
            "total_exposure": sum(exposures),
            "largest_position": max(position_sizes),
            "avg_position_size": sum(position_sizes) / len(position_sizes),
            "long_positions": long_count,
            "short_positions": short_count,
            "flat_positions": len(self.positions) - len(active_positions)
        }
    
    def check_cash_threshold(self, minimum_threshold: float) -> Dict[str, Any]:
        """
        Check if cash balance meets minimum threshold.
        
        Args:
            minimum_threshold: Minimum required cash balance
            
        Returns:
            Dict with threshold check results
        """
        try:
            is_sufficient = self.cash_balance >= minimum_threshold
            deficit = max(0, minimum_threshold - self.cash_balance)
            
            return {
                "sufficient": is_sufficient,
                "current_balance": self.cash_balance,
                "minimum_threshold": minimum_threshold,
                "deficit": deficit,
                "threshold_ratio": self.cash_balance / minimum_threshold if minimum_threshold > 0 else float('inf')
            }
            
        except Exception as e:
            logger.error(f"Error checking cash threshold: {e}")
            return {
                "sufficient": False,
                "current_balance": self.cash_balance,
                "minimum_threshold": minimum_threshold,
                "deficit": minimum_threshold,
                "threshold_ratio": 0.0,
                "error": str(e)
            }
    
    def estimate_position_liquidation_value(self, market_prices: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, Any]:
        """
        Estimate how much cash could be recovered by liquidating positions.
        
        Args:
            market_prices: Dict of {ticker: {"bid": float, "ask": float}} for current prices
            
        Returns:
            Dict with liquidation estimates
        """
        try:
            active_positions = self.get_active_positions()
            
            if not active_positions:
                return {
                    "estimated_recovery": 0.0,
                    "positions_count": 0,
                    "liquidation_details": [],
                    "potential_total_cash": self.cash_balance  # Just current cash since no positions
                }
            
            total_recovery = 0.0
            liquidation_details = []
            
            for ticker, position in active_positions.items():
                position_value = 0.0
                
                if market_prices and ticker in market_prices:
                    prices = market_prices[ticker]
                    yes_bid = prices.get("bid")
                    yes_ask = prices.get("ask")
                    
                    if yes_bid is not None and yes_ask is not None:
                        if position.is_long_yes:
                            # Long YES: sell at bid
                            position_value = position.contracts * (yes_bid / 100.0)
                        else:
                            # Long NO: close by buying YES at ask
                            position_value = abs(position.contracts) * ((100 - yes_ask) / 100.0)
                else:
                    # Conservative estimate: position worth cost basis
                    position_value = max(0, position.cost_basis * 0.5)  # 50% haircut
                
                total_recovery += position_value
                
                liquidation_details.append({
                    "ticker": ticker,
                    "contracts": position.contracts,
                    "cost_basis": position.cost_basis,
                    "estimated_value": position_value,
                    "type": "long_yes" if position.is_long_yes else "long_no"
                })
            
            return {
                "estimated_recovery": total_recovery,
                "positions_count": len(active_positions),
                "liquidation_details": liquidation_details,
                "potential_total_cash": self.cash_balance + total_recovery
            }
            
        except Exception as e:
            logger.error(f"Error estimating liquidation value: {e}")
            return {
                "estimated_recovery": 0.0,
                "positions_count": 0,
                "liquidation_details": [],
                "potential_total_cash": self.cash_balance,  # Default to current cash on error
                "error": str(e)
            }
    
    def get_cash_status(self, minimum_threshold: float, market_prices: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, Any]:
        """
        Get comprehensive cash status including threshold check and liquidation options.
        
        Args:
            minimum_threshold: Minimum required cash balance
            market_prices: Current market prices for liquidation estimates
            
        Returns:
            Dict with complete cash status
        """
        try:
            threshold_check = self.check_cash_threshold(minimum_threshold)
            liquidation_estimate = self.estimate_position_liquidation_value(market_prices)
            
            # Can we meet threshold by liquidating positions?
            potential_total = liquidation_estimate["potential_total_cash"]
            can_recover = potential_total >= minimum_threshold
            
            return {
                "cash_balance": self.cash_balance,
                "minimum_threshold": minimum_threshold,
                "threshold_check": threshold_check,
                "liquidation_estimate": liquidation_estimate,
                "can_recover_via_liquidation": can_recover,
                "recovery_needed": threshold_check["deficit"],
                "positions_available": len(self.get_active_positions()) > 0,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error getting cash status: {e}")
            return {
                "cash_balance": self.cash_balance,
                "minimum_threshold": minimum_threshold,
                "error": str(e),
                "timestamp": time.time()
            }
    
    def update_cash_balance_from_api(self, api_balance: float) -> bool:
        """
        Update cash balance from API sync (used by StateSync).
        
        Args:
            api_balance: Cash balance from Kalshi API
            
        Returns:
            True if balance was updated
        """
        try:
            old_balance = self.cash_balance
            self.cash_balance = api_balance
            
            if abs(old_balance - api_balance) > 0.01:  # 1 cent tolerance
                logger.info(f"Cash balance synced: ${old_balance:.2f} -> ${api_balance:.2f}")
                
            return True
            
        except Exception as e:
            logger.error(f"Error updating cash balance from API: {e}")
            return False
    
    def clear_all_positions(self) -> None:
        """Clear all positions (for testing or reset)."""
        logger.warning("Clearing all positions")
        self.positions.clear()
        self.trade_history.clear()
        self.cash_balance = 0.0
    
    async def _broadcast_position_update(self, ticker: str, position: Position, fill_info: FillInfo) -> None:
        """
        Broadcast individual position update via WebSocket.
        
        Args:
            ticker: Market ticker
            position: Updated position
            fill_info: Fill that triggered the update
        """
        if self.websocket_manager:
            try:
                position_data = {
                    "ticker": ticker,
                    "contracts": position.contracts,
                    "cost_basis": position.cost_basis,
                    "realized_pnl": position.realized_pnl,
                    "average_cost": position.average_cost_per_contract(),
                    "is_long_yes": position.is_long_yes,
                    "is_long_no": position.is_long_no,
                    "is_flat": position.is_flat,
                    "fill_trigger": {
                        "side": fill_info.side.name,
                        "contract_side": fill_info.contract_side.name,
                        "quantity": fill_info.quantity,
                        "fill_price": fill_info.fill_price
                    },
                    "timestamp": time.time()
                }
                
                await self.websocket_manager.broadcast_position_update(position_data)
            except Exception as e:
                logger.warning(f"Failed to broadcast position update: {e}")
    
    async def _broadcast_portfolio_update(self) -> None:
        """Broadcast overall portfolio update via WebSocket."""
        if self.websocket_manager:
            try:
                # Calculate total portfolio metrics
                total_unrealized = 0.0
                total_realized = 0.0
                position_count = 0
                
                for position in self.positions.values():
                    if not position.is_flat:
                        position_count += 1
                        total_unrealized += position.get_unrealized_pnl()
                        total_realized += position.realized_pnl
                
                portfolio_data = {
                    "cash_balance": self.cash_balance,
                    "position_count": position_count,
                    "total_positions": len(self.positions),
                    "total_unrealized_pnl": total_unrealized,
                    "total_realized_pnl": total_realized,
                    "total_pnl": total_unrealized + total_realized,
                    "trade_count": len(self.trade_history),
                    "timestamp": time.time()
                }
                
                await self.websocket_manager.broadcast_portfolio_update(portfolio_data)
            except Exception as e:
                logger.warning(f"Failed to broadcast portfolio update: {e}")