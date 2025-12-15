"""
Multi-Market Order Manager for Kalshi Trading Actor MVP.

Extends the base OrderManager to handle multi-market portfolio tracking and execution
with global constraints and per-market throttling. Integrates with SharedOrderbookState
via global registry and provides execution safety checks.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict

from .order_manager import OrderManager, Position
from .demo_client import KalshiDemoTradingClient
from ..data.orderbook_state import get_shared_orderbook_state
from ..config import config

logger = logging.getLogger("kalshiflow_rl.multi_market_order_manager")


@dataclass
class MarketThrottleState:
    """Per-market throttling state."""
    last_action_time: float = 0.0
    action_count: int = 0
    consecutive_errors: int = 0
    disabled: bool = False


@dataclass
class GlobalPortfolioState:
    """Global portfolio state across all markets."""
    total_cash: float = 10000.0  # Starting cash in dollars
    total_portfolio_value: float = 10000.0
    total_positions: int = 0  # Total contracts across all markets
    total_realized_pnl: float = 0.0
    
    # Risk limits
    max_position_per_market: int = 100  # Max contracts per market
    max_total_positions: int = 500      # Max total contracts across all markets  
    min_cash_reserve: float = 1000.0    # Minimum cash to maintain
    max_daily_loss: float = 2000.0      # Maximum daily loss limit
    
    # Daily tracking
    daily_pnl: float = 0.0
    daily_trades: int = 0
    session_start_value: float = 10000.0


class MultiMarketOrderManager:
    """
    Multi-market order manager with global portfolio tracking.
    
    Features:
    - Single manager handles all markets (up to 1000+)
    - Global portfolio value calculation across markets
    - Per-market throttling (250ms minimum between actions)
    - Market-specific queries (positions, orders, eligibility)
    - Comprehensive execution safety checks
    - Integration with KalshiDemoTradingClient
    - Global risk controls and position limits
    """
    
    def __init__(
        self,
        markets: Optional[List[str]] = None,
        throttle_ms: int = 250,
        starting_cash: float = 10000.0,
        demo_trading: bool = True
    ):
        """
        Initialize multi-market order manager.
        
        Args:
            markets: List of markets to manage (defaults to config)
            throttle_ms: Minimum milliseconds between actions per market
            starting_cash: Starting cash balance in dollars
            demo_trading: Use demo trading client (paper trading only)
        """
        self.markets = markets or config.RL_MARKET_TICKERS
        self.throttle_ms = throttle_ms
        self.demo_trading = demo_trading
        
        # Per-market order managers (composition over inheritance)
        self._market_managers: Dict[str, OrderManager] = {}
        
        # Per-market throttling
        self._market_throttle: Dict[str, MarketThrottleState] = {}
        
        # Global portfolio state
        self.global_state = GlobalPortfolioState(
            total_cash=starting_cash,
            total_portfolio_value=starting_cash,
            session_start_value=starting_cash
        )
        
        # Trading client (demo only)
        self._trading_client: Optional[KalshiDemoTradingClient] = None
        
        # Performance and safety monitoring
        self._execution_count = 0
        self._error_count = 0
        self._last_portfolio_update = 0.0
        self._safety_circuit_breaker = False
        
        # State change callbacks for UI updates
        self._state_change_callbacks = []
        
        logger.info(
            f"MultiMarketOrderManager initialized for {len(self.markets)} markets: "
            f"{', '.join(self.markets[:5])}{'...' if len(self.markets) > 5 else ''}"
        )
        logger.info(f"Starting cash: ${starting_cash:.2f}, Demo mode: {demo_trading}")
        
        # Initialize per-market states
        for market_ticker in self.markets:
            self._market_throttle[market_ticker] = MarketThrottleState()
    
    async def initialize(self) -> None:
        """Initialize the multi-market order manager."""
        logger.info("Initializing MultiMarketOrderManager...")
        
        # Initialize trading client (demo only for MVP)
        if self.demo_trading:
            try:
                self._trading_client = KalshiDemoTradingClient()
                await self._trading_client.connect()
                logger.info("✅ Demo trading client connected")
            except Exception as e:
                logger.error(f"Failed to initialize demo trading client: {e}")
                raise
        
        # Initialize per-market order managers
        # Use SimulatedOrderManager for demo trading (can't instantiate abstract OrderManager)
        for market_ticker in self.markets:
            try:
                if self.demo_trading:
                    # For demo/paper trading, use simulated manager
                    from .order_manager import SimulatedOrderManager
                    # Each market gets portion of total cash
                    cash_per_market = int(self.global_state.total_cash * 100 / len(self.markets))  # Convert to cents
                    manager = SimulatedOrderManager(initial_cash=cash_per_market)
                else:
                    # For live trading, would use KalshiOrderManager
                    # Not implemented in MVP
                    raise NotImplementedError("Live trading not yet implemented")
                
                self._market_managers[market_ticker] = manager
                logger.debug(f"Initialized OrderManager for {market_ticker}")
            except Exception as e:
                logger.error(f"Failed to initialize OrderManager for {market_ticker}: {e}")
                # Continue with other markets
        
        logger.info(f"✅ MultiMarketOrderManager initialized with {len(self._market_managers)} active markets")
    
    async def can_trade_market(self, market_ticker: str) -> bool:
        """
        Check if trading is allowed for a specific market.
        
        Args:
            market_ticker: Market to check
            
        Returns:
            True if trading is allowed, False otherwise
        """
        if self._safety_circuit_breaker:
            return False
        
        if market_ticker not in self.markets:
            return False
        
        # Check market-specific throttle state
        throttle_state = self._market_throttle.get(market_ticker)
        if not throttle_state or throttle_state.disabled:
            return False
        
        # Check throttling
        current_time = time.time()
        time_since_last = (current_time - throttle_state.last_action_time) * 1000
        
        if time_since_last < self.throttle_ms:
            return False
        
        # Check global risk limits
        if not self._check_global_risk_limits():
            return False
        
        # Check market-specific limits
        current_position = await self.get_position_for_market(market_ticker)
        if abs(current_position.get('position', 0)) >= self.global_state.max_position_per_market:
            return False
        
        return True
    
    async def get_position_for_market(self, market_ticker: str) -> Dict[str, Any]:
        """
        Get position data for a specific market.
        
        Args:
            market_ticker: Market ticker
            
        Returns:
            Position data dict with 'position', 'cost_basis', 'realized_pnl'
        """
        if market_ticker not in self._market_managers:
            return {'position': 0, 'cost_basis': 0.0, 'realized_pnl': 0.0}
        
        manager = self._market_managers[market_ticker]
        return manager.get_position()
    
    async def get_orders_for_market(self, market_ticker: str) -> List[Dict[str, Any]]:
        """
        Get open orders for a specific market.
        
        Args:
            market_ticker: Market ticker
            
        Returns:
            List of open orders for the market
        """
        if market_ticker not in self._market_managers:
            return []
        
        manager = self._market_managers[market_ticker]
        return manager.get_open_orders()
    
    async def execute_limit_order_action(
        self,
        action: int,
        market_ticker: str,
        orderbook_snapshot: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Execute limit order action for a specific market.
        
        Args:
            action: Action ID (0=HOLD, 1=BUY_YES_NOW, 2=SELL_YES_NOW, 3=BUY_NO_NOW, 4=SELL_NO_NOW)
            market_ticker: Market to trade
            orderbook_snapshot: Current orderbook state (optional)
            
        Returns:
            Execution result dict or None if execution failed/blocked
        """
        start_time = time.time()
        
        try:
            # Pre-execution safety checks
            if not await self._pre_execution_safety_checks(action, market_ticker):
                return None
            
            # Get orderbook snapshot if not provided
            if orderbook_snapshot is None:
                shared_state = await get_shared_orderbook_state(market_ticker)
                orderbook_snapshot = shared_state.get_snapshot()
                
                if not orderbook_snapshot:
                    logger.warning(f"No orderbook data available for {market_ticker}")
                    return None
            
            # Execute action through market-specific manager
            manager = self._market_managers.get(market_ticker)
            if not manager:
                logger.error(f"No manager found for {market_ticker}")
                return None
            
            # Execute the order
            execution_result = await manager.execute_limit_order_action(
                action=action,
                orderbook_snapshot=orderbook_snapshot,
                trading_client=self._trading_client
            )
            
            # Update throttling state
            self._update_market_throttle(market_ticker, True)
            
            # Update global portfolio state
            await self._update_global_portfolio()
            
            # Update metrics
            self._execution_count += 1
            execution_time = time.time() - start_time
            
            # Notify state change callbacks (for UI updates)
            await self._notify_state_change()
            
            logger.info(
                f"Executed action {action} for {market_ticker}: "
                f"result={execution_result.get('status', 'unknown') if execution_result else 'failed'}, "
                f"execution_time={execution_time*1000:.2f}ms"
            )
            
            return execution_result
            
        except Exception as e:
            self._error_count += 1
            self._update_market_throttle(market_ticker, False)
            logger.error(f"Error executing action {action} for {market_ticker}: {e}")
            return None
    
    async def get_total_portfolio_value(self) -> float:
        """
        Calculate total portfolio value across all markets.
        
        Returns:
            Total portfolio value including cash and positions
        """
        total_value = self.global_state.total_cash
        
        # Add unrealized PnL from all markets
        for market_ticker, manager in self._market_managers.items():
            try:
                position_data = manager.get_position()
                position = position_data.get('position', 0)
                cost_basis = position_data.get('cost_basis', 0.0)
                
                if position != 0:
                    # Get current market price for mark-to-market
                    shared_state = await get_shared_orderbook_state(market_ticker)
                    snapshot = shared_state.get_snapshot()
                    
                    if snapshot:
                        current_price = self._calculate_mark_price(snapshot, position > 0)
                        market_value = position * current_price / 100.0  # Convert cents to dollars
                        unrealized_pnl = market_value - cost_basis
                        total_value += cost_basis + unrealized_pnl
                    else:
                        # No current price available, use cost basis
                        total_value += cost_basis
                        
            except Exception as e:
                logger.error(f"Error calculating portfolio value for {market_ticker}: {e}")
        
        self.global_state.total_portfolio_value = total_value
        return total_value
    
    async def _pre_execution_safety_checks(self, action: int, market_ticker: str) -> bool:
        """
        Run comprehensive safety checks before executing any trade.
        
        Args:
            action: Action to execute
            market_ticker: Market for action
            
        Returns:
            True if safe to execute, False otherwise
        """
        # Check if action requires trading
        if action == 0:  # HOLD action
            return True  # HOLD always safe
        
        # Check if trading is allowed for this market
        if not await self.can_trade_market(market_ticker):
            logger.debug(f"Trading not allowed for {market_ticker}")
            return False
        
        # Check global circuit breaker
        if self._safety_circuit_breaker:
            logger.warning("Global safety circuit breaker active")
            return False
        
        # Check cash availability
        if self.global_state.total_cash < self.global_state.min_cash_reserve:
            logger.warning(f"Insufficient cash: ${self.global_state.total_cash:.2f} < ${self.global_state.min_cash_reserve:.2f}")
            return False
        
        # Check daily loss limits
        current_pnl = await self.get_total_portfolio_value() - self.global_state.session_start_value
        if current_pnl < -self.global_state.max_daily_loss:
            logger.warning(f"Daily loss limit exceeded: ${current_pnl:.2f}")
            self._safety_circuit_breaker = True
            return False
        
        # Check total position limits
        if self.global_state.total_positions >= self.global_state.max_total_positions:
            logger.warning(f"Total position limit reached: {self.global_state.total_positions}")
            return False
        
        return True
    
    def _update_market_throttle(self, market_ticker: str, success: bool) -> None:
        """Update throttling state for a market after execution attempt."""
        current_time = time.time()
        throttle_state = self._market_throttle.get(market_ticker)
        
        if throttle_state:
            throttle_state.last_action_time = current_time
            throttle_state.action_count += 1
            
            if success:
                throttle_state.consecutive_errors = 0
            else:
                throttle_state.consecutive_errors += 1
                
                # Disable market if too many consecutive errors
                if throttle_state.consecutive_errors >= 5:
                    logger.error(f"Disabling {market_ticker} due to consecutive errors")
                    throttle_state.disabled = True
    
    async def _update_global_portfolio(self) -> None:
        """Update global portfolio state from all market managers."""
        try:
            total_cash = self.global_state.total_cash
            total_positions = 0
            total_realized_pnl = 0.0
            
            # Aggregate from all market managers
            for market_ticker, manager in self._market_managers.items():
                position_data = manager.get_position()
                total_positions += abs(position_data.get('position', 0))
                total_realized_pnl += position_data.get('realized_pnl', 0.0)
            
            # Update global state
            self.global_state.total_positions = total_positions
            self.global_state.total_realized_pnl = total_realized_pnl
            self.global_state.total_portfolio_value = await self.get_total_portfolio_value()
            self.global_state.daily_pnl = self.global_state.total_portfolio_value - self.global_state.session_start_value
            
            self._last_portfolio_update = time.time()
            
        except Exception as e:
            logger.error(f"Error updating global portfolio: {e}")
    
    def _check_global_risk_limits(self) -> bool:
        """Check if global risk limits allow trading."""
        # Cash limits
        if self.global_state.total_cash < self.global_state.min_cash_reserve:
            return False
        
        # Position limits  
        if self.global_state.total_positions >= self.global_state.max_total_positions:
            return False
        
        # Daily loss limits
        if self.global_state.daily_pnl < -self.global_state.max_daily_loss:
            return False
        
        return True
    
    def _calculate_mark_price(self, snapshot: Dict[str, Any], is_yes_position: bool) -> float:
        """
        Calculate mark-to-market price from orderbook snapshot.
        
        Args:
            snapshot: Orderbook snapshot
            is_yes_position: True if pricing YES position, False for NO
            
        Returns:
            Mark price in cents
        """
        try:
            if is_yes_position:
                # For YES position, use mid of YES bid/ask
                yes_bids = snapshot.get('yes_bids', {})
                yes_asks = snapshot.get('yes_asks', {})
                
                if yes_bids and yes_asks:
                    best_bid = max(map(int, yes_bids.keys()))
                    best_ask = min(map(int, yes_asks.keys()))
                    return (best_bid + best_ask) / 2.0
                elif yes_bids:
                    return float(max(map(int, yes_bids.keys())))
                elif yes_asks:
                    return float(min(map(int, yes_asks.keys())))
            else:
                # For NO position, use mid of NO bid/ask  
                no_bids = snapshot.get('no_bids', {})
                no_asks = snapshot.get('no_asks', {})
                
                if no_bids and no_asks:
                    best_bid = max(map(int, no_bids.keys()))
                    best_ask = min(map(int, no_asks.keys()))
                    return (best_bid + best_ask) / 2.0
                elif no_bids:
                    return float(max(map(int, no_bids.keys())))
                elif no_asks:
                    return float(min(map(int, no_asks.keys())))
            
            # Default to 50 cents if no data
            return 50.0
            
        except Exception as e:
            logger.error(f"Error calculating mark price: {e}")
            return 50.0
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics for monitoring."""
        return {
            "execution_count": self._execution_count,
            "error_count": self._error_count,
            "error_rate": self._error_count / max(self._execution_count, 1),
            "active_markets": len(self._market_managers),
            "total_markets": len(self.markets),
            "disabled_markets": sum(1 for state in self._market_throttle.values() if state.disabled),
            "safety_circuit_breaker": self._safety_circuit_breaker,
            "global_portfolio": {
                "total_cash": self.global_state.total_cash,
                "total_portfolio_value": self.global_state.total_portfolio_value,
                "total_positions": self.global_state.total_positions,
                "daily_pnl": self.global_state.daily_pnl,
                "daily_trades": self.global_state.daily_trades
            },
            "last_portfolio_update": self._last_portfolio_update
        }
    
    def add_state_change_callback(self, callback):
        """
        Register callback for state changes.
        
        Args:
            callback: Async callable that takes state dict as argument
        """
        self._state_change_callbacks.append(callback)
        logger.debug(f"Added state change callback. Total callbacks: {len(self._state_change_callbacks)}")
    
    async def _notify_state_change(self):
        """Notify all callbacks of state change."""
        if not self._state_change_callbacks:
            return
        
        state = await self.get_current_state()
        for callback in self._state_change_callbacks:
            try:
                # Create async task to avoid blocking
                asyncio.create_task(callback(state))
            except Exception as e:
                logger.error(f"Error in state change callback: {e}")
    
    async def get_current_state(self) -> Dict[str, Any]:
        """
        Get complete current state snapshot for UI.
        
        Returns:
            Dict containing all trader state information
        """
        # Gather positions across all markets
        positions = {}
        open_orders = []
        
        for market_ticker in self._market_managers:
            # Get position for market
            position_data = await self.get_position_for_market(market_ticker)
            if position_data.get('position', 0) != 0:
                positions[market_ticker] = {
                    "contracts": position_data.get('position', 0),
                    "cost_basis": position_data.get('cost_basis', 0.0),
                    "realized_pnl": position_data.get('realized_pnl', 0.0),
                    "unrealized_pnl": 0.0  # Would need current prices to calculate
                }
            
            # Get open orders for market
            market_orders = await self.get_orders_for_market(market_ticker)
            for order in market_orders:
                open_orders.append({
                    "order_id": order.get("order_id"),
                    "ticker": market_ticker,
                    "side": order.get("side"),
                    "contract_side": order.get("contract_side", "YES"),  # Default to YES
                    "quantity": order.get("quantity"),
                    "limit_price": order.get("price"),
                    "placed_at": order.get("placed_at", time.time()),
                    "status": order.get("status", "PENDING")
                })
        
        # Calculate portfolio value (simplified - would need current prices)
        portfolio_value = self.global_state.total_cash
        for market, position in positions.items():
            # Add position value at cost basis for now
            portfolio_value += abs(position["contracts"]) * position.get("cost_basis", 0)
        
        return {
            "timestamp": time.time(),
            "cash_balance": self.global_state.total_cash,
            "promised_cash": sum(
                order["quantity"] * order["limit_price"] / 100  # Convert cents to dollars
                for order in open_orders
                if order.get("side") == "BUY"
            ),
            "portfolio_value": portfolio_value,
            "positions": positions,
            "open_orders": open_orders,
            "metrics": {
                "orders_placed": self._execution_count,
                "orders_filled": self._execution_count - len(open_orders),  # Approximation
                "orders_cancelled": 0,  # Would need to track this
                "fill_rate": (self._execution_count - len(open_orders)) / max(1, self._execution_count),
                "total_volume_traded": abs(self.global_state.total_realized_pnl) * 100,  # Approximation
                "daily_pnl": self.global_state.daily_pnl,
                "daily_trades": self.global_state.daily_trades
            }
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status for monitoring."""
        throttle_stats = {}
        for market, state in self._market_throttle.items():
            throttle_stats[market] = {
                "last_action": state.last_action_time,
                "action_count": state.action_count,
                "consecutive_errors": state.consecutive_errors,
                "disabled": state.disabled
            }
        
        return {
            "manager": "MultiMarketOrderManager",
            "demo_trading": self.demo_trading,
            "throttle_ms": self.throttle_ms,
            "metrics": self.get_metrics(),
            "market_throttle": throttle_stats,
            "risk_limits": {
                "max_position_per_market": self.global_state.max_position_per_market,
                "max_total_positions": self.global_state.max_total_positions,
                "min_cash_reserve": self.global_state.min_cash_reserve,
                "max_daily_loss": self.global_state.max_daily_loss
            }
        }


# Global multi-market order manager instance (singleton pattern)
_multi_market_order_manager: Optional[MultiMarketOrderManager] = None
_manager_lock = asyncio.Lock()


async def get_multi_market_order_manager() -> Optional[MultiMarketOrderManager]:
    """
    Get the global multi-market order manager instance.
    
    Returns:
        MultiMarketOrderManager instance or None if not initialized
    """
    async with _manager_lock:
        return _multi_market_order_manager


async def initialize_multi_market_order_manager(
    markets: Optional[List[str]] = None,
    **kwargs
) -> MultiMarketOrderManager:
    """
    Initialize the global multi-market order manager.
    
    Args:
        markets: Markets to manage
        **kwargs: Additional manager arguments
        
    Returns:
        Initialized MultiMarketOrderManager instance
    """
    global _multi_market_order_manager
    
    async with _manager_lock:
        if _multi_market_order_manager is not None:
            logger.warning("MultiMarketOrderManager already initialized")
            return _multi_market_order_manager
        
        logger.info("Initializing global MultiMarketOrderManager...")
        
        _multi_market_order_manager = MultiMarketOrderManager(
            markets=markets,
            **kwargs
        )
        
        await _multi_market_order_manager.initialize()
        
        logger.info("✅ Global MultiMarketOrderManager initialized")
        return _multi_market_order_manager