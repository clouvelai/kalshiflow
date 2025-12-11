"""
Trading Integration for RL Trading Subsystem.

This module provides integration between the demo trading client and the existing
RL infrastructure including SharedOrderbookState, trading action logging, and
trading session management.
"""

import asyncio
import logging
import time
import json
from typing import Dict, Any, Optional, List, Callable
from decimal import Decimal
from datetime import datetime

from .demo_client import KalshiDemoTradingClient, create_demo_trading_client
from .action_write_queue import ActionWriteQueue
from .trading_metrics import TradingMetricsCalculator
from ..data.database import RLDatabase
from ..data.orderbook_state import SharedOrderbookState
from ..environments.limit_order_action_space import ActionType
from ..config import config

logger = logging.getLogger("kalshiflow_rl.trading.integration")


class TradingSession:
    """
    Manages a trading session with demo account integration.
    
    Connects demo trading client with RL infrastructure for realistic
    trading that logs to the database and integrates with orderbook state.
    """
    
    def __init__(self, session_name: str, episode_id: Optional[int] = None):
        """
        Initialize trading session.
        
        Args:
            session_name: Descriptive name for the trading session
            episode_id: Optional episode ID for linking trades to training episodes
        """
        self.session_name = session_name
        self.episode_id = episode_id
        self.session_id = f"session_{session_name}_{int(time.time())}"
        
        # Core components
        self.demo_client: Optional[KalshiDemoTradingClient] = None
        self.database: Optional[RLDatabase] = None
        self.action_write_queue: Optional[ActionWriteQueue] = None
        self.orderbook_states: Dict[str, SharedOrderbookState] = {}
        
        # Unified metrics calculator for consistent P&L calculations
        self.metrics_calculator = TradingMetricsCalculator(
            reward_config={
                'trading_fee_rate': getattr(config, 'TRADING_FEE_RATE', 0.01),
                'pnl_scale': 0.01,
                'action_penalty': 0.001,
                'position_penalty_scale': 0.0001,
                'drawdown_penalty': 0.01,
                'diversification_bonus': 0.005,
                'min_reward': -10.0,
                'max_reward': 10.0,
                'normalize_rewards': False  # Don't normalize for inference
            },
            episode_config={
                'initial_cash': 10000.0  # Default starting cash
            }
        )
        
        # Session state
        self.is_active = False
        self.start_time: Optional[datetime] = None
        self.step_number = 0
        self.total_actions = 0
        
        # Callbacks for action execution events
        self.action_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        
        logger.info(f"Trading session '{session_name}' initialized")
    
    async def start(self, mode: str = "paper") -> None:
        """
        Start the trading session.
        
        Args:
            mode: Trading mode - must be "paper" for demo client
        """
        if mode != "paper":
            raise ValueError(f"Only 'paper' mode supported, got: {mode}")
        
        try:
            # Initialize database connection
            self.database = RLDatabase()
            await self.database.initialize()
            
            # Initialize demo trading client
            self.demo_client = await create_demo_trading_client(mode=mode)
            
            # Initialize action write queue
            self.action_write_queue = ActionWriteQueue(
                batch_size=config.ORDERBOOK_QUEUE_BATCH_SIZE,
                flush_interval=config.ORDERBOOK_QUEUE_FLUSH_INTERVAL,
                max_queue_size=config.ORDERBOOK_MAX_QUEUE_SIZE
            )
            await self.action_write_queue.start()
            
            # Initialize orderbook states for configured markets
            await self._initialize_orderbook_states()
            
            # Mark session as active
            self.is_active = True
            self.start_time = datetime.now()
            
            logger.info(f"Trading session '{self.session_name}' started in {mode} mode")
            
        except Exception as e:
            logger.error(f"Failed to start trading session: {e}")
            await self.stop()
            raise
    
    async def stop(self) -> None:
        """Stop the trading session and clean up resources."""
        try:
            self.is_active = False
            
            if self.demo_client:
                await self.demo_client.disconnect()
                self.demo_client = None
            
            if self.action_write_queue:
                await self.action_write_queue.stop()
                self.action_write_queue = None
            
            if self.database:
                await self.database.close()
                self.database = None
            
            # Clear orderbook states
            self.orderbook_states.clear()
            
            session_duration = None
            if self.start_time:
                session_duration = datetime.now() - self.start_time
            
            logger.info(f"Trading session '{self.session_name}' stopped. "
                       f"Duration: {session_duration}, Actions: {self.total_actions}")
            
        except Exception as e:
            logger.error(f"Error stopping trading session: {e}")
    
    async def _initialize_orderbook_states(self) -> None:
        """Initialize SharedOrderbookState for configured markets."""
        for ticker in config.RL_MARKET_TICKERS:
            orderbook_state = SharedOrderbookState(market_ticker=ticker)
            self.orderbook_states[ticker] = orderbook_state
            logger.debug(f"Initialized orderbook state for {ticker}")
    
    async def execute_trading_action(
        self,
        action_type: ActionType,
        ticker: str,
        quantity: int = 1,
        price: Optional[int] = None,
        side: str = "yes",
        observation: Optional[Dict[str, Any]] = None,
        model_confidence: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Execute a trading action through the demo client and log to database.
        
        Args:
            action_type: Type of action (ActionType enum)
            ticker: Market ticker to trade
            quantity: Number of contracts
            price: Limit price in cents (None for market orders)
            side: 'yes' or 'no' side
            observation: Market observation at action time
            model_confidence: Model confidence score [0,1]
            
        Returns:
            Dictionary containing execution results
        """
        if not self.is_active:
            raise ValueError("Trading session is not active")
        
        if not self.demo_client:
            raise ValueError("Demo client not initialized")
        
        if ticker not in self.orderbook_states:
            raise ValueError(f"Market {ticker} not configured in session")
        
        # Capture position before action
        position_before = self.metrics_calculator.get_positions_dict().get(ticker, {
            'position_yes': 0.0,
            'position_no': 0.0,
            'unrealized_pnl': 0.0
        }).copy()  # Copy to avoid mutation
        
        # Track trades for reward calculation
        trades_executed = []
        
        # Execute the action
        execution_result = await self._execute_action(
            action_type, ticker, quantity, price, side
        )
        
        # If trade was executed, update metrics calculator
        reward = 0.0
        if execution_result.get('executed', False) and action_type != ActionType.HOLD:
            # Determine direction from ActionType
            direction = 'buy' if action_type in [ActionType.BUY_YES, ActionType.BUY_NO] else 'sell'
            
            # Get execution price - must have a valid price for executed trades
            exec_price = execution_result.get('execution_price', price)
            if exec_price is None:
                logger.error(f"Executed trade missing execution price: {execution_result}")
                raise ValueError(f"Cannot record executed trade without price for {ticker}")
            
            # Execute through metrics calculator for consistent P&L tracking
            trade_result = self.metrics_calculator.execute_trade(
                market_ticker=ticker,
                side=side,
                direction=direction,
                quantity=quantity,
                price_cents=exec_price
            )
            trades_executed.append(trade_result)
            
            # Get current market prices for reward calculation
            market_prices = {}
            if ticker in self.orderbook_states:
                orderbook = await self.orderbook_states[ticker].get_snapshot()
                # Convert prices from dollars to cents for metrics calculator
                yes_mid_price = orderbook.get('yes_mid_price')
                no_mid_price = orderbook.get('no_mid_price')
                market_prices[ticker] = {
                    'yes_mid': yes_mid_price * 100 if yes_mid_price is not None else 50.0,
                    'no_mid': no_mid_price * 100 if no_mid_price is not None else 50.0
                }
            
            # Calculate reward using unified calculator
            reward = self.metrics_calculator.calculate_step_reward(trades_executed, market_prices)
        
        # Get position after action
        position_after = self.metrics_calculator.get_positions_dict().get(ticker, {
            'position_yes': 0.0,
            'position_no': 0.0,
            'unrealized_pnl': 0.0
        })
        
        # Log action to database (non-blocking via queue)
        # Convert ActionType to string for database logging
        if action_type == ActionType.HOLD:
            action_type_str = 'hold'
        elif action_type in [ActionType.BUY_YES, ActionType.BUY_NO]:
            action_type_str = f"buy_{side}"
        elif action_type in [ActionType.SELL_YES, ActionType.SELL_NO]:
            action_type_str = f"sell_{side}"
        elif action_type == ActionType.CLOSE_POSITION:
            action_type_str = 'close_position'
        else:
            action_type_str = action_type.name.lower()
        
        action_data = {
            'episode_id': self.episode_id,
            'action_timestamp_ms': int(time.time() * 1000),
            'step_number': self.step_number,
            'action_type': action_type_str,
            'price': price,
            'quantity': quantity if action_type != ActionType.HOLD else None,
            'position_before': position_before,
            'position_after': position_after,
            'reward': reward,
            'observation': observation or {},
            'model_confidence': model_confidence,
            'executed': execution_result.get('executed', False),
            'execution_price': execution_result.get('execution_price')
        }
        
        # Enqueue action for non-blocking database write
        action_logged = False
        if self.action_write_queue:
            action_logged = await self.action_write_queue.enqueue_action(action_data)
            if action_logged:
                logger.debug(f"Trading action enqueued for database write: episode={self.episode_id}, step={self.step_number}")
            else:
                logger.warning(f"Failed to enqueue trading action (queue full): episode={self.episode_id}, step={self.step_number}")
        else:
            logger.warning("Action write queue not available, action not logged")
        
        # Update step counter
        self.step_number += 1
        self.total_actions += 1
        
        # Notify callbacks
        for callback in self.action_callbacks:
            try:
                callback(action_data)
            except Exception as e:
                logger.warning(f"Action callback failed: {e}")
        
        # Return comprehensive result
        result = {
            'session_id': self.session_id,
            'step_number': self.step_number,
            'action_type': action_type.name,
            'ticker': ticker,
            'execution_result': execution_result,
            'position_before': position_before,
            'position_after': position_after,
            'reward': reward,
            'logged': action_logged
        }
        
        return result
    
    async def _execute_action(
        self,
        action_type: ActionType,
        ticker: str,
        quantity: int,
        price: Optional[int],
        side: str
    ) -> Dict[str, Any]:
        """Execute the actual trading action through demo client."""
        if action_type == ActionType.HOLD:
            return {
                'action': 'hold',
                'executed': False,
                'reason': 'Hold action - no execution'
            }
        
        try:
            if action_type in [ActionType.BUY_YES, ActionType.BUY_NO]:
                order_response = await self.demo_client.create_order(
                    ticker=ticker,
                    action="buy",
                    side=side,
                    count=quantity,
                    price=price,
                    type="limit" if price else "market"
                )
                
                return {
                    'action': 'buy',
                    'executed': 'simulated' not in str(order_response),
                    'order_id': order_response.get('order', {}).get('order_id'),
                    'execution_price': price,
                    'response': order_response
                }
            
            elif action_type in [ActionType.SELL_YES, ActionType.SELL_NO]:
                # For selling, we need to check if we have positions to sell
                order_response = await self.demo_client.create_order(
                    ticker=ticker,
                    action="sell",
                    side=side,
                    count=quantity,
                    price=price,
                    type="limit" if price else "market"
                )
                
                return {
                    'action': 'sell',
                    'executed': 'simulated' not in str(order_response),
                    'order_id': order_response.get('order', {}).get('order_id'),
                    'execution_price': price,
                    'response': order_response
                }
            
            elif action_type == ActionType.CLOSE_POSITION:
                # Close position by selling all holdings
                # Note: This is a simplified implementation
                # In a real scenario, we'd check current positions and close them
                return {
                    'action': 'close_position',
                    'executed': False,
                    'reason': 'Close position not fully implemented in demo mode'
                }
            
            else:
                raise ValueError(f"Unknown action type: {action_type.name}")
        
        except Exception as e:
            logger.error(f"Action execution failed: {e}")
            return {
                'action': action_type.name.lower(),
                'executed': False,
                'error': str(e)
            }
    
    async def _get_current_position(self, ticker: str) -> Dict[str, Any]:
        """Get current position for a ticker."""
        try:
            # Get positions from demo client
            positions_response = await self.demo_client.get_positions()
            positions = positions_response.get('positions', [])
            
            # Find position for this ticker
            ticker_position = None
            for pos in positions:
                if pos.get('ticker') == ticker:
                    ticker_position = pos
                    break
            
            # Get current balance
            balance = float(self.demo_client.balance)
            
            if ticker_position:
                return {
                    'ticker': ticker,
                    'yes_shares': ticker_position.get('yes_position', 0),
                    'no_shares': ticker_position.get('no_position', 0),
                    'balance': balance,
                    'total_cost': ticker_position.get('total_cost', 0),
                    'unrealized_pnl': ticker_position.get('unrealized_pnl', 0)
                }
            else:
                return {
                    'ticker': ticker,
                    'yes_shares': 0,
                    'no_shares': 0,
                    'balance': balance,
                    'total_cost': 0,
                    'unrealized_pnl': 0
                }
        
        except Exception as e:
            logger.warning(f"Failed to get position for {ticker}: {e}")
            return {
                'ticker': ticker,
                'yes_shares': 0,
                'no_shares': 0,
                'balance': float(self.demo_client.balance) if self.demo_client else 10000.0,
                'total_cost': 0,
                'unrealized_pnl': 0,
                'error': str(e)
            }
    
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of trading session."""
        duration = None
        if self.start_time:
            duration = datetime.now() - self.start_time
        
        demo_summary = {}
        if self.demo_client:
            demo_summary = self.demo_client.get_trading_summary()
        
        # Get metrics from unified calculator
        metrics_summary = self.metrics_calculator.get_metrics_summary()
        
        return {
            'session_id': self.session_id,
            'session_name': self.session_name,
            'is_active': self.is_active,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'duration': str(duration) if duration else None,
            'step_number': self.step_number,
            'total_actions': self.total_actions,
            'episode_id': self.episode_id,
            'configured_markets': list(self.orderbook_states.keys()),
            'demo_account': demo_summary,
            'metrics': metrics_summary  # Includes P&L, positions, etc.
        }
    
    def add_action_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Add a callback to be notified when actions are executed."""
        self.action_callbacks.append(callback)
    
    def remove_action_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Remove an action callback."""
        if callback in self.action_callbacks:
            self.action_callbacks.remove(callback)


class TradingSessionManager:
    """
    Manager for multiple trading sessions.
    
    Provides session lifecycle management, coordination with orderbook states,
    and integration with the broader RL infrastructure.
    """
    
    def __init__(self):
        """Initialize session manager."""
        self.active_sessions: Dict[str, TradingSession] = {}
        self.session_history: List[Dict[str, Any]] = []
    
    async def create_session(
        self,
        session_name: str,
        episode_id: Optional[int] = None
    ) -> TradingSession:
        """Create a new trading session."""
        if session_name in self.active_sessions:
            raise ValueError(f"Session '{session_name}' already exists")
        
        session = TradingSession(session_name=session_name, episode_id=episode_id)
        self.active_sessions[session_name] = session
        
        logger.info(f"Created trading session: {session_name}")
        return session
    
    async def start_session(self, session_name: str, mode: str = "paper") -> None:
        """Start a trading session."""
        if session_name not in self.active_sessions:
            raise ValueError(f"Session '{session_name}' does not exist")
        
        session = self.active_sessions[session_name]
        await session.start(mode=mode)
    
    async def stop_session(self, session_name: str) -> Dict[str, Any]:
        """Stop a trading session and return summary."""
        if session_name not in self.active_sessions:
            raise ValueError(f"Session '{session_name}' does not exist")
        
        session = self.active_sessions[session_name]
        summary = session.get_session_summary()
        
        await session.stop()
        del self.active_sessions[session_name]
        
        # Add to history
        self.session_history.append(summary)
        
        logger.info(f"Stopped trading session: {session_name}")
        return summary
    
    async def stop_all_sessions(self) -> List[Dict[str, Any]]:
        """Stop all active trading sessions."""
        summaries = []
        
        for session_name in list(self.active_sessions.keys()):
            summary = await self.stop_session(session_name)
            summaries.append(summary)
        
        logger.info(f"Stopped {len(summaries)} trading sessions")
        return summaries
    
    def get_active_sessions(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of all active sessions."""
        return {
            name: session.get_session_summary()
            for name, session in self.active_sessions.items()
        }
    
    def get_session_history(self) -> List[Dict[str, Any]]:
        """Get history of completed sessions."""
        return self.session_history.copy()


# Global session manager instance
session_manager = TradingSessionManager()


# Convenience functions for easy integration
async def create_trading_session(
    session_name: str,
    episode_id: Optional[int] = None,
    auto_start: bool = True
) -> TradingSession:
    """
    Create and optionally start a trading session.
    
    Args:
        session_name: Name for the session
        episode_id: Optional episode ID for linking to training
        auto_start: Whether to automatically start the session
        
    Returns:
        TradingSession instance
    """
    session = await session_manager.create_session(session_name, episode_id)
    
    if auto_start:
        await session_manager.start_session(session_name, mode="paper")
    
    return session


async def execute_trade(
    session_name: str,
    action_type: ActionType,
    ticker: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Execute a trade in the specified session.
    
    Args:
        session_name: Name of the trading session
        action_type: Type of action (ActionType enum)
        ticker: Market ticker
        **kwargs: Additional arguments for execute_trading_action
        
    Returns:
        Execution result dictionary
    """
    if session_name not in session_manager.active_sessions:
        raise ValueError(f"Session '{session_name}' does not exist or is not active")
    
    session = session_manager.active_sessions[session_name]
    return await session.execute_trading_action(action_type, ticker, **kwargs)


# Export key components
__all__ = [
    'TradingSession',
    'TradingSessionManager', 
    'session_manager',
    'create_trading_session',
    'execute_trade'
]