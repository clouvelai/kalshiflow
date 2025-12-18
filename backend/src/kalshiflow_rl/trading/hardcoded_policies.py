"""
Hard-Coded Trading Policies for Kalshi Trading Actor.

This module contains rule-based/hard-coded trading policies that implement the
ActionSelector interface. These policies use deterministic or heuristic strategies
rather than machine learning models.

Current Policies:
- QuantHardCodedAction: Aggressive quantitative strategy for low-volume environments
  using microprice divergence, mean reversion, momentum, and liquidity imbalance signals
- HardcodedSelector: Equal distribution strategy (25% each trading action) for
  comprehensive trading mechanics testing

Adding a New Policy:
1. Create a new class that inherits from ActionSelector
2. Implement the required abstract methods: select_action() and get_strategy_name()
3. Add the policy to the create_action_selector() factory function in action_selector.py
4. Optionally add a get_action_statistics() method for monitoring
"""

import logging
from typing import Optional, Dict, Any
import numpy as np

from ..environments.limit_order_action_space import LimitOrderActions
from .action_selector import ActionSelector

logger = logging.getLogger("kalshiflow_rl.trading.hardcoded_policies")


class QuantHardCodedAction(ActionSelector):
    """
    Moderate quantitative hardcoded action selector.
    
    Strategy Design (BALANCED TRADING - 40-50% trade rate):
    - Accepts moderate spreads (up to 8 cents)
    - Uses microprice divergence as primary signal (strongest predictor)
    - Mean reversion: Buy < 40¢, Sell > 60¢
    - Momentum following during activity bursts
    - Liquidity imbalance exploitation
    - Holds when spreads are wide (>8 cents) or signals are weak
    
    This strategy balances opportunity capture with risk management.
    """
    
    def __init__(self):
        """Initialize moderate quant selector with trading statistics tracking."""
        self.action_count = 0
        self.hold_count = 0
        self.action_history = []
        
        # Moderate thresholds for balanced trading
        self.max_acceptable_spread = 0.08  # 8 cents (moderate)
        self.mean_reversion_buy_threshold = 0.40  # Buy below 40 cents
        self.mean_reversion_sell_threshold = 0.60  # Sell above 60 cents
        self.microprice_divergence_threshold = 0.02  # 2 cent divergence triggers signal
        self.momentum_threshold = 0.3  # Moderate momentum threshold
        self.min_trade_score = 1.5  # Minimum score to execute trade (more selective)
        
        logger.info("Moderate QuantHardCodedAction initialized")
        logger.info(f"Max spread: {self.max_acceptable_spread*100:.0f}¢, "
                   f"Mean reversion: Buy<{self.mean_reversion_buy_threshold*100:.0f}¢, "
                   f"Sell>{self.mean_reversion_sell_threshold*100:.0f}¢, "
                   f"Min trade score: {self.min_trade_score}")
    
    async def select_action(
        self, 
        observation: np.ndarray, 
        market_ticker: str,
        position_info: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Select action using moderate quantitative strategy.
        
        Analyzes multiple market signals to make balanced trading decisions.
        Holds more frequently when conditions are unfavorable.
        
        Args:
            observation: 52-feature market observation
            market_ticker: Market ticker (for logging)
            position_info: Optional position information (ignored by this policy,
                          but maintained for interface compatibility)
        
        Returns:
            int: Action ID (0=HOLD, 1=BUY_YES, 2=SELL_YES, 3=BUY_NO, 4=SELL_NO)
        """
        self.action_count += 1
        
        # Extract key features from observation
        # Market features (indices 0-27)
        best_yes_bid = observation[0] if len(observation) > 0 else 0.5
        best_yes_ask = observation[1] if len(observation) > 1 else 0.5
        best_no_bid = observation[2] if len(observation) > 2 else 0.5
        best_no_ask = observation[3] if len(observation) > 3 else 0.5
        yes_mid = observation[4] if len(observation) > 4 else 0.5
        no_mid = observation[5] if len(observation) > 5 else 0.5
        
        # Spread features
        yes_spread = observation[6] if len(observation) > 6 else 0.02
        no_spread = observation[7] if len(observation) > 7 else 0.02
        
        # Advanced microstructure features
        yes_microprice = observation[14] if len(observation) > 14 else yes_mid
        no_microprice = observation[15] if len(observation) > 15 else no_mid
        volume_imbalance = observation[17] if len(observation) > 17 else 0.0
        bid_ask_depth_ratio = observation[19] if len(observation) > 19 else 0.0
        
        # Temporal features (indices 28-37)
        activity_burst = observation[32] if len(observation) > 32 else 0.0
        price_momentum = observation[34] if len(observation) > 34 else 0.0
        
        # === MODERATE TRADING LOGIC ===
        
        # 1. CHECK SPREAD CONDITIONS
        min_spread = min(yes_spread, no_spread)
        
        # Hold more often with wide spreads
        if min_spread > self.max_acceptable_spread:
            # Hold 80% of the time with wide spreads (only trade 20% of the time)
            if np.random.random() < 0.2:
                logger.info(f"Wide spread {min_spread*100:.1f}¢ but trading anyway for {market_ticker}")
                # Trade on the side with better spread
                if yes_spread < no_spread:
                    action = LimitOrderActions.BUY_YES_LIMIT.value if yes_mid < 0.5 else LimitOrderActions.SELL_YES_LIMIT.value
                else:
                    action = LimitOrderActions.BUY_NO_LIMIT.value if no_mid < 0.5 else LimitOrderActions.SELL_NO_LIMIT.value
            else:
                self.hold_count += 1
                logger.info(f"HOLD due to wide spread {min_spread*100:.1f}¢ for {market_ticker}")
                action = LimitOrderActions.HOLD.value
        
        # 2. CALCULATE TRADING SIGNALS
        else:
            # A. Microprice Divergence Signal (strongest predictor)
            yes_micro_divergence = yes_microprice - yes_mid
            no_micro_divergence = no_microprice - no_mid
            
            # B. Mean Reversion Signal
            yes_mean_reversion_buy = yes_mid < self.mean_reversion_buy_threshold
            yes_mean_reversion_sell = yes_mid > self.mean_reversion_sell_threshold
            no_mean_reversion_buy = no_mid < self.mean_reversion_buy_threshold
            no_mean_reversion_sell = no_mid > self.mean_reversion_sell_threshold
            
            # C. Momentum Signal (follow trends during high activity)
            momentum_signal = price_momentum if activity_burst > 0.5 else 0
            
            # D. Liquidity Imbalance Signal (trade against imbalance)
            liquidity_signal = -volume_imbalance  # Trade against the imbalance
            
            # === COMPOSITE SCORING SYSTEM ===
            buy_yes_score = 0.0
            sell_yes_score = 0.0
            buy_no_score = 0.0
            sell_no_score = 0.0
            
            # Microprice signals (strongest weight)
            if yes_micro_divergence > self.microprice_divergence_threshold:
                buy_yes_score += 3.0  # Microprice above mid = bullish
            elif yes_micro_divergence < -self.microprice_divergence_threshold:
                sell_yes_score += 3.0  # Microprice below mid = bearish
                
            if no_micro_divergence > self.microprice_divergence_threshold:
                buy_no_score += 3.0
            elif no_micro_divergence < -self.microprice_divergence_threshold:
                sell_no_score += 3.0
            
            # Mean reversion signals (strong weight)
            if yes_mean_reversion_buy:
                buy_yes_score += 2.5
            if yes_mean_reversion_sell:
                sell_yes_score += 2.5
            if no_mean_reversion_buy:
                buy_no_score += 2.5
            if no_mean_reversion_sell:
                sell_no_score += 2.5
            
            # Momentum signals (moderate weight during high activity)
            if activity_burst > 0.5:
                if momentum_signal > self.momentum_threshold:
                    buy_yes_score += 1.5
                    sell_no_score += 1.5
                elif momentum_signal < -self.momentum_threshold:
                    sell_yes_score += 1.5
                    buy_no_score += 1.5
            
            # Liquidity imbalance signals
            if liquidity_signal > 0.2:
                buy_yes_score += 1.0
                buy_no_score += 0.5
            elif liquidity_signal < -0.2:
                sell_yes_score += 1.0
                sell_no_score += 0.5
            
            # Spread-adjusted scoring (prefer tighter spreads)
            if yes_spread < no_spread:
                buy_yes_score *= 1.2
                sell_yes_score *= 1.2
            else:
                buy_no_score *= 1.2
                sell_no_score *= 1.2
            
            # === SELECT ACTION BASED ON SCORES ===
            scores = {
                'buy_yes': buy_yes_score,
                'sell_yes': sell_yes_score,
                'buy_no': buy_no_score,
                'sell_no': sell_no_score
            }
            
            max_score = max(scores.values())
            
            # More selective: Only trade with stronger signals
            if max_score < self.min_trade_score:
                # With weak signals, hold more often (only trade 30% of the time)
                if np.random.random() < 0.3:
                    # Random trade based on slight preferences
                    if yes_mid < 0.5:
                        action = LimitOrderActions.BUY_YES_LIMIT.value
                    elif yes_mid > 0.5:
                        action = LimitOrderActions.SELL_YES_LIMIT.value
                    else:
                        # Exactly at 0.5, trade based on spread
                        action = LimitOrderActions.BUY_YES_LIMIT.value if yes_spread < no_spread else LimitOrderActions.BUY_NO_LIMIT.value
                    logger.info(f"Weak signals but trading: action {action} for {market_ticker}")
                else:
                    self.hold_count += 1
                    logger.debug(f"HOLD due to weak signals (max_score={max_score:.2f} < {self.min_trade_score}) for {market_ticker}")
                    action = LimitOrderActions.HOLD.value
            else:
                # Execute highest scoring action
                best_action = max(scores, key=scores.get)
                action_map = {
                    'buy_yes': LimitOrderActions.BUY_YES_LIMIT.value,
                    'sell_yes': LimitOrderActions.SELL_YES_LIMIT.value,
                    'buy_no': LimitOrderActions.BUY_NO_LIMIT.value,
                    'sell_no': LimitOrderActions.SELL_NO_LIMIT.value
                }
                action = action_map[best_action]
                
                logger.info(f"Quant signal: {best_action} (score={max_score:.2f}) for {market_ticker} "
                          f"[mid={yes_mid:.2f}, micro={yes_microprice:.2f}, momentum={price_momentum:.2f}]")
        
        # Track action history
        self.action_history.append({
            'action': action,
            'market_ticker': market_ticker,
            'timestamp': self.action_count,
            'yes_mid': yes_mid,
            'yes_spread': yes_spread,
            'signals': {
                'microprice_div': yes_microprice - yes_mid if 'yes_microprice' in locals() else 0,
                'momentum': price_momentum if 'price_momentum' in locals() else 0,
                'volume_imb': volume_imbalance if 'volume_imbalance' in locals() else 0
            }
        })
        
        return action
    
    def get_strategy_name(self) -> str:
        """Return strategy name with statistics."""
        hold_rate = (self.hold_count / max(self.action_count, 1)) * 100
        return f"Moderate_Quant(HOLD={hold_rate:.1f}%_Target~50%)"
    
    def get_action_statistics(self) -> dict:
        """Get detailed statistics about action distribution and signals."""
        if not self.action_history:
            return {"total_actions": 0, "distribution": {}}
        
        total = len(self.action_history)
        action_counts = {}
        
        for entry in self.action_history:
            action = entry['action']
            action_counts[action] = action_counts.get(action, 0) + 1
        
        distribution = {
            action: (count / total * 100) for action, count in action_counts.items()
        }
        
        action_names = {
            0: "HOLD",
            1: "BUY_YES", 
            2: "SELL_YES",
            3: "BUY_NO",
            4: "SELL_NO"
        }
        
        named_distribution = {
            action_names.get(action, f"ACTION_{action}"): f"{percentage:.1f}%"
            for action, percentage in distribution.items()
        }
        
        # Calculate average signals
        recent_history = self.action_history[-100:] if len(self.action_history) > 100 else self.action_history
        avg_signals = {}
        if recent_history:
            avg_signals = {
                'avg_microprice_divergence': np.mean([e['signals']['microprice_div'] for e in recent_history]),
                'avg_momentum': np.mean([e['signals']['momentum'] for e in recent_history]),
                'avg_volume_imbalance': np.mean([e['signals']['volume_imb'] for e in recent_history])
            }
        
        return {
            "total_actions": total,
            "hold_rate": f"{(self.hold_count / total * 100):.1f}%",
            "trade_rate": f"{((total - self.hold_count) / total * 100):.1f}%",
            "distribution": named_distribution,
            "raw_counts": action_counts,
            "recent_actions": self.action_history[-10:] if len(self.action_history) >= 10 else self.action_history,
            "avg_signals": avg_signals,
            "strategy_type": "moderate_quantitative"
        }


class HardcodedSelector(ActionSelector):
    """
    Active trading hardcoded action selector for comprehensive trading mechanics testing.
    
    Strategy Distribution (NO HOLD ACTIONS):
    - 25% BUY_YES (action 1) - Test YES position building
    - 25% SELL_YES (action 2) - Test YES position closing 
    - 25% BUY_NO (action 3) - Test NO position building
    - 25% SELL_NO (action 4) - Test NO position closing
    
    This strategy forces active trading on every decision to test order submission,
    position synchronization, and fill processing across all action types without
    relying on HOLD actions. Designed for testing trading pipeline mechanics.
    """
    
    def __init__(self):
        """Initialize hardcoded selector with active trading distribution."""
        self.action_count = 0
        self.action_history = []
        logger.info("Active HardcodedSelector initialized - NO HOLD ACTIONS")
        logger.info("Strategy: 25% BUY_YES, 25% SELL_YES, 25% BUY_NO, 25% SELL_NO")
    
    async def select_action(
        self, 
        observation: np.ndarray, 
        market_ticker: str,
        position_info: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Select action using equal distribution across trading actions only.
        
        Uses simple cycling pattern to ensure equal distribution and predictable
        testing of all trading mechanics without any HOLD actions.
        
        Args:
            observation: Market observation (used for basic validity checks)
            market_ticker: Market ticker (used for logging)
            position_info: Optional position information (ignored by this policy,
                          but maintained for interface compatibility)
        
        Returns:
            int: Action ID (1-4) selected according to equal trading distribution
        """
        self.action_count += 1
        
        # Simple 4-cycle pattern for equal distribution
        # No HOLD actions - always force trading decisions
        cycle_position = self.action_count % 4
        
        # Equal distribution pattern over 4 cycles:
        # BUY_YES, SELL_YES, BUY_NO, SELL_NO (rotating)
        if cycle_position == 1:
            action = LimitOrderActions.BUY_YES_LIMIT.value
        elif cycle_position == 2:
            action = LimitOrderActions.SELL_YES_LIMIT.value
        elif cycle_position == 3:
            action = LimitOrderActions.BUY_NO_LIMIT.value
        else:  # cycle_position == 0
            action = LimitOrderActions.SELL_NO_LIMIT.value
        
        # Track action history for assessment
        self.action_history.append({
            'action': action,
            'cycle_position': cycle_position,
            'market_ticker': market_ticker,
            'timestamp': self.action_count
        })
        
        # Log all actions since they're all trading actions now
        action_names = {
            1: "BUY_YES",
            2: "SELL_YES", 
            3: "BUY_NO",
            4: "SELL_NO"
        }
        logger.info(f"Active trading action: {action_names[action]} for {market_ticker} "
                   f"(cycle {cycle_position}/4, total actions: {self.action_count})")
        
        return action
    
    def get_strategy_name(self) -> str:
        """Return strategy name with distribution info."""
        return "Active_Hardcoded(NO_HOLD_25%_Each_Trading)"
    
    def get_action_statistics(self) -> dict:
        """Get statistics about action distribution for assessment."""
        if not self.action_history:
            return {"total_actions": 0, "distribution": {}}
        
        total = len(self.action_history)
        action_counts = {}
        
        for entry in self.action_history:
            action = entry['action']
            action_counts[action] = action_counts.get(action, 0) + 1
        
        distribution = {
            action: (count / total * 100) for action, count in action_counts.items()
        }
        
        # Updated action names without HOLD (only trading actions)
        action_names = {
            1: "BUY_YES", 
            2: "SELL_YES",
            3: "BUY_NO",
            4: "SELL_NO"
        }
        
        named_distribution = {
            action_names.get(action, f"ACTION_{action}"): f"{percentage:.1f}%"
            for action, percentage in distribution.items()
        }
        
        return {
            "total_actions": total,
            "distribution": named_distribution,
            "raw_counts": action_counts,
            "recent_actions": self.action_history[-10:] if len(self.action_history) >= 10 else self.action_history,
            "strategy_type": "active_trading_no_hold"
        }


class PositionAwareSelector(ActionSelector):
    """
    Position-aware wrapper selector that prevents opening short positions.
    
    Wraps an existing ActionSelector and filters sell actions based on current positions:
    - SELL_YES (action 2): Only allowed if we have positive YES position (contracts > 0)
    - SELL_NO (action 4): Always allowed (can open new positions or close shorts)
    - Other actions: Passed through unchanged
    
    This ensures we only sell when we have matching positions to close, preventing
    accidental short positions.
    """
    
    def __init__(self, base_selector: ActionSelector):
        """
        Initialize position-aware selector.
        
        Args:
            base_selector: The underlying ActionSelector to wrap
        """
        self.base_selector = base_selector
        self.filtered_sell_yes_count = 0
        logger.info(f"PositionAwareSelector initialized, wrapping: {base_selector.get_strategy_name()}")
    
    async def select_action(
        self,
        observation: np.ndarray,
        market_ticker: str,
        position_info: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Select action with position-aware filtering of sell actions.
        
        Gets action from base selector, then filters SELL_YES if we don't have
        a positive position to close.
        
        Args:
            observation: 52-feature observation vector
            market_ticker: Market ticker (for logging)
            position_info: Position information dict with 'position' key containing
                          current contracts. Required - if None, returns HOLD to fail safe.
        
        Returns:
            int: Action ID (0=HOLD, 1=BUY_YES, 2=SELL_YES, 3=BUY_NO, 4=SELL_NO)
        """
        # Get action from base selector
        action = await self.base_selector.select_action(observation, market_ticker, position_info)
        
        # Apply position-aware filtering for sell actions
        if action == LimitOrderActions.SELL_YES_LIMIT.value:  # action 2
            # Only allow SELL_YES if we have positive YES position to close
            if position_info is None:
                # Position info is required for position-aware filtering
                logger.error(
                    f"PositionAwareSelector: position_info is None for {market_ticker}. "
                    f"Cannot determine if SELL_YES is allowed. Failing safe by returning HOLD."
                )
                return LimitOrderActions.HOLD.value
            
            current_position = position_info.get('position', 0)
            if current_position <= 0:
                # No position to close - filter out SELL_YES
                self.filtered_sell_yes_count += 1
                logger.debug(
                    f"PositionAwareSelector: Blocked SELL_YES for {market_ticker} "
                    f"(position={current_position}, need > 0). Returning HOLD."
                )
                return LimitOrderActions.HOLD.value
        
        # SELL_NO (action 4) is always allowed - it can open new positions or close shorts
        # All other actions (BUY, HOLD) pass through unchanged
        return action
    
    def get_strategy_name(self) -> str:
        """Return strategy name with base selector name."""
        base_name = self.base_selector.get_strategy_name()
        return f"PositionAware({base_name})"
    
    def get_action_statistics(self) -> dict:
        """Get statistics including filtered action count."""
        stats = {}
        if hasattr(self.base_selector, 'get_action_statistics'):
            stats = self.base_selector.get_action_statistics()
        
        stats['position_aware'] = {
            'filtered_sell_yes_count': self.filtered_sell_yes_count
        }
        return stats

