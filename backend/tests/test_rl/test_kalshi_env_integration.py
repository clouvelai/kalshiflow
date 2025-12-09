"""
Integration tests for KalshiTradingEnv with TradingMetricsCalculator.

These tests verify that the unified trading metrics calculation works correctly
when integrated with the KalshiTradingEnv. They test the public interface rather
than internal implementation details.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from typing import Dict, Any

from kalshiflow_rl.environments.kalshi_env import KalshiTradingEnv
from kalshiflow_rl.environments.observation_space import ObservationConfig
from kalshiflow_rl.environments.action_space import ActionConfig, ActionType
from kalshiflow_rl.environments.historical_data_loader import DataLoadConfig


class TestKalshiEnvIntegration:
    """Test KalshiTradingEnv integration with TradingMetricsCalculator."""
    
    @pytest.fixture
    def env_config(self):
        """Basic environment configuration for testing."""
        return {
            'market_tickers': ['TEST-MARKET'],
            'observation_config': ObservationConfig(),
            'action_config': ActionConfig(),
            'data_config': DataLoadConfig(window_hours=1, max_data_points=10),
            'episode_config': {
                'max_steps': 100,
                'initial_cash': 10000.0,
                'early_termination': False
            },
            'reward_config': {
                'reward_type': 'pnl_based',
                'trading_fee_rate': 0.01,  # 1% fee
                'pnl_scale': 0.01,
                'action_penalty': 0.001,
                'position_penalty_scale': 0.0001,
                'drawdown_penalty': 0.01,
                'diversification_bonus': 0.005,
                'win_rate_bonus_scale': 0.02,
                'min_reward': -10.0,
                'max_reward': 10.0,
                'normalize_rewards': True
            }
        }
    
    def test_trading_fee_calculation_through_step(self, env_config):
        """Test that trading fees are correctly calculated when executing trades through step()."""
        with patch.object(KalshiTradingEnv, '_preload_data'):
            env = KalshiTradingEnv(**env_config)
            env.historical_data = env._generate_dummy_data()
            env.episode_length = 10
            
            # Reset environment
            obs, _ = env.reset()
            initial_cash = env.cash_balance
            
            # Ensure we have orderbook data for the trade (keys must be strings)
            env.current_market_states['TEST-MARKET'].orderbook_state = {
                'yes_bids': {'45': 100},
                'yes_asks': {'55': 100},
                'no_bids': {'45': 100},
                'no_asks': {'55': 100},
                'yes_mid_price': 50.0,
                'no_mid_price': 50.0
            }
            
            # Execute a buy action
            # For discrete action space: action = action_type * num_position_sizes + position_size
            # BUY_YES = 1, FIXED_SMALL = 0
            from kalshiflow_rl.environments.action_space import PositionSizing
            action = np.array([ActionType.BUY_YES.value * len(PositionSizing) + PositionSizing.FIXED_SMALL.value], dtype=np.int32)
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Check that a trade was executed
            assert len(env.trade_history) == 1
            trade = env.trade_history[0]
            
            # Verify fee calculation
            expected_fee = trade['trade_value'] * env.reward_config['trading_fee_rate']
            assert abs(trade['fee'] - expected_fee) < 0.001
            
            # Verify cash was reduced by trade value + fee
            expected_cash = initial_cash - trade['trade_value'] - trade['fee']
            assert abs(env.cash_balance - expected_cash) < 0.01
            
            # Verify position was created
            positions = env.positions
            assert 'TEST-MARKET' in positions
            assert positions['TEST-MARKET']['position_yes'] > 0
    
    def test_cost_basis_tracking_through_multiple_trades(self, env_config):
        """Test that cost basis is correctly tracked through multiple trades via step()."""
        with patch.object(KalshiTradingEnv, '_preload_data'):
            env = KalshiTradingEnv(**env_config)
            env.historical_data = env._generate_dummy_data()
            env.episode_length = 100  # Need more steps for multiple trades
            
            # Reset environment
            obs, _ = env.reset()
            
            # Set up orderbook for consistent pricing (keys must be strings)
            env.current_market_states['TEST-MARKET'].orderbook_state = {
                'yes_bids': {'55': 1000},
                'yes_asks': {'60': 1000},
                'no_bids': {'40': 1000},
                'no_asks': {'45': 1000},
                'yes_mid_price': 57.5,
                'no_mid_price': 42.5
            }
            
            # First buy: Should establish position at 60 cents
            from kalshiflow_rl.environments.action_space import PositionSizing
            action1 = np.array([ActionType.BUY_YES.value * len(PositionSizing) + PositionSizing.FIXED_SMALL.value], dtype=np.int32)
            obs, reward, terminated, truncated, info = env.step(action1)
            
            # Check first position
            positions = env.positions
            first_quantity = positions['TEST-MARKET']['position_yes']
            first_avg_cost = positions['TEST-MARKET']['avg_cost_yes']
            assert first_quantity > 0
            assert abs(first_avg_cost - 0.60) < 0.01  # Should buy at ask price of 60
            
            # Update orderbook for second trade at different price (keys must be strings)
            env.current_market_states['TEST-MARKET'].orderbook_state = {
                'yes_bids': {'65': 1000},
                'yes_asks': {'70': 1000},
                'no_bids': {'30': 1000},
                'no_asks': {'35': 1000},
                'yes_mid_price': 67.5,
                'no_mid_price': 32.5
            }
            
            # Second buy: Should update weighted average cost basis
            action2 = np.array([ActionType.BUY_YES.value * len(PositionSizing) + PositionSizing.FIXED_SMALL.value], dtype=np.int32)
            obs, reward, terminated, truncated, info = env.step(action2)
            
            # Check updated position
            positions = env.positions
            second_quantity = positions['TEST-MARKET']['position_yes']
            second_avg_cost = positions['TEST-MARKET']['avg_cost_yes']
            
            # Debug output if test fails
            if second_quantity <= first_quantity:
                print(f"First quantity: {first_quantity}")
                print(f"Second quantity: {second_quantity}")
                print(f"Trade history length: {len(env.trade_history)}")
                print(f"Info from second step: {info}")
            
            # Verify position increased
            assert second_quantity > first_quantity, f"Expected second_quantity ({second_quantity}) > first_quantity ({first_quantity})"
            
            # Verify weighted average cost basis
            # Should be between first price (0.60) and second price (0.70)
            assert 0.60 < second_avg_cost < 0.70
            
            # Now sell part of the position
            action3 = np.array([ActionType.SELL_YES.value * len(PositionSizing) + PositionSizing.FIXED_SMALL.value], dtype=np.int32)
            obs, reward, terminated, truncated, info = env.step(action3)
            
            # Check position after selling
            positions = env.positions
            final_quantity = positions['TEST-MARKET']['position_yes']
            final_avg_cost = positions['TEST-MARKET']['avg_cost_yes']
            
            # Position should decrease
            assert final_quantity < second_quantity
            
            # Cost basis should remain unchanged after selling
            assert abs(final_avg_cost - second_avg_cost) < 0.001
            
            # Verify P&L was calculated
            assert len(env.trade_history) == 3
            sell_trade = env.trade_history[-1]
            assert 'immediate_pnl' in sell_trade
            # Should have profit since we're selling at 65 (bid) and avg cost is < 65
            assert sell_trade['immediate_pnl'] > 0
    
    def test_position_limits_and_overselling_prevention(self, env_config):
        """Test that the system prevents overselling and respects position limits."""
        with patch.object(KalshiTradingEnv, '_preload_data'):
            env = KalshiTradingEnv(**env_config)
            env.historical_data = env._generate_dummy_data()
            env.episode_length = 100
            
            # Reset environment
            obs, _ = env.reset()
            
            # Set up orderbook (keys must be strings)
            env.current_market_states['TEST-MARKET'].orderbook_state = {
                'yes_bids': {'50': 100},  # Only 100 available
                'yes_asks': {'55': 100},  # Only 100 available
                'no_bids': {'45': 100},
                'no_asks': {'50': 100},
                'yes_mid_price': 52.5,
                'no_mid_price': 47.5
            }
            
            # Buy a small position
            from kalshiflow_rl.environments.action_space import PositionSizing
            action_buy = np.array([ActionType.BUY_YES.value * len(PositionSizing) + PositionSizing.FIXED_SMALL.value], dtype=np.int32)
            obs, reward, terminated, truncated, info = env.step(action_buy)
            
            # Check we have a position
            positions = env.positions
            buy_quantity = positions['TEST-MARKET']['position_yes']
            assert buy_quantity > 0
            
            # Try to sell more than we own (multiple times)
            for _ in range(3):
                action_sell = np.array([ActionType.SELL_YES.value * len(PositionSizing) + PositionSizing.FIXED_MEDIUM.value], dtype=np.int32)
                obs, reward, terminated, truncated, info = env.step(action_sell)
            
            # Position should never go negative
            positions = env.positions
            final_position = positions['TEST-MARKET']['position_yes']
            assert final_position >= 0
            
            # Verify metrics calculator prevented overselling
            assert env.metrics_calculator.cash_balance >= 0