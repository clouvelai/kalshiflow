"""
Tests for trading fee and cost basis calculations in KalshiTradingEnv.

These tests ensure that:
1. Trading fees are configurable and applied correctly
2. Cost basis tracking works properly for both yes/no positions
3. P&L calculations use actual cost basis, not hardcoded values
4. Validation prevents unrealistic fee rates
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from typing import Dict, Any

from kalshiflow_rl.environments.kalshi_env import KalshiTradingEnv
from kalshiflow_rl.environments.observation_space import ObservationConfig
from kalshiflow_rl.environments.action_space import ActionConfig, ActionType
from kalshiflow_rl.environments.historical_data_loader import DataLoadConfig


class TestTradingCalculations:
    """Test trading fee and cost basis calculations."""
    
    @pytest.fixture
    def env_config(self):
        """Basic environment configuration for testing."""
        return {
            'market_tickers': ['TEST-MARKET'],
            'observation_config': ObservationConfig(),
            'action_config': ActionConfig(),
            'data_config': DataLoadConfig(window_hours=1, max_data_points=10),
            'episode_config': {
                'max_steps': 10,
                'initial_cash': 10000.0,
                'early_termination': False
            }
        }
    
    def test_configurable_fee_rate(self, env_config):
        """Test that trading fee rate is configurable."""
        # Test default fee rate
        with patch.object(KalshiTradingEnv, '_preload_data'):
            env = KalshiTradingEnv(**env_config)
            assert 'trading_fee_rate' in env.reward_config
            assert env.reward_config['trading_fee_rate'] == 0.01  # 1% default
        
        # Test custom fee rate
        custom_reward_config = {'trading_fee_rate': 0.005}  # 0.5%
        env_config['reward_config'] = custom_reward_config
        
        with patch.object(KalshiTradingEnv, '_preload_data'):
            env = KalshiTradingEnv(**env_config)
            assert env.reward_config['trading_fee_rate'] == 0.005
    
    def test_fee_calculation_uses_config(self, env_config):
        """Test that fee calculation uses the configured rate."""
        custom_reward_config = {'trading_fee_rate': 0.002}  # 0.2%
        env_config['reward_config'] = custom_reward_config
        
        with patch.object(KalshiTradingEnv, '_preload_data'):
            env = KalshiTradingEnv(**env_config)
            env.historical_data = env._generate_dummy_data()
            env.episode_length = 10
            
            # Setup market state for trade simulation
            market_state = {
                'market_ticker': 'TEST-MARKET',
                'timestamp_ms': 1640995200000,
                'yes_bids': {'45': 100},
                'yes_asks': {'55': 100},
                'no_bids': {'45': 100},
                'no_asks': {'55': 100}
            }
            
            # Simulate a trade
            trade = env._simulate_trade_execution(
                market_ticker='TEST-MARKET',
                action_type=ActionType.BUY_YES,
                quantity=100,
                market_state=market_state
            )
            
            # Check fee calculation
            trade_value = 100 * 55 / 100.0  # quantity * best_ask_price / 100
            expected_fee = trade_value * 0.002  # Using configured rate
            
            assert trade is not None
            assert abs(trade['fee'] - expected_fee) < 0.001
    
    def test_cost_basis_tracking_initialization(self, env_config):
        """Test that positions track cost basis properly."""
        with patch.object(KalshiTradingEnv, '_preload_data'):
            env = KalshiTradingEnv(**env_config)
            env.historical_data = env._generate_dummy_data()
            env.episode_length = 10
            
            # Reset environment
            env.reset()
            
            # Check position structure includes cost basis
            for market in env.positions:
                position = env.positions[market]
                assert 'position_yes' in position
                assert 'position_no' in position
                assert 'avg_cost_yes' in position
                assert 'avg_cost_no' in position
                assert position['avg_cost_yes'] == 0.0  # No position yet
                assert position['avg_cost_no'] == 0.0
    
    def test_cost_basis_update_on_buy(self, env_config):
        """Test that cost basis updates correctly when buying."""
        with patch.object(KalshiTradingEnv, '_preload_data'):
            env = KalshiTradingEnv(**env_config)
            env.historical_data = env._generate_dummy_data()
            env.episode_length = 10
            env.reset()
            
            market = 'TEST-MARKET'
            
            # First buy: 100 shares at 60 cents
            trade1 = {
                'market_ticker': market,
                'side': 'yes',
                'direction': 'buy',
                'quantity': 100,
                'price': 60,  # 60 cents
                'trade_value': 60.0,
                'fee': 0.6
            }
            env._update_position_from_trade(market, trade1)
            
            # Check cost basis after first trade
            assert env.positions[market]['position_yes'] == 100
            assert env.positions[market]['avg_cost_yes'] == 0.60
            
            # Second buy: 50 shares at 70 cents
            trade2 = {
                'market_ticker': market,
                'side': 'yes',
                'direction': 'buy',
                'quantity': 50,
                'price': 70,
                'trade_value': 35.0,
                'fee': 0.35
            }
            env._update_position_from_trade(market, trade2)
            
            # Check weighted average cost basis
            # (100 * 0.60 + 50 * 0.70) / 150 = 0.633...
            expected_avg = (100 * 0.60 + 50 * 0.70) / 150
            assert env.positions[market]['position_yes'] == 150
            assert abs(env.positions[market]['avg_cost_yes'] - expected_avg) < 0.001
    
    def test_cost_basis_unchanged_on_sell(self, env_config):
        """Test that cost basis doesn't change when selling."""
        with patch.object(KalshiTradingEnv, '_preload_data'):
            env = KalshiTradingEnv(**env_config)
            env.historical_data = env._generate_dummy_data()
            env.episode_length = 10
            env.reset()
            
            market = 'TEST-MARKET'
            
            # Buy position
            buy_trade = {
                'market_ticker': market,
                'side': 'yes',
                'direction': 'buy',
                'quantity': 100,
                'price': 50,
                'trade_value': 50.0,
                'fee': 0.5
            }
            env._update_position_from_trade(market, buy_trade)
            original_cost_basis = env.positions[market]['avg_cost_yes']
            
            # Sell part of position
            sell_trade = {
                'market_ticker': market,
                'side': 'yes',
                'direction': 'sell',
                'quantity': 40,
                'price': 60,
                'trade_value': 24.0,
                'fee': 0.24,
                'immediate_pnl': 4.0  # (60-50) * 40 / 100
            }
            env._update_position_from_trade(market, sell_trade)
            
            # Cost basis should remain the same
            assert env.positions[market]['position_yes'] == 60
            assert env.positions[market]['avg_cost_yes'] == original_cost_basis
    
    def test_pnl_calculation_uses_cost_basis(self, env_config):
        """Test that P&L calculations use actual cost basis, not hardcoded values."""
        with patch.object(KalshiTradingEnv, '_preload_data'):
            env = KalshiTradingEnv(**env_config)
            env.historical_data = env._generate_dummy_data()
            env.episode_length = 10
            env.reset()
            
            market = 'TEST-MARKET'
            
            # Buy at 30 cents (not the hardcoded 50 cents)
            buy_trade = {
                'market_ticker': market,
                'side': 'yes',
                'direction': 'buy',
                'quantity': 100,
                'price': 30,
                'trade_value': 30.0,
                'fee': 0.3
            }
            env._update_position_from_trade(market, buy_trade)
            
            # Calculate P&L when selling at 80 cents
            current_position = env.positions[market]
            pnl = env._calculate_trade_pnl(
                current_position=current_position,
                side='yes',
                direction='sell',
                quantity=50,
                price=80  # 80 cents
            )
            
            # P&L should be based on actual cost (30 cents), not hardcoded 50
            expected_pnl = 50 * (0.80 - 0.30)  # 50 shares * 50 cent profit
            assert abs(pnl - expected_pnl) < 0.001
    
    def test_unrealized_pnl_uses_cost_basis(self, env_config):
        """Test that unrealized P&L uses actual cost basis."""
        with patch.object(KalshiTradingEnv, '_preload_data'):
            env = KalshiTradingEnv(**env_config)
            env.historical_data = env._generate_dummy_data()
            env.episode_length = 10
            env.reset()
            
            market = 'TEST-MARKET'
            
            # Buy yes at 40 cents, no at 60 cents
            yes_trade = {
                'market_ticker': market,
                'side': 'yes',
                'direction': 'buy',
                'quantity': 100,
                'price': 40,
                'trade_value': 40.0,
                'fee': 0.4
            }
            env._update_position_from_trade(market, yes_trade)
            
            no_trade = {
                'market_ticker': market,
                'side': 'no',
                'direction': 'buy',
                'quantity': 50,
                'price': 60,
                'trade_value': 30.0,
                'fee': 0.3
            }
            env._update_position_from_trade(market, no_trade)
            
            # Set current market prices
            env.current_market_states[market] = MagicMock()
            env.current_market_states[market].orderbook_state = {
                'yes_mid_price': 70.0,  # 70 cents (up from 40)
                'no_mid_price': 30.0    # 30 cents (down from 60)
            }
            
            # Update position values
            env._update_position_values()
            
            # Check unrealized P&L calculation
            expected_yes_pnl = 100 * (0.70 - 0.40)  # 100 shares * 30 cent profit
            expected_no_pnl = 50 * (0.30 - 0.60)    # 50 shares * 30 cent loss
            expected_total_pnl = expected_yes_pnl + expected_no_pnl
            
            assert abs(env.positions[market]['unrealized_pnl'] - expected_total_pnl) < 0.001
    
    def test_fee_rate_validation(self, env_config):
        """Test that fee rate is validated to be realistic."""
        # Test invalid negative fee rate
        invalid_config = env_config.copy()
        invalid_config['reward_config'] = {'trading_fee_rate': -0.01}
        
        with patch.object(KalshiTradingEnv, '_preload_data'):
            with pytest.raises(ValueError, match="trading_fee_rate"):
                env = KalshiTradingEnv(**invalid_config)
        
        # Test unrealistically high fee rate
        high_fee_config = env_config.copy()
        high_fee_config['reward_config'] = {'trading_fee_rate': 0.5}  # 50%
        
        with patch.object(KalshiTradingEnv, '_preload_data'):
            with pytest.raises(ValueError, match="trading_fee_rate"):
                env = KalshiTradingEnv(**high_fee_config)
        
        # Test valid fee rates
        valid_rates = [0.0, 0.001, 0.01, 0.05, 0.1]
        for rate in valid_rates:
            valid_config = env_config.copy()
            valid_config['reward_config'] = {'trading_fee_rate': rate}
            
            with patch.object(KalshiTradingEnv, '_preload_data'):
                env = KalshiTradingEnv(**valid_config)
                assert env.reward_config['trading_fee_rate'] == rate
    
    def test_cost_basis_reset_on_zero_position(self, env_config):
        """Test that cost basis resets when position goes to zero."""
        with patch.object(KalshiTradingEnv, '_preload_data'):
            env = KalshiTradingEnv(**env_config)
            env.historical_data = env._generate_dummy_data()
            env.episode_length = 10
            env.reset()
            
            market = 'TEST-MARKET'
            
            # Buy position
            buy_trade = {
                'market_ticker': market,
                'side': 'yes',
                'direction': 'buy',
                'quantity': 100,
                'price': 50,
                'trade_value': 50.0,
                'fee': 0.5
            }
            env._update_position_from_trade(market, buy_trade)
            assert env.positions[market]['avg_cost_yes'] == 0.50
            
            # Sell entire position
            sell_trade = {
                'market_ticker': market,
                'side': 'yes',
                'direction': 'sell',
                'quantity': 100,
                'price': 60,
                'trade_value': 60.0,
                'fee': 0.6,
                'immediate_pnl': 10.0
            }
            env._update_position_from_trade(market, sell_trade)
            
            # Cost basis should reset to 0
            assert env.positions[market]['position_yes'] == 0
            assert env.positions[market]['avg_cost_yes'] == 0.0
            
            # New buy should establish new cost basis
            new_buy_trade = {
                'market_ticker': market,
                'side': 'yes',
                'direction': 'buy',
                'quantity': 50,
                'price': 70,
                'trade_value': 35.0,
                'fee': 0.35
            }
            env._update_position_from_trade(market, new_buy_trade)
            assert env.positions[market]['avg_cost_yes'] == 0.70


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])