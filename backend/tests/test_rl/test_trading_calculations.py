"""
Tests for trading fee and cost basis calculations in KalshiTradingEnv.

These tests ensure that:
1. Trading fees are configurable and applied correctly
2. Cost basis tracking works properly via the integrated TradingMetricsCalculator
3. Fee rate validation works correctly
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
            
            # Test fee calculation directly through metrics calculator
            trade_value = 100 * 55 / 100.0  # quantity * best_ask_price / 100
            expected_fee = trade_value * 0.002  # Using configured rate
            
            # Check that the calculator uses the configured fee rate
            actual_fee = env.metrics_calculator.calculate_trade_fee(trade_value)
            
            assert abs(actual_fee - expected_fee) < 0.001
    
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
            trade1 = env.metrics_calculator.execute_trade(
                market_ticker=market,
                side='yes',
                direction='buy',
                quantity=100,
                price_cents=60  # 60 cents
            )
            
            # Check cost basis after first trade
            positions = env.metrics_calculator.get_positions_dict()
            assert positions[market]['position_yes'] == 100
            assert positions[market]['avg_cost_yes'] == 0.60
            
            # Second buy: 50 shares at 70 cents
            trade2 = env.metrics_calculator.execute_trade(
                market_ticker=market,
                side='yes',
                direction='buy',
                quantity=50,
                price_cents=70  # 70 cents
            )
            
            # Check weighted average cost basis
            # (100 * 0.60 + 50 * 0.70) / 150 = 0.633...
            expected_avg = (100 * 0.60 + 50 * 0.70) / 150
            positions = env.metrics_calculator.get_positions_dict()
            assert positions[market]['position_yes'] == 150
            assert abs(positions[market]['avg_cost_yes'] - expected_avg) < 0.001
    
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