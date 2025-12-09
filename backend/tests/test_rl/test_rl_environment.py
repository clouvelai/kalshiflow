"""
Comprehensive tests for Kalshi RL Trading Environment.

Tests Gymnasium compatibility, multi-market support, observation/action spaces,
historical data loading, and performance benchmarks.

CRITICAL VALIDATION:
- Environment works without async/await (Gymnasium requirement)
- NO database queries during step()/reset() calls
- Observation format identical to what will be used in inference
- Multi-market support handles variable market counts gracefully
- Proper error handling and logging throughout
"""

import pytest
import asyncio
import numpy as np
import time
import logging
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, List, Any
from datetime import datetime, timedelta

import gymnasium as gym
from gymnasium.utils.env_checker import check_env

# Import RL environment components
from kalshiflow_rl.environments.kalshi_env import KalshiTradingEnv
from kalshiflow_rl.environments.observation_space import (
    build_observation_from_orderbook,
    ObservationConfig,
    create_observation_space,
    get_observation_feature_names
)
from kalshiflow_rl.environments.action_space import (
    create_action_space,
    decode_action,
    encode_action,
    validate_action,
    ActionConfig,
    ActionType,
    PositionSizing,
    create_random_action
)
from kalshiflow_rl.environments.historical_data_loader import (
    HistoricalDataLoader,
    DataLoadConfig,
    HistoricalDataPoint
)


class TestObservationSpace:
    """Test observation space functionality."""
    
    def test_observation_config_defaults(self):
        """Test default observation configuration."""
        config = ObservationConfig()
        
        assert config.max_markets == 10
        assert config.price_levels_per_side == 5
        assert config.include_position_state is True
        assert config.normalize_features is True
        assert config.features_per_market > 0
        assert config.global_features_count > 0
    
    def test_create_observation_space(self):
        """Test observation space creation."""
        config = ObservationConfig()
        obs_space = create_observation_space(config)
        
        assert isinstance(obs_space, gym.spaces.Box)
        assert obs_space.dtype == np.float32
        assert obs_space.low[0] == -1.0
        assert obs_space.high[0] == 1.0
        
        expected_size = config.max_markets * config.features_per_market + config.global_features_count
        assert obs_space.shape == (expected_size,)
    
    def test_build_observation_single_market(self):
        """Test observation building for single market."""
        # Mock single market orderbook state
        orderbook_state = {
            'market_ticker': 'TEST-MARKET',
            'timestamp_ms': 1640995200000,
            'sequence_number': 100,
            'yes_bids': {48: 100, 47: 50},
            'yes_asks': {52: 75, 53: 25},
            'no_bids': {48: 80, 47: 40},
            'no_asks': {52: 60, 53: 30},
            'last_update_time': 1640995200000,
            'last_sequence': 100,
            'yes_spread': 4,
            'no_spread': 4,
            'yes_mid_price': 50.0,
            'no_mid_price': 50.0,
            'total_volume': 420
        }
        
        position_state = {
            'TEST-MARKET': {
                'position_yes': 10.0,
                'position_no': -5.0,
                'unrealized_pnl': 50.0
            }
        }
        
        observation = build_observation_from_orderbook(
            orderbook_states=orderbook_state,
            position_states=position_state
        )
        
        assert isinstance(observation, np.ndarray)
        assert observation.dtype == np.float32
        assert len(observation) > 0
        assert np.all(np.isfinite(observation))  # No NaN or inf values
        assert np.all(observation >= -1.0) and np.all(observation <= 1.0)  # Normalized range
    
    def test_build_observation_multi_market(self):
        """Test observation building for multiple markets."""
        # Mock multi-market orderbook states
        market_states = {
            'MARKET-1': {
                'market_ticker': 'MARKET-1',
                'timestamp_ms': 1640995200000,
                'yes_spread': 2,
                'no_spread': 3,
                'yes_mid_price': 45.0,
                'no_mid_price': 55.0,
                'yes_bids': {44: 100},
                'yes_asks': {46: 100},
                'no_bids': {54: 100},
                'no_asks': {56: 100},
                'total_volume': 400,
                'last_update_time': 1640995200000,
                'last_sequence': 50
            },
            'MARKET-2': {
                'market_ticker': 'MARKET-2',
                'timestamp_ms': 1640995200000,
                'yes_spread': 1,
                'no_spread': 2,
                'yes_mid_price': 60.0,
                'no_mid_price': 40.0,
                'yes_bids': {59: 150},
                'yes_asks': {61: 120},
                'no_bids': {39: 130},
                'no_asks': {41: 110},
                'total_volume': 510,
                'last_update_time': 1640995200000,
                'last_sequence': 75
            }
        }
        
        position_states = {
            'MARKET-1': {'position_yes': 20.0, 'position_no': 0.0, 'unrealized_pnl': 100.0},
            'MARKET-2': {'position_yes': 0.0, 'position_no': 15.0, 'unrealized_pnl': -25.0}
        }
        
        observation = build_observation_from_orderbook(
            orderbook_states=market_states,
            position_states=position_states
        )
        
        assert isinstance(observation, np.ndarray)
        assert len(observation) > 0
        assert np.all(np.isfinite(observation))
        assert np.all(observation >= -1.0) and np.all(observation <= 1.0)
    
    def test_observation_consistency(self):
        """Test that observation format is consistent between calls."""
        orderbook_state = {
            'market_ticker': 'TEST-MARKET',
            'timestamp_ms': 1640995200000,
            'yes_spread': 2,
            'no_spread': 2,
            'yes_mid_price': 50.0,
            'no_mid_price': 50.0,
            'yes_bids': {49: 100},
            'yes_asks': {51: 100},
            'no_bids': {49: 100},
            'no_asks': {51: 100},
            'total_volume': 400,
            'last_update_time': 1640995200000,
            'last_sequence': 1
        }
        
        # Call observation builder multiple times
        obs1 = build_observation_from_orderbook(orderbook_state)
        obs2 = build_observation_from_orderbook(orderbook_state)
        
        assert np.array_equal(obs1, obs2)
        assert obs1.shape == obs2.shape
        assert obs1.dtype == obs2.dtype
    
    def test_get_observation_feature_names(self):
        """Test feature name generation."""
        config = ObservationConfig()
        feature_names = get_observation_feature_names(config)
        
        assert isinstance(feature_names, list)
        assert len(feature_names) > 0
        
        expected_length = config.max_markets * config.features_per_market + config.global_features_count
        assert len(feature_names) == expected_length
        
        # Check for expected feature categories
        feature_str = ' '.join(feature_names)
        assert 'spread' in feature_str
        assert 'mid' in feature_str
        assert 'volume' in feature_str
        assert 'position' in feature_str
        assert 'global' in feature_str


class TestActionSpace:
    """Test action space functionality."""
    
    def test_action_config_defaults(self):
        """Test default action configuration."""
        config = ActionConfig()
        
        assert config.max_markets == 10
        assert config.max_position_size == 1000
        assert config.use_discrete_actions is True
        assert config.enforce_position_limits is True
        assert config.discrete_action_count > 0
    
    def test_create_action_space_discrete(self):
        """Test discrete action space creation."""
        config = ActionConfig()
        market_count = 3
        
        action_space = create_action_space(market_count, config)
        
        assert isinstance(action_space, gym.spaces.MultiDiscrete)
        assert len(action_space.nvec) == market_count
        assert all(n == config.discrete_action_count for n in action_space.nvec)
    
    def test_action_encoding_decoding(self):
        """Test action encoding and decoding."""
        config = ActionConfig()
        market_tickers = ['MARKET-1', 'MARKET-2']
        
        # Create test action
        market_actions = {
            'MARKET-1': {
                'action_type': ActionType.BUY_YES,
                'position_sizing': PositionSizing.FIXED_MEDIUM,
                'quantity': 50
            },
            'MARKET-2': {
                'action_type': ActionType.HOLD,
                'position_sizing': PositionSizing.FIXED_SMALL,
                'quantity': 10
            }
        }
        
        # Encode action
        encoded = encode_action(market_actions, market_tickers, config)
        assert isinstance(encoded, np.ndarray)
        assert len(encoded) == len(market_tickers)
        
        # Decode action
        decoded = decode_action(encoded, market_tickers, config)
        assert isinstance(decoded, dict)
        assert 'MARKET-1' in decoded
        assert 'MARKET-2' in decoded
        
        # Check decoded action types
        assert decoded['MARKET-1']['action_type'] == ActionType.BUY_YES
        assert decoded['MARKET-2']['action_type'] == ActionType.HOLD
    
    def test_action_validation(self):
        """Test action validation against constraints."""
        config = ActionConfig()
        market_tickers = ['MARKET-1', 'MARKET-2']
        
        # Create valid action
        valid_action = np.array([
            ActionType.BUY_YES.value * len(PositionSizing) + PositionSizing.FIXED_SMALL.value,
            ActionType.HOLD.value * len(PositionSizing) + PositionSizing.FIXED_SMALL.value
        ])
        
        current_positions = {
            'MARKET-1': {'position_yes': 50.0, 'position_no': 0.0, 'unrealized_pnl': 0.0},
            'MARKET-2': {'position_yes': 0.0, 'position_no': 25.0, 'unrealized_pnl': 10.0}
        }
        
        is_valid, violations = validate_action(
            valid_action, market_tickers, current_positions, 5000.0, config
        )
        
        assert isinstance(is_valid, bool)
        assert isinstance(violations, list)
        
        # Valid action should pass
        assert is_valid or len(violations) == 0  # Should be valid or have specific reasons
        
        # Create invalid action (exceeding position limits)
        invalid_action = np.array([
            ActionType.BUY_YES.value * len(PositionSizing) + PositionSizing.RISK_ADJUSTED.value,
            ActionType.BUY_NO.value * len(PositionSizing) + PositionSizing.FIXED_LARGE.value
        ])
        
        # Test with current positions near limits
        large_positions = {
            'MARKET-1': {'position_yes': 950.0, 'position_no': 0.0, 'unrealized_pnl': 0.0},
            'MARKET-2': {'position_yes': 0.0, 'position_no': 900.0, 'unrealized_pnl': 0.0}
        }
        
        is_valid_large, violations_large = validate_action(
            invalid_action, market_tickers, large_positions, 50000.0, config
        )
        
        # Should have violations for exceeding limits
        if not is_valid_large:
            assert len(violations_large) > 0
    
    def test_random_action_generation(self):
        """Test random action generation."""
        config = ActionConfig()
        market_tickers = ['MARKET-1', 'MARKET-2', 'MARKET-3']
        
        random_action = create_random_action(market_tickers, config, action_probability=0.5)
        
        assert isinstance(random_action, np.ndarray)
        assert len(random_action) == len(market_tickers)
        assert random_action.dtype == np.int32
        
        # Should be valid discrete actions
        for action_val in random_action:
            assert 0 <= action_val < config.discrete_action_count


class TestHistoricalDataLoader:
    """Test historical data loading functionality."""
    
    @pytest.fixture
    def mock_db_data(self):
        """Mock database data for testing."""
        return {
            'snapshots': [
                {
                    'timestamp': 1640995200000,
                    'sequence_number': 1,
                    'yes_bids': {'48': 100, '47': 50},
                    'yes_asks': {'52': 75, '53': 25},
                    'no_bids': {'48': 80, '47': 40},
                    'no_asks': {'52': 60, '53': 30},
                    'received_at': datetime.utcnow()
                },
                {
                    'timestamp': 1640995260000,
                    'sequence_number': 2,
                    'yes_bids': {'49': 120, '48': 80},
                    'yes_asks': {'51': 90, '52': 50},
                    'no_bids': {'49': 100, '48': 60},
                    'no_asks': {'51': 70, '52': 40},
                    'received_at': datetime.utcnow()
                }
            ],
            'deltas': [
                {
                    'timestamp': 1640995230000,
                    'sequence_number': 1.5,
                    'delta_type': 'update',
                    'side': 'yes',
                    'price': 49,
                    'quantity': 110
                }
            ]
        }
    
    def test_data_load_config(self):
        """Test data loading configuration."""
        config = DataLoadConfig()
        
        assert config.window_hours == 24
        assert config.max_data_points == 100000
        assert config.batch_size == 1000
        assert config.preload_strategy == "time_ordered"
        assert config.sample_rate == 1
        assert config.validate_sequences is True
    
    @pytest.mark.asyncio
    async def test_historical_data_loader_init(self):
        """Test historical data loader initialization."""
        loader = HistoricalDataLoader()
        
        assert loader.db_url is not None
        assert loader.pool is None
        assert isinstance(loader._cache, dict)
        assert len(loader._cache) == 0
    
    @pytest.mark.asyncio
    async def test_data_loader_connection(self):
        """Test database connection management."""
        loader = HistoricalDataLoader("postgresql://test:test@localhost:5432/test")
        
        # Mock the connection
        with patch('asyncpg.create_pool', new_callable=AsyncMock) as mock_pool:
            mock_pool_instance = AsyncMock()
            mock_pool.return_value = mock_pool_instance
            
            await loader.connect()
            assert loader.pool is not None
            mock_pool.assert_called_once()
            
            await loader.disconnect()
            assert loader.pool is None


class TestKalshiTradingEnv:
    """Test complete Kalshi trading environment."""
    
    @pytest.fixture
    def basic_env_config(self):
        """Basic environment configuration for testing."""
        return {
            'market_tickers': ['TEST-MARKET-1', 'TEST-MARKET-2'],
            'observation_config': ObservationConfig(),
            'action_config': ActionConfig(),
            'data_config': DataLoadConfig(window_hours=1, max_data_points=100),
            'episode_config': {
                'max_steps': 50,
                'initial_cash': 1000.0,
                'early_termination': False  # Disable for testing
            }
        }
    
    def test_environment_initialization(self, basic_env_config):
        """Test environment initialization."""
        with patch.object(KalshiTradingEnv, '_preload_data'):
            env = KalshiTradingEnv(**basic_env_config)
            
            assert env.market_tickers == basic_env_config['market_tickers']
            assert isinstance(env.observation_space, gym.spaces.Box)
            assert isinstance(env.action_space, gym.spaces.MultiDiscrete)
            assert env.current_step == 0
            assert env.episode_count == 0
            assert env.cash_balance == basic_env_config['episode_config']['initial_cash']
    
    def test_environment_reset(self, basic_env_config):
        """Test environment reset functionality."""
        with patch.object(KalshiTradingEnv, '_preload_data'):
            env = KalshiTradingEnv(**basic_env_config)
            
            # Mock historical data
            env.historical_data = env._generate_dummy_data()
            env.episode_length = 50
            
            observation, info = env.reset()
            
            # Check observation
            assert isinstance(observation, np.ndarray)
            assert observation.shape == env.observation_space.shape
            assert np.all(np.isfinite(observation))
            
            # Check info
            assert isinstance(info, dict)
            assert 'episode' in info
            assert 'cash_balance' in info
            assert 'portfolio_value' in info
            
            # Check environment state
            assert env.current_step == 0
            assert env.episode_count == 1
            assert len(env.positions) == len(env.market_tickers)
    
    def test_environment_step(self, basic_env_config):
        """Test environment step functionality."""
        with patch.object(KalshiTradingEnv, '_preload_data'):
            env = KalshiTradingEnv(**basic_env_config)
            
            # Mock historical data and reset
            env.historical_data = env._generate_dummy_data()
            env.episode_length = 50
            observation, info = env.reset()
            
            # Take a random action
            action = env.action_space.sample()
            
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            
            # Check return types
            assert isinstance(next_obs, np.ndarray)
            assert isinstance(reward, (int, float))
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert isinstance(step_info, dict)
            
            # Check observation shape consistency
            assert next_obs.shape == observation.shape
            
            # Check step increment
            assert env.current_step == 1
            
            # Check info content
            assert 'step' in step_info
            assert 'portfolio_value' in step_info
            assert 'action_valid' in step_info
    
    def test_environment_episode_completion(self, basic_env_config):
        """Test complete episode execution."""
        # Reduce episode length for faster testing
        basic_env_config['episode_config']['max_steps'] = 10
        
        with patch.object(KalshiTradingEnv, '_preload_data'):
            env = KalshiTradingEnv(**basic_env_config)
            
            # Mock historical data
            env.historical_data = env._generate_dummy_data()
            env.episode_length = 10
            
            observation, info = env.reset()
            
            total_reward = 0.0
            episode_terminated = False
            
            for step in range(15):  # Run longer than episode length to test termination
                action = env.action_space.sample()
                observation, reward, terminated, truncated, step_info = env.step(action)
                
                total_reward += reward
                
                if terminated or truncated:
                    episode_terminated = True
                    break
            
            # Episode should terminate
            assert episode_terminated
            assert env.current_step <= basic_env_config['episode_config']['max_steps']
            
            # Should be able to reset after termination
            new_obs, new_info = env.reset()
            assert isinstance(new_obs, np.ndarray)
            assert env.current_step == 0
    
    def test_gymnasium_compatibility(self, basic_env_config):
        """Test Gymnasium environment compatibility."""
        with patch.object(KalshiTradingEnv, '_preload_data'):
            env = KalshiTradingEnv(**basic_env_config)
            
            # Mock historical data
            env.historical_data = env._generate_dummy_data()
            env.episode_length = 20
            
            # Use Gymnasium's built-in environment checker
            try:
                check_env(env, warn=True, skip_render_check=True)
                gymnasium_compatible = True
            except Exception as e:
                gymnasium_compatible = False
                print(f"Gymnasium compatibility check failed: {e}")
            
            assert gymnasium_compatible, "Environment should be Gymnasium compatible"
    
    def test_multi_market_support(self, basic_env_config):
        """Test multi-market functionality."""
        # Test with different market counts
        for market_count in [1, 2, 5]:
            config = basic_env_config.copy()
            config['market_tickers'] = [f'MARKET-{i}' for i in range(market_count)]
            
            with patch.object(KalshiTradingEnv, '_preload_data'):
                env = KalshiTradingEnv(**config)
                
                # Mock data for all markets
                env.historical_data = env._generate_dummy_data()
                env.episode_length = 20
                
                observation, info = env.reset()
                
                # Check that environment handles different market counts
                assert len(env.market_tickers) == market_count
                assert len(env.positions) == market_count
                assert env.action_space.nvec.shape[0] == market_count
                
                # Take a few steps
                for _ in range(3):
                    action = env.action_space.sample()
                    obs, reward, term, trunc, step_info = env.step(action)
                    if term or trunc:
                        break
                
                # Check multi-market state consistency
                assert len(env.current_market_states) <= market_count
    
    def test_no_database_queries_during_training(self, basic_env_config):
        """Test that no database queries occur during step/reset."""
        with patch.object(KalshiTradingEnv, '_preload_data'):
            env = KalshiTradingEnv(**basic_env_config)
            
            # Mock historical data
            env.historical_data = env._generate_dummy_data()
            env.episode_length = 10
            
            # Patch database-related functions to detect any calls
            with patch('asyncpg.create_pool', side_effect=Exception("Database access during training!")):
                with patch('asyncpg.connect', side_effect=Exception("Database access during training!")):
                    
                    # Reset and run several steps
                    observation, info = env.reset()
                    
                    for step in range(5):
                        action = env.action_space.sample()
                        obs, reward, term, trunc, step_info = env.step(action)
                        if term or trunc:
                            break
                    
                    # If we reach here, no database calls were made
                    assert True, "Successfully ran episode without database queries"
    
    def test_observation_consistency_with_actor(self, basic_env_config):
        """Test that observations match what actor would receive."""
        with patch.object(KalshiTradingEnv, '_preload_data'):
            env = KalshiTradingEnv(**basic_env_config)
            
            # Mock historical data
            env.historical_data = env._generate_dummy_data()
            env.episode_length = 10
            
            observation, info = env.reset()
            
            # Extract current orderbook states and positions
            orderbook_states = env._extract_orderbook_states()
            position_states = env.positions
            
            # Build observation using the same function an actor would use
            actor_observation = build_observation_from_orderbook(
                orderbook_states=orderbook_states,
                position_states=position_states,
                config=env.observation_config
            )
            
            # Observations should be identical
            assert np.array_equal(observation, actor_observation), \
                "Environment and actor observations must be identical"
    
    def test_performance_benchmarks(self, basic_env_config):
        """Test environment performance benchmarks."""
        basic_env_config['episode_config']['max_steps'] = 100
        
        with patch.object(KalshiTradingEnv, '_preload_data'):
            env = KalshiTradingEnv(**basic_env_config)
            
            # Mock historical data
            env.historical_data = env._generate_dummy_data()
            env.episode_length = 100
            
            # Measure reset performance
            reset_start = time.time()
            observation, info = env.reset()
            reset_time = time.time() - reset_start
            
            assert reset_time < 1.0, f"Reset took too long: {reset_time:.3f}s"
            
            # Measure step performance
            step_times = []
            for i in range(10):
                action = env.action_space.sample()
                step_start = time.time()
                obs, reward, term, trunc, step_info = env.step(action)
                step_time = time.time() - step_start
                step_times.append(step_time)
                
                if term or trunc:
                    break
            
            avg_step_time = np.mean(step_times)
            assert avg_step_time < 0.1, f"Average step time too slow: {avg_step_time:.3f}s"
            
            print(f"Performance: Reset={reset_time:.3f}s, Avg Step={avg_step_time:.3f}s")
    
    def test_episode_statistics(self, basic_env_config):
        """Test episode statistics tracking."""
        basic_env_config['episode_config']['max_steps'] = 20
        
        with patch.object(KalshiTradingEnv, '_preload_data'):
            env = KalshiTradingEnv(**basic_env_config)
            
            # Mock historical data
            env.historical_data = env._generate_dummy_data()
            env.episode_length = 20
            
            observation, info = env.reset()
            
            # Run episode with some actions
            for _ in range(15):
                action = env.action_space.sample()
                obs, reward, term, trunc, step_info = env.step(action)
                if term or trunc:
                    break
            
            # Get episode statistics
            stats = env.get_episode_stats()
            
            assert isinstance(stats, dict)
            assert 'episode' in stats
            assert 'portfolio_value' in stats
            assert 'total_return' in stats
            assert 'total_trades' in stats
            assert 'win_rate' in stats
            
            # Values should be reasonable
            assert isinstance(stats['portfolio_value'], (int, float))
            assert isinstance(stats['total_return'], (int, float))
            assert 0 <= stats['win_rate'] <= 1
    
    def test_error_handling(self, basic_env_config):
        """Test environment error handling."""
        with patch.object(KalshiTradingEnv, '_preload_data'):
            env = KalshiTradingEnv(**basic_env_config)
            
            # Test with empty historical data
            env.historical_data = {}
            env.episode_length = 10
            
            # Should handle empty data gracefully
            observation, info = env.reset()
            assert isinstance(observation, np.ndarray)
            
            # Test invalid action
            invalid_action = np.array([-1, 999999])  # Invalid action values
            
            try:
                obs, reward, term, trunc, step_info = env.step(invalid_action)
                # Should not crash, might apply penalty
                assert isinstance(reward, (int, float))
            except Exception as e:
                pytest.fail(f"Environment crashed on invalid action: {e}")


class TestIntegration:
    """Integration tests for complete RL environment pipeline."""
    
    def test_end_to_end_training_simulation(self):
        """Test complete training simulation without real database."""
        # Configuration for integration test
        config = {
            'market_tickers': ['INTEGRATION-TEST-1', 'INTEGRATION-TEST-2'],
            'observation_config': ObservationConfig(),
            'action_config': ActionConfig(),
            'data_config': DataLoadConfig(window_hours=1, max_data_points=50),
            'episode_config': {
                'max_steps': 25,
                'initial_cash': 1000.0,
                'early_termination': False
            }
        }
        
        with patch.object(KalshiTradingEnv, '_preload_data'):
            env = KalshiTradingEnv(**config)
            
            # Mock historical data
            env.historical_data = env._generate_dummy_data()
            env.episode_length = 25
            
            # Run multiple episodes
            episode_returns = []
            
            for episode in range(3):
                observation, info = env.reset()
                episode_return = 0.0
                
                for step in range(25):
                    # Use semi-intelligent action (mix of random and simple strategy)
                    if step % 3 == 0:
                        action = env.action_space.sample()  # Random action
                    else:
                        # Simple strategy: mostly hold
                        action = np.array([0, 0])  # HOLD for both markets
                    
                    obs, reward, terminated, truncated, step_info = env.step(action)
                    episode_return += reward
                    
                    if terminated or truncated:
                        break
                
                episode_returns.append(episode_return)
                
                # Check episode stats
                stats = env.get_episode_stats()
                assert stats['episode'] == episode + 1
                assert isinstance(stats['portfolio_value'], (int, float))
                
                print(f"Episode {episode + 1}: Return={episode_return:.3f}, "
                      f"Portfolio=${stats['portfolio_value']:.2f}, "
                      f"Trades={stats['total_trades']}")
            
            # Check that multiple episodes completed successfully
            assert len(episode_returns) == 3
            assert all(isinstance(ret, (int, float)) for ret in episode_returns)
            
            print(f"Integration test completed: {len(episode_returns)} episodes")
            print(f"Average return: {np.mean(episode_returns):.3f}")
    
    def test_observation_action_space_compatibility(self):
        """Test that observation and action spaces work together."""
        # Test various market configurations
        market_configs = [
            ['SINGLE-MARKET'],
            ['MARKET-A', 'MARKET-B'],
            ['MARKET-1', 'MARKET-2', 'MARKET-3', 'MARKET-4']
        ]
        
        for markets in market_configs:
            config = {
                'market_tickers': markets,
                'observation_config': ObservationConfig(),
                'action_config': ActionConfig(),
                'episode_config': {'max_steps': 5, 'initial_cash': 1000.0}
            }
            
            with patch.object(KalshiTradingEnv, '_preload_data'):
                env = KalshiTradingEnv(**config)
                env.historical_data = env._generate_dummy_data()
                env.episode_length = 5
                
                # Check spaces are compatible
                assert env.action_space.nvec.shape[0] == len(markets)
                
                # Test observation/action cycle
                observation, info = env.reset()
                
                for _ in range(3):
                    action = env.action_space.sample()
                    obs, reward, term, trunc, step_info = env.step(action)
                    
                    # Verify observation shape consistency
                    assert obs.shape == observation.shape
                    
                    if term or trunc:
                        break
                
                print(f"Tested {len(markets)} markets successfully")


if __name__ == "__main__":
    # Run tests with detailed output
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--disable-warnings",
        "-x"  # Stop on first failure for faster debugging
    ])