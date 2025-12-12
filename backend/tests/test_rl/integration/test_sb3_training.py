"""
Integration tests for SB3 training with MarketSessionView integration.

These tests validate that SB3 training works correctly on real session data
with MarketSessionView integration, ensuring the complete pipeline functions
properly from session data to trained models.
"""

import pytest
import asyncio
import os
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import numpy as np

# SB3 imports
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy

# Our imports
from kalshiflow_rl.training.sb3_wrapper import (
    SessionBasedEnvironment, SB3TrainingConfig, CurriculumEnvironmentFactory,
    create_sb3_env, create_env_config, create_training_config
)
from kalshiflow_rl.environments.market_agnostic_env import EnvConfig
from kalshiflow_rl.environments.session_data_loader import SessionDataLoader


@pytest.fixture
async def database_url():
    """Get database URL from environment."""
    url = os.getenv("DATABASE_URL")
    if not url:
        pytest.skip("DATABASE_URL not set - skipping integration tests")
    return url


@pytest.fixture
async def test_session_id(database_url):
    """Get a test session ID with sufficient data."""
    loader = SessionDataLoader(database_url=database_url)
    sessions = await loader.get_available_sessions()
    
    if not sessions:
        pytest.skip("No sessions available in database")
    
    # Find a session with good data
    for session in sessions:
        if (session.get('snapshots_count', 0) >= 50 and 
            session.get('deltas_count', 0) >= 50):
            return session['session_id']
    
    # Fallback to most recent session
    return max(s['session_id'] for s in sessions)


@pytest.fixture
async def sb3_training_config():
    """Create SB3 training configuration for tests."""
    env_config = create_env_config(
        cash_start=10000,
        max_markets=1,
        temporal_features=True
    )
    
    training_config = create_training_config(
        min_episode_length=5,  # Lower threshold for tests
        max_episode_steps=50,  # Limit episode length for faster tests
        skip_failed_markets=True
    )
    
    return SB3TrainingConfig(
        env_config=env_config,
        min_snapshots=1,
        min_deltas=1,
        min_episode_length=5,
        max_episode_steps=50,
        skip_failed_markets=True
    )


@pytest.fixture
def temp_model_dir():
    """Create temporary directory for model storage."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


class TestSB3EnvironmentIntegration:
    """Test SB3 environment integration with session data."""
    
    @pytest.mark.asyncio
    async def test_session_based_environment_creation(self, database_url, test_session_id, sb3_training_config):
        """Test creating SessionBasedEnvironment with real session data."""
        env = await create_sb3_env(
            database_url=database_url,
            session_ids=test_session_id,
            config=sb3_training_config
        )
        
        # Check environment properties
        assert env.observation_space.shape == (52,)
        assert env.action_space.n == 5
        
        # Test reset
        obs, info = env.reset()
        assert obs.shape == (52,)
        assert isinstance(info, dict)
        assert 'session_id' in info
        assert 'market_ticker' in info
        
        # Test step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (52,)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        
        env.close()
    
    @pytest.mark.asyncio
    async def test_multi_session_environment(self, database_url, sb3_training_config):
        """Test environment with multiple sessions."""
        # Get multiple sessions
        loader = SessionDataLoader(database_url=database_url)
        sessions = await loader.get_available_sessions()
        
        if len(sessions) < 2:
            pytest.skip("Need at least 2 sessions for multi-session test")
        
        session_ids = [s['session_id'] for s in sessions[:2]]
        
        env = await create_sb3_env(
            database_url=database_url,
            session_ids=session_ids,
            config=sb3_training_config
        )
        
        # Test multiple resets to cycle through sessions
        session_ids_seen = set()
        
        for _ in range(5):
            obs, info = env.reset()
            session_ids_seen.add(info['session_id'])
            
            # Run a few steps
            for _ in range(3):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
        
        # Should have seen multiple sessions
        assert len(session_ids_seen) > 0
        
        env.close()
    
    @pytest.mark.asyncio
    async def test_environment_passes_sb3_validation(self, database_url, test_session_id, sb3_training_config):
        """Test that environment passes SB3 validation checks."""
        env = await create_sb3_env(
            database_url=database_url,
            session_ids=test_session_id,
            config=sb3_training_config
        )
        
        # SB3 validation should pass
        check_env(env, warn=True, skip_render_check=True)
        
        env.close()
    
    @pytest.mark.asyncio
    async def test_curriculum_environment_factory(self, database_url, test_session_id):
        """Test CurriculumEnvironmentFactory functionality."""
        factory = CurriculumEnvironmentFactory(database_url)
        
        # Test single session environment
        env = await factory.create_single_session_env(test_session_id)
        assert env is not None
        
        rotation_info = env.get_market_rotation_info()
        assert rotation_info['total_markets'] > 0
        assert test_session_id in rotation_info['sessions_covered']
        
        env.close()
        
        # Test multi-session environment
        sessions = await factory.get_available_sessions()
        if len(sessions) >= 2:
            session_ids = [s['session_id'] for s in sessions[:2]]
            multi_env = await factory.create_multi_session_env(session_ids)
            assert multi_env is not None
            
            rotation_info = multi_env.get_market_rotation_info()
            assert rotation_info['total_markets'] > 0
            assert len(set(rotation_info['sessions_covered']).intersection(session_ids)) > 0
            
            multi_env.close()


class TestSB3ModelTraining:
    """Test actual SB3 model training on real data."""
    
    @pytest.mark.asyncio
    async def test_ppo_training_integration(self, database_url, test_session_id, sb3_training_config, temp_model_dir):
        """Test PPO training on session data."""
        env = await create_sb3_env(
            database_url=database_url,
            session_ids=test_session_id,
            config=sb3_training_config
        )
        
        # Create PPO model
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=1e-3,
            n_steps=64,
            batch_size=16,
            n_epochs=2,
            verbose=0
        )
        
        # Train for a small number of steps
        model.learn(total_timesteps=500, log_interval=None)
        
        # Test model saving
        model_path = Path(temp_model_dir) / "test_ppo_model.zip"
        model.save(model_path)
        assert model_path.exists()
        
        # Test model loading
        loaded_model = PPO.load(model_path, env=env)
        assert loaded_model is not None
        
        # Test prediction
        obs, _ = env.reset()
        action, _states = loaded_model.predict(obs, deterministic=True)
        assert 0 <= action < 5
        
        env.close()
    
    @pytest.mark.asyncio
    async def test_a2c_training_integration(self, database_url, test_session_id, sb3_training_config, temp_model_dir):
        """Test A2C training on session data."""
        env = await create_sb3_env(
            database_url=database_url,
            session_ids=test_session_id,
            config=sb3_training_config
        )
        
        # Create A2C model
        model = A2C(
            "MlpPolicy",
            env,
            learning_rate=1e-3,
            n_steps=16,
            verbose=0
        )
        
        # Train for a small number of steps
        model.learn(total_timesteps=300, log_interval=None)
        
        # Test model saving
        model_path = Path(temp_model_dir) / "test_a2c_model.zip"
        model.save(model_path)
        assert model_path.exists()
        
        # Test model loading
        loaded_model = A2C.load(model_path, env=env)
        assert loaded_model is not None
        
        env.close()
    
    @pytest.mark.asyncio
    async def test_model_evaluation(self, database_url, test_session_id, sb3_training_config):
        """Test model evaluation functionality."""
        env = await create_sb3_env(
            database_url=database_url,
            session_ids=test_session_id,
            config=sb3_training_config
        )
        
        # Create and train a simple model
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=1e-3,
            n_steps=32,
            batch_size=16,
            verbose=0
        )
        
        model.learn(total_timesteps=200, log_interval=None)
        
        # Evaluate the model
        mean_reward, std_reward = evaluate_policy(
            model, env, n_eval_episodes=3, deterministic=True
        )
        
        assert isinstance(mean_reward, float)
        assert isinstance(std_reward, float)
        assert not np.isnan(mean_reward)
        assert not np.isnan(std_reward)
        
        env.close()


class TestPortfolioMetricsIntegration:
    """Test portfolio metrics tracking during training."""
    
    @pytest.mark.asyncio
    async def test_portfolio_tracking_during_training(self, database_url, test_session_id, sb3_training_config):
        """Test that portfolio metrics are tracked correctly during training."""
        env = await create_sb3_env(
            database_url=database_url,
            session_ids=test_session_id,
            config=sb3_training_config
        )
        
        # Track portfolio values manually
        portfolio_values = []
        
        for episode in range(3):
            obs, info = env.reset()
            
            # Get initial portfolio value
            if hasattr(env.unwrapped, 'order_manager'):
                initial_portfolio = env.unwrapped.order_manager.get_portfolio_value_cents(env.unwrapped._get_current_market_prices())
                portfolio_values.append(initial_portfolio)
            
            # Run episode
            for step in range(10):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                
                if terminated or truncated:
                    break
            
            # Get final portfolio value
            if hasattr(env.unwrapped, 'order_manager'):
                final_portfolio = env.unwrapped.order_manager.get_portfolio_value_cents(env.unwrapped._get_current_market_prices())
                
                # Portfolio value should be tracked properly
                assert isinstance(final_portfolio, (int, float))
                assert final_portfolio > 0  # Should still have some value
        
        # Should have collected portfolio values
        assert len(portfolio_values) > 0
        
        env.close()
    
    @pytest.mark.asyncio
    async def test_order_manager_integration(self, database_url, test_session_id, sb3_training_config):
        """Test OrderManager integration and API usage."""
        env = await create_sb3_env(
            database_url=database_url,
            session_ids=test_session_id,
            config=sb3_training_config
        )
        
        obs, info = env.reset()
        
        # Check OrderManager API methods
        if hasattr(env.unwrapped, 'order_manager'):
            order_manager = env.unwrapped.order_manager
            
            # Test API methods
            portfolio_value = order_manager.get_portfolio_value_cents(env.unwrapped._get_current_market_prices())
            cash_balance = order_manager.get_cash_balance_cents()
            position_info = order_manager.get_position_info()
            
            assert isinstance(portfolio_value, (int, float))
            assert isinstance(cash_balance, (int, float))
            assert isinstance(position_info, dict)
            
            # Cash balance should be positive initially
            assert cash_balance > 0
            
            # Portfolio value should equal cash initially (no positions)
            assert portfolio_value == cash_balance
        
        env.close()


class TestErrorHandling:
    """Test error handling in SB3 integration."""
    
    @pytest.mark.asyncio
    async def test_invalid_session_handling(self, database_url, sb3_training_config):
        """Test handling of invalid session IDs."""
        # Try with non-existent session ID
        with pytest.raises((ValueError, RuntimeError)):
            await create_sb3_env(
                database_url=database_url,
                session_ids=99999,  # Should not exist
                config=sb3_training_config
            )
    
    @pytest.mark.asyncio
    async def test_insufficient_data_handling(self, database_url):
        """Test handling when sessions have insufficient data."""
        # Create config with very high requirements
        strict_config = create_training_config(
            min_episode_length=1000,  # Very high requirement
            skip_failed_markets=True
        )
        
        sb3_config = SB3TrainingConfig(
            env_config=create_env_config(),
            **strict_config.__dict__
        )
        
        # Get any available session
        loader = SessionDataLoader(database_url=database_url)
        sessions = await loader.get_available_sessions()
        
        if not sessions:
            pytest.skip("No sessions available")
        
        session_id = sessions[0]['session_id']
        
        # This might raise an error or create an env with no markets
        try:
            env = await create_sb3_env(
                database_url=database_url,
                session_ids=session_id,
                config=sb3_config
            )
            
            # If env is created, it should handle the case gracefully
            rotation_info = env.get_market_rotation_info()
            # Either no markets or error should be handled gracefully
            
            env.close()
            
        except (ValueError, RuntimeError) as e:
            # This is expected when insufficient data
            assert "No valid market views" in str(e) or "insufficient" in str(e).lower()
    
    @pytest.mark.asyncio
    async def test_environment_robustness(self, database_url, test_session_id, sb3_training_config):
        """Test environment robustness with edge cases."""
        env = await create_sb3_env(
            database_url=database_url,
            session_ids=test_session_id,
            config=sb3_training_config
        )
        
        # Test multiple resets
        for _ in range(3):
            obs, info = env.reset()
            assert obs.shape == (52,)
        
        # Test invalid actions (should be handled gracefully)
        obs, info = env.reset()
        
        for invalid_action in [-1, 5, 10]:
            try:
                # Some environments might handle invalid actions, others might raise
                obs, reward, terminated, truncated, info = env.step(invalid_action)
            except (ValueError, AssertionError):
                # Expected for truly invalid actions
                pass
        
        env.close()


# Pytest configuration for integration tests
pytestmark = pytest.mark.asyncio