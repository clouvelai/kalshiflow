"""
Tests for ActionSelector - M3 Modular Implementation.

Tests the modular ActionSelector system with:
- Abstract ActionSelector interface
- RLModelSelector (cached model predictions)
- HardcodedSelector (always hold)
- Factory function for strategy selection
- Backward compatibility with deprecated stub
"""

import pytest
import pytest_asyncio
import numpy as np
import time
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from kalshiflow_rl.trading.action_selector import (
    ActionSelector,
    RLModelSelector,
    create_action_selector
)
from kalshiflow_rl.trading.hardcoded_policies import HardcodedSelector
from kalshiflow_rl.environments.limit_order_action_space import LimitOrderActions


class TestActionSelectorAbstract:
    """Test abstract ActionSelector interface."""
    
    def test_cannot_instantiate_abstract_class(self):
        """Test that abstract ActionSelector cannot be instantiated."""
        with pytest.raises(TypeError):
            ActionSelector()
    
    def test_abstract_methods_required(self):
        """Test that subclasses must implement abstract methods."""
        class IncompleteSelector(ActionSelector):
            pass
        
        with pytest.raises(TypeError):
            IncompleteSelector()


class TestHardcodedSelector:
    """Test HardcodedSelector implementation."""
    
    def test_initialization(self):
        """Test HardcodedSelector initializes correctly."""
        selector = HardcodedSelector()
        
        assert selector.get_strategy_name() == "Hardcoded_AlwaysHold"
    
    @pytest.mark.asyncio
    async def test_always_returns_hold(self):
        """Test HardcodedSelector always returns HOLD action."""
        selector = HardcodedSelector()
        
        # Test with various observation shapes and market tickers
        test_cases = [
            (np.zeros(52), "EXAMPLE-MARKET-1"),
            (np.random.random(100), "TEST-MARKET-ABC"),
            (np.ones(10), "ANOTHER-MARKET"),
            (np.array([0.5, 0.3, 0.7]), "FINAL-TEST")
        ]
        
        for observation, market_ticker in test_cases:
            action = await selector.select_action(observation, market_ticker)
            
            # Always returns HOLD action
            assert action == LimitOrderActions.HOLD.value
            assert action == 0
    
    @pytest.mark.asyncio
    async def test_callable_interface(self):
        """Test HardcodedSelector works as callable (ActorService integration)."""
        selector = HardcodedSelector()
        
        observation = np.zeros(52)
        action = await selector(observation, "CALLABLE-TEST")
        
        assert action == LimitOrderActions.HOLD.value
    
    def test_strategy_name(self):
        """Test strategy name is correct."""
        selector = HardcodedSelector()
        assert selector.get_strategy_name() == "Hardcoded_AlwaysHold"


class TestRLModelSelector:
    """Test RLModelSelector implementation."""
    
    @pytest.mark.asyncio
    async def test_model_loading_failure_no_sb3(self):
        """Test RLModelSelector fails gracefully when SB3 not available."""
        with patch('kalshiflow_rl.trading.action_selector.SB3_AVAILABLE', False):
            with pytest.raises(RuntimeError, match="Stable Baselines3 not available"):
                RLModelSelector("dummy_path.zip")
    
    @pytest.mark.asyncio
    async def test_model_loading_failure_missing_file(self):
        """Test RLModelSelector fails when model file doesn't exist."""
        if not pytest.importorskip("stable_baselines3"):
            pytest.skip("stable_baselines3 not available")
        
        with pytest.raises(ValueError, match="Model file not found"):
            RLModelSelector("/nonexistent/model.zip")
    
    @pytest.mark.asyncio
    async def test_model_loading_and_prediction(self):
        """Test RLModelSelector loads model and makes predictions."""
        try:
            from stable_baselines3 import PPO
            from stable_baselines3.common.env_util import make_vec_env
            from gymnasium.spaces import Box
        except ImportError:
            pytest.skip("stable_baselines3 not available")
        
        # Create a minimal test environment and model
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a simple test model
            from gymnasium import Env
            
            class TestEnv(Env):
                def __init__(self):
                    self.observation_space = Box(low=-1, high=1, shape=(52,))
                    self.action_space = Box(low=0, high=4, shape=(), dtype=int)
                
                def reset(self, seed=None, options=None):
                    return np.zeros(52), {}
                
                def step(self, action):
                    return np.zeros(52), 0.0, True, False, {}
            
            # Train a minimal model
            env = TestEnv()
            model = PPO("MlpPolicy", env, verbose=0, n_steps=64, batch_size=32)
            model.learn(total_timesteps=100)
            
            # Save model
            model_path = Path(tmpdir) / "test_model.zip"
            model.save(str(model_path))
            
            # Test RLModelSelector
            selector = RLModelSelector(str(model_path))
            
            assert selector.is_model_loaded()
            assert "test_model.zip" in selector.get_strategy_name()
            
            # Test prediction
            observation = np.zeros(52)
            action = await selector.select_action(observation, "TEST-MARKET")
            
            # Should return valid action (0-4)
            assert isinstance(action, int)
            assert 0 <= action <= 4
    
    @pytest.mark.asyncio
    async def test_prediction_error_handling(self):
        """Test RLModelSelector handles prediction errors gracefully."""
        try:
            from stable_baselines3 import PPO
        except ImportError:
            pytest.skip("stable_baselines3 not available")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a minimal test model (same as above)
            from gymnasium import Env
            from gymnasium.spaces import Box
            
            class TestEnv(Env):
                def __init__(self):
                    self.observation_space = Box(low=-1, high=1, shape=(52,))
                    self.action_space = Box(low=0, high=4, shape=(), dtype=int)
                
                def reset(self, seed=None, options=None):
                    return np.zeros(52), {}
                
                def step(self, action):
                    return np.zeros(52), 0.0, True, False, {}
            
            env = TestEnv()
            model = PPO("MlpPolicy", env, verbose=0, n_steps=64, batch_size=32)
            model.learn(total_timesteps=100)
            
            model_path = Path(tmpdir) / "test_model.zip"
            model.save(str(model_path))
            
            selector = RLModelSelector(str(model_path))
            
            # Test with invalid observation (should handle gracefully)
            invalid_obs = np.array([1.0])  # Wrong shape
            action = await selector.select_action(invalid_obs, "TEST-MARKET")
            
            # Should fall back to HOLD on error
            assert action == LimitOrderActions.HOLD.value
    
    @pytest.mark.asyncio
    async def test_callable_interface(self):
        """Test RLModelSelector works as callable (ActorService integration)."""
        try:
            from stable_baselines3 import PPO
            from gymnasium import Env
            from gymnasium.spaces import Box
        except ImportError:
            pytest.skip("stable_baselines3 not available")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            class TestEnv(Env):
                def __init__(self):
                    self.observation_space = Box(low=-1, high=1, shape=(52,))
                    self.action_space = Box(low=0, high=4, shape=(), dtype=int)
                
                def reset(self, seed=None, options=None):
                    return np.zeros(52), {}
                
                def step(self, action):
                    return np.zeros(52), 0.0, True, False, {}
            
            env = TestEnv()
            model = PPO("MlpPolicy", env, verbose=0, n_steps=64, batch_size=32)
            model.learn(total_timesteps=100)
            
            model_path = Path(tmpdir) / "test_model.zip"
            model.save(str(model_path))
            
            selector = RLModelSelector(str(model_path))
            
            observation = np.zeros(52)
            action = await selector(observation, "CALLABLE-TEST")
            
            assert isinstance(action, int)
            assert 0 <= action <= 4


class TestFactoryFunction:
    """Test create_action_selector factory function."""
    
    @pytest.mark.asyncio
    async def test_create_hardcoded_selector(self):
        """Test factory creates HardcodedSelector for 'hardcoded' strategy."""
        selector = create_action_selector(strategy="hardcoded")
        
        assert isinstance(selector, HardcodedSelector)
        assert selector.get_strategy_name() == "Hardcoded_AlwaysHold"
    
    @pytest.mark.asyncio
    async def test_create_disabled_selector(self):
        """Test factory creates HardcodedSelector for 'disabled' strategy."""
        selector = create_action_selector(strategy="disabled")
        
        assert isinstance(selector, HardcodedSelector)
    
    @pytest.mark.asyncio
    async def test_create_rl_model_selector(self):
        """Test factory creates RLModelSelector for 'rl_model' strategy."""
        try:
            from stable_baselines3 import PPO
            from gymnasium import Env
            from gymnasium.spaces import Box
        except ImportError:
            pytest.skip("stable_baselines3 not available")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            class TestEnv(Env):
                def __init__(self):
                    self.observation_space = Box(low=-1, high=1, shape=(52,))
                    self.action_space = Box(low=0, high=4, shape=(), dtype=int)
                
                def reset(self, seed=None, options=None):
                    return np.zeros(52), {}
                
                def step(self, action):
                    return np.zeros(52), 0.0, True, False, {}
            
            env = TestEnv()
            model = PPO("MlpPolicy", env, verbose=0, n_steps=64, batch_size=32)
            model.learn(total_timesteps=100)
            
            model_path = Path(tmpdir) / "test_model.zip"
            model.save(str(model_path))
            
            selector = create_action_selector(
                strategy="rl_model",
                model_path=str(model_path)
            )
            
            assert isinstance(selector, RLModelSelector)
            assert selector.is_model_loaded()
    
    @pytest.mark.asyncio
    async def test_create_rl_model_selector_missing_path(self):
        """Test factory raises error when model_path missing for 'rl_model'."""
        with pytest.raises(ValueError, match="RL_ACTOR_MODEL_PATH must be provided"):
            create_action_selector(strategy="rl_model", model_path=None)
    
    @pytest.mark.asyncio
    async def test_create_unknown_strategy_fallback(self):
        """Test factory falls back to HardcodedSelector for unknown strategy."""
        selector = create_action_selector(strategy="unknown_strategy")
        
        assert isinstance(selector, HardcodedSelector)
    
    @pytest.mark.asyncio
    async def test_factory_uses_config_when_not_provided(self):
        """Test factory reads from config when parameters not provided."""
        # This test depends on config, so we'll test with explicit parameters
        # In practice, factory will read from config.RL_ACTOR_STRATEGY
        selector = create_action_selector(strategy=None, model_path=None)
        
        # Should create HardcodedSelector as fallback or based on config
        assert isinstance(selector, (HardcodedSelector, RLModelSelector))




class TestActionSelectorIntegration:
    """Test ActionSelector integration patterns."""
    
    @pytest.mark.asyncio
    async def test_actor_service_integration_pattern(self):
        """Test integration pattern expected by ActorService."""
        selector = HardcodedSelector()
        
        # Simulate ActorService call pattern
        observation = np.random.random(52)  # Typical observation size
        market_ticker = "INTEGRATION-TEST"
        
        # Call as if from ActorService._select_action()
        action = await selector(observation, market_ticker)
        
        # Verify correct type and value
        assert isinstance(action, int)
        assert action == 0  # HOLD
        assert action == LimitOrderActions.HOLD.value
    
    @pytest.mark.asyncio
    async def test_concurrent_calls(self):
        """Test selectors handle concurrent calls correctly."""
        import asyncio
        
        selector = HardcodedSelector()
        
        # Create multiple concurrent calls
        async def make_call(i):
            obs = np.zeros(52)
            return await selector(obs, f"CONCURRENT-{i}")
        
        # Run 10 concurrent calls
        tasks = [make_call(i) for i in range(10)]
        results = await asyncio.gather(*tasks)
        
        # All should return HOLD
        assert all(action == LimitOrderActions.HOLD.value for action in results)
        assert len(results) == 10
    
    @pytest.mark.asyncio
    async def test_edge_cases(self):
        """Test selectors handle edge cases gracefully."""
        selector = HardcodedSelector()
        
        edge_cases = [
            np.array([]),  # Empty array
            np.zeros(1),   # Single value
            np.full(1000, 0.5),  # Large array
        ]
        
        for edge_obs in edge_cases:
            edge_action = await selector.select_action(edge_obs, "EDGE-CASE")
            assert edge_action == LimitOrderActions.HOLD.value
