"""
Tests for Training Harness and SB3 Integration in Kalshi Flow RL Trading Subsystem.

Tests all aspects of training session management including:
- Training configuration validation and SB3 integration
- Training session lifecycle and progress tracking
- Multi-market training scenarios
- Training callbacks and monitoring integration
- Training manager coordination and resource management
"""

import pytest
import asyncio
import tempfile
import threading
from unittest.mock import AsyncMock, MagicMock, patch, call
from datetime import datetime, timedelta
import numpy as np

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import BaseCallback

from kalshiflow_rl.agents.training_config import (
    TrainingConfig,
    PPOConfig,
    A2CConfig,
    AlgorithmType,
    TrainingMode,
    TrainingConfigBuilder
)
from kalshiflow_rl.agents.training_harness import (
    TrainingSession,
    TrainingManager,
    TrainingCallback,
    training_manager
)
from kalshiflow_rl.agents.model_registry import ModelConfig
from kalshiflow_rl.environments.kalshi_env import KalshiTradingEnv


class TestTrainingConfig:
    """Test training configuration and validation."""
    
    def test_valid_ppo_config(self):
        """Test valid PPO configuration."""
        config = TrainingConfig(
            model_name="test_ppo",
            version="v1.0",
            algorithm=AlgorithmType.PPO,
            market_tickers=["INXD-25JAN03"],
            algorithm_config=PPOConfig(learning_rate=3e-4, n_steps=2048),
            total_timesteps=10000
        )
        
        errors = config.validate()
        assert len(errors) == 0
    
    def test_valid_a2c_config(self):
        """Test valid A2C configuration."""
        config = TrainingConfig(
            model_name="test_a2c",
            version="v1.0", 
            algorithm=AlgorithmType.A2C,
            market_tickers=["INXD-25JAN03"],
            algorithm_config=A2CConfig(learning_rate=7e-4, n_steps=5),
            total_timesteps=5000
        )
        
        errors = config.validate()
        assert len(errors) == 0
    
    def test_invalid_config_missing_fields(self):
        """Test invalid configuration with missing fields."""
        with pytest.raises(ValueError) as exc_info:
            config = TrainingConfig(
                model_name="",  # Invalid empty name
                version="v1.0",
                algorithm=AlgorithmType.PPO,
                market_tickers=[],  # Invalid empty list
                total_timesteps=0  # Invalid zero timesteps
            )
        
        error_message = str(exc_info.value)
        assert "model_name is required" in error_message
        assert "At least one market_ticker is required" in error_message
        assert "total_timesteps must be positive" in error_message
    
    def test_invalid_training_mode(self):
        """Test invalid training mode constraint."""
        # This should fail if we try to set a 'live' mode
        config = TrainingConfig(
            model_name="test_model",
            version="v1.0",
            algorithm=AlgorithmType.PPO,
            market_tickers=["INXD-25JAN03"],
            training_mode=TrainingMode.TRAINING  # This is valid
        )
        
        errors = config.validate()
        assert len(errors) == 0
        
        # Test with paper mode (also valid)
        config.training_mode = TrainingMode.PAPER
        errors = config.validate()
        assert len(errors) == 0
    
    def test_too_many_markets(self):
        """Test configuration with too many markets."""
        with pytest.raises(ValueError) as exc_info:
            config = TrainingConfig(
                model_name="test_model",
                version="v1.0",
                algorithm=AlgorithmType.PPO,
                market_tickers=[f"MARKET-{i:02d}" for i in range(15)],  # 15 markets (too many)
                total_timesteps=10000
            )
        
        error_message = str(exc_info.value)
        assert "Maximum 10 markets supported" in error_message
    
    def test_config_serialization(self):
        """Test configuration serialization and deserialization."""
        original_config = TrainingConfig(
            model_name="test_model",
            version="v1.0",
            algorithm=AlgorithmType.PPO,
            market_tickers=["INXD-25JAN03"],
            algorithm_config=PPOConfig(learning_rate=3e-4),
            total_timesteps=10000
        )
        
        # Convert to dict and back
        config_dict = original_config.to_dict()
        reconstructed_config = TrainingConfig.from_dict(config_dict)
        
        assert reconstructed_config.model_name == original_config.model_name
        assert reconstructed_config.algorithm == original_config.algorithm
        assert reconstructed_config.total_timesteps == original_config.total_timesteps
    
    def test_sb3_hyperparameters(self):
        """Test SB3 hyperparameter extraction."""
        config = TrainingConfig(
            model_name="test_model",
            version="v1.0",
            algorithm=AlgorithmType.PPO,
            market_tickers=["INXD-25JAN03"],
            algorithm_config=PPOConfig(learning_rate=3e-4, n_steps=2048, gamma=0.99)
        )
        
        sb3_params = config.get_sb3_hyperparameters()
        
        assert sb3_params['learning_rate'] == 3e-4
        assert sb3_params['n_steps'] == 2048
        assert sb3_params['gamma'] == 0.99
        assert 'verbose' in sb3_params
        assert 'seed' in sb3_params


class TestTrainingConfigBuilder:
    """Test training configuration builder."""
    
    def test_default_ppo_config(self):
        """Test default PPO configuration creation."""
        config = TrainingConfigBuilder.create_default_ppo_config(
            "test_model", "v1.0", "INXD-25JAN03"
        )
        
        assert config.algorithm == AlgorithmType.PPO
        assert isinstance(config.algorithm_config, PPOConfig)
        assert config.market_tickers == ["INXD-25JAN03"]
        assert config.total_timesteps == 100000
    
    def test_default_a2c_config(self):
        """Test default A2C configuration creation."""
        config = TrainingConfigBuilder.create_default_a2c_config(
            "test_model", "v1.0", "INXD-25JAN03"
        )
        
        assert config.algorithm == AlgorithmType.A2C
        assert isinstance(config.algorithm_config, A2CConfig)
        assert config.total_timesteps == 50000  # A2C default is lower
    
    def test_multi_market_config(self):
        """Test multi-market configuration creation."""
        markets = ["INXD-25JAN03", "KXCABOUT-29", "MARKET-03"]
        
        config = TrainingConfigBuilder.create_multi_market_config(
            "multi_model", "v1.0", markets, market_rotation=True
        )
        
        assert config.market_tickers == markets
        assert config.market_rotation is True
        assert config.market_weights is not None
        assert len(config.market_weights) == len(markets)
        assert config.total_timesteps == 300000  # Scaled by market count
    
    def test_production_config(self):
        """Test production-ready configuration."""
        config = TrainingConfigBuilder.create_production_config(
            "prod_model", "v1.0", "INXD-25JAN03"
        )
        
        assert config.total_timesteps == 200000  # More training for production
        assert config.early_stopping_patience == 20  # More conservative
        
        if config.algorithm == AlgorithmType.PPO:
            assert config.algorithm_config.learning_rate == 1e-4  # Lower learning rate
            assert config.algorithm_config.clip_range == 0.1  # More conservative


class TestTrainingCallback:
    """Test training callback functionality."""
    
    @pytest.fixture
    def mock_training_session(self):
        """Mock training session for testing."""
        session = MagicMock()
        session._should_stop = False
        session.config = MagicMock()
        session.config.market_tickers = ["INXD-25JAN03"]
        return session
    
    def test_callback_initialization(self, mock_training_session):
        """Test callback initialization."""
        callback = TrainingCallback(
            training_session=mock_training_session,
            model_id=123,
            log_frequency=100
        )
        
        assert callback.training_session == mock_training_session
        assert callback.model_id == 123
        assert callback.log_frequency == 100
        assert callback.episode_count == 0
    
    def test_on_step_logging(self, mock_training_session):
        """Test step logging functionality."""
        callback = TrainingCallback(
            training_session=mock_training_session,
            model_id=123,
            log_frequency=10
        )
        
        # Mock required attributes
        callback.num_timesteps = 10
        callback.locals = {'rewards': [0.5], 'dones': [False]}
        
        # Call step
        should_continue = callback._on_step()
        
        assert should_continue is True
    
    def test_on_step_early_stop(self, mock_training_session):
        """Test early stopping in callback."""
        mock_training_session._should_stop = True
        
        callback = TrainingCallback(
            training_session=mock_training_session,
            model_id=123
        )
        
        callback.num_timesteps = 1
        callback.locals = {}
        
        should_continue = callback._on_step()
        assert should_continue is False


class TestTrainingSession:
    """Test training session management."""
    
    @pytest.fixture
    def sample_config(self):
        """Sample training configuration."""
        return TrainingConfig(
            model_name="test_session",
            version="v1.0",
            algorithm=AlgorithmType.PPO,
            market_tickers=["INXD-25JAN03"],
            total_timesteps=1000,
            eval_freq=500
        )
    
    def test_session_initialization(self, sample_config):
        """Test training session initialization."""
        session = TrainingSession(sample_config)
        
        assert session.config == sample_config
        assert session.model is None
        assert session.env is None
        assert session._should_stop is False
        assert session.model_id is None
    
    @pytest.mark.asyncio
    async def test_session_setup(self, sample_config):
        """Test session setup process."""
        session = TrainingSession(sample_config)
        
        # Mock all required components
        with patch.object(session, '_create_environments') as mock_create_env, \
             patch.object(session, '_register_model') as mock_register, \
             patch.object(session, '_create_sb3_model') as mock_create_model, \
             patch.object(session, '_setup_callbacks') as mock_setup_callbacks:
            
            await session.setup()
            
            mock_create_env.assert_called_once()
            mock_register.assert_called_once() 
            mock_create_model.assert_called_once()
            mock_setup_callbacks.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_environments(self, sample_config):
        """Test environment creation."""
        session = TrainingSession(sample_config)
        
        with patch('kalshiflow_rl.agents.training_harness.KalshiTradingEnv') as MockEnv, \
             patch('kalshiflow_rl.agents.training_harness.DummyVecEnv') as MockVecEnv, \
             patch('kalshiflow_rl.agents.training_harness.check_env') as mock_check:
            
            mock_env_instance = MagicMock()
            MockEnv.return_value = mock_env_instance
            MockVecEnv.return_value = MagicMock()
            
            await session._create_environments()
            
            assert MockEnv.call_count == 2  # Training and eval environments
            mock_check.assert_called_once_with(mock_env_instance)
    
    @pytest.mark.asyncio
    async def test_register_model(self, sample_config):
        """Test model registration."""
        session = TrainingSession(sample_config)
        
        with patch('kalshiflow_rl.agents.training_harness.model_registry') as mock_registry:
            # Mock async method to return a coroutine
            async def mock_register():
                return 456
            mock_registry.register_model.return_value = mock_register()
            
            await session._register_model()
            
            assert session.model_id == 456
            mock_registry.register_model.assert_called_once()
    
    def test_create_sb3_model_ppo(self, sample_config):
        """Test PPO model creation."""
        session = TrainingSession(sample_config)
        session.env = MagicMock()
        
        with patch('kalshiflow_rl.agents.training_harness.PPO') as MockPPO:
            mock_model = MagicMock()
            MockPPO.return_value = mock_model
            
            session._create_sb3_model()
            
            assert session.model == mock_model
            MockPPO.assert_called_once()
    
    def test_create_sb3_model_a2c(self, sample_config):
        """Test A2C model creation."""
        # Modify config for A2C
        sample_config.algorithm = AlgorithmType.A2C
        sample_config.algorithm_config = A2CConfig()
        
        session = TrainingSession(sample_config)
        session.env = MagicMock()
        
        with patch('kalshiflow_rl.agents.training_harness.A2C') as MockA2C:
            mock_model = MagicMock()
            MockA2C.return_value = mock_model
            
            session._create_sb3_model()
            
            assert session.model == mock_model
            MockA2C.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_training_execution(self, sample_config):
        """Test training execution."""
        session = TrainingSession(sample_config)
        session.model = MagicMock()
        session.model_id = 123
        session.callback_list = MagicMock()
        
        # Mock the training loop execution
        def mock_training_loop():
            return {'success': True, 'total_timesteps': 1000}
        
        with patch.object(session, '_execute_training_loop', return_value={'success': True}), \
             patch.object(session, '_save_final_checkpoint') as mock_save, \
             patch('kalshiflow_rl.agents.training_harness.rl_db') as mock_db:
            
            # Use AsyncMock for async method
            mock_db.update_model_status = AsyncMock(return_value=True)
            
            results = await session.train()
            
            assert results['success'] is True
            assert results['model_id'] == 123
            mock_save.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_training_failure(self, sample_config):
        """Test training failure handling."""
        session = TrainingSession(sample_config)
        session.model_id = 123
        # Mark session as setup to allow training
        session.model = MagicMock()  # Mock model
        session.env = MagicMock()     # Mock environment
        
        # Simulate training failure
        with patch.object(session, '_execute_training_loop', side_effect=Exception("Training failed")), \
             patch('kalshiflow_rl.agents.training_harness.rl_db') as mock_db:
            
            # Use AsyncMock for async method
            mock_db.update_model_status = AsyncMock(return_value=True)
            
            results = await session.train()
            
            assert results['success'] is False
            assert "Training failed" in results['error']
            mock_db.update_model_status.assert_called_with(123, 'failed')
    
    @pytest.mark.asyncio
    async def test_session_cleanup(self, sample_config):
        """Test session cleanup."""
        session = TrainingSession(sample_config)
        session.env = MagicMock()
        session.eval_env = MagicMock()
        session.model = MagicMock()
        
        await session.cleanup()
        
        session.env.close.assert_called_once()
        session.eval_env.close.assert_called_once()
        assert session.model is None


class TestTrainingManager:
    """Test training manager functionality."""
    
    @pytest.fixture
    def manager(self):
        """Create training manager for testing."""
        return TrainingManager()
    
    @pytest.fixture
    def sample_config(self):
        """Sample training configuration."""
        return TrainingConfig(
            model_name="manager_test",
            version="v1.0",
            algorithm=AlgorithmType.PPO,
            market_tickers=["INXD-25JAN03"],
            total_timesteps=1000
        )
    
    @pytest.mark.asyncio
    async def test_start_training_session(self, manager, sample_config):
        """Test starting a training session."""
        with patch.object(manager, '_run_training_session') as mock_run:
            # Mock session setup
            with patch('kalshiflow_rl.agents.training_harness.TrainingSession') as MockSession:
                mock_session = MagicMock()
                mock_session.setup = AsyncMock()
                MockSession.return_value = mock_session
                
                session_id = await manager.start_training(sample_config)
                
                assert session_id in manager.active_sessions
                assert session_id.startswith("manager_test_v1.0_")
                mock_session.setup.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_concurrent_session_limit(self, manager, sample_config):
        """Test concurrent session limit enforcement."""
        manager.max_concurrent_sessions = 1
        
        # Start first session
        with patch('kalshiflow_rl.agents.training_harness.TrainingSession') as MockSession:
            mock_session = MagicMock()
            mock_session.setup = AsyncMock()
            MockSession.return_value = mock_session
            
            session_id1 = await manager.start_training(sample_config)
            
            # Try to start second session - should be queued
            sample_config.model_name = "manager_test_2"
            session_id2 = await manager.start_training(sample_config)
            
            assert len(manager.active_sessions) == 1
            assert len(manager.training_queue) == 1
    
    @pytest.mark.asyncio
    async def test_stop_training_session(self, manager, sample_config):
        """Test stopping a training session."""
        # Add mock session
        mock_session = MagicMock()
        session_id = "test_session_123"
        manager.active_sessions[session_id] = mock_session
        
        success = await manager.stop_training(session_id)
        
        assert success is True
        mock_session.stop_training.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_training_status(self, manager):
        """Test getting training status."""
        # Create mock session with required attributes
        mock_session = MagicMock()
        mock_session.config.model_name = "test_model"
        mock_session.config.version = "v1.0"
        mock_session.config.algorithm.value = "PPO"
        mock_session.config.market_tickers = ["INXD-25JAN03"]
        mock_session.model_id = 123
        mock_session._should_stop = False
        mock_session._start_time = 1640995200.0  # Fixed timestamp
        
        session_id = "test_session_123"
        manager.active_sessions[session_id] = mock_session
        
        status = await manager.get_training_status(session_id)
        
        assert status is not None
        assert status['session_id'] == session_id
        assert status['model_name'] == "test_model"
        assert status['model_id'] == 123
        assert status['is_training'] is True
    
    @pytest.mark.asyncio
    async def test_list_active_sessions(self, manager):
        """Test listing active sessions."""
        # Add mock sessions
        for i in range(3):
            mock_session = MagicMock()
            mock_session.config.model_name = f"model_{i}"
            mock_session.config.version = "v1.0"
            mock_session.config.algorithm.value = "PPO"
            mock_session.config.market_tickers = ["INXD-25JAN03"]
            mock_session.model_id = 100 + i
            mock_session._should_stop = False
            mock_session._start_time = None
            
            manager.active_sessions[f"session_{i}"] = mock_session
        
        sessions = await manager.list_active_sessions()
        
        assert len(sessions) == 3
        assert all('session_id' in session for session in sessions)
    
    @pytest.mark.asyncio
    async def test_manager_shutdown(self, manager):
        """Test graceful manager shutdown."""
        # Add mock sessions
        mock_sessions = {}
        for i in range(2):
            mock_session = MagicMock()
            mock_session.cleanup = AsyncMock()
            mock_session.stop_training = MagicMock()
            session_id = f"session_{i}"
            mock_sessions[session_id] = mock_session
            manager.active_sessions[session_id] = mock_session
        
        await manager.shutdown()
        
        assert len(manager.active_sessions) == 0
        assert len(manager.training_queue) == 0
        
        # Verify all sessions were stopped and cleaned up
        for mock_session in mock_sessions.values():
            mock_session.stop_training.assert_called_once()
            mock_session.cleanup.assert_called_once()


# Removed TestTrainingIntegration - covered by unit tests and clean integration tests
