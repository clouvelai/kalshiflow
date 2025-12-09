"""
Tests for Training Monitoring System in Kalshi Flow RL Trading Subsystem.

Tests all aspects of training progress tracking including:
- Performance metrics calculation and tracking
- Training progress callbacks and SB3 integration
- Training session lifecycle monitoring
- Performance threshold detection and automated actions
- Database persistence of training metrics
"""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from datetime import datetime, timedelta
from collections import deque
import numpy as np

from kalshiflow_rl.agents.training_monitor import (
    PerformanceMetrics,
    TrainingMonitor,
    TrainingProgressCallback,
    create_training_monitor
)
from kalshiflow_rl.agents.session_manager import (
    SessionManager,
    SessionStatus,
    SessionState
)
from kalshiflow_rl.agents.training_config import TrainingConfig, AlgorithmType


class TestPerformanceMetrics:
    """Test PerformanceMetrics dataclass."""
    
    def test_default_initialization(self):
        """Test default metrics initialization."""
        metrics = PerformanceMetrics()
        
        assert metrics.total_episodes == 0
        assert metrics.avg_episode_reward == 0.0
        assert metrics.win_rate == 0.0
        assert metrics.steps_per_second == 0.0
        assert isinstance(metrics.last_updated, datetime)
    
    def test_metrics_serialization(self):
        """Test metrics serialization to dictionary."""
        metrics = PerformanceMetrics(
            total_episodes=100,
            avg_episode_reward=5.5,
            win_rate=0.65,
            total_trades=250
        )
        
        metrics_dict = metrics.to_dict()
        
        assert metrics_dict['total_episodes'] == 100
        assert metrics_dict['avg_episode_reward'] == 5.5
        assert metrics_dict['win_rate'] == 0.65
        assert metrics_dict['total_trades'] == 250
        assert 'last_updated' in metrics_dict
        assert isinstance(metrics_dict['last_updated'], str)  # ISO format
    
    def test_metrics_with_timestamps(self):
        """Test metrics with custom timestamps."""
        training_start = datetime.utcnow() - timedelta(hours=2)
        last_update = datetime.utcnow()
        
        metrics = PerformanceMetrics(
            training_started=training_start,
            last_updated=last_update
        )
        
        metrics_dict = metrics.to_dict()
        assert 'training_started' in metrics_dict
        assert 'last_updated' in metrics_dict


class TestTrainingMonitor:
    """Test TrainingMonitor functionality."""
    
    @pytest.fixture
    def monitor(self):
        """Create training monitor for testing."""
        return TrainingMonitor(
            model_id=123,
            update_frequency=10,
            checkpoint_frequency=50,
            max_metrics_history=100
        )
    
    def test_monitor_initialization(self, monitor):
        """Test monitor initialization."""
        assert monitor.model_id == 123
        assert monitor.update_frequency == 10
        assert monitor.checkpoint_frequency == 50
        assert monitor.max_metrics_history == 100
        assert len(monitor.episode_rewards) == 0
        assert len(monitor.metrics_history) == 0
        assert monitor._step_count == 0
    
    def test_add_episode_data(self, monitor):
        """Test adding episode data."""
        episode_stats = {
            'total_trades': 15,
            'trade_returns': [0.02, -0.01, 0.03],
            'portfolio_value': 10250.0
        }
        
        monitor.add_episode_data(
            episode_reward=10.5,
            episode_length=200,
            portfolio_return=0.025,
            episode_stats=episode_stats
        )
        
        assert len(monitor.episode_rewards) == 1
        assert monitor.episode_rewards[0] == 10.5
        assert len(monitor.episode_lengths) == 1
        assert monitor.episode_lengths[0] == 200
        assert len(monitor.episode_returns) == 1
        assert monitor.episode_returns[0] == 0.025
        assert monitor.current_metrics.total_trades == 15
        assert len(monitor.trade_returns) == 3
        assert len(monitor.portfolio_values) == 1
    
    def test_add_training_step_data(self, monitor):
        """Test adding training step data."""
        monitor.add_training_step_data(
            step=25,
            policy_loss=0.15,
            value_loss=0.08,
            entropy=0.02,
            explained_variance=0.85
        )
        
        assert monitor._step_count == 25
        assert len(monitor.recent_losses['policy_loss']) == 1
        assert monitor.recent_losses['policy_loss'][0] == 0.15
        assert monitor.current_metrics.explained_variance == 0.85
    
    @pytest.mark.asyncio
    async def test_update_frequency_trigger(self, monitor):
        """Test metrics update on frequency trigger."""
        # Add some episode data
        monitor.add_episode_data(5.0, 100, 0.01)
        monitor.add_episode_data(7.0, 150, 0.02)
        
        with patch.object(monitor, '_save_metrics_async') as mock_save:
            # Step that triggers update (multiple of update_frequency)
            monitor.add_training_step_data(step=10)
            
            # Give async task time to be created
            await asyncio.sleep(0.01)
            
        # Verify metrics were updated
        assert monitor.current_metrics.total_episodes == 2
        assert monitor.current_metrics.avg_episode_reward == 6.0
        assert monitor.current_metrics.avg_episode_length == 125.0
    
    def test_metrics_calculation(self, monitor):
        """Test performance metrics calculation."""
        # Add multiple episodes
        episode_returns = [0.05, -0.02, 0.08, 0.01, 0.03]
        for i, ret in enumerate(episode_returns):
            monitor.add_episode_data(
                episode_reward=ret * 100,
                episode_length=100 + i * 10,
                portfolio_return=ret
            )
        
        # Trigger metrics update
        monitor._update_performance_metrics()
        
        # Check calculated metrics
        assert monitor.current_metrics.total_episodes == 5
        assert abs(monitor.current_metrics.avg_portfolio_return - 0.03) < 0.001
        assert monitor.current_metrics.win_rate == 0.8  # 4 out of 5 positive returns
        
        # Check Sharpe ratio calculation
        expected_sharpe = np.mean(episode_returns) / np.std(episode_returns)
        assert abs(monitor.current_metrics.avg_sharpe_ratio - expected_sharpe) < 0.01
    
    def test_trading_metrics_calculation(self, monitor):
        """Test trading-specific metrics calculation."""
        # Add trade returns
        trade_returns = [0.02, -0.01, 0.03, -0.005, 0.015, -0.02]
        monitor.trade_returns.extend(trade_returns)
        
        # Add drawdown data
        portfolio_values = [10000, 10200, 10150, 10300, 10290, 10350, 10250]
        monitor.portfolio_values.extend(portfolio_values)
        
        # Calculate drawdown history
        for i in range(1, len(portfolio_values)):
            peak = max(portfolio_values[:i+1])
            current_drawdown = (peak - portfolio_values[i]) / peak
            monitor.drawdown_history.append(current_drawdown)
        
        monitor._update_performance_metrics()
        
        # Check trading metrics
        assert abs(monitor.current_metrics.avg_trade_return - np.mean(trade_returns)) < 0.001
        assert monitor.current_metrics.max_drawdown > 0
        
        # Check profit factor
        profits = [r for r in trade_returns if r > 0]
        losses = [abs(r) for r in trade_returns if r < 0]
        expected_profit_factor = sum(profits) / sum(losses)
        assert abs(monitor.current_metrics.profit_factor - expected_profit_factor) < 0.01
    
    def test_threshold_callback_registration(self, monitor):
        """Test threshold callback registration and triggering."""
        callback_triggered = False
        callback_metrics = None
        
        def test_callback(metrics):
            nonlocal callback_triggered, callback_metrics
            callback_triggered = True
            callback_metrics = metrics
        
        # Register callback for win rate threshold
        monitor.add_threshold_callback('win_rate', 0.6, test_callback)
        
        # Add episodes to reach threshold
        for i in range(10):
            portfolio_return = 0.01 if i < 7 else -0.01  # 70% win rate
            monitor.add_episode_data(
                episode_reward=1.0,
                episode_length=100,
                portfolio_return=portfolio_return
            )
        
        monitor._update_performance_metrics()
        
        assert callback_triggered is True
        assert callback_metrics is not None
        assert callback_metrics.win_rate >= 0.6
    
    def test_metrics_history_management(self, monitor):
        """Test metrics history size management."""
        # Re-initialize metrics_history with smaller maxlen
        monitor.metrics_history = deque(maxlen=5)
        
        # Add more metrics than max history
        for i in range(10):
            monitor.add_episode_data(1.0, 100, 0.01)
            monitor._update_performance_metrics()
        
        assert len(monitor.metrics_history) == 5  # Should be capped
        assert len(monitor.episode_rewards) <= monitor.episode_rewards.maxlen
    
    def test_get_training_summary(self, monitor):
        """Test training summary generation."""
        # Set training_started to ensure positive duration with fixed timestamp
        start_time = datetime.utcnow() - timedelta(hours=1)
        monitor.current_metrics.training_started = start_time
        
        # Mock time.time() to return consistent value for duration calculation
        with patch('kalshiflow_rl.agents.training_monitor.time.time') as mock_time:
            # Set current time to be 1 hour after start_time
            mock_time.return_value = start_time.timestamp() + 3600  # 1 hour later
            
            # Add some training data
            for i in range(50):
                monitor.add_episode_data(
                    episode_reward=5.0 + np.random.normal(0, 1),
                    episode_length=100 + i,
                    portfolio_return=0.01 + np.random.normal(0, 0.005)
                )
            
            monitor._step_count = 5000
            
            summary = monitor.get_training_summary()
            
            assert summary['model_id'] == 123
            assert summary['current_step'] == 5000
            # Now duration should be positive (approximately 1 hour)
            assert 0.9 <= summary['training_duration_hours'] <= 1.1
            assert summary['total_episodes'] == 50
            assert 'recent_avg_reward' in summary
            assert 'best_episode_reward' in summary
            assert 'current_metrics' in summary
    
    @pytest.mark.asyncio
    async def test_save_metrics_async(self, monitor):
        """Test asynchronous metrics saving."""
        # Add some episodes for validation metrics
        for i in range(15):
            monitor.add_episode_data(
                episode_reward=5.0,
                episode_length=100,
                portfolio_return=0.02 if i % 2 == 0 else -0.01
            )
        
        with patch('kalshiflow_rl.agents.training_monitor.model_registry') as mock_registry:
            # Use AsyncMock for async method
            mock_registry.update_model_metrics = AsyncMock(return_value=True)
            
            await monitor._save_metrics_async()
            
            mock_registry.update_model_metrics.assert_called_once()
            call_args = mock_registry.update_model_metrics.call_args
            
            # Check positional arguments (model_id is first arg)
            assert call_args[0][0] == 123  # model_id
            # Check keyword arguments
            assert 'training_metrics' in call_args[1]
            assert 'validation_metrics' in call_args[1]
            assert call_args[1]['auto_deploy'] is True
    
    @pytest.mark.asyncio
    async def test_checkpoint_triggering(self, monitor):
        """Test checkpoint triggering on frequency."""
        with patch.object(monitor, '_trigger_checkpoint_async') as mock_checkpoint:
            # Step that triggers checkpoint
            monitor.add_training_step_data(step=50)  # checkpoint_frequency = 50
            
            # Give async task time to be created
            await asyncio.sleep(0.01)
        
        assert monitor._last_checkpoint_step == 0  # Would be updated in real checkpoint


class TestTrainingProgressCallback:
    """Test SB3 training progress callback."""
    
    @pytest.fixture
    def training_monitor(self):
        """Create mock training monitor."""
        return MagicMock(spec=TrainingMonitor)
    
    @pytest.fixture
    def callback(self, training_monitor):
        """Create training progress callback."""
        return TrainingProgressCallback(training_monitor, verbose=1)
    
    def test_callback_initialization(self, callback, training_monitor):
        """Test callback initialization."""
        assert callback.training_monitor == training_monitor
        assert callback.verbose == 1
        assert callback.current_episode_reward == 0.0
        assert callback.current_episode_length == 0
    
    def test_on_step(self, callback, training_monitor):
        """Test step processing."""
        # Mock SB3 callback attributes
        callback.num_timesteps = 100
        callback.locals = {
            'rewards': [0.5],
            'dones': [False]
        }
        # Mock the model attribute (set by SB3 during training)
        callback.model = MagicMock()
        callback.model.logger = None  # No logger in test
        
        result = callback._on_step()
        
        assert result is True
        assert callback.current_episode_reward == 0.5
        assert callback.current_episode_length == 1
        
        # Verify monitor was called
        training_monitor.add_training_step_data.assert_called_once_with(
            step=100,
            policy_loss=None,
            value_loss=None,
            entropy=None,
            explained_variance=None
        )
    
    def test_on_episode_end(self, callback, training_monitor):
        """Test episode end processing."""
        # Set up episode data
        callback.current_episode_reward = 15.5
        callback.current_episode_length = 200
        
        # Mock training environment with episode stats
        mock_env = MagicMock()
        mock_env.get_episode_stats.return_value = {
            'portfolio_value': 10300.0,
            'total_trades': 12,
            'win_rate': 0.75
        }
        
        # Mock the model (needed for training_env property access)
        callback.model = MagicMock()
        
        # Mock the training_env property (read-only in BaseCallback)
        mock_training_env = MagicMock()
        mock_training_env.envs = [mock_env]
        with patch.object(type(callback), 'training_env', new_callable=PropertyMock) as mock_prop:
            mock_prop.return_value = mock_training_env
            
            callback._on_episode_end()
        
        # Verify monitor was called with correct data
        training_monitor.add_episode_data.assert_called_once()
        call_args = training_monitor.add_episode_data.call_args[1]
        
        assert call_args['episode_reward'] == 15.5
        assert call_args['episode_length'] == 200
        assert call_args['portfolio_return'] == 0.03  # (10300-10000)/10000
        assert call_args['episode_stats']['total_trades'] == 12
        
        # Verify episode reset
        assert callback.current_episode_reward == 0.0
        assert callback.current_episode_length == 0
    
    def test_episode_end_without_stats(self, callback, training_monitor):
        """Test episode end when environment has no stats."""
        callback.current_episode_reward = 10.0
        callback.current_episode_length = 150
        
        # Mock environment without get_episode_stats method
        # Mock the training_env property (read-only in BaseCallback)
        mock_training_env = MagicMock()
        mock_env_no_stats = MagicMock()
        del mock_env_no_stats.get_episode_stats  # Remove the method
        mock_training_env.envs = [mock_env_no_stats]
        
        with patch.object(type(callback), 'training_env', new_callable=PropertyMock) as mock_prop:
            mock_prop.return_value = mock_training_env
            
            callback._on_episode_end()
            
            # Should still call monitor but with minimal data
            training_monitor.add_episode_data.assert_called_once()
            call_args = training_monitor.add_episode_data.call_args[1]
            
            assert call_args['episode_reward'] == 10.0
            assert call_args['episode_length'] == 150
            assert call_args['portfolio_return'] == 0.0  # Default when no stats


class TestSessionManager:
    """Test training session manager."""
    
    @pytest.fixture
    def session_manager(self):
        """Create session manager for testing."""
        return SessionManager()
    
    @pytest.fixture
    def sample_config(self):
        """Sample training configuration."""
        return TrainingConfig(
            model_name="session_test",
            version="v1.0",
            algorithm=AlgorithmType.PPO,
            market_tickers=["INXD-25JAN03"],
            total_timesteps=1000
        )
    
    @pytest.mark.asyncio
    async def test_create_session(self, session_manager, sample_config):
        """Test session creation."""
        session_id = await session_manager.create_session(sample_config)
        
        assert session_id in session_manager.active_sessions
        session_state = session_manager.active_sessions[session_id]
        
        assert session_state.config == sample_config
        assert session_state.status == SessionStatus.PENDING
        assert session_state.created_at is not None
        assert session_id.startswith("session_test_v1.0_")
    
    @pytest.mark.asyncio
    async def test_concurrent_session_limit(self, session_manager, sample_config):
        """Test concurrent session limit."""
        session_manager.max_concurrent_sessions = 1
        
        # Create first session
        session_id1 = await session_manager.create_session(sample_config)
        
        # Try to create second session - should fail
        sample_config.model_name = "session_test_2"
        
        with pytest.raises(RuntimeError, match="Maximum concurrent sessions"):
            await session_manager.create_session(sample_config)
    
    @pytest.mark.asyncio
    async def test_initialize_session(self, session_manager, sample_config):
        """Test session initialization."""
        session_id = await session_manager.create_session(sample_config)
        
        with patch('kalshiflow_rl.agents.session_manager.model_registry') as mock_registry, \
             patch('kalshiflow_rl.agents.session_manager.create_training_monitor') as mock_create_monitor:
            
            # Use AsyncMock for async method
            mock_registry.register_model = AsyncMock(return_value=456)
            mock_monitor = MagicMock()
            mock_callback = MagicMock()
            mock_create_monitor.return_value = (mock_monitor, mock_callback)
            
            success = await session_manager.initialize_session(session_id)
            
            assert success is True
            session_state = session_manager.active_sessions[session_id]
            assert session_state.model_id == 456
            assert session_id in session_manager.session_monitors
    
    @pytest.mark.asyncio
    async def test_update_session_progress(self, session_manager, sample_config):
        """Test session progress updates."""
        session_id = await session_manager.create_session(sample_config)
        
        metrics = {
            'avg_episode_reward': 5.5,
            'avg_portfolio_return': 0.03,
            'total_trades': 100
        }
        
        await session_manager.update_session_progress(
            session_id, 
            current_step=500,
            current_episode=25,
            metrics=metrics
        )
        
        session_state = session_manager.active_sessions[session_id]
        assert session_state.current_step == 500
        assert session_state.current_episode == 25
        assert session_state.current_metrics == metrics
        assert session_state.best_reward == 5.5
        assert session_state.best_return == 0.03
    
    @pytest.mark.asyncio
    async def test_complete_session_success(self, session_manager, sample_config):
        """Test successful session completion."""
        session_id = await session_manager.create_session(sample_config)
        session_state = session_manager.active_sessions[session_id]
        session_state.model_id = 789
        
        final_metrics = {'final_reward': 10.0}
        
        with patch('kalshiflow_rl.agents.session_manager.model_registry') as mock_registry:
            # Use AsyncMock for async method
            mock_registry.set_model_status = AsyncMock(return_value=True)
            
            await session_manager.complete_session(
                session_id,
                success=True,
                final_metrics=final_metrics
            )
            
            assert session_state.status == SessionStatus.COMPLETED
            assert session_state.completed_at is not None
            assert session_state.current_metrics == final_metrics
            
            mock_registry.set_model_status.assert_called_once_with(789, 'active')
    
    @pytest.mark.asyncio
    async def test_complete_session_failure(self, session_manager, sample_config):
        """Test failed session completion."""
        session_id = await session_manager.create_session(sample_config)
        session_state = session_manager.active_sessions[session_id]
        session_state.model_id = 789
        
        error_message = "Training failed due to convergence issues"
        
        with patch('kalshiflow_rl.agents.session_manager.model_registry') as mock_registry:
            # Use AsyncMock for async method
            mock_registry.set_model_status = AsyncMock(return_value=True)
            
            await session_manager.complete_session(
                session_id,
                success=False,
                error_message=error_message
            )
            
            assert session_state.status == SessionStatus.FAILED
            assert session_state.error_message == error_message
            
            mock_registry.set_model_status.assert_called_once_with(789, 'failed')
    
    @pytest.mark.asyncio
    async def test_cancel_session(self, session_manager, sample_config):
        """Test session cancellation."""
        session_id = await session_manager.create_session(sample_config)
        
        # Set session to training status
        session_state = session_manager.active_sessions[session_id]
        session_state.status = SessionStatus.TRAINING
        
        success = await session_manager.cancel_session(session_id)
        
        assert success is True
        assert session_state.status == SessionStatus.CANCELLED
    
    @pytest.mark.asyncio
    async def test_get_session_status(self, session_manager, sample_config):
        """Test getting session status."""
        session_id = await session_manager.create_session(sample_config)
        
        # Add mock monitor
        mock_monitor = MagicMock()
        mock_metrics = MagicMock()
        mock_metrics.to_dict.return_value = {'test': 'metrics'}
        mock_monitor.get_current_metrics.return_value = mock_metrics
        mock_monitor.get_training_summary.return_value = {'summary': 'data'}
        
        session_manager.session_monitors[session_id] = mock_monitor
        
        status = await session_manager.get_session_status(session_id)
        
        assert status is not None
        assert status['config']['model_name'] == "session_test"
        assert 'latest_metrics' in status
        assert 'training_summary' in status
    
    @pytest.mark.asyncio
    async def test_list_active_sessions(self):
        """Test listing active sessions."""
        # Create session manager with higher concurrent session limit
        session_manager = SessionManager(max_concurrent_sessions=3)
        
        configs = []
        for i in range(3):
            config = TrainingConfig(
                model_name=f"test_model_{i}",
                version="v1.0",
                algorithm=AlgorithmType.PPO,
                market_tickers=["INXD-25JAN03"],
                total_timesteps=1000
            )
            configs.append(config)
            await session_manager.create_session(config)
        
        sessions = await session_manager.list_active_sessions()
        
        assert len(sessions) == 3
        model_names = [s['config']['model_name'] for s in sessions]
        assert "test_model_0" in model_names
        assert "test_model_1" in model_names
        assert "test_model_2" in model_names
    
    @pytest.mark.asyncio
    async def test_cleanup_session(self, session_manager, sample_config):
        """Test session cleanup."""
        session_id = await session_manager.create_session(sample_config)
        
        # Add mock monitor
        session_manager.session_monitors[session_id] = MagicMock()
        
        await session_manager.cleanup_session(session_id)
        
        assert session_id not in session_manager.active_sessions
        assert session_id not in session_manager.session_locks
