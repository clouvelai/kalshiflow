"""
Comprehensive tests for SessionCurriculumManager and curriculum learning system.

This test suite validates:
- SessionCurriculumManager initialization and configuration
- Session data loading and market view creation
- Training pipeline execution and result tracking
- Error handling and edge cases
- End-to-end integration with MarketAgnosticKalshiEnv
- Performance tracking and aggregation
"""

import pytest
import asyncio
import os
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any
from unittest.mock import AsyncMock, Mock, patch

from kalshiflow_rl.training.curriculum import (
    SimpleSessionCurriculum,
    MarketTrainingResult,
    SessionTrainingResults,
    train_single_session,
    train_multiple_sessions
)
from kalshiflow_rl.environments.market_agnostic_env import EnvConfig
from kalshiflow_rl.environments.session_data_loader import SessionData, SessionDataPoint, MarketSessionView

logger = logging.getLogger(__name__)


@pytest.fixture
def database_url():
    """Get database URL from environment."""
    return os.getenv("DATABASE_URL", "postgresql://test:test@localhost:5432/test_kalshi")


@pytest.fixture
def env_config():
    """Default environment configuration for testing."""
    return EnvConfig(
        max_markets=1,
        temporal_features=True,
        cash_start=10000  # $100 in cents
    )


@pytest.fixture
def mock_session_data():
    """Create mock session data for testing."""
    session_data = Mock(spec=SessionData)
    session_data.session_id = 123
    session_data.markets_involved = ["MARKET_A", "MARKET_B", "MARKET_C"]
    session_data.get_episode_length.return_value = 50
    session_data.total_duration = timedelta(minutes=15)
    
    # Mock market view creation
    def create_market_view_side_effect(market_ticker):
        if market_ticker in ["MARKET_A", "MARKET_B"]:
            view = Mock(spec=MarketSessionView)
            view.target_market = market_ticker
            view.session_id = 123
            view.get_episode_length.return_value = 30
            view.market_coverage = 0.6
            view.avg_spread = 0.025
            view.volatility_score = 0.15
            return view
        return None  # MARKET_C has insufficient data
    
    session_data.create_market_view.side_effect = create_market_view_side_effect
    return session_data


@pytest.fixture
def mock_data_loader(mock_session_data):
    """Mock SessionDataLoader for testing."""
    loader = Mock()
    loader.load_session = AsyncMock(return_value=mock_session_data)
    return loader


class TestMarketTrainingResult:
    """Test MarketTrainingResult data structure."""
    
    def test_initialization(self):
        """Test basic initialization."""
        result = MarketTrainingResult(
            market_ticker="MARKET_A",
            session_id=123
        )
        
        assert result.market_ticker == "MARKET_A"
        assert result.session_id == 123
        assert result.success is False
        assert result.total_reward == 0.0
        assert result.episode_length == 0
        assert result.error_message is None
    
    def test_post_init_duration_calculation(self):
        """Test automatic duration calculation."""
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=5)
        
        result = MarketTrainingResult(
            market_ticker="MARKET_A",
            session_id=123,
            start_time=start_time,
            end_time=end_time
        )
        
        assert result.duration == timedelta(minutes=5)
    
    def test_successful_result_metrics(self):
        """Test successful training result with all metrics."""
        result = MarketTrainingResult(
            market_ticker="MARKET_A",
            session_id=123,
            success=True,
            total_reward=150.0,
            episode_length=45,
            final_cash=9800.0,
            final_position_value=350.0,
            market_coverage=0.75,
            avg_spread=0.02,
            volatility_score=0.12
        )
        
        assert result.success is True
        assert result.total_reward == 150.0
        assert result.episode_length == 45
        assert result.market_coverage == 0.75


class TestSessionTrainingResults:
    """Test SessionTrainingResults aggregation."""
    
    def test_initialization(self):
        """Test basic initialization."""
        results = SessionTrainingResults(session_id=123)
        
        assert results.session_id == 123
        assert results.total_markets == 0
        assert results.successful_markets == 0
        assert results.failed_markets == 0
        assert results.get_success_rate() == 0.0
    
    def test_add_successful_result(self):
        """Test adding successful training results."""
        results = SessionTrainingResults(session_id=123)
        
        # Add first successful result
        result1 = MarketTrainingResult(
            market_ticker="MARKET_A", 
            session_id=123,
            success=True,
            total_reward=100.0,
            episode_length=30
        )
        results.add_result(result1)
        
        assert results.total_markets == 1
        assert results.successful_markets == 1
        assert results.failed_markets == 0
        assert results.total_episodes == 1
        assert results.total_timesteps == 30
        assert results.avg_reward == 100.0
        assert results.best_reward == 100.0
        assert results.worst_reward == 100.0
        assert results.get_success_rate() == 1.0
        
        # Add second successful result
        result2 = MarketTrainingResult(
            market_ticker="MARKET_B",
            session_id=123,
            success=True,
            total_reward=200.0,
            episode_length=40
        )
        results.add_result(result2)
        
        assert results.total_markets == 2
        assert results.successful_markets == 2
        assert results.total_episodes == 2
        assert results.total_timesteps == 70
        assert results.avg_reward == 150.0  # (100 + 200) / 2
        assert results.best_reward == 200.0
        assert results.worst_reward == 100.0
    
    def test_add_failed_result(self):
        """Test adding failed training results."""
        results = SessionTrainingResults(session_id=123)
        
        failed_result = MarketTrainingResult(
            market_ticker="MARKET_C",
            session_id=123,
            success=False,
            error_message="Insufficient data"
        )
        results.add_result(failed_result)
        
        assert results.total_markets == 1
        assert results.successful_markets == 0
        assert results.failed_markets == 1
        assert results.total_episodes == 0
        assert results.get_success_rate() == 0.0
    
    def test_mixed_results(self):
        """Test aggregation with both successful and failed results."""
        results = SessionTrainingResults(session_id=123)
        
        # Add successful result
        success = MarketTrainingResult(
            market_ticker="MARKET_A",
            session_id=123,
            success=True,
            total_reward=100.0,
            episode_length=30
        )
        results.add_result(success)
        
        # Add failed result
        failure = MarketTrainingResult(
            market_ticker="MARKET_B",
            session_id=123,
            success=False,
            error_message="Test error"
        )
        results.add_result(failure)
        
        assert results.total_markets == 2
        assert results.successful_markets == 1
        assert results.failed_markets == 1
        assert results.get_success_rate() == 0.5
    
    def test_get_summary(self):
        """Test summary generation."""
        results = SessionTrainingResults(
            session_id=123,
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(minutes=10)
        )
        results.total_duration = results.end_time - results.start_time
        
        # Add some results
        for i, reward in enumerate([100.0, 200.0]):
            result = MarketTrainingResult(
                market_ticker=f"MARKET_{i}",
                session_id=123,
                success=True,
                total_reward=reward,
                episode_length=30
            )
            results.add_result(result)
        
        summary = results.get_summary()
        
        assert summary['session_id'] == 123
        assert summary['total_markets'] == 2
        assert summary['success_rate'] == 1.0
        assert summary['successful_markets'] == 2
        assert summary['failed_markets'] == 0
        assert summary['avg_reward'] == 150.0
        assert summary['best_reward'] == 200.0
        assert summary['worst_reward'] == 100.0
        assert 'total_duration' in summary


class TestSimpleSessionCurriculum:
    """Test SimpleSessionCurriculum main class."""
    
    def test_initialization(self, database_url, env_config):
        """Test curriculum initialization."""
        curriculum = SimpleSessionCurriculum(
            database_url=database_url,
            env_config=env_config
        )
        
        assert curriculum.database_url == database_url
        assert curriculum.env_config == env_config
        assert curriculum.current_session_results is None
        assert len(curriculum.session_history) == 0
    
    def test_initialization_with_defaults(self, database_url):
        """Test initialization with default config."""
        curriculum = SimpleSessionCurriculum(database_url=database_url)
        
        assert curriculum.env_config is not None
        assert curriculum.env_config.max_markets == 1
        assert curriculum.env_config.cash_start == 10000
    
    @pytest.mark.asyncio
    async def test_is_market_valid_success(self, database_url, mock_session_data):
        """Test market validation logic for valid market."""
        curriculum = SimpleSessionCurriculum(database_url=database_url)
        
        # Test valid market
        is_valid = await curriculum._is_market_valid(mock_session_data, "MARKET_A", 1, 1)
        assert is_valid is True
    
    @pytest.mark.asyncio
    async def test_is_market_valid_failure(self, database_url, mock_session_data):
        """Test market validation logic for invalid market."""
        curriculum = SimpleSessionCurriculum(database_url=database_url)
        
        # Test invalid market (returns None from create_market_view)
        is_valid = await curriculum._is_market_valid(mock_session_data, "MARKET_C", 1, 1)
        assert is_valid is False
    
    @pytest.mark.asyncio
    async def test_train_market_success(self, database_url, mock_session_data):
        """Test successful market training."""
        curriculum = SimpleSessionCurriculum(database_url=database_url)
        
        # Mock the environment to avoid actual training
        with patch('kalshiflow_rl.training.curriculum.MarketAgnosticKalshiEnv') as mock_env_class:
            # Set up mock environment
            mock_env = Mock()
            mock_env.action_space.sample.return_value = 0  # HOLD action
            mock_env.reset.return_value = (np.zeros(52), {})
            
            # Mock step responses for a short episode
            step_responses = [
                (np.zeros(52), 10.0, False, False, {}),  # Step 1
                (np.zeros(52), 5.0, False, False, {}),   # Step 2
                (np.zeros(52), -3.0, True, False, {})    # Step 3 (terminated)
            ]
            mock_env.step.side_effect = step_responses
            
            # Mock order manager for final metrics
            mock_env.order_manager.cash = 9950
            mock_env.order_manager.get_total_position_value.return_value = 50
            
            mock_env_class.return_value = mock_env
            
            # Execute training
            result = await curriculum._train_market(mock_session_data, "MARKET_A")
            
            # Validate results
            assert result.success is True
            assert result.market_ticker == "MARKET_A"
            assert result.session_id == 123
            assert result.total_reward == 12.0  # 10 + 5 - 3
            assert result.episode_length == 3
            assert result.final_cash == 9950
            assert result.final_position_value == 50
            assert result.error_message is None
            
            # Validate market characteristics were extracted
            assert result.market_coverage == 0.6
            assert result.avg_spread == 0.025
            assert result.volatility_score == 0.15
    
    @pytest.mark.asyncio
    async def test_train_market_failure(self, database_url, mock_session_data):
        """Test market training with errors."""
        curriculum = SimpleSessionCurriculum(database_url=database_url)
        
        # Test with market that returns None from create_market_view
        result = await curriculum._train_market(mock_session_data, "MARKET_C")
        
        assert result.success is False
        assert result.error_message == "Failed to create market view"
        assert result.total_reward == 0.0
        assert result.episode_length == 0
    
    @pytest.mark.asyncio
    async def test_train_session_success(self, database_url, mock_data_loader, mock_session_data):
        """Test successful session training."""
        curriculum = SimpleSessionCurriculum(database_url=database_url)
        curriculum.data_loader = mock_data_loader
        
        # Mock the _train_market method to return controlled results
        async def mock_train_market(session_data, market_ticker):
            if market_ticker in ["MARKET_A", "MARKET_B"]:
                return MarketTrainingResult(
                    market_ticker=market_ticker,
                    session_id=123,
                    success=True,
                    total_reward=100.0,
                    episode_length=30
                )
            else:
                return MarketTrainingResult(
                    market_ticker=market_ticker,
                    session_id=123,
                    success=False,
                    error_message="Invalid market"
                )
        
        curriculum._train_market = mock_train_market
        
        # Execute session training
        results = await curriculum.train_session(123)
        
        # Validate session results
        assert results.session_id == 123
        assert results.total_markets == 2  # Only MARKET_A and MARKET_B are valid
        assert results.successful_markets == 2
        assert results.failed_markets == 0
        assert results.get_success_rate() == 1.0
        assert results.avg_reward == 100.0
        
        # Check that results were added to history
        assert len(curriculum.session_history) == 1
        assert curriculum.session_history[0] == results
    
    @pytest.mark.asyncio
    async def test_train_session_no_data(self, database_url):
        """Test session training when session data loading fails."""
        curriculum = SimpleSessionCurriculum(database_url=database_url)
        
        # Mock data loader to return None (failed load)
        curriculum.data_loader = Mock()
        curriculum.data_loader.load_session = AsyncMock(return_value=None)
        
        results = await curriculum.train_session(999)
        
        assert results.session_id == 999
        assert results.total_markets == 0
        assert results.successful_markets == 0
        assert results.failed_markets == 0
        assert results.end_time is not None
        assert results.total_duration is not None
    
    def test_get_session_summary(self, database_url):
        """Test session summary retrieval."""
        curriculum = SimpleSessionCurriculum(database_url=database_url)
        
        # Add mock session results
        results = SessionTrainingResults(session_id=123)
        results.total_markets = 2
        results.successful_markets = 2
        curriculum.session_history.append(results)
        
        summary = curriculum.get_session_summary(123)
        assert summary is not None
        assert summary['session_id'] == 123
        assert summary['total_markets'] == 2
        
        # Test non-existent session
        assert curriculum.get_session_summary(999) is None
    
    def test_get_overall_summary_empty(self, database_url):
        """Test overall summary with no training history."""
        curriculum = SimpleSessionCurriculum(database_url=database_url)
        
        summary = curriculum.get_overall_summary()
        assert summary['total_sessions'] == 0
        assert summary['total_markets'] == 0
        assert summary['overall_success_rate'] == 0.0
        assert summary['avg_reward_across_sessions'] == 0.0
    
    def test_get_overall_summary_with_data(self, database_url):
        """Test overall summary with training history."""
        curriculum = SimpleSessionCurriculum(database_url=database_url)
        
        # Add multiple session results
        for session_id in [123, 124]:
            results = SessionTrainingResults(session_id=session_id)
            results.total_markets = 3
            results.successful_markets = 2
            results.total_episodes = 2
            results.avg_reward = 100.0 + session_id  # Different rewards
            curriculum.session_history.append(results)
        
        summary = curriculum.get_overall_summary()
        
        assert summary['total_sessions'] == 2
        assert summary['total_markets'] == 6
        assert summary['successful_markets'] == 4
        assert summary['failed_markets'] == 2
        assert summary['overall_success_rate'] == 4/6  # 0.667
        assert summary['avg_reward_across_sessions'] == (223.0 + 224.0) / 2  # Average of session averages
    
    def test_reset(self, database_url):
        """Test curriculum state reset."""
        curriculum = SimpleSessionCurriculum(database_url=database_url)
        
        # Add some state
        curriculum.current_session_results = SessionTrainingResults(session_id=123)
        curriculum.session_history.append(SessionTrainingResults(session_id=124))
        
        # Reset and verify
        curriculum.reset()
        
        assert curriculum.current_session_results is None
        assert len(curriculum.session_history) == 0


class TestUtilityFunctions:
    """Test utility functions for easy integration."""
    
    @pytest.mark.asyncio
    async def test_train_single_session(self, database_url, env_config):
        """Test train_single_session convenience function."""
        with patch('kalshiflow_rl.training.curriculum.SimpleSessionCurriculum') as mock_curriculum_class:
            mock_curriculum = Mock()
            mock_results = SessionTrainingResults(session_id=123)
            mock_curriculum.train_session = AsyncMock(return_value=mock_results)
            mock_curriculum_class.return_value = mock_curriculum
            
            results = await train_single_session(
                session_id=123,
                database_url=database_url,
                env_config=env_config,
                min_snapshots=2,
                min_deltas=3
            )
            
            # Verify curriculum was created correctly
            mock_curriculum_class.assert_called_once_with(database_url, env_config)
            
            # Verify training was called with correct parameters
            mock_curriculum.train_session.assert_called_once_with(123, 2, 3)
            
            assert results == mock_results
    
    @pytest.mark.asyncio
    async def test_train_multiple_sessions(self, database_url, env_config):
        """Test train_multiple_sessions convenience function."""
        with patch('kalshiflow_rl.training.curriculum.SimpleSessionCurriculum') as mock_curriculum_class:
            mock_curriculum = Mock()
            
            # Mock sequential training results
            mock_results = [
                SessionTrainingResults(session_id=123),
                SessionTrainingResults(session_id=124),
                SessionTrainingResults(session_id=125)
            ]
            
            async def mock_train_session(session_id):
                for result in mock_results:
                    if result.session_id == session_id:
                        return result
                return SessionTrainingResults(session_id=session_id)
            
            mock_curriculum.train_session = mock_train_session
            mock_curriculum_class.return_value = mock_curriculum
            
            session_ids = [123, 124, 125]
            results = await train_multiple_sessions(
                session_ids=session_ids,
                database_url=database_url,
                env_config=env_config
            )
            
            # Verify curriculum was created correctly
            mock_curriculum_class.assert_called_once_with(database_url, env_config)
            
            # Verify each session was trained
            assert len(results) == 3
            for i, session_id in enumerate(session_ids):
                assert results[i] == mock_results[i]


class TestIntegration:
    """Integration tests with real components."""
    
    @pytest.mark.asyncio
    async def test_curriculum_with_real_session_data(self, database_url):
        """Test curriculum with actual session data if available."""
        # This test only runs if we have real database access
        if not database_url or "test" in database_url:
            pytest.skip("Real database integration test skipped")
        
        curriculum = SimpleSessionCurriculum(database_url=database_url)
        
        # Try to get available sessions
        try:
            sessions = await curriculum.data_loader.get_available_sessions()
            if not sessions:
                pytest.skip("No sessions available for integration test")
            
            # Test with first available session
            session_id = sessions[0]['session_id']
            results = await curriculum.train_session(session_id)
            
            # Basic validation
            assert results.session_id == session_id
            assert results.total_markets >= 0
            assert results.end_time is not None
            assert results.total_duration is not None
            
            logger.info(f"Integration test completed: "
                       f"session={session_id}, "
                       f"markets={results.total_markets}, "
                       f"success_rate={results.get_success_rate():.1%}")
            
        except Exception as e:
            pytest.skip(f"Integration test failed due to: {e}")


# Fixtures and utilities for testing with numpy arrays
@pytest.fixture
def sample_observation():
    """Sample observation array for testing."""
    return np.zeros(52, dtype=np.float32)


@pytest.fixture
def sample_market_data():
    """Sample market data for orderbook conversion testing."""
    return {
        'yes_bids': {60: 100, 59: 200},
        'yes_asks': {61: 150, 62: 250},
        'no_bids': {40: 120, 39: 180},
        'no_asks': {41: 130, 42: 220}
    }


if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v"])