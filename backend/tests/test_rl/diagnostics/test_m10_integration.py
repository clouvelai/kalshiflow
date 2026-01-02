"""
Test M10 Diagnostics Integration.

Basic integration tests to verify that M10 diagnostics components
can be imported and initialized correctly.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import numpy as np

from kalshiflow_rl.diagnostics import (
    ActionTracker, 
    RewardAnalyzer,
    ObservationValidator,
    DiagnosticsLogger,
    M10DiagnosticsCallback
)
from kalshiflow_rl.diagnostics.diagnostics_logger import DiagnosticsConfig


class TestM10Integration:
    """Test M10 diagnostics component integration."""
    
    def test_action_tracker_basic(self):
        """Test ActionTracker basic functionality."""
        tracker = ActionTracker()
        
        # Track some actions
        market_state = {'spread': 0.05, 'total_liquidity': 1000}
        
        for i in range(5):
            tracker.track_action(
                action=i % 5,
                step=i,
                global_step=i,
                market_state=market_state
            )
        
        # End episode and get summary
        summary = tracker.end_episode()
        
        assert summary['episode'] == 1
        assert summary['episode_length'] == 5
        assert 'action_distribution' in summary
        assert 'action_entropy' in summary
        
        # Get overall statistics
        stats = tracker.get_overall_statistics()
        assert 'total_actions_tracked' in stats
        assert stats['total_actions_tracked'] == 5
        
        # Get console summary
        console_summary = tracker.get_action_console_summary()
        assert 'ACTION DISTRIBUTION SUMMARY' in console_summary
    
    def test_reward_analyzer_basic(self):
        """Test RewardAnalyzer basic functionality."""
        analyzer = RewardAnalyzer()
        
        # Track some rewards
        market_state = {'spread': 0.05}
        position_info = {'MARKET1': {'position': 0, 'cost_basis': 0}}
        
        for i in range(5):
            analyzer.track_reward(
                reward=float(i - 2),  # -2, -1, 0, 1, 2
                step=i,
                global_step=i,
                portfolio_value_change=float(i - 2),
                previous_portfolio_value=10000.0,
                current_portfolio_value=10000.0 + (i - 2),
                position_info=position_info,
                cash_balance=10000.0,
                market_state=market_state,
                action=0,
                action_name="HOLD"
            )
        
        # End episode and get summary
        summary = analyzer.end_episode()
        
        assert summary['episode'] == 1
        assert summary['episode_length'] == 5
        assert 'total_reward' in summary
        assert 'reward_sparsity_pct' in summary
        
        # Get overall statistics
        stats = analyzer.get_overall_statistics()
        assert 'total_rewards_tracked' in stats
        assert 'reward_quality' in stats
        
        # Get console summary
        console_summary = analyzer.get_reward_console_summary()
        assert 'REWARD SIGNAL ANALYSIS' in console_summary
    
    def test_observation_validator_basic(self):
        """Test ObservationValidator basic functionality."""
        validator = ObservationValidator(expected_obs_dim=52)
        
        # Validate some observations
        for i in range(5):
            obs = np.random.randn(52).astype(np.float32)
            
            result = validator.validate_observation(
                observation=obs,
                step=i,
                global_step=i
            )
            
            assert 'observation_valid' in result
            assert result['observation_valid'] == True  # Should be valid
        
        # End episode and get summary
        summary = validator.end_episode()
        
        assert summary['episode'] == 1
        assert 'observation_quality' in summary
        
        # Get overall statistics
        stats = validator.get_overall_statistics()
        assert 'total_observations_validated' in stats
        
        # Get console summary
        console_summary = validator.get_observation_console_summary()
        assert 'OBSERVATION VALIDATION' in console_summary
    
    def test_diagnostics_logger_basic(self):
        """Test DiagnosticsLogger basic functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = DiagnosticsConfig(
                session_id=9,
                algorithm="ppo",
                console_output=False,  # Disable for test
                file_output=True,
                json_output=True
            )
            
            logger = DiagnosticsLogger(
                output_dir=temp_dir,
                config=config
            )
            
            # Log some diagnostic events
            logger.log_action_event({
                'action': 0,
                'action_name': 'HOLD',
                'step': 1
            })
            
            logger.log_reward_event({
                'total_reward': 0.5,
                'step': 1
            })
            
            logger.log_episode_summary({
                'episode': 1,
                'total_reward': 0.5,
                'episode_length': 100
            })
            
            # Save final summary
            final_stats = {
                'training_completed': True,
                'total_episodes': 1
            }
            
            summary_path = logger.save_final_summary(final_stats)
            logger.close()
            
            # Verify files were created
            output_dir = Path(logger.get_output_directory())
            assert output_dir.exists()
            
            files = logger.get_diagnostics_files()
            assert Path(files['text_log']).exists()
            assert Path(files['json_log']).exists()
            assert Path(files['summary']).exists()
    
    def test_m10_callback_creation(self):
        """Test M10DiagnosticsCallback can be created."""
        with tempfile.TemporaryDirectory() as temp_dir:
            callback = M10DiagnosticsCallback(
                output_dir=temp_dir,
                session_id=9,
                algorithm="ppo",
                action_tracking=True,
                reward_analysis=True,
                observation_validation=True,
                console_output=False,  # Disable for test
                detailed_logging=True,
                verbose=0
            )
            
            # Verify components are created
            assert callback.action_tracker is not None
            assert callback.reward_analyzer is not None
            assert callback.observation_validator is not None
            # Note: callback.logger requires init_callback() to be called with a model
            # This is an SB3 implementation detail, so we don't test it here
            
            # Verify directory structure
            output_dir = callback.get_diagnostics_directory()
            assert Path(output_dir).exists()
    
    def test_invalid_observation_handling(self):
        """Test handling of invalid observations."""
        validator = ObservationValidator(expected_obs_dim=52)
        
        # Test NaN observation
        nan_obs = np.full(52, np.nan)
        result = validator.validate_observation(nan_obs, 0, 0)
        
        assert not result['observation_valid']
        assert any('NaN' in issue for issue in result['issues'])
        
        # Test infinite observation
        inf_obs = np.full(52, np.inf)
        result = validator.validate_observation(inf_obs, 1, 1)
        
        assert not result['observation_valid']
        assert any('infinite' in issue for issue in result['issues'])
        
        # Test wrong dimension
        wrong_dim_obs = np.random.randn(10)
        result = validator.validate_observation(wrong_dim_obs, 2, 2)
        
        assert not result['observation_valid']
        assert any('dimension' in issue for issue in result['issues'])


if __name__ == "__main__":
    pytest.main([__file__])