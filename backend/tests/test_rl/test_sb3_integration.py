"""
Clean integration tests for SB3 Integration in Kalshi Flow RL Trading Subsystem.

Properly mocked to avoid database and file system dependencies.
"""

import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from stable_baselines3 import PPO, A2C

from kalshiflow_rl.agents.training_config import (
    TrainingConfig, TrainingConfigBuilder, AlgorithmType
)
from kalshiflow_rl.agents.training_harness import TrainingSession, TrainingManager
from kalshiflow_rl.agents.model_registry import ModelRegistry, ModelConfig
from kalshiflow_rl.agents.session_manager import SessionManager


@pytest.mark.integration
class TestSB3WorkflowIntegration:
    """Clean integration tests for SB3 training workflows."""
    
    @pytest.mark.asyncio
    async def test_model_registration_workflow(self, tmp_path):
        """Test model registration and management workflow."""
        with patch('kalshiflow_rl.agents.model_registry.rl_db') as mock_db:
            # Setup all async mocks
            mock_db.get_model_by_name_version = AsyncMock(return_value=None)
            mock_db.create_model = AsyncMock(return_value=555)
            mock_db.update_model_metrics = AsyncMock(return_value=True)
            mock_db.update_model_status = AsyncMock(return_value=True)
            mock_db.get_model_by_id = AsyncMock(return_value={
                'id': 555,
                'status': 'ready',
                'algorithm': 'PPO',
                'market_ticker': 'INXD-25JAN03',
                'file_path': str(tmp_path / 'model.pkl')
            })
            mock_db.get_active_model_for_market = AsyncMock(return_value=None)  # No existing active model
            
            with patch('kalshiflow_rl.agents.model_registry.config.MODEL_STORAGE_PATH', str(tmp_path)):
                registry = ModelRegistry()
                
                # Register model
                config = ModelConfig(
                    model_name="workflow_test",
                    version="v1.0",
                    algorithm="PPO",
                    market_ticker="INXD-25JAN03",
                    hyperparameters={"learning_rate": 3e-4}
                )
                
                model_id = await registry.register_model(config)
                assert model_id == 555
                
                # Create mock model and save
                with patch('stable_baselines3.PPO.load') as mock_load:
                    mock_model = MagicMock()
                    mock_model.save = MagicMock()
                    mock_load.return_value = mock_model
                    
                    # Save checkpoint
                    checkpoint_path = tmp_path / 'checkpoint.pkl'
                    await registry.save_model_checkpoint(
                        model=mock_model,
                        file_path=str(checkpoint_path),
                        metadata={"step": 1000}
                    )
                    
                    mock_model.save.assert_called_once()
                
                # Update metrics and activate
                await registry.update_model_metrics(
                    model_id=555,
                    training_metrics={"loss": 0.5},
                    validation_metrics={"return": 0.1}
                )
                
                await registry.set_model_status(555, 'active')
                mock_db.update_model_status.assert_called_with(555, 'active')
    
    @pytest.mark.asyncio
    async def test_session_manager_integration(self):
        """Test session manager workflow with proper mocking."""
        config = TrainingConfigBuilder.create_default_ppo_config(
            "session_test", "v1.0", "INXD-25JAN03",
            total_timesteps=100
        )
        
        # Create session manager
        session_manager = SessionManager(max_concurrent_sessions=2)
        
        # Mock all external dependencies
        with patch('kalshiflow_rl.agents.session_manager.model_registry') as mock_registry, \
             patch('kalshiflow_rl.agents.session_manager.create_training_monitor') as mock_create_monitor:
            
            # Setup mocks
            mock_registry.register_model = AsyncMock(return_value=888)
            mock_registry.set_model_status = AsyncMock(return_value=True)
            
            mock_monitor = MagicMock()
            mock_callback = MagicMock()
            mock_create_monitor.return_value = (mock_monitor, mock_callback)
            
            # Create and initialize session
            session_id = await session_manager.create_session(config)
            assert session_id in session_manager.active_sessions
            
            # Initialize session
            success = await session_manager.initialize_session(session_id)
            assert success is True
            
            session_state = session_manager.active_sessions[session_id]
            assert session_state.model_id == 888
            
            # Complete session successfully
            await session_manager.complete_session(
                session_id,
                success=True,
                final_metrics={'final_reward': 10.0}
            )
            
            # Verify model was activated
            mock_registry.set_model_status.assert_called_with(888, 'active')