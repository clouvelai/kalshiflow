"""
Integration tests for Model Registry in Kalshi Flow RL Trading Subsystem.

These tests validate complete workflows with proper mocking to avoid real DB/file operations.
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from kalshiflow_rl.agents.model_registry import ModelRegistry, ModelConfig


@pytest.mark.integration
class TestModelRegistryIntegration:
    """Integration tests for model registry workflows."""
    
    @pytest.mark.asyncio
    async def test_full_model_lifecycle(self, tmp_path):
        """Test complete model lifecycle from registration to activation."""
        # Mock the entire database module to prevent any real DB calls
        with patch('kalshiflow_rl.agents.model_registry.rl_db') as mock_db:
            # Setup all database methods as AsyncMock
            mock_db.get_model_by_name_version = AsyncMock(return_value=None)
            mock_db.create_model = AsyncMock(return_value=456)
            mock_db.update_model_metrics = AsyncMock(return_value=True)
            mock_db.update_model_status = AsyncMock(return_value=True)
            mock_db.get_model_by_id = AsyncMock(return_value={
                'id': 456,
                'status': 'ready',
                'algorithm': 'PPO',
                'market_ticker': 'INXD-25JAN03',
                'file_path': str(tmp_path / 'test_model.pkl')
            })
            mock_db.get_active_model_for_market = AsyncMock(return_value=None)  # No existing active model
            
            # Also mock the config to use temp directory
            with patch('kalshiflow_rl.agents.model_registry.config.MODEL_STORAGE_PATH', str(tmp_path)):
                registry = ModelRegistry()
                
                # 1. Register a new model
                config = ModelConfig(
                    model_name="integration_test_model",
                    version="v1.0",
                    algorithm="PPO",
                    market_ticker="INXD-25JAN03",
                    hyperparameters={"learning_rate": 3e-4}
                )
                
                model_id = await registry.register_model(config)
                assert model_id == 456
                mock_db.create_model.assert_called_once()
                
                # 2. Update model metrics
                success = await registry.update_model_metrics(
                    model_id,
                    training_metrics={"loss": 0.3, "reward": 100},
                    validation_metrics={"avg_return": 0.08}
                )
                assert success is True
                mock_db.update_model_metrics.assert_called_once()
                
                # 3. Activate the model
                success = await registry.set_model_status(model_id, 'active')
                assert success is True
                mock_db.update_model_status.assert_called()
                
                # 4. Retrieve active model for market (update mock to return the active model)
                mock_db.get_active_model_for_market.return_value = {
                    'id': 456, 
                    'status': 'active',
                    'algorithm': 'PPO',
                    'market_ticker': 'INXD-25JAN03'
                }
                active_model = await registry.get_active_model_for_market('INXD-25JAN03')
                assert active_model['id'] == 456
                assert active_model['status'] == 'active'