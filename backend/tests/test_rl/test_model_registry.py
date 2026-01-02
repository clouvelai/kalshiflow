"""
Tests for Model Registry in Kalshi Flow RL Trading Subsystem.

Tests all aspects of model lifecycle management including:
- Model registration and database CRUD operations
- Model versioning and status management
- Performance metrics tracking and updates
- Hot-reload capabilities and callbacks
- Model cleanup and lifecycle management
"""

import pytest
import asyncio
import tempfile
import shutil
import json
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from stable_baselines3 import PPO

from kalshiflow_rl.agents.model_registry import (
    ModelRegistry,
    ModelConfig,
    model_registry
)
from kalshiflow_rl.data.database import rl_db


class TestModelConfig:
    """Test ModelConfig validation and functionality."""
    
    def test_valid_config(self):
        """Test valid model configuration."""
        config = ModelConfig(
            model_name="test_model",
            version="v1.0",
            algorithm="PPO",
            market_ticker="INXD-25JAN03",
            hyperparameters={"learning_rate": 3e-4, "n_steps": 2048}
        )
        
        errors = config.validate()
        assert len(errors) == 0
    
    def test_invalid_config_missing_required(self):
        """Test invalid configuration with missing required fields."""
        config = ModelConfig(
            model_name="",  # Invalid empty name
            version="v1.0",
            algorithm="PPO",
            market_ticker="INXD-25JAN03",
            hyperparameters={}
        )
        
        errors = config.validate()
        assert "model_name is required" in errors
    
    def test_invalid_algorithm(self):
        """Test invalid algorithm type."""
        config = ModelConfig(
            model_name="test_model",
            version="v1.0",
            algorithm="INVALID_ALGO",  # Unsupported algorithm
            market_ticker="INXD-25JAN03",
            hyperparameters={}
        )
        
        errors = config.validate()
        assert any("Unsupported algorithm" in error for error in errors)
    
    def test_invalid_hyperparameters(self):
        """Test invalid hyperparameters type."""
        config = ModelConfig(
            model_name="test_model",
            version="v1.0",
            algorithm="PPO",
            market_ticker="INXD-25JAN03",
            hyperparameters="not_a_dict"  # Should be dict
        )
        
        errors = config.validate()
        assert "hyperparameters must be a dictionary" in errors


class TestModelRegistry:
    """Test ModelRegistry functionality."""
    
    @pytest.fixture
    def temp_storage_path(self):
        """Create temporary storage path for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def registry(self, temp_storage_path):
        """Create ModelRegistry instance for testing."""
        with patch('kalshiflow_rl.agents.model_registry.config.MODEL_STORAGE_PATH', str(temp_storage_path)):
            registry = ModelRegistry()
            return registry
    
    @pytest.fixture
    def sample_config(self):
        """Sample model configuration for testing."""
        return ModelConfig(
            model_name="test_model",
            version="v1.0",
            algorithm="PPO",
            market_ticker="INXD-25JAN03",
            hyperparameters={"learning_rate": 3e-4, "n_steps": 2048}
        )
    
    @pytest.mark.asyncio
    async def test_register_model_success(self, registry, sample_config):
        """Test successful model registration."""
        # Mock database operations
        with patch.object(rl_db, 'get_model_by_name_version', return_value=None), \
             patch.object(rl_db, 'create_model', return_value=123):
            
            model_id = await registry.register_model(sample_config)
            
            assert model_id == 123
            assert f"{sample_config.model_name}:{sample_config.version}" in registry._model_cache
    
    @pytest.mark.asyncio
    async def test_register_duplicate_model(self, registry, sample_config):
        """Test registering duplicate model should fail."""
        # Mock existing model
        with patch.object(rl_db, 'get_model_by_name_version', return_value={'id': 123}):
            
            with pytest.raises(ValueError, match="already exists"):
                await registry.register_model(sample_config)
    
    @pytest.mark.asyncio
    async def test_register_invalid_config(self, registry):
        """Test registering invalid configuration should fail."""
        invalid_config = ModelConfig(
            model_name="",  # Invalid
            version="v1.0", 
            algorithm="PPO",
            market_ticker="INXD-25JAN03",
            hyperparameters={}
        )
        
        with pytest.raises(ValueError, match="Invalid model configuration"):
            await registry.register_model(invalid_config)
    
    @pytest.mark.asyncio
    async def test_save_model_checkpoint(self, registry, temp_storage_path):
        """Test saving model checkpoint."""
        # Create mock PPO model
        with patch('stable_baselines3.PPO') as MockPPO:
            mock_model = MagicMock()
            mock_model.save = MagicMock()
            
            file_path = str(temp_storage_path / "test_model.pkl")
            
            await registry.save_model_checkpoint(
                model=mock_model,
                file_path=file_path,
                metadata={"test": "data"}
            )
            
            # Verify model.save was called
            mock_model.save.assert_called_once_with(file_path.replace('.pkl', ''))
            
            # Verify metadata file created
            metadata_path = file_path.replace('.pkl', '_metadata.pkl')
            assert Path(metadata_path).exists()
    
    @pytest.mark.asyncio 
    async def test_load_model_success(self, registry):
        """Test successful model loading."""
        model_info = {
            'id': 123,
            'file_path': '/path/to/model.pkl',
            'algorithm': 'PPO'
        }
        
        with patch.object(rl_db, 'get_model_by_id', return_value=model_info), \
             patch('os.path.exists', return_value=True), \
             patch('stable_baselines3.PPO.load') as mock_load:
            
            mock_model = MagicMock()
            mock_load.return_value = mock_model
            
            loaded_model = await registry.load_model(123)
            
            assert loaded_model == mock_model
            mock_load.assert_called_once_with('/path/to/model')
    
    @pytest.mark.asyncio
    async def test_load_model_not_found(self, registry):
        """Test loading non-existent model."""
        with patch.object(rl_db, 'get_model_by_id', return_value=None):
            
            loaded_model = await registry.load_model(999)
            
            assert loaded_model is None
    
    @pytest.mark.asyncio
    async def test_load_model_file_missing(self, registry):
        """Test loading model with missing file."""
        model_info = {
            'id': 123,
            'file_path': '/nonexistent/model.pkl',
            'algorithm': 'PPO'
        }
        
        with patch.object(rl_db, 'get_model_by_id', return_value=model_info), \
             patch('os.path.exists', return_value=False):
            
            loaded_model = await registry.load_model(123)
            
            assert loaded_model is None
    
    @pytest.mark.asyncio
    async def test_update_model_metrics(self, registry):
        """Test updating model metrics."""
        training_metrics = {"loss": 0.5, "reward": 10.0}
        validation_metrics = {"avg_return": 0.05}
        
        with patch.object(rl_db, 'update_model_metrics', return_value=True), \
             patch.object(registry, '_check_auto_deployment') as mock_auto_deploy:
            
            success = await registry.update_model_metrics(
                model_id=123,
                training_metrics=training_metrics,
                validation_metrics=validation_metrics
            )
            
            assert success is True
            assert 123 in registry._performance_history
            mock_auto_deploy.assert_called_once_with(123, validation_metrics)
    
    @pytest.mark.asyncio
    async def test_set_model_status_activate(self, registry):
        """Test activating model and retiring previous active model."""
        model_info = {
            'id': 123,
            'market_ticker': 'INXD-25JAN03',
            'algorithm': 'PPO'
        }
        
        existing_active = {'id': 456}
        
        with patch.object(rl_db, 'get_model_by_id', return_value=model_info), \
             patch.object(rl_db, 'get_active_model_for_market', return_value=existing_active), \
             patch.object(rl_db, 'update_model_status', return_value=True), \
             patch.object(registry, '_trigger_hot_reload') as mock_reload:
            
            success = await registry.set_model_status(123, 'active')
            
            assert success is True
            mock_reload.assert_called_once_with(123)
    
    @pytest.mark.asyncio
    async def test_cleanup_old_models(self, registry):
        """Test cleaning up old models."""
        models = [
            {'id': 1, 'status': 'active', 'created_at': datetime.utcnow(), 'file_path': '/model1.pkl'},
            {'id': 2, 'status': 'retired', 'created_at': datetime.utcnow() - timedelta(days=1), 'file_path': '/model2.pkl'},
            {'id': 3, 'status': 'retired', 'created_at': datetime.utcnow() - timedelta(days=2), 'file_path': '/model3.pkl'},
        ]
        
        with patch.object(rl_db, 'list_models', return_value=models), \
             patch.object(rl_db, 'update_model_status', return_value=True), \
             patch('os.path.exists', return_value=True), \
             patch('os.remove') as mock_remove:
            
            cleaned_count = await registry.cleanup_old_models('INXD-25JAN03', keep_count=1)
            
            assert cleaned_count == 1  # Should clean up 1 old model (keeping 1 + active)
            mock_remove.assert_called_once_with('/model3.pkl')
    
    def test_hot_reload_callback(self, registry):
        """Test hot-reload callback registration and triggering."""
        callback_called = False
        callback_args = None
        
        def test_callback(model_id, file_path):
            nonlocal callback_called, callback_args
            callback_called = True
            callback_args = (model_id, file_path)
        
        registry.add_hot_reload_callback(test_callback)
        
        # Simulate triggering hot reload
        with patch.object(rl_db, 'get_model_by_id', return_value={'file_path': '/test/path'}):
            asyncio.run(registry._trigger_hot_reload(123))
        
        assert callback_called is True
        assert callback_args == ('123', '/test/path')
    
    @pytest.mark.asyncio
    async def test_check_for_updates(self, registry, temp_storage_path):
        """Test checking for file updates."""
        # Create test file
        test_file = temp_storage_path / "test_model.zip"
        test_file.write_text("test model data")
        
        # Add to file watchers with old timestamp
        registry._file_watchers[str(test_file)] = test_file.stat().st_mtime - 100
        
        # Mock finding model by file path
        with patch.object(registry, '_find_model_by_file_path', return_value={'id': 123}), \
             patch.object(registry, '_trigger_hot_reload') as mock_reload:
            
            updated_files = await registry.check_for_updates()
            
            assert str(test_file) in updated_files
            mock_reload.assert_called_once_with(123)
    
    def test_sanitize_filename(self, registry):
        """Test filename sanitization."""
        # Test normal filename
        result = registry._sanitize_filename("normal_model_name")
        assert result == "normal_model_name"
        
        # Test filename with unsafe characters
        result = registry._sanitize_filename("model/with\\unsafe:chars?")
        assert result == "model_with_unsafe_chars_"
        
        # Test empty filename
        result = registry._sanitize_filename("")
        assert result == "model"
        
        # Test very long filename
        long_name = "a" * 150
        result = registry._sanitize_filename(long_name)
        assert len(result) <= 100


class TestDatabaseCRUD:
    """Test database CRUD operations for models."""
    
    @pytest.mark.asyncio
    async def test_create_model(self):
        """Test creating model in database."""
        model_data = {
            'model_name': 'test_model',
            'version': 'v1.0',
            'algorithm': 'PPO',
            'market_ticker': 'INXD-25JAN03',
            'file_path': '/path/to/model.pkl',
            'hyperparameters': {'learning_rate': 3e-4},
            'training_metrics': {'loss': 0.5},
            'validation_metrics': {'accuracy': 0.8},
            'status': 'training'
        }
        
        with patch.object(rl_db, 'get_connection') as mock_get_conn:
            mock_conn = AsyncMock()
            mock_conn.fetchval.return_value = 123
            mock_get_conn.return_value.__aenter__.return_value = mock_conn
            
            model_id = await rl_db.create_model(model_data)
            
            assert model_id == 123
            mock_conn.fetchval.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_model_by_id(self):
        """Test retrieving model by ID."""
        mock_row = {
            'id': 123,
            'model_name': 'test_model',
            'hyperparameters': '{"learning_rate": 3e-4}',
            'training_metrics': '{"loss": 0.5}',
            'validation_metrics': '{"accuracy": 0.8}'
        }
        
        with patch.object(rl_db, 'get_connection') as mock_get_conn:
            mock_conn = AsyncMock()
            mock_conn.fetchrow.return_value = mock_row
            mock_get_conn.return_value.__aenter__.return_value = mock_conn
            
            model = await rl_db.get_model_by_id(123)
            
            assert model['id'] == 123
            assert model['model_name'] == 'test_model'
            assert isinstance(model['hyperparameters'], dict)
            assert model['hyperparameters']['learning_rate'] == 3e-4
    
    @pytest.mark.asyncio
    async def test_list_models_with_filters(self):
        """Test listing models with filters."""
        with patch.object(rl_db, 'get_connection') as mock_get_conn:
            mock_conn = AsyncMock()
            mock_conn.fetch.return_value = [
                {
                    'id': 123,
                    'model_name': 'test_model',
                    'market_ticker': 'INXD-25JAN03',
                    'status': 'active',
                    'hyperparameters': '{}',
                    'training_metrics': '{}',
                    'validation_metrics': '{}'
                }
            ]
            mock_get_conn.return_value.__aenter__.return_value = mock_conn
            
            models = await rl_db.list_models(
                market_ticker='INXD-25JAN03',
                status='active',
                limit=10
            )
            
            assert len(models) == 1
            assert models[0]['id'] == 123
            assert models[0]['status'] == 'active'
    
    @pytest.mark.asyncio
    async def test_update_model_status(self):
        """Test updating model status."""
        with patch.object(rl_db, 'get_connection') as mock_get_conn:
            mock_conn = AsyncMock()
            mock_conn.execute.return_value = "UPDATE 1"
            mock_get_conn.return_value.__aenter__.return_value = mock_conn
            
            success = await rl_db.update_model_status(123, 'active')
            
            assert success is True
    
    @pytest.mark.asyncio
    async def test_get_model_performance_summary(self):
        """Test getting model performance summary."""
        episode_stats = {
            'total_episodes': 100,
            'avg_return': 0.05,
            'max_return': 0.2,
            'min_return': -0.1,
            'avg_sharpe': 1.2,
            'avg_episode_length': 500,
            'total_trades': 1500,
            'win_rate': 0.6
        }
        
        model_info = {
            'model_name': 'test_model',
            'version': 'v1.0',
            'algorithm': 'PPO',
            'market_ticker': 'INXD-25JAN03',
            'status': 'active',
            'created_at': datetime.utcnow()
        }
        
        with patch.object(rl_db, 'get_connection') as mock_get_conn:
            mock_conn = AsyncMock()
            mock_conn.fetchrow.side_effect = [episode_stats, model_info]
            mock_get_conn.return_value.__aenter__.return_value = mock_conn
            
            summary = await rl_db.get_model_performance_summary(123)
            
            assert summary['model_id'] == 123
            assert summary['model_name'] == 'test_model'
            assert summary['performance']['total_episodes'] == 100
            assert summary['performance']['avg_return'] == 0.05


