"""
Model Registry for Kalshi Flow RL Trading Subsystem.

Provides high-level interface for managing trained RL models including
versioning, checkpoint management, hot-reload capabilities, and model
lineage tracking. Enforces architectural constraints for safe model
deployment and lifecycle management.
"""

import os
import asyncio
import hashlib
import logging
import shutil
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, asdict
import threading
import time

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.base_class import BaseAlgorithm

from ..config import config
from ..data.database import rl_db

logger = logging.getLogger("kalshiflow_rl.model_registry")


@dataclass
class ModelConfig:
    """Configuration for RL model training and deployment."""
    
    # Model identification
    model_name: str
    version: str
    algorithm: str  # 'PPO', 'A2C', etc.
    market_ticker: str
    
    # Training configuration
    hyperparameters: Dict[str, Any]
    
    # Model file management
    file_path: Optional[str] = None
    checkpoint_interval: int = 1000  # Save checkpoint every N steps
    
    # Performance tracking
    target_metrics: Optional[Dict[str, float]] = None  # Performance targets
    validation_frequency: int = 100  # Validate every N episodes
    
    # Deployment configuration
    auto_deploy_threshold: Optional[float] = None  # Auto-deploy if performance > threshold
    max_models_per_market: int = 5  # Keep max N models per market
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        if not self.model_name:
            errors.append("model_name is required")
        
        if not self.version:
            errors.append("version is required")
        
        if self.algorithm not in ['PPO', 'A2C']:
            errors.append(f"Unsupported algorithm: {self.algorithm}")
        
        if not self.market_ticker:
            errors.append("market_ticker is required")
        
        if not isinstance(self.hyperparameters, dict):
            errors.append("hyperparameters must be a dictionary")
        
        if self.checkpoint_interval <= 0:
            errors.append("checkpoint_interval must be positive")
        
        return errors


class ModelRegistry:
    """
    High-level model registry for RL Trading Subsystem.
    
    Provides:
    - Model versioning and checkpoint management
    - Hot-reload capabilities for inference actors
    - Model performance tracking and comparison
    - Automated deployment based on performance thresholds
    - Safe model lifecycle management
    """
    
    def __init__(self):
        """Initialize model registry."""
        self.storage_path = Path(config.MODEL_STORAGE_PATH)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache for fast model lookups
        self._model_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_lock = threading.Lock()
        
        # Hot-reload support
        self._file_watchers: Dict[str, float] = {}  # file_path -> last_modified
        self._reload_callbacks: List[Callable[[str, str], None]] = []  # (model_id, file_path) -> None
        
        # Performance tracking
        self._performance_history: Dict[int, List[Dict[str, Any]]] = {}  # model_id -> metrics_list
        
        logger.info(f"ModelRegistry initialized with storage: {self.storage_path}")
    
    async def register_model(
        self,
        model_config: ModelConfig,
        model: Optional[BaseAlgorithm] = None,
        initial_metrics: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Register a new model in the registry.
        
        Args:
            model_config: Model configuration
            model: Trained model instance (optional - can be saved later)
            initial_metrics: Initial performance metrics
            
        Returns:
            Model ID
        """
        # Validate configuration
        errors = model_config.validate()
        if errors:
            raise ValueError(f"Invalid model configuration: {', '.join(errors)}")
        
        # Check for existing model with same name/version
        existing = await rl_db.get_model_by_name_version(
            model_config.model_name, model_config.version
        )
        if existing:
            raise ValueError(f"Model {model_config.model_name}:{model_config.version} already exists")
        
        # Determine file path
        file_path = model_config.file_path
        if not file_path:
            safe_name = self._sanitize_filename(
                f"{model_config.model_name}_{model_config.version}_{model_config.algorithm}"
            )
            file_path = str(self.storage_path / f"{safe_name}.pkl")
        
        # Save model if provided
        if model:
            await self.save_model_checkpoint(model, file_path)
        
        # Create database record
        model_data = {
            'model_name': model_config.model_name,
            'version': model_config.version,
            'algorithm': model_config.algorithm,
            'market_ticker': model_config.market_ticker,
            'file_path': file_path,
            'hyperparameters': model_config.hyperparameters,
            'training_metrics': initial_metrics or {},
            'validation_metrics': {},
            'status': 'training'  # Start in training status
        }
        
        model_id = await rl_db.create_model(model_data)
        
        # Update cache
        with self._cache_lock:
            cache_key = f"{model_config.model_name}:{model_config.version}"
            self._model_cache[cache_key] = {
                'id': model_id,
                'config': model_config,
                'file_path': file_path,
                'last_updated': time.time()
            }
        
        logger.info(f"Registered model {model_config.model_name}:{model_config.version} with ID {model_id}")
        return model_id
    
    async def save_model_checkpoint(
        self,
        model: BaseAlgorithm,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save model checkpoint to filesystem.
        
        Args:
            model: Trained model instance
            file_path: Path to save model
            metadata: Additional metadata to save with model
        """
        try:
            # Ensure directory exists
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save model using Stable Baselines3 format
            model.save(file_path.replace('.pkl', ''))  # SB3 adds .zip extension
            actual_path = file_path.replace('.pkl', '.zip')
            
            # Save additional metadata if provided
            if metadata:
                metadata_path = file_path.replace('.pkl', '_metadata.pkl')
                with open(metadata_path, 'wb') as f:
                    pickle.dump(metadata, f)
            
            # Update file watcher only if file actually exists (handles mocked models)
            if os.path.exists(actual_path):
                self._file_watchers[actual_path] = os.path.getmtime(actual_path)
            
            logger.info(f"Saved model checkpoint: {actual_path}")
            
        except Exception as e:
            logger.error(f"Failed to save model checkpoint {file_path}: {e}")
            raise
    
    async def load_model(self, model_id: int) -> Optional[BaseAlgorithm]:
        """
        Load model from filesystem.
        
        Args:
            model_id: Model ID
            
        Returns:
            Loaded model instance or None if not found
        """
        try:
            # Get model info from database
            model_info = await rl_db.get_model_by_id(model_id)
            if not model_info:
                logger.error(f"Model {model_id} not found in database")
                return None
            
            file_path = model_info['file_path']
            algorithm = model_info['algorithm']
            
            # Convert .pkl to .zip if needed (SB3 format)
            if file_path.endswith('.pkl'):
                file_path = file_path.replace('.pkl', '.zip')
            
            if not os.path.exists(file_path):
                logger.error(f"Model file not found: {file_path}")
                return None
            
            # Load model using appropriate SB3 class
            if algorithm == 'PPO':
                model = PPO.load(file_path.replace('.zip', ''))
            elif algorithm == 'A2C':
                model = A2C.load(file_path.replace('.zip', ''))
            else:
                logger.error(f"Unsupported algorithm: {algorithm}")
                return None
            
            logger.info(f"Loaded model {model_id} from {file_path}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            return None
    
    async def update_model_metrics(
        self,
        model_id: int,
        training_metrics: Optional[Dict[str, Any]] = None,
        validation_metrics: Optional[Dict[str, Any]] = None,
        auto_deploy: bool = True
    ) -> bool:
        """
        Update model performance metrics.
        
        Args:
            model_id: Model ID
            training_metrics: Training performance metrics
            validation_metrics: Validation performance metrics
            auto_deploy: Whether to consider auto-deployment
            
        Returns:
            True if updated successfully
        """
        try:
            # Update database
            success = await rl_db.update_model_metrics(
                model_id, training_metrics, validation_metrics
            )
            
            if not success:
                return False
            
            # Track performance history
            if model_id not in self._performance_history:
                self._performance_history[model_id] = []
            
            metrics_record = {
                'timestamp': datetime.utcnow().isoformat(),
                'training_metrics': training_metrics,
                'validation_metrics': validation_metrics
            }
            self._performance_history[model_id].append(metrics_record)
            
            # Check auto-deployment
            if auto_deploy and validation_metrics:
                await self._check_auto_deployment(model_id, validation_metrics)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update metrics for model {model_id}: {e}")
            return False
    
    async def set_model_status(self, model_id: int, status: str) -> bool:
        """
        Set model status.
        
        Args:
            model_id: Model ID
            status: New status ('training', 'active', 'retired', 'failed')
            
        Returns:
            True if updated successfully
        """
        if status not in ['training', 'active', 'retired', 'failed']:
            raise ValueError(f"Invalid status: {status}")
        
        # Special handling for activating models
        if status == 'active':
            # Get model info
            model_info = await rl_db.get_model_by_id(model_id)
            if not model_info:
                logger.error(f"Model {model_id} not found")
                return False
            
            # Retire existing active model for the same market
            market_ticker = model_info['market_ticker']
            algorithm = model_info['algorithm']
            
            existing_active = await rl_db.get_active_model_for_market(market_ticker, algorithm)
            if existing_active and existing_active['id'] != model_id:
                await rl_db.update_model_status(existing_active['id'], 'retired')
                logger.info(f"Retired previous active model {existing_active['id']} for {market_ticker}")
        
        # Update status
        success = await rl_db.update_model_status(model_id, status)
        
        if success:
            logger.info(f"Updated model {model_id} status to {status}")
            
            # Trigger hot-reload callbacks if activating
            if status == 'active':
                await self._trigger_hot_reload(model_id)
        
        return success
    
    async def get_active_model_for_market(self, market_ticker: str, algorithm: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get active model for a market."""
        return await rl_db.get_active_model_for_market(market_ticker, algorithm)
    
    async def list_models(
        self,
        market_ticker: Optional[str] = None,
        status: Optional[str] = None,
        algorithm: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """List models with optional filtering."""
        return await rl_db.list_models(market_ticker, status, algorithm, limit)
    
    async def get_model_performance_summary(self, model_id: int) -> Optional[Dict[str, Any]]:
        """Get comprehensive performance summary for a model."""
        summary = await rl_db.get_model_performance_summary(model_id)
        
        # Add performance history if available
        if model_id in self._performance_history and summary:
            summary['performance_history'] = self._performance_history[model_id]
        
        return summary
    
    async def cleanup_old_models(self, market_ticker: str, keep_count: int = 5) -> int:
        """
        Clean up old models for a market, keeping only the most recent ones.
        
        Args:
            market_ticker: Market to clean up
            keep_count: Number of models to keep
            
        Returns:
            Number of models cleaned up
        """
        try:
            # Get all models for market, sorted by creation date
            models = await rl_db.list_models(
                market_ticker=market_ticker,
                limit=1000  # Get all models
            )
            
            # Filter out active models (never clean up active models)
            inactive_models = [m for m in models if m['status'] != 'active']
            
            if len(inactive_models) <= keep_count:
                return 0  # Nothing to clean up
            
            # Sort by creation date and identify models to remove
            inactive_models.sort(key=lambda x: x['created_at'], reverse=True)
            models_to_remove = inactive_models[keep_count:]
            
            cleaned_count = 0
            for model in models_to_remove:
                # Set status to retired
                await rl_db.update_model_status(model['id'], 'retired')
                
                # Remove model file if it exists
                file_path = model.get('file_path', '')
                if file_path and os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        logger.info(f"Removed model file: {file_path}")
                    except Exception as e:
                        logger.warning(f"Failed to remove model file {file_path}: {e}")
                
                cleaned_count += 1
            
            logger.info(f"Cleaned up {cleaned_count} old models for {market_ticker}")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup old models for {market_ticker}: {e}")
            return 0
    
    def add_hot_reload_callback(self, callback: Callable[[str, str], None]) -> None:
        """
        Add callback for hot-reload events.
        
        Args:
            callback: Function to call when model is hot-reloaded.
                     Takes (model_id, file_path) as arguments.
        """
        self._reload_callbacks.append(callback)
    
    async def check_for_updates(self) -> List[str]:
        """
        Check for updated model files and trigger hot-reload.
        
        Returns:
            List of model file paths that were updated
        """
        updated_files = []
        
        for file_path, last_modified in self._file_watchers.items():
            try:
                if os.path.exists(file_path):
                    current_modified = os.path.getmtime(file_path)
                    if current_modified > last_modified:
                        self._file_watchers[file_path] = current_modified
                        updated_files.append(file_path)
                        
                        # Find model ID for this file path
                        model_info = await self._find_model_by_file_path(file_path)
                        if model_info:
                            await self._trigger_hot_reload(model_info['id'])
                            
            except Exception as e:
                logger.warning(f"Error checking file {file_path} for updates: {e}")
        
        return updated_files
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe filesystem storage."""
        # Remove or replace unsafe characters
        safe_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-."
        sanitized = ''.join(c if c in safe_chars else '_' for c in filename)
        
        # Limit length and ensure it's not empty
        sanitized = sanitized[:100] or 'model'
        
        return sanitized
    
    async def _check_auto_deployment(self, model_id: int, validation_metrics: Dict[str, Any]) -> None:
        """Check if model should be auto-deployed based on performance."""
        try:
            # Get model info to check auto-deploy threshold
            model_info = await rl_db.get_model_by_id(model_id)
            if not model_info:
                return
            
            # Extract auto-deploy threshold from hyperparameters
            hyperparams = model_info.get('hyperparameters', {})
            auto_deploy_threshold = hyperparams.get('auto_deploy_threshold')
            
            if auto_deploy_threshold is None:
                return  # No auto-deployment configured
            
            # Check if performance exceeds threshold
            performance_metric = validation_metrics.get('avg_return', 0.0)
            
            if performance_metric >= auto_deploy_threshold:
                # Auto-deploy model
                success = await self.set_model_status(model_id, 'active')
                if success:
                    logger.info(f"Auto-deployed model {model_id} with performance {performance_metric:.4f}")
                else:
                    logger.warning(f"Failed to auto-deploy model {model_id}")
                    
        except Exception as e:
            logger.error(f"Error in auto-deployment check for model {model_id}: {e}")
    
    async def _find_model_by_file_path(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Find model info by file path."""
        # This is not efficient for large model counts, but works for now
        # Could be optimized with a reverse lookup cache if needed
        models = await rl_db.list_models(limit=1000)
        
        for model in models:
            if model.get('file_path') == file_path:
                return model
        
        return None
    
    async def _trigger_hot_reload(self, model_id: int) -> None:
        """Trigger hot-reload callbacks for a model."""
        try:
            model_info = await rl_db.get_model_by_id(model_id)
            if model_info:
                file_path = model_info['file_path']
                
                # Call all registered callbacks
                for callback in self._reload_callbacks:
                    try:
                        callback(str(model_id), file_path)
                    except Exception as e:
                        logger.warning(f"Hot-reload callback failed for model {model_id}: {e}")
                        
        except Exception as e:
            logger.error(f"Failed to trigger hot-reload for model {model_id}: {e}")


# Global model registry instance
model_registry = ModelRegistry()