"""
Configuration management for Kalshi Flow RL Trading Subsystem.

Loads environment variables and provides typed configuration objects
for all RL components including data ingestion, training, and trading.
"""

import os
import logging
from typing import Optional
from pathlib import Path


class RLConfig:
    """Configuration for RL Trading Subsystem."""
    
    def __init__(self):
        # Kalshi API Configuration (shared with main app - auth handled by KalshiAuth.from_env())
        # The auth component reads KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH/CONTENT directly
        self.KALSHI_WS_URL: str = os.getenv("KALSHI_WS_URL", "wss://api.elections.kalshi.com/trade-api/ws/v2")
        
        # Database Configuration (shared with main app)
        self.DATABASE_URL: str = self._get_required_env("DATABASE_URL")
        self.DATABASE_URL_POOLED: Optional[str] = os.getenv("DATABASE_URL_POOLED")
        
        # RL-Specific Configuration
        self.RL_MARKET_TICKER: str = os.getenv("RL_MARKET_TICKER", "INXD-25JAN03")  # Default market for RL training
        
        # Orderbook Ingestion Settings
        self.ORDERBOOK_QUEUE_BATCH_SIZE: int = int(os.getenv("RL_ORDERBOOK_BATCH_SIZE", "100"))
        self.ORDERBOOK_QUEUE_FLUSH_INTERVAL: float = float(os.getenv("RL_ORDERBOOK_FLUSH_INTERVAL", "1.0"))
        self.ORDERBOOK_DELTA_SAMPLE_RATE: int = int(os.getenv("RL_ORDERBOOK_SAMPLE_RATE", "1"))  # Keep 1 out of N deltas
        self.ORDERBOOK_MAX_QUEUE_SIZE: int = int(os.getenv("RL_ORDERBOOK_MAX_QUEUE_SIZE", "10000"))
        
        # WebSocket and Performance Settings
        self.WEBSOCKET_PING_INTERVAL: int = int(os.getenv("RL_WEBSOCKET_PING_INTERVAL", "30"))
        self.WEBSOCKET_TIMEOUT: int = int(os.getenv("RL_WEBSOCKET_TIMEOUT", "60"))
        self.WEBSOCKET_RECONNECT_DELAY: int = int(os.getenv("RL_WEBSOCKET_RECONNECT_DELAY", "5"))
        self.MAX_RECONNECT_ATTEMPTS: int = int(os.getenv("RL_MAX_RECONNECT_ATTEMPTS", "10"))
        
        # Database Pool Settings
        self.DB_POOL_MIN_SIZE: int = int(os.getenv("RL_DB_POOL_MIN_SIZE", "2"))
        self.DB_POOL_MAX_SIZE: int = int(os.getenv("RL_DB_POOL_MAX_SIZE", "10"))
        self.DB_POOL_TIMEOUT: int = int(os.getenv("RL_DB_POOL_TIMEOUT", "30"))
        
        # Logging Configuration
        self.LOG_LEVEL: str = os.getenv("RL_LOG_LEVEL", "INFO")
        self.LOG_FORMAT: str = os.getenv("RL_LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        
        # Application Settings
        self.ENVIRONMENT: str = os.getenv("ENVIRONMENT", "local")
        self.DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
        
        # RL System Safety - Only allow training and paper modes
        self.ALLOWED_TRADING_MODES = ["training", "paper"]  # Never allow 'live' mode
        
        # Model Storage and Registry
        self.MODEL_STORAGE_PATH: str = os.getenv("RL_MODEL_STORAGE_PATH", "./models")
        self.MODEL_REGISTRY_REFRESH_INTERVAL: int = int(os.getenv("RL_MODEL_REFRESH_INTERVAL", "30"))
        
        # Training Data Settings  
        self.TRAINING_DATA_WINDOW_HOURS: int = int(os.getenv("RL_TRAINING_WINDOW_HOURS", "24"))
        self.MAX_EPISODE_STEPS: int = int(os.getenv("RL_MAX_EPISODE_STEPS", "1000"))
        
        # Validate configuration
        self._validate_config()
    
    def _get_required_env(self, key: str) -> str:
        """Get required environment variable or raise error."""
        value = os.getenv(key)
        if not value:
            # Allow test environments to bypass required env vars
            if os.getenv("ENVIRONMENT") == "test":
                return f"test_{key.lower()}"
            raise ValueError(f"Required environment variable {key} is not set")
        return value
    
    def _validate_config(self) -> None:
        """Validate configuration values."""
        # Skip validation in test environment
        if self.ENVIRONMENT == "test":
            return
            
        # Kalshi auth validation is handled by KalshiAuth.from_env()
        
        # Validate numeric ranges
        if self.ORDERBOOK_QUEUE_BATCH_SIZE <= 0:
            raise ValueError("ORDERBOOK_QUEUE_BATCH_SIZE must be positive")
        
        if self.ORDERBOOK_QUEUE_FLUSH_INTERVAL <= 0:
            raise ValueError("ORDERBOOK_QUEUE_FLUSH_INTERVAL must be positive")
        
        if self.ORDERBOOK_DELTA_SAMPLE_RATE < 1:
            raise ValueError("ORDERBOOK_DELTA_SAMPLE_RATE must be >= 1")
        
        # Validate database URL format
        if not self.DATABASE_URL.startswith("postgresql://"):
            raise ValueError("DATABASE_URL must be a PostgreSQL connection string")
        
        # Ensure model storage directory exists
        Path(self.MODEL_STORAGE_PATH).mkdir(parents=True, exist_ok=True)
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.ENVIRONMENT in ["local", "development"]
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.ENVIRONMENT == "production"


# Global configuration instance
config = RLConfig()


def setup_logging() -> logging.Logger:
    """Set up logging configuration for RL subsystem."""
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format=config.LOG_FORMAT
    )
    
    # Create RL-specific logger
    logger = logging.getLogger("kalshiflow_rl")
    
    if config.is_development:
        # Add console handler for development
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(config.LOG_FORMAT)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


# Initialize logger
logger = setup_logging()